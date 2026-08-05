import json
import os
import socket
import time
from typing import Optional

import requests

from socket_server import serve_unix_socket, read_lines


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]


MAX_OUTPUT_TOKENS = 1024 # Max number of tokens the AI model can consume per scnene explanation request. More tokens: Better explanation but costs more.
TEMPERATURE = 0.0 # How creative you want model to be... try to keep it low for correctness

# Connect timeout, read timeout (seconds).
REQUEST_TIMEOUT = (10, 60)



# The GEMINI_MODEL env var sets the primary; the rest are automatic fallbacks
# used when the primary returns 503 (overloaded) or 429 (rate-limited).
_primary = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
_candidates = [
    _primary,
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]
# Deduplicate while preserving order.
_seen: set = set()
FALLBACK_MODELS: list[str] = [m for m in _candidates if not (m in _seen or _seen.add(m))]
del _primary, _candidates, _seen

VLM_SOCK = "./vlm.sock"





SYSTEM_PROMPT = """You are a navigation assistant for a visually impaired user.

Describe the image using this exact format:

Environment: <one short sentence describing the surroundings>
Hazard: <the most important visible object, obstacle, or hazard and its direction (left, center, or right)>

If there is no hazard, describe what is directly ahead instead.
Do not invent details. Use simple language."""

USER_PROMPT = "Describe this image."


# Reuse HTTP/TLS connections between requests.
HTTP_SESSION = requests.Session()
HTTP_SESSION.headers.update({"Content-Type": "application/json"})

# Index into FALLBACK_MODELS; rotated forward on 503/429.
_model_idx = 0


def _model_url(model: str) -> str:
    return (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def extract_text(response_json: dict) -> str:
    """
    Extract visible (non-thought) text from a Gemini response.
    Raises ValueError if no usable text is found.
    """
    candidates = response_json.get("candidates") or []
    if not candidates:
        raise ValueError(f"No candidates; promptFeedback={response_json.get('promptFeedback')}")

    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []

    text_parts = [
        part["text"].strip()
        for part in parts
        if isinstance(part, dict) and part.get("text") and not part.get("thought", False)
    ]

    result = " ".join(" ".join(text_parts).split())
    if result:
        return result

    raise ValueError(f"No visible text; finishReason={candidate.get('finishReason')}")


# ---------------------------------------------------------------------------
# Gemini request
# ---------------------------------------------------------------------------

def make_payload(image_b64: str, mime_type: str) -> dict:
    return {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": USER_PROMPT},
                    {"inlineData": {"mimeType": mime_type, "data": image_b64}},
                ],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "temperature": TEMPERATURE,
        },
    }


def run_inference(image_b64: str) -> Optional[str]:
    """
    Run Gemini inference. Returns the scene description string, or None on any
    error (error is logged but never forwarded to TTS).
    """
    global _model_idx

    start = time.perf_counter()
    scene_text = None

    try:
        mime_type = "image/jpeg"  # object_detector.py always encodes as JPEG

        model = FALLBACK_MODELS[_model_idx]
        url = _model_url(model)
        payload = make_payload(image_b64, mime_type)

        approx_bytes = (len(image_b64) * 3) // 4
        print(
            f"[VLM Worker] Request model={model} mime={mime_type} "
            f"image_b64_len={len(image_b64)} approx_bytes={approx_bytes}"
        )
        print(
            "[VLM Worker] generationConfig="
            + json.dumps(payload["generationConfig"], indent=2)
        )

        http_start = time.perf_counter()
        response = HTTP_SESSION.post(
            url, params={"key": GEMINI_API_KEY}, json=payload, timeout=REQUEST_TIMEOUT
        )
        http_ms = (time.perf_counter() - http_start) * 1000
        print(f"[VLM Worker] HTTP {response.status_code} from {model} in {http_ms:.0f}ms")

        # Model overloaded or rate-limited -- rotate to next fallback and retry once.
        if response.status_code in (429, 503):
            old_model = model
            _model_idx = (_model_idx + 1) % len(FALLBACK_MODELS)
            model = FALLBACK_MODELS[_model_idx]
            url = _model_url(model)
            print(f"[VLM Worker] {response.status_code} switching {old_model} -> {model}")
            http_start = time.perf_counter()
            response = HTTP_SESSION.post(
                url, params={"key": GEMINI_API_KEY}, json=payload, timeout=REQUEST_TIMEOUT
            )
            http_ms = (time.perf_counter() - http_start) * 1000
            print(f"[VLM Worker] HTTP {response.status_code} from {model} in {http_ms:.0f}ms")

        response.raise_for_status()
        response_json = response.json()

        try:
            candidate = response_json["candidates"][0]
            finish_reason = candidate.get("finishReason")
            usage = response_json.get("usageMetadata", {})
            print(f"[VLM Worker] finishReason={finish_reason}")
            print(
                "[VLM Worker] token usage:\n"
                + json.dumps(
                    {
                        "prompt": usage.get("promptTokenCount"),
                        "candidate": usage.get("candidatesTokenCount"),
                        "thoughts": usage.get("thoughtsTokenCount"),
                        "total": usage.get("totalTokenCount"),
                    },
                    indent=2,
                )
            )
        except Exception:
            pass

        scene_text = extract_text(response_json)

        if scene_text and not scene_text.rstrip().endswith((".", "!", "?")):
            print(f"[VLM Worker] WARNING: response appears truncated: {scene_text[:80]!r}")

    except requests.ConnectTimeout:
        print(
            f"[VLM Worker] CONNECT TIMEOUT model={model} timeout={REQUEST_TIMEOUT[0]}s"
        )

    except requests.ReadTimeout:
        old_model = model
        _model_idx = (_model_idx + 1) % len(FALLBACK_MODELS)
        model = FALLBACK_MODELS[_model_idx]
        print(
            f"[VLM Worker] READ TIMEOUT {REQUEST_TIMEOUT[1]}s -- "
            f"switching {old_model} -> {model}"
        )
        try:
            url = _model_url(model)
            http_start = time.perf_counter()
            response = HTTP_SESSION.post(
                url, params={"key": GEMINI_API_KEY}, json=payload, timeout=REQUEST_TIMEOUT
            )
            http_ms = (time.perf_counter() - http_start) * 1000
            print(f"[VLM Worker] HTTP {response.status_code} from {model} in {http_ms:.0f}ms")
            response.raise_for_status()
            response_json = response.json()
            try:
                candidate = response_json["candidates"][0]
                finish_reason = candidate.get("finishReason")
                usage = response_json.get("usageMetadata", {})
                print(f"[VLM Worker] finishReason={finish_reason}")
                print(
                    "[VLM Worker] token usage:\n"
                    + json.dumps(
                        {
                            "prompt": usage.get("promptTokenCount"),
                            "candidate": usage.get("candidatesTokenCount"),
                            "thoughts": usage.get("thoughtsTokenCount"),
                            "total": usage.get("totalTokenCount"),
                        },
                        indent=2,
                    )
                )
            except Exception:
                pass
            scene_text = extract_text(response_json)
            if scene_text and not scene_text.rstrip().endswith((".", "!", "?")):
                print(f"[VLM Worker] WARNING: response appears truncated: {scene_text[:80]!r}")
        except Exception as retry_error:
            print(f"[VLM Worker] Fallback {model} also failed: {retry_error}")

    except requests.RequestException as error:
        body = ""
        if getattr(error, "response", None) is not None:
            try:
                body = json.dumps(error.response.json(), indent=2)
            except Exception:
                body = error.response.text
        print(f"[VLM Worker] HTTP error: {error}\n[VLM Worker] Response body:\n{body}")

    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"[VLM Worker] Response error: {error}")

    except Exception as error:
        print(f"[VLM Worker] Error: {error}")

    elapsed_ms = (time.perf_counter() - start) * 1000
    current_model = FALLBACK_MODELS[_model_idx]
    print(
        f"[VLM Worker] model={current_model} inference={elapsed_ms:.0f} ms | "
        f"{scene_text[:160] if scene_text else 'error (not spoken)'}"
    )

    return scene_text


# ---------------------------------------------------------------------------
# Unix socket server
# ---------------------------------------------------------------------------

def main() -> None:
    def on_line(line: bytes, conn: socket.socket) -> None:
        image_value = line.decode("utf-8", errors="replace").strip()
        if not image_value:
            return

        result = run_inference(image_value)

        if result is None:
            return  # error already logged; never forward to TTS

        try:
            conn.sendall((result + "\n").encode("utf-8"))
        except Exception as error:
            print(f"[VLM Worker] Failed to send result: {error}")

    def handler(conn: socket.socket) -> None:
        read_lines(conn, on_line)

    primary = FALLBACK_MODELS[0]
    print(f"[VLM Worker] Primary model : {primary}")
    print(f"[VLM Worker] Fallback chain: {' -> '.join(FALLBACK_MODELS)}")
    print(f"[VLM Worker] Listening on  : {VLM_SOCK}")

    serve_unix_socket(VLM_SOCK, "VLM Worker", handler)


if __name__ == "__main__":
    main()