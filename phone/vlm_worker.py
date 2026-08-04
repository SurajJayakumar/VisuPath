import base64
import socket
import time
import requests

from socket_server import serve_unix_socket, read_lines

LLAMA_SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
MAX_TOKENS = 80
TEMPERATURE = 0.1
INFERENCE_TIMEOUT = 60

SYSTEM_PROMPT = (
    "You are a navigation assistant for a visually impaired person using a smart cane. "
    "Be precise, concise, and safety-focused. "
    "Describe only visible obstacles, hazards, people, vehicles, signs, crossings, "
    "and important navigation information. "
    "Prioritize objects directly ahead and state their relative direction "
    "such as left, center, or right. "
    "Mention distance only when it can be reasonably estimated. "
    "Do not invent details. "
    "Limit the response to 2 short sentences."
)


def run_inference(image_b64: str) -> str:
    start = time.perf_counter()

    # Add the MIME prefix required by the OpenAI-compatible image format.
    image_url = f"data:image/jpeg;base64,{image_b64}"

    payload = {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this scene for immediate navigation hazards."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False
    }

    try:
        response = requests.post(
            LLAMA_SERVER_URL,
            json=payload,
            timeout=INFERENCE_TIMEOUT
        )
        response.raise_for_status()

        data = response.json()
        scene_text = data["choices"][0]["message"]["content"]

        # Keep speech output on one line.
        scene_text = " ".join(scene_text.split())

    except requests.RequestException as error:
        scene_text = f"Scene analysis error: {error}"

    except (KeyError, IndexError, TypeError, ValueError) as error:
        scene_text = f"Invalid llama-server response: {error}"

    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"[VLM Worker] inference={elapsed_ms:.0f} ms: {scene_text[:120]}")

    return scene_text


VLM_SOCK = './vlm.sock'


def main() -> None:
    def on_line(line: bytes, conn: socket.socket) -> None:
        """Receive a base64 JPEG line, run inference, write result back."""
        image_b64 = line.decode('utf-8', errors='replace').strip()
        if not image_b64:
            return
        result = run_inference(image_b64)
        try:
            conn.sendall((result + '\n').encode('utf-8'))
        except Exception as e:
            print(f'[VLM Worker] Failed to send result: {e}')

    def handler(conn: socket.socket) -> None:
        read_lines(conn, on_line)

    serve_unix_socket(VLM_SOCK, 'VLM Worker', handler)


if __name__ == '__main__':
    main()
