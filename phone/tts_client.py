import socket
import subprocess
import threading
import time

from socket_server import serve_unix_socket, read_lines

# -- Config --------------------------------------------------------------------
SOCK_PATH       = './tts.sock'
MSG_TYPE_OBJECT = 0x01   # YOLO alert  — normal priority
MSG_TYPE_SCENE  = 0x02   # VLM scene   — high priority, spoken first
# -----------------------------------------------------------------------------


class PriorityAlertSlot:
    """
    Two-tier latest-only buffer.
    VLM scene descriptions (0x02) are spoken before YOLO alerts (0x01).
    Within each tier, only the latest message is kept — no backlog possible.
    """

    def __init__(self) -> None:
        self._scene: str | None = None
        self._alert: str | None = None
        self._lock  = threading.Lock()
        self._event = threading.Event()

    def put_scene(self, text: str) -> None:
        with self._lock:
            self._scene = text
        self._event.set()

    def put_alert(self, text: str) -> None:
        with self._lock:
            self._alert = text
        self._event.set()

    def take(self) -> str | None:
        self._event.wait()
        self._event.clear()
        with self._lock:
            if self._scene is not None:
                text = self._scene
                self._scene = None
                if self._alert is not None:
                    self._event.set()   # re-arm so pending alert is spoken next
                return text
            if self._alert is not None:
                text = self._alert
                self._alert = None
                return text
        return None


def speaker_thread(slot: PriorityAlertSlot) -> None:
    """
    Drains the priority slot and speaks each item.
    Always speaks the latest — never backlogs.
    """
    while True:
        text = slot.take()
        if not text:
            continue
        t0 = time.perf_counter()
        try:
            subprocess.run(['termux-tts-speak', text], timeout=30)
        except Exception as e:
            print(f'[TTS] termux tts error: {e}')
        finally:
            elapsed_ms = 1000 * (time.perf_counter() - t0)
            print(f'[TTS] Spoke in {elapsed_ms:.0f}ms: {text}')


def main() -> None:
    slot = PriorityAlertSlot()
    threading.Thread(target=speaker_thread, args=(slot,), daemon=True, name='Speaker').start()

    def on_line(line: bytes, conn: socket.socket) -> None:
        # Frame format: [1-byte type][utf8 text]
        # type 0x01 = YOLO alert (normal priority)
        # type 0x02 = VLM scene  (high priority, spoken first)
        msg_type = line[0]                                      # int, O(1) — no string scan
        text     = line[1:].decode('utf-8', errors='replace').strip()
        if not text:
            return
        if msg_type == MSG_TYPE_SCENE:
            slot.put_scene(text)
            print(f'[TTS] Scene queued: {text[:60]}')
        else:
            slot.put_alert(text)
            print(f'[TTS] Alert queued: {text[:60]}')

    def handler(conn: socket.socket) -> None:
        read_lines(conn, on_line)

    serve_unix_socket(SOCK_PATH, 'TTS Client', handler)


if __name__ == '__main__':
    main()