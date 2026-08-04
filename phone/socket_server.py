# Shared Unix socket server boilerplate used by tts_client.py and vlm_worker.py.

import os
import socket
import threading
from typing import Callable


def read_lines(
    conn: socket.socket,
    on_line: Callable[[bytes, socket.socket], None],
    recv_size: int = 4096,
) -> None:
    """
    Read from `conn` until closed, splitting on newlines.
    Calls `on_line(raw_line_bytes, conn)` for each complete line.
    `conn` is passed through so on_line can write a response if needed (e.g. VLM).
    """
    buf = b''
    try:
        with conn:
            while True:
                chunk = conn.recv(recv_size)
                if not chunk:
                    break
                buf += chunk
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    line = line.strip()
                    if line:
                        on_line(line, conn)
    except Exception as e:
        print(f'[Socket] Connection error: {e}')


def serve_unix_socket(
    sock_path: str,
    label: str,
    handler: Callable[[socket.socket], None],
    backlog: int = 5,
) -> None:
    """
    Bind a Unix domain socket, accept connections in a loop, and dispatch
    each connection to `handler` in a daemon thread.

    handler(conn) is responsible for reading, processing, and closing the connection.
    Blocks until KeyboardInterrupt, then cleans up the socket file.
    """
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(backlog)
    os.chmod(sock_path, 0o600)
    print(f'[{label}] Listening on {sock_path}')

    try:
        while True:
            conn, _ = server.accept()
            threading.Thread(target=handler, args=(conn,), daemon=True).start()
    except KeyboardInterrupt:
        print(f'[{label}] Shutting down.')
    finally:
        server.close()
        if os.path.exists(sock_path):
            os.unlink(sock_path)