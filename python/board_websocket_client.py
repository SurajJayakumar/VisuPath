from __future__ import annotations

import asyncio
import websockets
import multiprocessing
import queue
import signal
from dataclasses import dataclass
from typing import Union


# -------- WEBSOCKET CONFIG ----------
PHONE_IP = "192.168.1.251"
PHONE_PORT = 8765
URI = f"ws://{PHONE_IP}:{PHONE_PORT}/board"

RECONNECT_DELAY = 2 # Seconds before retry on disconnect
PING_INTERVAL = 10
PING_TIMEOUT = 20
CLOSE_TIMEOUT = 5

# ----------------------------
# ── Message type bytes ────────────────────────────────────────────────────────
MSG_TYPE_OBJECT = 0x01   # YOLO object detection text
MSG_TYPE_SCENE  = 0x02   # VLM scene image (base64)



# --- Dequeue result variants ---

@dataclass(frozen = True)
class QueueMessage:
    msg_type: int
    payload: str

@dataclass(frozen=True)
class QueueEmpty:
    pass

@dataclass(frozen=True)
class QueueShutdown:
    pass

DequeueResult = Union[QueueMessage, QueueEmpty, QueueShutdown]



def dequeue_with_priority(
        object_alert_queue: multiprocessing.Queue,
        scene_alert_queue: multiprocessing.Queue,
) -> DequeueResult:
    """
    Check scene_alert_queue (VLM) non-blocking first, then block on
    object_alert_queue for up to 50ms. Returns typed result — no globals.
    """

    try:
        msg = scene_alert_queue.get_nowait()

        if msg is None:
            return QueueShutdown()

        return QueueMessage(msg_type=MSG_TYPE_SCENE, payload=msg)

    except queue.Empty:
        pass

    try:

        text = object_alert_queue.get(timeout=0.05)

        if text is None:
            return QueueShutdown()

        return QueueMessage(msg_type=MSG_TYPE_OBJECT, payload=text)
    
    except queue.Empty:
        return QueueEmpty()


def frame_message(msg_type: int, payload: str) -> bytes:
    """Prepend 1-byte type prefix. Websocket handles length framing."""
    return bytes([msg_type]) + payload.encode('utf-8')



async def handle_exception(log: str, delay: float = 0) -> None:
    """
    Log error and (optional) sleep non-blocking for some delay
    """
    print(f"[Websocket Client] {log}. Retrying in {delay} seconds.")
    
    if delay > 0:   
        await asyncio.sleep(delay)
    
    

async def ws_sender(object_alert_queue: multiprocessing.Queue, scene_alert_queue: multiprocessing.Queue) -> None:
    """
    Websocket client sending data from a queue to websocket server.
    Attempts reconnection automatically on disconnect
    """
    
    loop = asyncio.get_running_loop()

    while True:
        try:

            print(f"[Websocket Client] Connecting to {URI}...")

            async with websockets.connect(
                URI,
                ping_interval = PING_INTERVAL,
                ping_timeout = PING_TIMEOUT,
                close_timeout = CLOSE_TIMEOUT
            ) as websocket:

                print("[Websocket Client] Connected to phone!")

                while True:

                    # bridges blocking queue.get() into async event loop
                    result: DequeueResult = await loop.run_in_executor(None, dequeue_with_priority, object_alert_queue, scene_alert_queue)

                    match result:

                        case QueueShutdown():
                            print("[WebSocket Client] Sentinel received, shutting down.")
                            return

                        case QueueEmpty():
                            continue

                        case QueueMessage(msg_type = t, payload = p):
                            await websocket.send(frame_message(t, p))
                            print(f"[Websocket client] sent type={t:#04x} len={len(p)}")

        
        except asyncio.CancelledError:

            await handle_exception("Cancelled, shutting down cleanly")
            return

        
        except (websockets.exceptions.ConnectionClosedError, OSError) as e:
            
            await handle_exception(f"Connection Error: {e}", RECONNECT_DELAY)
            

        except Exception as e:

            await handle_exception(f"Unexpected Error: {e}", RECONNECT_DELAY)
            


async def _run_board_client(object_alert_queue: multiprocessing.Queue, scene_alert_queue: multiprocessing.Queue) -> None:
    
    loop = asyncio.get_running_loop()

    loop.add_signal_handler(
        signal.SIGTERM,
        lambda: [task.cancel() for task in asyncio.all_tasks(loop)]
    )
    
    try:
        
        await ws_sender(object_alert_queue, scene_alert_queue)
        
    except asyncio.CancelledError:
        print("[Websocket Client] Exited Successfully.")


def run_board_client(object_alert_queue: multiprocessing.Queue, scene_alert_queue: multiprocessing.Queue) -> None:
    asyncio.run(_run_board_client(object_alert_queue, scene_alert_queue))
