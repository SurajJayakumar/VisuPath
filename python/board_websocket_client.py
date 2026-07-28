import asyncio
import websockets
import multiprocessing
import signal


# -------- WEBSOCKET CONFIG ----------
PHONE_IP = "192.168.1.251"
PHONE_PORT = 8765
URI = f"ws://{PHONE_IP}:{PHONE_PORT}/board"

RECONNECT_DELAY = 2 # Seconds before retry on disconnect
PING_INTERVAL = 10
PING_TIMEOUT = 20
CLOSE_TIMEOUT = 5

# ----------------------------

def handle_sigterm(signum, frame):
    """
    SIGTERM handler, cancel all asyncio tasks gracefully
    instead of dying immediately
    """
    print("[Websocket Client] SIGTERM received. Shutting down...")
    loop = asyncio.get_event_loop()
    for task in asyncio.all_tasks(loop):
        task.cancel()


def dequeue(ipc_queue: multiprocessing.Queue) -> str:
    """
    Blocking call that waits until a message is available.
    """
    return ipc_queue.get()


async def handle_exception(log: str, delay: float = 0):
    """
    Log error and (optional) sleep non-blocking for some delay
    """
    print(f"[Websocket Client] {log}. Retrying in {delay} seconds.")
    
    if delay > 0:   
        await asyncio.sleep(delay)
    
    

async def ws_sender(ipc_queue: multiprocessing.Queue):
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
                    message = await loop.run_in_executor(None, dequeue, ipc_queue)

                    await websocket.send(message)
                    print(f"[Websocket client] sent: {message}")

        
        except asyncio.CancelledError:

            handle_exception("Cancelled, shutting down cleanly")
            return

        
        except (websockets.exceptions.ConnectionClosedError, OSError) as e:
            
            handle_exception(f"Connection Error: {e}", RECONNECT_DELAY)
            

        except Exception as e:

            handle_exception(f"Unexpected Error: {e}", RECONNECT_DELAY)
            


async def _run_board_client(ipc_queue: multiprocessing.Queue):
    
    loop = asyncio.get_running_loop()

    loop.add_signal_handler(
        signal.SIGTERM,
        lambda: [task.cancel() for task in asyncio.all_tasks(loop)]
    )
    
    try:
        
        await ws_sender(ipc_queue)
        
    except asyncio.CancelledError:
        print("[Websocket Client] Exited Successfully.")


def run_board_client(ipc_queue: multiprocessing.Queue):
    asyncio.run(_run_board_client(ipc_queue))
