import multiprocessing
import asyncio
import random
import time
import queue # for queue.Full exception
import sys
import os

sys.path.append(os.path.dirname(__file__))
from board_websocket_client import run_board_client

# ----------- IPC QUEUE CONFIG ----------
QUEUE_SIZE = 3
# --------------------------------------

# -------- DUMMY DETECTIONS (replace with real model output) ----------
DUMMY_DETECTIONS = [
    "Tree detected to your right, 20 feet away",
    "Car approaching from the left",
    "Intersection ahead, 30 feet away",
    "Person walking towards you, 10 feet away",
    "Dog detected to your right, 8 feet away",
    "Steps detected ahead, 6 feet away",
    "Bicycle approaching from behind",
    "Pothole detected ahead, 3 feet away",
]
# ---------------------------------------------------------------------



def enqueue(ipc_queue: multiprocessing.Queue, text: str):
    """
    Drop oldest stale message if queue is full.
    Only keep freshest obstacle alerts for mobility cane.
    """

    try:

        ipc_queue.put_nowait(text)

    except queue.Full:

        try:

            dropped = ipc_queue.get_nowait()
            print(f"[Queue] dropped stale: '{dropped}'")

            ipc_queue.put_nowait(text)

        except queue.Empty:
            pass
            

def run_inference(ipc_queue: multiprocessing.Queue):
    """
    CPU-bound ML inference loop, runs in the main process.
    Replace random.choice() with the Actual CV computation.
    """

    print("[Visupath] Starting detection pipeline...")

    while True:

        detection = random.choice(DUMMY_DETECTIONS)

        enqueue(ipc_queue, detection)

        print(f"[Visupath] Queued: {detection}")

        # Remove this sleep when performing real computations.
        time.sleep(5)



def main():

    print("=== VisuPath Starting ===")

    # Shared queue between inference process and websocket process
    ipc_queue = multiprocessing.Queue(QUEUE_SIZE)


    # Spawn websocket client sending data to phone as a completely separate OS process
    # So that kernel can run it in parallel on another core

    board_process = multiprocessing.Process(
        target = run_board_client,
        args = (ipc_queue,),
        daemon = True,
        name = "BoardWebSocketClient"
    )

    board_process.start()
    print(f"[Visupath] Board Client process started(PID: {board_process.pid})")
    
    try:
    
        run_inference(ipc_queue)
    
    except KeyboardInterrupt:
        print("[Visupath] Shutting down...")
    
    finally:
        
        board_process.terminate()
        board_process.join() # Wait for clean exit
        print("[Visupath] Board client process stopped.")
        print("[visupath] Exited cleanly.")


if __name__ == "__main__":
    main()