import multiprocessing
import queue # for queue.Full exception
import sys
import os

sys.path.append(os.path.dirname(__file__))
from board_websocket_client import run_board_client
from object_detector import ObjectDetectionPipeline

# ----------- IPC QUEUE CONFIG ----------
OBJECT_ALERT_QUEUE_SIZE = 3
SCENE_ALERT_QUEUE_SIZE = 1
# --------------------------------------

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
            

def run_inference(object_alert_queue : multiprocessing.Queue, scene_alert_queue: multiprocessing.Queue):
    """
    CPU-bound object detection loop, runs in the main process.
    Defines _push_snapshot as a closure + callback so ObjectDetectionPipeline can invoke it passing the image for scene explanation
    """

    def _push_snapshot(image_b64: str) -> None:
        """
        Drop-on-full snapshot enqueue, only latest scene matters.
        """

        try:
            scene_alert_queue.put_nowait(image_b64)
        except queue.Full:
            try:
                scene_alert_queue.get_nowait()
                scene_alert_queue.put_nowait(image_b64)
            except queue.Empty:
                pass


    print("[VisuPath] Starting YOLO object detection pipeline...")
    pipeline = ObjectDetectionPipeline(on_snapshot=_push_snapshot)

    for detection in pipeline.alerts():
        enqueue(object_alert_queue, detection)
        print(f"[VisuPath] Queued: {detection}")



def main():

    print("=== VisuPath Starting ===")

    # Shared queue between inference process and websocket process
    object_alert_queue = multiprocessing.Queue(OBJECT_ALERT_QUEUE_SIZE)
    scene_alert_queue = multiprocessing.Queue(SCENE_ALERT_QUEUE_SIZE)

    # Spawn websocket client sending data to phone as a completely separate OS process
    # So that kernel can run it in parallel on another core

    board_process = multiprocessing.Process(
        target = run_board_client,
        args = (object_alert_queue, scene_alert_queue),
        daemon = True,
        name = "BoardWebSocketClient"
    )

    board_process.start()
    print(f"[Visupath] Board Client process started(PID: {board_process.pid})")
    
    try:
    
        run_inference(object_alert_queue, scene_alert_queue)
    
    except KeyboardInterrupt:
        pass
    
    finally:

        print("[Visupath] Shutting down...")
        
        try:
            object_alert_queue.put_nowait(None)  # sentinels to unblock the board websocket client
            scene_alert_queue.put_nowait(None)
        except Exception:
            pass

        board_process.terminate()
        board_process.join(timeout=5)

        if board_process.is_alive():
            print("[VisuPath] Force killing board process.")
            board_process.kill()
            board_process.join()

        print("[Visupath] Board client process stopped.")
        print("[visupath] Exited cleanly.")


if __name__ == "__main__":
    main()


# Camera index finding code keep the below code snippet commented
#import cv2
#
#print("Scanning camera indices 0–5...")
#for idx in range(6):
#    cap = cv2.VideoCapture(idx)
#    if cap.isOpened():
#        ret, frame = cap.read()
#        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#        print(f"  index={idx} | opened=True | read()={ret} | {w}x{h}")
#        cap.release()
#    else:
#        print(f"  index={idx} | opened=False")
