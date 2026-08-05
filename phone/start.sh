#!/data/data/com.termux/files/usr/bin/bash
# VisuPath phone-side startup script
# Run from ~/visupath/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_PATH="$HOME/models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
MMPROJ_PATH="$HOME/models/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"
LLAMA_SERVER="$(which llama-server)"

echo "[Visupath] Starting llama-server..."
"$LLAMA_SERVER" \
  -m "$MODEL_PATH" \
  --mmproj "$MMPROJ_PATH" \
  --port 8080 \
  --host 127.0.0.1 \
  --ctx-size 2048 \
  -lv 0 &          # -lv 0 = silent, run in background

LLAMA_PID=$!

echo "[Visupath] llama-server PID: $LLAMA_PID"

# Wait for llama-server to be ready (polls /health endpoint)
echo "[VisuPath] Waiting for llama-server to load model..."

until curl -sf http://127.0.0.1:8080/health > /dev/null 2>&1; do
  sleep 1
done
echo "[Visupath] llama-server ready."

echo "[Visupath] Starting VLM worker..."
python "$SCRIPT_DIR/vlm_worker.py" &
VLM_PID=$!

echo "[VisuPath] Starting TTS client..."
python "$SCRIPT_DIR/tts_client.py" &
TTS_PID=$!

echo "[VisuPath] Starting WebSocket server..."
node "$SCRIPT_DIR/server.js" &
NODE_PID=$!

echo ""
echo "=== Visupath running ==="
echo "  vlm_worker   : PID $VLM_PID"
echo "  tts_client   : PID $TTS_PID"
echo "  server.js    : PID $NODE_PID"
echo ""
echo "Press Ctrl+C to stop all."

trap "echo '[Visupath] Shutting down...'; kill $LLAMA_PID $VLM_PID $TTS_PID $NODE_PID 2>/dev/null" EXIT INT TERM
wait