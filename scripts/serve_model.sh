#!/bin/bash
# Serve one decision model with vllm-jev and run a script against it.
# usage: serve_model.sh <gpu> <port> <hf_model_id> <script.py> [args...]
# Env: VLLM_JEV_VENV (venv with the vllm-jev server), VLLM_JEV_WORKSPACE (server workspace).
GPU=$1; PORT=$2; HF=$3; SCRIPT=$4; shift 4
VJV=${VLLM_JEV_VENV:?set VLLM_JEV_VENV to the venv containing vllm-jev}
WS=${VLLM_JEV_WORKSPACE:?set VLLM_JEV_WORKSPACE to the vllm-jev workspace dir}
export CASCADE_URL=http://127.0.0.1:$PORT
export BATTERY_URL=http://127.0.0.1:$PORT
CUDA_VISIBLE_DEVICES=$GPU "$VJV/vllm-jev" serve "$HF" --workspace "$WS" --port "$PORT" \
  --served-model-name vllm-jev > server.log 2>&1 &
SRV=$!
OK=0
for i in $(seq 1 45); do
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && OK=1 && break
  sleep 10
done
if [ "$OK" = 1 ]; then
  "$VJV/python" "$SCRIPT" "$@"
else
  echo "[server failed] $HF"
fi
kill "$SRV" 2>/dev/null
