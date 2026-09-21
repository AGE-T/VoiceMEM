#!/usr/bin/env bash
# =============================================================================
# experimental/llama_b11073/start_experimental_8081.sh
#
# ISOLATED EXPERIMENTAL llama.cpp runtime launcher (sandbox / Linux side).
#
# Purpose: run llama.cpp llama-server **b11073** — the operator-tested newer
# build — next to the UNTOUCHED production runtime, on a SEPARATE executable
# path and a SEPARATE port (default 127.0.0.1:8081), so the VoiceMem
# application can be pointed at it with the EXISTING environment-only
# override:
#
#     LLAMA_SERVER_HOST=127.0.0.1 LLAMA_SERVER_PORT=8081
#
# (app/config.py apply_env, lines 502-505 — no application file changes.)
#
# EXPERIMENTAL PROFILE (the operator-tested baseline, first candidate —
# NOT assumed optimal):
#     ngl 99 | ctx 16000 | parallel 1 | threads = physical cores | reasoning off
#
# PRODUCTION IS NOT TOUCHED: the pinned b10717 binary, config/llm_config.yaml
# and scripts/start_llama_server.ps1 (port 8080) are never read or modified
# by this launcher. Reverting the experiment = stop this process + unset the
# two env variables.
#
# Sandbox adaptation (documented honestly): this machine has no GPU and
# (currently) 2 CPU cores, so -ngl 99 is a no-op on the CPU-only build and
# threads default to the physical core count instead of the target machine's
# 12. On the operator's Windows target use
# experimental/llama_b11073/start_llama_server_experimental.ps1, which
# carries the exact operator baseline (ngl 99, ctx 16000, threads 12).
#
# Every value is overridable through LLAMA_EXPERIMENTAL_* environment
# variables; nothing is hardcoded anywhere else.
# =============================================================================
set -euo pipefail

BIN_DIR="${LLAMA_EXPERIMENTAL_BIN:-/tmp/llama/llama-b11073}"
MODEL="${LLAMA_EXPERIMENTAL_MODEL:-/tmp/llama/qwen3-1.7b-q4km.gguf}"
HOST_BIND="${LLAMA_EXPERIMENTAL_HOST:-127.0.0.1}"
PORT="${LLAMA_EXPERIMENTAL_PORT:-8081}"
NGL="${LLAMA_EXPERIMENTAL_NGL:-99}"
CTX="${LLAMA_EXPERIMENTAL_CTX:-16000}"
THREADS="${LLAMA_EXPERIMENTAL_THREADS:-$(nproc)}"
LOG="${LLAMA_EXPERIMENTAL_LOG:-/tmp/llama/experimental-b11073.log}"

for f in "${BIN_DIR}/llama-server" "${MODEL}"; do
  if [[ ! -e "$f" ]]; then
    echo "ERROR: experimental runtime asset missing: $f" >&2
    echo "  (fetch: llama.cpp release b11073 'llama-b11073-bin-ubuntu-x64.tar.gz'" >&2
    echo "   + the stand-in GGUF; see experimental/llama_b11073/README.md)" >&2
    exit 2
  fi
done

mkdir -p "$(dirname "${LOG}")"
echo "[experimental] llama-server $(LD_LIBRARY_PATH="${BIN_DIR}" "${BIN_DIR}/llama-server" --version 2>&1 | head -1)"
echo "[experimental] model : ${MODEL}"
echo "[experimental] bind  : ${HOST_BIND}:${PORT}  (ngl ${NGL}, ctx ${CTX}, threads ${THREADS}, parallel 1, reasoning off)"
echo "[experimental] log   : ${LOG}"

# Foreground exec: the caller (benchmark harness / operator) waits for
# GET /health -> 200 before use, exactly like the production starter.
exec env LD_LIBRARY_PATH="${BIN_DIR}" "${BIN_DIR}/llama-server" \
  --model "${MODEL}" \
  --host "${HOST_BIND}" --port "${PORT}" \
  -ngl "${NGL}" -c "${CTX}" --parallel 1 -t "${THREADS}" \
  --cache-type-k q8_0 --cache-type-v q8_0 --temp 0.7 \
  --reasoning off --metrics --no-webui --verbose \
  >"${LOG}" 2>&1
