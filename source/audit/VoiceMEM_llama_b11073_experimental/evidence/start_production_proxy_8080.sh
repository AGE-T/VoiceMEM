#!/usr/bin/env bash
# =============================================================================
# audit/VoiceMEM_llama_b11073_experimental/evidence/start_production_proxy_8080.sh
#
# The A-SIDE of the A/B validation: llama.cpp b10717 (the PINNED production
# build) with the standing sandbox production-proxy flags — exactly what
# mini-services/llama-server/index.js has run for every prior live proof
# (v0.4.21 no-thinking proof, Hungarian benchmark, S8 forensic capture):
#
#   -ngl 0 -c 8192 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0
#   --temp 0.7 --metrics --no-webui --reasoning off --verbose
#
# (CPU-only sandbox; the 19 GB production model is never sandbox-resident —
# the same Qwen3-family hybrid-thinking stand-in is used on BOTH sides.)
#
# This launcher exists ONLY in the audit evidence directory: it documents
# how side A was run for reproducibility. It is NOT part of the product.
# =============================================================================
set -euo pipefail

BIN_DIR="${LLAMA_A_BIN:-/tmp/llama/llama-b10717}"
MODEL="${LLAMA_A_MODEL:-/tmp/llama/qwen3-1.7b-q4km.gguf}"
HOST_BIND="${LLAMA_A_HOST:-127.0.0.1}"
PORT="${LLAMA_A_PORT:-8080}"
CTX="${LLAMA_A_CTX:-8192}"
THREADS="${LLAMA_A_THREADS:-$(nproc)}"
LOG="${LLAMA_A_LOG:-/tmp/llama/production-proxy-b10717.log}"

for f in "${BIN_DIR}/llama-server" "${MODEL}"; do
  [[ -e "$f" ]] || { echo "ERROR: missing asset: $f" >&2; exit 2; }
done

echo "[A-side proxy] b10717 @ ${HOST_BIND}:${PORT} (ngl 0, ctx ${CTX}, threads ${THREADS}, reasoning off)"
exec env LD_LIBRARY_PATH="${BIN_DIR}" "${BIN_DIR}/llama-server" \
  --model "${MODEL}" \
  --host "${HOST_BIND}" --port "${PORT}" \
  -ngl 0 -c "${CTX}" --parallel 1 -t "${THREADS}" \
  --cache-type-k q8_0 --cache-type-v q8_0 --temp 0.7 \
  --reasoning off --metrics --no-webui --verbose \
  >"${LOG}" 2>&1
