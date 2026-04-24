#!/usr/bin/env bash

# DeepSeek-V4-Pro on H200, SGLang aggregated across 2 nodes (TP=16).
# Recipe from https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4
# (Pro variant, H200 section). Model is sgl-project/DeepSeek-V4-Pro-FP8 —
# an FP8 re-packaging of the upstream FP4 checkpoint for Hopper.
#
# This script is invoked once per node via
#   srun --ntasks=$NNODES --ntasks-per-node=1 bash benchmarks/multi_node/dsv4_fp8_h200_sglang.sh
# Rank 0 runs bench_serving once the server is ready; rank 1+ only hosts its
# sglang worker and exits when the server shuts down.

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    CONC_LIST \
    ISL \
    OSL \
    SPEC_DECODING \
    MODEL_PATH \
    DECODE_TP \
    DECODE_EP \
    DECODE_DP_ATTN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

NODE_RANK=${SLURM_PROCID:-0}
NNODES=${SLURM_NNODES:-2}

if [[ -z "$MASTER_ADDR" ]]; then
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
fi
MASTER_PORT=${MASTER_PORT:-29500}
PORT=${PORT:-8888}

echo "node_rank=$NODE_RANK nnodes=$NNODES master=$MASTER_ADDR:$MASTER_PORT host=$(hostname)"

export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0

# Low-latency (DP-attn=false): pure TP=16, EAGLE 3/1/4.
# Throughput  (DP-attn=true):  TP=16 + DP attention + deepep a2a, MTP 1/1/2.
SPEC_ARGS=""
if [[ "$SPEC_DECODING" == "mtp" ]]; then
    if [[ "$DECODE_DP_ATTN" == "true" ]]; then
        SPEC_ARGS="--speculative-algo EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2"
    else
        SPEC_ARGS="--speculative-algo EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
    fi
fi

DP_ARGS=""
if [[ "$DECODE_DP_ATTN" == "true" ]]; then
    DP_ARGS="--enable-dp-attention --moe-a2a-backend deepep"
fi

EVAL_CONTEXT_ARGS=""
if [[ "${EVAL_ONLY}" = "true" ]]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi

SERVER_LOG="$PWD/server_rank${NODE_RANK}.log"

if [[ "$NODE_RANK" == "0" ]]; then
    start_gpu_monitor --output "$PWD/gpu_metrics.csv"
fi

set -x
PYTHONNOUSERSITE=1 \
python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --trust-remote-code \
    --tp "$DECODE_TP" \
    --ep "$DECODE_EP" \
    --nnodes "$NNODES" \
    --node-rank "$NODE_RANK" \
    --dist-init-addr "${MASTER_ADDR}:${MASTER_PORT}" \
    --host 0.0.0.0 --port "$PORT" \
    --mem-fraction-static 0.82 \
    $DP_ARGS \
    $SPEC_ARGS \
    $EVAL_CONTEXT_ARGS \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
set +x

if [[ "$NODE_RANK" != "0" ]]; then
    # Worker nodes: sit on the server and exit when it does.
    wait "$SERVER_PID"
    exit 0
fi

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

for CONC in $CONC_LIST; do
    set -x
    run_benchmark_serving \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --backend vllm \
        --input-len "$ISL" \
        --output-len "$OSL" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts $((CONC * 10)) \
        --max-concurrency "$CONC" \
        --result-filename "${RESULT_FILENAME%.json}_conc${CONC}.json" \
        --result-dir "$PWD/"
    set +x
    curl -s "http://127.0.0.1:$PORT/flush_cache" >/dev/null || true
done

if [[ "${RUN_EVAL}" = "true" ]]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
