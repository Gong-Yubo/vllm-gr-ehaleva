#!/bin/bash

validate_metric() {
    local metric_name="$1"
    local metric_value="$2"
    local threshold="$3"
    local comparison_op="$4" # "le" or "ge"

    local awk_script
    if [[ "$comparison_op" == "ge" ]]; then
        awk_script='BEGIN {exit !(val >= th)}'
        condition_text="greater than or equal to"
    else
        awk_script='BEGIN {exit !(val <= th)}'
        condition_text="less than or equal to"
    fi

    if awk -v val="$metric_value" -v th="$threshold" "$awk_script"; then
        echo "SUCCESS: ${metric_name} (${metric_value}) is ${condition_text} ${threshold}."
    else
        echo "FAILURE: ${metric_name} (${metric_value}) is not ${condition_text} ${threshold}."
        exit 1
    fi
}

run_port=7000
timeout=600 # time for waiting server to go up
p99_ttft_th=600
p99_tpot_th=0.01
pass_1_th=0.09
pass_32_th=0.2
recall_32_th=0.04
# Ensure we clean up the background server process if the script exits early
trap 'if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null || true; sleep 2; pkill -P "$SERVER_PID" 2>/dev/null || true; fi' EXIT

vllm serve --gr \
    --port $run_port \
    --seed 123 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-logprobs 512 \
    --max_num_seqs 2048 \
    --max_num_batched_tokens 16384 \
    --scheduling-policy priority \
    --catalog-path "$CATALOG_PATH" \
    --attention-backend CUSTOM \
    OpenOneRec/OneRec-1.7B &

SERVER_PID=$!

echo "Waiting for server to be ready on port $run_port..."
# Check health/models endpoint until it responds
if ! timeout $timeout bash -c "
  until curl -s http://localhost:${run_port}/v1/models > /dev/null; do
    sleep 2
  done
"; then
  echo "Server failed to start within the timeout limit of $timeout seconds."
  exit 1
fi
echo "Server is ready! Running benchmark..."

# warmup
output=$(python -m benchmarks.open_one_rec.one_rec_main \
  bench serve \
  --endpoint /v1/chat/completions \
  --backend openai-chat  \
  --model OpenOneRec/OneRec-1.7B  \
  --dataset-name onerec \
  --dataset-path "$DATASET_PATH" \
  --task-type video  \
  --num-prompts 5 \
  --use-beam-search \
  --port $run_port \
  --request-rate 1 \
  --n 128)


output=$(python -m benchmarks.open_one_rec.one_rec_main \
  bench serve \
  --endpoint /v1/chat/completions \
  --backend openai-chat  \
  --model OpenOneRec/OneRec-1.7B  \
  --dataset-name onerec \
  --dataset-path "$DATASET_PATH" \
  --task-type video  \
  --num-prompts 10 \
  --use-beam-search \
  --port $run_port \
  --request-rate 0.5 \
  --n 128)



echo "$output"


# Extract P99 TTFT
p99_ttft=$(echo "$output" | grep "P99 TTFT (ms):" | awk '{print $4}')
p99_tpot=$(echo "$output" | grep "P99 TPOT (ms):" | awk '{print $4}')

if [ -z "$p99_ttft" ]; then
    echo "Error: Could not extract 'P99 TTFT (ms)' from the output."
    exit 1
fi

echo "Parsed P99 TTFT: ${p99_ttft} ms"

# Validate P99 ttft (Using awk to safely evaluate floating point comparison)
validate_metric "P99 TTFT" "$p99_ttft" "$p99_ttft_th" "le"

if [ -z "$p99_tpot" ]; then
    echo "Error: Could not extract 'P99 TPOT (ms)' from the output."
    exit 1
fi

echo "Parsed P99 TPOT: ${p99_tpot} ms"

# Validate P99 tpot (Using awk to safely evaluate floating point comparison)
validate_metric "P99 TPOT" "$p99_tpot" "$p99_tpot_th" "le"

output=$(python -m benchmarks.open_one_rec.one_rec_acc_test \
--model OpenOneRec/OneRec-1.7B \
--dataset-path "$DATASET_PATH" \
--dataset-name  onerec \
--num-prompts 100 \
--use-beam-search \
--backend openai-chat \
--task-type video \
--endpoint /v1/chat/completions \
--port $run_port \
--n 128)

echo "$output"
pass_1=$(echo "$output" | grep -E '^[[:space:]]*"pass@1":' | tail -1 | awk -F':' '{print $2}' | tr -d ' ,')
pass_32=$(echo "$output" | grep -E '^[[:space:]]*"pass@32":' | tail -1 | awk -F':' '{print $2}' | tr -d ' ,')
recall_32=$(echo "$output" | grep -E '^[[:space:]]*"recall@32":' | tail -1 | awk -F':' '{print $2}' | tr -d ' ,')

if [ -z "$pass_1" ] || [ -z "$pass_32" ] || [ -z "$recall_32" ]; then
    echo "Error: Could not extract accuracy metrics from the output."
    exit 1
fi

echo "Parsed pass@1: $pass_1"
echo "Parsed pass@32: $pass_32"
echo "Parsed recall@32: $recall_32"

# Validate pass@1
validate_metric "pass@1" "$pass_1" "$pass_1_th" "ge"

# Validate pass@32
validate_metric "pass@32" "$pass_32" "$pass_32_th" "ge"

# Validate recall@32
validate_metric "recall@32" "$recall_32" "$recall_32_th" "ge"

exit 0
