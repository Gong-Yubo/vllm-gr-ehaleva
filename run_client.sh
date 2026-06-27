export REC_MODEL=/workspace/ckpt/OneRec-1.7B
#export REC_MODEL=/workspace/ckpt/OneRec-8B-pro
export REC_DATA=/workspace/ckpt/data/benchmark_data
export ASCEND_RT_VISIBLE_DEVICES=7,8
export VLLM_GR_NPU_PROFILE=0

for bw in 64 128; do
  for length in 1024 2048 5120 10240; do
    for olen in 5; do
      echo "no prefill cache, input_seq=$length, BW=$bw, result:"
      python -m benchmarks.open_one_rec.one_rec_main bench serve \
        --port 9312 \
        --endpoint /v1/chat/completions \
        --backend openai-chat \
        --model $REC_MODEL \
        --dataset-name onerec \
        --dataset-path $REC_DATA \
        --task-types video \
        --num-prompts 200 \
        --output-len $olen \
        --max-concurrency 1 \
        --custom-input-len $length \
        --percentile-metrics ttft,tpot,itl,prefill,decode,e2el \
        --use-beam-search \
        --n $bw
    done
  done
done
