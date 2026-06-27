export REC_MODEL=/workspace/ckpt/OneRec-1.7B
#export REC_MODEL=/workspace/ckpt/OneRec-8B-pro
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_0/lib:$LD_LIBRARY_PATH
#export VLLM_TORCH_PROFILER_DIR=./vllm_profile
unset VLLM_TORCH_PROFILER_DIR
export REC_DATA=/workspace/ckpt/data/benchmark_data
export VLLM_CONFIGURE_LOGGING=0
export VLLM_TORCH_PROFILER_WITH_STACK=0

export ASCEND_RT_VISIBLE_DEVICES=12,13
export VLLM_GR_NPU_PROFILE=0
export VLLM_ASCEND_USE_NEW_QKV_RMSNORM_ROPE_IMPL=1

vllm serve $REC_MODEL \
  --gr \
  --port 9312 \
  --trust-remote-code \
  --max-logprobs 1024 \
  --max-num-seqs 1024 \
  --tensor-parallel-size 1 \
  --max-num-batched-tokens 16384 \
  --scheduling-policy priority \
  --attention-backend CUSTOM \
  --default-chat-template-kwargs '{"enable_thinking": false}'
