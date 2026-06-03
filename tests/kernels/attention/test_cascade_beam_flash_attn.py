# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from dataclasses import dataclass

import pytest
import torch
from vllm.utils.torch_utils import set_random_seed

try:
    from vllm_gr.v1.attention.backends.beam_attn import (
        BeamAttentionImpl,
    )
except ImportError:
    pytest.skip(
        "beam_attn backend is not supported.",
        allow_module_level=True,
    )

try:
    from vllm.vllm_flash_attn import (
        fa_version_unsupported_reason,
        flash_attn_varlen_func,
        is_fa_version_supported,
    )
except ImportError:
    pytest.skip(
        "vllm_flash_attn is not supported.",
        allow_module_level=True,
    )

NUM_HEADS = [(4, 4), (8, 2), (16, 2)]
HEAD_SIZES = [128, 256]
BLOCK_SIZES = [16, 32]
DTYPES = [torch.float16, torch.bfloat16]
CASES = [
    # Scenario: Standard Prefix Complete.
    # The shared prefix is fully cached, and beams independently decode their full suffixes.
    (8, 32, 8),
    
    # Scenario: No Shared Prefix.
    # Completely independent parallel sequence generation where shared_tokens_len = 0.
    (8, 8, 8),
    
    # Scenario: Large sequence general case with longer prefix.
    (4, 64, 4),
    
    # Scenario: General small context generation.
    (4, 16, 4),
    
    # Scenario: Chunked execution with strict budget (Partially Executed / Skipped beams).
    # Budget (1*16=16) runs out before all beams can process their 4-token suffixes.
    (1, 16, 4),
    
    # Scenario: Pure single-token decoding (Typical autoregressive decoding phase).
    # Each beam processes exactly 1 token over a large context.
    (1, 128, 1),
    
    # Scenario: Partial block-alignment anomaly.
    # Tests memory boundary safety when sequence lengths are prime/unaligned numbers.
    (4, 32, 7),
]


@dataclass
class BeamAttnInputs:
    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    scale: float
    num_reqs: int
    num_seqs: int
    max_seq_len: int
    block_size: int
    num_kv_heads: int
    batch_size: int
    beam_width: int
    q_seq_len: int
    head_size: int
    soft_cap: float
    fa_version: int
    shared_tokens_len: int
    suffix_kv_len: int


def prepare_inputs(
    batch_size: int,
    beam_width: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    q_seq_len: int,
    kv_seq_len: int,
    block_size: int,
    dtype: torch.dtype,
    soft_cap: float | None,
    fa_version: int,
    suffix_kv_len: int = 0,
    device: str = "cuda",
) -> BeamAttnInputs:
    num_seqs = batch_size * beam_width
    scale = float(1.0 / (head_size**0.5))

    shared_tokens_len = kv_seq_len - suffix_kv_len
    mega_seq_len = shared_tokens_len + beam_width * suffix_kv_len
    blocks_per_req = (mega_seq_len + block_size - 1) // block_size

    num_reqs = batch_size

    # Query: [num_seqs * q_seq_len, num_heads, head_size]
    query = (
        torch.randn(num_seqs * q_seq_len, num_heads, head_size, dtype=dtype, device=device) * scale
    )

    # KV Cache
    total_blocks = num_reqs * blocks_per_req

    key_cache = (
        torch.randn(total_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device)
        * scale
    )
    value_cache = (
        torch.randn(total_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device)
        * scale
    )

    # Block Tables
    block_tables = torch.zeros((num_reqs, blocks_per_req), dtype=torch.int32, device=device)

    current_block = 0
    for i in range(num_reqs):
        block_tables[i] = torch.arange(
            current_block, current_block + blocks_per_req, dtype=torch.int32, device=device
        )
        current_block += blocks_per_req

    seq_lens_list = []
    last_beam_suffix_start = shared_tokens_len + (beam_width - 1) * suffix_kv_len
    for i in range(num_reqs):
        seq_lens_list.append(mega_seq_len)
            
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)

    logits_soft_cap = soft_cap if soft_cap is not None else 0
    return BeamAttnInputs(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        scale=scale,
        num_reqs=num_reqs,
        num_seqs=num_seqs,
        max_seq_len=mega_seq_len,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        batch_size=batch_size,
        beam_width=beam_width,
        q_seq_len=q_seq_len,
        head_size=head_size,
        soft_cap=logits_soft_cap,
        fa_version=fa_version,
        shared_tokens_len=shared_tokens_len,
        suffix_kv_len=suffix_kv_len,
    )


def run_reference_attention(inputs: BeamAttnInputs) -> torch.Tensor:
    dense_k = []
    dense_v = []
    seqlens_k = []
    seqlens_q = []

    for i in range(inputs.batch_size):
        # Retrieve the flat layout of the key and value cache for this specific request
        phys_blocks = inputs.block_tables[i].cpu().tolist()
        k_seq = torch.cat([inputs.key_cache[b] for b in phys_blocks], dim=0)[:inputs.seq_lens[i]]
        v_seq = torch.cat([inputs.value_cache[b] for b in phys_blocks], dim=0)[:inputs.seq_lens[i]]

        prefix_len = inputs.shared_tokens_len
        steps = inputs.suffix_kv_len
        w = inputs.beam_width
        q_len = inputs.q_seq_len * inputs.beam_width
        
        cache_len = prefix_len
        
        # chunk_budget enforces the maximum number of new query tokens allowed in this iteration.
        chunk_budget = q_len
        # delta represents how many suffix tokens have already been cached across all beams
        delta = max(0, cache_len - prefix_len)

        for b in range(w):
            # Stop processing if we run out of token budget for this chunk
            if chunk_budget <= 0:
                break
                
            b_uncached = 0
            b_cache_len = 0
            b_suffix_start = 0
            b_suffix_len = 0
            
            if prefix_len >= cache_len:
                # Scenario A: Prefix Incomplete
                remaining_prefix = prefix_len - cache_len
                if b == 0:
                    # Leader Beam computes the remaining shared prefix on behalf of all beams,
                    # plus its own unique suffix tokens if there is leftover chunk budget.
                    b_uncached = min(chunk_budget, remaining_prefix + steps)
                    b_cache_len = cache_len
                    b_suffix_start = cache_len
                    b_suffix_len = b_uncached
                else:
                    # Follower Beams (1 to W-1) cannot compute shared prefix tokens.
                    # They must wait until the prefix is fully resolved. They are only allowed 
                    # to compute their unique suffixes if the prefix finishes within this chunk.
                    if remaining_prefix <= q_len:
                        b_uncached = min(chunk_budget, steps)
                        b_cache_len = prefix_len
                        b_suffix_start = prefix_len + b * steps
                        b_suffix_len = b_uncached
            else:
                # Scenario B: Prefix Complete
                past_suffix_b = max(0, min(steps, delta - b * steps))
                remaining_suffix_b = steps - past_suffix_b
                b_uncached = min(chunk_budget, remaining_suffix_b)
                if b_uncached > 0:
                    # Attend to the full shared prefix safely located in the cache.
                    b_cache_len = prefix_len
                    # Causal self-attention starts exactly where this beam's unique suffix begins.
                    b_suffix_start = prefix_len + b * steps
                    # Total suffix length includes past cached tokens + newly computed ones.
                    b_suffix_len = past_suffix_b + b_uncached
            
            # If there's no uncached tokens to compute for this beam, skip it
            if b_uncached <= 0:
                continue
                
            # Extract the shared prefix slice
            prefix_k = k_seq[:b_cache_len]
            prefix_v = v_seq[:b_cache_len]
            
            # Extract the unique suffix slice for the current beam
            suffix_k = k_seq[b_suffix_start:b_suffix_start + b_suffix_len]
            suffix_v = v_seq[b_suffix_start:b_suffix_start + b_suffix_len]

            # Concatenate prefix and suffix to form the contiguous beam key and value tensor
            beam_k = torch.cat([prefix_k, suffix_k], dim=0)
            beam_v = torch.cat([prefix_v, suffix_v], dim=0)

            dense_k.append(beam_k)
            dense_v.append(beam_v)
            seqlens_k.append(beam_k.shape[0])
            seqlens_q.append(b_uncached)
            
            # Deduct the processed tokens from the chunk budget
            chunk_budget -= b_uncached

    k_tensor = torch.cat(dense_k, dim=0)
    v_tensor = torch.cat(dense_v, dim=0)

    cu_seqlens_k = torch.tensor([0] + seqlens_k, dtype=torch.int32, device=inputs.query.device).cumsum(dim=0).to(torch.int32)
    cu_seqlens_q = torch.tensor([0] + seqlens_q, dtype=torch.int32, device=inputs.query.device).cumsum(dim=0).to(torch.int32)

    output = torch.zeros_like(inputs.query)

    def run_kernel() -> None:
        flash_attn_varlen_func(
            q=inputs.query[:cu_seqlens_q[-1]],
            k=k_tensor,
            v=v_tensor,
            out=output[:cu_seqlens_q[-1]],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max(seqlens_q),
            max_seqlen_k=max(seqlens_k),
            softmax_scale=inputs.scale,
            causal=True,
            window_size=(-1, -1),
            softcap=inputs.soft_cap,
            fa_version=inputs.fa_version,
        )

    run_kernel()
    torch.cuda.synchronize()
    return output


def run_cascade_beam_attention(inputs: BeamAttnInputs, shuffle_processing_order: bool = False) -> torch.Tensor:
    output = torch.zeros_like(inputs.query)
    device = inputs.query.device

    from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionMetadataBuilder, BeamAttentionImpl, MEGA_DATA_VAR
    
    class MockCommonAttentionMetadata:
        def __init__(self, inputs: BeamAttnInputs):
            self.num_reqs = inputs.num_reqs
            
            # If shuffle_processing_order is True, we can simulate out-of-order execution 
            # by reversing the query grouping lengths. But since we need them to match the
            # input shapes exactly, we just keep the sequential layout for queries.
            
            self.query_start_loc_cpu = torch.arange(0, (inputs.num_seqs + 1) * inputs.q_seq_len, inputs.beam_width * inputs.q_seq_len, dtype=torch.int32)
            self.query_start_loc = self.query_start_loc_cpu.to(device)
            self.seq_lens = inputs.seq_lens
            self.block_table_tensor = inputs.block_tables
            self.num_actual_tokens = inputs.num_seqs * inputs.q_seq_len
            self.max_query_len = inputs.q_seq_len * inputs.beam_width
            self.max_seq_len = inputs.max_seq_len
            self.slot_mapping = torch.empty(0)
            self.causal = True

    common_attn_metadata = MockCommonAttentionMetadata(inputs)
    
    mega_info = {
        "mega_data": {},
        "req_ids": [f"req_{i}" for i in range(inputs.num_reqs)]
    }
    
    for i in range(inputs.num_reqs):
        mega_info["mega_data"][f"req_{i}"] = {
            "is_mega_decode": True,
            "prefix_len": inputs.shared_tokens_len,
            "mega_beam_width": inputs.beam_width,
            "mega_decode_steps": inputs.suffix_kv_len,
            "cache_len": inputs.shared_tokens_len,
        }
        
    token = MEGA_DATA_VAR.set(mega_info)

    try:
        builder = BeamAttentionMetadataBuilder.__new__(BeamAttentionMetadataBuilder)
        builder.device = device
        builder.block_size = inputs.block_size
        
        attn_metadata = builder.build(
            common_prefix_len=inputs.shared_tokens_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=False,
        )
    finally:
        MEGA_DATA_VAR.reset(token)

    impl = BeamAttentionImpl(
        num_heads=inputs.query.shape[1],
        head_size=inputs.head_size,
        scale=inputs.scale,
        num_kv_heads=inputs.num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype='auto',
        logits_soft_cap=inputs.soft_cap,
    )

    impl.vllm_flash_attn_version = inputs.fa_version

    def run_kernel() -> None:
        BeamAttentionImpl.cascade_attention(
            output=output,
            query=inputs.query,
            key_cache=inputs.key_cache,
            value_cache=inputs.value_cache,
            attn_metadata=attn_metadata,
            sliding_window_size=impl.sliding_window,
            scale=impl.scale,
            logits_soft_cap=impl.logits_soft_cap,
            vllm_flash_attn_version=impl.vllm_flash_attn_version,
            alibi_slopes=impl.alibi_slopes,
        )

    run_kernel()
    torch.cuda.synchronize()
    return output


@pytest.mark.parametrize("batch_size", [8])  # type: ignore
@pytest.mark.parametrize("beam_width", [16, 32])  # type: ignore
@pytest.mark.parametrize("seq_lens_and_common_prefix", CASES)  # type: ignore
@pytest.mark.parametrize("num_heads", NUM_HEADS)  # type: ignore
@pytest.mark.parametrize("head_size", HEAD_SIZES)  # type: ignore
@pytest.mark.parametrize("dtype", DTYPES)  # type: ignore
@pytest.mark.parametrize("block_size", BLOCK_SIZES)  # type: ignore
@pytest.mark.parametrize("fa_version", [2, 3])  # type: ignore
@torch.inference_mode()  # type: ignore
def test_beam_cascade(
    batch_size: int,
    beam_width: int,
    seq_lens_and_common_prefix: tuple[int, int, int],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float,
    fa_version: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test.")
    device = torch.device("cuda")

    if not is_fa_version_supported(fa_version):
        pytest.skip(
            f"Flash attention version {fa_version} not supported due "
            f'to: "{fa_version_unsupported_reason(fa_version)}"'
        )

    set_random_seed(0)

    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    assert num_kv_heads != 0
    q_seq_len = seq_lens_and_common_prefix[0]
    kv_seq_len = seq_lens_and_common_prefix[1]
    suffix_kv_len = seq_lens_and_common_prefix[2]
    assert suffix_kv_len > 0

    inputs = prepare_inputs(
        batch_size=batch_size,
        beam_width=beam_width,
        num_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        block_size=block_size,
        dtype=dtype,
        soft_cap=soft_cap,
        fa_version=fa_version,
        suffix_kv_len=suffix_kv_len,
        device=device,
    )

    # Run the regular attention.
    ref_output = run_reference_attention(inputs)
    # Run beam attention
    output = run_cascade_beam_attention(inputs)
    # Compare the results.
    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("batch_size", [8])  # type: ignore
@pytest.mark.parametrize("beam_width", [16, 32, 128])  # type: ignore
@pytest.mark.parametrize("seq_lens_and_common_prefix", [(8, 2048, 8)])  # type: ignore
@pytest.mark.parametrize("num_heads", NUM_HEADS)  # type: ignore
@pytest.mark.parametrize("head_size", HEAD_SIZES)  # type: ignore
@pytest.mark.parametrize("dtype", DTYPES)  # type: ignore
@pytest.mark.parametrize("block_size", BLOCK_SIZES)  # type: ignore
@pytest.mark.parametrize("soft_cap", [None, 50])  # type: ignore
@pytest.mark.parametrize("fa_version", [2, 3])  # type: ignore
@pytest.mark.parametrize("iter_num", [100])  # type: ignore
@torch.inference_mode()  # type: ignore
def test_beam_cascade_perf(
    batch_size: int,
    beam_width: int,
    seq_lens_and_common_prefix: tuple[int, int, int],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float,
    fa_version: int,
    iter_num: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test.")
    device = torch.device("cuda")

    if not is_fa_version_supported(fa_version):
        pytest.skip(
            f"Flash attention version {fa_version} not supported due "
            f'to: "{fa_version_unsupported_reason(fa_version)}"'
        )

    set_random_seed(0)

    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    assert num_kv_heads != 0
    q_seq_len = seq_lens_and_common_prefix[0]
    kv_seq_len = seq_lens_and_common_prefix[1]
    suffix_kv_len = seq_lens_and_common_prefix[2]
    assert suffix_kv_len > 0

    inputs = prepare_inputs(
        batch_size=batch_size,
        beam_width=beam_width,
        num_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        block_size=block_size,
        dtype=dtype,
        soft_cap=soft_cap,
        fa_version=fa_version,
        suffix_kv_len=suffix_kv_len,
        device=device,
    )

    # Warmup
    for _ in range(5):
        _ = run_cascade_beam_attention(inputs)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iter_num):
        _ = run_cascade_beam_attention(inputs)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time_ms = start_event.elapsed_time(end_event) / iter_num
    print(f"\nPerf: batch_size={batch_size}, beam_width={beam_width}, "
          f"seq_lens={seq_lens_and_common_prefix}, num_heads={num_heads}, "
          f"head_size={head_size}, dtype={dtype}, block_size={block_size}, "
          f"avg_time={elapsed_time_ms:.3f} ms")
