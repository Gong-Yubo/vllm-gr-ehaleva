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
BLOCK_SIZES = [16]
DTYPES = [torch.float16, torch.bfloat16]
CASES = [
    # Case 1. A general case.
    #Q_SEQ, KV_SEQ, Suffix
    (8, 118, 8),
    (4, 64, 4),
    # Case 2. A general case + only suffix.
    (4, 16, 4),
    # # Case 3. Flash-decoding case.
    (1, 32, 8),
    # # Case 4. Flash-decoding case + only suffix.
    (1, 16, 4),
]


@dataclass
class BeamAttnInputs:
    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    scale: float
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
    suffix_blocks: int
    shared_blocks: int


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
    suffix_blocks = (suffix_kv_len + block_size - 1) // block_size
    num_blocks_per_seq = (kv_seq_len + block_size - 1) // block_size
    if suffix_blocks > num_blocks_per_seq:
        raise ValueError(
            f"suffix_blocks ({suffix_blocks}) cannot be larger than num_blocks_per_seq ({num_blocks_per_seq})"
        )

    shared_blocks = num_blocks_per_seq - suffix_blocks
    if shared_blocks < 0:
        raise ValueError(f"shared_blocks ({shared_blocks}) must be bigger than 0")

    # Query: [num_seqs * q_seq_len, num_heads, head_size]
    query = (
        torch.randn(num_seqs * q_seq_len, num_heads, head_size, dtype=dtype, device=device) * scale
    )

    # KV Cache
    # Shared blocks: batch_size * shared_blocks
    # Unique blocks: num_seqs * suffix_blocks
    total_blocks = batch_size * shared_blocks + num_seqs * suffix_blocks

    key_cache = (
        torch.randn(total_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device)
        * scale
    )
    value_cache = (
        torch.randn(total_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device)
        * scale
    )

    # Block Tables
    # [num_seqs, max_num_blocks_per_seq]
    block_tables = torch.zeros((num_seqs, num_blocks_per_seq), dtype=torch.int32, device=device)

    current_block = 0
    for i in range(batch_size):
        if shared_blocks > 0:
            shared_indices = torch.arange(
                current_block, current_block + shared_blocks, dtype=torch.int32, device=device
            )
            current_block += shared_blocks
        else:
            shared_indices = torch.tensor([], dtype=torch.int32, device=device)

        for j in range(beam_width):
            seq_idx = i * beam_width + j

            if suffix_blocks > 0:
                unique_indices = torch.arange(
                    current_block, current_block + suffix_blocks, dtype=torch.int32, device=device
                )
                current_block += suffix_blocks
            else:
                unique_indices = torch.tensor([], dtype=torch.int32, device=device)

            if shared_blocks > 0:
                block_tables[seq_idx, :shared_blocks] = shared_indices
            if suffix_blocks > 0:
                block_tables[seq_idx, shared_blocks:] = unique_indices

    seq_lens = torch.full((num_seqs,), kv_seq_len, dtype=torch.int32, device=device)

    logits_soft_cap = soft_cap if soft_cap is not None else 0
    return BeamAttnInputs(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        scale=scale,
        num_seqs=num_seqs,
        max_seq_len=kv_seq_len,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        batch_size=batch_size,
        beam_width=beam_width,
        q_seq_len=q_seq_len,
        head_size=head_size,
        soft_cap=logits_soft_cap,
        fa_version=fa_version,
        suffix_blocks=suffix_blocks,
        shared_blocks=shared_blocks,
    )


def run_reference_attention(inputs: BeamAttnInputs) -> torch.Tensor:
    output = torch.empty_like(inputs.query)

    cu_seqlens_q = torch.arange(
        0,
        (inputs.num_seqs + 1) * inputs.q_seq_len,
        inputs.q_seq_len,
        dtype=torch.int32,
        device=inputs.query.device,
    )

    def run_kernel() -> None:
        flash_attn_varlen_func(
            q=inputs.query,
            k=inputs.key_cache,
            v=inputs.value_cache,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=None,
            seqused_k=inputs.seq_lens,
            max_seqlen_q=inputs.q_seq_len,
            max_seqlen_k=inputs.max_seq_len,
            softmax_scale=inputs.scale,
            causal=True,
            window_size=(-1, -1),
            block_table=inputs.block_tables,
            softcap=inputs.soft_cap,
            fa_version=inputs.fa_version,
        )

    run_kernel()
    torch.cuda.synchronize()
    return output


def get_prefix_groups(block_tables: torch.Tensor) -> list[tuple[int, list[int]]]:
    """
    Group sequences by their longest common prefix of block indices.
    """
    block_tables_cpu = block_tables.cpu().tolist()

    groups_dict: dict[int, list[tuple[int, list[int]]]] = {}
    for i, blocks in enumerate(block_tables_cpu):
        first_block = blocks[0] if blocks else -1
        if first_block not in groups_dict:
            groups_dict[first_block] = []
        groups_dict[first_block].append((i, blocks))

    res = []
    for first_block, seqs in groups_dict.items():
        if len(seqs) == 1:
            res.append((0, [seqs[0][0]]))
            continue

        min_lcp = len(seqs[0][1])
        for i in range(1, len(seqs)):
            lcp = 0
            for b1, b2 in zip(seqs[0][1], seqs[i][1]):
                if b1 == b2:
                    lcp += 1
                else:
                    break
            min_lcp = min(min_lcp, lcp)

        res.append((min_lcp, [s[0] for s in seqs]))

    # Sort groups by the first request id to maintain deterministic order
    res.sort(key=lambda x: x[1][0])
    return res


def run_cascade_beam_attention(inputs: BeamAttnInputs) -> torch.Tensor:
    output = torch.empty_like(inputs.query)

    # Dynamically determine groups from block tables
    groups = get_prefix_groups(inputs.block_tables)

    std_indices = []
    std_q_lens = []
    std_seq_lens_list = []
    std_block_tables = []
    
    mega_prefix_indices = []
    mega_prefix_q_lens = []
    mega_prefix_seq_lens_list = []
    mega_suffix_q_lens = []
    mega_suffix_seq_lens_list = []
    mega_suffix_seq_idx = []
    mega_suffix_start_tok = []
    mega_suffix_out_offset = []
    mega_prefix_block_tables = []
    
    offset = 0
    device = inputs.query.device

    from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionMetadata

    for depth, seq_ids in groups:
        if depth == 0:
            for seq_idx in seq_ids:
                q_curr = seq_idx * inputs.q_seq_len
                std_indices.extend(range(q_curr, q_curr + inputs.q_seq_len))
                std_q_lens.append(inputs.q_seq_len)
                std_seq_lens_list.append(inputs.seq_lens[seq_idx].item())
                std_block_tables.append(inputs.block_tables[seq_idx])
        else:
            prefix_len_tokens = depth * inputs.block_size
            for seq_idx in seq_ids:
                q_curr = seq_idx * inputs.q_seq_len
                b_uncached = inputs.q_seq_len
                b_cache_len = prefix_len_tokens
                b_suffix_start = prefix_len_tokens
                b_suffix_len = inputs.seq_lens[seq_idx].item() - prefix_len_tokens

                mega_prefix_indices.extend(range(q_curr, q_curr + b_uncached))
                mega_prefix_q_lens.append(b_uncached)
                mega_prefix_seq_lens_list.append(b_cache_len)

                mega_suffix_q_lens.append(b_uncached)
                mega_suffix_seq_lens_list.append(b_suffix_len)
                mega_suffix_seq_idx.append(seq_idx)
                mega_suffix_start_tok.append(b_suffix_start)
                mega_suffix_out_offset.append(offset)

                offset += b_suffix_len
                mega_prefix_block_tables.append(inputs.block_tables[seq_idx])

    if std_q_lens:
        std_indices_tensor = torch.tensor(std_indices, dtype=torch.long, device=device)
        std_seq_lens = torch.tensor(std_seq_lens_list, dtype=torch.int32, device=device)
        std_cu_seqlens_q = torch.tensor([0] + std_q_lens, dtype=torch.int32, device=device).cumsum(dim=0).to(torch.int32)
        std_max_q_len = max(std_q_lens) if std_q_lens else 0
        std_max_seq_len = max(std_seq_lens_list) if std_seq_lens_list else 0
        std_block_table = torch.stack(std_block_tables).contiguous()
    else:
        std_indices_tensor = std_seq_lens = std_cu_seqlens_q = std_block_table = None
        std_max_q_len = std_max_seq_len = 0

    if mega_prefix_q_lens:
        mega_prefix_indices_tensor = torch.tensor(mega_prefix_indices, dtype=torch.long, device=device)
        mega_prefix_seq_lens = torch.tensor(mega_prefix_seq_lens_list, dtype=torch.int32, device=device)
        mega_prefix_max_seq_len = max(mega_prefix_seq_lens_list) if mega_prefix_seq_lens_list else 0
        mega_prefix_cu_seqlens_q = torch.tensor([0] + mega_prefix_q_lens, dtype=torch.int32, device=device).cumsum(dim=0).to(torch.int32)
        mega_prefix_max_q_len = max(mega_prefix_q_lens) if mega_prefix_q_lens else 0
        mega_prefix_block_table = torch.stack(mega_prefix_block_tables).contiguous()
        
        mega_suffix_seq_lens = torch.tensor(mega_suffix_seq_lens_list, dtype=torch.int32, device=device)
        mega_suffix_max_seq_len = max(mega_suffix_seq_lens_list) if mega_suffix_seq_lens_list else 0
        
        mega_suffix_cu_seqlens_q = torch.tensor([0] + mega_suffix_q_lens, dtype=torch.int32, device=device).cumsum(dim=0).to(torch.int32)
        mega_suffix_cu_seqlens_k = torch.tensor([0] + mega_suffix_seq_lens_list, dtype=torch.int32, device=device).cumsum(dim=0).to(torch.int32)
        mega_suffix_max_q_len = max(mega_suffix_q_lens) if mega_suffix_q_lens else 0
        
        mega_suffix_seq_idx = torch.tensor(mega_suffix_seq_idx, dtype=torch.int32, device=device)
        mega_suffix_start_tok = torch.tensor(mega_suffix_start_tok, dtype=torch.int32, device=device)
        mega_suffix_out_offset = torch.tensor(mega_suffix_out_offset, dtype=torch.int32, device=device)
    else:
        mega_prefix_indices_tensor = mega_prefix_seq_lens = mega_prefix_cu_seqlens_q = mega_prefix_block_table = None
        mega_prefix_max_q_len = mega_prefix_max_seq_len = 0
        mega_suffix_cu_seqlens_q = mega_suffix_cu_seqlens_k = mega_suffix_seq_lens = None
        mega_suffix_max_q_len = mega_suffix_max_seq_len = 0
        mega_suffix_seq_idx = mega_suffix_start_tok = mega_suffix_out_offset = None

    attn_metadata = BeamAttentionMetadata(
        num_actual_tokens=inputs.num_seqs * inputs.q_seq_len,
        max_query_len=inputs.q_seq_len,
        query_start_loc=torch.arange(0, (inputs.num_seqs + 1) * inputs.q_seq_len, inputs.q_seq_len, dtype=torch.int32, device=device),
        max_seq_len=inputs.max_seq_len,
        seq_lens=inputs.seq_lens,
        block_table=inputs.block_tables,
        slot_mapping=torch.empty(0),
        causal=True,
        std_indices=std_indices_tensor,
        std_cu_seqlens_q=std_cu_seqlens_q,
        std_seq_lens=std_seq_lens,
        std_max_q_len=std_max_q_len,
        std_max_seq_len=std_max_seq_len,
        std_block_table=std_block_table,
        mega_prefix_indices=mega_prefix_indices_tensor,
        mega_prefix_cu_seqlens_q=mega_prefix_cu_seqlens_q,
        mega_prefix_seq_lens=mega_prefix_seq_lens,
        mega_prefix_max_q_len=mega_prefix_max_q_len,
        mega_prefix_max_seq_len=mega_prefix_max_seq_len,
        mega_prefix_block_table=mega_prefix_block_table,
        mega_suffix_cu_seqlens_q=mega_suffix_cu_seqlens_q,
        mega_suffix_cu_seqlens_k=mega_suffix_cu_seqlens_k,
        mega_suffix_seq_lens=mega_suffix_seq_lens,
        mega_suffix_max_q_len=mega_suffix_max_q_len,
        mega_suffix_max_seq_len=mega_suffix_max_seq_len,
        mega_suffix_seq_idx=mega_suffix_seq_idx,
        mega_suffix_start_tok=mega_suffix_start_tok,
        mega_suffix_out_offset=mega_suffix_out_offset,
    )

    def run_kernel() -> None:
        if attn_metadata.std_indices is not None:
            std_query = inputs.query[attn_metadata.std_indices]
            std_out = torch.empty_like(std_query)
            flash_attn_varlen_func(
                q=std_query, k=inputs.key_cache, v=inputs.value_cache, out=std_out,
                cu_seqlens_q=attn_metadata.std_cu_seqlens_q, max_seqlen_q=attn_metadata.std_max_q_len,
                seqused_k=attn_metadata.std_seq_lens, max_seqlen_k=attn_metadata.std_max_seq_len,
                softmax_scale=inputs.scale, causal=attn_metadata.causal, alibi_slopes=None,
                window_size=(-1,-1), block_table=attn_metadata.std_block_table,
                softcap=inputs.soft_cap, fa_version=inputs.fa_version,
            )
            output[attn_metadata.std_indices] = std_out

        BeamAttentionImpl.cascade_attention(
            output=output,
            query=inputs.query,
            key_cache=inputs.key_cache,
            value_cache=inputs.value_cache,
            attn_metadata=attn_metadata,
            sliding_window_size=(-1, -1),
            scale=inputs.scale,
            logits_soft_cap=inputs.soft_cap,
            vllm_flash_attn_version=inputs.fa_version,
            alibi_slopes=None,
        )

    run_kernel()
    torch.cuda.synchronize()
    return output


@pytest.mark.parametrize("batch_size", [1, 4])  # type: ignore
@pytest.mark.parametrize("beam_width", [16, 32])  # type: ignore
@pytest.mark.parametrize("seq_lens_and_common_prefix", CASES)  # type: ignore
@pytest.mark.parametrize("num_heads", NUM_HEADS)  # type: ignore
@pytest.mark.parametrize("head_size", HEAD_SIZES)  # type: ignore
@pytest.mark.parametrize("dtype", DTYPES)  # type: ignore
@pytest.mark.parametrize("block_size", BLOCK_SIZES)  # type: ignore
@pytest.mark.parametrize("soft_cap", [None, 50])  # type: ignore
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
    assert suffix_kv_len < kv_seq_len
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


@pytest.mark.parametrize("batch_size", [3])  # type: ignore
@pytest.mark.parametrize("beam_width", [32])  # type: ignore
@pytest.mark.parametrize("seq_lens_and_common_prefix", CASES)  # type: ignore
@pytest.mark.parametrize("num_heads", NUM_HEADS)  # type: ignore
@pytest.mark.parametrize("head_size", HEAD_SIZES)  # type: ignore
@pytest.mark.parametrize("dtype", DTYPES)  # type: ignore
@pytest.mark.parametrize("block_size", BLOCK_SIZES)  # type: ignore
@pytest.mark.parametrize("non_shared_location", ["beginning", "middle", "end"])  # type: ignore
@torch.inference_mode()  # type: ignore
def test_beam_cascade_non_identity_path(
    batch_size: int,
    beam_width: int,
    seq_lens_and_common_prefix: tuple[int, int, int],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    non_shared_location: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test.")
    device = torch.device("cuda")
    soft_cap = None
    fa_version = 2
    if not is_fa_version_supported(fa_version):
        pytest.skip(
            f"Flash attention version {fa_version} not supported due "
            f'to: "{fa_version_unsupported_reason(fa_version)}"'
        )

    set_random_seed(0)

    num_query_heads, num_kv_heads = num_heads
    q_seq_len, kv_seq_len, suffix_kv_len = seq_lens_and_common_prefix

    # Prepare inputs with all sequences in a batch sharing prefixes initially.
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
        device="cuda",
    )

    # Determine which batch to make non-shared
    if batch_size < 3 and non_shared_location == "middle":
        pytest.skip("Batch size must be at least 3 for 'middle' location.")

    if non_shared_location == "beginning":
        target_batch_idx = 0
    elif non_shared_location == "middle":
        target_batch_idx = batch_size // 2
    else:  # "end"
        target_batch_idx = batch_size - 1

    # Modify block tables to create non-shared prefixes for the target batch
    shared_blocks = inputs.shared_blocks
    num_new_blocks_needed = beam_width * shared_blocks

    # Re-allocate KV cache with more blocks
    total_blocks = inputs.key_cache.shape[0] + num_new_blocks_needed
    new_key_cache = (
        torch.randn(total_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device)
        * inputs.scale
    )
    new_value_cache = (
        torch.randn(total_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device)
        * inputs.scale
    )

    new_key_cache[: inputs.key_cache.shape[0]] = inputs.key_cache
    new_value_cache[: inputs.value_cache.shape[0]] = inputs.value_cache

    inputs.key_cache = new_key_cache
    inputs.value_cache = new_value_cache

    # Assign new blocks to the target batch's sequences
    start_new_blocks = inputs.key_cache.shape[0] - num_new_blocks_needed
    for i in range(beam_width):
        seq_idx = target_batch_idx * beam_width + i
        new_block_start = start_new_blocks + i * shared_blocks
        inputs.block_tables[seq_idx, :shared_blocks] = torch.arange(
            new_block_start, new_block_start + shared_blocks, dtype=torch.int32, device=device
        )

    ref_output = run_reference_attention(inputs)
    output = run_cascade_beam_attention(inputs)
    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)
