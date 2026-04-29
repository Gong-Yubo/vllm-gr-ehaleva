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
HEAD_SIZES = [128, 192, 256]
BLOCK_SIZES = [16]
DTYPES = [torch.float16, torch.bfloat16]
CASES = [
    # Case 1. A general case.
    (8, 128, 8),
    (4, 64, 4),
    # Case 2. A general case + only suffix.
    (4, 16, 4),
    # Case 3. Flash-decoding case.
    (1, 32, 8),
    # Case 4. Flash-decoding case + only suffix.
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

    # Filter for actual sharing
    shared_groups = [g for g in groups if g[0] > 0]

    # Construct metadata from shared_groups
    # 1. cu_prefix_query_lens
    group_q_lens = [len(g[1]) * inputs.q_seq_len for g in shared_groups]
    cu_prefix_query_lens = torch.tensor(
        [0] + group_q_lens, dtype=torch.int32, device=inputs.query.device
    ).cumsum(0, dtype=torch.int32)

    # 2. prefix_kv_lens
    prefix_kv_lens = torch.tensor(
        [g[0] * inputs.block_size for g in shared_groups],
        dtype=torch.int32,
        device=inputs.query.device,
    )

    # 3. prefix_block_table
    if shared_groups:
        first_reqs = [g[1][0] for g in shared_groups]
        prefix_block_table = inputs.block_tables[first_reqs]
    else:
        prefix_block_table = torch.empty((0, 0), dtype=torch.int32, device=inputs.query.device)

    # 4. Suffix metadata
    shift_amounts = torch.zeros(inputs.num_seqs, dtype=torch.int32, device=inputs.query.device)
    for depth, req_ids in shared_groups:
        shift_amounts[req_ids] = depth

    max_blocks = inputs.block_tables.shape[1]
    col_indices = torch.arange(max_blocks, device=inputs.query.device, dtype=torch.int32).unsqueeze(
        0
    ) + shift_amounts.unsqueeze(1)
    col_indices = torch.clamp(col_indices, max=max_blocks - 1)
    suffix_block_table = torch.gather(inputs.block_tables, 1, col_indices)

    suffix_kv_lens = torch.clamp(inputs.seq_lens - (shift_amounts * inputs.block_size), min=0)

    cu_suffix_query_lens = torch.arange(
        0,
        (inputs.num_seqs + 1) * inputs.q_seq_len,
        inputs.q_seq_len,
        dtype=torch.int32,
        device=inputs.query.device,
    )

    # Max lengths
    prefix_query_max_len = int(max(group_q_lens)) if group_q_lens else 0
    prefix_kv_max_len = int(prefix_kv_lens.max().item()) if prefix_kv_lens.numel() > 0 else 0
    suffix_query_max_len = inputs.q_seq_len
    suffix_kv_max_len = int(suffix_kv_lens.max().item()) if suffix_kv_lens.numel() > 0 else 0

    # Prefix indices generation
    flat_req_ids_list = [req_id for _, req_ids in shared_groups for req_id in req_ids]
    prefix_indices_is_identity = len(flat_req_ids_list) == inputs.num_seqs

    prefix_indices = None
    if not prefix_indices_is_identity and flat_req_ids_list:
        flat_req_ids = torch.tensor(flat_req_ids_list, device=inputs.query.device, dtype=torch.long)
        # In this test, q_seq_len is constant per request
        starts = flat_req_ids * inputs.q_seq_len
        lengths = torch.full_like(starts, inputs.q_seq_len)

        total_prefix_tokens = lengths.sum().item()
        arrange_tensor = torch.arange(
            total_prefix_tokens, device=inputs.query.device, dtype=torch.long
        )
        cumsum_lengths = torch.zeros(
            lengths.numel() + 1, dtype=torch.long, device=inputs.query.device
        )
        torch.cumsum(lengths, dim=0, out=cumsum_lengths[1:])
        offsets = starts.repeat_interleave(lengths) - cumsum_lengths[:-1].repeat_interleave(lengths)
        prefix_indices = arrange_tensor + offsets

    def run_kernel() -> None:
        BeamAttentionImpl.cascade_attention(
            output=output,
            query=inputs.query,
            key_cache=inputs.key_cache,
            value_cache=inputs.value_cache,
            cu_prefix_query_lens=cu_prefix_query_lens,
            cu_suffix_query_lens=cu_suffix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_indices=prefix_indices,
            prefix_indices_is_identity=prefix_indices_is_identity,
            softmax_scale=inputs.scale,
            sliding_window=(-1, -1),
            logits_soft_cap=inputs.soft_cap,
            max_num_splits=0,
            fa_version=inputs.fa_version,
            block_table=inputs.block_tables,
            prefix_block_table=prefix_block_table,
            suffix_block_table=suffix_block_table,
            prefix_query_max_len=prefix_query_max_len,
            suffix_query_max_len=suffix_query_max_len,
            prefix_kv_max_len=prefix_kv_max_len,
            suffix_kv_max_len=suffix_kv_max_len,
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
