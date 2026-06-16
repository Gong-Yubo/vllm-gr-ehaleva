# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Beam-Consecutive Suffix Gather Attention Backend."""

import contextvars
import copy
from dataclasses import dataclass
from itertools import compress
from typing import ClassVar, Any

import torch
import triton
import triton.language as tl
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionType,
    MultipleOf,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_supports_fp8,
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_supports_sinks,
        flash_attn_varlen_func,
        get_scheduler_metadata,
        reshape_and_cache_flash,
    )
else:
    def _flash_attn_varlen_unavailable(*args, **kwargs):
        raise RuntimeError(
            "BeamAttentionBackend requires FlashAttention varlen support, "
        )
    flash_attn_varlen_func = _flash_attn_varlen_unavailable
    reshape_and_cache_flash = _flash_attn_varlen_unavailable
    get_scheduler_metadata = _flash_attn_varlen_unavailable

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    get_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)
MEGA_DATA_VAR = contextvars.ContextVar("mega_data", default=None)

@triton.jit
def paged_kv_to_contig_suffix_kernel(
    K_CACHE_PTR, V_CACHE_PTR, SLOT_MAPPING_PTR,
    SUFFIX_LENS_PTR, OUT_K_PTR, OUT_V_PTR, OUT_OFFSETS_PTR,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ok_tok, stride_ok_h, stride_ok_d,
    stride_ov_tok, stride_ov_h, stride_ov_d,  # Ensure this is here
    num_beams: tl.constexpr, max_suffix_len: tl.constexpr,
    n_kv_heads: tl.constexpr, head_dim: tl.constexpr,shared_gap_len: tl.constexpr
):
    pid_beam = tl.program_id(0)
    pid_tok = tl.program_id(1)
    pid_h = tl.program_id(2)

    suffix_len = tl.load(SUFFIX_LENS_PTR + pid_beam)
    if pid_tok >= suffix_len:
        return

    # LOGIC:
    # 0 to shared_gap_len-1: Load from the start of slot_mapping (Shared)
    # shared_gap_len to end: Load from beam-specific offsets
    is_shared = pid_tok < shared_gap_len
    
    slot_idx = tl.where(
        is_shared,
        tl.load(SLOT_MAPPING_PTR + pid_tok),
        tl.load(SLOT_MAPPING_PTR + shared_gap_len + (pid_beam * (suffix_len - shared_gap_len)) + (pid_tok - shared_gap_len))
    )
    
    # tl.static_print("Debg: Slot Index:", slot_idx, " pid_beam:", pid_beam, "pid_tok:", pid_tok)

    d_offsets = tl.arange(0, head_dim)
    k_ptrs = K_CACHE_PTR + slot_idx//16 * stride_kb + (slot_idx%16) * stride_kt + pid_h * stride_kh + d_offsets * stride_kd
    v_ptrs = V_CACHE_PTR + slot_idx//16 * stride_vb + (slot_idx%16) * stride_vt + pid_h * stride_vh + d_offsets * stride_vd

    k = tl.load(k_ptrs)
    v = tl.load(v_ptrs)

    out_offset = tl.load(OUT_OFFSETS_PTR + pid_beam)
    ok_ptrs = OUT_K_PTR + (out_offset + pid_tok) * stride_ok_tok + pid_h * stride_ok_h + d_offsets * stride_ok_d
    ov_ptrs = OUT_V_PTR + (out_offset + pid_tok) * stride_ov_tok + pid_h * stride_ov_h + d_offsets * stride_ov_d
    
    tl.store(ok_ptrs, k)
    tl.store(ov_ptrs, v)

# ---------------------------------------------------------------------------
# Python Launcher (Updated)
# ---------------------------------------------------------------------------
def extract_suffix_kv(
    k_cache, v_cache, slot_mapping, suffix_lens, out_offsets, 
    shared_gap_len, max_suffix_len, num_beams
):
    nheads = k_cache.shape[2]
    head_dim = k_cache.shape[3]
    total_tokens = suffix_lens.sum().item()
    
    out_k = torch.empty((total_tokens, nheads, head_dim), device=k_cache.device, dtype=k_cache.dtype)
    out_v = torch.empty((total_tokens, nheads, head_dim), device=v_cache.device, dtype=v_cache.dtype)
    
    grid = (num_beams, max_suffix_len, nheads)
    
    paged_kv_to_contig_suffix_kernel[grid](
        k_cache, v_cache, slot_mapping, suffix_lens, out_k, out_v, out_offsets,
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        out_k.stride(0), out_k.stride(1), out_k.stride(2),
        out_v.stride(0), out_v.stride(1), out_v.stride(2), # <--- THIS IS stride_ov_d
        num_beams=num_beams,
        max_suffix_len=max_suffix_len,
        n_kv_heads=nheads,
        head_dim=head_dim, shared_gap_len=shared_gap_len
    )
    return out_k, out_v

# ---------------------------------------------------------------------------
# METADATA STRUCTURE DEFINITIONS
# ---------------------------------------------------------------------------
@dataclass
class BeamAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = True

    # Standard Requests batch routing
    std_indices: torch.Tensor | None = None
    std_cu_seqlens_q: torch.Tensor | None = None
    std_seq_lens: torch.Tensor | None = None
    std_max_q_len: int = 0
    std_max_seq_len: int = 0
    std_block_table: torch.Tensor | None = None

    # Mega Requests shared context arrays
    mega_prefix_indices: torch.Tensor | None = None
    mega_prefix_cu_seqlens_q: torch.Tensor | None = None
    mega_prefix_seq_lens: torch.Tensor | None = None
    mega_prefix_max_q_len: int = 0
    mega_prefix_max_seq_len: int = 0
    mega_prefix_block_table: torch.Tensor | None = None

    # Mega Requests split-gather channels
    mega_suffix_cu_seqlens_q: torch.Tensor | None = None
    mega_suffix_cu_seqlens_k: torch.Tensor | None = None
    mega_suffix_seq_lens: torch.Tensor | None = None
    mega_suffix_max_q_len: int = 0
    mega_suffix_max_seq_len: int = 0
    mega_suffix_total_tokens: int = 0
    mega_suffix_shared_gap: int = 0
    mega_suffix_bw: int = 0

    mega_suffix_out_offset: torch.Tensor | None = None

    scheduler_metadata: dict | None = None
    prefix_scheduler_metadata: dict | None = None
    max_num_splits: int = 0


class BeamAttentionMetadataBuilder(AttentionMetadataBuilder[BeamAttentionMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    supports_update_block_table: bool = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config

        self.num_heads_q = self.model_config.get_num_attention_heads(self.parallel_config)
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size
        self.max_num_splits = 0
        self.aot_schedule = False

        self.use_full_cuda_graph = self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        self.max_cudagraph_size = self.compilation_config.max_cudagraph_capture_size

        if self.use_full_cuda_graph and self.aot_schedule:
            self.scheduler_metadata = torch.zeros(
                vllm_config.scheduler_config.max_num_seqs + 1,
                dtype=torch.int32,
                device=self.device,
            )
            if hasattr(vllm_config, "attention_config") and hasattr(vllm_config.attention_config, "flash_attn_max_num_splits_for_cuda_graph"):
                self.max_num_splits = vllm_config.attention_config.flash_attn_max_num_splits_for_cuda_graph

        self.aot_sliding_window: tuple[int, int] | None = None

    def _update_aot_schedule(self, fast_build: bool):
        self.aot_schedule = self.aot_schedule and not fast_build
        if self.aot_sliding_window is None:
            self.aot_sliding_window = (-1, -1)

    def _get_max_num_splits(self, num_actual_tokens: int) -> int:
        max_num_splits = 0
        if (
            self.use_full_cuda_graph
            and self.max_cudagraph_size is not None
            and num_actual_tokens <= self.max_cudagraph_size
        ):
            max_num_splits = self.max_num_splits

        if vllm_is_batch_invariant():
            max_num_splits = 1
        return max_num_splits

    def _get_schedule(
        self,
        batch_size: int,
        cu_query_lens: torch.Tensor,
        max_query_len: int,
        seqlens: torch.Tensor,
        max_seq_len: int,
        causal: bool,
        max_num_splits: int,
    ) -> dict | None:
        if not self.aot_schedule:
            return None

        cache_dtype = self.cache_config.cache_dtype
        if cache_dtype.startswith("fp8"):
            qkv_dtype = BeamAttentionBackend.get_fp8_dtype_for_flashattn(cache_dtype)
        else:
            qkv_dtype = self.kv_cache_dtype

        return get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_seq_len,
            num_heads_q=self.num_heads_q,
            num_heads_kv=self.num_heads_kv,
            headdim=self.headdim,
            cache_seqlens=seqlens,
            qkv_dtype=qkv_dtype,
            cu_seqlens_q=cu_query_lens,
            page_size=self.block_size,
            causal=causal,
            window_size=self.aot_sliding_window,
            num_splits=max_num_splits,
        )

    def _segment_requests(self, num_reqs: int, req_ids: list, mega_data: dict):
        std_req_idxs = []
        mega_req_idxs = []
        mega_prefix_lens = []
        mega_beam_widths = []
        mega_decode_steps = []
        mega_cache_lens = []

        for i in range(num_reqs):
            req_id = req_ids[i] if i < len(req_ids) else None
            mdata = mega_data.get(req_id, {}) if req_id else {}
            if mdata.get("is_mega_decode", False):
                mega_req_idxs.append(i)
                mega_prefix_lens.append(mdata["prefix_len"])
                mega_beam_widths.append(mdata["mega_beam_width"])
                mega_decode_steps.append(mdata["mega_decode_steps"])
                mega_cache_lens.append(mdata.get("cache_len", mdata["prefix_len"]))
            else:
                std_req_idxs.append(i)

        return std_req_idxs, mega_req_idxs, mega_prefix_lens, mega_beam_widths, mega_decode_steps, mega_cache_lens

    def _build_std_mappings(self, std_req_idxs: list, q_starts: list):
        std_indices = []
        std_q_lens = []

        for i in std_req_idxs:
            q_start = q_starts[i]
            q_end = q_starts[i+1]
            std_indices.extend(range(q_start, q_end))
            std_q_lens.append(q_end - q_start)

        return std_indices, std_q_lens

    def _build_mega_mappings(
        self, mega_req_idxs, mega_prefix_lens, mega_beam_widths,
        mega_decode_steps, mega_cache_lens, q_starts, common_attn_metadata
    ):
        mega_prefix_indices = []
        mega_prefix_q_lens = []
        mega_prefix_seq_lens_list = []
        
        mega_suffix_q_lens = []
        mega_suffix_seq_lens_list = []
        mega_suffix_out_offset = []
        mega_prefix_block_tables = []
        mega_suffix_bw_list = []
        mega_suffix_shared_gap_list = []
        offset = 0
        max_prefix_len = max(mega_prefix_lens) if mega_prefix_lens else 0
        max_prefix_blocks_count = (max_prefix_len + self.block_size - 1) // self.block_size

        for req_idx, prefix_len, w, steps, cache_len in zip(
            mega_req_idxs, mega_prefix_lens, mega_beam_widths,
            mega_decode_steps, mega_cache_lens
        ):
            q_start = q_starts[req_idx]
            q_end = q_starts[req_idx+1]
            q_len = q_end - q_start
            if w == 0 or steps == 0:
                continue
            
            mega_suffix_bw_list.append(w)
            q_curr = q_start
            bt = common_attn_metadata.block_table_tensor[req_idx]
            
            chunk_budget = q_len
            delta = max(0, cache_len - prefix_len)
            prefix_groups = []

            for b in range(w):
                if chunk_budget <= 0:
                    break
                    
                b_uncached = 0
                b_cache_len = 0
                b_suffix_len = 0
                if prefix_len >= cache_len:
                    # Scenario A: Prefix Incomplete
                    remaining_prefix = prefix_len - cache_len
                    mega_suffix_shared_gap_list.append(remaining_prefix)
                    if b == 0:
                        b_uncached = min(chunk_budget, remaining_prefix + steps)
                    else:
                        b_uncached = min(chunk_budget, steps)
                    
                    b_cache_len = cache_len
                    b_suffix_len = remaining_prefix + steps
                else:
                    # Scenario B: Prefix Complete
                    past_suffix_b = max(0, min(steps, delta - b * steps))
                    remaining_suffix_b = steps - past_suffix_b
                    b_uncached = min(chunk_budget, remaining_suffix_b)
                    
                    b_cache_len = prefix_len
                    b_suffix_len = past_suffix_b + b_uncached
                
                if b_uncached <= 0:
                    continue

                if prefix_groups and prefix_groups[-1][2] == b_cache_len:
                    prefix_groups[-1][1] += b_uncached
                else:
                    prefix_groups.append([q_curr, b_uncached, b_cache_len])
                
                mega_suffix_q_lens.append(b_uncached)
                mega_suffix_seq_lens_list.append(b_suffix_len)
                mega_suffix_out_offset.append(offset)
                
                offset += b_suffix_len
                q_curr += b_uncached
                chunk_budget -= b_uncached

            for group_q_start, group_q_len, group_cache_len in prefix_groups:
                mega_prefix_indices.extend(range(group_q_start, group_q_start + group_q_len))
                mega_prefix_q_lens.append(group_q_len)
                mega_prefix_seq_lens_list.append(group_cache_len)
                mega_prefix_block_tables.append(bt[:max_prefix_blocks_count])

        return (
            mega_prefix_indices, mega_prefix_q_lens, mega_prefix_seq_lens_list,
            mega_suffix_q_lens, mega_suffix_seq_lens_list,
            mega_suffix_out_offset, mega_prefix_block_tables, mega_suffix_bw_list, mega_suffix_shared_gap_list
        )

    def _create_std_tensors(self, std_indices, std_q_lens, std_req_idxs, common_attn_metadata, device):
        if std_q_lens:
            std_indices_tensor = torch.tensor(std_indices, dtype=torch.long, device=device)
            std_cu_seqlens_q = torch.tensor([0] + std_q_lens, dtype=torch.int32, device=device).cumsum(dim=0).to(torch.int32)
            std_seq_lens = common_attn_metadata.seq_lens[std_req_idxs]
            std_max_q_len = max(std_q_lens)
            std_max_seq_len = common_attn_metadata.max_seq_len
            std_block_table = common_attn_metadata.block_table_tensor[std_req_idxs].contiguous()
            return std_indices_tensor, std_cu_seqlens_q, std_seq_lens, std_max_q_len, std_max_seq_len, std_block_table
        return None, None, None, 0, 0, None

    def _create_mega_tensors(
        self, mega_prefix_indices, mega_prefix_q_lens, mega_prefix_seq_lens_list,
        mega_suffix_q_lens, mega_suffix_seq_lens_list,
        mega_suffix_out_offset, mega_prefix_block_tables, device, max_num_splits: int
    ):
        if mega_prefix_q_lens:
            mega_prefix_indices_tensor = torch.tensor(mega_prefix_indices, dtype=torch.long, device=device)
            mega_prefix_seq_lens = torch.tensor(mega_prefix_seq_lens_list, dtype=torch.int32, device=device)
            mega_prefix_max_seq_len = max(mega_prefix_seq_lens_list) if mega_prefix_seq_lens_list else 0
            mega_prefix_cu_seqlens_q = torch.tensor([0] + mega_prefix_q_lens, dtype=torch.int32, device=device).cumsum(dim=0).to(torch.int32)
            mega_prefix_max_q_len = max(mega_prefix_q_lens) if mega_prefix_q_lens else 0
            mega_prefix_block_table = torch.stack(mega_prefix_block_tables).contiguous()
            
            mega_suffix_seq_lens = torch.tensor(mega_suffix_seq_lens_list, dtype=torch.int32, device=device)
            mega_suffix_max_seq_len = max(mega_suffix_seq_lens_list) if mega_suffix_seq_lens_list else 0
            mega_suffix_total_tokens = sum(mega_suffix_seq_lens_list)
            
            mega_suffix_cu_seqlens_q = torch.tensor([0] + mega_suffix_q_lens, dtype=torch.int32, device=device).cumsum(dim=0).to(torch.int32)
            mega_suffix_cu_seqlens_k = torch.tensor([0] + mega_suffix_seq_lens_list, dtype=torch.int32, device=device).cumsum(dim=0).to(torch.int32)
            mega_suffix_max_q_len = max(mega_suffix_q_lens) if mega_suffix_q_lens else 0
            
            mega_suffix_out_offset_tensor = torch.tensor(mega_suffix_out_offset, dtype=torch.int32, device=device)

            prefix_scheduler_metadata = self._get_schedule(
                batch_size=len(mega_prefix_q_lens),
                cu_query_lens=mega_prefix_cu_seqlens_q,
                max_query_len=mega_prefix_max_q_len,
                seqlens=mega_prefix_seq_lens,
                max_seq_len=mega_prefix_max_seq_len,
                causal=False,
                max_num_splits=max_num_splits,
            )
        else:
            # Full structured initializations for profiling warm-up model passes
            mega_prefix_indices_tensor = mega_prefix_seq_lens = mega_prefix_cu_seqlens_q = mega_prefix_block_table = None
            mega_prefix_max_q_len = mega_prefix_max_seq_len = 0
            mega_suffix_cu_seqlens_q = mega_suffix_cu_seqlens_k = mega_suffix_seq_lens = None
            mega_suffix_max_q_len = mega_suffix_max_seq_len = mega_suffix_total_tokens = 0
            mega_suffix_out_offset_tensor = None
            prefix_scheduler_metadata = None
            
        return (
            mega_prefix_indices_tensor, mega_prefix_seq_lens, mega_prefix_max_seq_len,
            mega_prefix_cu_seqlens_q, mega_prefix_max_q_len, mega_prefix_block_table,
            mega_suffix_seq_lens, mega_suffix_max_seq_len, mega_suffix_total_tokens,
            mega_suffix_cu_seqlens_q, mega_suffix_cu_seqlens_k, mega_suffix_max_q_len,
            mega_suffix_out_offset_tensor, prefix_scheduler_metadata
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> BeamAttentionMetadata:
        torch.cuda.nvtx.range_push("BeamAttentionMetadataBuilder.build")
        self._update_aot_schedule(fast_build)
        max_num_splits = self._get_max_num_splits(common_attn_metadata.num_actual_tokens)
        device = self.device
        num_reqs = common_attn_metadata.num_reqs
        mega_info = MEGA_DATA_VAR.get()
        mega_data = mega_info["mega_data"] if mega_info else {}
        req_ids = mega_info["req_ids"] if mega_info else []

        (std_req_idxs, mega_req_idxs, mega_prefix_lens, mega_beam_widths,
         mega_decode_steps, mega_cache_lens) = self._segment_requests(num_reqs, req_ids, mega_data)

        q_starts = common_attn_metadata.query_start_loc_cpu.tolist()

        std_indices, std_q_lens = self._build_std_mappings(std_req_idxs, q_starts)

        (mega_prefix_indices, mega_prefix_q_lens, mega_prefix_seq_lens_list,
         mega_suffix_q_lens, mega_suffix_seq_lens_list,
         mega_suffix_out_offset, mega_prefix_block_tables, 
         mega_suffix_bw_list, mega_suffix_shared_gap_list) = self._build_mega_mappings(
            mega_req_idxs, mega_prefix_lens, mega_beam_widths,
            mega_decode_steps, mega_cache_lens, q_starts, common_attn_metadata
        )

        (std_indices_tensor, std_cu_seqlens_q, std_seq_lens,
         std_max_q_len, std_max_seq_len, std_block_table) = self._create_std_tensors(
            std_indices, std_q_lens, std_req_idxs, common_attn_metadata, device
        )

        (mega_prefix_indices_tensor, mega_prefix_seq_lens, mega_prefix_max_seq_len,
         mega_prefix_cu_seqlens_q, mega_prefix_max_q_len, mega_prefix_block_table,
         mega_suffix_seq_lens, mega_suffix_max_seq_len, mega_suffix_total_tokens,
         mega_suffix_cu_seqlens_q, mega_suffix_cu_seqlens_k, mega_suffix_max_q_len,
         mega_suffix_out_offset_tensor, prefix_scheduler_metadata) = self._create_mega_tensors(
            mega_prefix_indices, mega_prefix_q_lens, mega_prefix_seq_lens_list,
            mega_suffix_q_lens, mega_suffix_seq_lens_list,
            mega_suffix_out_offset, mega_prefix_block_tables, device, max_num_splits
        )

        torch.cuda.nvtx.range_pop()
        return BeamAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            causal=common_attn_metadata.causal,
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
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            mega_suffix_cu_seqlens_q=mega_suffix_cu_seqlens_q,
            mega_suffix_cu_seqlens_k=mega_suffix_cu_seqlens_k,
            mega_suffix_seq_lens=mega_suffix_seq_lens,
            mega_suffix_max_q_len=mega_suffix_max_q_len,
            mega_suffix_max_seq_len=mega_suffix_max_seq_len,
            mega_suffix_total_tokens=mega_suffix_total_tokens,
            mega_suffix_out_offset=mega_suffix_out_offset_tensor,
            mega_suffix_shared_gap=mega_suffix_shared_gap_list[0] if mega_suffix_shared_gap_list else 0,
            mega_suffix_bw = mega_suffix_bw_list[0] if mega_suffix_bw_list else 0,
            max_num_splits=max_num_splits,
        )

    def update_block_table(
        self,
        metadata: BeamAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> BeamAttentionMetadata:
        new_metadata = copy.copy(metadata)
        new_metadata.block_table = blk_table
        new_metadata.slot_mapping = slot_mapping
        return new_metadata


# ---------------------------------------------------------------------------
# CORE EXECUTION REFACTOR
# ---------------------------------------------------------------------------
class BeamAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        running_sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.attn_type = attn_type
        self.vllm_flash_attn_version = get_flash_attn_version()
        self.batch_invariant_enabled = vllm_is_batch_invariant()

        if is_quantized_kv_cache(self.kv_cache_dtype) and not flash_attn_supports_fp8():
            raise NotImplementedError("BeamAttention does not support fp8 kv-cache on this device.")

        self.sinks = running_sinks
        if self.sinks is not None:
            assert flash_attn_supports_sinks(), "Sinks are only supported in FlashAttention 3"
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of heads in the layer"
            )
        self.supports_quant_query_input = True
        self.use_cascade = True

    @staticmethod
    def cascade_attention(
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: BeamAttentionMetadata,
        sliding_window_size: list[int] | tuple[int, int] | None,
        scale: float,
        logits_soft_cap: float,
        vllm_flash_attn_version: int,
        s_aux: torch.Tensor | None = None,
        alibi_slopes: torch.Tensor | None = None,
    ):
        if attn_metadata.std_indices is not None:
            std_query = query[attn_metadata.std_indices]
            std_out = torch.empty_like(std_query)
            flash_attn_varlen_func(
                q=std_query, k=key_cache, v=value_cache, out=std_out,
                cu_seqlens_q=attn_metadata.std_cu_seqlens_q, max_seqlen_q=attn_metadata.std_max_q_len,
                seqused_k=attn_metadata.std_seq_lens, max_seqlen_k=attn_metadata.std_max_seq_len,
                softmax_scale=scale, causal=attn_metadata.causal, alibi_slopes=alibi_slopes,
                window_size=sliding_window_size, block_table=attn_metadata.std_block_table,
                softcap=logits_soft_cap, fa_version=vllm_flash_attn_version,
            )
            output[attn_metadata.std_indices] = std_out

        if attn_metadata.mega_prefix_indices is not None:
            mega_query = query[attn_metadata.mega_prefix_indices]

            mega_prefix_out, mega_prefix_lse = flash_attn_varlen_func(
                q=mega_query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=attn_metadata.mega_prefix_cu_seqlens_q,
                seqused_k=attn_metadata.mega_prefix_seq_lens,
                max_seqlen_q=attn_metadata.mega_prefix_max_q_len,
                max_seqlen_k=attn_metadata.mega_prefix_max_seq_len,
                softmax_scale=scale,
                causal=False,
                alibi_slopes=alibi_slopes,
                window_size=sliding_window_size,
                block_table=attn_metadata.mega_prefix_block_table,
                softcap=logits_soft_cap,
                return_softmax_lse=True,
                fa_version=vllm_flash_attn_version,
                scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
                num_splits=attn_metadata.max_num_splits,
                s_aux=s_aux,
            )
            
            total_suffix_tokens = attn_metadata.mega_suffix_total_tokens
            block_size = key_cache.shape[1]

            print("attn_metadata.mega_prefix_cu_seqlens_q", attn_metadata.mega_prefix_cu_seqlens_q)
            print("attn_metadata.mega_prefix_seq_lens", attn_metadata.mega_prefix_seq_lens)
            print("attn_metadata.mega_suffix_cu_seqlens_q", attn_metadata.mega_suffix_cu_seqlens_q)
            print("attn_metadata.mega_suffix_seq_lens", attn_metadata.mega_suffix_seq_lens)
            print("attn_metadata.slot_mapping", attn_metadata.slot_mapping)
            print("attn_metadata.mega_suffix_seq_lens", attn_metadata.mega_suffix_seq_lens)
            print("attn_metadata.mega_suffix_out_offset", attn_metadata.mega_suffix_out_offset)
            print("total_suffix_tokens", total_suffix_tokens)
            print("attn_metadata.mega_suffix_max_seq_len", attn_metadata.mega_suffix_max_seq_len)
            print("attn_metadata.mega_suffix_bw", attn_metadata.mega_suffix_bw)
            print("attn_metadata.mega_suffix_shared_gap", attn_metadata.mega_suffix_shared_gap)

            contig_k, contig_v = extract_suffix_kv(
            key_cache, value_cache, 
            attn_metadata.slot_mapping, # Global flat tensor
            attn_metadata.mega_suffix_seq_lens,
            attn_metadata.mega_suffix_out_offset,
            shared_gap_len=attn_metadata.mega_suffix_shared_gap,
            num_beams=attn_metadata.mega_suffix_bw,
            max_suffix_len=attn_metadata.mega_suffix_max_seq_len
            )
            
            if total_suffix_tokens > 0:
                mega_suffix_out = torch.empty_like(mega_query)
                mega_suffix_out, mega_suffix_lse = flash_attn_varlen_func(
                    q=mega_query, k=contig_k, v=contig_v, out=mega_suffix_out,
                    cu_seqlens_q=attn_metadata.mega_suffix_cu_seqlens_q, cu_seqlens_k=attn_metadata.mega_suffix_cu_seqlens_k,
                    seqused_k=None, max_seqlen_q=attn_metadata.mega_suffix_max_q_len,
                    max_seqlen_k=attn_metadata.mega_suffix_max_seq_len, softmax_scale=scale,
                    causal=True, alibi_slopes=alibi_slopes, window_size=sliding_window_size,
                    block_table=None, softcap=logits_soft_cap, return_softmax_lse=True, fa_version=vllm_flash_attn_version,
                )
                
                mega_out = torch.empty_like(mega_query)
                merge_attn_states(mega_out, mega_prefix_out, mega_prefix_lse, mega_suffix_out, mega_suffix_lse)
                output[attn_metadata.mega_prefix_indices] = mega_out
            else:
                output[attn_metadata.mega_prefix_indices] = mega_prefix_out

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: BeamAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        assert self.vllm_flash_attn_version is not None, "FlashAttention version not detected."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for BeamAttentionImpl"
            )
        if attn_metadata is None:
            return output.fill_(0)

        key_cache, value_cache = kv_cache.unbind(0)

        if self.kv_sharing_target_layer_name is None and key is not None and value is not None:
            reshape_and_cache_flash(
                key, value, key_cache, value_cache,
                attn_metadata.slot_mapping, self.kv_cache_dtype,
                layer._k_scale, layer._v_scale,
            )

        if self.kv_cache_dtype.startswith("fp8"):
            dtype = BeamAttentionBackend.get_fp8_dtype_for_flashattn(self.kv_cache_dtype)
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        sliding_window_size = list(self.sliding_window) if self.sliding_window is not None else None

        BeamAttentionImpl.cascade_attention(
            output=output,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            attn_metadata=attn_metadata,
            sliding_window_size=sliding_window_size,
            scale=self.scale,
            logits_soft_cap=self.logits_soft_cap,
            vllm_flash_attn_version=self.vllm_flash_attn_version,
            s_aux=self.sinks,
            alibi_slopes=self.alibi_slopes,
        )

        return output


@register_backend(AttentionBackendEnum.CUSTOM)
class BeamAttentionBackend(AttentionBackend):
    accept_output_buffer = True
    supported_dtypes = [torch.float16, torch.bfloat16]
    
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type in (AttentionType.DECODER, AttentionType.ENCODER, AttentionType.ENCODER_ONLY, AttentionType.ENCODER_DECODER)

    @staticmethod
    def get_impl_cls() -> type["BeamAttentionImpl"]:
        return BeamAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["BeamAttentionMetadataBuilder"]:
        return BeamAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int, head_size: int, cache_dtype_str: str = "auto") -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension: bool = False) -> tuple[int, ...]:
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            return (2, 0, 1, 3, 4, 5)
        elif cache_layout == "NHD":
            return (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            return (2, 4, 0, 1, 3, 5)
        elif cache_layout == "HND":
            return (0, 1, 3, 2, 4)
        raise ValueError(f"Unknown cache layout format {cache_layout}.")

    @staticmethod
    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size % 8 == 0 and head_size <= 256

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        if kv_cache_dtype.startswith("fp8"):
            return flash_attn_supports_fp8()
        return kv_cache_dtype in ["auto"]

    @classmethod
    def supports_sink(cls) -> bool:
        return is_flash_attn_varlen_func_available() and flash_attn_supports_sinks()

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)

    @classmethod
    def supports_combination(cls, head_size, dtype, kv_cache_dtype, block_size, use_mla, has_sink, use_sparse, device_capability) -> str | None:
        if not is_flash_attn_varlen_func_available(): return "No FlashAttention varlen support."
        if dtype not in cls.supported_dtypes: return f"Unsupported dtype {dtype}."
        if not cls.supports_head_size(head_size): return f"Unsupported head_size={head_size}."
        if not cls.supports_kv_cache_dtype(kv_cache_dtype): return f"Unsupported kv_cache_dtype={kv_cache_dtype}."
        if block_size is not None and block_size % 16 != 0: return f"block_size={block_size} not multiple of 16."
        if not cls.supports_compute_capability(device_capability): return f"Requires GPU compute capability >= 8.0."
        if has_sink and not cls.supports_sink(): return "No FlashAttention sink support."
        if use_mla: return "No MLA support."
        if use_sparse: return "No sparse support."
        return None