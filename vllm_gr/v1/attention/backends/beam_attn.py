# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Beam-Consecutive Suffix Gather Attention Backend."""

import contextvars
import copy
from dataclasses import dataclass
from itertools import compress
from typing import ClassVar

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
def _merge_indexed_attn_states_kernel(
    Out,
    Lse_s,
    Prefix_out,
    Prefix_lse,
    Indices,
    stride_out_n,
    stride_out_h,
    stride_out_d,
    stride_lse_s_h,
    stride_lse_s_n,
    stride_p_out_n,
    stride_p_out_h,
    stride_p_out_d,
    stride_p_lse_h,
    stride_p_lse_n,
    head_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    idx_out = tl.load(Indices + pid_n)

    lse_p = tl.load(Prefix_lse + pid_h * stride_p_lse_h + pid_n * stride_p_lse_n)
    lse_s = tl.load(Lse_s + pid_h * stride_lse_s_h + idx_out * stride_lse_s_n)
    # FA2 and FA3 have different behavior for when the sum-exp is 0, this namely
    # arises with 0 len seqlens. FA3 returns -inf here while FA2 returns inf.
    # If we see an inf assume FA2 and convert inf to -inf for consistency
    # and correctness. Inf generally doesn't make sense in this context outside
    # of undefined-behavior/FA2-case, so I think this a safe assumption.
    lse_p = tl.where(lse_p == float("inf"), float("-inf"), lse_p)
    lse_s = tl.where(lse_s == float("inf"), float("-inf"), lse_s)

    m = tl.maximum(lse_p, lse_s)
    se_p = tl.exp(lse_p - m)
    se_s = tl.exp(lse_s - m)
    sum_exp = se_p + se_s

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_size

    off_p_out = pid_n * stride_p_out_n + pid_h * stride_p_out_h + offs_d * stride_p_out_d
    val_p = tl.load(Prefix_out + off_p_out, mask=mask_d, other=0.0)

    off_s_out = idx_out * stride_out_n + pid_h * stride_out_h + offs_d * stride_out_d
    val_s = tl.load(Out + off_s_out, mask=mask_d, other=0.0)

    scale_p = se_p / sum_exp
    scale_s = se_s / sum_exp

    res = scale_p * val_p + scale_s * val_s
    res = tl.where(sum_exp == 0.0, 0.0, res)

    tl.store(Out + off_s_out, res, mask=mask_d)


def merge_indexed_attn_states(output, suffix_lse, prefix_out, prefix_lse, prefix_indices):
    num_prefix_tokens = prefix_indices.shape[0]
    num_heads = output.shape[1]
    head_size = output.shape[2]
    BLOCK_D = triton.next_power_of_2(head_size)
    grid = (num_prefix_tokens, num_heads)

    _merge_indexed_attn_states_kernel[grid](
        output,
        suffix_lse,
        prefix_out,
        prefix_lse,
        prefix_indices,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        suffix_lse.stride(0),
        suffix_lse.stride(1),
        prefix_out.stride(0),
        prefix_out.stride(1),
        prefix_out.stride(2),
        prefix_lse.stride(0),
        prefix_lse.stride(1),
        head_size=head_size,
        BLOCK_D=BLOCK_D,
    )

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

    prefix_indices: torch.Tensor | None = None
    prefix_indices_is_identity: bool = False
    cu_prefix_query_lens: torch.Tensor | None = None
    prefix_kv_lens: torch.Tensor | None = None
    prefix_block_table: torch.Tensor | None = None
    prefix_query_max_len: int = 0
    prefix_kv_max_len: int = 0

    cu_suffix_query_lens: torch.Tensor | None = None
    suffix_kv_lens: torch.Tensor | None = None
    suffix_block_table: torch.Tensor | None = None
    suffix_query_max_len: int = 0
    suffix_kv_max_len: int = 0

    scheduler_metadata: dict | None = None
    max_num_splits: int = 0

class BeamAttentionMetadataBuilder(AttentionMetadataBuilder[BeamAttentionMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    supports_update_block_table: bool = False

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

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> BeamAttentionMetadata:
        device = self.device
        num_reqs = common_attn_metadata.num_reqs
        
        mega_info = MEGA_DATA_VAR.get()
        mega_data = mega_info["mega_data"] if mega_info else {}
        req_ids = mega_info["req_ids"] if mega_info else []

        q_starts = common_attn_metadata.query_start_loc_cpu.tolist()

        prefix_indices = []
        prefix_q_lens = []
        prefix_kv_lens = []
        prefix_block_tables = []

        suffix_q_lens = []
        suffix_kv_lens = []
        suffix_block_tables = []

        for i in range(num_reqs):
            req_id = req_ids[i] if i < len(req_ids) else None
            mdata = mega_data.get(req_id, {}) if req_id else {}
            
            q_start = q_starts[i]
            q_end = q_starts[i+1]
            seq_len = q_end - q_start
            computed = common_attn_metadata._num_computed_tokens_cpu[i].item()
            step_start = computed
            step_end = computed + seq_len

            aligned_prefix_len = 0
            if mdata.get("is_mega_decode", False):
                prefix_len = mdata['prefix_len']
                aligned_prefix_len = (prefix_len // self.block_size) * self.block_size
                
            aligned_prefix_len = min(aligned_prefix_len, step_start)

            if aligned_prefix_len > 0:
                prefix_indices.extend(range(q_start, q_end))
                prefix_q_lens.append(seq_len)
                prefix_kv_lens.append(aligned_prefix_len)
                prefix_blocks = aligned_prefix_len // self.block_size
                prefix_block_tables.append(common_attn_metadata.block_table_tensor[i, :prefix_blocks])
            
            suffix_q_lens.append(seq_len)
            suffix_kv_lens.append(step_end - aligned_prefix_len)
            prefix_blocks = aligned_prefix_len // self.block_size
            suffix_block_tables.append(common_attn_metadata.block_table_tensor[i, prefix_blocks:])

        prefix_indices_tensor = None
        prefix_indices_is_identity = False
        if prefix_indices:
            if len(prefix_indices) == q_starts[-1]:
                prefix_indices_is_identity = True
            else:
                prefix_indices_tensor = torch.tensor(prefix_indices, dtype=torch.long, device=device)
        
        # Build prefix tensors
        if prefix_q_lens:
            cu_prefix_query_lens = torch.tensor([0] + prefix_q_lens, dtype=torch.int32, device=device).cumsum(dim=0).to(torch.int32)
            prefix_kv_lens_tensor = torch.tensor(prefix_kv_lens, dtype=torch.int32, device=device)
            prefix_query_max_len = max(prefix_q_lens)
            prefix_kv_max_len = max(prefix_kv_lens)
            
            max_prefix_blocks = max(bt.size(0) for bt in prefix_block_tables)
            prefix_block_table = torch.zeros((len(prefix_block_tables), max_prefix_blocks), dtype=torch.int32, device=device)
            for i, bt in enumerate(prefix_block_tables):
                prefix_block_table[i, :bt.size(0)] = bt
        else:
            cu_prefix_query_lens = prefix_kv_lens_tensor = prefix_block_table = None
            prefix_query_max_len = prefix_kv_max_len = 0

        # Build suffix tensors
        cu_suffix_query_lens = torch.tensor([0] + suffix_q_lens, dtype=torch.int32, device=device).cumsum(dim=0).to(torch.int32)
        suffix_kv_lens_tensor = torch.tensor(suffix_kv_lens, dtype=torch.int32, device=device)
        suffix_query_max_len = max(suffix_q_lens)
        suffix_kv_max_len = max(suffix_kv_lens)

        max_suffix_blocks = max(bt.size(0) for bt in suffix_block_tables) if suffix_block_tables else 0
        suffix_block_table = torch.zeros((num_reqs, max_suffix_blocks), dtype=torch.int32, device=device)
        for i, bt in enumerate(suffix_block_tables):
            suffix_block_table[i, :bt.size(0)] = bt

        return BeamAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            causal=common_attn_metadata.causal,
            prefix_indices=prefix_indices_tensor,
            prefix_indices_is_identity=prefix_indices_is_identity,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens_tensor,
            prefix_block_table=prefix_block_table,
            prefix_query_max_len=prefix_query_max_len,
            prefix_kv_max_len=prefix_kv_max_len,
            cu_suffix_query_lens=cu_suffix_query_lens,
            suffix_kv_lens=suffix_kv_lens_tensor,
            suffix_block_table=suffix_block_table,
            suffix_query_max_len=suffix_query_max_len,
            suffix_kv_max_len=suffix_kv_max_len,
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
        sinks: torch.Tensor | None = None,
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

        self.sinks = sinks
        if self.sinks is not None:
            assert flash_attn_supports_sinks(), "Sinks are only supported in FlashAttention 3"
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of heads in the layer"
            )
        self.supports_quant_query_input = True
        self.use_cascade = True

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

        # For decoder attention, use KV cache
        key_cache, value_cache = kv_cache.unbind(0)

        # Update KV cache with new keys and values
        if self.kv_sharing_target_layer_name is None and key is not None and value is not None:
            reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

        # Handle FP8 quantization if needed
        if self.kv_cache_dtype.startswith("fp8"):
            dtype = BeamAttentionBackend.get_fp8_dtype_for_flashattn(self.kv_cache_dtype)
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        sliding_window_size = list(self.sliding_window) if self.sliding_window is not None else None

        prefix_out, prefix_lse = None, None
        has_work = attn_metadata.prefix_indices_is_identity or (
            attn_metadata.prefix_indices is not None and attn_metadata.prefix_indices.numel() > 0
        )

        if has_work:
            if attn_metadata.prefix_indices_is_identity:
                prefix_query = query
            else:
                prefix_query = query[attn_metadata.prefix_indices]

            prefix_out, prefix_lse = flash_attn_varlen_func(
                q=prefix_query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=attn_metadata.cu_prefix_query_lens,
                seqused_k=attn_metadata.prefix_kv_lens,
                max_seqlen_q=attn_metadata.prefix_query_max_len,
                max_seqlen_k=attn_metadata.prefix_kv_max_len,
                softmax_scale=self.scale, causal=False, alibi_slopes=self.alibi_slopes,
                window_size=sliding_window_size, block_table=attn_metadata.prefix_block_table,
                softcap=self.logits_soft_cap, return_softmax_lse=True, fa_version=self.vllm_flash_attn_version,
            )
            
        suffix_out, suffix_lse = flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=attn_metadata.cu_suffix_query_lens,
            seqused_k=attn_metadata.suffix_kv_lens,
            max_seqlen_q=attn_metadata.suffix_query_max_len,
            max_seqlen_k=attn_metadata.suffix_kv_max_len,
            softmax_scale=self.scale, causal=True, alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size, block_table=attn_metadata.suffix_block_table,
            softcap=self.logits_soft_cap, return_softmax_lse=True, fa_version=self.vllm_flash_attn_version,
            out=output,
        )

        if prefix_out is not None:
            if attn_metadata.prefix_indices_is_identity:
                merge_attn_states(output, prefix_out, prefix_lse, suffix_out, suffix_lse)
            else:
                merge_indexed_attn_states(
                    output, suffix_lse, prefix_out, prefix_lse, attn_metadata.prefix_indices
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
