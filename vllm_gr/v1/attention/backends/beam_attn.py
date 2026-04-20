# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph Reuse Attention (Beam Attn) - Optimized attention with shared KV cache buffer separation."""

import copy
import contextvars
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
from vllm.v1.attention.backends.flash_attn import _get_sliding_window_configs
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
            "but it is not available in this environment. "
            "Ensure FlashAttention with varlen kernels is installed and "
            "that you are running on a supported GPU."
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
BEAM_PREFIX_GROUPS_VAR = contextvars.ContextVar("beam_prefix_groups", default=None)
PBSC_SHAPED_GROUPS_VAR = contextvars.ContextVar("pbsc_shaped_groups", default=None)

@register_backend(AttentionBackendEnum.CUSTOM)
class BeamAttentionBackend(AttentionBackend):
    """Beam (Graph Reuse) Attention Backend.

    Optimizes attention computation by separating KV cache into:
    1. Shared pages: Reused across multiple requests (copied once)
    2. Non-shared pages: Request-specific pages

    Processes them separately through flash attention and merges outputs
    using LSE-based normalization.
    """

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        if (
            model_config
            and model_config.is_hybrid
            and (
                cache_config.mamba_ssm_cache_dtype == "float32"
                or cache_config.mamba_cache_dtype == "float32"
            )
        ):
            # NOTE(tdoublep): while in principle, FA supports
            # MultipleOf(16), these are the block sizes that do not
            # suffer from the NaN propagation problem described here:
            # https://github.com/Dao-AILab/flash-attention/issues/1974
            return [16, 32, 64]
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """Beam Attention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @staticmethod
    def get_impl_cls() -> type["BeamAttentionImpl"]:
        return BeamAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["BeamAttentionMetadataBuilder"]:
        return BeamAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            return (2, 0, 1, 3, 4, 5)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            return (2, 4, 0, 1, 3, 5)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    @staticmethod
    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        else:
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
        if not is_flash_attn_varlen_func_available():
            return False
        return flash_attn_supports_sinks()

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        # Hard requirement: FlashAttention varlen support must be available.
        if not is_flash_attn_varlen_func_available():
            return (
                "BeamAttentionBackend requires FlashAttention varlen support, "
                "but it is not available in this environment."
            )
        # Dtype must be one of the backend's supported dtypes.
        if dtype not in cls.supported_dtypes:
            return (
                f"BeamAttentionBackend does not support dtype {dtype}; "
                f"supported dtypes are {cls.supported_dtypes}."
            )
        # Enforce head size constraints.
        if not cls.supports_head_size(head_size):
            return (
                f"BeamAttentionBackend does not support head_size={head_size}. "
                "Head size must be <= 256 and a multiple of 8."
            )
        # Enforce KV cache dtype constraints (including FP8 support).
        if not cls.supports_kv_cache_dtype(kv_cache_dtype):
            return f"BeamAttentionBackend does not support kv_cache_dtype={kv_cache_dtype}."
        # Enforce block size constraints if provided.
        if block_size is not None and block_size % 16 != 0:
            return (
                f"BeamAttentionBackend requires block_size to be a multiple of 16, "
                f"but got block_size={block_size}."
            )
        # Enforce compute capability constraints.
        if not cls.supports_compute_capability(device_capability):
            return (
                "BeamAttentionBackend requires GPU compute capability >= 8.0 "
                f"but got {device_capability}."
            )
        # Sinks are only supported when FlashAttention sink support is available.
        if has_sink and not cls.supports_sink():
            return (
                "BeamAttentionBackend was requested with sink attention, but "
                "FlashAttention sinks are not supported in this environment."
            )
        # MLA is not currently supported by this backend.
        if use_mla:
            return "BeamAttentionBackend does not support MLA (Mixture-of-Linear-Attention)."
        # Sparse attention is not currently supported by this backend.
        if use_sparse:
            return "BeamAttentionBackend does not support sparse attention."

        return None

@dataclass
class BeamAttentionMetadata:
    """Metadata for Beam (Graph Reuse) Attention.

    Extends cascade_attention metadata with shared/non-shared buffer tracking.
    """

    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # Cascade attention metadata (from FlashAttention)
    cu_prefix_query_lens: torch.Tensor | None = None
    cu_prefix_kv_lens: torch.Tensor | None = None
    prefix_kv_lens: torch.Tensor | None = None
    suffix_kv_lens: torch.Tensor | None = None
    common_prefix_lengths: torch.Tensor | None = None
    max_num_splits: int = 0
    prefix_scheduler_metadata: dict | None = None
    scheduler_metadata: dict | None = None

    prefix_block_table: torch.Tensor | None = None
    suffix_block_table: torch.Tensor | None = None

    # Beam-specific parameters for query and sequence lengths
    prefix_query_max_len: int = 0
    suffix_query_max_len: int = 0
    prefix_kv_max_len: int = 0
    suffix_kv_max_len: int = 0

    # Shared buffer metadata (Beam-specific optimization)
    shared_k_buffer: torch.Tensor | None = None
    shared_v_buffer: torch.Tensor | None = None
    prefix_indices: torch.Tensor | None = None
    prefix_indices_is_identity: bool = False
    group_first_reqs: torch.Tensor | None = None
    suffix_col_indices: torch.Tensor | None = None

    # PBSC KV Broadcast metadata
    pbsc_src_slots: torch.Tensor | None = None
    pbsc_dst_slots: torch.Tensor | None = None

    causal: bool = True
    use_cascade_attention: bool = True


class BeamAttentionMetadataBuilder(AttentionMetadataBuilder[BeamAttentionMetadata]):
    """Builder for Beam Attention metadata.

    Builds on FlashAttention's cascade_attention metadata structure,
    adding shared/non-shared buffer separation for optimization.
    """

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
        self.max_num_splits = 0  # No upper bound on the number of splits.
        self.aot_schedule = get_flash_attn_version() == 3

        self.cp_kv_cache_interleave_size = self.parallel_config.cp_kv_cache_interleave_size

        self.use_full_cuda_graph = self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        self.max_cudagraph_size = self.compilation_config.max_cudagraph_capture_size

        if self.use_full_cuda_graph and self.aot_schedule:
            self.scheduler_metadata = torch.zeros(
                vllm_config.scheduler_config.max_num_seqs + 1,
                dtype=torch.int32,
                device=self.device,
            )
            # When using cuda graph, we need to set the upper bound of the
            # number of splits so that large enough intermediate buffers are
            # pre-allocated during capture.
            self.max_num_splits = self.attention_config.flash_attn_max_num_splits_for_cuda_graph

        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: tuple[int, int] | None = None

        # Group thresholds
        self.min_tokens_for_sharing: int = 256
        self.tokens_threshold_for_sharing: int = 64

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> BeamAttentionMetadata:
        """Build Beam attention metadata with cascade_attention support.

        Inherits cascade_attention parameters and adds Beam-specific optimizations.
        """
        # 1. Setup AOT Schedule and Cuda Graph parameters
        self._update_aot_schedule(fast_build)
        max_num_splits = self._get_max_num_splits(common_attn_metadata.num_actual_tokens)

        beam_prefix_groups = BEAM_PREFIX_GROUPS_VAR.get()

        if beam_prefix_groups is None:
            return self._build_standard_metadata(common_attn_metadata, max_num_splits)

        common_prefix_groups = beam_prefix_groups
        # 3. Filter groups that have actual shared blocks
        shared_groups = [g for g in common_prefix_groups if g[0] > 0]

        # 4. Further filter groups based on query length constraints
        if shared_groups:
            shared_groups = self._filter_groups_by_query_len(
                shared_groups, common_attn_metadata.query_start_loc_cpu
            )

        # 5. Build Metadata (Shared vs Standard)
        if not shared_groups:
            return self._build_standard_metadata(common_attn_metadata, max_num_splits)

        return self._build_shared_metadata(common_attn_metadata, shared_groups, max_num_splits)

    def _update_aot_schedule(self, fast_build: bool):
        """Updates AOT schedule flags and sliding window configs."""
        self.aot_schedule = self.aot_schedule and not fast_build
        if self.aot_sliding_window is None:
            self.aot_sliding_window = (-1, -1)
            if self.aot_schedule:
                sliding_window_configs = _get_sliding_window_configs(self.vllm_config)
                if len(sliding_window_configs) == 1:
                    sliding_window_config = sliding_window_configs.pop()
                    if sliding_window_config is not None:
                        self.aot_sliding_window = sliding_window_config
                elif len(sliding_window_configs) > 1:
                    self.aot_schedule = False

    def _get_max_num_splits(self, num_actual_tokens: int) -> int:
        """Determines the maximum number of splits for Cuda Graph."""
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
        """Generates scheduler metadata for FlashAttention v3."""
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

    def _filter_groups_by_query_len(
        self, shared_groups: list, query_start_loc_cpu: torch.Tensor
    ) -> list:
        """Filters requests/groups based on valid query lengths."""
        # Flatten to get all req_ids
        flat_req_ids = [req_id for _, req_ids in shared_groups for req_id in req_ids]
        if not flat_req_ids:
            return []

        # Vectorized length calculation on CPU
        flat_req_ids_tensor = torch.tensor(flat_req_ids, dtype=torch.long)
        starts = query_start_loc_cpu[flat_req_ids_tensor]
        ends = query_start_loc_cpu[flat_req_ids_tensor + 1]
        lengths = ends - starts

        # Create mask: valid length and not too long (heuristic for beam optimization)
        mask = (lengths > 0) & (lengths <= self.block_size * 2)
        mask_np = mask.numpy()

        # Reconstruct groups efficiently
        new_groups = []
        curr_idx = 0
        for depth, req_ids in shared_groups:
            group_len = len(req_ids)
            group_mask = mask_np[curr_idx : curr_idx + group_len]
            curr_idx += group_len

            # Filter req_ids using the mask
            filtered_ids = list(compress(req_ids, group_mask))

            if filtered_ids:
                new_groups.append((depth, filtered_ids))

        return new_groups

    def _build_standard_metadata(
        self, common_meta: CommonAttentionMetadata, max_num_splits: int
    ) -> BeamAttentionMetadata:
        """Builds metadata for standard attention (no shared prefix optimization)."""
        scheduler_metadata = self._get_schedule(
            batch_size=common_meta.num_reqs,
            cu_query_lens=common_meta.query_start_loc,
            max_query_len=common_meta.max_query_len,
            seqlens=common_meta.seq_lens,
            max_seq_len=common_meta.max_seq_len,
            causal=common_meta.causal,
            max_num_splits=max_num_splits,
        )

        return BeamAttentionMetadata(
            num_actual_tokens=common_meta.num_actual_tokens,
            max_query_len=common_meta.max_query_len,
            query_start_loc=common_meta.query_start_loc,
            max_seq_len=common_meta.max_seq_len,
            seq_lens=common_meta.seq_lens,
            block_table=common_meta.block_table_tensor,
            slot_mapping=common_meta.slot_mapping,
            scheduler_metadata=scheduler_metadata,
            causal=common_meta.causal,
            use_cascade_attention=False,
            max_num_splits=max_num_splits,
        )

    def _build_shared_metadata(
        self, common_meta: CommonAttentionMetadata, shared_groups: list, max_num_splits: int
    ) -> BeamAttentionMetadata:
        """Builds metadata for split attention (shared prefix + suffix)."""

        # 1. Prepare Prefix Metadata
        (
            prefix_block_table,
            cu_prefix_query_lens,
            prefix_kv_lens,
            prefix_indices,
            prefix_query_max_len,
            prefix_kv_max_len,
            shift_amounts,
            shift_amounts_cpu,
            prefix_indices_is_identity,
            group_first_reqs,
        ) = self._create_prefix_metadata(common_meta, shared_groups)

        # 2. Prepare Suffix Metadata
        (
            suffix_block_table,
            suffix_kv_lens,
            suffix_query_max_len,
            suffix_kv_max_len,
            suffix_col_indices,
        ) = self._create_suffix_metadata(common_meta, shift_amounts, shift_amounts_cpu)

        # 3. Create Schedulers
        prefix_scheduler_metadata = self._get_schedule(
            batch_size=len(shared_groups),
            cu_query_lens=cu_prefix_query_lens,
            max_query_len=prefix_query_max_len,
            seqlens=prefix_kv_lens,
            max_seq_len=prefix_kv_max_len,
            causal=False,
            max_num_splits=max_num_splits,
        )

        scheduler_metadata = self._get_schedule(
            batch_size=common_meta.num_reqs,
            cu_query_lens=common_meta.query_start_loc,
            max_query_len=suffix_query_max_len,
            seqlens=suffix_kv_lens,
            max_seq_len=suffix_kv_max_len,
            causal=True,
            max_num_splits=max_num_splits,
        )

        # 4. Handle Cuda Graph Metadata padding
        if self.use_full_cuda_graph and scheduler_metadata is not None:
            n = scheduler_metadata.shape[0]
            self.scheduler_metadata[:n] = scheduler_metadata
            self.scheduler_metadata[n:] = 0
            scheduler_metadata = self.scheduler_metadata[:n]

        # Compute PBSC Broadcast Slots
        src_slots_list = []
        dst_slots_list = []

        pbsc_groups = PBSC_SHAPED_GROUPS_VAR.get()
        if pbsc_groups is not None:
            for t_shared, req_indices in pbsc_groups:
                t_block_aligned = (t_shared // self.block_size) * self.block_size
                tail_len = t_shared - t_block_aligned
                if tail_len == 0 or len(req_indices) <= 1:
                    continue

                leader_idx = req_indices[0]
                
                # Generate token indices for the tail
                tail_indices = torch.arange(t_block_aligned, t_shared, device=self.device)
                block_indices = tail_indices // self.block_size
                block_offsets = tail_indices % self.block_size

                # Leader's physical slots for the tail
                leader_blocks = common_meta.block_table_tensor[leader_idx, block_indices]
                leader_slots = leader_blocks * self.block_size + block_offsets

                for child_idx in req_indices[1:]:
                    child_blocks = common_meta.block_table_tensor[child_idx, block_indices]
                    child_slots = child_blocks * self.block_size + block_offsets
                    
                    src_slots_list.append(leader_slots)
                    dst_slots_list.append(child_slots)
                
        if src_slots_list:
            pbsc_src_slots = torch.cat(src_slots_list)
            pbsc_dst_slots = torch.cat(dst_slots_list)
        else:
            pbsc_src_slots = None
            pbsc_dst_slots = None

        return BeamAttentionMetadata(
            num_actual_tokens=common_meta.num_actual_tokens,
            max_query_len=common_meta.max_query_len,
            query_start_loc=common_meta.query_start_loc,
            max_seq_len=common_meta.max_seq_len,
            seq_lens=common_meta.seq_lens,
            block_table=common_meta.block_table_tensor,
            slot_mapping=common_meta.slot_mapping,
            # Shared/Split specific fields
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            scheduler_metadata=scheduler_metadata,
            prefix_block_table=prefix_block_table,
            suffix_block_table=suffix_block_table,
            prefix_query_max_len=prefix_query_max_len,
            suffix_query_max_len=suffix_query_max_len,
            prefix_kv_max_len=prefix_kv_max_len,
            suffix_kv_max_len=suffix_kv_max_len,
            prefix_indices=prefix_indices,
            prefix_indices_is_identity=prefix_indices_is_identity,
            group_first_reqs=group_first_reqs,
            suffix_col_indices=suffix_col_indices,
            pbsc_src_slots=pbsc_src_slots,
            pbsc_dst_slots=pbsc_dst_slots,
            causal=common_meta.causal,
            use_cascade_attention=True,
            max_num_splits=max_num_splits,
        )

    def _create_prefix_metadata(self, common_meta: CommonAttentionMetadata, shared_groups: list):
        """Constructs metadata for the shared prefix pass."""
        group_prefix_tokens, group_req_ids_list = zip(*shared_groups) if shared_groups else ([], [])

        # Convert token depths to block depths for cascade attention prefix
        group_prefix_tokens_tensor = torch.tensor(group_prefix_tokens, dtype=torch.int32)
        group_depths_tensor = group_prefix_tokens_tensor // self.block_size
        
        prefix_kv_lens_cpu = group_depths_tensor * self.block_size
        prefix_kv_max_len = (
            int(prefix_kv_lens_cpu.max().item()) if prefix_kv_lens_cpu.numel() > 0 else 0
        )
        max_prefix_blocks = (prefix_kv_max_len + self.block_size - 1) // self.block_size

        # Flatten req ids for vector operations
        flat_req_ids_list = [req_id for req_ids in group_req_ids_list for req_id in req_ids]

        num_reqs = common_meta.num_reqs
        prefix_indices_is_identity = (len(flat_req_ids_list) == num_reqs) and (
            flat_req_ids_list == list(range(num_reqs))
        )

        # Calculate query lengths for prefix indices on CPU to avoid sync
        flat_req_ids_tensor_cpu = torch.tensor(flat_req_ids_list, dtype=torch.long)
        starts_cpu = common_meta.query_start_loc_cpu[flat_req_ids_tensor_cpu]
        ends_cpu = common_meta.query_start_loc_cpu[flat_req_ids_tensor_cpu + 1]
        lengths_cpu = ends_cpu - starts_cpu

        cumsum_lengths_cpu = torch.zeros(lengths_cpu.numel() + 1, dtype=torch.long)
        torch.cumsum(lengths_cpu, dim=0, out=cumsum_lengths_cpu[1:])

        if not prefix_indices_is_identity:
            # Generate prefix indices (scatter map) on CPU and move to GPU async
            total_prefix_tokens = cumsum_lengths_cpu[-1].item()
            arrange_cpu = torch.arange(total_prefix_tokens, dtype=torch.long)

            # Vectorized offset calculation
            offsets_cpu = (starts_cpu - cumsum_lengths_cpu[:-1]).repeat_interleave(lengths_cpu)
            prefix_indices_cpu = arrange_cpu + offsets_cpu
            prefix_indices = prefix_indices_cpu.pin_memory().to(self.device, non_blocking=True)
        else:
            prefix_indices = None

        # Calculate CU lens for prefix groups on CPU (Vectorized)
        group_req_counts = [len(req_ids) for req_ids in group_req_ids_list]
        group_req_counts_tensor = torch.tensor(group_req_counts, dtype=torch.long)

        # Calculate boundaries and segment sums without split/loop or cat
        gather_indices = torch.zeros(len(group_req_counts) + 1, dtype=torch.long)
        torch.cumsum(group_req_counts_tensor, dim=0, out=gather_indices[1:])

        # Gather sums at boundaries
        cum_lens_at_boundaries = cumsum_lengths_cpu[gather_indices]
        group_token_lens_cpu = torch.diff(cum_lens_at_boundaries).to(torch.int32)

        cu_prefix_query_lens_cpu = torch.zeros(
            len(group_token_lens_cpu) + 1, dtype=torch.int32, pin_memory=True
        )
        torch.cumsum(group_token_lens_cpu, dim=0, out=cu_prefix_query_lens_cpu[1:])
        cu_prefix_query_lens = cu_prefix_query_lens_cpu.to(self.device, non_blocking=True)

        # Tensors
        prefix_kv_lens = (
            prefix_kv_lens_cpu.to(torch.int32).pin_memory().to(self.device, non_blocking=True)
        )

        # Prefix Block Table
        # Get first req indices from flattened list using gather_indices
        first_req_indices = gather_indices[:-1]
        group_first_reqs_tensor_cpu = flat_req_ids_tensor_cpu[first_req_indices]
        group_first_reqs_tensor = group_first_reqs_tensor_cpu.pin_memory().to(
            self.device, non_blocking=True
        )
        prefix_block_table = common_meta.block_table_tensor[group_first_reqs_tensor][
            :, :max_prefix_blocks
        ].contiguous()

        # Max lengths
        prefix_query_max_len = (
            int(group_token_lens_cpu.max().item()) if group_token_lens_cpu.numel() > 0 else 0
        )

        # Shift amounts for suffix calculation (Vectorized)
        shift_amounts_cpu = torch.zeros(num_reqs, dtype=torch.int32, pin_memory=True)
        if len(group_depths_tensor) > 0:
            depths_expanded = torch.repeat_interleave(group_depths_tensor, group_req_counts_tensor)
            shift_amounts_cpu[flat_req_ids_tensor_cpu] = depths_expanded
        shift_amounts = shift_amounts_cpu.to(self.device, non_blocking=True)

        return (
            prefix_block_table,
            cu_prefix_query_lens,
            prefix_kv_lens,
            prefix_indices,
            prefix_query_max_len,
            prefix_kv_max_len,
            shift_amounts,
            shift_amounts_cpu,
            prefix_indices_is_identity,
            group_first_reqs_tensor,
        )

    def _create_suffix_metadata(
        self,
        common_meta: CommonAttentionMetadata,
        shift_amounts: torch.Tensor,
        shift_amounts_cpu: torch.Tensor,
    ):
        """Constructs metadata for the individual suffix pass."""
        # Calculate suffix KV lengths
        suffix_kv_lens = torch.clamp(
            common_meta.seq_lens - (shift_amounts * self.block_size), min=0
        )

        seq_lens_cpu = getattr(common_meta, "_seq_lens_cpu", None)
        if seq_lens_cpu is not None:
            # Calculate suffix KV lengths on CPU (used for max length without sync)
            suffix_kv_lens_cpu = torch.clamp(
                seq_lens_cpu - (shift_amounts_cpu * self.block_size), min=0
            )
            suffix_kv_max_len = (
                int(suffix_kv_lens_cpu.max().item()) if suffix_kv_lens_cpu.numel() > 0 else 0
            )
        else:
            # Fallback to GPU tensor (will cause a host-device sync)
            suffix_kv_max_len = (
                int(suffix_kv_lens.max().item()) if suffix_kv_lens.numel() > 0 else 0
            )

        # Max lengths
        suffix_query_max_len = common_meta.max_query_len
        max_suffix_blocks = (suffix_kv_max_len + self.block_size - 1) // self.block_size

        # Create Suffix Block Table by shifting
        max_blocks_orig = common_meta.block_table_tensor.shape[1]
        col_indices = torch.arange(
            max_suffix_blocks, device=self.device, dtype=torch.int32
        ).unsqueeze(0) + shift_amounts.unsqueeze(1)
        col_indices = torch.clamp(col_indices, max=max_blocks_orig - 1)
        suffix_block_table = torch.gather(
            common_meta.block_table_tensor, 1, col_indices
        ).contiguous()

        return (
            suffix_block_table,
            suffix_kv_lens,
            suffix_query_max_len,
            suffix_kv_max_len,
            col_indices,
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

        if new_metadata.use_cascade_attention:
            if new_metadata.group_first_reqs is not None:
                new_metadata.prefix_block_table = blk_table[
                    new_metadata.group_first_reqs
                ].contiguous()

            if new_metadata.suffix_col_indices is not None:
                new_metadata.suffix_block_table = torch.gather(
                    blk_table, 1, new_metadata.suffix_col_indices
                ).contiguous()
        return new_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return True  # This backend is built upon cascade attention logic


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


class BeamAttentionImpl(AttentionImpl):
    """Graph Reuse Attention Implementation.

    Uses cascade_attention as default, with optional Beam buffer optimization.
    """

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

    @staticmethod
    def cascade_attention(
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cu_prefix_query_lens: torch.Tensor,
        cu_suffix_query_lens: torch.Tensor,
        prefix_kv_lens: torch.Tensor,
        suffix_kv_lens: torch.Tensor,
        softmax_scale: float,
        sliding_window: tuple[int, int] | None = None,
        logits_soft_cap: float | None = None,
        max_num_splits: int = 0,
        fa_version: int = 2,
        block_table: torch.Tensor | None = None,
        prefix_block_table: torch.Tensor | None = None,
        suffix_block_table: torch.Tensor | None = None,
        prefix_query_max_len: int = 0,
        suffix_query_max_len: int = 0,
        prefix_kv_max_len: int = 0,
        suffix_kv_max_len: int = 0,
        prefix_indices: torch.Tensor | None = None,
        prefix_indices_is_identity: bool = True,
        prefix_scheduler_metadata: dict | None = None,
        suffix_scheduler_metadata: dict | None = None,
        q_descale_prefix: torch.Tensor | None = None,
        k_descale_prefix: torch.Tensor | None = None,
        v_descale_prefix: torch.Tensor | None = None,
        q_descale_suffix: torch.Tensor | None = None,
        k_descale_suffix: torch.Tensor | None = None,
        v_descale_suffix: torch.Tensor | None = None,
        s_aux: torch.Tensor | None = None,
        alibi_slopes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. Run Prefix Attention (Shared Blocks)
        prefix_out, prefix_lse = None, None
        has_work = prefix_indices_is_identity or (
            prefix_indices is not None and prefix_indices.numel() > 0
        )

        if has_work:
            # Optimization: Group queries that share the same prefix into a single "sequence".
            # This allows the attention kernel to load the shared KV cache blocks once and
            # compute attention for all queries in the group simultaneously.
            if prefix_indices_is_identity:
                prefix_query = query
            else:
                prefix_query = query[prefix_indices]

            prefix_out, prefix_lse = flash_attn_varlen_func(
                q=prefix_query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=cu_prefix_query_lens,
                seqused_k=prefix_kv_lens,
                max_seqlen_q=prefix_query_max_len,
                max_seqlen_k=prefix_kv_max_len,
                softmax_scale=softmax_scale,
                causal=False,
                alibi_slopes=alibi_slopes,
                window_size=sliding_window,
                block_table=prefix_block_table,
                softcap=logits_soft_cap,
                return_softmax_lse=True,
                scheduler_metadata=prefix_scheduler_metadata,
                fa_version=fa_version,
                q_descale=q_descale_prefix,
                k_descale=k_descale_prefix,
                v_descale=v_descale_prefix,
                s_aux=s_aux,
                num_splits=1 if vllm_is_batch_invariant() else max_num_splits,
            )

        # 2. Run Suffix Attention (Non-shared Blocks)
        suffix_out, suffix_lse = flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_suffix_query_lens,
            seqused_k=suffix_kv_lens,
            max_seqlen_q=suffix_query_max_len,
            max_seqlen_k=suffix_kv_max_len,
            softmax_scale=softmax_scale,
            causal=True,
            alibi_slopes=alibi_slopes,
            window_size=sliding_window,
            block_table=suffix_block_table,
            softcap=logits_soft_cap,
            return_softmax_lse=True,
            scheduler_metadata=suffix_scheduler_metadata,
            fa_version=fa_version,
            q_descale=q_descale_suffix,
            k_descale=k_descale_suffix,
            v_descale=v_descale_suffix,
            num_splits=1 if vllm_is_batch_invariant() else max_num_splits,
            out=output,
        )

        # 3. Merge Results
        if prefix_out is not None:
            if prefix_indices_is_identity:
                merge_attn_states(output, prefix_out, prefix_lse, suffix_out, suffix_lse)
            else:
                merge_indexed_attn_states(
                    output, suffix_lse, prefix_out, prefix_lse, prefix_indices
                )

        return output

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
        """Forward pass using cascade_attention as default.

        Uses Beam buffer optimization when shared blocks are detected.

        Args:
            query: [num_tokens, num_heads, head_size]
            key: [num_tokens, num_kv_heads, head_size]
            value: [num_tokens, num_kv_heads, head_size]
            kv_cache: [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Beam attention metadata
            output: Pre-allocated output buffer
            output_scale: Optional FP8 output scale
            output_block_scale: Optional FP8 output block scale

        Returns:
            Attention output [num_tokens, num_heads, head_size]
        """
        assert output is not None, "Output tensor must be provided."
        assert self.vllm_flash_attn_version is not None, "FlashAttention version not detected."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for BeamAttentionImpl"
            )

        if attn_metadata is None:
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens
        attn_type = self.attn_type

        # Handle encoder attention
        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
                attn_metadata.max_num_splits,
            )

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
            
            # -------------------------------------------------------------
            # PBSC KV Hook: Broadcast Leader's Tail Tokens to Children
            # -------------------------------------------------------------
            if getattr(attn_metadata, "pbsc_src_slots", None) is not None:
                src_slots = attn_metadata.pbsc_src_slots
                dst_slots = attn_metadata.pbsc_dst_slots
                
                k_flat = key_cache.view(-1, self.num_kv_heads, self.head_size)
                v_flat = value_cache.view(-1, self.num_kv_heads, self.head_size)
                
                k_flat[dst_slots] = k_flat[src_slots]
                v_flat[dst_slots] = v_flat[src_slots]

        # Handle FP8 quantization if needed
        if self.kv_cache_dtype.startswith("fp8"):
            dtype = BeamAttentionBackend.get_fp8_dtype_for_flashattn(self.kv_cache_dtype)
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        sliding_window_size = list(self.sliding_window) if self.sliding_window is not None else None

        if attn_metadata.prefix_indices is not None or attn_metadata.prefix_indices_is_identity:
            descale_shape_prefix = (
                attn_metadata.cu_prefix_query_lens.shape[0] - 1,
                self.num_kv_heads,
            )
            descale_shape_suffix = (attn_metadata.query_start_loc.shape[0] - 1, self.num_kv_heads)

            self.cascade_attention(
                output=output[:num_actual_tokens],
                query=query[:num_actual_tokens],
                key_cache=key_cache,
                value_cache=value_cache,
                cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
                cu_suffix_query_lens=attn_metadata.query_start_loc,
                prefix_kv_lens=attn_metadata.prefix_kv_lens,
                suffix_kv_lens=attn_metadata.suffix_kv_lens,
                softmax_scale=self.scale,
                sliding_window=sliding_window_size,
                logits_soft_cap=self.logits_soft_cap,
                max_num_splits=attn_metadata.max_num_splits,
                fa_version=self.vllm_flash_attn_version,
                block_table=attn_metadata.block_table,
                prefix_block_table=attn_metadata.prefix_block_table,
                suffix_block_table=attn_metadata.suffix_block_table,
                prefix_query_max_len=attn_metadata.prefix_query_max_len,
                suffix_query_max_len=attn_metadata.suffix_query_max_len,
                prefix_kv_max_len=attn_metadata.prefix_kv_max_len,
                suffix_kv_max_len=attn_metadata.suffix_kv_max_len,
                prefix_indices=attn_metadata.prefix_indices,
                prefix_indices_is_identity=attn_metadata.prefix_indices_is_identity,
                prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
                suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
                q_descale_prefix=layer._q_scale.expand(descale_shape_prefix)
                if layer._q_scale is not None
                else None,
                k_descale_prefix=layer._k_scale.expand(descale_shape_prefix)
                if layer._k_scale is not None
                else None,
                v_descale_prefix=layer._v_scale.expand(descale_shape_prefix)
                if layer._v_scale is not None
                else None,
                q_descale_suffix=layer._q_scale.expand(descale_shape_suffix)
                if layer._q_scale is not None
                else None,
                k_descale_suffix=layer._k_scale.expand(descale_shape_suffix)
                if layer._k_scale is not None
                else None,
                v_descale_suffix=layer._v_scale.expand(descale_shape_suffix)
                if layer._v_scale is not None
                else None,
                s_aux=self.sinks,
                alibi_slopes=self.alibi_slopes,
            )
        else:
            # Standard Attention (Fallback)
            descale_shape = (attn_metadata.query_start_loc.shape[0] - 1, self.num_kv_heads)
            flash_attn_varlen_func(
                q=query[:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_actual_tokens],
                cu_seqlens_q=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                seqused_k=attn_metadata.seq_lens,
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,
                window_size=sliding_window_size,
                block_table=attn_metadata.block_table,
                softcap=self.logits_soft_cap,
                scheduler_metadata=attn_metadata.scheduler_metadata,
                fa_version=self.vllm_flash_attn_version,
                q_descale=layer._q_scale.expand(descale_shape)
                if layer._q_scale is not None
                else None,
                k_descale=layer._k_scale.expand(descale_shape)
                if layer._k_scale is not None
                else None,
                v_descale=layer._v_scale.expand(descale_shape)
                if layer._v_scale is not None
                else None,
                num_splits=attn_metadata.max_num_splits,
                s_aux=self.sinks,
            )

        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: BeamAttentionMetadata,
        layer: torch.nn.Module,
        max_num_splits: int = 0,
    ) -> torch.Tensor:
        """Forward pass for encoder attention."""
        assert self.vllm_flash_attn_version is not None

        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError("quantization is not supported for encoder attention")

        cu_seqlens_q = attn_metadata.query_start_loc
        cu_seqlens_k = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_query_len

        descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

        sliding_window_size = list(self.sliding_window) if self.sliding_window is not None else None

        flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            softcap=self.logits_soft_cap,
            scheduler_metadata=attn_metadata.scheduler_metadata,
            fa_version=self.vllm_flash_attn_version,
            q_descale=layer._q_scale.expand(descale_shape) if layer._q_scale is not None else None,
            k_descale=layer._k_scale.expand(descale_shape) if layer._k_scale is not None else None,
            v_descale=layer._v_scale.expand(descale_shape) if layer._v_scale is not None else None,
            num_splits=1 if vllm_is_batch_invariant() else max_num_splits,
        )

        return output
