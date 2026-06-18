# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for EngineCore / EngineCoreProc to support
ADD_BATCH and MEGA_REQUEST_STEP_UPDATE."""

from __future__ import annotations
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple
import torch

from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreRequestType

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# EngineCore.__init__ wrapper
# ---------------------------------------------------------------------------


def make_patched_engine_core_init(original_init):
    """Wrap EngineCore.__init__ to add beam state."""
    import threading

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.beam_cache: dict[str, dict] = {}
        self.beam_cache_lock = threading.Lock()

    return patched_init


# ---------------------------------------------------------------------------
# Helpers: add enum members to EngineCoreRequestType at runtime
# ---------------------------------------------------------------------------


def _add_enum_member(name: str, value: bytes) -> None:
    """Dynamically add a member to EngineCoreRequestType if absent.

    This uses object.__new__() to bypass enum's normal creation pathway,
    which is necessary because we're monkey-patching at runtime and cannot
    modify the original class definition in vLLM's codebase.

    While this approach manipulates enum internals, it:
    - Is a standard technique used by enum extension libraries (e.g. aenum)
    - Includes validation to detect incompatibilities early
    - Is contained to initialization time (not runtime performance critical)
    - Is tested to ensure correctness

    Args:
        name: The name of the new enum member (e.g., "ADD_BATCH")
        value: The byte value for the enum member (e.g., b"\\x05")

    Raises:
        RuntimeError: If the enum's internal structure is incompatible
    """
    # Skip if already added
    if hasattr(EngineCoreRequestType, name):
        return

    # Validate that the enum has expected internal structure
    required_attrs = ["_value2member_map_", "_member_map_", "_member_names_"]
    for attr in required_attrs:
        if not hasattr(EngineCoreRequestType, attr):
            raise RuntimeError(
                f"EngineCoreRequestType missing expected attribute '{attr}'. "
                f"This indicates an incompatible Python/vLLM version. "
                f"Please report this issue with your Python version."
            )

    # Verify the value type matches existing members
    if EngineCoreRequestType._value2member_map_:
        existing_value_type = type(next(iter(EngineCoreRequestType._value2member_map_.keys())))
        if not isinstance(value, existing_value_type):
            raise TypeError(
                f"Value type mismatch: expected {existing_value_type.__name__}, "
                f"got {type(value).__name__}"
            )

    # Create the member instance bypassing enum's __new__
    # (This is the standard approach for runtime enum extension)
    member = object.__new__(EngineCoreRequestType)
    member._name_ = name
    member._value_ = value

    # Register in enum's internal maps
    EngineCoreRequestType._value2member_map_[value] = member
    EngineCoreRequestType._member_map_[name] = member
    EngineCoreRequestType._member_names_.append(name)

    # Set as class attribute (must use type.__setattr__ to bypass enum's __setattr__)
    type.__setattr__(EngineCoreRequestType, name, member)


# ---------------------------------------------------------------------------
# Child-process patch applicator — called from patched run_engine_core
# ---------------------------------------------------------------------------


def apply_engine_core_child_patches():
    """Apply all patches inside the EngineCore child process.

    This must be called at the start of run_engine_core (the child process
    entry point) because multiprocessing 'spawn' starts a fresh Python
    interpreter that does not inherit monkey patches from the parent.
    """
    import vllm.v1.engine as engine_mod
    from vllm.v1.engine.core import EngineCore, EngineCoreProc

    from vllm_gr.v1.engine.core import (
        _cache_beam_request,
        _handle_mega_request_step_update,
        process_input_sockets,
    )
    

    # Add new enum members: ADD_BATCH, MEGA_REQUEST_STEP_UPDATE
    _add_enum_member("ADD_BATCH", b"\x05")
    _add_enum_member("MEGA_REQUEST_STEP_UPDATE", b"\x06")

    # Wrap EngineCore.__init__ to add beam state
    EngineCore.__init__ = make_patched_engine_core_init(EngineCore.__init__)

    # Bind _cache_beam_request and _handle_mega_request_step_update on both classes
    EngineCore._cache_beam_request = _cache_beam_request
    EngineCore._handle_mega_request_step_update = _handle_mega_request_step_update

    # Replace EngineCoreProc.process_input_sockets
    EngineCoreProc.process_input_sockets = process_input_sockets

    from vllm.sampling_params import SamplingParams
    # Force the greedy verification check to be a no-op to allow n > 1 under greedy beam search parameters
    SamplingParams._verify_greedy_sampling = lambda self: None

    # Apply scheduler patches in the child process
    apply_scheduler_patch()

    # Apply worker patches (e.g., GPUModelRunner)
    apply_worker_patches()

    logger.debug("Engine-core patches applied in child process.")


# Re-entry guard to detect recursive calls
_run_engine_core_active = False


def run_engine_core(*args, **kwargs):
    """Wrapper for EngineCoreProc.run_engine_core that applies patches
    before the engine starts.  This is a module-level function so that
    multiprocessing 'spawn' can pickle it by qualified name.

    In the spawned child process, all modules are freshly imported, so
    EngineCoreProc.run_engine_core is the original unpatched version.
    We apply our patches first, then delegate to the original.
    """
    global _run_engine_core_active

    # Re-entry guard: detect if we're already inside run_engine_core
    if _run_engine_core_active:
        raise RuntimeError(
            "run_engine_core re-entry detected. This indicates a recursive "
            "call, likely due to patching being applied multiple times. "
            "Check that patch_batch_and_fork() is only called once."
        )

    from vllm.v1.engine.core import EngineCoreProc

    # Guard against recursion: if vllm_gr/init.py was auto-imported in
    # the child process, run_patch() will have replaced
    # EngineCoreProc.run_engine_core with *this* function again.
    # Use the saved original stashed by patch_batch_and_fork() if
    # available; otherwise fall back to the current class attribute
    # (which is the unpatched original in a fresh child interpreter).
    original_run = getattr(run_engine_core, "_original_run_engine_core", None)

    # Validate the saved original
    if original_run is not None and original_run is not run_engine_core:
        # We have a saved original that isn't ourselves - validate it
        if not callable(original_run):
            raise RuntimeError(
                f"run_engine_core._original_run_engine_core is not callable: "
                f"type={type(original_run)}"
            )
    else:
        # No saved original (or it points to ourselves, which is invalid)
        # Fall back to the current class attribute (should be the original in a fresh process)
        if original_run is run_engine_core:
            logger.warning(
                "run_engine_core._original_run_engine_core points to itself. "
                "This may indicate corrupted patch state from a previous run. "
                "Falling back to EngineCoreProc.run_engine_core."
            )
        original_run = EngineCoreProc.run_engine_core

    # Final validation: ensure we didn't end up with ourselves
    if original_run is run_engine_core:
        raise RuntimeError(
            "run_engine_core recursion detected: EngineCoreProc.run_engine_core "
            "points to this wrapper. This indicates that patching was applied "
            "but the original was not saved, or the child process imported "
            "vllm_gr before starting."
        )

    # Validate that original_run looks like a real function from vllm
    # (Skip validation for test mocks)
    if hasattr(original_run, "__module__"):
        original_module = original_run.__module__
        expected_module = "vllm.v1.engine.core"
        # Skip warning for unittest mocks (used in tests)
        if original_module not in (expected_module, "unittest.mock", "mock"):
            logger.warning(
                "Unexpected module for original run_engine_core: %s (expected %s). "
                "Proceeding anyway, but this may indicate an issue.",
                original_module,
                expected_module,
            )

    # Set re-entry guard and apply patches
    _run_engine_core_active = True
    try:
        apply_engine_core_child_patches()
        original_run(*args, **kwargs)
    finally:
        _run_engine_core_active = False


def apply_scheduler_patch():
    """Monkey-patch Scheduler.schedule to propagate custom attributes via SchedulerOutput."""
    from functools import wraps
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler, "_patched_for_mega_requests", False):
        return

    _original_schedule = Scheduler.schedule

    @wraps(_original_schedule)
    def patched_schedule(self):
        
        _original_get_computed_blocks = self.kv_cache_manager.get_computed_blocks
        block_size = self.cache_config.block_size
        # TODO: Add support for cache > prefix
        def patch_get_computed_blocks(req):
            bn, tn = _original_get_computed_blocks(req)
            if req and getattr(req, 'is_mega_decode', False):
                pn = getattr(req, "prefix_len", 0) // block_size * block_size
                if tn > pn:
                    sliced_blocks = tuple(group[:pn//block_size] for group in bn.blocks)
                    return self.kv_cache_manager.create_kv_cache_blocks(sliced_blocks), pn
            return bn, tn

        self.kv_cache_manager.get_computed_blocks = patch_get_computed_blocks
        try:
            output = _original_schedule(self)
            
            # Propagate mega request attributes from Request objects to SchedulerOutput
            mega_data = {}
            if hasattr(output, "num_scheduled_tokens"):
                for req_id in output.num_scheduled_tokens.keys():
                    req = self.requests.get(req_id)
                    if req and getattr(req, 'is_mega_decode', False):
                        mega_data[req_id] = {
                            "is_mega_decode": True,
                            "mega_beam_width": getattr(req, "mega_beam_width", 1),
                            "mega_decode_steps": getattr(req, "mega_decode_steps", 0),
                            "prefix_len": getattr(req, "prefix_len", 0),
                            "cache_len":req.num_computed_tokens - output.num_scheduled_tokens[req_id],
                        }
            output.mega_data = mega_data
            return output
        finally:
            self.kv_cache_manager.get_computed_blocks = _original_get_computed_blocks

    Scheduler.schedule = patched_schedule
    Scheduler._patched_for_mega_requests = True
    logger.debug("Scheduler.schedule patched for mega-requests propagation.")


# ============================================================================
# TENSOR AND BUFFER GEOMETRY EXTRACTION HELPERS
# ============================================================================

def _get_positions_buffers(runner: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Safely extracts underlying hardware and host indexing position tensors."""
    gpu_positions = None
    cpu_positions = None
    if hasattr(runner, "positions") and runner.positions is not None:
        gpu_positions = getattr(runner.positions, "gpu", None)
        cpu_positions = getattr(runner.positions, "cpu", None)
    return gpu_positions, cpu_positions


def _compute_beam_bounds(
    b: int, 
    prefix_len: int, 
    cache_len: int, 
    decode_steps: int, 
    chunk_budget: int, 
    delta: int
) -> Tuple[int, int]:
    """Calculates active lookahead allocations and historical trace lengths."""
    past_suffix_b = max(0, min(decode_steps, delta - b * decode_steps))
    if prefix_len >= cache_len:
        remaining_prefix = prefix_len - cache_len
        if b == 0:
            return min(chunk_budget, remaining_prefix + decode_steps), past_suffix_b
        return min(chunk_budget, decode_steps), past_suffix_b

    remaining_suffix_b = decode_steps - past_suffix_b
    return min(chunk_budget, remaining_suffix_b), past_suffix_b


def _overwrite_position_ids(
    st: int, 
    b_uncached: int, 
    curr_offset: int, 
    gpu_pos: Optional[torch.Tensor], 
    cpu_pos: Optional[torch.Tensor]
) -> None:
    """Overwrites positional indexing structures in place on targets."""
    if gpu_pos is not None:
        gpu_pos[curr_offset : curr_offset + b_uncached] = torch.arange(
            st, st + b_uncached, dtype=gpu_pos.dtype, device=gpu_pos.device
        )
    if cpu_pos is not None:
        cpu_pos[curr_offset : curr_offset + b_uncached] = torch.arange(
            st, st + b_uncached, dtype=cpu_pos.dtype, device=torch.device("cpu")
        )


def _process_mega_decode_request(
    mdata: Dict[str, Any],
    seq_len: int,
    token_offset: int,
    gpu_pos: Optional[torch.Tensor],
    cpu_pos: Optional[torch.Tensor],
    new_logits_indices: List[int]
) -> Tuple[int, int, int]:
    """Traces allocation arrays and mutates position buffers for a speculative request."""
    prefix_len = mdata['prefix_len']
    beam_width = mdata['mega_beam_width']
    decode_steps = mdata['mega_decode_steps']
    cache_len = mdata['cache_len']

    chunk_budget = seq_len
    delta = max(0, cache_len - prefix_len)
    curr_offset = token_offset
    valid_rows_for_req = 0
    logits_row_cnt = 0
    # print(f"prefix_len={prefix_len}, cache_len={cache_len}, decode_steps={decode_steps}")
    for b in range(beam_width):
        if chunk_budget <= 0:
            break
            
        b_uncached, past_suffix_b = _compute_beam_bounds(
            b, prefix_len, cache_len, decode_steps, chunk_budget, delta
        )
        
        if b_uncached > 0:
            st = cache_len if (prefix_len >= cache_len and b == 0) else (
                prefix_len if prefix_len >= cache_len else prefix_len + past_suffix_b
            )
            _overwrite_position_ids(st, b_uncached, curr_offset, gpu_pos, cpu_pos)

            if past_suffix_b + b_uncached >= decode_steps:
                new_logits_indices.append(curr_offset + b_uncached - 1)
                logits_row_cnt += 1
                
            # print(f"\tb={b}, b_uncached={b_uncached}, past_suffix_b={past_suffix_b}, st={st}, curr_offset={curr_offset} logit {curr_offset + b_uncached - 1}")
            valid_rows_for_req += 1
            curr_offset += b_uncached
            chunk_budget -= b_uncached

    return valid_rows_for_req, logits_row_cnt, curr_offset


# ============================================================================
# CORE WORKER PIPELINE WELDER LOGIC
# ============================================================================

def _remux_logprobs_tensors(
    request_row_configs: List[Tuple[Any, int, int]], 
    max_chained_dim: int, 
    target_K: int, 
    orig_lps: torch.Tensor, 
    orig_lp_ids: Optional[torch.Tensor], 
    orig_ranks: Optional[torch.Tensor],
    st_ids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Merges and aligns sampler tensor slices sequentially across runtime waves."""
    if max_chained_dim == 0 or orig_lps.numel() == 0:
        return st_ids, orig_lps, orig_lp_ids, orig_ranks

    new_st_ids, new_lps, new_lp_ids, new_ranks = [], [], [], []
    gpu_row_cursor = 0

    for req_id, _, W_logits in request_row_configs:
        # --------------------------------------------------------------------
        # CASE A: INTERMEDIATE CHUNKED PREFILL (No output rows generated)
        # --------------------------------------------------------------------
        if W_logits == 0:
            dummy_st = torch.tensor([[-1]], dtype=st_ids.dtype, device=st_ids.device)
            new_st_ids.append(dummy_st)

            new_lps.append(torch.full((1, max_chained_dim), float('-inf'), dtype=orig_lps.dtype, device=orig_lps.device))
            if orig_lp_ids is not None:
                new_lp_ids.append(torch.zeros((1, max_chained_dim), dtype=orig_lp_ids.dtype, device=orig_lp_ids.device))
            if orig_ranks is not None:
                new_ranks.append(torch.zeros((1, max_chained_dim), dtype=orig_ranks.dtype, device=orig_ranks.device))
            continue

        # --------------------------------------------------------------------
        # CASE B: ACTIVE SAMPLING PATH (Captures the full multi-beam footprint)
        # --------------------------------------------------------------------
        # Retain leader token track identity
        new_st_ids.append(st_ids[gpu_row_cursor : gpu_row_cursor + 1, :])

        req_lps = orig_lps[gpu_row_cursor : gpu_row_cursor + W_logits, :target_K].reshape(1, W_logits * target_K)
        if W_logits * target_K < max_chained_dim:
            req_lps = torch.nn.functional.pad(req_lps, (0, max_chained_dim - (W_logits * target_K)), value=float('-inf'))
        new_lps.append(req_lps)

        if orig_lp_ids is not None:
            req_lp_ids = orig_lp_ids[gpu_row_cursor : gpu_row_cursor + W_logits, :target_K].reshape(1, W_logits * target_K)
            if W_logits * target_K < max_chained_dim:
                req_lp_ids = torch.nn.functional.pad(req_lp_ids, (0, max_chained_dim - (W_logits * target_K)), value=0)
            new_lp_ids.append(req_lp_ids)

        if orig_ranks is not None:
            req_ranks = orig_ranks[gpu_row_cursor : gpu_row_cursor + W_logits].reshape(1, W_logits)
            if W_logits < max_chained_dim:
                req_ranks = torch.nn.functional.pad(req_ranks, (0, max_chained_dim - W_logits), value=0)
            new_ranks.append(req_ranks)

        # Advance the pointer cursor past the complete multi-beam block size
        gpu_row_cursor += W_logits

    return (
        torch.cat(new_st_ids, dim=0),
        torch.cat(new_lps, dim=0),
        torch.cat(new_lp_ids, dim=0) if orig_lp_ids is not None else None,
        torch.cat(new_ranks, dim=0) if orig_ranks is not None else None
    )


# ============================================================================
# INLINE COMPILATION PROXIES FOR HIGH-PERFORMANCE INTERCEPTION
# ============================================================================

class MegaReqIdsProxy(list):
    def __init__(self, original_list, req_id_map):
        super().__init__(original_list)
        self.req_id_map = req_id_map
    def __getitem__(self, index):
        return self.req_id_map[index] if index < len(self.req_id_map) else (self.req_id_map[-1] if self.req_id_map else None)
    def copy(self): return list(self.req_id_map)


class Mock1DArray:
    def __init__(self, original, batch_idx_map):
        self.original = original
        self.batch_idx_map = batch_idx_map
    def __getitem__(self, idx):
        return self.original[self.batch_idx_map[idx]] if idx < len(self.batch_idx_map) else self.original[0]
    def __setitem__(self, idx, value):
        if idx < len(self.batch_idx_map):
            self.original[self.batch_idx_map[idx]] = value


class Mock2DArray:
    def __init__(self, original, batch_idx_map):
        self.original = original
        self.batch_idx_map = batch_idx_map
    def __setitem__(self, key, value):
        idx, slc = key
        if idx < len(self.batch_idx_map):
            self.original[self.batch_idx_map[idx], slc] = value
    def __getattr__(self, name): return getattr(self.original, name)
    def __getitem__(self, key):
        idx, slc = key
        actual_idx = self.batch_idx_map[idx] if idx < len(self.batch_idx_map) else 0
        return self.original[actual_idx, slc]


class InputBatchProxy:
    def __init__(self, obj, req_map, idx_map, logits_map):
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_req_id_map", req_map)
        object.__setattr__(self, "_batch_idx_map", idx_map)
        object.__setattr__(self, "_logit_idx_map", logits_map)
        
    def __getattr__(self, name):
        obj = object.__getattribute__(self, "_obj")
        idx_map = object.__getattribute__(self, "_batch_idx_map")
        logits_map = object.__getattribute__(self, "_logit_idx_map")
        req_map = object.__getattribute__(self, "_req_id_map")
        
        if name == "sampling_metadata":
            s_meta = getattr(obj, "sampling_metadata", None)
            if s_meta is None or logits_map is None:
                return s_meta
            
            orig_req_len = len(obj.req_ids) if hasattr(obj, 'req_ids') else 1
            for attr in ["temperature", "top_p", "top_k", "min_p", "frequency_penalties", "presence_penalties", "repetition_penalties"]:
                val = getattr(s_meta, attr, None)
                if isinstance(val, torch.Tensor) and val.numel() > 0:
                    if val.shape[0] == orig_req_len and val.shape[0] != len(logits_map):
                        object.__setattr__(s_meta, attr, val[logits_map])
            return s_meta
            
        elif name == "req_ids": return MegaReqIdsProxy(obj.req_ids, req_map)
        elif name == "num_tokens_no_spec": return Mock1DArray(obj.num_tokens_no_spec, idx_map)
        elif name == "token_ids_cpu": return Mock2DArray(obj.token_ids_cpu, idx_map)
        elif name == "is_token_ids": return Mock2DArray(obj.is_token_ids, idx_map) if hasattr(obj, 'is_token_ids') else Mock2DArray(obj.token_ids_cpu, idx_map)
        return getattr(obj, name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_obj"), name, value)


# ============================================================================
# MONKEY PATCH APPLICATOR REGISTRY
# ============================================================================

def apply_worker_patches():
    """Applies runtime overrides onto GPUModelRunner execution entries."""
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        from vllm_gr.v1.attention.backends.beam_attn import MEGA_DATA_VAR
    except ImportError:
        logger.warning("Failed to import GPUModelRunner segments, skipping patches.")
        return

    if getattr(GPUModelRunner, "_patched_for_mega_beam_pos_ids", False):
        return

    _original_execute_model = GPUModelRunner.execute_model
    _original_prepare_inputs = GPUModelRunner._prepare_inputs
    _original_bookkeeping_sync = GPUModelRunner._bookkeeping_sync

    @wraps(_original_execute_model)
    def patched_execute_model(self, scheduler_output, *args, **kwargs):
        self._current_mega_data = getattr(scheduler_output, "mega_data", {})
        mega_info = {"mega_data": self._current_mega_data, "req_ids": []}
        token = MEGA_DATA_VAR.set(mega_info)
        self._current_mega_info = mega_info
        try:
            return _original_execute_model(self, scheduler_output, *args, **kwargs)
        finally:
            self._current_mega_data = {}
            self._current_mega_info = None
            MEGA_DATA_VAR.reset(token)

    @wraps(_original_prepare_inputs)
    def patched_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        logits_indices, spec_decode_metadata = _original_prepare_inputs(self, scheduler_output, num_scheduled_tokens)

        if getattr(self, "_current_mega_info", None) is not None:
            self._current_mega_info["req_ids"] = self.input_batch.req_ids

        mega_data = getattr(scheduler_output, "mega_data", {})
        if not mega_data:
            return logits_indices, spec_decode_metadata
        
        self.input_batch.logprob_token_ids = {}
        new_logits_indices = []
        orig_req_ids = self.input_batch.req_ids
        
        token_offset = 0    
        row_to_req_id, row_to_batch_idx, logits_to_batch_idx, request_row_configs = [], [], [], []
        input_row_idx = 0

        gpu_positions, cpu_positions = _get_positions_buffers(self)

        for batch_idx, req_id in enumerate(orig_req_ids):
            seq_len = num_scheduled_tokens[batch_idx] if batch_idx < len(num_scheduled_tokens) else 0
            if req_id is None:
                row_to_req_id.append(None)
                row_to_batch_idx.append(batch_idx)
                input_row_idx += 1
                token_offset += seq_len
                continue
                
            mdata = mega_data.get(req_id)
            if mdata and mdata.get('is_mega_decode', False):
                W, W_logits, next_offset = _process_mega_decode_request(
                    mdata, seq_len, token_offset, gpu_positions, cpu_positions, new_logits_indices
                )
                row_to_req_id.extend([req_id] * W)
                row_to_batch_idx.extend([batch_idx] * W)
                logits_to_batch_idx.extend([batch_idx] * W_logits)
                
                request_row_configs.append((req_id, input_row_idx, W_logits))
                input_row_idx += W
                #if(W_logits != mdata['mega_beam_width']):print("Capture W_logits ", W_logits)
            else:
                num_computed_tokens = self.input_batch.num_computed_tokens_cpu[batch_idx]
                num_prompt_tokens = self.input_batch.num_prompt_tokens[batch_idx]


                is_last_chunk_prefill = bool(num_computed_tokens < num_prompt_tokens and num_computed_tokens + seq_len >= num_prompt_tokens)
                W_logits_count = 0
                W = 1
                if is_last_chunk_prefill:
                    new_logits_indices.append(token_offset + seq_len - 1)
                    W_logits_count = 1
                    logits_to_batch_idx.extend([batch_idx] * W)
                # else:
                #     print("Chunk Prefill")
                row_to_req_id.extend([req_id] * W)
                row_to_batch_idx.extend([batch_idx] * W)
                
                request_row_configs.append((req_id, input_row_idx, W_logits_count))
                input_row_idx += W

            token_offset += seq_len
        
        self.input_batch = InputBatchProxy(self.input_batch, row_to_req_id, row_to_batch_idx, logits_to_batch_idx)
        scheduler_output._gr_request_row_configs = request_row_configs
        
        return torch.tensor(new_logits_indices, dtype=torch.int32, device=self.device), spec_decode_metadata
  
    @wraps(_original_bookkeeping_sync)
    def patched_bookkeeping_sync(self, scheduler_output, *args, **kwargs):
        mega_data = getattr(scheduler_output, "mega_data", {})
        if not mega_data or not hasattr(scheduler_output, "_gr_request_row_configs"):
            res = _original_bookkeeping_sync(self, scheduler_output, *args, **kwargs)
            sampler_output = kwargs.get("sampler_output") or (args[0] if args else None)
            if sampler_output is not None:
                st_ids = sampler_output.sampled_token_ids
                lp_tensors = sampler_output.logprobs_tensors
                # print("st_ids: ", st_ids)
                # print("lp_tensors: ", lp_tensors)
            return res
        

        request_row_configs = scheduler_output._gr_request_row_configs
        orig_input_batch = object.__getattribute__(self.input_batch, "_obj")
        
        try:
            res = _original_bookkeeping_sync(self, scheduler_output, *args, **kwargs)
        finally:
            self.input_batch = orig_input_batch

        (num_nans_in_logits, logprobs_lists, valid_sampled_token_ids,
         prompt_logprobs_dict, req_ids_output_copy, req_id_to_index_output_copy, invalid_req_indices) = res

        collapsed_req_ids_copy = [r[0] for r in request_row_configs if r[0] is not None]
        collapsed_req_id_to_index = {req_id: idx for idx, req_id in enumerate(collapsed_req_ids_copy)}
        num_logical_requests = len(collapsed_req_ids_copy)

        sampler_output = kwargs.get("sampler_output") or (args[0] if args else None)
        if sampler_output is not None:
            st_ids = sampler_output.sampled_token_ids
            lp_tensors = sampler_output.logprobs_tensors
            # print("st_ids: ", st_ids)
            # print("lp_tensors: ", lp_tensors)
            if st_ids is not None and lp_tensors is not None and lp_tensors.logprobs is not None:
                orig_lps = lp_tensors.logprobs
                orig_lp_ids = lp_tensors.logprob_token_ids
                orig_ranks = getattr(lp_tensors, "selected_token_ranks", None)                
                actual_K = orig_lps.shape[1]
                max_chained_dim = max(W * actual_K for _, _, W in request_row_configs)

                final_st_ids, final_lps, final_lp_ids, final_ranks = _remux_logprobs_tensors(
                    request_row_configs, max_chained_dim, actual_K, orig_lps, orig_lp_ids, orig_ranks, st_ids
                )
                
                # Safely resizes the original tensor wrapper and copies the elements in place
                st_ids.resize_(final_st_ids.shape).copy_(final_st_ids)
                lp_tensors.logprobs.resize_(final_lps.shape).copy_(final_lps)

                if lp_tensors.logprob_token_ids is not None and final_lp_ids is not None:
                    lp_tensors.logprob_token_ids.resize_(final_lp_ids.shape).copy_(final_lp_ids)
                if orig_ranks is not None and final_ranks is not None:
                    clean_ranks_slice = final_ranks[:num_logical_requests, 0].contiguous()
                    lp_tensors.selected_token_ranks.resize_(clean_ranks_slice.shape).copy_(clean_ranks_slice)
                            
                if getattr(orig_input_batch, "prev_sampled_token_ids", None) is not None:
                    orig_input_batch.prev_sampled_token_ids.resize_(final_st_ids.shape).copy_(final_st_ids)

        return (
            num_nans_in_logits,
            [[] for _ in range(num_logical_requests)] if logprobs_lists is not None else None,
            [[] for _ in range(num_logical_requests)],
            prompt_logprobs_dict,
            collapsed_req_ids_copy,
            collapsed_req_id_to_index,
            invalid_req_indices
        )

    GPUModelRunner.execute_model = patched_execute_model
    GPUModelRunner._prepare_inputs = patched_prepare_inputs
    GPUModelRunner._bookkeeping_sync = patched_bookkeeping_sync
    GPUModelRunner._patched_for_mega_beam_pos_ids = True
    logger.debug("GPUModelRunner hooks patched successfully.")