# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monkey-patches for EngineCore / EngineCoreProc to support
ADD_BATCH and BEAM_FORK."""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreRequestType

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Helper: Pre-compute shared prefix groups for attention beam routing
# ---------------------------------------------------------------------------


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
    _add_enum_member("MEGA_REQUEST_STEP_UPDATE", b"\x08")

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
        import torch
        torch.cuda.nvtx.range_push("Scheduler.schedule")
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
            torch.cuda.nvtx.range_pop()

    Scheduler.schedule = patched_schedule
    Scheduler._patched_for_mega_requests = True
    logger.debug("Scheduler.schedule patched for mega-requests propagation.")


def apply_worker_patches():
    """Monkey-patch GPUModelRunner to fix position IDs, sampling metadata, and bookkeeping for mega-requests."""
    try:
        import torch
        import copy
        from vllm.v1.worker.gpu_input_batch import InputBatch
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        from vllm.sampling_params import SamplingType
    except ImportError:
        logger.warning("Failed to import GPUModelRunner components, skipping worker patches.")
        return

    if getattr(GPUModelRunner, "_patched_for_mega_beam_pos_ids", False):
        return

    if not hasattr(GPUModelRunner, '_prepare_inputs'):
        logger.warning("Required GPUModelRunner tracking methods not found, skipping patches.")
        return
        
    _original_execute_model = GPUModelRunner.execute_model
    _original_prepare_inputs = GPUModelRunner._prepare_inputs
    _original_bookkeeping_sync = GPUModelRunner._bookkeeping_sync

    def patched_execute_model(self, scheduler_output, *args, **kwargs):
        # Stash the mega_data temporarily on the runner instance for the lifecycle turn
        self._current_mega_data = getattr(scheduler_output, "mega_data", {})
        self._current_scheduler_output = scheduler_output
        from vllm_gr.v1.attention.backends.beam_attn import MEGA_DATA_VAR
        mega_info = {
            "mega_data": self._current_mega_data,
            "req_ids": []
        }
        token = MEGA_DATA_VAR.set(mega_info)
        self._current_mega_info = mega_info
        
        try:
            return _original_execute_model(self, scheduler_output, *args, **kwargs)
        finally:
            self._current_mega_data = {}
            self._current_scheduler_output = None
            self._current_mega_info = None
            MEGA_DATA_VAR.reset(token)

    def patched_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        # Initialize native vLLM parameters and populate baseline states
        logits_indices, spec_decode_metadata = _original_prepare_inputs(
            self, scheduler_output, num_scheduled_tokens
        )

        if getattr(self, "_current_mega_info", None) is not None:
            self._current_mega_info["req_ids"] = self.input_batch.req_ids

        mega_data = getattr(scheduler_output, "mega_data", {})
        if not mega_data:
            return logits_indices, spec_decode_metadata

        self.input_batch.logprob_token_ids = {}
        new_logits_indices = []

        token_offset = 0
        sample_slot_offset = 0
        
        # ---------------------------------------------------------------------
        # PHASE 1: PRE-COMPUTE DYNAMIC ROW MAPS & CPU_GPU_BUFFER POSITION IDS
        # ---------------------------------------------------------------------
        orig_req_ids = self.input_batch.req_ids
        row_to_req_id = []     
        row_to_batch_idx = []   
        request_row_configs = []
        input_row_idx = 0

        gpu_positions = None
        cpu_positions = None
        if hasattr(self, "positions") and self.positions is not None:
            gpu_positions = getattr(self.positions, "gpu", None)
            cpu_positions = getattr(self.positions, "cpu", None)

        for batch_idx, req_id in enumerate(orig_req_ids):
            seq_len = num_scheduled_tokens[batch_idx] if batch_idx < len(num_scheduled_tokens) else 0
            if req_id is None:
                row_to_req_id.append(None)
                row_to_batch_idx.append(batch_idx)
                input_row_idx += 1
                token_offset += seq_len
                continue
                
            request = self.requests[req_id]
            req_n = request.sampling_params.n if request.sampling_params else 1
            mdata = mega_data.get(req_id)

            if mdata and mdata.get('is_mega_decode', False):
                prefix_len = mdata['prefix_len']
                beam_width = mdata['mega_beam_width']
                decode_steps = mdata['mega_decode_steps']
                cache_len = mdata['cache_len']

                chunk_budget = seq_len  
                delta = max(0, cache_len - prefix_len)
                
                req_n = beam_width
                mdata['sample_slot_offset'] = sample_slot_offset

                curr_offset = token_offset
                valid_rows_for_req = 0

                for b in range(beam_width):
                    if chunk_budget <= 0:
                        break
                        
                    b_uncached = 0
                    if prefix_len >= cache_len:
                        remaining_prefix = prefix_len - cache_len
                        if b == 0:
                            b_uncached = min(chunk_budget, remaining_prefix + decode_steps)
                        else:
                            if remaining_prefix <= seq_len:
                                b_uncached = min(chunk_budget, decode_steps)
                    else:
                        past_suffix_b = max(0, min(decode_steps, delta - b * decode_steps))
                        remaining_suffix_b = decode_steps - past_suffix_b
                        b_uncached = min(chunk_budget, remaining_suffix_b)

                    if b_uncached > 0:
                        if prefix_len >= cache_len:
                            st = cache_len if b == 0 else prefix_len
                        else:
                            st = prefix_len + (b * decode_steps) + past_suffix_b

                        if gpu_positions is not None:
                            beam_positions_gpu = torch.arange(
                                st, st + b_uncached, 
                                dtype=gpu_positions.dtype, device=gpu_positions.device
                            )
                            gpu_positions[curr_offset : curr_offset + b_uncached] = beam_positions_gpu

                        if cpu_positions is not None:
                            beam_positions_cpu = torch.arange(
                                st, st + b_uncached, 
                                dtype=cpu_positions.dtype, device=torch.device("cpu")
                            )
                            cpu_positions[curr_offset : curr_offset + b_uncached] = beam_positions_cpu

                        new_logits_indices.append(curr_offset + b_uncached - 1)
                        curr_offset += b_uncached
                        chunk_budget -= b_uncached
                        valid_rows_for_req += 1

                W = max(1, valid_rows_for_req)
                row_to_req_id.extend([req_id] * W)
                row_to_batch_idx.extend([batch_idx] * W)
                input_row_idx += W
            else:
                if seq_len > 0:
                    new_logits_indices.append(token_offset + seq_len - 1)
                W = 1
                row_to_req_id.extend([req_id] * W)
                row_to_batch_idx.extend([batch_idx] * W)
                input_row_idx += W

            request_row_configs.append((req_id, input_row_idx - W, W))
            sample_slot_offset += req_n
            token_offset += seq_len

        # ---------------------------------------------------------------------
        # PHASE 2: ROW-MAPPED INTERCEPTOR PROTECTION PROXIES
        # ---------------------------------------------------------------------
        class MegaReqIdsProxy(list):
            def __init__(self, original_list, req_id_map):
                super().__init__(original_list)
                self.req_id_map = req_id_map
            def __getitem__(self, index):
                if index < len(self.req_id_map):
                    return self.req_id_map[index]
                return self.req_id_map[-1] if self.req_id_map else None
            def copy(self):
                return list(self.req_id_map)

        class Mock1DArray:
            def __init__(self, original, batch_idx_map):
                self.original = original
                self.batch_idx_map = batch_idx_map
            def __getitem__(self, idx):
                actual_idx = self.batch_idx_map[idx] if idx < len(self.batch_idx_map) else 0
                return self.original[actual_idx]
            def __setitem__(self, idx, value):
                actual_idx = self.batch_idx_map[idx] if idx < len(self.batch_idx_map) else 0
                self.original[actual_idx] = value

        class Mock2DArray:
            def __init__(self, original, batch_idx_map):
                self.original = original
                self.batch_idx_map = batch_idx_map
            def __setitem__(self, key, value):
                idx, slc = key
                actual_idx = self.batch_idx_map[idx] if idx < len(self.batch_idx_map) else 0
                self.original[actual_idx, slc] = value
            def __getattr__(self, name):
                return getattr(self.original, name)
            def __getitem__(self, key):
                idx, slc = key
                actual_idx = self.batch_idx_map[idx] if idx < len(self.batch_idx_map) else 0
                return self.original[actual_idx, slc]

        class InputBatchProxy:
            def __init__(self, obj, req_map, idx_map, runner):
                object.__setattr__(self, "_obj", obj)
                object.__setattr__(self, "_req_id_map", req_map)
                object.__setattr__(self, "_batch_idx_map", idx_map)
                object.__setattr__(self, "_runner", runner)
            def __getattr__(self, name):
                obj = object.__getattribute__(self, "_obj")
                req_map = object.__getattribute__(self, "_req_id_map")
                idx_map = object.__getattribute__(self, "_batch_idx_map")
                runner = object.__getattribute__(self, "_runner")
                
                if name == "sampling_metadata":
                    s_meta = getattr(obj, "sampling_metadata", None)
                    
                    # DYNAMIC RESOLUTION PASS: Calculate the row structure dynamically 
                    # whenever sampling_metadata is read to eliminate lifecycle races.
                    active_expansion = []
                    sched_out = getattr(runner, "_current_scheduler_output", None)
                    mega_data_ctx = getattr(runner, "_current_mega_data", {})
                    
                    if sched_out is not None and hasattr(obj, "req_ids"):
                        for b_idx, r_id in enumerate(obj.req_ids):
                            m_data = mega_data_ctx.get(r_id) if r_id else None
                            if m_data and m_data.get('is_mega_decode', False):
                                # Determine rows based on valid un-budgeted chunks
                                # If prepare_inputs has not changed the allocation grid yet,
                                # fall back gracefully to standard full beam-width allocation footprints.
                                num_allocated_rows = m_data.get('mega_beam_width', 1)
                                active_expansion.extend([b_idx] * num_allocated_rows)
                            else:
                                active_expansion.append(b_idx)
                    else:
                        active_expansion = idx_map

                    if s_meta is not None and active_expansion:
                        for attr in ["temperature", "top_p", "top_k", "min_p"]:
                            val = getattr(s_meta, attr, None)
                            if isinstance(val, torch.Tensor) and val.numel() > 0:
                                if val.shape[0] != len(active_expansion):
                                    object.__setattr__(s_meta, attr, val[active_expansion])
                    return s_meta
                elif name == "req_ids":
                    return MegaReqIdsProxy(obj.req_ids, req_map)
                elif name == "num_tokens_no_spec":
                    return Mock1DArray(obj.num_tokens_no_spec, idx_map)
                elif name == "token_ids_cpu":
                    return Mock2DArray(obj.token_ids_cpu, idx_map)
                elif name == "is_token_ids":
                    return Mock2DArray(obj.is_token_ids, idx_map) if hasattr(obj, 'is_token_ids') else Mock2DArray(obj.token_ids_cpu, idx_map)
                return getattr(obj, name)
            def __setattr__(self, name, value):
                obj = object.__getattribute__(self, "_obj")
                setattr(obj, name, value)

        # Hot-swap input batch parameters with our context-isolated proxy wrapper
        self.input_batch = InputBatchProxy(self.input_batch, row_to_req_id, row_to_batch_idx, self)
        
        self.input_batch._gr_request_row_configs = request_row_configs
        self.input_batch._gr_input_row_idx = input_row_idx
        
        logits_indices = torch.tensor(new_logits_indices, dtype=torch.int32, device=self.device)
        return logits_indices, spec_decode_metadata
  
    def patched_bookkeeping_sync(self, scheduler_output, *args, **kwargs):
        mega_data = getattr(scheduler_output, "mega_data", {})
        
        if not mega_data or not hasattr(self.input_batch, "_gr_request_row_configs"):
            return _original_bookkeeping_sync(self, scheduler_output, *args, **kwargs)

        request_row_configs = self.input_batch._gr_request_row_configs
        input_row_idx = self.input_batch._gr_input_row_idx

        orig_input_batch = object.__getattribute__(self.input_batch, "_obj")
        
        try:
            res = _original_bookkeeping_sync(self, scheduler_output, *args, **kwargs)
        finally:
            self.input_batch = orig_input_batch

        (num_nans_in_logits, logprobs_lists, valid_sampled_token_ids,
         prompt_logprobs_dict, req_ids_output_copy, req_id_to_index_output_copy, invalid_req_indices) = res

        # ---------------------------------------------------------------------
        # PHASE 3: ORIGINAL IN-PLACE DATA ASSIGNMENT (.DATA MUTATIONS UNCHANGED)
        # ---------------------------------------------------------------------
        sampler_output = kwargs.get("sampler_output") or (args[0] if args else None)
        if sampler_output is not None:
            st_ids = sampler_output.sampled_token_ids
            lp_tensors = sampler_output.logprobs_tensors
            
            if st_ids is not None and lp_tensors is not None and lp_tensors.logprobs is not None:
                orig_lps = lp_tensors.logprobs
                orig_lp_ids = lp_tensors.logprob_token_ids
                orig_ranks = getattr(lp_tensors, "selected_token_ranks", None)
                K = orig_lps.shape[1] 
                
                max_chained_dim = max(W * K for _, _, W in request_row_configs)

                new_st_ids_list = []
                new_lps_list = []
                new_lp_ids_list = []
                new_ranks_list = []

                for req_id, start_row, W in request_row_configs:
                    new_st_ids_list.append(st_ids[start_row : start_row + 1, :])

                    req_lps = orig_lps[start_row : start_row + W, :].reshape(1, W * K)
                    if W * K < max_chained_dim:
                        req_lps = torch.nn.functional.pad(req_lps, (0, max_chained_dim - W * K), value=float('-inf'))
                    new_lps_list.append(req_lps)

                    if orig_lp_ids is not None:
                        req_lp_ids = orig_lp_ids[start_row : start_row + W, :].reshape(1, W * K)
                        if W * K < max_chained_dim:
                            req_lp_ids = torch.nn.functional.pad(req_lp_ids, (0, max_chained_dim - W * K), value=0)
                        new_lp_ids_list.append(req_lp_ids)

                    if orig_ranks is not None:
                        req_ranks = orig_ranks[start_row : start_row + W].reshape(1, W)
                        if W < max_chained_dim:
                            req_ranks = torch.nn.functional.pad(req_ranks, (0, max_chained_dim - W), value=0)
                        new_ranks_list.append(req_ranks)

                final_st_ids = torch.cat(new_st_ids_list, dim=0)
                final_lps = torch.cat(new_lps_list, dim=0)
                final_lp_ids = torch.cat(new_lp_ids_list, dim=0) if orig_lp_ids is not None else None
                final_ranks = torch.cat(new_ranks_list, dim=0) if orig_ranks is not None else None

                st_ids.data = final_st_ids.data
                orig_lps.data = final_lps.data
                
                if orig_lp_ids is not None:
                    orig_lp_ids.data = final_lp_ids.data
                if orig_ranks is not None:
                    orig_ranks.data = final_ranks.view(-1).data

                if getattr(orig_input_batch, "prev_sampled_token_ids", None) is not None:
                    orig_input_batch.prev_sampled_token_ids.data = final_st_ids.data

                for req_id, _, _ in request_row_configs:
                    request = self.requests[req_id]
                    if request.sampling_params:
                        request.sampling_params.logprobs = max_chained_dim

                # -----------------------------------------------------------------
                # PHASE 4: SYNCHRONOUS LIST DATA RE-PACKAGING (FALLBACK)
                # -----------------------------------------------------------------
                if valid_sampled_token_ids and len(valid_sampled_token_ids) == input_row_idx:
                    new_valid_tokens = []
                    new_logprobs_lists = [] if logprobs_lists is not None else None
                    
                    for req_id, start_row, W in request_row_configs:
                        combined_tokens = []
                        for r in range(start_row, start_row + W):
                            combined_tokens.extend(valid_sampled_token_ids[r])
                        new_valid_tokens.append(combined_tokens)
                        
                        if new_logprobs_lists is not None:
                            combined_lps = []
                            for r in range(start_row, start_row + W):
                                if logprobs_lists[r] is not None:
                                    combined_lps.extend(logprobs_lists[r])
                            new_logprobs_lists.append(combined_lps)
                    
                    valid_sampled_token_ids = new_valid_tokens
                    logprobs_lists = new_logprobs_lists

        return (num_nans_in_logits, logprobs_lists, valid_sampled_token_ids,
                prompt_logprobs_dict, req_ids_output_copy, req_id_to_index_output_copy, invalid_req_indices)

    GPUModelRunner.execute_model = patched_execute_model
    GPUModelRunner._prepare_inputs = patched_prepare_inputs
    GPUModelRunner._bookkeeping_sync = patched_bookkeeping_sync
    GPUModelRunner._patched_for_mega_beam_pos_ids = True
    logger.debug("GPUModelRunner patched for mega-beam position IDs.")