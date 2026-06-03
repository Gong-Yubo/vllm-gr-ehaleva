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
    """Monkey-patch GPUModelRunner to fix position IDs for mega-requests."""
    try:
        import torch
        from vllm.v1.worker.gpu_input_batch import InputBatch
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        from vllm.sampling_params import SamplingType
    except ImportError:
        logger.warning("Failed to import GPUModelRunner, skipping worker patches.")
        return

    if getattr(GPUModelRunner, "_patched_for_mega_beam_pos_ids", False):
        return

    # In vLLM, prepare_model_input is where position_ids are finalized.
    # We need to find the mega request in the batch and overwrite its suffix positions.
    if not hasattr(GPUModelRunner, '_prepare_inputs'):
        logger.warning("GPUModelRunner._prepare_inputs not found, skipping position ID patch.")
        return
        
    _original_execute_model = GPUModelRunner.execute_model
    _original_prepare_inputs = GPUModelRunner._prepare_inputs

    def patched_execute_model(self, scheduler_output, *args, **kwargs):
        # Stash the mega_data temporarily on the runner so prepare_model_input can read it
        self._current_mega_data = getattr(scheduler_output, "mega_data", {})
        from vllm_gr.v1.attention.backends.beam_attn import MEGA_DATA_VAR
        mega_info = {
            "mega_data": self._current_mega_data,
            "req_ids": []  # To be populated dynamically inside prepare_model_input
        }
        token = MEGA_DATA_VAR.set(mega_info)
        self._current_mega_info = mega_info
        
        try:
            return _original_execute_model(self, scheduler_output, *args, **kwargs)
        finally:
            self._current_mega_data = {}
            self._current_mega_info = None
            MEGA_DATA_VAR.reset(token)

    def patched_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        # 1. Initialize native vLLM parameters and populate baseline states
        logits_indices, spec_decode_metadata = _original_prepare_inputs(
            self, scheduler_output, num_scheduled_tokens
        )

        # Synchronize dynamic tracking arrays if monitoring hooks are active
        if getattr(self, "_current_mega_info", None) is not None:
            self._current_mega_info["req_ids"] = self.input_batch.req_ids

        mega_data = getattr(self, "_current_mega_data", {})
        if not mega_data:
            return logits_indices, spec_decode_metadata

        # Initialize tracking registries for our unified layout pass
        self.input_batch.logprob_token_ids = {}
        new_logits_indices = []
        sample_routing = []

        token_offset = 0
        logit_row_idx = 0
        sample_slot_offset = 0
        
        
        # 2. Execute the Consolidated Single-Pass Batch Layout Engine
        for i, req_id in enumerate(self.input_batch.req_ids):
            seq_len = num_scheduled_tokens[i]            
            if req_id is None:
                token_offset += seq_len
                continue

            request = self.requests[req_id]
            req_n = request.sampling_params.n if request.sampling_params else 1
            mdata = mega_data.get(req_id)

            # ---------------------------------------------------------------------
            # PATH A: Grouped Mega-Request Branch Resolution
            # ---------------------------------------------------------------------
            if mdata and mdata.get('is_mega_decode', False):
                prefix_len = mdata['prefix_len']
                beam_width = mdata['mega_beam_width']
                decode_steps = mdata['mega_decode_steps']
                cache_len = mdata['cache_len']

                req_n = beam_width
                mdata['sample_slot_offset'] = sample_slot_offset

                if beam_width > 0 and decode_steps > 0:
                    req_positions = []
                    leaf_token_ids = []
                    beam_0_uncached = max(0, prefix_len + decode_steps - cache_len)
                    curr_offset = token_offset
                    for b in range(beam_width):
                        b_uncached = beam_0_uncached if b == 0 else decode_steps
                        if b_uncached > 0:
                            # Compute shared positional IDs
                            b_start_pos = prefix_len + decode_steps - b_uncached if b == 0 else prefix_len
                            b_pos = torch.arange(b_start_pos, b_start_pos + b_uncached, device=self.device, dtype=torch.long)
                            req_positions.append(b_pos)

                            # Pin active leaf logit row offsets
                            new_logits_indices.append(curr_offset + b_uncached - 1)
                            curr_offset += b_uncached

                            # Map logit calculations back to this specific request's batch slot
                            sample_routing.append([logit_row_idx, i])
                            logit_row_idx += 1

                        # Unified Strided Logprob Lookup (Resolves multi-beam chunk offsets)
                        leaf_idx = prefix_len + (b + 1) * decode_steps - 1
                        if leaf_idx < request.num_tokens:
                            tok_id = request.get_token_id(leaf_idx)
                            if tok_id >= 0:
                                leaf_token_ids.append(tok_id)

                    # # Commit layout geometries to memory channels concurrently
                    # if req_positions:
                    #     correct_suffix_positions = torch.cat(req_positions)
                    #     self.positions.gpu[token_offset:token_offset + seq_len] = correct_suffix_positions
                    #     self.positions.cpu[token_offset:token_offset + seq_len] = correct_suffix_positions.cpu()
            # ---------------------------------------------------------------------
            # PATH B: Standard Request Native Processing (Fallback)
            # ---------------------------------------------------------------------
            else:
                if seq_len > 0:
                    new_logits_indices.append(token_offset + seq_len - 1)
                    for _ in range(req_n):
                        sample_routing.append([logit_row_idx, i])
                    logit_row_idx += 1

            # Advance sequence offsets gracefully
            sample_slot_offset += req_n
            token_offset += seq_len

        # # 4. Finalize Sampler Metadata Enforcements
        # if new_logits_indices:
        #     logits_indices = torch.tensor(new_logits_indices, dtype=torch.int32, device=self.device)
        #     sampling_metadata = self.input_batch.sampling_metadata
        #     if sampling_metadata is not None:
        #         if not hasattr(sampling_metadata, 'categorized_sample_indices'):
        #             sampling_metadata.categorized_sample_indices = {}

        #         if sample_routing:
        #             routing_tensor = torch.tensor(sample_routing, dtype=torch.int32, device=self.device)
        #             sampling_metadata.logits_indices = logits_indices
        #             sampling_metadata.categorized_sample_indices[SamplingType.GREEDY] = routing_tensor
        #             sampling_metadata.categorized_sample_indices[SamplingType.RANDOM] = routing_tensor
        #             sampling_metadata.categorized_sample_indices[SamplingType.RANDOM_SEED] = routing_tensor

        return logits_indices, spec_decode_metadata
  
    GPUModelRunner.execute_model = patched_execute_model
    GPUModelRunner._prepare_inputs = patched_prepare_inputs
    GPUModelRunner._patched_for_mega_beam_pos_ids = True
    logger.debug("GPUModelRunner patched for mega-beam position IDs.")
