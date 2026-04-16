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


def _get_lcp(a: list, b: list, hint: int = 0) -> int:
    """Find longest common prefix between two hash lists using binary search."""
    m = min(len(a), len(b))
    if m == 0:
        return 0

    start = 0
    end = m - 1

    if hint > 0:
        h = min(hint, m)
        if a[h - 1] == b[h - 1]:
            if h == m or a[h] != b[h]:
                return h
            start = h
        else:
            end = h - 2

    if start <= end and a[end] == b[end]:
        return end + 1

    ans = start
    low, high = start, end
    while low <= high:
        mid = (low + high) // 2
        if a[mid] == b[mid]:
            ans = mid + 1
            low = mid + 1
        else:
            high = mid - 1
    return ans


def _compute_beam_prefix_groups(req_ids: list[str], requests: dict) -> list:
    """Group requests by beam priority and return shared prefix length in tokens."""
    if not req_ids:
        return []

    # Group requests by beam group id (priority)
    beam_groups: dict[int, list[str]] = {}
    for req_id in req_ids:
        g = requests[req_id].priority
        if g not in beam_groups:
            beam_groups[g] = []
        beam_groups[g].append(req_id)

    all_groups = []
    for g, group_req_ids in beam_groups.items():
        n = len(group_req_ids)
        if n <= 1:
            if n == 1:
                all_groups.append((0, [group_req_ids[0]]))
            continue

        # PBSC: We assume that the last 2 tokens are not shared and all others are shared
        req = requests[group_req_ids[0]]
        t_prefix = max(0, req.num_tokens - 2)
        all_groups.append((t_prefix, group_req_ids))

    return all_groups


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
        _handle_beam_fork,
        process_input_sockets,
    )
    from vllm_gr.v1.engine.types import BeamForkRequest

    # Inject types into vllm.v1.engine module
    engine_mod.BeamForkRequest = BeamForkRequest

    # Add new enum members: ADD_BATCH, BEAM_FORK
    _add_enum_member("ADD_BATCH", b"\x05")
    _add_enum_member("BEAM_FORK", b"\x06")

    # Wrap EngineCore.__init__ to add beam state
    EngineCore.__init__ = make_patched_engine_core_init(EngineCore.__init__)

    # Bind _cache_beam_request and _handle_beam_fork on both classes
    EngineCore._cache_beam_request = _cache_beam_request
    EngineCore._handle_beam_fork = _handle_beam_fork
    EngineCoreProc._cache_beam_request = _cache_beam_request
    EngineCoreProc._handle_beam_fork = _handle_beam_fork

    # Replace EngineCoreProc.process_input_sockets
    EngineCoreProc.process_input_sockets = process_input_sockets

    # Apply scheduler patches in the child process
    apply_scheduler_patch()

    # Apply worker patches (e.g., GPUModelRunner)
    apply_worker_patches()

    logger.debug("Engine-core patches (ADD_BATCH, BEAM_FORK) applied in child process.")


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
    """Monkey-patch Scheduler.schedule with the cache computed blocks version."""
    from functools import wraps
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler, "_patched_for_cache_computed_blocks", False):
        return

    _original_schedule = Scheduler.schedule

    @wraps(_original_schedule)
    def patched_schedule(self):
        # Cache for get_computed_blocks to optimize beam search
        computed_blocks_cache = {}
        last_key = None
        last_res = None
        block_size = self.cache_config.block_size
        _original_get_computed_blocks = self.kv_cache_manager.get_computed_blocks

        def get_cache_computed_blocks(req):
            # In some cases, the first beam must recompute all blocks if they were previously cleared.
            # However, later beams can skip this work because the first beam has already performed it.
            # By applying a threshold, we ensure that only heavily reused blocks—those with many cache hits—are stored in the cache.
            cache_th = 256

            def is_cache_worthy(r):
                return r[1] > cache_th

            nonlocal last_key, last_res
            # get_computed_blocks only accounts for hashes up to the final complete block.
            # Additionally, when every token is retrieved from the cache, the last token still
            # needs to be recomputed to produce the logits
            max_num_blocks = max(0, (req.num_tokens - 1) // block_size)
            hash_idx = min(max_num_blocks - 1, len(req.block_hashes) - 1)
            last_hash = req.block_hashes[hash_idx] if hash_idx >= 0 else None
            if last_hash is None:
                last_hash = req.request_id
            key = (req.num_tokens, req.skip_reading_prefix_cache, last_hash)

            served_from_sw_cache = True
            if key == last_key and last_res is not None and is_cache_worthy(last_res):
                res = last_res
            elif key in computed_blocks_cache:
                res = computed_blocks_cache[key]
                last_key = key
                last_res = res
            else:
                res = _original_get_computed_blocks(req)
                served_from_sw_cache = False
                if is_cache_worthy(res):
                    computed_blocks_cache[key] = res
                    last_key = key
                    last_res = res

            if served_from_sw_cache and self.kv_cache_manager.log_stats:
                if self.kv_cache_manager.prefix_cache_stats is not None:
                    self.kv_cache_manager.prefix_cache_stats.record(
                        num_tokens=req.num_tokens,
                        num_hits=res[1],
                        preempted=req.num_preemptions > 0,
                    )
            return res

        # Temporarily mock the get_computed_blocks method for this schedule pass
        self.kv_cache_manager.get_computed_blocks = get_cache_computed_blocks
        try:
            output = _original_schedule(self)
        finally:
            # Restore the original methods immediately after
            self.kv_cache_manager.get_computed_blocks = _original_get_computed_blocks

        req_ids = list(output.num_scheduled_tokens.keys())
        if req_ids:
            beam_groups = _compute_beam_prefix_groups(req_ids, self.requests)
            output.beam_prefix_groups = beam_groups
            
            # PBSC Tensor Shaping: Modify num_scheduled_tokens to ensure 
            # Leader computes the tail and Children only compute new tokens
            new_num_scheduled_tokens = {}
            for t_prefix, group_req_ids in beam_groups:
                if len(group_req_ids) == 1:
                    new_num_scheduled_tokens[group_req_ids[0]] = output.num_scheduled_tokens[group_req_ids[0]]
                    continue

                leader_id = group_req_ids[0]
                req = self.requests[leader_id]
                
                t_block_aligned = (t_prefix // block_size) * block_size
                
                # Leader computes everything after the block-aligned prefix
                leader_compute_len = req.num_tokens - t_block_aligned
                new_num_scheduled_tokens[leader_id] = leader_compute_len
                
                # Children compute everything after the token-aligned prefix (tail is broadcasted)
                child_compute_len = req.num_tokens - t_prefix
                for child_id in group_req_ids[1:]:
                    new_num_scheduled_tokens[child_id] = child_compute_len

            # Rebuild dictionary to keep Leader first, then Children (for contiguous 1D tensor shape)
            ordered_num_scheduled = {}
            for _, group_req_ids in beam_groups:
                for req_id in group_req_ids:
                    if req_id in new_num_scheduled_tokens:
                        ordered_num_scheduled[req_id] = new_num_scheduled_tokens[req_id]
            
            # Add any requests that were not in groups (just in case)
            for req_id, num in output.num_scheduled_tokens.items():
                if req_id not in ordered_num_scheduled:
                    ordered_num_scheduled[req_id] = num
                    
            output.num_scheduled_tokens = ordered_num_scheduled
            output.total_num_scheduled_tokens = sum(ordered_num_scheduled.values())

            # PBSC: Update num_computed_tokens for children so GPUModelRunner
            # generates correct Position IDs and slot mappings.
            child_prefix_map = {}
            for t_prefix, group_req_ids in beam_groups:
                if len(group_req_ids) > 1:
                    for child_id in group_req_ids[1:]:
                        child_prefix_map[child_id] = t_prefix

            for req_data in output.scheduled_new_reqs:
                if req_data.req_id in child_prefix_map:
                    req_data.num_computed_tokens = child_prefix_map[req_data.req_id]
                    
            if hasattr(output, "scheduled_cached_reqs"):
                req_data = output.scheduled_cached_reqs
                for i, req_id in enumerate(req_data.req_ids):
                    if req_id in child_prefix_map:
                        req_data.num_computed_tokens[i] = child_prefix_map[req_id]

        return output

    Scheduler.schedule = patched_schedule
    setattr(Scheduler, "_patched_for_cache_computed_blocks", True)
    logger.debug("Scheduler.schedule patched for cache computed blocks.")


def apply_worker_patches():
    """Monkey-patch GPUModelRunner to inject beam_prefix_groups into the attention builder."""
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        from vllm_gr.v1.attention.backends.beam_attn import BEAM_PREFIX_GROUPS_VAR
    except ImportError:
        return

    if getattr(GPUModelRunner, "_patched_for_beam_groups", False):
        return

    _original_execute_model = GPUModelRunner.execute_model
    _original_build_attention_metadata = GPUModelRunner._build_attention_metadata

    def patched_execute_model(self, scheduler_output, intermediate_tensors=None):
        beam_groups = getattr(scheduler_output, "beam_prefix_groups", None)
        self._current_beam_prefix_groups_str = beam_groups

        try:
            return _original_execute_model(self, scheduler_output, intermediate_tensors)
        finally:
            self._current_beam_prefix_groups_str = None

    def patched_build_attention_metadata(self, *args, **kwargs):
        beam_groups_str = getattr(self, "_current_beam_prefix_groups_str", None)
        if beam_groups_str is not None:
            req_id_to_idx = {
                r_id: i for i, r_id in enumerate(self.input_batch.req_ids) if r_id is not None
            }
            translated_groups = []
            for depth, group_req_ids in beam_groups_str:
                indices = [req_id_to_idx[r] for r in group_req_ids if r in req_id_to_idx]
                if indices:
                    translated_groups.append((depth, indices))

            token = BEAM_PREFIX_GROUPS_VAR.set(translated_groups)
            try:
                return _original_build_attention_metadata(self, *args, **kwargs)
            finally:
                BEAM_PREFIX_GROUPS_VAR.reset(token)
        else:
            return _original_build_attention_metadata(self, *args, **kwargs)

    GPUModelRunner.execute_model = patched_execute_model
    GPUModelRunner._build_attention_metadata = patched_build_attention_metadata
    GPUModelRunner._patched_for_beam_groups = True
    logger.debug("GPUModelRunner patched for beam groups via ContextVar.")
