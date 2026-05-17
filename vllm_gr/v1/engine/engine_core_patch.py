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


def _compute_beam_prefix_groups(req_ids: list[str], requests: dict, block_size: int) -> list:
    """Group requests by beam priority and find the common prefix length in tokens."""
    if not req_ids:
        return []

    # Group requests by beam group id (priority)
    beam_groups: dict[int, list[str]] = {}
    for req_id in req_ids:
        if req_id not in requests:
            continue
        g = requests[req_id].priority
        if g not in beam_groups:
            beam_groups[g] = []
        beam_groups[g].append(req_id)

    all_groups = []
    for g, group_req_ids in beam_groups.items():
        n = len(group_req_ids)
        if n == 0:
            continue
        if n == 1:
            all_groups.append((0, [group_req_ids[0]]))
            continue

        prev_num_common_blocks = 0
        # Find the longest common prefix (LCP) for the group.
        first_req = requests[group_req_ids[0]]
        lcp_tokens = first_req.num_computed_tokens
        for i in range(1, len(group_req_ids)):
            other_req = requests[group_req_ids[i]]
            # Use block hashes for a fast block-aligned LCP
            num_common_blocks = _get_lcp(
                first_req.block_hashes, other_req.block_hashes, hint=prev_num_common_blocks
            )
            prev_num_common_blocks = num_common_blocks
            # Refine to token-level LCP by checking inside the first differing block
            pairwise_lcp = min(num_common_blocks * block_size, other_req.num_computed_tokens)
            limit = min(lcp_tokens, other_req.num_computed_tokens)

            first_tokens = first_req._all_token_ids
            other_tokens = other_req._all_token_ids

            for i in range(pairwise_lcp, limit):
                if first_tokens[i] != other_tokens[i]:
                    limit = i
                    break
            pairwise_lcp = limit

            lcp_tokens = min(lcp_tokens, pairwise_lcp)
            if lcp_tokens == 0:
                break

        all_groups.append((lcp_tokens, group_req_ids))

    return all_groups


def _apply_pbsc_routing(
    enable_pbsc: bool, output, beam_groups: list, requests: dict, block_size: int
) -> None:
    """Applies Partial-Block Shared Compute (PBSC) routing logic to the scheduled requests."""

    new_num_scheduled_tokens = {}

    # PBSC Statistics
    pbsc_stats_groups_total = 0
    pbsc_stats_groups_active = 0
    pbsc_stats_children_total = 0
    pbsc_stats_children_valid = 0
    pbsc_stats_tokens_saved = 0

    # PBSC Drop Reasons
    drop_fully_cached = 0
    drop_no_prefix = 0
    drop_tail_too_long = 0

    pbsc_shaped_groups = []

    for orig_t_prefix, group_req_ids in beam_groups:
        if not enable_pbsc:
            break
        if len(group_req_ids) <= 1:
            continue
        t_block_aligned = (orig_t_prefix // block_size) * block_size
        # 2. Sub-group by exact cache_hit for PBSC tensor shaping
        cache_hit_groups = {}
        for req_id in group_req_ids:
            req = requests[req_id]
            # vLLM updates req.num_computed_tokens during schedule().
            # By subtracting the schedule, we get the exact number of tokens
            cache_hit = req.num_computed_tokens - output.num_scheduled_tokens[req_id]

            # If the request already has the entire shared prefix in cache,
            # it doesn't need PBSC tail sharing.
            if cache_hit >= orig_t_prefix:
                drop_fully_cached += 1
                logger.debug(
                    "[PBSC debug] Subgroup skipped: tokens=%d, schedule=%d cache_hit=%d, orig_t_prefix=%d)",
                    req.num_tokens,
                    output.num_scheduled_tokens[req_id],
                    cache_hit,
                    orig_t_prefix,
                )
                continue

            if cache_hit not in cache_hit_groups:
                cache_hit_groups[cache_hit] = []
            cache_hit_groups[cache_hit].append(req_id)

        for t_cache_hit, sub_group_req_ids in cache_hit_groups.items():
            if len(sub_group_req_ids) <= 1:
                continue

            leader_id = sub_group_req_ids[0]
            leader_req = requests[leader_id]
            valid_children = sub_group_req_ids[1:]

            # Safely cap t_prefix to what is actually scheduled/allocated this step to avoid OOB Seg Faults
            min_scheduled = output.num_scheduled_tokens[leader_id]
            for child_id in valid_children:
                min_scheduled = min(min_scheduled, output.num_scheduled_tokens[child_id])
            # Ensure each child computes at least 1 token to prevent num_scheduled_tokens == 0
            t_prefix = min(orig_t_prefix, t_cache_hit + min_scheduled - 1)

            # If the cache hit missed full blocks we thought were shared,
            # restrict the sharing to what is actually cached.
            if t_block_aligned > t_cache_hit:
                t_prefix = min(t_prefix, t_cache_hit)

            if t_prefix <= t_cache_hit:
                drop_no_prefix += len(valid_children)
                logger.debug(
                    "[PBSC debug] Subgroup skipped: t_prefix became <= t_cache_hit (orig_t_prefix=%d, t_cache_hit=%d)",
                    orig_t_prefix,
                    t_cache_hit,
                )
                continue

            # PBSC Safety Check: Active tail sharing is only for highly overlapping beams.
            # If the unshared part is large, these are likely unrelated requests.
            child_compute_len_estimate = leader_req.num_tokens - t_prefix
            if child_compute_len_estimate > block_size:
                drop_tail_too_long += len(valid_children)
                logger.debug(
                    "[PBSC debug] Subgroup skipped: tail too long (unshared=%d > block_size=%d)",
                    child_compute_len_estimate,
                    block_size,
                )
                continue

            pbsc_shaped_groups.append((t_prefix, sub_group_req_ids))
            pbsc_stats_groups_total += 1
            pbsc_stats_children_total += len(valid_children)

            # Leader computes exactly what the scheduler told it to (respects chunking)
            leader_scheduled = output.num_scheduled_tokens[leader_id]
            new_num_scheduled_tokens[leader_id] = leader_scheduled

            tokens_reused_from_leader = t_prefix - t_cache_hit

            # Children compute everything after the token-aligned prefix (tail is broadcasted)
            for child_id in valid_children:
                child_scheduled = output.num_scheduled_tokens[child_id]
                child_compute_len = max(0, child_scheduled - tokens_reused_from_leader)
                new_num_scheduled_tokens[child_id] = child_compute_len

                pbsc_stats_tokens_saved += max(0, child_scheduled - child_compute_len)

            pbsc_stats_groups_active += 1
            pbsc_stats_children_valid += len(valid_children)

    if pbsc_stats_groups_total > 0:
        logger.debug(
            "[PBSC Stats] Groups Active: %d / %d | Valid Children: %d / %d (%.1f%%) | Tokens Compute Saved: %d | Drops (Cached: %d, NoPrefix: %d, TailTooLong: %d)",
            pbsc_stats_groups_active,
            pbsc_stats_groups_total,
            pbsc_stats_children_valid,
            pbsc_stats_children_total,
            (pbsc_stats_children_valid / pbsc_stats_children_total * 100)
            if pbsc_stats_children_total > 0
            else 0,
            pbsc_stats_tokens_saved,
            drop_fully_cached,
            drop_no_prefix,
            drop_tail_too_long,
        )

    output.beam_prefix_groups = beam_groups
    output.pbsc_shaped_groups = pbsc_shaped_groups

    # Rebuild dictionary to keep Leader first, then Children (for contiguous 1D tensor shape)
    ordered_num_scheduled = {}
    for _, pbsc_group in pbsc_shaped_groups:
        for req_id in pbsc_group:
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
    # INVARIANT: `output.scheduled_new_reqs` and `output.scheduled_cached_reqs`
    # are DTOs created fresh during `schedule()`. Mutating `num_computed_tokens`
    # here only affects the message sent to the worker, instructing it to start
    # the child's local computation at `t_prefix` (skipping the broadcast tokens).
    # The Scheduler's internal state (`requests[child_id].num_computed_tokens`)
    # correctly reflects the total computed tokens by the end of the step (including
    # the broadcast ones), ensuring consistency with the KV-cache manager and
    # subsequent steps. Therefore, we do not mutate `requests[child_id]` here.
    child_prefix_map = {}
    for t_prefix, pbsc_group in pbsc_shaped_groups:
        for child_id in pbsc_group[1:]:
            if child_id in new_num_scheduled_tokens:
                child_prefix_map[child_id] = t_prefix

    for req_data in output.scheduled_new_reqs:
        if req_data.req_id in child_prefix_map:
            req_data.num_computed_tokens = child_prefix_map[req_data.req_id]

    if hasattr(output, "scheduled_cached_reqs"):
        req_data = output.scheduled_cached_reqs
        if isinstance(req_data.num_computed_tokens, tuple):
            req_data.num_computed_tokens = list(req_data.num_computed_tokens)
        for i, req_id in enumerate(req_data.req_ids):
            if req_id in child_prefix_map:
                req_data.num_computed_tokens[i] = child_prefix_map[req_id]


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

    import torch
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

        # Evaluate once per scheduler instance based on actual model configs
        if not hasattr(self, "_enable_pbsc"):
            attn_backend = "NONE"

            # 1. Check vllm_config for CLI arguments
            if hasattr(self, "vllm_config"):
                # Supports --attention-config.backend
                if hasattr(self.vllm_config, "attention_config") and hasattr(
                    self.vllm_config.attention_config, "backend"
                ):
                    attn_backend = getattr(self.vllm_config.attention_config, "backend")
                # Supports --attention-backend (sometimes mapped directly to vllm_config or model_config)
                if not attn_backend or "NONE" in str(attn_backend).upper():
                    attn_backend = getattr(self.vllm_config, "attention_backend", "NONE")
                if not attn_backend or "NONE" in str(attn_backend).upper():
                    if hasattr(self.vllm_config, "model_config"):
                        attn_backend = getattr(
                            self.vllm_config.model_config, "attn_backend", "NONE"
                        )

            # 2. Fallback to environment variables
            if not attn_backend or "NONE" in str(attn_backend).upper():
                try:
                    from vllm.envs import VLLM_ATTENTION_BACKEND

                    attn_backend = str(VLLM_ATTENTION_BACKEND)
                except ImportError:
                    import os

                    attn_backend = os.environ.get("VLLM_ATTENTION_BACKEND", "NONE")

            self._enable_pbsc = torch.cuda.is_available() and "CUSTOM" in str(attn_backend).upper()
            logger.info(
                "PBSC (Partial-Block Shared Compute) routing enabled: %s", self._enable_pbsc
            )
        enable_pbsc = self._enable_pbsc

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
            beam_groups = _compute_beam_prefix_groups(
                req_ids, self.requests, self.cache_config.block_size
            )
            _apply_pbsc_routing(
                enable_pbsc, output, beam_groups, self.requests, self.cache_config.block_size
            )

        return output

    Scheduler.schedule = patched_schedule
    setattr(Scheduler, "_patched_for_cache_computed_blocks", True)
    logger.debug("Scheduler.schedule patched for cache computed blocks.")


def apply_worker_patches():
    """Monkey-patch GPUModelRunner to inject beam_prefix_groups into the attention builder."""
    try:
        import numpy as np
        from vllm.v1.worker.gpu_input_batch import InputBatch
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        from vllm_gr.v1.attention.backends.beam_attn import (
            BEAM_PREFIX_GROUPS_VAR,
            PBSC_SHAPED_GROUPS_VAR,
        )
    except ImportError:
        return

    if getattr(GPUModelRunner, "_patched_for_beam_groups", False):
        return

    _original_execute_model = GPUModelRunner.execute_model
    _original_build_attention_metadata = GPUModelRunner._build_attention_metadata

    def patched_execute_model(self, scheduler_output, intermediate_tensors=None):
        beam_groups = getattr(scheduler_output, "beam_prefix_groups", None)
        pbsc_groups = getattr(scheduler_output, "pbsc_shaped_groups", None)
        self._current_beam_prefix_groups_str = beam_groups
        self._current_pbsc_shaped_groups_str = pbsc_groups

        # Clear fast prefix copy cache to prevent cross-step stale index usage
        if hasattr(self, "input_batch"):
            self.input_batch._last_prompt_token_ids_list = None
            self.input_batch._last_prompt_token_ids_index = None
            self.input_batch._last_req_id = None
            self.input_batch._current_beam_prefix_groups = beam_groups
            self.input_batch._last_beam_group = None

        try:
            return _original_execute_model(self, scheduler_output, intermediate_tensors)
        finally:
            self._current_beam_prefix_groups_str = None
            self._current_pbsc_shaped_groups_str = None
            if hasattr(self, "input_batch"):
                self.input_batch._current_beam_prefix_groups = None
                self.input_batch._last_beam_group = None

    def patched_build_attention_metadata(self, *args, **kwargs):
        beam_groups_str = getattr(self, "_current_beam_prefix_groups_str", None)
        pbsc_groups_str = getattr(self, "_current_pbsc_shaped_groups_str", None)

        tokens_to_reset = []
        if beam_groups_str is not None or pbsc_groups_str is not None:
            req_id_to_idx = {
                r_id: i for i, r_id in enumerate(self.input_batch.req_ids) if r_id is not None
            }
            if beam_groups_str is not None:
                translated_groups = [
                    (d, [req_id_to_idx[r] for r in g if r in req_id_to_idx])
                    for d, g in beam_groups_str
                ]
                translated_groups = [(d, g) for d, g in translated_groups if g]
                tokens_to_reset.append(
                    (BEAM_PREFIX_GROUPS_VAR, BEAM_PREFIX_GROUPS_VAR.set(translated_groups))
                )

            if pbsc_groups_str is not None:
                translated_pbsc = [
                    (d, [req_id_to_idx[r] for r in g if r in req_id_to_idx])
                    for d, g in pbsc_groups_str
                ]
                translated_pbsc = [(d, g) for d, g in translated_pbsc if g]
                tokens_to_reset.append(
                    (PBSC_SHAPED_GROUPS_VAR, PBSC_SHAPED_GROUPS_VAR.set(translated_pbsc))
                )
            try:
                return _original_build_attention_metadata(self, *args, **kwargs)
            finally:
                for var, token in tokens_to_reset:
                    var.reset(token)
        else:
            return _original_build_attention_metadata(self, *args, **kwargs)

    if not getattr(InputBatch, "_patched_for_fast_beam_add", False):
        _original_add_request = InputBatch.add_request

        def patched_add_request(self, request, *args, **kwargs):
            original_prompt_token_ids = request.prompt_token_ids
            if original_prompt_token_ids is not None:
                # Use a dummy array as a sentinel to skip the slow Python-to-numpy copy
                # in the upstream method, while still providing the correct length.
                dummy_tensor = np.empty(
                    len(original_prompt_token_ids), dtype=self.token_ids_cpu.dtype
                )
                request.prompt_token_ids = dummy_tensor

            try:
                req_index = _original_add_request(self, request, *args, **kwargs)
            finally:
                request.prompt_token_ids = original_prompt_token_ids

            assert isinstance(req_index, int), "Upstream add_request signature changed"

            try:
                if original_prompt_token_ids is not None:
                    prompt_len = len(original_prompt_token_ids)
                    req_id = request.req_id

                    # --- OPTIMIZED FAST PREFIX COPY ---
                    last_list = getattr(self, "_last_prompt_token_ids_list", None)
                    last_idx = getattr(self, "_last_prompt_token_ids_index", None)
                    last_req_id = getattr(self, "_last_req_id", None)

                    diverge_idx = -1
                    # Check for prefix sharing with the last added request
                    if (
                        last_list is not None
                        and last_idx is not None
                        and last_req_id is not None
                        and len(last_list) == prompt_len
                    ):
                        current_beam_groups = getattr(self, "_current_beam_prefix_groups", None)
                        if current_beam_groups is not None:
                            last_group = getattr(self, "_last_beam_group", None)
                            found_group = None

                            if last_group is not None:
                                lcp, req_ids = last_group
                                if req_id in req_ids and last_req_id in req_ids:
                                    found_group = last_group

                            if found_group is None:
                                for group in current_beam_groups:
                                    _, req_ids = group
                                    if req_id in req_ids and last_req_id in req_ids:
                                        found_group = group
                                        self._last_beam_group = group
                                        break

                            if found_group is not None:
                                group_lcp_tokens = found_group[0]
                                diverge_idx = group_lcp_tokens - 1

                    if diverge_idx >= 0:
                        # Copy shared prefix efficiently from previously populated numpy row
                        self.token_ids_cpu[req_index, : diverge_idx + 1] = self.token_ids_cpu[
                            last_idx, : diverge_idx + 1
                        ]
                        # Process the divergent tail
                        if diverge_idx + 1 < prompt_len:
                            self.token_ids_cpu[req_index, diverge_idx + 1 : prompt_len] = (
                                original_prompt_token_ids[diverge_idx + 1 :]
                            )
                    else:
                        # Fallback to slow Python-to-NumPy conversion
                        self.token_ids_cpu[req_index, :prompt_len] = original_prompt_token_ids

                    self.is_token_ids[req_index, :prompt_len] = True
            finally:
                if original_prompt_token_ids is not None:
                    # Cache this list state for the next sibling request
                    self._last_prompt_token_ids_list = original_prompt_token_ids
                    self._last_prompt_token_ids_index = req_index
                    self._last_req_id = request.req_id
                    # --- END OPTIMIZED ---
                else:
                    self._last_prompt_token_ids_list = None
                    self._last_prompt_token_ids_index = None
                    self._last_req_id = None

            return req_index

    InputBatch.add_request = patched_add_request
    InputBatch._patched_for_fast_beam_add = True
    GPUModelRunner.execute_model = patched_execute_model
    GPUModelRunner._build_attention_metadata = patched_build_attention_metadata
    GPUModelRunner._patched_for_beam_groups = True
    logger.debug("GPUModelRunner patched for beam groups via ContextVar.")
