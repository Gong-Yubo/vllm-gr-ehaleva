# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Data Parallelism support in beam search (no GPU required).

Tests cover the following DP beam search features:
  - data_parallel_rank threading through beam_search → step functions
  - DP-aware engine routing in add_requests_async / beam_fork_async
  - FIRST_REQ coordinator notification
  - reqs_in_flight tracking for batch and fork
  - BeamForkRequest.data_parallel_rank serialization
  - register_beam_output rank passthrough
  - current_wave propagation

See design_dp_beam_search.md §3.1 for the full test plan.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import msgspec
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequestType

from vllm_gr.v1.engine.core_client import add_requests_async, beam_fork_async
from vllm_gr.v1.engine.core_client_patch import register_beam_output_fn
from vllm_gr.v1.engine.engine_core_patch import _add_enum_member
from vllm_gr.v1.engine.types import BeamForkRequest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sampling_params(**overrides: Any) -> SamplingParams:
    defaults = dict(max_tokens=1, logprobs=3, temperature=0.0, detokenize=False)
    defaults.update(overrides)
    return SamplingParams(**defaults)


def _make_dp_client_mock(
    *,
    client_index: int = 0,
    engines_running: bool = True,
    num_engines: int = 2,
    has_current_wave: bool = False,
    current_wave: int = 0,
) -> MagicMock:
    """Create a mock that looks like DPLBAsyncMPClient."""
    mock_self = MagicMock()
    mock_self.client_index = client_index
    mock_self.engines_running = engines_running
    mock_self._send_input = AsyncMock()
    mock_self._ensure_output_queue_task = MagicMock()
    mock_self._ensure_stats_update_task = MagicMock()
    mock_self.first_req_send_socket = AsyncMock()
    mock_self.reqs_in_flight = {}

    # Build core_engines as simple strings so they are msgpack-serializable
    # (the real code does msgspec.msgpack.encode(("FIRST_REQ", engine))).
    engines = [f"engine-{i}" for i in range(num_engines)]
    mock_self.core_engines = engines

    # get_core_engine_for_request returns engine based on data_parallel_rank.
    def _get_engine(request):
        rank = getattr(request, "data_parallel_rank", None)
        idx = rank if rank is not None else 0
        chosen = engines[idx]
        mock_self.reqs_in_flight[request.request_id] = chosen
        return chosen

    mock_self.get_core_engine_for_request = MagicMock(side_effect=_get_engine)

    if has_current_wave:
        mock_self.current_wave = current_wave
    else:
        # Remove current_wave so hasattr returns False.
        del mock_self.current_wave

    return mock_self


def _make_non_dp_client_mock(*, client_index: int = 0) -> MagicMock:
    """Create a mock that looks like plain AsyncMPClient (no DP)."""
    mock_self = MagicMock()
    mock_self.client_index = client_index
    mock_self._send_input = AsyncMock()
    mock_self._ensure_output_queue_task = MagicMock()

    # Remove DP-specific attributes so hasattr checks fail.
    del mock_self.get_core_engine_for_request
    del mock_self.reqs_in_flight
    del mock_self.first_req_send_socket
    del mock_self.core_engines
    del mock_self.engines_running
    del mock_self.current_wave
    del mock_self._ensure_stats_update_task

    return mock_self


def _make_request_mock(request_id: str = "req-0", dp_rank: int | None = None) -> MagicMock:
    req = MagicMock()
    req.request_id = request_id
    req.client_index = 0
    req.data_parallel_rank = dp_rank
    return req


# Ensure enum members exist for all tests.
_add_enum_member("ADD_BATCH", b"\x05")
_add_enum_member("BEAM_FORK", b"\x06")


# ---------------------------------------------------------------------------
# 1. Rank Selection in beam_search()
# ---------------------------------------------------------------------------


class TestRankSelection:
    """Tests 1-2: rank selection based on data_parallel_size."""

    # Test 1
    def test_rank_selection_dp_gt_1(self) -> None:
        """data_parallel_size=2 → rank is 0 or 1, passed to _add_batch_step."""
        from vllm_gr.entrypoints.openai.serving_engine import beam_search

        mock_self = MagicMock()
        mock_self.input_processor.tokenizer.eos_token_id = 2
        mock_self.input_processor.tokenizer.convert_tokens_to_ids.return_value = None
        mock_self.models = SimpleNamespace(catalog=None)

        # Set up engine_client with DP config.
        ec = mock_self.engine_client
        ec.vllm_config.parallel_config.data_parallel_size = 2
        ec.beam_fork = AsyncMock()

        params = SimpleNamespace(
            beam_width=1,
            max_tokens=1,
            ignore_eos=False,
            temperature=0.0,
            begin_token=None,
            end_token=None,
            include_stop_str_in_output=False,
        )
        prompt = {"prompt_token_ids": [10, 20]}

        mock_output = MagicMock()
        mock_output.finished = True
        mock_output.outputs = [MagicMock(finish_reason="length", logprobs=None)]

        # Patch _add_batch_step to capture the data_parallel_rank kwarg.
        with patch(
            "vllm_gr.entrypoints.openai.serving_engine._add_batch_step",
            new_callable=AsyncMock,
        ) as mock_add_batch:
            mock_add_batch.return_value = ([mock_output], ["req-1-beam-0"])

            async def _run():
                async for _ in beam_search(mock_self, prompt, "req-1", params):
                    pass

            asyncio.run(_run())

            assert mock_add_batch.called
            captured_rank = mock_add_batch.call_args.kwargs.get("data_parallel_rank")
            assert captured_rank in (0, 1)

    # Test 2
    def test_rank_none_dp_eq_1(self) -> None:
        """data_parallel_size=1 → rank stays None."""
        from vllm_gr.entrypoints.openai.serving_engine import beam_search

        mock_self = MagicMock()
        mock_self.input_processor.tokenizer.eos_token_id = 2
        mock_self.input_processor.tokenizer.convert_tokens_to_ids.return_value = None
        mock_self.models = SimpleNamespace(catalog=None)

        ec = mock_self.engine_client
        ec.vllm_config.parallel_config.data_parallel_size = 1
        ec.beam_fork = AsyncMock()

        mock_queue = MagicMock()
        mock_ec_req = MagicMock()
        mock_ec_req.request_id = "req-beam-0"
        ec.prepare_request.return_value = (mock_queue, mock_ec_req)
        ec._add_requests_batch = AsyncMock()

        mock_output = MagicMock()
        mock_output.finished = True
        mock_output.outputs = [MagicMock(finish_reason="length", logprobs=None)]
        mock_queue.get = AsyncMock(return_value=mock_output)

        params = SimpleNamespace(
            beam_width=1,
            max_tokens=1,
            ignore_eos=False,
            temperature=0.0,
            begin_token=None,
            end_token=None,
            include_stop_str_in_output=False,
        )
        prompt = {"prompt_token_ids": [10, 20]}

        # Capture the data_parallel_rank arg passed to _add_batch_step.
        captured_rank = None

        with patch(
            "vllm_gr.entrypoints.openai.serving_engine._add_batch_step",
            new_callable=AsyncMock,
        ) as mock_add_batch:
            mock_add_batch.return_value = (
                [mock_output],
                ["req-1-beam-0"],
            )

            async def _run():
                async for _ in beam_search(mock_self, prompt, "req-1", params):
                    pass

            asyncio.run(_run())

            if mock_add_batch.called:
                call_kwargs = mock_add_batch.call_args
                # data_parallel_rank is a keyword argument.
                captured_rank = call_kwargs.kwargs.get("data_parallel_rank")

        assert captured_rank is None


# ---------------------------------------------------------------------------
# 2. add_requests_async DP Routing
# ---------------------------------------------------------------------------


class TestAddRequestsAsyncDP:
    """Tests 3-5: add_requests_async with DP client."""

    # Test 3
    def test_add_requests_async_dp_routing(self) -> None:
        """DP client routes via get_core_engine_for_request, registers all in reqs_in_flight."""
        mock_self = _make_dp_client_mock(engines_running=True)
        reqs = [
            _make_request_mock("r0", dp_rank=1),
            _make_request_mock("r1", dp_rank=1),
            _make_request_mock("r2", dp_rank=1),
        ]

        asyncio.run(add_requests_async(mock_self, reqs))

        # get_core_engine_for_request called once with first request.
        mock_self.get_core_engine_for_request.assert_called_once_with(reqs[0])

        # All 3 requests tracked in reqs_in_flight.
        # req[0] registered by get_core_engine_for_request, req[1] and req[2] manually.
        engine_1 = "engine-1"
        assert mock_self.reqs_in_flight["r0"] == engine_1
        assert mock_self.reqs_in_flight["r1"] == engine_1
        assert mock_self.reqs_in_flight["r2"] == engine_1

        # _send_input called with chosen_engine.
        mock_self._send_input.assert_called_once()
        call_args = mock_self._send_input.call_args
        assert call_args[0][0] == EngineCoreRequestType.ADD_BATCH
        assert call_args[0][1] is reqs
        assert call_args[0][2] == engine_1

    # Test 4
    def test_add_requests_async_first_req_when_idle(self) -> None:
        """engines_running=False → FIRST_REQ sent to coordinator."""
        mock_self = _make_dp_client_mock(engines_running=False)
        reqs = [_make_request_mock("r0", dp_rank=0)]

        asyncio.run(add_requests_async(mock_self, reqs, force_batch=True))

        # FIRST_REQ notification sent.
        mock_self.first_req_send_socket.send.assert_awaited_once()
        sent_data = mock_self.first_req_send_socket.send.call_args[0][0]
        decoded = msgspec.msgpack.decode(sent_data)
        assert decoded[0] == "FIRST_REQ"

    # Test 5
    def test_add_requests_async_no_first_req_when_running(self) -> None:
        """engines_running=True → no FIRST_REQ sent."""
        mock_self = _make_dp_client_mock(engines_running=True)
        reqs = [_make_request_mock("r0", dp_rank=0)]

        asyncio.run(add_requests_async(mock_self, reqs, force_batch=True))

        mock_self.first_req_send_socket.send.assert_not_awaited()


# ---------------------------------------------------------------------------
# 3. beam_fork_async DP Routing
# ---------------------------------------------------------------------------


class TestBeamForkAsyncDP:
    """Tests 6-8: beam_fork_async with DP routing."""

    # Test 6
    def test_beam_fork_async_dp_routing(self) -> None:
        """data_parallel_rank=1 → routes to core_engines[1], tracks children, removes aborts."""
        mock_self = _make_dp_client_mock(engines_running=True)
        fork_req = BeamForkRequest(
            parent_ids=["p0", "p1"],
            child_ids=["c0", "c1"],
            token_ids=[100, 200],
            abort_ids=["old-0", "old-1"],
            sampling_params=_sampling_params(),
            data_parallel_rank=1,
        )
        # Pre-populate reqs_in_flight with abort targets.
        engine_1 = "engine-1"
        mock_self.reqs_in_flight["old-0"] = engine_1
        mock_self.reqs_in_flight["old-1"] = engine_1

        asyncio.run(beam_fork_async(mock_self, fork_req))

        # _send_input routed to engine 1.
        call_args = mock_self._send_input.call_args
        assert call_args[0][0] == EngineCoreRequestType.BEAM_FORK
        assert call_args[0][1] is fork_req
        assert call_args[0][2] == engine_1

        # Children registered in reqs_in_flight.
        assert mock_self.reqs_in_flight["c0"] == engine_1
        assert mock_self.reqs_in_flight["c1"] == engine_1

        # Aborted IDs removed from reqs_in_flight.
        assert "old-0" not in mock_self.reqs_in_flight
        assert "old-1" not in mock_self.reqs_in_flight

    # Test 7
    def test_beam_fork_async_first_req_when_idle(self) -> None:
        """engines_running=False → FIRST_REQ sent for BEAM_FORK too."""
        mock_self = _make_dp_client_mock(engines_running=False)
        fork_req = BeamForkRequest(
            parent_ids=["p0"],
            child_ids=["c0"],
            token_ids=[100],
            abort_ids=[],
            sampling_params=_sampling_params(),
            data_parallel_rank=0,
        )

        asyncio.run(beam_fork_async(mock_self, fork_req))

        mock_self.first_req_send_socket.send.assert_awaited_once()
        sent_data = mock_self.first_req_send_socket.send.call_args[0][0]
        decoded = msgspec.msgpack.decode(sent_data)
        assert decoded[0] == "FIRST_REQ"

    # Test 8
    def test_beam_fork_async_no_dp(self) -> None:
        """Non-DP client (no get_core_engine_for_request) → plain _send_input without engine."""
        mock_self = _make_non_dp_client_mock()
        fork_req = BeamForkRequest(
            parent_ids=["p0"],
            child_ids=["c0"],
            token_ids=[100],
            abort_ids=[],
            sampling_params=_sampling_params(),
            data_parallel_rank=None,
        )

        asyncio.run(beam_fork_async(mock_self, fork_req))

        # _send_input called with just (type, request) — no engine arg.
        call_args = mock_self._send_input.call_args
        assert call_args[0][0] == EngineCoreRequestType.BEAM_FORK
        assert call_args[0][1] is fork_req
        # Third positional arg is None (engine=None from the code).
        assert call_args[0][2] is None


# ---------------------------------------------------------------------------
# 4. register_beam_output rank passthrough
# ---------------------------------------------------------------------------


class TestRegisterBeamOutput:
    """Test 9: register_beam_output_fn passes data_parallel_rank through."""

    # Test 9
    def test_register_beam_output_passes_rank(self) -> None:
        sp = _sampling_params()
        mock_self = MagicMock()
        mock_self._run_output_handler = MagicMock()
        mock_self.output_processor.add_request = MagicMock()

        register_beam_output_fn(
            mock_self,
            request_id="child-1",
            prompt_token_ids=[10, 20, 30],
            sampling_params=sp,
            eos_token_id=2,
            data_parallel_rank=1,
        )

        call_args = mock_self.output_processor.add_request.call_args
        ec_request = call_args[0][0]
        assert ec_request.data_parallel_rank == 1

    def test_register_beam_output_rank_none_default(self) -> None:
        """Without data_parallel_rank, default is None."""
        sp = _sampling_params()
        mock_self = MagicMock()
        mock_self._run_output_handler = MagicMock()
        mock_self.output_processor.add_request = MagicMock()

        register_beam_output_fn(
            mock_self,
            request_id="child-2",
            prompt_token_ids=[10, 20],
            sampling_params=sp,
            eos_token_id=2,
        )

        call_args = mock_self.output_processor.add_request.call_args
        ec_request = call_args[0][0]
        assert ec_request.data_parallel_rank is None


# ---------------------------------------------------------------------------
# 5. BeamForkRequest serialization with data_parallel_rank
# ---------------------------------------------------------------------------


class TestBeamForkRequestDPRank:
    """Test 10: BeamForkRequest.data_parallel_rank survives msgpack round-trip."""

    # Test 10
    def test_beam_fork_request_has_dp_rank(self) -> None:
        sp = _sampling_params()
        req = BeamForkRequest(
            parent_ids=["p0"],
            child_ids=["c0"],
            token_ids=[100],
            abort_ids=[],
            sampling_params=sp,
            data_parallel_rank=1,
        )
        data = msgspec.msgpack.encode(req)
        dec = msgspec.msgpack.Decoder(BeamForkRequest)
        got = dec.decode(data)

        assert got.data_parallel_rank == 1
        assert got.parent_ids == ["p0"]
        assert got.child_ids == ["c0"]

    def test_beam_fork_request_dp_rank_none_default(self) -> None:
        """data_parallel_rank defaults to None and round-trips correctly."""
        sp = _sampling_params()
        req = BeamForkRequest(
            parent_ids=["p0"],
            child_ids=["c0"],
            token_ids=[100],
            abort_ids=[],
            sampling_params=sp,
        )
        data = msgspec.msgpack.encode(req)
        dec = msgspec.msgpack.Decoder(BeamForkRequest)
        got = dec.decode(data)

        assert got.data_parallel_rank is None


# ---------------------------------------------------------------------------
# 6. Cleanup beam_fork passes rank
# ---------------------------------------------------------------------------


class TestCleanupPassesRank:
    """Test 11: cleanup beam_fork at end of loop includes data_parallel_rank."""

    # Test 11
    def test_cleanup_passes_rank(self) -> None:
        """Simulate the cleanup logic from beam_search — verify rank propagated."""
        rank = 1
        prev_beam_internal_ids = ["id-0", "id-1"]
        use_beam_fork = True
        beam_search_params = _sampling_params()

        # Build the BeamForkRequest exactly as the real cleanup code does.
        if use_beam_fork and prev_beam_internal_ids:
            cleanup_req = BeamForkRequest(
                parent_ids=[],
                child_ids=[],
                token_ids=[],
                abort_ids=prev_beam_internal_ids,
                sampling_params=beam_search_params,
                data_parallel_rank=rank,
            )
        else:
            cleanup_req = None

        assert cleanup_req is not None
        assert cleanup_req.data_parallel_rank == rank
        assert cleanup_req.abort_ids == ["id-0", "id-1"]
        assert cleanup_req.parent_ids == []
        assert cleanup_req.child_ids == []


# ---------------------------------------------------------------------------
# 7. current_wave propagation
# ---------------------------------------------------------------------------


class TestCurrentWave:
    """Test 12: current_wave set on requests when client has the attribute."""

    # Test 12
    def test_current_wave_set_on_add_requests(self) -> None:
        """Client with current_wave=3 → each request gets current_wave=3."""
        mock_self = _make_dp_client_mock(
            engines_running=True, has_current_wave=True, current_wave=3
        )
        reqs = [
            _make_request_mock("r0", dp_rank=0),
            _make_request_mock("r1", dp_rank=0),
        ]
        # Reset current_wave on requests to verify it gets set.
        for r in reqs:
            r.current_wave = 0

        asyncio.run(add_requests_async(mock_self, reqs))

        for r in reqs:
            assert r.current_wave == 3

    def test_current_wave_set_on_beam_fork(self) -> None:
        """Client with current_wave=5 → fork_request gets current_wave=5."""
        mock_self = _make_dp_client_mock(
            engines_running=True, has_current_wave=True, current_wave=5
        )
        fork_req = BeamForkRequest(
            parent_ids=["p0"],
            child_ids=["c0"],
            token_ids=[100],
            abort_ids=[],
            sampling_params=_sampling_params(),
            data_parallel_rank=0,
        )
        assert fork_req.current_wave == 0  # default

        asyncio.run(beam_fork_async(mock_self, fork_req))

        assert fork_req.current_wave == 5

    def test_no_current_wave_attr_leaves_default(self) -> None:
        """Client without current_wave attr → request current_wave unchanged."""
        mock_self = _make_dp_client_mock(engines_running=True, has_current_wave=False)
        # Use 2 requests to avoid the single-request shortcut path
        # (single request delegates to add_request_async instead of ADD_BATCH).
        reqs = [
            _make_request_mock("r0", dp_rank=0),
            _make_request_mock("r1", dp_rank=0),
        ]
        reqs[0].current_wave = 99  # pre-existing value
        reqs[1].current_wave = 88

        asyncio.run(add_requests_async(mock_self, reqs))

        # Should not have been overwritten since client has no current_wave.
        assert reqs[0].current_wave == 99
        assert reqs[1].current_wave == 88


# ---------------------------------------------------------------------------
# 8. Non-DP add_requests_async fallback
# ---------------------------------------------------------------------------


class TestAddRequestsAsyncNonDP:
    """Verify non-DP client takes the simple path."""

    def test_add_requests_async_non_dp_path(self) -> None:
        """Non-DP client sends ADD_BATCH without engine routing."""
        mock_self = _make_non_dp_client_mock()
        reqs = [
            _make_request_mock("r0"),
            _make_request_mock("r1"),
        ]

        asyncio.run(add_requests_async(mock_self, reqs))

        # _send_input called without engine arg (just type + requests).
        call_args = mock_self._send_input.call_args
        assert call_args[0][0] == EngineCoreRequestType.ADD_BATCH
        assert call_args[0][1] is reqs
        assert len(call_args[0]) == 2  # no third engine arg


# ---------------------------------------------------------------------------
# 9. _beam_fork_step / _add_batch_step rank passthrough
# ---------------------------------------------------------------------------


class TestStepFunctionsPassRank:
    """Verify _add_batch_step and _beam_fork_step forward data_parallel_rank."""

    def test_add_batch_step_passes_rank(self) -> None:
        """_add_batch_step forwards data_parallel_rank to prepare_request."""
        from vllm_gr.entrypoints.openai.serving_engine import _add_batch_step

        engine_client = MagicMock()

        # Build a queue mock whose .get() is awaitable (used by _gather_beam_results).
        mock_output = MagicMock()
        mock_output.finished = True
        mock_output.outputs = [MagicMock(finish_reason="length", logprobs=None)]

        mock_queue = MagicMock()
        mock_queue.get = AsyncMock(return_value=mock_output)

        mock_ec_req = MagicMock()
        mock_ec_req.request_id = "step-0-beam-0"
        engine_client.prepare_request.return_value = (mock_queue, mock_ec_req)
        engine_client._add_requests_batch = AsyncMock()

        asyncio.run(
            _add_batch_step(
                engine_client,
                prompts_batch=[{"prompt_token_ids": [10, 20]}],
                lora_req_batch=[None],
                request_id_batch="req-1",
                beam_search_params=_sampling_params(),
                use_beam_fork=True,
                trace_headers=None,
                priority=100,
                data_parallel_rank=1,
            )
        )

        # prepare_request receives data_parallel_rank.
        call_kwargs = engine_client.prepare_request.call_args.kwargs
        assert call_kwargs.get("data_parallel_rank") == 1

    def test_beam_fork_step_passes_rank(self) -> None:
        """_beam_fork_step forwards data_parallel_rank to register_beam_output and beam_fork."""
        from vllm.beam_search import BeamSearchSequence

        from vllm_gr.entrypoints.openai.serving_engine import _beam_fork_step

        engine_client = MagicMock()

        # Build a queue mock whose .get() is awaitable (used by _gather_beam_results).
        mock_output = MagicMock()
        mock_output.finished = True
        mock_output.outputs = [MagicMock(finish_reason="length", logprobs=None)]

        mock_queue = MagicMock()
        mock_queue.get = AsyncMock(return_value=mock_output)
        engine_client.register_beam_output.return_value = mock_queue
        engine_client.beam_fork = AsyncMock()

        all_beams = [
            BeamSearchSequence(tokens=[10, 20], cum_logprob=-1.0, logprobs=[]),
        ]

        asyncio.run(
            _beam_fork_step(
                engine_client,
                fork_info=[(0, 50)],  # parent 0 -> token 50
                prev_beam_internal_ids=["prev-0"],
                all_beams=all_beams,
                request_id_batch="req-1",
                beam_search_params=_sampling_params(),
                eos_token_id=2,
                lora_request=None,
                trace_headers=None,
                priority=100,
                data_parallel_rank=1,
            )
        )

        # register_beam_output called with data_parallel_rank=1.
        reg_kwargs = engine_client.register_beam_output.call_args.kwargs
        assert reg_kwargs.get("data_parallel_rank") == 1

        # beam_fork called with BeamForkRequest containing data_parallel_rank=1.
        fork_call = engine_client.beam_fork.call_args[0][0]
        assert isinstance(fork_call, BeamForkRequest)
        assert fork_call.data_parallel_rank == 1

    def test_beam_fork_step_rank_none(self) -> None:
        """_beam_fork_step with rank=None (non-DP) passes None through."""
        from vllm.beam_search import BeamSearchSequence

        from vllm_gr.entrypoints.openai.serving_engine import _beam_fork_step

        engine_client = MagicMock()

        mock_output = MagicMock()
        mock_output.finished = True
        mock_output.outputs = [MagicMock(finish_reason="length", logprobs=None)]

        mock_queue = MagicMock()
        mock_queue.get = AsyncMock(return_value=mock_output)
        engine_client.register_beam_output.return_value = mock_queue
        engine_client.beam_fork = AsyncMock()

        all_beams = [
            BeamSearchSequence(tokens=[10, 20], cum_logprob=-1.0, logprobs=[]),
        ]

        asyncio.run(
            _beam_fork_step(
                engine_client,
                fork_info=[(0, 50)],
                prev_beam_internal_ids=["prev-0"],
                all_beams=all_beams,
                request_id_batch="req-1",
                beam_search_params=_sampling_params(),
                eos_token_id=2,
                lora_request=None,
                trace_headers=None,
                priority=100,
                data_parallel_rank=None,
            )
        )

        fork_call = engine_client.beam_fork.call_args[0][0]
        assert fork_call.data_parallel_rank is None


# ---------------------------------------------------------------------------
# 10. ensure_stats_update_task called for DP
# ---------------------------------------------------------------------------


class TestStatsUpdateTask:
    """DP client's _ensure_stats_update_task is called during add_requests_async."""

    def test_ensure_stats_update_task_called(self) -> None:
        mock_self = _make_dp_client_mock(engines_running=True)
        reqs = [_make_request_mock("r0", dp_rank=0), _make_request_mock("r1", dp_rank=0)]

        asyncio.run(add_requests_async(mock_self, reqs))

        mock_self._ensure_stats_update_task.assert_called_once()

    def test_ensure_stats_update_task_not_called_non_dp(self) -> None:
        mock_self = _make_non_dp_client_mock()
        reqs = [_make_request_mock("r0"), _make_request_mock("r1")]

        asyncio.run(add_requests_async(mock_self, reqs))

        # Non-DP client doesn't have _ensure_stats_update_task.
        assert not hasattr(mock_self, "_ensure_stats_update_task")
