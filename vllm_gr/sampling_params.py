# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.sampling_params import BeamSearchParams as BaseBeamSearchParams


class BeamSearchParams(
    BaseBeamSearchParams,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.
    dict=True,
):  # type: ignore[call-arg]
    """Beam search parameters for text generation."""

    begin_token: str | None = None
    end_token: str | None = None
