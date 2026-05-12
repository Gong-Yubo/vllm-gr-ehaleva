# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optimized FlatLogprobs.append_fast: uses extend instead of
per-element append."""

import itertools
from collections.abc import Iterable


def append_fast(
    self,
    token_ids: list[int],
    logprobs: list[float],
    ranks: "Iterable[int | None] | list[int | None]",
    decoded_tokens: "Iterable[str | None]",
) -> None:
    self.start_indices.append(len(self.logprobs))
    self.token_ids.extend(token_ids)
    self.logprobs.extend(logprobs)
    # ranks/decoded_tokens may be lazy or infinite iterators
    # (e.g. itertools.chain, itertools.repeat) -- use islice to bound them.
    # When they are already lists, extend directly (C-level memcpy).
    if isinstance(ranks, list):
        self.ranks.extend(ranks)
    else:
        self.ranks.extend(itertools.islice(ranks, len(token_ids)))
    if isinstance(decoded_tokens, list):
        self.decoded_tokens.extend(decoded_tokens)
    else:
        self.decoded_tokens.extend(itertools.islice(decoded_tokens, len(token_ids)))
    self.end_indices.append(len(self.logprobs))


def extract_and_dedup_flat_logprobs(flat, logprobs_num):
    """Extract position-0 logprobs from FlatLogprobs, dedup sampled token.

    Flat arrays store [sampled, top1, ..., topK] — the only possible
    duplicate is position 0 (sampled token) when it also appears in the
    top-K (positions 1..K).  Returns exactly ``logprobs_num`` entries.
    """
    start = flat.start_indices[0]
    end = flat.end_indices[0]
    token_ids = flat.token_ids[start:end]
    lps = flat.logprobs[start:end]
    ranks = flat.ranks[start:end]
    decoded = flat.decoded_tokens[start:end]
    if len(token_ids) > logprobs_num:
        if any(token_ids[0] == token_ids[j] for j in range(1, logprobs_num + 1)):
            s = slice(1, logprobs_num + 1)
        else:
            s = slice(logprobs_num)
        token_ids, lps, ranks, decoded = (token_ids[s], lps[s], ranks[s], decoded[s])
    return token_ids, lps, ranks, decoded


def reconstruct_beam_logprobs(beam, initial_logprobs, sid_end_token_id=None):
    """Walk parent-pointer chain to build full FlatLogprobs for a beam.

    Replaces ``beam.logprobs`` in-place with a newly constructed
    ``FlatLogprobs`` containing the initial logprobs plus all step data
    collected via ``_lp_parent`` / ``_lp_step_data`` pointers.  If
    ``sid_end_token_id`` is given and the beam did not finish with EOS,
    an extra entry is appended for the session-end token.
    """
    from vllm.logprobs import FlatLogprobs

    steps = []
    ancestor = beam
    while getattr(ancestor, "_lp_step_data", None) is not None:
        steps.append(ancestor._lp_step_data)
        ancestor = ancestor._lp_parent
    steps.reverse()
    lp = FlatLogprobs(
        start_indices=list(initial_logprobs.start_indices),
        end_indices=list(initial_logprobs.end_indices),
        token_ids=list(initial_logprobs.token_ids),
        logprobs=list(initial_logprobs.logprobs),
        ranks=list(initial_logprobs.ranks),
        decoded_tokens=list(initial_logprobs.decoded_tokens),
    )
    for sd in steps:
        lp.append_fast(sd[0], sd[1], sd[2], sd[3])
    if sid_end_token_id is not None and beam.finish_reason != "stop":
        lp.append_fast([sid_end_token_id], [0.0], [None], [None])
    beam.logprobs = lp
