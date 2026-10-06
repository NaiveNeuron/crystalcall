"""Deterministic offline replay: crystallization analysis, oracle policies and
latency simulation.

Everything in this module is a pure function of a stored trajectory, so every
policy ablation and latency regime is reproducible from the append-only traces
without re-running the model.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canonicalize import CanonicalCall
from .effects import EffectRegistry, ResourceClass
from .parse_ptc import ParseResult, PtcExtractor
from .scheduler import (
    CostModel,
    GatePolicy,
    RemainingDecodeModel,
    RevisionTracker,
)
from .tokens import CanvasDecoder, DecodedCanvas
from .trace import CanvasRevision, TraceReader

__all__ = [
    "Trajectory",
    "RevisionView",
    "FinalCallAnalysis",
    "TrajectoryAnalysis",
    "analyze_trajectory",
    "PolicyLaunch",
    "PolicyOutcome",
    "simulate_policy",
    "LATENCY_CONDITIONS_MS",
    "BASELINE_POLICIES",
    "PREACT_POLICIES",
    "PREACT_ARM",
    "load_trajectories",
]

LATENCY_CONDITIONS_MS: tuple[float, ...] = (100.0, 500.0, 2000.0, 5000.0)

#: Offline policy for the P-ReAct-style arm.  It is NOT in
#: :data:`BASELINE_POLICIES` because it is not derivable from a standard trace:
#: it reads `tool_call_ready`, which only a priority-decoding server emits.
PREACT_POLICIES: tuple[str, ...] = ("preact_ready",)

#: The only arm whose server emits `tool_call_ready`.
PREACT_ARM = "preact_style"

#: Offline policies derived from a standard diffusion trace.
BASELINE_POLICIES: tuple[str, ...] = (
    "diffusion_final_response",
    "diffusion_final_canvas",
    "diffusion_earliest_committed",
    "oracle_earliest_invariant",
    "oracle_final_match",
)

#: Offline policies supplied by a script rather than by this module: name ->
#: launcher ``(traj, views, analysis, registry, tool_duration_ms) -> launches``,
#: where each launch names an identity, its launch revision (index into
#: ``views``) and the reason.  Final calls the launcher never fired on are
#: appended as ``not_speculated`` exactly as for a :class:`GatePolicy`.  Used by
#: ``scripts/learned_gate.py`` to replay a cross-validated predictor through the
#: unchanged latency simulation; nothing in the library registers here.
ExternalLaunch = tuple[str, int, str]
EXTERNAL_LAUNCHERS: dict[str, Callable[..., list[ExternalLaunch]]] = {}


# --------------------------------------------------------------------------- #
# trajectory container
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class RevisionView:
    """A revision plus its decoded canvas and parse result."""

    revision: CanvasRevision
    decoded: DecodedCanvas
    parse: ParseResult

    @property
    def step(self) -> int:
        return self.revision.step

    @property
    def ns(self) -> int:
        return self.revision.monotonic_ns


@dataclass(slots=True)
class Trajectory:
    """One model generation with its full revision history."""

    request_id: str
    prompt_id: str
    category: str
    seed: int
    arm: str
    revisions: list[CanvasRevision]
    final_text: str
    final_program: str | None
    request_start_ns: int
    response_done_ns: int
    canvas_done_ns: dict[int, int] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    #: ``tool_call_ready`` events recorded by a priority-decoding server (the
    #: P-ReAct-style arm).  Each entry is
    #: ``{"canvas_id","step","monotonic_ns","span_lo","span_hi"}``.  Empty for
    #: every other arm, which is how `preact_ready` refuses to run on them.
    ready: list[dict[str, Any]] = field(default_factory=list)
    #: Populated by the harness: the authoritative canonical calls actually
    #: executed by OMP, in execution order.
    authoritative_calls: list[dict[str, Any]] = field(default_factory=list)
    valid_final_parse: bool = False
    benchmark_correct: bool | None = None
    exclusion: str | None = None
    #: Per-call claim verification recorded by `HostToolRouter(verify_claims=True)`:
    #: {"claimed", "verified_match", "verified_divergent", "verification_error"}.
    #: A claim that was never verified, or whose verifier raised, is NOT evidence
    #: of losslessness, so it must be visible downstream.
    claims: dict[str, int] = field(default_factory=dict)
    #: What the ONLINE speculative runtime actually did on this request, as
    #: recorded by the future table.  Replaying the same gate over the same
    #: trace should reproduce it; a mismatch is a bug signal, not a result, so
    #: both are carried and compared rather than one being trusted.
    observed: dict[str, Any] = field(default_factory=dict)

    @property
    def decode_ms(self) -> float:
        return max(0.0, (self.response_done_ns - self.request_start_ns) / 1e6)

    def rel_ms(self, ns: int) -> float:
        return (ns - self.request_start_ns) / 1e6

    def to_json(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prompt_id": self.prompt_id,
            "category": self.category,
            "seed": self.seed,
            "arm": self.arm,
            "n_revisions": len(self.revisions),
            "n_ready": len(self.ready),
            "final_program": self.final_program,
            "request_start_ns": self.request_start_ns,
            "response_done_ns": self.response_done_ns,
            "decode_ms": self.decode_ms,
            "valid_final_parse": self.valid_final_parse,
            "benchmark_correct": self.benchmark_correct,
            "exclusion": self.exclusion,
            "authoritative_calls": self.authoritative_calls,
            "meta": self.meta,
        }


# --------------------------------------------------------------------------- #
# crystallization analysis
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class FinalCallAnalysis:
    """Crystallization timings for one authoritative final call."""

    identity: str
    tool: str
    args_digest: str
    snapshot_id: str | None
    effect: str
    eligible: bool

    # First revision index / monotonic ns at which each milestone holds.
    first_name_step: int = -1
    first_name_ns: int = 0
    first_args_step: int = -1
    first_args_ns: int = 0
    first_exec_step: int = -1
    first_exec_ns: int = 0
    invariant_step: int = -1
    invariant_ns: int = 0
    committed_step: int = -1
    committed_ns: int = 0
    canvas_done_ns: int = 0
    response_done_ns: int = 0
    mutations: int = 0
    appearances: int = 0
    disappearances: int = 0
    first_canvas_only: bool = True
    canvas_id: int = -1

    def lead_ms(self) -> float | None:
        """Oracle pre-commit lead: commit time minus earliest invariant time."""
        if self.invariant_ns and self.committed_ns:
            return (self.committed_ns - self.invariant_ns) / 1e6
        return None

    def lead_vs_response_ms(self) -> float | None:
        if self.invariant_ns and self.response_done_ns:
            return (self.response_done_ns - self.invariant_ns) / 1e6
        return None

    def lead_fraction_of_remaining(self) -> float | None:
        """Lead as a fraction of the decode time remaining at the launch point."""
        if not self.invariant_ns or not self.response_done_ns:
            return None
        remaining = self.response_done_ns - self.invariant_ns
        total_after = self.response_done_ns - self.invariant_ns
        if total_after <= 0:
            return 0.0
        if not self.committed_ns:
            return None
        lead = self.committed_ns - self.invariant_ns
        return max(0.0, lead / remaining) if remaining > 0 else None

    def to_json(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "tool": self.tool,
            "args_digest": self.args_digest,
            "snapshot_id": self.snapshot_id,
            "effect": self.effect,
            "eligible": self.eligible,
            "first_name_step": self.first_name_step,
            "first_args_step": self.first_args_step,
            "first_exec_step": self.first_exec_step,
            "invariant_step": self.invariant_step,
            "committed_step": self.committed_step,
            "first_name_ns": self.first_name_ns,
            "first_args_ns": self.first_args_ns,
            "first_exec_ns": self.first_exec_ns,
            "invariant_ns": self.invariant_ns,
            "committed_ns": self.committed_ns,
            "canvas_done_ns": self.canvas_done_ns,
            "response_done_ns": self.response_done_ns,
            "mutations": self.mutations,
            "appearances": self.appearances,
            "disappearances": self.disappearances,
            "first_canvas_only": self.first_canvas_only,
            "canvas_id": self.canvas_id,
            "lead_ms": self.lead_ms(),
            "lead_vs_response_ms": self.lead_vs_response_ms(),
        }


@dataclass(slots=True)
class TrajectoryAnalysis:
    trajectory_id: str
    prompt_id: str
    category: str
    seed: int
    valid_final_parse: bool
    final_calls: list[FinalCallAnalysis]
    revision_count: int
    candidate_revisions: int
    total_candidates: int
    mutations: int
    decode_ms: float
    parse_failures: int
    rejection_counts: dict[str, int]
    exclusion: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "prompt_id": self.prompt_id,
            "category": self.category,
            "seed": self.seed,
            "valid_final_parse": self.valid_final_parse,
            "revision_count": self.revision_count,
            "candidate_revisions": self.candidate_revisions,
            "total_candidates": self.total_candidates,
            "mutations": self.mutations,
            "decode_ms": self.decode_ms,
            "parse_failures": self.parse_failures,
            "rejection_counts": self.rejection_counts,
            "exclusion": self.exclusion,
            "final_calls": [c.to_json() for c in self.final_calls],
        }


def build_views(
    traj: Trajectory, extractor: PtcExtractor, decoder: CanvasDecoder
) -> list[RevisionView]:
    views: list[RevisionView] = []
    for rev in traj.revisions:
        decoded = decoder.decode(rev.predicted_token_ids)
        parse = extractor.extract(decoded.text)
        views.append(RevisionView(revision=rev, decoded=decoded, parse=parse))
    return views


def _observed_from_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Summarise the online future table from a trace's runtime report."""
    futures = runtime.get("futures") or []
    if not isinstance(futures, list):
        return {}
    states: dict[str, int] = {}
    for f in futures:
        if not isinstance(f, Mapping):
            continue
        st = str(f.get("state", "unknown"))
        states[st] = states.get(st, 0) + 1
    launched = len(futures)
    claimed = states.get("claimed", 0)
    wasted = states.get("discarded", 0) + states.get("cancelled", 0)
    out: dict[str, Any] = {
        "launched": launched,
        "claimed": claimed,
        "wasted": wasted,
        "failed": states.get("failed", 0),
        "by_state": states,
        "duplicate_hits": sum(
            int(f.get("duplicate_hits", 0) or 0) for f in futures if isinstance(f, Mapping)
        ),
        "precision": (claimed / launched) if launched else None,
    }
    stats = runtime.get("stats") or {}
    if isinstance(stats, Mapping):
        out["revisions"] = stats.get("revisions")
        out["parses"] = stats.get("parses")
        out["gate_launches"] = stats.get("launches")
    return out

def _final_calls(traj: Trajectory, registry: EffectRegistry) -> list[CanonicalCall]:
    from .canonicalize import canonicalize

    out: list[CanonicalCall] = []
    for rec in traj.authoritative_calls:
        tool = rec.get("tool")
        args = rec.get("args") or {}
        if not isinstance(tool, str):
            continue
        try:
            out.append(canonicalize(registry, tool, args))
        except Exception:  # noqa: BLE001 - preserved as an exclusion upstream
            continue
    return out


def analyze_trajectory(
    traj: Trajectory,
    views: Sequence[RevisionView],
    registry: EffectRegistry,
) -> TrajectoryAnalysis:
    """Compute every crystallization milestone for each authoritative call."""
    finals = _final_calls(traj, registry)
    final_ids = {c.identity for c in finals}

    # Per-revision identity sets and per-final-name / per-final-args presence.
    id_sets: list[frozenset[str]] = []
    for v in views:
        id_sets.append(frozenset(c.identity for c in v.parse.candidates))

    mutations = 0
    prev: frozenset[str] = frozenset()
    for s in id_sets:
        if s != prev:
            mutations += 1
        prev = s

    analyses: list[FinalCallAnalysis] = []
    for fc in finals:
        spec = registry.get(fc.tool)
        a = FinalCallAnalysis(
            identity=fc.identity,
            tool=fc.tool,
            args_digest=fc.args_digest,
            snapshot_id=fc.snapshot_id,
            effect=fc.effect.value,
            eligible=bool(spec and spec.speculation_eligible),
            response_done_ns=traj.response_done_ns,
        )
        required = {
            p.name for p in (spec.params if spec else ()) if p.required
        }
        final_args = fc.args_dict

        appearances = 0
        seen_ids: list[bool] = []
        canvases: set[int] = set()
        for v in views:
            rev = v.revision
            has_name = False
            has_args = False
            has_exact = False
            for cand in v.parse.candidates:
                if cand.call.tool != fc.tool:
                    continue
                has_name = True
                cargs = cand.call.args_dict
                if all(k in cargs and cargs[k] == final_args[k] for k in required):
                    has_args = True
                if cand.identity == fc.identity:
                    has_exact = True
                    canvases.add(rev.canvas_id)
                    if a.canvas_id < 0:
                        a.canvas_id = rev.canvas_id
                    tok_lo, tok_hi = v.decoded.token_range(*cand.canvas_span)
                    if a.committed_step < 0 and tok_hi > tok_lo:
                        span = rev.accepted_mask[tok_lo:tok_hi]
                        if span and all(span):
                            a.committed_step = rev.step
                            a.committed_ns = rev.monotonic_ns
            if has_name and a.first_name_step < 0:
                a.first_name_step, a.first_name_ns = rev.step, rev.monotonic_ns
            if has_args and a.first_args_step < 0:
                a.first_args_step, a.first_args_ns = rev.step, rev.monotonic_ns
            if has_exact and a.first_exec_step < 0:
                a.first_exec_step, a.first_exec_ns = rev.step, rev.monotonic_ns
            if has_exact:
                appearances += 1
            seen_ids.append(has_exact)

        a.appearances = appearances
        a.disappearances = sum(
            1 for i in range(1, len(seen_ids)) if seen_ids[i - 1] and not seen_ids[i]
        )
        # First invariant revision: the earliest index from which the exact
        # identity is present in every subsequent revision **of its own
        # canvas**.
        #
        # Scoping to the canvas is not a convenience, it is the physics: a
        # diffusion canvas commits as a unit and a committed canvas is
        # immutable, so once the canvas carrying the call has converged, no
        # later canvas can revise that call. Requiring presence through the
        # response's final revision would make the milestone unreachable by
        # construction for every multi-canvas response -- the continuation
        # canvas contains no call at all, so the scan would hit a `False`
        # immediately, report `invariant = 0` and leave every pre-commit lead
        # `None`.
        own = [i for i, v in enumerate(views) if v.revision.canvas_id == a.canvas_id]
        inv = -1
        for i in reversed(own):
            if seen_ids[i]:
                inv = i
            else:
                break
        if inv >= 0:
            a.invariant_step = views[inv].revision.step
            a.invariant_ns = views[inv].revision.monotonic_ns
        a.mutations = mutations
        a.first_canvas_only = len(canvases) <= 1
        if a.canvas_id >= 0:
            a.canvas_done_ns = traj.canvas_done_ns.get(a.canvas_id, traj.response_done_ns)
        else:
            a.canvas_done_ns = traj.response_done_ns
        if a.committed_ns == 0:
            # Never observed fully accepted inside a revision: commitment is at
            # the containing canvas's finalisation.
            a.committed_step = -1
            a.committed_ns = a.canvas_done_ns
        analyses.append(a)

    rejections: dict[str, int] = {}
    parse_failures = 0
    for v in views:
        if not v.parse.ok:
            parse_failures += 1
        for k, n in v.parse.rejection_counts.items():
            rejections[k] = rejections.get(k, 0) + n

    return TrajectoryAnalysis(
        trajectory_id=traj.request_id,
        prompt_id=traj.prompt_id,
        category=traj.category,
        seed=traj.seed,
        valid_final_parse=traj.valid_final_parse,
        final_calls=analyses,
        revision_count=len(views),
        candidate_revisions=sum(1 for s in id_sets if s),
        total_candidates=sum(len(s) for s in id_sets),
        mutations=mutations,
        decode_ms=traj.decode_ms,
        parse_failures=parse_failures,
        rejection_counts=rejections,
        exclusion=traj.exclusion,
    )


# --------------------------------------------------------------------------- #
# policy simulation
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class PolicyLaunch:
    identity: str
    tool: str
    launch_ns: int
    launch_step: int
    matched_final: bool
    resource: str
    reason: str = ""
    #: True when this launch was resolved from `analysis.final_calls` rather than
    #: from a forward gate observing revisions, i.e. `matched_final` is true by
    #: construction and the policy had no opportunity to mis-launch. Consumers
    #: MUST label precision computed over such launches.
    retrospective: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "tool": self.tool,
            "launch_ns": self.launch_ns,
            "launch_step": self.launch_step,
            "matched_final": self.matched_final,
            "resource": self.resource,
            "reason": self.reason,
            "retrospective": self.retrospective,
        }


@dataclass(slots=True)
class PolicyOutcome:
    """One (trajectory, policy, tool-duration) simulation result."""

    trajectory_id: str
    prompt_id: str
    category: str
    seed: int
    #: The experimental arm whose trace this outcome was replayed from.  The
    #: main comparison table selects (arm, policy) pairs, because the same
    #: offline policy applied to two arms' traces is two different results.
    arm: str
    policy: str
    tool_duration_ms: float
    completion_ms: float
    baseline_completion_ms: float
    launches: int
    true_launches: int
    false_launches: int
    claimed: int
    wasted_tool_ms: float
    hidden_overlap_ms: float
    #: Instrumentation + parse cost already contained in the replayed decode
    #: timeline, reported so the overhead is auditable without being charged
    #: twice.  Every diffusion policy shares this same timeline.
    instrumentation_in_timeline_ms: float
    eligible_final_calls: int
    covered_final_calls: int
    correct: bool | None
    semantic_divergence: bool
    #: Claim-verification accounting carried through from the trace.  S6 must
    #: fail closed: zero observed divergence is only evidence of losslessness
    #: when every claim was actually verified.
    claims_claimed: int = 0
    claims_verified_match: int = 0
    claims_verified_divergent: int = 0
    claims_verification_error: int = 0
    #: Online future-table observation and whether the offline replay of the
    #: same gate reproduced it.  Only meaningful for the arm's own gate policy.
    observed_launched: int | None = None
    observed_claimed: int | None = None
    observed_wasted: int | None = None
    observed_matches_replay: bool | None = None
    exclusion: str | None = None
    #: True when every launch this policy makes is resolved from
    #: `analysis.final_calls`, so its precision cannot fall below 100% by
    #: construction; ``None`` when the policy launched nothing here, which
    #: carries no information either way.
    retrospective: bool | None = None
    launch_records: list[PolicyLaunch] = field(default_factory=list)

    @property
    def speedup(self) -> float | None:
        if self.baseline_completion_ms <= 0:
            return None
        return 1.0 - self.completion_ms / self.baseline_completion_ms

    def to_json(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "prompt_id": self.prompt_id,
            "category": self.category,
            "seed": self.seed,
            "arm": self.arm,
            "policy": self.policy,
            "tool_duration_ms": self.tool_duration_ms,
            "completion_ms": self.completion_ms,
            "baseline_completion_ms": self.baseline_completion_ms,
            "speedup": self.speedup,
            "launches": self.launches,
            "retrospective": self.retrospective,
            "true_launches": self.true_launches,
            "false_launches": self.false_launches,
            "claimed": self.claimed,
            "wasted_tool_ms": self.wasted_tool_ms,
            "hidden_overlap_ms": self.hidden_overlap_ms,
            "instrumentation_in_timeline_ms": self.instrumentation_in_timeline_ms,
            "eligible_final_calls": self.eligible_final_calls,
            "covered_final_calls": self.covered_final_calls,
            "correct": self.correct,
            "semantic_divergence": self.semantic_divergence,
            "claims_claimed": self.claims_claimed,
            "claims_verified_match": self.claims_verified_match,
            "claims_verified_divergent": self.claims_verified_divergent,
            "claims_verification_error": self.claims_verification_error,
            "observed_launched": self.observed_launched,
            "observed_claimed": self.observed_claimed,
            "observed_wasted": self.observed_wasted,
            "observed_matches_replay": self.observed_matches_replay,
            "claims_unverified": max(
                0,
                self.claims_claimed
                - self.claims_verified_match
                - self.claims_verified_divergent,
            ),
            "verification_complete": (
                self.claims_verification_error == 0
                and self.claims_claimed
                == self.claims_verified_match + self.claims_verified_divergent
            ),
            "exclusion": self.exclusion,
        }


def _policy_launches(
    traj: Trajectory,
    views: Sequence[RevisionView],
    analysis: TrajectoryAnalysis,
    registry: EffectRegistry,
    policy: str | GatePolicy,
    *,
    remaining_model: RemainingDecodeModel,
    costs: CostModel,
    tool_duration_ms: float | None = None,
) -> list[PolicyLaunch]:
    """Compute the launch time of every call under one policy."""
    final_ids = {c.identity for c in analysis.final_calls}

    def _res(tool: str) -> str:
        spec = registry.get(tool)
        return spec.resource.value if spec else ResourceClass.CPU.value

    if isinstance(policy, str):
        if policy == "diffusion_final_response":
            return [
                PolicyLaunch(
                    identity=fc.identity,
                    tool=fc.tool,
                    launch_ns=traj.response_done_ns,
                    launch_step=-1,
                    matched_final=True,
                    retrospective=True,
                    resource=_res(fc.tool),
                    reason="final_response",
                )
                for fc in analysis.final_calls
            ]
        if policy == "diffusion_final_canvas":
            return [
                PolicyLaunch(
                    identity=fc.identity,
                    tool=fc.tool,
                    launch_ns=fc.canvas_done_ns or traj.response_done_ns,
                    launch_step=-1,
                    matched_final=True,
                    retrospective=True,
                    resource=_res(fc.tool),
                    reason="final_canvas",
                )
                for fc in analysis.final_calls
            ]
        if policy == "diffusion_earliest_committed":
            return [
                PolicyLaunch(
                    identity=fc.identity,
                    tool=fc.tool,
                    launch_ns=fc.committed_ns or traj.response_done_ns,
                    launch_step=fc.committed_step,
                    matched_final=True,
                    retrospective=True,
                    resource=_res(fc.tool),
                    reason="earliest_committed",
                )
                for fc in analysis.final_calls
            ]
        if policy == "oracle_earliest_invariant":
            return [
                PolicyLaunch(
                    identity=fc.identity,
                    tool=fc.tool,
                    launch_ns=fc.invariant_ns or fc.committed_ns or traj.response_done_ns,
                    launch_step=fc.invariant_step,
                    matched_final=True,
                    retrospective=True,
                    resource=_res(fc.tool),
                    reason="oracle_invariant",
                )
                for fc in analysis.final_calls
            ]
        if policy == "oracle_final_match":
            return [
                PolicyLaunch(
                    identity=fc.identity,
                    tool=fc.tool,
                    launch_ns=fc.first_exec_ns or fc.committed_ns or traj.response_done_ns,
                    launch_step=fc.first_exec_step,
                    matched_final=True,
                    retrospective=True,
                    resource=_res(fc.tool),
                    reason="oracle_final_match",
                )
                for fc in analysis.final_calls
            ]
        if policy == "preact_ready":
            # The P-ReAct-style arm launches on the sampler's readiness signal
            # and nothing else.  The signal is a property of *that arm's own*
            # trace, so the policy refuses to run on any other arm rather than
            # silently degrading into a different baseline -- unconditionally,
            # because a forged event on a standard trace must not be honoured.
            if traj.arm != PREACT_ARM:
                raise ValueError(
                    f"policy 'preact_ready' only applies to the {PREACT_ARM!r} arm; "
                    f"trajectory {traj.request_id!r} is arm {traj.arm!r}. Only a server "
                    f"running with CRYSTALCALL_PREACT=1 emits tool_call_ready."
                )
            # Step indices restart per canvas, so the join key must carry the
            # canvas: keying on `step` alone would attribute a readiness event
            # to a different canvas's revision.
            by_key = {(int(v.revision.canvas_id), int(v.revision.step)): v for v in views}
            out: list[PolicyLaunch] = []
            launched: set[str] = set()
            for ev in sorted(traj.ready, key=lambda e: int(e.get("monotonic_ns", 0))):
                cid = int(ev.get("canvas_id", 0) or 0)
                step = int(ev.get("step", -1))
                view = by_key.get((cid, step))
                if view is None or view.parse is None:
                    continue
                lo = int(ev.get("span_lo", -1))
                hi = int(ev.get("span_hi", -1))
                ns = int(ev.get("monotonic_ns", 0))
                for cand in view.parse.candidates:
                    ident = cand.call.identity
                    if ident in launched:
                        continue
                    # The event names one span.  A candidate elsewhere in the
                    # canvas was not signalled and must not inherit this
                    # timestamp, or an unrelated call would be credited with a
                    # launch the sampler never announced.
                    if lo >= 0 and hi > lo:
                        tok_lo, tok_hi = view.decoded.token_range(*cand.canvas_span)
                        if tok_hi <= tok_lo or tok_lo < lo or tok_hi > hi:
                            continue
                    spec = registry.get(cand.call.tool)
                    if not (spec and spec.speculation_eligible):
                        # Safety constraint 9 is global, not per-arm: no baseline
                        # launches a write or a live read early either.
                        continue
                    launched.add(ident)
                    out.append(
                        PolicyLaunch(
                            identity=ident,
                            tool=cand.call.tool,
                            launch_ns=ns,
                            launch_step=step,
                            # A readiness launch is still speculative: the span
                            # can settle to a call the final program does not
                            # contain.  Those are false launches and MUST be
                            # counted, or the baseline would look perfect by
                            # construction.
                            matched_final=ident in final_ids,
                            resource=_res(cand.call.tool),
                            reason=f"preact_ready[c{cid}:{lo}:{hi}]",
                        )
                    )
            for fc in analysis.final_calls:
                if fc.identity in launched:
                    continue
                # No readiness signal ever named this call, so the arm has
                # nothing to launch on and executes it when the turn's program
                # arrives.  Deliberately the *unflattering* fallback: this arm
                # does not watch canvas commits.
                out.append(
                    PolicyLaunch(
                        identity=fc.identity,
                        tool=fc.tool,
                        launch_ns=traj.response_done_ns,
                        launch_step=-1,
                        matched_final=True,
                        resource=_res(fc.tool),
                        reason="preact_no_ready_signal",
                    )
                )
            return out
        if policy in ("ar_streaming", "ar_sptc"):
            # An autoregressive prefix is monotonic, so the earliest executable
            # final-matching candidate is already committed: launch there.  The
            # two AR policies share this launch point and differ only in
            # execution discipline (serial vs concurrent futures), which
            # `simulate_policy` applies.
            return [
                PolicyLaunch(
                    identity=fc.identity,
                    tool=fc.tool,
                    launch_ns=fc.first_exec_ns or traj.response_done_ns,
                    launch_step=fc.first_exec_step,
                    matched_final=True,
                    resource=_res(fc.tool),
                    reason=policy,
                )
                for fc in analysis.final_calls
            ]
        launcher = EXTERNAL_LAUNCHERS.get(policy)
        if launcher is None:
            raise KeyError(f"unknown offline policy {policy!r}")
        launches = []
        launched = set()
        for identity, view_index, reason in launcher(
            traj, views, analysis, registry, tool_duration_ms
        ):
            if identity in launched:
                continue
            v = views[view_index]
            cand = next(c for c in v.parse.candidates if c.identity == identity)
            launched.add(identity)
            launches.append(
                PolicyLaunch(
                    identity=identity,
                    tool=cand.call.tool,
                    launch_ns=v.revision.monotonic_ns,
                    launch_step=v.revision.step,
                    matched_final=identity in final_ids,
                    resource=_res(cand.call.tool),
                    reason=reason,
                )
            )
        for fc in analysis.final_calls:
            if fc.identity not in launched:
                launches.append(
                    PolicyLaunch(
                        identity=fc.identity,
                        tool=fc.tool,
                        launch_ns=traj.response_done_ns,
                        launch_step=-1,
                        matched_final=True,
                        resource=_res(fc.tool),
                        reason="not_speculated",
                    )
                )
        return launches

    # An oracle entry is a *retrospective* policy, not a gate: dispatch it to
    # its offline resolver.  Running the tracker instead would silently make it
    # behave like the full gate.
    if policy.oracle is not None:
        alias = {
            "earliest_stable": "oracle_earliest_invariant",
            "final_match": "oracle_final_match",
        }.get(policy.oracle)
        if alias is None:
            raise KeyError(f"unknown oracle {policy.oracle!r} on policy {policy.name!r}")
        return _policy_launches(
            traj,
            views,
            analysis,
            registry,
            alias,
            remaining_model=remaining_model,
            costs=costs,
            tool_duration_ms=tool_duration_ms,
        )

    # A GatePolicy: replay the revision stream through the real tracker.
    tracker = RevisionTracker(
        registry=registry,
        policy=policy,
        remaining_model=remaining_model,
        costs=costs,
        request_id=traj.request_id,
        # The gate must judge overlap against the duration this replay will
        # actually charge, not a manifest's synthetic 0 ms.
        duration_override_ms=tool_duration_ms,
    )
    launches: list[PolicyLaunch] = []
    launched: set[str] = set()
    for v in views:
        for state, decision in tracker.observe(v.revision, v.parse, v.decoded):
            if not decision.launch or state.identity in launched:
                continue
            launched.add(state.identity)
            launches.append(
                PolicyLaunch(
                    identity=state.identity,
                    tool=state.call.tool,
                    launch_ns=v.revision.monotonic_ns,
                    launch_step=v.revision.step,
                    matched_final=state.identity in final_ids,
                    resource=_res(state.call.tool),
                    reason=decision.reason,
                )
            )
    # Final calls the gate never launched still execute authoritatively.
    for fc in analysis.final_calls:
        if fc.identity not in launched:
            launches.append(
                PolicyLaunch(
                    identity=fc.identity,
                    tool=fc.tool,
                    launch_ns=traj.response_done_ns,
                    launch_step=-1,
                    matched_final=True,
                    resource=_res(fc.tool),
                    reason="not_speculated",
                )
            )
    return launches


def simulate_policy(
    traj: Trajectory,
    views: Sequence[RevisionView],
    analysis: TrajectoryAnalysis,
    registry: EffectRegistry,
    policy: str | GatePolicy,
    tool_duration_ms: float,
    *,
    remaining_model: RemainingDecodeModel | None = None,
    costs: CostModel | None = None,
    claim_overhead_ms: float = 0.0,
    serial_execution: bool | None = None,
) -> PolicyOutcome:
    """Simulate end-to-end completion under one policy and tool duration.

    The authoritative program still executes at ``response_done_ns``.  A
    speculative launch can only *hide* tool time: a claimed call completes at
    ``max(t_auth, t_launch + D)``; an unclaimed or ineligible call completes at
    ``t_auth + D``.

    **Instrumentation is already in the timeline and is not charged again.**
    Every diffusion policy replays the *same* recorded trace, and that trace was
    collected with the sampler side channel and the parser live, so
    ``response_done_ns`` already contains the trace and parse cost.  Adding the
    measured :class:`CostModel` on top would double-charge it, and crediting it
    back to the final-response baseline would produce an unfair
    "fully instrumented candidate versus unrealistically stripped baseline"
    comparison.  The measured per-revision cost is
    therefore reported on its own in the Phase 0 overhead table and is used only
    by the launch gate's expected-overlap test, never to move a completion time.
    """
    rm = remaining_model or RemainingDecodeModel()
    cm = costs or CostModel()
    name = policy if isinstance(policy, str) else policy.name

    launches = _policy_launches(
        traj,
        views,
        analysis,
        registry,
        policy,
        remaining_model=rm,
        costs=cm,
        # This replay charges `tool_duration_ms` for every eligible call, so the
        # gate is told the same number.
        tool_duration_ms=tool_duration_ms,
    )
    t_auth = traj.response_done_ns
    final_ids = {c.identity for c in analysis.final_calls}
    eligible_finals = [c for c in analysis.final_calls if c.eligible]

    # Recorded for reporting only: this is what the shared decode timeline
    # already paid for instrumentation, not an extra charge (see the docstring).
    instrumentation_in_timeline_ms = (
        cm.trace_ms_per_revision + cm.parse_ms_per_revision
    ) * len(views)

    per_call_done: list[float] = []
    claimed = 0
    true_launches = 0
    false_launches = 0
    wasted_ms = 0.0
    hidden_ms = 0.0
    covered = 0

    speculated: dict[str, PolicyLaunch] = {}
    for lr in launches:
        if lr.reason == "not_speculated":
            continue
        speculated.setdefault(lr.identity, lr)

    for lr in speculated.values():
        spec = registry.get(lr.tool)
        eligible = bool(spec and spec.speculation_eligible)
        pre_commit = lr.launch_ns < t_auth
        if not eligible or not pre_commit:
            continue
        if lr.matched_final:
            true_launches += 1
        else:
            false_launches += 1
            wasted_ms += tool_duration_ms

    # Execution discipline.  `ar_streaming` runs the calls in stream order,
    # serially: the runtime blocks on each before starting the next.  Every
    # other policy overlaps independent calls, which is the sPTC contribution.
    serial = serial_execution if serial_execution is not None else (name == "ar_streaming")
    ordered = sorted(
        analysis.final_calls,
        key=lambda fc: (
            speculated[fc.identity].launch_ns if fc.identity in speculated else t_auth,
            fc.identity,
        ),
    )
    auth_ms = traj.rel_ms(t_auth)
    serial_cursor = 0.0
    for fc in ordered:
        lr = speculated.get(fc.identity)
        spec = registry.get(fc.tool)
        eligible = bool(spec and spec.speculation_eligible)
        if lr is not None and eligible and lr.launch_ns < t_auth:
            start_ms = traj.rel_ms(lr.launch_ns)
            if serial:
                start_ms = max(start_ms, serial_cursor)
            finish_ms = start_ms + tool_duration_ms
            serial_cursor = finish_ms
            done = max(auth_ms, finish_ms) + claim_overhead_ms
            hidden_ms += max(0.0, min(tool_duration_ms, auth_ms - start_ms))
            claimed += 1
            covered += 1 if fc.eligible else 0
        else:
            start_ms = max(auth_ms, serial_cursor) if serial else auth_ms
            finish_ms = start_ms + tool_duration_ms
            serial_cursor = finish_ms
            done = finish_ms
        per_call_done.append(done)

    completion = max(per_call_done) if per_call_done else traj.rel_ms(t_auth)

    # Compare the offline replay of this gate against what the online runtime
    # actually did.  Only the arm's own gate is comparable: a different policy
    # is expected to differ.
    obs = traj.observed or {}
    obs_launched = obs.get("launched")
    obs_claimed = obs.get("claimed")
    obs_wasted = obs.get("wasted")
    obs_matches: bool | None = None
    if obs_launched is not None and not isinstance(policy, str):
        frozen_gate = (traj.meta.get("policy") or {}).get("name")
        if frozen_gate and frozen_gate == name:
            # Cross-check every counter the fidelity block claims to compare,
            # not just the launch count: launches, claims and wasted work.
            checks = [int(obs_launched) == len(speculated)]
            if obs_claimed is not None:
                checks.append(int(obs_claimed) == claimed)
            if obs_wasted is not None:
                checks.append(int(obs_wasted) == false_launches)
            obs_matches = all(checks)
    baseline = traj.rel_ms(t_auth) + (tool_duration_ms if analysis.final_calls else 0.0)

    return PolicyOutcome(
        trajectory_id=traj.request_id,
        prompt_id=traj.prompt_id,
        category=traj.category,
        seed=traj.seed,
        retrospective=(
            all(lr.retrospective for lr in launches) if launches else None
        ),
        arm=traj.arm,
        policy=name,
        tool_duration_ms=tool_duration_ms,
        completion_ms=completion,
        baseline_completion_ms=baseline,
        launches=len(speculated),
        true_launches=true_launches,
        false_launches=false_launches,
        claimed=claimed,
        wasted_tool_ms=wasted_ms,
        hidden_overlap_ms=hidden_ms,
        instrumentation_in_timeline_ms=instrumentation_in_timeline_ms,
        eligible_final_calls=len(eligible_finals),
        covered_final_calls=covered,
        correct=traj.benchmark_correct,
        semantic_divergence=bool(traj.claims.get("verified_divergent", 0)),
        claims_claimed=int(traj.claims.get("claimed", 0)),
        claims_verified_match=int(traj.claims.get("verified_match", 0)),
        claims_verified_divergent=int(traj.claims.get("verified_divergent", 0)),
        claims_verification_error=int(traj.claims.get("verification_error", 0)),
        observed_launched=obs_launched,
        observed_claimed=obs_claimed,
        observed_wasted=obs_wasted,
        observed_matches_replay=obs_matches,
        exclusion=traj.exclusion,
        launch_records=list(speculated.values()),
    )


# --------------------------------------------------------------------------- #
# trace loading
# --------------------------------------------------------------------------- #


def load_trajectories(
    shards: Iterable[str | Path], *, verify: bool = True
) -> list[Trajectory]:
    """Reconstruct trajectories from append-only trace shards."""
    out: list[Trajectory] = []
    for shard in shards:
        reader = TraceReader(shard)
        if verify:
            reader.verify()
        current: dict[str, Trajectory] = {}
        shard_meta: dict[str, Any] = {}
        for rec in reader.records():
            k, p = rec.kind, rec.payload
            if k == "header":
                shard_meta = dict(p.get("meta") or {})
                continue
            if k == "request":
                current[p["request_id"]] = Trajectory(
                    request_id=p["request_id"],
                    prompt_id=p.get("prompt_id", p["request_id"]),
                    category=p.get("category", "unknown"),
                    seed=int(p.get("seed", 0)),
                    arm=p.get("arm", "unknown"),
                    revisions=[],
                    final_text="",
                    final_program=None,
                    request_start_ns=int(p.get("request_start_ns", 0)),
                    response_done_ns=0,
                    meta={**shard_meta, **(p.get("meta") or {})},
                )
            elif k == "revision":
                rev = CanvasRevision.from_json(p)
                traj = current.get(rev.request_id)
                if traj is not None:
                    traj.revisions.append(rev)
            elif k == "canvas_done":
                traj = current.get(p["request_id"])
                if traj is not None:
                    traj.canvas_done_ns[int(p["canvas_id"])] = int(p["monotonic_ns"])
            elif k == "tool_call_ready":
                traj = current.get(p["request_id"])
                if traj is not None:
                    traj.ready.append(
                        {
                            "canvas_id": int(p.get("canvas_id", 0) or 0),
                            "step": int(p.get("step", -1)),
                            "monotonic_ns": int(p.get("monotonic_ns", 0) or 0),
                            "span_lo": int(p.get("span_lo", -1)),
                            "span_hi": int(p.get("span_hi", -1)),
                        }
                    )
            elif k == "final":
                traj = current.get(p["request_id"])
                if traj is None:
                    continue
                traj.final_text = p.get("final_text", "")
                traj.final_program = p.get("final_program")
                traj.response_done_ns = int(p.get("response_done_ns", 0))
                traj.authoritative_calls = p.get("authoritative_calls", []) or []
                traj.valid_final_parse = bool(p.get("valid_final_parse", False))
                traj.benchmark_correct = p.get("benchmark_correct")
                traj.exclusion = p.get("exclusion")
                # `claims` also carries a nested `deferred_flush` record; keep
                # only the scalar counters here.
                traj.observed = _observed_from_runtime(p.get("runtime") or {})
                traj.claims = {
                    k: int(v)
                    for k, v in (p.get("claims") or {}).items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
                out.append(traj)
    return out


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
            fh.write("\n")
            n += 1
    return n
