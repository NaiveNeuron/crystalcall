"""Revision-aware launch gate and the online DenoisePTC scheduler.

The gate is training-free.  A candidate is launched at the first revision where
*all* of these hold:

1. the candidate came from a valid AST parse and is statically closed
   (guaranteed by :mod:`parse_ptc` -- a candidate cannot exist otherwise);
2. its effect is ``Pure`` or ``SnapshotRead``;
3. the identical canonical call appeared in the preceding ``k`` revisions;
4. confidence over the call's token span exceeds ``tau``;
5. expected remaining decode time is large enough to hide useful tool work;
6. the tool's resource class and concurrency limit permit overlap;
7. no identical future is already running or completed.

``k``, ``tau`` and the minimum useful overlap are the only tuned quantities and
are selected on development data only.
"""

from __future__ import annotations

import enum
import hashlib
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .canonicalize import CanonicalCall
from .effects import EffectRegistry, ResourceClass, ToolSpec
from .futures import FutureKey, FutureTable
from .parse_ptc import Candidate, ParseResult
from .tokens import DecodedCanvas
from .trace import CanvasRevision

__all__ = [
    "GatePolicy",
    "GateDecision",
    "CandidateState",
    "RevisionTracker",
    "RemainingDecodeModel",
    "CostModel",
    "POLICY_LIBRARY",
    "policy_by_name",
]


class ConfidenceMode(enum.Enum):
    MIN = "min"
    MEAN = "mean"


@dataclass(frozen=True, slots=True)
class CostModel:
    """Table-based cost accounting used by the expected-overlap test.

    All values are milliseconds and are measured in Phase 0, not guessed.
    """

    trace_ms_per_revision: float = 0.0
    parse_ms_per_revision: float = 0.0
    schedule_ms_per_launch: float = 0.0
    contention_ms_per_launch: Mapping[str, float] = field(default_factory=dict)

    def overhead_ms(self, resource: ResourceClass) -> float:
        return (
            self.trace_ms_per_revision
            + self.parse_ms_per_revision
            + self.schedule_ms_per_launch
            + float(self.contention_ms_per_launch.get(resource.value, 0.0))
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "trace_ms_per_revision": self.trace_ms_per_revision,
            "parse_ms_per_revision": self.parse_ms_per_revision,
            "schedule_ms_per_launch": self.schedule_ms_per_launch,
            "contention_ms_per_launch": dict(self.contention_ms_per_launch),
        }


@dataclass(frozen=True, slots=True)
class RemainingDecodeModel:
    """Expected remaining decode time, calibrated on development data."""

    expected_steps_per_canvas: float = 24.0
    expected_canvases: float = 1.0
    #: Fallback per-step duration used before enough intervals are observed.
    default_step_ms: float = 40.0

    def remaining_ms(
        self,
        revision: CanvasRevision,
        observed_step_ms: float | None,
    ) -> float:
        step_ms = observed_step_ms if observed_step_ms and observed_step_ms > 0 else self.default_step_ms
        steps_left_here = max(0.0, self.expected_steps_per_canvas - (revision.step + 1))
        canvases_left = max(0.0, self.expected_canvases - (revision.canvas_id + 1))
        return step_ms * (steps_left_here + canvases_left * self.expected_steps_per_canvas)

    def to_json(self) -> dict[str, Any]:
        return {
            "expected_steps_per_canvas": self.expected_steps_per_canvas,
            "expected_canvases": self.expected_canvases,
            "default_step_ms": self.default_step_ms,
        }


@dataclass(frozen=True, slots=True)
class GatePolicy:
    """A frozen launch policy.  Also used to express every gate ablation."""

    name: str
    k: int = 2
    tau: float = 0.9
    confidence_mode: ConfidenceMode = ConfidenceMode.MIN
    min_overlap_ms: float = 50.0
    # Ablation switches.
    use_persistence: bool = True
    use_confidence: bool = True
    use_overlap: bool = True
    use_resource: bool = True
    #: Gate on the entropy-bound accepted mask covering the whole span, i.e.
    #: `CandidateState.first_committed_step >= 0`. This is the signal the
    #: retrospective `diffusion_earliest_committed` reference uses; read forward
    #: it is a real policy whose false-launch rate is measurable, because this
    #: sampler renoises non-accepted positions and `accepted_count` is
    #: non-monotone in 84 of 100 development trajectories.
    use_acceptance: bool = False
    allowed_resources: frozenset[ResourceClass] = frozenset(
        {ResourceClass.CPU, ResourceClass.STORAGE, ResourceClass.GPU}
    )
    max_inflight: int = 4
    #: Ablation: launch every eligible candidate at exactly this denoising step.
    fixed_step: int | None = None
    #: Ablation: launch with this probability (deterministic hash of identity).
    random_rate: float | None = None
    #: Oracle policies are resolved offline in :mod:`replay`, not here.
    oracle: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "k": self.k,
            "tau": self.tau,
            "confidence_mode": self.confidence_mode.value,
            "min_overlap_ms": self.min_overlap_ms,
            "use_persistence": self.use_persistence,
            "use_confidence": self.use_confidence,
            "use_overlap": self.use_overlap,
            "use_resource": self.use_resource,
            "use_acceptance": self.use_acceptance,
            "allowed_resources": sorted(r.value for r in self.allowed_resources),
            "max_inflight": self.max_inflight,
            "fixed_step": self.fixed_step,
            "random_rate": self.random_rate,
            "oracle": self.oracle,
        }

    def fingerprint(self) -> str:
        import json

        return hashlib.sha256(
            json.dumps(self.to_json(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class GateDecision:
    launch: bool
    reason: str
    checks: Mapping[str, bool]
    remaining_ms: float = 0.0
    expected_overlap_ms: float = 0.0
    min_conf: float | None = None
    mean_conf: float | None = None
    consecutive: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "launch": self.launch,
            "reason": self.reason,
            "checks": dict(self.checks),
            "remaining_ms": round(self.remaining_ms, 3),
            "expected_overlap_ms": round(self.expected_overlap_ms, 3),
            "min_conf": self.min_conf,
            "mean_conf": self.mean_conf,
            "consecutive": self.consecutive,
        }


@dataclass(slots=True)
class CandidateState:
    """Per-identity history across the revision stream of one request."""

    identity: str
    call: CanonicalCall
    first_seen_step: int
    first_seen_ns: int
    first_seen_canvas: int
    last_seen_step: int = -1
    last_seen_ns: int = 0
    consecutive: int = 0
    max_consecutive: int = 0
    total_seen: int = 0
    disappearances: int = 0
    launched: bool = False
    launched_step: int = -1
    launched_ns: int = 0
    launch_decision: GateDecision | None = None
    last_min_conf: float | None = None
    last_mean_conf: float | None = None
    last_span: tuple[int, int] = (0, 0)
    last_token_span: tuple[int, int] = (0, 0)
    #: First revision at which every token of the candidate's span was accepted.
    first_committed_step: int = -1
    first_committed_ns: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "call": self.call.to_json(),
            "first_seen_step": self.first_seen_step,
            "first_seen_ns": self.first_seen_ns,
            "first_seen_canvas": self.first_seen_canvas,
            "last_seen_step": self.last_seen_step,
            "last_seen_ns": self.last_seen_ns,
            "max_consecutive": self.max_consecutive,
            "total_seen": self.total_seen,
            "disappearances": self.disappearances,
            "launched": self.launched,
            "launched_step": self.launched_step,
            "launched_ns": self.launched_ns,
            "launch_decision": self.launch_decision.to_json() if self.launch_decision else None,
            "last_min_conf": self.last_min_conf,
            "last_mean_conf": self.last_mean_conf,
            "first_committed_step": self.first_committed_step,
            "first_committed_ns": self.first_committed_ns,
        }


def _deterministic_uniform(identity: str, salt: str) -> float:
    h = hashlib.blake2b(f"{salt}|{identity}".encode(), digest_size=8).digest()
    (v,) = struct.unpack("<Q", h)
    return v / float(1 << 64)


@dataclass
class RevisionTracker:
    """Consumes the revision stream of one request and applies the gate.

    Pure bookkeeping: it decides, it never executes.  The caller (online
    scheduler or offline replay) performs the launch.
    """

    registry: EffectRegistry
    policy: GatePolicy
    remaining_model: RemainingDecodeModel = field(default_factory=RemainingDecodeModel)
    costs: CostModel = field(default_factory=CostModel)
    request_id: str = ""
    #: Tool duration the caller is actually going to charge, in ms.
    #:
    #: The gate's overlap condition asks "is there enough decode left to hide
    #: this tool's work?", so it must be told the duration the *simulation* will
    #: charge. Under a replayed latency condition the answer is that condition
    #: (100 ms, 500 ms, 2 s, 5 s); a benchmark manifest's synthetic tools
    #: declare `expected_duration_ms = 0.0`, and believing that while the
    #: simulator charges 500 ms makes the gate reject every candidate on
    #: `expected_overlap`.
    #:
    #: ``None`` keeps the declared per-tool duration, which is what the real
    #: local tools and the online runtime use.
    duration_override_ms: float | None = None
    states: dict[str, CandidateState] = field(default_factory=dict)
    revisions_seen: int = 0
    mutations: int = 0
    inflight: int = 0
    _last_identities: frozenset[str] = frozenset()
    _last_ns: int = 0
    _step_ms_ewma: float | None = None
    _launch_order: list[str] = field(default_factory=list)

    # ---- observation ---------------------------------------------------- #

    def observe(
        self,
        revision: CanvasRevision,
        parse: ParseResult,
        decoded: DecodedCanvas,
    ) -> list[tuple[CandidateState, GateDecision]]:
        self.revisions_seen += 1
        if self._last_ns:
            dt_ms = (revision.monotonic_ns - self._last_ns) / 1e6
            if dt_ms > 0:
                self._step_ms_ewma = (
                    dt_ms if self._step_ms_ewma is None else 0.6 * self._step_ms_ewma + 0.4 * dt_ms
                )
        self._last_ns = revision.monotonic_ns

        present: dict[str, Candidate] = {}
        for c in parse.candidates:
            present.setdefault(c.identity, c)
        ids = frozenset(present)
        if ids != self._last_identities:
            self.mutations += 1
        for gone in self._last_identities - ids:
            st = self.states.get(gone)
            if st is not None:
                st.consecutive = 0
                st.disappearances += 1
        self._last_identities = ids

        out: list[tuple[CandidateState, GateDecision]] = []
        for identity, cand in present.items():
            st = self.states.get(identity)
            if st is None:
                st = CandidateState(
                    identity=identity,
                    call=cand.call,
                    first_seen_step=revision.step,
                    first_seen_ns=revision.monotonic_ns,
                    first_seen_canvas=revision.canvas_id,
                )
                self.states[identity] = st
            st.last_seen_step = revision.step
            st.last_seen_ns = revision.monotonic_ns
            st.total_seen += 1
            st.consecutive += 1
            st.max_consecutive = max(st.max_consecutive, st.consecutive)
            st.last_span = cand.canvas_span

            tok_lo, tok_hi = decoded.token_range(*cand.canvas_span)
            st.last_token_span = (tok_lo, tok_hi)
            conf = revision.span_confidence(tok_lo, tok_hi)
            if conf is not None:
                st.last_min_conf, st.last_mean_conf = conf

            if st.first_committed_step < 0 and tok_hi > tok_lo:
                span_mask = revision.accepted_mask[tok_lo:tok_hi]
                if span_mask and all(span_mask):
                    st.first_committed_step = revision.step
                    st.first_committed_ns = revision.monotonic_ns

            decision = self.evaluate(st, cand, revision)
            out.append((st, decision))
            if decision.launch and not st.launched:
                st.launched = True
                st.launched_step = revision.step
                st.launched_ns = revision.monotonic_ns
                st.launch_decision = decision
                self.inflight += 1
                self._launch_order.append(identity)
        return out

    def release(self, identity: str) -> None:
        """Called when a launched future settles, freeing an in-flight slot."""
        if self.inflight > 0:
            self.inflight -= 1

    # ---- the gate ------------------------------------------------------- #

    def evaluate(
        self,
        state: CandidateState,
        candidate: Candidate,
        revision: CanvasRevision,
    ) -> GateDecision:
        p = self.policy
        checks: dict[str, bool] = {}

        checks["ast_valid"] = True  # a Candidate cannot exist without a valid parse
        checks["static_closure"] = True

        spec: ToolSpec | None = self.registry.get(candidate.call.tool)
        checks["effect_eligible"] = bool(spec and spec.speculation_eligible)
        checks["no_duplicate_future"] = not state.launched

        if p.use_persistence:
            checks["persistence"] = state.consecutive >= max(1, p.k)
        else:
            checks["persistence"] = True

        # `first_committed_step` is set by `observe` before this call, so the
        # check sees the current revision's accepted mask.
        if p.use_acceptance:
            checks["span_accepted"] = state.first_committed_step >= 0
        else:
            checks["span_accepted"] = True

        min_conf = state.last_min_conf
        mean_conf = state.last_mean_conf
        if p.use_confidence:
            value = min_conf if p.confidence_mode is ConfidenceMode.MIN else mean_conf
            checks["confidence"] = value is not None and value >= p.tau
        else:
            checks["confidence"] = True

        remaining = self.remaining_model.remaining_ms(revision, self._step_ms_ewma)
        if self.duration_override_ms is not None:
            duration = float(self.duration_override_ms)
        else:
            duration = spec.expected_duration_ms if spec else 0.0
        overhead = self.costs.overhead_ms(spec.resource) if spec else 0.0
        overlap = max(0.0, min(duration, remaining) - overhead)
        if p.use_overlap:
            checks["expected_overlap"] = overlap >= p.min_overlap_ms
        else:
            checks["expected_overlap"] = True

        if p.use_resource and spec is not None:
            checks["resource_class"] = spec.resource in p.allowed_resources
            checks["concurrency"] = self.inflight < p.max_inflight
        else:
            checks["resource_class"] = True
            checks["concurrency"] = self.inflight < max(1, p.max_inflight)

        if p.fixed_step is not None:
            checks["fixed_step"] = revision.step == p.fixed_step
        if p.random_rate is not None:
            checks["random_sample"] = (
                _deterministic_uniform(f"{self.request_id}|{state.identity}", p.name)
                < p.random_rate
            )

        launch = all(checks.values())
        reason = "gate_pass" if launch else ",".join(sorted(k for k, v in checks.items() if not v))
        return GateDecision(
            launch=launch,
            reason=reason,
            checks=checks,
            remaining_ms=remaining,
            expected_overlap_ms=overlap,
            min_conf=min_conf,
            mean_conf=mean_conf,
            consecutive=state.consecutive,
        )

    # ---- reporting ------------------------------------------------------ #

    def launched_states(self) -> tuple[CandidateState, ...]:
        return tuple(self.states[i] for i in self._launch_order if i in self.states)

    def to_json(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "policy": self.policy.to_json(),
            "revisions_seen": self.revisions_seen,
            "mutations": self.mutations,
            "candidates": [s.to_json() for s in self.states.values()],
        }


def make_future_key(request_id: str, call: CanonicalCall) -> FutureKey:
    return FutureKey.of(request_id, call)


# --------------------------------------------------------------------------- #
# the frozen policy library (ablation grid lives here, not in scripts)
# --------------------------------------------------------------------------- #

_BASE = GatePolicy(name="full_gate")

POLICY_LIBRARY: dict[str, GatePolicy] = {
    "syntax_effect_only": replace(
        _BASE,
        name="syntax_effect_only",
        use_persistence=False,
        use_confidence=False,
        use_overlap=False,
        use_resource=False,
        k=1,
    ),
    "persistence_only": replace(
        _BASE,
        name="persistence_only",
        use_confidence=False,
        use_overlap=False,
        use_resource=False,
    ),
    "confidence_only": replace(
        _BASE,
        name="confidence_only",
        use_persistence=False,
        use_overlap=False,
        use_resource=False,
        k=1,
    ),
    "persistence_confidence": replace(
        _BASE,
        name="persistence_confidence",
        use_overlap=False,
        use_resource=False,
    ),
    "full_gate": _BASE,
    "fixed_step_2": replace(
        _BASE,
        name="fixed_step_2",
        use_persistence=False,
        use_confidence=False,
        fixed_step=2,
    ),
    "random_matched": replace(
        _BASE,
        name="random_matched",
        use_persistence=False,
        use_confidence=False,
        random_rate=0.5,
    ),
    # The forward reading of `replay`'s retrospective
    # `diffusion_earliest_committed` reference: launch as soon as the whole span
    # lies inside the entropy-bound accepted set. The reference sets
    # `matched_final=True` for every final call, so its false-launch rate is zero
    # by construction; this policy makes the same signal measurable, including on
    # candidates that are accepted and then renoised.
    "acceptance_only": replace(
        _BASE,
        name="acceptance_only",
        use_persistence=False,
        use_confidence=False,
        use_overlap=False,
        use_resource=False,
        use_acceptance=True,
        k=1,
    ),
    # Acceptance plus the confidence floor, i.e. the cheapest safe-looking
    # combination that does not rely on identity repetition across revisions.
    "acceptance_confidence": replace(
        _BASE,
        name="acceptance_confidence",
        use_persistence=False,
        use_overlap=False,
        use_resource=False,
        use_acceptance=True,
        k=1,
    ),
    # The two retrospective oracles are deliberately NOT in this library: they
    # live in `replay.BASELINE_POLICIES` as `oracle_earliest_invariant` and
    # `oracle_final_match`.  Duplicating them here made `policies=("all",)`
    # emit two identical rows per trajectory and inflated the sample count.
}


def policy_by_name(name: str) -> GatePolicy:
    try:
        return POLICY_LIBRARY[name]
    except KeyError:
        raise KeyError(f"unknown policy {name!r}; have {sorted(POLICY_LIBRARY)}") from None


def k_sweep(base: GatePolicy, ks: Sequence[int]) -> list[GatePolicy]:
    return [replace(base, name=f"{base.name}_k{k}", k=k) for k in ks]


def tau_sweep(base: GatePolicy, taus: Sequence[float]) -> list[GatePolicy]:
    return [replace(base, name=f"{base.name}_tau{t:g}", tau=t) for t in taus]
