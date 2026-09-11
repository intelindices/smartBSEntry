"""Weight Calibrator: how much each STEngine contributes to the blend.

v0.9.5 ships **equal weights**. Fitted scoring (return-per-trade, drawdown,
Sharpe tilt, shrinkage, dominance cap) lost to 1/k on both the Yahoo 2y and
Dukascopy 10y holdouts: the fit window is only the preceding W days, so more
training history does not make the weights more trustworthy. ``compute_weights``
is therefore 1/k. ``compute_fitted_weights`` keeps the old scorer for research.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Trades needed to keep a participation floor in the blend.
MIN_TRADES_FOR_WEIGHT = 30
# Trades needed before an engine's edge is taken at face value rather than
# shrunk toward the field.
FULL_CONFIDENCE_TRADES = 300
# Beyond this, more trades stop buying additional confidence.
TRADE_COUNT_CAP = 400
# Drawdown, in percent, that halves an engine's score.
DD_TOLERANCE_PCT = 10.0
# How far Sharpe may move a score either way, as a fraction.
SHARPE_TILT = 0.25


@dataclass(frozen=True)
class EnginePerf:
    """One engine's result on one evaluation window."""

    engine: str
    total_return: float  # percent
    n_trades: int
    max_drawdown: float = 0.0
    sharpe: float = 0.0

    @property
    def return_per_trade(self) -> float:
        if self.n_trades <= 0:
            return 0.0
        return float(self.total_return) / float(self.n_trades)


@dataclass(frozen=True)
class WindowSplit:
    """Bar ranges for the three-way split. End indices are exclusive."""

    train: tuple[int, int]
    weight_fit: tuple[int, int]
    report: tuple[int, int]

    def as_dict(self) -> dict[str, tuple[int, int]]:
        return {"train": self.train, "weight_fit": self.weight_fit, "report": self.report}


def split_windows(n_bars: int, *, weight_fit_bars: int, report_bars: int) -> WindowSplit:
    """Carve a series into train / weight-fit / report, oldest to newest.

    The report window is the most recent data and must never be seen by either
    training or weight fitting.
    """
    if n_bars <= 0:
        raise ValueError("n_bars must be positive")
    if weight_fit_bars <= 0 or report_bars <= 0:
        raise ValueError("weight_fit_bars and report_bars must be positive")
    report_start = n_bars - int(report_bars)
    fit_start = report_start - int(weight_fit_bars)
    if fit_start <= 0:
        raise ValueError(
            f"n_bars={n_bars} too small for weight_fit={weight_fit_bars} + report={report_bars}"
        )
    return WindowSplit(
        train=(0, fit_start),
        weight_fit=(fit_start, report_start),
        report=(report_start, n_bars),
    )


def engine_score(perf: EnginePerf) -> float:
    """Risk-adjusted, sample-size-aware edge. Zero for engines that lost money.

    Edge per trade sets the scale, evidence (sqrt of trade count) rewards the
    better-sampled engine at equal edge, drawdown damps engines that earned it
    painfully, and Sharpe applies a bounded tilt. ``tanh`` keeps the tilt inside
    +/-``SHARPE_TILT`` and handles negative Sharpe without flipping the sign.
    """
    if perf.n_trades <= 0:
        return 0.0
    edge = max(0.0, perf.return_per_trade)
    if edge <= 0.0:
        return 0.0
    evidence = math.sqrt(min(int(perf.n_trades), TRADE_COUNT_CAP))
    dd_damp = 1.0 / (1.0 + abs(float(perf.max_drawdown)) / DD_TOLERANCE_PCT)
    sharpe_tilt = 1.0 + SHARPE_TILT * math.tanh(float(perf.sharpe))
    return edge * evidence * dd_damp * sharpe_tilt


def shrinkage_lambda(
    n_trades: int, full_confidence_trades: int = FULL_CONFIDENCE_TRADES
) -> float:
    """Fraction of its own score an engine keeps; the rest is borrowed from the field."""
    if full_confidence_trades <= 0:
        return 1.0
    return min(1.0, max(0.0, float(n_trades) / float(full_confidence_trades)))


def _apply_dominance_cap(weights: dict[str, float], cap: float) -> dict[str, float]:
    """Clip any engine above ``cap`` and redistribute to the rest.

    Without this, one engine with a thin but lucky sample can take the entire
    blend whenever every other engine happens to be flat or negative.
    """
    out = dict(weights)
    for _ in range(len(out)):
        over = {n: w for n, w in out.items() if w > cap + 1e-12}
        if not over:
            break
        spill = sum(w - cap for w in over.values())
        room = {n: w for n, w in out.items() if n not in over}
        for n in over:
            out[n] = cap
        head = sum(room.values())
        if head <= 0:  # everyone is capped; nothing left to absorb the spill
            break
        for n, w in room.items():
            out[n] = w + spill * (w / head)
    total = sum(out.values()) or 1.0
    return {n: w / total for n, w in out.items()}


def equal_weights(engines: list[str] | tuple[str, ...]) -> dict[str, float]:
    """1/k over the named engines. Empty input → empty dict."""
    names = [n for n in engines if n]
    if not names:
        return {}
    k = 1.0 / len(names)
    return {n: k for n in names}


def compute_weights(
    perfs: list[EnginePerf],
    *,
    min_trades: int = MIN_TRADES_FOR_WEIGHT,
    full_confidence_trades: int = FULL_CONFIDENCE_TRADES,
    max_weight_mult: float = 2.0,
    min_weight_mult: float = 0.25,
) -> dict[str, float]:
    """Production blend weights: equal, summing to 1.

    Scoring kwargs are accepted so existing callers keep working; they are
    ignored. Use ``compute_fitted_weights`` for the research scorer.
    """
    del min_trades, full_confidence_trades, max_weight_mult, min_weight_mult
    return equal_weights([p.engine for p in perfs])


def compute_fitted_weights(
    perfs: list[EnginePerf],
    *,
    min_trades: int = MIN_TRADES_FOR_WEIGHT,
    full_confidence_trades: int = FULL_CONFIDENCE_TRADES,
    max_weight_mult: float = 2.0,
    min_weight_mult: float = 0.25,
) -> dict[str, float]:
    """Research scorer: return-per-trade, drawdown, Sharpe, shrinkage, cap.

    Not used live. Kept so blend_eval can still print what fitted would have
    done next to the equal-weight production path.
    """
    if not perfs:
        return {}
    k = len(perfs)
    equal = 1.0 / k

    scores = {p.engine: engine_score(p) for p in perfs}
    if sum(scores.values()) <= 0:
        return {p.engine: equal for p in perfs}

    mean_score = sum(scores.values()) / k
    shrunk: dict[str, float] = {}
    for p in perfs:
        lam = shrinkage_lambda(p.n_trades, full_confidence_trades)
        shrunk[p.engine] = lam * scores[p.engine] + (1.0 - lam) * mean_score

    total = sum(shrunk.values())
    if total <= 0:
        return {p.engine: equal for p in perfs}
    weights = {name: val / total for name, val in shrunk.items()}

    floor = float(min_weight_mult) * equal
    for p in perfs:
        if p.n_trades >= min_trades:
            weights[p.engine] = max(weights[p.engine], floor)
    norm = sum(weights.values()) or 1.0
    weights = {n: w / norm for n, w in weights.items()}

    return _apply_dominance_cap(weights, min(1.0, float(max_weight_mult) * equal))


@dataclass
class WalkForwardWeights:
    """Weights re-fitted on a rolling schedule rather than frozen after training."""

    windows: list[dict[str, float]] = field(default_factory=list)

    def latest(self) -> dict[str, float]:
        return dict(self.windows[-1]) if self.windows else {}

    def averaged(self) -> dict[str, float]:
        if not self.windows:
            return {}
        names = sorted({n for w in self.windows for n in w})
        avg = {n: sum(w.get(n, 0.0) for w in self.windows) / len(self.windows) for n in names}
        total = sum(avg.values()) or 1.0
        return {n: v / total for n, v in avg.items()}


def walk_forward_weights(
    perf_by_window: list[list[EnginePerf]],
    *,
    min_trades: int = MIN_TRADES_FOR_WEIGHT,
    full_confidence_trades: int = FULL_CONFIDENCE_TRADES,
) -> WalkForwardWeights:
    """Fit weights independently on each successive window."""
    return WalkForwardWeights(
        windows=[
            compute_weights(
                perfs, min_trades=min_trades, full_confidence_trades=full_confidence_trades
            )
            for perfs in perf_by_window
        ]
    )


def describe_weights(
    weights: dict[str, float],
    perfs: list[EnginePerf],
    *,
    full_confidence_trades: int = FULL_CONFIDENCE_TRADES,
) -> str:
    by_name = {p.engine: p for p in perfs}
    lines = [
        f"{'engine':16s} {'weight':>7s} {'ret':>8s} {'n':>5s} {'ret/trade':>10s} "
        f"{'dd%':>6s} {'sharpe':>7s} {'keep':>5s}"
    ]
    for name, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        p = by_name.get(name)
        if p is None:
            lines.append(f"{name:16s} {w:7.3f}")
            continue
        lam = shrinkage_lambda(p.n_trades, full_confidence_trades)
        lines.append(
            f"{name:16s} {w:7.3f} {p.total_return:8.2f} {p.n_trades:5d} "
            f"{p.return_per_trade:10.4f} {abs(p.max_drawdown):6.2f} {p.sharpe:7.2f} "
            f"{lam:5.2f}"
        )
    return "\n".join(lines)
