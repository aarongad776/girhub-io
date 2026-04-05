"""test_coherence.py — Comprehensive coherence test suite for GlyphlockKernel.

Validates realistic synchronization behavior across multiple parameter
configurations and physics regimes.

Usage
-----
Run via pytest (standard mode)::

    pytest test_coherence.py -v

Run standalone for interactive debug output::

    python test_coherence.py

The standalone mode prints per-test metric breakdowns and pass/fail verdicts.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load glylock.kernell (non-standard extension) via SourceFileLoader
# ---------------------------------------------------------------------------

_KERNELL_PATH = Path(__file__).with_name("glylock.kernell")

_loader = importlib.machinery.SourceFileLoader("glylock_kernell", str(_KERNELL_PATH))
_spec = importlib.util.spec_from_loader("glylock_kernell", _loader)
_glylock = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["glylock_kernell"] = _glylock
_spec.loader.exec_module(_glylock)  # type: ignore[union-attr]

GlyphlockParams = _glylock.GlyphlockParams
GlyphlockKernel = _glylock.GlyphlockKernel

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CoherenceMetrics:
    """Measured coherence metrics computed from a completed simulation run.

    Attributes
    ----------
    final_r:
        Kuramoto order parameter at the last cycle (0 = incoherent, 1 = fully
        synchronised).
    convergence_cycles:
        First cycle number at which ``r`` reached ≥ 0.95, or ``None`` when
        synchronisation was never achieved during the run.
    tail_variance:
        Variance of ``r`` over the last 50 cycles (stability indicator).
    phase_diff:
        Maximum pairwise wrapped phase difference (radians) between any two
        oscillators at the end of the run.
    cluster_pct:
        Percentage of oscillators that belong to the largest phase-locked
        cluster (detected with ``phase_tol=0.3`` rad).
    spatial_spread:
        Mean standard deviation of oscillator positions over both spatial
        axes.  ``None`` when swarmalator mode is disabled.
    r_history:
        Full list of order-parameter values, one per cycle.
    """

    final_r: float
    convergence_cycles: Optional[int]
    tail_variance: float
    phase_diff: float
    cluster_pct: float
    spatial_spread: Optional[float]
    r_history: list[float] = field(default_factory=list)


@dataclass
class CoherenceExpectation:
    """Realistic tolerance bounds for a single test scenario.

    ``None`` for any bound means that bound is not checked.

    Attributes
    ----------
    r_min / r_max:
        Inclusive bounds on ``final_r``.
    convergence_max:
        Upper limit on ``convergence_cycles`` (cycles to reach r ≥ 0.95).
        ``None`` to skip.
    tail_var_max:
        Maximum allowed ``tail_variance``.
    phase_diff_max:
        Maximum allowed ``phase_diff`` in radians.
    cluster_pct_min:
        Minimum percentage of oscillators in the largest cluster.
    spatial_spread_max:
        Upper limit on ``spatial_spread``.  ``None`` to skip.
    """

    r_min: float
    r_max: float = 1.0
    convergence_max: Optional[int] = None
    tail_var_max: float = 1e-2
    phase_diff_max: float = 1.0
    cluster_pct_min: float = 70.0
    spatial_spread_max: Optional[float] = None


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

_TAIL_WINDOW = 50
_SYNC_THRESHOLD = 0.95


def _wrapped_phase_diff(a: float, b: float) -> float:
    """Return the wrapped phase difference |a − b| mapped to [0, π]."""
    delta = abs(a - b) % (2 * np.pi)
    if delta > np.pi:
        delta = 2 * np.pi - delta
    return delta


def compute_coherence_metrics(
    kernel: "GlyphlockKernel",  # type: ignore[name-defined]
    cycles: int,
) -> CoherenceMetrics:
    """Run *kernel* for *cycles* steps and return a :class:`CoherenceMetrics`.

    Parameters
    ----------
    kernel:
        A fully initialised :class:`GlyphlockKernel` instance (not yet run).
    cycles:
        Number of integration steps to execute.

    Returns
    -------
    CoherenceMetrics
        All metrics computed from the completed run.
    """
    r_history: list[float] = []
    for _ in range(cycles):
        r = kernel.step()
        r_history.append(r)

    final_r = r_history[-1]

    # Convergence: first cycle where r hit the sync threshold
    convergence_cycles: Optional[int] = next(
        (i + 1 for i, r in enumerate(r_history) if r >= _SYNC_THRESHOLD),
        None,
    )

    # Tail variance over the last TAIL_WINDOW cycles
    tail = r_history[-_TAIL_WINDOW:]
    tail_variance = float(np.var(tail))

    # Max pairwise wrapped phase difference
    phases = kernel.theta
    n = len(phases)
    max_pd = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            delta = _wrapped_phase_diff(float(phases[i]), float(phases[j]))
            if delta > max_pd:
                max_pd = delta

    # Largest cluster percentage via detect_regions
    regions = kernel.detect_regions(phase_tol=0.3)
    biggest = max(len(c) for c in regions) if regions else 0
    cluster_pct = biggest / n * 100.0

    # Spatial spread (swarmalator mode only)
    x = getattr(kernel, "x", None)
    spatial_spread: Optional[float] = (
        float(np.mean(np.std(x, axis=0))) if x is not None else None
    )

    return CoherenceMetrics(
        final_r=final_r,
        convergence_cycles=convergence_cycles,
        tail_variance=tail_variance,
        phase_diff=max_pd,
        cluster_pct=cluster_pct,
        spatial_spread=spatial_spread,
        r_history=r_history,
    )


def run_coherence_test(
    name: str,
    params_kwargs: dict,
    expectation: CoherenceExpectation,
    cycles: int = 800,
    swarmalator: bool = False,
    debug: bool = False,
) -> tuple[bool, CoherenceMetrics, list[str]]:
    """Build a kernel, run it, and validate metrics against *expectation*.

    Parameters
    ----------
    name:
        Human-readable test name (used in log / error messages).
    params_kwargs:
        Keyword arguments forwarded to :class:`GlyphlockParams`.
    expectation:
        Tolerance bounds to validate against.
    cycles:
        Number of simulation steps.
    swarmalator:
        Pass ``True`` to enable swarmalator spatial dynamics.
    debug:
        When ``True``, print detailed per-metric output to stdout.

    Returns
    -------
    passed : bool
        ``True`` when all checked metrics are within bounds.
    metrics : CoherenceMetrics
        The computed metrics for inspection.
    failures : list[str]
        Human-readable descriptions of any failed checks (empty on pass).
    """
    params = GlyphlockParams(**params_kwargs)
    kernel = GlyphlockKernel(params, swarmalator=swarmalator)
    metrics = compute_coherence_metrics(kernel, cycles)

    failures: list[str] = []

    def _check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    # --- individual checks ------------------------------------------------
    _check(
        expectation.r_min <= metrics.final_r <= expectation.r_max,
        f"final_r={metrics.final_r:.4f} not in [{expectation.r_min}, {expectation.r_max}]",
    )
    if expectation.convergence_max is not None:
        if metrics.convergence_cycles is None:
            failures.append(
                f"convergence_cycles=None (never reached r≥{_SYNC_THRESHOLD}) "
                f"— expected ≤{expectation.convergence_max}"
            )
        else:
            _check(
                metrics.convergence_cycles <= expectation.convergence_max,
                f"convergence_cycles={metrics.convergence_cycles} > {expectation.convergence_max}",
            )
    _check(
        metrics.tail_variance <= expectation.tail_var_max,
        f"tail_variance={metrics.tail_variance:.2e} > {expectation.tail_var_max:.2e}",
    )
    _check(
        metrics.phase_diff <= expectation.phase_diff_max,
        f"phase_diff={metrics.phase_diff:.4f} rad > {expectation.phase_diff_max} rad",
    )
    _check(
        metrics.cluster_pct >= expectation.cluster_pct_min,
        f"cluster_pct={metrics.cluster_pct:.1f}% < {expectation.cluster_pct_min}%",
    )
    if expectation.spatial_spread_max is not None and metrics.spatial_spread is not None:
        _check(
            metrics.spatial_spread <= expectation.spatial_spread_max,
            f"spatial_spread={metrics.spatial_spread:.3f} > {expectation.spatial_spread_max}",
        )
    if swarmalator and metrics.spatial_spread is not None:
        _check(
            np.isfinite(metrics.spatial_spread),
            "spatial_spread is not finite (NaN/inf in swarmalator positions)",
        )

    passed = len(failures) == 0

    if debug:
        verdict = "PASS" if passed else "FAIL"
        print(f"\n{'='*60}")
        print(f"[{verdict}] {name}")
        print(f"{'='*60}")
        print(f"  params          : {params_kwargs}")
        print(f"  cycles          : {cycles}  swarmalator={swarmalator}")
        print(f"  final_r         : {metrics.final_r:.6f}  "
              f"(expect [{expectation.r_min}, {expectation.r_max}])")
        conv_str = str(metrics.convergence_cycles) if metrics.convergence_cycles else "never"
        conv_exp = (f"≤{expectation.convergence_max}"
                    if expectation.convergence_max else "not checked")
        print(f"  convergence     : {conv_str} cycles  (expect {conv_exp})")
        print(f"  tail_variance   : {metrics.tail_variance:.2e}  "
              f"(expect ≤{expectation.tail_var_max:.2e})")
        print(f"  max_phase_diff  : {metrics.phase_diff:.4f} rad  "
              f"(expect ≤{expectation.phase_diff_max} rad)")
        print(f"  cluster_pct     : {metrics.cluster_pct:.1f}%  "
              f"(expect ≥{expectation.cluster_pct_min}%)")
        if metrics.spatial_spread is not None:
            ss_exp = (f"≤{expectation.spatial_spread_max}"
                      if expectation.spatial_spread_max else "finite only")
            print(f"  spatial_spread  : {metrics.spatial_spread:.3f}  (expect {ss_exp})")
        # r-history summary: min/max/mean over the tail
        tail = metrics.r_history[-_TAIL_WINDOW:]
        print(f"  r tail [{_TAIL_WINDOW} cyc] : "
              f"min={min(tail):.6f}  max={max(tail):.6f}  "
              f"mean={np.mean(tail):.6f}")
        if failures:
            print("  FAILED checks:")
            for msg in failures:
                print(f"    ✗ {msg}")

    return passed, metrics, failures


# ---------------------------------------------------------------------------
# Pytest parametrize cases
# ---------------------------------------------------------------------------

# Each entry: (test_id, params_kwargs, expectation, cycles, swarmalator)
_TEST_CASES = [
    (
        # TC-1: Standard Kuramoto — deterministic, strong coupling, no noise
        "standard_k8_nu0",
        dict(n=22, k=8.0, r_ruach=3.5, nu=0.0, seed=10841),
        CoherenceExpectation(
            r_min=0.95,
            r_max=1.0,
            convergence_max=400,
            tail_var_max=1e-4,
            phase_diff_max=0.5,
            cluster_pct_min=90.0,
        ),
        600,
        False,
    ),
    (
        # TC-2: Standard Kuramoto with moderate noise
        "standard_k8_nu03",
        dict(n=22, k=8.0, r_ruach=3.5, nu=0.3, seed=10841),
        CoherenceExpectation(
            r_min=0.90,
            r_max=1.0,
            convergence_max=400,
            tail_var_max=1e-3,
            phase_diff_max=0.6,
            cluster_pct_min=85.0,
        ),
        600,
        False,
    ),
    (
        # TC-3: High-noise regime — system should still synchronise
        "high_noise_k8_nu05",
        dict(n=22, k=8.0, r_ruach=3.5, nu=0.5, seed=10841),
        CoherenceExpectation(
            r_min=0.85,
            r_max=1.0,
            convergence_max=600,
            tail_var_max=5e-3,
            phase_diff_max=0.7,
            cluster_pct_min=80.0,
        ),
        800,
        False,
    ),
    (
        # TC-4: Low coupling — slower convergence, wider phase spread
        "low_coupling_k2_nu0",
        dict(n=22, k=2.0, r_ruach=3.5, nu=0.0, seed=10841),
        CoherenceExpectation(
            r_min=0.85,
            r_max=1.0,
            convergence_max=800,
            tail_var_max=5e-4,
            phase_diff_max=1.0,
            cluster_pct_min=70.0,
        ),
        1000,
        False,
    ),
    (
        # TC-5: High coupling — fast convergence, tight phase locking
        "high_coupling_k15_nu0",
        dict(n=22, k=15.0, r_ruach=3.5, nu=0.0, seed=10841),
        CoherenceExpectation(
            r_min=0.97,
            r_max=1.0,
            convergence_max=200,
            tail_var_max=1e-6,
            phase_diff_max=0.3,
            cluster_pct_min=95.0,
        ),
        400,
        False,
    ),
    (
        # TC-6: Sakaguchi phase lag α=0.5 — compatible with PR#3
        "sakaguchi_alpha05_k8_nu0",
        dict(n=22, k=8.0, r_ruach=3.5, nu=0.0, alpha=0.5, seed=10841),
        CoherenceExpectation(
            r_min=0.90,
            r_max=1.0,
            convergence_max=600,
            tail_var_max=1e-3,
            phase_diff_max=0.6,
            cluster_pct_min=85.0,
        ),
        600,
        False,
    ),
    (
        # TC-7: Small system n=5 — very fast convergence, near-perfect locking
        "small_n5_k8_nu0",
        dict(n=5, k=8.0, r_ruach=3.5, nu=0.0, seed=42),
        CoherenceExpectation(
            r_min=0.95,
            r_max=1.0,
            convergence_max=200,
            tail_var_max=1e-4,
            phase_diff_max=0.3,
            cluster_pct_min=90.0,
        ),
        400,
        False,
    ),
    (
        # TC-8: Swarmalator spatial dynamics — phase coherence + finite positions
        "swarmalator_k8_nu0",
        dict(n=22, k=8.0, r_ruach=3.5, nu=0.0, seed=10841),
        CoherenceExpectation(
            r_min=0.85,
            r_max=1.0,
            convergence_max=800,
            tail_var_max=1e-3,
            phase_diff_max=0.7,
            cluster_pct_min=80.0,
            spatial_spread_max=None,  # spread grows; just check finiteness
        ),
        800,
        True,  # swarmalator=True
    ),
]


@pytest.mark.parametrize(
    "test_id,params_kwargs,expectation,cycles,swarmalator",
    _TEST_CASES,
    ids=[tc[0] for tc in _TEST_CASES],
)
def test_coherence(
    test_id: str,
    params_kwargs: dict,
    expectation: CoherenceExpectation,
    cycles: int,
    swarmalator: bool,
) -> None:
    """Parametrised coherence test — validates synchronisation metrics."""
    passed, metrics, failures = run_coherence_test(
        name=test_id,
        params_kwargs=params_kwargs,
        expectation=expectation,
        cycles=cycles,
        swarmalator=swarmalator,
        debug=False,
    )
    failure_msg = (
        f"[{test_id}] Coherence test failed:\n"
        + "\n".join(f"  ✗ {f}" for f in failures)
        + f"\n  final_r={metrics.final_r:.4f}"
        f"  convergence={metrics.convergence_cycles}"
        f"  tail_var={metrics.tail_variance:.2e}"
        f"  phase_diff={metrics.phase_diff:.4f} rad"
        f"  cluster={metrics.cluster_pct:.1f}%"
    )
    assert passed, failure_msg


# ---------------------------------------------------------------------------
# Standalone debug mode
# ---------------------------------------------------------------------------


def _run_debug() -> None:
    """Execute all test cases with verbose per-metric output."""
    logging.basicConfig(level=logging.WARNING)  # suppress kernel INFO spam

    total = len(_TEST_CASES)
    n_pass = 0

    print(f"\nGlyphlock Coherence Debug Suite — {total} scenarios")
    print("=" * 60)

    for test_id, params_kwargs, expectation, cycles, swarmalator in _TEST_CASES:
        passed, metrics, _ = run_coherence_test(
            name=test_id,
            params_kwargs=params_kwargs,
            expectation=expectation,
            cycles=cycles,
            swarmalator=swarmalator,
            debug=True,
        )
        if passed:
            n_pass += 1

        # Per-cycle r-history: show every 100th entry
        print(f"\n  r-history (every 100 cycles, total={len(metrics.r_history)}):")
        for i in range(0, len(metrics.r_history), 100):
            print(f"    cycle {i+1:4d}: r={metrics.r_history[i]:.6f}")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {n_pass}/{total} tests PASSED")
    if n_pass == total:
        print("✓ All coherence tests PASSED")
    else:
        print(f"✗ {total - n_pass} test(s) FAILED")
    print("=" * 60)


if __name__ == "__main__":
    _run_debug()
