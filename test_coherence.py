"""test_coherence.py - Coherence metric tests for GlyphlockKernel.

Validates realistic coherence behavior under various parameter configurations,
comparing expected vs. actual results with tolerance margins.

Run modes
---------
pytest:
    pytest test_coherence.py -v

Standalone (debug/interactive):
    python test_coherence.py
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module loader helper
# ---------------------------------------------------------------------------
# glylock.kernell uses a non-standard extension, so we load it manually.

def _load_glylock() -> types.ModuleType:
    """Load glylock.kernell as a Python module."""
    mod_name = "glylock_kernel"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    import os
    kernel_path = os.path.join(os.path.dirname(__file__), "glylock.kernell")
    mod = types.ModuleType(mod_name)
    mod.__file__ = kernel_path
    sys.modules[mod_name] = mod
    with open(kernel_path) as fh:
        src = fh.read()
    exec(compile(src, kernel_path, "exec"), mod.__dict__)
    return mod


glylock = _load_glylock()
GlyphlockKernel = glylock.GlyphlockKernel
GlyphlockParams = glylock.GlyphlockParams

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s",
)
_LOG = logging.getLogger("test_coherence")


def _run(params: GlyphlockParams, cycles: int = 2000, swarmalator: bool = False):
    """Run simulation and return (r_history, kernel)."""
    kernel = GlyphlockKernel(params, swarmalator=swarmalator)
    r_history: list[float] = []
    for _ in range(cycles):
        r_history.append(kernel.step())
    return r_history, kernel


def _convergence_cycle(r_history: list[float], threshold: float = 0.95) -> Optional[int]:
    """Return first cycle where r >= threshold, or None if never reached."""
    for i, r in enumerate(r_history):
        if r >= threshold:
            return i + 1
    return None


def _phase_variance(kernel: GlyphlockKernel) -> float:
    """Wrapped phase variance relative to mean phase."""
    phases = kernel.theta % (2 * np.pi)
    mean_phase = np.angle(np.mean(np.exp(1j * phases)))
    diffs = np.angle(np.exp(1j * (phases - mean_phase)))
    return float(np.var(diffs))


def _largest_cluster_fraction(kernel: GlyphlockKernel, phase_tol: float = 0.3) -> float:
    """Fraction of oscillators in the largest phase-locked cluster."""
    regions = kernel.detect_regions(phase_tol=phase_tol)
    largest = max(len(c) for c in regions)
    return largest / kernel.N


def _spatial_spread(kernel: GlyphlockKernel) -> float:
    """Mean std of spatial positions (swarmalator mode only)."""
    assert kernel.x is not None, "spatial spread only available in swarmalator mode"
    return float(np.mean(np.std(kernel.x, axis=0)))


# ---------------------------------------------------------------------------
# Result container for debug/summary output
# ---------------------------------------------------------------------------

@dataclass
class CoherenceResult:
    label: str
    final_r: float
    r_expected_min: Optional[float]
    r_expected_max: Optional[float]
    convergence_cycle: Optional[int]
    phase_variance: float
    largest_cluster_pct: float
    spatial_spread: Optional[float]
    proctor_certified: Optional[bool]
    passed: bool

    def __str__(self) -> str:
        r_range = ""
        if self.r_expected_min is not None or self.r_expected_max is not None:
            lo = f"{self.r_expected_min:.2f}" if self.r_expected_min is not None else "-∞"
            hi = f"{self.r_expected_max:.2f}" if self.r_expected_max is not None else "+∞"
            r_range = f" [{lo}, {hi}]"
        conv = f"{self.convergence_cycle}" if self.convergence_cycle else "n/a"
        spread = f"{self.spatial_spread:.2f}" if self.spatial_spread is not None else "n/a"
        cert = str(self.proctor_certified) if self.proctor_certified is not None else "n/a"
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.label}\n"
            f"       final_r={self.final_r:.4f}{r_range}"
            f"  conv_cycle={conv}"
            f"  phase_var={self.phase_variance:.4f}"
            f"  cluster%={self.largest_cluster_pct:.1%}"
            f"  spread={spread}"
            f"  certified={cert}"
        )


# ---------------------------------------------------------------------------
# Core test function (used by both pytest and standalone script)
# ---------------------------------------------------------------------------

def run_coherence_test(
    label: str,
    params: GlyphlockParams,
    cycles: int = 2000,
    swarmalator: bool = False,
    r_min: Optional[float] = None,
    r_max: Optional[float] = None,
    check_proctors: bool = False,
    proctor_window: int = 50,
    convergence_tol: float = 1e-4,
    phase_tol: float = 0.3,
    strict: bool = True,
) -> CoherenceResult:
    """Run a simulation and collect coherence metrics.

    Parameters
    ----------
    label:
        Human-readable test name for logging.
    params:
        Simulation parameters.
    cycles:
        Number of integration steps.
    swarmalator:
        Enable swarmalator spatial dynamics.
    r_min, r_max:
        Inclusive bounds for the final order parameter ``r``.
    check_proctors:
        Whether to run the proctor validation suite.
    proctor_window:
        Tail window size for the convergence proctor.
    convergence_tol:
        Variance threshold for the convergence proctor.
    phase_tol:
        Phase tolerance (radians) for region detection.
    strict:
        When ``True`` (pytest mode) raise ``AssertionError`` on failure.
        When ``False`` (standalone/debug mode) only log.

    Returns
    -------
    CoherenceResult
        Full metrics snapshot for the run.
    """
    r_history, kernel = _run(params, cycles=cycles, swarmalator=swarmalator)

    final_r = r_history[-1]
    conv = _convergence_cycle(r_history, threshold=0.95)
    phase_var = _phase_variance(kernel)
    cluster_pct = _largest_cluster_fraction(kernel, phase_tol=phase_tol)
    spread = _spatial_spread(kernel) if swarmalator else None

    certified: Optional[bool] = None
    if check_proctors:
        report = kernel.run_proctors(
            r_history,
            window=proctor_window,
            convergence_tol=convergence_tol,
        )
        certified = report.certified

    # Determine pass/fail
    failures: list[str] = []

    if r_min is not None and final_r < r_min:
        failures.append(
            f"final_r={final_r:.4f} < expected_min={r_min:.4f}  (delta={final_r - r_min:.4f})"
        )
    if r_max is not None and final_r > r_max:
        failures.append(
            f"final_r={final_r:.4f} > expected_max={r_max:.4f}  (delta={final_r - r_max:.4f})"
        )
    if check_proctors and certified is False:
        failures.append("proctor suite: NOT certified")

    passed = len(failures) == 0

    result = CoherenceResult(
        label=label,
        final_r=final_r,
        r_expected_min=r_min,
        r_expected_max=r_max,
        convergence_cycle=conv,
        phase_variance=phase_var,
        largest_cluster_pct=cluster_pct,
        spatial_spread=spread,
        proctor_certified=certified,
        passed=passed,
    )

    _LOG.info("%s", result)

    if strict and failures:
        failure_summary = "\n  ".join(failures)
        raise AssertionError(
            f"Coherence test '{label}' FAILED:\n  {failure_summary}\n{result}"
        )

    return result


# ---------------------------------------------------------------------------
# pytest parametrized tests
# ---------------------------------------------------------------------------

_HIGH_COUPLING_PARAMS = GlyphlockParams(n=22, k=8.0, r_ruach=3.5, nu=0.3, seed=10841)
_LOW_COUPLING_PARAMS = GlyphlockParams(n=22, k=2.0, r_ruach=0.0, nu=0.3, seed=10841)
_HIGH_NOISE_CLEAN_PARAMS = GlyphlockParams(n=22, k=3.0, r_ruach=0.0, nu=0.0, seed=10841)
_HIGH_NOISE_NOISY_PARAMS = GlyphlockParams(n=22, k=3.0, r_ruach=0.0, nu=0.8, seed=10841)
_ALPHA_PARAMS = GlyphlockParams(n=22, k=8.0, r_ruach=3.5, nu=0.1, seed=10841, alpha=np.pi / 4)
_SWARM_PARAMS = GlyphlockParams(n=22, k=8.0, r_ruach=3.5, nu=0.1, seed=42)
_LOW_DIM_PARAMS = GlyphlockParams(n=5, k=8.0, r_ruach=3.5, nu=0.3, seed=10841)
_HIGH_DIM_PARAMS = GlyphlockParams(n=22, k=8.0, r_ruach=3.5, nu=0.3, seed=10841)


@pytest.mark.parametrize(
    "label,params,cycles,swarmalator,r_min,r_max,check_proctors",
    [
        # High coupling → fast convergence, r ≥ 0.95
        (
            "high_coupling_n22",
            _HIGH_COUPLING_PARAMS,
            2000,
            False,
            0.95,
            None,
            True,
        ),
        # Low coupling (no phase-pinning) → partial coherence, r ≤ 0.70
        (
            "low_coupling_n22",
            _LOW_COUPLING_PARAMS,
            2000,
            False,
            None,
            0.70,
            False,
        ),
        # Sakaguchi lag → still converges to high coherence
        (
            "sakaguchi_lag_alpha_pi4",
            _ALPHA_PARAMS,
            2000,
            False,
            0.80,
            None,
            False,
        ),
        # Low-dimensional (n=5) → very high coherence
        (
            "low_dim_n5",
            _LOW_DIM_PARAMS,
            2000,
            False,
            0.95,
            None,
            False,
        ),
        # High-dimensional (n=22) → high coherence
        (
            "high_dim_n22",
            _HIGH_DIM_PARAMS,
            2000,
            False,
            0.95,
            None,
            True,
        ),
        # Swarmalator mode → phase coherence alongside spatial dynamics
        (
            "swarmalator_n22",
            _SWARM_PARAMS,
            500,
            True,
            0.80,
            None,
            False,
        ),
    ],
)
def test_coherence_final_r(
    label, params, cycles, swarmalator, r_min, r_max, check_proctors
):
    """Parametrized test: final order parameter r is within expected bounds."""
    run_coherence_test(
        label=label,
        params=params,
        cycles=cycles,
        swarmalator=swarmalator,
        r_min=r_min,
        r_max=r_max,
        check_proctors=check_proctors,
        strict=True,
    )


def test_high_coupling_convergence_speed():
    """High coupling should reach r >= 0.95 well within 2000 cycles."""
    r_history, _ = _run(_HIGH_COUPLING_PARAMS, cycles=2000)
    conv = _convergence_cycle(r_history, threshold=0.95)
    assert conv is not None, (
        f"High coupling never reached r=0.95 in {len(r_history)} cycles"
    )
    assert conv <= 500, (
        f"High coupling took {conv} cycles to reach r=0.95; expected ≤ 500"
    )


def test_high_coupling_phase_variance():
    """High coupling should show tight phase clustering (low variance)."""
    _, kernel = _run(_HIGH_COUPLING_PARAMS, cycles=2000)
    var = _phase_variance(kernel)
    assert var < 0.1, (
        f"Phase variance={var:.4f} with high coupling; expected < 0.1"
    )


def test_high_coupling_single_region():
    """After convergence all oscillators should be in one phase-locked cluster."""
    _, kernel = _run(_HIGH_COUPLING_PARAMS, cycles=2000)
    regions = kernel.detect_regions(phase_tol=0.3)
    assert len(regions) == 1, (
        f"Expected 1 region with high coupling; got {len(regions)}"
    )
    assert regions[0] == list(range(kernel.N)), (
        f"Single region should contain all {kernel.N} oscillators"
    )


def test_low_coupling_multiple_regions():
    """Low coupling should produce multiple phase-locked clusters."""
    _, kernel = _run(_LOW_COUPLING_PARAMS, cycles=2000)
    regions = kernel.detect_regions(phase_tol=0.3)
    assert len(regions) > 1, (
        f"Expected multiple regions with low coupling; got {len(regions)}"
    )


def test_high_noise_reduces_coherence():
    """High noise (nu=0.8) should yield lower final r than noiseless case."""
    r_clean, _ = _run(_HIGH_NOISE_CLEAN_PARAMS, cycles=2000)
    r_noisy, _ = _run(_HIGH_NOISE_NOISY_PARAMS, cycles=2000)
    r_clean_final = r_clean[-1]
    r_noisy_final = r_noisy[-1]
    assert r_noisy_final < r_clean_final, (
        f"Expected noise to reduce r; clean={r_clean_final:.4f}, noisy={r_noisy_final:.4f}"
    )


def test_swarmalator_spatial_spread():
    """Swarmalator mode should show positive spatial spread."""
    _, kernel = _run(_SWARM_PARAMS, cycles=500, swarmalator=True)
    spread = _spatial_spread(kernel)
    assert spread > 0.0, f"Swarmalator spatial spread should be > 0; got {spread}"


def test_proctor_certifies_high_coupling():
    """Proctor suite should certify the high-coupling run."""
    r_history, kernel = _run(_HIGH_COUPLING_PARAMS, cycles=2000)
    report = kernel.run_proctors(r_history, window=50, convergence_tol=1e-4)
    assert report.certified, (
        f"Proctor not certified: r_finite={report.r_finite}, "
        f"r_in_range={report.r_in_range}, phases_finite={report.phases_finite}, "
        f"converged={report.converged}, trending_up={report.trending_up}, "
        f"final_r={report.final_r:.4f}, tail_var={report.r_tail_var:.2e}"
    )


def test_proctor_report_fields():
    """GlyphlockProctorReport fields should all be populated and valid."""
    r_history, kernel = _run(_HIGH_COUPLING_PARAMS, cycles=200)
    report = kernel.run_proctors(r_history, window=50, convergence_tol=1e-4)
    assert 0.0 <= report.final_r <= 1.0
    assert np.isfinite(report.r_tail_var)
    assert report.window == 50
    assert report.convergence_tol == 1e-4
    assert isinstance(report.certified, bool)


def test_detect_regions_all_singletons_after_reset():
    """With zero coupling and zero noise all oscillators diverge to singletons."""
    p = GlyphlockParams(n=5, k=0.0, r_ruach=0.0, nu=0.0, chirality=0.0, seed=42)
    kernel = GlyphlockKernel(p)
    # Let phases spread by running for many cycles (different natural frequencies)
    for _ in range(5000):
        kernel.step()
    regions = kernel.detect_regions(phase_tol=0.1)
    # Each oscillator should be in its own singleton (or very small cluster)
    largest = max(len(c) for c in regions)
    assert largest < p.n, (
        f"Expected oscillators to diverge without coupling; largest cluster={largest}"
    )


def test_params_validate_alpha():
    """GlyphlockParams.validate() should accept finite alpha values."""
    p = GlyphlockParams(alpha=np.pi / 3)
    p.validate()  # should not raise

    p_bad = GlyphlockParams(alpha=float("nan"))
    with pytest.raises(ValueError, match="alpha"):
        p_bad.validate()


def test_run_proctors_requires_nonempty_history():
    """run_proctors() should raise ValueError on empty r_history."""
    p = GlyphlockParams(n=5, seed=0)
    kernel = GlyphlockKernel(p)
    with pytest.raises(ValueError, match="non-empty"):
        kernel.run_proctors([])


# ---------------------------------------------------------------------------
# Standalone script mode — prints a debug summary table
# ---------------------------------------------------------------------------

def _run_all_standalone(strict: bool = False) -> None:
    """Run all coherence scenarios and print a summary table."""
    results: list[CoherenceResult] = []

    print("\n" + "=" * 70)
    print("GLYPHLOCK COHERENCE DEBUG SUITE")
    print("=" * 70)

    # 1. High coupling
    results.append(
        run_coherence_test(
            label="high_coupling_n22",
            params=_HIGH_COUPLING_PARAMS,
            cycles=2000,
            r_min=0.95,
            check_proctors=True,
            strict=strict,
        )
    )

    # 2. Low coupling
    results.append(
        run_coherence_test(
            label="low_coupling_n22",
            params=_LOW_COUPLING_PARAMS,
            cycles=2000,
            r_max=0.70,
            strict=strict,
        )
    )

    # 3. High noise clean baseline
    results.append(
        run_coherence_test(
            label="moderate_coupling_noiseless",
            params=_HIGH_NOISE_CLEAN_PARAMS,
            cycles=2000,
            strict=strict,
        )
    )

    # 4. High noise
    results.append(
        run_coherence_test(
            label="moderate_coupling_high_noise_nu08",
            params=_HIGH_NOISE_NOISY_PARAMS,
            cycles=2000,
            strict=strict,
        )
    )
    r_clean = results[-2].final_r
    r_noisy = results[-1].final_r
    noise_ok = r_noisy < r_clean
    print(
        f"\n  [NOISE CHECK] clean_r={r_clean:.4f}, noisy_r={r_noisy:.4f} "
        f"=> noise reduces r: {'PASS' if noise_ok else 'FAIL'}"
    )
    if strict and not noise_ok:
        raise AssertionError(
            f"Noise should reduce final r: clean={r_clean:.4f}, noisy={r_noisy:.4f}"
        )

    # 5. Sakaguchi lag
    results.append(
        run_coherence_test(
            label="sakaguchi_lag_alpha_pi4",
            params=_ALPHA_PARAMS,
            cycles=2000,
            r_min=0.80,
            strict=strict,
        )
    )

    # 6. Swarmalator
    results.append(
        run_coherence_test(
            label="swarmalator_n22",
            params=_SWARM_PARAMS,
            cycles=500,
            swarmalator=True,
            r_min=0.80,
            strict=strict,
        )
    )

    # 7. Low-dimensional n=5
    results.append(
        run_coherence_test(
            label="low_dim_n5",
            params=_LOW_DIM_PARAMS,
            cycles=2000,
            r_min=0.95,
            strict=strict,
        )
    )

    # 8. High-dimensional n=22
    results.append(
        run_coherence_test(
            label="high_dim_n22",
            params=_HIGH_DIM_PARAMS,
            cycles=2000,
            r_min=0.95,
            check_proctors=True,
            strict=strict,
        )
    )

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    header = (
        f"{'Label':<40} {'r_final':>8} {'r_exp':>12} "
        f"{'conv':>6} {'var':>8} {'clust%':>7} {'status':>6}"
    )
    print(header)
    print("-" * 70)
    for res in results:
        lo = f"{res.r_expected_min:.2f}" if res.r_expected_min is not None else " -"
        hi = f"{res.r_expected_max:.2f}" if res.r_expected_max is not None else " -"
        r_exp = f"[{lo},{hi}]"
        conv = str(res.convergence_cycle) if res.convergence_cycle else "n/a"
        status = "PASS" if res.passed else "FAIL"
        print(
            f"{res.label:<40} {res.final_r:>8.4f} {r_exp:>12} "
            f"{conv:>6} {res.phase_variance:>8.4f} {res.largest_cluster_pct:>6.1%} {status:>6}"
        )

    n_pass = sum(r.passed for r in results)
    n_fail = len(results) - n_pass
    print("=" * 70)
    print(f"  TOTAL: {len(results)} scenarios  |  PASSED: {n_pass}  |  FAILED: {n_fail}")
    print("=" * 70 + "\n")

    if strict and n_fail > 0:
        raise SystemExit(f"{n_fail} coherence test(s) failed")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run Glyphlock coherence debug suite")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on any threshold failure",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging",
    )
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger("test_coherence").setLevel(logging.INFO)
        logging.getLogger("test_coherence").handlers = []
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s | %(message)s",
        )

    _run_all_standalone(strict=args.strict)
