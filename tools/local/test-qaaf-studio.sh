#!/usr/bin/env bash
# test-qaaf-studio.sh — Validation rapide de QAAF Studio par layer
#
# Usage:
#   ./test-qaaf-studio.sh              → tous les layers
#   ./test-qaaf-studio.sh --layer 1    → Layer 1 uniquement
#   ./test-qaaf-studio.sh --layer 2    → Layer 2 uniquement
#   ./test-qaaf-studio.sh --layer 3    → Layer 3 uniquement
#   ./test-qaaf-studio.sh --layer 4    → Layer 4 uniquement
#   ./test-qaaf-studio.sh --smoke      → smoke tests uniquement (import + CNSR)

set -euo pipefail

LAYER=""
SMOKE_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --layer) LAYER="$2"; shift 2 ;;
        --smoke) SMOKE_ONLY=true; shift ;;
        *) echo "Usage: $0 [--layer 1|2|3|4] [--smoke]"; exit 1 ;;
    esac
done

PASS=0
FAIL=0

_run() {
    local label="$1"; shift
    if "$@" &>/dev/null; then
        echo "[OK]  $label"
        PASS=$((PASS+1))
    else
        echo "[FAIL] $label"
        FAIL=$((FAIL+1))
    fi
}

_pytest() {
    local label="$1"; shift
    if python -m pytest "$@" -q --tb=short 2>&1 | tail -1 | grep -q "passed"; then
        echo "[OK]  $label"
        PASS=$((PASS+1))
    else
        echo "[FAIL] $label"
        python -m pytest "$@" -q --tb=short 2>&1 | tail -5
        FAIL=$((FAIL+1))
    fi
}

echo ""
echo "════════════════════════════════════════"
echo "  QAAF Studio — Test Suite"
echo "════════════════════════════════════════"

# ── Smoke tests (toujours exécutés) ─────────────────────────────────────────
echo ""
echo "── Smoke tests ──"

_run "layer1_engine imports" python -c "
from layer1_engine import MetricsEngine, compute_cnsr, deflated_sharpe_ratio
from layer1_engine import Backtester, BenchmarkFactory
"

_run "layer2_qualification imports" python -c "
from layer2_qualification.paf.paf_d1_hierarchy import run_d1, D1Result
from layer2_qualification.paf.paf_d2_attribution import run_d2, D2Result
from layer2_qualification.paf.paf_d3_source import run_d3, D3Result
from layer2_qualification.paf.paf_runner import run_paf, load_paf_bundle
"

_run "layer3_validation imports" python -c "
from layer3_validation import METISRunner, METISReport
from layer3_validation.metis_q1_walkforward import run_q1
from layer3_validation.metis_q2_permutation import run_q2
from layer3_validation.metis_q3_ema_stability import run_q3
from layer3_validation.metis_q4_dsr import run_q4
"

_run "compute_cnsr smoke test" python -c "
import numpy as np, pandas as pd
from layer1_engine import compute_cnsr
r = pd.Series(np.random.randn(252) * 0.02)
result = compute_cnsr(r, r)
assert 'cnsr_usd_fed' in result
"

if $SMOKE_ONLY; then
    echo ""
    echo "────────────────────────────────────────"
    echo "  Smoke : $PASS OK, $FAIL FAIL"
    echo "════════════════════════════════════════"
    exit $([[ $FAIL -eq 0 ]] && echo 0 || echo 1)
fi

# ── Layer 1 ──────────────────────────────────────────────────────────────────
if [[ -z "$LAYER" || "$LAYER" == "1" ]]; then
    echo ""
    echo "── Layer 1 — Engine ──"
    _pytest "test_layer1_backtester" tests/test_layer1_backtester.py
    _pytest "test_layer1_metrics" tests/test_layer1_metrics.py 2>/dev/null || \
        _run "test_layer1_metrics (skip if absent)" true
    _pytest "test_benchmark_calibration" tests/test_benchmark_calibration.py
fi

# ── Layer 2 ──────────────────────────────────────────────────────────────────
if [[ -z "$LAYER" || "$LAYER" == "2" ]]; then
    echo ""
    echo "── Layer 2 — PAF ──"
    _pytest "test_layer2_paf" tests/test_layer2_paf.py
    _pytest "test_layer2_paf_adversarial (non-slow)" \
        tests/test_layer2_paf_adversarial.py -m "not slow"
fi

# ── Layer 3 ──────────────────────────────────────────────────────────────────
if [[ -z "$LAYER" || "$LAYER" == "3" ]]; then
    echo ""
    echo "── Layer 3 — MÉTIS ──"
    _pytest "test_layer3_metis" tests/test_layer3_metis.py
fi

# ── Layer 4 ──────────────────────────────────────────────────────────────────
if [[ -z "$LAYER" || "$LAYER" == "4" ]]; then
    echo ""
    echo "── Layer 4 — KB + D-SIG ──"
    _pytest "test_layer4_dsig (7 tests)" tests/test_layer4_dsig.py
    if python -m pytest tests/test_layer4_dsig.py::test_gate1_b5050_score_range -q --tb=short 2>&1 | grep -q "passed"; then
        echo "[OK]  Gate 1 — score(B_5050) ∈ [72,78] — v0.4→v1.0 autorisé"
    else
        echo "[FAIL] Gate 1 — score(B_5050) hors [72,78]"
        FAIL=$((FAIL+1))
    fi
fi

echo ""
echo "────────────────────────────────────────"
echo "  Résultat : $PASS OK, $FAIL FAIL"
echo "════════════════════════════════════════"
echo ""

[[ $FAIL -eq 0 ]]
