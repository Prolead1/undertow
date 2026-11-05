#!/usr/bin/env bash
#
# Main reproduction program for the Undertow report.
#
# The report's numbers come from long-running experiments, so the program is
# split into stages.  By default it re-runs every analysis, regenerates the
# appendix tables and figures, rebuilds the PDF and runs the checks.  Training
# is guarded because it takes hours; pass --train to include it.
#
# Usage (from anywhere):
#   scripts/report/reproduce.sh              # analyses + tables + figures + PDF + checks
#   scripts/report/reproduce.sh --train      # the above, after retraining both arms
#   scripts/report/reproduce.sh --analyses   # only the analysis CSVs
#   scripts/report/reproduce.sh --report     # only tables + figures + PDF
#
# The dataset snapshots and the recorded evaluation artefacts are distributed
# with the submission archive; they are excluded from version control because
# of their size.  The data pipeline is not re-run here because the fetch step
# needs API credentials.  See Appendix G of the report for the file inventory.
#
# Each stage loads a full market view, so the stages run one at a time and,
# where the host provides systemd user scopes, inside a memory-capped scope
# (override with UNDERTOW_MEM_MAX, systemd memory-limit syntax).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MEM_MAX="${UNDERTOW_MEM_MAX:-24G}"
if command -v systemd-run >/dev/null 2>&1 && systemctl --user is-system-running >/dev/null 2>&1; then
  RUN=(systemd-run --user --scope --quiet -p "MemoryMax=${MEM_MAX}")
else
  RUN=()
fi
UV="$(command -v uv)"

DO_TRAIN=0
DO_ANALYSES=0
DO_REPORT=0
for arg in "$@"; do
  case "$arg" in
    --train)    DO_TRAIN=1; DO_ANALYSES=1; DO_REPORT=1 ;;
    --analyses) DO_ANALYSES=1 ;;
    --report)   DO_REPORT=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done
if [[ $# -eq 0 ]]; then
  DO_ANALYSES=1
  DO_REPORT=1
fi

if [[ "$DO_TRAIN" == "1" ]]; then
  echo "== training the joint policy (5 seeds, ~2.5 h) =="
  "${RUN[@]}" "$UV" run undertow-sim train \
    --config configs/sim_eth_usdc_500_joint_rl.toml \
    --data-config configs/eth_usdc_500.toml \
    --output results/eth_usdc_500_joint_rl
  echo "== training the regime-feature ablation arm (5 seeds, ~3 h) =="
  "${RUN[@]}" "$UV" run undertow-sim train \
    --config configs/sim_eth_usdc_500_joint_noregime.toml \
    --data-config configs/eth_usdc_500.toml \
    --output results/eth_usdc_500_joint_noregime
fi

if [[ "$DO_ANALYSES" == "1" ]]; then
  echo "== analyses =="
  # Main joint band x hedge run: per-window metrics and paired intervals.
  "${RUN[@]}" "$UV" run python scripts/report/analyze_main_run.py
  # Static one-shot widths on train and eval (the frozen comparator).
  "${RUN[@]}" "$UV" run python scripts/report/static_widths.py
  # The two scripted recentring rules on the same evaluation windows.
  "${RUN[@]}" "$UV" run python scripts/report/extra_benchmarks.py
  # Gas and capital boundary sweep with the analytical LVR floor.
  "${RUN[@]}" "$UV" run python scripts/report/boundary_sweep.py
  # Fee-tier contrast ladder on the 0.30 % pool.
  "${RUN[@]}" "$UV" run python scripts/report/contrast_fee_tier.py
  # Regime-feature ablation arm (benchmarks are shared with the main run).
  "${RUN[@]}" "$UV" run python scripts/report/analyze_regime_ablation.py
fi

if [[ "$DO_REPORT" == "1" ]]; then
  echo "== appendix tables, figures and PDF =="
  "${RUN[@]}" "$UV" run python scripts/report/make_appendix_tables.py
  "${RUN[@]}" "$UV" run python report/figures/make_figures.py
  bash scripts/build_report.sh
  echo "== checks =="
  "$UV" run pytest -q
  "$UV" run python scripts/plagiarism_check.py --no-download
fi

echo "done"
