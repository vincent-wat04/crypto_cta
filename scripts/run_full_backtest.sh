#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Full Rule-Backtest Pipeline for Shortlisted Candidates
#
# Phase 1: Fetch 30 days of data (if needed)
# Phase 2: Run full backtest for all candidates × fill rates × strategy combos
#
# Runtime estimates (SOL/USDC, 30 days):
#   Data fetch:   ~40-60 min  (REST API, ~10K requests)
#   Full backtest: ~10-30 min (14 candidates × 3 fills × ~56 strategy combos)
#   Quick mode:    ~3-8 min   (14 candidates × 3 fills × 6 strategy combos)
#
# Usage:
#   ./scripts/run_full_backtest.sh                    # full run
#   ./scripts/run_full_backtest.sh --quick            # reduced grid
#   ./scripts/run_full_backtest.sh --tier 1           # Tier 1 only
#   ./scripts/run_full_backtest.sh --fetch            # force data re-fetch
#   ./scripts/run_full_backtest.sh --scan-only        # skip fetch
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ── Activate venv ──
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

SYMBOL="SOL/USDC"
SYMBOL_TAG="SOL_USDC"
DAYS=30

# ── Parse args ──
DO_FETCH="auto"
EXTRA_ARGS=""

for arg in "$@"; do
    case "$arg" in
        --fetch)      DO_FETCH="yes" ;;
        --scan-only)  DO_FETCH="no" ;;
        --quick)      EXTRA_ARGS="${EXTRA_ARGS} --quick" ;;
        --tier)       ;;  # handled by next arg
        1|2)          EXTRA_ARGS="${EXTRA_ARGS} --tier ${arg}" ;;
    esac
done

# Handle --tier N pattern
PREV=""
for arg in "$@"; do
    if [ "$PREV" = "--tier" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --tier ${arg}"
    fi
    PREV="$arg"
done

# ── Resolve data root ──
DATA_ROOT="${MR_DATA_ROOT:-${PROJECT_ROOT}/data}"
CACHE_DIR="${DATA_ROOT}/cache/${SYMBOL_TAG}/trades_perp"
BT_OUTPUT="${DATA_ROOT}/full_backtest_results/${SYMBOL_TAG}"

echo "═══════════════════════════════════════════════════════"
echo "Full Rule-Backtest Pipeline"
echo "  Symbol:    ${SYMBOL}"
echo "  Days:      ${DAYS}"
echo "  Data root: ${DATA_ROOT}"
echo "  Cache:     ${CACHE_DIR}"
echo "  Output:    ${BT_OUTPUT}"
echo "═══════════════════════════════════════════════════════"

# ── Phase 1: Fetch 30 days of data ──
fetch_data() {
    echo ""
    echo "── Phase 1: Fetching ${DAYS} days of aggTrades ──"
    echo "  Estimated time: ~40-60 minutes (REST API)"
    echo ""

    python scripts/fetch_perpetual_trades.py \
        --symbol "${SYMBOL}" \
        --days "${DAYS}"

    N_FILES=$(find "${CACHE_DIR}" -maxdepth 1 -name '*.parquet' ! -name '._*' 2>/dev/null | wc -l | tr -d ' ')
    echo "  Cached files: ${N_FILES}"
}

if [ "$DO_FETCH" = "yes" ]; then
    fetch_data
elif [ "$DO_FETCH" = "auto" ]; then
    mkdir -p "${CACHE_DIR}"
    N_CACHED=$(find "${CACHE_DIR}" -maxdepth 1 -name '*.parquet' ! -name '._*' 2>/dev/null | wc -l | tr -d ' ')
    if [ "$N_CACHED" -lt 25 ]; then
        echo "  Only ${N_CACHED} cached files found (need ~30 for 30 days). Fetching..."
        fetch_data
    else
        echo "  Found ${N_CACHED} cached files, skipping fetch."
    fi
fi

# ── Phase 2: Full backtest ──
echo ""
echo "── Phase 2: Full Rule Backtest ──"
echo "  14 candidates (Tier 1 + Tier 2)"
echo "  3 maker_fill_rates × multiple stop/exit/sizer combos"
echo ""

python scripts/run_full_backtest_candidates.py \
    --symbol "${SYMBOL}" \
    --output_dir "${BT_OUTPUT}" \
    ${EXTRA_ARGS}

echo ""
echo "── Results ──"
if [ -f "${BT_OUTPUT}/full_backtest_summary.txt" ]; then
    cat "${BT_OUTPUT}/full_backtest_summary.txt"
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "Done."
echo "═══════════════════════════════════════════════════════"
