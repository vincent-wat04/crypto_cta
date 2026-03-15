#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Signal-Quality Scan Pipeline
#
# Phase 1: Fetch data (if needed) → Phase 2: Run signal scan
#
# Runtime estimates (SOL/USDC, ~13 days, ~5M trades):
#   Data fetch:   ~20-30 min  (REST API rate-limited, ~5K requests)
#   Signal scan:  ~5-15 min   (21 factors × 4 levels × 2 sources × ~50 signals)
#   Total:        ~25-45 min  (first run with fetch)
#                 ~5-15 min   (subsequent runs, cached data)
#
# External drive for cache:
#   export MR_DATA_ROOT=/Volumes/YourDrive/mean_reversion_data
#   (affects both fetcher and scanner)
#
# Usage:
#   ./scripts/run_signal_scan.sh              # full run (skip fetch if cached)
#   ./scripts/run_signal_scan.sh --fetch      # force re-fetch data first
#   ./scripts/run_signal_scan.sh --scan-only  # skip fetch, scan only
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
DAYS=14

# ── Parse args ──
DO_FETCH="auto"
DO_SCAN=true

for arg in "$@"; do
    case "$arg" in
        --fetch)      DO_FETCH="yes" ;;
        --scan-only)  DO_FETCH="no" ;;
    esac
done

# ── Resolve data root ──
DATA_ROOT="${MR_DATA_ROOT:-${PROJECT_ROOT}/data}"
CACHE_DIR="${DATA_ROOT}/cache/${SYMBOL_TAG}/trades_perp"
SCAN_OUTPUT="${DATA_ROOT}/signal_scan_results/${SYMBOL_TAG}"

echo "═══════════════════════════════════════════════════════"
echo "Signal Scan Pipeline"
echo "  Symbol:    ${SYMBOL}"
echo "  Data root: ${DATA_ROOT}"
echo "  Cache:     ${CACHE_DIR}"
echo "  Output:    ${SCAN_OUTPUT}"
echo "═══════════════════════════════════════════════════════"

# ── Phase 1: Fetch data ──
fetch_data() {
    echo ""
    echo "── Phase 1: Fetching ${DAYS} days of aggTrades ──"
    echo "  Estimated time: ~20-30 minutes (REST API, ~5K requests)"
    echo ""

    python scripts/fetch_perpetual_trades.py \
        --symbol "${SYMBOL}" \
        --days "${DAYS}"

    N_FILES=$(find "${CACHE_DIR}" -maxdepth 1 -name '*.parquet' 2>/dev/null | wc -l | tr -d ' ')
    echo "  Cached files: ${N_FILES}"
}

if [ "$DO_FETCH" = "yes" ]; then
    fetch_data
elif [ "$DO_FETCH" = "auto" ]; then
    # Create cache dir if it doesn't exist
    mkdir -p "${CACHE_DIR}"
    N_CACHED=$(find "${CACHE_DIR}" -maxdepth 1 -name '*.parquet' 2>/dev/null | wc -l | tr -d ' ')
    if [ "$N_CACHED" -lt 5 ]; then
        echo "  Only ${N_CACHED} cached files found. Fetching data..."
        fetch_data
    else
        echo "  Found ${N_CACHED} cached files, skipping fetch."
    fi
fi

# ── Phase 2: Signal scan ──
if [ "$DO_SCAN" = true ]; then
    echo ""
    echo "── Phase 2: Signal Scan ──"
    echo "  4 trade-count levels: [150, 500, 2000, 6000]"
    echo "  2 trade sources: [raw, merged]"
    echo "  8 raw factors + 13 merged factors = 21 total"
    echo "  Estimated time: ~5-15 minutes"
    echo ""

    python scripts/run_signal_scan.py \
        --symbol "${SYMBOL}" \
        --trade_source both \
        --output_dir "${SCAN_OUTPUT}"

    echo ""
    echo "── Results ──"
    if [ -f "${SCAN_OUTPUT}/signal_scan_summary.txt" ]; then
        head -60 "${SCAN_OUTPUT}/signal_scan_summary.txt"
    fi
    echo ""
    echo "Full results: ${SCAN_OUTPUT}/signal_scan_combined.csv"
    echo "Summary:      ${SCAN_OUTPUT}/signal_scan_summary.txt"
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "Done."
echo "═══════════════════════════════════════════════════════"
