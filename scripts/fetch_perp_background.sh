#!/usr/bin/env bash
# Start perpetual aggTrades fetch detached from Terminal (macOS / Linux: nohup).
#
#   MR_DATA_ROOT   default: /Volumes/Lexar/mean_reversion_data
#   MR_LOG_DIR     default: repo notebooks/outputs/ofi_study
#   MR_FETCH_LOG   override log file path (default: timestamped under MR_LOG_DIR)
#   MR_FETCH_PIDFILE  override pid file (default: MR_LOG_DIR/fetch_perp_background.pid)
#
# Examples:
#   ./scripts/fetch_perp_background.sh
#   ./scripts/fetch_perp_background.sh --start-date 2026-01-01 --symbols SOL/USDC,BTC/USDC,ETH/USDC
#
# Stop: ./scripts/fetch_perp_stop.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${MR_LOG_DIR:-$ROOT/notebooks/outputs/ofi_study}"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="${MR_FETCH_LOG:-$LOG_DIR/fetch_perp_background_${STAMP}.log}"
PIDFILE="${MR_FETCH_PIDFILE:-$LOG_DIR/fetch_perp_background.pid}"
MR_DATA_ROOT="${MR_DATA_ROOT:-/Volumes/Lexar/mean_reversion_data}"
VENV_PY="${ROOT}/.venv/bin/python"
SCRIPT="${ROOT}/scripts/fetch_perpetual_trades.py"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Missing venv python: $VENV_PY" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  set -- --start-date 2026-01-01 --symbols SOL/USDC,BTC/USDC,ETH/USDC --pause-between-symbols 8
fi

if [[ -f "$PIDFILE" ]]; then
  oldpid=$(cat "$PIDFILE" 2>/dev/null || true)
  if [[ -n "${oldpid:-}" ]] && kill -0 "$oldpid" 2>/dev/null; then
    echo "fetch already running (pid $oldpid). Stop with: $ROOT/scripts/fetch_perp_stop.sh" >&2
    exit 1
  fi
fi

nohup env MR_DATA_ROOT="$MR_DATA_ROOT" \
  "$VENV_PY" -u "$SCRIPT" "$@" >>"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "Started background fetch pid=$(cat "$PIDFILE")"
echo "Log: $LOG"
echo "Stop:  $ROOT/scripts/fetch_perp_stop.sh"
