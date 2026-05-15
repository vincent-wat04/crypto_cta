#!/usr/bin/env bash
# Stop the job started by fetch_perp_background.sh (SIGTERM on recorded pid), and
# pkill any stray fetch_perpetual_trades.py from this repo.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${MR_LOG_DIR:-$ROOT/notebooks/outputs/ofi_study}"
PIDFILE="${MR_FETCH_PIDFILE:-$LOG_DIR/fetch_perp_background.pid}"

if [[ -f "$PIDFILE" ]]; then
  pid=$(cat "$PIDFILE" 2>/dev/null || true)
  if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    echo "Sent SIGTERM to pid $pid (from $PIDFILE)"
  else
    echo "Pid $pid not running; removing stale pidfile"
  fi
  rm -f "$PIDFILE"
else
  echo "No pidfile at $PIDFILE"
fi

# Best-effort: same script path in any terminal session
if pkill -f "$ROOT/scripts/fetch_perpetual_trades.py" 2>/dev/null; then
  echo "Also signalled remaining fetch_perpetual_trades.py for this repo path"
else
  true
fi
