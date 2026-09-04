#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <pid> <output-dir>" >&2
  exit 2
fi
PID="$1"
OUTDIR="$2"
mkdir -p "$OUTDIR"
WATCHER_LOG="$OUTDIR/watcher.log"
log(){ echo "[$(date +%F_%T)] $*" >> "$WATCHER_LOG"; }
log "watcher started for PID=$PID OUT=$OUTDIR"

# wait for profile.log to appear
while [ ! -f "$OUTDIR/profile.log" ]; do
  log "waiting for profile.log..."
  sleep 15
done
log "profile.log detected"

# wait for PID to exit (if PID is 0 or missing, skip)
if [ "$PID" -ne 0 ] 2>/dev/null; then
  while kill -0 "$PID" 2>/dev/null; do
    log "pid $PID still running"
    sleep 30
  done
  log "pid $PID exited"
else
  log "no pid provided or pid=0, skipping wait"
fi

# collect metrics
log "collecting metrics from profile.log"
grep -E "AEE:|AAE:|estimated_total_sops:|sops:|global_firing|firing:" "$OUTDIR/profile.log" | tee "$OUTDIR/summary.txt" >> "$WATCHER_LOG" || echo "no metrics found" >> "$WATCHER_LOG"
log "summary written to $OUTDIR/summary.txt"
log "watcher done"
