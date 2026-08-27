#!/usr/bin/env bash
# Start Mentra backend stack under nohup (receiver + cloudflared + watchdog).
#
#   scripts/mentra/start_stack.sh          # start if not already running
#   scripts/mentra/start_stack.sh --force  # restart watchdog (receiver/tunnel restarted only if dead)
#
# Logs: logs/keepalive.log, logs/receiver.log, logs/cloudflared.log
# Canonical public URL (30-day local lock): logs/tunnel_url.txt

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/logs"
PID_DIR="$LOG_DIR/pids"
KEEPALIVE_PID_FILE="$PID_DIR/keepalive.pid"
KEEPALIVE_SCRIPT="$ROOT/scripts/mentra/keepalive.sh"

mkdir -p "$LOG_DIR" "$PID_DIR"

log() {
  echo "[$(date -Iseconds)] $*"
}

keepalive_running() {
  if [[ -f "$KEEPALIVE_PID_FILE" ]]; then
    local pid
    pid="$(cat "$KEEPALIVE_PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  fi
  pgrep -f "$KEEPALIVE_SCRIPT" >/dev/null 2>&1
}

if [[ "${1:-}" == "--force" ]] && keepalive_running; then
  log "stopping existing keepalive"
  if [[ -f "$KEEPALIVE_PID_FILE" ]]; then
    kill "$(cat "$KEEPALIVE_PID_FILE")" 2>/dev/null || true
  fi
  pkill -f "scripts/mentra/keepalive.sh" 2>/dev/null || true
  rm -f "$KEEPALIVE_PID_FILE"
  sleep 1
fi

if keepalive_running; then
  log "keepalive already running (pid $(cat "$KEEPALIVE_PID_FILE" 2>/dev/null || pgrep -f "$KEEPALIVE_SCRIPT"))"
  if [[ -f "$LOG_DIR/tunnel_url.txt" ]]; then
    log "locked URL: $(cat "$LOG_DIR/tunnel_url.txt")"
  fi
  exit 0
fi

log "starting keepalive under nohup (1-min watchdog, 30-day local URL lock)"
nohup "$KEEPALIVE_SCRIPT" >>"$LOG_DIR/keepalive.log" 2>&1 &
echo $! >"$KEEPALIVE_PID_FILE"
log "keepalive pid $(cat "$KEEPALIVE_PID_FILE") — tail -f $LOG_DIR/keepalive.log"
