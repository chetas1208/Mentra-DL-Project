#!/usr/bin/env bash
# Keep Mentra backend + Cloudflare quick tunnel alive across SSH drops,
# OOM kills, and cloudflared restarts.
#
# Started by scripts/mentra/start_stack.sh (nohup). Do not run directly unless
# debugging.
#
# Watchdog interval: 60s (MENTRA_KEEPALIVE_INTERVAL override).
# URL lock: logs/tunnel_url.lock.json — canonical wss URL pinned locally for
# 30 days (MENTRA_URL_LOCK_DAYS). Does NOT push to Vercel; local reference
# only. Live URL (if different after tunnel restart) goes to tunnel_url.live.txt.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

RECEIVER_LOG="$LOG_DIR/receiver.log"
TUNNEL_LOG="$LOG_DIR/cloudflared.log"
TUNNEL_URL_FILE="$LOG_DIR/tunnel_url.txt"
TUNNEL_URL_LIVE_FILE="$LOG_DIR/tunnel_url.live.txt"
TUNNEL_LOCK_FILE="$LOG_DIR/tunnel_url.lock.json"
PID_DIR="$LOG_DIR/pids"
mkdir -p "$PID_DIR"

RECEIVER_PID_FILE="$PID_DIR/receiver.pid"
TUNNEL_PID_FILE="$PID_DIR/cloudflared.pid"

LISTEN="${MENTRA_LISTEN:-127.0.0.1}"
PORT="${MENTRA_PORT:-8765}"
CHECK_INTERVAL="${MENTRA_KEEPALIVE_INTERVAL:-60}"
URL_LOCK_DAYS="${MENTRA_URL_LOCK_DAYS:-30}"

log() {
  echo "[$(date -Iseconds)] $*"
}

port_open() {
  ss -tln 2>/dev/null | grep -q ":${PORT} "
}

receiver_running() {
  if [[ -f "$RECEIVER_PID_FILE" ]]; then
    local pid
    pid="$(cat "$RECEIVER_PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  fi
  pgrep -f "scripts/mentra/run_receiver.py" >/dev/null 2>&1
}

tunnel_running() {
  if [[ -f "$TUNNEL_PID_FILE" ]]; then
    local pid
    pid="$(cat "$TUNNEL_PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  fi
  pgrep -f "cloudflared tunnel --url http://${LISTEN}:${PORT}" >/dev/null 2>&1
}

lock_expires_epoch() {
  "$ROOT/.venv/bin/python3" - <<'PY' "$TUNNEL_LOCK_FILE"
import json, sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(0)
    raise SystemExit(0)
data = json.loads(path.read_text())
exp = data.get("expires_at", "")
if not exp:
    print(0)
    raise SystemExit(0)
dt = datetime.fromisoformat(exp.replace("Z", "+00:00"))
if dt.tzinfo is None:
    dt = dt.replace(tzinfo=timezone.utc)
print(int(dt.timestamp()))
PY
}

lock_valid() {
  [[ -f "$TUNNEL_LOCK_FILE" ]] || return 1
  local now exp
  now="$(date +%s)"
  exp="$(lock_expires_epoch)"
  [[ "$exp" -gt 0 && "$now" -lt "$exp" ]]
}

locked_https_url() {
  "$ROOT/.venv/bin/python3" - <<'PY' "$TUNNEL_LOCK_FILE"
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
print(json.loads(path.read_text()).get("https_url", ""))
PY
}

locked_wss_url() {
  "$ROOT/.venv/bin/python3" - <<'PY' "$TUNNEL_LOCK_FILE"
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
print(json.loads(path.read_text()).get("wss_url", ""))
PY
}

write_url_lock() {
  local https_url="$1"
  local wss_url="${https_url/https:\/\//wss://}"
  "$ROOT/.venv/bin/python3" - <<'PY' "$TUNNEL_LOCK_FILE" "$https_url" "$wss_url" "$URL_LOCK_DAYS"
import json, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

lock_path = Path(sys.argv[1])
https_url = sys.argv[2]
wss_url = sys.argv[3]
days = int(sys.argv[4])
now = datetime.now(timezone.utc)
payload = {
    "https_url": https_url,
    "wss_url": wss_url,
    "locked_at": now.isoformat(),
    "expires_at": (now + timedelta(days=days)).isoformat(),
    "lock_days": days,
    "note": "Local-only canonical URL. Does not auto-update Vercel.",
}
lock_path.write_text(json.dumps(payload, indent=2) + "\n")
print(wss_url)
PY
}

publish_canonical_url() {
  local live_https="${1:-}"
  local canonical_wss canonical_https

  if lock_valid; then
    canonical_https="$(locked_https_url)"
    canonical_wss="$(locked_wss_url)"
    echo "$canonical_wss" >"$TUNNEL_URL_FILE"
    if [[ -n "$live_https" && "$live_https" != "$canonical_https" ]]; then
      echo "$live_https" >"$TUNNEL_URL_LIVE_FILE"
      log "WARN: live tunnel URL differs from 30-day lock — locked=${canonical_https} live=${live_https} (update Vercel or wait for lock expiry)"
    else
      rm -f "$TUNNEL_URL_LIVE_FILE"
    fi
    return 0
  fi

  if [[ -z "$live_https" ]]; then
    return 0
  fi

  canonical_wss="$(write_url_lock "$live_https")"
  echo "$canonical_wss" >"$TUNNEL_URL_FILE"
  rm -f "$TUNNEL_URL_LIVE_FILE"
  log "URL lock created for ${URL_LOCK_DAYS} days: ${canonical_wss}"
}

start_receiver() {
  if receiver_running && port_open; then
    return 0
  fi
  log "starting receiver on ${LISTEN}:${PORT}"
  nohup "$ROOT/.venv/bin/python3" scripts/mentra/run_receiver.py \
    --listen "$LISTEN" --port "$PORT" >>"$RECEIVER_LOG" 2>&1 &
  echo $! >"$RECEIVER_PID_FILE"
  for _ in $(seq 1 60); do
    if port_open; then
      log "receiver ready (pid $(cat "$RECEIVER_PID_FILE"))"
      return 0
    fi
    sleep 1
  done
  log "ERROR: receiver failed to bind ${LISTEN}:${PORT} within 60s"
  return 1
}

extract_tunnel_url() {
  grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | tail -1 || true
}

start_tunnel() {
  local existing live
  if tunnel_running; then
    existing="$(extract_tunnel_url)"
    publish_canonical_url "$existing"
    return 0
  fi
  log "starting cloudflared quick tunnel -> http://${LISTEN}:${PORT}"
  : >"$TUNNEL_LOG"
  nohup cloudflared tunnel --url "http://${LISTEN}:${PORT}" --loglevel info \
    >>"$TUNNEL_LOG" 2>&1 &
  echo $! >"$TUNNEL_PID_FILE"
  for _ in $(seq 1 45); do
    live="$(extract_tunnel_url)"
    if [[ -n "$live" ]]; then
      publish_canonical_url "$live"
      log "tunnel ready: live=${live} canonical=$(cat "$TUNNEL_URL_FILE") (pid $(cat "$TUNNEL_PID_FILE"))"
      return 0
    fi
    sleep 1
  done
  log "ERROR: cloudflared did not publish a trycloudflare.com URL within 45s"
  return 1
}

# Seed lock from existing tunnel_url.txt if we have a live tunnel but no lock yet.
if [[ -f "$TUNNEL_URL_FILE" && ! -f "$TUNNEL_LOCK_FILE" ]]; then
  existing="$(cat "$TUNNEL_URL_FILE")"
  if [[ "$existing" == wss://* ]]; then
    publish_canonical_url "${existing/wss:\/\//https:\/\/}"
  elif [[ "$existing" == https://* ]]; then
    publish_canonical_url "$existing"
  fi
fi

log "keepalive starting (watchdog every ${CHECK_INTERVAL}s, URL lock ${URL_LOCK_DAYS}d local-only)"
start_receiver
start_tunnel

while true; do
  if ! receiver_running || ! port_open; then
    log "receiver down — restarting"
    rm -f "$RECEIVER_PID_FILE"
    start_receiver || true
  fi
  if ! tunnel_running; then
    log "tunnel down — restarting (live URL may change; canonical URL stays locked until expiry)"
    rm -f "$TUNNEL_PID_FILE"
    start_tunnel || true
  else
    publish_canonical_url "$(extract_tunnel_url)"
  fi
  sleep "$CHECK_INTERVAL"
done
