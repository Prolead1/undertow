#!/usr/bin/env bash
#
# smoke_ui.sh — end-to-end HTTP smoke test for the read-only dashboard API.
#
# Starts `undertow-ui serve` against the committed synthetic run under
# `scripts/smoke_fixture/runs/`, curls the contract endpoints, asserts the
# expected keys (no WebSocket, no browser), then shuts the server down.  This is
# the U11 integration smoke: it needs only `uv`, `curl` and `python3`.
#
# Usage:  bash scripts/smoke_ui.sh
# Env:    SMOKE_UI_PORT  (default 8137) — override if that port is busy.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNS_DIR="${SCRIPT_DIR}/smoke_fixture/runs"
HOST="127.0.0.1"
PORT="${SMOKE_UI_PORT:-8137}"
BASE_URL="http://${HOST}:${PORT}"

WORK_DIR="$(mktemp -d)"
SERVER_LOG="${WORK_DIR}/serve.log"
SERVER_PID=""

cleanup() {
  local status=$?
  if [[ -n "${SERVER_PID}" ]]; then
    # `uv run` may leave the uvicorn child behind; kill the tree.
    if command -v pkill >/dev/null 2>&1; then
      pkill -TERM -P "${SERVER_PID}" 2>/dev/null || true
    fi
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  rm -rf "${WORK_DIR}"
  exit "${status}"
}
trap cleanup EXIT INT TERM

fail() {
  printf 'smoke_ui: FAIL: %s\n' "$*" >&2
  if [[ -s "${SERVER_LOG}" ]]; then
    printf -- '--- server log ---\n' >&2
    cat "${SERVER_LOG}" >&2
  fi
  exit 1
}

pass() { printf 'smoke_ui: ok: %s\n' "$*"; }

require_key() {
  # require_key <file> <ere-pattern> <description>
  local file="$1" pattern="$2" description="$3"
  if ! grep -Eq "${pattern}" "${file}"; then
    printf -- '--- %s ---\n' "${description}" >&2
    cat "${file}" >&2
    printf '\n' >&2
    fail "${description}: missing pattern ${pattern}"
  fi
}

# --- start the server -------------------------------------------------------
( cd "${REPO_ROOT}" && exec uv run undertow-ui serve --runs "${RUNS_DIR}" \
    --host "${HOST}" --port "${PORT}" ) >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

health_ok=0
# `uv run` can take a while to resolve the environment on a cold cache.
for _ in $(seq 1 120); do
  if curl -fsS "${BASE_URL}/api/health" -o "${WORK_DIR}/health.json" 2>/dev/null; then
    health_ok=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    fail "server exited before becoming healthy"
  fi
  sleep 0.5
done
[[ "${health_ok}" == 1 ]] || fail "server did not become healthy within 60s"

# --- /api/health ------------------------------------------------------------
require_key "${WORK_DIR}/health.json" '"status"[[:space:]]*:[[:space:]]*"ok"' "health status"
require_key "${WORK_DIR}/health.json" '"schema_version"[[:space:]]*:[[:space:]]*1' "health schema_version"
pass "GET /api/health"

# --- /api/runs --------------------------------------------------------------
curl -fsS "${BASE_URL}/api/runs" -o "${WORK_DIR}/runs.json"
require_key "${WORK_DIR}/runs.json" '"run_id"' "runs run_id"
require_key "${WORK_DIR}/runs.json" '"state"' "runs state"
RUN_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[0]["run_id"])' "${WORK_DIR}/runs.json")"
[[ -n "${RUN_ID}" ]] || fail "no run discovered in fixture"
pass "GET /api/runs (${RUN_ID})"

# URL-encode the id for path segments (fixture ids are safe, but be defensive).
RUN_ID_ENC="$(python3 -c 'import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "${RUN_ID}")"

# --- /api/runs/<id>/curve ---------------------------------------------------
curl -fsS "${BASE_URL}/api/runs/${RUN_ID_ENC}/curve" -o "${WORK_DIR}/curve.json"
require_key "${WORK_DIR}/curve.json" '"run_id"' "curve run_id"
require_key "${WORK_DIR}/curve.json" '"timesteps"' "curve timesteps"
require_key "${WORK_DIR}/curve.json" '"mean"' "curve mean"
require_key "${WORK_DIR}/curve.json" '"std"' "curve std"
pass "GET /api/runs/<id>/curve"

# --- /api/runs/<id>/figures/learning ----------------------------------------
curl -fsS "${BASE_URL}/api/runs/${RUN_ID_ENC}/figures/learning" -o "${WORK_DIR}/figure.json"
require_key "${WORK_DIR}/figure.json" '"data"' "figure data"
require_key "${WORK_DIR}/figure.json" '"layout"' "figure layout"
require_key "${WORK_DIR}/figure.json" '"config"' "figure config"
if grep -Eq 'NaN|Infinity' "${WORK_DIR}/figure.json"; then
  cat "${WORK_DIR}/figure.json" >&2
  fail "figure JSON contains NaN/Infinity"
fi
pass "GET /api/runs/<id>/figures/learning"

# --- /api/runs/<id>/telemetry (WS-less polling fallback) --------------------
curl -fsS "${BASE_URL}/api/runs/${RUN_ID_ENC}/telemetry?since=0&limit=100" \
  -o "${WORK_DIR}/telemetry.json"
require_key "${WORK_DIR}/telemetry.json" '"points"' "telemetry points"
require_key "${WORK_DIR}/telemetry.json" '"next_offset"' "telemetry next_offset"
require_key "${WORK_DIR}/telemetry.json" '"status"' "telemetry status"
require_key "${WORK_DIR}/telemetry.json" '"timestep"' "telemetry timestep"
pass "GET /api/runs/<id>/telemetry"

printf 'smoke_ui: PASS\n'
