#!/usr/bin/env bash
# Launch a plan execution detached, so it survives this shell.
#
#   ./run_sim.sh <plan.json> <video.mp4> [extra execute_plan.py args...]
#
# Why a script: launching with `pkill ... && setsid ...` in one command kills the
# launcher before setsid runs, and nothing starts - silently, with no log file. Kill
# stale processes as a separate step, then launch.

set -euo pipefail

PLAN="${1:?usage: run_sim.sh <plan.json> <video.mp4> [args...]}"
VIDEO="${2:?usage: run_sim.sh <plan.json> <video.mp4> [args...]}"
shift 2

LOG="${VIDEO%.mp4}.log"
mkdir -p "$(dirname "$VIDEO")"
rm -f "$VIDEO" "$LOG"

source ~/safety_filter/setup_behavior_env.sh >/dev/null 2>&1
exec python -u execute_plan.py --plan "$PLAN" --video "$VIDEO" "$@"
