#!/usr/bin/env bash
# Weekly run: refresh data, retrain, predict, simulate, log the results.
# Intended to run from cron on Monday morning (US Eastern) -- after the
# weekend, before the US open. The pipeline scores from last Friday's
# close, so the output is the new week's trade list.
#
# Usage: ./scripts/weekly_run.sh
# Logs go to logs/weekly_YYYY-MM-DD.log.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG="logs/weekly_$(date +%Y-%m-%d).log"
{
  echo "=== Weekly run: $(date) ==="
  # --force-refresh re-downloads prices so the run uses the latest week.
  # Market-cap universe: trade the largest, most liquid names.
  .venv/bin/python main.py \
      --tickers 50 --years 5 \
      --universe market-cap \
      --force-refresh
  echo "=== Finished: $(date) ==="
} >> "$LOG" 2>&1
echo "Weekly run finished. Log: $LOG"
