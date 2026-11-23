#!/usr/bin/env bash
# Guarded test runner: pytest inside Docker with a time limit (TEST_TIMEOUT seconds, default 2700: the whole suite takes about 20 minutes).
cd "$(dirname "$0")/.." || exit 1
limit="${TEST_TIMEOUT:-2700}"
if ! [[ "$limit" =~ ^[1-9][0-9]*$ ]]; then   # 0 would mean no limit at all
  echo "TEST_TIMEOUT must be a whole number of seconds above 0 (got '$limit'); using 2700" >&2
  limit=2700
fi
exec docker compose --profile test run --rm tests timeout "$limit" pytest -q "$@"
