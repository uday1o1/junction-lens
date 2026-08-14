#!/bin/sh
set -eu

JL_SOURCE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)

if [ "$#" -ne 2 ]; then
  echo "Usage: run_remote_qualification.sh CONFIG_JSON RESULT_ROOT" >&2
  exit 2
fi

JL_CONFIG=$1
JL_RESULT_ROOT=$2

exec python3 "$JL_SOURCE_ROOT/scripts/gpu/remote_runner.py" \
  --source-root "$JL_SOURCE_ROOT" \
  --config "$JL_CONFIG" \
  --result-root "$JL_RESULT_ROOT"
