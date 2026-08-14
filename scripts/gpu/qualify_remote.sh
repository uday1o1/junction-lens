#!/bin/sh
set -eu

JL_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
JL_PROFILE=runtime-cuda

usage() {
  echo "Usage: ./scripts/gpu/qualify_remote.sh [--profile m0.3|runtime-cuda|core|full-v1]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile)
      [ "$#" -ge 2 ] || { usage; exit 2; }
      JL_PROFILE=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

case "$JL_PROFILE" in
  m0.3|runtime-cuda|core|full-v1) ;;
  *) echo "Invalid qualification profile: $JL_PROFILE" >&2; exit 2 ;;
esac

: "${JUNCTIONLENS_GPU_HOST:?Set JUNCTIONLENS_GPU_HOST to an SSH alias}"
JL_HOST=$JUNCTIONLENS_GPU_HOST
JL_REMOTE_ROOT=${JUNCTIONLENS_REMOTE_ROOT:-.junctionlens/qualification}
JL_REMOTE_DATA_ROOT=${JUNCTIONLENS_REMOTE_DATA_ROOT:-}
JL_GPU_UUID=${JUNCTIONLENS_GPU_UUID:-}
JL_POLL_SECONDS=${JUNCTIONLENS_GPU_POLL_SECONDS:-15}
JL_TIMEOUT_SECONDS=${JUNCTIONLENS_GPU_TIMEOUT_SECONDS:-43200}

case "$JL_HOST" in
  -*|*[!A-Za-z0-9_.:@\[\]-]*) echo "JUNCTIONLENS_GPU_HOST is not a safe SSH destination" >&2; exit 2 ;;
esac
case "$JL_REMOTE_ROOT" in
  ""|/*|*..*|*[!A-Za-z0-9_./-]*) echo "JUNCTIONLENS_REMOTE_ROOT must be a safe path below the remote home" >&2; exit 2 ;;
esac
case "$JL_POLL_SECONDS:$JL_TIMEOUT_SECONDS" in
  *[!0-9:]*|:*|*:) echo "GPU polling controls must be positive integers" >&2; exit 2 ;;
esac

for JL_TOOL in git ssh scp python3; do
  command -v "$JL_TOOL" >/dev/null 2>&1 || { echo "$JL_TOOL is required" >&2; exit 2; }
done

JL_TEMP=$(mktemp -d "${TMPDIR:-/tmp}/junctionlens-gpu.XXXXXX")
cleanup() {
  rm -rf -- "$JL_TEMP"
}
trap cleanup EXIT HUP INT TERM

python3 "$JL_ROOT/scripts/gpu/source_bundle.py" create \
  --root "$JL_ROOT" \
  --archive "$JL_TEMP/source.tar" \
  --manifest "$JL_TEMP/source-manifest.json" >/dev/null

set -- python3 "$JL_ROOT/scripts/gpu/source_bundle.py" make-config \
  --manifest "$JL_TEMP/source-manifest.json" \
  --output "$JL_TEMP/remote-config.json" \
  --profile "$JL_PROFILE"
if [ -n "$JL_REMOTE_DATA_ROOT" ]; then
  set -- "$@" --remote-data-root "$JL_REMOTE_DATA_ROOT"
fi
if [ -n "$JL_GPU_UUID" ]; then
  set -- "$@" --gpu-uuid "$JL_GPU_UUID"
fi
"$@" >/dev/null

JL_DIGEST=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["content_sha256"])' "$JL_TEMP/source-manifest.json")
case "$JL_DIGEST" in
  *[!0-9a-f]*) echo "Source bundle digest is invalid" >&2; exit 1 ;;
esac
[ "${#JL_DIGEST}" -eq 64 ] || { echo "Source bundle digest length is invalid" >&2; exit 1; }
JL_BUNDLE_REL=$JL_REMOTE_ROOT/bundles/$JL_DIGEST
JL_INCOMING_REL=$JL_BUNDLE_REL/incoming
JL_SOURCE_REL=$JL_BUNDLE_REL/source
JL_RESULT_REL=$JL_BUNDLE_REL/results
JL_SESSION=jl-${JL_DIGEST%${JL_DIGEST#????????????}}

ssh -- "$JL_HOST" "mkdir -p -- \"\$HOME/$JL_INCOMING_REL\""
scp -- \
  "$JL_TEMP/source.tar" \
  "$JL_TEMP/source-manifest.json" \
  "$JL_TEMP/remote-config.json" \
  "$JL_ROOT/scripts/gpu/source_bundle.py" \
  "$JL_HOST:$JL_INCOMING_REL/"

ssh -- "$JL_HOST" \
  "if [ ! -d \"\$HOME/$JL_SOURCE_REL\" ]; then python3 \"\$HOME/$JL_INCOMING_REL/source_bundle.py\" verify-extract --archive \"\$HOME/$JL_INCOMING_REL/source.tar\" --manifest \"\$HOME/$JL_INCOMING_REL/source-manifest.json\" --target \"\$HOME/$JL_SOURCE_REL\" >/dev/null; fi"

JL_REMOTE_COMMAND="cd \$HOME/$JL_SOURCE_REL && exec ./scripts/gpu/run_remote_qualification.sh \$HOME/$JL_INCOMING_REL/remote-config.json \$HOME/$JL_RESULT_REL > \$HOME/$JL_BUNDLE_REL/runner.log 2>&1"
ssh -- "$JL_HOST" \
  "if [ -f \"\$HOME/$JL_RESULT_REL/status.json\" ]; then exit 0; elif command -v tmux >/dev/null 2>&1; then tmux has-session -t $JL_SESSION 2>/dev/null || tmux new-session -d -s $JL_SESSION \"$JL_REMOTE_COMMAND\"; elif [ -f \"\$HOME/$JL_BUNDLE_REL/runner.pid\" ] && kill -0 \"\$(cat \"\$HOME/$JL_BUNDLE_REL/runner.pid\")\" 2>/dev/null; then exit 0; else nohup sh -c '$JL_REMOTE_COMMAND' </dev/null >/dev/null 2>&1 & echo \$! > \"\$HOME/$JL_BUNDLE_REL/runner.pid\"; fi"

JL_STARTED=$(date +%s)
JL_REMOTE_STATUS=RUNNING
while [ "$JL_REMOTE_STATUS" = RUNNING ]; do
  JL_STATUS_JSON=$(ssh -- "$JL_HOST" "if [ -f \"\$HOME/$JL_RESULT_REL/status.json\" ]; then cat \"\$HOME/$JL_RESULT_REL/status.json\"; else echo RUNNING; fi")
  case "$JL_STATUS_JSON" in
    RUNNING) JL_REMOTE_STATUS=RUNNING ;;
    *'"status":"PASSED"'*) JL_REMOTE_STATUS=PASSED ;;
    *'"status":"BLOCKED"'*) JL_REMOTE_STATUS=BLOCKED ;;
    *'"status":"FAILED"'*) JL_REMOTE_STATUS=FAILED ;;
    *) echo "Remote runner returned an invalid status payload" >&2; exit 1 ;;
  esac
  [ "$JL_REMOTE_STATUS" = RUNNING ] || break
  JL_NOW=$(date +%s)
  if [ $((JL_NOW - JL_STARTED)) -ge "$JL_TIMEOUT_SECONDS" ]; then
    echo "Remote qualification timed out without a terminal result" >&2
    exit 1
  fi
  sleep "$JL_POLL_SECONDS"
done

JL_LOCAL_PARENT=$JL_ROOT/.junctionlens/qualification
mkdir -p -- "$JL_LOCAL_PARENT"
JL_TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
JL_LOCAL_RESULT=$JL_LOCAL_PARENT/$JL_DIGEST-$JL_TIMESTAMP
[ ! -e "$JL_LOCAL_RESULT" ] || { echo "Local result path already exists" >&2; exit 1; }
scp -r -- "$JL_HOST:$JL_RESULT_REL" "$JL_LOCAL_RESULT"

if command -v sha256sum >/dev/null 2>&1; then
  (cd "$JL_LOCAL_RESULT" && sha256sum -c SHA256SUMS)
else
  (cd "$JL_LOCAL_RESULT" && shasum -a 256 -c SHA256SUMS)
fi

JL_LOCAL_STATUS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "$JL_LOCAL_RESULT/status.json")
echo "Remote qualification result: $JL_LOCAL_RESULT"
[ "$JL_LOCAL_STATUS" = PASSED ] || exit 1
