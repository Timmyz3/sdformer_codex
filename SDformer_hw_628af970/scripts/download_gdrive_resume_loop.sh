#!/usr/bin/env bash
set -euo pipefail

FILE_ID="${1:?file id required}"
OUTPUT="${2:?output path required}"
MAX_RETRIES="${3:-200}"

mkdir -p "$(dirname "$OUTPUT")"
LOG="$(mktemp)"

attempt=1
while (( attempt <= MAX_RETRIES )); do
  echo "[gdrive] attempt ${attempt}/${MAX_RETRIES}: ${OUTPUT}"
  set +e
  gdown --continue "${FILE_ID}" -O "${OUTPUT}" >"${LOG}" 2>&1
  status=$?
  set -e
  cat "${LOG}"

  if [[ "$status" -eq 0 ]]; then
    size=$(stat -c%s "${OUTPUT}")
    if python3 - <<PY
import zipfile
import sys
path = "${OUTPUT}"
try:
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
    if bad is not None:
        raise RuntimeError(f"corrupt member: {bad}")
except Exception as exc:
    print(f"[gdrive] invalid archive: {exc}", file=sys.stderr)
    sys.exit(1)
PY
    then
      echo "[gdrive] complete size=${size}"
      rm -f "${LOG}"
      exit 0
    fi
    echo "[gdrive] archive validation failed; retrying" >&2
    rm -f "${OUTPUT}" "${OUTPUT}"*.part
  fi

  if grep -qi "too many users" "${LOG}"; then
    echo "[gdrive] quota hit; sleep 600s then resume" >&2
    sleep 600
  else
    echo "[gdrive] failed; sleep 60s then resume" >&2
    sleep 60
  fi
  attempt=$((attempt + 1))
done

rm -f "${LOG}"
echo "[gdrive] exhausted retries" >&2
exit 1