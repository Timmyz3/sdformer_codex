#!/usr/bin/env bash
# Intentionally non-runnable template; root replaces every literal placeholder.
# No loop/queue. Preflight is default; use fresh output for each invocation.
exit 2

python -B run_native_lr.py \
  --runtime-manifest /REPLACE/runtime.json --runtime-manifest-sha256 REPLACE_SHA \
  --membership /REPLACE/manifest_v1.json --membership-sha256 REPLACE_SHA \
  --bn-buffers /REPLACE/new_bn.npz --bn-buffers-sha256 REPLACE_SHA \
  --bn-receipt /REPLACE/new_bn_receipt.json --bn-receipt-sha256 REPLACE_SHA \
  --baseline-result /REPLACE/diversity/recover/native/results.json --baseline-result-sha256 REPLACE_SHA \
  --runner-checks /REPLACE/native_lr_stability/runner_cpu_results.json \
  --mode smoke --output-dir /REPLACE/fresh_smoke --preflight-output /REPLACE/fresh_smoke_preflight.json \
  --wall-minutes 30

# After this new smoke is actually executed and independently checked, use the
# same source arguments, --mode recover and the one --smoke-receipt PATH.
# Actual execution uses --execute with fresh --output-dir; it does not accept
# recovered initial weights or an optimizer-resume option.
