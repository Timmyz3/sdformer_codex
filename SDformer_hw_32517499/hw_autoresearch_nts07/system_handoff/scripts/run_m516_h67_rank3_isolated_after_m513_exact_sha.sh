#!/usr/bin/env bash
set -euo pipefail

EXPECTED_RUNNER_SHA256="${M516_EXPECTED_RUNNER_SHA256:?set M516_EXPECTED_RUNNER_SHA256}"
RUNNER_PATH="$(readlink -f "$0")"
WORKTREE_ROOT="/root/private_data/work/m516_rank3_iso_20260827/SDformer"
REPO_ROOT="$WORKTREE_ROOT/SDformer"
ORIGINAL_ROOT="/root/private_data/work/sdformer_codex/SDformer"
M511_ROOT="/root/private_data/work/m511_capture_20260827/SDformer"
PYTHON_BIN="/opt/conda/envs/sdformerflow/bin/python"
PINNED_COMMIT="494593afa0ea81332ca21fcd68fdc9d6b72bbf1a"

BASE_REL="neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml"
CONFIG_REL="neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H67_ep35_M29_rank3_factor_atlif_ft5_20260822.yml"
CONFIG_RECEIPT_REL="${CONFIG_REL%.yml}.receipt.json"
CHECKPOINT_REL="neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth"
RESULT_REL="neuron_experiments/H9_bipolar_self_attention/results/m516_h67_ep35_rank3_factor_atlif_ft5_isolated_20260827"
M513_REL="hw_autoresearch_nts07/results/m513_h67_decoder_pgpr_tdr_fastkill_r1_20260827"
FINAL_PACKAGE_REL="hw_autoresearch_nts07/results/m516_h67_rank3_valid825_evidence_r1_20260827"

WATCHER_LOCK="/tmp/sdformer_m516_h67_rank3_isolated_20260827.lock"
ALGORITHM_LOCK="/tmp/sdformer_date_algorithm_evidence_queue_20260821.lock"
FACTORIAL_LOCK="/tmp/sdformer_date_fullres_factorial_controls_20260821.lock"
LOCAL5_LOCK="/tmp/sdformer_date_local5_same_parent_control_20260821.lock"
A800_LOCK="/tmp/sdformer_a800_training_global.lock"

die() {
    echo "M516_FAIL: $*" >&2
    exit 90
}

unset PYTHONOPTIMIZE PYTHONPATH LD_PRELOAD

runner_sha() {
    sha256sum "$RUNNER_PATH" | awk '{print $1}'
}

verify_repo_identity() {
    [[ -d "$WORKTREE_ROOT" && -d "$REPO_ROOT" ]] || die "isolated worktree/project missing"
    [[ "$(git -C "$WORKTREE_ROOT" rev-parse --show-toplevel)" == "$WORKTREE_ROOT" ]] \
        || die "isolated worktree root drift"
    [[ "$(git -C "$WORKTREE_ROOT" rev-parse HEAD)" == "$PINNED_COMMIT" ]] \
        || die "isolated commit drift"
    git -C "$WORKTREE_ROOT" diff --quiet || die "tracked worktree drift"
    git -C "$WORKTREE_ROOT" diff --cached --quiet || die "tracked index drift"
    [[ -z "$(git -C "$WORKTREE_ROOT" ls-files --others --exclude-standard -- \
        SDformer/neuron_experiments/H9_bipolar_self_attention/entrypoints \
        SDformer/neuron_experiments/H9_bipolar_self_attention/overlay \
        SDformer/third_party/SDformerFlow)" ]] || die "untracked executable/import shadow"
    [[ -L "$REPO_ROOT/data" ]] || die "isolated data link missing"
    [[ "$(readlink -f "$REPO_ROOT/data")" == "$ORIGINAL_ROOT/data" ]] \
        || die "isolated data target drift"
    [[ -L "$REPO_ROOT/$CHECKPOINT_REL" ]] || die "checkpoint link missing"
    [[ "$(readlink -f "$REPO_ROOT/$CHECKPOINT_REL")" == "$ORIGINAL_ROOT/$CHECKPOINT_REL" ]] \
        || die "checkpoint target drift"
}

verify_frozen_inputs() {
    cd "$REPO_ROOT"
    sha256sum --strict -c <<'EOF'
d9ee7e172f941a53ad1c031b0d5cdbbf7819f521c807e5bc54001a80c41b57f3  neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/atlif_ternary_psn/atlif_ternary_psn.py
5873063b98eb4a267afa6513d03b86621f3fb6a885b310b4c5569ef5448ae657  neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/atlif_ternary_psn/installer.py
f0e408c6bd136d7ce36b779881ca37a04de6f0cb6220701431b0a05b338f6d6b  neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/atlif_ternary_psn/__init__.py
172b3b8086cfe5c43bf9627fe92f947ca63148f9bbe8c50bca729b23c6273e68  neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/h9_load_audit.py
49c77538f2de2c54b709b05ae246da4cf7f36a147da990a03acb9e94a917446b  neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py
04f692c5bda6d1f88cdc932ce48f012767f22a2bb1ca161378971232f99c0684  neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_nts11_hardware_p0.py
5dbe838cabca7a1b47f7c9e3abde54b6a947bbbb39677fa432ef5dc936e475a6  neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_sops.py
ba555e897bf915319bc9976ce40b1b47abd5cd341472e3bfba0a6e68777a222a  neuron_experiments/H9_bipolar_self_attention/entrypoints/make_m29_h67_rank3_factor_config.py
55cabb82c64f59c6d30e83ac2b07e395f6c3162eb4f065314391ad21bc12621a  neuron_experiments/H9_bipolar_self_attention/entrypoints/verify_m29_h67_rank3_launch.py
331ec9b6ad62193ebe693bf930875b1af8db43ca1e4afac4e77793a567cfd714  neuron_experiments/H9_bipolar_self_attention/entrypoints/test_m29_atlif_temporal_factorization.py
b5aa4245c7237399ea49c65c2daae827e05120f760f4a72cf224dd7525dfdc29  neuron_experiments/H9_bipolar_self_attention/entrypoints/make_m29_h67_rank3_run_receipt.py
900cf8faa47cb4a2604ee0b500861b5d847cb074fe9c0ac6c09f010ebf955f3c  neuron_experiments/H9_bipolar_self_attention/entrypoints/run_h9_standard_valid825_eval.py
8be3f7bbffd75c4356d3abf5935679d80e15c1caefd307c19a727729659e6c49  neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml
4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158  neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth
919c79c61535eb499364ffe28fad3000441e25d1bddbf4fa9a0c27a78d4fdc10  data/Datasets/DSEC/saved_flow_data/sequence_lists/train_split_seq.csv
7f3dc2800653e12caca10379c51ee8e8988aaf6bb80c391224a454a5879325d0  data/Datasets/DSEC/saved_flow_data/sequence_lists/valid_split_seq.csv
EOF
}

verify_generated_config() {
    cd "$REPO_ROOT"
    sha256sum --strict -c <<'EOF'
bf0cb225afe6b29d21494f60cfda1be36eb6fe2bb47ac4ea578a8879b047b541  neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H67_ep35_M29_rank3_factor_atlif_ft5_20260822.yml
EOF
    "$PYTHON_BIN" - "$REPO_ROOT/$BASE_REL" "$REPO_ROOT/$CHECKPOINT_REL" \
        "$REPO_ROOT/$CONFIG_REL" "$REPO_ROOT/$CONFIG_RECEIPT_REL" <<'PY'
import hashlib, json, pathlib, sys
base, checkpoint, output, receipt_path = map(pathlib.Path, sys.argv[1:])
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
receipt = json.loads(receipt_path.read_text())
def require(condition, message):
    if not condition:
        raise RuntimeError(message)
require(receipt["schema"] == "m29_h67_rank3_factor_config_receipt_v1", "config receipt schema")
require(receipt["status"] == "READY_FLOATING_FACTOR_AMP_ACCURACY_SCREEN_NOT_INT8_NOT_SPEEDUP", "config receipt status")
require(pathlib.Path(receipt["base"]).resolve() == base.resolve(), "base path")
require(receipt["base_sha256"] == sha(base), "base SHA")
require(pathlib.Path(receipt["checkpoint"]).resolve() == checkpoint.resolve(), "checkpoint path")
require(receipt["checkpoint_sha256"] == sha(checkpoint), "checkpoint SHA")
require(pathlib.Path(receipt["output"]).resolve() == output.resolve(), "output path")
require(receipt["output_sha256"] == sha(output), "output SHA")
require(receipt["requested_rank"] == 3, "requested rank")
require(receipt["expected_t10_factorized_modules"] == 45, "T10 factorized count")
require(receipt["expected_t2_dense_fallback_modules"] == 60, "T2 fallback count")
require(receipt["headline_admitted"] is False, "headline boundary")
PY
}

RUNNER_SHA_START="$(runner_sha)"
[[ "$RUNNER_SHA_START" == "$EXPECTED_RUNNER_SHA256" ]] || die "runner SHA mismatch at start"
exec 209>"$WATCHER_LOCK"
flock -n 209 || die "another M516 watcher owns $WATCHER_LOCK"

verify_repo_identity
verify_frozen_inputs

# M513 is an independent capture/fast-kill chain queued behind the user's
# running 30-epoch job.  M516 starts only after that chain publishes and exits.
while pgrep -f '^m513_fastkill_watcher_tag ' >/dev/null; do
    sleep 60
done
[[ -f "$M511_ROOT/$M513_REL/RUN_COMPLETE.txt" ]] \
    || die "M513 completion missing after watcher exit"
grep -qx 'PASS_M513_EXACT_S10_DECODER_FASTKILL' \
    "$M511_ROOT/$M513_REL/RUN_COMPLETE.txt" || die "M513 completion status drift"
M513_IDENTITY_JSON="$("$PYTHON_BIN" - "$M511_ROOT/$M513_REL" <<'PY'
import hashlib, json, pathlib, re, sys

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def strict_json(path):
    def reject(token):
        raise RuntimeError("non-standard JSON token " + token)
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key " + key)
            out[key] = value
        return out
    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=reject)

directory = pathlib.Path(sys.argv[1])
require(directory.is_absolute(), "M513 directory must be absolute")
require(directory.is_dir() and not directory.is_symlink(), "M513 directory")
expected_files = {
    "m513_decoder_pgpr_tdr_fastkill.json", "RUN_COMPLETE.txt",
    "SHA256SUMS", "SHA256SUMS.seal.sha256",
}
actual_entries = {p.name for p in directory.iterdir()}
require(actual_entries == expected_files, "M513 exact top-level population")
for name in expected_files:
    p = directory / name
    require(p.is_file() and not p.is_symlink(), "M513 symlink/member " + name)
members = {}
for line in (directory / "SHA256SUMS").read_text().splitlines():
    fields = line.split("  ", 1)
    require(len(fields) == 2, "M513 member seal format")
    expected, name = fields
    require(name in {"m513_decoder_pgpr_tdr_fastkill.json", "RUN_COMPLETE.txt"}, "M513 sealed name")
    require(name not in members and re.fullmatch(r"[0-9a-f]{64}", expected), "M513 duplicate/hash")
    require(sha(directory / name) == expected, "M513 member mismatch " + name)
    members[name] = expected
require(set(members) == {"m513_decoder_pgpr_tdr_fastkill.json", "RUN_COMPLETE.txt"}, "M513 sealed set")
outer_fields = (directory / "SHA256SUMS.seal.sha256").read_text().strip().split("  ", 1)
require(len(outer_fields) == 2 and outer_fields[1] == "SHA256SUMS", "M513 outer format")
require(sha(directory / "SHA256SUMS") == outer_fields[0], "M513 outer mismatch")
require((directory / "RUN_COMPLETE.txt").read_text() == "PASS_M513_EXACT_S10_DECODER_FASTKILL\n", "M513 completion")
v = strict_json(directory / "m513_decoder_pgpr_tdr_fastkill.json")
require(v.get("schema") == "m513_h67_decoder_pgpr_tdr_fastkill_v1", "M513 schema")
require(v.get("status") == "PASS_EXACT_S10_DECODER_FASTKILL_NO_RTL_ADMISSION", "M513 status")
identity = v.get("identity", {})
require(identity.get("analyzer_sha256") == "9790f62d7a3e8fa4ca0ab98947bc6bfb49ae4720bbfb075ec75cebcd3cf7e299", "M513 analyzer")
require(identity.get("contract_sha256") == "e556743dd18804a7aba5be5b18f33823bbcd5e5be85d7715edcc43a4c314c28e", "M513 contract")
require(identity.get("payload_verifier_sha256") == "222d0402a57789671c975bac4a59a34a5188279b6b6a02319ddd26ad37c9ed1b", "M513 verifier")
require(identity.get("runner_sha256") == "788d674eb3df23f3af6cd8525b3a6471fd26596459e298ef8c9df7aa6369b7fa", "M513 runner")
for key in ("runner_final_seal_file_sha256", "capture_sha256sums_sha256", "payload_verify_sha256sums_sha256"):
    require(re.fullmatch(r"[0-9a-f]{64}", str(identity.get(key, ""))) is not None, "M513 upstream seal " + key)
require(v.get("decision", {}).get("new_performance_rtl_authorized") is False, "M513 RTL boundary")
for key in ("cycle_simulator_with_sram", "rtl", "vcs", "synopsys", "energy", "ppa", "system_speedup", "date_headline"):
    require(v.get("claim_boundary", {}).get(key) is False, "M513 claim " + key)
print(json.dumps({
    "directory": str(directory),
    "result_json_sha256": sha(directory / "m513_decoder_pgpr_tdr_fastkill.json"),
    "sha256sums_sha256": sha(directory / "SHA256SUMS"),
    "outer_seal_file_sha256": sha(directory / "SHA256SUMS.seal.sha256"),
    "upstream_identity": identity,
}, sort_keys=True))
PY
 )" || die "M513 sealed consumer failed"

cd "$REPO_ROOT"
"$PYTHON_BIN" -m unittest -q \
    neuron_experiments/H9_bipolar_self_attention/entrypoints/test_m29_atlif_temporal_factorization.py
"$PYTHON_BIN" \
    neuron_experiments/H9_bipolar_self_attention/entrypoints/make_m29_h67_rank3_factor_config.py \
    --base "$BASE_REL" --checkpoint "$CHECKPOINT_REL" --output "$CONFIG_REL" --force
verify_generated_config

exec 210>"$ALGORITHM_LOCK"
exec 211>"$FACTORIAL_LOCK"
exec 212>"$LOCAL5_LOCK"
exec 213>"$A800_LOCK"
while ! flock -n 210 || ! flock -n 211 || ! flock -n 212 || ! flock -n 213; do
    flock -u 210 || true
    flock -u 211 || true
    flock -u 212 || true
    flock -u 213 || true
    sleep 180
done

while true; do
    GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" \
        || die "nvidia-smi compute-app query failed"
    GPU_PIDS="$(printf '%s' "$GPU_PIDS" | tr -d '[:space:]')"
    [[ -z "$GPU_PIDS" ]] && break
    sleep 180
done

GPU_IDENTITY_JSON="$(nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total \
    --format=csv,noheader,nounits)" || die "nvidia-smi GPU identity query failed"
RUNNER_SHA_PRELAUNCH="$(runner_sha)"
[[ "$RUNNER_SHA_PRELAUNCH" == "$EXPECTED_RUNNER_SHA256" ]] || die "runner SHA drift before launch"
verify_repo_identity
verify_frozen_inputs
verify_generated_config
GPU_PIDS_AT_LAUNCH="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" \
    || die "nvidia-smi launch-race query failed"
GPU_PIDS_AT_LAUNCH="$(printf '%s' "$GPU_PIDS_AT_LAUNCH" | tr -d '[:space:]')"
[[ -z "$GPU_PIDS_AT_LAUNCH" ]] || die "GPU became busy after idle gate"

if [[ -e "$REPO_ROOT/$RESULT_REL" ]]; then
    die "M516 result directory already exists; refusing overwrite/resume"
fi
if [[ -e "$REPO_ROOT/$FINAL_PACKAGE_REL" ]]; then
    die "M516 final evidence package already exists; refusing overwrite"
fi
ATTEMPT_TAG="$(date -u +%Y%m%dT%H%M%SZ)_pid$$"
ATTEMPT_REL="hw_autoresearch_nts07/system_handoff/m516_receipts/$ATTEMPT_TAG"
PREFLIGHT_REL="$ATTEMPT_REL/preflight.json"
LAUNCH_RECEIPT_REL="$ATTEMPT_REL/launch_receipt.json"
POSTFLIGHT_RECEIPT_REL="$ATTEMPT_REL/postflight_receipt.json"
TRAIN_LOG_REL="$ATTEMPT_REL/train.log"
VALID_LOG_REL="$ATTEMPT_REL/valid825.log"
FACTOR_RECEIPT_REL="$ATTEMPT_REL/ep40_factor_checkpoint_receipt.json"
M513_RECEIPT_REL="$ATTEMPT_REL/m513_consumed_identity.json"
FINAL_RECEIPT_REL="$ATTEMPT_REL/final_valid825_receipt.json"
mkdir -p "$REPO_ROOT/$ATTEMPT_REL" "$REPO_ROOT/$RESULT_REL"
printf '%s\n' "$M513_IDENTITY_JSON" >"$REPO_ROOT/$M513_RECEIPT_REL"

CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" \
    neuron_experiments/H9_bipolar_self_attention/entrypoints/verify_m29_h67_rank3_launch.py \
    --config "$CONFIG_REL" --checkpoint "$CHECKPOINT_REL" \
    --receipt "${CONFIG_REL%.yml}.receipt.json" --output "$PREFLIGHT_REL"

"$PYTHON_BIN" \
    neuron_experiments/H9_bipolar_self_attention/entrypoints/make_m29_h67_rank3_run_receipt.py \
    --phase launch --config "$CONFIG_REL" --source-checkpoint "$CHECKPOINT_REL" \
    --preflight "$PREFLIGHT_REL" --result-dir "$RESULT_REL" \
    --train-log "$TRAIN_LOG_REL" --watcher-pid "$$" --output "$LAUNCH_RECEIPT_REL"

TRAIN_EXIT=0
CUDA_VISIBLE_DEVICES=0 SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PYTHON_BIN" -u \
    neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py \
    --config "$REPO_ROOT/$CONFIG_REL" --prev_runid "$REPO_ROOT/$CHECKPOINT_REL" \
    --save_path "$REPO_ROOT/$RESULT_REL/checkpoint_epoch{}.pth" --finetune 1 \
    >"$REPO_ROOT/$TRAIN_LOG_REL" 2>&1 || TRAIN_EXIT=$?

"$PYTHON_BIN" \
    neuron_experiments/H9_bipolar_self_attention/entrypoints/make_m29_h67_rank3_run_receipt.py \
    --phase postflight --config "$CONFIG_REL" --source-checkpoint "$CHECKPOINT_REL" \
    --preflight "$PREFLIGHT_REL" --result-dir "$RESULT_REL" \
    --train-log "$TRAIN_LOG_REL" --launch-receipt "$LAUNCH_RECEIPT_REL" \
    --exit-code "$TRAIN_EXIT" --watcher-pid "$$" --output "$POSTFLIGHT_RECEIPT_REL"
[[ "$TRAIN_EXIT" -eq 0 ]] || exit "$TRAIN_EXIT"

CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" - \
    "$REPO_ROOT/$CONFIG_REL" "$REPO_ROOT/$CHECKPOINT_REL" \
    "$REPO_ROOT/$RESULT_REL/checkpoint_epoch36.pth" \
    "$REPO_ROOT/$RESULT_REL/checkpoint_epoch40.pth" \
    "$REPO_ROOT/$RESULT_REL/checkpoint_epoch40_state_dict.pth" \
    "$REPO_ROOT/$FACTOR_RECEIPT_REL" <<'PY'
import hashlib, json, pathlib, sys
import torch

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

config_path, source_checkpoint_path, epoch36_path, checkpoint_path, state_path, output_path = map(pathlib.Path, sys.argv[1:])
for path in (config_path, epoch36_path, checkpoint_path, state_path):
    require(path.is_file() and not path.is_symlink(), "missing/symlinked factor-verifier input " + str(path))
require(source_checkpoint_path.is_symlink() and source_checkpoint_path.resolve().is_file(), "frozen source checkpoint link")
raw = torch.load(str(checkpoint_path), map_location="cpu")
require(isinstance(raw, dict) and isinstance(raw.get("model_state_dict"), dict), "ep40 model_state_dict container")
state = raw["model_state_dict"]
left = {key[:-len("temporal_factor_left")]: value for key, value in state.items() if key.endswith("temporal_factor_left")}
right = {key[:-len("temporal_factor_right")]: value for key, value in state.items() if key.endswith("temporal_factor_right")}
require(set(left) == set(right), "ep40 incomplete factor pairs")
require(len(left) == 45, "ep40 factor-pair count")
for prefix in sorted(left):
    require(tuple(left[prefix].shape) == (10, 3), "left factor shape " + prefix)
    require(tuple(right[prefix].shape) == (3, 10), "right factor shape " + prefix)
    require(torch.isfinite(left[prefix]).all().item(), "non-finite ep40 left factor " + prefix)
    require(torch.isfinite(right[prefix]).all().item(), "non-finite ep40 right factor " + prefix)

training_state = torch.load(str(state_path), map_location="cpu")
require(isinstance(training_state, dict), "ep40 training-state container")
require(int(training_state.get("epoch", -1)) == 4, "ep40 internal epoch index")
optimizer = training_state.get("optimizer")
require(isinstance(optimizer, dict) and len(optimizer.get("state", {})) > 0, "ep40 optimizer state")

entrypoints = config_path.resolve().parents[2] / "entrypoints"
sys.path.insert(0, str(entrypoints))
import profile_nts11_hardware_p0 as profiler
config, _ = profiler.load_config(config_path)
model = profiler.build_model(config, checkpoint_path, torch.device("cpu"))
audit = profiler.validate_h9_load_audit(model, config)
modules = [(name, module) for name, module in model.named_modules() if module.__class__.__name__ == "ATLIFTernaryPSN"]
factorized = [(name, module) for name, module in modules if int(getattr(module, "temporal_factor_rank", 0)) == 3]
fallback = [(name, module) for name, module in modules if int(getattr(module, "temporal_factor_requested_rank", 0)) == 3 and int(getattr(module, "temporal_factor_rank", 0)) == 0]
require(len(modules) == 105, "ep40 ATLIF module count")
require(len(factorized) == 45 and all(module.T == 10 for _, module in factorized), "ep40 T10 rank3 census")
require(len(fallback) == 60 and all(module.T == 2 for _, module in fallback), "ep40 T2 dense census")
require(all(module.temporal_factor_load_source == "checkpoint_factors" for _, module in factorized), "ep40 factor load source")
require(int(audit.get("missing_count", -1)) == 0 and int(audit.get("unexpected_count", -1)) == 0, "ep40 load audit")
model_prefixes = {name + "." for name, _ in factorized}
require(model_prefixes == set(left), "ep40 checkpoint/model factor prefix mismatch")

# Compare checkpoints emitted by the same training process.  This avoids a
# false update proof from CPU/GPU SVD sign or rounding differences.
raw36 = torch.load(str(epoch36_path), map_location="cpu")
require(isinstance(raw36, dict) and isinstance(raw36.get("model_state_dict"), dict), "ep36 model_state_dict container")
state36 = raw36["model_state_dict"]
left36 = {key[:-len("temporal_factor_left")]: value for key, value in state36.items() if key.endswith("temporal_factor_left")}
right36 = {key[:-len("temporal_factor_right")]: value for key, value in state36.items() if key.endswith("temporal_factor_right")}
require(set(left36) == model_prefixes and set(right36) == model_prefixes, "ep36/ep40 rank3 prefix mismatch")
changed_pairs = []
for prefix in sorted(model_prefixes):
    require(tuple(left36[prefix].shape) == (10, 3) and tuple(right36[prefix].shape) == (3, 10), "ep36 factor shape " + prefix)
    require(torch.isfinite(left36[prefix]).all().item() and torch.isfinite(right36[prefix]).all().item(), "non-finite ep36 factor " + prefix)
    left_changed = not torch.equal(left[prefix].detach().cpu(), left36[prefix].detach().cpu())
    right_changed = not torch.equal(right[prefix].detach().cpu(), right36[prefix].detach().cpu())
    if left_changed or right_changed:
        changed_pairs.append(prefix)
require(len(changed_pairs) == 45, "not all rank3 factor pairs changed between ep36 and ep40")

value = {
    "schema": "m516_ep40_factor_checkpoint_receipt_v1",
    "status": "PASS_EP40_CONTAINS_45_T10_RANK3_FACTOR_PAIRS__NOT_QUANTIZED_NOT_HARDWARE_ADMISSION",
    "config_sha256": sha(config_path),
    "source_checkpoint_sha256": sha(source_checkpoint_path),
    "checkpoint_epoch36_sha256": sha(epoch36_path),
    "checkpoint_sha256": sha(checkpoint_path),
    "training_state_sha256": sha(state_path),
    "checkpoint_factor_pair_count": len(left),
    "left_shape": [10, 3],
    "right_shape": [3, 10],
    "t10_rank3_modules": len(factorized),
    "t2_dense_fallback_modules": len(fallback),
    "factor_load_source": "checkpoint_factors",
    "missing_count": int(audit["missing_count"]),
    "unexpected_count": int(audit["unexpected_count"]),
    "optimizer_state_nonempty": True,
    "internal_epoch_index": 4,
    "factor_pairs_changed_epoch36_to_epoch40": len(changed_pairs),
    "factor_values_proven_changed_epoch36_to_epoch40": True,
    "accuracy_hardware_admitted": False,
}
with output_path.open("x") as f:
    json.dump(value, f, indent=2, sort_keys=True)
    f.write("\n")
PY

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -u \
    neuron_experiments/H9_bipolar_self_attention/entrypoints/run_h9_standard_valid825_eval.py \
    --config "$REPO_ROOT/$CONFIG_REL" --run-dir "$REPO_ROOT/$RESULT_REL" \
    --epoch 40 --ranking-mode aee >"$REPO_ROOT/$VALID_LOG_REL" 2>&1

RUNNER_SHA_END="$(runner_sha)"
[[ "$RUNNER_SHA_END" == "$EXPECTED_RUNNER_SHA256" ]] || die "runner SHA drift before final receipt"
"$PYTHON_BIN" - \
    "$REPO_ROOT/$CONFIG_REL" "$REPO_ROOT/$RESULT_REL/checkpoint_epoch40.pth" \
    "$REPO_ROOT/$RESULT_REL/standard_valid825/epoch40/spike_profile.json" \
    "$REPO_ROOT/$POSTFLIGHT_RECEIPT_REL" "$REPO_ROOT/$FACTOR_RECEIPT_REL" \
    "$REPO_ROOT/$M513_RECEIPT_REL" "$REPO_ROOT/$FINAL_RECEIPT_REL" \
    "$RUNNER_SHA_START" "$RUNNER_SHA_PRELAUNCH" "$RUNNER_SHA_END" \
    "$GPU_IDENTITY_JSON" <<'PY'
import hashlib, json, math, pathlib, sys

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def ident(path):
    path = pathlib.Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}

config, checkpoint, profile_path, postflight_path, factor_path, m513_path, output_path = map(pathlib.Path, sys.argv[1:8])
runner_start, runner_prelaunch, runner_end, gpu_identity = sys.argv[8:12]
profile = json.loads(profile_path.read_text())
postflight = json.loads(postflight_path.read_text())
factor = json.loads(factor_path.read_text())
m513 = json.loads(m513_path.read_text())
require(postflight.get("status") == "PASS_TRAIN_EXIT_AND_EPOCH36_TO40_PRESENT_NOT_ACCURACY_RESULT", "postflight status")
require(postflight.get("complete") is True, "postflight complete")
require(factor.get("status") == "PASS_EP40_CONTAINS_45_T10_RANK3_FACTOR_PAIRS__NOT_QUANTIZED_NOT_HARDWARE_ADMISSION", "factor receipt")
require(int(factor.get("checkpoint_factor_pair_count", -1)) == 45, "factor count")
require(int(factor.get("t10_rank3_modules", -1)) == 45 and int(factor.get("t2_dense_fallback_modules", -1)) == 60, "factor module census")
require(factor.get("factor_values_proven_changed_epoch36_to_epoch40") is True, "factor update proof")
require(int(factor.get("factor_pairs_changed_epoch36_to_epoch40", -1)) == 45, "factor update census")
require(int(profile.get("samples", -1)) == 825, "valid825 samples")
require(int(profile.get("checkpoint_load_audit", {}).get("missing_count", -1)) == 0, "valid825 missing")
require(int(profile.get("checkpoint_load_audit", {}).get("unexpected_count", -1)) == 0, "valid825 unexpected")
require(int(profile.get("module_counts", {}).get("ATLIFTernaryPSN", -1)) == 105, "valid825 ATLIF count")
require(int(profile.get("module_counts", {}).get("ShiftmaxAttention", -1)) == 12, "valid825 attention count")
required_metrics = ("AEE", "AAE", "AAE_Benchmark", "DSEC_Fl")
metrics = profile.get("metrics", {})
for key in required_metrics:
    value = float(metrics.get(key, float("nan")))
    require(math.isfinite(value) and value >= 0.0, "non-finite/negative metric " + key)
require(runner_start == runner_prelaunch == runner_end, "actual runner SHA mismatch")
require(len(runner_start) == 64, "runner SHA format")
require(isinstance(m513.get("upstream_identity"), dict), "M513 identity propagation")
value = {
    "schema": "m516_h67_rank3_isolated_valid825_receipt_v1",
    "status": "PASS_TRAIN_AND_VALID825_MEASUREMENT__HARDWARE_ADMISSION_REQUIRES_METRIC_GATE",
    "identity": {
        "runner_sha256_start": runner_start,
        "runner_sha256_prelaunch": runner_prelaunch,
        "runner_sha256_end": runner_end,
        "config": ident(config),
        "checkpoint_epoch40": ident(checkpoint),
        "spike_profile": ident(profile_path),
        "postflight_receipt": ident(postflight_path),
        "factor_checkpoint_receipt": ident(factor_path),
        "m513_consumed_identity": ident(m513_path),
        "gpu_identity": gpu_identity,
    },
    "metrics": {key: float(metrics[key]) for key in required_metrics},
    "samples": profile["samples"],
    "module_counts": profile["module_counts"],
    "checkpoint_load_audit": profile["checkpoint_load_audit"],
    "claim_boundary": {
        "five_epoch_run_completed": True,
        "ep40_factor_checkpoint_emitted": True,
        "factor_values_proven_changed_epoch36_to_epoch40": True,
        "valid825_measurement_complete": True,
        "accuracy_hardware_admitted": False,
        "cycle_speedup": False,
        "energy": False,
        "paper_ppa_ready": False,
        "date_headline": False,
    },
}
with output_path.open("x") as f:
    json.dump(value, f, indent=2, sort_keys=True)
    f.write("\n")
PY

verify_repo_identity
verify_frozen_inputs
verify_generated_config
[[ "$(runner_sha)" == "$RUNNER_SHA_END" ]] || die "runner SHA drift after completion"

FINAL_PARENT="$REPO_ROOT/$(dirname "$FINAL_PACKAGE_REL")"
FINAL_PATH="$REPO_ROOT/$FINAL_PACKAGE_REL"
mkdir -p "$FINAL_PARENT"
FINAL_STAGING="$(mktemp -d "$FINAL_PARENT/.m516_rank3_valid825_evidence.staging.XXXXXX")"
cp "$REPO_ROOT/$FINAL_RECEIPT_REL" "$FINAL_STAGING/final_valid825_receipt.json"
cp "$REPO_ROOT/$FACTOR_RECEIPT_REL" "$FINAL_STAGING/ep40_factor_checkpoint_receipt.json"
cp "$REPO_ROOT/$M513_RECEIPT_REL" "$FINAL_STAGING/m513_consumed_identity.json"
cp "$REPO_ROOT/$POSTFLIGHT_RECEIPT_REL" "$FINAL_STAGING/training_postflight_receipt.json"
printf '%s\n' 'PASS_M516_RANK3_TRAIN_AND_VALID825_MEASUREMENT_NOT_HARDWARE_ADMISSION' \
    >"$FINAL_STAGING/RUN_COMPLETE.txt"
"$PYTHON_BIN" - "$FINAL_STAGING" <<'PY'
import hashlib, pathlib, sys

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

directory = pathlib.Path(sys.argv[1]).resolve()
expected = {
    "final_valid825_receipt.json", "ep40_factor_checkpoint_receipt.json",
    "m513_consumed_identity.json", "training_postflight_receipt.json",
    "RUN_COMPLETE.txt",
}
actual = {p.name for p in directory.iterdir() if p.is_file()}
require(actual == expected, "M516 staging member population")
for name in expected:
    require(not (directory / name).is_symlink(), "M516 staging symlink " + name)
seal = directory / "SHA256SUMS"
seal.write_text("".join("{}  {}\n".format(sha(directory / name), name) for name in sorted(expected)))
outer = directory / "SHA256SUMS.seal.sha256"
outer.write_text("{}  SHA256SUMS\n".format(sha(seal)))
expected_full = expected | {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
actual_full = {p.name for p in directory.iterdir()}
require(actual_full == expected_full, "M516 sealed staging population")
for name in expected_full:
    require((directory / name).is_file() and not (directory / name).is_symlink(), "M516 sealed staging member " + name)
members = {}
for line in seal.read_text().splitlines():
    fields = line.split("  ", 1)
    require(len(fields) == 2, "M516 staging member seal format")
    digest, name = fields
    require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "M516 staging member digest")
    require(name in expected and name not in members, "M516 staging sealed member")
    require(sha(directory / name) == digest, "M516 staging member SHA " + name)
    members[name] = digest
require(set(members) == expected, "M516 staging sealed set")
outer_fields = outer.read_text().strip().split("  ", 1)
require(len(outer_fields) == 2 and outer_fields[1] == "SHA256SUMS", "M516 staging outer format")
require(sha(seal) == outer_fields[0], "M516 staging outer seal")
require((directory / "RUN_COMPLETE.txt").read_text() == "PASS_M516_RANK3_TRAIN_AND_VALID825_MEASUREMENT_NOT_HARDWARE_ADMISSION\n", "M516 staging completion")
PY
[[ ! -e "$FINAL_PATH" && ! -L "$FINAL_PATH" ]] \
    || die "M516 final package appeared before prepublish verification"
"$PYTHON_BIN" - "$FINAL_STAGING" <<'PY'
import hashlib, pathlib, sys
def require(condition, message):
    if not condition:
        raise RuntimeError(message)
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
directory = pathlib.Path(sys.argv[1]).resolve()
expected = {
    "final_valid825_receipt.json", "ep40_factor_checkpoint_receipt.json",
    "m513_consumed_identity.json", "training_postflight_receipt.json",
    "RUN_COMPLETE.txt", "SHA256SUMS", "SHA256SUMS.seal.sha256",
}
actual = {p.name for p in directory.iterdir()}
require(actual == expected and all((directory / n).is_file() and not (directory / n).is_symlink() for n in expected), "M516 prepublish population")
members = {}
for line in (directory / "SHA256SUMS").read_text().splitlines():
    digest, name = line.split("  ", 1)
    require(name not in members and name in expected - {"SHA256SUMS", "SHA256SUMS.seal.sha256"}, "M516 prepublish member")
    require(sha(directory / name) == digest, "M516 prepublish member SHA")
    members[name] = digest
require(set(members) == expected - {"SHA256SUMS", "SHA256SUMS.seal.sha256"}, "M516 prepublish sealed set")
outer_digest, outer_name = (directory / "SHA256SUMS.seal.sha256").read_text().strip().split("  ", 1)
require(outer_name == "SHA256SUMS" and sha(directory / "SHA256SUMS") == outer_digest, "M516 prepublish outer seal")
PY
[[ ! -e "$FINAL_PATH" && ! -L "$FINAL_PATH" ]] \
    || die "M516 final package appeared before atomic publish"
# This same-parent rename is deliberately the final command.  Before it, only
# a hidden fully verified staging directory exists.  After it, the canonical
# package is already fully verified; there is no post-publish failure window.
mv -T "$FINAL_STAGING" "$FINAL_PATH"
