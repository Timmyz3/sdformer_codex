#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent read-only M1120 candidate identity hammer."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import stat
import subprocess
import sys


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CONTRACT = HW / "contracts/m1120_motion_ep24_candidate_checkpoint_identity_r1_20260830.json"
CONTRACT_ID = ("9c8a2ae96a015d4a8ef82394caa6a7e250e7665208ce82e896f064f67f78a259",
               "9334cbfc1081976ee6443e3cbc03cfa538352e993be12a4cfa4e4c03f6143294",
               "dba10344a47ec8c57d6b667c7df7e81ecb983bf5dff98cbe1439f5685f69923c")
RECEIPT = HW / "reviews/m1120_motion_ep24_candidate_checkpoint_identity_receipt_r1_20260830"
RECEIPT_ID = ("4623c85d37d51a896eea86bc1c86de2cfb00c9ac99068b10460e7ab5047621da",
              "6e37f8e07d47b05ebb108e3c318d887e6242711aa1169c2227e15036fa0efc97",
              "5a5df419186ef60ae7b161ffdd5a387222c88f37ba2c8809a85be4414dcb2e6e")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
SOCKET = Path("/tmp/codex_m714_ssh.MFUzxMzZ/control.sock")
REMOTE = "root@ssh.sd5ai.scnet.cn"
PORT = "10037"
CKPT = "/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/date_two_contribution_full30_20260826/c12_binary_motion_ttx/checkpoint_epoch24.pth"
CONFIG = "/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_two_contrib_c12_binary_motion_ttx_nb0ep29_ft30_20260826.yml"
LOG = "/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/date_two_contribution_full30_20260826/c12_binary_motion_ttx/train.log"
OUT = HERE / "mechanical_checks.json"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path, expected):
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink() and
            sha(path) == expected, "identity drift: " + str(path))


def no_duplicate(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise RuntimeError("nonfinite JSON: " + value)


def strict_load_text(text):
    return json.loads(text, object_pairs_hook=no_duplicate, parse_constant=reject_constant,
                      parse_float=Decimal)


def verify_contract():
    side = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    regular(CONTRACT, CONTRACT_ID[0]); regular(side, CONTRACT_ID[1]); regular(outer, CONTRACT_ID[2])
    require(side.read_text(encoding="utf-8") == CONTRACT_ID[0] +
            "  m1120_motion_ep24_candidate_checkpoint_identity_r1_20260830.json\n",
            "contract side content")
    require(outer.read_text(encoding="utf-8") == CONTRACT_ID[1] +
            "  m1120_motion_ep24_candidate_checkpoint_identity_r1_20260830.json.sha256\n",
            "contract outer content")
    return strict_load_text(CONTRACT.read_text(encoding="utf-8"))


def verify_receipt():
    receipt, manifest, outer = RECEIPT / "receipt.md", RECEIPT / "SHA256SUMS", RECEIPT / "SHA256SUMS.seal.sha256"
    regular(receipt, RECEIPT_ID[0]); regular(manifest, RECEIPT_ID[1]); regular(outer, RECEIPT_ID[2])
    require(manifest.read_text(encoding="utf-8") == RECEIPT_ID[0] + "  receipt.md\n",
            "receipt manifest content")
    require(outer.read_text(encoding="utf-8") == RECEIPT_ID[1] + "  SHA256SUMS\n",
            "receipt outer content")
    actual = {path.name for path in RECEIPT.iterdir() if path.is_file() and
              path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == {"receipt.md"} and not any(path.is_symlink() for path in RECEIPT.iterdir()),
            "receipt exact flat coverage")


def exact(value, keys, label):
    require(isinstance(value, dict) and set(value) == set(keys), label + " keys")


def validate(value):
    exact(value, {"schema", "status", "observed_at", "remote_host", "run", "checkpoint",
                  "configuration", "validation_evidence", "training_state", "hardware_policy",
                  "paper_claims", "protected_file"}, "contract")
    require(value["schema"] == "m1120_motion_candidate_checkpoint_identity_r1" and
            value["status"] == "CANDIDATE_ONLY_NOT_FINAL_NOT_HARDWARE_ADMITTED" and
            value["remote_host"] == "ssh.sd5ai.scnet.cn:10037", "top identity")
    ckpt = value["checkpoint"]
    exact(ckpt, {"epoch", "absolute_path", "basename", "size_bytes", "mtime_epoch", "mtime",
                 "sha256", "stability_check"}, "checkpoint")
    require(ckpt["epoch"] == 24 and ckpt["absolute_path"] == CKPT and
            ckpt["basename"] == "checkpoint_epoch24.pth" and
            ckpt["size_bytes"] == 225504447 and ckpt["mtime_epoch"] == 1788037440 and
            ckpt["mtime"] == "2026-08-30 05:04:00.000000000 +0800" and
            ckpt["sha256"] == "1e55900cd0bb4e411d09a5e4cd7bd56c08c60874a1e4868f6494d18b3e691e32" and
            ckpt["stability_check"] == {"observations": 2, "interval_sec_at_least": 20,
                                         "size_unchanged": True, "mtime_unchanged": True},
            "checkpoint identity")
    require(value["configuration"] == {"absolute_path": CONFIG,
            "sha256": "c7b5b994cb9f9a43478f3cb7c09e52a7aecf529fcd6a590f982a291e9eeed955"},
            "configuration identity")
    val = value["validation_evidence"]
    exact(val, {"source", "absolute_log_path", "log_observed_size_bytes",
                "log_observed_mtime_epoch", "epoch23_loss", "epoch24_loss",
                "epoch24_valid_time_sec", "epoch24_valid_step_time_sec",
                "epoch24_valid_samples_per_sec", "epoch24_loss_minus_epoch23_loss",
                "epoch24_better_than_epoch23", "valid825"}, "validation")
    require(val["source"] == "remote train.log only" and val["absolute_log_path"] == LOG and
            val["epoch23_loss"] == Decimal("0.8988656344867888") and
            val["epoch24_loss"] == Decimal("0.8975050449371338") and
            val["epoch24_valid_time_sec"] == Decimal("15.40") and
            val["epoch24_valid_step_time_sec"] == Decimal("0.7335") and
            val["epoch24_valid_samples_per_sec"] == Decimal("2.7268") and
            val["epoch24_better_than_epoch23"] is True and val["valid825"] is False,
            "validation boundary")
    state = value["training_state"]
    require(state == {"running": True, "next_epoch_started": 25,
                      "predeclared_saved_epochs": [9, 14, 19, 24, 29]}, "training state")
    hardware = value["hardware_policy"]
    require(hardware["candidate_only"] is True and
            hardware["best_saved_candidate_observed_so_far"] is True and
            hardware["final_checkpoint_selected"] is False and
            hardware["checkpoint_downloaded"] is False and
            hardware["hardware_replay_started_for_this_candidate"] is False and
            hardware["intermediate_full_hardware_replay_authorized"] is False,
            "hardware policy")
    require(value["paper_claims"] == {"final_checkpoint": False, "hardware_speedup": False,
            "system_speedup": False, "energy": False, "accuracy": False}, "paper claims")
    require(value["protected_file"] == {"path": "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md",
            "sha256": DOCS359_SHA}, "protected identity")
    return True


contract = verify_contract(); verify_receipt(); require(validate(contract), "canonical contract")
regular(DOCS359, DOCS359_SHA)


def rejected(function):
    try:
        function()
    except RuntimeError:
        return True
    return False


mutations = {}
for name, path, replacement in (
        ("final_checkpoint", ("hardware_policy", "final_checkpoint_selected"), True),
        ("valid825", ("validation_evidence", "valid825"), True),
        ("hardware_replay", ("hardware_policy", "hardware_replay_started_for_this_candidate"), True),
        ("intermediate_replay_authority", ("hardware_policy", "intermediate_full_hardware_replay_authorized"), True),
        ("hardware_speedup", ("paper_claims", "hardware_speedup"), True),
        ("system_speedup", ("paper_claims", "system_speedup"), True),
        ("accuracy", ("paper_claims", "accuracy"), True),
        ("checkpoint_downloaded", ("hardware_policy", "checkpoint_downloaded"), True)):
    forged = copy.deepcopy(contract); forged[path[0]][path[1]] = replacement
    mutations[name] = rejected(lambda forged=forged: validate(forged))
mutations["duplicate_json"] = rejected(lambda: strict_load_text('{"a":1,"a":2}'))
mutations["nonfinite_json"] = rejected(lambda: strict_load_text('{"a":NaN}'))
require(all(mutations.values()), "claim/JSON mutation escaped")

require(SOCKET.exists() and stat.S_ISSOCK(SOCKET.lstat().st_mode), "SSH control socket unavailable")


def remote(command):
    completed = subprocess.run(["ssh", "-S", str(SOCKET), "-p", PORT, REMOTE, command],
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=90, check=False)
    require(completed.returncode == 0 and not completed.stderr.strip(),
            "remote read-only command failed: " + completed.stderr.strip())
    return completed.stdout


identity_command = """set -eu
ckpt='{}'
cfg='{}'
log='{}'
stat -c 'CKPT_BEFORE|%s|%Y|%y|%n' "$ckpt"
sha256sum "$ckpt"
stat -c 'CKPT_AFTER|%s|%Y|%y|%n' "$ckpt"
sha256sum "$cfg"
stat -c 'CONFIG|%s|%Y|%y|%n' "$cfg"
stat -c 'LOG|%s|%Y|%y|%n' "$log"
""".format(CKPT, CONFIG, LOG)
identity_lines = remote(identity_command).splitlines()
require(len(identity_lines) == 6, "remote identity line count")


def stat_line(line, prefix):
    fields = line.split("|", 4)
    require(len(fields) == 5 and fields[0] == prefix, prefix + " stat grammar")
    return {"size_bytes": int(fields[1]), "mtime_epoch": int(fields[2]),
            "mtime": fields[3], "path": fields[4]}


before = stat_line(identity_lines[0], "CKPT_BEFORE")
ckpt_sha, ckpt_path = identity_lines[1].split(None, 1)
after = stat_line(identity_lines[2], "CKPT_AFTER")
config_sha, config_path = identity_lines[3].split(None, 1)
config_stat = stat_line(identity_lines[4], "CONFIG")
log_stat = stat_line(identity_lines[5], "LOG")
require(before == after and before == {"size_bytes": 225504447, "mtime_epoch": 1788037440,
        "mtime": "2026-08-30 05:04:00.000000000 +0800", "path": CKPT}, "remote checkpoint stat")
require(ckpt_sha == contract["checkpoint"]["sha256"] and ckpt_path.strip() == CKPT,
        "remote checkpoint SHA")
require(config_sha == contract["configuration"]["sha256"] and config_path.strip() == CONFIG and
        config_stat["path"] == CONFIG, "remote config SHA")
require(log_stat["path"] == LOG and
        log_stat["size_bytes"] >= contract["validation_evidence"]["log_observed_size_bytes"] and
        log_stat["mtime_epoch"] >= contract["validation_evidence"]["log_observed_mtime_epoch"],
        "remote append-only log stat")

log_command = """log='{}'
awk '/^Epoch [0-9]+$/ || /^Epoch loss \\(Validation\\):/ || /^Validation stats:/ || /checkpoint_epoch24\\.pth/ {{print NR "|" $0}}' "$log"
""".format(LOG)
selected = remote(log_command).splitlines()
current_epoch = None
epochs_started = set()
validations = {}
validation_stats = {}
checkpoint_save_lines = []
evidence_lines = []
for item in selected:
    number_text, text = item.split("|", 1); number = int(number_text)
    match = re.fullmatch(r"Epoch ([0-9]+)", text.strip())
    if match:
        current_epoch = int(match.group(1)); epochs_started.add(current_epoch)
        if current_epoch in {23, 24, 25}:
            evidence_lines.append({"line": number, "text": text})
        continue
    match = re.fullmatch(r"Epoch loss \(Validation\): ([0-9.eE+-]+)\s*", text)
    if match and current_epoch is not None:
        validations.setdefault(current_epoch, []).append(Decimal(match.group(1)))
        if current_epoch in {23, 24}:
            evidence_lines.append({"line": number, "text": text})
        continue
    match = re.fullmatch(r"Validation stats: valid_time_sec=([0-9.]+), valid_step_time_sec=([0-9.]+), valid_samples_per_sec=([0-9.]+)", text.strip())
    if match and current_epoch is not None:
        validation_stats.setdefault(current_epoch, []).append(tuple(Decimal(v) for v in match.groups()))
        if current_epoch == 24:
            evidence_lines.append({"line": number, "text": text})
        continue
    if "checkpoint_epoch24.pth" in text:
        checkpoint_save_lines.append((number, text)); evidence_lines.append({"line": number, "text": text})

require(validations.get(23) == [Decimal("0.8988656344867888")] and
        validations.get(24) == [Decimal("0.8975050449371338")], "epoch23/24 validation parse")
require(validation_stats.get(24) == [(Decimal("15.40"), Decimal("0.7335"), Decimal("2.7268"))],
        "epoch24 validation stats parse")
require(25 in epochs_started and len(checkpoint_save_lines) == 1 and
        checkpoint_save_lines[0][1].endswith(CKPT), "epoch25/checkpoint save parse")
require(sha(DOCS359) == DOCS359_SHA, "docs359 drift")

output = {"schema": "m1121_m1120_motion_ep24_candidate_identity_independent_hammer_mechanical_v1",
    "status": "PASS_M1121_M1120_EP24_CANDIDATE_IDENTITY__NOT_FINAL_NO_REPLAY",
    "score": 100,
    "local_authority": {"contract_sha256": CONTRACT_ID[0],
        "contract_sidecar_sha256": CONTRACT_ID[1],
        "contract_outer_seal_file_sha256": CONTRACT_ID[2],
        "receipt_sha256": RECEIPT_ID[0], "receipt_manifest_sha256": RECEIPT_ID[1],
        "receipt_outer_seal_file_sha256": RECEIPT_ID[2], "docs359_sha256": DOCS359_SHA},
    "remote_readonly": {"host": "ssh.sd5ai.scnet.cn:10037", "control_socket": str(SOCKET),
        "checkpoint": {**before, "sha256": ckpt_sha, "before_after_stat_equal": True},
        "configuration": {**config_stat, "sha256": config_sha}, "train_log": log_stat,
        "checkpoint_downloaded": False, "remote_write": False, "gpu_intervention": False},
    "independent_log_parse": {"epoch23_validation_loss": "0.8988656344867888",
        "epoch24_validation_loss": "0.8975050449371338",
        "epoch24_validation_stats": {"valid_time_sec": "15.40",
            "valid_step_time_sec": "0.7335", "valid_samples_per_sec": "2.7268"},
        "epoch25_started": True, "checkpoint_epoch24_save_line_count": 1,
        "evidence_lines": evidence_lines},
    "mutations": {"rejected": sum(mutations.values()), "total": len(mutations),
                  "cases": mutations},
    "claim_boundary": {"candidate_only": True, "final_checkpoint": False,
        "valid825": False, "hardware_replay": False, "hardware_speedup": False,
        "system_speedup": False, "energy": False, "accuracy": False,
        "paper_citable_hardware_result": False},
    "execution": {"checkpoint_downloaded": False, "remote_write": False,
        "gpu_intervention": False, "hardware_replay": False, "source_modified": False,
        "docs359_modified": False},
    "observed_at_utc": datetime.now(timezone.utc).isoformat()}
OUT.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
               encoding="utf-8")
print(output["status"])
