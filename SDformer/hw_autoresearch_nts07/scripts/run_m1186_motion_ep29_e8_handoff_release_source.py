#!/opt/conda/envs/sdformerflow/bin/python
"""Inert handoff/remote-preflight/one-shot launcher source for M1183 E8.

The checked-in contract is source-only.  Transfer, preflight and launch require
a future fresh M1187 different-author hammer.  ``remote-preflight`` is the only
remote entry that does not require that local hammer; it is strictly read-only
and never creates or consumes the M1183 attempt marker.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "hw_autoresearch_nts07"
CONTRACT_REL = Path("hw_autoresearch_nts07/contracts/m1186_motion_ep29_e8_handoff_release_source_contract_r1_20260830.json")
CONTRACT = ROOT / CONTRACT_REL
SOURCE_REL = Path("hw_autoresearch_nts07/scripts/run_m1186_motion_ep29_e8_handoff_release_source.py")
TEST_REL = Path("hw_autoresearch_nts07/tests/test_run_m1186_motion_ep29_e8_handoff_release_source.py")
RUNTIME_REL = Path("hw_autoresearch_nts07/contracts/m1183_motion_ep29_e8_inert_launch_release_r1_20260830.json")
FUTURE_HAMMER_REL = Path("hw_autoresearch_nts07/reviews/m1187_m1186_motion_ep29_e8_handoff_release_hammer_r1_20260830/review.json")
REMOTE_REPO = Path("/root/private_data/work/sdformer_codex/SDformer")
REMOTE_INTERPRETER = Path("/opt/conda/envs/sdformerflow/bin/python")
REMOTE_HOST = "root@ssh.sd5ai.scnet.cn"
REMOTE_PORT = "10037"
SSH_CONTROL_PATH = Path("/tmp/codex_m714_ssh.MFUzxMzZ/control.sock")
RSYNC = Path("/usr/bin/rsync")
SSH = Path("/usr/bin/ssh")
DOCS359_REL = Path("hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md")
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
LEGACY_MARKERS = ("capture_m511_h67_convtranspose_binary_inputs.py",
                  "m511_capture_watcher", "run_m511_h67")


class HandoffError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HandoffError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           HandoffError("nonfinite JSON: " + token)))
    require(isinstance(value, dict), "JSON root must be object")
    return value


def regular(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise HandoffError("missing {}: {}".format(label, path)) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            "{} must be non-symlink regular file".format(label))


def exact_keys(value: Any, expected: set[str], label: str) -> None:
    require(isinstance(value, dict) and set(value) == expected,
            label + " exact key drift")


def repo_relative(text: str) -> Path:
    path = Path(text)
    require(not path.is_absolute() and ".." not in path.parts and path.parts,
            "unsafe repository-relative path")
    return path


def verify_row(row: dict[str, Any], *, root: Path, hash_bytes: bool) -> Path:
    exact_keys(row, {"class", "local_path", "remote_path", "size_bytes", "sha256"},
               "transfer row")
    relative = repo_relative(row["local_path"])
    require(row["remote_path"] == str(REMOTE_REPO / relative),
            "remote mapping drift")
    require(type(row["size_bytes"]) is int and row["size_bytes"] > 0 and
            isinstance(row["sha256"], str) and len(row["sha256"]) == 64,
            "row size/SHA malformed")
    path = root / relative
    if hash_bytes:
        regular(path, "transfer/preflight member")
        require(path.stat().st_size == row["size_bytes"] and
                sha256(path) == row["sha256"], "member byte identity drift: " + str(relative))
    return path


def validate_contract(contract: dict[str, Any], *, root: Path,
                      hash_small: bool, hash_large: bool) -> dict[str, Any]:
    exact_keys(contract, {"schema", "status", "date", "source", "tests",
                          "unchanged_runtime", "predecessor_authorities",
                          "remote_policy", "small_transfer", "preexisting_large",
                          "ep29", "runtime_namespaces", "future_hammer",
                          "execution_boundary"}, "handoff contract")
    require(contract["schema"] == "m1186_motion_ep29_e8_handoff_release_source_contract_r1_v1" and
            contract["status"] == "INERT_SOURCE_ONLY__M1185_P0_REPAIR__FRESH_M1187_HAMMER_REQUIRED",
            "handoff schema/status drift")
    source = contract["source"]
    tests = contract["tests"]
    exact_keys(source, {"path", "sha256"}, "source")
    exact_keys(tests, {"path", "sha256"}, "tests")
    require(source["path"] == str(SOURCE_REL) and tests["path"] == str(TEST_REL),
            "source/test path drift")
    if hash_small:
        require(sha256(root / SOURCE_REL) == source["sha256"] and
                sha256(root / TEST_REL) == tests["sha256"], "source/test SHA drift")
    runtime = contract["unchanged_runtime"]
    require(runtime == {"path": str(RUNTIME_REL),
                        "sha256": "3bc14a2e45837be5e1c5f4c2f0042634b8428f6beaa2152a1f818e0531aa43f5",
                        "inner_sha256": "6a220dc29606fe89e0fa2b977b52e92746d916c0ae75f7976562d660b86331b8",
                        "outer_sha256": "7ec2916d08265aeb0118ca53a34d0c24046485d786c0438caf77b8b30c4562e1",
                        "overwritten": False}, "M1183 runtime authority drift")
    predecessors = contract["predecessor_authorities"]
    require(predecessors == {
        "M1183_author": {
            "review_path": "hw_autoresearch_nts07/reviews/m1183_motion_ep29_e1e8_inert_launch_release_author_r1_20260830/e8_author_receipt.json",
            "review_sha256": "9d4deffa7d5bdb86e069d01856fbd6b3e9b89b6e67a5ffea1fe972e7b0c73e82",
            "manifest_sha256": "3009efce541bc816b58abf42a279e1be676237bd3ab07f95e39efd858f4db437",
            "outer_sha256": "505bd678768eef57a7ebc153132cb887d90842eee1da4f52e6d0655801b76c86"},
        "M1185_FAIL": {
            "review_path": "hw_autoresearch_nts07/reviews/m1185_m1183_motion_ep29_e8_inert_release_hammer_r1_20260830/review.json",
            "review_sha256": "7571d7ff97d2fe526c70e30b4a290ce72bd442a6375dba75c5083d1335b3d428",
            "manifest_sha256": "0a5294bec1b063c44cb9e9e54f31e5cd5df2590ea46b3081f3c0cf01aa49d4bd",
            "outer_sha256": "5bbf66d9b913d71e6a31684b0c0cc55d75d27a0d5340f99473d98af8424eebef",
            "required_status": "FAIL_CLOSED__REMOTE_TRANSFER_CLOSURE_REQUIRED__DO_NOT_TRANSFER_OR_LAUNCH",
            "required_p0": "P0_REMOTE_TRANSFER_CLOSURE_ABSENT"}},
        "M1183/M1185 predecessor authority drift")
    policy = contract["remote_policy"]
    require(policy == {"repo": str(REMOTE_REPO), "interpreter": str(REMOTE_INTERPRETER),
                       "python_version": "3.10.20", "host": REMOTE_HOST,
                       "port": 10037, "ssh_control_path": str(SSH_CONTROL_PATH),
                       "transfer_program": str(RSYNC), "ssh_program": str(SSH),
                       "canonical_lease": "hw_autoresearch_nts07/results/gpu_profile_lease.lock",
                       "automatic_retry": False}, "remote policy drift")
    small = contract["small_transfer"]
    large = contract["preexisting_large"]
    require(isinstance(small, list) and isinstance(large, list) and
            len(small) == 43 and len(large) == 40,
            "transfer populations drift")
    small_paths = [row.get("local_path") for row in small]
    large_paths = [row.get("local_path") for row in large]
    require(len(small_paths) == len(set(small_paths)) and
            len(large_paths) == len(set(large_paths)) and
            not set(small_paths) & set(large_paths), "duplicate transfer member")
    require([row.get("class") for row in small] == ["TRANSFER_SMALL"] * len(small),
            "small transfer class drift")
    require([row.get("class") for row in large] == ["REMOTE_PREEXISTING_HASH_ONLY"] * 40,
            "large preflight class drift")
    for row in small:
        verify_row(row, root=root, hash_bytes=hash_small)
    for row in large:
        verify_row(row, root=root, hash_bytes=hash_large)
    if hash_small:
        m1183_review = strict_json(root / predecessors["M1183_author"]["review_path"])
        m1185_review = strict_json(root / predecessors["M1185_FAIL"]["review_path"])
        require(m1183_review.get("status") ==
                "PASS_INERT_E8_RELEASE_AUTHOR__FRESH_DIFFERENT_AUTHOR_RELEASE_HAMMER_REQUIRED",
                "M1183 author semantic drift")
        require(m1185_review.get("status") == predecessors["M1185_FAIL"]["required_status"] and
                m1185_review.get("blocking_finding", {}).get("id") ==
                predecessors["M1185_FAIL"]["required_p0"],
                "M1185 FAIL/P0 semantic drift")
        canonical = strict_json(root / Path(
            "hw_autoresearch_nts07/contracts/m1177r2_motion_ep29_e8_canonical_40_source_manifest_r1_20260830.json"))
        expected_large = [(row["path"], row["bytes"], row["sha256"])
                          for row in canonical.get("rows", [])]
        actual_large = [(row["local_path"], row["size_bytes"], row["sha256"])
                        for row in large]
        require(actual_large == expected_large and len(expected_large) == 40,
                "exact40 canonical row identity/order drift")
    required = {str(SOURCE_REL), str(TEST_REL), str(RUNTIME_REL),
                "hw_autoresearch_nts07/system_handoff/scripts/run_m1177r2_motion_ep29_e1e8_closure_source.py",
                "hw_autoresearch_nts07/contracts/m1177r2_motion_ep29_e1e8_source_contract_r1_20260830.json",
                "hw_autoresearch_nts07/tests/test_run_m1177r2_motion_ep29_e1e8_closure_source.py",
                "hw_autoresearch_nts07/reviews/m1175_m1171_motion_final_checkpoint_binder_result_hammer_r1_20260830/review.json",
                "hw_autoresearch_nts07/reviews/m1181_m1177r2_motion_ep29_e1e8_source_hammer_r1_20260830/review.json",
                "hw_autoresearch_nts07/reviews/m1181_m1177r2_motion_ep29_e1e8_source_hammer_r1_20260830/SHA256SUMS",
                "hw_autoresearch_nts07/reviews/m1181_m1177r2_motion_ep29_e1e8_source_hammer_r1_20260830/SHA256SUMS.seal.sha256",
                "hw_autoresearch_nts07/contracts/m1177r2_motion_ep29_e8_canonical_40_source_manifest_r1_20260830.json",
                "hw_autoresearch_nts07/contracts/m1177r2_motion_ep29_e8_canonical_40_source_manifest_r1_20260830.json.sha256",
                "hw_autoresearch_nts07/contracts/m1177r2_motion_ep29_e8_canonical_40_source_manifest_r1_20260830.json.sha256.seal.sha256",
                "neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_nts11_hardware_p0.py",
                "third_party/SDformerFlow/eval_DSEC_flow_SNN.py",
                "hw_autoresearch_nts07/results/m36_h67_ep35_patch_embed_profile_s10_r1_20260822/sample_workload.csv",
                "hw_autoresearch_nts07/contracts/m699_h67_ep35_multisequence_decoder_payload_contract_r1_20260828.json",
                str(DOCS359_REL)}
    require(required <= set(small_paths), "minimum runtime/handoff closure absent")
    future = contract["future_hammer"]
    require(future == {"canonical_review_path": str(FUTURE_HAMMER_REL),
                       "required_schema": "m1187_m1186_motion_ep29_e8_handoff_release_hammer_r1_v1",
                       "required_status": "PASS_HANDOFF_RELEASE__TRANSFER_PREFLIGHT_LAUNCH_ALLOWED",
                       "production_authorized_by_contract": False}, "future hammer contract drift")
    return contract


def verify_future_hammer(contract: dict[str, Any], review_sha: str,
                         manifest_sha: str, outer_sha: str) -> None:
    review_path = ROOT / FUTURE_HAMMER_REL
    directory = review_path.parent
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    for path, digest in ((review_path, review_sha), (manifest, manifest_sha), (outer, outer_sha)):
        regular(path, "future hammer authority")
        require(sha256(path) == digest, "future hammer declared SHA drift")
    require(outer.read_text(encoding="utf-8").split() == [manifest_sha, "SHA256SUMS"],
            "future hammer outer seal drift")
    rows = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2, "future hammer manifest malformed")
        name = fields[1].lstrip("*")
        require(Path(name).name == name and name not in rows, "future hammer member unsafe")
        member = directory / name
        regular(member, "future hammer member")
        require(sha256(member) == fields[0], "future hammer member SHA drift")
        rows[name] = fields[0]
    require(rows.get("review.json") == review_sha, "future hammer review unsealed")
    review = strict_json(review_path)
    require(review.get("schema") == contract["future_hammer"]["required_schema"] and
            review.get("status") == contract["future_hammer"]["required_status"] and
            review.get("launch_authorized") is True and
            review.get("artifacts", {}).get("source_sha256") == contract["source"]["sha256"] and
            review.get("artifacts", {}).get("contract_sha256") == sha256(CONTRACT) and
            review.get("artifacts", {}).get("tests_sha256") == contract["tests"]["sha256"],
            "future hammer semantics/artifact binding drift")


def ssh_prefix() -> list[str]:
    return [str(SSH), "-p", REMOTE_PORT, "-o", "ControlPath=" + str(SSH_CONTROL_PATH),
            "-o", "BatchMode=yes", "--", REMOTE_HOST]


def transfer_argv(contract: dict[str, Any]) -> list[str]:
    # The self-referential contract cannot contain its own SHA.  Its exact bytes
    # are instead sealed by the author receipt and future M1187 review, while
    # this literal argv still transfers it alongside every inventoried member.
    sources = [str(ROOT) + "/./" + str(CONTRACT_REL)]
    sources.extend(str(ROOT) + "/./" + row["local_path"]
                   for row in contract["small_transfer"])
    remote_shell = (str(SSH) + " -p " + REMOTE_PORT + " -o ControlPath=" +
                    str(SSH_CONTROL_PATH) + " -o BatchMode=yes")
    return [str(RSYNC), "-a", "--relative", "--protect-args", "-e", remote_shell,
            *sources, REMOTE_HOST + ":" + str(REMOTE_REPO) + "/"]


def remote_preflight_argv() -> list[str]:
    return [*ssh_prefix(), str(REMOTE_INTERPRETER), str(REMOTE_REPO / SOURCE_REL),
            "--contract", str(REMOTE_REPO / CONTRACT_REL), "--action", "remote-preflight"]


def runtime_argv() -> list[str]:
    runtime_source = REMOTE_REPO / "hw_autoresearch_nts07/system_handoff/scripts/run_m1177r2_motion_ep29_e1e8_closure_source.py"
    return [*ssh_prefix(), str(REMOTE_INTERPRETER), str(runtime_source), "--contract",
            str(REMOTE_REPO / RUNTIME_REL)]


def running_legacy_watchers(proc_root: Path = Path("/proc")) -> list[int]:
    found = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(marker in command for marker in LEGACY_MARKERS):
            found.append(int(entry.name))
    return sorted(found)


def remote_preflight(contract: dict[str, Any]) -> None:
    require(ROOT == REMOTE_REPO, "remote repository identity drift")
    require(sys.version_info[:3] == (3, 10, 20), "remote Python version drift")
    validate_contract(contract, root=ROOT, hash_small=True, hash_large=True)
    ep29 = contract["ep29"]
    for label in ("checkpoint", "config"):
        row = ep29[label]
        path = Path(row["path"])
        regular(path, "ep29 " + label)
        require(path.stat().st_size == row["size_bytes"] and sha256(path) == row["sha256"],
                "ep29 {} bytes drift".format(label))
        if "mtime_ns" in row:
            require(path.stat().st_mtime_ns == row["mtime_ns"], label + " mtime drift")
    namespaces = contract["runtime_namespaces"]
    require(not os.path.lexists(ROOT / namespaces["output"]) and
            not os.path.lexists(ROOT / namespaces["attempt_marker"]),
            "output/attempt namespace is not fresh")
    lease = ROOT / contract["remote_policy"]["canonical_lease"]
    if lease.exists():
        regular(lease, "canonical GPU lease")
        descriptor = os.open(lease, os.O_RDWR)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise HandoffError("canonical GPU lease busy") from error
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
    else:
        require(lease.parent.is_dir(), "canonical lease parent absent")
    require(not running_legacy_watchers(), "legacy M511 watcher present including SIGSTOP")
    gpu = subprocess.run(["/usr/bin/nvidia-smi", "--query-compute-apps=pid",
                          "--format=csv,noheader,nounits"], shell=False, check=False,
                         text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    require(gpu.returncode == 0 and not gpu.stdout.strip(), "GPU is not idle")
    print(json.dumps({"status": "PASS_READONLY_REMOTE_PREFLIGHT",
                      "attempt_consumed": False, "output_created": False,
                      "small_members": len(contract["small_transfer"]),
                      "large_hash_rows": len(contract["preexisting_large"])}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--action", required=True,
                        choices=("show-transfer-argv", "transfer-small", "preflight", "launch",
                                 "remote-preflight"))
    parser.add_argument("--release-hammer-review-sha256")
    parser.add_argument("--release-hammer-manifest-sha256")
    parser.add_argument("--release-hammer-outer-sha256")
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    require(contract_path == CONTRACT.resolve(), "canonical handoff contract path required")
    contract = strict_json(contract_path)
    remote = args.action == "remote-preflight"
    validate_contract(contract, root=ROOT, hash_small=True, hash_large=False)
    if args.action == "show-transfer-argv":
        print(json.dumps(transfer_argv(contract)))
        return 0
    if remote:
        remote_preflight(contract)
        return 0
    require(all((args.release_hammer_review_sha256,
                 args.release_hammer_manifest_sha256,
                 args.release_hammer_outer_sha256)), "fresh M1187 hammer hashes required")
    verify_future_hammer(contract, args.release_hammer_review_sha256,
                         args.release_hammer_manifest_sha256,
                         args.release_hammer_outer_sha256)
    require(SSH_CONTROL_PATH.is_socket(), "exact SSH control socket absent")
    if args.action == "transfer-small":
        completed = subprocess.run(transfer_argv(contract), shell=False, check=False)
        require(completed.returncode == 0, "exact small transfer failed")
        return 0
    preflight = subprocess.run(remote_preflight_argv(), shell=False, check=False)
    require(preflight.returncode == 0, "read-only remote preflight failed")
    if args.action == "preflight":
        return 0
    require(args.action == "launch", "unreachable action")
    completed = subprocess.run(runtime_argv(), shell=False, check=False)
    require(completed.returncode == 0, "single zero-retry runtime launch failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
