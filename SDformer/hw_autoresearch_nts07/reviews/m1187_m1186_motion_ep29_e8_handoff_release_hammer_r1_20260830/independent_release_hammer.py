#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Read-only, different-author hammer for the M1186 E8 handoff release."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "scripts/run_m1186_motion_ep29_e8_handoff_release_source.py"
CONTRACT = HW / "contracts/m1186_motion_ep29_e8_handoff_release_source_contract_r1_20260830.json"
TEST = HW / "tests/test_run_m1186_motion_ep29_e8_handoff_release_source.py"
AUTHOR = HW / "reviews/m1186_motion_ep29_e8_handoff_release_source_author_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sealed_directory(directory: Path, review_name: str) -> None:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    need(outer.read_text(encoding="utf-8").split() == [sha(manifest), "SHA256SUMS"],
         "outer seal mismatch: " + str(directory))
    rows = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        name = name.lstrip("*")
        need(Path(name).name == name and name not in rows, "unsafe manifest member")
        need(sha(directory / name) == digest, "manifest byte mismatch: " + name)
        rows[name] = digest
    need(rows.get(review_name) == sha(directory / review_name), "receipt/review not sealed")


def rejected(callable_, pattern: str) -> None:
    try:
        callable_()
    except Exception as error:
        need(pattern in str(error), "wrong rejection: " + repr(error))
        return
    raise RuntimeError("attack accepted: " + pattern)


def main() -> int:
    spec = importlib.util.spec_from_file_location("m1186_hammered", SOURCE)
    need(spec is not None and spec.loader is not None, "cannot load M1186")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    contract = module.strict_json(CONTRACT)
    module.validate_contract(contract, root=ROOT, hash_small=True, hash_large=False)

    compile_run = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", "-m", "py_compile",
         str(SOURCE), str(TEST)], shell=False, check=False, capture_output=True, text=True)
    need(compile_run.returncode == 0, "Python 3.10 compile failed")
    tests = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", "-m", "unittest", "-v",
         str(TEST.relative_to(ROOT))], cwd=ROOT, shell=False, check=False,
        capture_output=True, text=True)
    need(tests.returncode == 0 and "Ran 11 tests" in tests.stderr and "OK" in tests.stderr,
         "controlled tests failed")

    receipt = json.loads((AUTHOR / "author_receipt.json").read_text(encoding="utf-8"))
    sealed_directory(AUTHOR, "author_receipt.json")
    artifacts = receipt["artifacts"]
    for label, path in (("source", SOURCE), ("contract", CONTRACT), ("tests", TEST)):
        need(artifacts[label]["sha256"] == sha(path) and
             artifacts[label]["size_bytes"] == path.stat().st_size,
             "author artifact binding drift: " + label)
    need(sha(CONTRACT.with_name(CONTRACT.name + ".sha256")) ==
         artifacts["contract"]["inner_sha256"], "contract inner seal drift")
    need(sha(CONTRACT.with_name(CONTRACT.name + ".sha256.seal.sha256")) ==
         artifacts["contract"]["outer_sha256"], "contract outer seal drift")

    runtime = ROOT / contract["unchanged_runtime"]["path"]
    need(sha(runtime) == contract["unchanged_runtime"]["sha256"] and
         contract["unchanged_runtime"]["overwritten"] is False,
         "M1183 runtime not preserved")
    small_bytes = sum(row["size_bytes"] for row in contract["small_transfer"])
    large_bytes = sum(row["size_bytes"] for row in contract["preexisting_large"])
    need((len(contract["small_transfer"]), small_bytes + CONTRACT.stat().st_size) ==
         (43, 400749), "44-file exact transfer population/bytes drift")
    need((len(contract["preexisting_large"]), large_bytes) == (40, 491525120),
         "40-row remote hash-only population/bytes drift")
    argv = module.transfer_argv(contract)
    argv_bytes = (json.dumps(argv, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    need(len(argv) == 51 and hashlib.sha256(argv_bytes).hexdigest() ==
         receipt["transfer_closure"]["transfer_argv_canonical_json_sha256"],
         "fixed transfer argv drift")
    need("shell=True" not in SOURCE.read_text(encoding="utf-8"), "shell=True present")
    need(module.remote_preflight_argv() == receipt["preflight"]["remote_argv"],
         "preflight argv drift")
    need(module.runtime_argv() == receipt["single_runtime"]["remote_argv"],
         "runtime argv drift")
    need(sha(DOCS359) == module.DOCS359_SHA256, "docs359 drift")

    m1181 = HW / "reviews/m1181_m1177r2_motion_ep29_e1e8_source_hammer_r1_20260830"
    listed = {Path(row["local_path"]).name for row in contract["small_transfer"]
              if "reviews/m1181_m1177r2" in row["local_path"]}
    actual = {path.name for path in m1181.iterdir() if path.is_file()}
    need(listed == actual and len(actual) == 8, "complete M1181 directory absent")
    for directory, review_name in (
            (m1181, "review.json"),
            (HW / "reviews/m1183_motion_ep29_e1e8_inert_launch_release_author_r1_20260830",
             "e8_author_receipt.json"),
            (HW / "reviews/m1185_m1183_motion_ep29_e8_inert_release_hammer_r1_20260830",
             "review.json")):
        sealed_directory(directory, review_name)

    # Minimal controlled attacks: all operate on in-memory copies or bogus hashes.
    broken = deepcopy(contract)
    broken["small_transfer"] = broken["small_transfer"][:-1]
    rejected(lambda: module.validate_contract(broken, root=ROOT, hash_small=False,
                                              hash_large=False), "transfer populations")
    broken = deepcopy(contract)
    broken["small_transfer"][0]["remote_path"] = "/tmp/redirection"
    rejected(lambda: module.validate_contract(broken, root=ROOT, hash_small=False,
                                              hash_large=False), "remote mapping")
    broken = deepcopy(contract)
    broken["preexisting_large"][0], broken["preexisting_large"][1] = (
        broken["preexisting_large"][1], broken["preexisting_large"][0])
    rejected(lambda: module.validate_contract(broken, root=ROOT, hash_small=True,
                                              hash_large=False), "canonical")
    broken = deepcopy(contract)
    broken["remote_policy"]["host"] = "host;touch /tmp/injected"
    rejected(lambda: module.validate_contract(broken, root=ROOT, hash_small=False,
                                              hash_large=False), "remote policy")
    rejected(lambda: module.verify_future_hammer(contract, "0" * 64, "0" * 64,
                                                 "0" * 64), "future hammer")
    broken = deepcopy(contract)
    broken["runtime_namespaces"]["output"] += "_redirected"
    need(hashlib.sha256((json.dumps(broken, sort_keys=True) + "\n").encode()).hexdigest()
         != sha(CONTRACT), "namespace mutation escaped exact contract seal")

    output = {
        "status": "PASS",
        "python310_compile": True,
        "controlled_tests": 11,
        "attacks_rejected": 6,
        "actual_transfer_files_including_self_contract": 44,
        "actual_transfer_bytes_including_self_contract": 400749,
        "remote_hash_only_files": 40,
        "remote_hash_only_bytes": 491525120,
        "transfer_argv_elements": 51,
        "transfer_argv_canonical_json_sha256": hashlib.sha256(argv_bytes).hexdigest(),
        "unchanged_runtime_sha256": sha(runtime),
        "docs359_sha256": sha(DOCS359),
        "execution": {"remote": False, "gpu": False, "checkpoint": False,
                      "range": False, "eda": False, "production": False},
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
