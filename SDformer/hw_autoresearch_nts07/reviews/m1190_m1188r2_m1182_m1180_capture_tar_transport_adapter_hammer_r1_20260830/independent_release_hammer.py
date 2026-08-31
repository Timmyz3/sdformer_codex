#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Local-only different-author hammer for the M1188R2 transport adapter."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "scripts/run_m1188r2_m1182_m1180_capture_tar_transport_adapter_source.py"
TEST = HW / "tests/test_run_m1188r2_m1182_m1180_capture_tar_transport_adapter_source.py"
R1_TEST = HW / "tests/test_run_m1188_m1182_m1180_capture_tar_transport_adapter_source.py"
CONTRACT = HW / "contracts/m1188r2_m1182_m1180_capture_tar_transport_adapter_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1188r2_m1182_m1180_capture_tar_transport_adapter_author_r1_20260830"
R1_FAIL = HW / "reviews/m1189_m1188_m1182_m1180_capture_tar_transport_adapter_hammer_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sealed(directory: Path, review_name: str) -> None:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    need(outer.read_text(encoding="utf-8").split() == [sha(manifest), "SHA256SUMS"],
         "outer seal mismatch")
    rows = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1); name = name.lstrip("*")
        need(Path(name).name == name and name not in rows, "unsafe manifest")
        need(sha(directory / name) == digest, "inner seal mismatch: " + name)
        rows[name] = digest
    need(rows.get(review_name) == sha(directory / review_name), "review not inner sealed")


def run_tests(path: Path, count: int) -> None:
    completed = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", "-m", "unittest", "-v",
         str(path.relative_to(ROOT))], cwd=ROOT, shell=False, check=False,
        capture_output=True, text=True)
    need(completed.returncode == 0 and "Ran {} tests".format(count) in completed.stderr and
         "OK" in completed.stderr, "controlled tests failed")


def main() -> int:
    spec = importlib.util.spec_from_file_location("m1188r2_hammered", SOURCE)
    need(spec is not None and spec.loader is not None, "cannot import R2")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    contract = module.load_contract()
    members = module.exact_members(contract)
    admitted = module.strict_m1184_admission(contract)
    run_tests(TEST, 10)
    run_tests(R1_TEST, 7)
    sealed(AUTHOR, "author_receipt.json")
    sealed(R1_FAIL, "review.json")
    fail = json.loads((R1_FAIL / "review.json").read_text(encoding="utf-8"))
    need(fail["status"] == "FAIL_CLOSED__M1184_SEMANTIC_STATUS_BINDING_MISMATCH__DO_NOT_TRANSFER" and
         fail["p0"] == 1 and fail["launch_authorized"] is False,
         "R1 FAIL authority drift")
    receipt = json.loads((AUTHOR / "author_receipt.json").read_text(encoding="utf-8"))
    for key, path in (("source", SOURCE), ("test", TEST), ("contract", CONTRACT)):
        need(receipt["artifacts"][key]["sha256"] == sha(path), "author binding drift")
    need((admitted["schema"], admitted["status"], admitted["verdict"]) ==
         tuple(contract["m1184_exact_semantics"][key]
               for key in ("schema", "status", "verdict")),
         "M1184 semantic tuple drift")
    need(admitted["bindings"] == contract["m1184_exact_semantics"]["bindings"] and
         admitted["authorization"] == contract["m1184_exact_semantics"]["authorization"],
         "M1184 exact object drift")
    need(len(members) == len({row["path"] for row in members}) == 51 and
         sum(row["class"] == "ORIGINAL_EXACT42" for row in members) == 42 and
         sum(row["class"] == "M1184_EXACT_SEAL" for row in members) == 9,
         "exact51 drift")
    for row in members:
        path = ROOT / row["path"]
        need(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink() and
             path.stat().st_size == row["size_bytes"] and sha(path) == row["sha256"],
             "member identity drift")

    ssh = module.R1.fixed_ssh_argv()
    scp = module.fixed_scp_argv(Path("/fixed/exact51.tar"))
    need(ssh == ["/usr/bin/ssh", "-p", "10037", "-o",
                 "ControlPath=/tmp/codex_m714_ssh.MFUzxMzZ/control.sock", "-o",
                 "BatchMode=yes", "root@ssh.sd5ai.scnet.cn",
                 "/opt/conda/envs/sdformerflow/bin/python", "-I", "-"], "SSH argv drift")
    need(scp == ["/usr/bin/scp", "-P", "10037", "-o",
                 "ControlPath=/tmp/codex_m714_ssh.MFUzxMzZ/control.sock", "-o",
                 "BatchMode=yes", "/fixed/exact51.tar",
                 "root@ssh.sd5ai.scnet.cn:/tmp/m1188r2_m1180_exact51_transport_r1.tar"],
         "SCP argv drift")
    need(Path("/tmp/codex_m714_ssh.MFUzxMzZ/control.sock").is_socket(),
         "bound control socket absent")
    source_text = SOURCE.read_text(encoding="utf-8")
    need("shell=True" not in source_text and source_text.count("strict_m1184_admission(contract)") >= 2,
         "shell or double semantic gate drift")
    tree = ast.parse(source_text)
    main_def = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = [node.func.id for node in ast.walk(main_def) if isinstance(node, ast.Call) and
             isinstance(node.func, ast.Name)]
    need(calls.index("exact_members") < calls.index("strict_m1184_admission") <
         calls.index("consume_attempt"), "pre-attempt gate order drift")
    r1_text = (ROOT / module.R1_SOURCE_REL).read_text(encoding="utf-8")
    for token in ("not member.isfile() or member.issym() or member.islnk()",
                  "unsafe path", "unsafe parent", "post-install identity"):
        need(token in r1_text, "safe extract/postverify token absent")
    need(sha(DOCS359) == module.DOCS359_SHA256 and not module.ATTEMPT.exists() and
         not module.RESULT.exists(), "protected bytes/namespaces drift")
    print(json.dumps({
        "status": "PASS", "r1_fail_bound": True, "r2_tests": 10,
        "inherited_transport_tests": 7, "semantic_mutation_axes": 8,
        "exact51": 51, "original42": 42, "m1184_exact9": 9,
        "double_semantic_gate_before_attempt": True, "fixed_argv_shell_false": True,
        "control_socket_present": True, "safe_extract_postverify": True,
        "docs359_sha256": sha(DOCS359),
        "execution": {"remote": False, "transfer": False, "gpu": False,
                      "capture": False, "checkpoint": False, "eda": False}},
        indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
