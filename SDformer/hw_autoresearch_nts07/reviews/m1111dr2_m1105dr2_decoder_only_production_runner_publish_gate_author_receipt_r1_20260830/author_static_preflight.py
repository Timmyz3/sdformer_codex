#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1111Dr2 author static and /tmp mutation preflight; never runs production."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUNNER = HW / "system_simulator/scripts/run_m1111dr2_m1105dr2_decoder_only_production_zero_arg.py"
RUNNER_SHA = "1167258c228631b73ca1784ae57db19e8f0fbe709efa34f369585c508bc9d746"
CONTRACT = HW / "contracts/m1111dr2_m1105dr2_decoder_only_production_runner_source_contract_r2_20260830.json"
CONTRACT_ID = ("821819b00503b91a8fb8dfca8fe000208e10746e751a3815131dc8ff1cbed515",
    "6f71af39ddd60ee1faaae350bc55a7145bfe0d6313ff878f742f23acebdf0bc6",
    "402fc2e2d7ea9da5fbadc33dea104a7ef3eae06e9e89e21a3244123d66298268")
M1112D = HW / "reviews/m1112d_m1111d_decoder_runner_final_independent_hammer_r1_20260830"
M1112D_ID = ("dc47d9fdb59c17531d7bd5d3f41734357064d7e90a355e7973ad30885e85112a",
    "1f341d6b862d5d72d40d208acf9de4b2dfda905908594fb713c82c6833a3256e",
    "d55667ad70f9946716fa76534196f7266d4f32a718ca5b5fa51f9a26b2cb9872")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
OUT = HERE / "mechanical_checks.json"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, expected: str) -> None:
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink() and
            sha(path) == expected, "regular identity drift " + str(path))


def rejected(function) -> bool:
    try:
        function()
    except (Exception, SystemExit):
        return True
    return False


def copy_flat(source: Path, destination: Path) -> None:
    destination.mkdir()
    for path in source.iterdir():
        if path.is_file():
            shutil.copyfile(path, destination / path.name)


regular(RUNNER, RUNNER_SHA)
regular(CONTRACT, CONTRACT_ID[0])
regular(Path(str(CONTRACT) + ".sha256"), CONTRACT_ID[1])
regular(Path(str(CONTRACT) + ".sha256.seal.sha256"), CONTRACT_ID[2])
regular(M1112D / "review.json", M1112D_ID[0])
regular(M1112D / "SHA256SUMS", M1112D_ID[1])
regular(M1112D / "SHA256SUMS.seal.sha256", M1112D_ID[2])
regular(DOCS359, DOCS359_SHA)
require(Path(str(CONTRACT) + ".sha256").read_text().split() ==
        [CONTRACT_ID[0], CONTRACT.name] and
        Path(str(CONTRACT) + ".sha256.seal.sha256").read_text().split() ==
        [CONTRACT_ID[1], CONTRACT.name + ".sha256"], "contract double seal drift")

source_text = RUNNER.read_text(encoding="utf-8")
tree = ast.parse(source_text)
functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
require("validate_publish_candidate" in functions and "publish_gate_mutation_self_test" in functions and
        "publish_result" in functions and source_text.index("checked = validate_publish_candidate(work)") <
            source_text.index("rename_noreplace(work, RESULT)") and
        'M1112D_ID = (' in source_text and M1112D_ID[2] in source_text,
        "publish repair/static trust structure drift")

spec = importlib.util.spec_from_file_location("m1111dr2_author_preflight", RUNNER)
require(spec is not None and spec.loader is not None, "runner import spec")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)

namespace_before = runner.namespace_fresh()
require(namespace_before, "r2 namespace not fresh before static test")
self_test = runner.source_static_self_test()
require(self_test["status"] == "PASS_M1111DR2_RUNNER_SOURCE_STATIC_SELF_TEST__NO_PRODUCTION" and
        self_test["publish_gate_mutation_self_test"]["valid_candidate_calls"] == 120 and
        self_test["publish_gate_mutation_self_test"]["valid_candidate_transactions"] == 720 and
        self_test["publish_gate_mutation_self_test"]["mutations_rejected"] == 13 and
        self_test["publish_gate_mutation_self_test"]["mutations_total"] == 13 and
        self_test["publish_gate_mutation_self_test"]["canonical_publish_called"] is False and
        self_test["publish_gate_mutation_self_test"]["canonical_attempt_created"] is False,
        "publish mutation self-test drift")

old_argv = list(sys.argv)
try:
    sys.argv[:] = [str(RUNNER), "forbidden"]
    extra_argv_rejected = rejected(runner.main)
finally:
    sys.argv[:] = old_argv
require(extra_argv_rejected, "extra argv accepted")

with tempfile.TemporaryDirectory(prefix="m1111dr2_flat_symlink.") as raw:
    root = Path(raw)
    manifest_case = root / "manifest_case"
    copy_flat(M1112D, manifest_case)
    real_manifest = root / "real_manifest"
    real_manifest.write_bytes((manifest_case / "SHA256SUMS").read_bytes())
    (manifest_case / "SHA256SUMS").unlink()
    (manifest_case / "SHA256SUMS").symlink_to(real_manifest)
    manifest_symlink_rejected = rejected(lambda: runner.verify_flat(
        manifest_case, M1112D_ID,
        "STOP_M1112D_PUBLISH_GATE_ACCEPTS_FORBIDDEN_CLAIMS_AND_INCOMPLETE_FILESET"))

    outer_case = root / "outer_case"
    copy_flat(M1112D, outer_case)
    real_outer = root / "real_outer"
    real_outer.write_bytes((outer_case / "SHA256SUMS.seal.sha256").read_bytes())
    (outer_case / "SHA256SUMS.seal.sha256").unlink()
    (outer_case / "SHA256SUMS.seal.sha256").symlink_to(real_outer)
    outer_symlink_rejected = rejected(lambda: runner.verify_flat(
        outer_case, M1112D_ID,
        "STOP_M1112D_PUBLISH_GATE_ACCEPTS_FORBIDDEN_CLAIMS_AND_INCOMPLETE_FILESET"))

    extra_case = root / "extra_case"
    copy_flat(M1112D, extra_case)
    (extra_case / "EXTRA").write_text("forged\n", encoding="utf-8")
    extra_flat_file_rejected = rejected(lambda: runner.verify_flat(
        extra_case, M1112D_ID,
        "STOP_M1112D_PUBLISH_GATE_ACCEPTS_FORBIDDEN_CLAIMS_AND_INCOMPLETE_FILESET"))
require(manifest_symlink_rejected and outer_symlink_rejected and extra_flat_file_rejected,
        "flat-root mutation escaped")

with tempfile.TemporaryDirectory(prefix="m1111dr2_contract_bytes.") as raw:
    root = Path(raw)
    changed = root / CONTRACT.name
    changed.write_bytes(CONTRACT.read_bytes() + b"\n")
    (root / (CONTRACT.name + ".sha256")).write_bytes(
        Path(str(CONTRACT) + ".sha256").read_bytes())
    (root / (CONTRACT.name + ".sha256.seal.sha256")).write_bytes(
        Path(str(CONTRACT) + ".sha256.seal.sha256").read_bytes())
    contract_bytes_rejected = rejected(lambda: runner.verify_double(changed, CONTRACT_ID))
require(contract_bytes_rejected, "contract byte mutation escaped")

require(runner.namespace_fresh() and sha(DOCS359) == DOCS359_SHA,
        "canonical namespace/docs359 changed during static test")

output: dict[str, Any] = {
    "schema": "m1111dr2_decoder_runner_publish_gate_author_static_checks_v1",
    "status": "PASS_M1111DR2_AUTHOR_STATIC_AND_TMP_MUTATION_TEST__NO_PRODUCTION",
    "score": 100,
    "identity": {"runner_sha256": RUNNER_SHA, "contract_sha256": CONTRACT_ID[0],
        "contract_sidecar_sha256": CONTRACT_ID[1],
        "contract_outer_seal_file_sha256": CONTRACT_ID[2],
        "m1112d_review_sha256": M1112D_ID[0],
        "m1112d_manifest_sha256": M1112D_ID[1],
        "m1112d_outer_seal_file_sha256": M1112D_ID[2],
        "docs359_sha256": DOCS359_SHA},
    "publish_gate": self_test["publish_gate_mutation_self_test"],
    "flat_root_repairs": {"manifest_symlink_rejected": manifest_symlink_rejected,
        "outer_symlink_rejected": outer_symlink_rejected,
        "extra_flat_file_rejected": extra_flat_file_rejected},
    "additional_attacks": {"extra_argv_rejected": extra_argv_rejected,
        "contract_bytes_rejected": contract_bytes_rejected},
    "execution": {"runner_main_executed": False, "canonical_payload_opened": False,
        "execute_production_executed": False, "publish_result_executed": False,
        "canonical_attempt_created": False, "canonical_result_created": False,
        "canonical_work_created": False, "canonical_quarantine_created": False,
        "temporary_synthetic_candidates_only": True, "namespace_fresh": True},
    "authorization": {"different_author_final_hammer_required": True,
        "production_launch_now": False, "attempt_now": False,
        "production_replay_now": False},
}
OUT.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
               encoding="utf-8")
print(output["status"])
