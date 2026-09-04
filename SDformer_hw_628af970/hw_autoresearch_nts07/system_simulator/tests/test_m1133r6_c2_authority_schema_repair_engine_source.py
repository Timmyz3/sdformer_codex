#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Author tests for M1133r6.  No EDA, launcher, or engine main is executed."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path


HW = Path(__file__).resolve().parents[2]
SOURCE = HW / "dc_handoff/scripts/m1133r6_c2_authority_schema_repair_engine_source_r1.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def seal_flat(directory: Path) -> str:
    members = sorted(path for path in directory.rglob("*") if path.is_file() and
                     path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    manifest = directory / "SHA256SUMS"
    manifest.write_text("".join(
        f"{sha(path)}  {path.relative_to(directory).as_posix()}\n" for path in members),
        encoding="utf-8")
    outer = directory / "SHA256SUMS.seal.sha256"
    outer.write_text(f"{sha(manifest)}  SHA256SUMS\n", encoding="utf-8")
    return sha(outer)


def seal_double(path: Path) -> str:
    side = Path(str(path) + ".sha256")
    side.write_text(f"{sha(path)}  {path.relative_to(HW).as_posix()}\n", encoding="utf-8")
    outer = Path(str(path) + ".sha256.seal.sha256")
    outer.write_text(f"{sha(side)}  {side.relative_to(HW).as_posix()}\n", encoding="utf-8")
    return sha(outer)


spec = importlib.util.spec_from_file_location("m1133r6_subject", SOURCE)
assert spec is not None and spec.loader is not None
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


def build_fixture(root: Path, mutate=None) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    launcher = root / "launcher.py"
    launcher.write_text("# controlled mock; never executed\n", encoding="utf-8")
    m1134 = root / "m1134"
    m1134.mkdir()
    author_outer = "a" * 64
    write_json(m1134 / "review.json", {
        "status": "PASS_M1134R6_M1133R6_ENGINE_HAMMER__AUTHOR_ZERO_ARG_LAUNCHER_ONLY__NO_EDA",
        "identity": {
            "engine_sha256": sha(SOURCE),
            "contract_sha256": E.CONTRACT_SHA256,
            "author_receipt_outer_seal_file_sha256": author_outer,
        },
    })
    m1134_outer = seal_flat(m1134)
    receipt = root / "receipt.json"
    value = {
        "schema": "m1133r6_c2_authority_schema_repair_authorized_launch_receipt_r1_v1",
        "status": "M1133R6_LAUNCH_SOURCE_FROZEN__M1136R6_REQUIRED__NO_EDA",
        "launcher_sha256": sha(launcher),
        "engine_sha256": sha(SOURCE),
        "engine_contract_sha256": E.CONTRACT_SHA256,
        "engine_contract_outer_seal_file_sha256": E.CONTRACT_OUTER_SHA256,
        "engine_author_receipt_outer_seal_file_sha256": author_outer,
        "m1121_outer_seal_file_sha256": E.M1121_OUTER_SHA256,
        "m1132r5_stop_outer_seal_file_sha256": E.M1132R5_STOP_OUTER_SHA256,
        "m1134r6_outer_seal_file_sha256": m1134_outer,
        "arguments": 0,
        "caller_selected_authority_allowed": False,
        "caller_environment_forwarded": False,
        "m1136r6_required": True,
        "launch_now": False,
        "attempt_now": False,
        "dc_now": False,
        "mapped_vcs_now": False,
        "maximum_attempts": 1,
        "automatic_retry": False,
        "paper_citable": False,
    }
    if mutate is not None:
        mutate(value)
    write_json(receipt, value)
    receipt_outer = seal_double(receipt)
    m1136 = root / "m1136"
    m1136.mkdir()
    write_json(m1136 / "review.json", {
        "status": "PASS_M1136R6_M1133R6_FINAL_LAUNCH_HAMMER__GO_ONE_ATTEMPT",
        "identity": {
            "launch_receipt_outer_seal_file_sha256": receipt_outer,
            "launcher_sha256": sha(launcher),
            "engine_sha256": sha(SOURCE),
            "engine_contract_outer_seal_file_sha256": E.CONTRACT_OUTER_SHA256,
            "engine_author_receipt_outer_seal_file_sha256": author_outer,
            "m1121_outer_seal_file_sha256": E.M1121_OUTER_SHA256,
            "m1132r5_stop_outer_seal_file_sha256": E.M1132R5_STOP_OUTER_SHA256,
            "m1134r6_outer_seal_file_sha256": m1134_outer,
        },
    })
    seal_flat(m1136)
    return {"launcher": launcher, "receipt": receipt, "m1134": m1134, "m1136": m1136}


def bind_fixture(paths: dict) -> None:
    E.LAUNCHER = paths["launcher"]
    E.LAUNCH_RECEIPT = paths["receipt"]
    E.M1134R6 = paths["m1134"]
    E.M1136R6 = paths["m1136"]
    E.BASE.LAUNCHER = paths["launcher"]
    E.BASE.LAUNCH_RECEIPT = paths["receipt"]
    # Only /proc parent identity is controlled; all sealed authority functions
    # and static source checks remain the real subject implementation.
    E.BASE.verify_parent_launcher = lambda _receipt: None


def expect_failure(paths: dict, label: str) -> None:
    bind_fixture(paths)
    try:
        E.verify_future_authority()
    except E.GateFailure:
        return
    raise AssertionError(label + " did not fail closed")


def main() -> int:
    original_argv = list(sys.argv)
    checks = 0
    with tempfile.TemporaryDirectory(prefix=".m1133r6_fixture.", dir=HW / "results") as tmp:
        paths = build_fixture(Path(tmp) / "valid")
        bind_fixture(paths)
        result = E.verify_future_authority()
        assert result["m1121_outer_seal_file_sha256"] == E.M1121_OUTER_SHA256
        checks += 1
        sys.argv[:] = [str(SOURCE), "--authorized-launch"]
        static = E.static_gate()
        assert static["m1121_exact_static_authority"] is True
        assert static["m1132r5_stop_exact_static_authority"] is True
        checks += 1

        def missing(value):
            del value["m1121_outer_seal_file_sha256"]

        def extra(value):
            value["unexpected_authority"] = False

        def wrong(value):
            value["m1121_outer_seal_file_sha256"] = "0" * 64

        for label, mutation in (("missing", missing), ("extra", extra), ("wrong", wrong)):
            attack_root = Path(tmp) / label
            expect_failure(build_fixture(attack_root, mutation), label)
            checks += 1
    sys.argv[:] = original_argv
    assert sha(E.BASE_ENGINE) == E.BASE_ENGINE_SHA256
    assert sha(E.BASE.RTL) == E.RTL_SHA256
    assert sha(E.BASE.TB) == E.TB_SHA256
    assert sha(E.BASE.FILELIST) == E.FILELIST_SHA256
    checks += 4
    print(f"PASS_M1133R6_AUTHOR_TEST checks={checks} future_success=1 static_return=1 attacks=3 no_eda=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
