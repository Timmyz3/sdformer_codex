#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author source-only blind hammer for the M1342 authority compiler."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/build_m1342_ep34_table_a_authority_compiler.py"
TEST = HW / "system_simulator/tests/test_m1342_ep34_table_a_authority_compiler.py"
M1340_TEST = HW / "system_simulator/tests/test_m1340_ep34_table_a_common_charge_compiler.py"
CONTRACT = HW / "contracts/m1342_ep34_table_a_authority_compiler_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1342_ep34_table_a_authority_compiler_source_author_r1_20260831"
M1341 = HW / "reviews/m1341_m1340_table_a_common_charge_compiler_source_blind_hammer_r1_20260831"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")

EXPECTED = {
    SOURCE: "ce0089a4af043e5ad74b6a38c142bdd74ad91c36527ffc4a3894e8e905954569",
    TEST: "0b3c9269fd3d07510de944e5300b0cca1cd6454855a80a0ed40f17d59949987f",
    CONTRACT: "45bf03972a72aa2ded7f15e4ac188af4168747f16f6ac55660cfd8a079442c47",
    AUTHOR / "review.json": "3b8adb84c56f4efb0a7b467cf5f9cba135042c21b35e3092f1405f50d6f5e14f",
    AUTHOR / "SHA256SUMS": "ab98de80a7d55a51e3786f4938075901efa1a56ecde9745f9ec00e305396b6f0",
    AUTHOR / "SHA256SUMS.seal.sha256": "30578c5d31f8e3c3c27823441fcb3d0b22692587b630ebf6a8d07e5486ad14d3",
    M1341 / "review.json": "823afc06b63cc24548136bb36f52f79967125a7e559eb113613a49ebe14b2844",
    M1341 / "SHA256SUMS": "fb77a27e122a55d71ce4b9188b370cd4799c7442d9e5d966ac257d357f915e2e",
    M1341 / "SHA256SUMS.seal.sha256": "2822e7e7e1f357f8474fd44127d7fbdcae39a3dcd055a32f755ceadb3eb46d75",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


class HammerError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise HammerError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "import spec failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_seal(root: Path, review_sha: str, manifest_sha: str,
                outer_sha: str) -> dict[str, str]:
    require(root.is_dir() and not root.is_symlink(), "sealed root invalid")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(sha(manifest) == manifest_sha and sha(outer) == outer_sha,
            "seal identity drift")
    require(outer.read_text().split() == [manifest_sha, "SHA256SUMS"],
            "outer content drift")
    rows: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        fields = line.split(maxsplit=1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]) is not None,
                "manifest grammar")
        name = fields[1].lstrip("*")
        rel = Path(name)
        require(not rel.is_absolute() and ".." not in rel.parts and name not in rows,
                "unsafe manifest member")
        member = root / rel
        require(member.is_file() and not member.is_symlink() and sha(member) == fields[0],
                "sealed member drift: " + name)
        rows[name] = fields[0]
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == set(rows), "sealed population drift")
    require(rows.get("review.json") == review_sha, "review member drift")
    return rows


def run_tests(path: Path, expected: int) -> str:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    run = subprocess.run([str(PYTHON), "-B", str(path)], cwd=str(HW.parent), env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, check=False)
    require(run.returncode == 0 and ("Ran %d tests" % expected) in run.stdout
            and "OK" in run.stdout,
            "test replay failed: " + path.name)
    return run.stdout


def rewrite_json(path: Path, payload: Any) -> None:
    path.chmod(0o644)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                               allow_nan=False) + "\n")
    path.chmod(0o444)


def refresh_payload(fixture: Any, role: str, name: str,
                    mutate: Callable[[dict[str, Any]], None]) -> None:
    member = fixture.role_payloads[role][name]["member"]
    path = fixture.role_dirs[role] / member
    payload = json.loads(path.read_text())
    mutate(payload)
    rewrite_json(path, payload)
    fixture.role_payloads[role][name]["sha256"] = sha(path)
    fixture.seal(role)


def make_memory_axis_fixture(T: Any, axis: str):
    fixture = T.Fixture()
    base_path = fixture.role_dirs["final_identity"] / "base_config.json"
    base = json.loads(base_path.read_text())
    for name, spec in fixture.role_payloads["charge_producer"].items():
        path = fixture.role_dirs["charge_producer"] / spec["member"]
        payload = json.loads(path.read_text())
        for charge in payload["population"].values():
            if axis == "dram":
                charge["dram_read_bytes"] = 0
                charge["dram_write_bytes"] = 0
            else:
                for access in charge["sram_bytes"].values():
                    access["read_bytes"] = 0
                    access["write_bytes"] = 0
        rewrite_json(path, payload)
        spec["sha256"] = sha(path)
        if name.startswith("common:"):
            category = name.split(":", 1)[1]
            base["common_operators"][category]["sha256"] = spec["sha256"]
        else:
            _, branch, row = name.split(":")
            base["direct_branches"][branch][row]["sha256"] = spec["sha256"]
    fixture.seal("charge_producer")
    rewrite_json(base_path, base)
    base_sha = sha(base_path)
    fixture.role_payloads["final_identity"]["base_config"]["sha256"] = base_sha
    fixture.config["base_config"]["sha256"] = base_sha
    fixture.seal("final_identity")

    trans_spec = fixture.role_payloads["transaction_receipt"]["transaction_receipt"]
    trans_path = fixture.role_dirs["transaction_receipt"] / trans_spec["member"]
    receipt = json.loads(trans_path.read_text())
    for rows in receipt["rows"].values():
        for charge in rows.values():
            if axis == "dram":
                charge["dram_read_bytes"] = 0
                charge["dram_write_bytes"] = 0
            else:
                for access in charge["sram_bytes"].values():
                    access["read_bytes"] = 0
                    access["write_bytes"] = 0
    rewrite_json(trans_path, receipt)
    trans_spec["sha256"] = sha(trans_path)
    fixture.seal("transaction_receipt")
    return fixture


def main() -> int:
    for path, digest in EXPECTED.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                "identity drift: " + str(path))
    verify_seal(AUTHOR, EXPECTED[AUTHOR / "review.json"],
                EXPECTED[AUTHOR / "SHA256SUMS"],
                EXPECTED[AUTHOR / "SHA256SUMS.seal.sha256"])
    verify_seal(M1341, EXPECTED[M1341 / "review.json"],
                EXPECTED[M1341 / "SHA256SUMS"],
                EXPECTED[M1341 / "SHA256SUMS.seal.sha256"])
    contract = json.loads(CONTRACT.read_text())
    require(contract["authority_model"]["production_allowlist_exact_entries"] == 0
            and contract["authority_model"]["production_execution_possible"] is False,
            "source-only production boundary drift")

    m1340_output = run_tests(M1340_TEST, 10)
    m1342_output = run_tests(TEST, 16)
    T = load("m1346_bound_m1342_tests", TEST)
    N = T.N
    require(N.PRODUCTION_AUTHORITY_ALLOWLIST == {}, "production allowlist not empty")

    rejected: list[str] = []
    accepted: list[str] = []

    def reject_expected(label: str, action: Callable[[], Any]) -> None:
        try:
            action()
        except Exception:
            rejected.append(label)
            return
        accepted.append(label)

    # Ordinary self-created production JSON cannot cross the empty code allowlist.
    production = T.Fixture()
    try:
        production.config["status"] = "PRODUCTION_CANDIDATE"
        reject_expected("self_created_production_with_fixture_allowlist", production.build)
    finally:
        production.close()

    # Positive predecessor closures.
    for label, kwargs in (
            ("per_key_numerator_cancel", {"numerator_cancel": True}),
            ("common_energy_row_rate", {"unfair_energy": True}),
            ("population_manifest_substitution", {"population_manifest_mismatch": True}),
            ("all_memory_zero", {"zero_memory": True}),
            ("transaction_mismatch", {"transaction_mismatch": True})):
        fixture = T.Fixture(**kwargs)
        try:
            reject_expected(label, fixture.build)
        finally:
            fixture.close()

    # FN1: lexical relative_to permits '..' to leave the supplied workspace.
    with tempfile.TemporaryDirectory(prefix="m1346_path_escape_") as temp_name:
        parent = Path(temp_name)
        workspace = parent / "workspace"; workspace.mkdir()
        outside = parent / "outside"; outside.mkdir()
        payload = outside / "forged.json"
        payload.write_text("{}\n"); payload.chmod(0o444)
        escaped = N.regular_readonly_single(workspace, Path("../outside/forged.json"),
                                            "escaped fixture")
        if escaped == workspace.absolute() / "../outside/forged.json":
            accepted.append("dotdot_workspace_escape")
        else:
            rejected.append("dotdot_workspace_escape")

    # FN2: direct per-row rates can assign B0 an arbitrarily high rate and Ours
    # an arbitrarily low rate while the common rate stays fair.
    energy = T.Fixture()
    try:
        def extreme(payload: dict[str, Any]) -> None:
            for branch in N.M.DIRECT_BRANCHES:
                payload["direct_logic_pj_per_cycle"][branch]["B0"] = 1.0e9
                payload["direct_logic_pj_per_cycle"][branch]["Ours"] = 1.0e-9
        refresh_payload(energy, "energy_producer", "partitioned_energy", extreme)
        result = energy.build()
        ours = next(row for row in result["rows"] if row["row_id"] == "Ours")
        require(ours["energy_reduction_vs_B0"] > 0.99,
                "extreme direct-rate exploit did not amplify energy")
        accepted.append("baseline_candidate_direct_energy_rate_mismatch")
    finally:
        energy.close()

    # FN3: address_trace_sha256 is only length-checked and has no sealed trace
    # payload behind it.
    trace = T.Fixture()
    try:
        refresh_payload(trace, "transaction_receipt", "transaction_receipt",
                        lambda payload: payload.__setitem__("address_trace_sha256", "g" * 64))
        result = trace.build()
        require(result["address_trace_sha256"] == "g" * 64,
                "nonhex trace digest not propagated")
        accepted.append("nonhex_unbound_address_trace_digest")
    finally:
        trace.close()

    # FN4/FN5: aggregate all-memory>0 permits an all-zero DRAM plane or an
    # all-zero 17-SRAM plane.
    for axis, label in (("dram", "all_dram_zero_but_sram_nonzero"),
                        ("sram", "all_sram_zero_but_dram_nonzero")):
        fixture = make_memory_axis_fixture(T, axis)
        try:
            result = fixture.build()
            require(len(result["rows"]) == 6, "memory-axis fixture did not build")
            accepted.append(label)
        finally:
            fixture.close()

    # FN6: no SRAM/DRAM latency coordinate or authority is represented.
    baseline = T.Fixture()
    try:
        result = baseline.build()
        resource = result["resource"]
        required_latency = {"sram_read_latency_cycles", "sram_write_latency_cycles",
                            "dram_read_latency_cycles", "dram_write_latency_cycles"}
        if required_latency.isdisjoint(resource):
            accepted.append("memory_latency_authority_absent")
        else:
            rejected.append("memory_latency_authority_absent")
    finally:
        baseline.close()

    result = {
        "schema": "m1346_m1342_table_a_authority_source_blind_hammer_output_r1",
        "status": ("PASS_SOURCE_ADMITTED" if not accepted else
                   "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED"),
        "score": 100 if not accepted else 54,
        "reviewer_independent_of_author": True,
        "inherited_m1340_tests": "10/10 PASS",
        "m1342_author_tests": "16/16 PASS",
        "author_double_seal_verified": True,
        "m1341_double_seal_verified": True,
        "production_allowlist_entries": 0,
        "production_candidate_emitted": False,
        "independent_attack_count": len(rejected) + len(accepted),
        "independent_rejected_count": len(rejected),
        "independent_false_negative_count": len(accepted),
        "rejected_attacks": rejected,
        "accepted_attacks": accepted,
        "execution": {"capture": False, "table_a": False, "gpu": False,
                      "vcs": False, "dc": False, "pt": False, "ptpx": False,
                      "eda": False, "remote": False},
        "docs359_sha256": sha(DOCS359),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("M1346_M1342_BLIND_HAMMER_ERROR: " + str(error), file=sys.stderr)
        raise
