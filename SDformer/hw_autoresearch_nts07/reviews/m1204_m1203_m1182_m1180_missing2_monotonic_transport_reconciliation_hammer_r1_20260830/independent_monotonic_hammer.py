#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent local-only M1204 hammer; never invokes SSH/SCP/GPU/capture/EDA."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "scripts/run_m1203_m1182_m1180_missing2_monotonic_transport_reconciliation_source.py"
TEST = HW / "tests/test_run_m1203_m1182_m1180_missing2_monotonic_transport_reconciliation_source.py"
CONTRACT = HW / "contracts/m1203_m1182_m1180_missing2_monotonic_transport_reconciliation_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1203_m1182_m1180_missing2_monotonic_transport_reconciliation_author_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    SOURCE: "0f54be0e8d14dace4a6bc36700798d3e45ee53db0a5d30557af1fcea822c1b7e",
    TEST: "c0666be2e4584a64610aad4bb9e94e4a3f4557922f5aa56b96b8b2824708dc93",
    CONTRACT: "bf9213a56d473aeb2511afdb2f20ce9bcaf3ee9cb6ac8d7ed4379aa0d14c9f57",
    AUTHOR / "SHA256SUMS": "b3db53ee073c62cd82373d97e3f223fa99e88a378b8a05ad363f867174f5fb67",
    AUTHOR / "SHA256SUMS.seal.sha256": "08529b2e049b787b6a8473cd90421d8f1afda50af38076a40e7dbfef0df4da45",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_module():
    spec = importlib.util.spec_from_file_location("m1203_independent", SOURCE)
    require(spec is not None and spec.loader is not None, "source import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(payload: bytes, path: str) -> dict:
    return {"path": path, "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def archive(path: Path, rows: list[dict], payloads: list[bytes], *,
            names: list[str] | None = None, link: bool = False) -> None:
    with tarfile.open(path, "w") as stream:
        for index, (item, payload) in enumerate(zip(rows, payloads)):
            info = tarfile.TarInfo((names or [value["path"] for value in rows])[index])
            if link and index == 0:
                info.type = tarfile.SYMTYPE
                info.linkname = "victim"
                info.size = 0
                stream.addfile(info)
            else:
                info.size = len(payload)
                stream.addfile(info, io.BytesIO(payload))


def main() -> None:
    checks: dict[str, object] = {}
    checks["identity"] = {str(path.relative_to(ROOT)): sha(path) for path in EXPECTED}
    require(all(checks["identity"][str(path.relative_to(ROOT))] == digest
                for path, digest in EXPECTED.items()), "declared identity drift")

    manifest = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=AUTHOR,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              check=False, text=True)
    outer = subprocess.run(["sha256sum", "-c", "SHA256SUMS.seal.sha256"], cwd=AUTHOR,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           check=False, text=True)
    require(manifest.returncode == outer.returncode == 0, "author recursive seal")
    checks["author_recursive_seal"] = True

    tests = subprocess.run(["/opt/anaconda3/envs/pytorch310/bin/python3.10", str(TEST)],
                           cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           check=False, text=True)
    require(tests.returncode == 0 and "Ran 10 tests" in tests.stdout and "OK" in tests.stdout,
            "author tests")
    checks["author_tests"] = "PASS_10_OF_10"

    m = load_module()
    contract = m.load_contract()
    m.verify_policy(contract)
    members = m.exact_members(contract)
    require(len(members) == 2, "exact two")
    checks["contract_and_local_members"] = "PASS_EXACT2"

    payloads = [b"independent-one", b"independent-two"]
    rows = [row(payloads[0], "a/one"), row(payloads[1], "b/two")]
    with tempfile.TemporaryDirectory(prefix="m1204_independent_") as temporary:
        temp = Path(temporary)
        staged = [temp / "s1", temp / "s2"]
        destinations = [temp / "a/one", temp / "b/two"]
        for path, payload in zip(staged, payloads):
            write(path, payload)
        for path in destinations:
            path.parent.mkdir(parents=True, exist_ok=True)

        write(destinations[0], payloads[0])
        inode = destinations[0].stat().st_ino
        require(m.reconcile_exact_files(staged, destinations, rows) == ["EXACT", "EXACT"],
                "subset reconciliation")
        require(destinations[0].stat().st_ino == inode, "exact subset not idempotent")
        checks["preexisting_exact_subset"] = "PASS_IDEMPOTENT"

        destinations[0].unlink()
        destinations[1].unlink()
        calls = 0
        def fail_second(source: Path, destination: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("M1204_INJECT_SECOND_PUBLISH")
            os.replace(source, destination)
        try:
            m.reconcile_exact_files(staged, destinations, rows, publish=fail_second)
            raise AssertionError("publish failure accepted")
        except OSError as error:
            require("M1204_INJECT" in str(error), "unexpected publish failure")
        require([m.exact_state(path, item) for path, item in zip(destinations, rows)] ==
                ["EXACT", "ABSENT"], "unsafe partial publication")
        require(m.reconcile_exact_files(staged, destinations, rows) == ["EXACT", "EXACT"],
                "partial exact not recoverable")
        checks["partial_exact_safe_and_recoverable"] = True

        destinations[0].unlink()
        destinations[1].unlink()
        write(destinations[0], b"wrong")
        try:
            m.reconcile_exact_files(staged, destinations, rows)
            raise AssertionError("wrong target accepted")
        except m.ReconcileError:
            pass
        require(destinations[0].read_bytes() == b"wrong" and not destinations[1].exists(),
                "wrong target mutated")
        destinations[0].unlink()
        destinations[0].symlink_to(temp / "victim")
        try:
            m.reconcile_exact_files(staged, destinations, rows)
            raise AssertionError("symlink target accepted")
        except m.ReconcileError:
            pass
        require(destinations[0].is_symlink() and not destinations[1].exists(),
                "symlink target mutated")
        checks["wrong_and_symlink_target"] = "PASS_REJECT_NO_MUTATION"

        good = temp / "good.tar"
        archive(good, rows, payloads)
        require(len(m.validate_archive_to_stage(good, temp / "stage-good", rows)) == 2,
                "good archive")
        attacks = []
        extra_rows = rows + [row(b"extra", "c/extra")]
        extra = temp / "extra.tar"
        archive(extra, extra_rows, payloads + [b"extra"])
        attacks.append((extra, temp / "stage-extra"))
        traversal = temp / "traversal.tar"
        archive(traversal, rows, payloads, names=["../escape", "b/two"])
        attacks.append((traversal, temp / "stage-traversal"))
        symlink_archive = temp / "symlink.tar"
        archive(symlink_archive, rows, payloads, link=True)
        attacks.append((symlink_archive, temp / "stage-link"))
        for candidate, stage in attacks:
            try:
                m.validate_archive_to_stage(candidate, stage, rows)
                raise AssertionError("archive attack accepted")
            except m.ReconcileError:
                pass
        checks["archive_member_attacks"] = "PASS_REJECT_3_OF_3"

    preflight = m.preflight_program(members).decode("utf-8")
    archive_covered = str(m.REMOTE_ARCHIVE) in preflight
    stage_covered = str(m.REMOTE_STAGE) in preflight
    checks["pre_scp_remote_archive_checked"] = archive_covered
    checks["pre_scp_remote_stage_checked"] = stage_covered
    checks["fixed_scp_destination"] = m.fixed_scp_argv(Path("/tmp/local.tar"))[-1]
    require(not archive_covered and not stage_covered,
            "expected pre-SCP counterexample no longer reproduces")

    attempt_paths = [m.LOCAL_ATTEMPT, m.LOCAL_RESULT,
                     ROOT / m.M1180_ATTEMPT_REL, ROOT / m.M1180_RESULT_REL]
    checks["attempt_result_namespaces_absent"] = all(
        not path.exists() and not path.is_symlink() for path in attempt_paths)
    require(checks["attempt_result_namespaces_absent"] is True, "namespace not inert")

    report = {
        "schema": "m1204_m1203_m1182_m1180_missing2_monotonic_transport_reconciliation_mechanical_r1_v1",
        "status": "STOP_PRE_SCP_ARCHIVE_PATH_NOT_PREFLIGHTED",
        "checks": checks,
        "counterexample": {
            "precondition": "remote /tmp archive path is a symlink/nonregular file before SCP",
            "observed_source_order": "preflight members -> mark local attempt -> SCP fixed path -> remote verifier",
            "failure": "preflight carries neither REMOTE_ARCHIVE nor REMOTE_STAGE; SCP can mutate/follow the archive target before reconciler type/SHA checks",
            "required_fix": "preflight and safely reserve an absent non-symlink archive endpoint, or stream bytes over the already authenticated SSH process without a remote pathname; fresh source identity and hammer required",
        },
        "execution_authorized": False,
        "remote_transfer_gpu_capture_eda_executed": False,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
