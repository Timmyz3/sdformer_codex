#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent local-only M1207 hammer; never invokes SSH/SCP/GPU/capture/EDA."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "scripts/run_m1206_m1182_m1180_missing2_secure_mktemp_monotonic_transport_source.py"
TEST = HW / "tests/test_run_m1206_m1182_m1180_missing2_secure_mktemp_monotonic_transport_source.py"
CONTRACT = HW / "contracts/m1206_m1182_m1180_missing2_secure_mktemp_monotonic_transport_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1206_m1182_m1180_missing2_secure_mktemp_transport_author_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    SOURCE: "09ee522c6524a31994e11718c3e597aea1a6c1a4cccb8b36b524ac3792cf0106",
    TEST: "38d835351ef019a5ed1daaa3148eb044fc610afb87c187e965cce87bd50e5e32",
    CONTRACT: "e1faadfc1253bf0c8bd3b581f47468a9236e6532fb7d5a50ec3505a4ae02869c",
    AUTHOR / "SHA256SUMS": "0b2b564090e1f6de13efe0dc251b1f43b8d24e5f03df9463e6b438fc2c15ecad",
    AUTHOR / "SHA256SUMS.seal.sha256": "af7db5fdf543d50481e1e7ac5f0b846a0e238ef656c44d06f9810cacdf879b9c",
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
    spec = importlib.util.spec_from_file_location("m1206_independent", SOURCE)
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


def expect_transport_error(m, operation, label: str) -> None:
    try:
        operation()
    except m.TransportError:
        return
    raise AssertionError(label + " accepted")


def main() -> None:
    checks: dict[str, object] = {}
    identities = {str(path.relative_to(ROOT)): sha(path) for path in EXPECTED}
    require(all(identities[str(path.relative_to(ROOT))] == digest
                for path, digest in EXPECTED.items()), "declared identity drift")
    checks["identity"] = identities

    author_manifest = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"], cwd=AUTHOR, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    author_outer = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS.seal.sha256"], cwd=AUTHOR, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    require(author_manifest.returncode == author_outer.returncode == 0,
            "author recursive seal")
    checks["author_recursive_seal"] = True

    tests = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", str(TEST)], cwd=ROOT,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    require(tests.returncode == 0 and "Ran 12 tests" in tests.stdout and "OK" in tests.stdout,
            "author tests")
    checks["author_tests"] = "PASS_12_OF_12"

    m = load_module()
    contract = m.load_contract()
    m.verify_policy(contract)
    members = m.exact_members(contract)
    require(len(members) == 2 and [item["path"] for item in members] ==
            [item["path"] for item in m.EXPECTED], "exact-two identity/order")
    checks["contract_members"] = "PASS_EXACT2_PINNED"

    valid_stdout = b"/tmp/m1206_m1180.Ab12Cd34Ef56\n"
    require(m.validate_temp_path_text(valid_stdout).as_posix() ==
            "/tmp/m1206_m1180.Ab12Cd34Ef56", "valid mktemp stdout")
    stdout_attacks = [
        b"relative\n", b"/tmp/m1206_m1180.short\n",
        valid_stdout + b"/tmp/m1206_m1180.Zz99Yy88Xx77\n",
        b"noise /tmp/m1206_m1180.Ab12Cd34Ef56\n",
        b"/tmp/m1206_m1180.Ab12Cd34Ef56", b"\xff\n",
    ]
    for attack in stdout_attacks:
        expect_transport_error(m, lambda attack=attack: m.validate_temp_path_text(attack),
                               "mktemp stdout attack")
    checks["mktemp_stdout"] = "PASS_ANCHORED_EXACT_ONE_LINE__REJECT_6_OF_6"

    with tempfile.TemporaryDirectory(prefix="m1207_tempdir_") as temporary:
        parent = Path(temporary)
        tempdir = parent / "m1206_m1180.Ab12Cd34Ef56"
        old_re = m.REMOTE_TEMP_RE
        m.REMOTE_TEMP_RE = re.compile(r"\A" + re.escape(tempdir.as_posix()) + r"\Z")
        try:
            tempdir.mkdir(mode=0o700)
            m.validate_temp_directory(tempdir, os.getuid())
            expect_transport_error(m, lambda: m.validate_temp_directory(tempdir, os.getuid()+1),
                                   "wrong temp owner")
            tempdir.chmod(0o750)
            expect_transport_error(m, lambda: m.validate_temp_directory(tempdir, os.getuid()),
                                   "wrong temp mode")
            tempdir.chmod(0o700)
            tempdir.rmdir()
            tempdir.write_bytes(b"file")
            expect_transport_error(m, lambda: m.validate_temp_directory(tempdir, os.getuid()),
                                   "temp regular file")
            tempdir.unlink()
            target = parent / "target"
            target.mkdir()
            tempdir.symlink_to(target, target_is_directory=True)
            expect_transport_error(m, lambda: m.validate_temp_directory(tempdir, os.getuid()),
                                   "temp symlink")
        finally:
            m.REMOTE_TEMP_RE = old_re
    checks["temp_lstat_owner_mode"] = "PASS_DIR_NONLINK_OWNER_MODE0700"

    payloads = [b"independent-one", b"independent-two-two"]
    rows = [row(payloads[0], "a/one"), row(payloads[1], "b/two")]
    with tempfile.TemporaryDirectory(prefix="m1207_independent_") as temporary:
        temp = Path(temporary)
        good = temp / m.REMOTE_ARCHIVE_BASENAME
        archive(good, rows, payloads)
        m.validate_archive_path(good, good.stat().st_size, sha(good))
        expect_transport_error(m, lambda: m.validate_archive_path(good, good.stat().st_size+1,
                                                                  sha(good)),
                               "archive size")
        expect_transport_error(m, lambda: m.validate_archive_path(good, good.stat().st_size,
                                                                  "0"*64),
                               "archive SHA")
        renamed = temp / "wrong.tar"
        renamed.write_bytes(good.read_bytes())
        expect_transport_error(m, lambda: m.validate_archive_path(renamed, renamed.stat().st_size,
                                                                  sha(renamed)),
                               "archive basename")
        link_path = temp / "link-parent" / m.REMOTE_ARCHIVE_BASENAME
        link_path.parent.mkdir()
        link_path.symlink_to(good)
        expect_transport_error(m, lambda: m.validate_archive_path(link_path,
                                                                  good.stat().st_size, sha(good)),
                               "archive symlink")

        staged = m.validate_archive_to_stage(good, temp / "stage-good", rows)
        require([m.exact_state(path, item) for path, item in zip(staged, rows)] ==
                ["EXACT", "EXACT"], "good archive extraction")
        attacks: list[tuple[Path, Path]] = []
        extra_rows = rows + [row(b"extra", "c/extra")]
        extra = temp / "extra.tar"
        archive(extra, extra_rows, payloads + [b"extra"])
        attacks.append((extra, temp / "stage-extra"))
        traversal = temp / "traversal.tar"
        archive(traversal, rows, payloads, names=["../escape", "b/two"])
        attacks.append((traversal, temp / "stage-traversal"))
        duplicate = temp / "duplicate.tar"
        archive(duplicate, rows, payloads, names=["a/one", "a/one"])
        attacks.append((duplicate, temp / "stage-duplicate"))
        link = temp / "member-link.tar"
        archive(link, rows, payloads, link=True)
        attacks.append((link, temp / "stage-link"))
        bad_sha = temp / "bad-sha.tar"
        archive(bad_sha, rows, [b"X" * len(payloads[0]), payloads[1]])
        attacks.append((bad_sha, temp / "stage-bad-sha"))
        for candidate, stage in attacks:
            expect_transport_error(
                m, lambda candidate=candidate, stage=stage:
                m.validate_archive_to_stage(candidate, stage, rows), "archive member attack")
            require(not stage.exists() and not stage.is_symlink(), "failed stage cleanup")
        checks["archive_gate"] = "PASS_LSTAT_SIZE_SHA_AND_SAFE_MEMBERS__REJECT_5_OF_5"

        local_staged = [temp / "s1", temp / "s2"]
        destinations = [temp / "dest/a/one", temp / "dest/b/two"]
        for path, payload in zip(local_staged, payloads):
            write(path, payload)
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)

        state_matrix = []
        for initial in ((False, False), (True, False), (False, True), (True, True)):
            for destination in destinations:
                if destination.exists() or destination.is_symlink(): destination.unlink()
            for present, destination, payload in zip(initial, destinations, payloads):
                if present: write(destination, payload)
            old_inodes = [destination.stat().st_ino if present else None
                          for present, destination in zip(initial, destinations)]
            final = m.reconcile_exact_files(local_staged, destinations, rows,
                                            "Ab12Cd34Ef56")
            require(final == ["EXACT", "EXACT"], "state matrix final")
            for index, old_inode in enumerate(old_inodes):
                if old_inode is not None:
                    require(destinations[index].stat().st_ino == old_inode,
                            "preexisting exact target replaced")
            state_matrix.append({"initial": ["EXACT" if x else "ABSENT" for x in initial],
                                 "final": final})
        checks["absent_exact_state_matrix"] = state_matrix

        for destination in destinations:
            if destination.exists() or destination.is_symlink(): destination.unlink()
        write(destinations[0], b"wrong")
        expect_transport_error(m, lambda: m.reconcile_exact_files(
            local_staged, destinations, rows, "Ab12Cd34Ef56"), "wrong destination")
        require(destinations[0].read_bytes() == b"wrong" and not destinations[1].exists(),
                "wrong destination mutated")
        destinations[0].unlink()
        destinations[0].symlink_to(temp / "missing-victim")
        expect_transport_error(m, lambda: m.reconcile_exact_files(
            local_staged, destinations, rows, "Ab12Cd34Ef56"), "symlink destination")
        require(destinations[0].is_symlink() and not destinations[1].exists(),
                "symlink destination mutated")
        checks["wrong_symlink_targets"] = "PASS_REJECT_NO_MUTATION"

        destinations[0].unlink()
        calls = 0
        def fail_second(source: Path, destination: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == 2: raise OSError("M1207_INJECT_SECOND_PUBLISH")
            os.replace(source, destination)
        try:
            m.reconcile_exact_files(local_staged, destinations, rows, "Ab12Cd34Ef56",
                                    publish=fail_second)
            raise AssertionError("publish failure accepted")
        except OSError as error:
            require("M1207_INJECT" in str(error), "unexpected publish failure")
        require([m.exact_state(path, item) for path, item in zip(destinations, rows)] ==
                ["EXACT", "ABSENT"], "unsafe partial publication")
        require(m.reconcile_exact_files(local_staged, destinations, rows,
                                        "Zz99Yy88Xx77") == ["EXACT", "EXACT"],
                "partial exact recovery")
        checks["partial_publish"] = "PASS_SAFE_EXACT_ABSENT_AND_RECOVERABLE"

        for destination in destinations:
            if destination.exists() or destination.is_symlink(): destination.unlink()
        marker = temp / "capture.marker"
        control_calls = 0
        def control() -> None:
            nonlocal control_calls
            control_calls += 1
            if marker.exists() or marker.is_symlink():
                raise m.TransportError("M1207_CAPTURE_MARKER_RACE")
        def race_after_first(index: int) -> None:
            if index == 0: marker.write_text("race", encoding="utf-8")
        expect_transport_error(m, lambda: m.reconcile_exact_files(
            local_staged, destinations, rows, "Ab12Cd34Ef56",
            after_publish=race_after_first, control_absent=control), "marker race")
        require([m.exact_state(path, item) for path, item in zip(destinations, rows)] ==
                ["EXACT", "ABSENT"], "marker race unsafe state")
        require(control_calls >= 2, "marker not checked around publish")
        checks["capture_marker_race"] = "PASS_FAIL_CLOSED_EXACT_ABSENT"

        marker.unlink()
        require(m.reconcile_exact_files(local_staged, destinations, rows,
                                        "Zz99Yy88Xx77", control_absent=control) ==
                ["EXACT", "EXACT"], "marker-race recovery")
        for destination in destinations: destination.unlink()
        def retain_temp(source: Path, destination: Path) -> None:
            os.link(source, destination)
        def reject_cleanup(path: Path) -> None:
            raise OSError("M1207_INJECT_CLEANUP_FAILURE")
        expect_transport_error(m, lambda: m.reconcile_exact_files(
            local_staged, destinations, rows, "Ab12Cd34Ef56",
            publish=retain_temp, cleanup=reject_cleanup), "cleanup failure")
        require([m.exact_state(path, item) for path, item in zip(destinations, rows)] ==
                ["EXACT", "EXACT"], "cleanup failure invalidated exact targets")
        checks["cleanup_failure"] = "PASS_NO_SUCCESS_EXACT_TARGETS_RETAINED"

    temp = Path("/tmp/m1206_m1180.Ab12Cd34Ef56")
    scp_argv = m.fixed_scp_argv(Path("/local/archive.tar"), temp)
    ssh_argv = m.fixed_ssh_python_argv()
    require(scp_argv[-1] == "root@ssh.sd5ai.scnet.cn:/tmp/m1206_m1180.Ab12Cd34Ef56/exact2.tar",
            "SCP destination not exclusive-dir child")
    require(ssh_argv[-3:] == [m.REMOTE_INTERPRETER, "-I", "-"], "SSH python argv")
    source_text = SOURCE.read_text(encoding="utf-8")
    require("shell=True" not in source_text and source_text.count("shell=False") >= 3,
            "shell execution boundary")
    require("/tmp/m1203_m1180_missing2_monotonic_transport_r1.tar" not in source_text,
            "fixed legacy archive path survived")
    require(re.search(r"/tmp/m1206_m1180\.[A-Za-z0-9]+/exact2\.tar", source_text) is None,
            "fixed M1206 archive path present")
    preflight = m.temp_preflight_program(temp, members).decode("utf-8")
    reconcile = m.reconciler_program(temp, members, 123, "0" * 64).decode("utf-8")
    cleanup = m.cleanup_program(temp, members).decode("utf-8")
    for token in ("tempdir(td)", "list(td.iterdir())", "archive symlink/nonregular",
                  "archive owner/size/SHA", "archive extra/path/order attack",
                  "final both-exact gate", "M1180 attempt/result must remain absent"):
        require(token in preflight + reconcile + cleanup, "remote gate missing: " + token)
    checks["argv_and_remote_programs"] = {
        "scp_exclusive_dir_child": scp_argv[-1],
        "shell_false": True,
        "no_fixed_archive_path": True,
        "preflight_reconcile_cleanup_gates": True,
    }

    namespaces = [m.LOCAL_ATTEMPT, m.LOCAL_RESULT,
                  ROOT / m.M1180_ATTEMPT_REL, ROOT / m.M1180_RESULT_REL]
    require(all(not path.exists() and not path.is_symlink() for path in namespaces),
            "attempt/result namespace not inert")
    checks["attempt_result_namespaces_absent"] = True
    checks["no_remote_transfer_gpu_capture_eda"] = True

    report = {
        "schema": "m1207_m1206_m1182_m1180_missing2_secure_mktemp_transport_mechanical_r1_v1",
        "status": "PASS_SECURE_MKTEMP_MONOTONIC_EXACT2_LOCAL_HAMMER",
        "checks": checks,
        "score": 100,
        "one_shot_transfer_authorizable": True,
        "automatic_retry": False,
        "remote_transfer_gpu_capture_eda_executed": False,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
