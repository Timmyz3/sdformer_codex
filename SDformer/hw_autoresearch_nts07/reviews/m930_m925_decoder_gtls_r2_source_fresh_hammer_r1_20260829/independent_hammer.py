#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent source/process-control hammer for M925 decoder GTLS R2.

This review is intentionally bounded.  It runs the frozen M896 tests only up
to the sealed real 100K prefix and exercises generic private process groups.
It cannot consume an M925 attempt, enumerate the full first row, or invoke
EDA, VCS, GPU, remote, network, or production work.
"""

from __future__ import annotations

import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Any, Dict, Iterable, List, Mapping, Tuple


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
SETSID = Path("/usr/bin/setsid")
DRIVER = HW / "system_simulator/scripts/execute_m925_m896_decoder_run_gtls_full_first_row_exact_scalability_r2.py"
RUNNER = HW / "system_simulator/scripts/run_m925_m896_decoder_run_gtls_full_first_row_exact_scalability_r2_one_shot.sh"
CONTRACT = HW / "contracts/m925_m896_decoder_run_gtls_full_first_row_exact_scalability_source_contract_r1_20260829.json"
M896 = HW / "system_simulator/scripts/analyze_m896_decoder_run_gtls_source_candidate.py"
M896_TESTS = HW / "system_simulator/tests/test_m896_decoder_run_gtls_source_candidate.py"
M902 = HW / "reviews/m902_m900_decoder_fullrow_failure_audit_r1_20260829"
M900_ATTEMPT = HW / "results/.m900_m896_decoder_run_gtls_full_first_row_runtime_gate_r1_attempt_consumed"
M900_FAILURE = HW / "results/m900_m896_decoder_run_gtls_full_first_row_runtime_gate_r1_20260829.failed_or_incomplete.3773893.17022.27057"
M900_PARTIAL_HEARTBEAT = HW / "results/m900_m896_decoder_run_gtls_full_first_row_runtime_gate_r1_20260829.failed_or_incomplete.3773893.17022.27057.partial_artifact/runtime_heartbeat.json"
REQUEST = HW / "reviews/m926_m925_decoder_gtls_r2_source_fresh_hammer_REQUEST_r1_20260829"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
FUTURE_RELEASE = HW / "contracts/m927_m925_decoder_run_gtls_full_first_row_exact_scalability_release_r1_20260829.json"

RESULT_NAME = "m925_m896_decoder_run_gtls_full_first_row_exact_scalability_r2_20260829"
ATTEMPT_NAME = ".m925_m896_decoder_run_gtls_full_first_row_exact_scalability_r2_attempt_consumed"
EXPECTED = {
    DRIVER: "e02d3c0dc8b47234b3c6b065ccb30f52d8684b3813fa7b2753a6eab2c2df6806",
    RUNNER: "b8f0dae1dd07423099d9d82cd3646b9343aa1623d9e39e9239ff30959cd18f05",
    CONTRACT: "7140d6cc7aa80f1f6016828d325f719abad594aff66cb13316564ef93256032e",
    M896: "c877f70849eb254bd5b227c79e8120773a9c48aa7405a2e6564b7eb4647aae39",
    M896_TESTS: "12c1e092253ff078b52f7b5f7fcce9e17d4cb721e0f0d5aad2d75e86ca4d90eb",
    M902 / "review.json": "6b25dae1ed54fb7b591472a3fd6b6ac9932772e13e60f4395895fd3526e2fc3b",
    M900_ATTEMPT / "attempt.json": "8515d32c05fdc03084f8633f304c5903b3a769afcc3b0c4b004c81a3bc70a561",
    M900_FAILURE / "failure.json": "56cd1f4d27cd0fe54fddd94ed9432e8abfaadb4a7c036c2e13badbb5cda71d6c",
    M900_PARTIAL_HEARTBEAT: "0a030c05410fc92a03a09bdbc3e2a13af3e17b206668545e5be11313795f4b03",
    REQUEST / "request.json": "dba53ce88e257dc22f12dc305d298cb748bbd240f015e57ec1897aa4bb1c1f6a",
    PYTHON: "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    SETSID: "827259531e3511bcc704143690d8a3afec043d24a7922bf3ebfacf917cd7e100",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
EXPECTED_SEALS = {
    M902: ("e6f1fe535227be4146b3563b481f7d3504b76352b93e585cff49b879fbb4fad9",
           "98b3c505534fec3904d2fb327c4050c6fc3ab3a4e975ca96a0fd7ec8ef91d4da"),
    M900_ATTEMPT: ("4584af13bfa85033aafb9d6ffd9881a1ce20f37e46c45e544454fa39128dff7b",
                   "01f582fefec1e4ba2f079c4e0057e02e108f934bde3b6863af4b46cd77eccedd"),
    M900_FAILURE: ("ed3cff05817be659093f2546b25180b75d0f5ac7432e16f27f643b23ed98206a",
                   "f36d2335e2dff5a2102e8c89a5b6c6b61181540519ccb2088ba3baf53d9d94c2"),
    REQUEST: ("1153b2591cffef315218932befc28e696920c5cd9454a219af54266f28c66033",
              "6440422a58b745ee58553dbb4dc9737f0a6c9ccdf19c81b51d2ae10350657135"),
}
EXPECTED_SIDECARS = {
    DRIVER: ("3947fef839bb36fef7a43a6377d410309975404a02ad63215712a000e57b8add",
             "2e03957c8ab0f45a2dd209e6017031e30a3530b86c28a0f8c21e2a436ee33f75"),
    RUNNER: ("de70a8424497deedef7d30a2fff81a9b3531af066e8f666da0859b10ee27a970",
             "4182b25f08ed9f769b13a694be5c2efb217c78a4db41128eff0876693fb1f91a"),
    CONTRACT: ("9190f1d2399fe8d02838d236b528b3b26a46e94fdeb1a508eef98c70798cdcdc",
               "4097067044d1946b38184e3bdd7896dc457472e3d6703ae6011ce319ccb74d9a"),
}


class HammerFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HammerFailure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_load(path: Path) -> Any:
    def pairs(rows: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise HammerFailure("duplicate JSON key: " + key)
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise HammerFailure("nonfinite JSON: " + value)

    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=constant)


def verify_sidecar(path: Path, expected: Tuple[str, str]) -> Dict[str, str]:
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    require(path.is_file() and not path.is_symlink(), "payload absent/symlink: " + str(path))
    require(sidecar.is_file() and not sidecar.is_symlink(), "sidecar absent/symlink")
    require(outer.is_file() and not outer.is_symlink(), "outer sidecar absent/symlink")
    require(sha256(sidecar) == expected[0], "sidecar SHA drift: " + path.name)
    require(sha256(outer) == expected[1], "outer-sidecar SHA drift: " + path.name)
    require(sidecar.read_text(encoding="ascii").strip().split() ==
            [sha256(path), path.name], "sidecar content drift: " + path.name)
    require(outer.read_text(encoding="ascii").strip().split() ==
            [sha256(sidecar), sidecar.name], "outer-sidecar content drift: " + path.name)
    return {"payload_sha256": sha256(path), "sidecar_sha256": expected[0],
            "outer_seal_file_sha256": expected[1]}


def verify_sealed_dir(path: Path, expected: Tuple[str, str]) -> Dict[str, Any]:
    require(path.is_dir() and not path.is_symlink(), "sealed dir absent/symlink: " + str(path))
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and not manifest.is_symlink(), "manifest absent/symlink")
    require(outer.is_file() and not outer.is_symlink(), "outer seal absent/symlink")
    require(sha256(manifest) == expected[0], "manifest SHA drift: " + path.name)
    require(sha256(outer) == expected[1], "outer seal-file SHA drift: " + path.name)
    require(outer.read_text(encoding="ascii").strip().split() ==
            [expected[0], "SHA256SUMS"], "outer seal content drift: " + path.name)
    listed: Dict[str, str] = {}
    for row in manifest.read_text(encoding="ascii").splitlines():
        fields = row.split("  ", 1)
        require(len(fields) == 2 and len(fields[0]) == 64, "malformed manifest row")
        digest, name = fields
        require(name not in listed and Path(name).name == name and
                "/" not in name and "\\" not in name and "\x00" not in name,
                "unsafe/duplicate manifest name")
        member = path / name
        require(member.is_file() and not member.is_symlink(), "member absent/symlink: " + name)
        require(sha256(member) == digest, "member drift: " + name)
        if member.suffix == ".json":
            strict_load(member)
        listed[name] = digest
    actual = {entry.name for entry in path.iterdir() if entry.is_file()}
    require(actual == set(listed) | {"SHA256SUMS", "SHA256SUMS.seal.sha256"},
            "sealed population drift: " + path.name)
    require(not [entry for entry in path.iterdir() if entry.is_symlink()],
            "symlink in sealed directory: " + path.name)
    return {"manifest_sha256": expected[0], "outer_seal_file_sha256": expected[1],
            "sealed_members": len(listed)}


def result_namespace() -> List[str]:
    names: List[str] = []
    for entry in (HW / "results").iterdir():
        if (entry.name in (RESULT_NAME, ATTEMPT_NAME) or
                entry.name.startswith(RESULT_NAME + ".stage.") or
                entry.name.startswith(ATTEMPT_NAME + ".stage.") or
                entry.name.startswith(RESULT_NAME + ".worker_") or
                entry.name.startswith(RESULT_NAME + ".runtime_resource_") or
                entry.name.startswith(RESULT_NAME + ".failed_or_incomplete.")):
            names.append(entry.name)
    return sorted(names)


def compile_sources() -> Dict[str, bool]:
    for path in (DRIVER, M896, M896_TESTS):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    completed = subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)],
                               cwd=str(HW), capture_output=True, text=True,
                               check=False, timeout=30)
    require(completed.returncode == 0, "runner bash -n failed: " + completed.stderr)
    return {"driver_compile": True, "m896_compile": True,
            "m896_tests_compile": True, "runner_bash_n": True}


def run_m896_pytest() -> Dict[str, Any]:
    started = time.monotonic()
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_ADDOPTS": "-p no:cacheprovider"}
    completed = subprocess.run(
        [str(PYTHON), "-m", "pytest", "-q", "-p", "no:cacheprovider", str(M896_TESTS)],
        cwd=str(HW), env=env, capture_output=True, text=True, check=False,
        timeout=180)
    require(completed.returncode == 0, "M896 pytest failed:\n" + completed.stdout + completed.stderr)
    require("11 passed" in completed.stdout, "M896 pytest did not report 11 passed")
    return {"status": "PASS_FROZEN_M896_11_OF_11", "elapsed_seconds": round(time.monotonic() - started, 6),
            "stdout_tail": completed.stdout.strip().splitlines()[-1]}


def runner_env() -> Dict[str, str]:
    return {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "M925_EXPECTED_RUNNER_SHA256": EXPECTED[RUNNER],
            "M925_EXPECTED_CONTRACT_SHA256": EXPECTED[CONTRACT]}


def run_refusal_attacks() -> Dict[str, Any]:
    before = result_namespace()
    require(not before, "M925 namespace occupied before refusal attacks: " + repr(before))
    require(not FUTURE_RELEASE.exists() and not FUTURE_RELEASE.is_symlink(),
            "future M927 release unexpectedly exists")

    good = subprocess.run([str(RUNNER), "--dry-run-no-work"], cwd=str(HW),
                          env=runner_env(), capture_output=True, text=True,
                          check=False, timeout=60)
    require(good.returncode == 0, "exact-pin no-work failed: " + good.stdout + good.stderr)
    require("PASS_M925_NO_WORK_DRY_RUN__NO_FILES_NO_ATTEMPT" in good.stdout,
            "no-work PASS token absent")

    attacks: Dict[str, int] = {}
    cases = []
    cases.append(("missing_all_pins", [str(RUNNER), "--dry-run-no-work"],
                  {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}))
    bad_runner = runner_env(); bad_runner["M925_EXPECTED_RUNNER_SHA256"] = "0" * 64
    cases.append(("wrong_runner_pin", [str(RUNNER), "--dry-run-no-work"], bad_runner))
    malformed_runner = runner_env(); malformed_runner["M925_EXPECTED_RUNNER_SHA256"] = "not-a-sha"
    cases.append(("malformed_runner_pin", [str(RUNNER), "--dry-run-no-work"], malformed_runner))
    bad_contract = runner_env(); bad_contract["M925_EXPECTED_CONTRACT_SHA256"] = "f" * 64
    cases.append(("wrong_contract_pin", [str(RUNNER), "--dry-run-no-work"], bad_contract))
    cases.append(("wrong_arguments", [str(RUNNER), "--dry-run-no-work", "extra"], runner_env()))
    future = runner_env()
    future.update({"M925_EXPECTED_RELEASE_SHA256": "0" * 64,
                   "M925_EXPECTED_FINAL_HAMMER_REVIEW_SHA256": "1" * 64,
                   "M925_EXPECTED_FINAL_HAMMER_OUTER_SHA256": "2" * 64})
    cases.append(("future_release_absent", [str(RUNNER)], future))
    for label, argv, env in cases:
        completed = subprocess.run(argv, cwd=str(HW), env=env,
                                   capture_output=True, text=True, check=False,
                                   timeout=60)
        require(completed.returncode != 0, label + " attack was accepted")
        attacks[label] = completed.returncode
    after = result_namespace()
    require(after == before, "refusal/no-work attacks changed M925 namespace")
    require(not FUTURE_RELEASE.exists() and not FUTURE_RELEASE.is_symlink(),
            "refusal attack created future release")
    return {"exact_pin_no_work": True, "refusal_return_codes": attacks,
            "namespace_before": before, "namespace_after": after,
            "future_release_absence_blocked_no_argument_launch": True}


def import_driver():
    spec = importlib.util.spec_from_file_location("m930_frozen_m925", DRIVER)
    require(spec is not None and spec.loader is not None, "cannot import M925 driver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def target_json_and_rename_attacks() -> Dict[str, bool]:
    module = import_driver()
    rejected: Dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="m930_json_rename_") as temporary:
        root = Path(temporary)
        for label, payload in (("duplicate", '{"a":1,"a":2}'),
                               ("nan", '{"a":NaN}'),
                               ("positive_infinity", '{"a":Infinity}'),
                               ("negative_infinity", '{"a":-Infinity}')):
            path = root / (label + ".json")
            path.write_text(payload, encoding="utf-8")
            refused = False
            try:
                module.strict_json(path)
            except Exception:
                refused = True
            require(refused, "target strict_json accepted " + label)
            rejected[label] = True
        source = root / "source"
        destination = root / "destination"
        source.write_bytes(b"source")
        destination.write_bytes(b"destination")
        collision = False
        try:
            module._rename_noreplace(source, destination)
        except module.Failure:
            collision = True
        require(collision and source.read_bytes() == b"source" and
                destination.read_bytes() == b"destination",
                "renameat2 no-replace collision was destructive/accepted")
        destination.unlink()
        module._rename_noreplace(source, destination)
        require(not source.exists() and destination.read_bytes() == b"source",
                "renameat2 no-replace success drift")
    return {**{key + "_rejected": value for key, value in rejected.items()},
            "renameat2_collision_rejected": True,
            "renameat2_success_same_directory": True}


def proc_identity(pid: int) -> Dict[str, Any] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        rest = raw[raw.rfind(")") + 2:].split()
        status = Path(f"/proc/{pid}/status").read_text(encoding="ascii").splitlines()
        uid = int(next(row.split()[1] for row in status if row.startswith("Uid:")))
        rss_rows = [row.split()[1] for row in status if row.startswith("VmRSS:")]
        exe = Path(os.readlink(f"/proc/{pid}/exe")).resolve()
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        return {"pid": pid, "state": rest[0], "ppid": int(rest[1]),
                "pgrp": int(rest[2]), "session": int(rest[3]),
                "start": int(rest[19]), "uid": uid,
                "rss_kib": int(rss_rows[0]) if rss_rows else 0,
                "exe": str(exe), "cmdline_hex": cmdline.hex()}
    except (FileNotFoundError, ProcessLookupError, PermissionError,
            StopIteration, ValueError, OSError):
        return None


def group_members(pgrp: int, session: int, uid: int, root_start: int,
                  include_zombie: bool = False) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        row = proc_identity(int(entry.name))
        if row is None:
            continue
        if (row["pgrp"] == pgrp and row["session"] == session and
                row["uid"] == uid and row["start"] >= root_start and
                (include_zombie or row["state"] != "Z")):
            rows.append(row)
    return sorted(rows, key=lambda row: row["pid"])


def capture_python_root(process: subprocess.Popen, timeout: float = 5.0) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = proc_identity(process.pid)
        if (row is not None and row["exe"] == str(PYTHON.resolve()) and
                row["pgrp"] == process.pid and row["session"] == process.pid and
                row["ppid"] == os.getpid()):
            return row
        if process.poll() is not None:
            break
        time.sleep(0.01)
    raise HammerFailure("setsid root did not resolve to actual Python worker")


def wait_group_empty(root: Mapping[str, Any], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not group_members(root["pgrp"], root["session"], root["uid"], root["start"],
                             include_zombie=False):
            return True
        time.sleep(0.05)
    return False


def process_control_attacks() -> Dict[str, Any]:
    worker_source = textwrap.dedent(r'''
        import os, signal, subprocess, sys, time
        mode = sys.argv[1]
        if mode == "normal":
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.15)"])
            child.wait()
            raise SystemExit(0)
        ignore = mode in ("kill", "rss")
        if ignore:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        child_code = "import signal,time,sys; " + (
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore else "") + (
            "x=bytearray(24*1024*1024); time.sleep(60)")
        child = subprocess.Popen([sys.executable, "-c", child_code])
        x = bytearray(8*1024*1024)
        time.sleep(60)
    ''')
    outputs: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="m930_process_") as temporary:
        worker = Path(temporary) / "worker.py"
        worker.write_text(worker_source, encoding="utf-8")

        def launch(mode: str) -> Tuple[subprocess.Popen, Dict[str, Any]]:
            process = subprocess.Popen(
                [str(SETSID), "--wait", "/usr/bin/env", "-i",
                 "PATH=/usr/bin:/bin", "LANG=C.UTF-8", "LC_ALL=C.UTF-8",
                 "PYTHONDONTWRITEBYTECODE=1", str(PYTHON), str(worker), mode],
                cwd=str(HW), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                text=True)
            return process, capture_python_root(process)

        normal, normal_root = launch("normal")
        normal_rc = normal.wait(timeout=10)
        require(normal_rc == 0, "normal private group failed")
        require(wait_group_empty(normal_root), "normal private group not empty after reap")
        outputs["normal"] = {"root_pid": normal_root["pid"], "root_is_python": True,
                             "pid_equals_pgrp_sid": True, "root_reaped": True,
                             "group_empty": True, "return_code": normal_rc}

        term, term_root = launch("term")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and len(group_members(
                term_root["pgrp"], term_root["session"], term_root["uid"], term_root["start"])) < 2:
            time.sleep(0.02)
        require(len(group_members(term_root["pgrp"], term_root["session"],
                                  term_root["uid"], term_root["start"])) >= 2,
                "TERM test did not create group descendant")
        os.killpg(term_root["pgrp"], signal.SIGTERM)
        term_rc = term.wait(timeout=10)
        require(term_rc != 0 and wait_group_empty(term_root),
                "TERM private-group drain failed")
        outputs["term"] = {"root_pid": term_root["pid"], "whole_group_signaled": True,
                           "root_reaped": True, "group_empty": True,
                           "return_code": term_rc}

        kill, kill_root = launch("kill")
        deadline = time.monotonic() + 5
        rows: List[Dict[str, Any]] = []
        while time.monotonic() < deadline:
            rows = group_members(kill_root["pgrp"], kill_root["session"],
                                 kill_root["uid"], kill_root["start"])
            if len(rows) >= 2:
                break
            time.sleep(0.02)
        require(len(rows) >= 2, "KILL test did not create group descendant")
        os.killpg(kill_root["pgrp"], signal.SIGTERM)
        time.sleep(0.35)
        require(kill.poll() is None and group_members(
            kill_root["pgrp"], kill_root["session"], kill_root["uid"], kill_root["start"]),
            "TERM-ignore group did not survive grace")
        os.killpg(kill_root["pgrp"], signal.SIGKILL)
        kill_rc = kill.wait(timeout=10)
        require(kill_rc != 0 and wait_group_empty(kill_root),
                "KILL fallback private-group drain failed")
        outputs["term_to_kill"] = {"root_pid": kill_root["pid"],
                                   "term_ignored": True, "kill_whole_group": True,
                                   "root_reaped": True, "group_empty": True,
                                   "return_code": kill_rc}

        rss, rss_root = launch("rss")
        deadline = time.monotonic() + 5
        rss_rows: List[Dict[str, Any]] = []
        while time.monotonic() < deadline:
            rss_rows = group_members(rss_root["pgrp"], rss_root["session"],
                                     rss_root["uid"], rss_root["start"])
            if len(rss_rows) >= 2 and sum(row["rss_kib"] for row in rss_rows) > rss_root["rss_kib"]:
                break
            time.sleep(0.05)
        group_rss = sum(row["rss_kib"] for row in rss_rows)
        require(len(rss_rows) >= 2 and group_rss > rss_root["rss_kib"],
                "process-group RSS did not aggregate actual members")
        os.killpg(rss_root["pgrp"], signal.SIGKILL)
        rss.wait(timeout=10)
        require(wait_group_empty(rss_root), "RSS private group not drained")
        outputs["rss"] = {"member_count": len(rss_rows),
                          "root_rss_kib": rss_root["rss_kib"],
                          "group_rss_kib": group_rss,
                          "group_exceeds_root": True, "group_empty_after_reap": True}
    return outputs


def static_semantics(contract: Mapping[str, Any], request: Mapping[str, Any]) -> Dict[str, Any]:
    runner = RUNNER.read_text(encoding="utf-8")
    driver = DRIVER.read_text(encoding="utf-8")
    require('"${m925_setsid}" --wait /usr/bin/env -i' in runner,
            "exact direct setsid launch absent")
    require('"${m925_python}" "${m925_driver}" --run-full-first-row' in runner,
            "actual Python driver not direct worker")
    require("m925_run()" not in runner and "m925_worker()" not in runner,
            "background shell worker function reintroduced")
    require(runner.count("m925_driver_env --consume-attempt") == 1,
            "attempt consumption count drift")
    consume = runner.index("m925_driver_env --consume-attempt")
    started = runner.index("m925_started=1", consume)
    launch = runner.index('"${m925_setsid}" --wait', started)
    reap = runner.index("m925_reap_root", launch)
    drained = runner.index('[[ "${m925_tree_drained}" -eq 1', reap)
    seal_phase = runner.index("m925_phase=SEAL_AND_PUBLISH", drained)
    publish = runner.index("m925_driver_env --publish-no-replace", seal_phase)
    require(consume < started < launch < reap < drained < seal_phase < publish,
            "normal publication order drift")
    drain_body = runner[runner.index("m925_drain_job()"):
                        runner.index("m925_fail_closed()")]
    require(drain_body.index("kill -TERM") < drain_body.index("kill -KILL") <
            drain_body.index("m925_reap_root") < drain_body.index("m925_tree_drained=1"),
            "TERM/KILL/reap/drain order drift")
    failure_body = runner[runner.index("m925_fail_closed()"):
                          runner.index("trap m925_fail_closed EXIT")]
    require(failure_body.index("m925_drain_job") < failure_body.index("mv -T --no-clobber") <
            failure_body.index("--write-failure-receipt"),
            "failure drain/rename/receipt order drift")
    for trap in ("trap m925_fail_closed EXIT", "trap 'm925_signal=HUP; exit 129' HUP",
                 "trap 'm925_signal=INT; exit 130' INT",
                 "trap 'm925_signal=TERM; exit 143' TERM"):
        require(trap in runner, "signal trap absent: " + trap)
    require('[[ -f "${m925_drain_receipt}" && ! -L "${m925_drain_receipt}" ]]' in runner,
            "failure drain receipt regularity gate absent")
    for name in ("runtime_resource_snapshots.tsv", "worker_identity.txt",
                 "job_tree_drain_receipt.txt"):
        require(name in driver and name in runner, "sealed process evidence absent: " + name)
    require("seal_directory(output, tuple(members))" in driver,
            "failure receipt member seal absent")
    require("_rename_noreplace(stage, destination)" in driver,
            "no-replace publication absent")
    require(contract["max_future_attempts_after_separate_release_and_final_hammer"] == 1 and
            contract["future_gate_sequence"]["m928"].startswith("fresh independent final-launch") and
            request["semantic_checks"]["future_max_attempts"] == 1,
            "one-attempt/final-hammer semantics drift")
    require(contract["timing_contract"]["scientific_100x_hypothesis_already_failed_by_m900"] is True and
            contract["timing_contract"]["r2_objective_is_100x_retry"] is False and
            contract["timing_contract"]["operational_safety_timeout_seconds"] == 2715,
            "scientific/operational threshold separation drift")
    false_keys = ("full_first_row", "full_population", "production", "decoder_complete",
                  "cycles_or_speedup_citable", "system_speedup", "energy",
                  "paper_ppa_ready", "paper_citable", "vcs_eda_license_gpu_remote_network")
    for key in false_keys:
        require(contract.get(key) is False, "source contract claim drift: " + key)
    return {"direct_actual_python_private_group": True,
            "background_shell_function_pid_forbidden": True,
            "consume_before_worker": True,
            "normal_reap_group_empty_before_seal_publish": True,
            "failure_term_kill_reap_before_rename_receipt": True,
            "exit_hup_int_term_traps_drain": True,
            "failure_receipt_binds_worker_identity_group_rss_drain": True,
            "renameat2_noreplace_publication": True,
            "future_attempts": 1,
            "scientific_100x_threshold_seconds": 9.320783571209759,
            "scientific_threshold_historical_status": "FAILED_BY_M900__NOT_RETRIED",
            "operational_safety_timeout_seconds": 2715,
            "all_publication_claims_false": True}


def main() -> int:
    started = time.monotonic()
    output: Dict[str, Any] = {
        "schema": "m930_m925_decoder_gtls_r2_source_fresh_hammer_output_v1",
        "date": "2026-08-29", "status": "FAIL_CLOSED_PENDING"}
    try:
        require(Path(sys.executable).resolve() == PYTHON.resolve(),
                "hammer must use pinned Python 3.10")
        require(not result_namespace(), "M925 namespace occupied before hammer")
        identities: Dict[str, str] = {}
        for path, expected in EXPECTED.items():
            require(path.is_file() and not path.is_symlink(), "input absent/symlink: " + str(path))
            require(sha256(path) == expected, "input SHA drift: " + str(path))
            identities[str(path.relative_to(HW)) if path.is_relative_to(HW) else str(path)] = expected
        output["identities"] = identities
        output["sidecars"] = {str(path.relative_to(HW)): verify_sidecar(path, expected)
                              for path, expected in EXPECTED_SIDECARS.items()}
        output["sealed_directories"] = {
            str(path.relative_to(HW)): verify_sealed_dir(path, expected)
            for path, expected in EXPECTED_SEALS.items()}
        contract = strict_load(CONTRACT)
        request = strict_load(REQUEST / "request.json")
        output["compile_checks"] = compile_sources()
        output["static_semantics"] = static_semantics(contract, request)
        output["strict_json_and_rename_attacks"] = target_json_and_rename_attacks()
        output["runner_refusal_attacks"] = run_refusal_attacks()
        output["m896_pytest"] = run_m896_pytest()
        output["process_control_attacks"] = process_control_attacks()
        require(not result_namespace(), "M925 namespace occupied after hammer")
        require(sha256(M900_ATTEMPT / "attempt.json") == EXPECTED[M900_ATTEMPT / "attempt.json"],
                "M900 consumed attempt modified")
        require(sha256(M900_FAILURE / "failure.json") == EXPECTED[M900_FAILURE / "failure.json"],
                "M900 failure receipt modified")
        require(sha256(DOCS359) == EXPECTED[DOCS359], "docs359 changed during hammer")
        output["postconditions"] = {
            "m925_result_attempt_stage_log_failure_namespace_absent": True,
            "m900_consumed_attempt_unmodified": True,
            "m900_failure_unmodified": True,
            "future_release_absent": not FUTURE_RELEASE.exists(),
            "no_full_row_or_attempt": True,
            "no_eda_vcs_gpu_remote_network": True,
            "docs359_sha256": EXPECTED[DOCS359]}
        output["status"] = "PASS100_M925_R2_SOURCE_PROCESS_CONTROL__ONLY_FRESH_M927_INERT_RELEASE_AUTHOR_AUTHORIZED"
        output["score"] = 100
        output["severity_counts"] = {"p0": 0, "p1": 0, "p2": 0}
        output["elapsed_seconds"] = round(time.monotonic() - started, 6)
    except Exception as error:
        output["status"] = "FAIL_M930_M925_SOURCE_HAMMER"
        output["score"] = 0
        output["severity_counts"] = {"p0": 1, "p1": 0, "p2": 0}
        output["error"] = type(error).__name__ + ": " + str(error)
        output["elapsed_seconds"] = round(time.monotonic() - started, 6)
        (HERE / "independent_hammer_output.json").write_text(
            json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        raise
    (HERE / "independent_hammer_output.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(output["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
