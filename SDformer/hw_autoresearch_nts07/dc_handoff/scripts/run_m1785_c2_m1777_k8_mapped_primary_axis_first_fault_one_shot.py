#!/usr/bin/env python3
"""One-shot M1785 mapped K8 first-fault diagnostic; M1786-gated."""

from __future__ import print_function

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


HW = Path(__file__).resolve().parents[2]
SELF = Path(__file__).resolve()
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
FILELIST = HW / "dc_handoff/filelists/iscas_m1785_c2_m1777_k8_mapped_primary_axis_first_fault_diagnostic.f"
TB = HW / "dc_handoff/tb/tb_m1785_c2_m1777_k8_mapped_primary_axis_first_fault_diagnostic.sv"
CHECKER = HW / "system_simulator/scripts/check_m1785_c2_m1777_k8_mapped_primary_axis_first_fault_source.py"
TEST = HW / "system_simulator/tests/test_m1785_c2_m1777_k8_mapped_primary_axis_first_fault_source.py"
CONTRACT = HW / "contracts/m1785_c2_m1777_k8_mapped_primary_axis_first_fault_source_contract_r1_20260902.json"
REVIEW = HW / "reviews/m1786_m1785_c2_m1777_k8_mapped_primary_axis_first_fault_source_hammer_r1_20260902"
RESULT = HW / "results/m1785_c2_m1777_k8_mapped_primary_axis_first_fault_r1_20260902"
ATTEMPT = HW / "results/.m1785_c2_m1777_k8_mapped_primary_axis_first_fault_attempt_consumed"
LOCK = Path("/tmp/date_dual_synopsys_same_uid_eda_queue.lock")
TOP = "tb_m1785_c2_m1777_k8_mapped_primary_axis_first_fault_diagnostic"

IMMUTABLE = {
    "vcs": (VCS, "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287"),
    "m1661_k8_netlist": (HW / "dc_handoff/runs/m1661_m1652_c2_resource_gate_successor_three_axis_logic_only_dc_3p000ns_r1_20260901/k8/netlist/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.v", "6c62d99b444ba25f8eb3f1e491479b44f5613b0323e032af8150e81c84f393c4"),
    "memory": (HW / "dc_handoff/tb/m1334_c2_production_activity_reset_safe_memory_model.sv", "f9b0d87dd3b951a24b79545555c09b32bbce695e85cc71df2948e5065981c7c3"),
    "m979_tb": (HW / "dc_handoff/tb/tb_m979_c2_three_axis_mapped_gate_case_saif.sv", "cce12a93c4c8fd8d424fbf9f6354ba30e2870a05a7480fc7de26b3b29c87266c"),
    "m1334_assert": (HW / "dc_handoff/tb/m1334_c2_production_activity_assertions.sv", "86be3fa541bf65afa6ada99aa3e2bd494ed689594fece18cfea135b91420c32a"),
    "m1684_assert": (HW / "dc_handoff/tb/m1684_c2_m1609_production_binary_fault_assertions.sv", "39fdc0f47628272a6f1a7b6887da52fdbf4d71f1f5fe6557d4a7022f06bc62b1"),
    "m1684_wrapper": (HW / "dc_handoff/tb/tb_m1684_c2_m1609_fresh_mapped_production_energy.sv", "034934d1cdb6dc683ffa51811bd363fadd02673a1311a5b715b5c4b0e3cb5a2e"),
    "docs359": (HW / "docs/359_DATE终局冻结_20260813.md", "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"),
}


class Failure(RuntimeError):
    pass


def need(value, message):
    if not value:
        raise Failure(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(path, expected):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "missing/nonregular: " + str(path))
    need(sha256(path) == expected, "SHA mismatch: " + str(path))


def strict_json(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            need(key not in value, "duplicate key: " + key)
            value[key] = item
        return value
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=pairs,
                         parse_constant=lambda token: (_ for _ in ()).throw(
                             Failure("nonfinite JSON: " + token)))


def verify_file_seal(path):
    path = Path(path)
    exact(path, sha256(path))
    digest = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    exact(digest, sha256(digest))
    exact(outer, sha256(outer))
    need(digest.read_text(encoding="utf-8").strip()
         == sha256(path) + "  " + path.name, "file digest")
    need(outer.read_text(encoding="utf-8").strip()
         == sha256(digest) + "  " + digest.name, "outer file seal")


def verify_dir_seal(path):
    path = Path(path)
    need(path.is_dir() and not path.is_symlink(), "sealed dir absent: " + str(path))
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    exact(manifest, sha256(manifest))
    exact(outer, sha256(outer))
    need(outer.read_text(encoding="utf-8").strip()
         == sha256(manifest) + "  SHA256SUMS", "directory outer seal")
    listed = set()
    for row in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = row.split("  ", 1)
        member = path / name
        exact(member, digest)
        listed.add(name)
    actual = set(member.relative_to(path).as_posix() for member in path.rglob("*")
                 if member.is_file() and member.name not in
                 {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    need(actual == listed, "sealed directory population")


def collision_gate():
    blocked = {"vcs", "vcs1", "vlogan", "simv", "dc_shell", "pt_shell", "fm_shell"}
    ancestry = set()
    pid = os.getpid()
    while pid > 1 and pid not in ancestry:
        ancestry.add(pid)
        try:
            pid = int((Path("/proc") / str(pid) / "stat").read_text().split()[3])
        except Exception:
            break
    hits = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in ancestry:
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            comm = (proc / "comm").read_text().strip()
            argv = {Path(item.decode(errors="replace")).name
                    for item in (proc / "cmdline").read_bytes().split(b"\0") if item}
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if comm in blocked or blocked.intersection(argv):
            hits.append((proc.name, comm, sorted(argv)))
    need(not hits, "same-UID EDA collision: " + repr(hits))


def seal_dir(path):
    rows = []
    for member in sorted((item for item in path.rglob("*") if item.is_file()),
                         key=lambda item: item.relative_to(path).as_posix()):
        if member.name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        rows.append(sha256(member) + "  " + member.relative_to(path).as_posix())
    (path / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (path / "SHA256SUMS.seal.sha256").write_text(
        sha256(path / "SHA256SUMS") + "  SHA256SUMS\n", encoding="utf-8")
    verify_dir_seal(path)


def main():
    need(len(sys.argv) == 1, "M1785 accepts no arguments")
    expected_self = os.environ.get("M1785_EXPECTED_RUNNER_SHA256", "")
    expected_review = os.environ.get("M1785_EXPECTED_M1786_REVIEW_SHA256", "")
    need(expected_self and sha256(SELF) == expected_self,
         "caller did not pin reviewed runner")
    for _name, pair in IMMUTABLE.items():
        exact(pair[0], pair[1])
    verify_file_seal(CONTRACT)
    verify_dir_seal(REVIEW)
    review_path = REVIEW / "review.json"
    need(expected_review and sha256(review_path) == expected_review,
         "caller did not pin M1786 review")
    review = strict_json(review_path)
    need(review["status"] ==
         "PASS_M1786_M1785_C2_K8_MAPPED_FIRST_FAULT_SOURCE_HAMMER__AUTHORIZE_ONE_ATTEMPT",
         "M1786 did not authorize")
    need(review["p0_count"] == 0 and review["p1_count"] == 0
         and review["score_over_100"] >= 95, "M1786 quality gate")
    identities = review["source_identity"]
    for key, path in (("runner_sha256", SELF), ("tb_sha256", TB),
                      ("filelist_sha256", FILELIST),
                      ("checker_sha256", CHECKER), ("test_sha256", TEST),
                      ("contract_sha256", CONTRACT)):
        exact(path, identities[key])
    contract = strict_json(CONTRACT)
    need(contract["future_execution"]["authorized_now"] is False,
         "source contract unexpectedly self-authorizes")

    spec = importlib.util.spec_from_file_location("m1785_runtime_check", str(CHECKER))
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    checker.main()

    need(not os.path.lexists(str(RESULT)) and not os.path.lexists(str(ATTEMPT)),
         "attempt/result namespace not fresh")
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        collision_gate()
        ATTEMPT.mkdir()
        (ATTEMPT / "attempt.json").write_text(json.dumps({
            "schema": "m1785_attempt_r1_v1", "attempt_consumed": True,
            "automatic_retry": False, "runner_sha256": sha256(SELF),
            "m1786_review_sha256": sha256(review_path)},
            sort_keys=True, indent=2) + "\n", encoding="utf-8")
        work = Path(tempfile.mkdtemp(prefix=".m1785_c2_first_fault_work.",
                                    dir=str(HW / "results")))
        try:
            compile_log = work / "compile.log"
            sim_log = work / "sim.log"
            simv = work / "simv"
            compile_command = [str(VCS), "-full64", "-sverilog",
                "-assert", "svaext", "-timescale=1ns/1ps",
                "-Mdir=" + str(work / "csrc"), "-f", str(FILELIST),
                "-top", TOP, "-o", str(simv)]
            collision_gate()
            with compile_log.open("wb") as output:
                compiled = subprocess.run(compile_command, cwd=str(HW),
                                          stdout=output, stderr=subprocess.STDOUT)
            need(compiled.returncode == 0 and simv.is_file(), "VCS compile failed")
            sim_command = [str(simv), "+M979_CASE=0", "+ntb_random_seed=1785",
                           "-assert", "report=" + str(work / "assert.report")]
            collision_gate()
            with sim_log.open("wb") as output:
                simulated = subprocess.run(sim_command, cwd=str(HW),
                                           stdout=output, stderr=subprocess.STDOUT)
            text = sim_log.read_text(encoding="utf-8", errors="replace")
            need(text.count("Fatal:") == 1 and "watchdog" not in text.lower(),
                 "unexpected fatal population")
            localized = checker.check_runtime_text(text)
            receipt = {
                "schema": "m1785_c2_m1777_k8_first_fault_result_r1_v1",
                "status": "PASS_DIAGNOSTIC_ONLY__M1777_K8_XZ_REPRODUCED_AND_LOCALIZED",
                "counts": {"vcs_compiles": 1, "simv_runs": 1,
                           "saif_files": 0, "ptpx_runs": 0},
                "simv_returncode": simulated.returncode,
                "localization": localized,
                "exact_m1684_assertion_preserved": True,
                "initreg": False, "force": False, "ignore_x": False,
                "mapped_functionality": False, "power": False,
                "energy": False, "paper_citable": False,
                "automatic_retry": False,
            }
            (work / "receipt.json").write_text(
                json.dumps(receipt, sort_keys=True, indent=2) + "\n",
                encoding="utf-8")
            (work / "RUN_COMPLETE.txt").write_text(
                "PASS_M1785_DIAGNOSTIC_ONLY_M1777_K8_XZ_LOCALIZED\n",
                encoding="utf-8")
            seal_dir(work)
            need(not os.path.lexists(str(RESULT)), "result appeared")
            os.rename(str(work), str(RESULT))
        except BaseException:
            quarantine = Path(str(RESULT) + ".failed_or_incomplete.quarantine")
            if work.exists() and not quarantine.exists():
                os.rename(str(work), str(quarantine))
            raise
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
    print("PASS_M1785_DIAGNOSTIC_ONLY_M1777_K8_XZ_LOCALIZED")


if __name__ == "__main__":
    try:
        main()
    except Failure as error:
        print("M1785_FAILURE: " + str(error), file=sys.stderr)
        raise SystemExit(3)
