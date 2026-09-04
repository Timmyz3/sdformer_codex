#!/usr/bin/env python3
"""Independent M862 execution of the M859/C2 R25 source hammer.

This program is source/synthetic-filesystem only.  It never invokes VCS,
simv, lmutil, DC, Formality, PT, PTPX, a CPU workload, a GPU workload, or a
remote job.  Python 3.6 compatibility is intentional because the release
runner pins platform-python3.6.
"""

import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
sys.path.insert(0, str(HW / "verif_m859"))
import m859_c2_r25_shared_whitelist_guard as guard  # noqa: E402


RUNNER = HW / "dc_handoff/scripts/run_vcs_m859_c2_r25_shared_whitelist_exact_sha.sh"
R24_RUNNER = HW / "dc_handoff/scripts/run_vcs_m851_c2_r24_recursive_seal_exact_sha.sh"
CONTRACT = HW / "contracts/m859_c2_r25_shared_whitelist_source_only_contract_r1_20260829.json"
CANDIDATE = HW / "contracts/m859_c2_r25_shared_whitelist_vcs_launch_candidate_source_only_r1_20260829.json"
M856 = HW / "reviews/m856_m851_c2_r24_recursive_seal_source_fresh_hammer_r1_20260829"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

RUNNER_SHA = "da423b17f6245b0e9af9cc6df05a846e221175da45bfbce9408fe91930a9f8d6"
GUARD_SHA = "5622a5ade16c18091e7f2facd37bcd3d39565c1c7bf6fe694f5c1362fc07e224"
CONTRACT_SHA = "a7458798f11b0ba02d83072d93cf6185508de0e882eb9bf4c02a0b7380e66c5f"
CANDIDATE_SHA = "bf8599efc0ebce9b7e11b6d2ca38061b869c6555bddd620acb93a0ae3332696e"
M856_REVIEW_SHA = "829aa823f431dc64727e3f16efcd1a200fc9e797846b8b5030957400e9362f1b"
M856_MANIFEST_SHA = "1115ae859642950870b6370acf61f50cd2780c66b0f4d098d9f598bd4c6b5903"
M856_OUTER_SHA = "96fd220ea2061390dccb0563ce3e0592a5d6ea0d7f0b067146032e32eeccda67"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

# Deliberately independent of guard.WHITELIST.  This is the exact population
# emitted by the two compile_and_run calls in the real runner.
PHASE_FILES = (
    "attack/compile.log", "attack/compile.rc",
    "attack/sim.log", "attack/sim.rc",
    "attack/assert.report", "attack/assert.report.disablelog",
    "equalbw/compile.log", "equalbw/compile.rc",
    "equalbw/sim.log", "equalbw/sim.rc",
    "equalbw/assert.report", "equalbw/assert.report.disablelog",
)
RUNNER_EVIDENCE = (
    "RUN_COMPLETE.txt", "launch_identity.txt",
    "m859_c2_r25_shared_whitelist_vcs_receipt_r1.json",
) + PHASE_FILES

EXPECTED = {
    "positive_runner_pipeline": "PASS",
    "shared_authority_static": "PASS",
    "obsolete_r24_receipt_substitution": "REJECT",
    "wrong_receipt_schema": "REJECT",
    "wrong_receipt_status": "REJECT",
    "missing_file": "REJECT",
    "extra_file": "REJECT",
    "extra_empty_directory": "REJECT",
    "nested_depth_drift": "REJECT",
    "source_file_symlink": "REJECT",
    "source_directory_symlink": "REJECT",
    "sealed_root_symlink": "REJECT",
    "sealed_nested_file_symlink": "REJECT",
    "source_path_toctou": "REJECT",
    "publish_path_toctou": "REJECT",
    "payload_mutation": "REJECT",
    "manifest_mutation": "REJECT",
    "outer_seal_mutation": "REJECT",
    "destination_collision": "REJECT",
    "receipt_writer_existing_file": "REJECT",
    "receipt_writer_symlink": "REJECT",
    "receipt_writer_bad_sha": "REJECT",
}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def rejected(callback):
    try:
        callback()
    except (guard.base.Failure, OSError, ValueError, RuntimeError):
        return "REJECT"
    return "ACCEPT"


def write_receipt(work):
    return guard.write_pending_receipt(
        work, "1" * 64, "2" * 64, "3" * 64, "4" * 64, "5" * 64)


def emit_runner_style(work, tool_extras=True):
    """Emit the real runner shape without using the guard whitelist."""
    work.mkdir()
    (work / "launch_identity.txt").write_text(
        "runner_sha256=" + "1" * 64 + "\n"
        "contract_sha256=" + "2" * 64 + "\n"
        "candidate_sha256=" + "3" * 64 + "\n"
        "release_sha256=" + "4" * 64 + "\n"
        "final_hammer_outer_seal_sha256=" + "5" * 64 + "\n",
        encoding="utf-8")
    for index, relative in enumerate(PHASE_FILES):
        path = work / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((relative + "\n").encode("utf-8") + bytes([index]))
    write_receipt(work)
    (work / "RUN_COMPLETE.txt").write_text(
        guard.RUN_COMPLETE_STATUS + "\n", encoding="utf-8")
    if tool_extras:
        # Ordinary VCS output is intentionally private and may contain links.
        tool = work / "attack" / "simv.daidir"
        tool.mkdir()
        (tool / "archive.so").write_bytes(b"private VCS artifact\n")
        os.symlink("simv.daidir/archive.so",
                   str(work / "attack" / "tool_archive.so"))


def sealed(root):
    work = root / "work"
    emit_runner_style(work)
    stage = root / "stage"
    guard.stage_result_whitelist(work, stage)
    guard.base.seal_directory(stage)
    return work, stage


def function_block(text, name, next_name):
    start = text.index(name + "() {")
    end = text.index(next_name + "() {", start)
    return text[start:end]


def between(text, begin, end):
    start = text.index(begin)
    stop = text.index(end, start)
    return text[start:stop]


def formal_population():
    results = HW / "results"
    names = []
    for path in results.iterdir():
        name = path.name
        if (name.startswith("m859_c2_r25_shared_whitelist_vcs_r1_20260829") or
                name.startswith(".m859_c2_r25_shared_whitelist_vcs_")):
            names.append(name)
    return sorted(names)


def run_matrix():
    results = {}

    # Positive path: the producer is independent of WHITELIST, then the exact
    # same receipt writer/stager/verifier/publisher used by the runner is used.
    with tempfile.TemporaryDirectory(prefix="m862_positive.") as raw:
        root = Path(raw)
        work = root / "work"
        emit_runner_style(work)
        explicit = set(RUNNER_EVIDENCE)
        require(explicit == set(guard.WHITELIST),
                "runner population differs from shared whitelist")
        stage = root / "stage"
        staged = guard.stage_result_whitelist(work, stage)
        require(staged["member_count"] == 15, "stage count drift")
        guard.base.seal_directory(stage)
        before = guard.verify_recursive_sealed_directory(stage)
        destination = root / "canonical"
        published = guard.publish_recursive_noreplace(stage, destination)
        post = guard.verify_recursive_sealed_directory(destination)
        actual = {
            p.relative_to(destination).as_posix()
            for p in destination.rglob("*") if p.is_file()
        }
        require(before == published == post, "publication identity drift")
        require(actual == set(guard.RESULT_MEMBERS),
                "canonical exact population drift")
        require(not stage.exists(), "stage survived no-replace publication")
        require((work / "attack/tool_archive.so").is_symlink(),
                "private tool symlink fixture missing")
        require(not (destination / "attack/tool_archive.so").exists(),
                "private tool symlink escaped whitelist")
        results["positive_runner_pipeline"] = "PASS"

    # Static single-authority proof.  Candidate duplicates the list only as a
    # sealed declaration; all executable consumers reference guard globals.
    guard_text = (HW / "verif_m859/m859_c2_r25_shared_whitelist_guard.py").read_text(
        encoding="utf-8")
    runner_text = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(guard_text)
    functions = {node.name: node for node in tree.body
                 if isinstance(node, ast.FunctionDef)}
    loads = {}
    for name in ("write_pending_receipt", "stage_result_whitelist",
                 "verify_recursive_sealed_directory",
                 "publish_recursive_noreplace"):
        loads[name] = {node.id for node in ast.walk(functions[name])
                       if isinstance(node, ast.Name) and
                       isinstance(node.ctx, ast.Load)}
    candidate = guard.base.strict_json(CANDIDATE)
    require(len(guard.WHITELIST) == len(set(guard.WHITELIST)) == 15,
            "whitelist count/uniqueness drift")
    require(tuple(candidate["single_shared_whitelist"]) == guard.WHITELIST,
            "candidate whitelist declaration drift")
    require(guard.RESULT_MEMBERS == guard.WHITELIST +
            ("SHA256SUMS", "SHA256SUMS.seal.sha256"),
            "recursive member derivation drift")
    require("RECEIPT_FILENAME" in loads["write_pending_receipt"] and
            "RECEIPT_SCHEMA" in loads["write_pending_receipt"] and
            "RECEIPT_STATUS" in loads["write_pending_receipt"],
            "receipt writer does not consume central identity")
    require("WHITELIST" in loads["stage_result_whitelist"],
            "stager does not consume central whitelist")
    require("RESULT_MEMBERS" in loads["verify_recursive_sealed_directory"],
            "recursive verifier does not consume derived members")
    require(runner_text.count("write-pending-receipt") == 1 and
            runner_text.count("stage-result-whitelist") == 1 and
            runner_text.count("verify-recursive-sealed-directory") == 2 and
            runner_text.count("publish-recursive-no-replace") == 1,
            "runner publication call graph drift")
    require(guard.RECEIPT_FILENAME not in runner_text and
            "m851_c2_r24_recursive_seal_vcs_receipt_r1.json" not in runner_text,
            "runner carries a private receipt filename authority")
    results["shared_authority_static"] = "PASS"

    with tempfile.TemporaryDirectory(prefix="m862_old_receipt.") as raw:
        root = Path(raw)
        work = root / "work"
        emit_runner_style(work)
        current = work / guard.RECEIPT_FILENAME
        current.rename(work / "m851_c2_r24_recursive_seal_vcs_receipt_r1.json")
        results["obsolete_r24_receipt_substitution"] = rejected(
            lambda: guard.stage_result_whitelist(work, root / "stage"))

    for key, wrong, label in (
            ("schema", "wrong_schema", "wrong_receipt_schema"),
            ("status", "wrong_status", "wrong_receipt_status")):
        with tempfile.TemporaryDirectory(prefix="m862_" + label + ".") as raw:
            root = Path(raw)
            work = root / "work"
            emit_runner_style(work)
            path = work / guard.RECEIPT_FILENAME
            value = json.loads(path.read_text(encoding="utf-8"))
            value[key] = wrong
            path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
            results[label] = rejected(
                lambda: guard.stage_result_whitelist(work, root / "stage"))

    with tempfile.TemporaryDirectory(prefix="m862_source_file_link.") as raw:
        root = Path(raw)
        work = root / "work"
        emit_runner_style(work)
        victim = work / "RUN_COMPLETE.txt"
        victim.rename(work / "RUN_COMPLETE.real")
        os.symlink("RUN_COMPLETE.real", str(victim))
        results["source_file_symlink"] = rejected(
            lambda: guard.stage_result_whitelist(work, root / "stage"))

    with tempfile.TemporaryDirectory(prefix="m862_source_dir_link.") as raw:
        root = Path(raw)
        work = root / "work"
        emit_runner_style(work)
        (work / "equalbw").rename(work / "equalbw.real")
        os.symlink("equalbw.real", str(work / "equalbw"))
        results["source_directory_symlink"] = rejected(
            lambda: guard.stage_result_whitelist(work, root / "stage"))

    with tempfile.TemporaryDirectory(prefix="m862_root_link.") as raw:
        root = Path(raw)
        _, stage = sealed(root)
        stage.rename(root / "real")
        os.symlink("real", str(stage))
        results["sealed_root_symlink"] = rejected(
            lambda: guard.verify_recursive_sealed_directory(stage))

    with tempfile.TemporaryDirectory(prefix="m862_nested_link.") as raw:
        root = Path(raw)
        _, stage = sealed(root)
        victim = stage / "attack/compile.log"
        victim.rename(stage / "attack/compile.real")
        os.symlink("compile.real", str(victim))
        results["sealed_nested_file_symlink"] = rejected(
            lambda: guard.verify_recursive_sealed_directory(stage))

    with tempfile.TemporaryDirectory(prefix="m862_source_toctou.") as raw:
        root = Path(raw)
        work = root / "work"
        emit_runner_style(work)
        original = guard.r848._hash_open_file
        calls = [0]

        def swap_after_first(handle):
            value = original(handle)
            calls[0] += 1
            if calls[0] == 1:
                path = work / "RUN_COMPLETE.txt"
                path.rename(work / "RUN_COMPLETE.old")
                path.write_bytes(b"replacement\n")
            return value

        guard.r848._hash_open_file = swap_after_first
        try:
            results["source_path_toctou"] = rejected(
                lambda: guard.stage_result_whitelist(work, root / "stage"))
        finally:
            guard.r848._hash_open_file = original

    with tempfile.TemporaryDirectory(prefix="m862_publish_toctou.") as raw:
        root = Path(raw)
        _, stage = sealed(root)
        attacker_work = root / "attacker_work"
        emit_runner_style(attacker_work)
        (attacker_work / "attack/compile.log").write_bytes(b"attacker\n")
        attacker = root / "attacker"
        guard.stage_result_whitelist(attacker_work, attacker)
        guard.base.seal_directory(attacker)
        original = guard.base._rename_noreplace

        def swap_then_rename(source, destination):
            source = Path(source)
            source.rename(root / "verified_old")
            attacker.rename(source)
            return original(source, destination)

        guard.base._rename_noreplace = swap_then_rename
        try:
            results["publish_path_toctou"] = rejected(
                lambda: guard.publish_recursive_noreplace(
                    stage, root / "canonical"))
        finally:
            guard.base._rename_noreplace = original

    mutators = {
        "missing_file": lambda stage:
            (stage / "equalbw/compile.log").unlink(),
        "extra_file": lambda stage:
            (stage / "attack/extra.txt").write_bytes(b"x"),
        "extra_empty_directory": lambda stage:
            (stage / "empty").mkdir(),
        "payload_mutation": lambda stage:
            (stage / "attack/compile.log").write_bytes(b"changed"),
        "manifest_mutation": lambda stage:
            (stage / "SHA256SUMS").write_bytes(b"changed"),
        "outer_seal_mutation": lambda stage:
            (stage / "SHA256SUMS.seal.sha256").write_bytes(b"changed"),
    }
    for label, mutate in mutators.items():
        with tempfile.TemporaryDirectory(prefix="m862_" + label + ".") as raw:
            root = Path(raw)
            _, stage = sealed(root)
            mutate(stage)
            results[label] = rejected(
                lambda: guard.verify_recursive_sealed_directory(stage))

    with tempfile.TemporaryDirectory(prefix="m862_depth.") as raw:
        root = Path(raw)
        work = root / "work"
        emit_runner_style(work)
        stage = root / "stage"
        guard.stage_result_whitelist(work, stage)
        (stage / "attack/deep").mkdir()
        (stage / "attack/compile.log").rename(
            stage / "attack/deep/compile.log")
        guard.base.seal_directory(stage)
        results["nested_depth_drift"] = rejected(
            lambda: guard.verify_recursive_sealed_directory(stage))

    with tempfile.TemporaryDirectory(prefix="m862_collision.") as raw:
        root = Path(raw)
        _, stage = sealed(root)
        destination = root / "canonical"
        destination.mkdir()
        marker = destination / "marker"
        marker.write_bytes(b"preserve\n")
        outcome = rejected(lambda: guard.publish_recursive_noreplace(
            stage, destination))
        results["destination_collision"] = (
            outcome if marker.read_bytes() == b"preserve\n" and stage.is_dir()
            else "CLOBBER")

    with tempfile.TemporaryDirectory(prefix="m862_writer_exists.") as raw:
        work = Path(raw) / "work"
        work.mkdir()
        (work / guard.RECEIPT_FILENAME).write_bytes(b"existing\n")
        results["receipt_writer_existing_file"] = rejected(
            lambda: write_receipt(work))

    with tempfile.TemporaryDirectory(prefix="m862_writer_link.") as raw:
        work = Path(raw) / "work"
        work.mkdir()
        (work / "target").write_bytes(b"preserve\n")
        os.symlink("target", str(work / guard.RECEIPT_FILENAME))
        results["receipt_writer_symlink"] = rejected(
            lambda: write_receipt(work))

    with tempfile.TemporaryDirectory(prefix="m862_writer_bad_sha.") as raw:
        work = Path(raw) / "work"
        work.mkdir()
        results["receipt_writer_bad_sha"] = rejected(
            lambda: guard.write_pending_receipt(
                work, "not-a-sha", "2" * 64, "3" * 64,
                "4" * 64, "5" * 64))

    return results


def source_checks():
    require(sha256(RUNNER) == RUNNER_SHA, "runner SHA drift")
    require(sha256(HW / "verif_m859/m859_c2_r25_shared_whitelist_guard.py") ==
            GUARD_SHA, "guard SHA drift")
    require(sha256(CONTRACT) == CONTRACT_SHA, "contract SHA drift")
    require(sha256(CANDIDATE) == CANDIDATE_SHA, "candidate SHA drift")
    require(sha256(M856 / "review.json") == M856_REVIEW_SHA,
            "M856 review SHA drift")
    require(sha256(M856 / "SHA256SUMS") == M856_MANIFEST_SHA,
            "M856 manifest SHA drift")
    require(sha256(M856 / "SHA256SUMS.seal.sha256") == M856_OUTER_SHA,
            "M856 outer seal SHA drift")
    require(sha256(DOCS359) == DOCS359_SHA, "docs/359 SHA drift")
    m856_identity = guard.base.verify_sealed_directory(M856)
    require(m856_identity["manifest_sha256"] == M856_MANIFEST_SHA and
            m856_identity["outer_seal_file_sha256"] == M856_OUTER_SHA,
            "M856 seal identity drift")
    m856 = guard.base.strict_json(M856 / "review.json")
    require(m856["status"] == guard.M856_STATUS and
            m856["score_out_of_100"] == 88 and
            (m856["p0_count"], m856["p1_count"], m856["p2_count"]) ==
            (1, 0, 0) and
            m856["claim_boundary"]["source_gate_passed"] is False and
            m856["claim_boundary"]["release_authorized"] is False,
            "M856 negative authority semantics drift")

    guard.base.verify_double_sealed_file(CONTRACT)
    guard.base.verify_double_sealed_file(CANDIDATE)
    source = guard.validate_source(HW, CONTRACT, CANDIDATE, RUNNER)
    require(source["status"] ==
            "PASS_M859_R25_SHARED_WHITELIST_SOURCE__NO_VCS_OR_EDA",
            "formal source validation status drift")

    contract = guard.base.strict_json(CONTRACT)
    for relative, expected in contract["source_sha256"].items():
        require(sha256(HW / relative) == expected,
                "contract source SHA drift: " + relative)
    m803 = [relative for relative in contract["source_sha256"]
            if relative.startswith(("rtl_m803/", "tb_m803/", "verif_m803/")) or
            relative.startswith("dc_handoff/filelists/date_m803")]
    require(len(m803) == 9, "M803 frozen source population drift")

    old = R24_RUNNER.read_text(encoding="utf-8")
    new = RUNNER.read_text(encoding="utf-8")
    old_compile = function_block(old, "compile_and_run", "publish_failure_receipt")
    new_compile = function_block(new, "compile_and_run", "publish_failure_receipt")
    require(old_compile == new_compile, "compile_and_run changed R24 to R25")
    old_matrix = between(old, "log_phase ATTACK_VCS", "log_phase RESULT_STAGE_SEAL")
    new_matrix = between(new, "log_phase ATTACK_VCS", "log_phase RESULT_STAGE_SEAL")
    require(old_matrix == new_matrix,
            "attack/equal-bandwidth commands or gates changed R24 to R25")
    require(hashlib.sha256(old_compile.encode("utf-8")).hexdigest() ==
            "b6f6753be90a2f5c8ab5a3ab2e7acdb1095bc2d31fe033fe7486ad4b998d9ad2",
            "compile block anchor drift")
    require(hashlib.sha256(old_matrix.encode("utf-8")).hexdigest() ==
            "261d47f0a57fd76176c63961b472d353f7e294e9b40a05a8571bed475005ef14",
            "attack/equal block anchor drift")
    require(contract["repair_scope"]["frozen_exact_cycles"] ==
            "51/53,131/133,486/499,1231/1246,14/14",
            "frozen cycle declaration drift")
    return source, m803


def source_dry_run():
    before = formal_population()
    require(before == [], "formal R25 identity exists before source hammer")
    with tempfile.TemporaryDirectory(prefix="m862_source_dry.") as raw:
        root = Path(raw)
        nonce = root / "nonce"
        trace = root / "trace.jsonl"
        nonce.write_text("M859_R25_SOURCE_HAMMER_ONLY\n", encoding="utf-8")
        env = {
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
            "M859_R25_EXPECTED_VCS_RUNNER_SHA256": RUNNER_SHA,
            "M859_R25_SOURCE_DRY_RUN": "1",
            "M859_R25_SOURCE_DRY_RUN_ROOT": str(root),
            "M859_R25_SOURCE_DRY_RUN_NONCE": str(nonce),
            "M859_R25_SOURCE_DRY_RUN_TRACE": str(trace),
        }
        completed = subprocess.run(
            ["/bin/bash", str(RUNNER)], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        require(completed.returncode == 86,
                "source dry-run rc drift: {} stderr={}".format(
                    completed.returncode,
                    completed.stderr.decode("utf-8", "replace")))
        require(b"STUB_REACHED_LIVE_VCS_LICENSE_BOUNDARY" in completed.stdout,
                "source dry-run stop marker absent")
        lines = trace.read_text(encoding="utf-8").splitlines()
        require(len(lines) == 4, "source dry-run event count drift")
        events = [json.loads(line) for line in lines]
        require([event["event"] for event in events] == [
            "source_contract_verified",
            "m856_negative_source_and_predecessor_authorities_verified",
            "whitelist_guard_selftest",
            "live_vcs_license_boundary_stop",
        ], "source dry-run event sequence drift")
        for event in events:
            require(set(event["totals"].values()) == {0},
                    "source dry-run nonzero action counter")
    after = formal_population()
    require(after == [], "formal R25 identity created by source hammer")
    return {
        "return_code": 86,
        "event_count": 4,
        "formal_population_before": before,
        "formal_population_after": after,
        "vcs_runs": 0,
        "simv_runs": 0,
        "license_queries": 0,
        "eda_runs": 0,
    }


def author_tests():
    completed = subprocess.run([
        "/usr/libexec/platform-python3.6", "-m", "unittest", "discover",
        "-s", str(HW / "verif_m859"), "-p", "test_m859*.py", "-v",
    ], env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8"}, stdout=subprocess.PIPE,
       stderr=subprocess.STDOUT)
    require(completed.returncode == 0,
            "author tests failed:\n" + completed.stdout.decode(
                "utf-8", "replace"))
    text = completed.stdout.decode("utf-8", "replace")
    require("Ran 5 tests" in text and text.rstrip().endswith("OK"),
            "author unittest count/status drift")
    return text


def main():
    source, m803 = source_checks()
    author_output = author_tests()
    matrix = run_matrix()
    failures = [name for name, expected in EXPECTED.items()
                if matrix.get(name) != expected]
    dry = source_dry_run()
    for name in sorted(EXPECTED):
        print("{} expected={} actual={}".format(
            name, EXPECTED[name], matrix.get(name, "MISSING")))
    print("matrix_count={} unexpected_count={}".format(
        len(EXPECTED), len(failures)))
    print("shared_whitelist_keys={}".format(len(guard.WHITELIST)))
    print("runner_sha256={}".format(source["runner_sha256"]))
    print("guard_sha256={}".format(GUARD_SHA))
    print("contract_sha256={}".format(source["contract_sha256"]))
    print("candidate_sha256={}".format(source["candidate_sha256"]))
    print("m856_outer_seal_file_sha256={}".format(
        source["m856_outer_seal_file_sha256"]))
    print("m803_frozen_files={}".format(len(m803)))
    print("author_unittest_methods=5 status=PASS")
    print("source_dry_run_rc={} events={} formal_before={} formal_after={}".format(
        dry["return_code"], dry["event_count"],
        len(dry["formal_population_before"]),
        len(dry["formal_population_after"])))
    print("docs359_sha256={}".format(sha256(DOCS359)))
    if failures:
        print("failures=" + ",".join(failures))
        return 1
    # Keep the subprocess output inspectable without confusing the formal
    # method count above with adversarial matrix rows.
    print("author_output_sha256={}".format(
        hashlib.sha256(author_output.encode("utf-8")).hexdigest()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
