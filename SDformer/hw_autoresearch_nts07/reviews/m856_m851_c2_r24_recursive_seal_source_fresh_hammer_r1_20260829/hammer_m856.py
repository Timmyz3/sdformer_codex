#!/usr/bin/env python3
"""Independent M856 source hammer for M851/C2 R24. Never invokes EDA."""

import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
sys.path.insert(0, str(HW / "verif_m851"))
import m851_c2_r24_recursive_seal_guard as guard  # noqa: E402


EXPECTED = {
    "complete_pipeline": "PASS",
    "actual_runner_receipt_population": "REJECT",
    "source_file_symlink": "REJECT",
    "source_directory_symlink": "REJECT",
    "sealed_root_symlink": "REJECT",
    "nested_file_symlink": "REJECT",
    "source_path_toctou": "REJECT",
    "publish_path_toctou": "REJECT",
    "missing_file": "REJECT",
    "extra_file": "REJECT",
    "extra_empty_directory": "REJECT",
    "nested_depth_drift": "REJECT",
    "payload_mutation": "REJECT",
    "manifest_mutation": "REJECT",
    "outer_seal_mutation": "REJECT",
    "destination_collision": "REJECT",
    "flat_verifier_nested_population": "REJECT",
}


def populate(work, receipt="m848_c2_r23_whitelist_vcs_receipt_r1.json"):
    for index, relative in enumerate(guard.r848.WHITELIST):
        actual = relative
        if relative == "m848_c2_r23_whitelist_vcs_receipt_r1.json":
            actual = receipt
        path = work / actual
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((actual + "\n").encode("utf-8") + bytes([index]))


def sealed(root):
    work = root / "work"
    work.mkdir()
    populate(work)
    stage = root / "stage"
    guard.r848.stage_result_whitelist(work, stage)
    guard.base.seal_directory(stage)
    return work, stage


def rejected(callback):
    try:
        callback()
    except (guard.base.Failure, OSError, ValueError):
        return "REJECT"
    return "ACCEPT"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def run():
    results = {}

    with tempfile.TemporaryDirectory(prefix="m856_complete.") as raw:
        root = Path(raw)
        _, stage = sealed(root)
        identity = guard.verify_recursive_sealed_directory(
            stage, guard.RESULT_MEMBERS)
        destination = root / "canonical"
        after = guard.publish_recursive_noreplace(
            stage, destination, guard.RESULT_MEMBERS)
        results["complete_pipeline"] = (
            "PASS" if identity == after and not stage.exists() else "FAIL")

    with tempfile.TemporaryDirectory(prefix="m856_runner_shape.") as raw:
        root = Path(raw)
        work = root / "work"
        work.mkdir()
        populate(work, "m851_c2_r24_recursive_seal_vcs_receipt_r1.json")
        results["actual_runner_receipt_population"] = rejected(
            lambda: guard.r848.stage_result_whitelist(work, root / "stage"))

    with tempfile.TemporaryDirectory(prefix="m856_file_link.") as raw:
        root = Path(raw)
        work = root / "work"
        work.mkdir()
        populate(work)
        target = work / "RUN_COMPLETE.real"
        (work / "RUN_COMPLETE.txt").rename(target)
        os.symlink(target.name, str(work / "RUN_COMPLETE.txt"))
        results["source_file_symlink"] = rejected(
            lambda: guard.r848.stage_result_whitelist(work, root / "stage"))

    with tempfile.TemporaryDirectory(prefix="m856_dir_link.") as raw:
        root = Path(raw)
        work = root / "work"
        work.mkdir()
        populate(work)
        (work / "attack").rename(work / "attack.real")
        os.symlink("attack.real", str(work / "attack"))
        results["source_directory_symlink"] = rejected(
            lambda: guard.r848.stage_result_whitelist(work, root / "stage"))

    with tempfile.TemporaryDirectory(prefix="m856_root_link.") as raw:
        root = Path(raw)
        _, stage = sealed(root)
        stage.rename(root / "real")
        os.symlink("real", str(stage))
        results["sealed_root_symlink"] = rejected(
            lambda: guard.verify_recursive_sealed_directory(
                stage, guard.RESULT_MEMBERS))

    with tempfile.TemporaryDirectory(prefix="m856_nested_link.") as raw:
        root = Path(raw)
        _, stage = sealed(root)
        victim = stage / "attack" / "compile.log"
        victim.rename(stage / "attack" / "compile.real")
        os.symlink("compile.real", str(victim))
        results["nested_file_symlink"] = rejected(
            lambda: guard.verify_recursive_sealed_directory(
                stage, guard.RESULT_MEMBERS))

    with tempfile.TemporaryDirectory(prefix="m856_copy_toctou.") as raw:
        root = Path(raw)
        work = root / "work"
        work.mkdir()
        populate(work)
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
                lambda: guard.r848.stage_result_whitelist(
                    work, root / "stage"))
        finally:
            guard.r848._hash_open_file = original

    with tempfile.TemporaryDirectory(prefix="m856_publish_toctou.") as raw:
        root = Path(raw)
        _, stage = sealed(root)
        attacker_work = root / "attacker_work"
        attacker_work.mkdir()
        populate(attacker_work)
        (attacker_work / "RUN_COMPLETE.txt").write_bytes(b"attacker\n")
        attacker = root / "attacker"
        guard.r848.stage_result_whitelist(attacker_work, attacker)
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
                    stage, root / "canonical", guard.RESULT_MEMBERS))
        finally:
            guard.base._rename_noreplace = original

    mutators = {
        "missing_file": lambda stage: (stage / "attack" / "compile.log").unlink(),
        "extra_file": lambda stage: (stage / "attack" / "extra.txt").write_bytes(b"x"),
        "extra_empty_directory": lambda stage: (stage / "empty").mkdir(),
        "payload_mutation": lambda stage: (stage / "attack" / "compile.log").write_bytes(b"changed"),
        "manifest_mutation": lambda stage: (stage / "SHA256SUMS").write_bytes(b"changed"),
        "outer_seal_mutation": lambda stage: (stage / "SHA256SUMS.seal.sha256").write_bytes(b"changed"),
    }
    for label, mutate in mutators.items():
        with tempfile.TemporaryDirectory(prefix="m856_" + label + ".") as raw:
            root = Path(raw)
            _, stage = sealed(root)
            mutate(stage)
            results[label] = rejected(
                lambda: guard.verify_recursive_sealed_directory(
                    stage, guard.RESULT_MEMBERS))

    with tempfile.TemporaryDirectory(prefix="m856_depth.") as raw:
        root = Path(raw)
        work = root / "work"
        work.mkdir()
        populate(work)
        stage = root / "stage"
        guard.r848.stage_result_whitelist(work, stage)
        (stage / "attack" / "deep").mkdir()
        (stage / "attack" / "compile.log").rename(
            stage / "attack" / "deep" / "compile.log")
        guard.base.seal_directory(stage)
        results["nested_depth_drift"] = rejected(
            lambda: guard.verify_recursive_sealed_directory(
                stage, guard.RESULT_MEMBERS))

    with tempfile.TemporaryDirectory(prefix="m856_collision.") as raw:
        root = Path(raw)
        _, stage = sealed(root)
        destination = root / "canonical"
        destination.mkdir()
        marker = destination / "marker"
        marker.write_bytes(b"preserve\n")
        outcome = rejected(lambda: guard.publish_recursive_noreplace(
            stage, destination, guard.RESULT_MEMBERS))
        results["destination_collision"] = (
            outcome if marker.read_bytes() == b"preserve\n" and stage.is_dir()
            else "CLOBBER")

    with tempfile.TemporaryDirectory(prefix="m856_flat.") as raw:
        root = Path(raw)
        _, stage = sealed(root)
        results["flat_verifier_nested_population"] = rejected(
            lambda: guard.base.verify_sealed_directory(
                stage, set(guard.RESULT_MEMBERS)))

    for key in sorted(EXPECTED):
        print("{} expected={} actual={}".format(
            key, EXPECTED[key], results.get(key, "MISSING")))
    print("runner_receipt=m851_c2_r24_recursive_seal_vcs_receipt_r1.json")
    print("whitelist_receipt={}".format(guard.r848.WHITELIST[2]))
    print("result_member_receipt={}".format(guard.RESULT_MEMBERS[2]))
    failures = [key for key in EXPECTED if results.get(key) != EXPECTED[key]]
    print("test_count={} unexpected_count={}".format(len(EXPECTED), len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run())
