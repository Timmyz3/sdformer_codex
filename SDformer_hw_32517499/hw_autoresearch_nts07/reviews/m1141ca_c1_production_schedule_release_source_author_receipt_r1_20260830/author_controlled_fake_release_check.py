#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1141CA author check: controlled fake canonical only; no M410/production/full/EDA."""
from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Callable
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/run_m1141ca_c1_production_schedule_release_source.py"
CONTRACT = HW / "contracts/m1141ca_c1_production_schedule_release_source_contract_r1_20260830.json"
ROWS = HW / "results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/m410r2_h67_q32_runtime_rows_32.memh"
RESULT = HW / "results/m1141ca_c1_production_schedule_release_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    "source": "e2f5d4e0bab472b3a5c7ec5259a805641b800efd3c0e82884e81152eb41cb611",
    "contract": "4fe7ba960516e889cb1f7140315e1e37a5b42dd00337f136b22a25f1c7ac06d4",
    "contract_side": "128d813d63cba813173a5e282dd6f3247ff2f443a5428878d76bef36230d0263",
    "contract_outer": "6e5561e52fab6b4ae3018f8995f4b71f4c8eaeaf02c83ea192421081b5af8184",
    "m1140ca_outer": "f73cafa73ed047abd59730749bf48fcb3f463fca77609aec6085f5b3389fa352",
    "rows": "6e03352b89eff1955825334b4dedd991db8c975a9ef6662fe0317e73ccfa8334",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
checks = 0
attacks: dict[str, str] = {}


class CheckFailure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise CheckFailure(message)


def rejected(label: str, action: Callable[[], Any], contains: str | None = None) -> None:
    try:
        action()
    except Exception as error:
        if contains is not None:
            require(contains in str(error), label + " wrong rejection: " + str(error))
        attacks[label] = type(error).__name__ + ": " + str(error)
        return
    raise CheckFailure("attack accepted: " + label)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_regular(path: Path, expected: str) -> None:
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink() and
            sha(path) == expected, "identity drift: " + str(path))


def strict_json(path: Path) -> Any:
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          CheckFailure("nonfinite JSON: " + token)))


def verify_frozen() -> dict[str, Any]:
    verify_regular(SOURCE, EXPECTED["source"])
    verify_regular(CONTRACT, EXPECTED["contract"])
    side = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    verify_regular(side, EXPECTED["contract_side"])
    verify_regular(outer, EXPECTED["contract_outer"])
    require(side.read_text(encoding="utf-8").split() ==
            [EXPECTED["contract"], CONTRACT.name] and
            outer.read_text(encoding="utf-8").split() ==
            [EXPECTED["contract_side"], side.name], "contract double seal drift")
    verify_regular(DOCS359, EXPECTED["docs359"])
    contract = strict_json(CONTRACT)
    require(contract["status"] ==
            "SOURCE_ONLY__CONTROLLED_FAKE_FIXTURE_ONLY__DIFFERENT_AUTHOR_HAMMER_REQUIRED__NO_PRODUCTION" and
            contract["source"]["arguments"] == 0 and
            contract["production_geometry"]["records"] == 2_436_480 and
            contract["authorization_root"]["m1140ca_outer_seal_file_sha256"] ==
            EXPECTED["m1140ca_outer"] and
            contract["this_milestone_execution"]["m410_opened"] is False and
            contract["authorization"]["production_execution"] is False,
            "contract semantic drift")
    return contract


def load_subject():
    spec = importlib.util.spec_from_file_location("m1141ca_author_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "subject module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def production_namespace() -> tuple[str, ...]:
    parent = RESULT.parent
    names = []
    if RESULT.exists() or RESULT.is_symlink():
        names.append(RESULT.name)
    names.extend(path.name for path in parent.glob(".m1141ca_c1_production_schedule_release_work.*"))
    names.extend(path.name for path in parent.glob(
        "m1141ca_c1_production_schedule_release_r1_20260830.failed_or_incomplete.*"))
    return tuple(sorted(names))


def static_checks(module, contract: dict[str, Any]) -> dict[str, Any]:
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    imports = [alias.name for node in ast.walk(tree)
               if isinstance(node, (ast.Import, ast.ImportFrom))
               for alias in node.names]
    require(not any("m1016" in name.lower() for name in imports),
            "M1016 runtime import forbidden")
    production = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
                  node.name == "production_main"]
    main = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
            node.name == "main"]
    require(len(production) == len(main) == 1 and
            len(production[0].args.args) == 0 and len(main[0].args.args) == 0,
            "zero-argument production/main drift")
    prod_text = ast.unparse(production[0])
    main_text = ast.unparse(main[0])
    require("_execute_release(ROWS, ROWS_SHA, ROWS_BYTES, PRODUCTION_GEOMETRY, RESULT)" in
            prod_text and "len(sys.argv) == 1" in main_text and
            "production_main()" in main_text and "argparse" not in text,
            "hardcoded zero-argument binding drift")
    recurrence = [node for node in tree.body if isinstance(node, ast.ClassDef) and
                  node.name == "ExactScheduleRecurrence"]
    require(len(recurrence) == 1, "exact recurrence class drift")
    forbidden_history = []
    for node in ast.walk(recurrence[0]):
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.Set)):
            forbidden_history.append(type(node).__name__)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                node.func.attr in {"append", "extend", "add"}):
            forbidden_history.append(node.func.attr)
    require(forbidden_history == [] and "O(axes)" in text,
            "O(N) recurrence history primitive")
    require("O_NOFOLLOW" in text and "_verify_open_identity" in text and
            "_fd_hash(fd) == expected_sha" in text and
            "_rename_noreplace(stage, result)" in text and
            "_rename_noreplace(stage, quarantine)" in text,
            "no-follow/atomic/quarantine implementation drift")
    require(contract["independent_derivation"]["imports_m1016_runtime_module"] is False and
            contract["input_security"]["single_open_file_descriptor"] is True and
            contract["output"]["automatic_retry"] is False,
            "static contract boundary drift")
    return {"zero_argument_entry": True, "m1016_runtime_imports": 0,
            "recurrence_history_primitives": forbidden_history,
            "no_follow_single_fd": True, "atomic_noreplace_publish": True,
            "failure_quarantine": True}


def verify_sealed_directory(module, directory: Path) -> tuple[str, str]:
    require(directory.is_dir() and not directory.is_symlink(), "sealed directory drift")
    manifest = directory / module.MANIFEST_NAME
    outer = directory / module.OUTER_NAME
    manifest_sha = sha(manifest)
    require(outer.read_text(encoding="utf-8").split() ==
            [manifest_sha, module.MANIFEST_NAME], "outer seal content drift")
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(name not in listed and sha(directory / name) == digest,
                "sealed member drift")
        listed.add(name)
    actual = {path.name for path in directory.iterdir()
              if path.name not in {module.MANIFEST_NAME, module.OUTER_NAME}}
    require(actual == listed, "exact sealed member set drift")
    return manifest_sha, sha(outer)


def fake_payload() -> bytes:
    return ("\n".join(("00000001", "00000003", "00000005", "00000000",
                       "00000007", "00000008", "0000000f", "00000010")) +
            "\n").encode()


def fixture_geometry(module):
    return module.Geometry(1, 1, 2, 4, 4, 8, 7, 12, 2)


def independent_record_provenance(module, record: dict[str, Any]) -> str:
    payload = b"".join((
        b"M1139CA_SCHEDULE_RECORD\x00\x01",
        bytes.fromhex(module.M1016_SOURCE_SHA), bytes.fromhex(module.M1102_SOURCE_SHA),
        bytes.fromhex(module.M1137_SOURCE_SHA),
        bytes((AXES.index(record["axis"]),)),
        *(int(record[field]).to_bytes(8, "big") for field in (
            "task_sequence_ordinal", "sample", "operator", "chunk", "partition",
            "requested_cycle_first")),
        bytes.fromhex(record["source_task_provenance_sha256"]),
    ))
    return hashlib.sha256(payload).hexdigest()


def positive_fixture(module, root: Path) -> dict[str, Any]:
    payload = fake_payload()
    rows = root / "fake_canonical.memh"
    rows.write_bytes(payload)
    result = root / "fake_result"
    summary = module._execute_release(
        rows, hashlib.sha256(payload).hexdigest(), len(payload),
        fixture_geometry(module), result)
    manifest_sha, outer_sha = verify_sealed_directory(module, result)
    release = strict_json(result / module.RELEASE_NAME)
    raw_lines = (result / module.RECORDS_NAME).read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in raw_lines]
    require(summary["records"] == release["records"]["count"] == len(records) == 6 and
            release["records"]["sha256"] == sha(result / module.RECORDS_NAME) and
            release["records"]["axis_counts"] == {axis: 2 for axis in AXES},
            "fake positive count/SHA drift")
    require([(record["task_sequence_ordinal"], record["axis"])
             for record in records] == [(task, axis) for task in range(2) for axis in AXES],
            "fake positive task/axis order drift")
    require([record["requested_cycle_first"] for record in records] ==
            [0, 0, 0, 34, 42, 42], "independent recurrence result drift")
    for record in records:
        require(record["schedule_record_provenance_sha256"] ==
                independent_record_provenance(module, record),
                "independent record provenance mismatch")
    provenance = hashlib.sha256(b"".join(bytes.fromhex(
        record["schedule_record_provenance_sha256"]) for record in records)).hexdigest()
    require(release["records"]["schedule_provenance_sha256"] == provenance and
            release["source_rows"]["no_follow_single_fd"] is True and
            release["source_rows"]["identity_reverified_after_stream"] is True and
            release["state_complexity"] == "O(axes) plus one bounded row tile" and
            not any(root.glob(".fake_result.private_staging.*")) and
            not any(root.glob("fake_result.failed_or_incomplete.*")),
            "fake positive provenance/atomic cleanup drift")
    return {"tasks": 2, "axes": 3, "records": 6,
            "requested_cycle_first": [0, 0, 0, 34, 42, 42],
            "records_sha256": release["records"]["sha256"],
            "schedule_provenance_sha256": provenance,
            "manifest_sha256": manifest_sha,
            "outer_seal_file_sha256": outer_sha,
            "private_staging_remaining": 0}


def make_record(module, axis: str, task: int = 0, provenance: str = "1" * 64):
    digest = module._record_provenance(axis, task, 0, 0, 0, task, 0, provenance)
    return module.ScheduleRecord(axis, task, 0, 0, 0, task, 0, provenance, digest)


def stream_attacks(module) -> None:
    sink = module._StreamingRecordSink(io.BytesIO())
    sink(make_record(module, "candidate"))
    sink(make_record(module, "strongest_zero"))
    rejected("missing_record", lambda: sink.finalize(1), "conservation")

    duplicate = module._StreamingRecordSink(io.BytesIO())
    duplicate(make_record(module, "candidate"))
    rejected("duplicate_record", lambda: duplicate(make_record(module, "candidate")),
             "out of order")

    out_axis = module._StreamingRecordSink(io.BytesIO())
    rejected("out_of_order_axis", lambda: out_axis(make_record(module, "strongest_zero")),
             "out of order")
    out_task = module._StreamingRecordSink(io.BytesIO())
    rejected("out_of_order_task", lambda: out_task(make_record(module, "candidate", 1)),
             "out of order")
    drift = replace(make_record(module, "candidate"),
                    schedule_record_provenance_sha256="0" * 64)
    rejected("record_provenance_drift", lambda:
             module._StreamingRecordSink(io.BytesIO())(drift), "provenance")


def run_failure_fixture(module, root: Path, label: str, payload: bytes,
                        expected_sha: str | None = None,
                        mutate=None) -> Path:
    case = root / label
    case.mkdir()
    rows = case / "fake.memh"
    rows.write_bytes(payload)
    result = case / "result"
    expected = expected_sha or hashlib.sha256(payload).hexdigest()
    action = lambda: module._execute_release(
        rows, expected, 72, fixture_geometry(module), result)
    if mutate is None:
        rejected(label, action)
    else:
        mutate(action, rows)
    require(not result.exists(), label + " published a result")
    quarantines = list(case.glob("result.failed_or_incomplete.*.quarantine"))
    require(len(quarantines) == 1 and
            not any(case.glob(".result.private_staging.*")),
            label + " staging/quarantine drift")
    verify_sealed_directory(module, quarantines[0])
    failure = strict_json(quarantines[0] / "failure.json")
    require(failure["status"] == "FAILED_OR_INCOMPLETE_DO_NOT_CITE" and
            failure["automatic_retry"] is False,
            label + " failure boundary drift")
    return quarantines[0]


def atomic_and_identity_attacks(module, root: Path) -> dict[str, Any]:
    payload = fake_payload()
    short = run_failure_fixture(module, root, "short_canonical", payload[:-9])
    malformed_payload = b"xxxxxxxx\n" + payload[9:]
    malformed = run_failure_fixture(module, root, "malformed_canonical", malformed_payload)
    wrong_sha = run_failure_fixture(module, root, "canonical_sha_identity_drift", payload,
                                    "0" * 64)

    symlink_case = root / "canonical_symlink"
    symlink_case.mkdir()
    real = symlink_case / "real.memh"; real.write_bytes(payload)
    link = symlink_case / "fake.memh"; link.symlink_to(real)
    result = symlink_case / "result"
    rejected("canonical_symlink", lambda: module._execute_release(
        link, hashlib.sha256(payload).hexdigest(), 72,
        fixture_geometry(module), result), "regular")
    require(not result.exists() and len(list(symlink_case.glob(
        "result.failed_or_incomplete.*.quarantine"))) == 1,
        "canonical symlink quarantine drift")

    replacement_case = root / "path_replacement"
    replacement_case.mkdir()
    rows = replacement_case / "fake.memh"; rows.write_bytes(payload)
    result = replacement_case / "result"
    original_hash = module._fd_hash
    calls = 0
    def replace_after_hash(fd):
        nonlocal calls
        value = original_hash(fd)
        calls += 1
        if calls == 1:
            rows.rename(replacement_case / "opened_original.memh")
            rows.write_bytes(payload)
        return value
    with patch.object(module, "_fd_hash", replace_after_hash):
        rejected("canonical_path_replacement", lambda: module._execute_release(
            rows, hashlib.sha256(payload).hexdigest(), 72,
            fixture_geometry(module), result), "replacement")
    require(calls == 1 and not result.exists() and len(list(replacement_case.glob(
        "result.failed_or_incomplete.*.quarantine"))) == 1,
        "path replacement quarantine drift")

    mid_case = root / "controlled_midstream_failure"
    mid_case.mkdir()
    rows = mid_case / "fake.memh"; rows.write_bytes(payload)
    result = mid_case / "result"
    original_call = module._StreamingRecordSink.__call__
    def fail_third(self, record):
        if self.count == 2:
            raise RuntimeError("controlled midstream fixture failure")
        return original_call(self, record)
    with patch.object(module._StreamingRecordSink, "__call__", fail_third):
        rejected("controlled_midstream_failure", lambda: module._execute_release(
            rows, hashlib.sha256(payload).hexdigest(), 72,
            fixture_geometry(module), result), "controlled midstream")
    quarantines = list(mid_case.glob("result.failed_or_incomplete.*.quarantine"))
    require(not result.exists() and len(quarantines) == 1 and
            (quarantines[0] / module.RECORDS_NAME).is_file(),
            "midstream atomic quarantine drift")
    verify_sealed_directory(module, quarantines[0])

    collision_case = root / "result_collision"
    collision_case.mkdir(); rows = collision_case / "fake.memh"; rows.write_bytes(payload)
    result = collision_case / "result"; result.mkdir()
    rejected("result_collision", lambda: module._execute_release(
        rows, hashlib.sha256(payload).hexdigest(), 72,
        fixture_geometry(module), result), "collision")
    require(not any(collision_case.glob(".result.private_staging.*")) and
            not any(collision_case.glob("result.failed_or_incomplete.*")),
            "collision created private namespace")

    return {"short_quarantine_outer": sha(short / module.OUTER_NAME),
            "malformed_quarantine_outer": sha(malformed / module.OUTER_NAME),
            "wrong_sha_quarantine_outer": sha(wrong_sha / module.OUTER_NAME),
            "path_replacement_rejected": True,
            "midstream_partial_staging_quarantined": True,
            "result_collision_pre_stage": True}


def main() -> None:
    before_namespace = production_namespace()
    before_docs = sha(DOCS359)
    contract = verify_frozen()
    module = load_subject()
    static = static_checks(module, contract)
    real_row_opens = 0
    original_os_open = os.open
    original_path_open = Path.open
    def watched_os_open(path, *args, **kwargs):
        nonlocal real_row_opens
        try:
            if Path(path) == ROWS:
                real_row_opens += 1
        except TypeError:
            pass
        return original_os_open(path, *args, **kwargs)
    def watched_path_open(path, *args, **kwargs):
        nonlocal real_row_opens
        if path == ROWS:
            real_row_opens += 1
        return original_path_open(path, *args, **kwargs)
    with patch.object(os, "open", watched_os_open), patch.object(Path, "open", watched_path_open):
        preflight = module.source_static_self_test()
        with tempfile.TemporaryDirectory(prefix="m1141ca_author_fake_") as temp:
            root = Path(temp)
            positive = positive_fixture(module, root)
            stream_attacks(module)
            atomic = atomic_and_identity_attacks(module, root)
    rejected("runtime_argument", lambda: _runtime_argument_attack(module), "zero arguments")
    require(real_row_opens == 0 and preflight["canonical_opened"] is False and
            preflight["production_records"] == 0,
            "real M410 was opened or production executed")
    require(production_namespace() == before_namespace == () and
            sha(DOCS359) == before_docs == EXPECTED["docs359"],
            "production namespace or docs359 changed")
    report = {
        "schema": "m1141ca_author_controlled_fake_release_checks_r1_v1",
        "status": "PASS_M1141CA_SOURCE_AND_CONTROLLED_FAKE_RELEASE__DIFFERENT_AUTHOR_HAMMER_ONLY",
        "checks_passed": checks,
        "attacks_rejected": len(attacks),
        "attacks": attacks,
        "static": static,
        "positive_fake_fixture": positive,
        "atomic_and_identity": atomic,
        "production_boundary": {"m410_open_count": real_row_opens,
                                "production_records": 0,
                                "production_result_created": False,
                                "digest_compiler": False, "real_driver": False,
                                "full_replay": False, "eda_gpu_remote": False},
        "authorization": {"different_author_hammer_only": True,
                          "production_execution": False,
                          "open_real_m410": False,
                          "digest_compiler_driver_full_eda": False},
        "identity": {"source_sha256": EXPECTED["source"],
                     "contract_sha256": EXPECTED["contract"],
                     "contract_sidecar_sha256": EXPECTED["contract_side"],
                     "contract_outer_seal_file_sha256": EXPECTED["contract_outer"],
                     "m1140ca_outer_seal_file_sha256": EXPECTED["m1140ca_outer"],
                     "docs359_sha256": EXPECTED["docs359"]},
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


def _runtime_argument_attack(module) -> None:
    with patch.object(sys, "argv", [str(SOURCE), "unexpected"]):
        module.main()


if __name__ == "__main__":
    main()
