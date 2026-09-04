#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1140CA independent hammer: bounded 2-task x 3-axis only; no production/full/EDA."""
from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
from typing import Any, Callable
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/build_m1139ca_c1_independent_per_task_schedule_authority_source.py"
CONTRACT = HW / "contracts/m1139ca_c1_independent_per_task_schedule_authority_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1139ca_c1_independent_per_task_schedule_authority_author_receipt_r1_20260830"
M1016 = HW / "system_simulator/scripts/run_m1016_c1_full_matched_address_replay.py"
M1102 = HW / "system_simulator/scripts/run_m1102_c1_work8_exact_1rw_source.py"
ROWS = HW / "results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/m410r2_h67_q32_runtime_rows_32.memh"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUTPUT = HERE / "mechanical_checks.json"
EXPECTED = {
    "source": "d18137661517538a8273b696b5f2ada09aff9847c16da0d3a723037e901153a9",
    "contract": "8c92bdd9b7e3668b47b97d2d8a85a0f1977980961470e3dabf7bb2c22d5d9973",
    "contract_side": "b9d12cb80136b674d9fe38794e7cc226eb9f318a3b3c00f866ba7114e15a751d",
    "contract_outer": "b8feb4f8394ddbe5445692efc14485ad818cbff719104c8d77aca29ea7a32d0b",
    "rows": "6e03352b89eff1955825334b4dedd991db8c975a9ef6662fe0317e73ccfa8334",
    "rows_bytes": 466560000,
    "author_review": "32ec860f8be0b46b3fcf099095246caa4a9b6050029d405fa78cc872002d7db8",
    "author_manifest": "0f79bf5928361c9a78d785ceb67d51fc5fbf9c601df0684198401fdb7445d08f",
    "author_outer": "de00e71c78870403d3aa5f710e09d0ca0d024d7dc4f8f4e77194a7df8728e3e6",
    "m1016": "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa",
    "m1102": "95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc",
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
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          CheckFailure("nonfinite JSON: " + token)))


def verify_flat(directory: Path, outer_sha: str) -> dict[str, Any]:
    require(stat.S_ISDIR(directory.lstat().st_mode) and not directory.is_symlink(),
            "sealed authority directory")
    manifest = directory / "SHA256SUMS"; outer = directory / "SHA256SUMS.seal.sha256"
    verify_regular(outer, outer_sha)
    require(outer.read_text(encoding="utf-8").split() == [sha(manifest), "SHA256SUMS"],
            "sealed outer content")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*"); rel = Path(name)
        require(name not in expected and not rel.is_absolute() and ".." not in rel.parts,
                "manifest member safety")
        expected[name] = digest
    actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}: continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "sealed symlink")
        if stat.S_ISREG(mode): actual.add(name)
        else: require(stat.S_ISDIR(mode), "sealed special member")
    require(actual == set(expected), "sealed exact member set")
    for name, digest in expected.items(): verify_regular(directory / name, digest)
    return strict_json(directory / "review.json")


def verify_frozen() -> None:
    verify_regular(SOURCE, EXPECTED["source"])
    verify_regular(CONTRACT, EXPECTED["contract"])
    side = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    verify_regular(side, EXPECTED["contract_side"])
    verify_regular(outer, EXPECTED["contract_outer"])
    require(side.read_text(encoding="utf-8").split() ==
            [EXPECTED["contract"], CONTRACT.name] and
            outer.read_text(encoding="utf-8").split() ==
            [EXPECTED["contract_side"], side.name], "contract double seal")
    require(stat.S_ISREG(ROWS.lstat().st_mode) and not ROWS.is_symlink() and
            ROWS.stat().st_size == EXPECTED["rows_bytes"],
            "M410 rows metadata drift without opening canonical rows")
    verify_regular(M1016, EXPECTED["m1016"])
    verify_regular(M1102, EXPECTED["m1102"])
    author = verify_flat(AUTHOR, EXPECTED["author_outer"])
    verify_regular(AUTHOR / "review.json", EXPECTED["author_review"])
    verify_regular(AUTHOR / "SHA256SUMS", EXPECTED["author_manifest"])
    require(author["status"] ==
            "PASS_M1139CA_INDEPENDENT_PER_TASK_SCHEDULE_AUTHORITY_AUTHOR__BOUNDED_ONLY_PRODUCTION_STOP",
            "author authority status")
    verify_regular(DOCS359, EXPECTED["docs359"])


def load_subject():
    spec = importlib.util.spec_from_file_location("m1140ca_independent_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "subject module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def static_checks(module) -> dict[str, Any]:
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    contract = strict_json(CONTRACT)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and
               node.name == "IndependentPerTaskScheduleAuthority"]
    require(len(classes) == 1, "one authority class")
    forbidden = []
    for node in ast.walk(classes[0]):
        if isinstance(node, (ast.Set, ast.SetComp, ast.ListComp)):
            forbidden.append(type(node).__name__)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"set", "list"}:
                forbidden.append(node.func.id + "()")
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                    "append", "extend", "add"}:
                forbidden.append(node.func.attr)
    require(forbidden == [], "O(N) history primitive in authority class")
    imported_names = [alias.name for node in ast.walk(tree)
                      if isinstance(node, (ast.Import, ast.ImportFrom))
                      for alias in node.names]
    require(not any("m1137" in name.lower() for name in imported_names) and
            "load_m1137" not in text and "producer_result" not in text and
            "event_result" not in text, "M1137 producer/result dependency")
    production = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "iter_production_schedule_authority_records")
    production_text = ast.unparse(production)
    require(production_text.index("require(PRODUCTION_RELEASE") <
            production_text.index("if False") and "os.open" not in production_text and
            "os.pread" not in production_text, "production gate precedes row access")
    require(contract["derivability_audit"]["m1016_raw_rows_plus_frozen_formulas_sufficient"] is True and
            contract["derivability_audit"]["m1102_record_alone_sufficient"] is False and
            contract["production_fail_closed"]["release_outer_seal_file_sha256"] is None and
            contract["input_schema"]["exact_fields"] == [
                "task_id", "sample", "operator", "chunk", "partition",
                "preprocess_by_axis", "work_by_axis", "source_raw_sha256",
                "source_task_provenance_sha256"], "contract derivability/schema drift")
    return {
        "authority_class_history_primitives": forbidden,
        "m1137_imports_or_result_reads": 0,
        "production_gate_before_row_open": True,
        "input_fields": 9,
        "state_complexity": "O(axes)",
    }


class CaptureSink:
    def __init__(self, fail_at: int | None = None):
        self.calls = 0
        self.accepted = 0
        self.fail_at = fail_at
        self.records = []

    def __call__(self, record) -> None:
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("controlled schedule sink failure")
        record.validate()
        self.records.append(record)
        self.accepted += 1


def independent_reference(primitives) -> dict[str, list[int]]:
    requested = {axis: [] for axis in AXES}
    previous_start = {axis: None for axis in AXES}
    previous_work = {axis: 0 for axis in AXES}
    offsets = {axis: 0 for axis in AXES}
    for primitive in primitives:
        for axis in AXES:
            preprocess = primitive.preprocess_by_axis[axis]
            start = (preprocess if previous_start[axis] is None else
                     previous_start[axis] + max(previous_work[axis], preprocess) + 2)
            requested[axis].append(offsets[axis] + start - preprocess)
            previous_start[axis] = start
            previous_work[axis] = primitive.work_by_axis[axis]
    return requested


def closed_form_and_derivability(module) -> dict[str, Any]:
    primitives = list(module.bounded_primitives())
    p0, p1 = primitives
    closed = {}
    for axis in AXES:
        pre0 = p0.preprocess_by_axis[axis]; work0 = p0.work_by_axis[axis]
        pre1 = p1.preprocess_by_axis[axis]
        closed[axis] = [0, pre0 + max(work0, pre1) + 2 - pre1]
    require(closed == {"candidate": [0, 22], "strongest_zero": [0, 12],
                       "same_coordinate_bit": [0, 14]},
            "closed-form bounded recurrence")
    m1016 = M1016.read_text(encoding="utf-8")
    m1102 = M1102.read_text(encoding="utf-8")
    require("def preprocess_for_design(" in m1016 and
            "self.work_start + max(self.previous_work, preprocess) + 2" in m1016 and
            "global_offsets[design] + start - preprocess" in m1016,
            "M1016 design-specific recurrence basis")
    require("record.shared_preprocess_cycles" in m1102 and
            "shared_preprocess_cycles" in m1102,
            "M1102 shared-maximum preprocess evidence")

    # Two hidden design-specific preprocess worlds have the same retained
    # shared maximum and identical work, yet produce different exact cycles.
    # Therefore the M1102 shared value alone cannot identify the schedule.
    world_a = {"candidate": 10, "strongest_zero": 8, "same_coordinate_bit": 6}
    world_b = {"candidate": 10, "strongest_zero": 10, "same_coordinate_bit": 10}
    next_pre = {axis: 6 for axis in AXES}; prior_work = {axis: 0 for axis in AXES}
    require(max(world_a.values()) == max(world_b.values()) == 10,
            "shared maximum witness")
    cycles_a = {axis: world_a[axis] + max(prior_work[axis], next_pre[axis]) + 2 - next_pre[axis]
                for axis in AXES}
    cycles_b = {axis: world_b[axis] + max(prior_work[axis], next_pre[axis]) + 2 - next_pre[axis]
                for axis in AXES}
    require(cycles_a != cycles_b and cycles_a["candidate"] == cycles_b["candidate"],
            "M1102-alone non-identifiability witness")
    contract = strict_json(CONTRACT)
    require(contract["frozen_authorities"]["m410_rows_sha256"] == EXPECTED["rows"] and
            contract["frozen_authorities"]["m1016_source_sha256"] == EXPECTED["m1016"] and
            contract["derivability_audit"]["m1102_record_alone_sufficient"] is False,
            "M410 plus M1016 derivability authority")
    return {"bounded_closed_form": closed,
            "m410_plus_m1016_sufficient": True,
            "m1102_alone_sufficient": False,
            "same_shared_max_world_a": world_a,
            "same_shared_max_world_b": world_b,
            "different_exact_cycles_a": cycles_a,
            "different_exact_cycles_b": cycles_b}


def bounded_positive(module) -> dict[str, Any]:
    primitives = list(module.bounded_primitives())
    reference = independent_reference(primitives)
    sink = CaptureSink()
    authority = module.IndependentPerTaskScheduleAuthority(module.BOUNDED_GEOMETRY, sink)
    initial_keys = tuple(sorted(authority.__dict__))
    for primitive in primitives:
        require(authority.consume_task(primitive) == 3, "three axis records per task")
    terminal = authority.finalize()
    observed = {axis: [record.requested_cycle_first for record in sink.records
                       if record.axis == axis] for axis in AXES}
    require(reference == observed == {
        "candidate": [0, 22], "strongest_zero": [0, 12],
        "same_coordinate_bit": [0, 14]}, "independent recurrence mismatch")
    require(tuple(record.axis for record in sink.records) == AXES + AXES,
            "axis order mismatch")
    require(tuple(sorted(authority.__dict__)) == initial_keys and
            len(authority._axis) == len(AXES), "state cardinality drift")
    return {
        "tasks": 2, "axes": 3, "records": sink.accepted,
        "requested_cycles": observed,
        "axis_order": list(AXES),
        "terminal": terminal,
        "independent_reference_match": True,
        "retained_record_or_key_history": False,
    }


def fail_closed_mutations(module) -> None:
    p0, p1 = list(module.bounded_primitives())
    rejected("wrong_exact_input_type", lambda:
             module.IndependentPerTaskScheduleAuthority(
                 module.BOUNDED_GEOMETRY, CaptureSink()).consume_task(object()), "exact")
    missing = module.IndependentPerTaskScheduleAuthority(module.BOUNDED_GEOMETRY, CaptureSink())
    missing.consume_task(p0)
    rejected("missing_task", missing.finalize, "conservation")
    duplicate = module.IndependentPerTaskScheduleAuthority(module.BOUNDED_GEOMETRY, CaptureSink())
    duplicate.consume_task(p0)
    rejected("duplicate_task", lambda: duplicate.consume_task(p0), "out of order")
    rejected("out_of_order_task", lambda:
             module.IndependentPerTaskScheduleAuthority(
                 module.BOUNDED_GEOMETRY, CaptureSink()).consume_task(p1), "out of order")
    rejected("wrong_coordinates", lambda:
             module.IndependentPerTaskScheduleAuthority(
                 module.BOUNDED_GEOMETRY, CaptureSink()).consume_task(
                     replace(p0, partition=1)), "coordinate")
    rejected("wrong_task_provenance", lambda:
             module.IndependentPerTaskScheduleAuthority(
                 module.BOUNDED_GEOMETRY, CaptureSink()).consume_task(
                     replace(p0, source_task_provenance_sha256="0" * 64)), "provenance")
    rejected("wrong_raw_provenance", lambda:
             module.IndependentPerTaskScheduleAuthority(
                 module.BOUNDED_GEOMETRY, CaptureSink()).consume_task(
                     replace(p0, source_raw_sha256="1" * 64)), "provenance")
    reversed_pre = dict(reversed(tuple(p0.preprocess_by_axis.items())))
    rejected("wrong_axis_map_order", lambda:
             module.IndependentPerTaskScheduleAuthority(
                 module.BOUNDED_GEOMETRY, CaptureSink()).consume_task(
                     replace(p0, preprocess_by_axis=reversed_pre)), "axis order")
    regression = module.IndependentPerTaskScheduleAuthority(module.BOUNDED_GEOMETRY, CaptureSink())
    regression.consume_task(p0)
    regression._axis["candidate"].last_requested_cycle = 10_000
    rejected("requested_cycle_regression", lambda: regression.consume_task(p1), "regressed")

    first_sink = CaptureSink(fail_at=1)
    first = module.IndependentPerTaskScheduleAuthority(module.BOUNDED_GEOMETRY, first_sink)
    before = first.snapshot()
    rejected("first_axis_sink_failure", lambda: first.consume_task(p0), "controlled")
    require(first.snapshot() == before, "first failed axis committed state")
    first._sink = CaptureSink()
    require(first.consume_task(p0) == 3, "first failed axis exact retry")

    middle_sink = CaptureSink(fail_at=2)
    middle = module.IndependentPerTaskScheduleAuthority(module.BOUNDED_GEOMETRY, middle_sink)
    rejected("middle_axis_sink_failure", lambda: middle.consume_task(p0), "controlled")
    paused = middle.snapshot()
    require(paused["axes"]["candidate"]["records"] == 1 and
            paused["axes"]["strongest_zero"]["records"] == 0 and
            paused["next_axis_index"] == 1 and paused["next_task_id"] == 0,
            "middle failure committed wrong axis state")
    resume_sink = CaptureSink(); middle._sink = resume_sink
    require(middle.consume_task(p0) == 2 and resume_sink.accepted == 2 and
            middle.snapshot()["next_task_id"] == 1,
            "middle failure did not resume failed axis")

    record = module.PerTaskScheduleAuthorityRecord(
        "candidate", 0, 0, 0, 0, 0, 0, p0.source_task_provenance_sha256,
        module.schedule_record_provenance(
            "strongest_zero", 0, 0, 0, 0, 0, 0,
            p0.source_task_provenance_sha256))
    rejected("record_axis_order_provenance", record.validate, "provenance")

    row_opens = 0
    original_open = module.os.open
    original_path_open = Path.open
    def watched_open(path, *args, **kwargs):
        nonlocal row_opens
        if Path(path) == ROWS:
            row_opens += 1
        return original_open(path, *args, **kwargs)
    def watched_path_open(path, *args, **kwargs):
        nonlocal row_opens
        if Path(path) == ROWS:
            row_opens += 1
        return original_path_open(path, *args, **kwargs)
    with patch.object(module.os, "open", watched_open), \
         patch.object(Path, "open", watched_path_open):
        rejected("production_release_absent", lambda:
                 next(module.iter_production_schedule_authority_records()), "release is absent")
        saved_release = module.PRODUCTION_RELEASE_OUTER_SEAL_FILE_SHA256
        module.PRODUCTION_RELEASE_OUTER_SEAL_FILE_SHA256 = "a" * 64
        try:
            rejected("fake_production_release_without_integrated_authority", lambda:
                     next(module.iter_production_schedule_authority_records()),
                     "production schedule release is absent")
        finally:
            module.PRODUCTION_RELEASE_OUTER_SEAL_FILE_SHA256 = saved_release
    require(row_opens == 0, "production row opened before release gate")


def main() -> None:
    verify_frozen()
    before = {path: sha(path) for path in (SOURCE, CONTRACT, M1016, M1102, DOCS359)}
    rows_metadata_before = (ROWS.lstat().st_mode, ROWS.stat().st_size,
                            ROWS.stat().st_mtime_ns)
    module = load_subject()
    m410_opens = 0
    original_path_open = Path.open
    def prohibit_m410_open(path, *args, **kwargs):
        nonlocal m410_opens
        if Path(path) == ROWS:
            m410_opens += 1
            raise CheckFailure("canonical M410 rows must not open in bounded hammer")
        return original_path_open(path, *args, **kwargs)
    with patch.object(Path, "open", prohibit_m410_open):
        static = static_checks(module)
        oracle = bounded_positive(module)
        derivability = closed_form_and_derivability(module)
        fail_closed_mutations(module)
        small = module.source_small_oracle()
    require(small["production_rows_opened"] is False and
            small["production_records"] == 0 and
            small["derivability"]["invented_requested_cycles"] is False and
            small["derivability"]["m1102_retained_preprocess"] == "shared maximum only",
            "source oracle claim boundary")
    require(m410_opens == 0 and module.PRODUCTION_RELEASE_OUTER_SEAL_FILE_SHA256 is None,
            "M410 open or production release escaped")
    verify_frozen()
    after = {path: sha(path) for path in (SOURCE, CONTRACT, M1016, M1102, DOCS359)}
    rows_metadata_after = (ROWS.lstat().st_mode, ROWS.stat().st_size,
                           ROWS.stat().st_mtime_ns)
    require(before == after and rows_metadata_before == rows_metadata_after,
            "frozen identity/metadata changed during hammer")
    result = {
        "schema": "m1140ca_independent_bounded_schedule_hammer_checks_r1_v1",
        "status": "PASS_M1140CA_INDEPENDENT_BOUNDED_SCHEDULE_HAMMER__AUTHOR_PRODUCTION_RELEASE_SOURCE_ONLY",
        "checks": checks,
        "attacks_rejected": attacks,
        "static": static,
        "bounded": oracle,
        "independent_derivability": derivability,
        "derivability": small["derivability"],
        "production": {
            "m410_rows_opened": m410_opens, "rows_opened": 0, "records": 0,
            "release_outer": None, "digest_compiler": False,
            "real_driver": False, "full_eda_gpu_remote": False,
        },
        "frozen_identity_before_after_equal": True,
        "source_sha256": EXPECTED["source"],
        "contract_identity": [EXPECTED["contract"], EXPECTED["contract_side"],
                              EXPECTED["contract_outer"]],
        "docs359_sha256": EXPECTED["docs359"],
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": checks,
                      "attacks": len(attacks)}, sort_keys=True))


if __name__ == "__main__":
    main()
