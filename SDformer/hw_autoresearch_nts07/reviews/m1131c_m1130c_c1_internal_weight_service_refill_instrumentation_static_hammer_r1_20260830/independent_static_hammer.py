#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author static/bounded M1131C hammer; never runs canonical/full/EDA."""
from __future__ import annotations

import ast
from dataclasses import fields, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import stat
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/build_m1130c_c1_internal_weight_service_refill_instrumentation_source.py"
CONTRACT = HW / "contracts/m1130c_c1_internal_weight_service_refill_instrumentation_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1130c_c1_internal_weight_service_refill_instrumentation_author_receipt_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUTPUT = HERE / "mechanical_checks.json"
SOURCE_SHA = "ce157e7b4b8b9507ba71948fd4b7fcef4145fb24e3252097b5e50b68cf519eaf"
CONTRACT_ID = (
    "20ff9026f8dbc25ad0e9813107a6e97a96f1e379244dcb26ffb51d3a972bcfab",
    "49c2e9599a2c87807717f87f7c117844ad056cefe660c71bffb564d0413de745",
    "efc4bc08d3634531b99c1e45d1ce20c362bb5ca74249d9f2a6877b857af9352a",
)
AUTHOR_OUTER = "f9ce60c54bc016378cd7c0727cb471b0629de4a2e43b24567a7aeb40163efa36"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


class Reject(RuntimeError): pass
checks = 0
attacks: list[str] = []


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value: raise Reject(message)


def rejected(label, action) -> None:
    try: action()
    except Exception:
        attacks.append(label); return
    raise Reject("attack accepted: " + label)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def strict_pairs(rows):
    out = {}
    for key, value in rows:
        require(key not in out, "duplicate JSON key"); out[key] = value
    return out


def load_json(path: Path):
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(), "direct JSON")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(Reject("nonfinite " + token)))


def manifest_rows(path: Path):
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.split(None, 1)
        require(len(item) == 2 and re.fullmatch(r"[0-9a-f]{64}", item[0]) is not None, "manifest row")
        name = item[1].lstrip("*"); rel = Path(name)
        require(name not in out and name == rel.as_posix() and not rel.is_absolute() and ".." not in rel.parts,
                "manifest path")
        out[name] = item[0]
    return out


def verify_flat(directory: Path, outer_sha: str):
    manifest = directory / "SHA256SUMS"; outer = directory / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and not directory.is_symlink() and sha(outer) == outer_sha and
            outer.read_text().split() == [sha(manifest), "SHA256SUMS"], "flat outer")
    expected = manifest_rows(manifest); actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}: continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "live symlink")
        if stat.S_ISREG(mode): actual.add(name)
        else: require(stat.S_ISDIR(mode), "special member")
    require(actual == set(expected), "exact members")
    for name, digest in expected.items(): require(sha(directory / name) == digest, "member identity")
    return load_json(directory / "review.json")


def load_subject():
    spec = importlib.util.spec_from_file_location("m1131c_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "subject spec")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    before = {path: sha(path) for path in (SOURCE, CONTRACT, DOCS359)}
    require(before[SOURCE] == SOURCE_SHA and before[CONTRACT] == CONTRACT_ID[0] and
            before[DOCS359] == DOCS359_SHA, "primary identities")
    side = Path(str(CONTRACT) + ".sha256"); outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    require(sha(side) == CONTRACT_ID[1] and sha(outer) == CONTRACT_ID[2] and
            side.read_text().split() == [CONTRACT_ID[0], CONTRACT.name] and
            outer.read_text().split() == [CONTRACT_ID[1], side.name], "contract double seal")
    author = verify_flat(AUTHOR, AUTHOR_OUTER); contract = load_json(CONTRACT)
    require(author["status"] ==
            "PASS_M1130C_EVENT_INPUT_INSTRUMENTATION_SOURCE_AUTHOR_RECEIPT__CANONICAL_STOP__DIFFERENT_AUTHOR_HAMMER_REQUIRED" and
            author["identity"]["source_sha256"] == SOURCE_SHA and
            author["identity"]["contract_outer_seal_file_sha256"] == CONTRACT_ID[2], "author receipt identity")
    require(contract["frozen_upstream_audit"]["real_per_beat_weight_event_object_available"] is False and
            contract["frozen_upstream_audit"]["canonical_ready"] is False and
            contract["fail_closed_behavior"] == {
                "canonical_iterator_stops_before_row_reader_open": True, "canonical_rows_read": 0,
                "canonical_events_emitted": 0, "aggregate_fields_not_expanded": True,
                "synthetic_events_never_labeled_real": True}, "canonical STOP contract")
    require(contract["direct_event_rule"]["all_fields_must_be_producer_supplied"] is True and
            contract["direct_event_rule"]["count_weight_beat_first_interval_or_capacity_inference_forbidden"] is True,
            "producer-only rule")

    text = SOURCE.read_text(encoding="utf-8"); tree = ast.parse(text)
    funcs = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    iterator_text = ast.get_source_segment(text, funcs["iter_canonical_internal_weight_service_refill_events"])
    consumer_text = ast.get_source_segment(text, funcs["instrument_real_event_inputs"])
    require("CanonicalRowReader" not in text and "iter_canonical_full_replay_results" not in text and
            'audit["canonical_ready"] is True' in iterator_text and
            iterator_text.index('audit["canonical_ready"] is True') < iterator_text.index("yield {}"),
            "STOP before any canonical row/event")
    require("type(event) is InternalWeightServiceRefillEvent" in consumer_text and
            "schedule_native_one_rw" in consumer_text and "validate_exact_once_and_conflicts" in consumer_text and
            not any(token in consumer_text for token in ("weight_beat_first", "interval", "capacity")),
            "direct typed events only")

    module = load_subject()
    audited = module.audit_frozen_internal_event_point()
    require(audited["status"] == "STOP_UPSTREAM_HAS_AGGREGATE_WEIGHT_INTERVAL_NOT_PER_BEAT_ADDRESSED_EVENT" and
            audited["real_internal_weight_service_refill_event_available"] is False and
            audited["canonical_rows_read"] == 0 and audited["canonical_events_emitted"] == 0 and
            audited["aggregate_expansion_allowed"] is False, "real producer absent fail closed")
    required = contract["minimum_upstream_event_fields"]
    actual_fields = [field.name for field in fields(module.InternalWeightServiceRefillEvent)]
    require(actual_fields == required, "all and only producer-supplied fields")
    synthetic = module.source_small_oracle()
    require(synthetic["status"] == "PASS_SYNTHETIC_DIRECT_EVENT_INSTRUMENTATION__CANONICAL_STOP" and
            synthetic["synthetic"] == {"events": 9, "writes": 6, "reads": 3,
                "unique_exact_once_write_ids": 6, "explicitly_stalled_transactions": 3,
                "final_native_1rw_conflicts": 0, "final_weight_half_slot_overlaps": 0} and
            synthetic["canonical_iterator_stopped_before_row_open"] is True and
            synthetic["full_51840000_replayed"] is False, "bounded 9/6/3/6/3/0 oracle")
    rejected("aggregate_count", lambda: module.instrument_real_event_inputs([{"count": 9}]))
    rejected("aggregate_first_beat", lambda: module.instrument_real_event_inputs([{"weight_beat_first": 0}]))
    rejected("aggregate_interval", lambda: module.instrument_real_event_inputs([{"start": 0, "beats": 6, "half_slot": 0}]))
    rejected("aggregate_capacity", lambda: module.instrument_real_event_inputs([{"capacity": 24}]))
    rejected("canonical_iterator", lambda: next(module.iter_canonical_internal_weight_service_refill_events()))
    digest = "0" * 64
    event = module.InternalWeightServiceRefillEvent(
        "candidate", 0, 0, 5, "WRITE", 0, 0, 0, 0, tuple(range(8)), 128,
        (0xffff,) * 8, 8, 0, 0, module.exact_once_id("candidate", 0, 0, 0, 0), digest)
    for name, value in (("logical_bank", 1), ("local_row", 1), ("bytes", 127),
                        ("native_macro_activations", 7), ("service_event_exact_once_id", "0" * 64)):
        rejected("producer_field_" + name, lambda name=name, value=value: replace(event, **{name: value}).validate())
    require(before == {path: sha(path) for path in before}, "subject/contract/docs unchanged")
    result = {
        "schema": "m1131c_m1130c_instrumentation_static_hammer_mechanical_r1_v1",
        "status": "PASS_M1131C_M1130C_STATIC_HAMMER__CANONICAL_STOP__AUTHOR_ADDITIVE_UPSTREAM_PRODUCER_SOURCE_ONLY",
        "checks_passed": checks, "attacks_rejected": len(attacks), "attack_labels": attacks,
        "identity": {"source_sha256": sha(SOURCE), "contract_sha256": sha(CONTRACT),
                     "contract_outer_seal_file_sha256": sha(outer),
                     "author_receipt_outer_seal_file_sha256": sha(AUTHOR / "SHA256SUMS.seal.sha256"),
                     "docs359_sha256": sha(DOCS359)},
        "canonical": {"real_per_beat_weight_event_object": False, "rows": 0, "events": 0,
                      "aggregate_inference_allowed": False, "verdict": "STOP"},
        "synthetic": synthetic["synthetic"],
        "execution": {"full_51840000": False, "eda": False, "gpu": False, "remote": False,
                      "subject_modified": False},
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": checks,
                      "attacks": len(attacks)}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
