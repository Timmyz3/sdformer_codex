#!/usr/bin/env python3
"""Static-only M997 hammer for the exact M996 release; never launches M998."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
RELEASE = HW / "contracts/m996_m994_decoder_canonical_attempt_release_r1_20260829.json"
CONTRACT = HW / "contracts/m994_m982_decoder_canonical_attempt_source_contract_r1_20260829.json"
DRIVER = HW / "system_simulator/scripts/execute_m994_m982_decoder_canonical_attempt_source_r1.py"
RUNNER = HW / "system_simulator/scripts/run_m998_m994_decoder_canonical_attempt_one_shot.sh"
M995 = HW / "reviews/m995_m994_decoder_canonical_attempt_source_hammer_r1_20260829"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED_RELEASE_SHA = "7140608515b165db358c1ccee23c3a23712aff5abd0172b3793993c89bc6fc03"
EXPECTED_M995 = (
    "cea74195cdcef8532e41e3dd6810bd5dbfc0cc225174d9a707383b3bd092f4b8",
    "8b3745a9c449438c6f0618e4514a28c39f05167f024e2cd860118b58724080ca",
    "9a6ea0f3fd321b6c23eb34246f75a6b4737607dc2bb2dc18bcf229981ba5b9c6",
)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(path):
    def pairs(values):
        out = {}
        for key, value in values:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs)


def load_driver():
    spec = importlib.util.spec_from_file_location("m997_independent_m994", DRIVER)
    require(spec is not None and spec.loader is not None, "cannot load M994 driver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_release_value(value, module):
    require(value.get("schema") == module.RELEASE_SCHEMA,
            "release schema drift")
    require(value.get("status") == "AUTHORIZE_ONE_M998_D2_THEN_D3_10K_RUN" and
            value.get("release") is True and value.get("launch_now") is False and
            value.get("max_attempts") == 1, "release cardinality drift")
    require(value.get("exact_rows") == [
        {"layer": "D2", "sample_id": 0, "config": "A1_OSG",
         "timestep": 0, "expanded_prefix": 10000},
        {"layer": "D3", "sample_id": 0, "config": "A1_OSG",
         "timestep": 0, "expanded_prefix": 10000}], "release row/order drift")
    source = value.get("source_binding", {})
    require(source == {
        "m994_contract_sha256": sha(CONTRACT),
        "m994_driver_sha256": sha(DRIVER),
        "m998_runner_sha256": sha(RUNNER),
        "m995_review_sha256": EXPECTED_M995[0],
        "m982_review_sha256": module.M982_ID[0],
    }, "release source binding drift")
    auth = value.get("authorization", {})
    require(auth == {
        "one_m998_d2_then_d3_10k": True,
        "retry": False,
        "d2_or_d3_100k": False,
        "full_row": False,
        "production": False,
        "eda_gpu_remote": False,
    }, "release authorization expansion")
    boundary = value.get("claim_boundary", {})
    require(boundary == {
        "source_prefix_diagnostic_only": True,
        "decoder_complete": False,
        "paper_citable": False,
        "table_a_row": False,
        "system_speedup": False,
    }, "release claim boundary drift")
    require(value.get("docs359_sha256") == module.DOCS359_SHA and
            sha(DOCS359) == module.DOCS359_SHA, "docs359 drift")
    return True


def reject_mutation(value, module, label, mutator):
    candidate = copy.deepcopy(value)
    mutator(candidate)
    try:
        validate_release_value(candidate, module)
    except RuntimeError:
        return label
    raise RuntimeError("release mutation accepted: " + label)


def result_namespace(module):
    parent = module.RESULT.parent
    return sorted(item.name for item in parent.iterdir()
                  if item.name == module.ATTEMPT.name or
                  item.name == module.RESULT.name or
                  item.name.startswith(module.RESULT.name + ".work.") or
                  item.name.startswith(module.FAILURE_PREFIX))


def main():
    require(sha(RELEASE) == EXPECTED_RELEASE_SHA, "M996 release SHA drift")
    module = load_driver()
    source = module.validate_source_contract(CONTRACT, RUNNER)
    release = strict_json(RELEASE)
    validate_release_value(release, module)
    m995_seal = module.verify_flat_review(M995, EXPECTED_M995, "M995")
    m995_review = strict_json(M995 / "review.json")
    require(m995_review.get("status") ==
            "PASS_M995_M994_CANONICAL_ATTEMPT_SOURCE_HAMMER" and
            m995_review.get("verdict") == "GO_AUTHOR_M996_RELEASE_ONLY",
            "M995 authority drift")

    before = result_namespace(module)
    require(not before, "M998 result namespace not fresh")
    mutations = [
        reject_mutation(release, module, "launch_now_true",
                        lambda x: x.__setitem__("launch_now", True)),
        reject_mutation(release, module, "max_attempts_two",
                        lambda x: x.__setitem__("max_attempts", 2)),
        reject_mutation(release, module, "rows_reversed",
                        lambda x: x["exact_rows"].reverse()),
        reject_mutation(release, module, "prefix_100k",
                        lambda x: x["exact_rows"][0].__setitem__("expanded_prefix", 100000)),
        reject_mutation(release, module, "retry_true",
                        lambda x: x["authorization"].__setitem__("retry", True)),
        reject_mutation(release, module, "full_row_true",
                        lambda x: x["authorization"].__setitem__("full_row", True)),
        reject_mutation(release, module, "production_true",
                        lambda x: x["authorization"].__setitem__("production", True)),
        reject_mutation(release, module, "eda_gpu_remote_true",
                        lambda x: x["authorization"].__setitem__("eda_gpu_remote", True)),
        reject_mutation(release, module, "paper_citable_true",
                        lambda x: x["claim_boundary"].__setitem__("paper_citable", True)),
        reject_mutation(release, module, "m995_binding_drift",
                        lambda x: x["source_binding"].__setitem__("m995_review_sha256", "0" * 64)),
    ]
    after = result_namespace(module)
    require(after == before, "static hammer mutated M998 namespace")
    runner = RUNNER.read_text(encoding="utf-8")
    require(runner.index("m998_auth --validate-authority") <
            runner.index("m998_auth --consume-attempt"),
            "attempt consumed before release authority")
    for name in ("M998_EXPECTED_M996_RELEASE_SHA", "M998_EXPECTED_M995_REVIEW_SHA",
                 "M998_EXPECTED_M995_MANIFEST_SHA", "M998_EXPECTED_M995_OUTER_SHA",
                 "M998_EXPECTED_M997_REVIEW_SHA", "M998_EXPECTED_M997_MANIFEST_SHA",
                 "M998_EXPECTED_M997_OUTER_SHA"):
        require(': "${' + name + ':?' in runner, "runner authority pin absent: " + name)
    return {
        "schema": "m997_m996_m994_independent_release_hammer_v1",
        "status": "PASS_M997_M996_M994_CANONICAL_ATTEMPT_RELEASE_HAMMER",
        "verdict": "GO_ONE_M998_RUN_ONLY",
        "release_sha256": EXPECTED_RELEASE_SHA,
        "source_validation": source,
        "m995_seal": m995_seal,
        "negative_mutations_rejected": mutations,
        "negative_mutation_count": len(mutations),
        "attempt_result_work_failure_fresh_before_after": True,
        "real_10k_executed": False,
        "eda_gpu_remote_used": False,
        "docs359_sha256": sha(DOCS359),
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
