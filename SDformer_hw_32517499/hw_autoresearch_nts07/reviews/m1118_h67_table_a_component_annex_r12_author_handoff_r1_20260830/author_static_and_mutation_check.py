#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Author-side M1118 handoff check. Static/CPU only; no EDA or production."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CONFIG = HW / "system_simulator/config/m1118_h67_table_a_component_annex_r12_20260830.json"
BUILDER = HW / "system_simulator/scripts/build_m1118_h67_table_a_component_annex_r12.py"
TESTS = HW / "system_simulator/tests/test_m1118_h67_table_a_component_annex_r12.py"
CONTRACT = HW / "contracts/m1118_h67_table_a_component_annex_r12_contract_r1_20260830.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
IDENTITY = {
    "config": "d4be661225df58a3906f015d867bdc272de48edc6901fefd744813047c513332",
    "builder": "1e72ccebdf8f0ee78a5a885b71fa873e02076b22854657ce69efbf5e4c942c78",
    "tests": "ca123d01123e8cf8591a891c4c8675a456b7b4fa09c412cc082ebd6c7e3df5ae",
    "contract": "e747724a6b5c0533692086eb49a9471cbaa681af04adb5f7df55f2ab2b4e1f11",
    "contract_sidecar": "4465225f2d7c51ab510ab107b4cc82ebb78e3dc499ca51931c3f877b5f0f4439",
    "contract_outer": "0e3507fdeb00e4709eb8ff8de4ed9e7cffd7e17059f29a472f3bcfa1a93e9733",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
OUT = HERE / "mechanical_checks.json"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path, expected):
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink() and
            sha(path) == expected, "identity drift: " + str(path))


for path, key in ((CONFIG, "config"), (BUILDER, "builder"), (TESTS, "tests"),
                  (CONTRACT, "contract"), (Path(str(CONTRACT) + ".sha256"), "contract_sidecar"),
                  (Path(str(CONTRACT) + ".sha256.seal.sha256"), "contract_outer"),
                  (DOCS359, "docs359")):
    regular(path, IDENTITY[key])

require(Path(str(CONTRACT) + ".sha256").read_text(encoding="utf-8") ==
        IDENTITY["contract"] + "  contracts/m1118_h67_table_a_component_annex_r12_contract_r1_20260830.json\n",
        "contract sidecar content")
require(Path(str(CONTRACT) + ".sha256.seal.sha256").read_text(encoding="utf-8") ==
        IDENTITY["contract_sidecar"] +
        "  contracts/m1118_h67_table_a_component_annex_r12_contract_r1_20260830.json.sha256\n",
        "contract outer content")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "import spec")
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


builder = module("m1118_author_builder", BUILDER)
tests = module("m1118_author_tests", TESTS)
suite = unittest.defaultTestLoader.loadTestsFromTestCase(tests.M1118Tests)
test_result = unittest.TestResult(); suite.run(test_result)
require(test_result.testsRun == 21 and not test_result.failures and not test_result.errors,
        "M1118 unit tests")
preview = builder.build(CONFIG)
require(preview["component_annex_row_count"] == 3 and
        preview["full_system_table_a_production_rows"] == 0 and
        set(preview["component_annex"]) == {builder.C1, builder.C2, builder.C3} and
        preview["system_speedup_admitted"] is False and
        preview["final_checkpoint_bound"] is False and
        preview["paper_ppa_ready"] is False, "canonical preview boundary")


base = json.loads(CONFIG.read_text(encoding="utf-8"))


def rejected(raw):
    with tempfile.TemporaryDirectory(prefix="m1118_author_mutation_", dir=HERE) as name:
        path = Path(name) / "config.json"; path.write_text(raw, encoding="utf-8")
        try:
            builder.build(path)
        except builder.AnnexError:
            return True
    return False


def mutation(change):
    value = copy.deepcopy(base); change(value)
    return rejected(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


attacks = {
    "duplicate_key": rejected('{"schema":"x","schema":"y"}\n'),
    "nonfinite_nan": rejected('{"schema":NaN}\n'),
    "c1_rtl_escalation": mutation(lambda v: v["additive_component_rows"][builder.C1]
                                  ["claim_boundary"].__setitem__("rtl_speedup", True)),
    "c1_final_checkpoint_escalation": mutation(lambda v: v["additive_component_rows"][builder.C1]
                                  ["claim_boundary"].__setitem__("final_checkpoint_bound", True)),
    "c3_second_row_area_mutation": mutation(lambda v: v["additive_component_rows"][builder.C3]
                                  ["dc_setup_area"].__setitem__("cell_area_um2", "1.0")),
    "c3_second_row_speedup_escalation": mutation(lambda v: v["additive_component_rows"][builder.C3]
                                  ["claim_boundary"].__setitem__("speedup", True)),
    "full_system_row_escalation": mutation(lambda v: v["admission_boundary"].__setitem__(
                                  "table_a_full_system_production_rows", 1)),
    "extra_component_row": mutation(lambda v: v["additive_component_rows"].__setitem__(
                                  "fake", copy.deepcopy(v["additive_component_rows"][builder.C1]))),
}
require(all(attacks.values()), "author mutation escaped")
require(sha(DOCS359) == IDENTITY["docs359"], "docs359 drift")

output = {
    "schema": "m1118_table_a_component_annex_r12_author_mechanical_checks_v1",
    "status": "PASS_M1118_R12_AUTHOR_HANDOFF__DIFFERENT_AUTHOR_HAMMER_REQUIRED__NO_EDA",
    "score": 100,
    "identity": IDENTITY,
    "canonical": {"component_rows": 3, "inherited_rows": 1, "additive_rows": 2,
                  "full_system_table_a_rows": 0, "system_speedup": False,
                  "final_checkpoint_bound": False, "paper_ppa_ready": False},
    "tests": {"unit_tests_run": test_result.testsRun, "unit_test_failures": 0,
              "unit_test_errors": 0, "bounded_attacks": attacks,
              "bounded_attacks_rejected": sum(attacks.values()),
              "bounded_attacks_total": len(attacks)},
    "execution": {"eda": False, "gpu": False, "remote": False,
                  "production": False, "full_system_row_created": False},
    "authorization": {"different_author_static_hammer": True,
                      "eda_or_production": False}}
OUT.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
               encoding="utf-8")
print(output["status"])
