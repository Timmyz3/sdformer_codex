#!/usr/bin/env python3
"""Static-only M994 checker; no decoder prefix, EDA, GPU, or remote work."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
DRIVER = HERE / "execute_m994_m982_decoder_canonical_attempt_source_r1.py"
RUNNER = HERE / "run_m998_m994_decoder_canonical_attempt_one_shot.sh"
TEST = HW / "system_simulator/tests/test_m994_m982_decoder_canonical_attempt_source.py"
CONTRACT = HW / "contracts/m994_m982_decoder_canonical_attempt_source_contract_r1_20260829.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load():
    spec = importlib.util.spec_from_file_location("m994_static_driver", DRIVER)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module); return module


def check(contract):
    driver, runner = DRIVER.read_text(), RUNNER.read_text()
    tree = ast.parse(driver)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    if any(isinstance(n.func, ast.Attribute) and n.func.attr in
           ("rmtree", "unlink", "remove", "removedirs") for n in calls):
        raise RuntimeError("destructive attempt cleanup primitive")
    for token in ("after_canonical_mkdir", "M998 attempt already consumed",
                  "os.mkdir(attempt, 0o700)", "B.fsync_dir(attempt.parent)",
                  "M994", "M995", "M996", "M997", "M998"):
        if token not in driver + runner:
            raise RuntimeError("missing M994 token: " + token)
    for forbidden in ("attempt_stage", "attempt-stage", "ATTEMPT_STAGE", ".stage.$$"):
        if forbidden in driver or forbidden in runner:
            raise RuntimeError("random attempt-stage protocol survived: " + forbidden)
    mkdir_pos = runner.index("m998_auth --consume-attempt")
    work_pos = runner.index('/usr/bin/mkdir -m 700 "${m998_work}"')
    if mkdir_pos >= work_pos:
        raise RuntimeError("work can start before canonical attempt consumption")
    module = load()
    source = module.validate_source_contract(contract, RUNNER)
    directed = module.source_self_test()
    return {"schema": "m994_canonical_attempt_source_static_check_v1",
            "status": "PASS_M994_STATIC_SOURCE__NO_REAL_10K",
            "driver_sha256": sha(DRIVER), "runner_sha256": sha(RUNNER),
            "test_sha256": sha(TEST), "contract_sha256": sha(contract),
            "source_status": source["status"],
            "canonical_mkdir_is_consumption": directed["canonical_mkdir_is_consumption"],
            "interrupted_attempt_blocks_retry":
                directed["interrupted_canonical_attempt_blocks_retry"],
            "random_precanonical_directory_created": False,
            "real_10k_executed": False, "eda_gpu_remote_used": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--contract", type=Path, default=CONTRACT)
    print(json.dumps(check(p.parse_args().contract), sort_keys=True))
