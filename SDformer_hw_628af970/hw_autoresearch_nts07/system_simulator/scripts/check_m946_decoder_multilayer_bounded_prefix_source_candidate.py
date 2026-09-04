#!/usr/bin/env python3
"""Static and identity checker for the M946 DRAFT source candidate."""

import ast
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Optional, Sequence


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HERE / "analyze_m946_decoder_multilayer_bounded_prefix_source_candidate.py"
TEST = HW / "system_simulator/tests/test_m946_decoder_multilayer_bounded_prefix_source_candidate.py"
PYTHON_PATH = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
PYTHON_SHA256 = "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"
PYTHON_VERSION = (3, 10, 18)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source():
    spec = importlib.util.spec_from_file_location("m946_static_source", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import M946 source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_interpreter() -> dict:
    executable = Path(sys.executable).resolve()
    if (executable != PYTHON_PATH or sha256(executable) != PYTHON_SHA256 or
            tuple(sys.version_info[:3]) != PYTHON_VERSION):
        raise RuntimeError(
            "M946 checker requires the exact frozen M925 Python interpreter")
    return {
        "path": str(executable),
        "sha256": PYTHON_SHA256,
        "version": list(PYTHON_VERSION),
    }


def static_check(contract: Path) -> dict:
    interpreter = validate_interpreter()
    source_text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(SOURCE))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    forbidden_imports = sorted(imported.intersection({
        "subprocess", "socket", "requests", "urllib", "paramiko",
        "torch", "tensorflow", "cupy",
    }))
    if forbidden_imports:
        raise RuntimeError("forbidden execution/network import: " +
                           ",".join(forbidden_imports))
    forbidden_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (isinstance(node.func.value, ast.Name) and
                    (node.func.value.id, node.func.attr) in {
                        ("os", "system"), ("os", "popen"),
                        ("subprocess", "run"), ("subprocess", "Popen"),
                    }):
                forbidden_calls.append(
                    node.func.value.id + "." + node.func.attr)
    if forbidden_calls:
        raise RuntimeError("forbidden external execution call")
    module = load_source()
    if tuple(module.ALLOWED_PREFIXES) != (1000, 10000, 100000):
        raise RuntimeError("bounded prefix set drift")
    if tuple(module.ALLOWED_LAYERS) != ("D1", "D2", "D3"):
        raise RuntimeError("layer selector scope drift")
    validation = module.validate_source_candidate(contract)
    self_test = module.source_self_test()
    return {
        "schema": "m946_decoder_multilayer_bounded_prefix_static_check_v1",
        "status": "PASS_M946_STATIC_AND_IDENTITY_CHECK__NO_EXECUTION_RELEASE",
        "source_sha256": sha256(SOURCE),
        "test_sha256": sha256(TEST),
        "contract_sha256": sha256(contract),
        "interpreter": interpreter,
        "forbidden_imports": forbidden_imports,
        "forbidden_calls": forbidden_calls,
        "allowed_prefixes": list(module.ALLOWED_PREFIXES),
        "allowed_layers": list(module.ALLOWED_LAYERS),
        "contract_validation": validation,
        "source_self_test_status": self_test["status"],
        "full_row_authorized": False,
        "paper_citable": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(static_check(args.contract), sort_keys=True,
                     allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
