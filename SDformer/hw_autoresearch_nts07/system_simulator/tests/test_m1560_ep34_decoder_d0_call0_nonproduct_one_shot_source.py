#!/usr/bin/env python3
"""Source-only tests for M1560; never call the run entry."""
from __future__ import print_function

import importlib.util
from pathlib import Path
import tempfile


SOURCE = Path(__file__).resolve().parent.parent / "scripts/run_m1560_ep34_decoder_d0_call0_nonproduct_one_shot.py"
SPEC = importlib.util.spec_from_file_location("m1560", str(SOURCE))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def rejects(function):
    try:
        function()
    except Exception:
        return
    raise AssertionError("invalid source condition accepted")


def result(config, cycles, commit="same", resource="same"):
    return {"configuration": config, "total_cycles": cycles,
            "commit_sequence_sha256": commit,
            "resource_manifest_sha256": resource}


def main():
    attacks = []
    description = M.describe()
    assert description["population"]["configurations"] == list(M.CONFIGS)
    assert description["execution"] == {"attempt_consumed": False,
        "pilot": False, "production": False, "product": False}
    review = M.verify_m1559()
    assert review["decision"]["separately_sealed_one_shot_release_required"] is True
    bound = M.load_bound_source()
    assert tuple(bound.M.CONFIGS) == M.CONFIGS
    with tempfile.TemporaryDirectory(prefix="m1560_test.") as directory:
        root = Path(directory)
        fresh, parent = M.validate_output(root / "fresh")
        assert fresh.name == "fresh" and parent == root.resolve()
        existing = root / "existing"; existing.mkdir()
        rejects(lambda: M.validate_output(existing)); attacks.append("existing")
        link = root / "link"; link.symlink_to(existing, target_is_directory=True)
        rejects(lambda: M.validate_output(link)); attacks.append("symlink")
    rows = [result(M.CONFIGS[0], 300), result(M.CONFIGS[1], 200),
            result(M.CONFIGS[2], 100)]
    values = M.comparisons(rows)
    assert values["dense_over_bit_equal_cycle_ratio"] == 1.5
    assert values["bit_equal_over_bit_typed_cycle_ratio"] == 2.0
    rejects(lambda: M.comparisons(rows[:2])); attacks.append("missing_axis")
    bad = list(rows); bad[2] = result("PRODUCT_CAPTURE_TYPED_K8", 100)
    rejects(lambda: M.comparisons(bad)); attacks.append("product")
    bad = list(rows); bad[0] = result(M.CONFIGS[0], 0)
    rejects(lambda: M.comparisons(bad)); attacks.append("zero_cycle")
    assert M.MIN_MEMORY_BYTES == M.MIN_DISK_BYTES == 16 * 1024 * 1024 * 1024
    assert len(attacks) == 5
    print("PASS M1560 source tests attacks=5 pilot=0 production=0 product=0")


if __name__ == "__main__":
    main()
