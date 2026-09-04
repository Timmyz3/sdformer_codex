#!/usr/bin/env python3
"""Read-only M1725 hammer for the M1721 TSBG/S2 decision source."""
from __future__ import print_function

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / ("system_simulator/scripts/analyze_m1721_m1707_ep34_tsbg_"
               "b4_b8_s2_fc1_patch_decision_source.py")
TEST = HW / ("system_simulator/tests/test_m1721_m1707_ep34_tsbg_b4_b8_"
             "s2_fc1_patch_decision_source.py")
CONTRACT = HW / ("contracts/m1721_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_"
                 "decision_source_contract_r1_20260901.json")
AUTHOR = HW / ("reviews/m1721_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_"
               "decision_source_author_receipt_r1_20260901")
EXPECTED = {
    SOURCE: "7564842899716491f3d8de9b47e6b2abcf6a1a4d39c8fbe6da4e8e4206812df7",
    TEST: "f27f88d1e215df3d744b3f3e352c56adf811df45e8563a864896f4fa8ff7b433",
    CONTRACT: "1953fb132047ac491ba9099f1fd90c3cdf13b5cea5f3dc9b2593fd8c81aa09b5",
    Path(str(CONTRACT) + ".sha256"): "0708ff2539babf3815a2466bbf051acd44cfd13f01dd1e035626f3b6667e1d3d",
    Path(str(CONTRACT) + ".sha256.seal.sha256"): "ff8177dcbc912940f16cddf3442f7109eadff97dc5552f53bde15aaa15aeb282",
    AUTHOR / "author_receipt.json": "18712b53d6b33124554f42f18d06dfc84c0fd14aa17081a93194009162a7e076",
    AUTHOR / "SHA256SUMS": "aef03fe8e8e9a9adfa2312f8d38432337facf914f691ba5bf12a36fae5157ac6",
    AUTHOR / "SHA256SUMS.seal.sha256": "18fc2f1c13644f8d5a0446070b26cd7cf5f2d04f5e7c0a8f122f2dd264ae4579",
    HW / "docs/359_DATE终局冻结_20260813.md":
        "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def exact(path, expected):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and
            stat.S_ISREG(path.lstat().st_mode) and sha(path) == expected,
            "identity drift: " + str(path))


def verify_seal(root):
    root = Path(root)
    sums = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(outer.read_text().split() == [sha(sums), "SHA256SUMS"],
            "outer seal drift")
    listed = set()
    for line in sums.read_text().splitlines():
        digest, name = line.split(None, 1)
        name = name.lstrip("*")
        require(name not in listed and ".." not in Path(name).parts and
                not Path(name).is_absolute(), "unsafe manifest")
        exact(root / name, digest)
        listed.add(name)
    actual = set(path.relative_to(root).as_posix() for path in root.rglob("*")
                 if path.is_file() and path.name not in
                 ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    require(listed == actual, "manifest coverage drift")


def main():
    for path, digest in EXPECTED.items():
        exact(path, digest)
    verify_seal(AUTHOR)
    contract = json.loads(CONTRACT.read_text())
    require(contract["authorization"]["analysis_run"] is False and
            contract["authorization"]["release"] is False,
            "source-only contract drift")
    spec = importlib.util.spec_from_file_location("m1725_target", str(SOURCE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import numpy as np
    rng = np.random.RandomState(1725)
    for _ in range(1000):
        rows = int(rng.randint(1, 60))
        width = int(rng.randint(1, 24))
        capacity = int(rng.randint(1, width + 1))
        tiles = int(rng.randint(1, 6))
        active = rng.rand(rows, width) < rng.uniform(0.02, 0.95)
        if not active.any():
            active[0, 0] = True
        got = module.exact_lru_entity_stats(active, tiles, capacity, np)
        accesses = []
        for row in range(rows):
            for tile in range(tiles):
                accesses.extend(tile * width + int(group)
                    for group in np.flatnonzero(active[row]).tolist())
        misses, _cache, hits = module._reference_lru(accesses, capacity)
        require((got["accesses"], got["misses"], got["hits"]) ==
                (len(accesses), misses, len(hits)), "vector LRU mismatch")

    active = np.array([[True]], dtype=np.bool_)
    nnz = np.array([[1]], dtype=np.int16)
    magnitude = np.array([[1]], dtype=np.int32)
    debt = module.s2_fc1_pair_metrics(
        active, nnz, magnitude, 32, [1, 1], 0.01, np)
    require(debt["sum_abs_output_code_debt"] == 2,
            "counterexample coordinate drift")
    true_sum_debt = 32
    require(debt["sum_abs_output_code_debt"] < true_sum_debt,
            "expected S2 sum-debt undercount not reproduced")

    class Sentinel(Exception):
        pass
    def stop(_root):
        raise Sentinel("capture reached before analysis authority")
    module.verify_capture_identity = stop
    try:
        module.run_analysis()
    except Sentinel:
        authority_bypass = True
    else:
        authority_bypass = False
    require(authority_bypass, "analysis authority bypass not reproduced")
    require(not os.path.lexists(str(module.RESULT)) and
            not os.path.lexists(str(module.WORK)), "result namespace not fresh")
    print(json.dumps({
        "status": "FAIL_M1725_M1721_SOURCE__NO_RELEASE_NO_ANALYSIS",
        "vector_lru_random_cases": 1000,
        "vector_lru_equivalent": True,
        "s2_reported_sum_debt": 2,
        "s2_true_sum_debt_counterexample": true_sum_debt,
        "analysis_authority_bypass_reproduced": authority_bypass,
        "capture_runs": 0, "analysis_runs": 0, "gpu_runs": 0,
        "eda_runs": 0, "result_writes": 0},
        indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
