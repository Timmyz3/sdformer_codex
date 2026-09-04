#!/usr/bin/env python3
"""M1560 source for one D0/call0 three-axis non-product diagnostic replay.

The run entry exists so an independent release review can pin it.  No run is
authorized by the author receipt.  A future released invocation consumes one
fresh output namespace and never retries automatically.
"""
from __future__ import print_function

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import time


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HERE / "build_m1543_ep34_decoder_nonproduct_streaming_single_call_pilot_source.py"
SOURCE_SHA256 = "a2fd0e3b1d5fbadcb18ccbadd7b4f709114abb22a19b6c92eec940afab5f9dfa"
M1559 = HW / "reviews/m1559_m1556_decoder_immutable_snapshot_regression_r1_20260901"
M1559_REVIEW_SHA256 = "9b34ec5d2e2fd7fb3a934e864b0cd975b6a9c2306c8f7fe80e5c77f6530c1185"
M1559_OUTER_SHA256 = "ae36fe2c2a6643623c6840577cf828587d07508dcc15f36d3ec4922fc0921399"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
LOCK_PATH = Path("/tmp/m1560_ep34_decoder_d0_call0_nonproduct.lock")
CONFIGS = ("DENSE_TYPED_K8", "BIT_EQUAL_SERVICE_K1X8", "BIT_TYPED_K8")
MIN_MEMORY_BYTES = 16 * 1024 * 1024 * 1024
MIN_DISK_BYTES = 16 * 1024 * 1024 * 1024
SCHEMA = "m1560_ep34_decoder_d0_call0_nonproduct_one_shot_source_r1_v1"
STATUS = "SOURCE_ONLY__INDEPENDENT_RELEASE_REVIEW_REQUIRED__ATTEMPT_NOT_CONSUMED"


class M1560Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1560Error(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path, expected, label):
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise M1560Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be a regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


def strict_json(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           M1560Error("nonfinite JSON " + token)))
    require(type(value) is dict, "JSON root must be object")
    return value


def verify_m1559():
    review = M1559 / "review.json"
    outer = M1559 / "SHA256SUMS.seal.sha256"
    sums = M1559 / "SHA256SUMS"
    regular_exact(review, M1559_REVIEW_SHA256, "M1559 review")
    regular_exact(outer, M1559_OUTER_SHA256, "M1559 outer seal")
    require(outer.read_text(encoding="ascii").split() ==
            [sha256(sums), "SHA256SUMS"], "M1559 outer content drift")
    value = strict_json(review)
    require(value.get("status") ==
            "PASS_M1559_M1556_ORDINARY_CONSISTENCY_REGRESSION__ONE_SHOT_DIAGNOSTIC_RELEASE_MAY_BE_AUTHORED__NO_EXECUTION" and
            value.get("decision", {}).get(
                "prerequisite_for_one_d0_call0_three_nonproduct_diagnostic_met") is True and
            value.get("decision", {}).get("actual_run_authorized_by_this_review") is False,
            "M1559 decision drift")
    return value


def load_bound_source():
    regular_exact(SOURCE, SOURCE_SHA256, "M1556 source")
    spec = importlib.util.spec_from_file_location("m1560_bound_m1556", str(SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot import M1556 source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(tuple(module.M.CONFIGS) == CONFIGS and
            module.FORBIDDEN_CONFIG == "PRODUCT_CAPTURE_TYPED_K8" and
            module.describe()["pilot"] == {"call_ordinal": 0,
                "sample_id": 10, "module_ordinal": 0,
                "timesteps": 10, "execution": False},
            "M1556 execution boundary drift")
    return module


def mem_available_bytes():
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise M1560Error("MemAvailable unavailable")


def validate_output(output):
    output = Path(output)
    require(not os.path.lexists(str(output)), "one-shot output already exists")
    parent = output.parent.resolve()
    require(parent.is_dir() and not parent.is_symlink(), "output parent invalid")
    return output, parent


def preflight(output, full_payload=False):
    output, parent = validate_output(output)
    regular_exact(DOCS359, DOCS359_SHA256, "docs359")
    verify_m1559()
    module = load_bound_source()
    memory = mem_available_bytes()
    disk = int(shutil.disk_usage(str(parent)).free)
    require(memory > MIN_MEMORY_BYTES, "less than strict 16 GiB memory headroom")
    require(disk > MIN_DISK_BYTES, "less than strict 16 GiB disk headroom")
    authorities = module.validate_authorities(bool(full_payload))
    return {"schema": SCHEMA,
        "status": "PASS_M1560_ONE_SHOT_SOURCE_PREFLIGHT__NO_EXECUTION",
        "output": str(output.resolve()), "memory_available_bytes": memory,
        "disk_free_bytes": disk, "configurations": list(CONFIGS),
        "authorities": authorities, "attempt_consumed": False,
        "actual_run_authorized_by_author_source": False}


def canonical_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def write_new(path, value):
    path = Path(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(str(path), flags, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value))


def comparisons(results):
    mapped = dict((row["configuration"], row) for row in results)
    require(set(mapped) == set(CONFIGS), "result configuration drift")
    dense = mapped["DENSE_TYPED_K8"]["total_cycles"]
    equal = mapped["BIT_EQUAL_SERVICE_K1X8"]["total_cycles"]
    typed = mapped["BIT_TYPED_K8"]["total_cycles"]
    require(min(dense, equal, typed) > 0, "nonpositive pilot cycle")
    return {"dense_over_bit_equal_cycle_ratio": float(dense) / float(equal),
        "dense_over_bit_typed_cycle_ratio": float(dense) / float(typed),
        "bit_equal_over_bit_typed_cycle_ratio": float(equal) / float(typed),
        "diagnostic_single_call_only": True,
        "paper_citable_performance": False}


def run_once(output):
    output, _parent = validate_output(output)
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise M1560Error("another M1560 attempt holds the lock") from error
        before = preflight(output, full_payload=True)
        output.mkdir()
        started = {"schema": SCHEMA, "status": "WORK_STARTED",
            "attempt_consumed": True, "automatic_retry": False,
            "pid": os.getpid(), "started_unix": time.time(),
            "configurations": list(CONFIGS),
            "source_sha256": SOURCE_SHA256,
            "m1559_review_sha256": M1559_REVIEW_SHA256}
        write_new(output / "WORK_STARTED.json", started)
        module = load_bound_source()
        results = []
        try:
            for ordinal, config in enumerate(CONFIGS):
                row = module.stream_actual_call(config)
                require(row["configuration"] == config and
                        row["pilot_call_ordinal"] == 0 and
                        row["module_ordinal"] == 0 and
                        row["timesteps"] == 10 and
                        row["diagnostic_only"] is True and
                        row["product_capture"] is False and
                        row["production"] is False,
                        "pilot result boundary drift")
                results.append(row)
                write_new(output / ("partial_%d_%s.json" % (ordinal, config)), row)
            require(len(set(row["commit_sequence_sha256"] for row in results)) == 1,
                    "configuration commit sequence mismatch")
            require(len(set(row["resource_manifest_sha256"] for row in results)) == 1,
                    "configuration resource identity mismatch")
            result = {"schema": "m1560_ep34_decoder_d0_call0_nonproduct_diagnostic_result_r1_v1",
                "status": "PASS_M1560_ONE_SHOT_DIAGNOSTIC__INDEPENDENT_RESULT_HAMMER_REQUIRED",
                "identity": {"source_sha256": SOURCE_SHA256,
                    "m1559_review_sha256": M1559_REVIEW_SHA256,
                    "docs359_sha256": DOCS359_SHA256},
                "population": {"call_ordinal": 0, "sample_id": 10,
                    "module_ordinal": 0, "timesteps": 10,
                    "configurations": list(CONFIGS)},
                "preflight": before, "results": results,
                "comparisons": comparisons(results),
                "claim_boundary": {"single_call_diagnostic": True,
                    "paper_citable_performance": False,
                    "system_speedup": False, "energy": False,
                    "rtl": False, "ppa": False, "production": False}}
            write_new(output / "result.json", result)
            write_new(output / "RUN_COMPLETE.json", {"status": result["status"]})
            names = sorted(path.name for path in output.iterdir()
                           if path.is_file())
            sums = "".join("{}  {}\n".format(sha256(output / name), name)
                           for name in names)
            (output / "SHA256SUMS").write_text(sums, encoding="ascii")
            (output / "SHA256SUMS.seal.sha256").write_text(
                "{}  SHA256SUMS\n".format(sha256(output / "SHA256SUMS")),
                encoding="ascii")
            print(result["status"])
            return result
        except Exception as error:
            failure = {"schema": SCHEMA, "status": "FAILED_OR_INCOMPLETE",
                "attempt_consumed": True, "automatic_retry": False,
                "completed_configurations": len(results),
                "exception_type": type(error).__name__,
                "exception": str(error)}
            write_new(output / "FAILED_OR_INCOMPLETE.json", failure)
            raise


def describe():
    return {"schema": SCHEMA, "status": STATUS,
        "population": {"call_ordinal": 0, "sample_id": 10,
            "module_ordinal": 0, "timesteps": 10,
            "configurations": list(CONFIGS)},
        "one_shot": {"fresh_output": True, "exclusive_lock": True,
            "automatic_retry": False, "partial_after_each_configuration": True},
        "execution": {"attempt_consumed": False, "pilot": False,
            "production": False, "product": False},
        "claim_boundary": {"source_only": True,
            "paper_citable_performance": False, "system_speedup": False,
            "energy": False, "rtl": False, "ppa": False}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-payload-members", action="store_true")
    args = parser.parse_args(argv)
    if args.describe:
        require(args.output is None and not args.verify_payload_members,
                "describe accepts no execution arguments")
        value = describe()
    elif args.preflight:
        require(args.output is not None, "preflight requires output")
        value = preflight(args.output, args.verify_payload_members)
    else:
        require(args.output is not None and args.verify_payload_members,
                "run requires output and --verify-payload-members")
        run_once(args.output)
        return 0
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
