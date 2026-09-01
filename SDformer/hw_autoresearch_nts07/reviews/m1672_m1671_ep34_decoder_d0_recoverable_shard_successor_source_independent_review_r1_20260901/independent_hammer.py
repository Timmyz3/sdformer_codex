#!/usr/bin/env python3
"""M1672 different-author source hammer for the M1671 full-D0 plan.

This checker imports only source/metadata and installs a hard guard against
opening canonical decoder bitpack payloads.  It never invokes the replay,
GPU, EDA, or a production reducer/publisher.
"""
from __future__ import print_function

import argparse
import ast
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import stat
import sys
from pathlib import Path


HW = Path(__file__).resolve().parents[2]
SOURCE = HW / "system_simulator/scripts/build_m1671_ep34_decoder_d0_recoverable_shard_successor_source.py"
TEST = HW / "system_simulator/tests/test_m1671_ep34_decoder_d0_recoverable_shard_successor_source.py"
CONTRACT = HW / "contracts/m1671_ep34_decoder_d0_recoverable_shard_successor_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1671_ep34_decoder_d0_recoverable_shard_successor_source_author_receipt_r1_20260901"
M1656_RESULT = HW / "results/m1656_decoder_d0_call0_actual_prefix_three_configuration_r1_20260901"
M1666 = HW / "reviews/m1666_m1656_decoder_actual_prefix_result_independent_hammer_r1_20260901"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
FUTURE_REVIEW = HW / "reviews/m1672_m1671_ep34_decoder_d0_recoverable_shard_successor_source_independent_review_r1_20260901"
FUTURE_RELEASE = HW / "contracts/m1673_m1672_m1671_ep34_decoder_d0_recoverable_shard_execution_release_r1_20260901.json"

EXPECTED = {
    "source": "f6f99909265acac768acf3f1f6340e25d422bde2726cc19b60b4a30c602b8e02",
    "test": "db1a64ae42b2885f7ebe7bfc7542cab695b63a7e24275da8858d52d98b2675f5",
    "contract": "5745fd1d1c44507cc20208144c78533bdc6838265cd0611b04cfed23eb90aa6f",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "checkpoint": "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48",
    "resource": "64661d825ee8ddbdccad9c3e09ca5e41c5ea9cfc75bcea394667dcfd91b4de10",
    "m1656_result": "badb856d74beb9a4a618a8e2cfa53f17f7fc08b42d73c98ec026258b2dfe0eb5",
    "m1666_review": "1acd2380365c1d89750f82cf1623d68ad77147355ebbba7b6d2c83597d6eda29",
}
CONFIGS = ("DENSE_TYPED_K8", "BIT_EQUAL_SERVICE_K1X8", "BIT_TYPED_K8")


class HammerError(Exception):
    pass


def need(value, message):
    if not value:
        raise HammerError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path):
    try:
        mode = path.lstat().st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode) and not stat.S_ISLNK(mode)


def strict_json(path):
    def pairs(rows):
        output = {}
        for key, value in rows:
            need(key not in output, "duplicate JSON key " + key)
            output[key] = value
        return output
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          HammerError("nonfinite JSON " + token)))


def verify_file_seal(path, expected):
    need(regular(path) and sha(path) == expected, "file identity drift " + str(path))
    sums = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    need(regular(sums) and regular(outer), "file seal absent " + str(path))
    need(sums.read_text() == expected + "  " + path.name + "\n", "file inner seal drift")
    need(outer.read_text() == sha(sums) + "  " + sums.name + "\n", "file outer seal drift")


def verify_tree(root, label, strict_population=True):
    need(root.is_dir() and not root.is_symlink(), label + " tree absent/symlink")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(regular(manifest) and regular(outer), label + " seal absent")
    need(outer.read_text() == sha(manifest) + "  SHA256SUMS\n", label + " outer drift")
    listed = {}
    for line in manifest.read_text().splitlines():
        fields = line.split("  ", 1)
        need(len(fields) == 2, label + " malformed manifest")
        digest, name = fields
        need(name not in listed and not Path(name).is_absolute() and
             ".." not in Path(name).parts, label + " unsafe/duplicate member")
        listed[name] = digest
        member = root / name
        need(regular(member) and sha(member) == digest, label + " member drift " + name)
    actual = set()
    for base, dirs, files in os.walk(str(root), followlinks=False):
        base_path = Path(base)
        for name in dirs:
            point = base_path / name
            need(not point.is_symlink() and any(point.iterdir()), label + " symlink/empty dir")
        for name in files:
            point = base_path / name
            need(not point.is_symlink(), label + " symlink file")
            if regular(point):
                actual.add(point.relative_to(root).as_posix())
    expected_population = set(listed) | {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    missing = sorted(expected_population - actual)
    extra = sorted(actual - expected_population)
    need(not missing, label + " sealed members missing")
    if strict_population:
        need(not extra, label + " unsealed extra population " + repr(extra))
    return {"manifest_entries": len(listed), "manifest_sha256": sha(manifest),
            "outer_seal_file_sha256": sha(outer),
            "unsealed_extra_files": extra,
            "strict_recursive_population": not extra}


def install_payload_open_guard():
    real_io_open = io.open
    real_os_open = os.open
    observations = {"allowed_opens": 0, "forbidden_payload_opens": 0}

    def forbidden(file_name):
        try:
            text = os.fspath(file_name)
        except TypeError:
            return False
        if isinstance(text, bytes):
            text = text.decode(errors="replace")
        return ("/m1521_ep34_decoder_positive_planes_s30_c120_r1_20260831/payloads/" in text or
                text.endswith(".positive.le.bitpack") or text.endswith(".negative.le.bitpack"))

    def guarded_io(file_name, *args, **kwargs):
        if forbidden(file_name):
            observations["forbidden_payload_opens"] += 1
            raise HammerError("canonical payload open forbidden by M1672")
        observations["allowed_opens"] += 1
        return real_io_open(file_name, *args, **kwargs)

    def guarded_os(file_name, *args, **kwargs):
        if forbidden(file_name):
            observations["forbidden_payload_opens"] += 1
            raise HammerError("canonical payload os.open forbidden by M1672")
        observations["allowed_opens"] += 1
        return real_os_open(file_name, *args, **kwargs)

    io.open = guarded_io
    os.open = guarded_os
    return observations, real_io_open, real_os_open


def load_source():
    spec = importlib.util.spec_from_file_location("m1672_reviewed_m1671", str(SOURCE))
    need(spec is not None and spec.loader is not None, "cannot load M1671 source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_rows(module, shard, ordinal):
    rows = []
    cycles = (100 + ordinal % 7, 70 + ordinal % 5, 50 + ordinal % 3)
    common = hashlib.sha256(("commit-%d" % ordinal).encode()).hexdigest()
    for config, cycle in zip(CONFIGS, cycles):
        rows.append({"configuration": config,
            "resource_manifest_sha256": EXPECTED["resource"],
            "per_request_miter": True, "per_destination_miter": True,
            "shard_reset_boundary": True, "total_cycles": cycle,
            "request_count": shard["destination_count"] * 4,
            "kind_counts": {"commit": shard["destination_count"] * 4},
            "byte_counts": {"commit": shard["destination_count"] * 384},
            "packed_commit_sequence_sha256": common})
    return rows


def complete_rows(module):
    rows = []
    for ordinal in range(module.TOTAL_SHARDS):
        shard = module.shard_descriptor(ordinal)
        rows.append({"shard": shard, "shard_ordinal": ordinal,
            "checkpoint_sha256": EXPECTED["checkpoint"],
            "resource_manifest_sha256": EXPECTED["resource"],
            "metrics": metric_rows(module, shard, ordinal)})
    return rows


def expect_reject(label, function):
    try:
        function()
    except Exception as error:
        need(type(error).__name__ in ("M1671Error", "HammerError", "ValueError", "KeyError", "TypeError"),
             "unexpected exception for mutation %s: %s" % (label, type(error).__name__))
        return label
    raise HammerError("mutation not rejected: " + label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    observations, real_io_open, real_os_open = install_payload_open_guard()
    errors = []
    try:
        verify_file_seal(CONTRACT, EXPECTED["contract"])
        need(regular(SOURCE) and sha(SOURCE) == EXPECTED["source"], "source SHA drift")
        need(regular(TEST) and sha(TEST) == EXPECTED["test"], "test SHA drift")
        need(regular(DOC359) and sha(DOC359) == EXPECTED["docs359"], "docs359 drift")
        author_seal = verify_tree(AUTHOR, "M1671 author receipt")
        m1656_seal = verify_tree(M1656_RESULT, "M1656 result")
        # M1671 calls this predecessor recursively sealed, but its own
        # verify_flat_tree checks only listed flat members.  Continue the
        # semantic audit while retaining any unsealed population as a P1.
        m1666_seal = verify_tree(M1666, "M1666 review", strict_population=False)
        need(sha(M1656_RESULT / "result.json") == EXPECTED["m1656_result"], "M1656 result drift")
        need(sha(M1666 / "review.json") == EXPECTED["m1666_review"], "M1666 review drift")

        contract = strict_json(CONTRACT)
        need(contract["status"] == "SOURCE_ONLY__FULL_D0_RECOVERABLE_SHARD_GRID__DIFFERENT_AUTHOR_REVIEW_REQUIRED__NO_PAYLOAD_NO_EXECUTION",
             "contract source-only status drift")
        need(contract["source"]["sha256"] == EXPECTED["source"] and
             contract["test"]["sha256"] == EXPECTED["test"], "contract source/test identity")
        need(contract["frozen_identity"]["checkpoint_sha256"] == EXPECTED["checkpoint"] and
             contract["frozen_identity"]["resource_manifest_sha256"] == EXPECTED["resource"],
             "contract frozen identity")
        need(contract["authorization"] == {"source_only": True,
             "different_author_review": True, "release_authoring": False,
             "payload_open": False, "attempt_write": False,
             "shard_execution": False, "reducer_execution": False,
             "automatic_retry": False, "gpu_runs": 0, "eda_runs": 0,
             "all_other_runs": 0}, "contract authorization drift")

        module = load_source()
        need(module.CONFIGS == CONFIGS and module.FORBIDDEN_CONFIG == "PRODUCT_CAPTURE_TYPED_K8",
             "configuration boundary drift")
        grid = module.validate_grid()
        need(grid == {"calls": 30, "timesteps": 300, "destinations": 360000,
                      "shards": 8700, "gap_count": 0, "overlap_count": 0},
             "grid conservation drift")
        first = module.shard_descriptor(0)
        last = module.shard_descriptor(8699)
        need((first["call_ordinal"], first["timestep"], first["destination_start"],
              first["destination_stop_exclusive"], first["destination_count"]) == (0, 0, 0, 42, 42),
             "first shard drift")
        need((last["call_ordinal"], last["timestep"], last["destination_start"],
              last["destination_stop_exclusive"], last["destination_count"]) == (116, 9, 1176, 1200, 24),
             "last shard drift")
        need(28 * 42 + 24 == 1200 and 30 * 10 * 29 == 8700, "shard arithmetic drift")

        synthetic = module.synthetic_shard()
        need(synthetic["status"] == "PASS_M1671_SYNTHETIC_SHARD__NO_PAYLOAD_NO_EXECUTION" and
             all(row["per_request_miter"] and row["per_destination_miter"]
                 for row in synthetic["metrics"]), "synthetic miter drift")

        rows = complete_rows(module)
        reduction = module.reduce_complete_shards(rows)
        need(reduction["complete_shards"] == 8700 and
             reduction["full_d0_population_covered"] is True and
             reduction["shard_isolated_cycle_model"] is True and
             reduction["monolithic_full_call"] is False and
             reduction["full_decoder"] is False and
             reduction["system_speedup"] is False, "complete reducer boundary drift")
        totals = reduction["configuration_totals"]
        need(reduction["ratio_of_sums"]["dense_to_bit_typed"] == {
             "numerator": totals[CONFIGS[0]]["cycles"],
             "denominator": totals[CONFIGS[2]]["cycles"]} and
             reduction["ratio_of_sums"]["bit_equal_to_bit_typed"] == {
             "numerator": totals[CONFIGS[1]]["cycles"],
             "denominator": totals[CONFIGS[2]]["cycles"]},
             "integer ratio-of-sums drift")

        rejected = []
        rejected.append(expect_reject("incomplete_8699", lambda: module.reduce_complete_shards(rows[:-1])))
        bad = copy.deepcopy(rows); bad[1]["shard_ordinal"] = 0
        rejected.append(expect_reject("wrong_shard_order", lambda: module.reduce_complete_shards(bad)))
        base_shard = module.shard_descriptor(0)
        base_metrics = metric_rows(module, base_shard, 0)
        for label, mutate in (
            ("configuration_order", lambda x: x.reverse()),
            ("resource_manifest", lambda x: x[0].update(resource_manifest_sha256="0" * 64)),
            ("request_miter", lambda x: x[0].update(per_request_miter=False)),
            ("destination_miter", lambda x: x[1].update(per_destination_miter=False)),
            ("commit_population", lambda x: x[2]["kind_counts"].update(commit=0)),
            ("cross_config_commit", lambda x: x[2].update(packed_commit_sequence_sha256="0" * 64)),
            ("nonpositive_cycles", lambda x: x[0].update(total_cycles=0)),
        ):
            candidate = copy.deepcopy(base_metrics); mutate(candidate)
            rejected.append(expect_reject(label, lambda c=candidate:
                                           module.validate_three_configuration_metrics(c, base_shard)))

        forged = copy.deepcopy(base_metrics)
        for row in forged:
            row["request_count"] = -7
            row["byte_counts"] = {"commit": -99}
            row.pop("packed_transaction_address_sha256", None)
            row.pop("destination_state_chain_sha256", None)
        module.validate_three_configuration_metrics(forged, base_shard)
        reducer_accepts_unsealed_or_incomplete_metric_rows = True

        tree = ast.parse(SOURCE.read_text())
        functions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        source_text = SOURCE.read_text()
        execution_functions = sorted(functions & {
            "consume_attempt", "run_shard", "_run_authorized_shard", "seal_shard",
            "publish_shard", "resume_shards", "execute_full_d0"})
        namespace_symbols = [name for name in ("RESULT", "ATTEMPT", "WORK", "FAILURE")
                             if re_search_assignment(tree, name)]
        main_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        main_text = "\n".join(source_text.splitlines()[main_node.lineno - 1:])
        cli_execution_modes = [flag for flag in ("--run", "--execute", "--reduce", "--resume", "--shard")
                               if flag in main_text]
        private_execution_target_present = bool(execution_functions)
        exact_runtime_namespaces_present = len(namespace_symbols) == 4
        attempt_before_payload_executable = ("consume_attempt" in functions and private_execution_target_present)
        atomic_sealed_shard_publisher_present = ("seal_shard" in functions and "publish_shard" in functions)

        p1 = [
            {"id": "P1_NO_EXECUTABLE_SHARD_OR_ATOMIC_PUBLISH_CLOSURE",
             "finding": "M1671 contains the grid, scheduler session and pure reducer, but no payload-to-shard execution target, RESULT/ATTEMPT/WORK/FAILURE namespaces, attempt consumer, atomic sealed-shard publisher or resume verifier. attempt-before-payload and no-retry are declarative only; M1673 cannot safely authorize execution of this identity."},
            {"id": "P1_REDUCER_ACCEPTS_UNSEALED_INCOMPLETE_METRICS",
             "finding": "reduce_complete_shards accepts in-memory dictionaries rather than recursively sealed shard receipts, while validate_three_configuration_metrics accepts negative request/byte counts and omits address/destination-chain requirements. A forged 8700-row population can therefore satisfy identity/order and alter totals."},
        ]
        if m1666_seal.get("unsealed_extra_files"):
            p1.append({"id": "P1_M1666_PREDECESSOR_NOT_RECURSIVELY_CLOSED",
                "finding": "The sealed M1666 flat members rehash, but its tree contains unsealed __pycache__ files. M1671 verify_flat_tree does not compare actual recursive population, so its claimed recursively sealed predecessor boundary is not executable."})
        p2 = ["The admitted model is intentionally shard-isolated, not monolithic full-call; D1 remains excluded and D2/D3 require separate geometry reviews."]
        need(observations["forbidden_payload_opens"] == 0, "payload guard recorded an attempted canonical payload open")
        status = "FAIL_M1672_M1671_DECODER_FULL_D0_SOURCE__NO_M1673_EXECUTION_RELEASE__SUCCESSOR_EXECUTION_CLOSURE_REQUIRED"
        score = 84 if len(p1) == 3 else 87
    except Exception as error:
        errors.append("%s: %s" % (type(error).__name__, error))
        author_seal = locals().get("author_seal", {})
        m1656_seal = locals().get("m1656_seal", {})
        m1666_seal = locals().get("m1666_seal", {})
        grid = locals().get("grid", {})
        reduction = locals().get("reduction", {})
        totals = locals().get("totals", {})
        rejected = locals().get("rejected", [])
        reducer_accepts_unsealed_or_incomplete_metric_rows = locals().get(
            "reducer_accepts_unsealed_or_incomplete_metric_rows", None)
        execution_functions = locals().get("execution_functions", [])
        namespace_symbols = locals().get("namespace_symbols", [])
        cli_execution_modes = locals().get("cli_execution_modes", [])
        private_execution_target_present = locals().get("private_execution_target_present", None)
        exact_runtime_namespaces_present = locals().get("exact_runtime_namespaces_present", None)
        attempt_before_payload_executable = locals().get("attempt_before_payload_executable", None)
        atomic_sealed_shard_publisher_present = locals().get("atomic_sealed_shard_publisher_present", None)
        p1 = []
        p2 = []
        status = "FAIL_M1672_M1671_SOURCE_AUDIT_INTERNAL_OR_IDENTITY_ERROR__NO_RELEASE"
        score = 0
    finally:
        io.open = real_io_open
        os.open = real_os_open

    review = {
        "schema": "m1672_m1671_ep34_decoder_d0_recoverable_shard_successor_source_independent_review_r1_v1",
        "milestone": "M1672", "date_cst": "2026-09-01", "status": status,
        "verdict": "FAIL_CLOSED_NO_M1673_EXECUTION_RELEASE", "score_out_of_100": score,
        "p0": errors, "p0_count": len(errors), "p1": p1, "p1_count": len(p1),
        "p2": p2, "p2_count": len(p2),
        "identity": {"source_sha256": sha(SOURCE), "test_sha256": sha(TEST),
            "contract_sha256": sha(CONTRACT), "docs359_sha256": sha(DOC359),
            "checkpoint_sha256": EXPECTED["checkpoint"],
            "resource_manifest_sha256": EXPECTED["resource"],
            "author_receipt_seal": author_seal, "m1656_result_seal": m1656_seal,
            "m1666_review_seal": m1666_seal},
        "verified_static_model": {"grid": grid,
            "calls": 30, "timesteps_per_call": 10,
            "destinations_per_timestep": 1200, "output_blocks": 4,
            "nominal_shard_destinations": 42, "last_shard_destinations": 24,
            "shards_per_timestep": 29, "total_shards": 8700,
            "configuration_order": list(CONFIGS),
            "per_request_reference_compact_miter_source_present": True,
            "per_destination_cumulative_miter_source_present": True,
            "rss_absolute_limit_kib": 2097152, "rss_increment_limit_kib": 524288,
            "pure_complete_reducer_ratio_of_sums": reduction.get("ratio_of_sums", {}),
            "shard_isolated_not_monolithic": True},
        "execution_closure_audit": {
            "private_execution_target_present": private_execution_target_present,
            "execution_function_names": execution_functions,
            "exact_result_attempt_work_failure_namespaces_present": exact_runtime_namespaces_present,
            "namespace_symbols_found": namespace_symbols,
            "attempt_before_payload_executable": attempt_before_payload_executable,
            "atomic_sealed_shard_publisher_present": atomic_sealed_shard_publisher_present,
            "cli_execution_modes": cli_execution_modes,
            "automatic_retry_executable_gate_present": False,
            "reducer_accepts_unsealed_or_incomplete_metric_rows": reducer_accepts_unsealed_or_incomplete_metric_rows,
        },
        "negative_testing": {"mutations_rejected": len(rejected),
            "mutation_labels": rejected,
            "forged_negative_request_and_byte_metric_accepted":
                reducer_accepts_unsealed_or_incomplete_metric_rows},
        "payload_guard": {"canonical_payload_open_attempts": observations["forbidden_payload_opens"],
            "canonical_payload_bytes_opened": False, "actual_replay_runs": 0,
            "gpu_runs": 0, "eda_runs": 0},
        "authorization": {"m1673_execution_release": False,
            "payload_open": False, "attempt_write": False, "shard_execution": False,
            "reducer_execution_on_production_rows": False,
            "successor_execution_closure_source": True,
            "successor_different_author_review_required": True},
        "required_successor_repairs": [
            "Bind an exact private payload-to-shard runner and immutable payload FD/hash before release.",
            "Define fresh RESULT/ATTEMPT/WORK/FAILURE namespaces; consume attempt before the first payload open; enforce no retry.",
            "Atomically double-seal each shard and verify recursive seals before resume/reduction.",
            "Validate nonnegative request/byte totals plus address, commit and destination-state digests for every configuration.",
            "Reduce only the exact complete 8700 sealed-shard population with integer ratio-of-sums."],
        "claim_boundary": {"source_only": True, "full_d0_executed": False,
            "cycles": False, "traffic": False, "speedup": False,
            "energy": False, "monolithic_full_call": False,
            "full_decoder": False, "D1_included": False,
            "system_speedup": False, "table_a": False, "paper_result": False},
        "review_execution": {"python_runtime": "%d.%d.%d" % sys.version_info[:3],
            "canonical_payload_bytes_opened": False, "actual_replay_runs": 0,
            "gpu_runs": 0, "eda_runs": 0, "git_commit": False, "git_push": False},
    }
    Path(args.output).write_text(json.dumps(review, ensure_ascii=False, indent=2,
                                           sort_keys=True, allow_nan=False) + "\n")
    print(status)
    return 0 if not errors else 2


def re_search_assignment(tree, name):
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
