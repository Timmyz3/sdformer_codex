#!/usr/bin/env python3
"""Independent dual-runtime rehammer of the M1574 permit provenance source.

This test never calls a remote host, loads a checkpoint, runs a capture, or
creates a production payload.  Its only produced payload is a tiny synthetic
three-layer roundtrip in a temporary directory.
"""

from __future__ import print_function

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zlib


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = ROOT / ("neuron_experiments/H9_bipolar_self_attention/entrypoints/"
                 "capture_m1558_motion_ep34_s2_tsbg_reduced_binary_source_r1.py")
AUTHOR_TEST = HW / "tests/test_m1558_motion_ep34_s2_tsbg_reduced_binary_source.py"
CONTRACT = HW / "contracts/m1574_m1565_reduced_binary_permit_provenance_successor_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1574_m1565_reduced_binary_permit_provenance_successor_author_receipt_r1_20260901"
PREDECESSOR = HW / "reviews/m1565_m1564_reduced_binary_permit_gate_successor_independent_rehammer_r1_20260901/review.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    SOURCE: "4bf055ff31510a41882de219898e583509ccab7e9dc841aabcb6d52b20a07bf9",
    AUTHOR_TEST: "bf09307c4afc837171dd51fa49102677f710c66c8d47ba1cc25b351ea2e883da",
    CONTRACT: "c86f8d656824aff89a5767c83b3fe7e9468fa7f2338a9053a9985f03a9d06a52",
    AUTHOR / "review.json": "caa944692d31067a7049209c2bc0bfc34e84daefd8e268b4973e42892774733c",
    AUTHOR / "SHA256SUMS": "5e5c85384e99ffb700399e60aca0395e128912676405100dc74a0c6e94815b6c",
    AUTHOR / "SHA256SUMS.seal.sha256": "73c851c6ff368e7841a93a335d589e6842ab35c7661ab3cda9cc2ba9ac87e334",
    PREDECESSOR: "b77da40d87fea49aab62ee56db129fbcc42f1f8063cd2d5800f690b8afd013ed",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          AssertionError("nonfinite JSON: " + token)))


for _path, _expected in EXPECTED.items():
    require(_path.is_file() and sha256(_path) == _expected,
            "identity drift: " + str(_path))
require((AUTHOR / "SHA256SUMS.seal.sha256").read_text(encoding="ascii").split() ==
        [EXPECTED[AUTHOR / "SHA256SUMS"], "SHA256SUMS"],
        "author outer seal drift")
for _line in (AUTHOR / "SHA256SUMS").read_text(encoding="ascii").splitlines():
    _digest, _name = _line.split(None, 1)
    require(sha256(AUTHOR / _name.strip()) == _digest,
            "author member seal drift: " + _name)

SPEC = importlib.util.spec_from_file_location("m1576_source", str(SOURCE))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class Handle(object):
    def __init__(self, hooks, hook):
        self.hooks = hooks
        self.hook = hook

    def remove(self):
        if self.hook in self.hooks:
            self.hooks.remove(self.hook)


class FakeModule(object):
    def __init__(self, inputs, outputs, beta):
        self.m1552_input_channels = int(inputs)
        self.m1552_output_channels = int(outputs)
        self.m1552_beta_by_tile = list(beta)
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return Handle(self.hooks, hook)

    def fire(self, value):
        for hook in list(self.hooks):
            hook(self, (value,), None)


class FakeModel(object):
    def __init__(self):
        self.table = {
            "ind.patch": FakeModule(4, 4, [3]),
            "ind.fc1": FakeModule(5, 4, [2]),
            "ind.fc2": FakeModule(5, 4, [2]),
        }

    def named_modules(self):
        return list(self.table.items())


class FakeTensor(object):
    def __init__(self, shape, rows):
        self.shape = tuple(shape)
        self.rows = [list(row) for row in rows]


def fake_specs(sample_count):
    return [
        {"layer_id": 0, "target": "PATCH", "module_name": "ind.patch",
         "operator": "Conv2d", "operator_order": 0,
         "input_shape": (1, 1, 4, 1, 2), "output_shape": (1, 1, 4, 1, 2),
         "channel_axis": 2, "input_channels": 4, "output_channels": 4,
         "tokens_per_call": 2, "tokens_s40": 2 * sample_count,
         "input_elements_s40": 8 * sample_count,
         "input_active_s40": 3 * sample_count},
        {"layer_id": 1, "target": "FC1", "module_name": "ind.fc1",
         "operator": "Linear", "operator_order": 1,
         "input_shape": (1, 1, 1, 3, 5), "output_shape": (1, 1, 1, 3, 4),
         "channel_axis": 4, "input_channels": 5, "output_channels": 4,
         "tokens_per_call": 3, "tokens_s40": 3 * sample_count,
         "input_elements_s40": 15 * sample_count,
         "input_active_s40": 6 * sample_count},
        {"layer_id": 2, "target": "FC2", "module_name": "ind.fc2",
         "operator": "Linear", "operator_order": 2,
         "input_shape": (1, 1, 1, 3, 5), "output_shape": (1, 1, 1, 3, 4),
         "channel_axis": 4, "input_channels": 5, "output_channels": 4,
         "tokens_per_call": 3, "tokens_s40": 3 * sample_count,
         "input_elements_s40": 15 * sample_count,
         "input_active_s40": 6 * sample_count},
    ]


def sample_order(count):
    authority = M.M1552.verify_bindings()
    return dict(authority, samples=list(authority["samples"][:count]))


def tensors():
    patch = FakeTensor((1, 1, 4, 1, 2),
                       [[0, 0, 0, 0], [1, -2, 0, 3]])
    fc = FakeTensor((1, 1, 1, 3, 5),
                    [[0, 0, 0, 0, 0], [1, -2, 0, 3, 0],
                     [1, 0, -1, 0, 2]])
    return patch, fc


def must_reject(function):
    try:
        function()
    except (M.M1558Error, AssertionError, AttributeError, TypeError,
            ValueError, zlib.error):
        return True
    raise AssertionError("attack was accepted")


def make_synthetic(root, specs, count):
    estimate = M.estimate_from_specs(specs, count)
    free = estimate["result_upper_bytes"] + M.MIN_FREE_AFTER_BYTES + 1
    return M.issue_synthetic_permit(root, specs, count, free)


def forge_without_constructor(cls, output, inventory, estimate, free_bytes):
    """Try the CPython object.__new__ path that constructor-only tests miss."""
    permit = object.__new__(cls)
    prefix = "_{}__".format(cls.__name__)
    object.__setattr__(permit, prefix + "output", str(Path(output).resolve()))
    object.__setattr__(permit, prefix + "inventory", str(inventory))
    object.__setattr__(permit, prefix + "estimate", dict(estimate))
    object.__setattr__(permit, prefix + "free", int(free_bytes))
    object.__setattr__(permit, prefix + "consumed", False)
    return permit


def run(output):
    contract = strict_json(CONTRACT)
    author = strict_json(AUTHOR / "review.json")
    require(contract["status"] ==
            "SUCCESSOR_SOURCE_ONLY__PRODUCTION_REAL_DISK_AND_TYPED_PROVENANCE__INDEPENDENT_REHAMMER_REQUIRED__NO_REMOTE_NO_CAPTURE",
            "contract status drift")
    require(author["status"] ==
            "PASS_AUTHOR_DUAL_RUNTIME_AND_TYPED_PROVENANCE_REGRESSION__INDEPENDENT_REHAMMER_REQUIRED__NO_REMOTE_NO_CAPTURE",
            "author status drift")

    attacks = []

    def attack(name, function):
        row = {"name": name, "passed": False, "error": None}
        try:
            require(function() is not False, "check returned false")
            row["passed"] = True
        except Exception as error:
            row["error"] = "{}: {}".format(type(error).__name__, str(error))
        attacks.append(row)

    with tempfile.TemporaryDirectory(prefix="m1576.", dir=str(Path(output).parent)) as directory:
        base = Path(directory)
        production_specs = M.frozen_layer_specs()
        production_inventory = M.canonical_sha(production_specs)
        production_estimate = M.estimate_from_specs(production_specs, 40)

        real_disk_usage = shutil.disk_usage
        real_queries = []

        def spying_real_disk_usage(path):
            value = real_disk_usage(path)
            real_queries.append({"path": str(Path(path).resolve()),
                                 "total": int(value.total), "used": int(value.used),
                                 "free": int(value.free)})
            return value

        M.shutil.disk_usage = spying_real_disk_usage
        try:
            production_root = base / "actual_production_permit"
            production_permit = M.issue_preload_permit(production_root)
        finally:
            M.shutil.disk_usage = real_disk_usage
        require(len(real_queries) == 1, "production did not issue exactly one real disk query")
        production_receipt = production_permit.consume(production_root,
                                                        production_inventory)

        attack("01_public_production_signature_output_only",
               lambda: list(inspect.signature(M.issue_preload_permit).parameters) == ["output"])
        attack("02_closure_production_signature_output_only",
               lambda: list(inspect.signature(M._issue_production_permit).parameters) == ["output"])
        attack("03_production_and_synthetic_exact_types_distinct",
               lambda: M._ProductionPreloadPermit is not M._SyntheticPreloadPermit)
        attack("04_authority_factory_deleted_after_closure",
               lambda: not hasattr(M, "_permit_authority"))
        attack("05_actual_disk_usage_queried_resolved_parent",
               lambda: real_queries[0]["path"] == str(base.resolve()))
        attack("06_actual_disk_free_copied_to_receipt",
               lambda: production_receipt["free_bytes_before"] == real_queries[0]["free"])
        attack("07_actual_production_receipt_typed_provenance",
               lambda: (type(production_permit) is M._ProductionPreloadPermit and
                        production_receipt["provenance"] == M.PRODUCTION_PROVENANCE))
        attack("08_public_free_space_override_rejected",
               lambda: must_reject(lambda: M.issue_preload_permit(
                   base / "public_override", real_queries[0]["free"])))
        attack("09_closure_free_space_override_rejected",
               lambda: must_reject(lambda: M._issue_production_permit(
                   base / "closure_override", real_queries[0]["free"])))
        attack("10_direct_production_constructor_rejected",
               lambda: must_reject(lambda: M._ProductionPreloadPermit(
                   base / "direct_prod", production_inventory, production_estimate,
                   real_queries[0]["free"], object())))

        fake = fake_specs(2)
        fake_estimate = M.estimate_from_specs(fake, 2)
        fake_free = fake_estimate["result_upper_bytes"] + M.MIN_FREE_AFTER_BYTES + 1
        attack("11_direct_synthetic_constructor_rejected",
               lambda: must_reject(lambda: M._SyntheticPreloadPermit(
                   base / "direct_syn", M.canonical_sha(fake), fake_estimate,
                   fake_free, object())))

        forgery_evidence = {}
        forged_prod_root = base / "forged_prod_object_new"
        forged_prod = forge_without_constructor(
            M._ProductionPreloadPermit, forged_prod_root, production_inventory,
            production_estimate, real_queries[0]["free"])

        def reject_forged_production():
            try:
                receipt = forged_prod.consume(forged_prod_root, production_inventory)
            except (M.M1558Error, AssertionError, AttributeError, TypeError,
                    ValueError, zlib.error):
                return True
            forgery_evidence["production"] = receipt
            raise AssertionError("forged exact production type issued a receipt")

        attack("12_object_new_production_forgery_rejected",
               reject_forged_production)

        forged_syn_root = base / "forged_syn_object_new"
        forged_syn = forge_without_constructor(
            M._SyntheticPreloadPermit, forged_syn_root, M.canonical_sha(fake),
            fake_estimate, fake_free)

        def reject_forged_synthetic():
            try:
                receipt = forged_syn.consume(forged_syn_root, M.canonical_sha(fake))
            except (M.M1558Error, AssertionError, AttributeError, TypeError,
                    ValueError, zlib.error):
                return True
            forgery_evidence["synthetic"] = receipt
            raise AssertionError("forged exact synthetic type issued a receipt")

        attack("13_object_new_synthetic_forgery_rejected",
               reject_forged_synthetic)

        exact_syn_root = base / "exact_production_inventory_synthetic"
        exact_syn = M.issue_synthetic_permit(
            exact_syn_root, production_specs, 40,
            production_estimate["result_upper_bytes"] + M.MIN_FREE_AFTER_BYTES + 1)
        attack("14_production_mode_rejects_exact_inventory_synthetic_type",
               lambda: must_reject(lambda: M.ReducedBinaryProducer(
                   object(), object(), exact_syn_root, production_specs,
                   {"samples": []}, exact_syn, production_inventory=True)))

        M.shutil.disk_usage = spying_real_disk_usage
        try:
            prod_for_syn_root = base / "production_for_synthetic"
            prod_for_syn = M.issue_preload_permit(prod_for_syn_root)
        finally:
            M.shutil.disk_usage = real_disk_usage
        attack("15_synthetic_mode_rejects_production_type",
               lambda: must_reject(lambda: M.ReducedBinaryProducer(
                   object(), object(), prod_for_syn_root, production_specs,
                   {"samples": []}, prod_for_syn, production_inventory=False)))
        attack("16_provenance_slot_not_caller_assignable",
               lambda: must_reject(lambda: setattr(exact_syn, "provenance",
                                                    M.PRODUCTION_PROVENANCE)))
        attack("17_production_permit_one_shot",
               lambda: must_reject(lambda: production_permit.consume(
                   production_root, production_inventory)))

        syn_once_root = base / "syn_once"
        syn_once = make_synthetic(syn_once_root, fake, 2)
        syn_once.consume(syn_once_root, M.canonical_sha(fake))
        attack("18_synthetic_permit_one_shot",
               lambda: must_reject(lambda: syn_once.consume(
                   syn_once_root, M.canonical_sha(fake))))

        wrong_path_permit = make_synthetic(base / "bound_a", fake, 2)
        attack("19_permit_wrong_output_rejected",
               lambda: must_reject(lambda: wrong_path_permit.consume(
                   base / "bound_b", M.canonical_sha(fake))))
        wrong_inventory_permit = make_synthetic(base / "wrong_inventory", fake, 2)
        attack("20_permit_wrong_inventory_rejected",
               lambda: must_reject(lambda: wrong_inventory_permit.consume(
                   base / "wrong_inventory", "0" * 64)))

        occupied = base / "occupied"
        occupied.mkdir()
        attack("21_fresh_namespace_required",
               lambda: must_reject(lambda: M.issue_synthetic_permit(
                   occupied, fake, 2, fake_free)))
        equal_free = fake_estimate["result_upper_bytes"] + M.MIN_FREE_AFTER_BYTES
        attack("22_strict_post_result_free_equality_rejected",
               lambda: must_reject(lambda: M.issue_synthetic_permit(
                   base / "equal_free", fake, 2, equal_free)))
        attack("23_synthetic_sample_count_zero_rejected",
               lambda: must_reject(lambda: M.issue_synthetic_permit(
                   base / "sample_zero", fake, 0, fake_free)))
        attack("24_synthetic_sample_count_over_40_rejected",
               lambda: must_reject(lambda: M.issue_synthetic_permit(
                   base / "sample_41", fake, 41, fake_free)))

        huge = [dict(row) for row in fake]
        huge[1]["input_active_s40"] = M.MAX_RUNTIME_BYTES
        attack("25_first_principles_estimate_under_12gib",
               lambda: must_reject(lambda: M.estimate_from_specs(huge, 2)))
        attack("26_runtime_raw_strict_cap",
               lambda: must_reject(lambda: M.RuntimeBudget(10).charge(10, 0)))
        attack("27_runtime_disk_strict_cap",
               lambda: must_reject(lambda: M.RuntimeBudget(10).charge(0, 10)))

        samples = sample_order(2)
        valid_root = base / "valid_synthetic"
        permit = make_synthetic(valid_root, fake, 2)
        model = FakeModel()
        producer = M.ReducedBinaryProducer(
            model, M.SyntheticBinaryAdapter(), valid_root, fake, samples, permit,
            production_inventory=False)
        patch, fc = tensors()
        for sample in samples["samples"]:
            producer.begin_sample(sample)
            model.table["ind.patch"].fire(patch)
            model.table["ind.fc1"].fire(fc)
            model.table["ind.fc2"].fire(fc)
            producer.end_sample()
        synthetic_result = producer.finalize_source_result()
        validation = M.validate_binary_result(synthetic_result, fake, samples)
        synthetic_receipt = strict_json(synthetic_result / "preload_permit_receipt.json")
        attack("28_synthetic_roundtrip_validates",
               lambda: (validation["frames"] == 4 and validation["fc_tokens"] == 12 and
                        validation["patch_histogram_rows"] == 2))
        attack("29_synthetic_result_keeps_synthetic_provenance",
               lambda: synthetic_receipt["provenance"] == M.SYNTHETIC_PROVENANCE)
        attack("30_production_release_remains_forbidden",
               lambda: must_reject(M.production_release))

        logical_bytes = sum(path.stat().st_size for path in synthetic_result.iterdir()
                            if path.is_file())
        allocated_bytes = sum(path.stat().st_blocks * 512
                              for path in synthetic_result.iterdir() if path.is_file())

    require(len(attacks) == 30, "attack population drift")
    failed = [row for row in attacks if not row["passed"]]
    result = {
        "schema": "m1576_m1574_permit_provenance_independent_rehammer_runtime_r1_v1",
        "status": ("PASS_M1576_30_ATTACKS" if not failed else
                   "NO_GO_M1576_PROVENANCE_FORGERY_SURVIVES"),
        "runtime": {"implementation": sys.implementation.name,
                    "version": ".".join(str(value) for value in sys.version_info[:3])},
        "attacks": {"count": len(attacks), "passed": len(attacks) - len(failed),
                    "failed": len(failed), "rows": attacks},
        "production_real_disk_usage": {
            "query_count": len(real_queries),
            "first_query": real_queries[0],
            "first_receipt_free_bytes_before": production_receipt["free_bytes_before"],
            "caller_supplied_free_argument": False,
        },
        "synthetic_result_disk_usage": {
            "logical_bytes": logical_bytes,
            "allocated_bytes_st_blocks_x_512": allocated_bytes,
            "result_validated": validation["status"],
            "provenance": synthetic_receipt["provenance"],
        },
        "forgery_evidence": forgery_evidence,
        "side_effects": {"remote": False, "ssh": False, "checkpoint_loaded": False,
                         "gpu": False, "capture": False, "production_payload": False,
                         "release": False, "rtl": False, "eda": False},
    }
    output = Path(output)
    require(not output.exists(), "output exists")
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    print(result["status"] + " attacks={}/30".format(len(attacks) - len(failed)))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return run(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
