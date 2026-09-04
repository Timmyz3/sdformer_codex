#!/usr/bin/env python3
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = (Path(__file__).resolve().parent.parent / "scripts" /
          "build_m1290_decoder_surrogate_production_adapter_r1.py")
SPEC = importlib.util.spec_from_file_location("m1290_adapter", str(SOURCE))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(label):
    return hashlib.sha256(label.encode()).hexdigest()


def summary(count, traffic, base):
    return {"count": count, "traffic_bytes": traffic,
            "address_first": base, "address_last": base + max(count - 1, 0),
            "issue_first": base, "issue_last": base + max(count - 1, 0),
            "return_first": base + 1, "return_last": base + count,
            "commit_first": base + 1, "commit_last": base + count,
            "stall_events": {"none": count}}


def make_rows():
    rows = []
    transaction = 0; cycle = 0
    constants = (17, 23, 31, 41)
    for ordinal in range(120):
        sample = ordinal // 4; module = ordinal % 4; groups = 1000 + ordinal
        terms = groups * (2 + ordinal % 6)
        commit = M.EXPECTED_COMMIT_BYTES[module]; commit_count = commit // 288
        counts = {"input_descriptor_read": groups, "weight_read": groups,
                  "psum_read": groups, "compute": groups, "psum_write": groups,
                  "output_commit": commit_count}
        traffic = {"input_descriptor_read": groups * 16,
                   "weight_read": terms * 16, "psum_read": groups * 288,
                   "compute": groups * 288, "psum_write": groups * 288,
                   "output_commit": commit}
        traffic.update({"total": sum(traffic.values()),
            "external": traffic["input_descriptor_read"] + commit,
            "onchip": traffic["weight_read"] + traffic["psum_read"] +
                      traffic["psum_write"]})
        transaction_count = sum(counts.values())
        cycles = 4 * groups + constants[module]
        row = {
            "schema": "m1111dr2_decoder_address_timed_call_schedule_v2",
            "global_call_ordinal": ordinal,
            "sequence_ordinal": sample // 10,
            "sequence": M.SEQUENCES[sample // 10],
            "sequence_sample_id": sample % 10,
            "module_ordinal": module,
            "module": M.MODULES[module],
            "configuration": M.CONFIGURATION,
            "d1_exact_theta": module == 1,
            "d1_theta_word_uint32": 1065353139 if module == 1 else None,
            "d1_weight_folding": False,
            "transaction_ordinal_first": transaction,
            "transaction_ordinal_last": transaction + transaction_count - 1,
            "transaction_count": transaction_count,
            "address_digest_sha256": digest("address-%d" % ordinal),
            "dependency_digest_sha256": digest("dependency-%d" % ordinal),
            "schedule_digest_sha256": digest("schedule-%d" % ordinal),
            "cycle_start": cycle,
            "cycle_end": cycle + cycles,
            "diagnostic_cycles": cycles,
            "diagnostic_traffic_bytes": traffic,
            "kind_summaries": {kind: summary(counts[kind], traffic[kind],
                                               ordinal * 1000000 + index * 1000)
                               for index, kind in enumerate(M.KINDS)},
            "claim_boundary": {"diagnostic_only": True,
                "speedup_admitted": False, "system_speedup_admitted": False,
                "paper_ppa_ready": False, "final_checkpoint_rebind_required": True},
        }
        rows.append(row); transaction += transaction_count; cycle += cycles
    return rows


def write_result(directory, rows):
    directory.mkdir()
    calls = directory / M.CALLS
    calls.write_text("".join(canonical(row) + "\n" for row in rows))
    calls_sha = sha(calls)
    transaction_count = sum(row["transaction_count"] for row in rows)
    contract = json.loads((M.HW / "contracts/m1111dr2_m1105dr2_decoder_only_production_runner_source_contract_r2_20260830.json").read_text())
    aggregate = {key: sum(row["diagnostic_traffic_bytes"][key] for row in rows)
                 for key in (*M.KINDS, "total", "external", "onchip")}
    payload = {
        "schema": "m1111dr2_m1105dr2_decoder_only_address_timed_result_v2",
        "status": "PASS_M1111DR2_DECODER_ONLY_DIAGNOSTIC_RESULT__FINAL_RESULT_HAMMER_REQUIRED",
        "identity": {"checkpoint": "H67_ep35",
            "checkpoint_sha256": "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158",
            "source_sha256": "b2d8ef4139283de06b7e332429bdf752ad16122ffbeda0ff7d75bce6d816a5c4",
            "contract_sha256": "821819b00503b91a8fb8dfca8fe000208e10746e751a3815131dc8ff1cbed515",
            "m1110d_outer_seal_file_sha256":
                "9caf64e422b4cb696a600b69415bd8265dc4694066fae7ec67a5f34019f39e23",
            "final_checkpoint_rebind_required": True},
        "population": {"calls": 120, "timesteps_per_call": 10,
            "transaction_count": transaction_count, "call_schedule_sha256": calls_sha,
            "call_row_stream_digest_sha256": calls_sha},
        "common_resource": contract["common_resource"],
        "diagnostic": {"cycles": rows[-1]["cycle_end"],
                       "traffic_bytes": aggregate, "ratios_or_speedups": None},
        "claim_boundary": {"decoder_only": True,
            "address_timed_transactions_complete": True,
            "same_resource_schedule_complete": True, "diagnostic_cycles_only": True,
            "diagnostic_traffic_only": True, "speedup_admitted": False,
            "system_speedup_admitted": False, "paper_ppa_ready": False,
            "paper_citable_performance": False,
            "final_checkpoint_rebind_required": True,
            "independent_result_hammer_required": True}}
    (directory / M.PAYLOAD).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (directory / M.COMPLETE).write_bytes(
        b"M1111DR2_DECODER_DIAGNOSTIC_COMPLETE__RESULT_HAMMER_REQUIRED\n")
    seal_result(directory)


def seal_result(directory):
    bundle = directory / M.SEAL_DIR
    if bundle.exists():
        for child in bundle.iterdir(): child.unlink()
    else: bundle.mkdir()
    lines = [sha(directory / name) + "  " + name for name in M.RESULT_FILES]
    (bundle / M.MANIFEST).write_text("\n".join(lines) + "\n")
    (bundle / M.OUTER).write_text(sha(bundle / M.MANIFEST) + "  " + M.MANIFEST + "\n")


def refresh_result(directory, rows):
    calls = directory / M.CALLS
    calls.write_text("".join(canonical(row) + "\n" for row in rows))
    payload = json.loads((directory / M.PAYLOAD).read_text())
    payload["population"]["call_schedule_sha256"] = sha(calls)
    payload["population"]["call_row_stream_digest_sha256"] = sha(calls)
    payload["population"]["transaction_count"] = sum(row["transaction_count"] for row in rows)
    payload["diagnostic"]["cycles"] = rows[-1]["cycle_end"]
    payload["diagnostic"]["traffic_bytes"] = {
        key: sum(row["diagnostic_traffic_bytes"][key] for row in rows)
        for key in (*M.KINDS, "total", "external", "onchip")}
    (directory / M.PAYLOAD).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    seal_result(directory)


def write_hammer(directory, result):
    directory.mkdir(exist_ok=True)
    seal = M.verify_result_seal(result)
    review = {"schema": M.HAMMER_SCHEMA, "status": M.HAMMER_STATUS,
        "identity": {"result_manifest_sha256": seal["manifest_sha256"],
            "result_outer_seal_file_sha256": seal["outer_seal_file_sha256"],
            "result_payload_sha256": seal["members"][M.PAYLOAD],
            "result_calls_sha256": seal["members"][M.CALLS],
            "result_completion_sha256": seal["members"][M.COMPLETE]},
        "verification": {"exact_three_payload_files": True,
            "result_manifest_and_outer_seal": True, "strict_120_call_rows": True,
            "kind_summaries_and_digests": True, "diagnostic_claim_boundary": True},
        "claim_boundary": {"diagnostic_only": True, "analytical_annex": False,
            "speedup": False, "system_speedup": False, "paper_ppa_ready": False}}
    (directory / "review.json").write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    (directory / "RUN_COMPLETE.txt").write_text(M.HAMMER_STATUS + "\n")
    lines = [sha(directory / name) + "  " + name
             for name in ("RUN_COMPLETE.txt", "review.json")]
    (directory / M.MANIFEST).write_text("\n".join(lines) + "\n")
    (directory / M.OUTER).write_text(sha(directory / M.MANIFEST) + "  " + M.MANIFEST + "\n")


class M1290RepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="m1290_fixture_")
        root = Path(self.temp.name)
        self.result = root / "result"; self.hammer = root / "hammer"
        self.rows = make_rows(); write_result(self.result, self.rows)
        write_hammer(self.hammer, self.result)

    def tearDown(self):
        self.temp.cleanup()

    def verified(self):
        return M.verify_production_authorities(self.result, self.hammer)

    def test_exact_production_shape_projects_real_semantics(self):
        rows, authority = self.verified()
        self.assertEqual(len(rows), 120)
        self.assertEqual({row["traffic"]["commit_bytes"] for row in rows},
                         set(M.EXPECTED_COMMIT_BYTES))
        self.assertTrue(all(row["group_count"] <= row["active_source_terms"] <=
                            8 * row["group_count"] for row in rows))
        self.assertEqual(authority["hammer"]["status"], M.HAMMER_STATUS)

    def test_fixture_and_production_apis_are_separate_and_exact_bool(self):
        self.assertEqual(len(inspect.signature(M.calibrate_production).parameters), 0)
        projected, _ = self.verified()
        payload = {"schema": "m1290_projected_fixture_v1", "calls": projected,
                   "claim_boundary": {"synthetic_fixture": True,
                                      "analytical_cycle_annex": False}}
        result = M.calibrate_fixture(payload, True)
        self.assertFalse(result["cycle_surrogate"]["analytical_cycle_annex_allowed"])
        for bad in (0, 1, None, False, "true"):
            with self.assertRaises(M.CalibrationError):
                M.calibrate_fixture(payload, bad)
        with self.assertRaises(TypeError):
            M.calibrate_production({"result_outer_seal_pass": True})

    def test_forged_naked_authority_cannot_enter_production(self):
        fake = Path(self.temp.name) / "fake"; fake.mkdir()
        (fake / "authority.json").write_text(json.dumps({
            "result_outer_seal_pass": True, "result_hammer_pass": True,
            "result_sha256": "0" * 64}) + "\n")
        with self.assertRaises(M.CalibrationError):
            M.verify_production_authorities(fake, self.hammer)

    def test_result_manifest_outer_and_hammer_are_mandatory(self):
        (self.result / M.SEAL_DIR / M.OUTER).write_text("0" * 64 + "  SHA256SUMS\n")
        with self.assertRaises(M.CalibrationError): self.verified()
        seal_result(self.result); write_hammer(self.hammer, self.result)
        (self.hammer / M.OUTER).write_text("0" * 64 + "  SHA256SUMS\n")
        with self.assertRaises(M.CalibrationError): self.verified()

    def test_digest_substitution_breaks_result_seal(self):
        rows = copy.deepcopy(self.rows)
        rows[4]["address_digest_sha256"] = rows[0]["address_digest_sha256"]
        (self.result / M.CALLS).write_text("".join(canonical(row) + "\n" for row in rows))
        with self.assertRaises(M.CalibrationError): self.verified()

    def test_coordinated_row_forgery_resealed_but_old_hammer_rejected(self):
        rows = copy.deepcopy(self.rows); row = rows[0]
        row["kind_summaries"]["weight_read"]["traffic_bytes"] += 16
        row["diagnostic_traffic_bytes"]["weight_read"] += 16
        row["diagnostic_traffic_bytes"]["total"] += 16
        row["diagnostic_traffic_bytes"]["onchip"] += 16
        refresh_result(self.result, rows)
        with self.assertRaises(M.CalibrationError): self.verified()

    def test_group_term_bounds_rejected_even_with_fresh_hammer(self):
        rows = copy.deepcopy(self.rows); row = rows[0]
        row["kind_summaries"]["weight_read"]["traffic_bytes"] = 0
        row["diagnostic_traffic_bytes"]["weight_read"] = 0
        row["diagnostic_traffic_bytes"]["total"] = sum(
            row["diagnostic_traffic_bytes"][kind] for kind in M.KINDS)
        row["diagnostic_traffic_bytes"]["onchip"] = (
            row["diagnostic_traffic_bytes"]["psum_read"] +
            row["diagnostic_traffic_bytes"]["psum_write"])
        refresh_result(self.result, rows); write_hammer(self.hammer, self.result)
        with self.assertRaisesRegex(M.CalibrationError, "active_source_terms"):
            self.verified()

    def test_sequence_and_module_swaps_rejected(self):
        for field, value in (("sequence", M.SEQUENCES[1]),
                             ("module", M.MODULES[1])):
            with self.subTest(field=field):
                rows = copy.deepcopy(self.rows); rows[0][field] = value
                refresh_result(self.result, rows); write_hammer(self.hammer, self.result)
                with self.assertRaises(M.CalibrationError): self.verified()
                self.result = Path(self.temp.name) / ("result_" + field)
                self.hammer = Path(self.temp.name) / ("hammer_" + field)
                write_result(self.result, self.rows); write_hammer(self.hammer, self.result)

    def test_each_layer_requires_30_distinct_observations(self):
        rows = copy.deepcopy(self.rows)
        source = rows[0]
        target = rows[4]
        for key in ("address_digest_sha256", "dependency_digest_sha256",
                    "schedule_digest_sha256", "kind_summaries",
                    "diagnostic_traffic_bytes", "diagnostic_cycles"):
            target[key] = copy.deepcopy(source[key])
        target["cycle_end"] = target["cycle_start"] + target["diagnostic_cycles"]
        # Restore global cycle continuity after changing one observation.
        delta = self.rows[4]["diagnostic_cycles"] - target["diagnostic_cycles"]
        for index in range(5, 120):
            rows[index]["cycle_start"] -= delta; rows[index]["cycle_end"] -= delta
        refresh_result(self.result, rows); write_hammer(self.hammer, self.result)
        with self.assertRaises(M.CalibrationError): self.verified()

    def test_all_four_commit_values_are_enforced(self):
        for module, expected in enumerate(M.EXPECTED_COMMIT_BYTES):
            self.assertEqual(self.rows[module]["diagnostic_traffic_bytes"]["output_commit"],
                             expected)
        rows = copy.deepcopy(self.rows); row = rows[3]
        row["diagnostic_traffic_bytes"]["output_commit"] -= 288
        row["kind_summaries"]["output_commit"]["traffic_bytes"] -= 288
        row["kind_summaries"]["output_commit"]["count"] -= 1
        row["kind_summaries"]["output_commit"]["stall_events"]["none"] -= 1
        row["transaction_count"] -= 1; row["transaction_ordinal_last"] -= 1
        # This mutation need not repair downstream ordinals: any route must reject.
        row["diagnostic_traffic_bytes"]["total"] -= 288
        row["diagnostic_traffic_bytes"]["external"] -= 288
        refresh_result(self.result, rows); write_hammer(self.hammer, self.result)
        with self.assertRaises(M.CalibrationError): self.verified()


if __name__ == "__main__":
    unittest.main()
