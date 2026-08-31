#!/usr/bin/python3.12
"""Independent local-only hammer for the staged M1306 remote selection result."""
from __future__ import annotations

import csv
from decimal import Decimal
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
import tarfile
from types import ModuleType, SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "hw_autoresearch_nts07/system_handoff/incoming/m1306_remote_selection_result_20260830"
ARCHIVE = STAGE / "m1306_remote_selection_result_20260830.tar"
ARCHIVE_SIDECAR = STAGE / "m1306_remote_selection_result_20260830.tar.sha256"
ARCHIVE_SHA = "0524a94ccb36adc7ebc17603dedc322810141d8b14dc743923c5b942a5c6c36f"
RESULTS = STAGE / "hw_autoresearch_nts07/results"
OUT = RESULTS / "m1257_motion_cross_run_final_checkpoint_selection_r5_20260830"
ATTEMPT = RESULTS / ".m1257_motion_cross_run_final_checkpoint_selection_r5_attempt_consumed"
LOG = RESULTS / "m1257_motion_cross_run_final_checkpoint_selection_r5_20260830.launch.log"

M1257_SOURCE = ROOT / "hw_autoresearch_nts07/scripts/run_m1257_m1253_motion_cross_run_final_checkpoint_binder_one_shot_release_successor.py"
M1257_SHA = "ce539d625c0583542dd795a0fdfacff2050c4475995b40371ce599109ce001b6"
M1306_SOURCE = ROOT / "hw_autoresearch_nts07/scripts/run_m1306_m1301_motion_final_checkpoint_binder_inherited_authority_successor.py"
M1306_TEST = ROOT / "hw_autoresearch_nts07/tests/test_run_m1306_m1301_motion_final_checkpoint_binder_inherited_authority_successor.py"
M1306_CONTRACT = ROOT / "hw_autoresearch_nts07/contracts/m1306_m1301_motion_final_checkpoint_binder_inherited_authority_successor_source_contract_r1_20260830.json"
M1306_SHA = "99b2f4f895b28bdc15ca3a3fa75e3364658751132cde4cc47cf01c778fc16548"
M1306_TEST_SHA = "6154b311472a1f99a2a90bfea5461c78b06b43099251105f131400121de8ff5f"
M1306_CONTRACT_SHA = "478c005b7cd5db47971e9a0fa621f4901f0e087a170a2ccc6fd4a33ae404d4bf"
M1307_ROOT = ROOT / "hw_autoresearch_nts07/reviews/m1307_m1306_inherited_authority_successor_receipt_blind_hammer_r1_20260830"
M1307_MANIFEST_SHA = "b58d346102d5aaaa1a21573276c70c7afc45bbd408cca717c807d8d592d51dcd"
M1307_OUTER_SHA = "ef91ccfcf77e393df68bb37178a0ff41f61fd65b113a93cd2718a323ec0dbbad"
M1307_CONTRACT = ROOT / "hw_autoresearch_nts07/contracts/m1307_m1306_inherited_authority_successor_receipt_blind_hammer_contract_r1_20260830.json"
M1307_CONTRACT_SHA = "737f4a090bf8a5d482ef5e5079868ed09c0e582c968b049ecee8bcd8f8f5906f"
DOCS359 = ROOT / "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

OUT_REL = "hw_autoresearch_nts07/results/m1257_motion_cross_run_final_checkpoint_selection_r5_20260830"
STAGED_OUT_REL = str(OUT.relative_to(ROOT))
EXPECTED_RESULT_MEMBERS = {
    "RUN_COMPLETE.txt", "e0_e8_activation_rebind_targets.json",
    "final_checkpoint_selection.json", "four_checkpoint_metrics.csv",
    "selected_checkpoint_and_config.json", "SHA256SUMS", "SHA256SUMS.seal.sha256",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module)
    return module


S = load("m1309_frozen_m1257", M1257_SOURCE)
R = load("m1309_frozen_m1306", M1306_SOURCE)


def strict_json(path: Path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value: raise AssertionError("duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          AssertionError("nonfinite JSON: " + value)))


RESULT = strict_json(OUT / "final_checkpoint_selection.json")


def snapshot(identity):
    return SimpleNamespace(
        absolute_path=identity["absolute_path"], sha256=identity["sha256"],
        size_bytes=identity["size_bytes"], mtime_ns=identity["mtime_ns"],
        device=identity["device"], inode=identity["inode"], mode=identity["mode"])


def staged_prepared():
    values = {"manifest": snapshot(RESULT["new_run_manifest"])}
    candidates = S.PRODUCTION_POLICY.base.candidates
    for row, candidate in zip(RESULT["candidate_population"], candidates):
        values[candidate.candidate_id + ":checkpoint"] = snapshot(row["checkpoint"])
        values[candidate.candidate_id + ":profile"] = snapshot(row["profile"])
        values["config:" + candidate.config_key] = snapshot(row["configuration"])
    assert len(values) == 11
    return SimpleNamespace(policy=S.PRODUCTION_POLICY.base, snapshots=values)


class Hammer(unittest.TestCase):
    def test_01_archive_sha_sidecar_and_exact_safe_population(self):
        self.assertEqual(sha(ARCHIVE), ARCHIVE_SHA)
        self.assertEqual(ARCHIVE_SIDECAR.read_text().split(),
                         [ARCHIVE_SHA, ARCHIVE.name])
        prefix = OUT_REL + "/"
        expected_files = {prefix + name for name in EXPECTED_RESULT_MEMBERS} | {
            "hw_autoresearch_nts07/results/" + ATTEMPT.name,
            "hw_autoresearch_nts07/results/" + LOG.name,
        }
        with tarfile.open(ARCHIVE, "r") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            self.assertEqual(len(names), len(set(names)))
            self.assertEqual({member.name for member in members if member.isfile()},
                             expected_files)
            self.assertEqual({member.name for member in members if member.isdir()},
                             {OUT_REL})
            for member in members:
                self.assertFalse(member.issym() or member.islnk())
                path = Path(member.name)
                self.assertFalse(path.is_absolute())
                self.assertNotIn("..", path.parts)

    def test_02_extracted_bytes_equal_archive_and_modes_are_closed(self):
        with tarfile.open(ARCHIVE, "r") as archive:
            for member in archive.getmembers():
                if not member.isfile(): continue
                payload = archive.extractfile(member).read()
                local = STAGE / member.name
                self.assertEqual(hashlib.sha256(payload).hexdigest(), sha(local))
        self.assertEqual(stat.S_IMODE(ATTEMPT.stat().st_mode), 0o400)
        self.assertEqual(stat.S_IMODE(LOG.stat().st_mode), 0o400)

    def test_03_exact_result_set_manifest_outer_and_frozen_verifier(self):
        self.assertEqual({path.name for path in OUT.iterdir()}, EXPECTED_RESULT_MEMBERS)
        manifest_sha = sha(OUT / "SHA256SUMS")
        outer_sha = sha(OUT / "SHA256SUMS.seal.sha256")
        self.assertEqual(manifest_sha,
                         "ae4a61f5e79b0d6e308174c00567fff6e25a07a6f065cd7ee3acec2faabcf458")
        self.assertEqual(outer_sha,
                         "d0afaea457958752b9d76c21746c0796145a91466cf93ecd20a56d27bd5ef7e4")
        verified = S.verify_receipt(OUT, staged_prepared())
        self.assertEqual(verified, RESULT)

    def test_04_four_candidates_strict_profiles_and_identity_shapes(self):
        rows = RESULT["candidate_population"]
        self.assertEqual([(row["candidate_id"], row["epoch"]) for row in rows],
                         [("legacy_ep29", 29), ("resume_ep30", 30),
                          ("resume_ep32", 32), ("resume_ep34", 34)])
        for row in rows:
            profile = row["profile"]
            self.assertIs(type(profile["samples"]), int); self.assertEqual(profile["samples"], 825)
            self.assertIs(profile["load_audit_exact_zero"], True)
            self.assertIs(profile["artifact_identity_exact"], True)
            self.assertEqual(profile["module_counts"], {
                "ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12})
            for name in ("checkpoint", "configuration", "profile"):
                identity = row[name]
                self.assertEqual(len(identity["sha256"]), 64)
                self.assertGreater(identity["size_bytes"], 0)
                self.assertGreater(identity["mtime_ns"], 0)

    def test_05_AEE_minimum_tie_break_and_csv_are_recomputed(self):
        rows = RESULT["candidate_population"]
        ranked = sorted(rows, key=lambda row: (Decimal(row["accuracy_metrics"]["AEE"]),
                                               row["epoch"]))
        self.assertEqual([(row["candidate_id"], row["accuracy_metrics"]["AEE"])
                          for row in ranked], [
            ("resume_ep34", "1.1995140134204518"),
            ("resume_ep30", "1.2072849134242896"),
            ("legacy_ep29", "1.209876834190253"),
            ("resume_ep32", "1.2172589833086187")])
        self.assertEqual(RESULT["selected"]["candidate_id"], ranked[0]["candidate_id"])
        self.assertEqual(RESULT["selected"]["epoch"], ranked[0]["epoch"])
        csv_rows = list(csv.DictReader(io.StringIO(
            (OUT / "four_checkpoint_metrics.csv").read_text(encoding="utf-8"))))
        self.assertEqual(len(csv_rows), 4)
        for row, csv_row in zip(rows, csv_rows):
            self.assertEqual(csv_row["candidate_id"], row["candidate_id"])
            self.assertEqual(csv_row["epoch"], str(row["epoch"]))
            self.assertEqual(csv_row["AEE"], row["accuracy_metrics"]["AEE"])
            self.assertEqual(csv_row["checkpoint_sha256"], row["checkpoint"]["sha256"])
            self.assertEqual(csv_row["config_sha256"], row["configuration"]["sha256"])
            self.assertEqual(csv_row["profile_sha256"], row["profile"]["sha256"])

    def test_06_selected_projection_sidecar_and_exact_tuple(self):
        selected = RESULT["selected"]
        winner = next(row for row in RESULT["candidate_population"]
                      if row["candidate_id"] == "resume_ep34")
        self.assertEqual(selected, winner)
        sidecar = strict_json(OUT / "selected_checkpoint_and_config.json")
        self.assertEqual(sidecar["candidate_id"], "resume_ep34")
        self.assertEqual(sidecar["epoch"], 34)
        for key in ("checkpoint", "configuration", "profile"):
            self.assertEqual(sidecar[key], selected[key])
        self.assertEqual(selected["checkpoint"]["sha256"],
                         "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48")
        self.assertEqual(selected["configuration"]["sha256"],
                         "630e735c8fe1d643b524ecd82ecf69d514df548d36380144cef442541daa4d39")
        self.assertEqual(selected["profile"]["sha256"],
                         "144ba2d94eeafd2b6549a7b0aa7d0c89d2b334fe814a7d45f71d6990670e379c")

    def test_07_activity_derived_values_and_selected_projection_recompute(self):
        for row in RESULT["candidate_population"]:
            activity = row["activity"]
            dense = activity["dense_flops"]; effective = activity["effective_flops"]
            recomputed = 1.0 - effective / dense
            self.assertTrue(math.isclose(activity["effective_sparsity"], recomputed,
                                         rel_tol=0.0, abs_tol=1e-15))
            self.assertTrue(math.isclose(activity["global_firing_rate"],
                                         effective / dense, rel_tol=0.0, abs_tol=1e-15))
            self.assertGreater(activity["total_spikes"], 0)
            self.assertGreater(activity["spike_energy_proxy_uj"], 0)
            self.assertEqual(activity["energy_scope"],
                             "spike_activity_proxy_not_hardware_energy")
        self.assertEqual(RESULT["selected"]["activity"],
                         RESULT["candidate_population"][3]["activity"])

    def test_08_E0_E8_and_claim_boundary_are_exact_and_nonpromoting(self):
        expected = S.exact_rebind_targets()
        self.assertEqual(RESULT["e0_e8_activation_dependent_invalidation_and_rebind_targets"],
                         expected)
        self.assertEqual(strict_json(OUT / "e0_e8_activation_rebind_targets.json"),
                         expected)
        self.assertEqual(RESULT["claim_boundary"], dict(S.B.EXACT_CLAIM_BOUNDARY))
        claims = RESULT["claim_boundary"]
        self.assertIs(claims["selection_bound_after_execution"], True)
        self.assertIs(claims["fresh_result_hammer_required"], True)
        for key in ("hardware_rebind_authorized", "hardware_replay_complete",
                    "hardware_speedup", "system_speedup", "power_or_energy",
                    "checkpoint_copied", "gpu_started_by_binder",
                    "remote_access_by_binder", "eda_started_by_binder"):
            self.assertIs(claims[key], False)

    def test_09_attempt_is_unique_and_binds_inputs_and_interpreter(self):
        self.assertEqual([path.name for path in RESULTS.iterdir()].count(ATTEMPT.name), 1)
        body = dict(line.split("=", 1) for line in ATTEMPT.read_text().splitlines()[1:])
        self.assertEqual(body["automatic_retry"], "false")
        prepared = staged_prepared()
        population = {key: S._snapshot_identity(prepared.snapshots[key])
                      for key in sorted(prepared.snapshots)}
        input_digest = S.B.sha256_bytes(json.dumps(
            population, sort_keys=True, separators=(",", ":")).encode())
        self.assertEqual(body["input_snapshot_sha256"], input_digest)
        entity_digest = S.B.sha256_bytes(json.dumps(
            R.M.M.TARGET_ENTITY, sort_keys=True, separators=(",", ":")).encode())
        self.assertEqual(body["interpreter_entity_sha256"], entity_digest)
        self.assertEqual(len(body["command_sha256"]), 64)

    def test_10_log_returncode_and_exact_child_stdout_hash(self):
        self.assertEqual([path.name for path in RESULTS.iterdir()].count(LOG.name), 1)
        body = dict(line.split("=", 1) for line in LOG.read_text().splitlines()[1:])
        self.assertEqual(body["returncode"], "0")
        child_stdout = (
            S.CHILD_TOKEN + "\nselected_candidate=resume_ep34\nselected_epoch=34\n")
        self.assertEqual(body["stdout_sha256"],
                         hashlib.sha256(child_stdout.encode()).hexdigest())
        self.assertEqual(body["stderr_sha256"], hashlib.sha256(b"").hexdigest())
        self.assertEqual((OUT / "RUN_COMPLETE.txt").read_text(), S.B.RUN_COMPLETE)

    def test_11_M1306_M1307_authority_chain_and_namespaces_match(self):
        self.assertEqual(sha(M1306_SOURCE), M1306_SHA)
        self.assertEqual(sha(M1306_TEST), M1306_TEST_SHA)
        self.assertEqual(sha(M1306_CONTRACT), M1306_CONTRACT_SHA)
        self.assertEqual(sha(M1307_CONTRACT), M1307_CONTRACT_SHA)
        self.assertEqual(sha(M1307_ROOT / "SHA256SUMS"), M1307_MANIFEST_SHA)
        self.assertEqual(sha(M1307_ROOT / "SHA256SUMS.seal.sha256"), M1307_OUTER_SHA)
        review = strict_json(M1307_ROOT / "review.json")
        self.assertEqual(review["score"]["points"], 100)
        self.assertEqual(review["authority"]["exactly_one_remote_production_execution"],
                         "GO_AFTER_EXACT_TRANSFER_AND_ROOT_LIVE_PREFLIGHT_MATCH")
        self.assertTrue(review["authority"]["automatic_retry"] is False)
        self.assertEqual(R.PRODUCTION_POLICY.base.output_rel, Path(OUT_REL))
        self.assertEqual(R.PRODUCTION_POLICY.base.attempt_rel.name, ATTEMPT.name)
        self.assertEqual(R.PRODUCTION_POLICY.base.log_rel.name, LOG.name)

    def test_12_docs_and_scope_remain_non_hardware(self):
        self.assertEqual(sha(DOCS359), DOCS359_SHA)
        self.assertFalse(RESULT["claim_boundary"]["hardware_rebind_authorized"])
        self.assertFalse(RESULT["claim_boundary"]["hardware_speedup"])
        self.assertFalse(RESULT["claim_boundary"]["power_or_energy"])


if __name__ == "__main__": unittest.main(verbosity=2)
