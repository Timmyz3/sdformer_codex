#!/usr/bin/env python3
"""Source-only admission checks for M2020/M2025; never launch EDA."""
from __future__ import print_function

import hashlib
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RTL = HW / "rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv"
ADAPTER = HW / "rtl_m2020/m2020_m2018_vcs_public_name_adapter.sv"
M803 = HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
SVA = HW / "verif_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend_assertions.sv"
TB = HW / "tb_m1984/tb_m1984_c2_tsbg_b4_parseable_pass.sv"
FILELIST = HW / "dc_handoff/filelists/iscas_m2020_m2018_c2_tsbg_b4_divfree_directed_vcs.f"
RUNNER = HW / "dc_handoff/scripts/run_m2025_m2024_m2020_m2018_c2_tsbg_b4_divfree_directed_vcs_one_shot.sh"
M2019_DIR = HW / "reviews/m2019_m2018_c2_tsbg_b4_divfree_fair_scheduler_source_hammer_r1_20260902"
M2019_REVIEW = M2019_DIR / "review.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
CONTRACT = HW / "contracts/m2020_m2018_c2_tsbg_b4_divfree_directed_vcs_source_contract_r1_20260902.json"
CONTRACT_MANIFEST = Path(str(CONTRACT) + ".sha256")
CONTRACT_SEAL = Path(str(CONTRACT_MANIFEST) + ".seal.sha256")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stripped_header(text):
    begin = text.index("#(", text.index("module "))
    end = text.index(");", begin) + 2
    header = text[begin:end]
    header = re.sub(r"/\*.*?\*/", "", header, flags=re.S)
    header = re.sub(r"//.*", "", header)
    return re.sub(r"\s+", " ", header).strip()


def source_without_comments_or_strings(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    return "\n".join(row for row in text.splitlines()
                     if not row.lstrip().startswith("`timescale"))


def verify_sealed_directory(directory):
    manifest = directory / "SHA256SUMS"
    seal = directory / "SHA256SUMS.seal.sha256"
    assert manifest.is_file() and not manifest.is_symlink()
    assert seal.is_file() and not seal.is_symlink()
    seal_fields = seal.read_text().strip().split()
    assert seal_fields == [sha(manifest), manifest.name]
    for row in manifest.read_text().splitlines():
        expected, relative = row.split(None, 1)
        relative = relative.lstrip(" *")
        target = directory / relative
        assert target.is_file() and not target.is_symlink()
        assert sha(target) == expected


class M2020M2018DivfreeDirectedVcsSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rtl = RTL.read_text()
        cls.adapter = ADAPTER.read_text()
        cls.filelist = FILELIST.read_text().splitlines()
        cls.runner = RUNNER.read_text()
        cls.review = json.loads(M2019_REVIEW.read_text())
        cls.contract = json.loads(CONTRACT.read_text())

    def test_01_frozen_dependency_identities(self):
        expected = {
            RTL: "96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21",
            ADAPTER: "dfd24f7dbb4122140be8bdb945fe5346c60cc2431a1def7e25f1085df319293c",
            M803: "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
            SVA: "e5519a75c14d68dfc273c3a7e9930560fa8a3c7779ab5ed7f22f294a14be58c2",
            TB: "d46a47dada89e16cdc3f2593020a89e3513060a8a1a03ae3a1963d0483b96081",
            FILELIST: "759a18d9c975ed912b8c75eeeb92b527afb46185c8f8e64f50a8e83f76d86996",
            DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
            M2019_REVIEW: "bf1cfc2d1090f5932419e19921a3cd1966adbbec5585ad446e43f9bcb266477d",
        }
        for path, expected_sha in expected.items():
            self.assertTrue(path.is_file() and not path.is_symlink(), path)
            self.assertEqual(sha(path), expected_sha, path)

    def test_02_adapter_parameter_port_semantics_are_exact(self):
        self.assertEqual(stripped_header(self.adapter), stripped_header(self.rtl))
        self.assertIn(
            "module m1880_c2_tsbg_b4_real_channel_signed_frontend #(",
            self.adapter)
        self.assertEqual(
            self.adapter.count(
                "m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend #("), 1)
        for parameter in (
                "SCHEDULE_MODE", "BUNDLE", "SOURCE_GROUPS",
                "SOURCES_PER_GROUP", "OUTPUT_SLICES", "CACHE_ROWS",
                "TAG_BITS", "CHANNEL_BITS", "EPOCH_BITS",
                "GENERATION_BITS", "LANES"):
            self.assertEqual(
                self.adapter.count(".{0}({0})".format(parameter)), 1)
        self.assertEqual(self.adapter.count(") implementation (.*);"), 1)
        self.assertEqual(self.adapter.count(".*"), 1)

    def test_03_adapter_is_only_a_public_name_shell(self):
        body = self.adapter.split(");", 1)[1].split("endmodule", 1)[0]
        self.assertNotRegex(body, r"\balways(?:_comb|_ff|_latch)?\b")
        self.assertNotRegex(body, r"\bassign\b")
        self.assertNotRegex(body, r"\b(initial|final|force|release)\b")
        self.assertEqual(body.count("implementation"), 1)

    def test_04_filelist_is_exact_five_source_rows(self):
        self.assertEqual(self.filelist, [
            "hw_autoresearch_nts07/rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv",
            "hw_autoresearch_nts07/rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv",
            "hw_autoresearch_nts07/rtl_m2020/m2020_m2018_vcs_public_name_adapter.sv",
            "hw_autoresearch_nts07/verif_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend_assertions.sv",
            "hw_autoresearch_nts07/tb_m1984/tb_m1984_c2_tsbg_b4_parseable_pass.sv",
        ])
        self.assertNotIn("m1995", FILELIST.read_text().lower())
        self.assertNotIn("+incdir", FILELIST.read_text())

    def test_05_m2019_is_sealed_pass_and_divrem_stay_absent(self):
        verify_sealed_directory(M2019_DIR)
        self.assertEqual(
            self.review["status"],
            "PASS_M2019_M2018_C2_TSBG_B4_DIVFREE_FAIR_SCHEDULER_SOURCE_HAMMER")
        self.assertGreaterEqual(self.review["score_over_100"], 95)
        self.assertEqual(self.review["severity_counts"],
                         {"p0": 0, "p1": 0, "p2": 0})
        active = source_without_comments_or_strings(self.rtl)
        self.assertNotRegex(active, r"(?<!/)/(?!/)")
        self.assertNotIn("%", active)
        self.assertNotIn("active_q", active)
        self.assertIn("logic row_live_q", active)
        self.assertEqual(self.review["scheduler_audit"]["runtime_division_operators"], 0)
        self.assertEqual(self.review["scheduler_audit"]["runtime_remainder_operators"], 0)
        self.assertTrue(self.review["disclosure"]["common_payload_mux_remains"])

    def test_06_runner_is_exact_one_shot_with_future_review_pin(self):
        required = (
            "M2025_EXPECTED_RUNNER_SHA256",
            "M2025_EXPECTED_M2024_REVIEW_SHA256",
            "verify_dir_seal \"${M2019_DIR}\"",
            "verify_dir_seal \"${M2024_DIR}\"",
            '"${LMUTIL}" lmstat -a -c "${LICENSE_SERVER}"',
            '"${VCS}" -full64 -sverilog -assert svaext',
            "-assert global_finish_maxfail=1",
            "automatic_retry\": false",
            "production_g48_dynamic\": false",
        )
        for token in required:
            self.assertIn(token, self.runner)
        self.assertEqual(self.runner.count('"${LMUTIL}" lmstat -a'), 1)
        self.assertEqual(self.runner.count('"${VCS}" -full64'), 1)
        self.assertEqual(self.runner.count("simv_runs\": 1"), 1)
        self.assertNotRegex(self.runner, r"\b(dc_shell|pt_shell|fm_shell)\s+-")
        self.assertNotIn("for attempt", self.runner.lower())

    def test_07_runner_gates_identity_namespace_collision_and_memory(self):
        for token in (
                "sha_exact 96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21",
                "sha_exact dfd24f7dbb4122140be8bdb945fe5346c60cc2431a1def7e25f1085df319293c",
                "sha_exact 759a18d9c975ed912b8c75eeeb92b527afb46185c8f8e64f50a8e83f76d86996",
                "[[ ! -e \"${RESULT}\" && ! -e \"${ATTEMPT}\"",
                "for proc in /proc/[0-9]*",
                "[[ \"${real_uid}\" == \"${EUID}\" ]]",
                "mem_available",
                "commit_limit",
                "16777216",
                "FAILED_OR_INCOMPLETE_DO_NOT_CITE",
                "failed_or_incomplete.$$.quarantine",
                "seal_dir \"${WORK}\"",
                "mv -T -- \"${WORK}\" \"${RESULT}\""):
            self.assertIn(token, self.runner)

    def test_08_runner_checks_unique_pass_phase_load_and_claim_boundaries(self):
        self.assertIn(
            "PASS_M1880_C2_TSBG_B4_REAL_M803_TYPED_SIGNED_DIRECTED rows=48 issues=576 products=9216 commits=24 bundles_base=576 bundles_tsbg=144 scalar_base=4608 scalar_tsbg=1152 stale=1 retired_replay=1 replay_accept=0 reset=2 recovery=1",
            self.runner)
        for phase in (
                "reset", "full_load", "full_execute", "retired_replay",
                "replay_reset_recovery", "stale_attack",
                "stale_reset_recovery", "recovery_load",
                "recovery_execute", "final_checks"):
            self.assertIn(phase, self.runner)
        for token in (
                "M1970_LOAD_BEGIN", "M1970_LOAD_COMPLETE",
                "M1970_LOAD_TIMEOUT", '"same_area": false',
                '"exact_cycle_speedup": false', '"system_speedup": false',
                '"paper_admitted": false', '"headline": false'):
            self.assertIn(token, self.runner)

    def test_09_double_sealed_contract_and_exact_authored_sources(self):
        self.assertEqual(CONTRACT_MANIFEST.read_text().strip().split(),
                         [sha(CONTRACT), CONTRACT.name])
        self.assertEqual(CONTRACT_SEAL.read_text().strip().split(),
                         [sha(CONTRACT_MANIFEST), CONTRACT_MANIFEST.name])
        sources = self.contract["source_sha256"]
        for path in (RTL, ADAPTER, M803, SVA, TB, FILELIST, RUNNER,
                     Path(__file__).resolve(), M2019_REVIEW, DOCS359):
            relative = str(path.relative_to(ROOT))
            self.assertEqual(sources[relative], sha(path), relative)

    def test_10_contract_is_source_only_and_reserves_m2024_m2025(self):
        self.assertEqual(
            self.contract["status"],
            "SOURCE_ONLY_M2020_M2018_DIVFREE_DIRECTED_VCS_TOOLCHAIN__NO_EDA")
        self.assertEqual(self.contract["author_execution"], {
            "license_queries": 0, "vcs_compiles": 0, "simv_runs": 0,
            "dc_runs": 0, "pt_runs": 0, "fm_runs": 0,
            "gpu_runs": 0, "attempts": 0, "results": 0,
            "reviews_m2024": 0})
        future = self.contract["future_authority"]
        self.assertEqual(future["source_review"], "M2024")
        self.assertEqual(future["one_shot_execution"], "M2025")
        self.assertTrue(future["exact_m2024_review_sha_pin_required"])
        self.assertTrue(future["different_author_result_review_required"])
        for key in ("vcs", "same_area", "exact_cycle_speedup",
                    "production_g48_dynamic", "system_speedup",
                    "paper_admitted", "headline"):
            self.assertFalse(self.contract["claim_boundary"][key])


if __name__ == "__main__":
    unittest.main(verbosity=2)
