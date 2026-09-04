#!/usr/bin/env python3
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
CHECKER = HERE.parent / "scripts/check_m1334_c2_headline_mapped_production_activity_source.py"
SPEC = importlib.util.spec_from_file_location("m1334_checker", CHECKER)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fake_saif(top, duration, endpoint_tc=4, tx=0, reset_tc=0,
              activity_inside_dut=True):
    payload = '''(PORT
      (clk_core (T0 1) (T1 1) (TX {tx}) (TC 20))
      (rst_core (T0 1) (T1 0) (TX 0) (TC {reset_tc}))
      (raw_valid (T0 1) (T1 1) (TX 0) (TC 2))
      (raw_accept (T0 1) (T1 1) (TX 0) (TC 2))
      (mem_req_accept[0] (T0 1) (T1 1) (TX 0) (TC {endpoint_tc}))
      (mem_rsp_accept[0] (T0 1) (T1 1) (TX 0) (TC {endpoint_tc}))
      (result_accumulator[0] (T0 1) (T1 1) (TX 0) (TC 6))
      (result_accept (T0 1) (T1 1) (TX 0) (TC 4))
      (token_done_accept (T0 1) (T1 1) (TX 0) (TC 2)))'''.format(
          tx=tx, reset_tc=reset_tc, endpoint_tc=endpoint_tc)
    if activity_inside_dut:
        body = "(INSTANCE core (INSTANCE dut " + payload + "))"
    else:
        body = "(INSTANCE core (INSTANCE dut)) (INSTANCE checks " + payload + ")"
    return "(SAIFILE (DURATION {0}) (INSTANCE {1} {2}))".format(
        duration, top, body)


def pass_log(case_id, endpoint=4):
    return ("PASS M1334 coverage case={0} source=3 endpoint={1} "
            "commit=2 stall=1 done=1 unknown=0 fatal=0\n").format(
                case_id, endpoint)


class M1334SourceTest(unittest.TestCase):
    def write(self, directory, name, text):
        path = Path(directory) / name
        path.write_text(text)
        return path

    def test_static_source(self):
        out = M.validate_static()
        self.assertEqual(out["status"], "PASS_M1334_SOURCE_ONLY__NO_EDA")
        self.assertEqual(out["closed_predecessor_false_negatives"], 10)

    def test_exact_filelist_path_and_provider_allowlist(self):
        text = M.FILELISTS["k8"].read_text()
        self.assertEqual(M.validate_filelist(text, "k8"), M.exact_paths("k8"))
        leaf = "m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.v"
        official = next(line for line in text.splitlines() if line.endswith(leaf))
        forged = text.replace(official, "/tmp/forged/k8/netlist/" + leaf)
        with self.assertRaisesRegex(RuntimeError, "exact ordered allowlist"):
            M.validate_filelist(forged, "k8")
        with self.assertRaisesRegex(RuntimeError, "exact ordered allowlist"):
            M.validate_filelist(text + "/tmp/legacy/m349_fc2_scalar_bank_memory_model.sv\n", "k8")

    def test_comment_stripping_and_memory_reset_structure(self):
        text = M.MEM.read_text()
        mutant = text.replace("epoch_q[slot] <= '0;",
                              "// epoch_q[slot] <= '0;")
        self.assertNotEqual(text, mutant)
        with self.assertRaisesRegex(RuntimeError, "reset omits payload"):
            M.validate_memory_source(mutant)

    def test_comment_stripping_and_cover_structure(self):
        text = M.SVA.read_text()
        mutant = text.replace("cp_source: cover property (raw_accept);",
                              "// cp_source: cover property (raw_accept);")
        self.assertNotEqual(text, mutant)
        with self.assertRaisesRegex(RuntimeError, "runtime cover absent"):
            M.validate_assertion_source(mutant)

    def test_ucli_comments_cannot_substitute_active_scope(self):
        text = M.UCLI.read_text()
        scope = "tb_m1334_c2_headline_mapped_production_activity.core.dut"
        mutant = text.replace("power " + scope,
            "# power " + scope + "\npower tb_m1334_c2_headline_mapped_production_activity.core")
        mutant = mutant.replace(
            "power -report $::env(M1334_SAIF_FILE) 1e-9 " + scope,
            "# power -report $::env(M1334_SAIF_FILE) 1e-9 " + scope
            + "\npower -report $::env(M1334_SAIF_FILE) 1e-9 tb_m1334_c2_headline_mapped_production_activity.core")
        with self.assertRaisesRegex(RuntimeError, "exact DUT-only"):
            M.validate_ucli(mutant)

    def test_invalid_or_x_valid_accept_cannot_reach_state_write(self):
        text = M.MEM.read_text()
        mutant = text.replace("&& mem_req_valid === 1'b1 && mem_req_ready === 1'b1",
                              "&& mem_req_ready === 1'b1")
        self.assertNotEqual(text, mutant)
        with self.assertRaisesRegex(RuntimeError, "clean request gate omits"):
            M.validate_memory_source(mutant)
        mutant = text.replace("if (request_fire_clean) begin",
                              "if (mem_req_accept === 1'b1) begin")
        with self.assertRaisesRegex(RuntimeError, "structure marker"):
            M.validate_memory_source(mutant)

    def test_payload_and_stability_are_explicitly_fatal(self):
        text = M.SVA.read_text()
        mutant = text.replace(
            'else $fatal(1, "M1334 result payload unknown");', "else ;", 1)
        with self.assertRaisesRegex(RuntimeError, "fail-closed SVA absent"):
            M.validate_assertion_source(mutant)
        mutant = text.replace(
            '$fatal(1, "M1334 result stability violation");',
            '$display("M1334 result stability violation");', 1)
        with self.assertRaisesRegex(RuntimeError, "fatal path absent"):
            M.validate_assertion_source(mutant)

    def test_saif_exact_dut_scope_and_case4_zero(self):
        with tempfile.TemporaryDirectory(prefix="m1334_saif_") as td:
            top = "tb_m1334_c2_headline_mapped_production_activity"
            good = self.write(td, "good.saif", fake_saif(top, 153, 4))
            out = M.validate_saif(good, "k8", 0, 51)
            self.assertEqual(out["major_cone_tc"]["endpoint"], 8.0)
            sibling = self.write(td, "sibling.saif",
                                 fake_saif(top, 153, 4,
                                           activity_inside_dut=False))
            with self.assertRaisesRegex(RuntimeError, "no activity"):
                M.validate_saif(sibling, "k8", 0, 51)
            bad_zero = self.write(td, "bad_zero.saif", fake_saif(top, 42, 9))
            with self.assertRaisesRegex(RuntimeError, "must equal zero"):
                M.validate_saif(bad_zero, "k1x8", 4, 14)
            good_zero = self.write(td, "good_zero.saif", fake_saif(top, 42, 0))
            out = M.validate_saif(good_zero, "k1x8", 4, 14)
            self.assertEqual(out["major_cone_tc"]["endpoint"], 0.0)

    def make_inventory(self, td):
        root = Path(td)
        top = "tb_m1334_c2_headline_mapped_production_activity"
        entries = []
        for axis in ("k8", "k1x8"):
            for case_id, cycles in enumerate(M.AXES[axis]["cycles"]):
                endpoint = 0 if case_id == 4 else 4
                sp = self.write(root, "{0}_case{1}.saif".format(axis, case_id),
                    fake_saif(top, cycles * 3, endpoint))
                lp = self.write(root, "{0}_case{1}.log".format(axis, case_id),
                    pass_log(case_id, endpoint))
                entries.append({"axis": axis, "case": case_id,
                    "cycles": cycles, "saif": sp.name,
                    "saif_sha256": digest(sp), "runtime_log": lp.name,
                    "runtime_log_sha256": digest(lp)})
        manifest = self.write(root, "inventory.json", json.dumps({
            "schema": "m1334_c2_production_activity_inventory_r1",
            "status": "CANDIDATE_UNSEALED_DO_NOT_CITE",
            "entries": entries}, sort_keys=True))
        return manifest

    def test_exact_ten_file_cartesian_inventory(self):
        with tempfile.TemporaryDirectory(prefix="m1334_inventory_") as td:
            manifest = self.make_inventory(td)
            out = M.validate_inventory(manifest)
            self.assertEqual(out["entry_count"], 10)
            data = json.loads(manifest.read_text())
            data["entries"] = data["entries"][:-1]
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(RuntimeError, "exactly ten"):
                M.validate_inventory(manifest)

    def test_inventory_reuse_and_extra_saif_rejected(self):
        with tempfile.TemporaryDirectory(prefix="m1334_inventory_") as td:
            manifest = self.make_inventory(td)
            data = json.loads(manifest.read_text())
            data["entries"][1]["saif"] = data["entries"][0]["saif"]
            data["entries"][1]["saif_sha256"] = data["entries"][0]["saif_sha256"]
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(RuntimeError, "reused"):
                M.validate_inventory(manifest)
        with tempfile.TemporaryDirectory(prefix="m1334_inventory_") as td:
            manifest = self.make_inventory(td)
            self.write(td, "extra.saif", "(SAIFILE)")
            with self.assertRaisesRegex(RuntimeError, "missing/extra"):
                M.validate_inventory(manifest)

    def test_contract_source_set_is_exact(self):
        data = json.loads(M.CONTRACT.read_text())
        data["source_files"] = data["source_files"][:-1]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(data, fh); path = Path(fh.name)
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(RuntimeError, "exact key set"):
            M.validate_static(path)

    def test_runtime_log_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix="m1334_log_") as td:
            path = self.write(td, "good.log", pass_log(4, 0))
            M.validate_runtime_log(path, "k8", 4)
            path.write_text(pass_log(4, 0) + "Fatal: payload unknown\n")
            with self.assertRaisesRegex(RuntimeError, "fatal/assertion"):
                M.validate_runtime_log(path, "k8", 4)


if __name__ == "__main__":
    unittest.main()
