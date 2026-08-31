#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import sys,tempfile,unittest

HERE=Path(__file__).resolve().parent;HW=HERE.parent.parent
DRIVER=HERE.parent/"scripts/execute_m981_m977_decoder_d2d3_10k_atomic_evidence_r1.py"
RUNNER=HERE.parent/"scripts/run_m985_m981_decoder_d2d3_10k_atomic_evidence_one_shot.sh"
CONTRACT=HW/"contracts/m981_m977_decoder_d2d3_10k_atomic_evidence_source_contract_r1_20260829.json"
S=importlib.util.spec_from_file_location("m981_test",DRIVER)
M=importlib.util.module_from_spec(S);sys.modules[S.name]=M;S.loader.exec_module(M)

def fake(layer,*unused):
    return {"row_identity":{"layer":layer,"numerical_route":"EXACT_BINARY_SUPPORT"},
      "prefix":10000,"exact_miter":{
      "status":"PASS_M768_M861_M890_M896_EXACT_MITER",
      "expanded_request_count":10000,"compressed_transaction_count":157,
      "commit_requests_in_prefix":9}}

class M981Test(unittest.TestCase):
    def test_chain_names_are_additive_and_collision_free(self):
        paths=M.canonical_paths();joined=" ".join(paths.values())
        for token in ("m981_","m982_","m983_","m984_","m985_"):
            self.assertIn(token,joined)
        for stale in ("m973_","m974_","m975_"):self.assertNotIn(stale,joined)

    def test_atomic_bundle_never_exposes_lone_outer(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/"payload";root.mkdir();M.write_exclusive(root/"x",b"x")
            with self.assertRaisesRegex(RuntimeError,"after manifest"):
                M.atomic_seal(root,"after_manifest")
            self.assertFalse((root/M.SEAL_DIR).exists())
            self.assertTrue(M.partial_seal_stages(root))
            seal=M.atomic_seal(root)
            self.assertEqual(M.verify_atomic_seal(root),seal)
            self.assertTrue((root/M.SEAL_DIR/M.SEAL_MANIFEST).is_file())
            self.assertTrue((root/M.SEAL_DIR/M.SEAL_OUTER).is_file())

    def test_row_exception_is_persisted_and_atomically_sealed(self):
        with tempfile.TemporaryDirectory() as t:
            work=Path(t)/(M.RESULT.name+".work.row");work.mkdir()
            with self.assertRaisesRegex(RuntimeError,"injected"):
                M.run_row("D2",work/"D2",lambda *a:(_ for _ in ()).throw(
                    RuntimeError("injected")))
            self.assertTrue((work/"D2/traceback.log").is_file())
            M.verify_atomic_seal(work/"D2")

    def test_lone_legacy_manifest_cannot_skip_quarantine(self):
        with tempfile.TemporaryDirectory() as t:
            parent=Path(t);work=parent/(M.RESULT.name+".work.legacy");work.mkdir()
            M.write_exclusive(work/"SHA256SUMS",b"partial\n")
            target=parent/(M.FAILURE_PREFIX+"legacy")
            M.quarantine_work(work,target,130,allowed_parent=parent)
            self.assertFalse(work.exists());M.verify_atomic_seal(target)

    def test_cleanup_failure_retains_work_unmoved(self):
        with tempfile.TemporaryDirectory() as t:
            parent=Path(t);work=parent/(M.RESULT.name+".work.keep");work.mkdir()
            M.write_exclusive(work/"payload",b"keep")
            target=parent/(M.FAILURE_PREFIX+"keep")
            with self.assertRaisesRegex(RuntimeError,"work retained"):
                M.quarantine_work(work,target,137,"before_root_seal",
                                  allowed_parent=parent)
            self.assertTrue(work.exists());self.assertFalse(target.exists())

    def test_multi_transaction_geometry_and_source_contract(self):
        self.assertEqual(M.source_geometry("D2"),
                         {"source_bytes":231600,"source_fetch_requests":1207})
        self.assertEqual(M.source_geometry("D3"),
                         {"source_bytes":465600,"source_fetch_requests":2425})
        value=M.summarize_row(fake("D2"))
        self.assertEqual(value["observed_compressed_transaction_count"],157)
        self.assertEqual(value["observed_commit_requests_in_prefix"],9)
        self.assertEqual(M.validate_source_contract(CONTRACT,RUNNER)["status"],
                         "PASS_M981_SOURCE__NO_10K_EXECUTED")

    def test_source_selftest_no_real_prefix(self):
        value=M.source_self_test();self.assertFalse(value["real_10k_executed"])
        self.assertFalse(value["eda_gpu_remote_used"])

if __name__=="__main__":unittest.main()
