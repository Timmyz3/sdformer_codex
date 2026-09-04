#!/usr/bin/env python3
from __future__ import annotations
import copy, importlib.util, os, sys, tempfile, unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "hw_autoresearch_nts07/scripts/run_m1329_m1327_motion_ep34_production_one_shot.py"
spec = importlib.util.spec_from_file_location("test_m1329", SOURCE)
M = importlib.util.module_from_spec(spec); sys.modules[spec.name] = M; spec.loader.exec_module(M)


class Tests(unittest.TestCase):
    def test_01_authorities_runtime_and_release(self):
        M.verify_authorities(); runtime = M.validate_runtime_file(); release = M.validate_release_static()
        self.assertEqual(set(runtime), {"contract_path", "capture", "cohort", "output"})
        self.assertEqual(runtime["capture"], {"attention_windows_per_call": 100})
        self.assertFalse(release["one_shot"]["automatic_retry"])

    def test_02_preflight_never_consumes_or_delegates(self):
        runtime = M.validate_runtime_file(); binding = {"identity": {}}
        with mock.patch.object(M, "validate_release_static"), mock.patch.object(M, "ensure_fresh") as fresh, \
             mock.patch.object(M, "validate_runtime_file", return_value=runtime), \
             mock.patch.object(M.M1327, "validate_identity_and_project", return_value=(runtime,binding)), \
             mock.patch.object(M, "consume_attempt") as consume, \
             mock.patch.object(M.M1327, "delegate_for_future_release") as delegate:
            self.assertEqual(M.read_only_preflight(), runtime)
        self.assertEqual(fresh.call_count, 2); consume.assert_not_called(); delegate.assert_not_called()

    def test_03_under_lease_order_and_single_new_attempt(self):
        runtime = M.validate_runtime_file(); binding = {"identity": {}}
        events=[]
        class Lease:
            def __enter__(self): events.append("lease_enter")
            def __exit__(self,*a): events.append("lease_exit")
        substrate=mock.Mock(); substrate.exclusive_gpu_lease.return_value=Lease()
        with mock.patch.object(M.M1327,"validate_identity_and_project",side_effect=lambda:(events.append("revalidate") or (runtime,binding))), \
             mock.patch.object(M,"validate_runtime_file",return_value=runtime), \
             mock.patch.object(M,"ensure_fresh",side_effect=lambda:events.append("fresh")), \
             mock.patch.object(M,"consume_attempt",side_effect=lambda:events.append("consume")) as consume, \
             mock.patch.object(M.M1327,"delegate_for_future_release",side_effect=lambda *a:(events.append("delegate") or M.CANONICAL_RESULT)), \
             mock.patch.object(M.M1327.M1249.R1,"verify_double_seal",side_effect=lambda p:events.append("seal")):
            M.execute_under_lease(runtime,substrate)
        self.assertEqual(events,["lease_enter","revalidate","fresh","consume","delegate","lease_exit","seal"])
        consume.assert_called_once()

    def test_04_attempt_is_O_EXCL_and_no_retry(self):
        with tempfile.TemporaryDirectory() as raw:
            marker=Path(raw)/"attempt"
            with mock.patch.object(M,"CANONICAL_ATTEMPT",marker): M.consume_attempt()
            self.assertEqual(marker.read_bytes(),M.ATTEMPT_TOKEN)
            with mock.patch.object(M,"CANONICAL_ATTEMPT",marker), self.assertRaises(FileExistsError): M.consume_attempt()

    def test_05_nonroot_rejected(self):
        with mock.patch.object(M.os,"geteuid",return_value=1000), self.assertRaisesRegex(M.M1329Error,"root_agent"):
            M.execute_once(Path("/tmp/no"))

    def test_06_atomic_log_no_replace(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw); canonical=root/"production.log"; temp=root/"production.log.tmp.x"; temp.write_bytes(b"ok")
            with mock.patch.object(M,"CANONICAL_LOG",canonical): M.publish_no_replace(temp)
            self.assertEqual(canonical.read_bytes(),b"ok"); self.assertFalse(temp.exists())
            temp.write_bytes(b"new")
            with mock.patch.object(M,"CANONICAL_LOG",canonical), self.assertRaises(M.M1329Error): M.publish_no_replace(temp)
            self.assertEqual(canonical.read_bytes(),b"ok")

    def test_07_runtime_drift_under_lease_rejected_before_attempt(self):
        runtime=M.validate_runtime_file(); changed=copy.deepcopy(runtime); changed["capture"]["attention_windows_per_call"]=99
        class Lease:
            def __enter__(self): return None
            def __exit__(self,*a): return None
        substrate=mock.Mock(); substrate.exclusive_gpu_lease.return_value=Lease()
        with mock.patch.object(M.M1327,"validate_identity_and_project",return_value=(changed,{})), \
             mock.patch.object(M,"consume_attempt") as consume, self.assertRaises(M.M1329Error):
            M.execute_under_lease(runtime,substrate)
        consume.assert_not_called()

    def test_08_old_attempt_consumer_not_used(self):
        source=SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("M1249.consume_attempt",source)
        self.assertIn("os.O_EXCL",source)

if __name__ == "__main__": unittest.main(verbosity=2)
