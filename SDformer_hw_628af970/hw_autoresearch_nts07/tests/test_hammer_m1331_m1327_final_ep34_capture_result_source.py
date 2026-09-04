#!/usr/bin/env python3
from __future__ import annotations
import copy, hashlib, importlib.util, json, shutil, sys, tempfile, unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"hw_autoresearch_nts07/scripts/hammer_m1331_m1327_final_ep34_capture_result_source.py"
spec=importlib.util.spec_from_file_location("test_m1331",SOURCE); M=importlib.util.module_from_spec(spec); sys.modules[spec.name]=M; spec.loader.exec_module(M)

def write_json(path,value): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,sort_keys=True)+"\n")
def seal(root):
    for p in (root/"SHA256SUMS",root/"SHA256SUMS.seal.sha256"):
        if p.exists(): p.unlink()
    members=sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    manifest=root/"SHA256SUMS"; manifest.write_text("".join(f"{M.sha(root/n)}  {n}\n" for n in members))
    (root/"SHA256SUMS.seal.sha256").write_text(f"{M.sha(manifest)}  SHA256SUMS\n")

class Fixture:
    def __init__(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)/"result"; self.root.mkdir()
        cohort=copy.deepcopy(M.expected_cohort())
        selected={"candidate_id":"resume_ep34","epoch":34,
                  "checkpoint":{"sha256":M.CHECKPOINT_SHA256},
                  "configuration":{"sha256":M.CONFIG_SHA256},
                  "profile":{"sha256":M.PROFILE_SHA256,"samples":825,"module_counts":{"ATLIFTernaryPSN":105,"ShiftmaxAttention":12}}}
        final={"epoch":34,"checkpoint_sha256":M.CHECKPOINT_SHA256,"config_sha256":M.CONFIG_SHA256,
               "profile_sha256":M.PROFILE_SHA256,"selection_sha256":M.SELECTION_SHA256}
        self.manifest={"schema":"m1227_motion_final_checkpoint_unified_hardware_capture_r1_v1",
          "status":"CAPTURE_COMPLETE__FRESH_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM",
          "identity":{"contract_sha256":M.RUNTIME_SHA256,"selection":{"selected":selected},
                      "checkpoint_load_audit":{"missing_count":0,"unexpected_count":0},
                      "module_counts":{"ATLIFTernaryPSN":105,"ShiftmaxAttention":12}},
          "cohort":{"samples":cohort,"population":40,"c1_samples":10,
                    "decoder_sequences":["interlaken_01_a","thun_01_b","zurich_city_12_a"],"decoder_samples_per_sequence":10},
          "m1227_runtime_contract":{"static_modules":259,"static_atlif":105,"live_modules_per_sample":247,
                    "live_atlif":93,"dead_sn_v":list(M.M1227.DEAD_SN_V),"dead_calls_per_sample":0,
                    "ordered_records":9880,"attention_records":480,"payload_files":640,"final_selection_identity":final},
          "claim_boundary":{"capture_only":True,"accuracy":False,"cycles":False,"speedup":False,
                    "system_speedup":False,"energy":False,"rtl":False,"ppa":False,"fresh_result_hammer_required":True}}
        self.admission={"schema":"m1227_final_capture_admission_r1_v1","status":"PASS","ordered":9880,
          "attention":480,"payload_files":640,"execution":7360,"operator_rows":79,"atlif_live_rows":93,
          "atlif_static":105,"dead_sn_v":list(M.M1227.DEAD_SN_V),
          "claim_boundary":{"capture_only":True,"paper_result":False,"cycles":False,"speedup":False,"energy":False,"ppa":False}}
        self.write_all()
    def write_all(self):
        write_json(self.root/"manifest.json",self.manifest); write_json(self.root/"m1227_admission.json",self.admission)
        with (self.root/"unified_ordered_records.jsonl").open("w") as f:
            for s in range(40):
                for n in range(247): f.write(json.dumps({"global_sample_id":s,"category":"fixture","name":f"module.{n}"},sort_keys=True)+"\n")
        write_json(self.root/"attention_qk/manifest.json",{"records":[{"i":i} for i in range(480)]})
        payload=self.root/"payloads"; payload.mkdir(exist_ok=True)
        hashes={hashlib.sha256(n.encode()).hexdigest()[:12] for n in M.M1227.C1_TARGETS+M.M1227.DECODER_TARGETS}
        for s in range(40):
            for h in hashes:
                for suffix in ("fp32.zlib","support_sign.le.bitpack"):
                    (payload/f"s{s:02d}_o00000_{h}.{suffix}").write_bytes(b"x")
        write_json(self.root/"execution_trace.json",[{"sample_id":i//184} for i in range(7360)])
        write_json(self.root/"operator_runtime.json",[{"name":f"op.{i}","calls":40} for i in range(79)])
        write_json(self.root/"atlif_activity.json",[{"name":f"live.{i}","calls":40} for i in range(93)])
        (self.root/"RUN_COMPLETE.txt").write_text("PASS_M1174_UNIFIED_CAPTURE__FRESH_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM\n")
        seal(self.root)
    def close(self): self.tmp.cleanup()

class Tests(unittest.TestCase):
    def setUp(self): self.fx=Fixture()
    def tearDown(self): self.fx.close()
    def reject(self):
        seal(self.fx.root)
        with self.assertRaises(M.M1331Error): M.validate_result(self.fx.root)
    def test_01_positive_fixture(self):
        out=M.validate_result(self.fx.root); self.assertEqual(out["population"]["ordered"],9880)
    def test_02_source_policy_has_no_actual_seal(self):
        policy=M.validate_source_policy(); self.assertFalse(policy["actual_result_seal_prefilled"])
        self.assertNotIn("actual_result_seal",policy)
    def test_03_recursive_seal_tamper_rejected(self):
        (self.fx.root/"RUN_COMPLETE.txt").write_text("bad\n")
        with self.assertRaises(M.M1331Error): M.validate_result(self.fx.root)
    def test_04_ep34_identity_drift_rejected(self):
        self.fx.manifest["identity"]["selection"]["selected"]["epoch"]=35; write_json(self.fx.root/"manifest.json",self.fx.manifest); self.reject()
    def test_05_ordered_population_rejected(self):
        p=self.fx.root/"unified_ordered_records.jsonl"; lines=p.read_text().splitlines(); p.write_text("\n".join(lines[:-1])+"\n"); self.reject()
    def test_06_attention_population_rejected(self):
        write_json(self.fx.root/"attention_qk/manifest.json",{"records":[]}); self.reject()
    def test_07_payload_population_rejected(self):
        next((self.fx.root/"payloads").iterdir()).unlink(); self.reject()
    def test_08_execution_operator_atlif_rejected(self):
        write_json(self.fx.root/"operator_runtime.json",[]); self.reject()
    def test_09_cohort_sha_order_rejected(self):
        self.fx.manifest["cohort"]["samples"][0]["sha256"]="0"*64; write_json(self.fx.root/"manifest.json",self.fx.manifest); self.reject()
    def test_10_claim_boundary_rejected(self):
        self.fx.manifest["claim_boundary"]["speedup"]=True; write_json(self.fx.root/"manifest.json",self.fx.manifest); self.reject()

if __name__=="__main__": unittest.main(verbosity=2)
