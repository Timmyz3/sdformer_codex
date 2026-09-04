#!/usr/bin/env python3
"""Post-result, read-only independent hammer source for final M1327 capture.

The canonical result must already exist and carry its own recursive double
seal.  No result seal is predicted or embedded here.  Author self-checks use
only disposable fixtures and never inspect or create the canonical result.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, re, stat, sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "hw_autoresearch_nts07"
M1327_SOURCE = ROOT / "neuron_experiments/H9_bipolar_self_attention/entrypoints/capture_m1327_motion_ep34_consumed_namespace_bridge_r1.py"
M1327_SOURCE_SHA256 = "2ab5024a11a81f7bb3ed75956114cc95e07dbe0782328414f2bd3c79342c3ac9"
M1313 = HW / "contracts/m1313_motion_ep34_final_unified_capture_production_launch_r1_20260831.json"
M1313_SHA256 = "eeb0a8380e51610652ec6cdf1c2bb58c22395c9d72608e98f6a88a18f5c6bbda"
RUNTIME_CONTRACT = HW / "contracts/m1327_motion_ep34_consumed_namespace_bridge_production_launch_r1_20260831.json"
RUNTIME_SHA256 = "10c1f9ef06976846ee39f88efbb5c5df1e8bcd6f1d9db4542ecc09b43aae72d7"
CANONICAL_RESULT = HW / "results/m1327_motion_ep34_unified_hardware_capture_s40_r1_20260831"
SOURCE_CONTRACT = HW / "contracts/m1331_m1327_final_ep34_capture_result_hammer_source_contract_r1_20260831.json"
TEST = HW / "tests/test_hammer_m1331_m1327_final_ep34_capture_result_source.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
CHECKPOINT_SHA256 = "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"
CONFIG_SHA256 = "630e735c8fe1d643b524ecd82ecf69d514df548d36380144cef442541daa4d39"
PROFILE_SHA256 = "144ba2d94eeafd2b6549a7b0aa7d0c89d2b334fe814a7d45f71d6990670e379c"
SELECTION_SHA256 = "4af7b7e1b4a174440331268fcfffda44896d86d02c7d20195e7a49d73eae6cd0"
SOURCE_SCHEMA = "m1331_m1327_final_ep34_capture_result_hammer_source_r1_v1"
SOURCE_STATUS = "SOURCE_ONLY__CANONICAL_RESULT_MUST_PREEXIST__NO_REMOTE_NO_GPU_NO_CAPTURE"
PASS_TOKEN = "PASS_M1331_SOURCE_SELF_CHECK__FIXTURES_ONLY_NO_CANONICAL_RESULT"

class M1331Error(RuntimeError): pass
def require(ok,msg):
    if not ok: raise M1331Error(msg)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def regular(path,label):
    try: mode=Path(path).lstat().st_mode
    except FileNotFoundError as e: raise M1331Error("missing "+label) from e
    require(stat.S_ISREG(mode) and not Path(path).is_symlink(),label+" must be regular non-symlink")
def strict_text(raw):
    def pairs(items):
        out={}
        for k,v in items: require(k not in out,"duplicate JSON key"); out[k]=v
        return out
    try: value=json.loads(raw,object_pairs_hook=pairs,
        parse_constant=lambda x:(_ for _ in ()).throw(M1331Error("nonfinite JSON")))
    except (ValueError,TypeError) as e: raise M1331Error("invalid JSON") from e
    return value
def strict(path): regular(path,str(path)); return strict_text(Path(path).read_text(encoding="utf-8"))

def _load_m1327():
    regular(M1327_SOURCE,"M1327 source"); require(sha(M1327_SOURCE)==M1327_SOURCE_SHA256,"M1327 SHA")
    spec=importlib.util.spec_from_file_location("m1331_sealed_m1327",M1327_SOURCE)
    require(spec and spec.loader,"cannot load M1327"); module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module; spec.loader.exec_module(module); return module
M1327=_load_m1327(); M1227=M1327.M1325.M1227

def verify_recursive_seal(root:Path):
    require(root.is_dir() and not root.is_symlink(),"result root")
    manifest=root/"SHA256SUMS"; outer=root/"SHA256SUMS.seal.sha256"
    regular(manifest,"result manifest seal"); regular(outer,"result outer seal")
    rows={}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts=line.split("  ",1); require(len(parts)==2,"seal row")
        digest,name=parts; require(re.fullmatch(r"[0-9a-f]{64}",digest) and name not in rows,"seal syntax")
        rel=Path(name); require(not rel.is_absolute() and ".." not in rel.parts and name not in {manifest.name,outer.name},"unsafe seal path")
        member=root/rel; regular(member,"sealed member"); require(sha(member)==digest,"sealed member SHA"); rows[name]=digest
    actual={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name not in {manifest.name,outer.name}}
    require(actual==set(rows),"recursive sealed population")
    require(outer.read_text(encoding="ascii")==sha(manifest)+"  SHA256SUMS\n","outer seal mismatch")
    return rows,{"manifest_sha256":sha(manifest),"outer_file_sha256":sha(outer)}

def expected_cohort():
    require(sha(M1313)==M1313_SHA256,"M1313 SHA drift"); return strict(M1313)["cohort"]["samples"]

def validate_identity(manifest):
    require(manifest.get("schema")=="m1227_motion_final_checkpoint_unified_hardware_capture_r1_v1" and
            manifest.get("status")=="CAPTURE_COMPLETE__FRESH_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM","M1227 manifest status")
    identity=manifest["identity"]; require(identity["contract_sha256"]==RUNTIME_SHA256,"runtime contract identity")
    require(identity["checkpoint_load_audit"].get("missing_count",0)==0 and identity["checkpoint_load_audit"].get("unexpected_count",0)==0,"checkpoint load audit")
    require(identity["module_counts"]=={"ATLIFTernaryPSN":105,"ShiftmaxAttention":12},"module counts")
    selected=identity["selection"]["selected"]
    require(selected["candidate_id"]=="resume_ep34" and selected["epoch"]==34,"ep34 selection")
    require(selected["checkpoint"]["sha256"]==CHECKPOINT_SHA256 and selected["configuration"]["sha256"]==CONFIG_SHA256 and selected["profile"]["sha256"]==PROFILE_SHA256,"selected artifact SHA")
    require(selected["profile"]["samples"]==825 and selected["profile"]["module_counts"]=={"ATLIFTernaryPSN":105,"ShiftmaxAttention":12},"selected profile")
    final=manifest["m1227_runtime_contract"]["final_selection_identity"]
    require(final["epoch"]==34 and final["checkpoint_sha256"]==CHECKPOINT_SHA256 and final["config_sha256"]==CONFIG_SHA256 and final["profile_sha256"]==PROFILE_SHA256 and final["selection_sha256"]==SELECTION_SHA256,"final selection identity")

def validate_result(root:Path):
    rows,seal=verify_recursive_seal(root)
    required={"manifest.json","m1227_admission.json","unified_ordered_records.jsonl","attention_qk/manifest.json","execution_trace.json","operator_runtime.json","atlif_activity.json","RUN_COMPLETE.txt"}
    require(required<=set(rows),"required sealed members")
    manifest=strict(root/"manifest.json"); validate_identity(manifest)
    admission=strict(root/"m1227_admission.json")
    require(admission=={"schema":"m1227_final_capture_admission_r1_v1","status":"PASS","ordered":9880,"attention":480,"payload_files":640,"execution":7360,"operator_rows":79,"atlif_live_rows":93,"atlif_static":105,"dead_sn_v":list(M1227.DEAD_SN_V),"claim_boundary":{"capture_only":True,"paper_result":False,"cycles":False,"speedup":False,"energy":False,"ppa":False}},"M1227 admission")
    runtime=manifest["m1227_runtime_contract"]
    require(runtime["static_modules"]==259 and runtime["static_atlif"]==105 and runtime["live_modules_per_sample"]==247 and runtime["live_atlif"]==93 and runtime["dead_sn_v"]==list(M1227.DEAD_SN_V) and runtime["dead_calls_per_sample"]==0 and runtime["ordered_records"]==9880 and runtime["attention_records"]==480 and runtime["payload_files"]==640,"M1227 runtime admission")
    observed=manifest["cohort"]["samples"]; expected=expected_cohort()
    core=lambda r:{k:r[k] for k in expected[0]}
    require(len(observed)==40 and [core(r) for r in observed]==expected,"cohort SHA/order")
    ordered=[strict_text(line) for line in (root/"unified_ordered_records.jsonl").read_text().splitlines()]
    require(len(ordered)==9880,"ordered population")
    ids=[r.get("global_sample_id",r.get("sample_id")) for r in ordered]
    require(Counter(ids)==Counter({i:247 for i in range(40)}),"40x247 ordered matrix")
    inventory={i:{(r.get("category"),r.get("name")) for r,rid in zip(ordered,ids) if rid==i}
               for i in range(40)}
    require(all(len(inventory[i])==247 for i in range(40)) and all(inventory[i]==inventory[0] for i in range(1,40)),"ordered live inventory")
    attention=strict(root/"attention_qk/manifest.json"); require(len(attention["records"])==480,"attention 480")
    try: payloads=M1227.validate_payload_population(root)
    except Exception as e: raise M1331Error("payload population") from e
    require(len(payloads)==640,"payload 640")
    execution=strict(root/"execution_trace.json"); operators=strict(root/"operator_runtime.json"); atlif=strict(root/"atlif_activity.json")
    require(len(execution)==7360,"execution 7360")
    require(len(operators)==79 and len({r["name"] for r in operators})==79 and all(int(r["calls"])==40 for r in operators),"operator 79")
    require(len(atlif)==93 and len({r["name"] for r in atlif})==93 and all(int(r["calls"])==40 for r in atlif) and not ({r["name"] for r in atlif}&set(M1227.DEAD_SN_V)),"ATLIF 93")
    require((root/"RUN_COMPLETE.txt").read_text()=="PASS_M1174_UNIFIED_CAPTURE__FRESH_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM\n","completion")
    claim=manifest["claim_boundary"]; require(claim=={"capture_only":True,"accuracy":False,"cycles":False,"speedup":False,"system_speedup":False,"energy":False,"rtl":False,"ppa":False,"fresh_result_hammer_required":True},"claim boundary")
    return {"status":"PASS_M1331_M1327_EP34_CAPTURE_RESULT","seal":seal,"population":{"ordered":9880,"attention":480,"payload":640,"execution":7360,"operator":79,"atlif":93},"identity":{"checkpoint_sha256":CHECKPOINT_SHA256,"config_sha256":CONFIG_SHA256,"profile_sha256":PROFILE_SHA256},"claim_boundary":{"capture_only":True,"paper_result":False}}

def validate_source_policy():
    policy=strict(SOURCE_CONTRACT)
    require(policy.get("schema")==SOURCE_SCHEMA and policy.get("status")==SOURCE_STATUS,"source policy")
    require(policy.get("source")=={"path":str(Path(__file__).resolve().relative_to(ROOT)),"sha256":sha(Path(__file__).resolve())},"source identity")
    require(policy.get("test")=={"path":str(TEST.relative_to(ROOT)),"sha256":sha(TEST)},"test identity")
    require(policy.get("actual_result_seal_prefilled") is False and policy.get("production_authorized") is False,"source boundary")
    require(sha(DOCS359)==DOCS359_SHA256,"docs359"); return policy

def main():
    parser=argparse.ArgumentParser(description=__doc__); group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-self-check",action="store_true"); group.add_argument("--validate-canonical-result",action="store_true")
    args=parser.parse_args()
    if args.source_self_check:
        validate_source_policy(); require("actual_result_seal" not in strict(SOURCE_CONTRACT),"actual seal forbidden"); print(PASS_TOKEN); return 0
    require(CANONICAL_RESULT.exists(),"canonical M1327 result does not yet exist")
    print(json.dumps(validate_result(CANONICAL_RESULT),sort_keys=True,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
