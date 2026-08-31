#!/usr/bin/env python3
"""Different-author, read-only hammer for the sealed M1111DR2 result."""
from __future__ import annotations
import hashlib, json, math, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RESULT = HW / "results/m1111dr2_m1105dr2_decoder_only_address_timed_production_r2_20260830"
SEAL = RESULT / ".m1111dr2_atomic_seal"
PAYLOADS = {"RUN_COMPLETE.txt", "m1111dr2_decoder_call_schedule.jsonl", "m1111dr2_decoder_result.json"}
SOURCE = HW / "system_simulator/scripts/build_m1105dr2_decoder_only_address_timed_source.py"
CONTRACT = HW / "contracts/m1111dr2_m1105dr2_decoder_only_production_runner_source_contract_r2_20260830.json"
M1110_OUTER = HW / "reviews/m1110d_m1105dr2_decoder_source_contract_receipt_independent_hammer_r1_20260830/SHA256SUMS.seal.sha256"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
KINDS = {"input_descriptor_read", "weight_read", "psum_read", "compute", "psum_write", "output_commit"}
SEQUENCES = ["interlaken_01_a", "thun_01_b", "zurich_city_12_a"]
MODULES = [f"sttmultires_unet.decoders.{i}.deconv.0" for i in range(4)]
ROW_CLAIM = {"diagnostic_only": True, "final_checkpoint_rebind_required": True,
             "paper_ppa_ready": False, "speedup_admitted": False,
             "system_speedup_admitted": False}

def require(ok, msg):
    if not ok: raise RuntimeError(msg)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def strict(raw):
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,"duplicate JSON key") ; out[k]=v
        return out
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(RuntimeError("nonfinite")))
def hex64(value): return type(value) is str and len(value)==64 and all(c in "0123456789abcdef" for c in value)

def verify_seal():
    require(RESULT.is_dir() and not RESULT.is_symlink(),"result root")
    require({p.name for p in RESULT.iterdir()} == PAYLOADS | {SEAL.name},"top-level fileset")
    require(SEAL.is_dir() and not SEAL.is_symlink(),"seal dir")
    require({p.name for p in SEAL.iterdir()} == {"SHA256SUMS","SHA256SUMS.seal.sha256"},"seal fileset")
    rows={}
    for line in (SEAL/"SHA256SUMS").read_text().splitlines():
        digest,name=line.split("  ",1); require(name in PAYLOADS and name not in rows and hex64(digest),"manifest row")
        rows[name]=digest
    require(set(rows)==PAYLOADS,"manifest population")
    for name,digest in rows.items():
        p=RESULT/name; require(p.is_file() and not p.is_symlink() and sha(p)==digest,"payload SHA")
    outer=(SEAL/"SHA256SUMS.seal.sha256").read_text().strip().split("  ")
    require(outer==[sha(SEAL/"SHA256SUMS"),"SHA256SUMS"],"outer seal")
    return rows, sha(SEAL/"SHA256SUMS"), sha(SEAL/"SHA256SUMS.seal.sha256")

def main():
    rows_sha, manifest_sha, outer_file_sha = verify_seal()
    require(sha(SOURCE)=="b2d8ef4139283de06b7e332429bdf752ad16122ffbeda0ff7d75bce6d816a5c4","source")
    require(sha(CONTRACT)=="821819b00503b91a8fb8dfca8fe000208e10746e751a3815131dc8ff1cbed515","contract")
    require(sha(M1110_OUTER)=="9caf64e422b4cb696a600b69415bd8265dc4694066fae7ec67a5f34019f39e23","M1110 outer")
    require(sha(DOCS359)=="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4","docs359")
    contract=strict(CONTRACT.read_text()); result=strict((RESULT/"m1111dr2_decoder_result.json").read_text())
    raw=(RESULT/"m1111dr2_decoder_call_schedule.jsonl").read_text()
    calls=[strict(line) for line in raw.splitlines()]
    require(len(calls)==120 and sha(RESULT/"m1111dr2_decoder_call_schedule.jsonl")==rows_sha["m1111dr2_decoder_call_schedule.jsonl"],"120 rows")
    require(result["schema"]=="m1111dr2_m1105dr2_decoder_only_address_timed_result_v2" and
            result["status"]=="PASS_M1111DR2_DECODER_ONLY_DIAGNOSTIC_RESULT__FINAL_RESULT_HAMMER_REQUIRED","result status")
    require(result["identity"]=={
        "checkpoint":"H67_ep35","checkpoint_sha256":"4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158",
        "contract_sha256":"821819b00503b91a8fb8dfca8fe000208e10746e751a3815131dc8ff1cbed515",
        "final_checkpoint_rebind_required":True,
        "m1110d_outer_seal_file_sha256":"9caf64e422b4cb696a600b69415bd8265dc4694066fae7ec67a5f34019f39e23",
        "source_sha256":"b2d8ef4139283de06b7e332429bdf752ad16122ffbeda0ff7d75bce6d816a5c4"},"identity")
    require(result["common_resource"]==contract["common_resource"],"same-resource projection")
    require(sum(result["common_resource"]["partitions"].values())==245760,"SRAM partition")
    claim=result["claim_boundary"]
    require(claim=={"address_timed_transactions_complete":True,"decoder_only":True,
                    "diagnostic_cycles_only":True,"diagnostic_traffic_only":True,
                    "final_checkpoint_rebind_required":True,"independent_result_hammer_required":True,
                    "paper_citable_performance":False,"paper_ppa_ready":False,
                    "same_resource_schedule_complete":True,"speedup_admitted":False,
                    "system_speedup_admitted":False},"result claim boundary")
    traffic_keys={"compute","external","input_descriptor_read","onchip","output_commit",
                  "psum_read","psum_write","total","weight_read"}
    agg={k:0 for k in traffic_keys}; tx=0; cycles=0; prev_tx=0; prev_cycle=0
    for i,row in enumerate(calls):
        seq_i=i//40; sample=(i%40)//4; mod=i%4
        require(row["schema"]=="m1111dr2_decoder_address_timed_call_schedule_v2" and
                row["global_call_ordinal"]==i and row["sequence_ordinal"]==seq_i and
                row["sequence"]==SEQUENCES[seq_i] and row["sequence_sample_id"]==sample and
                row["module_ordinal"]==mod and row["module"]==MODULES[mod],"call Cartesian order")
        require(row["configuration"]=="M1105DR2_EXACT_TYPED_K8" and
                row["claim_boundary"]==ROW_CLAIM and row["d1_weight_folding"] is False,"row boundary")
        require(row["d1_exact_theta"] is (mod==1) and
                row["d1_theta_word_uint32"]==(1065353139 if mod==1 else None),"D1 theta")
        require(row["cycle_start"]==prev_cycle and row["cycle_end"]>prev_cycle and
                row["diagnostic_cycles"]==row["cycle_end"]-row["cycle_start"],"cycle continuity")
        require(row["transaction_ordinal_first"]==prev_tx and
                row["transaction_ordinal_last"]-prev_tx+1==row["transaction_count"],"transaction continuity")
        require(set(row["kind_summaries"])==KINDS and all(hex64(row[k]) for k in
                ("address_digest_sha256","dependency_digest_sha256","schedule_digest_sha256")),"kind/digests")
        require(sum(v["count"] for v in row["kind_summaries"].values())==row["transaction_count"],"transaction sum")
        tr=row["diagnostic_traffic_bytes"]; require(set(tr)==traffic_keys,"traffic schema")
        require(all(row["kind_summaries"][k]["traffic_bytes"]==tr[k] for k in KINDS),"kind traffic")
        require(tr["external"]==tr["input_descriptor_read"]+tr["output_commit"] and
                tr["onchip"]==tr["weight_read"]+tr["psum_read"]+tr["psum_write"] and
                tr["total"]==tr["external"]+tr["onchip"]+tr["compute"],"traffic arithmetic")
        for summary in row["kind_summaries"].values():
            require(summary["count"]>0 and summary["address_first"]<=summary["address_last"] and
                    row["cycle_start"]<=summary["issue_first"]<=summary["issue_last"]<row["cycle_end"] and
                    row["cycle_start"]<summary["return_first"]<=summary["return_last"]<row["cycle_end"] and
                    row["cycle_start"]<summary["commit_first"]<=summary["commit_last"]<row["cycle_end"] and
                    sum(summary["stall_events"].values())==summary["count"],"timing/stall summary")
        for k in traffic_keys: agg[k]+=tr[k]
        tx+=row["transaction_count"]; cycles+=row["diagnostic_cycles"]
        prev_tx=row["transaction_ordinal_last"]+1; prev_cycle=row["cycle_end"]
    require(tx==4537773925 and cycles==3604247976 and prev_tx==tx and prev_cycle==cycles,"global transaction/cycle")
    require(agg==result["diagnostic"]["traffic_bytes"] and
            result["diagnostic"]["cycles"]==cycles and result["diagnostic"]["ratios_or_speedups"] is None,"aggregate diagnostic")
    require(result["population"]=={"call_row_stream_digest_sha256":rows_sha["m1111dr2_decoder_call_schedule.jsonl"],
            "call_schedule_sha256":rows_sha["m1111dr2_decoder_call_schedule.jsonl"],"calls":120,
            "timesteps_per_call":10,"transaction_count":tx},"population")
    require((RESULT/"RUN_COMPLETE.txt").read_text()=="M1111DR2_DECODER_DIAGNOSTIC_COMPLETE__RESULT_HAMMER_REQUIRED\n","completion")
    print(json.dumps({"status":"PASS_M1330_M1111DR2_DIAGNOSTIC_RESULT_HAMMER",
        "checks":{"recursive_seal":True,"calls":120,"transactions":tx,"cycles":cycles,
                  "traffic_bytes":agg["total"],"same_resource":True,"ep35":True,
                  "final_rebind_required":True,"paper_citable_performance":False},
        "result_seal":{"manifest_sha256":manifest_sha,"outer_file_sha256":outer_file_sha,
                       "result_sha256":rows_sha["m1111dr2_decoder_result.json"]},
        "execution":{"remote":False,"gpu":False,"eda":False}},sort_keys=True,indent=2))

if __name__=="__main__": main()
