#!/usr/bin/env python3
"""M1763 additive analyzer with layer-private S2 witness identities.

The exact M1747 analyzer is imported as the implementation base.  TSBG math,
S2 keep/drop metrics, decision hashes, capture identity and claim boundaries are
unchanged.  Only S2's diagnostic same-block keep/drop witness is repaired:
drop/keep state is ORed per (epsilon, report scope, layer_id), multiplied by
that layer's real output-block count, then summed into layer/sequence/all rows.
No padding or cross-layer source-group OR is permitted.
"""
from __future__ import print_function

import hashlib, importlib.util, json, os
from pathlib import Path
import stat, struct, sys

ROOT = Path(__file__).resolve().parents[3]; HW = ROOT / "hw_autoresearch_nts07"
SOURCE = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1763_m1707_ep34_tsbg_layer_private_s2_witness_source.py"
CONTRACT = HW / "contracts/m1763_m1707_ep34_tsbg_layer_private_s2_witness_source_contract_r1_20260902.json"
CONTRACT_SIDECAR = Path(str(CONTRACT)+".sha256"); CONTRACT_OUTER = Path(str(CONTRACT)+".sha256.seal.sha256")
M1747_SOURCE = HW / "system_simulator/scripts/analyze_m1747_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_source.py"
M1748_REVIEW = HW / "reviews/m1748_m1747_m1727_ep34_tsbg_schema_identity_successor_source_hammer_r1_20260901"
M1749_RELEASE = HW / "contracts/m1749_m1748_m1747_m1727_ep34_tsbg_schema_identity_successor_analysis_release_r1_20260901.json"
M1762_FAILURE = HW / "results/m1762_m1756_m1754_m1747_tsbg_shape_failure_receipt_r1_20260901.json"
M1762_FAILURE_SIDECAR = Path(str(M1762_FAILURE)+".sha256"); M1762_FAILURE_OUTER = Path(str(M1762_FAILURE)+".sha256.seal.sha256")
M1762_REVIEW = HW / "reviews/m1762_m1756_m1754_m1747_tsbg_shape_failure_independent_diagnosis_r1_20260901"
M1744_REVIEW = HW / "reviews/m1744_m1707_ep34_tsbg_capture_result_independent_hammer_r1_20260901"
FUTURE_REVIEW = HW / "reviews/m1764_m1763_m1707_ep34_tsbg_layer_private_s2_witness_source_hammer_r1_20260902"
FUTURE_RELEASE = HW / "contracts/m1765_m1764_m1763_ep34_tsbg_layer_private_s2_witness_analysis_release_r1_20260902.json"
RESULT = HW / "results/m1763_m1707_ep34_tsbg_layer_private_s2_witness_r1_20260902"
WORK = HW / "results/.m1763_m1707_ep34_tsbg_layer_private_s2_witness_r1_20260902.work"
ATTEMPT = HW / "results/.m1763_m1707_ep34_tsbg_layer_private_s2_witness_attempt_consumed"

M1747_SHA="3bc48502ab1cccf579cfc65dc0cba2747e5bd38a8a4df82dda3f626f7283683b"
M1748_TRIPLE=("f9c3e152bb10d67a1e0b2421565e0f72469804fab4330dae9c00518b684e1c47","10683d2a63035841ef17572a5ca8b57a98eb260cb5b8c39d8d5eabbfb132e594","d1ba7c36dff713385fc30817877f3228516f9a6fa862805a44e5f7d6355e07cc")
M1749_SHA="6114020ab8d4da7c9a7c6f149496ee3efb1e7d19aeff5e34becaf60c1d465806"
M1762_FAILURE_TRIPLE=("42c6771cf1b585174e0d9b3198392bc6761b18d5324eeb26033043f738d559d7","9d2f2bce905184b56ac41fbdf84e4e40cac869fb8a1d25558cb7aacd952c5987","e0357977836659782990ba193641dfff5425bfe070820212648bede1f1c1e501")
M1762_REVIEW_TRIPLE=("ecf3fbfc595efb56b404699f0eacdfb278aa5a7008cf4166005bfacfca0642ff","7e6935f74d9100407695fe7892ab075561dfbf483c587ca6e240c73c3538ba00","18f0862dbca4ce4761e4290448c2a7102f9689203c28531b7236fb45c6863e39")
M1744_TRIPLE=("d237b3a64cf47313873a84a4749465b7cc7361bd8cf57dde5a0b6275f336dbc7","df15fe385bc7f5eccde2fecd19f5fe478dbc0480653cec5aab208c59a8a6b1f4","40c3e5f2c4a98be985bf225fe6cf3a3cda88c3a32047a372c84ca0608baaf1d2")
M1707_CAPTURE_SEALS=("be0b89f9b8084baf0c2cd959805530a4e4f41c437a446335142a65bd73960a8f","8d63d1054452377836c333c9848f771a6fc964e4ed4b00ed9a22f1537bd73c85")
SCHEMA="m1763_m1707_ep34_tsbg_layer_private_s2_witness_r1_v1"
STATUS="DIAGNOSTIC_SCREENING_ONLY__LAYER_PRIVATE_S2_WITNESS__TSBG_UNCHANGED__NO_PAPER_RESULT"
REVIEW_SCHEMA="m1764_m1763_m1707_ep34_tsbg_layer_private_s2_witness_source_hammer_r1_v1"
REVIEW_STATUS="PASS_M1764_M1763_SOURCE_HAMMER__M1765_RELEASE_MAY_BE_CREATED"
RELEASE_SCHEMA="m1765_m1764_m1763_ep34_tsbg_layer_private_s2_witness_analysis_release_r1_v1"
RELEASE_STATUS="AUTHORIZE_ONE_M1763_LAYER_PRIVATE_S2_WITNESS_ANALYSIS"

class M1763Error(RuntimeError): pass
def require(v,m):
    if not v: raise M1763Error(m)
def sha256(p):
    d=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): d.update(b)
    return d.hexdigest()
def regular_exact(p,e,l):
    p=Path(p)
    try: mode=p.lstat().st_mode
    except OSError as x: raise M1763Error("missing "+l) from x
    require(stat.S_ISREG(mode) and not p.is_symlink(),l+" nonregular"); require(sha256(p)==e,l+" SHA drift")
def strict_json(p):
    def pairs(rows):
        d={}
        for k,v in rows: require(k not in d,"duplicate JSON key"); d[k]=v
        return d
    v=json.loads(Path(p).read_text(encoding="utf-8"),object_pairs_hook=pairs,parse_constant=lambda t:(_ for _ in()).throw(M1763Error("nonfinite")))
    require(type(v) is dict,"JSON root"); return v
def verify_sidecar(p,s,o,l):
    p,s,o=Path(p),Path(s),Path(o); require(s.is_file() and o.is_file() and not s.is_symlink() and not o.is_symlink(),l+" seals")
    require(s.read_text().split()==[sha256(p),p.name] and o.read_text().split()==[sha256(s),s.name],l+" seal drift")
def verify_dir(root, triple, label):
    root=Path(root); sums=root/"SHA256SUMS"; outer=root/"SHA256SUMS.seal.sha256"
    require(root.is_dir() and sums.is_file() and outer.is_file(),label+" missing")
    require((sha256(root/"review.json"),sha256(sums),sha256(outer))==triple,label+" triple drift")
    require(outer.read_text().split()==[sha256(sums),sums.name],label+" outer")
    for line in sums.read_text().splitlines():
        d,n=line.split(None,1); n=n.strip().lstrip("*"); regular_exact(root/n,d,label+" member")
    return {"review_sha256":triple[0],"manifest_sha256":triple[1],"outer_seal_file_sha256":triple[2]}

regular_exact(M1747_SOURCE,M1747_SHA,"M1747 source")
_spec=importlib.util.spec_from_file_location("m1763_exact_m1747",str(M1747_SOURCE)); require(_spec and _spec.loader,"import M1747")
M1747=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M1747); regular_exact(M1747_SOURCE,M1747_SHA,"M1747 after import")
BASE=M1747.BASE; _BASE_CANONICAL=M1747._BASE_CANONICAL_JSON_BYTES; _BASE_DECISION_ACCUMULATOR=BASE.DecisionAccumulator; _ACTIVE=None

class DecisionAccumulator(_BASE_DECISION_ACCUMULATOR):
    """Exact predecessor work metrics with layer-private S2 witness state."""
    def __init__(self, layer_rows, sample_rows, betas, np):
        _BASE_DECISION_ACCUMULATOR.__init__(self,layer_rows,sample_rows,betas,np)
        self.s2_seen={}
    def consume_pair(self,sample_id,layer_id,codes):
        np=self.np; row=self.layers[int(layer_id)]; sample=self.samples[int(sample_id)]; target=row["target"]
        BASE.require(target in ("FC1","FC2"),"binary frame target drift")
        value=np.asarray(codes,dtype=np.int8); channels=int(row["input_channels"])
        BASE.require(value.ndim==2 and int(value.shape[1])==channels and int(value.shape[0])==int(row["tokens_per_call"]),"pair code matrix shape drift")
        padded=BASE.ceil_div(channels,BASE.GROUP_WIDTH)*BASE.GROUP_WIDTH
        if padded!=channels: value=np.pad(value,((0,0),(0,padded-channels)),mode="constant")
        shaped=value.reshape(value.shape[0],-1,BASE.GROUP_WIDTH); nnz=(shaped!=0).sum(axis=2).astype(np.int16); active=nnz>0
        magnitude=np.abs(shaped.astype(np.int16)).sum(axis=2).astype(np.int32)
        output_tiles=BASE.ceil_div(int(row["output_channels"]),BASE.TSBG_OUTPUT_TILE); layout=row["weight_layout"]
        row_bytes=int(layout["row_bytes"]); base=int(layout["base_address"])
        BASE.require(row_bytes==BASE.GROUP_WIDTH*BASE.TSBG_OUTPUT_TILE*4 and base%row_bytes==0 and int(layout["source_group_count"])==int(active.shape[1]) and int(layout["output_tile_count"])==output_tiles and int(layout["bank_count"])==BASE.WEIGHT_BANKS,"static weight layout drift")
        sequence=sample["sequence"]
        for bundle in BASE.BUNDLES:
            metric=BASE.tsbg_pair_metrics(active,nnz,output_tiles,row_bytes,base//row_bytes,bundle,np)
            for st,sc in (("all","FC1_FC2"),("sequence",sequence),("family",target),("layer",row["module_name"])):
                BASE._add_sum_metric(self.tsbg,(bundle,st,sc),metric)
        if target=="FC1":
            for epsilon in BASE.S2_EPSILON_RATIO:
                metric=BASE.s2_fc1_pair_metrics(active,nnz,magnitude,int(row["output_channels"]),self.betas[int(layer_id)],epsilon,np)
                self.s2_hash[epsilon].update(struct.pack("<IId",int(sample_id),int(layer_id),float(epsilon))); self.s2_hash[epsilon].update(metric.pop("decision_payload"))
                drop=metric.pop("drop_seen_by_source_group"); keep=metric.pop("keep_seen_by_source_group"); ob=BASE.ceil_div(int(row["output_channels"]),BASE.S2_OUTPUT_TILE)
                for st,sc in (("all","FC1"),("sequence",sequence),("layer",row["module_name"])):
                    report=(epsilon,st,sc); BASE._add_sum_metric(self.s2,report,metric,max_fields=("max_dropped_block_abs_output_code_debt","max_accumulated_abs_output_code_debt_per_token"))
                    key=(epsilon,st,sc,int(layer_id)); seen=self.s2_seen.setdefault(key,{"drop":np.zeros(drop.shape,dtype=np.bool_),"keep":np.zeros(keep.shape,dtype=np.bool_),"output_blocks":ob})
                    BASE.require(seen["drop"].shape==drop.shape and int(seen["output_blocks"])==ob,"layer-private S2 witness drift")
                    seen["drop"]|=drop; seen["keep"]|=keep
        self.pairs+=1; self.tokens+=int(value.shape[0]); self.nonzero_codes+=int(nnz.sum())
    def finalize_s2_rows(self):
        rows=[]
        for key in sorted(self.s2,key=lambda x:(x[0],x[1],x[2])):
            epsilon,st,sc=key; metric=dict(self.s2[key]); layer_seen=[v for k,v in self.s2_seen.items() if k[:3]==key]
            BASE.require(layer_seen,"missing layer-private S2 witness")
            witness=sum(int((v["drop"]&v["keep"]).sum())*int(v["output_blocks"]) for v in layer_seen)
            bb=metric["baseline_nonzero_blocks"]; bw=metric["baseline_weight_bytes"]
            metric.update({"epsilon_ratio":epsilon,"scope_type":st,"scope":sc,"drop_fraction_of_remaining_nonzero_blocks":BASE._ratio(metric["dropped_blocks"],bb),"extra_nonzero_product_reduction":BASE._ratio(metric["saved_nonzero_products"],metric["baseline_nonzero_products"]),"metadata_to_baseline_weight_bytes":BASE._ratio(metric["metadata_bytes"],bw),"dynamic_same_block_keep_drop_witness_count":witness,"layer_private_witness_identity":True,"layer_private_witness_layers":len(layer_seen),"paired_aee_present":False,"overall_delta_aee":None,"max_sequence_delta_aee":None,"same_resource_cycle_speedup":None,"passes_fixed_gate":False,"paper_admission":False}); rows.append(metric)
        return rows

def identities():
    return {"source_sha256":sha256(SOURCE),"test_sha256":sha256(TEST),"contract_sha256":sha256(CONTRACT),"contract_sidecar_sha256":sha256(CONTRACT_SIDECAR),"contract_outer_seal_file_sha256":sha256(CONTRACT_OUTER),"m1747_source_sha256":M1747_SHA,"m1748_review_sha256":M1748_TRIPLE[0],"m1749_release_sha256":M1749_SHA,"m1762_failure_sha256":M1762_FAILURE_TRIPLE[0],"m1762_review_sha256":M1762_REVIEW_TRIPLE[0],"m1744_review_sha256":M1744_TRIPLE[0],"m1707_capture_manifest_sha256":M1707_CAPTURE_SEALS[0],"m1707_capture_outer_seal_file_sha256":M1707_CAPTURE_SEALS[1]}
def validate_static():
    regular_exact(M1749_RELEASE,M1749_SHA,"M1749"); regular_exact(M1762_FAILURE,M1762_FAILURE_TRIPLE[0],"M1762 failure"); regular_exact(M1762_FAILURE_SIDECAR,M1762_FAILURE_TRIPLE[1],"M1762 sidecar"); regular_exact(M1762_FAILURE_OUTER,M1762_FAILURE_TRIPLE[2],"M1762 outer"); verify_sidecar(M1762_FAILURE,M1762_FAILURE_SIDECAR,M1762_FAILURE_OUTER,"M1762 failure")
    verify_dir(M1762_REVIEW,M1762_REVIEW_TRIPLE,"M1762 review"); verify_dir(M1748_REVIEW,M1748_TRIPLE,"M1748 review"); verify_dir(M1744_REVIEW,M1744_TRIPLE,"M1744 review")
    f=strict_json(M1762_FAILURE); require(f.get("root_cause",{}).get("s2_cross_layer_witness_aggregation_implicated") is True and f.get("root_cause",{}).get("tsbg_algorithm_implicated") is False and f.get("absence_and_budget",{}).get("m1756_authority_consumed") is True,"M1762 semantics")
def validate_contract():
    verify_sidecar(CONTRACT,CONTRACT_SIDECAR,CONTRACT_OUTER,"M1763 contract"); r=strict_json(CONTRACT)
    require(r.get("schema")=="m1763_m1707_ep34_tsbg_layer_private_s2_witness_source_contract_r1_v1" and r.get("source")=={"path":str(SOURCE.relative_to(ROOT)),"sha256":sha256(SOURCE)} and r.get("test")=={"path":str(TEST.relative_to(ROOT)),"sha256":sha256(TEST)} and r.get("authorization",{}).get("analysis_run") is False and r.get("claim_boundary",{}).get("paper_result") is False,"M1763 contract")
def validate_future_review(root,ids):
    b=verify_dir_dynamic(root,"M1764 review"); r=strict_json(Path(root)/"review.json")
    require(r.get("schema")==REVIEW_SCHEMA and r.get("status")==REVIEW_STATUS and r.get("identity")==ids and r.get("authorization")=={"m1765_release_may_be_created":True,"analysis_run":False,"capture_verify":False} and r.get("claim_boundary",{}).get("paper_result") is False,"M1764 review"); return b
def verify_dir_dynamic(root,label):
    root=Path(root); sums=root/"SHA256SUMS"; outer=root/"SHA256SUMS.seal.sha256"; require(root.is_dir() and sums.is_file() and outer.is_file(),label+" missing"); require(outer.read_text().split()==[sha256(sums),sums.name],label+" seal")
    for line in sums.read_text().splitlines(): d,n=line.split(None,1); regular_exact(root/n.strip().lstrip("*"),d,label+" member")
    return {"review_sha256":sha256(root/"review.json"),"manifest_sha256":sha256(sums),"outer_seal_file_sha256":sha256(outer)}
def validate_future_release(path,review,ids):
    path=Path(path); s=Path(str(path)+".sha256"); o=Path(str(path)+".sha256.seal.sha256"); verify_sidecar(path,s,o,"M1765 release"); r=strict_json(path); expected=dict(ids); expected.update({"m1764_review_sha256":review["review_sha256"],"m1764_review_outer_seal_file_sha256":review["outer_seal_file_sha256"]})
    require(r.get("schema")==RELEASE_SCHEMA and r.get("status")==RELEASE_STATUS and r.get("identity")==expected and r.get("authorization")=={"analysis_runs":1,"capture_verifications":1,"result_publications":1,"attempts":1,"automatic_retry":False,"gpu_runs":0,"eda_runs":0,"all_other_runs":0} and r.get("claim_boundary",{}).get("paper_result") is False,"M1765 release"); return {"release_sha256":sha256(path)}
def verify_authority():
    validate_contract(); validate_static(); ids=identities(); rv=validate_future_review(FUTURE_REVIEW,ids); rl=validate_future_release(FUTURE_RELEASE,rv,ids); return ids,rv,rl
def canonical(value):
    if type(value) is dict and value.get("schema")==SCHEMA:
        require(_ACTIVE is not None,"M1763 authority absent"); value.setdefault("identity",{}).update({"m1763_contract_sha256":_ACTIVE[0]["contract_sha256"],"m1762_failure_sha256":M1762_FAILURE_TRIPLE[0],"m1764_review_sha256":_ACTIVE[1]["review_sha256"],"m1765_release_sha256":_ACTIVE[2]["release_sha256"]}); value["s2_witness_repair"]={"layer_private_identity":True,"cross_layer_padding":False,"tsbg_changed":False}
    return _BASE_CANONICAL(value)

BASE.SOURCE=SOURCE; BASE.TEST=TEST; BASE.CONTRACT=CONTRACT; BASE.RESULT=RESULT; BASE.WORK=WORK; BASE.SCHEMA=SCHEMA; BASE.STATUS=STATUS; BASE.DecisionAccumulator=DecisionAccumulator; BASE.canonical_json_bytes=canonical
def run_analysis():
    global _ACTIVE; require(_ACTIVE is None,"active"); authority=verify_authority(); require(not os.path.lexists(str(RESULT)) and not os.path.lexists(str(WORK)) and not os.path.lexists(str(ATTEMPT)),"fresh M1763 namespaces")
    ATTEMPT.mkdir()
    _ACTIVE=authority
    try: return BASE.run_analysis()
    finally: _ACTIVE=None
def source_self_check():
    validate_contract(); validate_static(); require(not any(os.path.lexists(str(p)) for p in (RESULT,WORK,ATTEMPT)),"fresh namespaces")
    return {"status":"PASS_M1763_SOURCE_SELF_CHECK__NO_ANALYSIS","tsbg_math_changed":False,"s2_layer_private_witness":True,"analysis_runs":0,"capture_touched":False,"gpu_runs":0,"eda_runs":0,"network_access":False,"paper_result":False}
def main(argv=None):
    parser=BASE.argparse.ArgumentParser(description=__doc__); g=parser.add_mutually_exclusive_group(required=True); g.add_argument("--source-self-check",action="store_true"); g.add_argument("--run-analysis",action="store_true"); a=parser.parse_args(argv); v=source_self_check() if a.source_self_check else run_analysis(); print(json.dumps(v,indent=2,sort_keys=True)); return 0
if __name__=="__main__": sys.exit(main())
