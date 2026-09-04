#!/usr/bin/env python3
from __future__ import print_function
import hashlib, importlib.util, json, os, struct
from pathlib import Path
import tempfile, unittest

SOURCE=Path(__file__).resolve().parents[1]/"scripts/analyze_m1763_m1707_ep34_tsbg_layer_private_s2_witness_source.py"
S=importlib.util.spec_from_file_location("m1763",str(SOURCE)); M=importlib.util.module_from_spec(S); S.loader.exec_module(M)

class TestM1763(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  try: import numpy as np
  except ImportError: np=None
  cls.np=np
 def fixture(self):
  gs=(6,12,24,48); obs=(24,48,96,192); layers=[]; betas={}; samples=[]
  for i,(g,ob) in enumerate(zip(gs,obs)):
   lid=8+i*2; out=ob*16; layers.append({"layer_id":lid,"target":"FC1","module_name":"fc1_%d"%lid,"input_channels":g*16,"output_channels":out,"tokens_per_call":1,"weight_layout":{"row_bytes":6144,"base_address":i*48*6144,"source_group_count":g,"output_tile_count":M.BASE.ceil_div(out,96),"bank_count":8}}); betas[lid]=[1]*ob
   samples.extend([{"global_sample_id":i*2,"sequence":"same"},{"global_sample_id":i*2+1,"sequence":"same"}])
  return layers,samples,betas,gs,obs
 def run_fixture(self):
  if self.np is None: self.skipTest("numpy")
  layers,samples,betas,gs,obs=self.fixture(); a=M.DecisionAccumulator(layers,samples,betas,self.np)
  for i,g in enumerate(gs):
   low=self.np.zeros((1,g*16),dtype=self.np.int8); low[0,0]=1
   high=self.np.zeros((1,g*16),dtype=self.np.int8); high[0,0]=127
   a.consume_pair(i*2,8+i*2,low); a.consume_pair(i*2+1,8+i*2,high)
  return a,obs
 def test_heterogeneous_6_12_24_48_completes(self):
  a,obs=self.run_fixture(); rows=a.finalize_s2_rows(); self.assertTrue(rows); self.assertEqual(sorted({v["drop"].shape[0] for v in a.s2_seen.values()}),[6,12,24,48])
 def test_layer_private_output_block_weight(self):
  a,obs=self.run_fixture(); rows=a.finalize_s2_rows(); allrow=[r for r in rows if r["epsilon_ratio"]==0.01 and r["scope_type"]=="all"][0]
  self.assertEqual(allrow["dynamic_same_block_keep_drop_witness_count"],sum(obs)); self.assertEqual(allrow["layer_private_witness_layers"],4)
 def test_sequence_equals_sum_of_layers(self):
  a,obs=self.run_fixture(); rows=a.finalize_s2_rows(); seq=[r for r in rows if r["epsilon_ratio"]==0.01 and r["scope_type"]=="sequence"][0]; layer=sum(r["dynamic_same_block_keep_drop_witness_count"] for r in rows if r["epsilon_ratio"]==0.01 and r["scope_type"]=="layer")
  self.assertEqual(seq["dynamic_same_block_keep_drop_witness_count"],layer)
 def test_no_padding_or_cross_layer_or(self):
  text=SOURCE.read_text(); self.assertNotIn("np.pad(drop",text); self.assertNotIn("np.pad(keep",text); self.assertIn("int(layer_id)",text)
 def test_output_block_mutation_changes_only_own_layer(self):
  a,obs=self.run_fixture(); before=[r for r in a.finalize_s2_rows() if r["epsilon_ratio"]==0.01 and r["scope_type"]=="all"][0]["dynamic_same_block_keep_drop_witness_count"]
  key=next(k for k in a.s2_seen if k[0]==0.01 and k[1]=="all" and k[3]==8); a.s2_seen[key]["output_blocks"]+=1
  after=[r for r in a.finalize_s2_rows() if r["epsilon_ratio"]==0.01 and r["scope_type"]=="all"][0]["dynamic_same_block_keep_drop_witness_count"]; self.assertEqual(after-before,1)
 def test_decision_hash_exact_predecessor_formula(self):
  a,obs=self.run_fixture(); layers,samples,betas,gs,_=self.fixture(); expected=dict((e,hashlib.sha256()) for e in M.BASE.S2_EPSILON_RATIO)
  for i,g in enumerate(gs):
   for sid,val in ((i*2,1),(i*2+1,127)):
    codes=self.np.zeros((1,g*16),dtype=self.np.int8); codes[0,0]=val; shaped=codes.reshape(1,g,16); nnz=(shaped!=0).sum(2).astype(self.np.int16); active=nnz>0; mag=self.np.abs(shaped.astype(self.np.int16)).sum(2).astype(self.np.int32); lid=8+i*2
    for e in M.BASE.S2_EPSILON_RATIO:
     row=M.BASE.s2_fc1_pair_metrics(active,nnz,mag,layers[i]["output_channels"],betas[lid],e,self.np); expected[e].update(struct.pack("<IId",sid,lid,float(e))); expected[e].update(row["decision_payload"])
  self.assertEqual({e:a.s2_hash[e].hexdigest() for e in expected},{e:expected[e].hexdigest() for e in expected})
 def test_tsbg_math_object_identity(self):
  self.assertIs(M.BASE.tsbg_pair_metrics,M.M1747.BASE.tsbg_pair_metrics); self.assertIs(M.DecisionAccumulator.finalize_tsbg_rows,M._BASE_DECISION_ACCUMULATOR.finalize_tsbg_rows)
 def seal_dir(self,r):
  names=sorted(p.name for p in r.iterdir() if p.is_file() and p.name not in ("SHA256SUMS","SHA256SUMS.seal.sha256")); s=r/"SHA256SUMS"; s.write_text("".join("{}  {}\n".format(M.sha256(r/n),n) for n in names)); (r/"SHA256SUMS.seal.sha256").write_text("{}  SHA256SUMS\n".format(M.sha256(s)))
 def seal_file(self,p):
  s=Path(str(p)+".sha256");o=Path(str(p)+".sha256.seal.sha256");s.write_text("{}  {}\n".format(M.sha256(p),p.name));o.write_text("{}  {}\n".format(M.sha256(s),s.name))
 def test_future_authority_mutations_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); ids=M.identities(); rv=root/"review";rv.mkdir(); (rv/"review.json").write_text(json.dumps({"schema":M.REVIEW_SCHEMA,"status":M.REVIEW_STATUS,"identity":ids,"authorization":{"m1765_release_may_be_created":True,"analysis_run":False,"capture_verify":False},"claim_boundary":{"paper_result":False}},sort_keys=True));self.seal_dir(rv);b=M.validate_future_review(rv,ids)
   rel=root/"release.json";ri=dict(ids);ri.update({"m1764_review_sha256":b["review_sha256"],"m1764_review_outer_seal_file_sha256":b["outer_seal_file_sha256"]});rel.write_text(json.dumps({"schema":M.RELEASE_SCHEMA,"status":M.RELEASE_STATUS,"identity":ri,"authorization":{"analysis_runs":1,"capture_verifications":1,"result_publications":1,"attempts":1,"automatic_retry":False,"gpu_runs":0,"eda_runs":0,"all_other_runs":0},"claim_boundary":{"paper_result":False}},sort_keys=True));self.seal_file(rel);M.validate_future_release(rel,b,ids);d=json.loads(rel.read_text());d["authorization"]["attempts"]=2;rel.write_text(json.dumps(d));self.seal_file(rel)
   with self.assertRaises(M.M1763Error):M.validate_future_release(rel,b,ids)
 def test_source_self_check_inert(self):
  r=M.source_self_check();self.assertFalse(r["tsbg_math_changed"]);self.assertEqual(r["analysis_runs"],0);self.assertFalse(os.path.lexists(str(M.ATTEMPT)))

if __name__=="__main__":unittest.main()
