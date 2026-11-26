"""Counterexample to cheap features uniquely determining deferred features.

No real dataset, GT, fitted head, certificate outcome, GPU or fitting is used.
"""
from pathlib import Path
import hashlib,importlib.util,json,os
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 source=ROOT/'research/flow_story_20261021/round34_deployability/numeric/core.py'
 assert sha(source)=='8895b5ced6719e6d104e7d36dab87010d8ef519ba218b3f2b68d8380d3ac60fc'
 sp=importlib.util.spec_from_file_location('_old_fp32_features',source);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 pattern=np.where(np.arange(640)%160<80,np.float32(1),np.float32(-1))
 a=np.broadcast_to(pattern,(10,480,640)).copy();b=a.copy();b[1::2]*=np.float32(-1)
 fa=m.voxel_features(a)[:128].reshape(16,8);fb=m.voxel_features(b)[:128].reshape(16,8)
 cheap=[0,1,2,3,6,7];assert np.array_equal(fa[:,cheap],fb[:,cheap])
 assert np.all(fa[:,4]==1) and np.all(fa[:,5]==0) and np.all(fb[:,4]==0) and np.all(fb[:,5]==2)
 output=HERE/'SYNTHETIC_COUNTEREXAMPLE_V1.json';assert not output.exists()
 report=dict(status='PASS_CHEAP_FEATURE_INFORMATION_COUNTEREXAMPLE',auditor_sha256=sha(__file__),numeric_reference_sha256=sha(source),input_shape=list(a.shape),input_dtype=str(a.dtype),
  arrays='A: each cell half pixels+1,half-1 fixed for all10 bins. B: identical spatial pattern with full sign flip at each adjacent bin.',
  all16_cells_cheap_values_bitwise_equal=True,cheap_indices=cheap,cheap_first_cell=fa[0,cheap].tolist(),late_A=dict(x4=1.0,x5=0.0),late_B=dict(x4=0.0,x5=2.0),
  input_hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in [a,b]],GPU_calls=0,real_inputs_read=0,GT_values_read=0,head_models_read=0,new_fits=0,
  scope='This rules out general unique reconstruction of x4/x5 from cheap features. It does not prove frozen-head decisions differ, bound certification always fails, or any dataset-level rate.')
 output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
