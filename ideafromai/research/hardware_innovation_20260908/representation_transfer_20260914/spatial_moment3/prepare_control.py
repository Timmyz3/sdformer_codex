"""Prepare the specified unconstrained U2=0 phase3 control for the same RTL.

There is no native q2 and no native-g relation admission for this control.
The physical three planes drive the existing M3 datapath; its p2 stays whole.
"""
from pathlib import Path
import subprocess
H=Path(__file__).resolve().parent;D=H/'unconstrained';D.mkdir(exist_ok=True)
s=(H/'prepare.py').read_text()
s=s.replace("N=H.parent;F=N/'spatial_winograd_pruning/moment'", "N=H.parents[1];F=N/'spatial_winograd_pruning/unconstrained'")
s=s.replace("q2=f['q2'].astype(np.int64)","q2=f['physical_coeff3'].astype(np.int64)")
s=s.replace('assert np.all(q2[:,:,1]==q2[:,:,0]+q2[:,:,2])',"assert bool(f['raw_p_is_unhalved']) and np.max(abs(q2))<4096")
s=s.replace("(H/'spatial_core.sv').read_text()", "(H.parent/'spatial_core.sv').read_text()")
s=s.replace("W=np.einsum('orx,rcy->ocyx',q2,q1);lo=", "phase=np.stack([np.stack([q2[:,:,0],q2[:,:,1],q2[:,:,1]-q2[:,:,0],np.zeros_like(q2[:,:,0])],axis=2),np.stack([np.zeros_like(q2[:,:,0]),q2[:,:,1]-q2[:,:,2],q2[:,:,1],q2[:,:,2]],axis=2)])\nW=np.einsum('aorx,rcy->aocyx',phase,q1);assert np.array_equal(W,f['expanded_phase_int32']);lo=")
s=s.replace(" native=np.stack([np.einsum('trpx,orx->top',z[:,:,:,x:x+3],q2) for x in range(2)],axis=3)\n assert np.array_equal(p,native)\n",'')
s=s.replace("np.einsum('tcij,ocij->to',ev[:,:,y:y+3,x:x+3],W)","np.einsum('tcij,ocij->to',ev[:,:,y:y+3,:],W[x])")
s=s.replace("seq=np.load(Q/'sequence_tiles.npz');meta=", "seq=np.load(F/'gold_sequences.npz');meta=")
s=s.replace("seq['identity_fp32_bits'][i]);sets['sequences']", "seq['identity_fp32_bits'][i],{key:seq[key][i] for key in ['z_halo_int','p_int','J_q20','wide_int64','i24']});sets['sequences']")
s=s.replace("function='frozen moment native3tap with g1=g0+g2',all_scalar_constraints=1536,q2_range=", "function='unconstrained U2=0 physical3 planes; unhalved p2 and newly rounded half-scale consumer',all_scalar_constraints=0,physical_coeff3_range=")
s=s.replace('new moment CPU gold; not new 36-frame moment network gold','new unconstrained phase3 CPU gold; not new 36-frame control network gold')
assert "q2=f['q2']" not in s and 'assert np.all(q2[:,:,1]' not in s
(D/'prepare.py').write_text(s)
subprocess.run(['/opt/anaconda3/bin/python3.12','-B',str(D/'prepare.py')],check=True)
