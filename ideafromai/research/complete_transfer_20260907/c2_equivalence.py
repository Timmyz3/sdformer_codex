"""Exact full-source duplicate opportunity, with a zero-aware baseline."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent
PARSER=ROOT.parent/'mechanism_rebuild_gh_20260906/scripts'
sys.path.insert(0,str(PARSER))
from screen_threshold_packets import sources,EXPECTED

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    out=ROOT/'c2_equivalence_r1.json';assert not out.exists()
    plan_path=ROOT/'c2_equivalence_plan.json';plan=json.loads(plan_path.read_text());plan_sha=sha(plan_path)
    arrays=sources();rows=[]
    for stage in [0,3]:
        name=f'sttmultires_unet.encoders.swin3d.layers.{stage}.swin_blocks.0.mlp.fc2'
        spec,S=arrays[name];M,C=S.shape;H=spec['output_channels']
        packed=np.packbits(S,axis=1,bitorder='little')
        unique,index,inverse,counts=np.unique(packed,axis=0,return_index=True,return_inverse=True,return_counts=True)
        pop=S.sum(1,dtype=np.int64);upop=pop[index]
        valid=upop>0;U=int(valid.sum());active=int((pop>0).sum());zero=M-active
        assert int(counts.sum())==M and np.array_equal(unique[inverse],packed)
        index_bits=max(1,U.bit_length());count_bits=M.bit_length()
        zero_baseline_bits=active*H*32+M
        class_Y_bits=U*H*32
        keys_bits=U*((C+7)//8)*8
        index_payload_bits=M*index_bits
        count_payload_bits=U*count_bits
        total_bits=class_Y_bits+keys_bits+index_payload_bits+count_payload_bits
        # Small signed rational-numerator diagnostics keep squared sums in int64.
        c=np.arange(C,dtype=np.int64)
        theta_num=9+c%5
        phi=np.stack([((c*7+3)%15-7)*theta_num,((c*11+2)%13-6)*theta_num],axis=1)
        assert int(np.abs(phi).sum(0).max())**2*M < (1<<63)
        Y=S.astype(np.int64)@phi;UY=Y[index]
        assert np.array_equal(Y,UY[inverse])
        sum1=Y.sum(0);sum2=(Y*Y).sum(0)
        weighted1=(UY*counts[:,None]).sum(0);weighted2=((UY*UY)*counts[:,None]).sum(0)
        assert np.array_equal(sum1,weighted1) and np.array_equal(sum2,weighted2)
        direct_adds=int(np.maximum(pop-1,0).sum());unique_adds=int(np.maximum(upop-1,0).sum())
        r={'module':name,'M':M,'T':10,'P':M//10,'C':C,'H':H,'zero_source_rows':zero,
           'nonzero_rows':active,'unique_nonzero_rows':U,'repeated_nonzero_rows':active-U,
           'nonzero_repetition_fraction':(active-U)/max(1,active),
           'largest_nonzero_class':int(counts[valid].max()),
           'nonzero_class_multiplicity_histogram':{str(int(n)):int((counts[valid]==n).sum()) for n in np.unique(counts[valid])},
           'direct_dot_adds_per_output_channel_before_other_reuse':direct_adds,
           'unique_dot_adds_per_output_channel_before_other_reuse':unique_adds,
           'storage_bits':{'uncompressed_full_Y':M*H*32,'strong_zero_baseline_Y_and_bitmap':zero_baseline_bits,
              'unique_Y':class_Y_bits,'explicit_full_source_keys':keys_bits,'row_class_indices':index_payload_bits,
              'class_counts':count_payload_bits,'candidate_before_hash_table_ports_and_scheduler':total_bits},
           'candidate_explicit_bits_over_zero_baseline':total_bits/zero_baseline_bits,
           'integer_moment_diagnostics':{'lane_values':M*2,'sum':sum1.tolist(),'sum_squares':sum2.tolist(),'errors':0}}
        rows.append(r)
        print(stage,'nonzero repeats',r['nonzero_repetition_fraction'],'explicit storage ratio',r['candidate_explicit_bits_over_zero_baseline'],flush=True)
    assert sha(plan_path)==plan_sha
    report={'status':'EXPLORATORY_EXACT_SOURCE_CLASS_OPPORTUNITY','date':'2026-09-07','plan':plan,
        'plan_sha256':plan_sha,'script_sha256':sha(Path(__file__)),'parser_sha256':sha(PARSER/'screen_threshold_packets.py'),
        'capture_sha256':EXPECTED,'layers':rows,'limits':['No approximate source matching; all-zero advantage already assigned to baseline.',
        'Compute counts are before Prosperity/RSR/partial reuse and are not an incremental measured speedup against those works.',
        'Explicit class metadata is charged; dictionary construction, hashing occupancy, ports and scheduling are additional costs.',
        'Real shortcut remains per position; multiplicity does not merge the residual output.',
        'Exact integer moments are not frozen FP32 reduction-order or AEE proof.']}
    with out.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')

if __name__=='__main__':main()
