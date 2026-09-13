"""Bounded signed offset + N:M execution on actual matched-dense conv2 U16.
CPU payload, byte-address requests and explicit instruction census; no cycles/PPA.
Calibration is frame 0001; three other captured frames are untouched holdout.
"""
from pathlib import Path
from itertools import combinations
from collections import Counter
import argparse, json, sys, hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
OPEN=HERE.parents[1]
CAP=OPEN/'fusion_followthrough_20260913/capture_owned'
sys.dont_write_bytecode=True
sys.path.insert(0,str(OPEN/'review'))
from phase_channel_probe import downstream, metrics


def load(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def samples(data, label):
    geo=json.loads(str(data['window_geometry_json']))[label]
    gate=data[label+'_sn2_gate']; raw=data[label+'_I24']
    gy,gx=geo['gate_origin']; sy,sx=geo['source_origin']
    oy,ox=geo['output_origin']; oh,ow=geo['output_shape']
    xs=[];rs=[];u=[];pg=[];ped=[];addresses=[]
    for y in range(oh):
        for x in range(ow):
            cy,cx=2*(oy+y),2*(ox+x)
            a=np.zeros((10,96,3,3),np.int64);addr=np.full(864,-1,np.int64)
            for ky in range(3):
                for kx in range(3):
                    yy,xx=cy+ky-1,cx+kx-1
                    if 0<=yy<240 and 0<=xx<320:
                        ly,lx=yy-gy,xx-gx
                        assert 0<=ly<gate.shape[2] and 0<=lx<gate.shape[3]
                        a[:,:,ky,kx]=gate[:,:,ly,lx]
                        # One uint16 T10 word, current common spatial-major C layout.
                        addr[np.arange(96)*9+ky*3+kx]=2*((ly*gate.shape[3]+lx)*96+np.arange(96))
            xs.append(a.reshape(10,864));addresses.append(addr)
            rs.append(raw[:,:,cy-sy,cx-sx])
            u.append(data[label+'_updated_I24'][:,:,cy-gy,cx-gx])
            pg.append(data[label+'_proj_gate'][:,:,cy-gy,cx-gx])
            ped.append(data[label+'_continuous_q24'][:,:,y,x])
    source_words=sum(gate[t].astype(np.uint16)<<t for t in range(10)).transpose(1,2,0).copy()
    return dict(source_blob=source_words.astype('<u2').tobytes(),x=np.stack(xs),raw=np.stack(rs),gold=(np.stack(u),np.stack(pg),np.stack(ped)),
                addresses=np.stack(addresses), label=label)


def permutation(w,x,kind):
    if kind=='natural':return np.arange(w.shape[1])
    # Fixed, calibration-only co-layout within physical K16; both arms receive it.
    # Greedy nearest W-vector and activation support, no evaluation labels.
    wn=w/np.maximum(np.linalg.norm(w,axis=0,keepdims=True),1.)
    xn=x/np.maximum(np.linalg.norm(x,axis=0,keepdims=True),1.)
    out=[]
    for start in range(0,w.shape[1],16):
        remaining=list(range(start,min(start+16,w.shape[1])))
        while remaining:
            first=remaining.pop(0);out.append(first)
            distance=[(float(np.sum((wn[:,j]-wn[:,first])**2)+np.sum((xn[:,j]-xn[:,first])**2)),j) for j in remaining]
            selected=[j for _,j in sorted(distance)[:3]]
            out.extend(selected);remaining=[j for j in remaining if j not in selected]
    return np.array(out)


def fit(w,x,mode):
    """Enumerate ALL legal supports; equal local Gram least-squares recovery.
    Integer coefficients with original exponent; no optimizer/backprop/retraining.
    """
    h,k=w.shape;groups=k//4
    n={'dense':4,'ordinary24':2,'ordinary34':3,'shifted24':2,'shifted14':1}[mode]
    offset=mode.startswith('shifted');slots=n+int(offset)
    vals=np.zeros((groups,h,slots),np.int16);idx=np.zeros((groups,h,n),np.uint8)
    reconstructed=np.zeros_like(w)
    scores=[]
    for g in range(groups):
        wg=w[:,g*4:g*4+4];xx=x[:,g*4:g*4+4]
        gram=xx.T@xx
        # Stable ridge scaled to this group; common permission for all arms.
        ridge=max(float(np.trace(gram))/4,1)*1e-4
        target=xx@wg.T
        if mode=='dense':
            vals[g]=wg;idx[g]=np.arange(4);reconstructed[:,g*4:g*4+4]=wg;continue
        best=np.full(h,np.inf)
        for support in combinations(range(4),n):
            basis=np.eye(4)[:,support]
            if offset:basis=np.column_stack((np.ones(4),basis))
            coeff=np.linalg.solve(basis.T@(gram+ridge*np.eye(4))@basis,
                                   basis.T@(gram+ridge*np.eye(4))@wg.T).T
            coeff=np.clip(np.rint(coeff),-32768,32767).astype(np.int16)
            if offset:
                c=coeff[:,0].astype(np.int64)
                coeff[:,1:]=np.clip(coeff[:,1:].astype(np.int64),np.maximum(-32768,-32768-c)[:,None],np.minimum(32767,32767-c)[:,None]).astype(np.int16)
            candidate=coeff.astype(np.int64)@basis.T.astype(np.int64)
            error=np.sum((target-xx@candidate.T)**2,axis=0)
            # Include original-weight distance only as deterministic tie-break.
            error=error+1e-10*np.sum((candidate-wg)**2,axis=1)
            choose=error<best
            best[choose]=error[choose];vals[g,choose]=coeff[choose]
            idx[g,choose]=support;reconstructed[choose,g*4:g*4+4]=candidate[choose]
        scores.append(float(best.sum()))
    assert reconstructed.min()>=-32768 and reconstructed.max()<=32767
    return dict(values=vals,indices=idx,reconstructed=reconstructed,offset=offset,n=n,
                mode=mode,local_calibration_SSE=sum(scores))


def encode(f):
    # K4 -> H8 -> component-slot -> contiguous eight int16 weights.
    val=f['values']; ng,h,slots=val.shape
    v=val.reshape(ng,h//8,8,slots).transpose(0,1,3,2).copy().astype('<i2').tobytes()
    if f['mode']=='dense':meta=b''
    else:
        codes=np.zeros((ng,h),np.uint8)
        for j in range(f['n']):codes|=f['indices'][:,:,j]<<(2*j)
        # 2 or 3 positions per output: 4 or 6 bits. Pack actual bits densely.
        bit=0;packed=bytearray((ng*h*f['n']*2+7)//8)
        for c in codes.flat:
            for j in range(f['n']*2):
                packed[bit//8]|=((int(c)>>j)&1)<<(bit%8);bit+=1
        meta=bytes(packed)
    return v,meta


class Port:
    """One 32B response latch per coefficient stream; real aligned addresses."""
    def __init__(self,blob,counter,key):self.blob=blob;self.last=-1;self.c=counter;self.key=key
    def read(self,address,size):
        for word in range(address//32,(address+size-1)//32+1):
            if self.last!=word:self.c[self.key]+=1;self.last=word
        return self.blob[address:address+size]


def execute(f,x,addresses,perm,source_blob):
    """P2 x T10, H16 all live in 40 SIMD8 accumulators, no partial-sum spill.
    Index stream read before value requests; masks gather independently per lane.
    Every source packet is physically built, written, read and decoded.
    """
    vb,mb=encode(f);counter=Counter();out=np.zeros((len(x),10,16),np.int64)
    inverse=np.argsort(perm);xp=x[:,:,perm];ng=len(perm)//4
    counter['coefficient_cold_fill_CR256_words']=(len(vb)+31)//32+(len(mb)+31)//32
    counter['permutation_cold_fill_SR64_words']=(len(perm)*2+7)//8 if not np.array_equal(perm,np.arange(len(perm))) else 0
    counter['coefficient_payload_bytes']=len(vb)+len(mb)
    counter['permutation_payload_bytes']=len(perm)*2 if not np.array_equal(perm,np.arange(len(perm))) else 0
    # Replay each horizontal pair of output anchors with the same common latches.
    for p0 in range(0,len(x),2):
        pair=xp[p0:p0+2].reshape(-1,len(perm));tokens=len(pair)
        acc=np.zeros((tokens,16),np.int64)
        vp=Port(vb,counter,'coefficient_value_CR256_requests')
        mp=Port(mb,counter,'coefficient_metadata_CR256_requests')
        counter['accumulator_clear_SIMD8']+=tokens*2
        for g in range(ng):
            reference_a=pair[:,g*4:g*4+4]
            # Read each source physical SR64 word once inside the quartet/P2 gather.
            addrs=addresses[p0:p0+2][:,perm[g*4:g*4+4]]
            source_cache={word:source_blob[word*8:word*8+8] for word in set(int(z)//8 for z in addrs.flat if z>=0)}
            counter['source_gate_SR64_requests']+=len(source_cache)
            a=np.zeros_like(reference_a)
            for ip in range(len(addrs)):
                for j,address in enumerate(addrs[ip]):
                    if address>=0:
                        raw=source_cache[int(address)//8]
                        word=int.from_bytes(raw[int(address)%8:int(address)%8+2],'little')
                        a[ip*10:ip*10+10,j]=(word>>np.arange(10))&1
            assert np.array_equal(a,reference_a), 'physical source decode mismatch'
            counter['source_address_select_slots']+=int(np.count_nonzero(addrs>=0))
            if counter['permutation_payload_bytes']:counter['permutation_SR64_requests']+=1
            counter['source_quartet_pack_slots']+=tokens
            if not np.any(a):counter['empty_quartets_skipped']+=1;continue
            mask=np.sum(a.astype(np.uint8)*(1<<np.arange(4,dtype=np.uint8)),axis=1)
            live=int(sum(int(v!=0)<<t for t,v in enumerate(mask)))
            payload=int(g).to_bytes(2,'little')+live.to_bytes(4,'little')+bytes(int(mask[t])|(int(mask[t+1])<<4) for t in range(0,tokens,2))
            payload=payload.ljust(16,b'\0');assert len(payload)==16
            counter['source_packet_SW64_writes']+=2;counter['source_packet_SR64_reads']+=2
            assert int.from_bytes(payload[:2],'little')==g
            decoded=np.array([(payload[6+t//2]>>(4*(t%2)))&15 for t in range(tokens)],np.uint8)
            assert np.array_equal(decoded,mask)
            activation=((decoded[:,None]>>np.arange(4))&1).astype(np.int64)
            counter['source_packet_decode_slots']+=1
            pop=activation.sum(1)
            if f['offset']:counter['shared_popcount_slots']+=int(np.count_nonzero(pop))
            for hb in range(2):
                if mb:
                    bits=f['n']*2;bitstart=(g*16+hb*8)*bits
                    raw=mp.read(bitstart//8,bits)
                    stream=int.from_bytes(raw,'little');support=np.array([[(stream>>(lane*bits+2*j))&3 for j in range(f['n'])] for lane in range(8)])
                    counter['support_decode_SIMD8_slots']+=1
                else:support=np.tile(np.arange(4),(8,1))
                for slot in range(f['n']+int(f['offset'])):
                    if f['offset'] and slot==0:act=np.broadcast_to(pop[:,None],(tokens,8))
                    else:
                        j=slot-int(f['offset']);act=activation[:,support[:,j]]
                        if f['mode']!='dense':counter['activation_gather_SIMD8_slots']+=int(np.count_nonzero(pop))
                    # Early activation test is based on decoded source/support only.
                    if not np.any(act):counter['empty_component_value_request_skipped']+=1;continue
                    off=((g*2+hb)*(f['n']+int(f['offset']))+slot)*16
                    weight=np.frombuffer(vp.read(off,16),'<i2').astype(np.int64)
                    live_t=np.any((act!=0)&(weight[None,:]!=0),axis=1)
                    acc[:,hb*8:hb*8+8]+=act*weight
                    counter['active_component_SIMD8_arithmetic_slots']+=int(live_t.sum())
                    counter['active_scalar_products']+=int(np.count_nonzero((act!=0)&(weight[None,:]!=0)))
        counter['accumulator_write_SR64_words']+=tokens*16*6//8 # physical signed48
        out[p0:p0+2]=acc.reshape(-1,10,16)
    gold=x@f['reconstructed'][:,inverse].T
    assert np.array_equal(out,gold),'packed executor mismatch'
    return out,dict(counter)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--capture',type=Path,default=CAP);ap.add_argument('--output',type=Path,default=HERE);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    q=load(args.capture/'deployed_constants.npz');w=q['U_conv2_theta_q16'].astype(np.int64)
    datasets=[]
    for p in sorted(args.capture.glob('[0-9][0-9][0-9]_*.npz')):
        d=load(p)
        for label in ('corner','interior'):
            row=samples(d,label);row['file']=p.name;datasets.append(row)
    train=np.concatenate([r['x'].reshape(-1,864) for r in datasets[:2]])
    result=dict(scope=__doc__,module='r1 conv2 fixed U16: actual matched dense stage320, K864',
                calibration_files=[datasets[0]['file']],holdout_files=sorted(set(r['file'] for r in datasets[2:])),
                NB0_diverse10_gate=1.45460286107,NB0_valid825_frame_mean=1.44535253468097,AEE_measured=False,
                no_training=True,fit='All supports, identical local Gram ridge=1e-4 mean diagonal, nearest int16, unchanged original exponent; no global full-HiNM claim.',
                resource='40 SIMD8 signed48 accumulators; 2x32B coefficient latches; 16B packet latch; 64B quartet source gather latch; source and coefficient accesses counted separately; no cycle/area model',
                axes={})
    for r in datasets:
        ref=downstream(r['x']@w.T,r['raw'],q)
        check=[int(np.count_nonzero(a!=b)) for a,b in zip(ref,r['gold'])]
        assert check==[0,0,0],(r['file'],r['label'],check)
    result['unmodified_baseline_capture_mismatches']=0
    for layout in ('natural','joint_k16'):
        perm=permutation(w,train,layout);ww=w[:,perm];xx=train[:,perm]
        entry={};result['axes'][layout]=entry
        for mode in ('dense','ordinary24','ordinary34','shifted24','shifted14'):
            f=fit(ww,xx,mode);vb,mb=encode(f)
            effective=f['reconstructed'][:,np.argsort(perm)].astype(np.int16)
            np.savez_compressed(args.output/f'{layout}_{mode}.npz',U_conv2_theta_q16=effective,
                U_conv2_theta_exponent=q['U_conv2_theta_exponent'],permutation=perm.astype(np.uint16),
                values=f['values'],indices=f['indices'],offset=np.array(f['offset']),
                value_payload=np.frombuffer(vb,np.uint8),metadata_payload=np.frombuffer(mb,np.uint8))
            rec=dict(coefficient_bytes=len(vb)+len(mb),dense_int16_bytes=w.nbytes//4,
                     effective_weight_nnz=int(np.count_nonzero(effective)),logical_value_count=f['values'].size,negative_components=int(np.count_nonzero(f['values']<0)),
                     opposite_sign_base_residual_pairs=int(np.count_nonzero(f['values'][:,:,1:].astype(np.int64)*f['values'][:,:,:1].astype(np.int64)<0)) if f['offset'] else 0,
                     mean_removed_weight_relative_sq_error=float(np.sum((effective-w)**2)/np.sum(w*w)),samples=[])
            entry[mode]=rec
            for r in datasets:
                acc,counts=execute(f,r['x'],r['addresses'],perm,r['source_blob'])
                value=downstream(acc,r['raw'],q)
                rec['samples'].append(dict(file=r['file'],window=r['label'],holdout=r['file']!=datasets[0]['file'],
                                          metrics=metrics(value,r['gold']),counts=counts,
                                          packed_vs_reconstructed_acc_mismatches=0,
                                          signed48_acc_maxabs=int(np.max(np.abs(acc)))))
            for split,held in [('calibration',False),('holdout',True)]:
                chosen=[s for s in rec['samples'] if s['holdout']==held]
                rec[split]=dict(PED_relative_squared_error_mean=float(np.mean([s['metrics']['PED_relative_squared_error'] for s in chosen])),
                    gate_flip_fraction=float(sum(s['metrics']['gate_flips'] for s in chosen)/sum(s['metrics']['gate_count'] for s in chosen)),
                    request_and_instruction_totals={k:sum(s['counts'].get(k,0) for s in chosen) for k in chosen[0]['counts']})
            print(layout,mode,rec['holdout'],flush=True)
            (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    result['complete']=True
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
