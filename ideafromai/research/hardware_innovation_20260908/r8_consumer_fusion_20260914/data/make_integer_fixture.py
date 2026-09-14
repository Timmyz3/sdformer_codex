"""Freeze exactly one hardware consumer format and integer fixture."""
import json
import numpy as np
from model_access import HERE

def rne(x,shift):
    q=x//(1<<shift);r=x-q*(1<<shift)
    return q+((r>(1<<(shift-1)))|((r==(1<<(shift-1)))&((q&1)!=0)))

def main():
    z=np.load(HERE/'consumer_first8.npz');c=np.load(HERE/'consumer_coefficients.npz')
    a=c['a_q40'].astype(np.int64);b=c['b_q20'].astype(np.int64)
    raw=z['p_int'].astype(np.int64)
    j=np.clip(np.rint(z['identity_fp32'].astype(np.float64)*2**20),-2**31,2**31-1).astype(np.int64)
    wide=raw*a[None,None,:,None,None]+((j+b[None,None,:,None,None])<<20)
    new=np.clip(rne(wide,26),-2**23,2**23-1).astype(np.int32)
    np.savez_compressed(HERE/'consumer_integer_first8.npz',source_bits=z['source_bits'],q1=z['q1'],q2=z['q2'],
        p_int=raw.astype(np.int32),identity_q20=j.astype(np.int32),a_q40=a.astype(np.int32),b_q20=b.astype(np.int32),
        wide_int64=wide,i24_new=new,i24_original=z['I24'],output_origin_yx=z['output_origin_yx'],input_origin_yx=z['input_origin_yx'])
    meta=dict(complete=True,format_frozen=True,a_fraction=40,b_fraction=20,identity_fraction=20,final_shift=26,I24_fraction=14,
        a_range=[int(a.min()),int(a.max())],b_range=[int(b.min()),int(b.max())],identity_q20_range=[int(j.min()),int(j.max())],
        wide_range=[int(wide.min()),int(wide.max())],I24_range=[int(new.min()),int(new.max())],
        different_from_original_I24=int((new!=z['I24']).sum()),elements=int(new.size),max_abs_delta=int(np.abs(new.astype(np.int64)-z['I24']).max()),
        p_abs_bound=679477248,wide_abs_bound_for_all_J32=int(679477248*np.abs(a).max()+((2**31+np.abs(b).max())<<20)),
        gold='consumer_integer_first8.npz',gold_arithmetic='numpy int64 product/add; floorquotient tie-even signed rounding; signed24 saturation')
    assert meta['wide_abs_bound_for_all_J32']<2**63
    (HERE/'consumer_integer_definition.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
