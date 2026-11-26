"""Root independent vectorized geometry/actual-report audit, no result rewriting."""
import hashlib,json,re
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;H=HERE.parent/'hardware'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    sources={};summary=read(H/'results/SUMMARY_V1.json');raw=read(H/'results/SUPPLY_REPLAY_V1.json')['results']
    words=np.asarray(read(H/'BITMAP_CONTENT_V1.json')['words'],dtype=np.uint32)
    assert len(words)==192 and len(raw)==14
    rows=np.arange(10*240*320,dtype=np.int64);y=(rows//320)%240;x=rows%320
    complete=(y<234)&(x<315);tile=np.where(complete,y//9*35+x//9,910)
    start=(rows%128)==0;checks=[]
    for action in range(7):
        key=np.where(complete,911,tile) if action==6 else tile
        changed=start|(key!=np.roll(key,1));full=changed&complete&(action<6);bypass=changed&~full
        fullcount=int(full.sum())*6;bypasscount=int(bypass.sum())*6
        mask=np.zeros(1024,dtype=bool)
        if action<6:
            i=np.arange(1024);mask=((words[action*32+i//32]>>(i%32))&1).astype(bool)
            assert np.array_equal(mask,(i>=action*151)&(i<(action+1)*151))
        high=(~complete)|(mask[tile]&complete);n8=int(high.sum());n4=len(rows)-n8
        hist=np.bincount((tile[full]//32),minlength=32)*6
        for dyn in (False,True):
            a=next(r for r in raw if r['selected_action']==action and r['dynamic_selection']==dyn)
            assert (a['q8_rows'],a['q4_rows'])==(n8,n4)
            assert (a['full_bitmap_queries'],a['registered_bypass_queries'])==(fullcount,bypasscount)
            assert a['query_service_ticks']==3*fullcount+bypasscount
            expected=np.zeros(192,dtype=np.int64)
            if action<6:expected[(32*action if dyn else 0):(32*action if dyn else 0)+32]=hist
            assert np.array_equal(a['bitmap_word_read_histogram'],expected)
            assert a['arithmetic_accepted_beats']==(n8*8+n4*4)*7*6
            assert a['descriptor_RF_read_bytes']==fullcount*4
            checks.append(dict(action=action,dynamic=dyn,Q8_rows=n8,RNE4_rows=n4,full_queries=fullcount,bypass_queries=bypasscount,status='PASS'))
        c=summary['candidates'][action]
        assert c['dynamic_affine_BN_ticks']-c['static_affine_BN_ticks']==(640//64+640//4 if action<6 else 0)
    actualreports={}
    for kind in ('dynamic','static','none'):
        rdir=H/'results'/('dc_'+kind);dc=read(rdir/'summary.json');area=(rdir/'area.rpt').read_text();sdc=(rdir/'constraints.sdc').read_text();timing=(rdir/'timing.rpt').read_text();log=(rdir/'run.log').read_text()
        assert float(re.search(r'Total cell area:\s+([0-9.]+)',area).group(1))==dc['area_library_units']
        slacks=[float(v) for v in re.findall(r'slack \(MET\)\s+([0-9.]+)',timing)]
        assert slacks and abs(min(slacks)-dc['worst_setup_slack_ns'])<=0.0051 # report prints two decimal places
        exact=float(re.search(r'FLOW_BITMAP_CONTROL_SYNTH_DONE top=\S+ area=\S+ slack=([0-9.]+)',log).group(1))
        assert exact==dc['worst_setup_slack_ns']
        assert '-period 4 ' in sdc and 'FLOW_BITMAP_CONTROL_SYNTH_DONE' in log and 'Error:' not in log
        rtl=read(H/'results'/('rtl_'+kind)/'RESULT.json');assert rtl['status']=='PASS' and rtl['all910_tiles'] and rtl['no_stall_bypass_span_ticks']==1
        if kind!='none':assert rtl['no_stall_active_span_ticks']==3
        actualreports[kind]=dict(area_library_units=dc['area_library_units'],slack_ns=dc['worst_setup_slack_ns'],clock_ns=4,RTL_queries=rtl['queries'])
        for n in ('area.rpt','timing.rpt','constraints.sdc','run.log','summary.json'):sources[str((rdir/n).relative_to(H))]=sha(rdir/n)
    for p in ('rtl/bitmap_controller.sv','rtl/tb_bitmap.cpp','rtl/synth_ctrl.tcl','bitmap_supply.cpp','results/SUMMARY_V1.json','results/SUPPLY_REPLAY_V1.json','BITMAP_CONTENT_V1.json'):sources[p]=sha(H/p)
    out=dict(schema='flow_r33_root_hardware_independent_audit_v1',status='PASS_WITH_ISOLATED_SCOPE',sources=sources,vectorized_geometry_checks=checks,actual_reports=actualreports,
        dynamic_minus_same_static_per_conv_ticks=170,four_independent_convs_extra_init_ticks=680,
        interpretation='3/1 are request-accept to response-accept latency, not throughput. PE work separates span queries. Address/span generator and RF are modeled externally, not part of synthesized controller.',
        exclusions=['predictor feature/readout/storage','RF/SRAM macros','PE and scale/BN datapath','complete network','postlayout timing/power'],new_GPU_calls=0,full_network_speedup_claim=False)
    dest=HERE/'HARDWARE_ROOT_INDEPENDENT_AUDIT_V1.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print(out['status'])
if __name__=='__main__':main()
