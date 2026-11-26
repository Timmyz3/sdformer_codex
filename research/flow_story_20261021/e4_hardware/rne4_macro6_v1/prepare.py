#!/usr/bin/env python3
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;R=P.parents[2]
SRC=P.parent/'rne4_supply_v1/rne4_supply.cpp';C=R/'flow_story_20261021/e3_rne_probe/PREPARED_CONTRACT_V1.json'
d=json.loads(C.read_text()); targets=d['family_targets'];ids=[6]*910
for i,g in enumerate(d['groups']):
 assert g['id']==i and g['added_core_ticks_vs_Q4']==82192320
 ref=g['tiles'][targets[0]]; assert len(ref)==151
 for name in targets:assert g['tiles'][name]==ref
 for y,x in ref:
  idx=y*35+x;assert ids[idx]==6;ids[idx]=i
assert [ids.count(i) for i in range(7)]==[151]*6+[4]
(P/'macro_groups.h').write_text('static const int group_ids[910]={'+','.join(map(str,ids))+'};\n')
s=SRC.read_text().replace('#include "rne_masks.h"','#include "macro_groups.h"')
s=s.replace('int mode;','int mode;int macro;bool dynamic;U decode_ticks=0,selection_latch_ticks=0;')
s=s.replace('Simulator(Geometry gg,int arm):g(gg),parallel(arm==0),precision(arm<2?8:4),mode(arm){','Simulator(Geometry gg,int arm,int selected,bool dyn):g(gg),parallel(arm==0),precision(arm<2?8:4),mode(arm),macro(selected),dynamic(dyn){\n  need((mode==3&&macro==-1&&!dynamic)||(mode==4&&macro>=0&&macro<6),"legal selected macro0..5 in3bits");')
s=s.replace('return selected_high(g.h,g.w,y,x)?8:4;','return group_ids[(y/9)*35+x/9]==macro?8:4;')
s=s.replace('event("STATS_RF","fixed_M_descriptor",a,now,(id/4)*4,4,slot);previous=id;', 'event("STATS_RF",dynamic?"groupID_descriptor":"static_plane_descriptor",a,now,(id/4)*4,4,slot);if(dynamic){now++;decode_ticks++;}previous=id;')
s=s.replace('  if(resident)transfer(', '  if(dynamic){transfer("selected_macro3bits",0,1024,1,0);dma++;selection_latch_ticks++;}\n  if(resident)transfer(')
s=s.replace('"M_fixed_RNE4_Q8_dual"','"one_macro_RNE4_Q8_dual"')
s=s.replace('std::cout<<"{\\"arm\\":\\""<<names[mode]', 'std::cout<<"{\\"selected_macro\\":"<<macro<<",\\"dynamic_selection\\":"<<(dynamic?"true":"false")<<",\\"group_decode_ticks\\":"<<decode_ticks<<",\\"selection_latch_ticks\\":"<<selection_latch_ticks<<",\\"arm\\":\\""<<names[mode]')
s=s[:s.index('int main()')]+r'''int main(){try{
 Geometry g={"patch.res0.conv0",240,320,96,96,true};
 std::cout<<"{\"schema\":\"flow_e4_rne4_macro6_finite_replay_v1\",\"status\":\"MODEL_PREPARED_CONTRACT_NOT_ENDPOINT_BRIDGE\",\"results\":[";
 Simulator(g,3,-1,false).run();
 for(int m=0;m<6;m++)for(int d=0;d<2;d++){std::cout<<",";Simulator(g,4,m,d!=0).run();}
 std::cout<<"],\"GPU_used\":false,\"GT_read\":false,\"numerical_BN_verified\":false,\"whole_network\":false}\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
'''
(P/'macro6_supply.cpp').write_text(s)
(P/'SOURCE_BINDINGS_V1.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SRC,C,P.parent/'rne4_supply_v1/FREEZE_V1.json',R/'flow_story_20261021/e3_rne_probe/probe_contract.py']},indent=2)+'\n')
