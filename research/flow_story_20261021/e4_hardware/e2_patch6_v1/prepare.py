#!/usr/bin/env python3
"""Derive a versioned finite-supply extension from frozen E4 and E2 geometry."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent
E4=HERE.parent
CONTRACT=E4.parent/'e2_group_damage/PATCH6_CONTRACT_V1.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(CONTRACT)=='a6926959cd24167e73fff38c1f7bb0024955cf9c035a50fd10ed081703bb862a'
c=json.loads(CONTRACT.read_text());targets=c['family_targets'];assert len(targets)==4
lookup=[6]*910
for g in c['groups']:
 assert all(g['tiles'][x]==g['tiles'][targets[0]] for x in targets)
 assert len(g['tiles'][targets[0]])==151
 for y,x in g['tiles'][targets[0]]:
  assert 0<=y<26 and 0<=x<35 and lookup[y*35+x]==6
  lookup[y*35+x]=g['id']
assert lookup.count(6)==4
header='// Frozen E2 group IDs: one byte per complete output tile; group6 = leftover.\n'
header+='static constexpr unsigned char tile_group[910]={'+','.join(map(str,lookup))+'};\n'
(HERE/'tile_groups.h').write_text(header)
s=(E4/'supply_replay.cpp').read_text()
s=s.replace('#include <vector>','#include <vector>\n#include "tile_groups.h"')
s=s.replace('struct Simulator{','struct Simulator{\n int pair_a=-1,pair_b=-1,control=0;U descriptor_reads=0,descriptor_ticks=0,policy_setup_ticks=0;')
s=s.replace('Simulator(Geometry gg,bool p,int q):g(gg),parallel(p),precision(q){','Simulator(Geometry gg,bool p,int q,int a=-1,int b=-1,int ctl=0):g(gg),parallel(p),precision(q),pair_a(a),pair_b(b),control(ctl){')
# Initialization order corrected to declaration order: no behavior change.
s=s.replace(':g(gg),parallel(p),precision(q),pair_a(a),pair_b(b),control(ctl)', ':pair_a(a),pair_b(b),control(ctl),g(gg),parallel(p),precision(q)')
needle='int y=int(r/g.w)%g.h,x=int(r%g.w);int bits=(y<(g.h/9)*9&&x<(g.w/9)*9)?precision:8;'
replacement='''int y=int(r/g.w)%g.h,x=int(r%g.w);bool complete=y<(g.h/9)*9&&x<(g.w/9)*9;int bits=complete?precision:8;
    if(complete&&pair_a>=0){
     int tile=(y/9)*35+x/9,group=tile_group[tile];need(tile>=0&&tile<910,"Actual complete tile descriptor address");
     if(group==pair_a||group==pair_b)bits=4;
     if(control&&((x%9)==0||r==first)){
      // Fixed stats-RF group/precision table: one32b bank read per tile span.
      // control1: precompiled precision lookup; control2: group-ID lookup then
      // 6-bit membership decode. Serialized to avoid pretending free logic.
      descriptor_reads++;descriptor_ticks+=control;U a=now;now+=control;
      event("STATS_RF","tile_descriptor",a,now,U(tile/4)*4,4,0);
     }
    }'''
assert s.count(needle)==1;s=s.replace(needle,replacement)
s=s.replace('initial_ready=dma;', '''if(control){
   need(910+16+12*U(g.cout)*4<=262144,"Table + selectors coexist withBN stats RF");
   transfer("tile_descriptor_table",dma,0,910,0);
   if(control==2){policy_setup_ticks=3;pe=dma+policy_setup_ticks;}
  }
  initial_ready=std::max(dma,pe);''')
s=s.replace('std::cout<<"{\\\"name\\\":\\\""<<g.name', 'std::cout<<"{\\\"pair\\\":["<<pair_a<<","<<pair_b<<"],\\\"control\\\":"<<control<<",\\\"descriptor_word_reads\\\":"<<descriptor_reads<<",\\\"descriptor_read_bytes\\\":"<<descriptor_reads*4<<",\\\"descriptor_control_ticks\\\":"<<descriptor_ticks<<",\\\"pair_mask_setup_ticks\\\":"<<policy_setup_ticks<<",\\\"name\\\":\\\""<<g.name')
start=s.index('int main()')
s=s[:start]+'''int main(){try{
 Geometry g={"patch.res0.conv0",240,320,96,96,true};
 std::cout<<"{\\\"schema\\\":\\\"flow_e2_patch6_finite_supply_v1\\\",\\\"status\\\":\\\"COMPLETE_OPERATOR_MODEL_NOT_NETWORK_RTL\\\",\\\"results\\\":[";
 Simulator(g,false,6).run();
 for(int a=0;a<6;a++)for(int b=a+1;b<6;b++)for(int ctl=0;ctl<3;ctl++){
  std::cout<<",";Simulator(g,false,6,a,b,ctl).run();}
 std::cout<<"],\\\"identical_geometry_and_tile_mapping_targets\\\":4,\\\"GPU_used\\\":false,\\\"GT_read\\\":false,\\\"predictor_implemented\\\":false,\\\"whole_network\\\":false}\\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\\n";return 1;}}
'''
(HERE/'patch6_supply.cpp').write_text(s)
(HERE/'PREPARE_RECEIPT_V1.json').write_text(json.dumps({'schema':'flow_e2_supply_source_derivation_v1','contract_sha256':sha(CONTRACT),'frozen_E4_cpp_sha256':sha(E4/'supply_replay.cpp'),'derived_cpp_sha256':sha(HERE/'patch6_supply.cpp'),'tile_header_sha256':sha(HERE/'tile_groups.h'),'all_four_target_geometry_and_groups_identical':True,'group_sizes':[lookup.count(i) for i in range(7)],'new_GPU_forwards':0,'GT_read':False},indent=2)+'\n')
print('Prepared46 complete single-conv operator replays; four targets share the same mapping.')
