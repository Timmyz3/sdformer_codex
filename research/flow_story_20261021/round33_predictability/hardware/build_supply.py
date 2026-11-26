#!/usr/bin/env python3
from pathlib import Path
import json
P=Path(__file__).resolve().parent;R=P.parents[2]
# Actual measured controller service, not chosen to improve the supply result.
for kind in ['dynamic','static','none']:
 d=json.loads((P/'results'/('rtl_'+kind)/'RESULT.json').read_text());assert d['status']=='PASS' and d['no_stall_bypass_span_ticks']==1
 if kind!='none':assert d['no_stall_active_span_ticks']==3
words=json.loads((P/'BITMAP_CONTENT_V1.json').read_text())['words']
(P/'bitmap_words.h').write_text('static const uint32_t bitmap_words[192]={'+','.join(str(x)+'u' for x in words)+'};\n')
s=(R/'flow_story_20261021/e4_hardware/rne4_supply_v1/rne4_supply.cpp').read_text().replace('#include "rne_masks.h"','#include "bitmap_words.h"')
s=s.replace('int mode;','int mode;int action;bool dynamic;U full_queries=0,bypass_queries=0,query_service_ticks=0,descriptor_init_writes=0;std::array<U,192> bitmap_word_hits{};')
s=s.replace('Simulator(Geometry gg,int arm):g(gg),parallel(arm==0),precision(arm<2?8:4),mode(arm){','Simulator(Geometry gg,int selected,bool dyn):g(gg),parallel(false),precision(4),mode(selected==6?3:4),action(selected),dynamic(dyn){\n  need(action>=0&&action<=6,"action0..5 or none6");')
s=s.replace('return selected_high(g.h,g.w,y,x)?8:4;','U id=U(y/9)*35+x/9;return ((bitmap_words[action*32+id/32]>>(id%32))&1u)?8:4;')
a=s.index('  auto desc=');b=s.index('  std::array<std::vector<U>,2> rows;',a)
s=s[:a]+r'''  auto desc=[&](U r,U& previous){
   int y=int(r/g.w)%g.h,x=int(r%g.w);bool complete=y<(g.h/9)*9&&x<(g.w/9)*9;
   U id=complete?U(y/9)*(g.w/9)+x/9:910;
   // Cache only one contiguous span. None has a constant complete-region mode.
   U key=complete&&action==6?911:id;
   if(key==previous)return;
   U a=now;
   if(complete&&action<6){
    U address=(dynamic?U(action)*128:0)+(id/32)*4;
    need(address%4==0&&address+4<=descriptor_storage&&address+4<=2048,"bitmap32bit1RW address bounds");
    uint32_t data=bitmap_words[action*32+id/32];int mode_bits=((data>>(id%32))&1u)?8:4;
    need(mode_bits==row_bits(r),"controller output/row representation agreement");
    full_queries++;bitmap_word_hits[address/4]++;descriptor_reads+=4;now+=3;query_service_ticks+=3;descriptor_ticks+=3;
    c.hash=(c.hash^address^data)*1099511628211ULL;
    event("STATS_RF","bitmap_request_read_extract_response",a,now,address,4,slot);
   }else{
    bypass_queries++;now++;query_service_ticks++;descriptor_ticks++;
    need(row_bits(r)==(complete?4:8),"none/tail registered bypass mode");
    event("CONTROL","registered_none_or_tail_response",a,now,0,0,slot);
   }
   previous=key;
  };
''' + s[b:]
s=s.replace('    if(mode>=3&&bits!=previous_bits){now++;control_ticks++;previous_bits=bits;}','    // Query response latches the mode; no duplicate precision-switch tick.\n    previous_bits=bits;')
s=s.replace('int previous_bits=-1;U previous_desc=~U(0);','U previous_desc=~U(0);').replace('    previous_bits=bits;','')
a=s.index('  if(mode==4){descriptor_storage=');b=s.index('  if(resident)',a)
s=s[:a]+r'''  if(action<6){
   descriptor_storage=dynamic?768:128;need(descriptor_storage<=2048,"existing2KiB descriptor reservation");
   transfer(dynamic?"six_bitmaps_load":"one_fixed_bitmap_load",0,0,descriptor_storage,0);
   U a=dma;dma+=descriptor_storage/4;descriptor_init_writes=descriptor_storage/4;descriptor_ticks+=descriptor_init_writes;
   event("STATS_RF","bitmap_init32bit1RW",a,dma,0,descriptor_storage,0);
  }
''' +s[b:]
s=s.replace('  initial_ready=dma;','  initial_ready=dma+1; // start handshake; selector is an interface, not a predictor')
s=s.replace('  matrix_done=std::max(pe,dma);','  matrix_done=std::max(pe,dma)+1; // finish handshake after all affine retirement')
s=s.replace('  const char* names[]={"Q8_parallel_single","Q8_serial_single","Q4_single_allrows_COMPONENT_ONLY","Q4_complete_Q8_tail_dual","M_fixed_RNE4_Q8_dual"};','  const char* arm=action==6?(dynamic?"dynamic_none":"static_none"):(dynamic?"dynamic_bitmap":"static_bitmap");')
s=s.replace('std::cout<<"{\\"arm\\":\\""<<names[mode]', 'std::cout<<"{\\"selected_action\\":"<<action<<",\\"dynamic_selection\\":"<<(dynamic?"true":"false")<<",\\"full_bitmap_queries\\":"<<full_queries<<",\\"registered_bypass_queries\\":"<<bypass_queries<<",\\"query_service_ticks\\":"<<query_service_ticks<<",\\"descriptor_init32bit_writes\\":"<<descriptor_init_writes<<",\\"boundary_handshake_ticks\\":2,\\"predictor_ticks_included\\":false,\\"arm\\":\\""<<arm')
s=s[:s.index('int main()')]+r'''int main(){try{
 Geometry g={"patch.res0.conv0",240,320,96,96,true};
 std::cout<<"{\"schema\":\"flow_r33_precomputed_bitmap_supply_v1\",\"status\":\"OPERATOR_MODEL_RTL_CONTROL_PROTOCOL_NOT_NETWORK_RTL\",\"results\":[";
 bool first=true;for(int action=0;action<=6;action++)for(int dyn=0;dyn<2;dyn++){
  if(!first){std::cout<<",";}
  first=false;Simulator(g,action,dyn!=0).run();
 }
 std::cout<<"],\"GPU_used\":false,\"GT_read\":false,\"numerical_BN_verified\":false,\"whole_network\":false}\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
'''
s=s.replace('One explicit geometry/mode descriptor cycle.','One explicit row-gather geometry cycle; mode lookup is separate.')
s=s.replace('  std::cout<<"{\\"selected_action\\":', '  std::cout<<"{\\"bitmap_word_read_histogram\\":[";for(int wi=0;wi<192;wi++){if(wi)std::cout<<",";std::cout<<bitmap_word_hits[wi];}std::cout<<"],\\"selected_action\\":')
(P/'bitmap_supply.cpp').write_text(s)
