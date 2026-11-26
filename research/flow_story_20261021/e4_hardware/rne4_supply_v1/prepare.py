#!/usr/bin/env python3
"""Generate isolated RNE supply model from frozen E4; never mutate its source."""
from pathlib import Path
import json, hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=HERE.parent/'supply_replay.cpp'
MASK=ROOT/'flow_story_20261020/round30/hardware/MASK_CONTRACT_V1.json'
d=json.loads(MASK.read_text()); geos=d['geometries']['full9']; tiles=d['evaluation_tiles']
if 'full9' in tiles: tiles=tiles['full9']
keys=list(geos)
selected=[]
for h,w in [(240,320),(30,40),(120,160)]:
 k=next(k for k,v in geos.items() if v['H']==h and v['W']==w)
 t=tiles[k]
 selected.append((h,w,t))
with (HERE/'rne_masks.h').open('w') as f:
 f.write('// Exact frozen evaluation mask; one mask reused by all same-geometry operator cases.\n')
 f.write('static bool selected_high(int h,int w,int y,int x){\n')
 for h,w,t in selected:
  f.write(f'if(h=={h}&&w=={w}){{ static const int ids[]={{'+','.join(str(int(a)*(w//9)+int(b)) for a,b in t)+'}; int id=(y/9)*(w/9)+x/9; for(int z:ids)if(z==id)return true; return false;}\n')
 f.write('throw std::runtime_error("unknown mask geometry");}\n')
s=OLD.read_text()
s=s.replace('struct Geometry{','#include "rne_masks.h"\nstruct Geometry{')
s=s.replace(' Geometry g;bool parallel;int precision;',' Geometry g;bool parallel;int precision;int mode;\n U q8rows=0,q4rows=0,scale_rf_reads=0,scale_sram_reads=0,descriptor_reads=0,descriptor_ticks=0,control_ticks=0,index_read_bytes=0,index_write_bytes=0,index_ticks=0,output_store_ticks=0,high_storage=0,low_storage=0,descriptor_storage=0,scale_fill_ticks=0; bool grouped=false;\n')
s=s.replace('Simulator(Geometry gg,bool p,int q):g(gg),parallel(p),precision(q){','Simulator(Geometry gg,int arm):g(gg),parallel(arm==0),precision(arm<2?8:4),mode(arm){')
s=s.replace('resident=g.rows()*g.cin/8<=RES;', 'U chunks=ceildiv(g.k(),parallel?16:128); grouped=(mode>=3 && (ceildiv(chunks*8*256+64,256)+ceildiv(chunks*4*256+64,256))*256>WRF);\n  resident=g.rows()*g.cin/8<=RES;')
s=s.replace('peak_in=std::max(peak_in,packed+halo);need(packed+halo<=IOS,','peak_in=std::max(peak_in,packed+halo+(grouped?count*4:0));need(packed+halo+(grouped?count*4:0)<=IOS,')
start=s.index(' U matrix('); end=s.index(' U bn_phase(',start)
s=s[:start]+r'''
 int row_bits(U r)const{
  if(mode<2)return 8;
  if(mode==2)return 4; // component reference, NOT actual E0 Q4 true-tail semantics
  int y=int(r/g.w)%g.h,x=int(r%g.w);
  if(y>=(g.h/9)*9||x>=(g.w/9)*9)return 8;
  if(mode==3)return 4;
  return selected_high(g.h,g.w,y,x)?8:4;
 }
 U matrix(U first,U count,U ready,int slot){
  U packed_base=source_bytes(first,count),now=ready,padk=ceildiv(g.k(),parallel?16:128)*(parallel?16:128),chunks=ceildiv(g.k(),parallel?16:128);
  auto wb=[&](int bits){return chunks*U(parallel?1:bits)*256;};
  auto alloc=[&](int bits){return ceildiv(wb(bits)+64,256)*256;};
  U high_rf=0,low_rf=mode<2?0:(mode==2||grouped?0:alloc(8));
  U rf_used=mode<2?alloc(8):mode==2?alloc(4):grouped?std::max(alloc(8),alloc(4)):alloc(8)+alloc(4);
  peak_rf=std::max(peak_rf,rf_used);need(rf_used<=WRF,"RNE representations and scales fit32KiB RF");
  auto prefill=[&](int ob,int bits){
   U weightbytes=wb(bits),srcbase=(bits==8?0:high_storage)+U(ob)*(weightbytes+64),rfbase=(bits==8?high_rf:low_rf);
   need(srcbase+weightbytes+64<=high_storage+low_storage,"variant source SRAM range");
   U a=now;now+=ceildiv(weightbytes,256);c.rf_fill_ticks+=now-a;c.weight_sram_reads+=weightbytes;
   event("WEIGHT_SRAM",bits==8?"Q8_RF_refill":"RNE4_RF_refill",a,now,srcbase,weightbytes,slot);
   a=now;now++;scale_fill_ticks++;scale_sram_reads+=64;
   event("WEIGHT_SRAM","scale16_RF_refill",a,now,srcbase+weightbytes,64,slot);
   need(rfbase+weightbytes+64<=WRF,"scale RF bounds");
  };
  auto desc=[&](U r,U& previous){
   int y=int(r/g.w)%g.h,x=int(r%g.w);
   if(mode!=4||y>=(g.h/9)*9||x>=(g.w/9)*9){previous=~U(0);return;}
   // Reuse one byte descriptor only through a contiguous horizontal tile span.
   U id=U(y/9)*(g.w/9)+x/9;
   if(id!=previous){U a=now;now++;descriptor_reads+=4;descriptor_ticks++;need((id/4)*4+4<=ceildiv(descriptor_storage,4)*4,"descriptor address");event("STATS_RF","fixed_M_descriptor",a,now,(id/4)*4,4,slot);previous=id;}
  };
  std::array<std::vector<U>,2> rows;std::vector<U> index_contents;
  if(grouped){
   U previous=~U(0),indexbase=packed_base+count*ceildiv(padk,8);
   for(U r=first;r<first+count;r++){
    desc(r,previous);int bits=row_bits(r);rows[bits==8?0:1].push_back(r);
    now++;control_ticks++; // classify row; no future predictor is assumed
   }
   U pos=0;
   for(int v=0;v<2;v++)for(U r:rows[v]){
    now=avoid(now,input_dma_busy,c.input_dma_block_ticks);
    U addr=indexbase+(pos++)*4;need(addr+4<=IOS,"row partition index write range");index_contents.push_back(r);
    now++;index_ticks++;index_write_bytes+=4;c.hash=(c.hash^addr^r)*1099511628211ULL;
   }
  }
  for(int ob=0;ob<g.cout/16;ob++){
   U section_start=now;
   if(!grouped){if(mode<2)prefill(ob,8);else if(mode==2)prefill(ob,4);else{prefill(ob,8);prefill(ob,4);}}
   int previous_bits=-1;U previous_desc=~U(0);
   auto processrow=[&](U r,U listpos){
    int bits=row_bits(r);if(grouped){
     now=avoid(now,input_dma_busy,c.input_dma_block_ticks);
     U addr=packed_base+count*ceildiv(padk,8)+listpos*4;
     need(addr+4<=IOS,"row index read address");need(listpos<index_contents.size()&&index_contents[listpos]==r,"row index data/canonical scatter identity");now++;index_ticks++;index_read_bytes+=4;c.hash=(c.hash^addr^r)*1099511628211ULL;
    }else desc(r,previous_desc);
    if(mode>=3&&bits!=previous_bits){now++;control_ticks++;previous_bits=bits;}
    U blockbytes=wb(bits),rfbase=(bits==8?high_rf:low_rf),rowbeats=parallel?chunks:chunks*bits;
    for(U b=0;b<rowbeats;b++){
     U rfaddr=rfbase+b*256;need(rfaddr%256==0&&rfaddr+256<=rfbase+blockbytes,"weight RF alignment range");
     U chunk=parallel?b:b%chunks,actaddr=packed_base+(r-first)*ceildiv(padk,8)+chunk*(parallel?2:16);
     need(actaddr+U(parallel?2:16)<=packed_base+count*ceildiv(padk,8),"activation slot bounds");
     now=avoid(now,input_dma_busy,c.input_dma_block_ticks);
     if(sampled[bits]<16){beats_trace.push_back({now,r,rfaddr,actaddr,ob,bits,parallel?-1:int(8-bits+b/chunks),int(chunk)});sampled[bits]++;}
     c.hash=(c.hash^rfaddr^(actaddr<<1)^U(ob))*1099511628211ULL;now++;c.beats++;
     c.weight_rf_reads+=256;c.input_rf_reads+=parallel?4:16;
    }
    // One64B scale RF read serialized against weight access, then convert/multiply.
    U scaleaddr=rfbase+blockbytes;need(scaleaddr+64<=WRF&&scaleaddr%64==0,"scale16 RF range/alignment");
    now++;scale_rf_reads+=64;now+=2;c.scale_ticks+=2;
    now=avoid(now,output_dma_busy,c.output_dma_block_ticks);
    U outaddr=((r-first)*g.cout+ob*16)*4;need(outaddr%64==0&&outaddr+64<=IOS,"canonical output scatter range");
    now++;output_store_ticks++;
   };
   if(grouped){U listpos=0;for(int v=0;v<2;v++)if(!rows[v].empty()){prefill(ob,v==0?8:4);for(U r:rows[v])processrow(r,listpos++);}}
   else for(U r=first;r<first+count;r++)processrow(r,0);
   event("PE","variant_integer_scale_scatter",section_start,now,0,rf_used,slot);
  }
  for(U r=first;r<first+count;r++){if(row_bits(r)==8)q8rows++;else q4rows++;}
  peak_out=std::max(peak_out,count*g.cout*4);return now;
 }
''' +s[end:]
start=s.index('  U padk=',s.index(' void run()'));end=s.index('  if(resident)',start)
s=s[:start]+r'''  U padk=ceildiv(g.k(),parallel?16:128)*(parallel?16:128);
  high_storage=mode==2?0:padk*g.cout+4*g.cout;
  low_storage=mode<2?0:padk*g.cout/2+4*g.cout;
  need(high_storage+low_storage<=WSLOT,"independent weights and scales fit1MiB weight SRAM slot");
  if(high_storage)transfer("Q8_weights_and_scales",0,0,high_storage,0);
  if(low_storage)transfer("RNE4_weights_and_scales",0,high_storage,low_storage,0);
  if(mode==4){descriptor_storage=U(g.h/9)*(g.w/9);need(descriptor_storage<=2048,"fixed M descriptor reserved stats RF");transfer("fixed_M_descriptor",0,0,descriptor_storage,0);U a=dma;dma+=ceildiv(descriptor_storage,64);descriptor_ticks+=dma-a;event("STATS_RF","descriptor_refill",a,dma,0,descriptor_storage,0);}
''' +s[end:]
# Insert new metadata ahead of existing JSON fields.
s=s.replace('  std::cout<<"{\\"name\\":\\""<<g.name',r'''  const char* names[]={"Q8_parallel_single","Q8_serial_single","Q4_single_allrows_COMPONENT_ONLY","Q4_complete_Q8_tail_dual","M_fixed_RNE4_Q8_dual"};
  std::cout<<"{\"arm\":\""<<names[mode]<<"\",\"row_schedule\":\""<<(grouped?"batch_local_two_variant_partition":"natural_row_order")<<"\",\"q8_rows\":"<<q8rows<<",\"q4_rows\":"<<q4rows<<",\"Q8_weight_scale_storage_bytes\":"<<high_storage<<",\"RNE4_weight_scale_storage_bytes\":"<<low_storage<<",\"descriptor_storage_bytes\":"<<descriptor_storage<<",\"scale_RF_read_bytes\":"<<scale_rf_reads<<",\"scale_SRAM_read_bytes\":"<<scale_sram_reads<<",\"scale_RF_refill_ticks\":"<<scale_fill_ticks<<",\"descriptor_RF_read_bytes\":"<<descriptor_reads<<",\"descriptor_service_ticks\":"<<descriptor_ticks<<",\"variant_control_ticks\":"<<control_ticks<<",\"row_index_read_bytes\":"<<index_read_bytes<<",\"row_index_write_bytes\":"<<index_write_bytes<<",\"row_index_service_ticks\":"<<index_ticks<<",\"output_store_ticks\":"<<output_store_ticks<<",\"name\":\""<<g.name''')
s=s.replace('<<",\\"tail_precision\\":8,\\"rows\\":"','<<",\\"tail_precision\\":"<<(mode==2?4:8)<<",\\"rows\\":"')
s=s.replace('flow_e4_finite_address_replay_v1','flow_e4_rne4_dual_finite_replay_v1')
s=s.replace('arm<4','arm<5').replace('Simulator(g,arm==0,arm<2?8:arm==2?6:4).run()','Simulator(g,arm).run()')
(HERE/'rne4_supply.cpp').write_text(s)
sources=[OLD,MASK,ROOT/'flow_story_20261021/e0_ordinary/mixed_integer_core.py',ROOT/'flow_story_20261021/e0_ordinary/mask_contract.py',ROOT/'flow_story_20261021/e0_ordinary/SOURCE_FREEZE_V1.json',ROOT/'flow_story_20261019/round27/hardware/UNIFIED_SERVICE_V1.json']
(HERE/'SOURCE_BINDINGS_V1.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
