// Finite single-context-PE schedule. Address transactions, not network/FP RTL.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
using U=uint64_t;
static U ceildiv(U n,U d){return (n+d-1)/d;}
static void need(bool b,const char*s){if(!b)throw std::runtime_error(s);}
static uint32_t bitmap_words[192]={};
struct Geometry{std::string name;int h,w,cin,cout;bool conv; int k()const{return cin*(conv?9:1);}U rows()const{return U(10)*h*w;}};
struct Counters{U beats=0,weight_rf_reads=0,weight_sram_reads=0,input_rf_reads=0,source_reads=0,packed_writes=0,bank_stall=0,scale_ticks=0,pack_ticks=0,rf_fill_ticks=0,dma_bytes=0,dma_ticks=0,dma_transactions=0,input_dma_block_ticks=0,output_dma_block_ticks=0,hash=1469598103934665603ULL;};
struct Beat{U tick,row,weight_address,activation_address;int out16,bits,plane,chunk;};
struct Event{std::string resource,kind;U start,end,address,bytes;int slot;};
struct Simulator{
 Geometry g;bool parallel;int precision;int mode;int action;bool dynamic;U full_queries=0,bypass_queries=0,query_service_ticks=0,descriptor_init_writes=0;std::array<U,192> bitmap_word_hits{};
 U q8rows=0,q4rows=0,scale_rf_reads=0,scale_sram_reads=0,descriptor_reads=0,descriptor_ticks=0,control_ticks=0,index_read_bytes=0,index_write_bytes=0,index_ticks=0,output_store_ticks=0,high_storage=0,low_storage=0,descriptor_storage=0,scale_fill_ticks=0; bool grouped=false;
Counters c;U dma=0,pe=0,pack=0;std::array<U,2> in_free{},out_free{};std::vector<Event> trace;std::vector<Beat> beats_trace;std::array<int,9> sampled{};U matrix_done=0;U peak_in=0,peak_out=0,peak_rf=0;int batch;bool resident;std::vector<std::pair<U,U>> input_dma_busy,output_dma_busy;U initial_ready=0;
 static constexpr U IOS=524288,RES=4194304,WRF=32768,WSLOT=1048576;
 Simulator(Geometry gg,int selected,bool dyn):g(gg),parallel(false),precision(4),mode(selected==6?3:4),action(selected),dynamic(dyn){
  need(action>=0&&action<=6,"action0..5 or none6");
  batch=128;while(U(batch)*g.cout*4>IOS)batch/=2;
  U chunks=ceildiv(g.k(),parallel?16:128); grouped=(mode>=3 && (ceildiv(chunks*8*256+64,256)+ceildiv(chunks*4*256+64,256))*256>WRF);
  resident=g.rows()*g.cin/8<=RES;need(g.cin%32==0&&g.cout%16==0,"source word/output alignment");
 }
 void event(const std::string&r,const std::string&k,U a,U b,U addr,U bytes,int slot){if(trace.size()<160)trace.push_back({r,k,a,b,addr,bytes,slot});}
 U transfer(const std::string&kind,U ready,U addr,U n,int slot){if(!n)return ready;U a=std::max(dma,ready),b=a+32+ceildiv(n,64);dma=b;c.dma_bytes+=ceildiv(n,64)*64;c.dma_ticks+=b-a;c.dma_transactions++;if(kind=="halo_input")input_dma_busy.push_back({a+32,b});if(kind=="affine_store")output_dma_busy.push_back({a+32,b});event("DMA",kind,a,b,addr,n,slot);return b;}
 std::vector<std::pair<U,U>> source_segments(U first,U count){
  std::vector<std::pair<U,U>> v;if(!g.conv){if(!resident)v.push_back({first*g.cin/8,count*g.cin/8});return v;}
  U left=count,r=first;while(left){U t=r/(g.h*g.w),local=r%(g.h*g.w),take=std::min(left,U(g.h*g.w)-local);
   int y0=int(local/g.w),y1=int((local+take-1)/g.w);int top=std::max(0,y0-1),bot=std::min(g.h-1,y1+1);
   v.push_back({(t*g.h+top)*g.w*g.cin/8,U(bot-top+1)*g.w*g.cin/8});r+=take;left-=take;}return v;
 }
 U source_bytes(U first,U count){U n=0;for(auto s:source_segments(first,count))n+=s.second;return n;}
 U avoid(U now,const std::vector<std::pair<U,U>>& busy,U& stalled){for(auto b:busy)if(now>=b.first&&now<b.second){stalled+=b.second-now;now=b.second;}return now;}
 U packing(U first,U count){
  U ticks=0,padk=ceildiv(g.k(),parallel?16:128)*(parallel?16:128),packed=ceildiv(count*padk,8),halo=source_bytes(first,count);
  peak_in=std::max(peak_in,packed+halo+(grouped?count*4:0));need(packed+halo+(grouped?count*4:0)<=IOS,"halo + packed rows fit input slot");
  for(U r=first;r<first+count;r++){
   std::array<U,16> reads{};int y=int(r/g.w)%g.h,x=int(r%g.w);U t=r/(g.h*g.w);
   if(g.conv){
    for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
     int yy=y+dy,xx=x+dx;if(yy<0||yy>=g.h||xx<0||xx>=g.w)continue;
     for(int cw=0;cw<g.cin/32;cw++){U addr=((t*g.h+yy)*g.w+xx)*(g.cin/32)+cw;reads[addr%16]++;c.source_reads+=4;c.hash=(c.hash^addr)*1099511628211ULL;}
    }
   }else for(int cw=0;cw<g.cin/32;cw++){U addr=r*(g.cin/32)+cw;reads[addr%16]++;c.source_reads+=4;c.hash=(c.hash^addr)*1099511628211ULL;}
   U readcycles=*std::max_element(reads.begin(),reads.end());U totalwords=0;for(auto z:reads)totalwords+=z;
   c.bank_stall+=readcycles-ceildiv(totalwords,16);
   U writewords=ceildiv(padk,32),writes=ceildiv(writewords,16);c.packed_writes+=writewords*4;
   // 1RW: reads precede writes. One explicit row-gather geometry cycle; mode lookup is separate.
   ticks+=readcycles+writes+1;
  }c.pack_ticks+=ticks;return ticks;
 }

 int row_bits(U r)const{
  if(mode<2)return 8;
  if(mode==2)return 4; // component reference, NOT actual E0 Q4 true-tail semantics
  int y=int(r/g.w)%g.h,x=int(r%g.w);
  if(y>=(g.h/9)*9||x>=(g.w/9)*9)return 8;
  if(mode==3)return 4;
  U id=U(y/9)*35+x/9;return ((bitmap_words[action*32+id/32]>>(id%32))&1u)?8:4;
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
   U previous_desc=~U(0);
   auto processrow=[&](U r,U listpos){
    int bits=row_bits(r);if(grouped){
     now=avoid(now,input_dma_busy,c.input_dma_block_ticks);
     U addr=packed_base+count*ceildiv(padk,8)+listpos*4;
     need(addr+4<=IOS,"row index read address");need(listpos<index_contents.size()&&index_contents[listpos]==r,"row index data/canonical scatter identity");now++;index_ticks++;index_read_bytes+=4;c.hash=(c.hash^addr^r)*1099511628211ULL;
    }else desc(r,previous_desc);
    // Query response latches the mode; no duplicate precision-switch tick.

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
 U bn_phase(const std::string&name,int operations,bool output,U ready){
  U elems=g.rows()*g.cout,phase=ready;constexpr U chunk=65536;
  for(U first=0;first<elems;first+=chunk){U n=std::min(chunk,elems-first);int slot=(first/chunk)%2;
   U loaded=transfer("BN_"+name+"_read",phase,first*4,n*4,slot);
   U a=std::max(phase,loaded),b=a+ceildiv(n,16)*operations;
   // Fixed1RW bank groups: prefetch is not assumed concurrent with vector
   // reads. Three cycles for normalization cover input read/affine/write.
   event("VECTOR","BN_"+name,a,b,first*4,n*4,slot);phase=b;
   if(output)phase=transfer("BN_normalized_store",phase,first*4,n*4,slot);
  }return std::max(phase,dma);
 }
 void run(){
  U padk=ceildiv(g.k(),parallel?16:128)*(parallel?16:128);
  high_storage=mode==2?0:padk*g.cout+4*g.cout;
  low_storage=mode<2?0:padk*g.cout/2+4*g.cout;
  need(high_storage+low_storage<=WSLOT,"independent weights and scales fit1MiB weight SRAM slot");
  if(high_storage)transfer("Q8_weights_and_scales",0,0,high_storage,0);
  if(low_storage)transfer("RNE4_weights_and_scales",0,high_storage,low_storage,0);
  if(action<6){
   descriptor_storage=dynamic?768:128;need(descriptor_storage<=2048,"existing2KiB descriptor reservation");
   transfer(dynamic?"six_bitmaps_load":"one_fixed_bitmap_load",0,0,descriptor_storage,0);
   U a=dma;dma+=descriptor_storage/4;descriptor_init_writes=descriptor_storage/4;descriptor_ticks+=descriptor_init_writes;
   event("STATS_RF","bitmap_init32bit1RW",a,dma,0,descriptor_storage,0);
  }
  if(resident)transfer("resident_binary_input",0,0,g.rows()*g.cin/8,0);
  initial_ready=dma+1; // start handshake; selector is an interface, not a predictor
  U jobs=ceildiv(g.rows(),batch);std::vector<U> ends(jobs),counts(jobs),starts(jobs),load_ready(jobs);
  for(U j=0;j<jobs;j++){starts[j]=j*batch;counts[j]=std::min<U>(batch,g.rows()-starts[j]);}
  auto load=[&](U j,U after){U ready=std::max({initial_ready,in_free[j%2],after});for(auto seg:source_segments(starts[j],counts[j]))ready=transfer("halo_input",ready,seg.first,seg.second,j%2);return ready;};
  auto prepare=[&](U j){U a=std::max({load_ready[j],pack,pe,in_free[j%2]}),b=a+packing(starts[j],counts[j]);pack=b;
   event("INPUT_BANKS","gather_and_pack",a,b,starts[j]*g.cin/8,counts[j]*ceildiv(padk,8),j%2);return b;};
  load_ready[0]=load(0,0);U prepared=prepare(0);input_dma_busy.clear();
  for(U j=0;j<jobs;j++){
   U a=std::max({pe,prepared,out_free[j%2]});input_dma_busy.clear();output_dma_busy.clear();
   if(j)out_free[(j-1)%2]=transfer("affine_store",ends[j-1],starts[j-1]*g.cout*4,counts[j-1]*g.cout*4,(j-1)%2);
   if(j+1<jobs)load_ready[j+1]=load(j+1,a);
   U b=matrix(starts[j],counts[j],a,j%2);ends[j]=b;pe=b;in_free[j%2]=b;
   if(j+1<jobs)prepared=prepare(j+1);
  }
  out_free[(jobs-1)%2]=transfer("affine_store",ends.back(),starts.back()*g.cout*4,counts.back()*g.cout*4,(jobs-1)%2);
  matrix_done=std::max(pe,dma)+1; // finish handshake after all affine retirement
  U mean=bn_phase("mean",1,false,matrix_done);
  // Statistics finalize/normalization coefficients are explicitly serialized.
  U finalize_mean=ceildiv(g.cout,16)+4;mean+=finalize_mean;
  U variance=bn_phase("variance",2,false,mean);
  U finalize_var=U(g.cout)+16+5*ceildiv(g.cout,16);variance+=finalize_var;
  U norm=bn_phase("normalization",3,true,variance); // read, affine transform, write
  pe=norm;
  const char* arm=action==6?(dynamic?"dynamic_none":"static_none"):(dynamic?"dynamic_bitmap":"static_bitmap");
  std::cout<<"{\"bitmap_word_read_histogram\":[";for(int wi=0;wi<192;wi++){if(wi)std::cout<<",";std::cout<<bitmap_word_hits[wi];}std::cout<<"],\"selected_action\":"<<action<<",\"dynamic_selection\":"<<(dynamic?"true":"false")<<",\"full_bitmap_queries\":"<<full_queries<<",\"registered_bypass_queries\":"<<bypass_queries<<",\"query_service_ticks\":"<<query_service_ticks<<",\"descriptor_init32bit_writes\":"<<descriptor_init_writes<<",\"boundary_handshake_ticks\":2,\"predictor_ticks_included\":false,\"arm\":\""<<arm<<"\",\"row_schedule\":\""<<(grouped?"batch_local_two_variant_partition":"natural_row_order")<<"\",\"q8_rows\":"<<q8rows<<",\"q4_rows\":"<<q4rows<<",\"Q8_weight_scale_storage_bytes\":"<<high_storage<<",\"RNE4_weight_scale_storage_bytes\":"<<low_storage<<",\"descriptor_storage_bytes\":"<<descriptor_storage<<",\"scale_RF_read_bytes\":"<<scale_rf_reads<<",\"scale_SRAM_read_bytes\":"<<scale_sram_reads<<",\"scale_RF_refill_ticks\":"<<scale_fill_ticks<<",\"descriptor_RF_read_bytes\":"<<descriptor_reads<<",\"descriptor_service_ticks\":"<<descriptor_ticks<<",\"variant_control_ticks\":"<<control_ticks<<",\"row_index_read_bytes\":"<<index_read_bytes<<",\"row_index_write_bytes\":"<<index_write_bytes<<",\"row_index_service_ticks\":"<<index_ticks<<",\"output_store_ticks\":"<<output_store_ticks<<",\"name\":\""<<g.name<<"\",\"engine\":\""<<(parallel?"parallel16":"bitplane128")<<"\",\"precision_complete\":"<<precision<<",\"tail_precision\":"<<(mode==2?4:8)<<",\"rows\":"<<g.rows()<<",\"K\":"<<g.k()<<",\"Cout\":"<<g.cout<<",\"batch_rows\":"<<batch<<",\"arithmetic_accepted_beats\":"<<c.beats<<",\"scale_vector_ticks\":"<<c.scale_ticks<<",\"affine_completion_ticks\":"<<matrix_done<<",\"affine_plus_BN_completion_ticks\":"<<pe<<",\"input_gather_pack_ticks\":"<<c.pack_ticks<<",\"input_bank_conflict_extra_ticks\":"<<c.bank_stall<<",\"input_DMA_blocks_PE_ticks\":"<<c.input_dma_block_ticks<<",\"output_DMA_blocks_PE_ticks\":"<<c.output_dma_block_ticks<<",\"weight_RF_prefill_busy_ticks\":"<<c.rf_fill_ticks<<",\"weight_RF_read_bytes\":"<<c.weight_rf_reads<<",\"weight_SRAM_read_bytes\":"<<c.weight_sram_reads<<",\"activation_RF_read_bytes\":"<<c.input_rf_reads<<",\"source_input_word_read_bytes\":"<<c.source_reads<<",\"packed_input_write_bytes\":"<<c.packed_writes<<",\"DMA_bytes\":"<<c.dma_bytes<<",\"DMA_busy_ticks\":"<<c.dma_ticks<<",\"DMA_transactions\":"<<c.dma_transactions<<",\"peak_input_slot_bytes\":"<<peak_in<<",\"peak_output_slot_bytes\":"<<peak_out<<",\"peak_weight_RF_bytes\":"<<peak_rf<<",\"BN_full_domain_passes\":3,\"BN_dependencies\":{\"all_affine_done\":"<<matrix_done<<",\"mean_finalized\":"<<mean<<",\"variance_finalized\":"<<variance<<",\"normalized_output_done\":"<<norm<<"},\"address_fnv64\":\""<<c.hash<<"\",\"trace_prefix\":[";
  for(size_t i=0;i<trace.size();i++){auto&e=trace[i];if(i)std::cout<<",";std::cout<<"{\"resource\":\""<<e.resource<<"\",\"kind\":\""<<e.kind<<"\",\"start\":"<<e.start<<",\"end\":"<<e.end<<",\"address\":"<<e.address<<",\"bytes\":"<<e.bytes<<",\"slot\":"<<e.slot<<"}";}std::cout<<"],\"accepted_beat_trace\":[";
  for(size_t i=0;i<beats_trace.size();i++){auto&b=beats_trace[i];if(i)std::cout<<",";std::cout<<"{\"tick\":"<<b.tick<<",\"row\":"<<b.row<<",\"out16\":"<<b.out16<<",\"bits\":"<<b.bits<<",\"plane\":"<<b.plane<<",\"chunk\":"<<b.chunk<<",\"weight_RF_address\":"<<b.weight_address<<",\"activation_slot_address\":"<<b.activation_address<<"}";}
  std::cout<<"]}";
 }
};
int main(int argc,char**argv){try{
 need(argc==2,"complete upgraded tile count required");int count=std::stoi(argv[1]);need(count>=151&&count<=910,"151..910 tiles");
 std::vector<int> order;for(int i=302;i<453;i++)order.push_back(i);for(int i=0;i<910;i++)if(i<302||i>=453)order.push_back(i);
 for(int j=0;j<count;j++)bitmap_words[order[j]/32]|=1u<<(order[j]%32);
 Geometry g={"patch.res0.conv0",240,320,96,96,true};
 std::cout<<"{\"schema\":\"r34_static_neighbor_operator_service_v1\",\"complete_Q8_tiles\":"<<count<<",\"mask_words\":[";
 for(int i=0;i<32;i++){if(i)std::cout<<",";std::cout<<bitmap_words[i];}
 std::cout<<"],\"result\":";Simulator(g,0,false).run();std::cout<<",\"AEE_measured\":false,\"whole_network\":false}\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
