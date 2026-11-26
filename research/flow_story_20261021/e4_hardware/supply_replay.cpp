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
struct Geometry{std::string name;int h,w,cin,cout;bool conv; int k()const{return cin*(conv?9:1);}U rows()const{return U(10)*h*w;}};
struct Counters{U beats=0,weight_rf_reads=0,weight_sram_reads=0,input_rf_reads=0,source_reads=0,packed_writes=0,bank_stall=0,scale_ticks=0,pack_ticks=0,rf_fill_ticks=0,dma_bytes=0,dma_ticks=0,dma_transactions=0,input_dma_block_ticks=0,output_dma_block_ticks=0,hash=1469598103934665603ULL;};
struct Beat{U tick,row,weight_address,activation_address;int out16,bits,plane,chunk;};
struct Event{std::string resource,kind;U start,end,address,bytes;int slot;};
struct Simulator{
 Geometry g;bool parallel;int precision;Counters c;U dma=0,pe=0,pack=0;std::array<U,2> in_free{},out_free{};std::vector<Event> trace;std::vector<Beat> beats_trace;std::array<int,9> sampled{};U matrix_done=0;U peak_in=0,peak_out=0,peak_rf=0;int batch;bool resident;std::vector<std::pair<U,U>> input_dma_busy,output_dma_busy;U initial_ready=0;
 static constexpr U IOS=524288,RES=4194304,WRF=32768,WSLOT=1048576;
 Simulator(Geometry gg,bool p,int q):g(gg),parallel(p),precision(q){
  batch=128;while(U(batch)*g.cout*4>IOS)batch/=2;
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
  peak_in=std::max(peak_in,packed+halo);need(packed+halo<=IOS,"halo + packed rows fit input slot");
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
   // 1RW: reads precede writes. One explicit geometry/mode descriptor cycle.
   ticks+=readcycles+writes+1;
  }c.pack_ticks+=ticks;return ticks;
 }
 U matrix(U first,U count,U ready,int slot){
  U packed_base=source_bytes(first,count);U now=ready,padk=ceildiv(g.k(),parallel?16:128)*(parallel?16:128),chunks=ceildiv(g.k(),parallel?16:128);
  U blockbytes=parallel?chunks*256:chunks*8*256;peak_rf=std::max(peak_rf,blockbytes);need(blockbytes<=WRF,"one output16 weight block fits32KiB RF");
  U rf_calendar=ready;U next_prefill=ready;
  auto prefill=[&](int ob,U earliest){U a=std::max(rf_calendar,earliest),b=a+ceildiv(blockbytes,256);rf_calendar=b;c.rf_fill_ticks+=b-a;c.weight_sram_reads+=blockbytes;event("WEIGHT_SRAM", "prefill_RF",a,b,U(ob)*blockbytes,blockbytes,ob%2);return b;};
  next_prefill=prefill(0,ready);
  for(int ob=0;ob<g.cout/16;ob++){
   now=std::max(now,next_prefill);
   // RF uses one64-bank1RW group. Even the other capacity half cannot be
   // filled concurrently with an all-bank256B PE read. No extra RF ports.
   U start=now;
   for(U r=first;r<first+count;r++){
    int y=int(r/g.w)%g.h,x=int(r%g.w);int bits=(y<(g.h/9)*9&&x<(g.w/9)*9)?precision:8;
    U rowbeats=parallel?chunks:chunks*bits;
    // Each accepted beat reads exactly one aligned256B RF line:64 distinct
    // 32-bit banks. Activation chunk reads1(parallel) or4(serial) distinct banks.
    // Count/address-hash every beat; no implied byte/bandwidth overlap.
    for(U b=0;b<rowbeats;b++){
     U rfaddr=(parallel?b:U(8-bits)*chunks+b)*256;
     need(rfaddr%256==0&&rfaddr+256<=blockbytes,"weight RF aligned and in range");
     U chunk=parallel?b:b%chunks,actaddr=packed_base+(r-first)*ceildiv(padk,8)+chunk*(parallel?2:16);
     need(actaddr+U(parallel?2:16)<=packed_base+count*ceildiv(padk,8),"activation slot address");
     now=avoid(now,input_dma_busy,c.input_dma_block_ticks);
     if(sampled[bits]<16){beats_trace.push_back({now,r,rfaddr,actaddr,ob,bits,parallel?-1:int(8-bits+b/chunks),int(chunk)});sampled[bits]++;}
     c.hash=(c.hash^rfaddr^(actaddr<<1)^U(ob))*1099511628211ULL;now++;
     c.weight_rf_reads+=256;c.input_rf_reads+=parallel?4:16;c.beats++;
    }
    // Shared16-lane conversion then scale. Output64B occupies16 distinct banks.
    now+=2;c.scale_ticks+=2;now=avoid(now,output_dma_busy,c.output_dma_block_ticks);U outaddr=((r-first)*g.cout+ob*16)*4;
    need(outaddr%64==0&&outaddr+64<=IOS,"output bank address and capacity");
   }
   event("PE", "integer+scale",start,now,U(ob)*blockbytes,blockbytes,slot);
   if(ob+1<g.cout/16)next_prefill=prefill(ob+1,now);
  }
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
  U padk=ceildiv(g.k(),parallel?16:128)*(parallel?16:128),weights=padk*g.cout+4*g.cout;need(weights<=WSLOT,"weights + scales fit single1MiB SRAM slot");
  transfer("weights_and_scales",0,0,weights,0);
  if(resident)transfer("resident_binary_input",0,0,g.rows()*g.cin/8,0);
  initial_ready=dma;
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
  matrix_done=std::max(pe,dma);
  U mean=bn_phase("mean",1,false,matrix_done);
  // Statistics finalize/normalization coefficients are explicitly serialized.
  U finalize_mean=ceildiv(g.cout,16)+4;mean+=finalize_mean;
  U variance=bn_phase("variance",2,false,mean);
  U finalize_var=U(g.cout)+16+5*ceildiv(g.cout,16);variance+=finalize_var;
  U norm=bn_phase("normalization",3,true,variance); // read, affine transform, write
  pe=norm;
  std::cout<<"{\"name\":\""<<g.name<<"\",\"engine\":\""<<(parallel?"parallel16":"bitplane128")<<"\",\"precision_complete\":"<<precision<<",\"tail_precision\":8,\"rows\":"<<g.rows()<<",\"K\":"<<g.k()<<",\"Cout\":"<<g.cout<<",\"batch_rows\":"<<batch<<",\"arithmetic_accepted_beats\":"<<c.beats<<",\"scale_vector_ticks\":"<<c.scale_ticks<<",\"affine_completion_ticks\":"<<matrix_done<<",\"affine_plus_BN_completion_ticks\":"<<pe<<",\"input_gather_pack_ticks\":"<<c.pack_ticks<<",\"input_bank_conflict_extra_ticks\":"<<c.bank_stall<<",\"input_DMA_blocks_PE_ticks\":"<<c.input_dma_block_ticks<<",\"output_DMA_blocks_PE_ticks\":"<<c.output_dma_block_ticks<<",\"weight_RF_prefill_busy_ticks\":"<<c.rf_fill_ticks<<",\"weight_RF_read_bytes\":"<<c.weight_rf_reads<<",\"weight_SRAM_read_bytes\":"<<c.weight_sram_reads<<",\"activation_RF_read_bytes\":"<<c.input_rf_reads<<",\"source_input_word_read_bytes\":"<<c.source_reads<<",\"packed_input_write_bytes\":"<<c.packed_writes<<",\"DMA_bytes\":"<<c.dma_bytes<<",\"DMA_busy_ticks\":"<<c.dma_ticks<<",\"DMA_transactions\":"<<c.dma_transactions<<",\"peak_input_slot_bytes\":"<<peak_in<<",\"peak_output_slot_bytes\":"<<peak_out<<",\"peak_weight_RF_bytes\":"<<peak_rf<<",\"BN_full_domain_passes\":3,\"BN_dependencies\":{\"all_affine_done\":"<<matrix_done<<",\"mean_finalized\":"<<mean<<",\"variance_finalized\":"<<variance<<",\"normalized_output_done\":"<<norm<<"},\"address_fnv64\":\""<<c.hash<<"\",\"trace_prefix\":[";
  for(size_t i=0;i<trace.size();i++){auto&e=trace[i];if(i)std::cout<<",";std::cout<<"{\"resource\":\""<<e.resource<<"\",\"kind\":\""<<e.kind<<"\",\"start\":"<<e.start<<",\"end\":"<<e.end<<",\"address\":"<<e.address<<",\"bytes\":"<<e.bytes<<",\"slot\":"<<e.slot<<"}";}std::cout<<"],\"accepted_beat_trace\":[";
  for(size_t i=0;i<beats_trace.size();i++){auto&b=beats_trace[i];if(i)std::cout<<",";std::cout<<"{\"tick\":"<<b.tick<<",\"row\":"<<b.row<<",\"out16\":"<<b.out16<<",\"bits\":"<<b.bits<<",\"plane\":"<<b.plane<<",\"chunk\":"<<b.chunk<<",\"weight_RF_address\":"<<b.weight_address<<",\"activation_slot_address\":"<<b.activation_address<<"}";}
  std::cout<<"]}";
 }
};
int main(){try{
 std::vector<Geometry> cases={{"patch.res0.conv0",240,320,96,96,true},{"stage2.fc1",30,40,384,1536,false},{"stage2.fc2",30,40,1536,384,false},{"stage0.fc1_K96",120,160,96,384,false}};
 std::cout<<"{\"schema\":\"flow_e4_finite_address_replay_v1\",\"status\":\"COMPLETE_MODEL_NOT_NETWORK_RTL\",\"results\":[";bool first=true;
 for(auto g:cases)for(int arm=0;arm<4;arm++){if(!first)std::cout<<",";first=false;Simulator(g,arm==0,arm<2?8:arm==2?6:4).run();}
 std::cout<<"],\"GPU_used\":false,\"GT_read\":false,\"numerical_BN_verified\":false,\"whole_network\":false}\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
