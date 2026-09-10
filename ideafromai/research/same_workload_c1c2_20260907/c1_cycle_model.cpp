// Complete-operator finite-service CPU model. Not RTL or a frequency/PPA model.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using U = uint64_t;
using I = int64_t;
using Bits = std::array<U,12>;
constexpr int TT=10, HH=15, WW=20, CC=768, NN=768, MM=3000, KK=6912;
constexpr int MT=64, KT=16, NT=96, NK=432, NM=47, NS=8, DD=8;
const int diag_lane[DD]={0,1,15,95,96,383,511,767};
enum Count {RAW_DMA, PREP_READ, PREP_WRITE, COMMON_WRITE, BITMAP_AND, BITMAP_XOR,
 LOAD_WORD, LOAD_ROW, MATCH_ROW, DISPATCH, W_READ, W_FILL, W_HIT, W_DMA,
 P_READ,P_WRITE,S_ADD,Y_READ,Y_WRITE,Y_ADD,P_HIT,P_INHERIT,RESID_ISSUE,
 COMMON_BUILD,COMMON_ISSUE,COMMON_SCATTER,COMMON_REUSE,ADDR_RECORD,COMMON_READ,
 COMMIT,ZERO_COMMIT,OUT_DMA,NUMCOUNT};
const char* labels[NUMCOUNT]={"raw_input_dma_beats","preparation_input_word_reads","preparation_residual_word_writes",
 "common_bitmap_word_writes","bitmap_AND_ops","bitmap_XOR_ops","im2col_source_word_reads","descriptor_rows_loaded",
 "matcher_rows","row_dispatches","coefficient_vector_reads","coefficient_vector_fills","coefficient_prefill_hits",
 "coefficient_dma_beats","parent_vector_reads","parent_vector_writes","source_vector_adds","destination_vector_reads",
 "destination_vector_writes","destination_vector_adds","parent_hits","parent_inherited_coefficients",
 "residual_direct_coefficient_issues","common_G_tap_builds","common_coefficient_issues","common_scatter_consumers",
 "common_coefficient_reuse_with_M_rebuild","address_G_tap_records","common_bitmap_word_reads","output_vector_commits",
 "zero_output_markers","output_dma_beats"};
using Counts=std::array<U,NUMCOUNT>;
struct Port { U free=0; U use(U t,U duration=1){t=std::max(t,free);free=t+duration;return free;} };
struct Memory {
 bool single; Port r,w,rw;
 explicit Memory(bool s=false):single(s){}
 U read(U t){return single?rw.use(t):r.use(t);}
 U write(U t){return single?rw.use(t):w.use(t);}
 U end() const{return single?rw.free:std::max(r.free,w.free);}
};
struct Tile {
 std::array<uint16_t,64> mask{};
 std::array<int16_t,64> parent{};
 std::array<uint8_t,64> order{},child{};
 U load_cycles=0,source_words=0;
 int real=0;
};
struct Tap {int delta;std::vector<int> dest;};
struct GroupView {int group;std::vector<Tap> taps;};
struct Data {
 int g; bool product; std::string name;
 std::vector<Bits> residual,common;
 std::vector<std::array<int,4>> members;
 std::array<std::vector<GroupView>,NM> views;
 std::vector<Tile> tiles;
 U raw_duplicate_events=0,global_common_build_issues=0;
};
std::vector<Bits> original;
std::vector<uint16_t> original_masks;
std::array<bool,MM> original_nonzero{};
U theta_num=0,theta_den=0;
int pop(uint16_t x){return __builtin_popcount(unsigned(x));}
bool bit(const Bits& b,int c){return (b[c/64]>>(c%64))&1;}
int input_q(int r,int delta){
 int t=r/300,p=r%300,y=p/20,x=p%20;
 int iy=y+delta/3-1,ix=x+delta%3-1;
 return (iy<0||iy>=15||ix<0||ix>=20)?-1:t*300+iy*20+ix;
}
int output_r(int q,int delta){
 int t=q/300,p=q%300,y=p/20,x=p%20;
 int oy=y-delta/3+1,ox=x-delta%3+1;
 return (oy<0||oy>=15||ox<0||ox>=20)?-1:t*300+oy*20+ox;
}
I phi(int f,int lane){return I(((f*17+lane*13+11)&255)-128)*I(theta_num);}
void read_input(const char* path){
 std::ifstream in(path,std::ios::binary);assert(in);
 char magic[8];uint32_t dims[8];U theta[2];
 in.read(magic,8);in.read(reinterpret_cast<char*>(dims),32);in.read(reinterpret_cast<char*>(theta),16);
 assert(std::memcmp(magic,"C1UNION1",8)==0);
 const uint32_t expected[8]={10,15,20,768,768,3000,6912,16};
 for(int i=0;i<8;++i)assert(dims[i]==expected[i]);
 theta_num=theta[0];theta_den=theta[1];assert(theta_den && !(theta_den&(theta_den-1)));
 assert(theta_num*U(128)*KK < (U(1)<<47));
 original.resize(MM);original_masks.resize(MM*NK);
 in.read(reinterpret_cast<char*>(original.data()),MM*96);
 in.read(reinterpret_cast<char*>(original_masks.data()),MM*NK*2);
 assert(in && in.peek()==std::ifstream::traits_type::eof());
 for(int r=0;r<MM;++r)for(int k=0;k<NK;++k){
  uint16_t mask=0;
  for(int j=0;j<16;++j){int f=k*16+j,q=input_q(r,f%9);if(q>=0&&bit(original[q],f/9))mask|=uint16_t(1u<<j);}
  assert(mask==original_masks[r*NK+k]);if(mask)original_nonzero[r]=true;
 }
}
Data build_data(int g,bool product,const std::string& name){
 Data d;d.g=g;d.product=product;d.name=name;d.residual=original;
 if(g){
  assert(g==2||g==4);
  for(int t=0;t<10;++t)for(int y=0;y<15;++y)for(int x=0;x<20;x+=g){
   std::array<int,4> qs{{-1,-1,-1,-1}};Bits common;
   common.fill(~U(0));
   for(int a=0;a<g;++a){qs[a]=t*300+y*20+x+a;for(int w=0;w<12;++w)common[w]&=original[qs[a]][w];}
   U pc=0;for(auto v:common)pc+=__builtin_popcountll(v);
   d.raw_duplicate_events+=U(g-1)*pc;
   int id=int(d.common.size());d.common.push_back(common);d.members.push_back(qs);
   for(int a=0;a<g;++a)for(int w=0;w<12;++w){
    d.residual[qs[a]][w]&=~common[w];
    assert((d.residual[qs[a]][w]&common[w])==0);
    assert((d.residual[qs[a]][w]|common[w])==original[qs[a]][w]);
   }
   for(int delta=0;delta<9;++delta){
    std::array<std::vector<int>,NM> per;
    bool any=false;
    for(int a=0;a<g;++a){int r=output_r(qs[a],delta);if(r>=0){per[r/64].push_back(r%64);any=true;}}
    if(any)d.global_common_build_issues+=pc;
    for(int m=0;m<NM;++m)if(!per[m].empty()){
     if(d.views[m].empty()||d.views[m].back().group!=id)d.views[m].push_back(GroupView{id,{}});
     d.views[m].back().taps.push_back(Tap{delta,per[m]});
    }
   }
  }
 }
 d.tiles.reserve(NM*NK);
 for(int m=0;m<NM;++m)for(int k=0;k<NK;++k){
  Tile a;a.real=std::min(64,MM-m*64);a.parent.fill(-1);
  for(int i=0;i<64;++i){
   a.order[i]=uint8_t(i);
   if(i>=a.real){++a.load_cycles;continue;}
   int r=m*64+i;std::array<int,16> addrs{};int na=0;std::array<int,8> bank{};
   for(int j=0;j<16;++j){
    int f=k*16+j,q=input_q(r,f%9);if(q<0)continue;
    if(bit(d.residual[q],f/9))a.mask[i]|=uint16_t(1u<<j);
    int addr=q*6+(f/9)/128;
    bool found=false;for(int z=0;z<na;++z)if(addrs[z]==addr)found=true;
    if(!found){addrs[na++]=addr;++bank[addr%8];}
   }
   a.source_words+=na;
   a.load_cycles+=std::max(1,*std::max_element(bank.begin(),bank.end()));
  }
  ++a.load_cycles; // final descriptor assembly response; preceding rows pipeline.
  if(product){
   for(int i=0;i<a.real;++i)if(pop(a.mask[i])>=2){
    int best=0;
    for(int p=0;p<a.real;++p){
     int pc=pop(a.mask[p]);if(!pc||pc<=best)continue;
     if((a.mask[i]&a.mask[p])!=a.mask[p])continue;
     if(a.mask[i]==a.mask[p]&&p>=i)continue;
     best=pc;a.parent[i]=int16_t(p);
    }
    if(a.parent[i]>=0)++a.child[a.parent[i]];
   }
   std::stable_sort(a.order.begin(),a.order.end(),[&](uint8_t x,uint8_t y){return pop(a.mask[x])<pop(a.mask[y]);});
   std::array<bool,64> seen{};
   for(auto r:a.order){int p=a.parent[r];if(p>=0){assert(seen[p]);assert((a.mask[r]&a.mask[p])==a.mask[p]);}seen[r]=true;}
  }
  d.tiles.push_back(a);
 }
 return d;
}
struct Engine {
 Counts c{}; Memory parent,out;
 Port bus,weight_read,weight_write,tag,source_alu,dest_alu,dispatch,common_port;
 std::array<int,16> cache;std::array<U,16> age{},weight_ready{};U clock=0;
 std::array<U,64> parent_ready{},y_ready{};std::array<bool,64> y_live{};
 U source_free=1,read_operand_free=0,pending_free=0;
 explicit Engine(bool single):parent(single),out(single){cache.fill(-1);}
 U ensure(int f,U t){
  t=tag.use(t);int slot=-1;
  for(int i=0;i<16;++i)if(cache[i]==f)slot=i;
  if(slot>=0){++c[W_HIT];age[slot]=++clock;return std::max(t,weight_ready[slot]);}
  for(int i=0;i<16;++i)if(cache[i]<0){slot=i;break;}
  if(slot<0)slot=int(std::min_element(age.begin(),age.end())-age.begin());
  t=bus.use(t,6);c[W_DMA]+=6;++c[W_FILL];t=weight_write.use(t);
  cache[slot]=f;age[slot]=++clock;weight_ready[slot]=t;return t;
 }
 U weight(int f,U t){
  int slot=-1;for(int i=0;i<16;++i)if(cache[i]==f)slot=i;
  assert(slot>=0);age[slot]=++clock;++c[W_READ];return weight_read.use(std::max(t,weight_ready[slot]));
 }
 U dest(int row,U value_ready,U earliest){
  U rhs_ready=earliest;
  if(y_live[row]){rhs_ready=out.read(std::max({earliest,y_ready[row],read_operand_free}));++c[Y_READ];}
  U t=std::max(value_ready,rhs_ready);
  if(pending_free)t=std::max(t,pending_free-1);
  U result=dest_alu.use(t); // copy and add both occupy destination pipeline stage.
  if(y_live[row])++c[Y_ADD];
  read_operand_free=result;
  U end=out.write(result);++c[Y_WRITE];y_live[row]=true;y_ready[row]=end;pending_free=end;
  return result; // source value consumed; pending result retains it until write.
 }
 U common_phase(const Data& d,int m){
  for(const auto& view:d.views[m]){
   U ready=source_free;
   for(int word=0;word<6;++word){ready=common_port.use(ready);++c[COMMON_READ];}
   const Bits& b=d.common[view.group];std::vector<int> channels;
   for(int w=0;w<12;++w){U bits=b[w];while(bits){int z=__builtin_ctzll(bits);channels.push_back(w*64+z);bits&=bits-1;}}
   source_free=std::max(source_free,ready);
   for(const auto& tap:view.taps){
    U start=dispatch.use(source_free);++c[ADDR_RECORD];
    if(channels.empty()){source_free=start;continue;}
    ++c[COMMON_BUILD];c[COMMON_REUSE]+=U(tap.dest.size()-1)*channels.size();
    U value=start;bool live=false;
    for(int channel:channels){
     int f=channel*9+tap.delta;
     U earliest=live?std::max(start,value-1):start;
     U available=ensure(f,earliest);U w=weight(f,available);++c[COMMON_ISSUE];
     if(live){value=source_alu.use(std::max(value,w));++c[S_ADD];}else{value=w;live=true;}
    }
    U consumed=value;
    for(int r:tap.dest){++c[COMMON_SCATTER];consumed=std::max(consumed,dest(r,value,value));}
    source_free=consumed;
   }
  }
  return std::max({source_free,out.end(),pending_free});
 }
 U residual_tile(const Tile& a,int k,U start,bool product){
  assert(parent.end()<=start); // previous K writes drain before parent epoch reset.
  U t=start+1; // valid-map/parent-epoch clear.
  parent_ready.fill(0);
  uint16_t used=0;for(int r=0;r<a.real;++r){uint16_t p=a.parent[r]>=0?a.mask[a.parent[r]]:0;used|=uint16_t(a.mask[r]^p);}
  U filled=t;
  for(int j=0;j<16;++j)if((used>>j)&1)filled=ensure(k*16+j,filled);
  source_free=std::max(source_free,filled);
  for(auto row:a.order){
   if(row>=a.real||!a.mask[row])continue;
   U begin=dispatch.use(source_free);++c[DISPATCH];
   int p=product?a.parent[row]:-1;
   uint16_t rem=uint16_t(a.mask[row]^(p>=0?a.mask[p]:0));
   U value=begin;bool live=false;
   if(p>=0){
    assert(parent_ready[p]>0);value=parent.read(std::max(begin,parent_ready[p]));live=true;
    ++c[P_READ];++c[P_HIT];c[P_INHERIT]+=pop(a.mask[p]);
   }
   for(int j=0;j<16;++j)if((rem>>j)&1){
    U earliest=live?std::max(begin,value-1):begin;
    U w=weight(k*16+j,earliest);++c[RESID_ISSUE];
    if(live){value=source_alu.use(std::max(value,w));++c[S_ADD];}else{value=w;live=true;}
   }
   assert(live);
   if(product&&a.child[row]){parent_ready[row]=parent.write(value);++c[P_WRITE];}
   source_free=dest(row,value,begin);
  }
  return std::max({source_free,parent.end(),out.end(),pending_free,weight_write.free,bus.free});
 }
 U commit(int m,int real,U start){
  U end=start;
  for(int r=0;r<real;++r){
   U ready=end;
   if(original_nonzero[m*64+r]){assert(y_live[r]);ready=out.read(std::max(ready,y_ready[r]));++c[Y_READ];ready=bus.use(ready,6);c[OUT_DMA]+=6;}
   else{assert(!y_live[r]);ready=bus.use(ready);++c[ZERO_COMMIT];++c[OUT_DMA];}
   ++c[COMMIT];end=ready;
  }
  return std::max({end,parent.end(),out.end(),pending_free,bus.free});
 }
};
U preparation(const Data& d,Counts& counts){
 U t=3000;counts[RAW_DMA]=3000; // 3000*768 source bits /768bit shared bus.
 if(!d.g)return t;
 std::array<Memory,8> source;Port logic,common;
 for(const auto& qs:d.members)for(int word=0;word<6;++word){
  U read_done=t;
  for(int q:qs)if(q>=0){read_done=std::max(read_done,source[(q*6+word)%8].read(t));++counts[PREP_READ];}
  U v=read_done;
  for(int a=1;a<d.g;++a){v=logic.use(v);++counts[BITMAP_AND];}
  U common_end=common.use(v);++counts[COMMON_WRITE];U write_done=common_end;
  for(int q:qs)if(q>=0){v=logic.use(v);++counts[BITMAP_XOR];write_done=std::max(write_done,source[(q*6+word)%8].write(v));++counts[PREP_WRITE];}
  t=std::max(v,write_done);
 }
 return t;
}
std::array<std::array<I,DD>,MM> gold;
void make_gold(){
 for(int r=0;r<MM;++r)for(int k=0;k<NK;++k){uint16_t mask=original_masks[r*NK+k];
  while(mask){int j=__builtin_ctz(unsigned(mask));for(int n=0;n<DD;++n)gold[r][n]+=phi(k*16+j,diag_lane[n]);mask&=uint16_t(mask-1);}
 }
}
U check_numeric(const Data& d){
 U checked=0;
 for(int m=0;m<NM;++m){
  std::array<std::array<I,DD>,64> y{};
  for(const auto& view:d.views[m])for(const auto& tap:view.taps){
   std::array<I,DD> common{};
   for(int c=0;c<CC;++c)if(bit(d.common[view.group],c))for(int n=0;n<DD;++n)common[n]+=phi(c*9+tap.delta,diag_lane[n]);
   for(int r:tap.dest)for(int n=0;n<DD;++n)y[r][n]+=common[n];
  }
  for(int k=0;k<NK;++k){
   const Tile& a=d.tiles[m*NK+k];std::array<std::array<I,DD>,64> parent{};std::array<bool,64> live{};
   for(auto r:a.order){
    if(r>=a.real||!a.mask[r])continue;
    int p=d.product?a.parent[r]:-1;uint16_t rem=uint16_t(a.mask[r]^(p>=0?a.mask[p]:0));
    if(p>=0){assert(live[p]);parent[r]=parent[p];}
    while(rem){int j=__builtin_ctz(unsigned(rem));for(int n=0;n<DD;++n)parent[r][n]+=phi(k*16+j,diag_lane[n]);rem&=uint16_t(rem-1);}
    live[r]=true;for(int n=0;n<DD;++n)y[r][n]+=parent[r][n];
   }
  }
  for(int r=0;r<std::min(64,MM-m*64);++r)for(int n=0;n<DD;++n){assert(y[r][n]==gold[m*64+r][n]);++checked;}
 }
 return checked;
}
struct Result {std::string mode,ports;Counts c{};U prep=0,cycles=0,common=0,residual=0,commit=0,frontend_wait=0,checks=0,raw_dup=0,global_common=0;};
Result run(const Data& d,bool single,U checks){
 Result z;z.mode=d.name;z.ports=single?"1RW":"1R1W";z.checks=checks;z.raw_dup=d.raw_duplicate_events;z.global_common=d.global_common_build_issues;
 z.prep=preparation(d,z.c);z.cycles=z.prep;
 for(int m=0;m<NM;++m)for(int n=0;n<NS;++n){
  Engine e(single);U common_end=e.common_phase(d,m);z.common+=common_end;
  std::array<U,NK> ends{};U load_end=0,match_end=0,backend_end=common_end;
  for(int k=0;k<NK;++k){
   const Tile& a=d.tiles[m*NK+k];
   U release=k>=2?ends[k-2]:0;
   U load_start=std::max(load_end,release);load_end=load_start+a.load_cycles;
   match_end=std::max(load_end,match_end)+(d.product?67:0);
   U start=std::max(backend_end,match_end);z.frontend_wait+=start-backend_end;
   e.c[LOAD_WORD]+=a.source_words;e.c[LOAD_ROW]+=64;if(d.product)e.c[MATCH_ROW]+=64;
   U end=e.residual_tile(a,k,start,d.product);z.residual+=end-start;
   ends[k]=end;backend_end=end;
  }
  U done=e.commit(m,d.tiles[m*NK].real,backend_end);z.commit+=done-backend_end;z.cycles+=done;
  for(int i=0;i<NUMCOUNT;++i)z.c[i]+=e.c[i];
 }
 assert(z.c[COMMIT]==MM*NS);
 assert(z.cycles==z.prep+z.common+z.frontend_wait+z.residual+z.commit);
 return z;
}
void emit(std::ostream& out,const std::vector<Result>& results){
 out<<"{\n\"status\":\"COMPLETE_OPERATOR_CPU_SERVICE_MODEL_NOT_RTL\",\n\"theta_numerator\":"<<theta_num<<",\"theta_denominator\":"<<theta_den<<",\n\"points\":[\n";
 for(size_t i=0;i<results.size();++i){const auto& r=results[i];if(i)out<<",\n";
  out<<"{\"mode\":\""<<r.mode<<"\",\"value_memory_ports\":\""<<r.ports<<"\",\"total_cycles\":"<<r.cycles
   <<",\"input_and_preparation_cycles\":"<<r.prep<<",\"common_phase_cycles\":"<<r.common
   <<",\"residual_execution_cycles\":"<<r.residual<<",\"frontend_wait_cycles\":"<<r.frontend_wait
   <<",\"final_commit_cycles\":"<<r.commit<<",\"diagnostic_lane_values_checked\":"<<r.checks
   <<",\"input_apec_duplicate_events\":"<<r.raw_dup<<",\"common_coefficient_builds_if_no_M_repetition_one_N_slice\":"<<r.global_common<<",\"counts\":{";
  for(int j=0;j<NUMCOUNT;++j){if(j)out<<',';out<<'"'<<labels[j]<<"\":"<<r.c[j];}out<<"}}";
 }
 out<<"\n],\"limitations\":[\"New declared CPU resource model, not mapped m935\",\"Diagnostic signedINT8 W with exact theta; no trainedFP32 or AEE\",\"Full source support and complete K are covered; all output-vector identities commit\",\"Geometric address generator throughput is declared, not RTL timing-proven\",\"Same allocated resources do not establish ASIC iso-area or energy\"]}\n";
}
int main(int argc,char**argv){
 assert(argc==3);std::ifstream existing(argv[2]);assert(!existing.good());
 read_input(argv[1]);make_gold();std::vector<Result> results;
 const char* names[]={"bit","prosperity","apec_g2_bit_residual","apec_g2_prosperity_residual","apec_g4_prosperity_residual"};
 const int groups[]={0,0,2,2,4};const bool products[]={false,true,false,true,true};
 for(int mode=0;mode<5;++mode){
  Data d=build_data(groups[mode],products[mode],names[mode]);U checks=check_numeric(d);
  std::cerr<<d.name<<" complete support and "<<checks<<" diagnostic values pass\n";
  for(bool single:{false,true}){auto r=run(d,single,checks);std::cerr<<r.mode<<' '<<r.ports<<" cycles "<<r.cycles<<'\n';results.push_back(r);}
 }
 std::ofstream out(argv[2]);assert(out);emit(out,results);out.close();assert(out.good());
}
