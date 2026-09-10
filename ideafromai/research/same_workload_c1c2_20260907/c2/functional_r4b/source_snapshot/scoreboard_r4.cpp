#include "Vdut.h"
#include "verilated.h"
#include <array>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdint>
#ifndef PRODUCTION
#define PRODUCTION 0
#endif
static void need(bool ok,const std::string& s){if(!ok)throw std::runtime_error(s);}
static int signed24(uint32_t x){return (x&0x800000u)?int(x|0xff000000u):int(x);}
struct Fixture {
 int n=0,t=0,c=0,h=0,first=0;
 std::vector<int> masks,weights,golden;
 explicit Fixture(const std::string& dir){
  std::ifstream sf(dir+"/support.txt"); need(bool(sf),"support open");
  sf>>n>>t>>c>>h>>first;need(n==32&&t==10&&c==384&&h==96&&first==256,"fixture geometry");
  int v;while(sf>>std::hex>>v)masks.push_back(v);
  need(int(masks.size())==n*c,"support length");
  std::ifstream wf(dir+"/weights.txt");while(wf>>v)weights.push_back(v);
  std::ifstream gf(dir+"/golden.txt");while(gf>>v)golden.push_back(v);
  need(int(weights.size())==c*h&&int(golden.size())==n*t*h,"weight/golden length");
  for(int x:weights)need(x>=-128&&x<=127,"stored numerator outside INT8");
 }
 int mask(int p,int ch)const{return masks.at((p-first)*c+ch);}
 int truth(int p,int tm,int hc)const{return golden.at(((p-first)*t+tm)*h+hc);}
};
struct Request {
 int b=0,c=0,s=0,slot=0,block=0;uint32_t epoch=0,gen=0,tag=0;
 bool operator==(const Request&o)const{return b==o.b&&c==o.c&&s==o.s&&slot==o.slot&&block==o.block&&epoch==o.epoch&&gen==o.gen&&tag==o.tag;}
};
struct Commit {
 int context=0,slice=0,terminal=0;uint32_t tag=0;std::array<int,16> v{};
 bool operator==(const Commit&o)const{return context==o.context&&slice==o.slice&&terminal==o.terminal&&tag==o.tag&&v==o.v;}
};
struct Memory {
 const Fixture& f;
 struct Bank {
  std::deque<Request> q;
  std::optional<Request> active,response;
  std::array<int,1536> row{};
  std::array<int,16> response_values{};
  int cached=-1,remaining=0,phase=0;
 };
 std::array<Bank,8> banks;
 std::array<bool,8> owned{};
 uint64_t requests=0,responses=0,row_fetches=0,fetch_cycles=0,extract_cycles=0,response_hold_cycles=0;
 std::array<uint64_t,8> physical_rows{},physical_vectors{},fetch_by_bank{},extract_by_bank{},queue_peak{},pending_peak{};
 uint64_t max_owned=0,max_service_banks=0,multi_bank_service_cycles=0,max_pending_banks=0;
 uint64_t max_request_accepts=0,max_response_accepts=0;
 explicit Memory(const Fixture& fx):f(fx){}
 static int physical(const Request&r){
  need(r.c>=0&&r.c<384&&r.s>=0&&r.s<6,"request geometry");
  uint64_t address=393216u+uint64_t(r.c/16)*6144;
  int p=int((address/6144)%8);
  need(p==(r.c/16)%8,"canonical address/bank");return p;
 }
 Request get(const Vdut&m,int b)const{
  Request r;r.b=b;r.c=m.mem_req_source_channel[b];r.s=m.mem_req_slice[b];
  r.slot=m.mem_req_slot[b];r.block=m.mem_req_output_block[b];r.epoch=m.mem_req_epoch[b];
  r.gen=m.mem_req_generation[b];r.tag=m.mem_req_tag[b];return r;
 }
 void drive(Vdut&m){
  m.mem_req_ready=0;m.mem_rsp_valid=0;
  for(int b=0;b<8;b++){
   m.mem_rsp_epoch[b]=0;m.mem_rsp_slot[b]=0;m.mem_rsp_generation[b]=0;m.mem_rsp_tag[b]=0;
   for(int l=0;l<16;l++)m.mem_rsp_weight[b][l]=0;
   // At most eight total owners implies the eight-entry physical-bank queues
   // can accept every newly unowned port, even if they target the same bank.
   if((m.mem_req_valid&(1u<<b))&&!owned[b])m.mem_req_ready|=1u<<b;
  }
  for(int p=0;p<8;p++)if(banks[p].response){
   const Request&r=*banks[p].response;
   need(owned[r.b]&&!(m.mem_rsp_valid&(1u<<r.b)),"response port ownership/collision");
   m.mem_rsp_valid|=1u<<r.b;
   m.mem_rsp_epoch[r.b]=r.epoch;m.mem_rsp_slot[r.b]=r.slot;
   m.mem_rsp_generation[r.b]=r.gen;m.mem_rsp_tag[r.b]=r.tag;
   for(int l=0;l<16;l++)m.mem_rsp_weight[r.b][l]=uint8_t(banks[p].response_values[l]);
  }
 }
 void edge(const std::vector<Request>& accepted,unsigned rsp_accept){
  unsigned consumed=0;
  for(int p=0;p<8;p++){
   Bank&k=banks[p];
   if(k.response){
    if(rsp_accept&(1u<<k.response->b)){
     need(!(consumed&(1u<<k.response->b)),"two response consumers on one port");
     consumed|=1u<<k.response->b;owned[k.response->b]=false;k.response.reset();responses++;
    }else response_hold_cycles++;
   }
  }
  need(consumed==rsp_accept,"response accepted without producer valid");
  max_response_accepts=std::max(max_response_accepts,uint64_t(__builtin_popcount(consumed)));
  max_request_accepts=std::max(max_request_accepts,uint64_t(accepted.size()));
  for(const auto&r:accepted){
   int p=physical(r);
#if PRODUCTION
   need((r.c%8)==r.b,"m803 logical return channel");
#else
   need(p==r.b,"parallel engine must own its canonical bank");
#endif
   need(!owned[r.b],"return-port reused while outstanding");
   owned[r.b]=true;banks[p].q.push_back(r);requests++;physical_vectors[p]++;
  }
  unsigned pending_banks=0;
  for(int p=0;p<8;p++){
   Bank&k=banks[p];
   uint64_t pending=k.q.size()+bool(k.active)+bool(k.response);
   need(pending<=8,"physical bank metadata capacity");
   queue_peak[p]=std::max(queue_peak[p],uint64_t(k.q.size()));
   pending_peak[p]=std::max(pending_peak[p],pending);
   if(pending)pending_banks++;
  }
  max_owned=std::max(max_owned,uint64_t(std::count(owned.begin(),owned.end(),true)));
  max_pending_banks=std::max(max_pending_banks,uint64_t(pending_banks));
  unsigned servicing=0;
  for(int p=0;p<8;p++){
   Bank&k=banks[p];
   if(k.active){
    need(k.remaining>0&&!k.response,"service phase overlap");servicing++;
    if(k.phase==1){fetch_cycles++;fetch_by_bank[p]++;}
    else{need(k.phase==2,"service phase");extract_cycles++;extract_by_bank[p]++;}
    k.remaining--;
    if(k.remaining==0){
     if(k.phase==1){
      k.cached=k.active->c/16;
      need(k.cached%8==p,"cache owns wrong physical bank");
      for(int local=0;local<16;local++)for(int h=0;h<96;h++)
       k.row[local*96+h]=f.weights.at((k.cached*16+local)*96+h);
      k.phase=2;k.remaining=4;
     }else{
      k.response=*k.active;
      for(int l=0;l<16;l++)k.response_values[l]=k.row[(k.active->c%16)*96+k.active->s*16+l];
      k.active.reset();k.phase=0;
     }
    }
   }
   if(!k.active&&!k.response&&!k.q.empty()){
    k.active=k.q.front();k.q.pop_front();
    if(k.cached!=k.active->c/16){k.phase=1;k.remaining=384;row_fetches++;physical_rows[p]++;}
    else{k.phase=2;k.remaining=4;}
   }
  }
  max_service_banks=std::max(max_service_banks,uint64_t(servicing));
  if(servicing>=2)multi_bank_service_cycles++;
 }
 bool empty()const{
  if(std::any_of(owned.begin(),owned.end(),[](bool x){return x;}))return false;
  for(const auto&k:banks)if(!k.q.empty()||k.active||k.response)return false;
  return true;
 }
};
struct Simulation {
 const Fixture& f;int first;bool shared;
 Vdut m;Memory mem;
 uint64_t cycle=0,last_commit=0,commits=0,integer_checks=0,load_accepts=0,load_active_cycles=0;
 uint64_t output_stalls=0,request_stalls=0;
 uint64_t max_bank_updates=0,multi_bank_update_cycles=0,max_temporal_writes=0,multi_temporal_update_bank_cycles=0;
 uint64_t candidate_temporal_vector_writes=0;
 bool last_load=false,last_source=false,last_start=false,last_done=false;
 std::vector<bool> seen;
 std::optional<Commit> held_commit;
 std::array<std::optional<Request>,8> held_request;
 Simulation(const Fixture&fx,int begin,bool share):f(fx),first(begin),shared(share),mem(fx),seen(4*10*96,false){
  m.clk_core=0;m.rst_core=1;m.mem_req_ready=0;m.mem_rsp_valid=0;
  m.bridge_ready=1;m.commit_ready=1;m.bundle_done_ready=1;
#if PRODUCTION
  m.load_valid=0;m.load_context=0;m.load_tag=0;m.load_group=0;
  m.load_source_active=0;m.load_source_sign=0;m.load_last=0;
#else
  m.start_valid=0;m.start_tag=0;m.share_enable=shared;
  m.source_valid=0;m.source_index=0;m.source_mask=0;
#endif
  for(int b=0;b<8;b++){
   m.mem_rsp_epoch[b]=0;m.mem_rsp_slot[b]=0;m.mem_rsp_generation[b]=0;m.mem_rsp_tag[b]=0;
   for(int l=0;l<16;l++)m.mem_rsp_weight[b][l]=0;
  }
  for(int i=0;i<5;i++){m.clk_core=0;m.eval();m.clk_core=1;m.eval();}
  m.rst_core=0;m.clk_core=0;m.eval();
 }
 Commit current_commit()const{Commit r;r.context=m.commit_context;r.slice=m.commit_slice;r.tag=m.commit_tag;r.terminal=m.commit_terminal;
  for(int l=0;l<16;l++)r.v[l]=signed24(m.commit_accumulator[l]);return r;}
 void consume(const Commit&r){
  int p=int(r.tag>>4),t=int(r.tag&15);
#if PRODUCTION
  need(r.context==p-first,"production context vs tag");
#else
  need((r.tag&15)==0,"FTP tag format");t=r.context;
#endif
  need(p>=first&&p<first+4&&t>=0&&t<10&&r.slice>=0&&r.slice<6,"commit identity");
  need(r.terminal==(r.slice==5),"terminal meaning");
  for(int l=0;l<16;l++){
   int h=r.slice*16+l;int idx=((p-first)*10+t)*96+h;
   need(!seen.at(idx),"duplicate commit");seen[idx]=true;
   if(r.v[l]!=f.truth(p,t,h)){
    std::ostringstream s;s<<"numeric mismatch p="<<p<<" t="<<t<<" h="<<h<<" got="<<r.v[l]<<" expected="<<f.truth(p,t,h);throw std::runtime_error(s.str());}
   integer_checks++;
  }
  commits++;last_commit=cycle+1;
 }
 void step(){
  need(cycle<3000000,"bounded quad timeout");
  m.clk_core=0;m.bridge_ready=(cycle%11!=3);m.commit_ready=(cycle%13!=5);m.bundle_done_ready=1;
  mem.drive(m);m.eval();mem.drive(m);m.eval();
  for(int b=0;b<8;b++){
   if(held_request[b])need((m.mem_req_valid&(1u<<b))&&mem.get(m,b)==*held_request[b],"request changed under backpressure");
   held_request[b].reset();
   if((m.mem_req_valid&(1u<<b))&&!(m.mem_req_ready&(1u<<b))){held_request[b]=mem.get(m,b);request_stalls++;}
  }
  if(held_commit)need(m.commit_valid&&current_commit()==*held_commit,"output changed or withdrawn under backpressure");
  held_commit.reset();
  if(m.commit_valid&&!m.commit_ready){held_commit=current_commit();output_stalls++;}
  if(m.commit_valid&&m.commit_ready)consume(current_commit());
  std::vector<Request> accepted;
  for(int b=0;b<8;b++)if((m.mem_req_valid&m.mem_req_ready)&(1u<<b))accepted.push_back(mem.get(m,b));
  unsigned rsp_accept=m.mem_rsp_valid&m.mem_rsp_ready;
#if !PRODUCTION
  unsigned updating=__builtin_popcount(unsigned(m.debug_bank_update_mask));
  max_bank_updates=std::max(max_bank_updates,uint64_t(updating));
  if(updating>=2)multi_bank_update_cycles++;
  for(int b=0;b<8;b++){
   unsigned writing=__builtin_popcount(unsigned(m.debug_time_write_mask[b]));
   max_temporal_writes=std::max(max_temporal_writes,uint64_t(writing));
   if(writing>=2)multi_temporal_update_bank_cycles++;
   candidate_temporal_vector_writes+=writing;
  }
#endif
  last_done=m.bundle_done_valid&&m.bundle_done_ready;
  last_load=false;last_source=false;last_start=false;
#if PRODUCTION
  last_load=m.load_valid&&m.load_accept;
  if(m.load_valid)load_active_cycles++;
  if(last_load)load_accepts++;
#else
  last_start=m.start_valid&&m.start_ready;
  last_source=m.source_valid&&m.source_ready;
  if(m.source_valid)load_active_cycles++;
  if(last_source)load_accepts++;
#endif
  m.clk_core=1;m.eval();cycle++;
  mem.edge(accepted,rsp_accept);
  // The behavioral registered producer must withdraw an accepted response
  // on this same edge, like the NBA updates in the original SV memory TB.
  // Inspect only after those input updates have settled combinationally.
  mem.drive(m);m.eval();
  if(m.protocol_error||m.numeric_overflow){
   std::ostringstream e;e<<"DUT flag cycle="<<cycle<<" protocol="<<int(m.protocol_error)<<" overflow="<<int(m.numeric_overflow)
    <<" reqs="<<mem.requests<<" rsps="<<mem.responses<<" load_accepts="<<load_accepts;
#if PRODUCTION
   e<<" load_valid="<<int(m.load_valid)<<" load_context="<<int(m.load_context)<<" load_group="<<int(m.load_group)
    <<" load_last="<<int(m.load_last)<<" load_accept="<<int(m.load_accept)<<" stale="<<int(m.stale_response_seen);
#endif
   for(const auto&r:accepted)e<<" accepted_req_port="<<r.b<<" c="<<r.c<<" slot="<<r.slot<<" gen="<<r.gen;
   for(const auto&k:mem.banks)if(k.response)e<<" response_port="<<k.response->b<<" slot="<<k.response->slot<<" gen="<<k.response->gen;
   throw std::runtime_error(e.str());
  }
  need(!Verilated::gotFinish(),"unexpected RTL finish");
 }
 void wait_until(uint64_t c){while(cycle<c)step();}
 void execute(){
#if PRODUCTION
  for(int t=0;t<10;t++){
   wait_until(uint64_t(t+1)*12);
   for(int ctx=0;ctx<4;ctx++){
    std::vector<std::pair<int,int>> desc;
    for(int g=0;g<24;g++){int bits=0;for(int s=0;s<16;s++)if((f.mask(first+ctx,g*16+s)>>t)&1)bits|=1<<s;
     if(bits)desc.emplace_back(g,bits);}
    if(desc.empty())desc.emplace_back(0,0);
    for(size_t i=0;i<desc.size();i++){
     m.load_valid=1;m.load_context=ctx;m.load_tag=((first+ctx)<<4)|t;
     m.load_group=desc[i].first;m.load_source_active=desc[i].second;m.load_source_sign=0;m.load_last=(i+1==desc.size());
     do{step();}while(!last_load);m.load_valid=0;
    }
   }
   do{step();}while(!last_done);
  }
#else
  wait_until(120);
  for(int p=first;p<first+4;p++){
   m.start_valid=1;m.start_tag=p<<4;do{step();}while(!last_start);m.start_valid=0;
   for(int c=0;c<384;c++){
    m.source_valid=1;m.source_index=c;m.source_mask=f.mask(p,c);
    do{step();}while(!last_source);m.source_valid=0;
   }
   do{step();}while(!last_done);
  }
#endif
  need(commits==240&&integer_checks==3840,"missing output cardinality");
  need(std::all_of(seen.begin(),seen.end(),[](bool x){return x;}),"missing output coordinates");
  need(mem.empty()&&mem.requests==mem.responses,"memory not drained");
  need(mem.fetch_cycles==mem.row_fetches*384&&mem.extract_cycles==mem.requests*4,"service cycle conservation");
  for(int b=0;b<8;b++)need(mem.fetch_by_bank[b]==mem.physical_rows[b]*384&&mem.extract_by_bank[b]==mem.physical_vectors[b]*4,"per-bank service conservation");
#if !PRODUCTION
  uint64_t direct_adds=0,saving=0,expected_reduction=0;
  for(int p=first;p<first+4;p++){
   for(int t=0;t<10;t++){
    int count=0,live_banks=0;
    for(int b=0;b<8;b++){
     int local=0;for(int c=0;c<384;c++)if((c/16)%8==b&&((f.mask(p,c)>>t)&1))local++;
     count+=local;live_banks+=local>0;
    }
    direct_adds+=uint64_t(std::max(count-1,0))*96;
    expected_reduction+=uint64_t(std::max(live_banks-1,0))*96;
   }
   for(int b=0;b<8;b++){
    int selected=0;for(int c=0;c<384;c++)if((c/16)%8==b&&f.mask(p,c)==0x140)selected++;
    saving+=uint64_t(std::max(selected-1,0))*96;
   }
  }
  need(m.debug_additions==direct_adds-(shared?saving:0),"independent total-addition conservation including reduction");
  need(m.debug_reduction_additions==expected_reduction,"independent final reduction addition count");
  need(m.debug_source_loads==4*384,"all source masks actually loaded");
  need(mem.max_owned>1&&mem.max_service_banks>1&&max_bank_updates>1&&max_temporal_writes>1,"parallelism not exercised");
#endif
 }
 void report(std::ostream&o)const{
  o<<"{\"first_position\":"<<first<<",\"positions\":4,\"end_to_end_cycles\":"<<last_commit
   <<",\"host_driver_last_cycle\":"<<cycle<<",\"source_transfer_cycles_model\":120"
   <<",\"descriptor_or_signature_accepts\":"<<load_accepts<<",\"load_valid_cycles\":"<<load_active_cycles
   <<",\"canonical_row_fetches\":"<<mem.row_fetches<<",\"canonical_fetch_bank_lane_cycles\":"<<mem.fetch_cycles
   <<",\"vector_extract_bank_lane_cycles\":"<<mem.extract_cycles<<",\"logical_vector_requests\":"<<mem.requests
   <<",\"logical_vector_responses\":"<<mem.responses<<",\"response_hold_cycles\":"<<mem.response_hold_cycles
   <<",\"request_stall_lane_cycles\":"<<request_stalls<<",\"output_stall_cycles\":"<<output_stalls
   <<",\"committed_vectors\":"<<commits<<",\"integer_checks\":"<<integer_checks<<",\"errors\":0,\"row_fetches_by_canonical_bank\":[";
  for(int b=0;b<8;b++){if(b)o<<",";o<<mem.physical_rows[b];}o<<"]";
  const auto array_field=[&](const char*name,const std::array<uint64_t,8>&a){o<<",\""<<name<<"\":[";for(int b=0;b<8;b++){if(b)o<<",";o<<a[b];}o<<"]";};
  array_field("vector_requests_by_canonical_bank",mem.physical_vectors);
  array_field("fetch_lane_cycles_by_canonical_bank",mem.fetch_by_bank);
  array_field("extract_lane_cycles_by_canonical_bank",mem.extract_by_bank);
  array_field("queue_peak_by_canonical_bank",mem.queue_peak);
  array_field("pending_peak_by_canonical_bank",mem.pending_peak);
  o<<",\"max_owned_requests\":"<<mem.max_owned<<",\"max_service_banks\":"<<mem.max_service_banks
   <<",\"multi_bank_service_cycles\":"<<mem.multi_bank_service_cycles<<",\"max_pending_banks\":"<<mem.max_pending_banks
   <<",\"max_request_accepts_per_cycle\":"<<mem.max_request_accepts<<",\"max_response_accepts_per_cycle\":"<<mem.max_response_accepts;
#if !PRODUCTION
  o<<",\"rtl_scalar_additions\":"<<m.debug_additions<<",\"rtl_final_reduction_additions\":"<<m.debug_reduction_additions
   <<",\"max_bank_updates_per_cycle\":"<<max_bank_updates<<",\"multi_bank_update_cycles\":"<<multi_bank_update_cycles
   <<",\"max_temporal_vectors_written_per_bank_cycle\":"<<max_temporal_writes
   <<",\"multi_temporal_update_bank_cycles\":"<<multi_temporal_update_bank_cycles
   <<",\"temporal_vector_writes\":"<<candidate_temporal_vector_writes;
#endif
  o<<"}";
 }
};
int main(int argc,char**argv){
 try{
  need(argc==4,"arguments: fixture_dir shared_mode output_json");Verilated::commandArgs(argc,argv);
  Fixture f(argv[1]);bool shared=std::stoi(argv[2])!=0;
  std::ifstream existing(argv[3]);need(!existing.good(),"output exists");existing.close();
  std::ostringstream records;uint64_t cycles=0,rows=0,requests=0,checks=0;
  for(int p=f.first;p<f.first+f.n;p+=4){
   Simulation s(f,p,shared);s.execute();if(p!=f.first)records<<",";s.report(records);
   cycles+=s.last_commit;rows+=s.mem.row_fetches;requests+=s.mem.requests;checks+=s.integer_checks;
   std::cerr<<"PASS quad="<<p<<" cycles="<<s.last_commit<<" rows="<<s.mem.row_fetches<<" requests="<<s.mem.requests<<" checks="<<s.integer_checks<<"\n";
  }
  std::ofstream o(argv[3]);need(bool(o),"output open");
  o<<"{\"status\":\"PASS_BOUNDED_VERILATOR_RTL_WITH_BEHAVIORAL_IO_ONLY\",\"axis\":\""
   <<(PRODUCTION?"production_m2018_schedule1":shared?"full_T_parallel_shared_static_layer_mode":"full_T_parallel_direct")
   <<"\",\"positions\":32,\"quad_workloads\":8,\"complete_frame\":false,\"end_to_end_cycle_sum\":"<<cycles
   <<",\"canonical_row_fetches\":"<<rows<<",\"logical_vector_requests\":"<<requests
   <<",\"integer_checks\":"<<checks<<",\"errors\":0,\"quads\":["<<records.str()<<"]}\n";
  return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}
}
