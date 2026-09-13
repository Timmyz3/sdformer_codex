#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
using U=uint64_t;
struct Totals {
 U psum_r=0,psum_w=0,dram_r=0,dram_w=0,weight_r=0,source_r=0;
 U residual_adds=0,global_adds=0,parent_reads=0,parent_writes=0;
 U global_hits=0,global_misses=0,dirty_evictions=0,stream_updates=0;
 U detector_cycles=0,detector_tiles=0,weight_refills=0,completed_rows=0;
 U clear_loads=0,zero_K_rows=0,metadata_bytes=0,cohorts=0;
 int parent_peak=0,global_peak=0,combined_peak=0;
};
struct Engine {
 int bytes,cap,policy,vector_bytes,beats,pin_budget;
 U now=0,rport=0,wport=0,wread=0,alu=0,dram=0,xready=0,detector=0;
 int parents_live=0,global_live=0,head=-1,tail=-1;
 std::array<bool,256> valid{},ever{},pin{};
 std::array<U,256> gready{},backready{},pready{},touch{};
 std::array<unsigned,256> version{},gversion{},backversion{};
 std::array<int,256> prev{},next{};
 std::array<unsigned,1800> complete{};
 Totals s;
 Engine(int b,int p):bytes(b),policy(p){
  vector_bytes=128*bytes;beats=vector_bytes/128;
  // 96KiB includes P, G and writeback-transfer vector registers.
  cap=(98304-3*vector_bytes)/vector_bytes;pin_budget=std::max(0,cap-48);
  prev.fill(-1);next.fill(-1);
 }
 U read(U t){U a=std::max(t,rport);rport=a+1;s.psum_r+=vector_bytes;return a+1;}
 U write(U t){U a=std::max(t,wport);wport=a+1;s.psum_w+=vector_bytes;return a+1;}
 U add(U t){U a=std::max(t,alu);alu=a+1;return a+1;}
 U dr(U t,int n){U a=std::max(t,dram);dram=a+(n+127)/128;s.dram_r+=n;return dram;}
 U dw(U t,int n){U a=std::max(t,dram);dram=a+(n+127)/128;s.dram_w+=n;return dram;}
 void peak(){s.parent_peak=std::max(s.parent_peak,parents_live);s.global_peak=std::max(s.global_peak,global_live);
  s.combined_peak=std::max(s.combined_peak,parents_live+global_live);assert(parents_live+global_live<=cap);}
 void unlink(int r){if(prev[r]>=0)next[prev[r]]=next[r];else head=next[r];
  if(next[r]>=0)prev[next[r]]=prev[r];else tail=prev[r];prev[r]=next[r]=-1;}
 void touch_global(int r){if(head==r||prev[r]>=0||next[r]>=0)unlink(r);
  prev[r]=tail;if(tail>=0)next[tail]=r;else head=r;tail=r;}
 U evict(U t){assert(head>=0);int r=head;
  U rd=read(std::max({t,xready,gready[r]}));U end=dw(rd,vector_bytes);
  xready=end;backready[r]=end;valid[r]=false;unlink(r);--global_live;++s.dirty_evictions;
  assert(gversion[r]==version[r]);backversion[r]=gversion[r];
  return rd;
 }
 U room(U t){while(parents_live+global_live>=cap)t=std::max(t,evict(t));return t;}
 U global_load(int r,U t,bool keep){
  if(valid[r]){assert(gversion[r]==version[r]);++s.global_hits;touch_global(r);return read(std::max(t,gready[r]));}
  ++s.global_misses;
  if(keep){t=room(t);valid[r]=true;++global_live;touch_global(r);peak();}
  if(ever[r]){assert(backversion[r]==version[r]);return dr(std::max(t,backready[r]),vector_bytes);}
  return t; // Known zero at first update; no read or materialized clear.
 }
 U spill_current(int r,U t){U copied=std::max(t,xready)+1;
  backready[r]=dw(copied,vector_bytes);backversion[r]=version[r];xready=backready[r];return copied;
 }
 U preload(int rows,U t){
  // Current/next 16x128 weight buffers consume32KiB at INT64, <=that at8bit.
  U act=dr(t,rows*2);s.source_r+=rows*2;
  U weights=dr(act,16*vector_bytes);s.weight_refills+=16*vector_bytes;
  // A conservative legal TCAM fill then detector schedule, double-buffered.
  U dstart=std::max(detector,act+rows);detector=dstart+rows+4;
  s.detector_cycles+=rows+4;++s.detector_tiles;
  s.metadata_bytes+=rows*5;
  return std::max(weights,detector);
 }
 void reset(int m,int rows){
  assert(parents_live==0&&global_live==0);valid.fill(false);ever.fill(false);pin.fill(false);
  gready.fill(0);backready.fill(0);pready.fill(0);prev.fill(-1);next.fill(-1);head=tail=-1;
  version.fill(0);gversion.fill(0);backversion.fill(0);
  if(policy==1){for(int r=0;r<std::min(rows,pin_budget);++r)pin[r]=true;}
  if(policy==2){
   if(rows<=pin_budget){for(int r=0;r<rows;++r)pin[r]=true;}
   else {int count=0;for(int r=0;r<rows;++r)if((m+r)%10==0&&r+10<=rows&&count+10<=pin_budget){
     for(int j=0;j<10;++j)pin[r+j]=true;count+=10;}}
  }
 }
 void tile(int m,int n,int k,int rows,const int16_t* par,const uint16_t* residual,const uint16_t* order){
  std::array<int,256> uses{};for(int r=0;r<rows;++r)if(par[r]>=0)++uses[par[r]];
  for(int ix=0;ix<rows;++ix){int r=order[ix],p=par[r];unsigned bits=residual[r];int nnz=__builtin_popcount(bits);
   if(p<0&&!nnz){++s.zero_K_rows;if(k==431)finish(m,n,r);continue;}
   U begin=now,pr=begin;
   if(p>=0){assert(pready[p]>0);pr=read(std::max(pr,pready[p]));++s.parent_reads;
    if(--uses[p]==0){--parents_live;pready[p]=0;}}
   bool keep=(policy==0)||pin[r];bool was_ever=ever[r];U gr=global_load(r,begin,keep);
   U local=pr;bool first=(p<0);
   while(bits){int bit=__builtin_ctz(bits);(void)bit;bits&=bits-1;
    U wr=std::max(wread,local);wread=wr+1;s.weight_r+=vector_bytes;
    if(first){local=wr+1;first=false;++s.clear_loads;}
    else {local=add(wr+1);++s.residual_adds;}
   }
   U psave=local;
   if(uses[r]>0){psave=room(psave);++parents_live;peak();pready[r]=write(psave);psave=pready[r];++s.parent_writes;}
   U result=std::max(local,gr);if(was_ever){result=add(result);++s.global_adds;}
   ever[r]=true;++version[r];
   if(keep){gready[r]=write(result);gversion[r]=version[r];now=std::max(psave,gready[r]);}
   else {++s.stream_updates;now=std::max(psave,spill_current(r,result));}
   if(k==431)finish(m,n,r);
  }
  assert(parents_live==0);for(int r=0;r<rows;++r)assert(uses[r]==0);
 }
 void finish(int m,int n,int r){
  // Exact final-K certificate. Parent's current-K value has an independent
  // lifetime; finishing global output never destroys a live parent.
  if(valid[r]){U rr=read(std::max({now,xready,gready[r]}));backready[r]=dw(rr,vector_bytes);
   assert(gversion[r]==version[r]);backversion[r]=gversion[r];
   xready=backready[r];valid[r]=false;unlink(r);--global_live;now=std::max(now,rr);}
  else if(!ever[r]){now=spill_current(r,now);ever[r]=true;}
  assert(backversion[r]==version[r]);
  ++s.completed_rows;int logical=m+r;int p=logical/10,t=logical%10,slot=p*6+n;
  complete[slot]|=1u<<t;if(complete[slot]==1023){++s.cohorts;complete[slot]=2047;}
 }
 void emit(const std::string& name){
  now=std::max({now,dram,wport,rport,xready});
  std::cout<<"{\"policy\":\""<<name<<"\",\"scalar_bytes\":"<<bytes<<",\"cycles\":"<<now
   <<",\"capacity_bytes\":98304,\"temporary_vector_bytes\":"<<3*vector_bytes<<",\"cache_vector_slots\":"<<cap
   <<",\"reserved_parent_slots\":48,\"generic_pin_budget\":"<<pin_budget;
  #define F(x) std::cout<<",\""#x"\":"<<s.x
  F(psum_r);F(psum_w);F(dram_r);F(dram_w);F(weight_r);F(source_r);F(residual_adds);F(global_adds);
  F(parent_reads);F(parent_writes);F(global_hits);F(global_misses);F(dirty_evictions);F(stream_updates);
  F(detector_cycles);F(detector_tiles);F(weight_refills);F(completed_rows);F(clear_loads);F(zero_K_rows);
  F(metadata_bytes);F(cohorts);F(parent_peak);F(global_peak);F(combined_peak);
  #undef F
  std::cout<<"}";
 }
};
int main(int argc,char**argv){assert(argc==2);std::ifstream f(argv[1],std::ios::binary);assert(f.good());
 const int count=3000*432;std::vector<int16_t> p(count);std::vector<uint16_t> r(count),o(count);
 f.read((char*)p.data(),count*2);f.read((char*)r.data(),count*2);f.read((char*)o.data(),count*2);assert(f.good());
 std::cout<<"[";bool comma=false;
 for(int b:{1,8})for(int policy=0;policy<3;++policy){Engine e(b,policy);int off=0;
  for(int m=0;m<3000;m+=256){int rows=std::min(256,3000-m);
   for(int n=0;n<6;++n){e.reset(m,rows);U ready=e.preload(rows,e.now);
    for(int k=0;k<432;++k){e.now=std::max(e.now,ready);U next=0;if(k<431)next=e.preload(rows,e.now);
     e.tile(m,n,k,rows,p.data()+off+k*rows,r.data()+off+k*rows,o.data()+off+k*rows);ready=next;}
    assert(e.global_live==0&&e.parents_live==0);}
   off+=rows*432;
  }
  assert(e.s.completed_rows==18000&&e.s.cohorts==1800);
  if(comma)std::cout<<",";comma=true;e.emit(policy==0?"ordinary_unified_LRU":policy==1?"ordinary_resident_set":"T10_cohort_resident_set");
 }
 std::cout<<"]\n";
}
