#include "Vpe_compare_top.h"
#include "verilated.h"
#include <array>
#include <vector>
#include <random>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <string>
struct Case {
 int k; bool q8; std::vector<int> x,w; std::array<int32_t,16> expected;
 std::string source;
};
static void need(bool x,const std::string& msg){if(!x)throw std::runtime_error(msg);}
static int low4(int v){return v>=0?(v/16)*16:-(((-v+15)/16)*16);}
static void golden(Case& c){
 c.expected.fill(0);
 for(int o=0;o<16;o++)for(int j=0;j<c.k;j++)c.expected[o]+=c.x[j]*(c.q8?c.w[o*c.k+j]:low4(c.w[o*c.k+j]));
}
static std::vector<Case> vectors(){
 std::mt19937 rng(20261020);std::vector<Case> v;
 for(int k:{1,16,96,128,129,256,384,768,864,1536,3072})for(bool q8:{false,true})for(int pattern=0;pattern<5;pattern++){
  Case c{k,q8,std::vector<int>(k),std::vector<int>(16*k),{},"synthetic"};
  for(int j=0;j<k;j++)c.x[j]=pattern==0?0:pattern==1?1:int(rng()%2);
  for(int j=0;j<16*k;j++)c.w[j]=pattern==1?-128:pattern==2?127:pattern==3?-1:int(rng()%255)-127;
  golden(c);v.push_back(std::move(c));
 }
 return v;
}
static void append_actual(std::vector<Case>& cases,const char* path){
 if(!path)return;std::ifstream f(path);need(bool(f),"Actual vectors readable");int count;f>>count;
 for(int z=0;z<count;z++){
  int k,q8;f>>k>>q8;Case c{k,bool(q8),std::vector<int>(k),std::vector<int>(k*16),{},"audited_R26"};
  for(int&x:c.x)f>>x;for(int&w:c.w)f>>w;
  need(bool(f),"Actual complete payload");golden(c);cases.push_back(std::move(c));
 }
}
static void tick(Vpe_compare_top& d){d.clk=0;d.eval();d.clk=1;d.eval();d.clk=0;d.eval();}
static void reset(Vpe_compare_top&d,bool parallel){
 d.in_valid=0;d.out_ready=0;d.row_start=0;d.mode_q8=0;d.k_chunks=1;d.parallel_mode=parallel;d.rst_n=0;
 for(int j=0;j<4;j++)d.activation[j]=0;for(int j=0;j<64;j++)d.weights[j]=0;
 tick(d);tick(d);d.rst_n=1;d.eval();
}
struct Receipt{uint64_t cycles=0,beats=0,retired=0,blocked=0,bubbles=0,values=0;};
static Receipt run(Vpe_compare_top&d,const std::vector<Case>&cases,bool parallel,bool stalls){
 reset(d,parallel);std::mt19937 rng(20261021+parallel+2*stalls);
 size_t send=0,receive=0;int beat=0;Receipt r;std::array<uint32_t,16> held{};bool holding=false;
 while(receive<cases.size()){
  need(r.cycles<2000000,"Bounded completion");d.out_ready=!stalls||rng()%5!=0;
  bool offer=send<cases.size()&&(!stalls||rng()%7!=0);d.in_valid=offer;
  if(send<cases.size()){
   const Case&c=cases[send];int width=parallel?16:128,chunks=(c.k+width-1)/width;
   int plane=parallel?0:(c.q8?0:4)+beat/chunks,chunk=parallel?beat:beat%chunks;
   d.row_start=beat==0;d.mode_q8=c.q8;d.k_chunks=chunks;
   for(int j=0;j<4;j++)d.activation[j]=0;for(int j=0;j<64;j++)d.weights[j]=0;
   for(int j=0;j<width;j++){
    int idx=chunk*width+j;if(idx>=c.k)continue;
    d.activation[j/32]|=uint32_t(c.x[idx])<<(j%32);
    for(int o=0;o<16;o++){
     uint32_t q=uint8_t(parallel&&!c.q8?low4(c.w[o*c.k+idx]):c.w[o*c.k+idx]);
     if(parallel){int bit=o*128+j*8;d.weights[bit/32]|=q<<(bit%32);}
     else {int bit=o*128+j;d.weights[bit/32]|=((q>>plane)&1u)<<(bit%32);}
    }
   }
  }else d.row_start=0;
  d.eval();
  if(holding){need(d.out_valid,"Output valid persists under backpressure");for(int o=0;o<16;o++)need(d.result[o]==held[o],"Output payload stable");}
  bool accept=d.in_valid&&d.in_ready,retire=d.out_valid&&d.out_ready;
  if(d.in_valid&&!d.in_ready)r.blocked++;if(!d.in_valid)r.bubbles++;
  if(retire){
   const auto&c=cases[receive];int chunks=(c.k+(parallel?15:127))/(parallel?16:128);
   int expected_beats=parallel?chunks:chunks*(c.q8?8:4);
   need(d.row_beats==uint32_t(expected_beats),"Per-row actual accepted beat count");
   for(int o=0;o<16;o++){need(int32_t(d.result[o])==c.expected[o],"Independent signed integer golden mismatch");r.values++;}
   receive++;r.retired++;
  }
  holding=d.out_valid&&!d.out_ready;if(holding)for(int o=0;o<16;o++)held[o]=d.result[o];
  if(accept){
   const auto&c=cases[send];int chunks=(c.k+(parallel?15:127))/(parallel?16:128);
   if(!parallel){need(d.expected_plane==uint32_t((c.q8?0:4)+beat/chunks),"Low planes never requested forQ4");need(d.expected_chunk==uint32_t(beat%chunks),"Exact reduction chunk");}
   else need(d.expected_chunk==uint32_t(beat),"Ordinary reduction chunk");
   r.beats++;beat++;if(beat==chunks*(parallel?1:(c.q8?8:4))){beat=0;send++;}
  }
  tick(d);r.cycles++;need(!d.fault,"No normal protocol fault");
 }
 need(send==cases.size(),"Every descriptor fully consumed");need(d.accepted_beats==r.beats,"Hardware total beat counter");
 if(!stalls)need(r.cycles==r.beats+1,"One accepted beat/tick steady stream with one final retire tick");
 return r;
}
static void invalid(Vpe_compare_top&d,bool p){
 reset(d,p);d.in_valid=1;d.row_start=0;d.out_ready=1;tick(d);
 need(d.fault&&!d.out_valid&&!d.in_ready,"Missing descriptor rejected");
 reset(d,p);d.row_start=1;d.k_chunks=0;d.in_valid=1;tick(d);
 need(d.fault&&!d.out_valid,"Zero chunks rejected");
 reset(d,p);d.row_start=1;d.k_chunks=2;d.mode_q8=1;d.in_valid=1;tick(d);
 d.row_start=1;tick(d);need(d.fault&&!d.out_valid,"Mid-row restart rejected");
}
int main(int argc,char**argv){
 try{
  Verilated::commandArgs(argc,argv);Vpe_compare_top d;auto cases=vectors();if(argc>1)append_actual(cases,argv[1]);
  std::cout<<"{\"schema\":\"flow_r30_integer_PE_RTL_v1\",\"status\":\"PASS\",\"case_count\":"<<cases.size()<<",\"runs\":[";
  bool first=true;for(bool p:{false,true})for(bool stall:{false,true}){
   auto r=run(d,cases,p,stall);if(!first)std::cout<<",";first=false;
   std::cout<<"{\"engine\":\""<<(p?"ordinary_gateadd16":"binary_popcount128")<<"\",\"stalls\":"<<(stall?"true":"false")<<",\"cycles\":"<<r.cycles<<",\"accepted_beats\":"<<r.beats<<",\"retired_rows\":"<<r.retired<<",\"checked_int32_values\":"<<r.values<<",\"input_backpressure_ticks\":"<<r.blocked<<"}";
  }
  invalid(d,false);invalid(d,true);std::cout<<"],\"malformed_protocol_rejections\":6,\"full_BN_or_network\":false,\"SRAM_or_scale\":false}\n";return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}
}
