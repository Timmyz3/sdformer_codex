#include "Vpair_execution.h"
#include "verilated.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<uint32_t> read_hex(const std::string &p) {
  std::ifstream f(p); if(!f) throw std::runtime_error(p);
  std::vector<uint32_t> v; uint64_t x;
  while(f >> std::hex >> x) v.push_back(static_cast<uint32_t>(x));
  return v;
}
int main(int argc,char **argv) {
  Verilated::commandArgs(argc,argv);
  if(argc!=4) return 2;
  const std::string dir=argv[1]; const int mode=std::stoi(argv[2]), stall=std::stoi(argv[3]);
  const auto source=read_hex(dir+"/source.hex"),map=read_hex(dir+"/mapping.hex"),
             weights=read_hex(dir+"/weights.hex"),gold=read_hex(dir+"/gold.hex");
  if(source.size()!=864 || map.size()!=432 || weights.size()%432!=0 ||
     gold.size()!=20*(weights.size()/432)) return 3;
  Vpair_execution d;
  auto tick=[&](){d.clk=0;d.eval();d.clk=1;d.eval();d.clk=0;d.eval();};
  d.reset_n=0; d.start=0;d.map_we=0;d.source_valid=0;d.weight_ready=0;
  d.weight_valid=0;d.result_ready=0;d.mode=mode;tick();tick();d.reset_n=1;
  for(size_t i=0;i<map.size();++i) {d.map_we=1;d.map_addr=i;d.map_data=map[i];tick();}
  d.map_we=0;d.start=1;tick();d.start=0;
  size_t si=0,outputs=0;bool pending=false;int remaining=0;uint32_t response=0;
  unsigned input_stall=0,output_stall=0,request_stall=0;
  for(unsigned n=0;n<2000000;++n) {
    d.source_valid=(si<source.size() && (!stall || n%11!=3));
    d.source_data=(si<source.size()?source[si]:0);
    d.weight_ready=(!pending && (!stall || n%5!=2));
    d.weight_valid=(pending && remaining==0);d.weight_data=response;
    d.result_ready=(!stall || n%7!=1);
    d.clk=0;d.eval();
    const bool take_source=d.source_valid && d.source_ready;
    const bool take_req=d.weight_req && d.weight_ready;
    const bool sent_response=d.weight_valid;
    const uint32_t request_addr=d.weight_addr;
    if(d.source_ready && !d.source_valid) ++input_stall;
    if(d.weight_req && !d.weight_ready) ++request_stall;
    if(d.result_valid && !d.result_ready) ++output_stall;
    if(d.result_valid && d.result_ready) {
      if(d.result_addr!=outputs || d.result_data!=gold.at(outputs)) {
        std::cerr<<"Mismatch at "<<outputs<<" addr="<<d.result_addr<<" got="
          <<static_cast<int32_t>(d.result_data)<<" expected="
          <<static_cast<int32_t>(gold.at(outputs))<<"\n";return 4;
      }
      ++outputs;
    }
    d.clk=1;d.eval();d.clk=0;d.eval();
    if(take_source) ++si;
    if(sent_response) pending=false;
    else if(pending && remaining>0) --remaining;
    if(take_req) {if(pending) return 5;pending=true;remaining=stall?2:1;response=weights.at(request_addr);}
    if(d.done) {
      if(outputs!=gold.size() || si!=source.size()) return 6;
      std::cout<<"{\"mode\":"<<mode<<",\"stall\":"<<stall
        <<",\"cycles\":"<<d.cycles<<",\"weight_words\":"<<d.weight_words
        <<",\"issue\":"<<d.issue_cycles<<",\"replay\":"<<d.replay_cycles
        <<",\"outputs\":"<<outputs<<",\"input_stall\":"<<input_stall
        <<",\"output_stall\":"<<output_stall<<",\"request_stall\":"<<request_stall<<"}\n";
      d.final();return 0;
    }
  }
  std::cerr<<"Timeout\n";return 7;
}
