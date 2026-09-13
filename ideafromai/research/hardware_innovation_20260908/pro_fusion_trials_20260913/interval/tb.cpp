#include "Vinterval_tile.h"
#include "verilated.h"
#include <array>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

template<size_t N> std::vector<std::array<uint32_t,N>> load_hex(const char* name) {
  std::ifstream in(name); if(!in) throw std::runtime_error(name);
  std::vector<std::array<uint32_t,N>> values; std::string line;
  while(in>>line) {
    std::array<uint32_t,N> x{};
    for(size_t i=0;i<N && line.size()>i*8;i++) {
      size_t end=line.size()-i*8, begin=end>8?end-8:0;
      x[i]=static_cast<uint32_t>(std::stoul(line.substr(begin,end-begin),nullptr,16));
    }
    values.push_back(x);
  }
  return values;
}
int main(int argc,char** argv) {
  Verilated::commandArgs(argc,argv);
  Vinterval_tile d;
  auto s=load_hex<1>("spikes.mem"); auto w=load_hex<8>("weights.mem");
  auto gold=load_hex<6>("golden.mem");
  if(s.size()!=4320 || w.size()!=864 || gold.size()!=200) throw std::runtime_error("memory size");
  std::ofstream csv("rtl_results.csv");
  csv<<"case,mode,stress,selected_endpoint,cycles,clear,source_reads,source_stalls,weight_reads,weight_stalls,event_add,prefix,round,output_beats,output_stalls,selection_cycles,mismatched_beats\n";
  int case_id=0, stress=0, errors=0, checked=0, run_errors=0, received=0;
  bool held=false; std::array<uint32_t,6> held_data{}; int held_index=0;
  // Only ROM/port stimulus and expected-value checking live in the TB.
  // No endpoint construction, partial sum, clearing, prefix or RNE is done here.
  auto tick=[&]() {
    d.clk=0; d.eval();
    d.source_valid=d.source_req && (!stress || d.cycles%11!=3);
    d.source_mask=s.at(case_id*864+d.source_k)[0];
    d.weight_valid=d.weight_req && (!stress || (d.cycles%7!=2 && d.cycles%7!=3));
    for(int i=0;i<8;i++) d.weight_data[i]=w.at(d.weight_k)[i];
    d.out_ready=!stress || (d.cycles%5!=1 && d.cycles%5!=2);
    d.eval();
    if(!d.rst && d.busy) {
      if(held) {
        bool mismatch=!d.out_valid || d.out_index!=held_index;
        for(int i=0;i<6;i++) mismatch|=d.out_data[i]!=held_data[i];
        if(mismatch) throw std::runtime_error("output changed under backpressure");
      }
      if(d.out_valid && d.out_ready) {
        bool bad=d.out_index!=received;
        for(int i=0;i<6;i++) bad|=d.out_data[i]!=gold.at(case_id*40+d.out_index)[i];
        if(bad) { errors++; run_errors++; std::cerr<<"Mismatch case "<<case_id<<" mode "<<int(d.mode)<<" group "<<int(d.out_index)<<"\n"; }
        checked++; received++;
      }
      held=d.out_valid && !d.out_ready; held_index=d.out_index;
      for(int i=0;i<6;i++) held_data[i]=d.out_data[i];
    } else held=false;
    d.clk=1; d.eval();
  };
  d.rst=1; d.start=0; d.mode=0; tick();tick();tick();d.rst=0;
  // Thirty full commands without resetting the state between commands.
  for(case_id=0;case_id<5;case_id++) for(stress=0;stress<2;stress++) for(int mode=0;mode<3;mode++) {
    d.mode=mode; run_errors=0;received=0; d.start=1;tick();d.start=0;
    int guard=0; while(!d.done && guard++<100000) tick();
    if(!d.done || d.output_beats!=40 || received!=40) throw std::runtime_error("timeout or missing outputs");
    int sum=d.clear_cycles+d.source_reads+d.source_stalls+d.weight_reads+d.weight_stalls+
      d.event_add_cycles+d.prefix_cycles+d.round_cycles+d.output_beats+d.output_stalls+1+(mode==2);
    if(int(d.cycles)!=sum) throw std::runtime_error("cycle accounting mismatch");
    csv<<case_id<<','<<mode<<','<<stress<<','<<int(d.selected_endpoint)<<','<<d.cycles<<','<<d.clear_cycles<<','
       <<d.source_reads<<','<<d.source_stalls<<','<<d.weight_reads<<','<<d.weight_stalls<<','<<d.event_add_cycles<<','
       <<d.prefix_cycles<<','<<d.round_cycles<<','<<d.output_beats<<','<<d.output_stalls<<','<<d.selection_cycles<<','<<run_errors<<'\n';
    std::cout<<"RUN case="<<case_id<<" mode="<<mode<<" stress="<<stress<<" selected="<<int(d.selected_endpoint)
             <<" cycles="<<d.cycles<<" bad="<<run_errors<<'\n';
  }
  d.final();
  if(errors) throw std::runtime_error("functional mismatches");
  std::cout<<"PASS "<<checked*8<<" continuous signed24 values; "<<checked<<" beats; 30 commands without intervening reset\n";
  return 0;
}
