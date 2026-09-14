#ifdef OS_ARM
#include "Vos_core.h"
using Core = Vos_core;
#else
#include "Vdirect_core.h"
using Core = Vdirect_core;
#endif
#include "verilated.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
double sc_time_stamp(){return 0;}
static std::vector<uint32_t> readhex(const std::string&p){std::ifstream f(p);if(!f)throw std::runtime_error(p);std::vector<uint32_t>a;uint64_t x;while(f>>std::hex>>x)a.push_back(uint32_t(x));return a;}
int main(int argc,char**argv){try{
 Verilated::commandArgs(argc,argv);if(argc!=4)return 2;
 const std::string root=argv[1];int stall=std::stoi(argv[3]);
 std::ifstream manifest(argv[2]);std::vector<std::string>fixtures;std::string path;while(manifest>>path)fixtures.push_back(path);if(fixtures.empty())return 3;
 auto w=readhex(root+"/parameters/weight.hex");if(w.size()!=82944)return 4;
 Core d;auto tick=[&](){d.clk=0;d.eval();d.clk=1;d.eval();d.clk=0;d.eval();};
 d.reset_n=0;d.start=0;d.cfg_valid=0;d.source_allow=1;d.weight_allow=1;d.result_ready=1;tick();tick();d.reset_n=1;
 uint32_t data[8]={0};unsigned cfg_cycles=0;
 auto cfg=[&](int kind,int addr){d.cfg_valid=1;d.cfg_kind=kind;d.cfg_addr=addr;for(int i=0;i<8;i++)d.cfg_data[i]=data[i];tick();cfg_cycles++;};
 for(unsigned a=0;a<10368;a++){for(int i=0;i<8;i++)data[i]=w[a*8+i];cfg(4,a);}
 d.cfg_valid=0;unsigned static_cycles=cfg_cycles;
 for(unsigned command=0;command<2*fixtures.size();command++){
  const auto&dir=fixtures[command%fixtures.size()];auto src=readhex(dir+"/source.hex"),origin=readhex(dir+"/origin.hex"),gold=readhex(dir+"/gold.hex");
  if(src.size()!=1536||origin.size()!=2||gold.size()!=3840)return 5;
  cfg_cycles=0;for(unsigned a=0;a<1536;a++){data[0]=src[a];cfg(0,a);}
  data[0]=(origin[0]&65535)|((origin[1]&65535)<<16);cfg(3,0);d.cfg_valid=0;
  const unsigned this_cfg=cfg_cycles+(command==0?static_cycles:0);d.start=1;tick();d.start=0;
  unsigned outputs=0;bool held=false;uint32_t hold[8];unsigned hold_addr=0;uint64_t states[64]={};bool finished=false;
  for(unsigned n=0;n<2000000;n++){
   d.source_allow=(!stall||n%11!=3);d.weight_allow=(!stall||(n%7!=2&&n%7!=3));d.result_ready=(!stall||(n%5!=1&&n%5!=2));d.clk=0;d.eval();
   if(held){if(!d.result_valid||d.result_addr!=hold_addr)return 6;for(int i=0;i<8;i++)if(d.result_data[i]!=hold[i])return 7;}
   held=d.result_valid&&!d.result_ready;if(held){hold_addr=d.result_addr;for(int i=0;i<8;i++)hold[i]=d.result_data[i];}
   if(d.result_valid&&d.result_ready){if(d.result_addr!=outputs)return 10;
    for(int i=0;i<8;i++)if(d.result_data[i]!=gold[outputs*8+i]){std::cerr<<"P mismatch "<<dir<<" row="<<outputs<<" lane="<<i<<" got="<<int32_t(d.result_data[i])<<" exp="<<int32_t(gold[outputs*8+i])<<"\n";return 11;}outputs++;}
   states[d.debug_state]++;tick();
   if(d.done){if(outputs!=480)return 12;uint64_t sum=0;for(auto v:states)sum+=v;if(sum!=d.cycles)return 13;
    std::cout<<"{\"fixture\":\""<<dir<<"\",\"stall\":"<<stall<<",\"command\":"<<command<<",\"cycles\":"<<d.cycles<<",\"configuration_cycles\":"<<this_cfg<<",\"start_cycles\":1,\"service_cycles\":"<<(uint64_t(d.cycles)+this_cfg+1)<<",\"outputs\":"<<outputs*8;
    auto field=[&](const char*n,uint64_t v){std::cout<<",\""<<n<<"\":"<<v;};
    field("source_words",d.source_words);field("weight_words",d.weight_words);field("local_gathers",d.local_gathers);field("add_issues",d.add_issues);
    field("psum_reads",d.psum_reads);field("psum_writes",d.psum_writes);field("psum_clears",d.psum_clears);
#ifdef OS_ARM
    field("bitmap_reads",d.bitmap_reads);field("bitmap_writes",d.bitmap_writes);
#endif
    field("source_stalls",d.source_stalls);field("weight_stalls",d.weight_stalls);field("output_stalls",d.output_stalls);
    std::cout<<",\"state_cycles\":[";for(int i=0;i<64;i++)std::cout<<(i?",":"")<<states[i];std::cout<<"]}\n";finished=true;break;}
  }
  if(!finished)return 14;
 }
 d.final();return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 20;}}
