#include "Vnative_sparse.h"
#include "verilated.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
static std::vector<uint32_t> read(const std::string& path) {
  std::ifstream f(path);if(!f)throw std::runtime_error(path);
  std::vector<uint32_t> a;uint64_t x;while(f>>std::hex>>x)a.push_back(uint32_t(x));return a;
}
int main(int argc,char** argv) {
  Verilated::commandArgs(argc,argv);if(argc!=4)return 2;
  const std::string dir=argv[1];const int mode=std::stoi(argv[2]),stall=std::stoi(argv[3]);
  auto src=read(dir+"/source.hex"),w=read(dir+"/weight.hex"),mask=read(dir+"/mask.hex"),gold=read(dir+"/gold.hex");
  auto origin=read(dir+"/origin.hex");
  if(src.size()!=1536 || w.size()!=82944 || mask.size()!=288 || gold.size()!=3840)return 3;
  Vnative_sparse d;
  auto tick=[&](){d.clk=0;d.eval();d.clk=1;d.eval();d.clk=0;d.eval();};
  d.reset_n=0;d.start=0;d.cfg_valid=0;d.mode=mode;
  d.source_allow=1;d.weight_allow=1;d.result_ready=1;tick();tick();d.reset_n=1;
  unsigned cfg_cycles=0;
  auto cfg=[&](int kind,int addr,const uint32_t *data) {
    d.cfg_valid=1;d.cfg_kind=kind;d.cfg_addr=addr;
    for(int j=0;j<4;++j)d.cfg_data[j]=data[j];tick();++cfg_cycles;
  };
  uint32_t data[4]={0,0,0,0};
  for(unsigned i=0;i<src.size();++i){data[0]=src[i];cfg(0,i,data);}
  for(unsigned i=0;i<mask.size();++i){data[0]=mask[i];cfg(2,i,data);}
  if(origin.size()!=2)return 11;
  data[0]=(origin[0]&65535)|((origin[1]&65535)<<16);cfg(3,0,data);
  for(unsigned row=0;row<10368;++row){
    for(int j=0;j<4;++j)data[j]=(w[row*8+j*2]&65535)|((w[row*8+j*2+1]&65535)<<16);
    cfg(1,row,data);
  }
  d.cfg_valid=0;unsigned checked=0;
  // Two commands without reset exercise psum clear, persistent configuration,
  // source cache lifetime, and output stability under global-time backpressure.
  for(unsigned command=0;command<2;++command) {
    d.start=1;tick();d.start=0;unsigned outputs=0;bool held=false;uint32_t held_data[8];unsigned held_addr=0;
    for(unsigned n=0;n<4000000;++n) {
      d.source_allow=(!stall || n%11!=3);
      d.weight_allow=(!stall || n%7!=2 && n%7!=3);
      d.result_ready=(!stall || n%5!=1 && n%5!=2);
      d.clk=0;d.eval();
      if(held) {
        if(!d.result_valid || d.result_addr!=held_addr)return 8;
        for(int lane=0;lane<8;++lane)if(d.result_data[lane]!=held_data[lane])return 9;
      }
      held=d.result_valid&&!d.result_ready;
      if(held){held_addr=d.result_addr;for(int lane=0;lane<8;++lane)held_data[lane]=d.result_data[lane];}
      if(d.result_valid&&d.result_ready) {
        if(d.result_addr!=outputs)return 4;
        for(int lane=0;lane<8;++lane)if(d.result_data[lane]!=gold.at(outputs*8+lane)) {
          std::cerr<<"Mismatch mode "<<mode<<" row "<<outputs<<" lane "<<lane<<" got "
            <<int32_t(d.result_data[lane])<<" gold "<<int32_t(gold.at(outputs*8+lane))<<"\n";return 5;
        }
        ++outputs;checked+=8;
      }
      tick();
      if(d.done) {
        if(outputs!=480)return 6;
        std::cout<<"{\"mode\":"<<mode<<",\"stall\":"<<stall<<",\"command\":"<<command
          <<",\"cycles\":"<<d.cycles<<",\"configuration_cycles\":"<<cfg_cycles
          <<",\"source_words\":"<<d.source_words<<",\"weight_words\":"<<d.weight_words
          <<",\"psum_reads\":"<<d.psum_reads<<",\"psum_writes\":"<<d.psum_writes
          <<",\"sum_issues\":"<<d.sum_issues<<",\"update_issues\":"<<d.update_issues
          <<",\"source_stalls\":"<<d.source_stalls<<",\"weight_stalls\":"<<d.weight_stalls
          <<",\"output_stalls\":"<<d.output_stalls<<",\"masked_contexts\":"<<d.masked_contexts
          <<",\"zero_contexts\":"<<d.zero_contexts<<",\"outputs\":"<<outputs*8<<"}\n";
        break;
      }
      if(n==3999999){std::cerr<<"Timeout\n";return 7;}
    }
  }
  d.final();return checked==7680?0:10;
}
