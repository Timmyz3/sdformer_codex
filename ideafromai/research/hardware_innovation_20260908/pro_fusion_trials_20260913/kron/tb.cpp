#include "Vped_kron.h"
#include "verilated.h"
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

std::vector<int64_t> read(const std::string &p){std::ifstream f(p);if(!f)throw std::runtime_error(p);std::vector<int64_t> v;int64_t x;while(f>>x)v.push_back(x);return v;}
template<class T> void put(T &bus,int index,int bits,int64_t val){
 uint64_t x=uint64_t(val)&((uint64_t(1)<<bits)-1);int bit=index*bits,w=bit/32,s=bit%32;
 bus[w]|=uint32_t(x<<s);if(s+bits>32)bus[w+1]|=uint32_t(x>>(32-s));
}
template<class T> int64_t get(T &bus,int index,int bits){
 int bit=index*bits,w=bit/32,s=bit%32;uint64_t x=uint64_t(bus[w])>>s;
 if(s+bits>32)x|=uint64_t(bus[w+1])<<(32-s);x&=(uint64_t(1)<<bits)-1;
 return (x&(uint64_t(1)<<(bits-1)))?int64_t(x)-int64_t(uint64_t(1)<<bits):int64_t(x);
}
struct Counts{uint64_t total=0,state[6]={0},cr_load=0,cr_reads=0,bias_load=0,input_words=0,input_stall=0,output_words=0,output_stall=0,latent_commit=0,result_commit=0,checked=0,held_checks=0;};
int main(int argc,char**argv){
 Verilated::commandArgs(argc,argv);auto inputs=read("inputs.txt"),bias=read("bias.txt");
 std::ofstream report("rtl_results.json");report<<"{\"verilator_executed\":true,\"cases\":[";bool first=true;
 for(auto setting:std::vector<std::pair<std::string,int>>{{"original",0},{"expanded_k1",0},{"kron1",1},{"expanded_k2",0},{"kron2",2}}){
  for(int stress=0;stress<2;stress++){
   Vped_kron d;Counts c;auto co=read(setting.first+"_coeff.txt"),gold=read(setting.first+"_gold.txt");
   d.clk=0;d.rst=1;d.mode=setting.second;d.start=0;d.cfg_valid=0;d.bias_valid=0;d.in_valid=0;d.out_ready=0;d.eval();d.clk=1;d.eval();d.rst=0;
   auto tick=[&](){d.clk=0;d.eval();int s=d.debug_state;c.total++;c.state[s]++;
    if(d.cfg_valid)c.cr_load++;if(d.bias_valid)c.bias_load++;
    if(d.debug_cr_read)c.cr_reads++;
    if(s==3 && !d.debug_cr_read)throw std::runtime_error("empty MAC must be skipped");
    if(d.in_ready){if(d.in_valid)c.input_words++;else c.input_stall++;}
    if(d.out_valid){if(d.out_ready)c.output_words++;else c.output_stall++;}
    if(s==4){if(setting.second && !d.debug_phase)c.latent_commit++;else c.result_commit++;}
    d.clk=1;d.eval();};
   for(size_t i=0;i<co.size()/8;i++){
    if(stress && i%7==0){d.cfg_valid=0;tick();}
    for(int w=0;w<4;w++)d.cfg_data[w]=0;
    for(int j=0;j<8;j++)put(d.cfg_data,j,16,co[i*8+j]);d.cfg_addr=i;d.cfg_valid=1;tick();
   }d.cfg_valid=0;
   for(int i=0;i<96;i++){d.bias_valid=1;d.bias_addr=i;d.bias_data=bias[i]&0xffffff;tick();}d.bias_valid=0;
   size_t vectors=inputs.size()/24;
   for(size_t n=0;n<vectors;n++){
    d.start=1;tick();d.start=0;size_t ib=0,ob=0;uint64_t limit=c.total+2000;bool held=false;std::array<uint32_t,6> held_data;
    while(!d.done){
     if(c.total>limit)throw std::runtime_error("timeout");
     d.clk=0;d.in_valid=(d.debug_state==1 && (!stress || c.total%11>=3));
     d.out_ready=(!stress || c.total%13>=4);
     for(int w=0;w<6;w++)d.in_data[w]=0;
     if(ib<3)for(int j=0;j<8;j++)put(d.in_data,j,24,inputs[n*24+ib*8+j]);
     d.eval();bool accepted=d.in_ready&&d.in_valid;
     if(held){if(!d.out_valid)throw std::runtime_error("output valid dropped under backpressure");for(int j=0;j<6;j++)if(held_data[j]!=d.out_data[j])throw std::runtime_error("unstable stalled output");c.held_checks++;}
     held=d.out_valid&&!d.out_ready;if(held)for(int j=0;j<6;j++)held_data[j]=d.out_data[j];
     if(d.out_valid&&d.out_ready){
      for(int j=0;j<8;j++){int64_t actual=get(d.out_data,j,24),expected=gold[n*96+ob*8+j];
       if(actual!=expected){std::cerr<<setting.first<<" vector="<<n<<" output="<<ob*8+j<<" actual="<<actual<<" expected="<<expected<<"\n";return 1;}c.checked++;}
      ob++;
     }
     tick();if(accepted)ib++;
    }
    if(ib!=3||ob!=12)throw std::runtime_error("incomplete transaction");d.in_valid=0;d.out_ready=0;
   }
   if(!first)report<<",";first=false;
   report<<"{\"mode\":\""<<setting.first<<"\",\"stress\":"<<stress<<",\"vectors\":"<<vectors
    <<",\"total_cycles\":"<<c.total<<",\"IDLE_config_start\":"<<c.state[0]<<",\"LOAD\":"<<c.state[1]<<",\"CLEAR\":"<<c.state[2]<<",\"MAC\":"<<c.state[3]<<",\"COMMIT\":"<<c.state[4]<<",\"OUTPUT\":"<<c.state[5]
    <<",\"coefficient_load_words\":"<<c.cr_load<<",\"coefficient_read_words\":"<<c.cr_reads<<",\"bias_load_words\":"<<c.bias_load<<",\"input_words\":"<<c.input_words<<",\"input_stall_cycles\":"<<c.input_stall<<",\"output_words\":"<<c.output_words<<",\"output_stall_cycles\":"<<c.output_stall
    <<",\"latent_writeback_cycles\":"<<c.latent_commit<<",\"result_writeback_cycles\":"<<c.result_commit<<",\"checked_values\":"<<c.checked<<",\"held_output_checks\":"<<c.held_checks<<",\"mismatches\":0}";
   std::cout<<setting.first<<" stress="<<stress<<" cycles="<<c.total<<" checked="<<c.checked<<"\n";
  }
 }report<<"]}\n";return 0;
}
