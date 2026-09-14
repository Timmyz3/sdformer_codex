#include "Vos_stream.h"
#include "verilated.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
double sc_time_stamp(){return 0;}
static std::vector<uint32_t> readhex(const std::string& p){std::ifstream f(p);if(!f)throw std::runtime_error(p);std::vector<uint32_t>a;uint64_t x;while(f>>std::hex>>x)a.push_back(uint32_t(x));return a;}
int main(int argc,char**argv){try{
 Verilated::commandArgs(argc,argv);if(argc!=4)return 2;const std::string root=argv[1];int stall=std::stoi(argv[3]);
 std::ifstream manifest(argv[2]);std::vector<std::string> fixtures;std::string path;while(manifest>>path)fixtures.push_back(path);if(fixtures.empty())return 3;
 auto w=readhex(root+"/parameters/weight.hex"),ab=readhex(root+"/parameters/consumer.hex");
 if(w.size()!=82944||ab.size()!=192)return 4;
 Vos_stream d;auto tick=[&](){d.clk=0;d.eval();d.clk=1;d.eval();d.clk=0;d.eval();};
 d.reset_n=0;d.start=0;d.cfg_valid=0;d.source_allow=1;d.weight_allow=1;d.raw_allow=1;d.result_ready=1;d.identity_valid=0;for(int i=0;i<8;i++)d.identity_data[i]=0;tick();tick();d.reset_n=1;
 uint32_t data[8]={0};unsigned cfg_cycles=0;
 auto cfg=[&](int kind,int addr){d.cfg_valid=1;d.cfg_kind=kind;d.cfg_addr=addr;for(int i=0;i<8;i++)d.cfg_data[i]=data[i];tick();cfg_cycles++;};
 for(unsigned a=0;a<10368;a++){for(int i=0;i<8;i++)data[i]=w[a*8+i];cfg(4,a);}
 for(unsigned a=0;a<24;a++){for(int i=0;i<8;i++)data[i]=ab[a*8+i];cfg(6,a);}
 d.cfg_valid=0;unsigned static_cycles=cfg_cycles;
 for(unsigned command=0;command<2*fixtures.size();command++){
  const auto &dir=fixtures[command%fixtures.size()];
  auto src=readhex(dir+"/source.hex"),origin=readhex(dir+"/origin.hex"),gold=readhex(dir+"/gold.hex");
  auto identity=readhex(dir+"/identity.hex"),j=readhex(dir+"/j.hex"),wide=readhex(dir+"/wide.hex"),i24=readhex(dir+"/i24.hex");
  if(src.size()!=1536||origin.size()!=2||gold.size()!=3840||identity.size()!=3840||j.size()!=3840||wide.size()!=7680||i24.size()!=3840)return 5;
  cfg_cycles=0;for(unsigned a=0;a<1536;a++){data[0]=src[a];cfg(0,a);}
  data[0]=(origin[0]&65535)|((origin[1]&65535)<<16);cfg(3,0);d.cfg_valid=0;
  const unsigned this_cfg=cfg_cycles+(command==0?static_cycles:0);d.start=1;tick();d.start=0;
  unsigned outputs=0,rawoutputs=0,joutputs=0,wideoutputs=0,idoutputs=0;bool held=false;uint32_t hold[8];unsigned hold_addr=0;uint64_t states[64]={};bool finished=false;
  for(unsigned n=0;n<2000000;n++){
   d.source_allow=(!stall||n%11!=3);d.weight_allow=(!stall||(n%7!=2&&n%7!=3));d.raw_allow=(!stall||n%13!=4);d.result_ready=(!stall||(n%5!=1&&n%5!=2));d.identity_valid=0;
   d.clk=0;d.eval();
   if(d.identity_request_valid&&(!stall||(n%17!=4&&n%17!=5&&n%17!=6))){
    if(d.identity_address!=idoutputs)return 6;
    d.identity_valid=1;for(int i=0;i<8;i++)d.identity_data[i]=identity[idoutputs*8+i];idoutputs++;
   }
   d.eval();
   if(held){if(!d.result_valid||d.result_addr!=hold_addr)return 7;for(int i=0;i<8;i++)if(d.result_data[i]!=hold[i])return 8;}
   held=d.result_valid&&!d.result_ready;if(held){hold_addr=d.result_addr;for(int i=0;i<8;i++)hold[i]=d.result_data[i];}
   if(d.raw_monitor_valid){if(d.raw_monitor_addr!=rawoutputs)return 11;for(int i=0;i<8;i++)if(d.raw_monitor_data[i]!=gold[rawoutputs*8+i]){std::cerr<<"raw mismatch "<<dir<<" row="<<rawoutputs<<" lane="<<i<<"\n";return 12;}rawoutputs++;}
   if(d.j_monitor_valid){if(d.j_monitor_address!=joutputs)return 13;for(int i=0;i<8;i++)if(d.j_monitor_data[i]!=j[joutputs*8+i]){std::cerr<<"J mismatch "<<dir<<" row="<<joutputs<<" lane="<<i<<"\n";return 14;}joutputs++;}
   if(d.wide_monitor_valid){if(d.wide_monitor_address!=wideoutputs)return 15;for(int i=0;i<16;i++)if(d.wide_monitor_data[i]!=wide[wideoutputs*16+i]){std::cerr<<"wide mismatch "<<dir<<" row="<<wideoutputs<<" word="<<i<<" got="<<d.wide_monitor_data[i]<<" exp="<<wide[wideoutputs*16+i]<<"\n";return 16;}wideoutputs++;}
   if(d.result_valid&&d.result_ready){if(d.result_addr!=outputs)return 17;for(int i=0;i<8;i++)if(d.result_data[i]!=i24[outputs*8+i]){std::cerr<<"I24 mismatch "<<dir<<" row="<<outputs<<" lane="<<i<<" got="<<int32_t(d.result_data[i])<<" exp="<<int32_t(i24[outputs*8+i])<<"\n";return 18;}outputs++;}
   states[d.debug_state]++;tick();
   if(d.error)return 19;
   if(d.done){if(outputs!=480||rawoutputs!=480||joutputs!=480||wideoutputs!=480||idoutputs!=480)return 20;
    uint64_t sum=0;for(auto v:states)sum+=v;if(sum!=d.c_cycles||sum-states[0]!=d.cycles)return 21;
    std::cout<<"{\"fixture\":\""<<dir<<"\",\"stall\":"<<stall<<",\"command\":"<<command<<",\"configuration_cycles\":"<<this_cfg<<",\"raw_values\":3840,\"j_values\":3840,\"wide_values\":3840,\"i24_values\":3840";
    auto field=[&](const char*n,uint64_t v){std::cout<<",\""<<n<<"\":"<<v;};
#define F(n) field(#n,d.n)
    F(cycles);F(source_words);F(weight_words);F(add_issues);F(bitmap_reads);F(bitmap_writes);F(psum_clears);F(local_gathers);F(psum_reads);F(psum_writes);F(source_stalls);F(weight_stalls);F(output_stalls);
    F(c_cycles);F(c_raw_words);F(c_identity_words);F(c_coefficient_words);F(c_mul_issues);F(c_add_issues);F(c_round_issues);F(c_output_words);F(c_identity_stalls);F(c_raw_wait_cycles);F(c_join_wait_cycles);F(c_output_stalls);F(c_saturations);F(c_conversion_issues);F(c_conversion_saturations);F(c_wide_waits);
#undef F
    std::cout<<",\"state_cycles\":[";for(int i=0;i<64;i++)std::cout<<(i?",":"")<<states[i];std::cout<<"]}\n";finished=true;break;
   }
  }
  if(!finished)return 22;
 }
 d.final();return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 30;}}
