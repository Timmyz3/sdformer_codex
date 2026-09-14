#include "Vdecomp_core.h"
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
int main(int argc,char** argv){try{
 Verilated::commandArgs(argc,argv);if(argc!=4)return 2;
 const std::string dir=argv[1];const int mode=std::stoi(argv[2]),stall=std::stoi(argv[3]);
 auto src=read(dir+"/source.hex"),gold=read(dir+"/gold.hex"),origin=read(dir+"/origin.hex");
 auto q1=read(dir+"/q1.hex"),q2=read(dir+"/q2.hex"),klive=read(dir+"/k_live.hex");
 if(src.size()!=1536||gold.size()!=3840||origin.size()!=2||q1.size()!=6912||q2.size()!=768||klive.size()!=864)return 3;
 Vdecomp_core d;auto tick=[&](){d.clk=0;d.eval();d.clk=1;d.eval();d.clk=0;d.eval();};
 d.reset_n=0;d.start=0;d.cfg_valid=0;d.mode=mode;d.source_allow=1;d.weight_allow=1;d.result_ready=1;tick();tick();d.reset_n=1;
 unsigned cfg_cycles=0;uint32_t data[8]={0};
 auto cfg=[&](int kind,int addr){d.cfg_valid=1;d.cfg_kind=kind;d.cfg_addr=addr;for(int j=0;j<8;j++)d.cfg_data[j]=data[j];tick();cfg_cycles++;};
 for(unsigned i=0;i<src.size();i++){data[0]=src[i];cfg(0,i);}
 data[0]=(origin[0]&65535)|((origin[1]&65535)<<16);cfg(3,0);
 for(unsigned i=0;i<864;i++){for(int j=0;j<8;j++)data[j]=q1[i*8+j];cfg(4,i);}
 for(unsigned i=0;i<96;i++){for(int j=0;j<8;j++)data[j]=q2[i*8+j];cfg(5,i);}
 for(unsigned i=0;i<864;i++){data[0]=klive[i];cfg(6,i);}
 if(mode==15){
 auto klass=read(dir+"/class.hex"),rep=read(dir+"/representative.hex"),ng=read(dir+"/ngroups.hex");
 if(klass.size()!=864||rep.size()!=256||ng.size()!=1)return 21;
 for(unsigned i=0;i<32;i++){for(int l=0;l<8;l++)data[l]=rep[i*8+l];cfg(7,i);}
 for(unsigned i=0;i<864;i++){data[0]=klass[i];cfg(7,32+i);}
 data[0]=ng[0];cfg(7,1023);
 }
 d.cfg_valid=0;unsigned checked=0;
 for(unsigned command=0;command<2;command++){
  d.start=1;tick();d.start=0;unsigned outputs=0;bool held=false;uint32_t hold[8];unsigned hold_addr=0;uint64_t states[64]={};
  for(unsigned n=0;n<4000000;n++){
   d.source_allow=(!stall||n%11!=3);d.weight_allow=(!stall||(n%7!=2&&n%7!=3));d.result_ready=(!stall||(n%5!=1&&n%5!=2));
   d.clk=0;d.eval();
   if(held){if(!d.result_valid||d.result_addr!=hold_addr)return 8;for(int l=0;l<8;l++)if(d.result_data[l]!=hold[l])return 9;}
   held=d.result_valid&&!d.result_ready;
   if(held){hold_addr=d.result_addr;for(int l=0;l<8;l++)hold[l]=d.result_data[l];}
   if(d.result_valid&&d.result_ready){
    if(d.result_addr!=outputs)return 4;
    for(int l=0;l<8;l++)if(d.result_data[l]!=gold.at(outputs*8+l)){
     std::cerr<<"mismatch mode="<<mode<<" row="<<outputs<<" lane="<<l<<" got="<<int32_t(d.result_data[l])<<" gold="<<int32_t(gold.at(outputs*8+l))<<"\n";return 5;}
    outputs++;checked+=8;
   }
   if(d.debug_state>=64)return 14;states[d.debug_state]++;tick();
   if(d.done){
    if(outputs!=480)return 6;uint64_t state_sum=0;for(auto v:states)state_sum+=v;if(state_sum!=d.cycles)return 15;
    std::cout<<"{\"mode\":"<<mode<<",\"stall\":"<<stall<<",\"command\":"<<command<<",\"cycles\":"<<d.cycles
      <<",\"configuration_cycles\":"<<(command==0?cfg_cycles:0)<<",\"outputs\":"<<outputs*8
      <<",\"source_words\":"<<d.source_words<<",\"weight_words\":"<<d.weight_words<<",\"second_weight_words\":"<<d.second_weight_words
      <<",\"z_vector_reads\":"<<d.z_vector_reads<<",\"z_scalar_reads\":"<<d.z_scalar_reads<<",\"z_writes\":"<<d.z_writes<<",\"first_issues\":"<<d.first_issues
      <<",\"psum_reads\":"<<d.psum_reads<<",\"psum_writes\":"<<d.psum_writes<<",\"mac_issues\":"<<d.mac_issues
      <<",\"source_stalls\":"<<d.source_stalls<<",\"weight_stalls\":"<<d.weight_stalls<<",\"output_stalls\":"<<d.output_stalls
      <<",\"local_source_reads\":"<<d.local_source_reads<<",\"dual_updates\":"<<d.dual_updates<<",\"aux_reads\":"<<d.aux_reads<<",\"aux_writes\":"<<d.aux_writes<<",\"aux_issues\":"<<d.aux_issues<<",\"aux_weight_words\":"<<d.aux_weight_words<<",\"aux_events\":"<<d.aux_events<<",\"metadata_reads\":"<<d.metadata_reads<<",\"count_checks\":"<<d.count_checks<<",\"count_bank_reads\":"<<d.count_bank_reads<<",\"count_bank_writes\":"<<d.count_bank_writes<<",\"state_cycles\":[";
    for(int i=0;i<64;i++)std::cout<<(i?",":"")<<states[i];std::cout<<"]}\n";break;
   }
   if(n==3999999)return 7;
  }
 }
 d.final();return checked==7680?0:10;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 20;}}
