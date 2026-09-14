#include "Vlossy_tile.h"
#include "verilated.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
static std::vector<uint32_t> read(const std::string& path){std::ifstream f(path);if(!f)throw std::runtime_error(path);std::vector<uint32_t>a;uint64_t x;while(f>>std::hex>>x)a.push_back(uint32_t(x));return a;}
int main(int argc,char**argv){try{
 Verilated::commandArgs(argc,argv);if(argc!=5)return 2;
 std::string dir=argv[1],con=argv[2];int mode=std::stoi(argv[3]),stall=std::stoi(argv[4]);
 auto src=read(dir+"/source.hex"),gold=read(dir+"/gold_"+std::to_string(mode)+".hex"),igold=read(dir+"/i24_"+std::to_string(mode)+".hex"),identity=read(dir+"/identity.hex"),jg=read(dir+"/j.hex"),origin=read(dir+"/origin.hex");
 Vlossy_tile d;auto tick=[&](){d.clk=0;d.eval();d.clk=1;d.eval();d.clk=0;d.eval();};
 d.reset_n=0;d.start=0;d.cfg_valid=0;d.mode=mode;d.source_allow=1;d.weight_allow=1;d.result_ready=1;d.identity_valid=0;tick();tick();d.reset_n=1;
 unsigned cfg_cycles=0;auto cfg=[&](int kind,int addr,const uint32_t*data){d.cfg_valid=1;d.cfg_kind=kind;d.cfg_addr=addr;for(int l=0;l<8;l++)d.cfg_data[l]=data[l];tick();cfg_cycles++;};
 uint32_t data[8]={};for(unsigned i=0;i<src.size();i++){data[0]=src[i];cfg(0,i,data);}data[0]=(origin[0]&65535)|((origin[1]&65535)<<16);cfg(3,0,data);
 auto table=[&](const std::string&file,int kind,int width){auto a=read(con+"/"+file+".hex");for(unsigned i=0;i<a.size()/width;i++){for(int l=0;l<8;l++)data[l]=(l<width)?a[i*width+l]:0;cfg(kind,i,data);}};
 table("q1",4,8);table("q2",5,8);table("k_live",6,1);table("parameters",7,8);table("codebook",8,8);table("prototypes",9,8);table("consumer",10,8);d.cfg_valid=0;
 for(unsigned command=0;command<2;command++){
 d.start=1;tick();d.start=0;unsigned output=0,raw=0,jcount=0;bool held=false;uint32_t hv[8];unsigned ha=0;uint64_t states[29]={};bool coredone=false;
 for(unsigned n=0;n<4000000;n++){
 d.source_allow=!stall||n%11!=3;d.weight_allow=!stall||(n%7!=2&&n%7!=3);d.result_ready=!stall||(n%5!=1&&n%5!=2);
 d.identity_valid=0;d.clk=0;d.eval();if(d.identity_request_valid&&(!stall||n%13!=4)){d.identity_valid=1;for(int l=0;l<8;l++)d.identity_data[l]=identity.at(d.identity_address*8+l);}d.eval();
 if(held){if(!d.result_valid||d.result_addr!=ha)throw std::runtime_error("held address");for(int l=0;l<8;l++)if(d.result_data[l]!=hv[l])throw std::runtime_error("held data");}
 held=d.result_valid&&!d.result_ready;if(held){ha=d.result_addr;for(int l=0;l<8;l++)hv[l]=d.result_data[l];}
 auto compare=[&](const auto& data,const std::vector<uint32_t>&g,unsigned addr,const char*what){for(int l=0;l<8;l++)if(data[l]!=g.at(addr*8+l)){std::cerr<<what<<" mode="<<mode<<" cmd="<<command<<" row="<<addr<<" lane="<<l<<" got="<<int32_t(data[l])<<" gold="<<int32_t(g.at(addr*8+l))<<" state="<<unsigned(d.debug_state)<<"\n";throw std::runtime_error("mismatch");}};
 if(d.raw_monitor_valid){if(d.raw_monitor_addr!=raw)throw std::runtime_error("raw order");compare(d.raw_monitor_data,gold,raw++,"raw");}
 if(d.j_monitor_valid){if(d.j_monitor_address!=jcount)throw std::runtime_error("J order");compare(d.j_monitor_data,jg,jcount++,"J");}
 if(d.result_valid&&d.result_ready){if(d.result_addr!=output)throw std::runtime_error("output order");compare(d.result_data,igold,output++,"I24");}
 if(!coredone){if(d.debug_state>=29)throw std::runtime_error("state");states[d.debug_state]++;if(d.debug_state==18)coredone=true;}
 tick();if(d.error)throw std::runtime_error("consumer error");
 if(d.done){if(output!=480||raw!=480||jcount!=480)throw std::runtime_error("count");uint64_t sum=0;for(auto v:states)sum+=v;if(sum!=d.cycles)throw std::runtime_error("cycle sum");
 std::cout<<"{\"mode\":"<<mode<<",\"stall\":"<<stall<<",\"command\":"<<command<<",\"configuration_cycles\":"<<(command==0?cfg_cycles:0)<<",\"outputs\":"<<output*8;
 #define FIELD(x) std::cout<<",\"" #x "\":"<<d.x
 FIELD(cycles);FIELD(total_cycles);FIELD(source_words);FIELD(weight_words);FIELD(second_weight_words);FIELD(local_source_reads);FIELD(z_vector_reads);FIELD(z_scalar_reads);FIELD(z_writes);FIELD(first_issues);FIELD(dual_updates);FIELD(psum_reads);FIELD(psum_writes);FIELD(mac_issues);FIELD(source_stalls);FIELD(weight_stalls);FIELD(output_stalls);FIELD(encoder_cycles);FIELD(prototype_reads);FIELD(held_vectors);FIELD(consumer_cycles);FIELD(consumer_raw_words);FIELD(consumer_identity_words);FIELD(consumer_mul_issues);FIELD(consumer_add_issues);FIELD(consumer_round_issues);FIELD(consumer_output_words);FIELD(consumer_conversion_issues);FIELD(consumer_coefficient_words);FIELD(consumer_join_wait);FIELD(consumer_output_stalls);
 std::cout<<",\"state_cycles\":[";for(int i=0;i<29;i++)std::cout<<(i?",":"")<<states[i];std::cout<<"]}\n";break;}
 if(n==3999999)throw std::runtime_error("timeout");
 }
 }
 d.final();return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 20;}}
