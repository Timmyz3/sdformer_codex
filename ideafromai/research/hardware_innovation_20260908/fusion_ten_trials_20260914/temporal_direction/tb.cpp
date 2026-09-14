#include "Vconsumer_stream.h"
#include "verilated.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <chrono>
template<class T> std::vector<T> readbin(const std::string& path) {
 std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error(path);
 auto n=f.tellg();std::vector<T> v(size_t(n)/sizeof(T));f.seekg(0);f.read((char*)v.data(),n);return v;
}
int main(int argc,char**argv) {
 Verilated::commandArgs(argc,argv);if(argc!=5)return 2;
 std::string dir=argv[1];int mode=std::stoi(argv[2]),stall=std::stoi(argv[3]),tile=std::stoi(argv[4]);
 auto src=readbin<uint16_t>(dir+"/source.bin");auto raw=readbin<uint32_t>(dir+"/raw.bin");
 auto jgold=readbin<uint32_t>(dir+"/identity.bin");auto id=readbin<uint32_t>(dir+"/identity_fp32.bin");auto gold=readbin<uint32_t>(dir+"/gold.bin");
 std::vector<uint32_t> param[8];for(int k:{4,5,6,7})param[k]=readbin<uint32_t>(dir+"/param"+std::to_string(k)+".bin");
 if(src.size()!=1536 || raw.size()!=3840 || id.size()!=3840 || gold.size()!=3840)return 3;
 Vconsumer_stream d;auto tick=[&](){d.clk=0;d.eval();d.clk=1;d.eval();d.clk=0;d.eval();};
 d.reset_n=0;d.go=0;d.parameter_valid=0;d.source_valid=0;d.identity_valid=0;
 d.result_ready=1;d.compute_source_allow=1;d.compute_weight_allow=1;tick();tick();d.reset_n=1;
 auto started=std::chrono::steady_clock::now();
 for(unsigned command=0;command<2;++command) {
  d.mode=mode;d.first_tile=tile;d.tile_count=1;d.go=1;tick();d.go=0;
  unsigned outputs=0,raws=0,js=0;bool held=false,heldp=false,helds=false,heldi=false;
  uint32_t hd[8]={};unsigned ht=0,ha=0,pa=0,pk=0,sa=0,ia=0,it=0;
  uint64_t param_count=0,source_count=0,identity_count=0;
  for(uint64_t n=0;n<6000000;++n) {
   d.parameter_valid=(!stall || n%13!=2);
   d.source_valid=(!stall || n%7!=1 && n%7!=2);
   d.identity_valid=(!stall || n%11!=2 && n%11!=3);
   d.compute_source_allow=(!stall || n%11!=3);
   d.compute_weight_allow=(!stall || n%7!=2 && n%7!=3);
   d.result_ready=(!stall || n%5!=1 && n%5!=2);d.eval();
   if(heldp && (!d.parameter_request_valid || d.parameter_kind!=pk || d.parameter_address!=pa))return 14;
   if(helds && (!d.source_request_valid || d.source_address!=sa))return 15;
   if(heldi && (!d.identity_request_valid || d.identity_address!=ia || d.identity_tile!=it))return 16;
   if(d.parameter_request_valid)for(int l=0;l<8;++l)d.parameter_data[l]=param[d.parameter_kind].at(d.parameter_address*8+l);
   if(d.source_request_valid) {
    unsigned a=d.source_address,c=a/76800,y=(a%76800)/320,x=a%320;
    int ly=int(y)-(2*(tile/160)-1),lx=int(x)-(2*(tile%160)-1);
    if(c>=96 || ly<0 || ly>=4 || lx<0 || lx>=4)return 17;
    d.source_data=src.at(c*16+ly*4+lx);
   }
   if(d.identity_request_valid) {
    if(d.identity_tile!=tile)return 18;
    for(int l=0;l<8;++l)d.identity_data[l]=id.at(d.identity_address*8+l);
   }
   d.eval();
   if(held) {
    if(!d.result_valid || d.result_tile!=ht || d.result_address!=ha)return 8;
    for(int l=0;l<8;++l)if(d.result_data[l]!=hd[l])return 9;
   }
   held=d.result_valid&&!d.result_ready;
   if(held){ht=d.result_tile;ha=d.result_address;for(int l=0;l<8;++l)hd[l]=d.result_data[l];}
   heldp=d.parameter_request_valid&&!d.parameter_valid;pk=d.parameter_kind;pa=d.parameter_address;
   helds=d.source_request_valid&&!d.source_valid;sa=d.source_address;
   heldi=d.identity_request_valid&&!d.identity_valid;ia=d.identity_address;it=d.identity_tile;
   param_count+=d.parameter_request_valid&&d.parameter_valid;
   source_count+=d.source_request_valid&&d.source_valid;
   identity_count+=d.identity_request_valid&&d.identity_valid;
   if(d.j_monitor_valid) {
    if(d.j_monitor_address!=js)return 27;
    for(int l=0;l<8;++l)if(d.j_monitor_data[l]!=jgold.at(js*8+l))return 28;
    ++js;
   }
   if(d.raw_monitor_valid) {
    if(d.raw_monitor_address!=raws)return 19;
    for(int l=0;l<8;++l)if(d.raw_monitor_data[l]!=raw.at(raws*8+l))return 20;
    ++raws;
   }
   if(d.result_valid&&d.result_ready) {
    if(d.result_tile!=tile || d.result_address!=outputs || bool(d.result_tile_last)!=(outputs==479) || bool(d.result_job_last)!=(outputs==479))return 4;
    for(int l=0;l<8;++l)if(d.result_data[l]!=gold.at(outputs*8+l)) {
     std::cerr<<"I24 mismatch "<<mode<<" "<<outputs<<" "<<l<<" got "<<int32_t(d.result_data[l])<<" gold "<<int32_t(gold.at(outputs*8+l))<<"\n";return 5;
    }
    ++outputs;
   }
   tick();if(d.error)return 21;
   if(d.done) {
    if(outputs!=480 || raws!=480 || js!=480 || d.retired_tiles!=1)return 6;
    if(param_count!=(command?0:1848) || param_count!=d.static_words || source_count!=d.external_source_words || identity_count!=480)return 22;
    std::cout<<"{\"mode\":"<<mode<<",\"stall\":"<<stall<<",\"command\":"<<command<<",\"outputs\":"<<outputs*8<<",\"raw_outputs\":"<<raws*8<<",\"J_outputs\":"<<js*8;
#define SHOW(x) std::cout<<",\"" #x "\":"<<d.x
    SHOW(total_cycles);SHOW(static_words);SHOW(parameter_stalls);SHOW(source_load_words);SHOW(external_source_words);SHOW(padding_words);SHOW(origin_words);SHOW(source_load_stalls);SHOW(output_beats);SHOW(retired_tiles);
    SHOW(core_encoder_cycles);SHOW(core_candidate_trials);SHOW(core_range_rejects);SHOW(core_base_reads);SHOW(core_base_negations);SHOW(core_reference_writes);SHOW(core_choice_prev);SHOW(core_choice_anchor);SHOW(core_choice_neg1);SHOW(core_choice_pos2);SHOW(core_choice_neg2);SHOW(core_anchor_count);SHOW(core_cycles);SHOW(core_source_words);SHOW(core_weight_words);SHOW(core_second_weight_words);SHOW(core_local_source_reads);SHOW(core_z_vector_reads);SHOW(core_z_scalar_reads);SHOW(core_z_writes);SHOW(core_first_issues);SHOW(core_dual_updates);SHOW(core_psum_reads);SHOW(core_psum_writes);SHOW(core_mac_issues);SHOW(core_source_stalls);SHOW(core_weight_stalls);SHOW(core_output_stalls);
    SHOW(consumer_cycles);SHOW(consumer_raw_words);SHOW(consumer_identity_words);SHOW(consumer_coefficient_words);SHOW(consumer_mul_issues);SHOW(consumer_add_issues);SHOW(consumer_round_issues);SHOW(consumer_output_words);SHOW(consumer_identity_stalls);SHOW(consumer_raw_wait_cycles);SHOW(consumer_join_wait_cycles);SHOW(consumer_output_stalls);SHOW(consumer_saturations);SHOW(consumer_conversion_issues);SHOW(consumer_conversion_saturations);
    std::cout<<",\"wall_seconds_so_far\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<"}\n";break;
   }
   if(n==5999999)return 7;
  }
 }
 d.final();return 0;
}
