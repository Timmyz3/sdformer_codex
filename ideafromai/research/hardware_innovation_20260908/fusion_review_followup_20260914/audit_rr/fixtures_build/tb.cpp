double sc_time_stamp(){return 0.0;}
#include "Vinterleave_stream.h"
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
 Verilated::commandArgs(argc,argv);if(argc!=6)return 2;
 std::string dir=argv[1];int mode=std::stoi(argv[2]),stall=std::stoi(argv[3]),tile=std::stoi(argv[4]),count=std::stoi(argv[5]);
 auto src=readbin<uint16_t>(dir+"/source.bin");auto raw=readbin<uint32_t>(dir+"/raw.bin");
 auto jgold=readbin<uint32_t>(dir+"/identity.bin");auto id=readbin<uint32_t>(dir+"/identity_fp32.bin");auto gold=readbin<uint32_t>(dir+"/gold.bin");
 std::vector<uint32_t> param[8];for(int k:{4,5,6,7})param[k]=readbin<uint32_t>(dir+"/param"+std::to_string(k)+".bin");
 if(src.size()!=1536 || raw.size()!=3840 || id.size()!=3840 || gold.size()!=3840)return 3;
 Vinterleave_stream d;auto tick=[&](){d.clk=0;d.eval();d.clk=1;d.eval();d.clk=0;d.eval();};
 d.reset_n=0;d.go=0;d.parameter_valid=0;d.source_valid=0;d.identity_valid=0;
 d.result_ready=1;d.compute_source_allow=1;d.compute_weight_allow=1;tick();tick();d.reset_n=1;
 auto started=std::chrono::steady_clock::now();
 for(unsigned command=0;command<2;++command) {
  unsigned runmode=(mode==3?1+command%2:mode==4?2-command%2:mode);d.mode=runmode;d.first_tile=tile;d.tile_count=count;d.go=1;tick();d.go=0;
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
    if(c>=96 || ly<0 || ly>=4 || lx<0 || lx>=4+2*(count-1))return 17;
    if(count==1)d.source_data=src.at(c*16+ly*4+lx);
    else {for(int pos=1;pos<16;++pos)if(src.at(c*16+pos)!=src.at(c*16))return 40;d.source_data=src.at(c*16);}
   }
   if(d.identity_request_valid) {
    if(d.identity_tile<tile || d.identity_tile>=tile+count)return 18;
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
    if(d.j_monitor_address!=js%480)return 27;
    for(int l=0;l<8;++l)if(d.j_monitor_data[l]!=jgold.at((js%480)*8+l))return 28;
    ++js;
   }
   if(d.raw_monitor_valid) {
    if(d.raw_monitor_address!=raws%480)return 19;
    for(int l=0;l<8;++l)if(d.raw_monitor_data[l]!=raw.at((raws%480)*8+l))return 20;
    ++raws;
   }
   if(d.result_valid&&d.result_ready) {
    if(d.result_tile!=tile+outputs/480 || d.result_address!=outputs%480 || bool(d.result_tile_last)!=(outputs%480==479) || bool(d.result_job_last)!=(outputs==480*count-1))return 4;
    for(int l=0;l<8;++l)if(d.result_data[l]!=gold.at((outputs%480)*8+l)) {
     std::cerr<<"I24 mismatch "<<mode<<" "<<outputs<<" "<<l<<" got "<<int32_t(d.result_data[l])<<" gold "<<int32_t(gold.at((outputs%480)*8+l))<<"\n";return 5;
    }
    ++outputs;
   }
   tick();if(d.error)return 21;
   if(d.done) {
    if(d.proof_issues!=(command?0:864))return 30;
    bool expected_range=true;
    for(int lane=0;lane<8;++lane){
     int pos=0,neg=0;
     for(int k=0;k<864;++k){int q=int32_t(param[4][k*8+lane]);if(q>=0)pos+=q;else neg+=q;}
     auto field=[&](const auto& bus){unsigned bit=lane*13,word=bit/32;uint64_t v=bus[word];if(word<3)v|=uint64_t(bus[word+1])<<32;int z=(v>>(bit%32))&8191;return z&4096?z-8192:z;};
     if(field(d.proof_positive_bounds)!=pos || field(d.proof_negative_bounds)!=neg)return 31;
     expected_range &= pos<=511 && neg>=-512;
    }
    if(bool(d.proof_range_ok)!=expected_range)return 32;
    if(d.range_fallback_tiles!=((runmode==1&&!expected_range)?d.retired_tiles:0))return 33;
    if(d.core_normalization_issues!=(runmode==2?10*d.retired_tiles:0))return 34;
    if(runmode!=2&&(d.core_repair_issues||d.core_repair_fields))return 35;


    if(d.shared_alu_grants!=d.proof_issues+d.core_first_issues+d.core_mac_issues+d.core_repair_issues+d.core_normalization_issues)return 36;
    if(d.shared_z_grants!=d.core_z_vector_reads+d.core_z_scalar_reads+d.core_z_writes)return 37;
    if(d.shared_source_grants!=d.core_source_words || d.shared_weight_grants!=d.core_weight_words)return 38;
    if(d.shared_psum_grants!=d.core_psum_reads+d.core_psum_writes || d.conflict_cycles!=d.core_arbitration_stalls)return 39;

    if(outputs!=480*count || raws!=480*count || js!=480*count || d.retired_tiles!=count)return 6;
    if(param_count!=(command?0:1848) || param_count!=d.static_words || source_count!=d.external_source_words || identity_count!=480*count)return 22;
    std::cout<<"{\"mode\":"<<mode<<",\"stall\":"<<stall<<",\"command\":"<<command<<",\"outputs\":"<<outputs*8<<",\"raw_outputs\":"<<raws*8<<",\"J_outputs\":"<<js*8;
#define SHOW(x) std::cout<<",\"" #x "\":"<<uint64_t(d.x)
    SHOW(proof_issues);SHOW(range_fallback_tiles);SHOW(proof_range_ok);SHOW(core_repair_issues);SHOW(core_repair_fields);SHOW(core_normalization_issues);SHOW(core_repair_arbitration_stalls);SHOW(core_normalization_arbitration_stalls);SHOW(total_cycles);SHOW(window_cycles);SHOW(launch_cycles);SHOW(batches);SHOW(conflict_cycles);SHOW(both_compute_cycles);SHOW(shared_source_grants);SHOW(shared_weight_grants);SHOW(shared_z_grants);SHOW(shared_psum_grants);SHOW(shared_alu_grants);SHOW(static_words);SHOW(parameter_stalls);SHOW(source_load_words);SHOW(external_source_words);SHOW(padding_words);SHOW(origin_words);SHOW(source_load_stalls);SHOW(output_beats);SHOW(retired_tiles);
    SHOW(core_cycles);SHOW(core_source_words);SHOW(core_weight_words);SHOW(core_second_weight_words);SHOW(core_local_source_reads);SHOW(core_z_vector_reads);SHOW(core_z_scalar_reads);SHOW(core_z_writes);SHOW(core_first_issues);SHOW(core_merged_updates);SHOW(core_psum_reads);SHOW(core_psum_writes);SHOW(core_mac_issues);SHOW(core_source_stalls);SHOW(core_weight_stalls);SHOW(core_output_stalls);SHOW(core_arbitration_stalls);
    SHOW(consumer_cycles);SHOW(consumer_raw_words);SHOW(consumer_identity_words);SHOW(consumer_coefficient_words);SHOW(consumer_mul_issues);SHOW(consumer_add_issues);SHOW(consumer_round_issues);SHOW(consumer_output_words);SHOW(consumer_identity_stalls);SHOW(consumer_raw_wait_cycles);SHOW(consumer_join_wait_cycles);SHOW(consumer_output_stalls);SHOW(consumer_saturations);SHOW(consumer_conversion_issues);SHOW(consumer_conversion_saturations);
    std::cout<<",\"wall_seconds_so_far\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<"}\n";break;
   }
   if(n==5999999)return 7;
  }
 }
 d.final();return 0;
}
