#include "Vinterleave_stream.h"
#include "verilated.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
template<class T> std::vector<T> readbin(const std::string& path) {
 std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error(path);
 auto n=f.tellg();std::vector<T> v(size_t(n)/sizeof(T));f.seekg(0);f.read((char*)v.data(),n);return v;
}
struct Fixture {
 std::vector<uint16_t> source;
 std::vector<uint32_t> raw,id,j,gold,wide,origin;
 Fixture(const std::string& p):source(readbin<uint16_t>(p+"/source.bin")),raw(readbin<uint32_t>(p+"/raw.bin")),id(readbin<uint32_t>(p+"/identity_fp32.bin")),j(readbin<uint32_t>(p+"/identity.bin")),gold(readbin<uint32_t>(p+"/gold.bin")),wide(readbin<uint32_t>(p+"/wide.bin")),origin(readbin<uint32_t>(p+"/origin.bin")) {
  if(source.size()!=1536||raw.size()!=3840||id.size()!=3840||j.size()!=3840||gold.size()!=3840||wide.size()!=7680||origin.size()!=2)throw std::runtime_error("fixture size");
 }
};
double sc_time_stamp(){return 0;}
int main(int argc,char**argv) {
 Verilated::commandArgs(argc,argv);if(argc!=5 && argc!=6)return 2;
 std::string dir=argv[1];std::ifstream manifest(argv[2]);std::string p;std::vector<Fixture> fixtures;
 while(manifest>>p)fixtures.emplace_back(p);
 unsigned mode=std::stoul(argv[3]),stall=std::stoul(argv[4]),first=0,count=fixtures.size(),repeats=2;
 uint64_t limit=12000000;if(!count)return 3;
 std::vector<uint32_t> param[12];for(int k:{4,5,6,7,8,9,10,11})param[k]=readbin<uint32_t>(dir+"/param"+std::to_string(k)+".bin");
 Vinterleave_stream d;auto tick=[&](){d.clk=0;d.eval();d.clk=1;d.eval();d.clk=0;d.eval();};
 d.reset_n=0;d.go=0;d.parameter_valid=0;d.source_valid=0;d.identity_valid=0;d.origin_valid=0;
 d.result_ready=1;d.compute_source_allow=1;d.compute_weight_allow=1;tick();tick();d.reset_n=1;
 auto started=std::chrono::steady_clock::now();
 bool class_loaded=false,permutation_loaded=false;
 for(unsigned command=0;command<repeats;++command) {
  unsigned active_mode=command && argc>5?std::stoul(argv[5]):mode;
  unsigned expected_static=command?0:1848;
  if(active_mode>=20&&!class_loaded){expected_static+=897;class_loaded=true;}
  if(active_mode==21&&!permutation_loaded){expected_static++;permutation_loaded=true;}
  d.mode=active_mode;d.first_tile=first;d.tile_count=count;d.go=1;tick();d.go=0;
  uint64_t outputs=0,raws=0,js=0,wides=0;bool held=false,heldp=false,helds=false,heldi=false,heldo=false;
  uint32_t hd[8]={};unsigned ht=0,ha=0,pa=0,pk=0,sa=0,st=0,ia=0,it=0,ot=0;
  uint64_t param_count=0,source_count=0,identity_count=0,origin_count=0;
  for(uint64_t n=0;n<limit;++n) {
   d.origin_valid=(!stall || (n%17!=4 && n%17!=5));
   d.parameter_valid=(!stall || n%13!=2);
   d.source_valid=(!stall || n%7!=1 && n%7!=2);
   d.identity_valid=(!stall || n%11!=2 && n%11!=3);
   d.compute_source_allow=(!stall || n%11!=3);
   d.compute_weight_allow=(!stall || n%7!=2 && n%7!=3);
   d.result_ready=(!stall || n%5!=1 && n%5!=2);d.eval();
   if(heldp && (!d.parameter_request_valid || d.parameter_kind!=pk || d.parameter_address!=pa))return 14;
   if(helds && (!d.source_request_valid || (d.source_address!=sa||d.source_tile!=st)))return 15;
   if(heldi && (!d.identity_request_valid || d.identity_address!=ia || d.identity_tile!=it))return 16;
   if(heldo&&(!d.origin_request_valid||d.origin_tile!=ot))return 43;
   if(d.origin_request_valid){const auto& f=fixtures.at(d.origin_tile);d.origin_data=(f.origin[0]&65535)|((f.origin[1]&65535)<<16);}
   if(d.parameter_request_valid)for(int l=0;l<8;++l)d.parameter_data[l]=param[d.parameter_kind].at(d.parameter_address*8+l);
   if(d.source_request_valid)d.source_data=fixtures.at(d.source_tile).source.at(d.source_address);
   if(d.identity_request_valid)for(int l=0;l<8;++l)d.identity_data[l]=fixtures.at(d.identity_tile).id.at(d.identity_address*8+l);

   d.eval();
   if(held) {
    if(!d.result_valid || d.result_tile!=ht || d.result_address!=ha)return 8;
    for(int l=0;l<8;++l)if(d.result_data[l]!=hd[l])return 9;
   }
   held=d.result_valid&&!d.result_ready;
   if(held){ht=d.result_tile;ha=d.result_address;for(int l=0;l<8;++l)hd[l]=d.result_data[l];}
   heldp=d.parameter_request_valid&&!d.parameter_valid;pk=d.parameter_kind;pa=d.parameter_address;
   helds=d.source_request_valid&&!d.source_valid;sa=d.source_address;st=d.source_tile;
   heldo=d.origin_request_valid&&!d.origin_valid;ot=d.origin_tile;
   heldi=d.identity_request_valid&&!d.identity_valid;ia=d.identity_address;it=d.identity_tile;
   param_count+=d.parameter_request_valid&&d.parameter_valid;
   source_count+=d.source_request_valid&&d.source_valid;
   origin_count+=d.origin_request_valid&&d.origin_valid;
   identity_count+=d.identity_request_valid&&d.identity_valid;
   if(d.j_monitor_valid) {
    unsigned tile=first+js/480,row=js%480;
    if(d.result_tile!=tile || d.j_monitor_address!=row)return 27;
    for(int l=0;l<8;++l)if(d.j_monitor_data[l]!=fixtures.at(tile).j.at(row*8+l))return 28;
    ++js;
   }
   if(d.raw_monitor_valid) {
    unsigned tile=first+raws/480,row=raws%480;
    if(d.result_tile!=tile || d.raw_monitor_address!=row)return 19;
    for(int l=0;l<8;++l)if(d.raw_monitor_data[l]!=fixtures.at(tile).raw.at(row*8+l)){std::cerr<<"raw mismatch mode="<<active_mode<<" row="<<raws%480<<" lane="<<l<<" got="<<int32_t(d.raw_monitor_data[l])<<"\n";return 20;}
    ++raws;
   }
   if(d.wide_monitor_valid) {
    unsigned tile=first+wides/480,row=wides%480;
    if(d.result_tile!=tile||d.wide_monitor_address!=row)return 44;
    for(int l=0;l<16;++l)if(d.wide_monitor_data[l]!=fixtures.at(tile).wide.at(row*16+l))return 45;
    ++wides;
   }
   if(d.result_valid&&d.result_ready) {
    unsigned tile=first+outputs/480,row=outputs%480;
    if(d.result_tile!=tile || d.result_address!=row || bool(d.result_tile_last)!=(row==479) || bool(d.result_job_last)!=(tile==first+count-1 && row==479))return 4;
    for(int l=0;l<8;++l)if(d.result_data[l]!=fixtures.at(tile).gold.at(row*8+l)) {
     std::cerr<<"I24 mismatch "<<active_mode<<" "<<outputs<<" "<<l<<" got "<<int32_t(d.result_data[l])<<" gold "<<int32_t(fixtures.at(tile).gold.at(row*8+l))<<"\n";return 5;
    }
    ++outputs;
    if(outputs%(480*512)==0)std::cerr<<"PASS tiles="<<outputs/480<<" cycles="<<n<<"\n";
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
    if(d.range_fallback_tiles!=((active_mode==1&&!expected_range)?d.retired_tiles:0))return 33;
    if(d.core_normalization_issues!=(active_mode==2?10*d.retired_tiles:0))return 34;
    if(active_mode!=2&&(d.core_repair_issues||d.core_repair_fields))return 35;


    if(d.shared_alu_grants!=d.proof_issues+((active_mode==3||active_mode==4)?0:d.core_first_issues)+d.core_mac_issues+d.core_repair_issues+d.core_normalization_issues+((active_mode==20||active_mode==21)?d.core_aux_issues:0))return 36;
    if(d.borrow_grants!=((active_mode==3||active_mode==4)?d.core_first_issues:0))return 41;
    if(d.shared_wide_grants!=d.borrow_grants+d.consumer_add_issues)return 42;
    if(d.shared_z_grants!=d.core_z_vector_reads+d.core_z_scalar_reads+d.core_z_writes)return 37;
    if(d.shared_source_grants!=d.core_source_words || d.shared_weight_grants!=d.core_weight_words+d.core_metadata_reads)return 38;
    if(d.shared_psum_grants!=d.core_psum_reads+d.core_psum_writes+((active_mode==20||active_mode==21)?d.core_aux_reads+d.core_aux_writes:0) || d.conflict_cycles+d.borrow_consumer_stalls!=d.core_arbitration_stalls)return 39;

    if(outputs!=480ULL*count || raws!=480ULL*count || js!=480ULL*count || wides!=480ULL*count || d.retired_tiles!=count)return 6;
    if(param_count!=expected_static || param_count!=d.static_words || source_count!=d.external_source_words || identity_count!=480ULL*count || origin_count!=count || source_count!=1536ULL*count)return 22;
    if(d.total_cycles!=n+1)return 23;
    if(d.total_cycles!=d.window_cycles+d.launch_cycles+d.static_words+d.parameter_stalls+d.source_load_words+d.origin_words+d.source_load_stalls+d.origin_stalls+1)return 24;
    std::cout<<"{\"first_tile\":"<<first<<",\"tiles\":"<<count<<",\"mode\":"<<active_mode<<",\"stall\":"<<stall<<",\"command\":"<<command<<",\"outputs\":"<<outputs*8<<",\"raw_outputs\":"<<raws*8<<",\"J_outputs\":"<<js*8;
#define SHOW(x) std::cout<<",\"" #x "\":"<<uint64_t(d.x)
    SHOW(origin_stalls);
    std::cout<<",\"wide_outputs\":"<<wides*8;
    SHOW(shared_wide_grants);SHOW(borrow_grants);SHOW(borrow_consumer_stalls);SHOW(borrow_rr_stalls);SHOW(wide_conflict_cycles);SHOW(consumer_wide_waits);SHOW(proof_issues);SHOW(range_fallback_tiles);SHOW(proof_range_ok);SHOW(core_repair_issues);SHOW(core_repair_fields);SHOW(core_normalization_issues);SHOW(core_repair_arbitration_stalls);SHOW(core_normalization_arbitration_stalls);SHOW(total_cycles);SHOW(window_cycles);SHOW(launch_cycles);SHOW(batches);SHOW(conflict_cycles);SHOW(both_compute_cycles);SHOW(shared_source_grants);SHOW(shared_weight_grants);SHOW(shared_z_grants);SHOW(shared_psum_grants);SHOW(shared_alu_grants);SHOW(static_words);SHOW(parameter_stalls);SHOW(source_load_words);SHOW(external_source_words);SHOW(padding_words);SHOW(origin_words);SHOW(source_load_stalls);SHOW(output_beats);SHOW(retired_tiles);
    SHOW(core_bitmap_native_reads);SHOW(core_bitmap_native_issues);SHOW(core_cache_reads);SHOW(core_cache_writes);SHOW(core_bitmap_z_arbitration_stalls);SHOW(core_bitmap_alu_arbitration_stalls);SHOW(core_bitmap_weight_arbitration_stalls);SHOW(core_bitmap_hold_writes);SHOW(core_bitmap_hold_reads);SHOW(core_bitmap_row_z_arbitration_stalls);
    SHOW(core_aux_reads);SHOW(core_aux_writes);SHOW(core_aux_issues);SHOW(core_aux_weight_words);SHOW(core_aux_events);SHOW(core_metadata_reads);SHOW(core_count_checks);SHOW(core_count_bank_reads);SHOW(core_count_bank_writes);
    SHOW(core_cycles);SHOW(core_source_words);SHOW(core_weight_words);SHOW(core_second_weight_words);SHOW(core_local_source_reads);SHOW(core_z_vector_reads);SHOW(core_z_scalar_reads);SHOW(core_z_writes);SHOW(core_first_issues);SHOW(core_merged_updates);SHOW(core_psum_reads);SHOW(core_psum_writes);SHOW(core_mac_issues);SHOW(core_source_stalls);SHOW(core_weight_stalls);SHOW(core_output_stalls);SHOW(core_arbitration_stalls);
    SHOW(consumer_cycles);SHOW(consumer_raw_words);SHOW(consumer_identity_words);SHOW(consumer_coefficient_words);SHOW(consumer_mul_issues);SHOW(consumer_add_issues);SHOW(consumer_round_issues);SHOW(consumer_output_words);SHOW(consumer_identity_stalls);SHOW(consumer_raw_wait_cycles);SHOW(consumer_join_wait_cycles);SHOW(consumer_output_stalls);SHOW(consumer_saturations);SHOW(consumer_conversion_issues);SHOW(consumer_conversion_saturations);
    std::cout<<",\"wall_seconds_so_far\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<"}\n";break;
   }
   if(n==limit-1)return 7;
  }
 }
 d.final();return 0;
}
