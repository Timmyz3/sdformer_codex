#include "Vconsumer_stream.h"
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
struct NpyMap {
 int fd;size_t size,offset;void* base;
 NpyMap(const std::string& path,const std::string& dtype,size_t bytes) {
  fd=open(path.c_str(),O_RDONLY);if(fd<0)throw std::runtime_error(path);
  struct stat st;if(fstat(fd,&st))throw std::runtime_error("fstat");size=st.st_size;
  base=mmap(nullptr,size,PROT_READ,MAP_PRIVATE,fd,0);if(base==MAP_FAILED)throw std::runtime_error("mmap");
  auto p=static_cast<const uint8_t*>(base);if(std::memcmp(p,"\x93NUMPY",6))throw std::runtime_error("npy magic");
  uint32_t n=p[8]|(uint32_t(p[9])<<8);unsigned start=10;
  if(p[6]>=2){n|=(uint32_t(p[10])<<16)|(uint32_t(p[11])<<24);start=12;}
  std::string header(reinterpret_cast<const char*>(p+start),n);
  if(header.find(dtype)==std::string::npos || header.find("False")==std::string::npos)throw std::runtime_error("npy dtype/layout");
  offset=start+n;if(size-offset!=bytes)throw std::runtime_error("npy size");
 }
 template<class T> const T* data()const{return reinterpret_cast<const T*>(static_cast<const uint8_t*>(base)+offset);}
 ~NpyMap(){munmap(base,size);close(fd);}
};
size_t index(unsigned tile,unsigned row,unsigned lane) {
 unsigned t=row%10, p=(row/10)%4, n=(row/40)*8+lane;
 return size_t(tile)*3840+t*384+n*4+p;
}
int main(int argc,char**argv) {
 Verilated::commandArgs(argc,argv);if(argc!=13)return 2;
 NpyMap sm(argv[1],"<u2",96ULL*240*320*2),im(argv[2],"<f4",10ULL*96*240*320*4),
        pm(argv[3],"<i4",10ULL*96*240*320*4),gm(argv[4],"<i4",10ULL*96*240*320*4),jm(argv[5],"<i4",10ULL*96*240*320*4);
 auto src=sm.data<uint16_t>();auto id=im.data<uint32_t>();auto raw=pm.data<uint32_t>();auto gold=gm.data<uint32_t>();auto jgold=jm.data<uint32_t>();
 std::string dir=argv[6];unsigned mode=std::stoul(argv[7]),first=std::stoul(argv[8]),count=std::stoul(argv[9]);
 unsigned stall=std::stoul(argv[10]),repeats=std::stoul(argv[11]);uint64_t limit=std::stoull(argv[12]);
 std::vector<uint32_t> param[8];for(int k:{4,5,6,7})param[k]=readbin<uint32_t>(dir+"/param"+std::to_string(k)+".bin");
 Vconsumer_stream d;auto tick=[&](){d.clk=0;d.eval();d.clk=1;d.eval();d.clk=0;d.eval();};
 d.reset_n=0;d.go=0;d.parameter_valid=0;d.source_valid=0;d.identity_valid=0;
 d.result_ready=1;d.compute_source_allow=1;d.compute_weight_allow=1;tick();tick();d.reset_n=1;
 auto started=std::chrono::steady_clock::now();
 for(unsigned command=0;command<repeats;++command) {
  d.mode=mode;d.first_tile=first;d.tile_count=count;d.go=1;tick();d.go=0;
  uint64_t outputs=0,raws=0,js=0;bool held=false,heldp=false,helds=false,heldi=false;
  uint32_t hd[8]={};unsigned ht=0,ha=0,pa=0,pk=0,sa=0,ia=0,it=0;
  uint64_t param_count=0,source_count=0,identity_count=0;
  for(uint64_t n=0;n<limit;++n) {
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
    if(d.source_address>=7372800)return 17;
    d.source_data=src[d.source_address];
   }
   if(d.identity_request_valid) {
    if(d.identity_tile<first || d.identity_tile>=first+count)return 18;
    for(int l=0;l<8;++l)d.identity_data[l]=id[index(d.identity_tile,d.identity_address,l)];
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
    unsigned tile=first+js/480,row=js%480;
    if(d.result_tile!=tile || d.j_monitor_address!=row)return 27;
    for(int l=0;l<8;++l)if(d.j_monitor_data[l]!=jgold[index(tile,row,l)])return 28;
    ++js;
   }
   if(d.raw_monitor_valid) {
    unsigned tile=first+raws/480,row=raws%480;
    if(d.result_tile!=tile || d.raw_monitor_address!=row)return 19;
    for(int l=0;l<8;++l)if(d.raw_monitor_data[l]!=raw[index(tile,row,l)])return 20;
    ++raws;
   }
   if(d.result_valid&&d.result_ready) {
    unsigned tile=first+outputs/480,row=outputs%480;
    if(d.result_tile!=tile || d.result_address!=row || bool(d.result_tile_last)!=(row==479) || bool(d.result_job_last)!=(tile==first+count-1 && row==479))return 4;
    for(int l=0;l<8;++l)if(d.result_data[l]!=gold[index(tile,row,l)]) {
     std::cerr<<"I24 mismatch "<<mode<<" "<<outputs<<" "<<l<<" got "<<int32_t(d.result_data[l])<<" gold "<<int32_t(gold[index(tile,row,l)])<<"\n";return 5;
    }
    ++outputs;
    if(outputs%(480*512)==0)std::cerr<<"PASS tiles="<<outputs/480<<" cycles="<<n<<"\n";
   }
   tick();if(d.error)return 21;
   if(d.done) {
    if(outputs!=480ULL*count || raws!=480ULL*count || js!=480ULL*count || d.retired_tiles!=count)return 6;
    if(param_count!=(command?0:1848) || param_count!=d.static_words || source_count!=d.external_source_words || identity_count!=480ULL*count)return 22;
    if(d.total_cycles!=n+1)return 23;
    if(d.total_cycles!=d.consumer_cycles+d.static_words+d.parameter_stalls+d.source_load_words+d.origin_words+d.source_load_stalls+2ULL*count+1)return 24;
    std::cout<<"{\"first_tile\":"<<first<<",\"tiles\":"<<count<<",\"mode\":"<<mode<<",\"stall\":"<<stall<<",\"command\":"<<command<<",\"outputs\":"<<outputs*8<<",\"raw_outputs\":"<<raws*8<<",\"J_outputs\":"<<js*8;
#define SHOW(x) std::cout<<",\"" #x "\":"<<d.x
    SHOW(retained_source_words);SHOW(halo_tiles);SHOW(total_cycles);SHOW(static_words);SHOW(parameter_stalls);SHOW(source_load_words);SHOW(external_source_words);SHOW(padding_words);SHOW(origin_words);SHOW(source_load_stalls);SHOW(output_beats);SHOW(retired_tiles);
    SHOW(core_cycles);SHOW(core_source_words);SHOW(core_weight_words);SHOW(core_second_weight_words);SHOW(core_local_source_reads);SHOW(core_z_vector_reads);SHOW(core_z_scalar_reads);SHOW(core_z_writes);SHOW(core_first_issues);SHOW(core_dual_updates);SHOW(core_psum_reads);SHOW(core_psum_writes);SHOW(core_mac_issues);SHOW(core_source_stalls);SHOW(core_weight_stalls);SHOW(core_output_stalls);
    SHOW(consumer_cycles);SHOW(consumer_raw_words);SHOW(consumer_identity_words);SHOW(consumer_coefficient_words);SHOW(consumer_mul_issues);SHOW(consumer_add_issues);SHOW(consumer_round_issues);SHOW(consumer_output_words);SHOW(consumer_identity_stalls);SHOW(consumer_raw_wait_cycles);SHOW(consumer_join_wait_cycles);SHOW(consumer_output_stalls);SHOW(consumer_saturations);SHOW(consumer_conversion_issues);SHOW(consumer_conversion_saturations);
    std::cout<<",\"wall_seconds_so_far\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<"}\n";break;
   }
   if(n==limit-1)return 7;
  }
 }
 d.final();return 0;
}
