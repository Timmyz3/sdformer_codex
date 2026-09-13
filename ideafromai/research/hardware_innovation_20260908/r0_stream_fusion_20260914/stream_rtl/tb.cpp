#include "Vstream_wrapper.h"
#include "verilated.h"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct NpyMap {
  int fd;size_t size,offset;void* base;
  NpyMap(const std::string& path,const std::string& dtype) {
    fd=open(path.c_str(),O_RDONLY);if(fd<0)throw std::runtime_error(path);
    struct stat st;if(fstat(fd,&st))throw std::runtime_error("fstat");size=st.st_size;
    base=mmap(nullptr,size,PROT_READ,MAP_PRIVATE,fd,0);if(base==MAP_FAILED)throw std::runtime_error("mmap");
    auto p=static_cast<const uint8_t*>(base);if(std::memcmp(p,"\x93NUMPY",6))throw std::runtime_error("npy magic");
    uint32_t hlen=p[8]|(uint32_t(p[9])<<8);unsigned start=10;
    if(p[6]>=2){hlen|=(uint32_t(p[10])<<16)|(uint32_t(p[11])<<24);start=12;}
    std::string header(reinterpret_cast<const char*>(p+start),hlen);
    if(header.find(dtype)==std::string::npos || header.find("False")==std::string::npos)throw std::runtime_error("npy dtype/layout");
    offset=start+hlen;
  }
  template<class T> const T* data()const{return reinterpret_cast<const T*>(static_cast<const uint8_t*>(base)+offset);}
  ~NpyMap(){munmap(base,size);close(fd);}
};
template<class T> std::vector<T> binary(const std::string& path,size_t count) {
  std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error(path);
  std::vector<T> out(count);f.read(reinterpret_cast<char*>(out.data()),count*sizeof(T));
  if(size_t(f.gcount())!=count*sizeof(T) || f.peek()!=EOF)throw std::runtime_error("binary size");return out;
}
int main(int argc,char** argv) {
  Verilated::commandArgs(argc,argv);if(argc!=10)return 2;
  const std::string source_path=argv[1],gold_path=argv[2],parameter_dir=argv[3];
  const unsigned mode=std::stoul(argv[4]),first=std::stoul(argv[5]),count=std::stoul(argv[6]);
  const bool stall=std::stoul(argv[7]);const unsigned repeats=std::stoul(argv[8]);
  const uint64_t limit=std::stoull(argv[9]);
  NpyMap src_file(source_path,"<u2"),gold_file(gold_path,"<i4");
  const auto src=src_file.data<uint16_t>();const auto gold=gold_file.data<int32_t>();
  if(src_file.size-src_file.offset!=96ULL*240*320*2 || gold_file.size-gold_file.offset!=19200ULL*3840*4)return 3;
  const auto w=binary<uint16_t>(parameter_dir+"/weight.bin",82944);
  const auto mask=binary<uint8_t>(parameter_dir+"/mask.bin",288);
  Vstream_wrapper d;d.reset_n=0;d.go=0;d.clk=0;d.mode=mode;d.first_tile=first;d.tile_count=count;
  d.parameter_valid=0;d.source_valid=0;d.compute_source_allow=1;d.compute_weight_allow=1;d.result_ready=1;
  auto edge=[&](){d.clk=0;d.eval();d.clk=1;d.eval();};edge();edge();d.reset_n=1;
  for(unsigned command=0;command<repeats;++command) {
    d.go=1;edge();d.go=0;
    uint64_t outputs=0;bool held=false,held_source=false,held_parameter=false;uint32_t held_data[8];
    unsigned held_tile=0,held_address=0,held_source_address=0,held_parameter_address=0;
    bool held_tile_last=false,held_job_last=false,held_parameter_mask=false;
    auto begin=std::chrono::steady_clock::now();bool finished=false;
    for(uint64_t cycle=0;cycle<limit;++cycle) {
      // Requests below are already settled after the previous rising edge.
      // All source addressing/tiling is RTL-owned. TB is a one-word elastic
      // memory responder and never supplies a gathered tile or origin.
      if(held_source && (!d.source_request_valid || d.source_address!=held_source_address))return 10;
      if(held_parameter && (!d.parameter_request_valid || d.parameter_address!=held_parameter_address || d.parameter_request_mask!=held_parameter_mask))return 11;
      d.parameter_valid=(!stall || cycle%17!=4);
      d.source_valid=(!stall || (cycle%13!=2 && cycle%13!=3));
      d.compute_source_allow=(!stall || cycle%11!=3);
      d.compute_weight_allow=(!stall || (cycle%7!=2 && cycle%7!=3));
      d.result_ready=(!stall || (cycle%5!=1 && cycle%5!=2));
      for(int j=0;j<4;++j)d.parameter_data[j]=0;
      if(d.parameter_request_valid) {
        unsigned a=d.parameter_address;
        if(d.parameter_request_mask){if(a>=288)return 12;d.parameter_data[0]=mask[a];}
        else {if(a>=10368)return 13;for(int j=0;j<4;++j)d.parameter_data[j]=w[a*8+j*2]|(uint32_t(w[a*8+j*2+1])<<16);}
      }
      if(d.source_request_valid){if(d.source_address>=7372800)return 14;d.source_data=src[d.source_address];}
      else d.source_data=0;
      d.clk=0;d.eval();
      held_source=d.source_request_valid&&!d.source_valid;
      held_source_address=d.source_address;
      held_parameter=d.parameter_request_valid&&!d.parameter_valid;
      held_parameter_address=d.parameter_address;held_parameter_mask=d.parameter_request_mask;
      if(held) {
        if(!d.result_valid || d.result_tile!=held_tile || d.result_address!=held_address || d.result_tile_last!=held_tile_last || d.result_job_last!=held_job_last)return 15;
        for(unsigned lane=0;lane<8;++lane)if(d.result_data[lane]!=held_data[lane])return 16;
      }
      held=d.result_valid&&!d.result_ready;
      if(held) {
        held_tile=d.result_tile;held_address=d.result_address;held_tile_last=d.result_tile_last;held_job_last=d.result_job_last;
        for(unsigned lane=0;lane<8;++lane)held_data[lane]=d.result_data[lane];
      }
      if(d.result_valid&&d.result_ready) {
        const unsigned tile=first+outputs/480,addr=outputs%480;
        if(tile>=first+count || d.result_tile!=tile || d.result_address!=addr)return 17;
        if(d.result_tile_last!=(addr==479) || d.result_job_last!=(tile==first+count-1 && addr==479))return 18;
        const unsigned og=addr/40,p=(addr/10)%4,t=addr%10;
        for(unsigned lane=0;lane<8;++lane) {
          const size_t index=size_t(tile)*3840+t*384+(og*8+lane)*4+p;
          if(int32_t(d.result_data[lane])!=gold[index]) {
            std::cerr<<"Mismatch mode="<<mode<<" tile="<<tile<<" addr="<<addr<<" lane="<<lane<<" got="<<int32_t(d.result_data[lane])<<" expected="<<gold[index]<<"\n";return 19;
          }
        }
        ++outputs;
        if(outputs%(480*64)==0)std::cerr<<"STREAM_PASS tiles="<<outputs/480<<" cycles="<<cycle<<"\n";
      }
      d.clk=1;d.eval();
      if(d.error)return 20;
      if(d.done) {
        if(outputs!=uint64_t(count)*480 || d.retired_tiles!=count || d.output_beats!=outputs)return 21;
        if(d.total_cycles!=cycle+1)return 22;
        if(d.static_weight_words!=(command?0:10368) || d.static_mask_words!=(command?0:288))return 23;
        const uint64_t accounted=d.core_cycles+d.static_weight_words+d.static_mask_words+d.parameter_stalls+
          d.source_load_words+d.origin_words+d.source_load_stalls+2ULL*count+1;
        if(d.total_cycles!=accounted){std::cerr<<"cycle ledger "<<d.total_cycles<<" != "<<accounted<<"\n";return 24;}
        if(d.source_load_words!=1536ULL*count || d.origin_words!=count || d.external_source_words+d.padding_words!=d.source_load_words)return 25;
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        std::cout<<"{\"mode\":"<<mode<<",\"first_tile\":"<<first<<",\"tiles\":"<<count<<",\"stall\":"<<stall<<",\"command\":"<<command;
#define FIELD(x) std::cout<<",\"" #x "\":"<<d.x
        FIELD(total_cycles);FIELD(static_weight_words);FIELD(static_mask_words);FIELD(parameter_stalls);
        FIELD(source_load_words);FIELD(external_source_words);FIELD(padding_words);FIELD(origin_words);FIELD(source_load_stalls);
        FIELD(output_beats);FIELD(retired_tiles);FIELD(core_cycles);FIELD(core_source_words);FIELD(core_weight_words);
        FIELD(core_psum_reads);FIELD(core_psum_writes);FIELD(core_sum_issues);FIELD(core_update_issues);FIELD(core_merge_issues);
        FIELD(core_source_stalls);FIELD(core_weight_stalls);FIELD(core_output_stalls);
#undef FIELD
        std::cout<<",\"checked_outputs\":"<<outputs*8<<",\"wall_seconds\":"<<seconds<<"}\n";finished=true;break;
      }
    }
    if(!finished){std::cerr<<"Timeout\n";return 26;}
  }
  d.final();return 0;
}
