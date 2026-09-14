from pathlib import Path
H=Path(__file__).resolve().parent
s=(H/'tb.cpp').read_text()
s=s.replace('#include <chrono>','#include <chrono>\n#include <cstring>\n#include <fcntl.h>\n#include <sys/mman.h>\n#include <sys/stat.h>\n#include <unistd.h>')
pos=s.index('int main(')
s=s[:pos]+'''struct NpyMap {
 int fd;size_t size,offset;void* base;
 NpyMap(const std::string& path,const std::string& dtype,size_t bytes) {
  fd=open(path.c_str(),O_RDONLY);if(fd<0)throw std::runtime_error(path);
  struct stat st;if(fstat(fd,&st))throw std::runtime_error("fstat");size=st.st_size;
  base=mmap(nullptr,size,PROT_READ,MAP_PRIVATE,fd,0);if(base==MAP_FAILED)throw std::runtime_error("mmap");
  auto p=static_cast<const uint8_t*>(base);if(std::memcmp(p,"\\x93NUMPY",6))throw std::runtime_error("npy magic");
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
'''+s[pos:]
start=s.index(' Verilated::commandArgs');end=s.index(' Vconsumer_stream d;')
s=s[:start]+''' Verilated::commandArgs(argc,argv);if(argc!=12)return 2;
 NpyMap sm(argv[1],"<u2",96ULL*240*320*2),im(argv[2],"<i4",10ULL*96*240*320*4),
        pm(argv[3],"<i4",10ULL*96*240*320*4),gm(argv[4],"<i4",10ULL*96*240*320*4);
 auto src=sm.data<uint16_t>();auto id=im.data<uint32_t>();auto raw=pm.data<uint32_t>();auto gold=gm.data<uint32_t>();
 std::string dir=argv[5];unsigned mode=std::stoul(argv[6]),first=std::stoul(argv[7]),count=std::stoul(argv[8]);
 unsigned stall=std::stoul(argv[9]),repeats=std::stoul(argv[10]);uint64_t limit=std::stoull(argv[11]);
 std::vector<uint32_t> param[8];for(int k:{1,2,4,5,6,7})param[k]=readbin<uint32_t>(dir+"/param"+std::to_string(k)+".bin");
'''+s[end:]
s=s.replace('command<2','command<repeats').replace('d.first_tile=tile;d.tile_count=1','d.first_tile=first;d.tile_count=count').replace('n<6000000','n<limit')
s=s.replace('unsigned outputs=0,raws=0;','uint64_t outputs=0,raws=0;')
a=s.index('    unsigned a=d.source_address');b=s.index('\n   d.eval();',a)
s=s[:a]+'''    if(d.source_address>=7372800)return 17;
    d.source_data=src[d.source_address];
   }
   if(d.identity_request_valid) {
    if(d.identity_tile<first || d.identity_tile>=first+count)return 18;
    for(int l=0;l<8;++l)d.identity_data[l]=id[index(d.identity_tile,d.identity_address,l)];
   }
'''+s[b:]
s=s.replace('if(d.raw_monitor_address!=raws)return 19;','unsigned tile=first+raws/480,row=raws%480;\n    if(d.result_tile!=tile || d.raw_monitor_address!=row)return 19;')
s=s.replace('raw.at(raws*8+l)','raw[index(tile,row,l)]')
s=s.replace('if(d.result_tile!=tile || d.result_address!=outputs || bool(d.result_tile_last)!=(outputs==479) || bool(d.result_job_last)!=(outputs==479))return 4;',
'''unsigned tile=first+outputs/480,row=outputs%480;
    if(d.result_tile!=tile || d.result_address!=row || bool(d.result_tile_last)!=(row==479) || bool(d.result_job_last)!=(tile==first+count-1 && row==479))return 4;''')
s=s.replace('gold.at(outputs*8+l)','gold[index(tile,row,l)]')
s=s.replace('++outputs;','++outputs;\n    if(outputs%(480*512)==0)std::cerr<<"PASS tiles="<<outputs/480<<" cycles="<<n<<"\\n";')
s=s.replace('outputs!=480 || raws!=480 || d.retired_tiles!=1','outputs!=480ULL*count || raws!=480ULL*count || d.retired_tiles!=count')
s=s.replace('identity_count!=480','identity_count!=480ULL*count')
s=s.replace('std::cout<<"{\\\"mode\\\":"<<mode', 'std::cout<<"{\\\"first_tile\\\":"<<first<<",\\\"tiles\\\":"<<count<<",\\\"mode\\\":"<<mode')
s=s.replace('if(n==5999999)return 7;','if(n==limit-1)return 7;')
s=s.replace('    std::cout<<"{\\\"first_tile', '''    if(d.total_cycles!=n+1)return 23;
    if(d.total_cycles!=d.consumer_cycles+d.static_words+d.parameter_stalls+d.source_load_words+d.origin_words+d.source_load_stalls+2ULL*count+1)return 24;
    std::cout<<"{\\\"first_tile''')
(H/'stream_tb.cpp').write_text(s)
