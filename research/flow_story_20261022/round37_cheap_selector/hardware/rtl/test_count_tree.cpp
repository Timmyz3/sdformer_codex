#include "Vcount_tree.h"
#include "verilated.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using U=uint64_t;
static void req(bool x,const char* s){if(!x)throw std::runtime_error(s);}
static std::vector<uint32_t> words(const std::string& p,size_t n){std::ifstream f(p,std::ios::binary);req(bool(f),"open vector");std::vector<uint32_t> v(n);f.read((char*)v.data(),n*4);req(f.gcount()==std::streamsize(n*4),"complete vector");return v;}
static std::vector<uint32_t> npy(const std::string&p){std::ifstream f(p,std::ios::binary);char m[8];f.read(m,8);req(!std::memcmp(m,"\x93NUMPY",6)&&m[6]==1,"npy");uint16_t n;f.read((char*)&n,2);f.seekg(n,std::ios::cur);std::vector<uint32_t> v(3072000);f.read((char*)v.data(),v.size()*4);req(f.gcount()==std::streamsize(v.size()*4),"full actual frame");return v;}
struct Bench {
 Vcount_tree d;U cycle=0;uint32_t rng=937;
 void tick(){d.clk=0;d.eval();d.clk=1;d.eval();cycle++;d.clk=0;d.eval();}
 uint32_t random(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
 Bench(){d.rst_n=0;d.cfg_we=0;d.frame_start=0;d.in_valid=0;d.in_last=0;d.out_ready=0;d.debug_count_index=0;tick();tick();d.rst_n=1;tick();}
 void configure(const std::vector<uint32_t>&c){req(d.cfg_ready,"configuration only idle");for(int i=0;i<16;i++){d.cfg_we=1;d.cfg_addr=i;d.cfg_data=c[i];tick();}d.cfg_we=0;d.eval();}
 void start(){req(d.frame_ready,"start ready");d.frame_start=1;tick();d.frame_start=0;for(int j=0;j<31;j++)tick();req(d.in_ready,"32 count clear cycles, first at start acceptance");}
 void run(const std::vector<uint32_t>&data,const std::vector<uint32_t>&expected,int action,bool stalls){
  start();U begin=cycle,first=0,last=0,next=0,valid_at=0,accepted_at=0,held=0,offer_at=cycle,idle_gaps=0,backpressure=0;bool pending=false;
  while(true){
   req(cycle-begin<5000000,"bounded test cycles");
   if(!pending&&next<192000&&cycle>=offer_at)pending=true;
   d.in_valid=pending;
   if(pending){for(int j=0;j<16;j++)d.in_data[j]=data[next*16+j];d.in_last=next==191999;}
   d.out_ready=valid_at&&(!stalls||held>=17);
   d.eval();bool accept=d.in_valid&&d.in_ready,done=d.out_valid&&d.out_ready;
   if(d.out_valid){if(!valid_at)valid_at=cycle;req(!d.error&&d.out_action==action,"actual tree action");held++;}
   if(d.in_ready&&!d.in_valid&&next<192000)idle_gaps++;
   if(d.in_valid&&!d.in_ready)backpressure++;
   if(accept){if(next==0)first=cycle;else if(!stalls)req(cycle-last==11,"intrinsic 11-cycle beat interval");last=cycle;next++;pending=false;offer_at=cycle+1+(stalls?random()%17:0);}
   tick();
   if(done){accepted_at=cycle;break;}
  }
  d.in_valid=0;d.out_ready=0;d.eval();req(next==192000,"all actual beats accepted");
  for(int i=0;i<32;i++){d.debug_count_index=i;d.eval();req(d.debug_count==expected[i],"32 exact RTL counters");}
  if(stalls)req(idle_gaps>0&&backpressure>0,"both actual input gaps and held valid exercised");
  std::cout<<"{\"kind\":\""<<(stalls?"random_stalls":"continuous")<<"\",\"accepted_beats\":"<<next<<",\"first_accept_cycle\":"<<first<<",\"last_accept_cycle\":"<<last<<",\"first_output_valid_cycle\":"<<valid_at<<",\"output_accepted_cycle\":"<<accepted_at<<",\"output_hold_cycles\":"<<held<<",\"input_ready_idle_cycles\":"<<idle_gaps<<",\"input_valid_backpressure_cycles\":"<<backpressure<<",\"action\":"<<action<<",\"counter_exact\":true}\n";
 }
 void bad_last(){start();d.in_valid=1;d.in_last=1;d.out_ready=0;tick();req(d.out_valid&&d.error&&d.out_action==7,"reject premature last");d.in_valid=0;d.out_ready=1;tick();d.out_ready=0;std::cout<<"{\"kind\":\"premature_last\",\"rejected\":true}\n";}
};
int main(int argc,char**argv){try{Verilated::commandArgs(argc,argv);req(argc==5,"input.npy config.u32 expectedcounts.u32 action");auto data=npy(argv[1]),config=words(argv[2],16),counts=words(argv[3],32);int action=std::stoi(argv[4]);Bench b;b.configure(config);b.run(data,counts,action,false);b.run(data,counts,action,true);b.bad_last();return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
