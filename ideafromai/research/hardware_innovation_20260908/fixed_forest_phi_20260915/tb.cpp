#include "Vforest_eval.h"
#include "verilated.h"
#include <array>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>
#include <cstdint>
using U=uint64_t;
static U tick=0;
static void need(bool x,const std::string&s){if(!x)throw std::runtime_error(s+" tick="+std::to_string(tick));}
template<class T>static void read(std::istream&f,T&x){f.read(reinterpret_cast<char*>(&x),sizeof(x));need(bool(f),"read");}
static void edge(Vforest_eval&d){d.clk=0;d.eval();d.clk=1;d.eval();tick++;}
struct Case{std::string name;uint8_t kind,held;uint16_t mask[40];int8_t parent[40],w[16][8];uint16_t orig[8],delta[8],sub[8];};
static void bits(std::array<uint32_t,8>&a,unsigned bit,unsigned width,U v){
 for(unsigned b=0;b<width;b++)if(v&(U(1)<<b))a[(bit+b)/32]|=1u<<((bit+b)%32);
}
static auto memory(const Case&c,int mode){
 std::array<std::array<uint32_t,8>,5> mem{};
 for(unsigned k=0;k<16;k++)for(unsigned n=0;n<8;n++)for(unsigned b=0;b<3;b++){
   unsigned bit=(k*8+n)*3+b;if((uint8_t(c.w[k][n])>>b)&1)bits(mem[bit/256],bit%256,1,1);
 }
 const auto*center=(mode==2||mode==7)?c.orig:(mode==5?c.sub:c.delta);
 for(unsigned q=0;q<8;q++){
   bits(mem[2],q*16,16,center[q]);
   for(unsigned n=0;n<8;n++){
     int sum=0;for(unsigned k=0;k<16;k++)if(center[q]&(1u<<k))sum+=c.w[k][n];
     need(sum>=-64&&sum<=48,"PWP bound");bits(mem[3+q/4],(q%4)*64+n*8,8,uint8_t(sum));
   }
 }
 return mem;
}
static void clear(Vforest_eval&d){d.cfg_valid=0;d.start_valid=0;d.mem_req_ready=0;d.mem_rsp_valid=0;d.out_ready=0;d.done_ready=0;}
int main(int argc,char**argv){try{
 Verilated::commandArgs(argc,argv);need(argc==3,"usage runs.bin results.tsv");
 std::ifstream f(argv[1],std::ios::binary);char magic[4];f.read(magic,4);need(std::string(magic,4)=="FFP1","magic");uint32_t nr;read(f,nr);
 std::ofstream out(argv[2]);out<<"case\tkind\theld\tmode\tlazy\tbp\tcycles\tconfiguration\tmem_beats\tw_beats\tcenter_beats\tpwp_beats\tquery\talu\tfields\tbuild\tparent_reads\tjoins\tnegative_fields\tpwp_rows\tquery_native_rows\tmem_wait\tout_wait\n";
 Vforest_eval d;clear(d);d.rst_n=0;edge(d);edge(d);d.rst_n=1;edge(d);
 U tasks=0,values=0,neg=0,build=0,backpressure=0,pwpbeats=0;
 for(unsigned cidx=0;cidx<nr;cidx++){
   Case c;uint16_t len;read(f,len);c.name.resize(len);f.read(c.name.data(),len);read(f,c.kind);read(f,c.held);read(f,c.mask);read(f,c.parent);read(f,c.w);read(f,c.orig);read(f,c.delta);read(f,c.sub);
   int gold[40][8]{};
   for(int r=0;r<40;r++){
     int p=c.parent[r];need(p<40,"parent range");
     if(p>=0){need((c.mask[p]&~c.mask[r])==0,"not subset");need(c.mask[p]!=c.mask[r]||p<r,"EM order");}
     for(int n=0;n<8;n++)for(int k=0;k<16;k++)if(c.mask[r]&(1u<<k))gold[r][n]+=c.w[k][n];
   }
   for(int mode=0;mode<8;mode++)for(int lazy=0;lazy<2;lazy++)for(int bp=0;bp<2;bp++){
     clear(d);need(d.start_ready,"not idle");
     for(int r=0;r<40;r++){d.cfg_valid=1;d.cfg_row=r;d.cfg_mask=c.mask[r];d.cfg_parent=c.parent[r]<0?63:c.parent[r];edge(d);}
     d.cfg_valid=0;d.start_mode=mode;d.start_lazy=lazy;d.start_valid=1;edge(d);d.start_valid=0;
     auto mem=memory(c,mode);bool pending=false;unsigned addr=0;U due=0;bool seen[40]{};unsigned retired=0,beats[5]{},mw=0,ow=0;
     bool held_output=false;unsigned savedrow=0;std::array<uint32_t,8>saved{};U cycles=0;
     for(unsigned s=0;s<100000;s++){
       d.mem_req_ready=!pending && (!bp || s%7!=1);
       d.mem_rsp_valid=pending && tick>=due;
       if(d.mem_rsp_valid)for(int n=0;n<8;n++)d.mem_rsp_data[n]=mem[addr][n];
       d.out_ready=!bp || (s%11>=4);d.done_ready=1;d.clk=0;d.eval();
       if(held_output){need(d.out_valid&&d.out_row==savedrow,"output identity hold");for(int n=0;n<8;n++)need(d.out_data[n]==saved[n],"output payload hold");}
       held_output=d.out_valid&&!d.out_ready;
       if(held_output){savedrow=d.out_row;for(int n=0;n<8;n++)saved[n]=d.out_data[n];}
       if(d.mem_req_valid&&!d.mem_req_ready)mw++;
       if(d.out_valid&&!d.out_ready)ow++;
       if(d.mem_req_valid&&d.mem_req_ready){need(!pending,"two in flight");addr=d.mem_req_addr;need(addr<5,"address");beats[addr]++;pending=true;due=tick+1+(bp?2+(s%3):0);}
       if(d.mem_rsp_valid&&d.mem_rsp_ready){need(pending,"unowned response");pending=false;}
       if(d.out_valid&&d.out_ready){
         unsigned r=d.out_row;need(r<40&&!seen[r],"duplicate row");
         if(mode!=0&&mode!=2&&mode!=7&&c.parent[r]>=0)need(seen[c.parent[r]],"child before parent commit");
         for(int n=0;n<8;n++)need(int32_t(d.out_data[n])==gold[r][n],c.name+" mode="+std::to_string(mode)+" row="+std::to_string(r)+" n="+std::to_string(n)+" value="+std::to_string(int32_t(d.out_data[n]))+" gold="+std::to_string(gold[r][n]));
         seen[r]=true;retired++;values+=8;
       }
       bool done=d.done_valid&&d.done_ready;
       edge(d);cycles++;
       if(done){need(retired==40&&!pending,"early done");break;}
       need(s<99999,c.name+" timeout");
     }
     need(beats[0]==1&&beats[1]==1,"W load once");need(beats[2]<=unsigned(mode>=2),"center bytes");if(!lazy)need(beats[2]==unsigned(mode>=2),"eager center bytes");
     if(mode==4)need(beats[3]+beats[4]==0,"local build read external PWP");
     neg+=d.dbg_neg_fields;build+=d.dbg_build;backpressure+=ow;pwpbeats+=beats[3]+beats[4];tasks++;
     out<<c.name<<'\t'<<int(c.kind)<<'\t'<<int(c.held)<<'\t'<<mode<<'\t'<<lazy<<'\t'<<bp<<'\t'<<cycles<<"\t40\t"<<beats[0]+beats[1]+beats[2]+beats[3]+beats[4]<<'\t'<<beats[0]+beats[1]<<'\t'<<beats[2]<<'\t'<<beats[3]+beats[4]<<'\t'<<d.dbg_query<<'\t'<<d.dbg_alu<<'\t'<<d.dbg_fields<<'\t'<<d.dbg_build<<'\t'<<d.dbg_parent_reads<<'\t'<<d.dbg_joins<<'\t'<<d.dbg_neg_fields<<'\t'<<d.dbg_pwp_rows<<'\t'<<d.dbg_native_rows<<'\t'<<mw<<'\t'<<ow<<'\n';
   }
 }
 need(neg>0&&build>0&&pwpbeats>0&&backpressure>0,"missing directed execution path");
 std::cout<<"PASS tasks="<<tasks<<" exact_values="<<values<<" negative_fields="<<neg<<" build="<<build<<" output_BP="<<backpressure<<" PWP_beats="<<pwpbeats<<"\n";
 return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}}
