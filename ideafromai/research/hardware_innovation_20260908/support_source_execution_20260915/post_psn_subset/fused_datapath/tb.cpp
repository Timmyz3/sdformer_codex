#include "Vsubset_psn.h"
#include "verilated.h"
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <algorithm>
struct Case{std::string name;uint32_t real;std::vector<int32_t> y;std::vector<int64_t> u,tau;std::vector<uint8_t> positive,constant,cgate,gold;};
template<class T>void readvec(std::ifstream&f,std::vector<T>&v,size_t n){v.resize(n);f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T));if(!f)throw std::runtime_error("truncated fixture");}
uint64_t bits(const WData*x,int pos,int n){uint64_t v=0;for(int i=0;i<n;i++)if((x[(pos+i)/32]>>((pos+i)%32))&1)v|=uint64_t(1)<<i;return v;}
int64_t signedbits(const WData*x,int pos,int n){uint64_t v=bits(x,pos,n);return int64_t(v<<(64-n))>>(64-n);}
void put(WData*x,int pos,int n,uint64_t v){for(int i=0;i<n;i++){uint32_t m=uint32_t(1)<<((pos+i)%32);if(v>>i&1)x[(pos+i)/32]|=m;else x[(pos+i)/32]&=~m;}}
void require(bool v,const std::string&s){if(!v)throw std::runtime_error(s);}
int main(int argc,char**argv){
 try{
  Verilated::commandArgs(argc,argv);std::ifstream f(argc>3?argv[3]:"../cases.bin",std::ios::binary);uint32_t magic,n;f.read((char*)&magic,4);f.read((char*)&n,4);require(magic==0x50534e35,"fixture magic");
  std::array<int16_t,100>a;f.read((char*)a.data(),200);std::vector<Case>cases;
  for(unsigned j=0;j<n;j++){Case c;char name[64];f.read(name,64);c.name=name;f.read((char*)&c.real,4);readvec(f,c.y,30720);readvec(f,c.u,30720);readvec(f,c.tau,960);readvec(f,c.positive,96);readvec(f,c.constant,96);readvec(f,c.cgate,960);readvec(f,c.gold,30720);cases.push_back(std::move(c));}
  int limit=argc>1?std::stoi(argv[1]):int(n);std::string tag=argc>2?argv[2]:"all";
  std::ofstream out("results_"+tag+".jsonl");Vsubset_psn d;
  auto tick=[&](){d.clk=0;d.eval();d.clk=1;d.eval();d.clk=0;d.eval();};
  d.rst_n=0;d.start_valid=0;d.param_req_ready=0;d.param_rsp_valid=0;d.y_valid=0;d.out_ready=0;d.done_ready=0;for(int i=0;i<3;i++)tick();d.rst_n=1;tick();
  uint64_t total_gates=0,total_u=0,total_y=0,total_bounds=0,total_exp=0,commands=0;
  for(int ci=0;ci<limit;ci++)for(int bp=0;bp<2;bp++)for(int mode=0;mode<2;mode++)for(int warm=0;warm<2;warm++){
   const Case&c=cases[ci];std::array<std::array<uint32_t,4>,503>params{};
   for(int j=0;j<100;j++)put(params[j/8].data(),(j%8)*16,16,uint16_t(a[j]));
   for(int j=0;j<960;j++)put(params[13+j/2].data(),(j%2)*48,48,uint64_t(c.tau[j]));
   for(int h=0;h<96;h++){put(params[493].data(),h,1,c.positive[h]);put(params[494].data(),h,1,c.constant[h]);}
   for(int j=0;j<960;j++)put(params[495+j/128].data(),j%128,1,c.cgate[j]);
   require(d.start_ready,"start not ready");d.start_valid=1;d.start_cert=mode;d.start_reload_a=!warm;tick();d.start_valid=0;
   uint64_t cyc=1;std::array<uint64_t,21>sc{};int source_row=0,outputs=0,reqs=0,rsps=0,expected_addr=warm?13:0;
   bool pending=false;int pending_addr=0;uint64_t due=0;bool offered_y=false;
   bool req_stall=false,out_stall=false,y_stall=false;int req_hold=0;std::array<uint32_t,3>gate_hold{};std::array<uint32_t,120>u_hold{};int op_hold=0,og_hold=0;
   uint64_t ycheck=0,ucheck=0,gcheck=0,bcheck=0,echeck=0;
   while(true){
    require(cyc<200000,"watchdog "+c.name);
    d.param_req_ready=!bp || cyc%7!=1;
    d.param_rsp_valid=pending && cyc>=due;
    if(pending)for(int k=0;k<4;k++)d.param_rsp_data[k]=params[pending_addr][k];
    if(source_row<320 && !offered_y && (!bp || (cyc%11!=2 && cyc%11!=3))){
      offered_y=true;for(int k=0;k<72;k++)d.y_data[k]=0;
      for(int h=0;h<96;h++)put(d.y_data,24*h,24,uint32_t(c.y[source_row*96+h]));
    }
    d.y_valid=offered_y;d.out_ready=!bp || (cyc%13!=3 && cyc%13!=4 && cyc%13!=5);d.done_ready=!bp || (cyc%19!=5 && cyc%19!=6);d.eval();
    int st=d.dbg_state;require(st<21,"state");sc[st]++;
    if(req_stall)require(d.param_req_valid && d.param_req_addr==req_hold,"request instability");
    if(out_stall){require(d.out_valid && d.out_p==op_hold && d.out_hgroup==og_hold,"output address instability");for(int k=0;k<3;k++)require(d.out_gate[k]==gate_hold[k],"gate instability");for(int k=0;k<120;k++)require(d.out_u[k]==u_hold[k],"U instability");}
    req_stall=d.param_req_valid&&!d.param_req_ready;req_hold=d.param_req_addr;
    out_stall=d.out_valid&&!d.out_ready;op_hold=d.out_p;og_hold=d.out_hgroup;for(int k=0;k<3;k++)gate_hold[k]=d.out_gate[k];for(int k=0;k<120;k++)u_hold[k]=d.out_u[k];
    if(st==3){int half=d.mon_table_addr/32,code=d.mon_table_addr%32;for(int t=0;t<10;t++){int ref=0;for(int b=0;b<5;b++)if(code>>b&1)ref+=a[t*10+half*5+b];require(signedbits(d.mon_table,t*16,16)==ref,"table construction mismatch");}}
    if(st==9){int row=d.mon_yrow;for(int h=0;h<96;h++){require(signedbits(d.mon_y,h*24,24)==c.y[row*96+h],"Y memory mismatch");ycheck++;}}
    if(st==10){int exp=0;for(int s=0;s<10;s++)for(int j=0;j<8;j++){int64_t v=std::abs(int64_t(c.y[(d.out_p*10+s)*96+d.out_hgroup*8+j]));int e=0;while(v){v>>=1;e++;}exp=std::max(exp,e);}require(int(d.dbg_exponent)==exp,"runtime exponent mismatch");echeck++;}
    if(d.mon_lower && d.mon_upper){
      for(int i=0;i<80;i++){
        int h=d.out_hgroup*8+i%8,t=i/8;int64_t ref=c.u[(d.out_p*10+t)*96+h];
        require(signedbits(d.mon_bound,i*48,48)<=ref && signedbits(d.mon_bound_hi,i*48,48)>=ref,"simultaneous certificate bounds "+c.name);bcheck+=2;
      }
      for(int t=0;t<10;t++)for(int sign=0;sign<2;sign++){
        int64_t coefficient=0;for(int s=0;s<10;s++)coefficient+=sign?std::min(int64_t(a[t*10+s]),int64_t(0)):std::max(int64_t(a[t*10+s]),int64_t(0));
        int64_t tail=coefficient*((int64_t(1)<<d.dbg_m)-1),observed=signedbits(d.mon_tail,(2*t+sign)*48,48);
        require(observed==tail,"actual tail recurrence mismatch");
        if(d.dbg_m){int64_t delta=signedbits(d.mon_tail_delta,(2*t+sign)*48,48);require(delta==tail-coefficient && delta%2==0,"tail recurrence even exactness");}
      }
    }
    if(d.out_valid && d.out_ready){
      for(int i=0;i<80;i++)require(bits(d.subset_psn__DOT__locked,i,1),"unlocked output");
      require(int(d.out_p)*12+int(d.out_hgroup)==outputs,"output order/count");
      for(int i=0;i<80;i++){
       int h=d.out_hgroup*8+i%8,t=i/8,idx=(d.out_p*10+t)*96+h;
       if(bits(d.out_gate,i,1)!=c.gold[idx])throw std::runtime_error("gate mismatch "+c.name+" mode"+std::to_string(mode)+" p"+std::to_string(d.out_p)+" t"+std::to_string(t)+" h"+std::to_string(h));gcheck++;
       if(!mode){require(signedbits(d.out_u,i*48,48)==c.u[idx],"full U mismatch "+c.name);ucheck++;}
      }outputs++;
    }
    bool take_req=d.param_req_valid&&d.param_req_ready,take_rsp=d.param_rsp_valid&&d.param_rsp_ready,take_y=d.y_valid&&d.y_ready,done=d.done_valid&&d.done_ready;
    int ra=d.param_req_addr;
    if(take_req){require(!pending,"more than one parameter request");require(ra==expected_addr++,"parameter address");pending=true;pending_addr=ra;due=cyc+(bp?1+(cyc%4):1);reqs++;}
    if(take_rsp){require(pending,"orphan response");pending=false;rsps++;}
    if(take_y){source_row++;offered_y=false;}
    uint32_t planes=d.dbg_planes,early=d.dbg_early,yreads=d.dbg_y_reads,twrites=d.dbg_table_writes;
    tick();cyc++;
    if(done){
      require(outputs==384 && source_row==320,"retirement counts");require(reqs==(warm?490:503)&&rsps==reqs,"parameter count");require(yreads==320&&twrites==(warm?0:64),"resource counts");
      uint64_t service=0;for(int z=8;z<=19;z++)service+=sc[z];
      out<<"{\"case\":\""<<c.name<<"\",\"real\":"<<c.real<<",\"mode\":"<<mode<<",\"bp\":"<<bp<<",\"warm\":"<<warm<<",\"cycles\":"<<cyc-1<<",\"psn_service\":"<<service<<",\"param_words\":"<<reqs<<",\"y_rows_written\":"<<source_row<<",\"y_rows_read\":"<<yreads<<",\"table_rows_written\":"<<twrites<<",\"planes\":"<<planes<<",\"early_groups\":"<<early<<",\"gate_checks\":"<<gcheck<<",\"u_checks\":"<<ucheck<<",\"y_checks\":"<<ycheck<<",\"bound_checks\":"<<bcheck<<",\"exponent_checks\":"<<echeck<<",\"state_cycles\":[";
      for(int z=0;z<21;z++)out<<(z?",":"")<<sc[z];out<<"]}\n";out.flush();
      total_gates+=gcheck;total_u+=ucheck;total_y+=ycheck;total_bounds+=bcheck;total_exp+=echeck;commands++;break;
    }
   }
   if(warm&&mode&&bp)std::cout<<"PASS "<<ci<<" "<<c.name<<"\n";
  }
  std::cout<<"ALL PASS commands="<<commands<<" gates="<<total_gates<<" full_U="<<total_u<<" Y="<<total_y<<" bounds="<<total_bounds<<" exponents="<<total_exp<<"\n";
 }catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}return 0;
}
