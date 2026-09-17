#include "Vpatch_core.h"
#include "verilated.h"
#include <array>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <algorithm>
using namespace std;using Word=array<uint32_t,4>;double sc_time_stamp(){return 0;}
void need(bool x,const string&s){if(!x)throw runtime_error(s);}
template<class T>void rd(ifstream&f,T*p,size_t n){f.read(reinterpret_cast<char*>(p),n*sizeof(T));need(bool(f),"fixture EOF");}
uint64_t bits(const uint32_t*p,int at,int n){uint64_t v=0;for(int i=0;i<n;i++)v|=uint64_t((p[(at+i)/32]>>((at+i)%32))&1)<<i;return v;}
int64_t sg(const uint32_t*p,int at,int n){auto v=bits(p,at,n);return v&(1ULL<<(n-1))?int64_t(v)-int64_t(1ULL<<n):int64_t(v);}
struct Params{int R;vector<int8_t>u;vector<uint8_t>v;array<int16_t,100>a;array<int64_t,960>tau;array<uint32_t,8>masks;};
struct Case{string name;uint32_t group,real;array<uint16_t,3456>source;vector<int16_t>z;vector<int32_t>y,q;vector<int64_t>out;vector<uint8_t>gate;};
int64_t coeff(uint8_t c){return c&32?(c&16?-1LL:1LL)*(1LL<<(c&15)):0;}
void oracle(const Params&p,const Case&c){
 vector<int64_t>z(40*p.R),y(3840),q(40*p.R),out(3840);uint32_t mask=p.masks[c.group/2400];
 for(int pos=0;pos<40;pos++)for(int r=0;r<p.R;r++)if((mask>>(r/4))&1){
  int64_t s=0;for(int k=0;k<864;k++)if((c.source[k*4+pos/10]>>(pos%10))&1)s+=p.u[k*p.R+r];z[pos*p.R+r]=s;
 }
 for(int pos=0;pos<40;pos++)for(int h=0;h<96;h++)for(int r=0;r<p.R;r++)y[pos*96+h]+=z[pos*p.R+r]*coeff(p.v[r*96+h]);
 for(int pos=0;pos<40;pos++){
  for(int r=0;r<p.R;r++)for(int s=0;s<10;s++)q[pos*p.R+r]+=int64_t(p.a[(pos%10)*10+s])*z[(pos/10*10+s)*p.R+r];
  for(int h=0;h<96;h++){
   int64_t a=0,b=0;for(int s=0;s<10;s++)a+=int64_t(p.a[(pos%10)*10+s])*y[(pos/10*10+s)*96+h];
   for(int r=0;r<p.R;r++)b+=q[pos*p.R+r]*coeff(p.v[r*96+h]);need(a==b,"independent integer VA/AV");out[pos*96+h]=a;
  }
 }
 need(equal(z.begin(),z.end(),c.z.begin())&&equal(y.begin(),y.end(),c.y.begin())&&equal(q.begin(),q.end(),c.q.begin())&&out==c.out,"serialized gold differs from independent C++ source oracle");
 for(int i=0;i<3840;i++)need(c.gate[i]==(out[i]>=p.tau[((i/96)%10)*96+i%96]),"independent gate");
}
struct Memory{vector<Word>m;vector<uint8_t>valid;Memory():m(16384),valid(16384){}
 void put(int addr,int bit,int n,int64_t value){for(int i=0;i<n;i++){int a=addr+(bit+i)/128,b=(bit+i)%128;need(a<16384,"memory capacity");valid[a]=1;if((uint64_t(value)>>i)&1)m[a][b/32]|=1U<<(b%32);}}
};
Memory memory(const Params&p,const Case&c){Memory m;m.put(0,0,8,p.R);
 for(int g=0;g<8;g++)m.put(1+(g/4),(g%4)*32,24,p.masks[g]);
 for(int i=0;i<100;i++)m.put(3,i*16,16,p.a[i]);
 for(int hb=0;hb<12;hb++)for(int t=0;t<10;t++)for(int l=0;l<8;l++)m.put(16+(hb*10+t)*3,l*48,48,p.tau[t*96+hb*8+l]);
 for(int k=0;k<864;k++)for(int r=0;r<p.R;r++)m.put(512+k*(p.R/16)+r/16,(r%16)*8,8,p.u[k*p.R+r]);
 for(int r=0;r<p.R;r++)for(int hb=0;hb<12;hb++)for(int l=0;l<8;l++)m.put(6000+r*12+hb,l*16,6,p.v[r*96+hb*8+l]);
 for(int k=0;k<864;k++)for(int pt=0;pt<4;pt++)m.put(8192+k/3,(k%3)*40+pt*10,10,c.source[k*4+pt]);return m;
}
void run(Vpatch_core&v,const Params&p,const Case&c,const string&fn,int tc,int av,int bp,int pass,ofstream&csv){
 auto mem=memory(p,c);vector<int>ranks;for(int r=0;r<p.R;r++)if(!tc||((p.masks[c.group/2400]>>(r/4))&1))ranks.push_back(r);int n=ranks.size(),nb=n/8;
 vector<int64_t>zs(480*8),ss(480*8);vector<bool>zi(480),si(480),uf(480);int zphase=0,sphase=0,finals=0,outs=0,tcseen=0,zwrites=0,swrites=0;
 bool started=false,pending=false,held=false,outhold=false,donehold=false;int due=0,addr=0,lastaddr=0,holdrow=0;array<uint32_t,3>holdgate{};
 int firstsource=-1,lastgate=-1,words=0,reqstall=0,outstall=0;array<int,5>phase{};array<int,64>state{};array<int,6>wordkind{};
 v.start_group=c.group;v.start_tc=tc;v.start_av=av;
 for(int cyc=0;cyc<3000000;cyc++){
  v.clk=0;v.start_valid=!started;v.req_ready=!bp||cyc%9>1;v.rsp_valid=pending&&due<=cyc;v.out_ready=!bp||cyc%11>2;v.done_ready=!bp||cyc%13>1;
  if(v.rsp_valid)copy_n(mem.m[addr].data(),4,v.rsp_data);v.eval();need(v.debug_phase<5&&v.debug_state<64,"debug range");phase[v.debug_phase]++;state[v.debug_state]++;
  if(v.start_valid&&v.start_ready)started=true;
  if(held)need(v.req_valid&&v.req_addr==lastaddr,"request changed while stalled");held=v.req_valid&&!v.req_ready;lastaddr=v.req_addr;if(held)reqstall++;
  if(v.rsp_valid&&v.rsp_ready){need(pending,"response owner");pending=false;}
  if(v.req_valid&&v.req_ready){need(!pending,"more than one outstanding shared 128bit service");addr=v.req_addr;need(addr<16384&&mem.valid[addr],"uninitialized read addr="+to_string(addr));pending=true;due=cyc+1+(bp?cyc%5:0);words++;
   int kind=addr>=8192?0:addr>=6000?2:addr>=512?1:addr>=16?4:addr>=3?3:5;wordkind[kind]++;if(kind==0&&firstsource<0)firstsource=cyc;
  }
  if(v.tc_valid){int want=-1,j=0;for(int k=0;k<p.R/4;k++)if((p.masks[c.group/2400]>>k)&1){if(j==tcseen)want=k;j++;}
   need(int(v.tc_compact)==tcseen&&int(v.tc_global)==want,"TC actual mask decode");tcseen++;
  }
  if(v.z_write_valid){int row=v.z_write_row;need(row<40*nb,"Z capacity/address");zi[row]=true;zwrites++;for(int l=0;l<8;l++)zs[row*8+l]=sg(v.z_write_data,l*16,16);}
  if(v.scratch_write_valid){int row=v.scratch_write_row;need(row<(av?40*nb:480),"scratch capacity/address");si[row]=true;swrites++;for(int l=0;l<8;l++)ss[row*8+l]=sg(v.scratch_write_data,l*48,48);}
  if(v.phase_valid){
   if(v.phase_kind==1){need(!zphase,"duplicate Z phase");zphase++;
    for(int pos=0;pos<40;pos++)for(int r=0;r<n;r++){int row=pos*nb+r/8;need(zi[row]&&zs[row*8+r%8]==c.z[pos*p.R+ranks[r]],"actual compact/dense Z mismatch "+c.name+" pos="+to_string(pos)+" r="+to_string(r));}
   }else {need(!sphase&&int(v.phase_kind)==(av?3:2),"scratch phase kind");sphase++;
    for(int pos=0;pos<40;pos++)for(int col=0;col<(av?n:96);col++){int row=pos*(av?nb:12)+col/8;int64_t want=av?c.q[pos*p.R+ranks[col]]:c.y[pos*96+col];
     need(si[row]&&ss[row*8+col%8]==want,"actual Y/Q mismatch "+c.name+" pos="+to_string(pos)+" col="+to_string(col)+" got="+to_string(ss[row*8+col%8])+" want="+to_string(want));}
   }
  }
  if(v.final_valid){int pos=v.final_row,hb=v.final_hblock;need(pos<40&&hb<12&&!uf[pos*12+hb],"duplicate final H8");uf[pos*12+hb]=true;finals++;
   for(int l=0;l<8;l++){int i=pos*96+hb*8+l;need(sg(v.final_u,l*48,48)==c.out[i]&&((v.final_gate>>l)&1)==c.gate[i],"final U/gate mismatch "+c.name+" pos="+to_string(pos)+" h="+to_string(hb*8+l)+" got="+to_string(sg(v.final_u,l*48,48))+" want="+to_string(c.out[i]));}
  }
  if(outhold)need(v.out_valid&&int(v.out_row)==holdrow&&equal(holdgate.begin(),holdgate.end(),v.out_gate),"output not held");
  outhold=v.out_valid&&!v.out_ready;if(outhold){outstall++;holdrow=v.out_row;copy_n(v.out_gate,3,holdgate.begin());}
  if(v.out_valid&&v.out_ready){need(v.out_row==outs,"ordered 96bit gate pack");for(int h=0;h<96;h++)need(bits(v.out_gate,h,1)==c.gate[outs*96+h],"final packed gate");outs++;lastgate=cyc;}
  if(donehold)need(v.done_valid,"done not held");donehold=v.done_valid&&!v.done_ready;
  if(v.done_valid&&v.done_ready){
   need(!pending&&!held&&outs==40&&finals==480&&zphase==1&&sphase==1,"complete endpoint transaction");
   need(tcseen==(tc?__builtin_popcount(p.masks[c.group/2400]):0)&&int(v.debug_nlatent)==n,"TC count and compact extent");
   need(wordkind[0]==288&&wordkind[2]==__builtin_popcount(p.masks[c.group/2400])*4*12&&wordkind[3]==13&&wordkind[4]==360&&wordkind[5]==2,"fixed semantic physical traffic");
   need(v.count_source_words==wordkind[0]&&v.count_U_words==wordkind[1]&&v.count_V_words==wordkind[2]&&v.count_A_words==wordkind[3]&&v.count_tau_words==wordkind[4]&&v.count_config_words==wordkind[5],"word counters");
   need(int(v.count_U_updates)==zwrites-40*nb&&swrites==(av?40*nb:480),"actual intermediate write accounting");
   csv<<fn<<','<<c.name<<','<<c.real<<','<<c.group<<','<<tc<<','<<av<<','<<bp<<','<<pass<<','<<lastgate+1<<','<<cyc+1<<','<<lastgate-firstsource+1<<','<<words<<','<<words*16;
   for(int x:wordkind)csv<<','<<x;csv<<','<<v.count_U_updates<<','<<v.count_V_updates<<','<<v.count_A_updates<<','<<v.count_A_scalar_mac<<','<<v.count_U_cache_hits<<','<<v.count_zero_source<<','<<zwrites<<','<<swrites<<','<<reqstall<<','<<outstall;
   for(int x:phase)csv<<','<<x;for(int x:state)csv<<','<<x;csv<<'\n';csv.flush();
   v.clk=1;v.eval();v.clk=0;v.start_valid=0;v.rsp_valid=0;v.done_ready=0;v.eval();need(v.start_ready,"no-reset command reuse");return;
  }
  v.clk=1;v.eval();
 }
 throw runtime_error("timeout "+c.name+" state="+to_string(v.debug_state));
}
int main(int argc,char**argv){try{
 Verilated::commandArgs(argc,argv);need(argc>=6,"usage input output function small|full passes");ifstream f(argv[1],ios::binary);array<char,8>magic;rd(f,magic.data(),8);need(string(magic.data(),8)=="PATCH001","fixture version");
 array<uint32_t,3>head;rd(f,head.data(),3);Params p;p.R=head[0];vector<uint32_t>small(head[2]);rd(f,small.data(),small.size());p.u.resize(864*p.R);p.v.resize(p.R*96);
 rd(f,p.u.data(),p.u.size());rd(f,p.v.data(),p.v.size());rd(f,p.a.data(),100);rd(f,p.tau.data(),960);rd(f,p.masks.data(),8);
 vector<Case>cases(head[1]);for(auto&c:cases){uint32_t len;rd(f,&c.group,1);rd(f,&c.real,1);rd(f,&len,1);c.name.resize(len);rd(f,c.name.data(),len);rd(f,c.source.data(),3456);c.z.resize(40*p.R);c.y.resize(3840);c.q.resize(40*p.R);c.out.resize(3840);c.gate.resize(3840);
  rd(f,c.z.data(),c.z.size());rd(f,c.y.data(),c.y.size());rd(f,c.q.data(),c.q.size());rd(f,c.out.data(),c.out.size());rd(f,c.gate.data(),c.gate.size());}
 vector<uint32_t>selection=small;if(string(argv[4])=="full"){selection.clear();for(unsigned i=0;i<cases.size();i++)if(cases[i].real)selection.push_back(i);}
 for(auto i:selection)oracle(p,cases[i]);
 Vpatch_core v;v.clk=0;v.rst_n=0;v.start_valid=0;v.req_ready=0;v.rsp_valid=0;v.out_ready=0;v.done_ready=0;for(int i=0;i<4;i++)v.rsp_data[i]=0;
 for(int i=0;i<3;i++){v.clk=0;v.eval();v.clk=1;v.eval();}v.clk=0;v.rst_n=1;
 ofstream csv(argv[2]);csv<<"function,case,real,group,tc,av,bp,pass,cycles_last_gate,cycles_done,first_source_to_last_gate,words,bytes,source_words,U_words,V_words,A_words,tau_words,config_words,U_updates,V_updates,A_updates,A_scalar_mac,U_cache_reads,zero_source_words,Z_writes,scratch_writes,req_stall,out_stall";
 for(int i=0;i<5;i++)csv<<",phase"<<i;for(int i=0;i<64;i++)csv<<",state"<<i;csv<<'\n';int commands=0;
 for(int pass=0;pass<stoi(argv[5]);pass++)for(auto i:selection){for(int tc=0;tc<2;tc++)if(argc<=6||tc==stoi(argv[6]))for(int av=0;av<2;av++)for(int bp=0;bp<2;bp++){run(v,p,cases[i],argv[3],tc,av,bp,pass,csv);commands++;}
  cout<<"PASS "<<argv[3]<<' '<<cases[i].name<<" pass="<<pass<<endl;
 }
 cout<<"ALL REAL U / TC-TR / Z / Y-or-Q / FINAL U / 96BIT GATE / SHARED128 PASS; commands="<<commands<<" final_values="<<commands*3840LL<<" initial_reset_only=1"<<endl;
 }catch(exception&e){cerr<<e.what()<<endl;return 1;}return 0;
}
