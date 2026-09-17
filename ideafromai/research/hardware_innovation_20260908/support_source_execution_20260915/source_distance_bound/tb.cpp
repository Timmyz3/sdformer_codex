#include "Vfrontier_source.h"
#include "verilated.h"
#include <array>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <algorithm>
double sc_time_stamp(){return 0;}
using namespace std;using Word=array<uint32_t,4>;
template<class T>void rd(ifstream&f,T*p,size_t n){f.read(reinterpret_cast<char*>(p),n*sizeof(T));if(!f)throw runtime_error("fixture EOF");}
void need(bool x,const string&s){if(!x)throw runtime_error(s);}
uint64_t bits(const uint32_t*p,int at,int n){uint64_t v=0;for(int i=0;i<n;i++)v|=uint64_t((p[(at+i)/32]>>((at+i)%32))&1)<<i;return v;}
int64_t sg(const uint32_t*p,int at,int n){uint64_t v=bits(p,at,n);return v&(1ULL<<(n-1))?int64_t(v)-int64_t(1ULL<<n):int64_t(v);}
struct Params{
 array<uint8_t,1536>D;array<int16_t,100>A0,A1;array<int64_t,10>tau0;
 array<int64_t,3840>tau;array<uint8_t,384>positive,constant;array<uint8_t,3840>cgate;
 array<int8_t,36864>W,Wp;vector<uint64_t>code_nodes,class_nodes;array<uint16_t,12>roots;
 array<uint8_t,96>rank,canonical;
};
struct Case{string name;uint32_t real;vector<int32_t>x;};
struct Gold{vector<int64_t>u;vector<uint8_t>raw;array<uint8_t,1920>code,cl;};
Gold oracle(const Params&p,const Case&c){
 Gold o;o.u.resize(30720);o.raw.resize(30720);
 // Verify the loaded class metadata against every full H384 integer response.
 for(int g=0;g<6;g++)for(int k=0;k<16;k++)for(int h=0;h<384;h++){
  int a=0,b=0;for(int j=0;j<16;j++){a+=int(p.D[(g*16+k)*16+j])*p.Wp[(g*16+j)*384+h];b+=int(p.D[(g*16+p.canonical[g*16+k])*16+j])*p.Wp[(g*16+j)*384+h];}
  need(a==b,"canonical full H384 response mismatch");
 }
 for(int pt=0;pt<32;pt++)for(int ch=0;ch<96;ch++)for(int t=0;t<10;t++){
  int64_t u=0;for(int s=0;s<10;s++){auto x=c.x[(pt*96+ch)*10+s];need(x>=-(1<<23)&&x<(1<<23),"X24");u+=int64_t(p.A0[t*10+s])*x;}
  need(u>=-(1LL<<47)&&u<(1LL<<47),"U48 bound");o.u[(pt*96+ch)*10+t]=u;o.raw[(pt*10+t)*96+ch]=u>=p.tau0[t];
 }
 for(int pt=0;pt<32;pt++)for(int g=0;g<6;g++)for(int t=0;t<10;t++){
  int best=17,kbest=0;for(int k=0;k<16;k++){
   int d=0;for(int j=0;j<16;j++)d+=o.raw[(pt*10+t)*96+g*16+j]!=p.D[(g*16+k)*16+j];
   if(d<best){best=d;kbest=k;}
  }o.code[(pt*6+g)*10+t]=kbest;o.cl[(pt*6+g)*10+t]=p.canonical[g*16+kbest];
 }return o;
}
struct Memory{
 array<Word,256>m{};array<bool,256>initialized{};
 void put(int addr,int bit,int n,int64_t value){for(int i=0;i<n;i++){
  int a=addr+(bit+i)/128,b=(bit+i)%128;need(a>=0&&a<256,"memory address");initialized[a]=true;
  if((uint64_t(value)>>i)&1)m[a][b/32]|=1U<<(b%32);
 }}
};
Memory memory(const Params&p,const Case&c,int pt){
 Memory m;
 for(int ch=0;ch<96;ch++)for(int s=0;s<10;s++)m.put(ch*2+s/5,(s%5)*24,24,c.x[(pt*96+ch)*10+s]);
 for(int i=0;i<100;i++)m.put(192,i*16,16,p.A0[i]);
 for(int i=0;i<10;i++)m.put(205,i*48,48,p.tau0[i]);
 for(int g=0;g<6;g++){
  int vary=0,first=0;for(int k=0;k<16;k++){int mask=0;for(int j=0;j<16;j++)mask|=int(p.D[(g*16+k)*16+j])<<j;
   if(k==0)first=mask;vary|=mask^first;m.put(209,(g*16+k)*16,16,mask);
  }m.put(221,g*16,16,vary);
 }
 for(int i=0;i<96;i++){m.put(224,i*4,4,p.rank[i]);m.put(227,i*4,4,p.canonical[i]);}
 return m;
}
void run(Vfrontier_source&v,const Params&p,const Case&c,const Gold&o,const string&fn,int mode,int bp,int pass,int pt,ofstream&csv){
 auto mem=memory(p,c,pt);
 struct Pending{bool live=false;int due=0,addr=0;};array<Pending,8>pending;array<bool,8>held{};array<int,8>lastaddr{};
 array<bool,960>produced{};array<bool,6>codes{};bool started=false,outhold=false,donehold=false;uint64_t holdcode=0;int holdgroup=0;
 int outputs=0,nproduced=0,nrefs=0,nbatch=0,lastout=-1,firstx=-1,words=0,xwords=0,config=0,reqstall=0,outstall=0;array<int,17>states{};
 v.start_mode=mode;v.start_reuse_config=pt!=0;v.start_pack32=1;v.start_prefetch=1;v.start_frontier=1;v.start_resident=1;v.start_active_prefetch=1;
 for(int cyc=0;cyc<100000;cyc++){
  v.clk=0;v.start_valid=!started;v.done_ready=!bp||cyc%13>1;v.out_ready=!bp||cyc%11>2;v.req_ready=0;v.rsp_valid=0;
  for(int b=0;b<8;b++){
   if(!bp||(cyc+3*b)%9>1)v.req_ready|=1<<b;
   if(pending[b].live&&pending[b].due<=cyc){v.rsp_valid|=1<<b;copy_n(mem.m[pending[b].addr].data(),4,v.rsp_data[b]);}
  }
  v.eval();int st=v.debug_state;need(st<17,"state");states[st]++;
  if(v.start_valid&&v.start_ready)started=true;
  for(int b=0;b<8;b++){
   bool rv=(v.req_valid>>b)&1,rr=(v.req_ready>>b)&1;
   if(held[b])need(rv&&int(v.req_addr[b])==lastaddr[b],"request changed while stalled");held[b]=rv&&!rr;lastaddr[b]=v.req_addr[b];if(held[b])reqstall++;
   if(((v.rsp_valid&v.rsp_ready)>>b)&1){need(pending[b].live,"unowned response");pending[b].live=false;}
   if(rv&&rr){
    need(!pending[b].live,"two outstanding requests in one bank");int a=v.req_addr[b];need(a%8==b&&a<256,"bank/address");need(mem.initialized[a],"uninitialized word "+to_string(a));
    pending[b]={true,cyc+1+(bp?(cyc+2*b)%5:0),a};words++;
    if(a<192){xwords++;if(firstx<0)firstx=cyc;}else {need(st==1,"configuration outside boot");config++;}
   }
  }
  if(v.producer_valid){
   need(v.producer_active,"empty producer");array<bool,96> refs{};nbatch++;
   for(int t=0;t<10;t++)if((v.producer_active>>t)&1){
    int ch=v.producer_channel[t],i=(pt*96+ch)*10+t;
    need(ch<96&&!produced[ch*10+t],"duplicate produced (channel,t)");produced[ch*10+t]=1;nproduced++;if(!refs[ch]){refs[ch]=true;nrefs++;}
    need(sg(v.producer_u,t*48,48)==o.u[i]&&((v.producer_gate>>t)&1)==o.raw[(pt*10+t)*96+ch],"actual source MAC/gate mismatch");
   }
  }
  if(outhold)need(v.out_valid&&v.out_group==holdgroup&&v.out_code==holdcode,"output not held");
  outhold=v.out_valid&&!v.out_ready;if(outhold){outstall++;holdgroup=v.out_group;holdcode=v.out_code;}
  if(v.out_valid&&v.out_ready){int g=v.out_group;need(g==outputs&&!codes[g],"code group order");codes[g]=true;outputs++;lastout=cyc;
   for(int t=0;t<10;t++){int got=(v.out_code>>(t*4))&15,expected=mode==3?o.cl[(pt*6+g)*10+t]:o.code[(pt*6+g)*10+t];
    need(got==expected,"exact nearest/class mismatch "+c.name+" P="+to_string(pt)+" g="+to_string(g)+" t="+to_string(t)+" got="+to_string(got)+" expected="+to_string(expected));}
  }
  if(donehold)need(v.done_valid,"done not held");donehold=v.done_valid&&!v.done_ready;
  if(v.done_valid&&v.done_ready){
   need(outputs==6,"incomplete output");for(auto&a:pending)need(!a.live,"pending at done");for(auto h:held)need(!h,"request held at done");
   need(v.count_mac==nproduced*10&&v.count_channels==nrefs&&v.count_batches==nbatch&&v.count_pairs==nproduced,"source compute counts");
   need(v.count_source_words==xwords&&v.count_graph_words==0&&words==config+xwords,"physical word counts");
   need(config==(pt==0?(mode==1?30:mode==2?32:35):0),"actual configuration cost");
   if(mode>=2){int evals=60+nproduced;
    need(v.count_reduce_cycles==evals&&v.count_dominance_tests==evals*256&&v.count_popcount_ops==evals*272,"bounded 16-pop engine work");
    need(v.count_bound_cycles==18*evals+nbatch+6,"bound cycle identity");
   }else need(v.count_bound_cycles==0&&v.count_popcount_ops==960,"ordinary shared popcount cost");
   need(v.count_bound_cycles==states[14]+states[15]+states[16],"bound states equal counter");
   csv<<fn<<','<<c.name<<','<<c.real<<','<<mode<<','<<bp<<','<<pass<<','<<pt<<','<<lastout+1<<','<<cyc+1<<','<<lastout-firstx+1<<','<<words<<','<<words*16<<','<<config<<','<<xwords<<','<<v.count_prefetch_words<<','<<nproduced<<','<<nrefs<<','<<nbatch<<','<<v.count_mac<<','<<v.count_bound_cycles<<','<<v.count_popcount_ops<<','<<v.count_dominance_tests<<','<<v.count_reduce_cycles<<','<<v.count_cache_hits<<','<<v.count_refetches<<','<<v.count_peak_slots<<','<<reqstall<<','<<outstall;
   for(int x:states)csv<<','<<x;csv<<'\n';csv.flush();
   v.clk=1;v.eval();v.clk=0;v.start_valid=0;v.done_ready=0;v.req_ready=0;v.rsp_valid=0;v.eval();need(v.start_ready,"not reusable after done");return;
  }
  v.clk=1;v.eval();
 }
 throw runtime_error("source timeout "+c.name+" P="+to_string(pt)+" state="+to_string(v.debug_state));
}
int main(int argc,char**argv){try{
 Verilated::commandArgs(argc,argv);need(argc>=8,"usage input output label nc passes first_mode last_mode");
 ifstream f(argv[1],ios::binary);array<char,8>magic;rd(f,magic.data(),8);need(string(magic.data(),8)=="JOIN0001","fixture version");Params p;
 rd(f,p.D.data(),1536);rd(f,p.A0.data(),100);rd(f,p.tau0.data(),10);rd(f,p.A1.data(),100);rd(f,p.tau.data(),3840);rd(f,p.positive.data(),384);rd(f,p.constant.data(),384);rd(f,p.cgate.data(),3840);rd(f,p.W.data(),36864);rd(f,p.Wp.data(),36864);
 uint32_t n;rd(f,&n,1);p.code_nodes.resize(n);rd(f,p.code_nodes.data(),n);rd(f,&n,1);p.class_nodes.resize(n);rd(f,p.class_nodes.data(),n);rd(f,p.roots.data(),12);rd(f,p.rank.data(),96);rd(f,p.canonical.data(),96);rd(f,&n,1);
 vector<Case>cases(n);for(auto&c:cases){uint32_t len;rd(f,&c.real,1);rd(f,&len,1);c.name.resize(len);rd(f,c.name.data(),len);c.x.resize(30720);rd(f,c.x.data(),30720);}
 int nc=min<int>(n,stoi(argv[4])),passes=stoi(argv[5]),first=stoi(argv[6]),last=stoi(argv[7]);need(first>=1&&last<=3&&first<=last,"mode range");
 Vfrontier_source v;v.clk=0;v.rst_n=0;v.start_valid=0;v.req_ready=0;v.rsp_valid=0;v.out_ready=0;v.done_ready=0;
 for(int b=0;b<8;b++)for(int j=0;j<4;j++)v.rsp_data[b][j]=0;
 for(int i=0;i<3;i++){v.clk=0;v.eval();v.clk=1;v.eval();}v.clk=0;v.rst_n=1;
 ofstream csv(argv[2]);csv<<"function,case,real,mode,bp,pass,p,cycles_to_last_code,cycles_to_done,first_X_request_to_last_code,words,bytes,config_words,X_words,prefetch_words,produced_pairs,channel_refs,batches,scalar_mac,bound_cycles,popcount16_ops,dominance_tests,reduce_cycles,cache_hits,refetches,peak_slots,req_stall,out_stall";for(int i=0;i<17;i++)csv<<",state"<<i;csv<<'\n';
 int commands=0;for(int pass=0;pass<passes;pass++)for(int ci=0;ci<nc;ci++){
  auto o=oracle(p,cases[ci]);for(int mode=first;mode<=last;mode++)for(int bp=0;bp<2;bp++)for(int pt=0;pt<32;pt++){run(v,p,cases[ci],o,argv[3],mode,bp,pass,pt,csv);commands++;}
  cout<<"PASS "<<argv[3]<<' '<<cases[ci].name<<" pass="<<pass<<endl;
 }
 cout<<"ALL REAL SOURCE MAC / LOWEST-TIE CODE / FULL-H384 CLASS / BANK / BOUND COUNTS PASS; commands="<<commands<<" labels="<<commands*60<<" initial_reset_only=1"<<endl;
 }catch(exception&e){cerr<<e.what()<<endl;return 1;}return 0;
}
