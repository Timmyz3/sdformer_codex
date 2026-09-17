#include "Vsource_classifier.h"
#include "verilated.h"
#include <array>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <algorithm>
using namespace std;using Word=array<uint32_t,4>;
template<class T>void rd(ifstream&f,T*p,size_t n){f.read(reinterpret_cast<char*>(p),n*sizeof(T));if(!f)throw runtime_error("fixture EOF");}
void need(bool x,const string&s){if(!x)throw runtime_error(s);}
void put(vector<Word>&m,int addr,int bit,int n,int64_t v){for(int i=0;i<n;i++)if((uint64_t(v)>>i)&1)m[addr+(bit+i)/128][((bit+i)%128)/32]|=1U<<((bit+i)%32);}
uint64_t get(const uint32_t*p,int bit,int n){uint64_t x=0;for(int i=0;i<n;i++)x|=uint64_t((p[(bit+i)/32]>>((bit+i)%32))&1)<<i;return x;}
int64_t sg(const uint32_t*p,int bit,int n){uint64_t x=get(p,bit,n);return x&(1ULL<<(n-1))?int64_t(x)-int64_t(1ULL<<n):int64_t(x);}
struct Profile{vector<uint64_t>code,cl;array<uint16_t,12>roots;array<uint8_t,96>canonical,rank;};
struct Case{string name;uint32_t real;array<int32_t,960>x;};
vector<Word> memory(const Case&c,const Profile&q,int packed,const array<int16_t,100>&A,const array<int64_t,10>&tau,const array<uint8_t,1536>&D){
 vector<Word>m(8192);
 for(int ch=0;ch<96;ch++)for(int s=0;s<10;s++)put(m,ch*2+s/5,(s%5)*24,24,c.x[ch*10+s]);
 for(int i=0;i<100;i++)put(m,192,i*16,16,A[i]);
 for(int i=0;i<10;i++)put(m,205,i*48,48,tau[i]);
 for(int g=0;g<6;g++){int mask=0;for(int k=0;k<16;k++){int w=0;for(int j=0;j<16;j++)w|=int(D[(g*16+k)*16+j])<<j;mask|=w;put(m,209,(g*16+k)*16,16,w);}put(m,221,g*16,16,mask);}
 for(int i=0;i<12;i++)put(m,222+i/6,(i%6)*16,16,q.roots[i]);
 for(int i=0;i<96;i++)put(m,224,i*4,4,q.rank[i]);
 need(q.code.size()<=2180&&q.cl.size()<=2180,"graph capacity");
 auto packnode=[](uint64_t x){return (x&4095)|(((x>>16)&4095)<<12)|(((x>>32)&15)<<24);};
 for(unsigned i=0;i<q.code.size();i++)put(m,256,i*(packed?32:64),packed?32:64,packed?packnode(q.code[i]):q.code[i]);
 for(unsigned i=0;i<q.cl.size();i++)put(m,1536,i*(packed?32:64),packed?32:64,packed?packnode(q.cl[i]):q.cl[i]);
 return m;
}
void run(const Case&c,const Profile&q,int order,int mode,int bp,int packed,int prefetch,const array<int16_t,100>&A,const array<int64_t,10>&tau,const array<uint8_t,1536>&D,ofstream&csv){
 array<int64_t,960>u{};array<uint16_t,96>gate{};array<uint8_t,60>gold{};
 for(int ch=0;ch<96;ch++)for(int t=0;t<10;t++){
  int64_t v=0;for(int s=0;s<10;s++)v+=int64_t(A[t*10+s])*c.x[ch*10+s];u[ch*10+t]=v;if(v>=tau[t])gate[ch]|=1<<t;
 }
 for(int g=0;g<6;g++)for(int t=0;t<10;t++){
  int best=17,code=0;
  for(int k=0;k<16;k++){int dist=0;for(int j=0;j<16;j++)dist+=int((gate[g*16+j]>>t)&1)!=int(D[(g*16+k)*16+j]);if(dist<best){best=dist;code=k;}}
  gold[g*10+t]=mode==3?q.canonical[g*16+code]:code;
 }
 auto mem=memory(c,q,packed,A,tau,D);Vsource_classifier v;v.clk=0;v.rst_n=0;v.start_valid=0;v.req_ready=0;v.rsp_valid=0;v.out_ready=0;v.done_ready=0;v.start_pack32=packed;v.start_prefetch=prefetch;
 for(int b=0;b<8;b++)for(int j=0;j<4;j++)v.rsp_data[b][j]=0;
 for(int i=0;i<3;i++){v.clk=0;v.eval();v.clk=1;v.eval();}v.rst_n=1;v.start_mode=mode;
 struct P{bool live=false;int due=0,addr=0;};array<P,8>pnd{};array<bool,8>held{};array<int,8>prev{};
 array<bool,96>produced{};array<bool,6>received{};array<int,14>states{};
 bool started=false,outhold=false;uint64_t holdcode=0;int holdgroup=0,products=0,outs=0,words=0,gw=0,xw=0,pfw=0,reqstall=0,outstall=0;
 for(int cyc=0;cyc<20000;cyc++){
  v.clk=0;v.start_valid=!started;v.out_ready=!bp||cyc%11>2;v.req_ready=0;v.rsp_valid=0;
  for(int b=0;b<8;b++){
   if(!bp||(cyc+3*b)%9>1)v.req_ready|=1<<b;
   if(pnd[b].live&&pnd[b].due<=cyc){v.rsp_valid|=1<<b;for(int j=0;j<4;j++)v.rsp_data[b][j]=mem[pnd[b].addr][j];}
  }
  v.eval();int state=v.debug_state;states[state]++;
  if(v.start_valid&&v.start_ready)started=true;
  for(int b=0;b<8;b++){
   bool rv=(v.req_valid>>b)&1,rr=(v.req_ready>>b)&1;
   if(held[b])need(rv&&v.req_addr[b]==prev[b],"request changed during stall");held[b]=rv&&!rr;prev[b]=v.req_addr[b];if(held[b])reqstall++;
   if(((v.rsp_valid&v.rsp_ready)>>b)&1){need(pnd[b].live,"response");pnd[b].live=false;}
   if(rv&&rr){need(!pnd[b].live,"bank pending >1");int addr=v.req_addr[b];need(addr%8==b,"bank mapping");
    pnd[b]={true,cyc+1+(bp?(cyc+2*b)%5:0),addr};words++;if(addr>=256)gw++;if(state==8||state==9)pfw++;if(addr<192)xw++;
   }
  }
  if(v.producer_valid){int ch=v.producer_channel;need(ch<96&&!produced[ch],"duplicate producer channel");produced[ch]=true;products++;
   need(v.producer_gate==gate[ch],"producer gate "+c.name+" ch="+to_string(ch));
   for(int t=0;t<10;t++)need(sg(v.producer_u,t*48,48)==u[ch*10+t],"producer actual MAC");
  }
  if(outhold)need(v.out_valid&&v.out_group==holdgroup&&v.out_code==holdcode,"output unstable");
  outhold=v.out_valid&&!v.out_ready;if(outhold){holdcode=v.out_code;holdgroup=v.out_group;outstall++;}
  if(v.out_valid&&v.out_ready){int g=v.out_group;need(g<6&&!received[g],"duplicate group");received[g]=true;outs++;
   for(int t=0;t<10;t++)if(((v.out_code>>(t*4))&15)!=gold[g*10+t])throw runtime_error("classification "+c.name+" mode="+to_string(mode)+" g="+to_string(g)+" t="+to_string(t));
  }
  if(v.done_valid){need(outs==6,"incomplete");need(products==int(v.count_channels)&&int(v.count_mac)==products*100,"producer accounting");need(xw==products*2&&xw==int(v.count_source_words)&&gw==int(v.count_graph_words),"word accounting");
   if(mode==0)need(products==96,"full source");if(mode==1)need(products==64,"static source");
   need(pfw==int(v.count_prefetch_words),"prefetch count");need(words==gw+xw+(mode==0?29:mode==1?30:21),"configuration count");
   csv<<c.name<<','<<c.real<<','<<order<<','<<mode<<','<<bp<<','<<packed<<','<<prefetch<<','<<cyc<<','<<products<<','<<v.count_mac<<','<<words<<','<<xw<<','<<gw<<','<<pfw<<','<<v.count_graph_hits<<','<<reqstall<<','<<outstall;
   for(int i=0;i<14;i++)csv<<','<<states[i];csv<<'\n';return;
  }
  v.clk=1;v.eval();
 }
 throw runtime_error("timeout "+c.name+" st="+to_string(v.debug_state));
}
int main(int argc,char**argv){try{
 Verilated::commandArgs(argc,argv);ifstream f("source.bin",ios::binary);array<int16_t,100>A;array<int64_t,10>tau;array<uint8_t,1536>D;rd(f,A.data(),100);rd(f,tau.data(),10);rd(f,D.data(),1536);array<Profile,2>profiles;
 for(auto&q:profiles){uint32_t n;rd(f,&n,1);q.code.resize(n);rd(f,q.code.data(),n);rd(f,&n,1);q.cl.resize(n);rd(f,q.cl.data(),n);rd(f,q.roots.data(),12);rd(f,q.canonical.data(),96);rd(f,q.rank.data(),96);}
 uint32_t count;rd(f,&count,1);ofstream csv("source_cycles.csv");csv<<"case,real,order,mode,bp,packed32,prefetch,cycles,channels,scalar_mac,words,xwords,graph_words,prefetch_words,graph_hits,req_stall,out_stall";for(int i=0;i<14;i++)csv<<",state"<<i;csv<<'\n';
 for(unsigned i=0;i<count;i++){
  Case c;uint32_t n;rd(f,&c.real,1);rd(f,&n,1);c.name.resize(n);rd(f,c.name.data(),n);rd(f,c.x.data(),960);
  for(int o=0;o<2;o++)for(int m=0;m<4;m++)for(int b=0;b<2;b++){
   if(argc>2){
    string select=argv[2];need(select=="--strong-only"||select=="--class-only","selection flag");
    if(o!=1||m==0||(select=="--class-only"&&m!=3))continue;
    run(c,profiles[o],o,m,b,1,1,A,tau,D,csv);continue;
   }
   if(m<2){run(c,profiles[o],o,m,b,1,0,A,tau,D,csv);run(c,profiles[o],o,m,b,1,1,A,tau,D,csv);}
   else{run(c,profiles[o],o,m,b,0,0,A,tau,D,csv);run(c,profiles[o],o,m,b,1,0,A,tau,D,csv);run(c,profiles[o],o,m,b,1,1,A,tau,D,csv);}
  }
  cout<<"PASS "<<i<<' '<<c.name<<endl;if(argc>1&&i+1>=unsigned(stoi(argv[1])))break;
 }
 cout<<"ALL SOURCE MAC / CLASSIFICATION / PROTOCOL CHECKS PASS\n";
 }catch(exception&e){cerr<<e.what()<<endl;return 1;}return 0;}
