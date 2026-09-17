#include "Vjoined_core.h"
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
uint64_t bits(const uint32_t*p,int at,int n){uint64_t v=0;for(int i=0;i<n;i++)v|=uint64_t((p[(at+i)/32]>>((at+i)%32))&1)<<i;return v;}
int64_t sg(const uint32_t*p,int at,int n){uint64_t v=bits(p,at,n);return v&(1ULL<<(n-1))?int64_t(v)-int64_t(1ULL<<n):int64_t(v);}
struct Params{
 array<uint8_t,1536>D;array<int16_t,100>A0,A1;array<int64_t,10>tau0;
 array<int64_t,3840>tau;array<uint8_t,384>positive,constant;array<uint8_t,3840>cgate;
 array<int8_t,36864>W,Wp;vector<uint64_t>code_nodes,class_nodes;array<uint16_t,12>roots;
 array<uint8_t,96>rank,canonical;
};
struct Case{string name;uint32_t real;vector<int32_t>x;};
struct Gold{
 vector<int64_t>source_u,U;vector<uint8_t>raw,S,Sc,gate;vector<int32_t>Y;
 array<uint8_t,1920>code,cl;
};
Gold oracle(const Params&p,const Case&c,bool projected){
 const auto&W=projected?p.Wp:p.W;Gold o;o.source_u.resize(30720);o.raw.resize(30720);o.S.resize(30720);o.Sc.resize(30720);
 o.Y.resize(122880);o.U.resize(122880);o.gate.resize(122880);
 for(int pt=0;pt<32;pt++)for(int ch=0;ch<96;ch++)for(int t=0;t<10;t++){
  int64_t u=0;for(int s=0;s<10;s++)u+=int64_t(p.A0[t*10+s])*c.x[(pt*96+ch)*10+s];
  need(u>=-(1LL<<47)&&u<(1LL<<47),"source U48 bound");o.source_u[(pt*96+ch)*10+t]=u;o.raw[(pt*10+t)*96+ch]=u>=p.tau0[t];
 }
 for(int pt=0;pt<32;pt++)for(int g=0;g<6;g++)for(int t=0;t<10;t++){
  int best=17,kbest=0;for(int k=0;k<16;k++){
   int dist=0;for(int j=0;j<16;j++)dist+=o.raw[(pt*10+t)*96+g*16+j]!=p.D[(g*16+k)*16+j];
   if(dist<best){best=dist;kbest=k;}
  }
  int cl=p.canonical[g*16+kbest];o.code[(pt*6+g)*10+t]=kbest;o.cl[(pt*6+g)*10+t]=cl;
  for(int j=0;j<16;j++){o.S[(pt*10+t)*96+g*16+j]=p.D[(g*16+kbest)*16+j];o.Sc[(pt*10+t)*96+g*16+j]=p.D[(g*16+cl)*16+j];}
 }
 for(int r=0;r<320;r++)for(int h=0;h<384;h++){
  int y=0,yc=0;for(int ch=0;ch<96;ch++){y+=int(o.S[r*96+ch])*int(W[ch*384+h]);yc+=int(o.Sc[r*96+ch])*int(W[ch*384+h]);}
  need(y>=-(1<<23)&&y<(1<<23),"Y24 bound");if(projected)need(y==yc,"full H384 response class equality");o.Y[r*384+h]=y;
 }
 for(int pt=0;pt<32;pt++)for(int t=0;t<10;t++)for(int h=0;h<384;h++){
  int64_t u=0;for(int s=0;s<10;s++)u+=int64_t(p.A1[t*10+s])*o.Y[(pt*10+s)*384+h];
  need(u>=-(1LL<<47)&&u<(1LL<<47),"backend U48 bound");int i=(pt*10+t)*384+h;o.U[i]=u;
  o.gate[i]=p.constant[h]?p.cgate[t*384+h]:(p.positive[h]?u>=p.tau[t*384+h]:u<=p.tau[t*384+h]);
 }
 return o;
}
struct Memory{
 vector<Word>m;vector<uint8_t>initialized;
 Memory():m(16384),initialized(16384){}
 void put(int addr,int bit,int n,int64_t value){for(int i=0;i<n;i++){
  int a=addr+(bit+i)/128,b=(bit+i)%128;need(a>=0&&a<16384,"memory address");initialized[a]=1;
  if((uint64_t(value)>>i)&1)m[a][b/32]|=1U<<(b%32);
 }}
};
Memory memory(const Params&p,const Case&c,bool projected){
 Memory m;const auto&W=projected?p.Wp:p.W;
 for(int pt=0;pt<32;pt++)for(int ch=0;ch<96;ch++)for(int s=0;s<10;s++)m.put(pt*192+ch*2+s/5,(s%5)*24,24,c.x[(pt*96+ch)*10+s]);
 for(int i=0;i<100;i++)m.put(6144,i*16,16,p.A0[i]);
 for(int i=0;i<10;i++)m.put(6157,i*48,48,p.tau0[i]);
 for(int g=0;g<6;g++){
  int info=0;for(int k=0;k<16;k++){int mask=0;for(int j=0;j<16;j++)mask|=int(p.D[(g*16+k)*16+j])<<j;
   info|=mask;m.put(6161,(g*16+k)*16,16,mask);m.put(8192+8077,(g*16+k)*16,16,mask);
  }m.put(6173,g*16,16,info);
 }
 for(int i=0;i<12;i++)m.put(6174+i/6,(i%6)*16,16,p.roots[i]);
 for(int i=0;i<96;i++)m.put(6176,i*4,4,p.rank[i]);
 auto putgraph=[&](int addr,const vector<uint64_t>&v){for(unsigned i=0;i<v.size();i++){
  uint64_t x=v[i];need((x&65535)<4096&&((x>>16)&65535)<4096,"node 12bit admission");
  uint32_t word=(x&4095)|(((x>>16)&4095)<<12)|(((x>>32)&15)<<24);m.put(addr,i*32,32,word);
 }};
 putgraph(6208,p.code_nodes);putgraph(7488,p.class_nodes);
 for(int hb=0;hb<4;hb++){
  array<array<int,96>,96>response{};
  for(int g=0;g<6;g++)for(int k=0;k<16;k++)for(int h=0;h<96;h++)for(int j=0;j<16;j++)
   response[g*16+k][h]+=int(p.D[(g*16+k)*16+j])*int(W[(g*16+j)*384+hb*96+h]);
  for(int g=0;g<6;g++)for(int k=0;k<16;k++){
   int canonical=k;for(int j=1;j<k;j++)if(response[g*16+j]==response[g*16+k]){canonical=j;break;}
   m.put(8192+8150+hb*3,(g*16+k)*4,4,canonical);
  }
  for(int ch=0;ch<96;ch++)for(int h=0;h<96;h++)m.put(8192+(hb*96+ch)*6,h*8,8,W[ch*384+hb*96+h]);
  int lid=0;for(int g=0;g<6;g++)for(int k=0;k<16;k++){
   int count=0;for(int j=0;j<16;j++)count+=p.D[(g*16+k)*16+j];if(!count)continue;
   for(int h=0;h<96;h++){
    int v=0;for(int j=0;j<16;j++)v+=int(p.D[(g*16+k)*16+j])*int(W[(g*16+j)*384+hb*96+h]);
    need(v>=-512&&v<=511,"LUT INT10 bound");m.put(8192+2304+(hb*90+lid)*8+h/12,(h%12)*10,10,v);
   }lid++;
  }need(lid==90,"LUT cardinality");
  for(int t=0;t<10;t++)for(int h=0;h<96;h++)m.put(8192+6624+hb*360,(t*96+h)*48,48,p.tau[t*384+hb*96+h]);
  for(int h=0;h<96;h++){m.put(8192+8089+hb*9,h,1,p.positive[hb*96+h]);m.put(8192+8089+hb*9,96+h,1,p.constant[hb*96+h]);}
  for(int t=0;t<10;t++)for(int h=0;h<96;h++)m.put(8192+8089+hb*9,192+t*96+h,1,p.cgate[t*384+hb*96+h]);
 }
 for(int i=0;i<100;i++)m.put(8192+8064,i*16,16,p.A1[i]);
 return m;
}
void run(Vjoined_core&v,const Params&p,const Case&c,const Gold&o,bool projected,const string&function_name,int mode,int bp,int pass,int nh,bool dedup,ofstream&csv){
 auto mem=memory(p,c,projected);const auto&S=mode==3?o.Sc:o.S;
 struct Pending{bool live=false;int due=0,addr=0;};array<Pending,8>pending;array<bool,8>held{};array<int,8>lastaddr{};
 vector<uint8_t>produced(3072),codes(192),writes(1920);vector<int>reads(nh);
 bool started=false,outhold=false,donehold=false;int firstx=-1,lastout=-1,outputs=0,nproduced=0,ncodes=0,nwrites=0,nreads=0;
 int words=0,reqstall=0,outstall=0,bridge_cycles=0,source_cycles=0,backend_cycles=0,backend_boot=0,backend_fc=0,backend_psn=0;
 int dwords=0,xwords=0,gwords=0,scfg=0,bcfg=0,bcoeff=0,source_starts=0,backend_starts=0;
 array<int,10>states{};vector<uint32_t>hy(72),hu(144),hg(3);int hrow=0,hb=0;
 v.start_mode=mode;v.start_last_hblock=nh-1;v.start_backend_dedup=dedup;
 for(int cyc=0;cyc<500000;cyc++){
  v.clk=0;v.start_valid=!started;v.done_ready=!bp||cyc%13>1;v.out_ready=!bp||cyc%11>2;v.req_ready=0;v.rsp_valid=0;
  for(int b=0;b<8;b++){
   if(!bp||(cyc+3*b)%9>1)v.req_ready|=1<<b;
   if(pending[b].live&&pending[b].due<=cyc){v.rsp_valid|=1<<b;copy_n(mem.m[pending[b].addr].data(),4,v.rsp_data[b]);}
  }
  v.eval();int st=v.debug_state;need(st<10,"wrapper state");states[st]++;
  if(st==1||st==2)bridge_cycles++;if(st>=3&&st<=5)source_cycles++;if(st>=6&&st<=8)backend_cycles++;
  if(st>=6&&st<=8){int bs=v.debug_backend_state;if(bs==1||bs==2)backend_boot++;if(bs==4)backend_fc++;if(bs>=5&&bs<=9)backend_psn++;}
  if(v.start_valid&&v.start_ready)started=true;
  if(st==3&&v.debug_source_state==0)source_starts++;
  if(st==6&&v.debug_backend_state==0)backend_starts++;
  for(int b=0;b<8;b++){
   bool rv=(v.req_valid>>b)&1,rr=(v.req_ready>>b)&1;
   if(held[b])need(rv&&int(v.req_addr[b])==lastaddr[b],"request changed while stalled");held[b]=rv&&!rr;lastaddr[b]=v.req_addr[b];if(held[b])reqstall++;
   if(((v.rsp_valid&v.rsp_ready)>>b)&1){need(pending[b].live,"unowned response");pending[b].live=false;}
   if(rv&&rr){
    need(!pending[b].live,"more than one outstanding per physical bank");int a=v.req_addr[b];need(a%8==b&&a<16384,"bank address");need(mem.initialized[a],"uninitialized physical memory word "+to_string(a));
    pending[b]={true,cyc+1+(bp?(cyc+2*b)%5:0),a};words++;
    if(st==1)dwords++;
    else if(st>=3&&st<=5){if(a<6144){xwords++;if(firstx<0)firstx=cyc;}else if(a>=6208)gwords++;else scfg++;}
    else if(st>=6&&st<=8){if(v.debug_backend_state==4)bcoeff++;else bcfg++;}
    else throw runtime_error("request outside grant");
   }
  }
  if(v.producer_valid){
   int pt=v.producer_p,ch=v.producer_channel;need(pt<32&&ch<96&&!produced[pt*96+ch],"source duplicate channel/P");produced[pt*96+ch]=1;nproduced++;
   for(int t=0;t<10;t++)need(sg(v.producer_u,t*48,48)==o.source_u[(pt*96+ch)*10+t]&&((v.producer_gate>>t)&1)==o.raw[(pt*10+t)*96+ch],"source actual MAC/gate mismatch");
  }
  if(v.code_valid){int pt=v.producer_p,g=v.code_group;need(pt<32&&g<6&&!codes[pt*6+g],"duplicate code group");codes[pt*6+g]=1;ncodes++;
   for(int t=0;t<10;t++)need(((v.code_data>>(t*4))&15)==(mode==3?o.cl[(pt*6+g)*10+t]:o.code[(pt*6+g)*10+t]),"nearest/class code mismatch");
  }
  if(v.bridge_write){int r=v.bridge_row,g=v.bridge_group;need(r<320&&g<6&&!writes[r*6+g],"bridge duplicate write");writes[r*6+g]=1;nwrites++;
   int expected=0;for(int j=0;j<16;j++)expected|=int(S[r*96+g*16+j])<<j;need(v.bridge_data==expected,"RTL D expansion mismatch");
  }
  if(v.bridge_read){int b=v.out_hblock,r=v.bridge_read_row;need(b<nh&&r==reads[b],"bridge read order");reads[b]++;nreads++;
   for(int ch=0;ch<96;ch++)need(bits(v.bridge_read_data,ch,1)==S[r*96+ch],"bridge read mismatch");
  }
  if(outhold)need(v.out_valid&&int(v.out_row)==hrow&&int(v.out_hblock)==hb&&equal(hy.begin(),hy.end(),v.out_y)&&equal(hu.begin(),hu.end(),v.out_u)&&equal(hg.begin(),hg.end(),v.out_gate),"output not held under BP");
  outhold=v.out_valid&&!v.out_ready;if(outhold){outstall++;hrow=v.out_row;hb=v.out_hblock;copy_n(v.out_y,72,hy.begin());copy_n(v.out_u,144,hu.begin());copy_n(v.out_gate,3,hg.begin());}
  if(v.out_valid&&v.out_ready){int b=v.out_hblock,r=v.out_row;need(b==outputs/320&&r==outputs%320,"output block/row order");outputs++;lastout=cyc;
   for(int h=0;h<96;h++){int i=r*384+b*96+h;int64_t y=sg(v.out_y,h*24,24),u=sg(v.out_u,h*48,48);bool gate=bits(v.out_gate,h,1);
    if(y!=o.Y[i]||u!=o.U[i]||gate!=bool(o.gate[i]))throw runtime_error("Y/U/gate "+c.name+" m="+to_string(mode)+" hb="+to_string(b)+" r="+to_string(r)+" h="+to_string(h)+" Y="+to_string(y)+"/"+to_string(o.Y[i])+" U="+to_string(u)+"/"+to_string(o.U[i]));
   }
  }
  if(donehold)need(v.done_valid,"done not held");donehold=v.done_valid&&!v.done_ready;
  if(v.done_valid&&v.done_ready){
   need(outputs==nh*320&&ncodes==192&&nwrites==1920&&nreads==nh*320,"incomplete joined transaction");
   need(source_starts==32&&backend_starts==nh,"start/done without reset");for(auto&a:pending)need(!a.live,"pending at done");for(auto h:held)need(!h,"held request at done");
   need(nproduced==int(v.count_source_channels)&&int(v.count_source_mac)==nproduced*100,"source MAC accounting");
   need(dwords==12&&dwords==int(v.count_dictionary_words)&&xwords==int(v.count_source_x_words)&&gwords==int(v.count_source_graph_words)&&scfg==int(v.count_source_config_words),"source word accounting");
   need(bcfg==int(v.count_backend_config_words)&&bcoeff==int(v.count_backend_coeff_words)&&words==dwords+xwords+gwords+scfg+bcfg+bcoeff,"shared word accounting");
   need(nwrites==int(v.count_bridge_writes)&&nreads==int(v.count_bridge_reads),"bridge accounting");
   if(mode<2)need(nproduced==2048&&scfg==960,"strong static64 source");else need(scfg==672,"graph per-P configuration");
   need(bcfg==nh*(mode==0?382:dedup?397:394),"backend configuration");
   csv<<function_name<<','<<c.name<<','<<c.real<<','<<mode<<','<<bp<<','<<pass<<','<<nh<<','<<lastout+1<<','<<cyc+1<<','<<lastout-firstx+1<<','<<bridge_cycles<<','<<source_cycles<<','<<backend_cycles<<','<<backend_boot<<','<<backend_fc<<','<<backend_psn<<','<<words<<','<<words*16<<','<<dwords<<','<<scfg<<','<<xwords<<','<<gwords<<','<<v.count_source_prefetch_words<<','<<bcfg<<','<<bcoeff<<','<<nproduced<<','<<v.count_source_mac<<','<<v.count_backend_mac<<','<<v.count_backend_updates<<','<<nwrites<<','<<nreads<<','<<reqstall<<','<<outstall;
   for(int n:states)csv<<','<<n;csv<<','<<(mode==0?0:dedup?4:2)<<'\n';csv.flush();
   v.clk=1;v.eval();v.clk=0;v.start_valid=0;v.done_ready=0;v.req_ready=0;v.rsp_valid=0;v.eval();need(v.start_ready,"top not reusable after done");
   return;
  }
  v.clk=1;v.eval();
 }
 throw runtime_error("joined timeout "+c.name+" st="+to_string(v.debug_state)+" src="+to_string(v.debug_source_state)+" backend="+to_string(v.debug_backend_state));
}
int main(int argc,char**argv){try{
 Verilated::commandArgs(argc,argv);string input_path=argc>6?argv[6]:"inputs.bin",output_path=argc>7?argv[7]:"cycles.csv",projected_label=argc>8?argv[8]:"response_class";
 ifstream f(input_path,ios::binary);array<char,8>magic;rd(f,magic.data(),8);need(string(magic.data(),8)=="JOIN0001","fixture version");Params p;
 rd(f,p.D.data(),1536);rd(f,p.A0.data(),100);rd(f,p.tau0.data(),10);rd(f,p.A1.data(),100);rd(f,p.tau.data(),3840);rd(f,p.positive.data(),384);rd(f,p.constant.data(),384);rd(f,p.cgate.data(),3840);rd(f,p.W.data(),36864);rd(f,p.Wp.data(),36864);
 uint32_t n;rd(f,&n,1);p.code_nodes.resize(n);rd(f,p.code_nodes.data(),n);rd(f,&n,1);p.class_nodes.resize(n);rd(f,p.class_nodes.data(),n);rd(f,p.roots.data(),12);rd(f,p.rank.data(),96);rd(f,p.canonical.data(),96);rd(f,&n,1);
 vector<Case>cases(n);for(auto&c:cases){uint32_t len;rd(f,&c.real,1);rd(f,&len,1);c.name.resize(len);rd(f,c.name.data(),len);c.x.resize(30720);rd(f,c.x.data(),30720);}
 int nc=argc>1?min<int>(n,stoi(argv[1])):n,nh=argc>2?stoi(argv[2]):4,passes=argc>3?stoi(argv[3]):2,functions=argc>4?stoi(argv[4]):2,first_mode=argc>5?stoi(argv[5]):0;bool dedup=argc>9?stoi(argv[9]):0;
 need(nh>=1&&nh<=4&&passes>=1&&functions>=1&&functions<=2&&first_mode>=0&&first_mode<=2,"arguments");
 Vjoined_core v;v.clk=0;v.rst_n=0;v.start_valid=0;v.req_ready=0;v.rsp_valid=0;v.out_ready=0;v.done_ready=0;
 for(int b=0;b<8;b++)for(int j=0;j<4;j++)v.rsp_data[b][j]=0;
 for(int i=0;i<3;i++){v.clk=0;v.eval();v.clk=1;v.eval();}v.clk=0;v.rst_n=1;
 ofstream csv(output_path);csv<<"function,case,real,mode,bp,pass,hblocks,cycles_to_last_gate,cycles_to_done,first_X_request_to_last_gate,dictionary_load_cycles,source_phase_cycles,backend_phase_cycles,backend_boot_cycles,backend_fc_cycles,backend_psn_cycles,words,bytes,dictionary_words,source_config_words,source_X_words,source_graph_words,source_prefetch_words,backend_config_words,backend_coeff_words,source_channels,source_scalar_mac,backend_vector_mac,backend_vector_updates,bridge_writes,bridge_reads,req_stall,out_stall";for(int i=0;i<10;i++)csv<<",state"<<i;csv<<",backend_mode\n";
 int commands=0;for(int pass=0;pass<passes;pass++)for(int fn=0;fn<functions;fn++)for(int ci=0;ci<nc;ci++){
  bool projected=fn==0;auto o=oracle(p,cases[ci],projected);
  string function_name=projected?projected_label:"original";
  for(int mode=first_mode;mode<(projected?4:3);mode++)for(int bp=0;bp<2;bp++){run(v,p,cases[ci],o,projected,function_name,mode,bp,pass,nh,dedup,csv);commands++;}
  cout<<"PASS "<<function_name<<' '<<cases[ci].name<<" pass="<<pass<<" H="<<nh*96<<endl;
 }
 cout<<"ALL JOINED SOURCE / D EXPANSION / FC1 / PSN / SHARED BANK CHECKS PASS; commands="<<commands<<" outputs="<<int64_t(commands)*nh*320*96<<" initial_reset_only=1"<<endl;
 }catch(exception&e){cerr<<e.what()<<endl;return 1;}return 0;
}
