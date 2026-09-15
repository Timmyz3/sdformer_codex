#include "Vsupport_fc1.h"
#include "verilated.h"
#include <array>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <algorithm>
using namespace std;
using Word=array<uint32_t,4>;
template<class T> void rd(ifstream& f,T* p,size_t n){f.read(reinterpret_cast<char*>(p),n*sizeof(T));if(!f)throw runtime_error("fixture EOF");}
void need(bool v,const string& s){if(!v)throw runtime_error(s);}
uint64_t bits(const uint32_t* p,int at,int n){uint64_t v=0;for(int i=0;i<n;i++)v|=uint64_t((p[(at+i)/32]>>((at+i)%32))&1)<<i;return v;}
int64_t signedbits(const uint32_t* p,int at,int n){uint64_t v=bits(p,at,n);return (v&(1ULL<<(n-1)))?int64_t(v)-int64_t(1ULL<<n):int64_t(v);}
void put(vector<Word>& m,int addr,int bit,int n,int64_t x){for(int i=0;i<n;i++){int a=addr+(bit+i)/128,b=(bit+i)%128;need(a>=0&&a<8192,"address");if((uint64_t(x)>>i)&1)m[a][b/32]|=1U<<(b%32);}}
struct Case{uint32_t real,hb;string name;vector<uint8_t>S,positive,constant,cgate,gold;vector<int8_t>W;vector<int64_t>tau,U;vector<int32_t>Y;};
vector<Word> memory(const Case& c,const vector<uint8_t>&D,const vector<int16_t>&A,int mode){
 vector<Word> m(8192);int lid=0;vector<vector<int>> response(96,vector<int>(96));
 for(int g=0;g<6;g++)for(int k=0;k<16;k++)for(int h=0;h<96;h++)
  for(int j=0;j<16;j++)response[g*16+k][h]+=int(D[(g*16+k)*16+j])*int(c.W[(g*16+j)*96+h]);
 // Ordinary compile-time content dedup. No future source activity supplied.
 // One class map for this H96 task; full-layer maps may differ by H block.
 for(int g=0;g<6;g++)for(int k=0;k<16;k++){
  int canonical=k;
  for(int j=1;j<k;j++)if(response[g*16+j]==response[g*16+k]){canonical=j;break;}
  put(m,8150+c.hb*3,(g*16+k)*4,4,canonical);
 }
 for(int ch=0;ch<96;ch++)for(int h=0;h<96;h++)put(m,(c.hb*96+ch)*6,h*8,8,c.W[ch*96+h]);
 for(int g=0;g<6;g++)for(int k=0;k<16;k++){
  int mask=0;for(int j=0;j<16;j++)mask|=int(D[(g*16+k)*16+j])<<j;
  put(m,8077,(g*16+k)*16,16,mask);if(!mask)continue;
  int nw=mode==1?12:8,dir=0;
  for(int h=0;h<96;h++){
   int v=0;for(int j=0;j<16;j++)v+=int(D[(g*16+k)*16+j])*int(c.W[(g*16+j)*96+h]);
   need(v>=-512&&v<=511,"INT10 table admission");
   if(mode==1)put(m,2304+(c.hb*90+lid)*nw,h*16,16,v);
   else put(m,2304+(c.hb*90+lid)*nw+h/12,(h%12)*10,10,v);
   if(v)dir|=1<<(h/12);
  }
  put(m,8125+c.hb*6,lid*8,8,dir);lid++;
 }
 need(lid==90,"dictionary size");
 for(int i=0;i<100;i++)put(m,8064,i*16,16,A[i]);
 for(int i=0;i<960;i++)put(m,6624+c.hb*360,i*48,48,c.tau[i]);
 for(int h=0;h<96;h++){put(m,8089+c.hb*9,h,1,c.positive[h]);put(m,8089+c.hb*9,96+h,1,c.constant[h]);}
 for(int i=0;i<960;i++)put(m,8089+c.hb*9,192+i,1,c.cgate[i]);
 return m;
}
void check_dense(Case& c,const vector<int16_t>&A){
 for(int r=0;r<320;r++)for(int h=0;h<96;h++){
  int64_t v=0;for(int ch=0;ch<96;ch++)v+=int(c.S[r*96+ch])*int(c.W[ch*96+h]);
  need(v==c.Y[r*96+h],"fixture dense Y");
 }
 for(int p=0;p<32;p++)for(int t=0;t<10;t++)for(int h=0;h<96;h++){
  int64_t u=0;for(int s=0;s<10;s++)u+=int64_t(A[t*10+s])*c.Y[(p*10+s)*96+h];
  bool gate=c.constant[h]?c.cgate[t*96+h]:(c.positive[h]?u>=c.tau[t*96+h]:u<=c.tau[t*96+h]);
  need(u==c.U[(p*10+t)*96+h]&&gate==bool(c.gold[(p*10+t)*96+h]),"fixture dense U/gate");
 }
}
void run(const Case& c,const vector<uint8_t>&D,const vector<int16_t>&A,int mode,int bp,ofstream& csv){
 Vsupport_fc1 v;v.clk=0;v.rst_n=0;v.start_valid=0;v.source_valid=0;v.mem_req_ready=0;v.mem_rsp_valid=0;v.out_ready=0;v.done_ready=0;
 for(int b=0;b<8;b++)for(int w=0;w<4;w++)v.mem_rsp_data[b][w]=0;
 for(int k=0;k<3;k++){v.clk=0;v.eval();v.clk=1;v.eval();}v.clk=0;v.rst_n=1;
 v.start_mode=mode;v.start_hblock=c.hb;
 auto mem=memory(c,D,A,mode);
 struct Pending{bool live=false;int due=0,addr=0;};array<Pending,8> pending;
 array<bool,8> reqhold{};array<int,8> prevaddr{};
 bool started=false,outhold=false,source_offer=false;int source=0,outputs=0,firstsource=-1,lastout=-1,fc_cycles=0,psn_cycles=0,boot_cycles=0;
 int words=0,coeff=0,outstall=0,reqstall=0;array<bool,320> seen{};
 vector<uint32_t> heldY(72),heldU(144),heldG(3);int heldrow=0;
 for(int cyc=0;cyc<200000;cyc++){
  v.clk=0;v.start_valid=!started;
  if(!source_offer&&source<320&&v.dbg_state==3&&(!bp||cyc%7!=2))source_offer=true;
  v.source_valid=source_offer;
  for(int w=0;w<3;w++)v.source_data[w]=0;
  if(source<320)for(int ch=0;ch<96;ch++)v.source_data[ch/32]|=uint32_t(c.S[source*96+ch])<<(ch%32);
  v.out_ready=!bp||cyc%11>2;v.done_ready=0;v.mem_req_ready=0;v.mem_rsp_valid=0;
  for(int b=0;b<8;b++){
   if(!bp||(cyc+3*b)%9>1)v.mem_req_ready|=1<<b;
   if(pending[b].live&&pending[b].due<=cyc){v.mem_rsp_valid|=1<<b;for(int w=0;w<4;w++)v.mem_rsp_data[b][w]=mem[pending[b].addr][w];}
  }
  v.eval();int st=v.dbg_state;
  if(st==1||st==2)boot_cycles++;if(st==4)fc_cycles++;if(st>=5&&st<=9)psn_cycles++;
  if(v.start_valid&&v.start_ready)started=true;
  if(v.source_valid&&v.source_ready){if(firstsource<0)firstsource=cyc;source++;source_offer=false;}
  for(int b=0;b<8;b++){
   bool rv=(v.mem_req_valid>>b)&1,rr=(v.mem_req_ready>>b)&1;
   if(reqhold[b])need(rv&&v.mem_req_addr[b]==prevaddr[b],"request changed while stalled");
   reqhold[b]=rv&&!rr;prevaddr[b]=v.mem_req_addr[b];if(rv&&!rr)reqstall++;
   bool rsp=((v.mem_rsp_valid&v.mem_rsp_ready)>>b)&1;
   if(rsp){need(pending[b].live,"unsolicited response");pending[b].live=false;}
   if(rv&&rr){need(!pending[b].live,"bank outstanding >1");int addr=v.mem_req_addr[b];need(addr%8==b,"bank mapping");
    pending[b]={true,cyc+1+(bp?(cyc+2*b)%5:0),addr};words++;if(st==4)coeff++;
   }
  }
  if(outhold){need(v.out_valid&&int(v.out_row)==heldrow,"output valid/row stalled");
   need(equal(heldY.begin(),heldY.end(),v.out_y)&&equal(heldU.begin(),heldU.end(),v.out_u)&&equal(heldG.begin(),heldG.end(),v.out_gate),"output data stalled");}
  outhold=v.out_valid&&!v.out_ready;
  if(outhold){outstall++;heldrow=v.out_row;copy_n(v.out_y,72,heldY.begin());copy_n(v.out_u,144,heldU.begin());copy_n(v.out_gate,3,heldG.begin());}
  if(v.out_valid&&v.out_ready){int r=v.out_row;need(r<320&&!seen[r],"output row duplicate");seen[r]=true;outputs++;lastout=cyc;
   for(int h=0;h<96;h++){
    int i=r*96+h;int64_t y=signedbits(v.out_y,h*24,24),u=signedbits(v.out_u,h*48,48);bool gate=bits(v.out_gate,h,1);
    if(y!=c.Y[i]||u!=c.U[i]||gate!=bool(c.gold[i]))throw runtime_error(c.name+" mode="+to_string(mode)+" bp="+to_string(bp)+" r="+to_string(r)+" h="+to_string(h)+" Y="+to_string(y)+"/"+to_string(c.Y[i])+" U="+to_string(u)+"/"+to_string(c.U[i]));
   }
  }
  if(v.done_valid){need(source==320&&outputs==320,"incomplete");need(coeff==int(v.dbg_coeff_words),"request count");need(v.dbg_peak_words<=24&&v.dbg_peak_desc<=4,"prefetch limit");
   csv<<c.name<<','<<c.real<<','<<mode<<','<<bp<<','<<cyc<<','<<lastout-firstsource+1<<','<<boot_cycles<<','<<fc_cycles<<','<<psn_cycles<<','<<words<<','<<coeff<<','<<v.dbg_updates<<','<<v.dbg_mac<<','<<v.dbg_jobs<<','<<v.dbg_zero_jobs<<','<<v.dbg_bank_skips<<','<<v.dbg_psn_wait<<','<<v.dbg_peak_words<<','<<v.dbg_peak_desc<<','<<outstall<<','<<reqstall<<'\n';
   return;
  }
  v.clk=1;v.eval();
 }
 throw runtime_error("timeout "+c.name+" mode="+to_string(mode)+" state="+to_string(v.dbg_state));
}
int main(int argc,char**argv){try{
 Verilated::commandArgs(argc,argv);ifstream f("cases.bin",ios::binary);uint32_t count;rd(f,&count,1);
 vector<uint8_t>D(1536);vector<int16_t>A(100);rd(f,D.data(),D.size());rd(f,A.data(),A.size());
 ofstream csv("rtl_cycles.csv");csv<<"case,real,mode,bp,cycles,source_to_gate,boot_cycles,fc_cycles,psn_cycles,words,coeff_words,updates,mac,jobs,zero_jobs,bank_skips,psn_wait,peak_words,peak_desc,out_stall,bank_req_stall\n";
 for(unsigned i=0;i<count;i++){
  Case c;uint32_t n;rd(f,&c.real,1);rd(f,&c.hb,1);rd(f,&n,1);c.name.resize(n);rd(f,c.name.data(),n);
  c.S.resize(30720);c.W.resize(9216);c.tau.resize(960);c.positive.resize(96);c.constant.resize(96);c.cgate.resize(960);c.Y.resize(30720);c.U.resize(30720);c.gold.resize(30720);
  rd(f,c.S.data(),c.S.size());rd(f,c.W.data(),c.W.size());rd(f,c.tau.data(),c.tau.size());rd(f,c.positive.data(),96);rd(f,c.constant.data(),96);rd(f,c.cgate.data(),960);rd(f,c.Y.data(),30720);rd(f,c.U.data(),30720);rd(f,c.gold.data(),30720);
  check_dense(c,A);for(int mode=0;mode<5;mode++)for(int bp=0;bp<2;bp++)run(c,D,A,mode,bp,csv);
  cout<<"PASS "<<i<<' '<<c.name<<endl;
  if(argc>1&&i+1>=unsigned(stoi(argv[1])))break;
 }
 cout<<"ALL REQUEST/NUMERIC CHECKS PASS\n";
 }catch(exception&e){cerr<<e.what()<<endl;return 1;}return 0;}
