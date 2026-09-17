#include "Vtest_top.h"
#include "verilated.h"
#include <array>
#include <vector>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <fstream>
using I=__int128_t;using namespace std;
constexpr int64_t MIN48=-(INT64_C(1)<<47),MAX48=(INT64_C(1)<<47)-1;
void need(bool b,const char* s){if(!b)throw runtime_error(s);}
bool fits(I x){return x>=MIN48&&x<=MAX48;}
uint64_t enc(int64_t x){return uint64_t(x)&((UINT64_C(1)<<48)-1);}
struct Bench{
 Vtest_top d;uint64_t groups=0,steps=0,ties=0,endpoints=0;array<int,32>lo{},hi{};
 void tick(){d.clk_core=0;d.eval();d.clk_core=1;d.eval();d.clk_core=0;d.eval();}
 Bench(){d.rst_core=1;d.cfg_we=0;d.context_we=0;d.codes_we=0;d.group_go=0;d.step_go=0;tick();d.rst_core=0;}
 void table(const array<int,10>&a){
  for(int half=0;half<2;half++)for(int c=0;c<32;c++){
   int z=0;for(int k=0;k<5;k++)if(c>>k&1)z+=a[half*5+k];need(z>=-32768&&z<=32767,"LUT16 admission");
   (half?hi:lo)[c]=z;d.cfg_we=1;d.cfg_half=half;d.cfg_addr=c;d.cfg_data=uint16_t(z);tick();
  }d.cfg_we=0;
 }
 void run(int64_t pref,int mm,int64_t tau,int64_t p,int64_t n,bool positive,bool constant,bool cg,vector<pair<int,int>> codes){
  need(mm>=0&&mm<=23&&p>=0&&n<=0,"context admission");
  d.context_we=1;d.prefix_in=enc(pref);d.tau_in=enc(tau);d.p_in=enc(p);d.n_in=enc(n);d.m_in=mm;
  d.lo_code_in=codes[0].first;d.hi_code_in=codes[0].second;d.positive_in=positive;d.constant_in=constant;d.constant_gate_in=cg;
  tick();d.context_we=0;d.group_go=1;tick();d.group_go=0;groups++;endpoints+=tau==MIN48||tau==MAX48;
  bool lock=constant,gate=constant&&cg;
  for(size_t i=0;i<codes.size();i++){
   if(i){d.codes_we=1;d.lo_code_in=codes[i].first;d.hi_code_in=codes[i].second;tick();d.codes_we=0;}
   I first=I(pref)*2+lo[codes[i].first],nv=first+hi[codes[i].second];I scale=I(1)<<mm;
   I base=nv*scale,tn=I(n)*(scale-1),tp=I(p)*(scale-1),L=base+tn,H=base+tp;
   need(fits(I(pref)*2)&&fits(first)&&fits(nv)&&fits(base)&&fits(tn)&&fits(tp)&&fits(L)&&fits(H),"original intermediate48 admission");
   bool lh=positive?L>=tau:L>tau,uh=positive?H<tau:H<=tau;
   if(!lock){if(lh){gate=positive;lock=true;}else if(uh){gate=!positive;lock=true;}}
   d.step_go=1;tick();d.step_go=0;
   need(d.lower_hit_out==(lh?3:0)&&d.upper_hit_out==(uh?3:0),"independent L/H hits");
   need(d.gate_out==(gate?3:0)&&d.locked_out==(lock?3:0),"registered gate/lock");
   ties+=L==tau||H==tau;steps++;pref=int64_t(nv);if(mm)mm--;
  }
 }
};
int main(int argc,char**argv){try{
 Verilated::commandArgs(argc,argv);Bench b;uint64_t rejected=0;
 vector<array<int,10>> tables={{100,-121,37,-41,12,-15,78,-82,19,-9},{32767,0,0,0,0,-32768,0,0,0,0},{0,0,0,0,0,0,0,0,0,0}};
 for(auto a:tables){b.table(a);int64_t p=0,n=0;for(int x:a){p+=max(x,0);n+=min(x,0);}
  for(int m=0;m<24;m++)for(int positive=0;positive<2;positive++)for(int c=0;c<32;c++){
   int h=(c*13+7)%32;int64_t pref=(int64_t(c)-16)*983;I nv=I(pref)*2+b.lo[c]+b.hi[h],base=nv*(I(1)<<m),L=base+I(n)*((I(1)<<m)-1),H=base+I(p)*((I(1)<<m)-1);
   vector<I> ts={MIN48,MAX48,0,-1,1,L-1,L,L+1,H-1,H,H+1};
   for(I t:ts)if(fits(t))b.run(pref,m,int64_t(t),p,n,positive,false,false,{{c,h}});
   for(int cg=0;cg<2;cg++)b.run(pref,m,0,p,n,positive,true,cg,{{c,h}});
  }
  // Every table address is read in both halves; multi-plane tails must follow the exact recurrence.
  for(int seed=0;seed<32;seed++)for(int positive=0;positive<2;positive++){
   vector<pair<int,int>>codes;for(int m=23;m>=0;m--)codes.push_back({(seed+m*3)%32,(seed*7+m*11)%32});
   int64_t pref=(seed&1)?-7:11;for(int64_t tau:{MIN48,MAX48,INT64_C(-1),INT64_C(0),INT64_C(1)})b.run(pref,23,tau,p,n,positive,false,false,codes);
  }
 }
 // Large registered prefixes near both signed48 boundaries; m0 and m1 with legal original sums.
 b.table(tables[2]);for(int m=0;m<2;m++)for(int pos=0;pos<2;pos++)for(int sign:{-1,1}){
  int64_t pref=sign*((INT64_C(1)<<44)-1);for(int64_t t:{MIN48,MAX48,INT64_C(0)})b.run(pref,m,t,0,0,pos,false,false,{{0,0}});
 }
 ofstream f("FUNCTIONAL.json");f<<"{\"status\":\"PASS\",\"contexts\":"<<b.groups<<",\"decision_steps_each_arm\":"<<b.steps<<",\"L_or_H_equal_tau_steps\":"<<b.ties<<",\"tau_endpoint_contexts\":"<<b.endpoints<<",\"m_min\":0,\"m_max\":23,\"runtime_LUT_tables\":4,\"reference\":\"independent signed128 original L/H, legal signed48 original intermediates\"}\n";
 cout<<"PASS contexts="<<b.groups<<" decision_steps_each_arm="<<b.steps<<" ties="<<b.ties<<" tau_endpoints="<<b.endpoints<<endl;
 }catch(exception&e){cerr<<"FAIL "<<e.what()<<endl;return 1;}return 0;}

