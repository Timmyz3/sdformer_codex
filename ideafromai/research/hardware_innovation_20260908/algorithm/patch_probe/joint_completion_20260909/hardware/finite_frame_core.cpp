// Abstract issue-service reference, not an RTL clock model.
// Same 32 conditional FP add lanes / 8 FP FMA lanes for all routes.
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using U=uint64_t;
template<class T> std::vector<T> read(const std::string&p){
 std::ifstream f(p,std::ios::binary|std::ios::ate); if(!f)throw std::runtime_error(p);
 auto n=f.tellg(); std::vector<T>v(size_t(n)/sizeof(T));f.seekg(0);f.read((char*)v.data(),n);return v;
}
struct Phase {U ticks=0,vec=0,w=0,reads=0;};
unsigned time_mask(U s,U need){U v=s&need;return unsigned((v|(v>>10)|(v>>20)|(v>>30))&1023);}
unsigned packed_issue(U v,int packing){
 if(!packing)return __builtin_popcount(time_mask(v,(U(1)<<40)-1));
 unsigned biggest=0;
 static constexpr unsigned tm[4]={273,546,68,136};
 for(int bank=0;bank<4;++bank){unsigned bits=0;
  if(packing==1)bits=unsigned((v>>(bank*10))&1023);
  else bits=unsigned((v&tm[bank])|((v>>10)&tm[(bank+3)&3])|((v>>20)&tm[(bank+2)&3])|((v>>30)&tm[(bank+1)&3]));
  biggest=std::max(biggest,unsigned(__builtin_popcount(bits)));
 }return biggest;
}
Phase scan(const U*src,const std::array<U,4>&need,bool time_major,int packing=0){
 Phase all; U header=0; for(int k=0;k<864;++k)header|=src[k];
 U any=need[0]|need[1]|need[2]|need[3]; if(!(any&header))return all;
 std::vector<int>rec;for(int d=0;d<288;++d)if(src[d*3]|src[d*3+1]|src[d*3+2])rec.push_back(d);
 // One 60-bit aligned c/kh source word feeds three kw, four H8 contexts.
 // Two payload read slots; at most one source read and one W256 return/step.
 // A current W register can be replaced after its final T use.
 const int passes=time_major?10:1;
 for(int pass=0;pass<passes;++pass){
  std::array<U,4>q=need;
  if(time_major){U tm=0;for(int p=0;p<4;++p)tm|=U(1)<<(10*p+pass);for(auto&v:q)v&=tm;}
  if(!((q[0]|q[1]|q[2]|q[3])&header))continue;
  U compute=0,source_port=0;std::array<U,2>released{0,0};int ix=0;
  for(int d:rec){
   U ready=std::max(source_port,released[ix%2])+1;source_port=ready;
   U n=0,w=0;
   for(int kw=0;kw<3;++kw)for(int h=0;h<4;++h){
    U v=src[d*3+kw]&q[h];n+=packed_issue(v,packing);w+=(v!=0);
   }
   // The first distinct W request completes one step after source decode.
   // Subsequent W requests overlap the last use of the preceding W.
   compute=n?std::max(compute,ready+1)+n:std::max(compute,ready);
   released[ix++%2]=compute;all.vec+=n;all.w+=w;all.reads++;
  }
  all.ticks+=compute;
 }
 return all;
}
struct PSN {U ticks=0,fma=0,cmp=0,param=0;};
PSN psn(const uint8_t*acc,const uint8_t*a,int h0,unsigned prefix,bool prediction,bool ordinary){
 PSN r;U fma_end=0,cmp_end=0;
 // Constant A plus same-h/time predictor parameters are resident. A single
 // 256-bit local parameter read supplies offset/radius to all four p.
 // Two parameter-vector registers overlap the next row's reads with FMA.
 for(int h=0;h<4;++h)for(int t=0;t<10;++t){
  int nn=0,pn=0;for(int s=0;s<10;++s)if(a[t*10+s]){nn++;pn+=bool(prefix&(1u<<s));}
  if(prediction)r.param+=2;
  for(int p=0;p<4;++p){
   bool failed=acc[((h0+h)*4+p)*10+t]!=255;
   if(!ordinary&&!prediction&&(!failed||nn==pn))continue;
   U n=(prediction?pn:nn)+1; // temporal bias / predicted offset is paid.
   U c=prediction?2:1;
   // One dependent U accumulator, one result register, separate comparator.
   U ready=std::max(fma_end+n,cmp_end);fma_end=ready;cmp_end=ready+c;
   r.fma+=n;r.cmp+=c;
  }
 }
 // Two initial parameter-vector reads; remaining reads fit the >=4p row work.
 r.ticks=std::max(fma_end,cmp_end)+(prediction?2:0);return r;
}
enum {C_PRE_K,C_PRE_T,C_TAIL_K,C_TAIL_T,F_PRE,F_TAIL,CONV_VEC,W_K,W_T,SRC_K,SRC_T,
      FMA,CMP,PARAM,BN_STEPS,BN_NONEMPTY,DIR_READS,PSN_TICKS,PSN_FMA,PSN_CMP,
      C2_TIME0,C_PRE_P=C2_TIME0+10,C_TAIL_P,CONV_VEC_P,C_PRE_R,C_TAIL_R,CONV_VEC_R,NFIELDS};
int main(int argc,char**argv){try{
 if(argc!=9)throw std::runtime_error("src need accept A prefix conditional mode output");
 auto src=read<U>(argv[1]);auto needs=read<U>(argv[2]);auto acc=read<uint8_t>(argv[3]);auto a=read<uint8_t>(argv[4]);
 unsigned prefix=std::stoul(argv[5]);bool conditional=std::stoi(argv[6]);bool c2=std::stoi(argv[7]);
 size_t G=src.size()/864;if(src.size()!=G*864||needs.size()!=G*12||acc.size()!=G*12*4*10||a.size()!=100)throw std::runtime_error("shape");
 std::vector<U>out(G*3*NFIELDS,0);U allmask=(U(1)<<40)-1,pm=0;for(int p=0;p<4;++p)pm|=U(prefix)<<(p*10);
 for(size_t g=0;g<G;++g){const U*s=&src[g*864];const uint8_t*ac=&acc[g*480];U header=0;for(int k=0;k<864;++k)header|=s[k];
  for(int stripe=0;stripe<3;++stripe){U*r=&out[(g*3+stripe)*NFIELDS];int h0=stripe*4;
   std::array<U,4>pre,tail;for(int h=0;h<4;++h){pre[h]=conditional?pm:allmask;tail[h]=conditional?(needs[g*12+h0+h]&~pm):0;}
   Phase pk=scan(s,pre,false),pt=scan(s,pre,true),tk=scan(s,tail,false),tt=scan(s,tail,true);
   r[C_PRE_K]=pk.ticks;r[C_PRE_T]=pt.ticks;r[C_TAIL_K]=tk.ticks;r[C_TAIL_T]=tt.ticks;
   Phase pp=scan(s,pre,false,1),tp=scan(s,tail,false,1),pr=scan(s,pre,false,2),tr=scan(s,tail,false,2);
   r[C_PRE_P]=pp.ticks;r[C_TAIL_P]=tp.ticks;r[CONV_VEC_P]=pp.vec+tp.vec;
   r[C_PRE_R]=pr.ticks;r[C_TAIL_R]=tr.ticks;r[CONV_VEC_R]=pr.vec+tr.vec;
   r[CONV_VEC]=pk.vec+tk.vec;r[W_K]=pk.w+tk.w;r[W_T]=pt.w+tt.w;r[SRC_K]=pk.reads+tk.reads;r[SRC_T]=pt.reads+tt.reads;
   r[DIR_READS]=6; // 288-bit live-c/kh map plus the complete40-bit P4 header.
   if(c2){for(int t=0;t<10;++t){std::array<U,4>only;U m=0;for(int p=0;p<4;++p)m|=U(1)<<(p*10+t);only.fill(m);std::array<U,864>st;for(int k=0;k<864;++k)st[k]=s[k]&m;r[C2_TIME0+t]=scan(st.data(),only,false).ticks;}continue;}
   U bpre=0,btail=0,bactive=0;for(int h=0;h<4;++h){bpre+=__builtin_popcountll(pre[h]);btail+=__builtin_popcountll(tail[h]);bactive+=__builtin_popcountll((pre[h]|tail[h])&header);}
   PSN pred=psn(ac,a.data(),h0,prefix,true,false),full=psn(ac,a.data(),h0,prefix,false,!conditional);
   r[F_PRE]=bpre+(conditional?pred.ticks:full.ticks);r[F_TAIL]=conditional?(btail+full.ticks+1):0;
   r[FMA]=bactive+full.fma+(conditional?pred.fma:0);r[CMP]=full.cmp+(conditional?pred.cmp:0);r[PARAM]=conditional?pred.param:0;
   r[BN_STEPS]=bpre+btail;r[BN_NONEMPTY]=bactive;r[PSN_TICKS]=full.ticks+(conditional?pred.ticks:0);
   r[PSN_FMA]=full.fma+(conditional?pred.fma:0);r[PSN_CMP]=r[CMP];
  }
 }
 std::ofstream f(argv[8],std::ios::binary);f.write((char*)out.data(),out.size()*sizeof(U));
 std::cout<<"groups "<<G<<" fields "<<NFIELDS<<"\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
