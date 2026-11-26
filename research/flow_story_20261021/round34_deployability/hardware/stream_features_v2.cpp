#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
using U=uint64_t;
static void require(bool b,const char* s){if(!b)throw std::runtime_error(s);}
static U divup(U a,U b){return (a+b-1)/b;}
// Records are explicitly serialized, not compiler-layout dependent:
// start:u64,end:u64,byte_address:u32,words:u32,mem:u8,write:u8,vector:u8,reserved:u8.
struct Machine {
 U now=0,hash=1469598103934665603ULL,dma_bytes=0,dma_transactions=0,dma_ticks=0;
 std::map<std::string,U> ops;std::array<U,3> reads{},writes{},peak{},memory_ticks{};
 std::array<std::array<U,16>,3> bank_read{},bank_write{};
 std::ofstream trace,dma_trace;U records=0;
 const std::array<U,3> capacity{{524288,4194304,2048}};
 Machine(const std::string& path){if(path!="-"){trace.open(path,std::ios::binary);dma_trace.open(path+".dma.csv");dma_trace<<"start,end,external_byte_address,bytes\n";}}
 void charge(const std::string& op,U n=1){ops[op]+=n;now+=n*((op=="add"||op=="sub"||op=="mul")?3:(op=="div"||op=="sqrt")?16:1);}
 float add(float a,float b){charge("add");volatile float c=a+b;return c;}
 float sub(float a,float b){charge("sub");volatile float c=a-b;return c;}
 float mul(float a,float b){charge("mul");volatile float c=a*b;return c;}
 float divide(float a,float b){charge("div");volatile float c=a/b;return c;}
 float square_root(float a){charge("sqrt");volatile float c=std::sqrt(a);return c;}
 float absolute(float a){charge("abs");return std::abs(a);}
 void mem(int id,bool write,U addr,U n,bool vec=false){
  require(id>=0&&id<3&&addr%4==0&&addr+n*4<=capacity[id],"memory bounds/alignment");
  peak[id]=std::max(peak[id],addr+n*4);U start=now;U ticks=vec?divup(n,16):n;
  now+=ticks;memory_ticks[id]+=ticks;(write?writes[id]:reads[id])+=n*4;
  for(U k=0;k<n;k++){U a=addr+k*4;(write?bank_write[id]:bank_read[id])[(a/4)%16]++;hash=(hash^U(id)^U(write?0x100:0)^a)*1099511628211ULL;}
  if(trace){uint32_t a=uint32_t(addr),nn=uint32_t(n);uint8_t flags[4]={uint8_t(id),uint8_t(write),uint8_t(vec),0};trace.write((char*)&start,8);trace.write((char*)&now,8);trace.write((char*)&a,4);trace.write((char*)&nn,4);trace.write((char*)flags,4);}records++;
 }
 void dma(U external,U bytes){
  require(external%64==0&&bytes%64==0,"aligned DMA");U start=now;now+=32+bytes/64;dma_bytes+=bytes;dma_transactions++;dma_ticks+=now-start;
  if(dma_trace)dma_trace<<start<<','<<now<<','<<external<<','<<bytes<<'\n';
  // DMA ingress writes IOS 16 banks at one beat/tick; accounted within DMA.
  U before=now;now=start+32;mem(0,true,0,bytes/4,true);require(now==before,"DMA ingress schedule");memory_ticks[0]-=bytes/64;
  // Address binding includes every true external word; distinct from on-chip hash.
  hash=(hash^external^bytes^0xD0)*1099511628211ULL;
 }
 void report(const std::string& path,U resident_end,bool directional,const std::string& layout){
  std::ofstream o(path);o<<"{\"schema\":\"r34_full_stream_service_v1\",\"layout\":\""<<layout<<"\",\"directional\":"<<(directional?"true":"false")<<",\"feature_ready_tick\":"<<now<<",\"DMA_bytes\":"<<dma_bytes<<",\"DMA_transactions\":"<<dma_transactions<<",\"DMA_ticks\":"<<dma_ticks<<",\"address_fnv64\":\""<<hash<<"\",\"trace_records\":"<<records<<",\"resident_allocation_bytes\":"<<resident_end<<",\"model_arithmetic_not_RTL\":true,\"ops\":{";
  bool comma=false;for(auto p:ops){if(comma)o<<',';comma=true;o<<'"'<<p.first<<"\":"<<p.second;}o<<"},\"memory\":[";
  for(int j=0;j<3;j++){if(j)o<<',';o<<"{\"id\":"<<j<<",\"capacity\":"<<capacity[j]<<",\"high_water_address\":"<<peak[j]<<",\"read_bytes\":"<<reads[j]<<",\"write_bytes\":"<<writes[j]<<",\"serial_service_ticks\":"<<memory_ticks[j]<<",\"bank_reads\":[";for(int k=0;k<16;k++){if(k)o<<',';o<<bank_read[j][k];}o<<"],\"bank_writes\":[";for(int k=0;k<16;k++){if(k)o<<',';o<<bank_write[j][k];}o<<"]}";}o<<"]}\n";
 }
};

// NumPy-style pairwise C-contiguous float32 reduction without storing the input.
// At most eight lanes plus a logarithmic tree stack per active reducer.
struct Reducer {
 struct Frame {int n;float left;int stage;};
 Machine& m;std::vector<Frame> stack;int pos=0,len=0;std::array<float,8> lane{};float tail=-0.0f,result=0;bool done=false;
 void descend(int n){while(n>128){stack.push_back({n,0,0});int a=n/2;a-=a%8;n=a;}require(stack.size()<=11,"bounded reducer stack");len=n;}
 Reducer(Machine& mm,int n):m(mm){stack.reserve(11);descend(n);}
 void complete(float v){while(!stack.empty()){Frame& f=stack.back();if(f.stage==0){f.left=v;f.stage=1;int a=f.n/2;a-=a%8;int right=f.n-a;descend(right);return;}v=m.add(f.left,v);stack.pop_back();}result=v;done=true;}
 void push(float x){require(!done,"reducer excess input");
  if(len<8){tail=m.add(tail,x);}else if(pos<8)lane[pos]=x;else if(pos<len-len%8)lane[pos%8]=m.add(lane[pos%8],x);else{if(pos==len-len%8)tail=collapse();tail=m.add(tail,x);}
  pos++;if(pos==len){float v=len<8?tail:(len%8?tail:collapse());pos=0;tail=-0.0f;complete(v);}
 }
 float collapse(){return m.add(m.add(m.add(lane[0],lane[1]),m.add(lane[2],lane[3])),m.add(m.add(lane[4],lane[5]),m.add(lane[6],lane[7])));}
 float finish(){require(done&&pos==0,"reducer missing input");return result;}
};

static std::vector<float> read_npy(const std::string& path){std::ifstream f(path,std::ios::binary);require(bool(f),"open input");char magic[8];f.read(magic,8);require(std::memcmp(magic,"\x93NUMPY",6)==0&&magic[6]==1,"npy v1 expected");uint16_t n;f.read((char*)&n,2);std::string h(n,' ');f.read(&h[0],n);require(h.find("'<f4'")!=std::string::npos&&h.find("(10, 480, 640)")!=std::string::npos&&h.find("False")!=std::string::npos,"FP32 T10 C-order source required");std::vector<float> z(10*480*640);f.read((char*)z.data(),z.size()*4);require(f.gcount()==std::streamsize(z.size()*4),"full input payload");for(float x:z)require(std::isfinite(x),"finite raw input");return z;}

int main(int argc,char**argv){try{
 require(argc==7,"input.npy output.f32 result.json trace.bin-or- layout(rowband/cell) directional(1/0)");
 auto input=read_npy(argv[1]);std::string layout=argv[5];require(layout=="rowband"||layout=="cell","layout");bool direction=std::string(argv[6])=="1";Machine m(argv[4]);
 std::vector<float> res(4194304/4),ios(524288/4);std::array<float,160> output{};
 const int H=120,W=160,N=H*W,T=10;U rawbytes=(layout=="rowband"?10*120*640:10*N)*4;U prevbase=rawbytes,sumbase=prevbase+N*4,resend=sumbase+N*4;
 require(resend<=4194304,"RES allocated capacity");
 const U RAW=0,PREV=640,SUM=1280,MID=1920,IT=3840,FEATURE=400000;
 // 160 features, 10 mass values and bounded reducer lanes/stack fit declared
 // 2KiB new scalar register bank; host tree metadata is control, not data RAM.
 // Maximum nine active reducers, each eight lanes and <=11 bounded
 // (n,left,stage) frames; 8-pixel work chunks and moments fit 2KiB.
 U memory_end=FEATURE+160*4;require(memory_end<=524288,"IOS lifetime bound");
 std::array<float,10> edges{};for(int t=0;t<10;t++)edges[t]=float(-1.0+2.0*t/9.0);
 auto gather=[&](int cy,int cx){
  if(layout=="rowband")for(int t=0;t<T;t++){
   U src=U(t*480+cy*H)*640,words=H*640,dst=U(t)*words;m.dma(src*4,words*4);
   std::copy(input.begin()+src,input.begin()+src+words,ios.begin());m.mem(0,false,0,words,true);m.mem(1,true,dst*4,words,true);std::copy(ios.begin(),ios.begin()+words,res.begin()+dst);
  }else for(int t=0;t<T;t++)for(int r=0;r<H;r++){
   U src=U(t*480+cy*H+r)*640+cx*W,dst=U(t*H+r)*W;m.dma(src*4,W*4);std::copy(input.begin()+src,input.begin()+src+W,ios.begin());m.mem(0,false,0,W,true);m.mem(1,true,dst*4,W,true);std::copy(ios.begin(),ios.begin()+W,res.begin()+dst);
  }
 };
 for(int cy=0;cy<4;cy++){
  if(layout=="rowband")gather(cy,0);
  for(int cx=0;cx<4;cx++){
   if(layout=="cell")gather(cy,cx);
   Reducer positive(m,T*N),negative(m,T*N),diff(m,(T-1)*N);std::array<float,10> masses{};U positives=0,negatives=0;
   Reducer it2(m,9*118*158),ixt(m,9*118*158),iyt(m,9*118*158),ix2(m,9*118*158),iy2(m,9*118*158);
   for(int t=0;t<T;t++){
    Reducer mass(m,N);
    for(int r=0;r<H;r++){
     U raw=((layout=="rowband")?U(t*H+r)*640+cx*W:U(t*H+r)*W)*4;
     for(int begin=0;begin<W;begin+=8){int count=std::min(8,W-begin);U idx=U(r*W+begin);
      m.mem(1,false,raw+begin*4,count);m.mem(0,true,RAW+begin*4,count);std::copy(res.begin()+raw/4+begin,res.begin()+raw/4+begin+count,ios.begin()+RAW/4+begin);
      if(t){m.mem(1,false,prevbase+idx*4,count);m.mem(0,true,PREV+begin*4,count);m.mem(1,false,sumbase+idx*4,count);m.mem(0,true,SUM+begin*4,count);std::copy(res.begin()+prevbase/4+idx,res.begin()+prevbase/4+idx+count,ios.begin()+PREV/4+begin);std::copy(res.begin()+sumbase/4+idx,res.begin()+sumbase/4+idx+count,ios.begin()+SUM/4+begin);}
      m.mem(0,false,RAW+begin*4,count);if(t){m.mem(0,false,PREV+begin*4,count);m.mem(0,false,SUM+begin*4,count);}
      for(int c=begin;c<begin+count;c++){
       float a=ios[RAW/4+c],p=t?ios[PREV/4+c]:0;positive.push(std::max(a,0.0f));negative.push(std::max(-a,0.0f));m.charge("compare",2);m.charge("sign",1);m.charge("integer_count_update",2);positives+=a>0;negatives+=a<0;mass.push(m.absolute(a));
       ios[SUM/4+c]=t?m.add(ios[SUM/4+c],a):a;
       if(t){float delta=m.sub(a,p);diff.push(m.absolute(delta));if(direction){ios[MID/4+(r%3)*W+c]=m.mul(m.add(a,p),0.5f);ios[IT/4+(r%2)*W+c]=delta;}}
      }
      m.mem(0,true,SUM+begin*4,count);m.mem(0,false,SUM+begin*4,count);m.mem(1,true,sumbase+idx*4,count);std::copy(ios.begin()+SUM/4+begin,ios.begin()+SUM/4+begin+count,res.begin()+sumbase/4+idx);
      m.mem(1,true,prevbase+idx*4,count);std::copy(ios.begin()+RAW/4+begin,ios.begin()+RAW/4+begin+count,res.begin()+prevbase/4+idx);
      if(t&&direction){m.mem(0,true,MID+(r%3)*W*4+begin*4,count);m.mem(0,true,IT+(r%2)*W*4+begin*4,count);}
     }
     if(t&&direction&&r>=2){int center=r-1;for(int begin=1;begin<W-1;begin+=8){int count=std::min(8,W-1-begin);
      // Four spatial neighbors and delayed temporal center: each actual scalar
      // read is explicitly paid. No multiport line-buffer assumption.
      m.mem(0,false,MID+(center%3)*W*4+(begin-1)*4,count);m.mem(0,false,MID+(center%3)*W*4+(begin+1)*4,count);m.mem(0,false,MID+((r-2)%3)*W*4+begin*4,count);m.mem(0,false,MID+(r%3)*W*4+begin*4,count);m.mem(0,false,IT+(center%2)*W*4+begin*4,count);
      for(int c=begin;c<begin+count;c++){
       float ix=m.mul(m.sub(ios[MID/4+(center%3)*W+c+1],ios[MID/4+(center%3)*W+c-1]),0.5f);
       float iy=m.mul(m.sub(ios[MID/4+(r%3)*W+c],ios[MID/4+((r-2)%3)*W+c]),0.5f),it=ios[IT/4+(center%2)*W+c];
       it2.push(m.mul(it,it));ixt.push(m.mul(ix,it));iyt.push(m.mul(iy,it));ix2.push(m.mul(ix,ix));iy2.push(m.mul(iy,iy));
      }
     }}
    }masses[t]=mass.finish();
   }
   Reducer temporal(m,N);for(int i=0;i<N;i+=16){int count=std::min(16,N-i);m.mem(1,false,sumbase+i*4,count);for(int j=0;j<count;j++)temporal.push(m.absolute(res[sumbase/4+i+j]));}
   int id=cy*4+cx;float nn=float(T*N);m.charge("convert",2);output[id*8]=m.divide(positive.finish(),nn);output[id*8+1]=m.divide(negative.finish(),nn);output[id*8+2]=m.divide(float(positives),nn);output[id*8+3]=m.divide(float(negatives),nn);output[id*8+4]=m.divide(m.divide(temporal.finish(),float(N)),float(T));output[id*8+5]=m.divide(diff.finish(),float((T-1)*N));
   Reducer total(m,10),first(m,10),second(m,10);for(int t=0;t<T;t++){total.push(masses[t]);first.push(m.mul(edges[t],masses[t]));second.push(m.mul(m.mul(edges[t],edges[t]),masses[t]));}float mass=total.finish(),one=first.finish(),two=second.finish();m.charge("compare");output[id*8+6]=mass!=0?m.divide(one,mass):0;output[id*8+7]=mass!=0?m.divide(two,mass):0;
   if(direction){float tt=it2.finish();std::array<float,2> numer{{ixt.finish(),iyt.finish()}},energy{{ix2.finish(),iy2.finish()}};for(int a=0;a<2;a++){float den=m.square_root(m.mul(energy[a],tt));m.charge("compare");output[128+id*2+a]=den>float(1e-12)?m.divide(numer[a],den):0;}}
   m.mem(0,true,FEATURE+id*8*4,8);std::copy(output.begin()+id*8,output.begin()+id*8+8,ios.begin()+FEATURE/4+id*8);if(direction){m.mem(0,true,FEATURE+(128+id*2)*4,2);std::copy(output.begin()+128+id*2,output.begin()+128+id*2+2,ios.begin()+FEATURE/4+128+id*2);}
  }
 }
 for(int i=0;i<(direction?160:128);i++){require(output[i]==ios[FEATURE/4+i],"persistent IOS feature lifetime");output[i]=ios[FEATURE/4+i];require(std::isfinite(output[i]),"finite output");}
 std::ofstream handoff(std::string(argv[2])+".ios",std::ios::binary);handoff.write((char*)ios.data(),ios.size()*4);
 std::ofstream out(argv[2],std::ios::binary);out.write((char*)output.data(),(direction?160:128)*4);m.report(argv[3],resend,direction,layout);return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
