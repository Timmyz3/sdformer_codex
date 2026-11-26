#include <array>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
using U=uint64_t;
static void req(bool b,const char* why){if(!b)throw std::runtime_error(why);}
struct Machine {
 U now=0,records=0,dma_bytes=0,bursts=0; std::map<std::string,U> ops;
 std::array<U,2> reads{},writes{},peak{}; std::array<U,16> ios_bank_r{},ios_bank_w{};
 std::array<uint32_t,32> ios{}; std::array<uint32_t,48> rf{};
 std::ofstream tr,dma;
 Machine(const std::string& p){if(p!="-"){tr.open(p,std::ios::binary);dma.open(p+".dma.csv");dma<<"start,end_startup,external_offset,bytes\n";}}
 void op(const std::string& n,U ticks=1){ops[n]+=ticks;now+=ticks;}
 void event(int id,bool write,U addr,U words,U ticks){
  req(addr%4==0&&addr+4*words<=(id==0?128:192),"actual memory bounds");
  U st=now;now+=ticks;records++;peak[id]=std::max(peak[id],addr+4*words);(write?writes[id]:reads[id])+=4*words;
  if(id==0)for(U i=0;i<words;i++)(write?ios_bank_w:ios_bank_r)[(addr/4+i)%16]++;
  if(tr){uint32_t a=addr,n=words;uint8_t flags[4]={uint8_t(id),uint8_t(write),uint8_t(words==16),0};tr.write((char*)&st,8);tr.write((char*)&now,8);tr.write((char*)&a,4);tr.write((char*)&n,4);tr.write((char*)flags,4);}
 }
 uint32_t read(U word){event(1,false,word*4,1,1);return rf.at(word);}
 void write(U word,uint32_t v){event(1,true,word*4,1,1);rf.at(word)=v;}
 void burst(U addr,U bytes){U st=now;op("DMA_startup",32);bursts++;dma_bytes+=bytes;if(dma)dma<<st<<','<<now<<','<<addr<<','<<bytes<<'\n';}
 void report(const std::string& p,U feature_ready,int action,int depth,int classifiers){
  std::ofstream o(p);o<<"{\"feature_ready_ticks\":"<<feature_ready<<",\"action_ready_ticks\":"<<now<<",\"action\":"<<action<<",\"tree_depth_visited\":"<<depth<<",\"classifier_lanes\":"<<classifiers<<",\"DMA_bytes_including_config\":"<<dma_bytes<<",\"DMA_bursts_including_config\":"<<bursts<<",\"trace_records\":"<<records<<",\"ops\":{";bool comma=false;for(auto p:ops){if(comma)o<<',';comma=true;o<<'"'<<p.first<<"\":"<<p.second;}o<<"},\"memory\":[";
  for(int i=0;i<2;i++){if(i)o<<',';o<<"{\"id\":"<<i<<",\"high_water\":"<<peak[i]<<",\"read_bytes\":"<<reads[i]<<",\"write_bytes\":"<<writes[i]<<'}';}o<<"],\"ios_bank_reads\":[";for(int i=0;i<16;i++){if(i)o<<',';o<<ios_bank_r[i];}o<<"],\"ios_bank_writes\":[";for(int i=0;i<16;i++){if(i)o<<',';o<<ios_bank_w[i];}o<<"],\"model_not_full_accelerator\":true}\n";
 }
};
static std::ifstream source(const std::string& path){std::ifstream f(path,std::ios::binary);req(bool(f),"open input");char m[8];f.read(m,8);req(!std::memcmp(m,"\x93NUMPY",6)&&m[6]==1,"npy v1");uint16_t n;f.read((char*)&n,2);std::string h(n,' ');f.read(&h[0],n);req(h.find("'<f4'")!=std::string::npos&&h.find("(10, 480, 640)")!=std::string::npos&&h.find("False")!=std::string::npos,"source format");return f;}
int main(int argc,char**argv){try{
 req(argc==7,"input.npy config.u32 output.u32 result.json trace-or- classifier-lanes(16/1)");
 int lanes=std::stoi(argv[6]);req(lanes==16||lanes==1,"finite classifier choice");Machine m(argv[5]);auto input=source(argv[1]);
 std::array<uint32_t,16> config{};std::ifstream cf(argv[2],std::ios::binary);cf.read((char*)config.data(),64);req(cf.gcount()==64,"64B actual tree config");
 m.burst(12288000,64);m.event(0,true,0,16,1);std::copy(config.begin(),config.end(),m.ios.begin());m.event(0,false,0,16,1);
 for(int j=0;j<16;j++)m.write(32+j,m.ios[j]);
 for(int j=0;j<32;j++)m.write(j,0);
 const U beats=192000;std::array<uint32_t,16> latch{};U beat=0;
 for(U offset=0;offset<12288000;offset+=65536){U bytes=std::min<U>(65536,12288000-offset);m.burst(offset,bytes);
  for(U j=0;j<bytes/64;j++,beat++){
   U slot=(beat%2)*16;m.event(0,true,slot*4,16,1);input.read((char*)(m.ios.data()+slot),64);req(input.gcount()==64,"complete payload beat");
   m.event(0,false,slot*4,16,1);std::copy(m.ios.begin()+slot,m.ios.begin()+slot+16,latch.begin());
   uint16_t positive=0,negative=0;for(int lane=0;lane<16;lane++){uint32_t a=latch[lane];req((a&0x7f800000U)!=0x7f800000U,"finite input contract");if(a&0x7fffffffU){if(a>>31)negative|=uint16_t(1U<<lane);else positive|=uint16_t(1U<<lane);}}
   m.op("classify",16/lanes);U row=(beat/40)%480,cx=(beat%40)/10,cy=row/120,index=(cy*4+cx)*2;
   for(int p=0;p<2;p++){uint32_t before=m.read(index+p);uint32_t count=__builtin_popcount(p?negative:positive);m.op("popcount16");uint32_t after=before+count;m.op("integer_add");req(after<=192000,"18-bit counter bound");m.write(index+p,after);}
   m.op("position_advance");
  }
 }
 req(beat==beats,"exact frame beats");U ready=m.now;uint32_t node=m.read(44);int depth=0;
 while(!(node&0x80000000U)){req(node<3&&depth<2,"legal depth-two tree");U a=32+node*4;uint32_t feature=m.read(a),threshold=m.read(a+1),left=m.read(a+2),right=m.read(a+3);req(feature<32&&threshold<=192000,"legal count threshold");uint32_t value=m.read(feature);m.op("integer_compare");bool take=value<=threshold;m.op("branch");node=take?left:right;depth++;}
 int action=int(node&7);req(action<=6,"legal leaf");m.op("action_output");
 std::ofstream out(argv[3],std::ios::binary);out.write((char*)m.rf.data(),128);std::ofstream backing(std::string(argv[3])+".rf",std::ios::binary);backing.write((char*)m.rf.data(),192);
 m.report(argv[4],ready,action,depth,lanes);return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
