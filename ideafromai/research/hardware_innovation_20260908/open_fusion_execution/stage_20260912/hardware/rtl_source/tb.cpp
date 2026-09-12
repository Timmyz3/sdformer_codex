#include "Vtemporal_source.h"
#include "verilated.h"
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Vec=std::array<int64_t,8>;
struct Instruction {
    int kind,dst,a,b,sa,sb,na,nb,rne,t;
    int64_t threshold;
    int negative,constant;
};
struct Writeback {int dst; Vec value;};
static void require(bool yes,const std::string &why) {if(!yes)throw std::runtime_error(why);}
template<class T> std::vector<T> binary(const std::string &p) {
    std::ifstream f(p,std::ios::binary|std::ios::ate);require(bool(f),"open "+p);
    auto bytes=f.tellg();require(size_t(bytes)%sizeof(T)==0,"file alignment");
    std::vector<T> v(size_t(bytes)/sizeof(T));f.seekg(0);f.read((char*)v.data(),bytes);return v;
}
static int64_t exact48(__int128 x) {
    require(x>=-(__int128(1)<<47)&&x<(__int128(1)<<47),"reference signed48 overflow");
    return int64_t(x);
}
static int64_t norm(int64_t x,int s) {
    int64_t d=int64_t(1)<<s,q=x/d,r=x%d;
    if(r<0){--q;r+=d;}
    if(s && (r>d/2 || (r==d/2 && (q&1)))) ++q;
    return q< -8388608 ? -8388608 : q>8388607 ? 8388607 : q;
}
static std::vector<Writeback> reference(const std::vector<Instruction>& p,const int32_t *x,
                                      std::array<uint16_t,8>& gates) {
    std::array<Vec,96> rf{};std::vector<Writeback> wb;gates.fill(0);
    for(auto i:p){
        if(i.kind==0||i.kind==5)continue;
        Vec value{};
        for(int l=0;l<8;++l){
            if(i.kind==1)value[l]=x[i.t*8+l];
            else {
                auto a=exact48(__int128(rf[i.a][l])*(int64_t(1)<<i.sa)*(i.na?-1:1));
                if(i.kind==2){auto b=exact48(__int128(rf[i.b][l])*(int64_t(1)<<i.sb)*(i.nb?-1:1));value[l]=exact48(__int128(a)+b);}
                else if(i.kind==3)value[l]=norm(a,i.rne);
                else if(i.kind==4)value[l]=i.constant ? i.constant==2 : i.negative ? a<=i.threshold : a>=i.threshold;
                else throw std::runtime_error("unknown instruction");
            }
        }
        rf[i.dst]=value;wb.push_back({i.dst,value});
        if(i.kind==4)for(int l=0;l<8;++l)gates[l]|=uint16_t(value[l])<<i.t;
    }
    return wb;
}
static __uint128_t encode(Instruction i){
    __uint128_t r=0;
    auto set=[&](uint64_t v,int off,int bits){r|=(__uint128_t(v)&((__uint128_t(1)<<bits)-1))<<off;};
    set(i.kind,0,3);set(i.dst,3,7);set(i.a,10,7);set(i.b,17,7);
    set(i.sa,24,6);set(i.sb,30,6);set(i.na,36,1);set(i.nb,37,1);
    set(i.rne,38,6);set(i.t,44,4);set(i.threshold,48,48);set(i.negative,96,1);set(i.constant,97,2);
    return r;
}
static int64_t lane48(const WData *w,int lane){
    int off=lane*48,word=off/32,shift=off%32;
    uint64_t v=uint64_t(w[word]) | (uint64_t(w[word+1])<<32);
    v=(v>>shift)&((uint64_t(1)<<48)-1);
    return (v&(uint64_t(1)<<47)) ? int64_t(v|0xffff000000000000ULL) : int64_t(v);
}

int main(int argc,char**argv){
 try {
    Verilated::commandArgs(argc,argv);
    require(argc>=5,"program.txt input_i24.bin gold_u16.bin mode[ready|stress|observed] [waits]");
    std::vector<Instruction> program;std::ifstream pf(argv[1]);std::string line;
    while(std::getline(pf,line)){Instruction i{};std::istringstream s(line);s>>i.kind>>i.dst>>i.a>>i.b>>i.sa>>i.sb>>i.na>>i.nb>>i.rne>>i.t>>i.threshold>>i.negative>>i.constant;require(bool(s),"program fields");program.push_back(i);}
    require(!program.empty()&&program.size()<=512,"program length");
    auto inputs=binary<int32_t>(argv[2]);auto gold=binary<uint16_t>(argv[3]);
    require(inputs.size()%80==0&&gold.size()==inputs.size()/10,"fixture shape");
    std::string mode=argv[4];std::vector<int> waits;
    if(mode=="observed"){require(argc==6,"observed waits required");std::ifstream wf(argv[5]);int n;while(wf>>n)waits.push_back(n);require(waits.size()==inputs.size()/40,"observed wait count");}
    Vtemporal_source d;d.clk=0;d.rst_n=0;d.cfg_we=0;d.start=0;d.rd_ready=0;d.rsp_valid=0;d.wr_ready=0;d.input_base=0;d.output_base=4096;d.input_stride=288;
    auto edge=[&](){d.clk=1;d.eval();d.clk=0;d.eval();};
    edge();edge();d.rst_n=1;edge();
    for(size_t pc=0;pc<program.size();++pc){auto v=encode(program[pc]);d.cfg_we=1;d.cfg_addr=pc;for(int k=0;k<4;++k)d.cfg_data[k]=uint32_t(v>>(32*k));edge();}
    d.cfg_we=0;
    uint64_t cycle=0,reads=0,writes=0,rd_stall=0,wr_stall=0,wb_count=0;
    int64_t due=-1;uint64_t response=0;size_t wait_index=0;int delay=-1;
    bool held=false;uint64_t held_data=0;uint32_t held_addr=0;
    for(size_t tile=0;tile<inputs.size()/80;++tile){
        std::array<uint16_t,8> refgate{};
        auto expected=reference(program,&inputs[tile*80],refgate);
        for(int l=0;l<8;++l)require(refgate[l]==gold[tile*8+l],"interpreter/captured gate mismatch");
        std::array<uint8_t,2880> memory{};
        for(int t=0;t<10;++t)for(int l=0;l<8;++l){uint32_t v=inputs[tile*80+t*8+l];for(int j=0;j<3;++j)memory[t*288+l*3+j]=v>>(8*j);}
        size_t wi=0;int stores=0;uint64_t begin=cycle;
        require(d.idle,"start when busy");d.start=1;edge();++cycle;d.start=0;
        while(true){
            require(cycle-begin<20000,"tile timeout");
            d.rsp_valid=(int64_t(cycle)==due);d.rsp_data=response;
            d.rd_ready=mode!="stress"||cycle%32<24;
            d.wr_ready=mode!="stress"||cycle%32<28;
            d.eval();
            if(mode=="observed"&&d.wr_valid){if(delay<0)delay=waits[wait_index];d.wr_ready=delay==0;d.eval();}
            if(d.wb_valid){
                require(wi<expected.size(),"extra RF writeback");auto &e=expected[wi++];
                require(int(d.wb_dst)==e.dst,"writeback destination");
                for(int l=0;l<8;++l)if(lane48(d.wb_data,l)!=e.value[l])throw std::runtime_error("WB value tile="+std::to_string(tile)+" step="+std::to_string(wi-1)+" lane="+std::to_string(l));
                ++wb_count;
            }
            if(d.rd_valid){
                if(!d.rd_ready)++rd_stall;
                else {
                    require(d.rd_addr+8<=memory.size(),"read address");
                    response=0;for(int j=0;j<8;++j)response|=uint64_t(memory[d.rd_addr+j])<<(8*j);
                    due=int64_t(cycle)+1+(mode=="stress"?int(cycle%3):0);++reads;
                }
            }
            if(held)require(d.wr_valid&&d.wr_addr==held_addr&&d.wr_data==held_data,"blocked write changed");
            held=d.wr_valid&&!d.wr_ready;
            if(held){held_addr=d.wr_addr;held_data=d.wr_data;++wr_stall;}
            if(d.wr_valid&&d.wr_ready){
                require(stores<2&&d.wr_addr==4096+stores*8,"gate write address/order");
                for(int l=0;l<4;++l)require(uint16_t(d.wr_data>>(l*16))==gold[tile*8+stores*4+l],"RTL/capture gate mismatch");
                ++stores;++writes;if(mode=="observed"){++wait_index;delay=-1;}
            }else if(mode=="observed"&&d.wr_valid&&delay>0)--delay;
            edge();++cycle;
            if(d.done){require(stores==2&&wi==expected.size(),"incomplete tile");break;}
        }
    }
    std::cout<<"{\"mode\":\""<<mode<<"\",\"tiles\":"<<inputs.size()/80<<",\"cycles\":"<<cycle
        <<",\"SR64_reads\":"<<reads<<",\"SW64_writes\":"<<writes<<",\"read_stall_cycles\":"<<rd_stall
        <<",\"write_stall_cycles\":"<<wr_stall<<",\"RF_vector_writebacks_checked\":"<<wb_count
        <<",\"input_values\":"<<inputs.size()<<",\"gate_bits_checked\":"<<gold.size()*10
        <<",\"differences\":0}"<<std::endl;
    return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}
}
