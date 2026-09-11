// Actual full-domain FP32 BN payload execution; no captured statistics input.
// External arrays are DRAM. On-chip state/coefficient arrays are128KiB each.
#include <array>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <cassert>
#include <cstdint>

using V=std::array<float,8>;
enum Kind { NONE,ZERO,LOAD,COPY,ADD,SUB,MUL,SQUARE,FMA_SQUARE,SEED,NEWTON_RESIDUAL };
struct Wb {bool valid=false; uint64_t due=0;int dst=0;V value{};};
struct Engine {
    uint64_t t=0,reads=0,writes=0,cr=0,cw=0,alu=0,dma=0,waits=0,wb_count=0;
    int phase=0;bool stress;std::array<uint64_t,5> stages{};
    std::array<V,96> rf{};std::array<uint64_t,96> ready{};
    std::array<Wb,8> pending{};
    std::array<char,131072> state{},coef{};
    std::array<char,8> response{},sword{};bool response_valid=false;
    V load{};
    explicit Engine(bool blocked):stress(blocked){}
    void tick(int read=-1,int write=-1,const char* payload=nullptr,Kind op=NONE,int dst=0,int a=-1,int b=-1) {
        int latency=op==SEED?2:4;
        while ((stress&&((read>=0&&t%32>=24)||(write>=0&&t%32>=28))) ||
               (op!=NONE&&pending[(t+latency)%8].valid)) {++waits;tick();}
        Wb& due=pending[t%8];
        if(due.valid&&due.due==t){rf[due.dst]=due.value;due.valid=false;++wb_count;}
        if(response_valid){sword=response;response_valid=false;}
        if(read>=0){assert(read%8==0&&read+8<=131072);std::memcpy(response.data(),state.data()+read,8);response_valid=true;++reads;}
        if(write>=0){assert(write%8==0&&write+8<=131072);std::memcpy(state.data()+write,payload,8);++writes;}
        if(op!=NONE){
            assert(ready[dst]<=t);if(a>=0)assert(ready[a]<=t);if(b>=0)assert(ready[b]<=t);
            V v{};
            for(int l=0;l<8;++l){
                float x=a<0?0:rf[a][l],y=b<0?0:rf[b][l];
                switch(op){
                case ZERO:v[l]=0;break;case LOAD:v[l]=load[l];break;case COPY:v[l]=x;break;
                case ADD:v[l]=x+y;break;case SUB:v[l]=x-y;break;case MUL:v[l]=x*y;break;
                case SQUARE:v[l]=x*x;break;
                case FMA_SQUARE:v[l]=std::fma(x,x,rf[dst][l]);break;
                case NEWTON_RESIDUAL:v[l]=std::fma(-x,y,1.5f);break;
                case SEED:{uint32_t bits;std::memcpy(&bits,&x,4);bits=0x5f3759dfu-(bits>>1);std::memcpy(&v[l],&bits,4);break;}
                default:assert(false);
                }
                assert(std::isfinite(v[l]));
            }
            Wb& next=pending[(t+latency)%8];next={true,t+uint64_t(latency),dst,v};
            ready[dst]=t+latency;++alu;
        }
        ++stages[phase];++t;
    }
    void wait(int r){while(ready[r]>=t){bool found=false;for(auto&w:pending)found|=w.valid&&w.dst==r;if(!found)break;tick();}}
    void op(Kind kind,int dst,int a=-1,int b=-1){wait(dst);if(a>=0)wait(a);if(b>=0)wait(b);tick(-1,-1,nullptr,kind,dst,a,b);}
    void transfer5(){for(int i=0;i<5;++i){tick();++dma;}}
    V gather(int address){
        V value{};char* out=reinterpret_cast<char*>(value.data());
        for(int i=0;i<4;++i){tick(address+i*8);if(i>0)std::memcpy(out+(i-1)*8,sword.data(),8);}
        tick();std::memcpy(out+24,sword.data(),8);return value;
    }
    void input(const float* external){
        transfer5();const char* in=reinterpret_cast<const char*>(external);
        for(int j=0;j<4;++j)tick(-1,j*8,in+j*8);
        load=gather(0);op(LOAD,80);wait(80);
    }
    void output(int r,std::ofstream& file){
        wait(r);const char* src=reinterpret_cast<const char*>(rf[r].data());
        for(int j=0;j<4;++j)tick(-1,32+j*8,src+j*8);
        V value=gather(32);transfer5();file.write(reinterpret_cast<const char*>(value.data()),32);
    }
    void coefficients(const std::vector<float>& values){
        assert(values.size()*4<=131072&&values.size()%8==0);
        for(size_t j=0;j<values.size();j+=8){transfer5();std::memcpy(coef.data()+j*4,values.data()+j,32);tick();++cw;}
    }
    void constant(int r,int address){
        assert(address%32==0);tick();++cr;std::memcpy(load.data(),coef.data()+address,32);tick();op(LOAD,r);wait(r);
    }
    void clear_sums(){for(int i=0;i<48;++i)op(ZERO,i);}
    void reduce(int hg,bool partial=true){if(partial)for(int stripe=1;stripe<4;++stripe)op(ADD,hg,hg,stripe*12+hg);op(MUL,hg,hg,72);wait(hg);}
    void state_load(int r,int address){load=gather(address);op(LOAD,r);wait(r);}
    void state_store(int r,int address){wait(r);const char* p=reinterpret_cast<const char*>(rf[r].data());for(int j=0;j<4;++j)tick(-1,address+j*8,p+j*8);}
    void fold_block(unsigned& occupied){
        for(int hg=0;hg<12;++hg)for(int stripe=1;stripe<4;++stripe)op(ADD,hg,hg,stripe*12+hg);
        int level=0;tick();
        while(occupied&(1u<<level)){
            for(int hg=0;hg<12;++hg){state_load(84,4096+level*384+hg*32);op(ADD,hg,hg,84);}
            occupied&=~(1u<<level);++level;tick();
        }
        assert(level<10);
        for(int hg=0;hg<12;++hg)state_store(hg,4096+level*384+hg*32);
        occupied|=1u<<level;tick();
    }
    void finish_tree(unsigned occupied){
        for(int hg=0;hg<12;++hg)op(ZERO,hg);
        for(int level=0;level<10;++level){tick();if(occupied&(1u<<level))for(int hg=0;hg<12;++hg){state_load(84,4096+level*384+hg*32);op(ADD,hg,hg,84);}}
    }
};

static std::vector<float> read(const char* path){std::ifstream f(path,std::ios::binary|std::ios::ate);assert(f);size_t n=f.tellg();assert(n%4==0);std::vector<float>x(n/4);f.seekg(0);f.read(reinterpret_cast<char*>(x.data()),n);assert(f);return x;}
int main(int argc,char**argv){
    assert(argc==7);bool pairwise=std::stoi(argv[6])!=0;auto x=read(argv[1]);auto coefficients=read(argv[2]);assert(x.size()%96==0);
    size_t n=x.size()/96;assert(n==192000);Engine e(std::stoi(argv[5])!=0);
    e.coefficients(coefficients);e.constant(72,768);e.constant(73,800);e.constant(74,832);
    e.phase=1;e.clear_sums();unsigned occupied=0;
    for(size_t p=0;p<n;++p){
        for(int hg=0;hg<12;++hg){e.input(x.data()+p*96+hg*8);int r=(p%4)*12+hg;e.op(ADD,r,r,80);}
        if(pairwise&&(p+1)%256==0){e.fold_block(occupied);if(p+1<n)e.clear_sums();}
    }
    if(pairwise)e.finish_tree(occupied);
    std::array<float,96> mean{},variance{},rsqrt{};
    for(int hg=0;hg<12;++hg){e.reduce(hg,!pairwise);for(int l=0;l<8;++l)mean[hg*8+l]=e.rf[hg][l];e.op(COPY,60+hg,hg);e.wait(60+hg);}
    e.phase=2;e.clear_sums();occupied=0;
    for(size_t p=0;p<n;++p){
        for(int hg=0;hg<12;++hg){e.input(x.data()+p*96+hg*8);e.op(SUB,81,80,60+hg);int r=(p%4)*12+hg;e.op(FMA_SQUARE,r,81);}
        if(pairwise&&(p+1)%256==0){e.fold_block(occupied);if(p+1<n)e.clear_sums();}
    }
    if(pairwise)e.finish_tree(occupied);
    e.phase=3;
    for(int hg=0;hg<12;++hg){
        e.reduce(hg,!pairwise);for(int l=0;l<8;++l)variance[hg*8+l]=e.rf[hg][l];
        e.op(ADD,hg,hg,74);e.op(MUL,82,hg,73);e.op(SEED,83,hg);
        for(int it=0;it<3;++it){e.op(SQUARE,84,83);e.op(NEWTON_RESIDUAL,85,82,84);e.op(MUL,83,83,85);}
        e.wait(83);for(int l=0;l<8;++l)rsqrt[hg*8+l]=e.rf[83][l];
        e.constant(86,hg*32);e.constant(87,384+hg*32);
        e.op(MUL,12+hg,83,86);e.op(MUL,84,60+hg,12+hg);e.op(SUB,24+hg,87,84);
    }
    e.phase=4;std::ofstream output(argv[3],std::ios::binary);assert(output);
    for(size_t p=0;p<n;++p)for(int hg=0;hg<12;++hg){e.input(x.data()+p*96+hg*8);e.op(MUL,81,80,12+hg);e.op(ADD,82,81,24+hg);e.output(82,output);}
    std::ofstream stats(argv[4],std::ios::binary);stats.write(reinterpret_cast<const char*>(mean.data()),384);stats.write(reinterpret_cast<const char*>(variance.data()),384);stats.write(reinterpret_cast<const char*>(rsqrt.data()),384);
    std::cout<<"{\"service_slots\":"<<e.t<<",\"stages\":[";
    for(int i=0;i<5;++i)std::cout<<(i?",":"")<<e.stages[i];
    std::cout<<"],\"SR64_reads\":"<<e.reads<<",\"SW64_writes\":"<<e.writes<<",\"CR256_reads\":"<<e.cr<<",\"CW256_writes\":"<<e.cw
             <<",\"ALU_vector_issues\":"<<e.alu<<",\"DMA_slots\":"<<e.dma<<",\"port_or_writeback_waits\":"<<e.waits<<",\"RF_writebacks\":"<<e.wb_count<<"}\n";
}
