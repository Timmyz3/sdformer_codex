#include "default_stream.hpp"

constexpr int TWO_TREE=16384;
static void fold(Engine& e,unsigned& occupied) {
    for(int hg=0;hg<12;++hg){e.op(ADD,hg,hg,12+hg);e.op(ADD,24+hg,24+hg,36+hg);}
    int level=0;e.tick();
    while(occupied&(1u<<level)) {
        for(int type=0;type<2;++type)for(int hg=0;hg<12;++hg){
            int r=type*24+hg;
            e.state_load(84,TWO_TREE+level*768+type*384+hg*32);
            e.op(ADD,r,r,84);
        }
        occupied&=~(1u<<level);++level;e.tick();
    }
    assert(level<10);
    for(int type=0;type<2;++type)for(int hg=0;hg<12;++hg)
        e.state_store(type*24+hg,TWO_TREE+level*768+type*384+hg*32);
    occupied|=1u<<level;e.tick();
}

static void finish(Engine& e,unsigned occupied) {
    for(int type=0;type<2;++type)for(int hg=0;hg<12;++hg)e.op(ZERO,type*24+hg);
    for(int level=0;level<10;++level) {
        e.tick();
        if(occupied&(1u<<level))for(int type=0;type<2;++type)for(int hg=0;hg<12;++hg){
            int r=type*24+hg;
            e.state_load(84,TWO_TREE+level*768+type*384+hg*32);
            e.op(ADD,r,r,84);
        }
    }
}

int main(int argc,char** argv) {
    // Dense, coefficients, tags, live, output, stats, stress, tagged.
    assert(argc==9);
    auto dense=read(argv[1]),coef=read(argv[2]),live=read(argv[4]);
    auto tags=read_tags(argv[3]);bool tagged=std::stoi(argv[8]);
    const size_t n=192000;assert(dense.size()==n*96);
    Engine e(std::stoi(argv[7]));CodeStream stream(e,tags);
    e.coefficients(coef);e.constant(72,768);e.constant(73,800);e.constant(74,832);
    if(tagged)e.op(ZERO,95);
    uint64_t payload_bytes=0;std::array<uint64_t,2> zeros{};
    e.phase=1;e.clear_sums();unsigned occupied=0;size_t cursor=0;
    for(size_t p=0;p<n;++p) {
        bool zero=tagged&&stream.zero(p);zeros[0]+=zero;
        if(!zero)for(int hg=0;hg<12;++hg) {
            e.input((tagged?live.data()+cursor*96:dense.data()+p*96)+hg*8);payload_bytes+=32;
            int r=(p%2)*12+hg;e.op(ADD,r,r,80);e.op(FMA_SQUARE,24+r,80);
        }
        cursor+=!zero;
        if((p+1)%256==0){fold(e,occupied);if(p+1<n)e.clear_sums();}
    }
    if(tagged)assert(cursor*96==live.size());
    finish(e,occupied);
    e.phase=2;
    std::array<float,96> mean{},variance{},rsqrt{};
    for(int hg=0;hg<12;++hg) {
        e.op(MUL,hg,hg,72);e.wait(hg);
        for(int l=0;l<8;++l)mean[hg*8+l]=e.rf[hg][l];
        e.op(MUL,24+hg,24+hg,72);
        e.op(SQUARE,81,hg);e.op(SUB,24+hg,24+hg,81);e.wait(24+hg);
        for(int l=0;l<8;++l){variance[hg*8+l]=e.rf[24+hg][l];assert(variance[hg*8+l]>=0);}
        e.op(ADD,24+hg,24+hg,74);e.op(MUL,82,24+hg,73);e.op(SEED,83,24+hg);
        for(int it=0;it<3;++it){e.op(SQUARE,84,83);e.op(NEWTON_RESIDUAL,85,82,84);e.op(MUL,83,83,85);}
        e.wait(83);for(int l=0;l<8;++l)rsqrt[hg*8+l]=e.rf[83][l];
        e.constant(86,hg*32);e.constant(87,384+hg*32);
        e.op(MUL,12+hg,83,86);e.op(MUL,84,hg,12+hg);e.op(SUB,24+hg,87,84);
        if(tagged){e.op(MUL,81,95,12+hg);e.op(ADD,48+hg,81,24+hg);}
    }
    e.phase=4;cursor=0;std::ofstream output(argv[5],std::ios::binary);assert(output);
    for(size_t p=0;p<n;++p) {
        bool zero=tagged&&stream.zero(p);zeros[1]+=zero;
        for(int hg=0;hg<12;++hg) {
            if(zero){e.output(48+hg,output);continue;}
            e.input((tagged?live.data()+cursor*96:dense.data()+p*96)+hg*8);payload_bytes+=32;
            e.op(MUL,81,80,12+hg);e.op(ADD,82,81,24+hg);e.output(82,output);
        }
        cursor+=!zero;
    }
    if(tagged)assert(cursor*96==live.size());
    std::ofstream stats(argv[6],std::ios::binary);
    stats.write(reinterpret_cast<const char*>(mean.data()),384);
    stats.write(reinterpret_cast<const char*>(variance.data()),384);
    stats.write(reinterpret_cast<const char*>(rsqrt.data()),384);
    std::cout<<"{\"service_slots\":"<<e.t<<",\"stages\":[";
    for(int i=0;i<5;++i)std::cout<<(i?",":"")<<e.stages[i];
    std::cout<<"],\"SR64_reads\":"<<e.reads<<",\"SW64_writes\":"<<e.writes
        <<",\"CR256_reads\":"<<e.cr<<",\"CW256_writes\":"<<e.cw
        <<",\"ALU_vector_issues\":"<<e.alu<<",\"DMA_slots\":"<<e.dma
        <<",\"port_or_writeback_waits\":"<<e.waits<<",\"RF_writebacks\":"<<e.wb_count
        <<",\"payload_bytes\":"<<payload_bytes<<",\"tag_bytes\":"<<stream.tag_bytes
        <<",\"tag_decode_slots\":"<<stream.decode_slots<<",\"tag_unpack\":"<<stream.tag_unpack
        <<",\"zero_positions\":["<<zeros[0]<<","<<zeros[1]<<"]}\n";
}
