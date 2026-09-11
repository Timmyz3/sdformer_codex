// global_bn_engine.hpp is generated verbatim from the existing global_bn.cpp
// prefix (Engine plus file reader), stopping before its main. No old file edits.
#include "global_bn_engine.hpp"

constexpr int TAG_STATE=8192, DEFAULT_STRIPE_STATE=12288;
struct CodeStream {
    Engine& e;
    const std::vector<uint8_t>& external;
    uint64_t tag_bytes=0,decode_slots=0,block_test_slots=0,tag_unpack=0;
    CodeStream(Engine& engine,const std::vector<uint8_t>& tags):e(engine),external(tags){}
    bool zero(size_t p) {
        if(p%256==0) {
            e.transfer5();
            for(int j=0;j<4;++j)e.tick(-1,TAG_STATE+j*8,reinterpret_cast<const char*>(external.data()+p/8+j*8));
            V loaded=e.gather(TAG_STATE);
            std::array<uint8_t,32> bytes{};std::memcpy(bytes.data(),loaded.data(),32);tag_bytes+=32;
            // Four of the SAME96 RF vectors retain the256 tag bits as byte
            // codes. Packed-byte expansion and RF writes are charged LOADs.
            // No hidden persistent tag cache beside the stated RF/state.
            for(int j=0;j<4;++j) {
                for(int lane=0;lane<8;++lane)e.load[lane]=float(bytes[j*8+lane]);
                e.op(LOAD,88+j);++tag_unpack;
            }
        }
        int byte=(p%256)/8,r=88+byte/8;e.wait(r);
        e.tick();++decode_slots;
        return (int(e.rf[r][byte%8])>>(p%8))&1;
    }
    bool all_zero() {
        bool yes=true;
        for(int j=0;j<4;++j) {
            e.wait(88+j);
            for(int lane=0;lane<8;++lane)yes=bool(yes&&(int(e.rf[88+j][lane])==255));
            e.tick();++block_test_slots;
        }
        return yes;
    }
};

static std::vector<uint8_t> read_tags(const char* path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);assert(f);
    size_t n=f.tellg();f.seekg(0);std::vector<uint8_t>x(n);
    f.read(reinterpret_cast<char*>(x.data()),n);assert(f);return x;
}

int main(int argc,char**argv) {
    // dense input, coefficients, tags, packed live payload, output, stats,
    // stress, mode:0=dense,1=tag_same_arithmetic,2=default_ops,3=zero_block.
    assert(argc==9);
    auto dense=read(argv[1]),coefficients=read(argv[2]);
    auto tags=read_tags(argv[3]);auto packed=read(argv[4]);
    const int mode=std::stoi(argv[8]);assert(mode>=0&&mode<=3);
    bool tagged=mode>0;const size_t n=dense.size()/96;
    assert(n==192000&&tags.size()==n/8&&packed.size()%96==0);
    Engine e(std::stoi(argv[7])!=0);CodeStream stream(e,tags);
    std::array<uint64_t,3> zero_positions{};
    uint64_t active_bytes=0,default_prepare_slots=0,negative_mu_prepare_slots=0,block_replayed=0;
    e.coefficients(coefficients);e.constant(72,768);e.constant(73,800);e.constant(74,832);
    if(tagged)e.op(ZERO,95);
    e.phase=1;e.clear_sums();unsigned occupied=0;size_t cursor=0;
    for(size_t p=0;p<n;++p) {
        bool zero=tagged&&stream.zero(p);zero_positions[0]+=zero;
        for(int hg=0;hg<12;++hg) {
            if(zero&&mode>=2)continue;
            int source=95;
            if(!zero) {e.input((tagged?packed.data()+cursor*96:dense.data()+p*96)+hg*8);source=80;active_bytes+=32;}
            int r=(p%4)*12+hg;e.op(ADD,r,r,source);
        }
        cursor+=!zero;
        if((p+1)%256==0){e.fold_block(occupied);if(p+1<n)e.clear_sums();}
    }
    if(tagged)assert(cursor*96==packed.size());
    e.finish_tree(occupied);
    std::array<float,96> mean{},variance{},rsqrt{};
    for(int hg=0;hg<12;++hg) {
        e.reduce(hg,false);for(int l=0;l<8;++l)mean[hg*8+l]=e.rf[hg][l];
        e.op(COPY,60+hg,hg);e.wait(60+hg);
    }
    e.phase=2;e.clear_sums();occupied=0;cursor=0;
    if(mode>=2) {
        uint64_t begin=e.t;
        // Ordinary constant propagation belongs to BOTH strong controls.
        // RF48..59 is unused by the variance sums. These12 vectors are
        // overwritten by normalized defaults only after variance completes.
        for(int hg=0;hg<12;++hg)e.op(SUB,48+hg,95,60+hg);
        negative_mu_prepare_slots=e.t-begin;
    }
    if(mode==3) {
        uint64_t begin=e.t;
        // Reproduce one COMPLETE 64-term branch from +0 exactly. There is
        // no count multiplication, altered reduction tree or mixed-branch reuse.
        for(int hg=0;hg<12;++hg) {
            e.op(ZERO,84);
            for(int k=0;k<64;++k)e.op(FMA_SQUARE,84,48+hg);
            e.state_store(84,DEFAULT_STRIPE_STATE+hg*32);
        }
        default_prepare_slots=e.t-begin;
    }
    for(size_t p=0;p<n;++p) {
        bool zero=tagged&&stream.zero(p);
        if(mode==3&&p%256==0&&stream.all_zero()) {
            zero_positions[1]+=256;++block_replayed;
            for(int stripe=0;stripe<4;++stripe)for(int hg=0;hg<12;++hg)
                e.state_load(stripe*12+hg,DEFAULT_STRIPE_STATE+hg*32);
            e.fold_block(occupied);p+=255;
            if(p+1<n)e.clear_sums();
            e.tick(); // logical position jump / block completion control
            continue;
        }
        zero_positions[1]+=zero;
        for(int hg=0;hg<12;++hg) {
            if(zero&&mode>=2) {
                int r=(p%4)*12+hg;e.op(FMA_SQUARE,r,48+hg);continue;
            }
            int source=95;
            if(!zero){e.input((tagged?packed.data()+cursor*96:dense.data()+p*96)+hg*8);source=80;active_bytes+=32;}
            e.op(SUB,81,source,60+hg);int r=(p%4)*12+hg;e.op(FMA_SQUARE,r,81);
        }
        cursor+=!zero;
        if((p+1)%256==0){e.fold_block(occupied);if(p+1<n)e.clear_sums();}
    }
    if(tagged)assert(cursor*96==packed.size());
    e.finish_tree(occupied);e.phase=3;
    for(int hg=0;hg<12;++hg) {
        e.reduce(hg,false);for(int l=0;l<8;++l)variance[hg*8+l]=e.rf[hg][l];
        e.op(ADD,hg,hg,74);e.op(MUL,82,hg,73);e.op(SEED,83,hg);
        for(int it=0;it<3;++it){e.op(SQUARE,84,83);e.op(NEWTON_RESIDUAL,85,82,84);e.op(MUL,83,83,85);}
        e.wait(83);for(int l=0;l<8;++l)rsqrt[hg*8+l]=e.rf[83][l];
        e.constant(86,hg*32);e.constant(87,384+hg*32);
        e.op(MUL,12+hg,83,86);e.op(MUL,84,60+hg,12+hg);e.op(SUB,24+hg,87,84);
        if(mode>=2){e.op(MUL,81,95,12+hg);e.op(ADD,48+hg,81,24+hg);}
    }
    e.phase=4;std::ofstream output(argv[5],std::ios::binary);assert(output);cursor=0;
    for(size_t p=0;p<n;++p) {
        bool zero=tagged&&stream.zero(p);zero_positions[2]+=zero;
        for(int hg=0;hg<12;++hg) {
            if(zero&&mode>=2){e.output(48+hg,output);continue;}
            int source=95;
            if(!zero){e.input((tagged?packed.data()+cursor*96:dense.data()+p*96)+hg*8);source=80;active_bytes+=32;}
            e.op(MUL,81,source,12+hg);e.op(ADD,82,81,24+hg);e.output(82,output);
        }
        cursor+=!zero;
    }
    if(tagged)assert(cursor*96==packed.size());
    std::ofstream stats(argv[6],std::ios::binary);
    stats.write(reinterpret_cast<const char*>(mean.data()),384);
    stats.write(reinterpret_cast<const char*>(variance.data()),384);
    stats.write(reinterpret_cast<const char*>(rsqrt.data()),384);
    std::cout<<"{\"service_slots\":"<<e.t<<",\"stages\":[";
    for(int i=0;i<5;++i)std::cout<<(i?",":"")<<e.stages[i];
    std::cout<<"],\"SR64_reads\":"<<e.reads<<",\"SW64_writes\":"<<e.writes<<",\"CR256_reads\":"<<e.cr<<",\"CW256_writes\":"<<e.cw
        <<",\"ALU_vector_issues\":"<<e.alu<<",\"DMA_slots\":"<<e.dma<<",\"port_or_writeback_waits\":"<<e.waits
        <<",\"RF_writebacks\":"<<e.wb_count<<",\"active_payload_read_bytes\":"<<active_bytes
        <<",\"tag_read_bytes\":"<<stream.tag_bytes<<",\"tag_decode_slots\":"<<stream.decode_slots
        <<",\"tag_unpack_vector_issues\":"<<stream.tag_unpack
        <<",\"block_test_slots\":"<<stream.block_test_slots<<",\"default_prepare_slots\":"<<default_prepare_slots
        <<",\"negative_mu_prepare_slots\":"<<negative_mu_prepare_slots
        <<",\"zero_blocks_replayed\":"<<block_replayed<<",\"zero_positions_per_pass\":["
        <<zero_positions[0]<<","<<zero_positions[1]<<","<<zero_positions[2]<<"]}\n";
}
