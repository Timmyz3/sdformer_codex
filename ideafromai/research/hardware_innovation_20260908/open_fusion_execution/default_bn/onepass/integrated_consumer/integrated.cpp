// Engine/CodeStream and one-pass fold/finish are imported verbatim by run.py.
// Join is imported with its only address change: PED_STATE=24576.
#include "onepass_core.hpp"
#include "consumer_join.hpp"

using Profile=std::array<uint64_t,7>;
static void mark(Profile& profile,int stage,uint64_t begin,const Engine& e) {
    profile[stage]+=e.t-begin;
}

int main(int argc,char** argv) {
    // live, coefficients, source-proven tags, actual PED packed24,
    // BN intermediate, final output, computed stats, stress, mode1/2.
    assert(argc==10);int mode=std::stoi(argv[9]);assert(mode==1||mode==2);
    auto live=read(argv[1]),coef=read(argv[2]);auto tags=read_tags(argv[3]);
    auto ped=read_bytes(argv[4]);const size_t n=192000;
    assert(tags.size()==24000&&ped.size()==55296000&&live.size()%96==0);
    assert(PED_STATE>=TWO_TREE+10*768&&PED_STATE+288<=131072);
    Engine e(std::stoi(argv[8])!=0);CodeStream stream(e,tags);Join join(e,ped);
    Profile stages{};uint64_t begin=e.t;
    uint64_t moment_bytes=0,normalize_bytes=0,intermediate_read=0,intermediate_write=0;
    uint64_t final_adds=0;std::array<uint64_t,2> zero_positions{};
    e.coefficients(coef);e.constant(72,768);e.constant(73,800);e.constant(74,832);e.op(ZERO,95);
    mark(stages,0,begin,e);

    begin=e.t;e.phase=1;e.clear_sums();unsigned occupied=0;size_t cursor=0;
    for(size_t p=0;p<n;++p) {
        bool zero=stream.zero(p);zero_positions[0]+=zero;
        if(!zero)for(int hg=0;hg<12;++hg) {
            e.input(live.data()+cursor*96+hg*8);moment_bytes+=32;
            int r=(p%2)*12+hg;e.op(ADD,r,r,80);e.op(FMA_SQUARE,24+r,80);
        }
        cursor+=!zero;
        if((p+1)%256==0){fold(e,occupied);if(p+1<n)e.clear_sums();}
    }
    assert(cursor*96==live.size());finish(e,occupied);
    const uint64_t tree_last_access_finished=e.t;
    mark(stages,1,begin,e);

    begin=e.t;e.phase=2;std::array<float,96> mean{},variance{},rsqrt{};
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
        e.op(MUL,81,95,12+hg);e.op(ADD,48+hg,81,24+hg);
    }
    // Every eps read was issued before this point. Reload through the real
    // coefficient port; wait(74) is built into constant(), not a host alias.
    const uint64_t eps_last_use_finished=e.t;
    e.constant(74,864);e.wait(74);
    for(float value:e.rf[74])assert(value==0x1p-14f);
    const uint64_t f14_scale_ready=e.t;
    mark(stages,2,begin,e);

    std::ofstream intermediate;
    if(mode==1){intermediate.open(argv[5],std::ios::binary);assert(intermediate);}
    std::ofstream output(argv[6],std::ios::binary);assert(output);
    cursor=0;uint64_t first_PED_input=0;
    for(size_t p=0;p<n;++p) {
        begin=e.t;e.phase=3;bool zero=stream.zero(p);zero_positions[1]+=zero;
        if(!zero)for(int hg=0;hg<12;++hg) {
            e.input(live.data()+cursor*96+hg*8);normalize_bytes+=32;
            e.op(MUL,81,80,12+hg);e.op(ADD,hg,81,24+hg);
        }
        cursor+=!zero;mark(stages,3,begin,e);
        if(mode==1) {
            begin=e.t;e.phase=4;
            for(int hg=0;hg<12;++hg){e.output(zero?48+hg:hg,intermediate);intermediate_write+=32;}
            mark(stages,6,begin,e);continue;
        }
        begin=e.t;e.phase=4;if(!p)first_PED_input=e.t;join.input(p);mark(stages,4,begin,e);
        for(int hg=0;hg<12;++hg) {
            begin=e.t;join.convert(hg);mark(stages,4,begin,e);
            begin=e.t;e.op(ADD,84,zero?48+hg:hg,83);++final_adds;e.output(84,output);mark(stages,5,begin,e);
        }
    }
    assert(cursor*96==live.size());
    if(mode==1) {
        intermediate.close();auto normalized=read(argv[5]);assert(normalized.size()==n*96);
        for(size_t p=0;p<n;++p) {
            begin=e.t;e.phase=4;if(!p)first_PED_input=e.t;join.input(p);mark(stages,4,begin,e);
            for(int hg=0;hg<12;++hg) {
                begin=e.t;e.input(normalized.data()+p*96+hg*8);intermediate_read+=32;mark(stages,6,begin,e);
                begin=e.t;join.convert(hg);mark(stages,4,begin,e);
                begin=e.t;e.op(ADD,84,80,83);++final_adds;e.output(84,output);mark(stages,5,begin,e);
            }
        }
    }
    assert(first_PED_input>tree_last_access_finished&&first_PED_input>=f14_scale_ready);
    uint64_t total=0;for(auto v:stages)total+=v;assert(total==e.t);
    std::ofstream stats(argv[7],std::ios::binary);assert(stats);
    stats.write(reinterpret_cast<const char*>(mean.data()),384);
    stats.write(reinterpret_cast<const char*>(variance.data()),384);
    stats.write(reinterpret_cast<const char*>(rsqrt.data()),384);
    std::cout<<"{\"service_slots\":"<<e.t<<",\"stages\":[";
    for(int i=0;i<7;++i)std::cout<<(i?",":"")<<stages[i];
    std::cout<<"],\"SR64_reads\":"<<e.reads<<",\"SW64_writes\":"<<e.writes
        <<",\"CR256_reads\":"<<e.cr<<",\"CW256_writes\":"<<e.cw
        <<",\"ALU_vector_issues\":"<<e.alu<<",\"DMA_slots\":"<<e.dma
        <<",\"port_or_writeback_waits\":"<<e.waits<<",\"RF_writebacks\":"<<e.wb_count
        <<",\"moment_payload_bytes\":"<<moment_bytes<<",\"normalization_payload_bytes\":"<<normalize_bytes
        <<",\"PED_packed24_read_bytes\":"<<join.bytes_read<<",\"PED_address_slots\":"<<join.address_slots
        <<",\"PED_decode_slots\":"<<join.decode_slots<<",\"PED_conversion_vector_issues\":"<<join.conversion_issues
        <<",\"final_ADD_vector_issues\":"<<final_adds
        <<",\"intermediate_BN_read_bytes\":"<<intermediate_read<<",\"intermediate_BN_write_bytes\":"<<intermediate_write
        <<",\"final_write_bytes\":"<<n*384<<",\"tag_read_bytes\":"<<stream.tag_bytes
        <<",\"tag_decode_slots\":"<<stream.decode_slots<<",\"tag_unpack_vector_issues\":"<<stream.tag_unpack
        <<",\"zero_positions\":["<<zero_positions[0]<<","<<zero_positions[1]<<"]"
        <<",\"lifetime\":{\"tree_last_access_finished\":"<<tree_last_access_finished
        <<",\"eps_last_use_finished\":"<<eps_last_use_finished<<",\"f14_scale_ready\":"<<f14_scale_ready
        <<",\"first_PED_input\":"<<first_PED_input<<"}}\n";
}
