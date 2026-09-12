#include "onepass_core.hpp"
#include "consumer_join.hpp"
#include "native.hpp"

using Profile=std::array<uint64_t,10>;
static void mark(Profile&s,int i,uint64_t begin,Engine&e){s[i]+=e.t-begin;}
static void save(const std::string&path,const std::vector<float>&v){std::ofstream f(path,std::ios::binary);assert(f);f.write(reinterpret_cast<const char*>(v.data()),v.size()*4);}

int main(int argc,char**argv) {
    // HWC packed T10-gate words, H/K TF32 weights, BN coefficients,
    // spatial/T10/C packed24 PED, work folder. One complete ordinary ready frame.
    assert(argc==6);const std::string dir=argv[5];
    auto gate=read_bytes(argv[1]),ped=read_bytes(argv[4]);auto weight=read(argv[2]),coef=read(argv[3]);
    assert(ped.size()==55296000&&coef.size()==224);
    Engine e(false);NativeCost native_cost;Profile common{};uint64_t begin=e.t;
    auto raw=native(e,gate,weight,native_cost);mark(common,0,begin,e);save(dir+"/native_raw.f32",raw);
    auto ref=native_reference(gate,weight);uint64_t mismatch=0;
    for(size_t i=0;i<raw.size();++i)mismatch+=std::memcmp(&raw[i],&ref[i],4)!=0;
    assert(mismatch==0);save(dir+"/native_reference.f32",ref);ref.clear();ref.shrink_to_fit();
    std::cerr<<"complete native reference bits equal\n";

    begin=e.t;e.phase=1;e.coefficients(coef);e.constant(72,768);e.constant(73,800);e.constant(74,832);
    mark(common,1,begin,e);begin=e.t;e.clear_sums();unsigned occupied=0;
    uint64_t moment_bytes=0;constexpr int n=192000;
    for(int p=0;p<n;++p) {
        for(int hg=0;hg<12;++hg){e.input(raw.data()+p*96+hg*8);moment_bytes+=32;
            int r=(p%2)*12+hg;e.op(ADD,r,r,80);e.op(FMA_SQUARE,24+r,80);}
        if((p+1)%256==0){fold(e,occupied);if(p+1<n)e.clear_sums();}
    }
    finish(e,occupied);mark(common,2,begin,e);begin=e.t;e.phase=2;
    std::vector<float> stats(480);
    for(int hg=0;hg<12;++hg) {
        e.op(MUL,hg,hg,72);e.wait(hg);for(int l=0;l<8;++l)stats[hg*8+l]=e.rf[hg][l];
        e.op(MUL,24+hg,24+hg,72);e.op(SQUARE,81,hg);e.op(SUB,24+hg,24+hg,81);e.wait(24+hg);
        for(int l=0;l<8;++l){stats[96+hg*8+l]=e.rf[24+hg][l];assert(stats[96+hg*8+l]>=0);}
        e.op(ADD,24+hg,24+hg,74);e.op(MUL,82,24+hg,73);e.op(SEED,83,24+hg);
        for(int it=0;it<3;++it){e.op(SQUARE,84,83);e.op(NEWTON_RESIDUAL,85,82,84);e.op(MUL,83,83,85);}
        e.wait(83);for(int l=0;l<8;++l)stats[192+hg*8+l]=e.rf[83][l];
        e.constant(86,hg*32);e.constant(87,384+hg*32);
        e.op(MUL,12+hg,83,86);e.op(MUL,84,hg,12+hg);e.op(SUB,24+hg,87,84);
        e.wait(12+hg);e.wait(24+hg);
        for(int l=0;l<8;++l){stats[288+hg*8+l]=e.rf[12+hg][l];stats[384+hg*8+l]=e.rf[24+hg][l];}
    }
    e.constant(74,864);e.wait(74);save(dir+"/statistics.f32",stats);mark(common,3,begin,e);
    const Engine prefix=e;const uint64_t prefix_end=e.t;
    std::cerr<<"sameEngine native+computed statistics prefix "<<prefix_end<<" slots\n";
    std::cout<<"{\"complete\":true,\"native_reference_bit_differences\":"<<mismatch
        <<",\"native_reference_values\":"<<raw.size()<<",\"prefix_slots\":"<<prefix_end
        <<",\"native\":{\"gate_read_bytes\":"<<native_cost.gate_bytes<<",\"weight_fill_bytes\":"<<native_cost.weight_bytes
        <<",\"raw_spill_bytes\":"<<native_cost.raw_bytes<<",\"directory_entries\":"<<native_cost.directory_entries
        <<",\"directory_decode_slots\":"<<native_cost.directory_decode<<",\"active_ADD_vector_issues\":"<<native_cost.active_adds
        <<",\"gate_address_slots\":"<<native_cost.gate_address<<",\"spill_address_slots\":"<<native_cost.spill_address
        <<",\"phase_slots\":["<<native_cost.coefficient_slots<<","<<native_cost.input_slots<<","<<native_cost.directory_slots
        <<","<<native_cost.compute_slots<<","<<native_cost.output_slots<<"]},\"arms\":[";
    for(int mode=1;mode<=2;++mode) {
        // Complete actual shared Machine state is copied, including RF,
        // pending writes, memory contents, cycle, ports and counters.
        e=prefix;Profile phases=common;Join join(e,ped);uint64_t normalized_read=0,intermediate_write=0,intermediate_read=0;
        std::string name=mode==1?"materialized":"fused";
        std::ofstream intermediate,output(dir+"/"+name+"_final.f32",std::ios::binary);assert(output);
        if(mode==1){intermediate.open(dir+"/materialized_BN.f32",std::ios::binary);assert(intermediate);}
        for(int p=0;p<n;++p) {
            begin=e.t;e.phase=3;
            for(int hg=0;hg<12;++hg){e.input(raw.data()+p*96+hg*8);normalized_read+=32;
                e.op(MUL,81,80,12+hg);e.op(ADD,hg,81,24+hg);}
            mark(phases,4,begin,e);
            if(mode==1){begin=e.t;for(int hg=0;hg<12;++hg){e.output(hg,intermediate);intermediate_write+=32;}mark(phases,8,begin,e);continue;}
            begin=e.t;e.phase=4;join.input(p);mark(phases,5,begin,e);
            for(int hg=0;hg<12;++hg){begin=e.t;join.convert(hg);mark(phases,6,begin,e);
                begin=e.t;e.op(ADD,84,hg,83);e.output(84,output);mark(phases,7,begin,e);}
        }
        if(mode==1){intermediate.close();auto normalized=read((dir+"/materialized_BN.f32").c_str());assert(normalized.size()==raw.size());
            for(int p=0;p<n;++p){begin=e.t;e.phase=4;join.input(p);mark(phases,5,begin,e);
                for(int hg=0;hg<12;++hg){begin=e.t;e.input(normalized.data()+p*96+hg*8);intermediate_read+=32;mark(phases,9,begin,e);
                    begin=e.t;join.convert(hg);mark(phases,6,begin,e);
                    begin=e.t;e.op(ADD,84,80,83);e.output(84,output);mark(phases,7,begin,e);}}
        }
        uint64_t sum=0;for(auto v:phases)sum+=v;assert(sum==e.t);
        std::cout<<(mode==2?",":"")<<"{\"mode\":\""<<name<<"\",\"service_slots\":"<<e.t<<",\"phase_slots\":[";
        for(int i=0;i<10;++i)std::cout<<(i?",":"")<<phases[i];
        std::cout<<"],\"SR64_reads\":"<<e.reads<<",\"SW64_writes\":"<<e.writes<<",\"CR256_reads\":"<<e.cr<<",\"CW256_writes\":"<<e.cw
            <<",\"ALU_vector_issues\":"<<e.alu<<",\"DMA_slots\":"<<e.dma<<",\"port_or_writeback_waits\":"<<e.waits
            <<",\"moment_read_bytes\":"<<moment_bytes<<",\"normalize_read_bytes\":"<<normalized_read
            <<",\"PED_read_bytes\":"<<join.bytes_read<<",\"PED_address_slots\":"<<join.address_slots
            <<",\"PED_decode_slots\":"<<join.decode_slots<<",\"PED_conversion_issues\":"<<join.conversion_issues
            <<",\"intermediate_BN_write_bytes\":"<<intermediate_write<<",\"intermediate_BN_read_bytes\":"<<intermediate_read
            <<",\"final_write_bytes\":73728000,\"same_Machine_prefix_copied\":true}";
        std::cerr<<name<<" complete "<<e.t<<" slots\n";
    }
    std::cout<<"]}\n";
}
