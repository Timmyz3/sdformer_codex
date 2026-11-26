#include <verilated.h>
#if defined(TOP_GP)
#include "Vnative_gp.h"
using DUT = Vnative_gp;
#elif defined(TOP_PR)
#include "Vnative_pr.h"
using DUT = Vnative_pr;
#elif defined(TOP_GP4)
#include "Vnative_gp4.h"
using DUT = Vnative_gp4;
#else
#include "Vnative_fc.h"
using DUT = Vnative_fc;
#endif
#include <array>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>
#include <cstdint>

struct Case {
    std::string name; int c, words;
    std::vector<std::array<uint32_t,4>> s;
    std::vector<std::array<int,8>> w, y;
};
static DUT d;
static void tick() { d.clk=0;d.eval();d.clk=1;d.eval();d.clk=0;d.eval(); }
static int signed24(uint32_t x) { x&=0xffffff;return x&0x800000 ? int(x)-0x1000000 : int(x); }
static int value(int h) {
    int p=h*24, j=p/32, b=p%32;
    uint64_t v=d.out_data[j];
    if(j+1<6)v|=uint64_t(d.out_data[j+1])<<32;
    return signed24(uint32_t(v>>b));
}
static void config(bool kind,int addr,const std::array<uint32_t,4>& v) {
    if(!d.cfg_ready)throw std::runtime_error("config while busy");
    d.cfg_valid=1;d.cfg_kind=kind;d.cfg_addr=addr;
    for(int i=0;i<4;i++)d.cfg_data[i]=v[i];
    tick();d.cfg_valid=0;
}
static int load_source(const Case& c,int rotate) {
    for(int r=0;r<80;r++)for(int k=0;k<c.words;k++)
        config(false,r*3+k,c.s[((r+rotate)%80)*c.words+k]);
    return 80*c.words;
}
static int load_weights(const Case& c) {
    for(int k=0;k<c.c;k++) {
        std::array<uint32_t,4> v={};
        for(int h=0;h<8;h++)v[h/4]|=uint32_t(uint8_t(c.w[k][h]))<<((h%4)*8);
        config(true,k,v);
    }
    return c.c;
}
static void run(const Case& c,int mode,int calendar,int rotate,const char* name,int cfg,int table_writes=0) {
    d.c_count=c.c;d.mode=mode;d.start=1;d.out_ready=0;
    d.src_enable=1;d.coef_enable=1;tick();d.start=0;
    unsigned cycles=1,received=0,stall=0;
    std::array<uint32_t,6> held={}; int heldrow=0;bool holding=false;
    for(;cycles<300000;cycles++) {
        unsigned t=cycles-1;
        d.src_enable=calendar==0 || t%4!=0;
        d.coef_enable=calendar==0 || t%3!=1;
        d.out_ready=calendar==0 || t%7<4;
        d.eval();
        if(holding) {
            if(!d.out_valid || d.out_row!=heldrow)throw std::runtime_error("unstable output valid/row");
            for(int j=0;j<6;j++)if(held[j]!=d.out_data[j])throw std::runtime_error("unstable output data");
        }
        if(d.out_valid && d.out_ready) {
            if(d.out_row!=received)throw std::runtime_error("unexpected row order");
            for(int h=0;h<8;h++)if(value(h)!=c.y[(received+rotate)%80][h]) {
                std::cerr<<c.name<<" "<<name<<" row="<<received<<" h="<<h
                         <<" got="<<value(h)<<" expected="<<c.y[(received+rotate)%80][h]<<"\n";
                throw std::runtime_error("integer mismatch");
            }
            received++;
        }
        holding=d.out_valid && !d.out_ready;
        if(holding) {
            stall++;heldrow=d.out_row;
            for(int j=0;j<6;j++)held[j]=d.out_data[j];
        }
        tick();
        if(d.done) {
            if(received!=80 || d.busy)throw std::runtime_error("bad completion");
            cycles++;
            break;
        }
    }
    if(received!=80)throw std::runtime_error("timeout");
    std::cout<<c.name<<","<<name<<","<<calendar<<","<<c.c<<","<<cycles<<","<<cfg<<","<<cycles+cfg
             <<","<<d.dbg_source_reads<<","<<d.dbg_coef_reads<<","<<d.dbg_add_cycles
             <<","<<d.dbg_build_cycles<<","<<stall<<","<<table_writes;
#if defined(TOP_PR)
    std::cout<<","<<d.dbg_match_cycles<<","<<d.dbg_sort_cycles
             <<","<<d.dbg_parent_reads<<","<<d.dbg_parent_writes
             <<","<<d.dbg_y_reads<<","<<d.dbg_y_writes<<","<<d.dbg_k_tiles;
#else
    std::cout<<",0,0,0,0,0,0,0";
#endif
    std::cout<<"\n";
}
int main(int argc,char**argv) {
    Verilated::commandArgs(argc,argv);
    try {
        std::ifstream f(argc>1?argv[1]:"cases.txt");
        if(!f)throw std::runtime_error("no case file");
        int n;f>>n;std::vector<Case> cases(n);
        for(auto& c:cases) {
            f>>c.name>>c.c;c.words=(c.c+127)/128;
            c.s.resize(80*c.words);c.w.resize(c.c);c.y.resize(80);
            for(auto&v:c.s)for(auto&x:v)f>>std::hex>>x;
            for(auto&v:c.w)for(auto&x:v)f>>std::dec>>x;
            for(auto&v:c.y)for(auto&x:v)f>>x;
            // Independent scalar check, before any RTL execution.
            for(int r=0;r<80;r++)for(int h=0;h<8;h++) {
                int ref=0;
                for(int k=0;k<c.c;k++)
                    if(c.s[r*c.words+k/128][(k%128)/32] & (1u<<(k%32)))ref+=c.w[k][h];
                if(ref!=c.y[r][h])throw std::runtime_error("host reference mismatch");
            }
        }
        if(!f)throw std::runtime_error("truncated cases");
        d.rst_n=0;d.cfg_valid=0;d.start=0;tick();tick();d.rst_n=1;tick();
        std::cout<<"case,mode,calendar,C,cycles,config_cycles,total_cycles,source_reads,coef_reads,add_cycles,build_cycles,out_stalls,table_writes,match_cycles,sort_cycles,parent_reads,parent_writes,y_reads,y_writes,k_tiles\n";
        for(const auto&c:cases)for(int cal=0;cal<2;cal++) {
            int cfg=load_weights(c)+load_source(c,0);
#if defined(TOP_GP)
            run(c,0,cal,0,"GP",cfg);
            cfg=load_source(c,8);
            run(c,0,cal,8,"GP_reloaded",cfg);
#elif defined(TOP_PR)
            run(c,0,cal,0,"PR",cfg);
            cfg=load_source(c,8);
            run(c,0,cal,8,"PR_reloaded",cfg);
#elif defined(TOP_GP4)
            run(c,0,cal,0,"GP4_cold",cfg,4*c.c);
            cfg=load_source(c,8);
            run(c,1,cal,8,"GP4_warm",cfg);
#else
            run(c,0,cal,0,"AAC",cfg);
            // Both memories reloaded: charge actual common cold configuration.
            cfg=load_weights(c)+load_source(c,0);
            run(c,1,cal,0,"G4_cold",cfg,4*c.c);
            cfg=load_source(c,8); // Permute whole P8 groups; preserve their spatial membership.
            run(c,1,cal,8,"G4_warm",cfg);
            run(c,0,cal,8,"AAC_reloaded",cfg);
#endif
        }
        // Real sequence of DIFFERENT spatial slices: retain W/LUT only while W is unchanged.
        // This measures construction amortization instead of assuming a warm table forever.
#if defined(TOP_GP) || defined(TOP_PR) || defined(TOP_GP4)
        const int variants=1;
#else
        const int variants=2;
#endif
        for(int cal=0;cal<2;cal++)for(int variant=0;variant<variants;variant++) {
            std::vector<std::array<int,8>> previous_w;
            for(const auto&c:cases) {
                bool changed=c.w!=previous_w;
                int cfg=changed ? load_weights(c):0;
                cfg+=load_source(c,0);
#if defined(TOP_GP)
                run(c,0,cal,0,"GP_stream",cfg);
#elif defined(TOP_PR)
                run(c,0,cal,0,"PR_stream",cfg);
#elif defined(TOP_GP4)
                run(c,1,cal,0,"GP4_stream",cfg,changed?4*c.c:0);
#else
                run(c,variant,cal,0,variant?"G4_stream":"AAC_stream",cfg,
                    changed&&variant?4*c.c:0);
#endif
                previous_w=c.w;
            }
        }
        std::cerr<<"PASS cases="<<cases.size()<<"; exact signed integer outputs and backpressure stability\n";
    }catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}
}
