#include "Vnative_gp.h"
#include "verilated.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static void tick(Vnative_gp &d) {
    d.clk=0; d.eval(); d.clk=1; d.eval();
}
static int32_t lane24(const Vnative_gp &d, int h) {
    int bit=h*24, word=bit/32, off=bit%32;
    uint64_t v=d.out_data[word];
    if(word+1<6) v|=uint64_t(d.out_data[word+1])<<32;
    uint32_t q=(v>>off)&0xffffffu;
    return (q&0x800000u)?int32_t(q|0xff000000u):int32_t(q);
}
static void require(bool v, const char *msg) {
    if(!v) { std::fprintf(stderr,"FAIL %s\n",msg); std::exit(1); }
}
int main(int argc,char **argv) {
    Verilated::commandArgs(argc,argv);
    Vnative_gp d;
    d.clk=0; d.rst_n=0; d.cfg_valid=0; d.start=0;
    d.src_enable=d.coef_enable=d.out_ready=1; d.mode=0;
    for(int i=0;i<4;i++) tick(d);
    d.rst_n=1; tick(d);
    uint64_t checked=0;
    for(int C: {96,384}) for(int pattern=0;pattern<5;pattern++) for(int bp=0;bp<2;bp++) {
        std::mt19937 gen(12345+C*7+pattern);
        std::vector<uint8_t> s(80*C);
        std::vector<int8_t> w(C*8);
        for(int r=0;r<80;r++) for(int c=0;c<C;c++) {
            unsigned rnd=gen()%100;
            s[r*C+c]=pattern==0?0:pattern==1?(c==((r*13)%C)):
                       pattern==2?(rnd<5):pattern==3?(rnd<33):1;
        }
        for(int c=0;c<C;c++) for(int h=0;h<8;h++)
            w[c*8+h]=int8_t(int(gen()%256)-128);
        // Include both INT8 extremes explicitly, including negative extension.
        w[0]=-128; w[1]=127;
        std::array<std::array<int32_t,8>,80> gold{};
        uint32_t nadd=0,nw=0,nbuild=0;
        for(int r=0;r<80;r++) for(int c=0;c<C;c++) if(s[r*C+c]) {
            nadd++;
            for(int h=0;h<8;h++) gold[r][h]+=w[c*8+h];
        }
        for(int t=0;t<10;t++) for(int wc=0;wc<(C+127)/128;wc++) {
            unsigned nc=0;
            for(int c=wc*128;c<C && c<(wc+1)*128;c++) {
                bool any=false;
                for(int p=0;p<8;p++) any=any||s[(t*8+p)*C+c];
                if(any) { nc++; nw++; }
            }
            nbuild+=(nc+3)/4;
        }
        d.cfg_valid=1; d.cfg_kind=0;
        for(int r=0;r<80;r++) for(int wc=0;wc<(C+127)/128;wc++) {
            for(int q=0;q<4;q++) d.cfg_data[q]=0;
            for(int b=0;b<128 && wc*128+b<C;b++)
                if(s[r*C+wc*128+b]) d.cfg_data[b/32]|=uint32_t(1)<<(b%32);
            d.cfg_addr=r*3+wc; tick(d);
        }
        d.cfg_kind=1;
        for(int c=0;c<C;c++) {
            for(int q=0;q<4;q++) d.cfg_data[q]=0;
            for(int h=0;h<8;h++)
                d.cfg_data[h/4]|=uint32_t(uint8_t(w[c*8+h]))<<((h%4)*8);
            d.cfg_addr=c; tick(d);
        }
        d.cfg_valid=0; d.c_count=C; d.start=1; tick(d); d.start=0;
        unsigned got=0, cycles=0;
        bool held=false;
        std::array<uint32_t,6> held_data{};
        unsigned held_row=0;
        for(;cycles<1000000;cycles++) {
            d.src_enable=!bp || cycles%5!=0;
            d.coef_enable=!bp || cycles%7<5;
            d.out_ready=!bp || cycles%11<7;
            d.clk=0; d.eval();
            if(held) {
                require(d.out_valid && d.out_row==held_row,"hold valid/row");
                for(int q=0;q<6;q++) require(d.out_data[q]==held_data[q],"hold data");
            }
            bool accept=d.out_valid&&d.out_ready;
            if(accept) {
                require(d.out_row==got,"output row order");
                for(int h=0;h<8;h++) {
                    int32_t actual=lane24(d,h);
                    if(actual!=gold[got][h]) {
                        std::fprintf(stderr,"C=%d pat=%d bp=%d r=%u h=%d got=%d exp=%d\n",C,pattern,bp,got,h,actual,gold[got][h]);
                        return 1;
                    }
                    checked++;
                }
                got++;
            }
            held=d.out_valid&&!d.out_ready;
            if(held) {
                held_row=d.out_row;
                for(int q=0;q<6;q++) held_data[q]=d.out_data[q];
            }
            d.clk=1; d.eval();
            if(d.done) {
                require(accept && got==80,"done on last accept");
                require(!d.busy,"busy clears with done");
                break;
            }
        }
        require(cycles<1000000,"timeout");
        require(d.dbg_source_reads==uint32_t(80*((C+127)/128)),"source read count");
        require(d.dbg_coef_reads==nw,"coefficient read count");
        require(d.dbg_add_cycles==nadd,"add cycle count");
        require(d.dbg_build_cycles==nbuild,"NRV build count");
        std::printf("PASS C=%d pattern=%d bp=%d cycles=%u src=%u coef=%u add=%u build=%u\n",C,pattern,bp,cycles+1,d.dbg_source_reads,d.dbg_coef_reads,d.dbg_add_cycles,d.dbg_build_cycles);
        tick(d); require(!d.done,"done pulse");
    }
    std::printf("GP_SMOKE_PASS cases=20 output_values=%llu\n",(unsigned long long)checked);
    return 0;
}
