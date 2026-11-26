#include "Vbn_reduce_addr.h"
#include "verilated.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Op {int s0,s1,dst,level; bool carry,last;};
struct Word {int64_t value=0; int channel=-1, first=-1, last=-1, count=0;};
static void need(bool b,const char* msg) {if(!b)throw std::runtime_error(msg);}
static uint32_t rng=0x20261016;
static uint32_t random32(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static std::vector<Op> golden(int samples,int channels,int& finalbase) {
    std::vector<Op> out;int source=0,destination=4096,level=0;
    for(int n=samples;n>1;n=(n+1)/2) {
        for(int pair=0;pair<(n+1)/2;pair++) for(int c=0;c<channels;c++) {
            bool carry=(2*pair+1==n);
            out.push_back({source+2*pair*channels+c,carry?0:source+(2*pair+1)*channels+c,
                           destination+pair*channels+c,level,carry,
                           pair==(n+1)/2-1 && c==channels-1});
        }
        std::swap(source,destination);level++;
    }
    finalbase=source;return out;
}
static std::array<int,10> descriptor(Vbn_reduce_addr& d) {
    return {d.cmd_src0,d.cmd_src1,d.cmd_dst,d.cmd_carry,d.cmd_level,d.cmd_level_end,
            d.cmd_src0_bank,d.cmd_src1_bank,d.cmd_dst_bank,d.cmd_valid};
}
static void tick(Vbn_reduce_addr& d) {d.clk=1;d.eval();d.clk=0;d.eval();}
int main(int argc,char** argv) {
 try {
    Verilated::commandArgs(argc,argv);need(argc==3,"usage: tb cases.txt output.csv");
    std::ifstream cases(argv[1]);need(bool(cases),"cases input");
    std::ofstream report(argv[2]);need(bool(report),"report output");
    report<<"samples,channels,valid,ops,pairs,carries,command_stalls,retire_wait_cycles,cycles\n";
    int samples,channels,test_count=0;uint64_t allops=0,allcarry=0,stalls=0;
    Vbn_reduce_addr d;d.clk=0;d.rst_n=0;d.cfg_valid=0;d.cmd_ready=0;
    d.retire_valid=0;d.finish_ready=0;tick(d);tick(d);d.rst_n=1;d.eval();
    while(cases>>samples>>channels) {
        bool valid=samples>0 && channels>0 && int64_t(samples)*channels<=4096;
        int finalbase=0;auto ops=valid?golden(samples,channels,finalbase):std::vector<Op>{};
        std::array<Word,8192> memory{};std::vector<int64_t> sums(valid?channels:0);
        if(valid)for(int s=0;s<samples;s++)for(int c=0;c<channels;c++) {
            int value=((s*17+c*31)%257)-128;
            memory[s*channels+c]={value,c,s,s,1};sums[c]+=value;
        }
        bool configured=false,pending=false,completed=false,held=false;
        std::array<int,10> held_descriptor{};Word pending_word{},operand0{},operand1{};
        Op pending_op{};int pending_dst=0,delay=0,consumer_phase=0;
        size_t accepted=0,retired=0;int cycles=0,backpressure=0,retire_wait=0,finish_wait=0,pairs=0,carries=0;
        while(cycles++<300000) {
            // No reset between chunks; accepted configuration must be latched.
            d.cfg_samples=configured?(random32()%8192):samples;
            d.cfg_channels=configured?(random32()%8192):channels;d.cfg_valid=!configured;
            d.cmd_ready=(random32()%4)!=0;
            d.retire_valid=pending && consumer_phase==3 && delay==0;
            d.finish_ready=finish_wait>=3;
            d.eval();
            if(held)need(d.cmd_valid && descriptor(d)==held_descriptor,"command changed under backpressure");
            held=d.cmd_valid && !d.cmd_ready;
            if(held){held_descriptor=descriptor(d);backpressure++;}
            if(d.cfg_valid && d.cfg_ready)configured=true;
            if(pending)need(!d.cmd_valid && !d.cfg_ready && !d.finish_valid,"outstanding operation/barrier violation");
            if(d.cmd_valid && d.cmd_ready) {
                need(valid && !pending && accepted<ops.size(),"extra command or capacity fault");
                Op op=ops[accepted++];
                need(d.cmd_src0==op.s0 && d.cmd_src1==op.s1 && d.cmd_dst==op.dst &&
                     bool(d.cmd_carry)==op.carry && d.cmd_level==op.level && bool(d.cmd_level_end)==op.last,
                     "mathematical address sequence mismatch");
                need(op.s0>=0 && op.s0<8192 && op.dst>=0 && op.dst<8192 && op.s0/4096!=op.dst/4096,
                     "finite buffer or read-before-overwrite violation");
                need(d.cmd_src0_bank==op.s0%16 && d.cmd_src0_row==op.s0/16 &&
                     d.cmd_src1_bank==op.s1%16 && d.cmd_src1_row==op.s1/16 &&
                     d.cmd_dst_bank==op.dst%16 && d.cmd_dst_row==op.dst/16,"bank/row mapping");
                if(!op.carry) {
                    need(op.s1>=0 && op.s1<8192 && op.s1/4096==op.s0/4096,"second source capacity");
                    pairs++;
                } else carries++;
                pending=true;pending_op=op;pending_dst=op.dst;consumer_phase=0;delay=1+random32()%5;
            }
            if(d.retire_valid && d.retire_ready) {
                need(pending && retired<accepted,"spurious retire");
                memory[pending_dst]=pending_word;pending=false;retired++;
            } else if(d.retire_ready)retire_wait++;
            // Behavioral consumer: at most one read word in a cycle, even when
            // two source addresses share a bank. No free dual read or copy.
            // Read0 -> optional read1 -> integer add/copy -> acknowledged write.
            // Latencies/stalls are synthetic protocol stress, not SRAM/FP timing.
            if(pending && delay==0 && consumer_phase<3) {
                if(consumer_phase==0) {
                    operand0=memory[pending_op.s0];need(operand0.count>0,"read0 uninitialized");
                    consumer_phase=pending_op.carry?2:1;
                } else if(consumer_phase==1) {
                    operand1=memory[pending_op.s1];
                    need(operand1.count>0 && operand0.channel==operand1.channel && operand0.last+1==operand1.first,
                         "mixed channel, missing or duplicate coverage");
                    consumer_phase=2;
                } else {
                    pending_word=pending_op.carry?operand0:
                        Word{operand0.value+operand1.value,operand0.channel,operand0.first,operand1.last,operand0.count+operand1.count};
                    consumer_phase=3;
                }
                delay=1+random32()%3;
            }
            if(d.finish_valid) {
                need(configured && !pending && retired==ops.size() && accepted==retired,"early finish");
                need(bool(d.finish_fault)==!valid && d.result_channels==channels,"finish contract");
                if(valid) {
                    need(d.result_base==finalbase,"final region");
                    for(int c=0;c<channels;c++) {
                        Word w=memory[finalbase+c];
                        need(w.value==sums[c] && w.channel==c && w.first==0 && w.last==samples-1 && w.count==samples,
                             "independent integer sum/coverage golden");
                    }
                }
                finish_wait++;
                if(d.finish_ready){tick(d);need(d.cfg_ready,"finish did not return idle");completed=true;break;}
            }
            tick(d);if(pending && delay>0)delay--;
        }
        need(completed,"timeout");need(!valid || pairs==(samples-1)*channels,"add count");
        report<<samples<<','<<channels<<','<<valid<<','<<accepted<<','<<pairs<<','<<carries<<','<<backpressure<<','<<retire_wait<<','<<cycles<<'\n';
        test_count++;allops+=accepted;allcarry+=carries;stalls+=backpressure;
    }
    need(test_count>0,"empty cases");
    std::cout<<"PASS cases="<<test_count<<" ops="<<allops<<" carry="<<allcarry<<" command_stalls="<<stalls<<"\n";
    return 0;
 }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}
}
