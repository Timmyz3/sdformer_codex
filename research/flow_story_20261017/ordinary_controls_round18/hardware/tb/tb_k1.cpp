#include "Vk1_f64.h"
#include "verilated.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
static uint64_t bits(double x){uint64_t q;std::memcpy(&q,&x,8);return q;}
static double val(uint64_t x){double q;std::memcpy(&q,&x,8);return q;}
static void need(bool x,const char*m){if(!x)throw std::runtime_error(m);}
int main(int argc,char**argv){
 Verilated::commandArgs(argc,argv);Vk1_f64 d;std::mt19937_64 rng(180017);
 uint64_t cycles=0,accepted=0,stalls=0;unsigned tests=0,bad=0,mincy=99999,maxcy=0;
 auto tick=[&](){d.clk=0;d.eval();d.clk=1;d.eval();cycles++;};
 d.rst_n=0;d.start_valid=0;d.score_valid=0;d.result_ready=0;tick();d.rst_n=1;
 auto test=[&](const std::array<uint64_t,12>&a,bool stress){
   unsigned expected=0;bool fault=false;
   for(unsigned j=0;j<12;j++){fault|=!std::isfinite(val(a[j]));if(j && val(a[j])>val(a[expected]))expected=j;}
   need(d.start_ready,"not idle between transactions");auto begin=cycles;
   d.start_valid=1;tick();d.start_valid=0;
   need(d.score_ready&&!d.result_valid,"start handshake");
   for(unsigned j=0;j<12;j++){
     unsigned pause=stress?rng()%4:0;
     for(unsigned k=0;k<pause;k++){d.score_valid=0;d.score_bits=rng();tick();stalls++;need(d.score_ready&&!d.result_valid,"count changed on invalid input");}
     d.score_valid=1;d.score_bits=a[j];need(d.score_ready,"score not accepted");tick();accepted++;
     if(j<11)need(!d.result_valid,"early decision");
   }
   d.score_valid=0;need(d.result_valid&&!d.score_ready,"missing completed decision");
   need(bool(d.fault)==fault,"fault disagreement");
   if(fault){bad++;need(d.fine_mask==0,"invalid data emitted runnable mask");}
   else {need(d.winner==expected,"strict-first numeric argmax mismatch");need(d.fine_mask==(1u<<(expected+expected/4)),"real-tail grid mask mismatch");}
   unsigned payload=d.fine_mask,winner=d.winner;
   unsigned pause=stress?rng()%6:0;
   for(unsigned k=0;k<pause;k++){d.score_valid=1;d.score_bits=rng();d.start_valid=1;tick();stalls++;need(d.result_valid&&d.fine_mask==payload&&d.winner==winner,"result unstable under backpressure");}
   d.score_valid=0;d.start_valid=0;d.result_ready=1;tick();d.result_ready=0;
   need(d.start_ready&&!d.result_valid,"result handshake failed");
   unsigned elapsed=cycles-begin;mincy=std::min(mincy,elapsed);maxcy=std::max(maxcy,elapsed);if(!stress)need(elapsed==14,"un-stalled transaction must be start+12scores+retire=14");tests++;
 };
 std::array<uint64_t,12>a;
 for(unsigned winner=0;winner<12;winner++){a.fill(bits(-2));a[winner]=bits(3);test(a,false);}
 a.fill(bits(1));test(a,false);a.fill(bits(-0.0));a[9]=bits(0.0);test(a,true);
 for(unsigned i=0;i<12;i++)a[i]=bits(-double(i+1));test(a,true);
 for(unsigned i=0;i<12;i++)a[i]=bits(double(i+1));test(a,true);
 for(unsigned k=0;k<2000;k++){
   for(auto&x:a){x=rng();if((x>>52&2047)==2047)x^=uint64_t(1)<<52;}
   if(k%3==0)a[11]=a[0];test(a,true);
 }
 for(uint64_t invalid:{0x7ff0000000000000ULL,0xfff0000000000000ULL,0x7ff8000000000001ULL}){a.fill(bits(1));a[5]=invalid;test(a,true);}
 unsigned synthetic=tests;
 if(argc>1){std::ifstream f(argv[1]);need(bool(f),"cannot read actual score vectors");while(true){if(!(f>>std::hex>>a[0]))break;for(unsigned i=1;i<12;i++)need(bool(f>>std::hex>>a[i]),"truncated actual score vector");test(a,true);}}
 std::cout<<"{\"status\":\"PASS\",\"synthetic_cases\":"<<synthetic<<",\"actual_score_cases\":"<<tests-synthetic<<",\"total_cases\":"<<tests<<",\"invalid_cases\":"<<bad<<",\"accepted_scores\":"<<accepted<<",\"backpressure_and_input_pause_ticks\":"<<stalls<<",\"total_test_cycles\":"<<cycles<<",\"minimum_transaction_ticks\":"<<mincy<<",\"maximum_stressed_transaction_ticks\":"<<maxcy<<",\"unstalled_start_to_decision_ticks\":13,\"unstalled_transaction_ticks\":14,\"FP_score_arithmetic_implemented\":false}"<<std::endl;
 d.final();return 0;
}
