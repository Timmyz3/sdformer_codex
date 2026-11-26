#include "Vdot_residual.h"
#include "verilated.h"
#include <array>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <string>
#ifndef PE_LANES
#define PE_LANES 4
#endif
static constexpr unsigned LANES=PE_LANES;
static uint32_t rng=9417;
static uint32_t next(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static void need(bool v,const char*m){if(!v)throw std::runtime_error(m);}
int main(int argc,char**argv){
 Verilated::commandArgs(argc,argv);Vdot_residual d;
 std::ofstream trace("cycle_trace.csv");
 trace<<"cycle,case_id,cmd_fire,read_fire,response_fire,write_fire,read_index,write_tag\n";
 uint64_t cycle=0,reads=0,writes=0,cases=0,stall=0;
 auto edge=[&](){d.clk=0;d.eval();d.clk=1;d.eval();++cycle;};
 d.rst=1;d.cmd_valid=0;d.read_ready=0;d.response_valid=0;d.write_ready=0;edge();edge();d.rst=0;
 std::vector<unsigned> counts={0,1,2,10,16,96,384};
 std::vector<unsigned> nostall_cycles;
 for(unsigned test=0;test<90;++test){
   bool keep=(test%5)!=0;unsigned K=counts[test%counts.size()];unsigned mask=(test%7==0)?1:((test%7==1)?((1u<<(LANES-1))-1):((1u<<LANES)-1));
   bool backpressure=test>=14;unsigned due=0,pending_index=0;bool pending=false,sent=false,done=false;
   std::vector<int8_t>a(K);std::vector<std::array<int8_t,LANES>>b(K);
   std::array<uint32_t,LANES> gold,bias,res;
   for(unsigned l=0;l<LANES;++l){bias[l]=next();res[l]=next();gold[l]=keep?bias[l]+res[l]:res[l];d.cmd_bias[l]=bias[l];d.cmd_residual[l]=res[l];}
   for(unsigned k=0;k<K;++k){a[k]=(int8_t)next();for(unsigned l=0;l<LANES;++l){b[k][l]=(int8_t)next();if(keep)gold[l]+=(uint32_t)((int32_t)a[k]*(int32_t)b[k][l]);}}
   for(unsigned l=0;l<LANES;++l)if(!(mask>>l&1))gold[l]=0;
   d.cmd_keep=keep;d.cmd_count=K;d.cmd_tag=test;d.cmd_lane_mask=mask;
   uint64_t begin=cycle;unsigned localreads=0;bool held=false;std::array<uint32_t,LANES> previous{};
   while(!done){
     need(cycle-begin<10000,"timeout");
     d.clk=0;d.cmd_valid=!sent;
     d.read_ready=!backpressure||(cycle%7!=2&&cycle%7!=3);
     d.write_ready=!backpressure||(cycle%11>=4);
     d.response_valid=pending&&cycle>=due;
     if(pending){d.response_scalar=(uint8_t)a[pending_index];
#if PE_LANES == 4
       uint32_t pack=0;for(unsigned l=0;l<LANES;++l)pack|=(uint32_t)(uint8_t)b[pending_index][l]<<(8*l);d.response_vector=pack;
#else
       for(unsigned word=0;word<LANES/4;++word){uint32_t pack=0;for(unsigned j=0;j<4;++j)pack|=(uint32_t)(uint8_t)b[pending_index][word*4+j]<<(8*j);d.response_vector[word]=pack;}
#endif
     }
     d.eval();
     bool cf=d.cmd_valid&&d.cmd_ready,rf=d.read_valid&&d.read_ready,sf=d.response_valid&&d.response_ready,wf=d.write_valid&&d.write_ready;
     if(held){need(d.write_valid,"write valid dropped under stall");for(unsigned l=0;l<LANES;++l)need(d.write_data[l]==previous[l],"unstable write");need(d.write_tag==test,"unstable tag");}
     held=d.write_valid&&!d.write_ready;if(held){++stall;for(unsigned l=0;l<LANES;++l)previous[l]=d.write_data[l];}
     if(cf)sent=true;
     if(rf){need(!pending,"more than one outstanding read");need(d.read_index==localreads&&localreads<K,"read index mismatch");pending_index=d.read_index;pending=true;due=cycle+1+(backpressure?next()%4:0);++localreads;++reads;}
     if(sf){need(pending,"unsolicited response");pending=false;}
     if(wf){need(d.write_tag==test&&d.write_lane_mask==mask,"write descriptor mismatch");for(unsigned l=0;l<LANES;++l)need(d.write_data[l]==gold[l],"arithmetic mismatch");need(localreads==(keep?K:0),"bypass read or missing MAC");done=true;++writes;}
     trace<<cycle<<','<<test<<','<<cf<<','<<rf<<','<<sf<<','<<wf<<','<<d.read_index<<','<<d.write_tag<<'\n';edge();
   }
   if(!backpressure){unsigned actual=cycle-begin;unsigned expected=keep?2*K+2:2;need(actual==expected,"no-stall cycle mismatch");nostall_cycles.push_back(actual);}
   ++cases;
 }
 // Reset while a response is outstanding: memory/testbench flushes its pending
 // response at the same reset. Accepted writes from earlier cases remain valid.
 d.cmd_keep=1;d.cmd_count=10;d.cmd_lane_mask=(1u<<LANES)-1;d.cmd_valid=1;d.read_ready=1;d.response_valid=0;d.write_ready=0;edge();d.cmd_valid=0;edge();
 need(d.response_ready,"reset test not waiting for response");d.rst=1;d.eval();need(!d.read_valid&&!d.write_valid&&!d.response_ready,"reset handshake suppression");edge();d.rst=0;d.eval();need(d.cmd_ready&&!d.write_valid,"reset did not clear pending op");
 std::ofstream out("summary.json");out<<"{\n\"status\":\"passed\",\"lanes\":"<<LANES<<",\"completed_cases\":"<<cases<<",\"read_handshakes\":"<<reads<<",\"write_handshakes\":"<<writes<<",\"blocked_write_cycles\":"<<stall<<",\"testbench_cycles\":"<<cycle<<",\"reset_outstanding_tests\":1,\"nostall_checked_cases\":"<<nostall_cycles.size()<<",\"nostall_formula\":\"keep: 2*K+2; bypass: 2\"}\n";
 std::cout<<"PASS "<<cases<<" cases, "<<cycle<<" cycles\n";
}
