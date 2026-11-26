#include <verilated.h>
#if defined(STATIC_ENGINE)
#include "Vbitmap_ctrl_static.h"
using DUT=Vbitmap_ctrl_static;
static constexpr int engine=1;
#elif defined(NONE_ENGINE)
#include "Vbitmap_ctrl_none.h"
using DUT=Vbitmap_ctrl_none;
static constexpr int engine=2;
#else
#include "Vbitmap_ctrl_dynamic.h"
using DUT=Vbitmap_ctrl_dynamic;
static constexpr int engine=0;
#endif
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
using U=uint64_t;
static void need(bool p,const char*s){if(!p)throw std::runtime_error(s);}
struct Fired{bool q,m,r;unsigned plane,addr;U tick;};
struct Test {
 DUT d;std::array<uint32_t,192> memory{},maps{};
 U cycle=0,queries=0,reads=0,responses=0,init_writes=0,busy_rejections=0,finish_rejections=0,invalid_cases=0,stalled_responses=0;
 bool pending=false;U due=0;uint32_t pending_data=0;std::mt19937 rng{20261021};
 std::set<U> active_lat,bypass_lat;
 Test(const char*path){std::ifstream f(path);need(bool(f),"open bitmap file");for(auto&x:maps){f>>std::hex>>x;need(bool(f),"192 bitmap words");}}
 Fired tick(bool memstall=false,bool outstall=false,bool unsolicited=false){
  d.clk=0;d.mem_ready=!memstall;d.rsp_ready=!outstall;
  d.mem_rvalid=(pending&&due==cycle)||unsolicited;d.mem_rdata=pending_data;d.eval();
  Fired e{bool(d.req_valid&&d.req_ready),bool(d.mem_req&&d.mem_ready),bool(d.rsp_valid&&d.rsp_ready),unsigned(d.start_plane),unsigned(d.mem_addr),cycle};
  if(d.rsp_valid&&outstall)stalled_responses++;
  if(pending&&due==cycle)pending=false;
  if(e.m){need(!pending,"one outstanding SRAM read");need(e.addr<192,"descriptor RF bounds");pending=true;due=cycle+1;pending_data=memory[e.addr];reads++;}
  if(e.q)queries++;if(e.r)responses++;
  d.clk=1;d.eval();cycle++;return e;
 }
 void reset(){d.rst_n=0;d.start_valid=0;d.finish_valid=0;d.req_valid=0;d.action=0;d.tile_id=0;d.complete_tile=0;pending=false;tick();tick();d.rst_n=1;tick();need(!d.fault,"reset clears fault");}
 void init(int action){memory.fill(0);int n=(action==6?0:engine==0?192:32);for(int i=0;i<n;i++){
   need(!d.active,"RF writes before convolution");memory[i]=maps[engine==0?i:action*32+i];tick(true,false);need(!d.mem_req,"no concurrent1RW load and read");init_writes++;
  }}
 void start(int action){d.action=action;d.start_valid=1;auto e=tick();(void)e;d.start_valid=0;need(d.active&&!d.fault,"accepted legal start");}
 void finish(){d.finish_valid=1;tick();d.finish_valid=0;need(!d.active&&!d.fault,"finish convolution");}
 unsigned expected(int action,int tile,bool complete){return !complete?0:(action<6&&tile>=action*151&&tile<(action+1)*151)?0:4;}
 void query(int action,int tile,bool complete,bool random_stalls,bool inject_busy=false,bool inject_finish=false){
  U accepted=~U(0);bool done=false,finish_injected=false,seen_rsp=false;unsigned held=0;
  d.req_valid=1;d.complete_tile=complete;d.tile_id=tile;
  if(inject_busy){d.start_valid=1;d.action=(action+1)%7;}
  for(unsigned guard=0;guard<200&&!done;guard++){
   bool ms=random_stalls&&(rng()%3==0),os=random_stalls&&(rng()%2==0);
   if(inject_finish&&accepted!=~U(0)&&!finish_injected){d.finish_valid=1;finish_injected=true;}
   auto e=tick(ms,os);
   if(inject_busy){need(d.reject_start,"busy action update rejected");busy_rejections++;d.start_valid=0;inject_busy=false;}
   if(d.finish_valid){need(d.reject_finish,"pending finish rejected");finish_rejections++;d.finish_valid=0;}
   need(!d.fault,"legal query no fault");
   if(e.q){need(accepted==~U(0),"single request acceptance");accepted=e.tick;d.req_valid=0;}
   // External action changes must not change the latched convolution choice.
   d.action=rng()%8;
   if(d.rsp_valid){if(seen_rsp)need(d.start_plane==held,"stable response under backpressure");held=d.start_plane;seen_rsp=true;}
   if(e.r){need(e.plane==expected(action,tile,complete),"golden plane equality");need(accepted!=~U(0),"response belongs to request");
    if(!random_stalls){U elapsed=e.tick-accepted;if(complete&&action<6){need(elapsed==3,"active span latency3");active_lat.insert(elapsed);}else{need(elapsed==1,"bypass span latency1");bypass_lat.insert(elapsed);}}
    done=true;
   }
  }need(done,"query completion bounded");
 }
 void run(){
  int lo=engine==2?6:0,hi=engine==1?5:6;
  for(int a=lo;a<=hi;a++)for(int pass=0;pass<2;pass++){
   reset();init(a);start(a);
   for(int id=0;id<910;id++)query(a,id,true,pass==1,id%137==0,pass==1&&id%173==0);
   for(int id=0;id<1024;id+=17)query(a,id,false,pass==1);
   finish();
  }
  // Every non-complete/padding ID is invalid when marked complete.
  for(int id=910;id<1024;id++){
   reset();start(engine==2?6:0);U before=reads;d.req_valid=1;d.complete_tile=1;d.tile_id=id;tick();d.req_valid=0;
   need(d.fault&&reads==before,"invalid complete ID rejected before RF");invalid_cases++;
  }
  if(engine==0){reset();d.start_valid=1;d.action=7;tick();d.start_valid=0;need(d.fault&&!d.active,"action7 rejected");invalid_cases++;}
  reset();start(engine==2?6:0);tick(false,false,true);need(d.fault,"unsolicited RF response rejected");invalid_cases++;
  reset();d.finish_valid=1;tick();need(d.reject_finish&&!d.active,"finish while idle rejected");d.finish_valid=0;invalid_cases++;
  std::cout<<"{\"engine\":"<<engine<<",\"status\":\"PASS\",\"queries\":"<<queries<<",\"RF_reads\":"<<reads<<",\"responses\":"<<responses<<",\"initialization_1RW_word_writes\":"<<init_writes<<",\"busy_start_rejections\":"<<busy_rejections<<",\"pending_finish_rejections\":"<<finish_rejections<<",\"invalid_cases\":"<<invalid_cases<<",\"response_backpressure_ticks\":"<<stalled_responses<<",\"no_stall_active_span_ticks\":"<<(active_lat.empty()?"null":"3")<<",\"no_stall_bypass_span_ticks\":1,\"all910_tiles\":true,\"prediction_tested\":false,\"whole_network\":false}\n";
 }
};
int main(int argc,char**argv){try{Verilated::commandArgs(argc,argv);need(argc==2,"usage tb bitmap_words");Test t(argv[1]);t.run();return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
