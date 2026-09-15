#include "Vinterleave_stream.h"
#include "verilated.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
double sc_time_stamp(){return 0;}
using Vec=std::vector<uint32_t>;
static Vec hex(const std::string&p){std::ifstream f(p);if(!f)throw std::runtime_error(p);Vec a;uint64_t x;while(f>>std::hex>>x)a.push_back(uint32_t(x));return a;}
struct Fixture{std::string name;Vec source,origin,raw,z,d,id,j,wide,i24;Fixture(std::string p):name(p),source(hex(p+"/source.hex")),origin(hex(p+"/origin.hex")),raw(hex(p+"/gold.hex")),z(hex(p+"/z.hex")),d(hex(p+"/d.hex")),id(hex(p+"/identity.hex")),j(hex(p+"/j.hex")),wide(hex(p+"/wide.hex")),i24(hex(p+"/i24.hex")){if(source.size()!=1536||origin.size()!=2||raw.size()!=3840||z.size()!=640||d.size()!=640||id.size()!=3840||j.size()!=3840||wide.size()!=7680||i24.size()!=3840)throw std::runtime_error("fixture shape "+p);}};
int main(int argc,char**argv){try{
 Verilated::commandArgs(argc,argv);if(argc<5)return 2;std::string root=argv[1];int stall=std::stoi(argv[2]),borrow=std::stoi(argv[3]);
 auto q1=hex(root+"/parameters/q1.hex"),q2=hex(root+"/parameters/q2.hex"),ab=hex(root+"/parameters/consumer.hex");
 if(q1.size()!=4608||q2.size()!=4608||ab.size()!=192)return 3;
 Vinterleave_stream d;auto tick=[&](){d.clk=0;d.eval();d.clk=1;d.eval();d.clk=0;d.eval();};
 d.reset_n=0;d.go=0;d.parameter_valid=0;d.source_valid=0;d.origin_valid=0;d.identity_valid=0;d.compute_source_allow=1;d.compute_weight_allow=1;d.result_ready=1;d.borrow_enable=borrow;tick();tick();d.reset_n=1;
 unsigned command=0;
 for(int arg=4;arg<argc;arg++){
  std::ifstream mf(argv[arg]);std::vector<Fixture> f;std::string p;while(mf>>p)f.emplace_back(p);if(f.empty())return 4;
  for(unsigned repeat=0;repeat<2;repeat++,command++){
   d.first_tile=0;d.tile_count=f.size();d.go=1;tick();d.go=0;
   std::vector<unsigned> raw(f.size()),j(f.size()),wide(f.size()),out(f.size()),z(f.size()),dd(f.size()),id(f.size()),source(f.size()),origin(f.size());
   std::vector<Vec> dseen(f.size(),Vec(80));
   bool req_held[4]={};uint64_t req_key[4]={},request_hold_checks=0;
   uint64_t states[16]={},cstates[2][64]={{}},denied[2][64]={{}},grants[6]={};
   bool held=false;uint32_t held_data[8]={};unsigned held_tile=0,held_addr=0;bool finished=false;
   for(unsigned n=0;n<30000000;n++){
    d.parameter_valid=0;d.source_valid=0;d.origin_valid=0;d.identity_valid=0;
    d.compute_source_allow=!stall||n%11!=3;d.compute_weight_allow=!stall||(n%7!=2&&n%7!=3);d.result_ready=!stall||(n%5!=1&&n%5!=2);d.clk=0;d.eval();
    uint64_t current_key[4]={uint64_t(d.parameter_kind)*16384+d.parameter_address,uint64_t(d.source_tile)*2048+d.source_address,uint64_t(d.origin_tile),uint64_t(d.identity_tile)*512+d.identity_address};
    bool current_valid[4]={bool(d.parameter_request_valid),bool(d.source_request_valid),bool(d.origin_request_valid),bool(d.identity_request_valid)};
    for(int q=0;q<4;q++)if(req_held[q]){request_hold_checks++;if(!current_valid[q]||current_key[q]!=req_key[q])throw std::runtime_error("request moved under backpressure");}
    if(d.parameter_request_valid&&(!stall||n%13!=2)){
     const Vec *v=d.parameter_kind==4?&q1:d.parameter_kind==5?&q2:d.parameter_kind==7?&ab:nullptr;
     if(!v||unsigned(d.parameter_address)*8+7>=v->size())return 5;
     d.parameter_valid=1;for(int k=0;k<8;k++)d.parameter_data[k]=(*v)[d.parameter_address*8+k];
    }
    if(d.source_request_valid&&(!stall||(n%7!=1&&n%7!=2))){
     unsigned t=d.source_tile,a=d.source_address;if(t>=f.size()||a!=source[t]++)return 6;
     d.source_valid=1;d.source_data=f[t].source[a]&1023;
    }
    if(d.origin_request_valid&&(!stall||(n%17!=4&&n%17!=5))){
     unsigned t=d.origin_tile;if(t>=f.size()||origin[t]++)return 7;
     d.origin_valid=1;d.origin_data=(f[t].origin[0]&65535)|((f[t].origin[1]&65535)<<16);
    }
    if(d.identity_request_valid&&(!stall||(n%11!=2&&n%11!=3))){
     unsigned t=d.identity_tile,a=d.identity_address;if(t>=f.size()||a!=id[t]++)return 8;
     d.identity_valid=1;for(int k=0;k<8;k++)d.identity_data[k]=f[t].id[a*8+k];
    }
    d.eval();
    bool accepted_req[4]={bool(d.parameter_valid),bool(d.source_valid),bool(d.origin_valid),bool(d.identity_valid)};
    for(int q=0;q<4;q++){req_held[q]=current_valid[q]&&!accepted_req[q];req_key[q]=current_key[q];}
    if(held){if(!d.result_valid||d.result_tile!=held_tile||d.result_address!=held_addr)return 9;for(int k=0;k<8;k++)if(d.result_data[k]!=held_data[k])return 10;}
    held=d.result_valid&&!d.result_ready;if(held){held_tile=d.result_tile;held_addr=d.result_address;for(int k=0;k<8;k++)held_data[k]=d.result_data[k];}
    for(unsigned c=0;c<2;c++){
     unsigned t=(d.context_tile>>(15*c))&32767;
     if((d.z_monitor_valid>>c)&1){
      unsigned a=((d.z_monitor_stripe>>c)&1)*40+((d.z_monitor_address>>(6*c))&63);
      if(t>=f.size()||a!=z[t]++)return 11;
      for(int k=0;k<8;k++)if(d.z_monitor_data[c*8+k]!=f[t].z[a*8+k]){std::cerr<<"Z mismatch "<<f[t].name<<" a="<<a<<" lane="<<k<<" got="<<d.z_monitor_data[c*8+k]<<" exp="<<f[t].z[a*8+k]<<"\n";return 12;}
     }
     if((d.d_monitor_valid>>c)&1){
      unsigned a=((d.z_monitor_stripe>>c)&1)*40+((d.d_monitor_address>>(6*c))&63);
      if(t>=f.size()||a>=80||dseen[t][a]++)return 13;dd[t]++;
      for(int k=0;k<8;k++)if(d.d_monitor_data[c*8+k]!=f[t].d[a*8+k]){std::cerr<<"D mismatch "<<f[t].name<<" a="<<a<<" lane="<<k<<"\n";return 14;}
     }
     unsigned cs=(d.debug_core_state>>(6*c))&63,rq=(d.debug_request>>(6*c))&63;
     if(cs)cstates[c][cs]++;
     if(rq&&!((d.debug_grant>>c)&1))denied[c][cs]++;
     if((d.debug_grant>>c)&1)for(int k=0;k<6;k++)if((rq>>k)&1)grants[k]++;
    }
    unsigned t=d.result_tile;
    auto check=[&](const Vec&gold,const uint32_t *data,unsigned a,unsigned width,const char*kind){for(unsigned k=0;k<width;k++)if(data[k]!=gold[a*width+k]){std::cerr<<kind<<" mismatch "<<f[t].name<<" row="<<a<<" lane="<<k<<" got="<<int32_t(data[k])<<" expected="<<int32_t(gold[a*width+k])<<"\n";throw std::runtime_error(kind);}};
    if(d.raw_monitor_valid){if(t>=f.size()||d.raw_monitor_address!=raw[t]++)return 15;check(f[t].raw,d.raw_monitor_data,d.raw_monitor_address,8,"raw");}
    if(d.j_monitor_valid){if(t>=f.size()||d.j_monitor_address!=j[t]++)return 16;check(f[t].j,d.j_monitor_data,d.j_monitor_address,8,"J");}
    if(d.wide_monitor_valid){if(t>=f.size()||d.wide_monitor_address!=wide[t]++)return 17;check(f[t].wide,d.wide_monitor_data,d.wide_monitor_address,16,"wide");}
    if(d.result_valid&&d.result_ready){
     if(t>=f.size()||d.result_address!=out[t]++)return 18;check(f[t].i24,d.result_data,d.result_address,8,"I24");
     if(bool(d.result_tile_last)!=(d.result_address==479)||bool(d.result_job_last)!=(t+1==f.size()&&d.result_address==479))return 19;
    }
    states[d.debug_state]++;tick();if(d.error)return 20;
    if(d.done){
     for(unsigned t=0;t<f.size();t++)if(raw[t]!=480||j[t]!=480||wide[t]!=480||out[t]!=480||z[t]!=80||dd[t]!=80||id[t]!=480||source[t]!=1536||origin[t]!=1)return 21;
     uint64_t ts=0,cs=0;for(auto v:states)ts+=v;for(auto &a:cstates)for(auto v:a)cs+=v;
     if(ts!=d.total_cycles||cs!=d.core_cycles||d.retired_tiles!=f.size())return 22;
     std::cout<<"{\"manifest\":\""<<argv[arg]<<"\",\"command\":"<<command<<",\"repeat\":"<<repeat<<",\"stall\":"<<stall<<",\"borrow\":"<<borrow<<",\"tiles\":"<<f.size()<<",\"raw_values\":"<<f.size()*3840<<",\"J_values\":"<<f.size()*3840<<",\"wide_values\":"<<f.size()*3840<<",\"I24_values\":"<<f.size()*3840<<",\"Z_fields\":"<<f.size()*1280<<",\"D_fields\":"<<f.size()*1280<<",\"go_cycles\":1";
     auto field=[&](const char*n,uint64_t v){std::cout<<",\""<<n<<"\":"<<v;};
     field("request_hold_checks",request_hold_checks);
     field("total_cycles",d.total_cycles);
     field("static_words",d.static_words);
     field("parameter_stalls",d.parameter_stalls);
     field("source_load_words",d.source_load_words);
     field("source_load_stalls",d.source_load_stalls);
     field("origin_words",d.origin_words);
     field("origin_stalls",d.origin_stalls);
     field("output_beats",d.output_beats);
     field("window_cycles",d.window_cycles);
     field("launch_cycles",d.launch_cycles);
     field("batches",d.batches);
     field("conflict_cycles",d.conflict_cycles);
     field("both_compute_cycles",d.both_compute_cycles);
     field("shared_source_grants",d.shared_source_grants);
     field("shared_weight_grants",d.shared_weight_grants);
     field("shared_z_grants",d.shared_z_grants);
     field("shared_psum_grants",d.shared_psum_grants);
     field("shared_alu_grants",d.shared_alu_grants);
     field("shared_wide_grants",d.shared_wide_grants);
     field("borrow_grants",d.borrow_grants);
     field("wide_conflict_cycles",d.wide_conflict_cycles);
     field("core_cycles",d.core_cycles);
     field("core_source_words",d.core_source_words);
     field("core_q1_words",d.core_q1_words);
     field("core_q2_words",d.core_q2_words);
     field("core_local_gathers",d.core_local_gathers);
     field("core_q1_issues",d.core_q1_issues);
     field("core_q2_issues",d.core_q2_issues);
     field("core_z_vector_reads",d.core_z_vector_reads);
     field("core_z_scalar_reads",d.core_z_scalar_reads);
     field("core_z_writes",d.core_z_writes);
     field("core_psum_reads",d.core_psum_reads);
     field("core_psum_writes",d.core_psum_writes);
     field("core_cache_writes",d.core_cache_writes);
     field("core_source_stalls",d.core_source_stalls);
     field("core_weight_stalls",d.core_weight_stalls);
     field("core_output_stalls",d.core_output_stalls);
     field("core_transform_issues",d.core_transform_issues);
     field("core_transform_reads",d.core_transform_reads);
     field("core_transform_writes",d.core_transform_writes);
     field("core_reconstruction_issues",d.core_reconstruction_issues);
     field("core_stripe_add_issues",d.core_stripe_add_issues);
     field("core_cache_reads",d.core_cache_reads);
     field("core_exact_halves",d.core_exact_halves);
     field("core_arbitration_stalls",d.core_arbitration_stalls);
     field("consumer_cycles",d.consumer_cycles);
     field("consumer_raw_words",d.consumer_raw_words);
     field("consumer_identity_words",d.consumer_identity_words);
     field("consumer_coefficient_words",d.consumer_coefficient_words);
     field("consumer_mul_issues",d.consumer_mul_issues);
     field("consumer_add_issues",d.consumer_add_issues);
     field("consumer_round_issues",d.consumer_round_issues);
     field("consumer_output_words",d.consumer_output_words);
     field("consumer_identity_stalls",d.consumer_identity_stalls);
     field("consumer_raw_wait_cycles",d.consumer_raw_wait_cycles);
     field("consumer_join_wait_cycles",d.consumer_join_wait_cycles);
     field("consumer_output_stalls",d.consumer_output_stalls);
     field("consumer_saturations",d.consumer_saturations);
     field("consumer_conversion_issues",d.consumer_conversion_issues);
     field("consumer_conversion_saturations",d.consumer_conversion_saturations);
     field("consumer_wide_waits",d.consumer_wide_waits);
     std::cout<<",\"top_states\":[";for(int i=0;i<16;i++)std::cout<<(i?",":"")<<states[i];std::cout<<"],\"context_states\":[";
     for(int c=0;c<2;c++){std::cout<<(c?",[":"[");for(int i=0;i<64;i++)std::cout<<(i?",":"")<<cstates[c][i];std::cout<<"]";}
     std::cout<<"],\"context_denied\":[";for(int c=0;c<2;c++){std::cout<<(c?",[":"[");for(int i=0;i<64;i++)std::cout<<(i?",":"")<<denied[c][i];std::cout<<"]";}
     std::cout<<"],\"observed_grants\":[";for(int i=0;i<6;i++)std::cout<<(i?",":"")<<grants[i];std::cout<<"]}\n";finished=true;break;
    }
   }
   if(!finished)return 23;
  }
 }
 d.final();return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 30;}}
