#include "Vwinograd_tile.h"
#include "verilated.h"
#include <array>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <iterator>
static std::vector<std::array<uint32_t,4> > readhex(const std::string &p){
 std::ifstream f(p);if(!f)throw std::runtime_error(p);std::vector<std::array<uint32_t,4> > v;std::string s;
 while(f>>s){std::array<uint32_t,4>a{};for(int j=0;j<4;j++)a[j]=std::stoul(s.substr(24-8*j,8),nullptr,16);v.push_back(a);}return v;
}
int main(int argc,char **argv){try{
 Verilated::commandArgs(argc,argv);if(argc!=4)throw std::runtime_error("fixture-directory kind stress");
 std::string dir=argv[1],kind=argv[2];bool stress=std::stoi(argv[3]);int mode=kind=="direct"?0:(kind=="masked_expanded"?2:1);
 auto meta=readhex(dir+"/"+kind+".meta"),input=readhex(dir+"/input.hex");
 std::ifstream bf(dir+"/"+kind+".bin",std::ios::binary);std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(bf)),std::istreambuf_iterator<char>());
 std::ifstream gf(dir+"/"+kind+".gold");std::vector<int32_t> gold;int64_t g;while(gf>>g)gold.push_back(int32_t(g));
 if(input.size()!=120||gold.size()!=3840||bytes.empty())throw std::runtime_error("fixture contract");
 Vwinograd_tile d;uint64_t cycles=0,states[19]={},requests=0,responses=0,hits=0,adds=0,halfzero=0,input_stall=0,output_stall=0,request_stall=0;
 auto edge=[&](bool count){d.clk=0;d.eval();if(count){cycles++;if(d.debug_state>18)throw std::runtime_error("state");states[d.debug_state]++;hits+=d.debug_cache_hit;adds+=d.debug_add;halfzero+=d.debug_half_zero;}d.clk=1;d.eval();d.clk=0;d.eval();};
 d.rst=1;d.start=0;d.meta_valid=0;d.in_valid=0;d.weight_ready=0;d.weight_valid=0;d.out_ready=0;edge(false);edge(false);d.rst=0;
 for(size_t i=0;i<meta.size();i++){d.meta_valid=1;d.meta_addr=i;for(int l=0;l<4;l++)d.meta_data[l]=meta[i][l];edge(true);}d.meta_valid=0;d.mode=mode;d.start=1;edge(true);d.start=0;
 size_t ip=0,op=0;bool pending=false,held=false;uint32_t address=0,held_addr=0;uint64_t due=0;std::array<uint32_t,8> hold{};
 while(cycles<5000000){
  d.in_valid=ip<input.size()&&(!stress||cycles%11!=3);if(ip<input.size())for(int l=0;l<4;l++)d.in_data[l]=input[ip][l];
  d.weight_ready=!stress||cycles%7!=2;d.out_ready=!stress||cycles%9>=3;
  d.weight_valid=pending&&cycles>=due;
  if(d.weight_valid)for(int l=0;l<8;l++){uint32_t word=0;for(int b=0;b<4;b++){size_t ix=size_t(address)*32+l*4+b;if(ix<bytes.size())word|=uint32_t(bytes[ix])<<(8*b);}d.weight_data[l]=word;}
  d.clk=0;d.eval();
  if(d.done){edge(true);break;}
  if(d.in_ready){if(d.in_valid)ip++;else input_stall++;}
  if(d.weight_req){if(d.weight_ready){if(pending)throw std::runtime_error("multiple outstanding CR");if(size_t(d.weight_addr)*32>=bytes.size())throw std::runtime_error("CR bounds");address=d.weight_addr;pending=true;due=cycles+(stress?4:1);requests++;}else request_stall++;}
  if(d.weight_valid){if(d.debug_state!=8)throw std::runtime_error("response outside RESPONSE");pending=false;responses++;}
  if(held){if(!d.out_valid||d.out_addr!=held_addr)throw std::runtime_error("output hold address");for(int l=0;l<8;l++)if(d.out_data[l]!=hold[l])throw std::runtime_error("output hold data");}
  if(d.out_valid){if(d.out_ready){if(d.out_addr!=op)throw std::runtime_error("output address");for(int l=0;l<8;l++){int32_t got=int32_t(d.out_data[l]);if(got!=gold[op*8+l]){std::cerr<<"mismatch beat="<<op<<" lane="<<l<<" got="<<got<<" gold="<<gold[op*8+l]<<"\n";throw std::runtime_error("integer mismatch");}}op++;held=false;}else{held=true;held_addr=d.out_addr;for(int l=0;l<8;l++)hold[l]=d.out_data[l];output_stall++;}}
  edge(true);
 }
 if(op!=480||ip!=120||pending||cycles>=5000000)throw std::runtime_error("completion contract");
 std::cout<<"{\"fixture\":\""<<dir.substr(dir.find_last_of('/')+1)<<"\",\"kind\":\""<<kind<<"\",\"stress\":"<<stress<<",\"cycles\":"<<cycles<<",\"configuration_cycles\":"<<meta.size()<<",\"coefficient_bytes\":"<<bytes.size()<<",\"weight_requests\":"<<requests<<",\"weight_responses\":"<<responses<<",\"cr_bytes\":"<<requests*32<<",\"cache_hits\":"<<hits<<",\"alu_vector_cycles\":"<<adds<<",\"half_zero_scan_cycles\":"<<halfzero<<",\"input_stall\":"<<input_stall<<",\"output_stall\":"<<output_stall<<",\"request_stall\":"<<request_stall<<",\"values_checked\":"<<op*8<<",\"mismatches\":0,\"state_cycles\":[";
 for(int i=0;i<19;i++)std::cout<<(i?",":"")<<states[i];std::cout<<"]}\n";
 d.final();return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
