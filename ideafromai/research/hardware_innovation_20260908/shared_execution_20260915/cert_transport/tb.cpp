#include "Vcert_transport.h"
#include "verilated.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
using Vec=std::array<int32_t,10>;
double sc_time_stamp(){return 0;}
struct Group {int h; Vec y{}; bool directed=false;};
static int64_t sx(uint64_t x,int n) {return int64_t(x<<(64-n))>>(64-n);}
static std::vector<uint64_t> hexfile(const std::string&p){
    std::ifstream f(p); if(!f) throw std::runtime_error(p);
    std::vector<uint64_t> v; std::string l; while(f>>l)v.push_back(std::stoull(l,nullptr,16));return v;
}
static int exponent(const Vec&v){uint32_t a=0;for(auto x:v)a=std::max(a,uint32_t(x<0?-int64_t(x):x));
    int e=0;while(a){++e;a>>=1;}return e;}
int main(int argc,char**argv){try{
    Verilated::commandArgs(argc,argv);
    std::string dir, outf;int bp=0,limit=20000,directed=0;
    for(int i=1;i<argc;i++){std::string a=argv[i];
        if(a.find("+dir=")==0)dir=a.substr(5);
        if(a.find("+out=")==0)outf=a.substr(5);
        if(a.find("+bp=")==0)bp=std::stoi(a.substr(4));
        if(a.find("+directed=")==0)directed=std::stoi(a.substr(10));
        if(a.find("+limit=")==0)limit=std::stoi(a.substr(7));}
    auto av=hexfile(dir+"/a.hex"),pnv=hexfile(dir+"/pn.hex");
    std::array<std::vector<uint64_t>,10> taus;
    for(int t=0;t<10;t++)taus[t]=hexfile(dir+"/tau_t"+std::to_string(t)+".hex");
    std::ifstream fs(dir+"/stim_fx.txt");if(!fs)throw std::runtime_error("stim");
    std::string line;std::vector<Group> groups;int plane=23;
    while(std::getline(fs,line)){
        if(line.empty())continue;
        if(line[0]=='G'){
            if(int(groups.size())==limit)break;
            std::istringstream s(line.substr(1));Group g;s>>g.h;groups.push_back(g);plane=23;
        }else if(line[0]=='B'){
            unsigned b=std::stoul(line.substr(2),nullptr,16);
            if(plane<0)throw std::runtime_error("extra plane");
            for(int j=0;j<10;j++)groups.back().y[j]|=((b>>j)&1)<<plane;
            plane--;
        }
    }
    for(auto&g:groups)for(auto&y:g.y)y=int32_t(sx(uint32_t(y),24));
    // Boundary groups use independent dot-product thresholds at V and V+1,
    // alternating comparison direction. No dynamic-BN interpretation claimed.
    const int edges[]={0,-1,1,-8388608,8388607,-4194304,4194304};
    for(int i=0;i<(directed?35:0);i++){Group g;g.h=i%int(taus[0].size());g.directed=true;
        for(int j=0;j<10;j++)g.y[j]=edges[(i+(i<7?0:j))%7];groups.push_back(g);}
    Vcert_transport d;
    auto edge=[&](){d.clk=0;d.eval();d.clk=1;d.eval();};
    d.rst_n=0;d.cmd_valid=0;d.req_ready=0;d.rsp_valid=0;d.out_ready=0;edge();edge();d.rst_n=1;
    std::ofstream out(outf);out<<"mode,groups,cycles,header_planes,source_reads,tau_reads,request_stall,response_wait,output_stall,dec_mismatch,plane_mismatch,exponent_mismatch\n";
    const char*names[]={"fx_full","fx_cert","bf_full","bf_cert"};
    for(int mode=0;mode<4;mode++){
        uint64_t cycles=0,payload=0,yr=0,tr=0,rqstall=0,rswait=0,ostall=0,dm=0,pm=0,em=0;
        for(size_t gi=0;gi<groups.size();gi++){
            const auto&g=groups[gi];std::array<uint64_t,10> tw;
            std::array<int64_t,10> dot{},thr{},vtop{};unsigned expected=0,locked=0;
            for(int t=0;t<10;t++){
                for(int j=0;j<10;j++)dot[t]+=sx(av[t*10+j],16)*int64_t(g.y[j]);
                tw[t]=taus[t][g.h];
                if(g.directed){int64_t th=dot[t]+((gi+t)&1);tw[t]=(uint64_t((t&1)==0)<<63)|(uint64_t(th)&((1ULL<<48)-1));}
                thr[t]=sx(tw[t],48);
                bool dec=(dot[t]>=thr[t])^!(tw[t]>>63);expected|=unsigned(dec)<<t;
                for(int j=0;j<10;j++)if(g.y[j]<0)vtop[t]-=sx(av[t*10+j],16);
            }
            int e=(mode&2)?exponent(g.y):23,refplanes=0;
            for(int m=std::max(e-1,0);m>=0;m--){
                for(int t=0;t<10;t++){
                    int64_t z=0;for(int j=0;j<10;j++)if((uint32_t(g.y[j])>>m)&1)z+=sx(av[t*10+j],16);
                    vtop[t]=vtop[t]*2+z;
                    int64_t lo=vtop[t]*(1LL<<m)+sx(pnv[2*t+1],48)*((1LL<<m)-1);
                    int64_t hi=vtop[t]*(1LL<<m)+sx(pnv[2*t],48)*((1LL<<m)-1);
                    if(lo>=thr[t]||hi<thr[t])locked|=1u<<t;
                }
                refplanes++;if((mode&1)&&locked==1023)break;
            }
            bool sent=false,pending=false,finished=false;uint64_t pendingdata=0;
            unsigned due=0,local=0,requests=0;uint16_t lastheld=0;bool held=false;
            while(!finished){
                if(local>1000)throw std::runtime_error("timeout");
                d.clk=0;d.cmd_valid=!sent;d.cmd_mode=mode;d.cmd_h=g.h;
                d.req_ready=(!bp || ((cycles%7)!=0 && (cycles%7)!=1));
                d.rsp_valid=pending && local>=due;d.rsp_data=pendingdata;
                d.out_ready=(!bp || (cycles%5)<3);d.eval();
                bool cmd=d.cmd_valid&&d.cmd_ready,req=d.req_valid&&d.req_ready;
                bool rsp=d.rsp_valid&&d.rsp_ready,ret=d.out_valid&&d.out_ready;
                if(held && (!d.out_valid || d.out_dec!=lastheld))throw std::runtime_error("unstable retirement");
                held=d.out_valid&&!d.out_ready;lastheld=d.out_dec;
                if(d.req_valid&&!d.req_ready)rqstall++;
                if(d.rsp_ready&&!d.rsp_valid)rswait++;
                if(d.out_valid&&!d.out_ready)ostall++;
                if(req){
                    if(pending)throw std::runtime_error("second outstanding");
                    if(d.req_h!=g.h)throw std::runtime_error("wrong h");
                    int idx=d.req_index;
                    if(d.req_tau){if(idx>=10)throw std::runtime_error("tau index");pendingdata=tw[idx];tr++;}
                    else {if(idx>=5)throw std::runtime_error("Y index");pendingdata=(uint32_t(g.y[2*idx])&0xffffffULL)|((uint64_t(uint32_t(g.y[2*idx+1]))&0xffffffULL)<<24);yr++;}
                    pending=true;due=local+1+(bp?1+(requests%3):0);requests++;
                }
                if(ret){
                    dm+=__builtin_popcount(unsigned(d.out_dec)^expected);
                    pm+=(d.out_planes!=refplanes);em+=(d.out_exp!=exponent(g.y));
                    payload+=1+d.out_planes;
                    if(requests!=15)throw std::runtime_error("request count");
                    finished=true;
                }
                d.clk=1;d.eval();if(cmd)sent=true;if(rsp)pending=false;
                local++;cycles++;
            }
        }
        out<<names[mode]<<","<<groups.size()<<","<<cycles<<","<<payload<<","<<yr<<","<<tr<<","<<rqstall<<","<<rswait<<","<<ostall<<","<<dm<<","<<pm<<","<<em<<"\n";
        printf("%s bp%d groups%zu cycles%llu payload%llu mismatch %llu/%llu/%llu\n",names[mode],bp,groups.size(),(unsigned long long)cycles,(unsigned long long)payload,(unsigned long long)dm,(unsigned long long)pm,(unsigned long long)em);
        if(dm||pm||em)return 2;
    }
    return 0;
}catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}
