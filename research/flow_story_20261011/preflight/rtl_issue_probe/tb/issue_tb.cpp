#include "Vrisk_group_issue.h"
#include "verilated.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Group {
    std::string name;
    unsigned id, base, stride, threshold;
    std::vector<unsigned> risk;
    bool stalls;
};
struct Runner {
    Vrisk_group_issue d;
    uint64_t cycle = 0, computes = 0, bypasses = 0, stalled = 0;
    unsigned aborted_computes = 0, aborted_bypasses = 0;
    unsigned completed = 0, reset_tests = 0;
    bool forced;
    std::ofstream trace, groups, aborted_trace;
    Runner(bool f, const std::string& dir): forced(f),
        trace(dir+"/issue_trace.csv"), groups(dir+"/group_results.csv"),
        aborted_trace(dir+"/aborted_issue_trace.csv") {
        trace << "cycle,group_id,member,window_id,channel,last\n";
        aborted_trace << "cycle,group_id,member,window_id,channel,last\n";
        groups << "name,group_id,members,max_risk,threshold,keep,start_cycle,last_cycle,elapsed_cycles,compute_count,bypass_count,backpressure_cycles\n";
        d.clk=0; d.rst=0; d.group_valid=0; d.risk_valid=0;
        d.compute_issue_ready=0; d.bypass_issue_ready=0;
        d.group_id=0; d.group_base=0; d.group_stride=0; d.group_four=0;
        d.group_threshold=0; d.risk_value=0;
    }
    void ck(bool v,const std::string& msg) {
        if(!v) throw std::runtime_error(msg+" at cycle "+std::to_string(cycle));
    }
    void settle() {d.clk=0;d.eval();}
    void edge() {d.clk=1;d.eval();++cycle;d.clk=0;d.eval();}
    void reset() {
        d.rst=1;settle();
        ck(!d.compute_issue_valid&&!d.bypass_issue_valid&&!d.group_ready&&!d.risk_ready,"reset suppresses handshakes");
        edge();edge();d.rst=0;d.group_valid=0;d.risk_valid=0;settle();
        ck(d.group_ready&&!d.risk_ready&&!d.compute_issue_valid&&!d.bypass_issue_valid,"reset clears in-flight transaction");
    }
    void command(const Group& g) {
        ck(g.risk.size()==1||g.risk.size()==4,"legal group length");
        ck(g.base+(g.risk.size()-1)*g.stride<=65535,"address overflow excluded by contract");
        d.group_valid=1;d.group_id=g.id;d.group_base=g.base;d.group_stride=g.stride;
        d.group_four=g.risk.size()==4;d.group_threshold=g.threshold;
        settle();ck(d.group_ready,"new group accepted in idle");edge();d.group_valid=0;
    }
    void run(const Group& g) {
        auto start=cycle;command(g);
        unsigned sent=0, emitted=0, c=0,b=0,waits=0;
        unsigned maximum=*std::max_element(g.risk.begin(),g.risk.end());
        bool keep=forced||maximum>=g.threshold;
        bool blocked=false;unsigned old_gid=0,old_win=0,old_member=0,old_last=0,old_cv=0,old_bv=0;
        while(emitted<g.risk.size()) {
            ck(cycle-start<200,"transaction timeout");
            // Both sources and sinks may independently withhold valid/ready.
            bool source_on=!g.stalls||(cycle%4!=1);
            d.risk_valid=!forced&&sent<g.risk.size()&&source_on;
            d.risk_value=sent<g.risk.size()?g.risk[sent]:0;
            d.compute_issue_ready=!g.stalls||(cycle%7!=2&&cycle%7!=3&&cycle%7!=4);
            d.bypass_issue_ready=!g.stalls||(cycle%5!=1&&cycle%5!=2);
            settle();
            ck(!(d.compute_issue_valid&&d.bypass_issue_valid),"exclusive channels");
            ck(!d.group_ready,"one active group only");
            if(blocked) ck(d.issue_group_id==old_gid&&d.issue_window_id==old_win&&d.issue_member==old_member&&d.issue_last==old_last&&d.compute_issue_valid==old_cv&&d.bypass_issue_valid==old_bv,"stable stalled descriptor");
            if(forced) ck(!d.risk_ready,"forced-full consumes no risk");
            if(d.risk_valid&&d.risk_ready) ++sent;
            bool valid=d.compute_issue_valid||d.bypass_issue_valid;
            if(valid) {
                ck(forced||sent==g.risk.size(),"issue requires all group risks");
                ck(bool(d.compute_issue_valid)==keep,"group decision differs from reference maximum");
                ck(d.issue_group_id==g.id,"group id mismatch");
                ck(d.issue_member==emitted,"member order mismatch");
                ck(d.issue_window_id==g.base+emitted*g.stride,"nonadjacent descriptor address mismatch");
                ck(bool(d.issue_last)==(emitted+1==g.risk.size()),"last flag mismatch");
            }
            bool ready=keep?d.compute_issue_ready:d.bypass_issue_ready;
            blocked=valid&&!ready;
            if(blocked) {
                ++waits;++stalled;old_gid=d.issue_group_id;old_win=d.issue_window_id;
                old_member=d.issue_member;old_last=d.issue_last;old_cv=d.compute_issue_valid;old_bv=d.bypass_issue_valid;
            }
            if(valid&&ready) {
                trace<<cycle<<','<<g.id<<','<<emitted<<','<<d.issue_window_id<<','<<(keep?"compute":"bypass")<<','<<unsigned(d.issue_last)<<'\n';
                ++emitted;if(keep){++c;++computes;}else{++b;++bypasses;}
            }
            edge();
        }
        d.risk_valid=0;settle();ck(d.group_ready,"last issue retires transaction");
        ck(c+b==g.risk.size()&&c==(keep?g.risk.size():0),"issue counts match reference");
        if(!g.stalls) ck(cycle-start==1+g.risk.size()+(forced?0:g.risk.size()),"no-stall cycle formula");
        groups<<g.name<<','<<g.id<<','<<g.risk.size()<<','<<maximum<<','<<g.threshold<<','<<keep<<','<<start<<','<<cycle-1<<','<<cycle-start<<','<<c<<','<<b<<','<<waits<<'\n';
        ++completed;
    }
    void reset_midflight() {
        Group g={"aborted",65500,65000,100,100,{0,255,0,0},false};
        if(!forced) {
            command(g);d.risk_valid=1;d.risk_value=20;settle();ck(d.risk_ready,"partial risk collection");edge();
            reset();++reset_tests;
        }
        command(g);
        if(!forced) for(unsigned r:g.risk){d.risk_valid=1;d.risk_value=r;settle();ck(d.risk_ready,"risk accepted before reset issue test");edge();}
        d.risk_valid=0;d.compute_issue_ready=0;d.bypass_issue_ready=0;settle();
        ck(d.compute_issue_valid,"pending kept descriptor");edge();
        reset();++reset_tests;
        if(!forced) {
            g.risk={0,1,2,3};command(g);
            for(unsigned risk:g.risk){d.risk_valid=1;d.risk_value=risk;settle();ck(d.risk_ready,"risk accepted before bypass reset");edge();}
            d.risk_valid=0;d.compute_issue_ready=0;d.bypass_issue_ready=0;settle();
            ck(d.bypass_issue_valid,"pending bypass descriptor");edge();
            reset();++reset_tests;
        }
        // An accepted prefix is externally visible and is never rolled back.
        for(unsigned prefix=1;prefix<=2;++prefix) {
            g.id=65200+prefix;
            g.risk=prefix==1?std::vector<unsigned>{0,255,0,0}:std::vector<unsigned>{0,1,2,3};
            const bool keep=forced||prefix==1;
            command(g);
            if(!forced) for(unsigned risk:g.risk){d.risk_valid=1;d.risk_value=risk;settle();ck(d.risk_ready,"risk accepted before partial retirement");edge();}
            d.risk_valid=0;d.compute_issue_ready=1;d.bypass_issue_ready=1;
            for(unsigned k=0;k<prefix;++k) {
                settle();
                ck(bool(d.compute_issue_valid)==keep&&bool(d.bypass_issue_valid)==!keep,"partial retirement channel");
                ck(d.issue_group_id==g.id&&d.issue_member==k&&d.issue_window_id==g.base+k*g.stride&&!d.issue_last,"partial retirement descriptor");
                aborted_trace<<cycle<<','<<g.id<<','<<k<<','<<d.issue_window_id<<','<<(keep?"compute":"bypass")<<",0\n";
                if(keep){++computes;++aborted_computes;}else{++bypasses;++aborted_bypasses;}
                edge();
            }
            d.compute_issue_ready=0;d.bypass_issue_ready=0;settle();
            ck(d.issue_member==prefix&&(d.compute_issue_valid||d.bypass_issue_valid),"remaining descriptor stalled before reset");
            edge();reset();++reset_tests;
        }
        run({"after_reset",65535,65535,0,0,{0},false});
    }
};
int main(int argc,char**argv) {
    Verilated::commandArgs(argc,argv);
    try {
        bool forced=argc>1&&std::string(argv[1])=="forced";
        std::string out=argc>2?argv[2]:".";
        Runner r(forced,out);r.reset();
        std::vector<Group> cases={
            {"all_keep_four",1,0,63,100,{200,200,200,200},false},
            {"all_bypass_four",2,7,63,100,{0,10,99,20},false},
            {"mixed_member_group_keep",3,4,5,100,{0,0,101,0},false},
            {"threshold_equal_four",4,10,19,100,{0,100,0,0},false},
            {"threshold_zero",5,0,1,0,{0,0,0,0},false},
            {"threshold_255_equal",6,0,17,255,{0,254,255,1},false},
            {"threshold_255_below",7,0,17,255,{0,254,254,1},false},
            {"mlp_keep",8,4095,0,128,{128},false},
            {"mlp_bypass",9,4096,0,128,{127},false},
            {"address_boundary",10,65532,1,1,{0,0,0,1},false},
            {"compute_backpressure",11,100,63,10,{0,10,0,0},true},
            {"bypass_backpressure",12,200,63,200,{0,10,0,0},true}
        };
        for(const auto& g:cases)r.run(g);
        uint32_t seed=0x6110cafe;
        for(unsigned i=0;i<128;++i) {
            auto next=[&](){seed=1664525u*seed+1013904223u;return seed;};
            unsigned n=(next()&4)?4:1,th=(next()>>8)&255;
            std::vector<unsigned> risks;for(unsigned k=0;k<n;++k)risks.push_back((next()>>16)&255);
            r.run({"deterministic_stress_"+std::to_string(i),100+i,next()%1000,1+next()%127,th,risks,true});
        }
        r.reset_midflight();
        std::ofstream summary(out+"/simulation_summary.json");
        summary<<"{\n  \"passed\": true,\n  \"forced_full\": "<<(forced?"true":"false")<<",\n  \"completed_groups\": "<<r.completed<<",\n  \"compute_issues\": "<<r.computes<<",\n  \"bypass_issues\": "<<r.bypasses<<",\n  \"completed_group_compute_issues\": "<<r.computes-r.aborted_computes<<",\n  \"completed_group_bypass_issues\": "<<r.bypasses-r.aborted_bypasses<<",\n  \"accepted_compute_in_aborted_groups\": "<<r.aborted_computes<<",\n  \"accepted_bypass_in_aborted_groups\": "<<r.aborted_bypasses<<",\n  \"backpressure_cycles_completed_groups\": "<<r.stalled<<",\n  \"reset_midflight_tests\": "<<r.reset_tests<<",\n  \"total_testbench_cycles\": "<<r.cycle<<"\n}\n";
        std::cout<<"PASS groups="<<r.completed<<" compute="<<r.computes<<" bypass="<<r.bypasses<<" cycles="<<r.cycle<<'\n';
        return 0;
    } catch(const std::exception& e) {std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
