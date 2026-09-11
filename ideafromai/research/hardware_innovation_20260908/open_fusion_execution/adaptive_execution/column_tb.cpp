#include "Vcolumn_relation_matcher.h"
#include "verilated.h"
#include <array>
#include <bitset>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

constexpr int T=20, K=864;
struct Column {unsigned k,mask;};
struct Case {std::string name,kind; int positions; std::vector<Column> columns;};
struct Gold {std::array<std::bitset<K>,T> rows; std::array<int,T> parent,count;
             std::vector<unsigned> residual;};
static void require(bool condition,const std::string& message) {
    if(!condition) throw std::runtime_error(message);
}
static Gold reference(const Case& test) {
    Gold g{}; std::bitset<K> supplied;
    for(const auto& col:test.columns) {
        require(col.k<K && !(col.mask>>T) && !supplied[col.k],"bad/distinct-K input");
        supplied.set(col.k);
        for(int row=0;row<T;++row) if(col.mask&(1u<<row)) g.rows[row].set(col.k);
    }
    for(int child=0;child<T;++child) {
        g.count[child]=g.rows[child].count(); g.parent[child]=-1; int best=0;
        // Independent row-bitset inclusion, not the RTL's column elimination.
        // The original ProSparsity kernel disables reuse for nnz<2, excludes
        // empty parents, and uses increasing original row index for ties.
        if(g.count[child]<2) continue;
        for(int parent=0;parent<T;++parent) {
            const int count=g.rows[parent].count();
            if(parent==child || count<=best) continue;
            if(count==g.count[child] && parent>=child) continue;
            if((g.rows[parent]&~g.rows[child]).none()) {
                best=count; g.parent[child]=parent;
            }
        }
    }
    std::array<std::bitset<K>,T> residual_rows;
    for(const auto& col:test.columns) {
        unsigned mask=0;
        for(int child=0;child<T;++child) {
            const bool keep=g.rows[child][col.k] &&
                (g.parent[child]<0 || !g.rows[g.parent[child]][col.k]);
            if(keep) { mask|=1u<<child; residual_rows[child].set(col.k); }
        }
        g.residual.push_back(mask);
    }
    for(int child=0;child<T;++child) {
        auto reconstructed=residual_rows[child];
        if(g.parent[child]>=0) reconstructed|=g.rows[g.parent[child]];
        require(reconstructed==g.rows[child],"gold row reconstruction failed");
    }
    return g;
}
struct Report {
    uint64_t cycles=0,start=0,fill=0,parent=0,replay=0;
    unsigned columns=0,parents=0,residual_in=0,residual_out=0,has_parent=0;
    unsigned parent_stall=0,residual_stall=0,column_bubbles=0,replay_bubbles=0;
    unsigned rejected_start=0;
};
class Sim {
public:
    Vcolumn_relation_matcher d;
    uint64_t cycles=0;
    void eval() { d.clk=0; d.eval(); }
    void tick() { eval(); d.clk=1; d.eval(); ++cycles; d.clk=0; d.eval(); }
    void defaults() {
        d.start=0; d.column_valid=0; d.column_mask=0; d.finish=0;
        d.parent_ready=0; d.residual_valid=0; d.residual_mask=0; d.residual_out_ready=0;
    }
    Sim() { defaults(); d.rst_n=0; tick();tick();d.rst_n=1;tick(); }
    Report run(const Case& test,const Gold& g,bool stress,bool last_finish) {
        defaults(); eval();
        require(d.start_ready && !d.parent_valid && !d.residual_out_valid,"task clear boundary not ready");
        const uint64_t begin=cycles;
        Report r; d.start=1;tick();d.start=0;r.start=1;
        const uint64_t fill_begin=cycles;
        size_t index=0;unsigned step=0;
        while(index<test.columns.size()) {
            require(++step<10000,"fill timeout");
            const bool issue=!stress || step%5!=1;
            d.column_valid=issue;d.column_mask=test.columns[index].mask;
            d.finish=last_finish && index+1==test.columns.size() && issue;
            eval();
            require(d.column_ready,"FILL unexpectedly not ready");
            if(issue && d.column_ready) {++index;++r.columns;} else ++r.column_bubbles;
            tick();
        }
        d.column_valid=0;
        if(!last_finish || test.columns.empty()) {d.finish=1;tick();}
        d.finish=0;r.fill=cycles-fill_begin;
        const uint64_t parent_begin=cycles;
        unsigned child=0;step=0;
        bool held=false;std::tuple<unsigned,unsigned,unsigned,unsigned,unsigned> previous;
        while(child<T) {
            require(++step<10000,"parent timeout");
            d.parent_ready=!stress || !(step%7==1 || step%7==2 || step%7==5);
            eval();
            require(d.parent_valid,"missing parent output");
            auto now=std::make_tuple(unsigned(d.child_index),unsigned(d.has_parent),
                unsigned(d.parent_index),unsigned(d.child_popcount),unsigned(d.parent_last));
            if(held) require(now==previous,"parent changed while stalled");
            require(d.child_index==child && d.child_popcount==g.count[child],"child/count mismatch");
            require(bool(d.has_parent)==(g.parent[child]>=0),"has_parent mismatch");
            if(g.parent[child]>=0) require(d.parent_index==unsigned(g.parent[child]),"parent choice mismatch");
            else require(d.parent_index==0,"invalid parent index was not zero");
            require(bool(d.parent_last)==(child==T-1),"parent_last mismatch");
            held=!d.parent_ready;previous=now;
            if(d.parent_ready) {++r.parents;r.has_parent+=d.has_parent;++child;}
            else ++r.parent_stall;
            tick();
        }
        d.parent_ready=0;r.parent=cycles-parent_begin;
        const uint64_t replay_begin=cycles;
        index=0;step=0;std::deque<unsigned> expected;bool held_residual=false;
        unsigned previous_mask=0,last_holds=0;
        while(index<test.columns.size() || !expected.empty() || d.residual_out_valid) {
            require(++step<10000,"replay timeout");
            d.residual_valid=index<test.columns.size() && (!stress || step%4!=1);
            d.residual_mask=index<test.columns.size()?test.columns[index].mask:0;
            const bool last_block=stress && index==test.columns.size() && !expected.empty() && last_holds<2;
            d.residual_out_ready=last_block?0:(!stress || !(step%6==1 || step%6==2));
            d.start=last_block;
            eval();
            if(last_block) {require(!d.start_ready,"accepted new start over pending residual");++last_holds;++r.rejected_start;}
            if(held_residual) require(d.residual_out_valid && d.residual_out_mask==previous_mask,
                                      "residual changed/dropped while stalled");
            held_residual=d.residual_out_valid && !d.residual_out_ready;
            previous_mask=d.residual_out_mask;
            if(d.residual_out_valid) {
                require(!expected.empty(),"unexpected residual output");
                require(d.residual_out_mask==expected.front(),"residual mask mismatch");
                if(d.residual_out_ready) {expected.pop_front();++r.residual_out;}
                else ++r.residual_stall;
            }
            if(d.residual_valid && d.residual_ready) {
                expected.push_back(g.residual[index]);++index;++r.residual_in;
            } else if(!d.residual_valid && index<test.columns.size()) ++r.replay_bubbles;
            tick();
        }
        d.start=0;d.residual_valid=0;d.residual_out_ready=0;eval();
        require(d.start_ready && !d.parent_valid && !d.residual_out_valid,"end task did not drain");
        require(r.residual_in==test.columns.size() && r.residual_out==test.columns.size(),"residual lost/duplicated");
        r.replay=cycles-replay_begin;r.cycles=cycles-begin;
        return r;
    }
};
int main(int argc,char**argv) {
    try {
        Verilated::commandArgs(argc,argv);
        require(argc==3,"usage: tb cases.txt results.json");
        std::ifstream input(argv[1]);size_t n;input>>n;
        std::vector<Case> cases;
        for(size_t i=0;i<n;++i) {
            Case c;size_t cols;input>>c.name>>c.kind>>c.positions>>cols;
            for(size_t j=0;j<cols;++j) {Column x;input>>x.k>>x.mask;c.columns.push_back(x);}
            cases.push_back(c);
        }
        require(bool(input),"case parse failed");
        Sim sim;std::ofstream out(argv[2]);out<<"{\"complete\":true,\"cases\":[";
        bool comma=false;
        for(size_t i=0;i<cases.size();++i) {
            const auto gold=reference(cases[i]);
            for(int profile=0;profile<2;++profile) {
                const bool last_finish=!cases[i].columns.empty() && ((i+profile)%2==0);
                auto r=sim.run(cases[i],gold,profile,last_finish);
                if(comma)out<<',';comma=true;
                out<<"{\"name\":\""<<cases[i].name<<"\",\"kind\":\""<<cases[i].kind
                   <<"\",\"positions\":"<<cases[i].positions<<",\"profile\":\""<<(profile?"bubbles_backpressure":"ready")
                   <<"\",\"finish_with_last_column\":"<<(last_finish?"true":"false")
                   <<",\"leaf_cycles\":"<<r.cycles<<",\"start_cycles\":"<<r.start
                   <<",\"fill_cycles\":"<<r.fill<<",\"parent_cycles\":"<<r.parent<<",\"replay_cycles\":"<<r.replay
                   <<",\"accepted_columns\":"<<r.columns<<",\"parent_comparisons\":"<<r.parents
                   <<",\"selected_parents\":"<<r.has_parent<<",\"accepted_replay_columns\":"<<r.residual_in
                   <<",\"residual_comparisons\":"<<r.residual_out<<",\"parent_stall_cycles\":"<<r.parent_stall
                   <<",\"residual_stall_cycles\":"<<r.residual_stall<<",\"column_bubble_cycles\":"<<r.column_bubbles
                   <<",\"replay_bubble_cycles\":"<<r.replay_bubbles<<",\"busy_start_rejections\":"<<r.rejected_start
                   <<",\"mismatches\":0}";
            }
        }
        out<<"]}\n";out.close();
        std::cout<<"COLUMN_RTL_PASS "<<cases.size()<<" input_cases "<<cases.size()*2<<" profile_runs\n";
    }catch(const std::exception& e){std::cerr<<"COLUMN_RTL_FAIL "<<e.what()<<'\n';return 1;}
}
