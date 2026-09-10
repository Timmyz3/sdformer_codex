#include "Vgp_temporal_pe.h"
#include "verilated.h"
#include <array>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool x, const std::string& msg) {
    if (!x) throw std::runtime_error(msg);
}
static uint8_t byte(std::istream& in) {
    char x; require(bool(in.get(x)), "truncated input"); return uint8_t(x);
}
static uint16_t u16(std::istream& in) {
    uint16_t x=byte(in); return x | uint16_t(byte(in))<<8;
}
static uint32_t u32(std::istream& in) {
    uint32_t x=0; for(int i=0;i<4;i++) x |= uint32_t(byte(in))<<(8*i); return x;
}
static int signed_bits(unsigned x, unsigned bits) {
    return (x & (1U<<(bits-1))) ? int(x)-int(1U<<bits) : int(x);
}
struct Packet {
    unsigned count; bool last;
    std::array<uint8_t,32> codes;
    std::array<int8_t,8> weight;
};
struct Case {
    std::string name;
    bool p4f2,real; unsigned rank;
    std::array<uint8_t,8> decode;
    std::vector<Packet> packets;
    std::array<int32_t,56> expected;
    std::array<uint8_t,56> valid;
};
struct PacketReference {
    std::array<uint8_t,56> members{};
    std::array<int32_t,56> delta{};
};
static PacketReference packet_reference(const Case& c,const Packet& packet) {
    PacketReference ref;
    unsigned P=c.p4f2?4:8,F=c.p4f2?2:1;
    for(unsigned f=0;f<F;f++) for(unsigned p=0;p<P;p++)
        for(unsigned r=0;r<c.rank;r++) for(unsigned s=0;s<packet.count;s++) {
            unsigned a=(f*P+p)*7+r;
            if (((c.decode[packet.codes[s*8+p]]>>r)&1U) && packet.weight[f*4+s]) {
                ref.delta[a] += packet.weight[f*4+s];
                ref.members[a] |= uint8_t(1U<<s);
            }
        }
    return ref;
}
static std::vector<Case> read_cases(const char* path) {
    std::ifstream in(path,std::ios::binary); require(bool(in),"cannot read cases");
    require(u32(in)==0x31545047U,"wrong case magic");
    unsigned n=u32(in); std::vector<Case> cases;
    for(unsigned i=0;i<n;i++) {
        Case c; unsigned len=u16(in); while(len--) c.name.push_back(char(byte(in)));
        c.p4f2=byte(in);c.rank=byte(in);c.real=byte(in);
        for(auto& x:c.decode)x=byte(in);
        unsigned count=u32(in);
        while(count--) {
            Packet p;p.count=byte(in);p.last=byte(in);
            for(auto& x:p.codes)x=byte(in);
            for(auto& x:p.weight)x=int8_t(byte(in));
            c.packets.push_back(p);
        }
        for(auto& x:c.expected)x=int32_t(u32(in));
        for(auto& x:c.valid)x=byte(in);
        // Second direct integer reference reconstructed from packet content;
        // NumPy generated expected values from the unpacketized dense matrices.
        std::array<int64_t,56> sums{};
        std::array<bool,56> valid{};
        for(const auto& p:c.packets) {
            auto ref=packet_reference(c,p);
            for(unsigned a=0;a<56;a++) {
                sums[a]+=ref.delta[a];valid[a]=valid[a]||ref.members[a];
                require(sums[a]>=-16384 && sums[a]<=16383,c.name+" prefix outside S15");
            }
        }
        for(unsigned a=0;a<56;a++)
            require(sums[a]==c.expected[a] && valid[a]==bool(c.valid[a]),c.name+" C++/NumPy mismatch");
        cases.push_back(c);
    }
    return cases;
}

struct Sim {
    Vgp_temporal_pe d;
    uint64_t cycle=0;
    void settle(){d.clk=0;d.eval();}
    void tick(){settle();d.clk=1;d.eval();cycle++;}
    void idle_inputs(){
        d.start_valid=0;d.packet_valid=0;d.rd_valid=0;d.rsp_ready=0;
        d.consumer_done_valid=0;d.cancel=0;
    }
    void initialize(){idle_inputs();d.rst_n=0;tick();d.rst_n=1;tick();}
    void configure(const Case& c,bool reduce,unsigned tag){
        idle_inputs();settle();require(d.start_ready && !d.busy,"new task before release");
        d.start_p4f2=c.p4f2;d.start_reduce=reduce;d.start_rank=c.rank;d.start_tag=tag;
        uint64_t table=0;for(unsigned k=0;k<8;k++)table|=uint64_t(c.decode[k])<<(k*7);
        d.start_decode=table;d.start_valid=1;tick();d.start_valid=0;
    }
    void pack(const Packet& p){
        for(unsigned w=0;w<3;w++)d.packet_codes[w]=0;
        for(unsigned i=0;i<32;i++)for(unsigned b=0;b<3;b++)
            if((p.codes[i]>>b)&1U)d.packet_codes[(i*3+b)/32]|=1U<<((i*3+b)%32);
        uint64_t weights=0;for(unsigned i=0;i<8;i++)weights|=uint64_t(uint8_t(p.weight[i]))<<(i*8);
        d.packet_weights=weights;d.packet_sources=p.count;d.packet_last=p.last;
    }
};
struct Count {
    uint64_t task_beats=0,issues=0,commits=0,members=0,packet_accepts=0;
    uint64_t source_gaps=0,input_backpressure=0,output_backpressure=0,read_responses=0;
};
static void check_empty(const PacketReference& ref,const std::string& name){
    for(auto x:ref.members)require(!x,name+" NR4 window retired with pending members");
}

static Count accumulate(Sim& sim,const Case& c,bool reduce,bool stress,unsigned tag){
    Count count;auto begin=sim.cycle;
    sim.configure(c,reduce,tag);
    size_t sent=0;bool offering=false,active=false;
    PacketReference current;int previous_commit=-1;
    uint64_t expected_events=0,expected_commits=0;
    for(const auto& p:c.packets){auto r=packet_reference(c,p);
        for(auto mask:r.members){expected_events+=__builtin_popcount(mask);expected_commits+=bool(mask);}}
    sim.settle();
    while(!sim.d.state_complete){
        require(sim.cycle-begin<1000000,c.name+" execution timeout");
        if(!offering && sent<c.packets.size() && (!stress || (sim.cycle+sent)%5>=2)){
            sim.pack(c.packets[sent]);offering=true;
        }
        sim.d.packet_valid=offering;
        // Probe the consumer boundary while computing; no state can leak early.
        sim.d.rd_valid=1;sim.d.rd_address=0;sim.d.rsp_ready=1;sim.settle();
        require(!sim.d.rd_ready && !sim.d.rsp_valid,c.name+" early state read");
        require(!sim.d.start_ready,c.name+" task start exposed during execution");
        if(active && sim.d.packet_ready){check_empty(current,c.name);active=false;}
        bool accept=sim.d.packet_valid && sim.d.packet_ready;
        if(accept){
            require(!active,c.name+" second packet in flight");
            current=packet_reference(c,c.packets[sent]);active=true;previous_commit=-1;
            sent++;offering=false;count.packet_accepts++;
        }
        if(sim.d.packet_ready && !sim.d.packet_valid)count.source_gaps++;
        if(sim.d.packet_valid && !sim.d.packet_ready)count.input_backpressure++;
        if(sim.d.execute_valid){
            require(active,c.name+" execution with no accepted NR4 packet");
            unsigned a=sim.d.commit_address,mask=sim.d.execute_members;
            require(a<56 && mask && !(mask & ~current.members[a]),c.name+" duplicate/wrong members");
            if(reduce)require(mask==current.members[a],c.name+" incomplete member reduction");
            else require((mask&(mask-1))==0,c.name+" scalar processed multiple members");
            current.members[a]&=uint8_t(~mask);
            count.issues++;count.members+=__builtin_popcount(mask);
            if(sim.d.commit_valid){
                require(!current.members[a],c.name+" partial destination committed early");
                require(signed_bits(sim.d.commit_delta,10)==current.delta[a],c.name+" wrong signed delta");
                require(int(a)>previous_commit,c.name+" destination/F order or duplicate commit");
                previous_commit=int(a);count.commits++;
            }
        }else require(!sim.d.commit_valid,c.name+" unissued commit");
        sim.tick();sim.settle();
        require(!sim.d.overflow,c.name+" S15 overflow");
    }
    sim.d.packet_valid=0;sim.d.rd_valid=0;sim.d.rsp_ready=0;
    require(sent==c.packets.size(),c.name+" last completed before all packets");
    check_empty(current,c.name);
    require(count.commits==expected_commits && count.members==expected_events,c.name+" event accounting mismatch");
    require(count.issues==(reduce?expected_commits:expected_events),c.name+" issue accounting mismatch");
    count.task_beats=sim.cycle-begin;
    return count;
}

static void read_and_release(Sim& sim,const Case& c,bool stress,unsigned tag,Count& count){
    // Let completed state live without a reader, while upstream offers a packet.
    sim.d.packet_valid=1;
    for(unsigned i=0;i<5;i++){
        sim.settle();require(sim.d.state_complete && sim.d.busy && !sim.d.packet_ready
                            && !sim.d.start_ready,c.name+" completed state reused without consumer");
        sim.tick();
    }
    sim.d.packet_valid=0;
    std::vector<unsigned> addresses;
    for(unsigned a=0;a<56;a++)addresses.push_back((17*a)%56);
    for(unsigned a=0;a<56;a++)addresses.push_back(55-a); // non-destructive reread
    size_t issued=0,received=0;std::deque<unsigned> expected;
    bool offering=false,held=false;
    unsigned held_addr=0,held_value=0,held_present=0,held_tag=0;
    auto begin=sim.cycle;
    while(received<addresses.size()){
        require(sim.cycle-begin<10000,c.name+" read timeout");
        if(!offering && issued<addresses.size() && (!stress || (sim.cycle+issued)%4!=0)){
            sim.d.rd_address=addresses[issued];offering=true;
        }
        sim.d.rd_valid=offering;
        sim.d.rsp_ready=!stress || (sim.cycle+received)%7>=3;
        sim.settle();
        // A release offered with an outstanding response must be blocked.
        sim.d.consumer_done_valid=sim.d.rsp_valid;sim.settle();
        if(sim.d.rsp_valid)require(!sim.d.consumer_done_ready,c.name+" release with outstanding response");
        if(held)require(sim.d.rsp_valid && sim.d.rsp_address==held_addr
                        && sim.d.rsp_value==held_value && sim.d.rsp_present==held_present
                        && sim.d.rsp_tag==held_tag,c.name+" response changed under backpressure");
        held=sim.d.rsp_valid && !sim.d.rsp_ready;
        if(held){held_addr=sim.d.rsp_address;held_value=sim.d.rsp_value;
            held_present=sim.d.rsp_present;held_tag=sim.d.rsp_tag;count.output_backpressure++;}
        if(sim.d.rd_valid && sim.d.rd_ready){expected.push_back(addresses[issued]);issued++;offering=false;}
        if(sim.d.rsp_valid && sim.d.rsp_ready){
            require(!expected.empty(),c.name+" unsolicited response");unsigned a=expected.front();expected.pop_front();
            require(sim.d.rsp_address==a && sim.d.rsp_tag==tag,c.name+" wrong read identity");
            require(signed_bits(sim.d.rsp_value,15)==c.expected[a],c.name+" numeric S mismatch at "+std::to_string(a));
            require(sim.d.rsp_present==c.valid[a],c.name+" valid-zero/empty mismatch");
            received++;count.read_responses++;
        }
        sim.tick();
    }
    sim.d.rd_valid=0;sim.d.consumer_done_valid=0;sim.d.rsp_ready=1;sim.settle();
    require(expected.empty() && !sim.d.rsp_valid && sim.d.state_complete,c.name+" read drained task unexpectedly");
    sim.d.consumer_done_valid=1;sim.settle();
    require(sim.d.consumer_done_ready,c.name+" consumer cannot release");sim.tick();
    sim.d.consumer_done_valid=0;sim.settle();require(sim.d.start_ready && !sim.d.busy,c.name+" task not released");
}

static void cancel_trials(Sim& sim,const Case& c){
    for(unsigned location=0;location<3;location++){
        if(location==2){(void)accumulate(sim,c,false,false,0xff00+location);
            sim.d.rd_valid=1;sim.d.rd_address=0;sim.d.rsp_ready=0;sim.tick();sim.d.rd_valid=0;
            sim.settle();require(sim.d.rsp_valid,"cancel test lacks held read response");
        }else{
            sim.configure(c,false,0xff00+location);
            if(location==1){sim.pack(c.packets.front());sim.d.packet_valid=1;sim.tick();
                sim.d.packet_valid=0;sim.settle();require(sim.d.execute_valid,"cancel test lacks execution");
                sim.tick(); // first member of a four-member scalar reduction
            }
        }
        sim.d.cancel=1;sim.d.packet_valid=1;sim.d.start_valid=1;sim.settle();
        require(!sim.d.packet_ready && !sim.d.start_ready && !sim.d.execute_valid
                && !sim.d.commit_valid && !sim.d.rsp_valid,"cancel leaked handshake");
        sim.tick();sim.idle_inputs();sim.settle();
        require(sim.d.start_ready && !sim.d.busy && !sim.d.rsp_valid,"cancel did not release context");
        auto result=accumulate(sim,c,true,true,0xfe00+location);
        read_and_release(sim,c,true,0xfe00+location,result);
    }
}

int main(int argc,char** argv){
    try{
        require(argc==3,"usage: gp_temporal cases.bin results.tsv");
        Verilated::commandArgs(argc,argv);auto cases=read_cases(argv[1]);Sim sim;sim.initialize();
        std::ofstream out(argv[2]);require(bool(out),"cannot write results");
        out<<"case\treal\tp4f2\treduce\tstress\tstart_to_complete_beats\tissue_beats\tcommits\tscalar_members\tpackets\tsource_gap_beats\tinput_backpressure_beats\toutput_backpressure_beats\tread_responses\n";
        unsigned tasks=0;
        for(const auto& c:cases)for(bool reduce:{false,true})for(bool stress:{false,true}){
            unsigned tag=tasks+1;auto count=accumulate(sim,c,reduce,stress,tag);
            read_and_release(sim,c,stress,tag,count);
            out<<c.name<<'\t'<<c.real<<'\t'<<c.p4f2<<'\t'<<reduce<<'\t'<<stress<<'\t'
               <<count.task_beats<<'\t'<<count.issues<<'\t'<<count.commits<<'\t'<<count.members<<'\t'
               <<count.packet_accepts<<'\t'<<count.source_gaps<<'\t'<<count.input_backpressure<<'\t'
               <<count.output_backpressure<<'\t'<<count.read_responses<<'\n';tasks++;
        }
        const Case* directed=nullptr;
        for(const auto& c:cases)if(c.name=="directed_S15_negative_boundary_P4F2")directed=&c;
        require(directed,"missing cancellation reference");cancel_trials(sim,*directed);
        std::cout<<"PASS tasks="<<tasks<<" cancellation_locations=3 cancellation_followup_tasks=3"
                 <<" cycles="<<sim.cycle<<" no_reset_between_tasks=1\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
