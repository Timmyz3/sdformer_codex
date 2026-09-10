#include "Vpacket_test_top.h"
#include "verilated.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

constexpr int B=B_TEST, K=K_TEST, W=W_TEST, T=T_TEST;
constexpr int log2ceil(int n) { int r=0; for (int v=n-1;v;v>>=1) ++r; return r; }
constexpr int NW=W+1, IW=log2ceil(B), EW=2*NW+IW+1;
constexpr int AO=B, BO=AO+NW, AV=BO+NW, BV=AV+1, KO=BV+1;
constexpr int MO=KO+K*EW, PW=MO+82, WORDS=(PW+31)/32;
constexpr int64_t MIN_U=-(int64_t(1)<<(W-1)), MAX_U=(int64_t(1)<<(W-1))-1;
using Packet=std::array<uint32_t, WORDS>;
struct Interval { int64_t lo,hi; };
struct Example {
    std::vector<Interval> u;
    int64_t prediction;
    bool reverse;
    uint32_t context,theta;
    uint16_t epoch;
};
struct Result { unsigned status; uint32_t bits,theta,context; uint16_t epoch; };
Vpacket_test_top dut;
std::mt19937 rng(0x1d3f);
uint64_t cycles=0, encodes=0, decodes=0, commits=0, replays=0, errors=0;
uint64_t proven_bits=0, group_tests=0, group_commits=0, group_errors=0;
uint64_t replay_requests=0, stalled_cycles=0, reset_tests=0;
uint64_t replacement_checks=0;
void require(bool yes, const std::string& why) {
    if(!yes) throw std::runtime_error(why+" at clock "+std::to_string(cycles));
}
template<class Bus> uint64_t read_bits(const Bus& bus,int off,int width) {
    uint64_t value=0;
    for(int i=0;i<width;++i) {
        bool b;
        if constexpr(std::is_integral_v<Bus>) b=(uint64_t(bus)>>(off+i))&1;
        else b=(bus[(off+i)/32]>>((off+i)%32))&1;
        value|=uint64_t(b)<<i;
    }
    return value;
}
template<class Bus> void write_bits(Bus& bus,int off,int width,uint64_t value) {
    for(int i=0;i<width;++i) {
        if constexpr(std::is_integral_v<Bus>) {
            uint64_t temp=bus,mask=uint64_t(1)<<(off+i);
            temp=(temp&~mask)|(((value>>i)&1)<<(off+i)); bus=Bus(temp);
        } else {
            uint32_t mask=uint32_t(1)<<((off+i)%32);
            bus[(off+i)/32]=(bus[(off+i)/32]&~mask)
                |(uint32_t((value>>i)&1)<<((off+i)%32));
        }
    }
}
int64_t signed_field(const Packet&p,int off,int width) {
    uint64_t v=read_bits(p,off,width);
    return (v&(uint64_t(1)<<(width-1))) ? int64_t(v)-int64_t(uint64_t(1)<<width) : int64_t(v);
}
Packet exported() {
    Packet p{}; for(int i=0;i<PW;++i) write_bits(p,i,1,read_bits(dut.e_packet,i,1)); return p;
}
void import_packet(const Packet&p) {
    for(int i=0;i<PW;++i) write_bits(dut.v_packet,i,1,read_bits(p,i,1));
}
void tick() { dut.clk=0;dut.eval();dut.clk=1;dut.eval();dut.clk=0;dut.eval();++cycles; }
void reset() {
    dut.e_start_valid=0;dut.e_in_valid=0;dut.e_packet_ready=0;
    dut.v_valid=0;dut.v_result_ready=0;
    dut.g_start_valid=0;dut.g_packet_valid=0;dut.g_replay_ready=0;
    dut.g_repair_valid=0;dut.g_result_ready=0;
    dut.rst_n=0;tick();dut.rst_n=1;tick();
    require(dut.e_start_ready && dut.v_ready && dut.g_start_ready,"reset ready");
    require(!dut.e_packet_valid&&!dut.v_result_valid&&!dut.g_result_valid,"reset cancels");
}
Interval normalize(Interval x,bool reverse) { return reverse?Interval{-x.hi,-x.lo}:x; }
int64_t distance(Interval x,int64_t p) { return std::max({x.lo-p,p-x.hi,int64_t(0)}); }
bool malformed(const Example&e) {
    return std::any_of(e.u.begin(),e.u.end(),[](auto x){return x.lo>x.hi;});
}
std::vector<int> selected(const Example&e) {
    std::vector<int> indices(B);std::iota(indices.begin(),indices.end(),0);
    int64_t pred=e.reverse?-e.prediction:e.prediction;
    std::stable_sort(indices.begin(),indices.end(),[&](int a,int b) {
        return distance(normalize(e.u[a],e.reverse),pred)<distance(normalize(e.u[b],e.reverse),pred);
    });
    indices.resize(K);return indices;
}
void check_record(const Example&e,const Packet&p) {
    require(read_bits(p,MO,32)==e.context && read_bits(p,MO+32,16)==e.epoch
        && read_bits(p,MO+48,32)==e.theta && read_bits(p,MO+80,1)==e.reverse,"metadata export");
    require(read_bits(p,MO+81,1)==malformed(e),"malformed input retained");
    if(malformed(e)) return;
    auto keep=selected(e);std::vector<int> actual;
    for(int k=0;k<K;++k) {
        int idx=read_bits(p,KO+k*EW+2*NW,IW);
        require(idx<B && read_bits(p,KO+k*EW+2*NW+IW,1),"valid retained index");
        auto x=normalize(e.u[idx],e.reverse);
        require(signed_field(p,KO+k*EW,NW)==x.lo && signed_field(p,KO+k*EW+NW,NW)==x.hi,"retained exact interval");
        actual.push_back(idx);
    }
    std::sort(actual.begin(),actual.end());auto sorted=keep;std::sort(sorted.begin(),sorted.end());
    require(actual==sorted,"offline top K agrees; ties retain earlier index");
    bool av=false,bv=false;int64_t a=0,b=0,pred=e.reverse?-e.prediction:e.prediction;
    for(int i=0;i<B;++i) {
        auto x=normalize(e.u[i],e.reverse);bool bit=x.lo>=pred;
        require(read_bits(p,i,1)==bit,"candidate bit position");
        if(std::find(keep.begin(),keep.end(),i)!=keep.end()) continue;
        if(bit) { if(!bv||x.lo<b)b=x.lo;bv=true; }
        else { if(!av||x.hi>a)a=x.hi;av=true; }
    }
    require(read_bits(p,AV,1)==av && read_bits(p,BV,1)==bv,"empty classes use flags");
    if(av)require(signed_field(p,AO,NW)==a,"discarded off upper envelope includes evictions");
    if(bv)require(signed_field(p,BO,NW)==b,"discarded on lower envelope includes evictions");
}
Packet encode(const Example&e,bool stalls=false) {
    require(dut.e_start_ready,"encoder available after export");
    dut.e_context=e.context;dut.e_epoch=e.epoch;dut.e_theta=e.theta;
    dut.e_reverse=e.reverse;dut.e_prediction=uint32_t(e.prediction);
    dut.e_start_valid=1;tick();dut.e_start_valid=0;
    for(auto x:e.u) {
        if(stalls) for(unsigned i=0,n=rng()%3;i<n;++i){tick();++stalled_cycles;}
        require(dut.e_in_ready&&!dut.e_packet_valid,"collect ready, not premature export");
        dut.e_lo=uint32_t(x.lo);dut.e_hi=uint32_t(x.hi);dut.e_in_valid=1;
        tick();dut.e_in_valid=0;
    }
    require(dut.e_packet_valid&&!dut.e_in_ready,"exact B inputs before export");
    Packet p=exported();
    if(stalls) for(int i=0;i<3;++i) {
        dut.e_context=rng();dut.e_lo=rng();dut.e_hi=rng();tick();++stalled_cycles;
        require(dut.e_packet_valid && exported()==p && !dut.e_start_ready,"export stable under backpressure");
    }
    dut.e_packet_ready=1;tick();dut.e_packet_ready=0;
    ++encodes;check_record(e,p);return p;
}
Result expected(const Example&e,Interval tau,bool bad_identity=false) {
    Result r{0,0,e.theta,e.context,e.epoch};
    if(malformed(e)||tau.lo>tau.hi||bad_identity){r.status=2;return r;}
    tau=normalize(tau,e.reverse);auto keep=selected(e);
    int64_t pred=e.reverse?-e.prediction:e.prediction;
    for(int i=0;i<B;++i) {
        auto x=normalize(e.u[i],e.reverse);
        bool bit=x.lo>=pred;
        if(std::find(keep.begin(),keep.end(),i)!=keep.end()) {
            if(x.lo>=tau.hi)bit=true;
            else if(x.hi<tau.lo)bit=false;
            else r.status=1;
        } else if(bit ? x.lo<tau.hi : x.hi>=tau.lo) r.status=1;
        r.bits|=uint32_t(bit)<<i;
    }
    if(r.status)r.bits=0;
    return r;
}
Result decode(const Example&e,const Packet&p,Interval tau,bool stalls=false,
              unsigned identity_error=0,bool corrupt_error=false) {
    require(dut.v_ready,"verifier ready");
    import_packet(p);dut.v_context=e.context^(identity_error==1);dut.v_epoch=e.epoch^(identity_error==2);
    dut.v_tau_lo=uint32_t(tau.lo);dut.v_tau_hi=uint32_t(tau.hi);dut.v_valid=1;
    tick();dut.v_valid=0;
    require(dut.v_result_valid,"verifier response");
    Result r{dut.v_status,dut.v_bits,dut.v_theta,dut.v_out_context,dut.v_out_epoch};
    auto ref=expected(e,tau,identity_error!=0);
    if(corrupt_error){ref.status=2;ref.bits=0;}
    require(r.status==ref.status && r.bits==ref.bits,"decoder vs independent interval oracle");
    require(r.theta==e.theta && r.context==e.context && r.epoch==e.epoch,"amplitude identity preserved");
    if(r.status==0) {
        // Check both endpoints and both possible threshold extremes directly in
        // the original direction; this does not reuse the packet certificate.
        for(int i=0;i<B;++i) for(auto u:{e.u[i].lo,e.u[i].hi}) for(auto th:{tau.lo,tau.hi})
            require(bool((r.bits>>i)&1)==(e.reverse?u<=th:u>=th),"every interval realization has same firing");
        proven_bits+=B;++commits;
    } else if(r.status==1)++replays;else ++errors;
    if(stalls) for(int i=0;i<3;++i) {
        dut.v_context=rng();dut.v_tau_lo=rng();dut.v_tau_hi=rng();tick();++stalled_cycles;
        require(dut.v_result_valid && !dut.v_ready && dut.v_status==r.status
            && dut.v_bits==r.bits && dut.v_theta==r.theta && dut.v_out_context==r.context
            && dut.v_out_epoch==r.epoch,"response stable while stalled");
    }
    dut.v_result_ready=1;tick();dut.v_result_ready=0;++decodes;return r;
}
Example make_example(bool reverse=false) {
    Example e;e.reverse=reverse;e.prediction=0;e.context=rng();e.epoch=uint16_t(rng());
    // Includes fractional floating representations as opaque amplitude payloads.
    static const uint32_t theta[]={0x3f7ff852,0x3f200000,0x3f800000,0x3eaaaaab};
    e.theta=theta[rng()%4];e.u.resize(B);return e;
}
void exhaustive_small() {
    if constexpr(B==4 && W<=8) {
        for(int code=0;code<625;++code) for(int reverse=0;reverse<2;++reverse)
        for(int pred=-2;pred<=2;++pred) {
            auto e=make_example(reverse);e.prediction=pred;int n=code;
            for(auto &x:e.u){x.lo=x.hi=n%5-2;n/=5;}
            auto p=encode(e);
            for(int th=-3;th<=3;++th)decode(e,p,{th,th});
        }
    }
}
void random_and_boundaries() {
    std::vector<std::pair<Example,Packet>> stored;
    for(int trial=0;trial<600;++trial) {
        auto e=make_example(trial&1);
        auto number=[&](){return MIN_U+int64_t(uint64_t(rng())%(uint64_t(1)<<W));};
        e.prediction=number();
        for(auto &x:e.u) {
            x.lo=number();x.hi=x.lo;
            if(trial%3==0)x.hi=std::min(MAX_U,x.lo+int64_t(rng()%4));
        }
        if(trial<10) {
            e.prediction=(trial&2)?MIN_U:MAX_U;
            for(int i=0;i<B;++i)e.u[i]={i&1?MIN_U:MAX_U,i&1?MIN_U:MAX_U};
        }
        auto p=encode(e,true);
        for(int q=0;q<12;++q) {
            int64_t lo=q<2?(q?MAX_U:MIN_U):q<4?e.prediction:number();
            int64_t hi=q%3==0?std::min(MAX_U,lo+2):lo;
            decode(e,p,{lo,hi},q==0);
        }
        if(trial<32)stored.emplace_back(e,p);
    }
    // Encoder has exported many contexts before final statistics arrive.
    for(auto it=stored.rbegin();it!=stored.rend();++it)decode(it->first,it->second,{0,0},true);
    for(int reverse=0;reverse<2;++reverse) {
        auto e=make_example(reverse);for(auto &x:e.u)x={-1,1};
        auto p=encode(e,true);decode(e,p,{0,0},true);
        decode(e,p,{1,-1});decode(e,p,{0,0},false,1);decode(e,p,{0,0},false,2);
        if(K>1) {
            auto damaged=p;
            write_bits(damaged,KO+EW+2*NW,IW,read_bits(p,KO+2*NW,IW));
            decode(e,damaged,{0,0},true,0,true);
        }
        if(K>0) {
            auto damaged=p;write_bits(damaged,KO,NW,1);write_bits(damaged,KO+NW,NW,uint64_t(-1));
            decode(e,damaged,{0,0},true,0,true);
            damaged=p;write_bits(damaged,KO+2*NW+IW,1,0);decode(e,damaged,{0,0},false,0,true);
            if((1<<IW)>B) {
                damaged=p;write_bits(damaged,KO+2*NW,IW,B);
                decode(e,damaged,{0,0},false,0,true);
            }
        }
        e.u.back()={2,-2};p=encode(e,true);decode(e,p,{0,0},true);
    }
}
void group_case(int mode) {
    // mode 0: all certified. 1: last time step requests full-T recovery.
    // mode 2: last input identity mismatch. 3: recovery identity mismatch.
    uint32_t base=rng()&0xfffffff0u,theta=0x3f7ff852;
    uint32_t supplied_base=base|(mode==9?1u:0u);
    uint16_t epoch=uint16_t(rng());
    dut.g_base=supplied_base;dut.g_epoch=epoch;dut.g_theta=theta;dut.g_start_valid=1;
    require(dut.g_start_ready,"group ready");tick();dut.g_start_valid=0;
    std::array<uint32_t,T> final_bits{};
    uint32_t failure_mask=0;
    bool needs_repair=mode==1||mode==3||mode==4||mode==7||mode==8;
    for(int t=0;t<T;++t) {
        auto e=make_example();e.context=base|t;e.epoch=epoch;e.theta=theta;
        for(int i=0;i<B;++i)e.u[i]={i&1?-2:2,i&1?-2:2};
        bool fail=needs_repair && (t==T-1 || (mode==4&&t==0));
        if(fail)failure_mask|=uint32_t(1)<<t;
        if(fail)for(auto &x:e.u)x={-1,1};
        auto p=encode(e,t==0);auto r=decode(e,p,{0,0},t==0);
        require(r.status==(fail?1u:0u),"constructed group case");
        // The external recovery oracle selects an admitted concrete U (the
        // midpoint), then performs the original dense >= comparison directly.
        // It does not reuse packet bits, retained values, or certificate bounds.
        for(int i=0;i<B;++i) {
            int64_t concrete_u=(e.u[i].lo+e.u[i].hi)/2;
            final_bits[t]|=uint32_t(concrete_u>=0)<<i;
        }
        require(dut.g_packet_ready&&!dut.g_result_valid,"group accepts before seal");
        dut.g_packet_status=(mode==10&&t==T-1)?3:r.status;dut.g_packet_bits=r.bits;
        dut.g_packet_context=r.context^((mode==2&&t==T-1)?16u:0u);
        dut.g_packet_epoch=r.epoch^((mode==6&&t==T-1)?1:0);
        dut.g_packet_theta=r.theta^((mode==5&&t==T-1)?1u:0u);
        dut.g_packet_valid=1;tick();dut.g_packet_valid=0;
        require(!dut.g_result_valid,"last packet cannot prematurely commit");
        for(int i=0;i<T*B;++i)require(!read_bits(dut.g_bits,i,1),"no time-step result leaks before group seal");
    }
    tick();
    if(needs_repair) {
        require(dut.g_replay_valid&&!dut.g_result_valid,"full T replay request before commit");
        require(dut.g_replay_mask==failure_mask,"all failure causes survive register timing");
        for(int i=0;i<3;++i){tick();++stalled_cycles;require(dut.g_replay_valid,"replay holds");}
        dut.g_replay_ready=1;tick();dut.g_replay_ready=0;++replay_requests;
        require(dut.g_repair_ready&&!dut.g_result_valid,"await external recovery");
        for(int t=0;t<T;++t)write_bits(dut.g_repair_bits,t*B,B,final_bits[t]);
        dut.g_repair_context=base^(mode==8?16u:0u);
        dut.g_repair_epoch=epoch^(mode==3);
        dut.g_repair_theta=theta^(mode==7?1u:0u);
        dut.g_repair_valid=1;tick();dut.g_repair_valid=0;
    }
    unsigned expected_status=(mode==0||mode==1||mode==4)?0:2;
    require(dut.g_result_valid && dut.g_status==expected_status,"whole group result");
    require(dut.g_out_context==supplied_base && dut.g_out_epoch==epoch && dut.g_out_theta==theta,"group amplitude and identity");
    for(int i=0;i<3;++i) {
        for(int t=0;t<T;++t)require(read_bits(dut.g_bits,t*B,B)==(expected_status?0:final_bits[t]),"full T commit or zero-on-error");
        require(dut.g_result_valid&&!dut.g_start_ready,"group result stable");tick();++stalled_cycles;
    }
    dut.g_result_ready=1;tick();dut.g_result_ready=0;
    require(!dut.g_result_valid&&dut.g_start_ready,"one group consumption only");
    for(int i=0;i<3;++i){tick();require(!dut.g_result_valid,"no duplicate commit after replay");}
    ++group_tests;if(expected_status)++group_errors;else ++group_commits;
}
void consecutive_verifier_requests() {
    auto e=make_example();for(auto &x:e.u)x={-2,-2};e.prediction=0;
    auto p=encode(e);
    auto apply=[&](unsigned identity) {
        import_packet(p);dut.v_context=e.context^identity;dut.v_epoch=e.epoch;
        dut.v_tau_lo=0;dut.v_tau_hi=0;dut.v_valid=1;
    };
    apply(0);tick();require(dut.v_result_valid&&dut.v_status==0,"first pipelined result");
    dut.v_result_ready=1;apply(1);tick();
    require(dut.v_result_valid&&dut.v_status==2&&dut.v_bits==0,"consume plus replace same cycle");
    ++replacement_checks;
    apply(0);tick();require(dut.v_result_valid&&dut.v_status==0,"replace error with verified packet");
    ++replacement_checks;
    dut.v_result_ready=0;apply(1);tick();
    require(dut.v_result_valid&&!dut.v_ready&&dut.v_status==0,"backpressure refuses pending replacement");
    ++replacement_checks;
    dut.v_result_ready=1;tick();
    require(dut.v_result_valid&&dut.v_status==2,"pending replacement accepted on ready");
    ++replacement_checks;
    dut.v_valid=0;tick();dut.v_result_ready=0;
    require(!dut.v_result_valid,"pipeline drains exactly once");
}
void reset_group_states() {
    for(int target=0;target<4;++target) {
        dut.g_base=0x12340;dut.g_epoch=7;dut.g_theta=0x3f200000;
        dut.g_start_valid=1;tick();dut.g_start_valid=0;
        if(target>0) {
            for(int t=0;t<T;++t) {
                dut.g_packet_context=0x12340|t;dut.g_packet_epoch=7;
                dut.g_packet_theta=0x3f200000;dut.g_packet_bits=0;
                dut.g_packet_status=(target<3&&t==T-1)?1:0;
                dut.g_packet_valid=1;tick();dut.g_packet_valid=0;
            }
            tick();
            if(target==1)require(dut.g_replay_valid,"reset at replay request");
            if(target==2) {
                dut.g_replay_ready=1;tick();dut.g_replay_ready=0;
                require(dut.g_repair_ready,"reset awaiting full-T repair");
            }
            if(target==3)require(dut.g_result_valid,"reset unconsumed group commit");
        } else require(dut.g_packet_ready,"reset during group collection");
        reset();++reset_tests;
        require(!dut.g_replay_valid&&!dut.g_repair_ready,"reset cancels group services");
        for(int i=0;i<T*B;++i)require(!read_bits(dut.g_bits,i,1),"reset leaks no group bits");
    }
}
void reset_in_flight() {
    auto e=make_example();for(auto &x:e.u)x={0,0};
    // Abandon a partially collected packet, then prove a new context works.
    dut.e_start_valid=1;dut.e_context=0;dut.e_epoch=0;dut.e_theta=0;tick();
    dut.e_start_valid=0;dut.e_in_valid=1;dut.e_lo=0;dut.e_hi=0;tick();
    reset();++reset_tests;auto p=encode(e);decode(e,p,{0,0});
    // Reset while a complete packet is stalled at export.
    dut.e_start_valid=1;tick();dut.e_start_valid=0;
    dut.e_in_valid=1;for(int i=0;i<B;++i)tick();dut.e_in_valid=0;
    require(dut.e_packet_valid,"reset export test setup");reset();++reset_tests;
    // Reset an unconsumed verifier response.
    import_packet(p);dut.v_context=e.context;dut.v_epoch=e.epoch;dut.v_tau_lo=0;dut.v_tau_hi=0;
    dut.v_valid=1;tick();dut.v_valid=0;require(dut.v_result_valid,"reset response setup");reset();++reset_tests;
}
int main(int argc,char**argv) {
    Verilated::commandArgs(argc,argv);
    try {
        reset();exhaustive_small();random_and_boundaries();
        consecutive_verifier_requests();
        for(int i=0;i<22;++i)group_case(i%11);
        reset_in_flight();reset_group_states();group_case(0);dut.final();
        std::cout<<"{\"status\":\"PASS_FUNCTIONAL_RTL_ONLY\",\"B\":"<<B<<",\"K\":"<<K
          <<",\"W\":"<<W<<",\"T\":"<<T<<",\"packet_bits\":"<<PW
          <<",\"encodes\":"<<encodes<<",\"verifier_requests\":"<<decodes
          <<",\"certified_packets\":"<<commits<<",\"replay_packets\":"<<replays
          <<",\"error_packets\":"<<errors<<",\"certified_bits_checked\":"<<proven_bits
          <<",\"group_tests\":"<<group_tests<<",\"group_commits\":"<<group_commits
          <<",\"group_errors\":"<<group_errors<<",\"external_repair_requests\":"<<replay_requests
          <<",\"stalled_test_clocks\":"<<stalled_cycles<<",\"reset_tests\":"<<reset_tests
          <<",\"same_cycle_replacement_checks\":"<<replacement_checks
          <<",\"test_clocks_not_workload_cycles\":"<<cycles
          <<",\"PPA_ADMISSION\":0,\"RTL_SPEEDUP_ADMISSION\":0,\"FROZEN_FP_EQUIVALENCE\":0}\n";
        return 0;
    } catch(const std::exception&e) {std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}
}
