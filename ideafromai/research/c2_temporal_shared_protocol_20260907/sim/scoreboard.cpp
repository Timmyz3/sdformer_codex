#include "Vc2_temporal_shared_sum.h"
#include "verilated.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef TEST_T
#define TEST_T 10
#endif
#ifndef TEST_D
#define TEST_D 8
#endif
constexpr int T = TEST_T, D = TEST_D, L = 8, W = 48;
constexpr uint32_t ALL = (1u << T) - 1;
constexpr uint64_t MASK = (1ULL << W) - 1;
using Vec = std::array<int64_t, L>;
using Outputs = std::array<Vec, T>;
struct Source { uint32_t q; Vec value; };
struct Frame { uint32_t id; std::vector<Source> sources; };
struct Event { uint32_t pending; Vec value; };
struct Entry { Vec value; uint64_t stamp; };

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
Vec plus(const Vec& a, const Vec& b) {
    Vec out{};
    for (int h=0; h<L; ++h) {
        out[h] = a[h] + b[h];
        require(out[h] >= -(1LL << (W-1)) && out[h] < (1LL << (W-1)),
                "diagnostic exceeds declared 48-bit domain");
    }
    return out;
}
Vec negative(Vec a) { for (auto& v:a) v=-v; return a; }
Vec diagnostic(int c) {
    Vec a{};
    for (int h=0; h<L; ++h) {
        int weight = ((c+3)*(h+5)*13)%256 - 128;
        if (h==0) weight=-128;
        if (h==1) weight=127;
        a[h] = int64_t(weight) * (999883 + 17*c);
    }
    return a;
}
void pack(WData* words, const Vec& a) {
    std::fill(words, words+12, 0);
    for (int h=0; h<L; ++h) {
        uint64_t bits = uint64_t(a[h]) & MASK;
        int k = h*W/32, shift = h*W%32;
        words[k] |= uint32_t(bits << shift);
        words[k+1] |= uint32_t(bits >> (32-shift));
    }
}
Vec unpack(const WData* words) {
    Vec a{};
    for (int h=0; h<L; ++h) {
        int k=h*W/32, shift=h*W%32;
        uint64_t bits=(uint64_t(words[k]) >> shift) |
                      (uint64_t(words[k+1]) << (32-shift));
        bits &= MASK;
        a[h]=(bits & (1ULL << (W-1))) ? int64_t(bits)-(1LL << W) : int64_t(bits);
    }
    return a;
}

// Independent software model: timestamp map, no RTL FSM or rank-array reuse.
struct Reference {
    std::map<uint32_t, Entry> classes;
    std::deque<Event> events;
    Outputs direct{};
    uint32_t direct_valid=0;
    uint64_t stamp=0, hits=0, assignments=0, evictions=0, singles=0, silent=0;
    uint64_t destination_adds=0, drained=0;
    void drain(bool eviction) {
        require(!classes.empty(), "software empty drain");
        auto oldest=std::min_element(classes.begin(),classes.end(),
            [](const auto& a,const auto& b){return a.second.stamp < b.second.stamp;});
        events.push_back({oldest->first,oldest->second.value});
        classes.erase(oldest);
        ++drained;
        evictions+=eviction;
    }
    void accept(const Source& s, bool last) {
        for (int t=0;t<T;++t) if ((s.q >> t)&1) direct[t]=plus(direct[t],s.value);
        direct_valid |= s.q;
        if (!s.q) ++silent;
        else if (__builtin_popcount(s.q)==1) { events.push_back({s.q,s.value}); ++singles; }
        else {
            auto found=classes.find(s.q);
            if (found!=classes.end()) {
                found->second.value=plus(found->second.value,s.value);
                found->second.stamp=++stamp;
                ++hits;
            } else {
                if (classes.size()==D) drain(true);
                classes.emplace(s.q,Entry{s.value,++stamp});
                ++assignments;
            }
        }
        if (last) while (!classes.empty()) drain(false);
    }
};

class Bench {
public:
    Vc2_temporal_shared_sum dut;
    std::mt19937 rng{0x341d3f67u};
    uint64_t cycles=0, frames=0, sources=0, segments=0, consumer_accepts=0;
    uint64_t partial_stall_cycles=0, all_stall_cycles=0, input_stall_cycles=0;
    uint64_t done_stall_cycles=0, numeric_zero_segments=0, identity_rejections=0;
    uint64_t resets_during_pending=0, hits=0, evictions=0, assignments=0;
    uint64_t destination_adds=0, coefficient_sources=0, silent_sources=0;
    uint64_t lane_outputs=0;

    void low() { dut.clk=0; dut.eval(); }
    void edge() { dut.clk=1; dut.eval(); ++cycles; }
    void reset() {
        dut.in_valid=0; dut.out_ready=0; dut.done_ready=0; dut.rst_n=0;
        for (int n=0;n<3;++n) {low();require(!dut.out_valid&&!dut.done_valid&&!dut.in_ready,"reset visibility");edge();}
        dut.rst_n=1; low();
    }
    void drive(const Frame& f, size_t c) {
        const auto& s=f.sources.at(c);
        dut.in_signature=s.q;pack(dut.in_phi,s.value);
        dut.in_frame=f.id;dut.in_index=c;dut.in_last=(c+1==f.sources.size());
    }
    void run(const Frame& f, bool stalls, bool test_identity=false) {
        require(!f.sources.empty() && f.sources.size()<65536 && f.id<65536,"invalid test frame");
        Reference ref;
        Outputs actual{};
        uint32_t actual_valid=0;
        size_t cursor=0;
        bool offered=false, done_seen=false;
        uint32_t must_hold_mask=0;
        Vec must_hold_value{};
        std::array<unsigned,T> blocked{};
        unsigned done_wait=0;
        size_t completed_events=0;
        unsigned invalid_remaining=test_identity?3:0;
        unsigned recovery_cycles=0;
        bool mid_rejection_done=false;
        uint64_t start=cycles;
        const uint64_t budget=f.sources.size()*200+20000;
        while (true) {
            require(cycles-start<budget,"bounded liveness timeout at frame "+std::to_string(f.id));
            bool malformed=invalid_remaining>0;
            bool recovery=!malformed && recovery_cycles>0;
            if (!malformed && !recovery && !offered && cursor<f.sources.size() && (!stalls || rng()%4)) offered=true;
            if (cursor<f.sources.size()) drive(f,cursor);
            dut.in_valid=offered && !recovery;
            if (malformed) {
                dut.in_valid=1;
                if (!cursor) dut.in_index=9;
                else dut.in_frame=f.id^0x4000;
            }
            uint32_t ready=0;
            for (int t=0;t<T;++t) {
                bool accept=!stalls || blocked[t]>=17 || rng()%5==0;
                if (accept) {ready|=1u<<t;blocked[t]=0;} else ++blocked[t];
            }
            dut.out_ready=ready;
            dut.done_ready=done_seen && (!stalls || done_wait>=7);
            low();
            if (must_hold_mask) {
                require(dut.out_valid==must_hold_mask,"valid withdrawn before every consumer accepted");
                require(unpack(dut.out_value)==must_hold_value && dut.out_frame==f.id,
                        "stalled output payload or identity changed");
            }
            if (done_seen) require(dut.done_valid && dut.done_frame==f.id,
                                   "done withdrawn or changed while stalled");
            if (malformed) {
                require(!dut.in_ready,"wrong identity accepted");
                --invalid_remaining;++identity_rejections;
                if (!invalid_remaining) recovery_cycles=1;
            }
            if (recovery) --recovery_cycles;
            bool take=dut.in_valid && dut.in_ready;
            if (take) {
                require(!malformed && cursor<f.sources.size(),"unexpected source accept");
                ref.accept(f.sources[cursor],cursor+1==f.sources.size());
                coefficient_sources += f.sources[cursor].q != 0;
                silent_sources += f.sources[cursor].q == 0;
                ++cursor;++sources;offered=false;
            } else if (dut.in_valid && !malformed) ++input_stall_cycles;
            uint32_t valid=dut.out_valid;
            must_hold_mask=valid & ~ready;
            if (must_hold_mask) must_hold_value=unpack(dut.out_value);
            if (valid) {
                require(!ref.events.empty(),"unexpected/duplicate output segment");
                auto& expected=ref.events.front();
                require(valid==expected.pending,"consumer valid mask differs from unaccepted reference mask");
                Vec value=unpack(dut.out_value);
                require(value==expected.value,"wrong segment payload");
                require(dut.out_frame==f.id,"output frame changed");
                uint32_t accepted=valid&ready;
                if (!accepted) ++all_stall_cycles;
                if (accepted && accepted!=valid) ++partial_stall_cycles;
                for (int t=0;t<T;++t) if ((accepted>>t)&1) {
                    destination_adds += (actual_valid>>t)&1;
                    actual[t]=plus(actual[t],value);actual_valid|=1u<<t;++consumer_accepts;
                }
                expected.pending &= ~accepted;
                if (!expected.pending) {
                    numeric_zero_segments += std::all_of(value.begin(),value.end(),[](int64_t v){return v==0;});
                    ref.events.pop_front();++segments;++completed_events;
                }
            }
            bool finished=dut.done_valid && dut.done_ready;
            if (dut.done_valid) {
                require(cursor==f.sources.size()&&ref.events.empty()&&ref.classes.empty(),"premature done");
                require(!dut.in_ready&&!dut.out_valid,"done overlaps source/output");
                require(dut.done_frame==f.id,"done identity unstable");
                require(actual==ref.direct && actual_valid==ref.direct_valid,"direct integer outputs/valid mismatch");
                done_seen=true;
                if (!dut.done_ready) {++done_wait;++done_stall_cycles;}
            }
            edge();
            if (finished) {
                ++frames;lane_outputs+=T*L;
                hits+=ref.hits;evictions+=ref.evictions;assignments+=ref.assignments;
                require(completed_events==ref.drained+ref.singles,"segment count mismatch");
                break;
            }
            // Malformed identities are deliberately out-of-contract inputs, tested
            // only while no legal offer is held. No malformed transaction is accepted.
            if (test_identity && !mid_rejection_done && cursor>=1 && !offered) {
                mid_rejection_done=true;invalid_remaining=3;
            }
        }
        dut.in_valid=0;dut.done_ready=0;low();
    }
    void reset_pending() {
        Frame f{60000,{{3,diagnostic(17)}}};
        drive(f,0);dut.in_valid=1;dut.out_ready=0;dut.done_ready=0;
        bool accepted=false,pending_seen=false;
        for (unsigned n=0;n<100;++n) {
            low();bool take=dut.in_valid&&dut.in_ready;
            if (dut.out_valid) {require(dut.out_valid==3,"reset target mask");pending_seen=true;break;}
            edge();if(take){accepted=true;dut.in_valid=0;}
        }
        require(accepted&&pending_seen,"did not reach pending drain");
        // Accept just one of two time consumers, then cancel the frame.
        dut.out_ready=1;low();require(dut.out_valid==3,"missing pre-reset partial");edge();
        dut.out_ready=0;low();require(dut.out_valid==2,"acknowledged bit repeated");
        ++resets_during_pending;reset();
        run(Frame{60001,{{0,diagnostic(4)},{1,diagnostic(6)},{3,diagnostic(7)}}},true);
    }
    void report() {
        std::cout << "{\"T\":"<<T<<",\"D\":"<<D<<",\"lanes\":8,\"value_bits\":48"
                  <<",\"frames\":"<<frames<<",\"sources\":"<<sources
                  <<",\"segment_events\":"<<segments<<",\"consumer_accepts\":"<<consumer_accepts
                  <<",\"lane_outputs_verified\":"<<lane_outputs
                  <<",\"dictionary_hits\":"<<hits<<",\"dictionary_assignments\":"<<assignments
                  <<",\"evictions\":"<<evictions<<",\"destination_adds\":"<<destination_adds
                  <<",\"coefficient_sources\":"<<coefficient_sources<<",\"silent_sources\":"<<silent_sources
                  <<",\"partial_accept_stall_cycles\":"<<partial_stall_cycles
                  <<",\"all_consumers_stalled_cycles\":"<<all_stall_cycles
                  <<",\"input_stall_cycles\":"<<input_stall_cycles
                  <<",\"done_stall_cycles\":"<<done_stall_cycles
                  <<",\"zero_valued_valid_segments\":"<<numeric_zero_segments
                  <<",\"identity_rejections\":"<<identity_rejections
                  <<",\"reset_during_pending\":"<<resets_during_pending
                  <<",\"simulator_cycles_only\":"<<cycles
                  <<",\"errors\":0,\"status\":\"PASS_FUNCTIONAL_ONLY\"}\n";
    }
};

std::vector<Frame> read_fixture(const std::string& path) {
    std::ifstream f(path);require(bool(f),"cannot open fixture");
    std::string magic;int t,lanes,n;f>>magic>>t>>lanes>>n;
    require(magic=="C2PROTO1"&&t==T&&lanes==L&&n>0,"fixture header mismatch");
    std::vector<Frame> out;
    for (int i=0;i<n;++i) {
        Frame frame;int count;f>>frame.id>>count;require(count>0,"empty fixture frame");
        for (int c=0;c<count;++c) {
            Source s{};f>>s.q;require(!(s.q&~ALL),"fixture signature width");
            for (auto& x:s.value) f>>x;
            frame.sources.push_back(s);
        }
        require(bool(f),"truncated fixture");out.push_back(frame);
    }
    std::string excess;require(!(f>>excess),"trailing fixture data");return out;
}

int main(int argc,char** argv) {
    try {
        Verilated::commandArgs(argc,argv);
        Bench b;b.reset();
        uint32_t id=1;
        std::vector<uint32_t> multi;
        for (uint32_t q=1;q<=ALL;++q) if (__builtin_popcount(q)>1) multi.push_back(q);
        // First/last zero and a valid zero produced by signed cancellation.
        b.run(Frame{id++,{{0,diagnostic(0)},{0,diagnostic(1)}}},true,true);
        b.run(Frame{id++,{{3,diagnostic(2)},{3,negative(diagnostic(2))},{0,diagnostic(9)}}},true,true);
        Frame singles{id++,{}};
        for (int c=0;c<4*T;++c) singles.sources.push_back({1u<<(c%T),diagnostic(c)});
        b.run(singles,true);
        Frame churn{id++,{}};
        for (int c=0;c<400;++c) {
            uint32_t q=multi[size_t(c) % std::min(multi.size(),size_t(D+1))];
            churn.sources.push_back({q,diagnostic(c)});
            if (c%7==0) churn.sources.push_back({1u<<(c%T),negative(diagnostic(c))});
            if (c%11==0) churn.sources.push_back({q,negative(diagnostic(c+1))});
        }
        b.run(churn,true);
        // Exercise sign extension around bit 47 without overflow at any partial.
        Vec large{};for(int h=0;h<L;++h) large[h]=(h%2?-1:1)*((1LL<<45)+h);
        b.run(Frame{id++,{{3,large},{3,negative(large)},{ALL,large},{ALL,negative(large)}}},true);
        for (int frame=0;frame<128;++frame) {
            Frame f{id++,{}};
            int n=1+int(b.rng()%240);
            for (int c=0;c<n;++c) {
                uint32_t q;
                switch (b.rng()%6) {
                    case 0:q=0;break;
                    case 1:q=1u<<(b.rng()%T);break;
                    case 2:q=3;break;
                    default:q=b.rng()&ALL;break;
                }
                Vec value=diagnostic(c+frame);
                if (b.rng()%3==0) value=negative(value);
                f.sources.push_back({q,value});
            }
            b.run(f,frame%4!=0);
        }
        if(argc==2) for(const auto& frame:read_fixture(argv[1])) b.run(frame,true);
        b.reset_pending();
        require(b.numeric_zero_segments&&b.partial_stall_cycles&&b.all_stall_cycles&&b.hits&&
                b.done_stall_cycles&&b.input_stall_cycles&&b.identity_rejections&&b.resets_during_pending,
                "missing required protocol coverage");
        if(T==10) require(b.evictions>0,"eviction not exercised");
        b.report();b.dut.final();return 0;
    } catch (const std::exception& e) {
        std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;
    }
}
