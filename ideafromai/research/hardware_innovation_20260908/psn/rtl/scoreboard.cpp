#include "Vshared_fc1_psn.h"
#include "verilated.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

template<class T> T scalar(std::istream& in) {
    T value{}; in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("truncated case input");
    return value;
}
template<class T> std::vector<T> array(std::istream& in, size_t count) {
    std::vector<T> value(count);
    in.read(reinterpret_cast<char*>(value.data()), sizeof(T)*count);
    if (!in) throw std::runtime_error("truncated case array");
    return value;
}
template<class T> void clear(T& value, unsigned width) {
    for (unsigned i=0; i<(width+31)/32; ++i) value[i]=0;
}
template<class T> void put(T& target, unsigned offset, unsigned width, uint32_t value) {
    for (unsigned b=0; b<width; ++b) {
        const unsigned bit=offset+b;
        const uint32_t mask=uint32_t(1)<<(bit%32);
        target[bit/32]=(target[bit/32]&~mask)|(((value>>b)&1)<<(bit%32));
    }
}
template<class T> uint32_t bit(const T& value, unsigned offset) {
    return (value[offset/32]>>(offset%32))&1;
}

struct Case {
    std::string name;
    unsigned positions, channels, lanes, length;
    std::vector<uint16_t> decode, uops;
    std::vector<int32_t> tau;
    std::vector<int8_t> weights;
    std::vector<uint8_t> codes, expected;
};
Case read_case(std::istream& in) {
    Case c;
    const auto len=scalar<uint32_t>(in);
    c.name.resize(len); in.read(c.name.data(), len);
    c.positions=scalar<uint32_t>(in); c.channels=scalar<uint32_t>(in);
    c.lanes=scalar<uint32_t>(in); c.length=scalar<uint32_t>(in);
    if (c.lanes!=96 || c.positions>32 || c.length>256)
        throw std::runtime_error("case dimensions do not match compiled prototype");
    c.decode=array<uint16_t>(in,8); c.uops=array<uint16_t>(in,c.length);
    c.tau=array<int32_t>(in,10*c.lanes);
    c.weights=array<int8_t>(in,c.channels*c.lanes);
    c.codes=array<uint8_t>(in,c.channels*c.positions);
    c.expected=array<uint8_t>(in,c.positions*10*c.lanes);
    return c;
}

struct Simulation {
    Vshared_fc1_psn dut;
    uint64_t time=0;
    void edge() {
        dut.clk=0; dut.eval();
        dut.clk=1; dut.eval(); ++time;
        if (Verilated::gotFinish()) throw std::runtime_error("RTL terminated early");
    }
    Simulation() {
        dut.rst_n=0; dut.start=0; dut.src_valid=0;
        dut.cfg_decode_valid=0; dut.cfg_uop_valid=0; dut.cfg_tau_valid=0;
        dut.out_ready=1;
        for (int i=0;i<3;++i) edge();
        dut.rst_n=1; edge();
    }
    void configure(const Case& c) {
        dut.src_valid=0;
        if (dut.busy) throw std::runtime_error("previous tile still busy");
        dut.cfg_decode_valid=1;
        for (unsigned k=0;k<8;++k) {
            dut.cfg_code=k; dut.cfg_decode=c.decode[k]; edge();
        }
        dut.cfg_decode_valid=0; dut.cfg_uop_valid=1;
        for (unsigned i=0;i<c.length;++i) {
            dut.cfg_uop_addr=i; dut.cfg_uop=c.uops[i]; edge();
        }
        dut.cfg_uop_valid=0; dut.cfg_tau_valid=1;
        for (unsigned t=0;t<10;++t) {
            dut.cfg_tau_row=t; clear(dut.cfg_tau,96*24);
            for (unsigned h=0;h<96;++h)
                put(dut.cfg_tau,h*24,24,uint32_t(c.tau[t*96+h]));
            edge();
        }
        dut.cfg_tau_valid=0;
        dut.positions=c.positions; dut.program_length=c.length;
        dut.start=1; edge(); dut.start=0;
    }
    struct Metrics {
        uint64_t cycles=0, first_output=0, source_wait=0, output_stall=0, decisions=0;
    };
    Metrics run(const Case& c, bool gaps) {
        configure(c);
        Metrics m;
        unsigned sent=0, received=0;
        std::vector<bool> seen(c.positions,false);
        bool held=false, previous_stall=false;
        unsigned previous_position=0;
        std::vector<uint32_t> previous_packet(30);
        for (;m.cycles<2000000;++m.cycles) {
            if (!held && sent<c.channels &&
                (!gaps || (m.cycles%11!=3 && m.cycles%17!=0))) {
                clear(dut.src_weight,96*8); clear(dut.src_codes,32*3);
                for (unsigned h=0;h<96;++h)
                    put(dut.src_weight,h*8,8,uint8_t(c.weights[sent*96+h]));
                for (unsigned p=0;p<c.positions;++p)
                    put(dut.src_codes,p*3,3,c.codes[sent*c.positions+p]);
                dut.src_last=(sent+1==c.channels); held=true;
            }
            dut.src_valid=held;
            dut.out_ready=(!gaps || (m.cycles%13>=3 && m.cycles%53<43));
            dut.clk=0; dut.eval();
            if (previous_stall) {
                if (!dut.out_valid || dut.out_position!=previous_position)
                    throw std::runtime_error("output valid/position changed under backpressure");
                for (unsigned w=0;w<30;++w)
                    if (dut.out_spikes[w]!=previous_packet[w])
                        throw std::runtime_error("output data changed under backpressure");
            }
            if (dut.out_valid) {
                if (!received && !m.first_output) m.first_output=m.cycles;
                if (!dut.out_ready) ++m.output_stall;
                else {
                    const unsigned p=dut.out_position;
                    if (p>=c.positions || seen[p])
                        throw std::runtime_error("duplicate or invalid output position");
                    for (unsigned t=0;t<10;++t) for (unsigned h=0;h<96;++h) {
                        const auto expected=c.expected[(p*10+t)*96+h];
                        if (bit(dut.out_spikes,t*96+h)!=expected) {
                            std::cerr<<c.name<<" p="<<p<<" t="<<t<<" h="<<h
                                     <<" expected="<<unsigned(expected)<<" cycle="<<m.cycles<<"\n";
                            throw std::runtime_error("integer gate mismatch");
                        }
                        ++m.decisions;
                    }
                    seen[p]=true; ++received;
                }
            }
            previous_stall=dut.out_valid && !dut.out_ready;
            if (previous_stall) {
                previous_position=dut.out_position;
                for (unsigned w=0;w<30;++w) previous_packet[w]=dut.out_spikes[w];
            }
            if (held && !dut.src_ready) ++m.source_wait;
            if (held && dut.src_ready) {++sent; held=false;}
            if (dut.done) {
                if (sent!=c.channels || received!=c.positions)
                    throw std::runtime_error("done without all inputs/outputs");
                dut.src_valid=0;
                return m;
            }
            edge();
        }
        throw std::runtime_error("tile timeout");
    }
};

int main(int argc, char** argv) {
    Verilated::commandArgs(argc,argv);
    if (argc!=3) {std::cerr<<"usage: scoreboard cases.bin result.tsv\n";return 2;}
    try {
        std::ifstream input(argv[1],std::ios::binary);
        const auto count=scalar<uint32_t>(input);
        std::vector<Case> cases;
        for (unsigned i=0;i<count;++i) cases.push_back(read_case(input));
        std::ofstream output(argv[2]);
        output<<"case\tbackpressure\tcycles\tfirst_output\tsource_wait\toutput_stall\tgate_decisions\n";
        Simulation sim;
        uint64_t decisions=0;
        for (bool gaps:{false,true}) for (const auto& c:cases) {
            const auto m=sim.run(c,gaps); decisions+=m.decisions;
            output<<c.name<<'\t'<<gaps<<'\t'<<m.cycles<<'\t'<<m.first_output<<'\t'
                  <<m.source_wait<<'\t'<<m.output_stall<<'\t'<<m.decisions<<'\n';
            std::cout<<"PASS "<<c.name<<" backpressure="<<gaps<<" cycles="<<m.cycles<<'\n';
        }
        sim.dut.final();
        std::cout<<"PASS "<<count*2<<" tile replays, "<<decisions<<" gate comparisons\n";
    } catch (const std::exception& e) {
        std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1;
    }
}
