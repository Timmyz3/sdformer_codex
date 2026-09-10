#include "Vdirect_temporal_code.h"
#include "verilated.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Vector {
    std::array<int16_t, 10> x;
    uint8_t address, expected;
};
struct Group {
    std::string name;
    std::array<int8_t, 30> weight;
    std::array<int32_t, 3> tau;
    std::array<uint8_t, 3> variable, fixed, starts, lengths;
    std::array<uint8_t, 8> mapping;
    std::vector<uint8_t> program;
    std::vector<Vector> vectors;
};

static void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
static uint8_t byte(std::istream& in) {
    char c;
    require(bool(in.get(c)), "truncated case file");
    return static_cast<uint8_t>(c);
}
static uint16_t u16(std::istream& in) {
    uint16_t value = byte(in);
    return value | (uint16_t(byte(in)) << 8);
}
static uint32_t u32(std::istream& in) {
    uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i) value |= uint32_t(byte(in)) << (8*i);
    return value;
}
template<size_t N> static void bytes(std::istream& in, std::array<uint8_t, N>& v) {
    for (auto& x: v) x = byte(in);
}
static std::vector<Group> read_groups(const char* path) {
    std::ifstream in(path, std::ios::binary);
    require(bool(in), "cannot open case file");
    require(u32(in) == 0x31435444U, "wrong case file magic");
    uint32_t count = u32(in);
    std::vector<Group> groups;
    for (uint32_t n = 0; n < count; ++n) {
        Group g;
        uint16_t length = u16(in);
        for (uint16_t i = 0; i < length; ++i) g.name.push_back(char(byte(in)));
        for (auto& w: g.weight) w = static_cast<int8_t>(byte(in));
        for (auto& t: g.tau) t = static_cast<int32_t>(u32(in));
        bytes(in, g.variable); bytes(in, g.fixed); bytes(in, g.mapping);
        bytes(in, g.starts); bytes(in, g.lengths);
        length = u16(in);
        for (uint16_t i = 0; i < length; ++i) g.program.push_back(byte(in));
        uint32_t vectors = u32(in);
        for (uint32_t i = 0; i < vectors; ++i) {
            Vector v;
            for (auto& x: v.x) x = static_cast<int16_t>(u16(in));
            v.address = byte(in); v.expected = byte(in);
            // A second ordinary dot reference never reads the CSD program.
            unsigned address = 0;
            for (unsigned row = 0; row < 3; ++row) {
                int64_t dot = 0;
                for (unsigned t = 0; t < 10; ++t) dot += int64_t(v.x[t])*g.weight[row*10+t];
                bool gate = g.variable[row] ? dot >= g.tau[row] : bool(g.fixed[row]);
                address |= unsigned(gate) << row;
            }
            require(address == v.address && g.mapping[address] == v.expected,
                    g.name+" C++ dot/NumPy reference mismatch");
            g.vectors.push_back(v);
        }
        groups.push_back(g);
    }
    return groups;
}

struct Simulation {
    Vdirect_temporal_code dut;
    uint64_t cycle = 0;
    void settle() { dut.clk = 0; dut.eval(); }
    void tick() {
        settle(); dut.clk = 1; dut.eval(); ++cycle;
    }
    void configuration_clear() {
        dut.cfg_uop_valid = 0; dut.cfg_row_valid = 0; dut.cfg_map_valid = 0;
    }
    void pack(const Vector& v) {
        for (unsigned i = 0; i < 4; ++i) dut.in_values[i] = 0;
        for (unsigned t = 0; t < 10; ++t) {
            uint16_t value = uint16_t(v.x[t]) & 0xfffU;
            for (unsigned bit = 0; bit < 12; ++bit)
                if ((value >> bit) & 1U)
                    dut.in_values[(12*t+bit)/32] |= uint32_t(1) << ((12*t+bit)%32);
        }
    }
    uint64_t configure(const Group& g) {
        settle(); require(!dut.busy, "configuration attempted while busy");
        dut.in_valid = 0; dut.out_ready = 1;
        uint64_t start = cycle;
        configuration_clear();
        for (size_t i = 0; i < g.program.size(); ++i) {
            dut.cfg_uop_valid = 1; dut.cfg_uop_address = i; dut.cfg_uop = g.program[i]; tick();
        }
        configuration_clear();
        for (unsigned row = 0; row < 3; ++row) {
            dut.cfg_row_valid = 1; dut.cfg_row = row;
            dut.cfg_row_start = g.starts[row]; dut.cfg_row_length = g.lengths[row];
            dut.cfg_tau = uint32_t(g.tau[row]) & 0xffffffU;
            dut.cfg_variable = g.variable[row]; dut.cfg_constant = g.fixed[row]; tick();
        }
        configuration_clear();
        for (unsigned i = 0; i < 8; ++i) {
            dut.cfg_map_valid = 1; dut.cfg_map_address = i; dut.cfg_map_value = g.mapping[i]; tick();
        }
        configuration_clear();
        return cycle-start;
    }
};

struct Result {
    uint64_t cycles = 0, latency_sum = 0, latency_min = UINT64_MAX, latency_max = 0;
    uint64_t input_wait = 0, source_gap = 0, output_backpressure = 0;
};
static Result run(Simulation& sim, const Group& g, bool stress, unsigned group_index) {
    Result result;
    size_t sent = 0, received = 0;
    bool offering = false, held_output = false;
    uint8_t held_code = 0;
    std::vector<uint64_t> accepted(g.vectors.size());
    uint64_t start = sim.cycle;
    while (received < g.vectors.size()) {
        require(sim.cycle-start < 1000000, g.name+" timeout");
        if (!offering && sent < g.vectors.size()) {
            bool allow = !stress || ((sim.cycle+sent+group_index)%5 >= 2);
            if (allow) { sim.pack(g.vectors[sent]); offering = true; }
        }
        sim.dut.in_valid = offering;
        sim.dut.out_ready = !stress || ((sim.cycle+3*received+group_index)%7 >= 3);
        sim.settle();
        if (held_output)
            require(sim.dut.out_valid && sim.dut.out_code == held_code,
                    g.name+" output changed under backpressure");
        held_output = sim.dut.out_valid && !sim.dut.out_ready;
        if (held_output) { held_code = sim.dut.out_code; ++result.output_backpressure; }
        if (sim.dut.in_valid && !sim.dut.in_ready) ++result.input_wait;
        if (!sim.dut.in_valid && sim.dut.in_ready && sent < g.vectors.size()) ++result.source_gap;
        bool accept = sim.dut.in_valid && sim.dut.in_ready;
        bool retire = sim.dut.out_valid && sim.dut.out_ready;
        if (retire) {
            require(received < sent, g.name+" output without accepted input");
            if (sim.dut.out_code != g.vectors[received].expected)
                throw std::runtime_error(g.name+" vector "+std::to_string(received)+" expected "+
                                         std::to_string(g.vectors[received].expected)+" got "+std::to_string(sim.dut.out_code));
            uint64_t latency = sim.cycle-accepted[received];
            result.latency_sum += latency;
            result.latency_min = std::min(result.latency_min, latency);
            result.latency_max = std::max(result.latency_max, latency);
            require(latency >= g.program.size()+5, g.name+" impossible early output");
            if (!stress) require(latency == g.program.size()+5, g.name+" unexpected unstalled latency");
            ++received;
        }
        if (accept) { accepted[sent++] = sim.cycle; offering = false; }
        sim.tick();
    }
    sim.dut.in_valid = 0;
    sim.dut.out_ready = 1;
    sim.settle();
    require(!sim.dut.busy && !sim.dut.out_valid, g.name+" failed to return idle");
    result.cycles = sim.cycle-start;
    if (!stress)
        require(result.cycles == g.vectors.size()*(g.program.size()+6), g.name+" unstalled service cycle mismatch");
    return result;
}

int main(int argc, char** argv) {
    try {
        Verilated::commandArgs(argc, argv);
        const char* path = argc > 1 ? argv[1] : "direct_temporal_cases.bin";
        auto groups = read_groups(path);
        Simulation sim;
        sim.configuration_clear(); sim.dut.in_valid = 0; sim.dut.out_ready = 1;
        sim.dut.rst_n = 0; sim.tick(); sim.tick(); sim.dut.rst_n = 1;
        std::ofstream table("direct_temporal_results.tsv");
        table << "mode\tgroup\tvectors\tuops\tconfiguration_cycles\tservice_cycles\tlatency_min\tlatency_max\tinput_wait\tsource_gap\toutput_backpressure\n";
        uint64_t tested = 0, backpressure = 0, source_gaps = 0, configurations = 0;
        for (unsigned mode = 0; mode < 2; ++mode) {
            for (unsigned i = 0; i < groups.size(); ++i) {
                const auto& g = groups[i];
                uint64_t config_cycles = sim.configure(g);
                Result r = run(sim, g, mode == 1, i);
                tested += g.vectors.size(); backpressure += r.output_backpressure;
                source_gaps += r.source_gap; ++configurations;
                table << (mode ? "stalled" : "unstalled") << '\t' << g.name << '\t' << g.vectors.size()
                      << '\t' << g.program.size() << '\t' << config_cycles << '\t' << r.cycles
                      << '\t' << r.latency_min << '\t' << r.latency_max << '\t' << r.input_wait
                      << '\t' << r.source_gap << '\t' << r.output_backpressure << '\n';
            }
        }
        require(backpressure && source_gaps, "stall scenarios were not exercised");
        sim.dut.final();
        std::ofstream summary("direct_temporal_result.json");
        summary << "{\n  \"status\": \"PASS\",\n  \"tested_vectors\": " << tested
                << ",\n  \"unique_vectors\": " << tested/2
                << ",\n  \"configurations_without_reset\": " << configurations
                << ",\n  \"reset_events\": 1,\n  \"mismatches\": 0,\n  \"output_backpressure_cycles\": " << backpressure
                << ",\n  \"idle_source_gap_cycles\": " << source_gaps
                << ",\n  \"total_simulated_cycles\": " << sim.cycle
                << ",\n  \"boundary\": \"Verilator source encoding leaf only; not GP layer cycles or EDA\"\n}\n";
        std::cout << "PASS " << tested << " vectors, " << configurations << " configurations without reset, "
                  << backpressure << " backpressure cycles, " << source_gaps << " source-gap cycles\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n'; return 1;
    }
    return 0;
}
