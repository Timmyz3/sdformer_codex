#include "Vmetadata_primitive.h"
#include "verilated.h"
#include <array>
#include <cstdint>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Request {
    unsigned operation = 0, group = 0, valid = 0;
    uint32_t summary = 0, geometry0 = 0, geometry1 = 0;
    std::array<uint16_t, 8> gate{};
};
struct Result {
    unsigned operation, occupied, permission;
    uint32_t summary;
    bool operator==(const Result& b) const {
        return operation == b.operation && occupied == b.occupied &&
               permission == b.permission && summary == b.summary;
    }
};
struct Expected { Result result; uint64_t issued; };

// Independent lane-wise reference from the stated bit contract.
Result reference(const Request& r) {
    Result e{r.operation, 0, 0, 0};
    if (r.operation < 2) {
        e.summary = r.summary & 0xffffffu;
        const unsigned lanes = r.operation == 0 ? 8 : 4;
        for (unsigned lane = 0; lane < lanes; ++lane) {
            bool any_time = false;
            for (unsigned time = 0; time < 10; ++time)
                any_time = any_time || ((r.gate[lane] >> time) & 1u);
            if (any_time) {
                e.occupied |= 1u << (lane / 4);
                e.summary |= 1u << (r.group + lane / 4);
            }
        }
    } else {
        const uint32_t geometry[2] = {r.geometry0, r.geometry1};
        for (unsigned position = 0; position < 2; ++position)
            if ((r.valid & (1u << position)) &&
                (geometry[position] & (1u << r.group)))
                e.permission |= 1u << position;
    }
    return e;
}

uint32_t random_word() {
    static uint32_t state = 0x20260913u;
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}
void require(bool condition, const std::string& reason) {
    if (!condition) throw std::runtime_error(reason);
}

class Harness {
    Vmetadata_primitive dut;
    std::deque<Expected> expected;
    Result held{};
    bool was_stalled = false;
    uint64_t previous_accept = UINT64_MAX;
    Result observed() const {
        return Result{dut.operation_o, dut.occupied_o, dut.permission_o, dut.summary_o};
    }
public:
    uint64_t cycles = 0, accepted = 0, retired = 0, stalled = 0;
    uint64_t consecutive_accepts = 0, reset_discarded = 0;
    uint64_t rf_writes = 0, last_rf_write_boundary = 0, last_rf_issue = 0;
    uint32_t summary_rf = 0;
    bool track_rf = false;

    Harness() { reset(); }
    void reset() {
        reset_discarded += expected.size(); expected.clear();
        dut.clk = 0; dut.rst_n = 0; dut.in_valid = 0; dut.out_ready = 0;
        dut.eval(); dut.clk = 1; dut.eval(); dut.clk = 0; dut.eval();
        require(!dut.out_valid, "reset failed to invalidate pending result");
        dut.rst_n = 1; dut.eval(); was_stalled = false;
    }
    bool empty() const { return expected.empty(); }
    bool tick(const Request* request, bool ready) {
        dut.clk = 0; dut.in_valid = request != nullptr; dut.out_ready = ready;
        if (request) {
            dut.operation_i = request->operation; dut.group_i = request->group;
            dut.summary_i = request->summary;
            dut.geometry0_i = request->geometry0; dut.geometry1_i = request->geometry1;
            dut.position_valid_i = request->valid;
            for (unsigned i = 0; i < 4; ++i)
                dut.gate_words_i[i] = uint32_t(request->gate[2*i]) |
                                     (uint32_t(request->gate[2*i+1]) << 16);
        }
        dut.eval();
        if (was_stalled)
            require(dut.out_valid && observed() == held,
                    "output changed or vanished under backpressure");
        was_stalled = dut.out_valid && !ready;
        if (was_stalled) { held = observed(); ++stalled; }

        // The existing RF writeback wrapper consumes the registered result.
        // Issue in slot t -> result at boundary t+1 -> RF at boundary t+2.
        if (dut.out_valid && ready) {
            require(!expected.empty(), "unissued or zero-latency result");
            Expected e = expected.front(); expected.pop_front();
            require(observed() == e.result, "primitive disagreed with lane-wise reference");
            require(cycles >= e.issued + 1, "registered result arrived too early");
            if (track_rf && e.result.operation < 2) {
                summary_rf = dut.summary_o; ++rf_writes;
                last_rf_write_boundary = cycles + 1; last_rf_issue = e.issued;
            }
            ++retired;
        }
        bool took = request && dut.in_ready;
        if (took) {
            expected.push_back(Expected{reference(*request), cycles}); ++accepted;
            if (previous_accept != UINT64_MAX && previous_accept + 1 == cycles)
                ++consecutive_accepts;
            previous_accept = cycles;
        }
        require(expected.size() <= 1, "more than one result register in flight");
        dut.clk = 1; dut.eval();
        require(bool(dut.out_valid) == !expected.empty(), "result-register valid mismatch");
        if (!expected.empty())
            require(observed() == expected.front().result, "incorrect registered result");
        dut.clk = 0; dut.eval(); ++cycles;
        return took;
    }
    void send(const Request& r, unsigned stalls = 0) {
        unsigned attempts = 0;
        while (!tick(&r, true)) require(++attempts < 16, "input deadlock");
        for (unsigned i = 0; i < stalls; ++i) tick(nullptr, false);
        tick(nullptr, true);
        require(empty(), "transaction failed to drain");
    }
};

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    try {
        Harness h;
        std::vector<Request> cases;
        // Every bit of every uint16 lane, at every producer summary pair.
        // Includes all 48 upper/reserved bits, which must contribute nothing.
        for (unsigned group = 0; group < 24; group += 2)
            for (unsigned bit = 0; bit < 128; ++bit) {
                Request r; r.group = group; r.summary = random_word() & 0xffffffu;
                r.gate[bit/16] = uint16_t(1u << (bit%16)); cases.push_back(r);
            }
        // Consumer-built summary mode consumes only the low 64 input bits.
        for (unsigned group = 0; group < 24; ++group)
            for (unsigned bit = 0; bit < 128; ++bit) {
                Request r; r.operation = 1; r.group = group;
                r.summary = random_word() & 0xffffffu;
                r.gate[bit/16] = uint16_t(1u << (bit%16)); cases.push_back(r);
            }
        // Every indexed permission truth table, including invalid positions
        // whose metadata bits are one and valid positions whose bits are zero.
        for (unsigned group = 0; group < 24; ++group)
            for (unsigned valid = 0; valid < 4; ++valid)
                for (unsigned bits = 0; bits < 4; ++bits) {
                    Request r; r.operation = 2; r.group = group; r.valid = valid;
                    r.geometry0 = (random_word() & 0xffffffu & ~(1u << group)) |
                                  (((bits & 1u) != 0) ? 1u << group : 0);
                    r.geometry1 = (random_word() & 0xffffffu & ~(1u << group)) |
                                  (((bits & 2u) != 0) ? 1u << group : 0);
                    cases.push_back(r);
                }
        for (unsigned i = 0; i < 2048; ++i) {
            Request r; r.operation = random_word() % 3;
            r.group = r.operation == 0 ? 2*(random_word()%12) : random_word()%24;
            r.summary = random_word() & 0xffffffu; r.valid = random_word() & 3;
            r.geometry0 = random_word() & 0xffffffu; r.geometry1 = random_word() & 0xffffffu;
            for (auto& word : r.gate) word = uint16_t(random_word());
            cases.push_back(r);
        }
        size_t next = 0;
        while (next < cases.size() || !h.empty()) {
            // Long holds plus full-throughput periods; the request remains
            // stable until in_ready, including simultaneous retire/replace.
            bool ready = (h.cycles % 19) >= 7;
            if (h.tick(next < cases.size() ? &cases[next] : nullptr, ready)) ++next;
            require(h.cycles < 30000, "stream failed to make progress");
        }

        // A real externally owned RF24 model: clear once per pixel, then each
        // H8 pair exactly once. Only a retired DUT result updates this RF.
        h.track_rf = true;
        unsigned pixels = 64;
        for (unsigned pixel = 0; pixel < pixels; ++pixel) {
            h.summary_rf = 0;
            uint32_t pixel_gold = 0;
            for (unsigned group = 0; group < 24; group += 2) {
                Request r; r.summary = h.summary_rf; r.group = group;
                for (unsigned lane = 0; lane < 8; ++lane) {
                    unsigned occupied = (pixel + group + lane/4) % 4;
                    r.gate[lane] = uint16_t(random_word() & 0xfc00u);
                    if (occupied == 1 || occupied == 3)
                        r.gate[lane] |= uint16_t(1u << (random_word()%10));
                    if (r.gate[lane] & 0x3ffu) pixel_gold |= 1u << (group + lane/4);
                }
                unsigned hold = (pixel % 7 == 0) ? 5 : 0;
                h.send(r, hold);
                require(h.summary_rf == pixel_gold, "sequential RF summary mismatch");
                require(h.last_rf_write_boundary == h.last_rf_issue + 2 + hold,
                        "RF writeback did not follow the two-slot wrapper timing");
            }
            // The sidecar format uses one uint64 with only the low 24 bits set.
            uint64_t sidecar = h.summary_rf;
            require((sidecar >> 24) == 0 && sidecar == pixel_gold, "sidecar packing mismatch");
        }
        h.track_rf = false;
        Request pending; pending.gate[0] = 1;
        require(h.tick(&pending, false), "empty result register rejected reset test");
        for (unsigned i = 0; i < 8; ++i) h.tick(nullptr, false);
        h.reset(); h.tick(nullptr, true);
        require(h.accepted == h.retired + h.reset_discarded, "lost or duplicated result");
        require(h.stalled > 100 && h.consecutive_accepts > 100,
                "backpressure/full-throughput coverage was not exercised");
        std::cout << "{\"status\":\"PASS\",\"seed\":\"0x20260913\","
                  << "\"producer_bit_basis_cases\":1536,\"consumer_word_bit_basis_cases\":3072,"
                  << "\"permission_truth_table_cases\":384,\"random_mixed_cases\":2048,"
                  << "\"sequential_pixels\":" << pixels << ",\"sequential_rf_writes\":" << h.rf_writes
                  << ",\"accepted\":" << h.accepted << ",\"retired\":" << h.retired
                  << ",\"reset_discarded\":" << h.reset_discarded
                  << ",\"stalled_result_cycles\":" << h.stalled
                  << ",\"consecutive_accept_pairs\":" << h.consecutive_accepts
                  << ",\"cycles\":" << h.cycles << "}" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << std::endl; return 1;
    }
}
