#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int T = 10;
constexpr int C = 768;
constexpr int H = 15;
constexpr int W = 20;
constexpr int SPATIAL = H * W;
constexpr int TILE_BITS = 256;
constexpr int TILES = (C * 9 + TILE_BITS - 1) / TILE_BITS;
constexpr int OUTPUT_BLOCKS = 8;
constexpr int WINDOWS[7] = {1, 2, 4, 8, 16, 32, 64};
constexpr std::uint64_t BANK_MASK = 0x0101010101010101ULL;

struct Mask {
    std::array<std::uint64_t, 4> q{{0, 0, 0, 0}};
};

struct Metric {
    int cycles = 0;
    int population = 0;
};

struct Choice {
    Metric metric;
    int priority = 0;
    int origin = 0;  // zero,left,up,previous,window
    int distance = 0;
    const Mask* parent = nullptr;
};

struct Totals {
    std::uint64_t source = 0;
    std::uint64_t logical = 0;
    std::uint64_t add = 0;
    std::uint64_t subtract = 0;
    std::uint64_t exact_copy = 0;
    std::uint64_t zero_parent = 0;
    std::uint64_t nonzero_parent = 0;
    std::uint64_t queries = 0;
    std::uint64_t matcher = 0;
    std::uint64_t lower = 0;
    std::array<std::uint64_t, 5> origins{{0, 0, 0, 0, 0}};
    int maximum_window_distance = 0;
};

int pop64(std::uint64_t value) {
    return __builtin_popcountll(value);
}

bool is_zero(const Mask& value) {
    return (value.q[0] | value.q[1] | value.q[2] | value.q[3]) == 0;
}

Metric metric(const Mask& current, const Mask& parent) {
    int maximum = 0;
    int population = 0;
    for (int bank = 0; bank < 8; ++bank) {
        int count = 0;
        const std::uint64_t bank_mask = BANK_MASK << bank;
        for (int word = 0; word < 4; ++word) {
            count += pop64((current.q[word] ^ parent.q[word]) & bank_mask);
        }
        maximum = std::max(maximum, count);
        population += count;
    }
    return {maximum, population};
}

bool better(const Choice& candidate, const Choice& incumbent) {
    if (candidate.metric.cycles != incumbent.metric.cycles)
        return candidate.metric.cycles < incumbent.metric.cycles;
    if (candidate.metric.population != incumbent.metric.population)
        return candidate.metric.population < incumbent.metric.population;
    return candidate.priority < incumbent.priority;
}

int signed_add_count(const Mask& current, const Mask& parent) {
    int count = 0;
    for (int word = 0; word < 4; ++word)
        count += pop64(current.q[word] & ~parent.q[word]);
    return count;
}

int signed_subtract_count(const Mask& current, const Mask& parent) {
    int count = 0;
    for (int word = 0; word < 4; ++word)
        count += pop64(parent.q[word] & ~current.q[word]);
    return count;
}

std::vector<Mask> unpack(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open packed source: " + path);
    std::vector<unsigned char> raw((std::istreambuf_iterator<char>(input)),
                                   std::istreambuf_iterator<char>());
    constexpr std::size_t PLANE_BYTES = (T * C * SPATIAL) / 8;
    if (raw.size() != 3 * PLANE_BYTES)
        throw std::runtime_error("packed source extent drift");
    for (std::size_t index = PLANE_BYTES; index < 2 * PLANE_BYTES; ++index)
        if (raw[index] != 0)
            throw std::runtime_error("negative plane is nonzero");
    std::vector<Mask> masks(T * SPATIAL * TILES);
    const std::size_t total_bits = T * C * SPATIAL;
    for (std::size_t byte_index = 0; byte_index < PLANE_BYTES; ++byte_index) {
        unsigned value = raw[byte_index];
        while (value != 0) {
            const unsigned bit = __builtin_ctz(value);
            const std::size_t flat = byte_index * 8 + bit;
            if (flat >= total_bits) throw std::runtime_error("nonzero tail bit");
            const int spatial = static_cast<int>(flat % SPATIAL);
            const std::size_t tc = flat / SPATIAL;
            const int channel = static_cast<int>(tc % C);
            const int timestep = static_cast<int>(tc / C);
            const int input_y = spatial / W;
            const int input_x = spatial % W;
            for (int kernel_y = 0; kernel_y < 3; ++kernel_y) {
                const int output_y = input_y - kernel_y + 1;
                if (output_y < 0 || output_y >= H) continue;
                for (int kernel_x = 0; kernel_x < 3; ++kernel_x) {
                    const int output_x = input_x - kernel_x + 1;
                    if (output_x < 0 || output_x >= W) continue;
                    const int feature = channel * 9 + kernel_y * 3 + kernel_x;
                    const int tile = feature / TILE_BITS;
                    const int tile_bit = feature % TILE_BITS;
                    const int row = timestep * SPATIAL + output_y * W + output_x;
                    masks[row * TILES + tile].q[tile_bit / 64] |=
                        std::uint64_t(1) << (tile_bit % 64);
                }
            }
            value &= value - 1;
        }
    }
    return masks;
}

void add_choice(Totals& total, const Mask& current, const Choice& choice) {
    const int add = signed_add_count(current, *choice.parent);
    const int subtract = signed_subtract_count(current, *choice.parent);
    if (add + subtract != choice.metric.population)
        throw std::runtime_error("signed conservation mismatch");
    total.source += std::uint64_t(choice.metric.cycles) * OUTPUT_BLOCKS;
    total.logical += std::uint64_t(choice.metric.population) * OUTPUT_BLOCKS;
    total.add += std::uint64_t(add) * OUTPUT_BLOCKS;
    total.subtract += std::uint64_t(subtract) * OUTPUT_BLOCKS;
    total.exact_copy += choice.metric.population == 0 && !is_zero(*choice.parent);
    total.zero_parent += is_zero(*choice.parent);
    total.nonzero_parent += !is_zero(*choice.parent);
    ++total.origins.at(choice.origin);
    if (choice.origin == 4)
        total.maximum_window_distance = std::max(total.maximum_window_distance,
                                                 choice.distance);
}

void analyze(const std::vector<Mask>& masks, std::uint64_t& local,
             std::uint64_t& canonical, std::array<Totals, 7>& totals) {
    static const Mask zero;
    for (int timestep = 0; timestep < T; ++timestep) {
        const int row_base = timestep * SPATIAL;
        for (int tile = 0; tile < TILES; ++tile) {
            std::array<std::uint64_t, 7> tile_source{{0, 0, 0, 0, 0, 0, 0}};
            for (int spatial = 0; spatial < SPATIAL; ++spatial) {
                const int row = row_base + spatial;
                const int index = row * TILES + tile;
                const Mask& current = masks.at(index);
                local += std::uint64_t(metric(current, zero).cycles) * OUTPUT_BLOCKS;

                Choice canonical_choice{metric(current, zero), 0, 0, 0, &zero};
                const int x = spatial % W;
                const int y = spatial / W;
                if (x > 0) {
                    Choice candidate{metric(current, masks.at(index - TILES)),
                                     1, 1, 1, &masks.at(index - TILES)};
                    if (better(candidate, canonical_choice)) canonical_choice = candidate;
                }
                if (y > 0) {
                    const int parent_index = index - W * TILES;
                    Choice candidate{metric(current, masks.at(parent_index)),
                                     2, 2, W, &masks.at(parent_index)};
                    if (better(candidate, canonical_choice)) canonical_choice = candidate;
                }
                if (timestep > 0) {
                    const int parent_index = index - SPATIAL * TILES;
                    Choice candidate{metric(current, masks.at(parent_index)),
                                     3, 3, SPATIAL, &masks.at(parent_index)};
                    if (better(candidate, canonical_choice)) canonical_choice = candidate;
                }
                canonical += std::uint64_t(canonical_choice.metric.cycles) * OUTPUT_BLOCKS;

                std::array<Choice, 7> selected;
                selected.fill(canonical_choice);
                Choice best = canonical_choice;
                const int limit = std::min(64, spatial);
                for (int distance = 1; distance <= limit; ++distance) {
                    const int parent_index = (row_base + spatial - distance) * TILES + tile;
                    Choice candidate{metric(current, masks.at(parent_index)),
                                     4 + distance, 4, distance,
                                     &masks.at(parent_index)};
                    if (better(candidate, best)) best = candidate;
                    for (int wi = 0; wi < 7; ++wi)
                        if (std::min(WINDOWS[wi], spatial) == distance)
                            selected[wi] = best;
                }
                for (int wi = 0; wi < 7; ++wi) {
                    add_choice(totals[wi], current, selected[wi]);
                    tile_source[wi] +=
                        std::uint64_t(selected[wi].metric.cycles) * OUTPUT_BLOCKS;
                }
            }
            for (int wi = 0; wi < 7; ++wi) {
                int log2_window = 0;
                for (int value = WINDOWS[wi]; value > 1; value >>= 1)
                    ++log2_window;
                const std::uint64_t matcher = SPATIAL + log2_window + 3;
                totals[wi].queries += SPATIAL;
                totals[wi].matcher += matcher;
                totals[wi].lower += std::max(tile_source[wi], matcher);
            }
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error("expected four sample-0 files");
        std::uint64_t local = 0;
        std::uint64_t canonical = 0;
        std::array<Totals, 7> totals;
        for (int arg = 1; arg < argc; ++arg) {
            const auto masks = unpack(argv[arg]);
            analyze(masks, local, canonical, totals);
        }
        std::cout << "LOCAL " << local << "\n";
        std::cout << "CANONICAL " << canonical << "\n";
        for (int wi = 0; wi < 7; ++wi) {
            const Totals& item = totals[wi];
            std::cout << "W " << WINDOWS[wi] << " " << item.source << " "
                      << item.logical << " " << item.add << " "
                      << item.subtract << " " << item.exact_copy << " "
                      << item.zero_parent << " " << item.nonzero_parent << " "
                      << item.queries << " " << item.matcher << " "
                      << item.lower;
            for (auto count : item.origins) std::cout << " " << count;
            std::cout << " " << item.maximum_window_distance << "\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL M69 sample0 independent replay: " << error.what() << "\n";
        return 1;
    }
}
