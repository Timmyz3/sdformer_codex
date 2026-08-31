#include <stddef.h>
#include <stdint.h>

/*
 * Independent O(1)-per-reference fully-associative LRU oracle.
 *
 * active is row-major [destination_count][input_tile_count].  The reference
 * order is destination, output block, increasing active input tile.  Cache
 * state begins empty for each plane, matching the M712 per-row lifetime.
 */
uint64_t m718_lru_misses(const uint8_t *active, size_t destination_count,
                         int input_tile_count, int output_blocks, int capacity,
                         uint64_t *reference_count) {
    enum { MAX_KEYS = 1024 };
    int previous[MAX_KEYS];
    int next[MAX_KEYS];
    uint8_t resident[MAX_KEYS];
    const int key_count = input_tile_count * output_blocks;
    int head = -1;
    int tail = -1;
    int occupancy = 0;
    uint64_t misses = 0;
    uint64_t references = 0;
    size_t destination;
    int key;

    if (!active || !reference_count || input_tile_count <= 0 ||
        output_blocks <= 0 || capacity <= 0 || key_count > MAX_KEYS) {
        if (reference_count) *reference_count = 0;
        return UINT64_MAX;
    }
    for (key = 0; key < key_count; ++key) {
        previous[key] = -1;
        next[key] = -1;
        resident[key] = 0;
    }

    for (destination = 0; destination < destination_count; ++destination) {
        int block;
        for (block = 0; block < output_blocks; ++block) {
            int tile;
            for (tile = 0; tile < input_tile_count; ++tile) {
                int old_head;
                if (!active[destination * (size_t)input_tile_count + tile])
                    continue;
                key = block * input_tile_count + tile;
                ++references;
                if (resident[key]) {
                    if (key == tail) continue;
                    if (previous[key] >= 0) next[previous[key]] = next[key];
                    else head = next[key];
                    if (next[key] >= 0) previous[next[key]] = previous[key];
                    previous[key] = tail;
                    next[key] = -1;
                    next[tail] = key;
                    tail = key;
                    continue;
                }

                ++misses;
                if (occupancy == capacity) {
                    old_head = head;
                    head = next[old_head];
                    if (head >= 0) previous[head] = -1;
                    else tail = -1;
                    resident[old_head] = 0;
                    previous[old_head] = -1;
                    next[old_head] = -1;
                    --occupancy;
                }
                resident[key] = 1;
                previous[key] = tail;
                next[key] = -1;
                if (tail >= 0) next[tail] = key;
                else head = key;
                tail = key;
                ++occupancy;
            }
        }
    }
    *reference_count = references;
    return misses;
}
