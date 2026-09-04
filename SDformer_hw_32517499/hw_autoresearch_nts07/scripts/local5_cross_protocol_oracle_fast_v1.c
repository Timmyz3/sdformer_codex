#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FNV1A64_SEED UINT64_C(0xcbf29ce484222325)
#define FNV1A64_PRIME UINT64_C(0x00000100000001b3)
#define DJB2XOR64_SEED UINT64_C(0x00001505d3c4b2a1)

static const char *const DOMAIN_TAG = "LOCAL5_PHASE_SUMMARY_V2";
static const char *const LEDGER_SCHEMA = "local5_cross_acc_protocol_ledger_v2";
static const char *const LEDGER_RESOURCE = "CROSS_ACC_PROTOCOL_LEDGER";

struct rolling_pair {
    uint64_t first;
    uint64_t second;
};

static void rolling_update_byte(struct rolling_pair *state, uint8_t value)
{
    state->first = (state->first ^ value) * FNV1A64_PRIME;
    state->second = ((state->second << 5) + state->second) ^ value;
}

static void rolling_update_bytes(struct rolling_pair *state,
                                 const uint8_t *data,
                                 size_t length)
{
    size_t index;

    for (index = 0; index < length; ++index)
        rolling_update_byte(state, data[index]);
}

static void rolling_update_u16le(struct rolling_pair *state, uint16_t value)
{
    rolling_update_byte(state, (uint8_t)(value & UINT16_C(0xff)));
    rolling_update_byte(state, (uint8_t)((value >> 8) & UINT16_C(0xff)));
}

static void rolling_update_u64le(struct rolling_pair *state, uint64_t value)
{
    unsigned int shift;

    for (shift = 0; shift < 64; shift += 8)
        rolling_update_byte(state, (uint8_t)((value >> shift) & UINT64_C(0xff)));
}

static int rolling_update_length_prefixed(struct rolling_pair *state,
                                          const char *value)
{
    size_t length = strlen(value);

    if (length == 0 || length > UINT16_MAX)
        return -1;
    rolling_update_u16le(state, (uint16_t)length);
    rolling_update_bytes(state, (const uint8_t *)value, length);
    return 0;
}

static int parse_positive_u64(const char *text, uint64_t *value)
{
    char *end = NULL;
    unsigned long long parsed;

    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0)
        return -1;
    *value = (uint64_t)parsed;
    return 0;
}

static int compare_u64(const void *left, const void *right)
{
    const uint64_t a = *(const uint64_t *)left;
    const uint64_t b = *(const uint64_t *)right;

    return (a > b) - (a < b);
}

static int read_u64le(FILE *stream, uint64_t *value)
{
    uint8_t bytes[8];
    unsigned int index;
    uint64_t decoded = 0;

    if (fread(bytes, 1, sizeof(bytes), stream) != sizeof(bytes))
        return -1;
    for (index = 0; index < sizeof(bytes); ++index)
        decoded |= ((uint64_t)bytes[index]) << (8 * index);
    *value = decoded;
    return 0;
}

static void update_event(struct rolling_pair *state,
                         uint64_t sequence,
                         uint64_t rw,
                         uint64_t address)
{
    rolling_update_u64le(state, sequence);
    rolling_update_u64le(state, rw);
    rolling_update_u64le(state, address);
}

static int check_product(uint64_t left, uint64_t right, uint64_t *product)
{
    if (left != 0 && right > UINT64_MAX / left)
        return -1;
    *product = left * right;
    return 0;
}

static int verify_no_trailing_bytes(FILE *stream)
{
    uint8_t byte;
    size_t count = fread(&byte, 1, 1, stream);

    if (count != 0)
        return -1;
    return ferror(stream) ? -1 : 0;
}

int main(int argc, char **argv)
{
    struct rolling_pair state = {FNV1A64_SEED, DJB2XOR64_SEED};
    const char *target_instance;
    uint64_t heads;
    uint64_t output_tiles;
    uint64_t addresses_per_tile;
    uint64_t half_count;
    uint64_t total_count;
    uint64_t sequence = 0;
    uint64_t reads = 0;
    uint64_t writes = 0;
    uint64_t tile;
    FILE *order_stream = NULL;
    uint64_t *order = NULL;
    uint64_t *sorted = NULL;

    if (argc != 5 && argc != 6) {
        fprintf(stderr,
                "usage: %s TARGET HEADS OUTPUT_TILES ADDRESSES_PER_TILE [ORDER_U64LE]\n",
                argv[0]);
        return 2;
    }
    target_instance = argv[1];
    if (parse_positive_u64(argv[2], &heads) != 0 ||
        parse_positive_u64(argv[3], &output_tiles) != 0 ||
        parse_positive_u64(argv[4], &addresses_per_tile) != 0) {
        fprintf(stderr, "dimensions must be positive unsigned integers\n");
        return 2;
    }
    if (strlen(target_instance) == 0 || strlen(target_instance) > UINT16_MAX) {
        fprintf(stderr, "target instance length is invalid\n");
        return 2;
    }
    if (check_product(heads, output_tiles, &half_count) != 0 ||
        check_product(half_count, addresses_per_tile, &half_count) != 0 ||
        check_product(half_count, UINT64_C(2), &total_count) != 0) {
        fprintf(stderr, "event count overflows uint64\n");
        return 2;
    }
    if (addresses_per_tile > SIZE_MAX / sizeof(*order)) {
        fprintf(stderr, "address order is too large\n");
        return 2;
    }
    if (argc == 6) {
        order_stream = fopen(argv[5], "rb");
        if (order_stream == NULL) {
            fprintf(stderr, "cannot open address-order file: %s\n", strerror(errno));
            return 2;
        }
        order = malloc((size_t)addresses_per_tile * sizeof(*order));
        sorted = malloc((size_t)addresses_per_tile * sizeof(*sorted));
        if (order == NULL || sorted == NULL) {
            fprintf(stderr, "cannot allocate address-order buffers\n");
            fclose(order_stream);
            free(order);
            free(sorted);
            return 2;
        }
    }
    if (rolling_update_length_prefixed(&state, DOMAIN_TAG) != 0 ||
        rolling_update_length_prefixed(&state, LEDGER_SCHEMA) != 0 ||
        rolling_update_length_prefixed(&state, LEDGER_RESOURCE) != 0 ||
        rolling_update_length_prefixed(&state, target_instance) != 0) {
        fprintf(stderr, "ledger prefix is invalid\n");
        return 2;
    }

    for (tile = 0; tile < output_tiles; ++tile) {
        uint64_t index;
        uint64_t head;

        if (order_stream != NULL) {
            for (index = 0; index < addresses_per_tile; ++index) {
                if (read_u64le(order_stream, &order[index]) != 0) {
                    fprintf(stderr, "address-order file is truncated at tile %" PRIu64 "\n", tile);
                    goto fail;
                }
            }
            memcpy(sorted, order, (size_t)addresses_per_tile * sizeof(*order));
            qsort(sorted, (size_t)addresses_per_tile, sizeof(*sorted), compare_u64);
            for (index = 1; index < addresses_per_tile; ++index) {
                if (sorted[index - 1] == sorted[index]) {
                    fprintf(stderr, "address-order file aliases an address at tile %" PRIu64 "\n", tile);
                    goto fail;
                }
            }
        }

        for (index = 0; index < addresses_per_tile; ++index) {
            uint64_t address = order_stream == NULL ? index : order[index];
            update_event(&state, sequence++, UINT64_C(1), address);
            ++writes;
        }
        for (head = 1; head < heads; ++head) {
            for (index = 0; index < addresses_per_tile; ++index) {
                uint64_t address = order_stream == NULL ? index : order[index];
                update_event(&state, sequence++, UINT64_C(0), address);
                ++reads;
                update_event(&state, sequence++, UINT64_C(1), address);
                ++writes;
            }
        }
        for (index = 0; index < addresses_per_tile; ++index) {
            uint64_t address = order_stream == NULL ? index : order[index];
            update_event(&state, sequence++, UINT64_C(0), address);
            ++reads;
        }
    }

    if (order_stream != NULL && verify_no_trailing_bytes(order_stream) != 0) {
        fprintf(stderr, "address-order file has trailing bytes or an I/O error\n");
        goto fail;
    }
    if (sequence != total_count || reads != half_count || writes != half_count) {
        fprintf(stderr, "internal closed-form count mismatch\n");
        goto fail;
    }
    if (order_stream != NULL)
        fclose(order_stream);
    free(order);
    free(sorted);
    printf("{\"schema\":\"local5_cross_protocol_fast_oracle_v1\","
           "\"count\":%" PRIu64 ",\"read_count\":%" PRIu64
           ",\"write_count\":%" PRIu64 ",\"digest0\":\"%016" PRIx64
           "\",\"digest1\":\"%016" PRIx64 "\"}\n",
           sequence, reads, writes, state.first, state.second);
    return 0;

fail:
    if (order_stream != NULL)
        fclose(order_stream);
    free(order);
    free(sorted);
    return 2;
}
