`timescale 1ns/1ps
`default_nettype none

// Transaction-level lower-bound baseline for head-major projection. Each head
// is decoded once; partial sums spill between heads and the last head finalizes.
module gatestack_head_major_spill_scheduler #(
    parameter int TOKENS = 162,
    parameter int MAX_HEADS = 24,
    parameter int MAX_OUTPUT_TILES = 24,
    parameter int BANKS = 2,
    parameter int OUT_TILE = 32,
    parameter int ACC_W = 32,
    parameter int TAG_W = 32,
    parameter int COUNTER_W = 32,
    parameter int TOKEN_ID_W = (TOKENS <= 1) ? 1 : $clog2(TOKENS),
    parameter int HEAD_W = (MAX_HEADS <= 1) ? 1 : $clog2(MAX_HEADS),
    parameter int TILE_W = (MAX_OUTPUT_TILES <= 1) ? 1 : $clog2(MAX_OUTPUT_TILES)
) (
    input  logic                         clk_core,
    input  logic                         rst_core,
    input  logic                         start_valid,
    output logic                         start_ready,
    input  logic [TAG_W-1:0]             start_tag,
    input  logic [HEAD_W:0]              start_head_count,
    input  logic [TILE_W:0]              start_output_tile_count,
    output logic                         decode_req_valid,
    input  logic                         decode_req_ready,
    output logic [TAG_W-1:0]             decode_req_tag,
    output logic [HEAD_W-1:0]            decode_req_head,
    input  logic                         decode_done_valid,
    output logic                         decode_done_ready,
    input  logic                         decode_done_error,
    output logic                         spill_read_valid,
    input  logic                         spill_read_ready,
    output logic [TILE_W-1:0]            spill_read_tile,
    output logic [TOKEN_ID_W-1:0]        spill_read_token_base,
    output logic [BANKS-1:0]             spill_read_token_valid,
    output logic                         spill_write_valid,
    input  logic                         spill_write_ready,
    output logic [TILE_W-1:0]            spill_write_tile,
    output logic [TOKEN_ID_W-1:0]        spill_write_token_base,
    output logic [BANKS-1:0]             spill_write_token_valid,
    output logic                         final_valid,
    input  logic                         final_ready,
    output logic [TILE_W-1:0]            final_tile,
    output logic [TOKEN_ID_W-1:0]        final_token_base,
    output logic [BANKS-1:0]             final_token_valid,
    output logic                         done_valid,
    input  logic                         done_ready,
    output logic [TAG_W-1:0]             done_tag,
    output logic                         protocol_error,
    output logic [COUNTER_W-1:0]         count_decodes,
    output logic [COUNTER_W-1:0]         count_spill_reads,
    output logic [COUNTER_W-1:0]         count_spill_writes,
    output logic [COUNTER_W-1:0]         count_final_batches,
    output logic [63:0]                  count_spill_value_bytes
);
    typedef enum logic [2:0] {
        ST_IDLE, ST_DECODE_REQ, ST_DECODE_WAIT, ST_ACCESS, ST_DONE
    } state_t;
    localparam logic [63:0] VALUE_BYTES = 64'(ACC_W) / 64'd8;
    state_t state_q;
    logic [TAG_W-1:0] tag_q;
    logic [HEAD_W:0] head_count_q;
    logic [TILE_W:0] tile_count_q;
    logic [HEAD_W-1:0] head_q;
    logic [TILE_W-1:0] tile_q;
    logic [TOKEN_ID_W-1:0] token_base_q;
    logic read_phase_q;
    logic last_head, last_tile, last_batch;
    logic access_fire, read_fire, write_fire, final_fire;
    logic [BANKS-1:0] token_mask;
    logic [31:0] active_tokens;
    logic [63:0] batch_bytes;

    assign start_ready = state_q == ST_IDLE && start_head_count != 0 &&
                         start_output_tile_count != 0 &&
                         32'(start_head_count) <= MAX_HEADS &&
                         32'(start_output_tile_count) <= MAX_OUTPUT_TILES;
    assign decode_req_valid = state_q == ST_DECODE_REQ;
    assign decode_req_tag = tag_q;
    assign decode_req_head = head_q;
    assign decode_done_ready = state_q == ST_DECODE_WAIT;
    assign last_head = 32'(head_q) + 1 == 32'(head_count_q);
    assign last_tile = 32'(tile_q) + 1 == 32'(tile_count_q);
    assign last_batch = 32'(token_base_q) + BANKS >= TOKENS;

    always_comb begin
        token_mask = '0;
        active_tokens = 0;
        for (int bank = 0; bank < BANKS; bank = bank + 1) begin
            if (32'(token_base_q) + bank < TOKENS) begin
                token_mask[bank] = 1'b1;
                active_tokens = active_tokens + 1;
            end
        end
    end
    assign batch_bytes = 64'(active_tokens) * OUT_TILE * VALUE_BYTES;

    assign spill_read_valid = state_q == ST_ACCESS && head_q != 0 && read_phase_q;
    assign spill_write_valid = state_q == ST_ACCESS && !last_head &&
                               (head_q == 0 || !read_phase_q);
    assign final_valid = state_q == ST_ACCESS && last_head &&
                         (head_q == 0 || !read_phase_q);
    assign spill_read_tile = tile_q;
    assign spill_write_tile = tile_q;
    assign final_tile = tile_q;
    assign spill_read_token_base = token_base_q;
    assign spill_write_token_base = token_base_q;
    assign final_token_base = token_base_q;
    assign spill_read_token_valid = token_mask;
    assign spill_write_token_valid = token_mask;
    assign final_token_valid = token_mask;
    assign read_fire = spill_read_valid && spill_read_ready;
    assign write_fire = spill_write_valid && spill_write_ready;
    assign final_fire = final_valid && final_ready;
    assign access_fire = write_fire || final_fire;
    assign done_valid = state_q == ST_DONE;
    assign done_tag = tag_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            tag_q <= '0;
            head_count_q <= '0;
            tile_count_q <= '0;
            head_q <= '0;
            tile_q <= '0;
            token_base_q <= '0;
            read_phase_q <= 1'b0;
            protocol_error <= 1'b0;
            count_decodes <= '0;
            count_spill_reads <= '0;
            count_spill_writes <= '0;
            count_final_batches <= '0;
            count_spill_value_bytes <= '0;
        end else begin
            if (state_q == ST_IDLE) begin
                if (start_valid && start_ready) begin
                    tag_q <= start_tag;
                    head_count_q <= start_head_count;
                    tile_count_q <= start_output_tile_count;
                    head_q <= '0;
                    tile_q <= '0;
                    token_base_q <= '0;
                    state_q <= ST_DECODE_REQ;
                end else if (start_valid && !start_ready) begin
                    protocol_error <= 1'b1;
                end
            end else if (state_q == ST_DECODE_REQ) begin
                if (decode_req_valid && decode_req_ready)
                    state_q <= ST_DECODE_WAIT;
            end else if (state_q == ST_DECODE_WAIT) begin
                if (decode_done_valid && decode_done_ready) begin
                    count_decodes <= count_decodes + 1'b1;
                    if (decode_done_error)
                        protocol_error <= 1'b1;
                    read_phase_q <= head_q != 0;
                    state_q <= ST_ACCESS;
                end
            end else if (state_q == ST_ACCESS) begin
                if (read_fire) begin
                    read_phase_q <= 1'b0;
                    count_spill_reads <= count_spill_reads + 1'b1;
                    count_spill_value_bytes <= count_spill_value_bytes + batch_bytes;
                end
                if (write_fire) begin
                    count_spill_writes <= count_spill_writes + 1'b1;
                    count_spill_value_bytes <= count_spill_value_bytes + batch_bytes;
                end
                if (final_fire)
                    count_final_batches <= count_final_batches + 1'b1;
                if (access_fire) begin
                    if (!last_batch) begin
                        token_base_q <= token_base_q + TOKEN_ID_W'(BANKS);
                        read_phase_q <= head_q != 0;
                    end else if (!last_tile) begin
                        tile_q <= tile_q + 1'b1;
                        token_base_q <= '0;
                        read_phase_q <= head_q != 0;
                    end else if (!last_head) begin
                        head_q <= head_q + 1'b1;
                        tile_q <= '0;
                        token_base_q <= '0;
                        state_q <= ST_DECODE_REQ;
                    end else begin
                        state_q <= ST_DONE;
                    end
                end
            end else if (state_q == ST_DONE) begin
                if (done_valid && done_ready) begin
                    protocol_error <= 1'b0;
                    state_q <= ST_IDLE;
                end
            end else begin
                protocol_error <= 1'b1;
                state_q <= ST_IDLE;
            end
        end
    end
endmodule

`default_nettype wire
