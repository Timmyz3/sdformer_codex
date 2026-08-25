`timescale 1ns/1ps
`default_nettype none

// Turn one packed binary activation tile into a source-owned weight stream.
// Local emits every current one-bit with +W.  Motion emits the XOR frontier:
// 0->1 as +W and 1->0 as -W.  The output can directly address a weight column
// and multicast that column across resident output accumulators.
module qfit_dual_line_source_streamer #(
    parameter int TILE_BITS = 256,
    parameter int TAG_W = 24,
    parameter int INDEX_W = (TILE_BITS <= 1) ? 1 : $clog2(TILE_BITS),
    parameter int COUNT_W = $clog2(TILE_BITS + 1),
    parameter int PERF_W = 32
) (
    input  logic                     clk_core,
    input  logic                     rst_core,

    input  logic                     command_valid,
    output logic                     command_ready,
    input  logic [TAG_W-1:0]         command_tag,
    input  logic                     command_use_motion,
    input  logic [TILE_BITS-1:0]     command_current_bits,
    input  logic [TILE_BITS-1:0]     command_previous_bits,

    output logic                     source_valid,
    input  logic                     source_ready,
    output logic [TAG_W-1:0]         source_tag,
    output logic [INDEX_W-1:0]       source_index,
    output logic                     source_negative,
    output logic                     source_use_motion,
    output logic                     source_last,

    output logic                     done_valid,
    input  logic                     done_ready,
    output logic [TAG_W-1:0]         done_tag,
    output logic                     done_use_motion,
    output logic [COUNT_W-1:0]       done_source_count,

    output logic [PERF_W-1:0]        perf_commands,
    output logic [PERF_W-1:0]        perf_local_commands,
    output logic [PERF_W-1:0]        perf_motion_commands,
    output logic [PERF_W-1:0]        perf_sources,
    output logic [PERF_W-1:0]        perf_positive_sources,
    output logic [PERF_W-1:0]        perf_negative_sources
);
    typedef enum logic [1:0] {
        ST_IDLE = 2'd0,
        ST_SCAN = 2'd1,
        ST_DONE = 2'd2
    } state_t;

    state_t state_q;
    logic [TAG_W-1:0] tag_q;
    logic use_motion_q;
    logic [TILE_BITS-1:0] remaining_q;
    logic [TILE_BITS-1:0] negative_q;
    logic [COUNT_W-1:0] emitted_q;
    logic selected_valid;
    logic [INDEX_W-1:0] selected_index;
    logic command_fire;
    logic source_fire;
    logic done_fire;

    always_comb begin
        selected_valid = 1'b0;
        selected_index = '0;
        for (integer bit_index = 0; bit_index < TILE_BITS; bit_index = bit_index + 1) begin
            if (!selected_valid && remaining_q[bit_index]) begin
                selected_valid = 1'b1;
                selected_index = INDEX_W'(bit_index);
            end
        end
    end

    assign command_ready = state_q == ST_IDLE;
    assign command_fire = command_valid && command_ready;
    assign source_valid = state_q == ST_SCAN && selected_valid;
    assign source_fire = source_valid && source_ready;
    assign source_tag = tag_q;
    assign source_index = selected_index;
    assign source_negative = source_valid && negative_q[selected_index];
    assign source_use_motion = use_motion_q;
    assign source_last = source_valid
                       && (remaining_q & (remaining_q - TILE_BITS'(1))) == '0;
    assign done_valid = state_q == ST_DONE;
    assign done_fire = done_valid && done_ready;
    assign done_tag = tag_q;
    assign done_use_motion = use_motion_q;
    assign done_source_count = emitted_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            tag_q <= '0;
            use_motion_q <= 1'b0;
            remaining_q <= '0;
            negative_q <= '0;
            emitted_q <= '0;
            perf_commands <= '0;
            perf_local_commands <= '0;
            perf_motion_commands <= '0;
            perf_sources <= '0;
            perf_positive_sources <= '0;
            perf_negative_sources <= '0;
        end else begin
            if (command_fire) begin : capture_command
                logic [TILE_BITS-1:0] selected;
                selected = command_use_motion
                    ? (command_current_bits ^ command_previous_bits)
                    : command_current_bits;
                tag_q <= command_tag;
                use_motion_q <= command_use_motion;
                remaining_q <= selected;
                negative_q <= command_use_motion
                    ? (command_previous_bits & ~command_current_bits)
                    : '0;
                emitted_q <= '0;
                state_q <= selected == '0 ? ST_DONE : ST_SCAN;
            end

            if (source_fire) begin
                remaining_q[selected_index] <= 1'b0;
                emitted_q <= emitted_q + COUNT_W'(1);
                if (source_last)
                    state_q <= ST_DONE;
            end

            if (done_fire) begin
                perf_commands <= perf_commands + PERF_W'(1);
                perf_sources <= perf_sources + PERF_W'(emitted_q);
                if (use_motion_q)
                    perf_motion_commands <= perf_motion_commands + PERF_W'(1);
                else
                    perf_local_commands <= perf_local_commands + PERF_W'(1);
                state_q <= ST_IDLE;
            end

            if (source_fire) begin
                if (source_negative)
                    perf_negative_sources <= perf_negative_sources + PERF_W'(1);
                else
                    perf_positive_sources <= perf_positive_sources + PERF_W'(1);
            end
        end
    end
endmodule

`default_nettype wire
