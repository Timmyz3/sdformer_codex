`timescale 1ns/1ps
`default_nettype none

// M231 converts the exact two-time-row x 16-channel event word produced by
// M167 BACK (or an equivalent binary ATLIF producer) into the raw4 x 96-bit
// token ingress consumed by M216.  Two pair slots decouple the phase-major
// ATLIF replay from FC2 backpressure.  Storage is bounded by 4*INPUT_WIDTH
// bits; no spatial feature map is materialized in this bridge.
module m231_atlif32_to_fc2_raw4_pingpong_bridge #(
    parameter int INPUT_WIDTH = 384,
    parameter int TAG_BITS = 24
) (
    input  logic                    clk_core,
    input  logic                    rst_core,

    input  logic                    pair_header_valid,
    output logic                    pair_header_ready,
    input  logic [TAG_BITS-2:0]     pair_header_tag,
    output logic                    pair_header_accept,

    input  logic                    event_valid,
    output logic                    event_ready,
    input  logic [TAG_BITS-2:0]     event_pair_tag,
    input  logic [7:0]              event_group_index,
    input  logic [31:0]             event_bits,
    input  logic                    event_last_group,
    output logic                    event_accept,

    output logic                    header_valid,
    input  logic                    header_ready,
    output logic [TAG_BITS-1:0]     header_tag,
    output logic [5:0]              header_raw_beat_count,
    output logic [3:0]              header_window_depth,
    output logic [3:0]              header_output_blocks,
    output logic                    header_accept,

    output logic                    raw_valid,
    input  logic                    raw_ready,
    output logic [3:0]              raw_lane_valid,
    output logic [4:0]              raw_beat_index [0:3],
    output logic [95:0]             raw_bitmap [0:3],
    output logic                    raw_last,
    output logic                    raw_accept,

    output logic                    protocol_error,
    output logic                    busy,
    output logic [1:0]              debug_full_slots,
    output logic [7:0]              debug_fill_group,
    output logic [31:0]             debug_pair_count,
    output logic [31:0]             debug_token_count,
    output logic [31:0]             debug_raw_packet_count
);
    localparam int GROUPS = INPUT_WIDTH / 16;
    localparam int RAW_BEATS = INPUT_WIDTH / 96;
    localparam int RAW_PACKETS = RAW_BEATS / 4;
    localparam int OUTPUT_BLOCKS = INPUT_WIDTH / 384;
    localparam bit PARAMETERS_LEGAL = TAG_BITS >= 2
        && (INPUT_WIDTH == 384 || INPUT_WIDTH == 768
            || INPUT_WIDTH == 1536 || INPUT_WIDTH == 3072)
        && GROUPS <= 256 && RAW_BEATS <= 32 && RAW_PACKETS <= 8
        && OUTPUT_BLOCKS <= 8;
    localparam logic [1:0] DRAIN_IDLE = 2'd0;
    localparam logic [1:0] DRAIN_HEADER = 2'd1;
    localparam logic [1:0] DRAIN_RAW = 2'd2;

    logic [INPUT_WIDTH-1:0] row0_q [0:1];
    logic [INPUT_WIDTH-1:0] row1_q [0:1];
    logic [TAG_BITS-2:0] pair_tag_q [0:1];
    logic [1:0] full_q;
    logic fill_ptr_q, drain_ptr_q;
    logic fill_active_q;
    logic [7:0] fill_group_q;
    logic [TAG_BITS-2:0] fill_tag_q;
    logic [1:0] drain_state_q;
    logic drain_row_q;
    logic [2:0] drain_packet_q;
    logic fault_q;
    logic event_shape_legal, illegal_event;
    logic [31:0] pair_count_q, token_count_q, raw_packet_count_q;

    generate
        if (!PARAMETERS_LEGAL) begin : g_bad_parameters
            initial $fatal(1, "M231 unsupported geometry");
        end
    endgenerate

    always_comb begin : event_shape
        event_shape_legal = fill_active_q
            && event_pair_tag == fill_tag_q
            && event_group_index == fill_group_q
            && event_last_group == (fill_group_q == GROUPS-1);
        illegal_event = event_valid && !event_shape_legal;
    end

    always_comb begin : interfaces
        protocol_error = fault_q || illegal_event;
        // Quarantine every interface in the same combinational cycle that a
        // malformed event is observed.  Gating only with the registered fault
        // would advertise a downstream accept that the sequential block then
        // intentionally refuses to commit.
        pair_header_ready = !protocol_error
            && !fill_active_q && !full_q[fill_ptr_q];
        pair_header_accept = pair_header_valid && pair_header_ready;
        event_ready = !protocol_error && fill_active_q && event_shape_legal;
        event_accept = event_valid && event_ready;

        header_valid = !protocol_error && drain_state_q == DRAIN_HEADER;
        header_tag = header_valid
            ? {pair_tag_q[drain_ptr_q], drain_row_q} : '0;
        header_raw_beat_count = RAW_BEATS;
        case (OUTPUT_BLOCKS)
            1: header_window_depth = 4'd2;
            2: header_window_depth = 4'd4;
            default: header_window_depth = 4'd8;
        endcase
        header_output_blocks = OUTPUT_BLOCKS;
        header_accept = header_valid && header_ready;

        raw_valid = !protocol_error && drain_state_q == DRAIN_RAW;
        raw_lane_valid = raw_valid ? 4'hf : 4'h0;
        raw_last = raw_valid && drain_packet_q == RAW_PACKETS-1;
        raw_accept = raw_valid && raw_ready;
        for (int lane = 0; lane < 4; lane++) begin
            raw_beat_index[lane] = raw_valid
                ? (drain_packet_q * 4 + lane) : '0;
            raw_bitmap[lane] = '0;
            if (raw_valid) begin
                if (!drain_row_q)
                    raw_bitmap[lane] = row0_q[drain_ptr_q]
                        [(drain_packet_q*384)+(lane*96)+:96];
                else
                    raw_bitmap[lane] = row1_q[drain_ptr_q]
                        [(drain_packet_q*384)+(lane*96)+:96];
            end
        end
        busy = fill_active_q || (|full_q) || drain_state_q != DRAIN_IDLE;
        debug_full_slots = full_q;
        debug_fill_group = fill_group_q;
        debug_pair_count = pair_count_q;
        debug_token_count = token_count_q;
        debug_raw_packet_count = raw_packet_count_q;
    end

    always_ff @(posedge clk_core) begin : state
        if (rst_core) begin
            full_q <= '0;
            fill_ptr_q <= 1'b0;
            drain_ptr_q <= 1'b0;
            fill_active_q <= 1'b0;
            fill_group_q <= '0;
            fill_tag_q <= '0;
            drain_state_q <= DRAIN_IDLE;
            drain_row_q <= 1'b0;
            drain_packet_q <= '0;
            fault_q <= 1'b0;
            pair_count_q <= '0;
            token_count_q <= '0;
            raw_packet_count_q <= '0;
            for (int slot = 0; slot < 2; slot++) begin
                row0_q[slot] <= '0;
                row1_q[slot] <= '0;
                pair_tag_q[slot] <= '0;
            end
        end else begin
            if (illegal_event)
                fault_q <= 1'b1;
            if (!protocol_error) begin
                if (pair_header_accept) begin
                    fill_active_q <= 1'b1;
                    fill_group_q <= '0;
                    fill_tag_q <= pair_header_tag;
                    pair_tag_q[fill_ptr_q] <= pair_header_tag;
                    row0_q[fill_ptr_q] <= '0;
                    row1_q[fill_ptr_q] <= '0;
                end
                if (event_accept) begin
                    row0_q[fill_ptr_q][fill_group_q*16+:16]
                        <= event_bits[15:0];
                    row1_q[fill_ptr_q][fill_group_q*16+:16]
                        <= event_bits[31:16];
                    if (event_last_group) begin
                        fill_active_q <= 1'b0;
                        full_q[fill_ptr_q] <= 1'b1;
                        fill_ptr_q <= !fill_ptr_q;
                        pair_count_q <= pair_count_q + 1'b1;
                    end else begin
                        fill_group_q <= fill_group_q + 1'b1;
                    end
                end

                case (drain_state_q)
                    DRAIN_IDLE: begin
                        if (full_q[drain_ptr_q]) begin
                            drain_row_q <= 1'b0;
                            drain_packet_q <= '0;
                            drain_state_q <= DRAIN_HEADER;
                        end
                    end
                    DRAIN_HEADER: begin
                        if (header_accept)
                            drain_state_q <= DRAIN_RAW;
                    end
                    DRAIN_RAW: begin
                        if (raw_accept) begin
                            raw_packet_count_q <= raw_packet_count_q + 1'b1;
                            if (raw_last) begin
                                token_count_q <= token_count_q + 1'b1;
                                drain_packet_q <= '0;
                                if (!drain_row_q) begin
                                    drain_row_q <= 1'b1;
                                    drain_state_q <= DRAIN_HEADER;
                                end else begin
                                    full_q[drain_ptr_q] <= 1'b0;
                                    drain_ptr_q <= !drain_ptr_q;
                                    drain_state_q <= DRAIN_IDLE;
                                end
                            end else begin
                                drain_packet_q <= drain_packet_q + 1'b1;
                            end
                        end
                    end
                    default: fault_q <= 1'b1;
                endcase
            end
        end
    end
endmodule

`default_nettype wire
