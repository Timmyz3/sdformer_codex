`timescale 1ns/1ps
`default_nettype none

// Exact fallback decoder for 162 token-major {gate9, K32} records packed into
// 104x64-bit head slots. The final physical word carries 50 valid bits.
module gatestack_raw41_replay_decoder #(
    parameter int TOKENS          = 162,
    parameter int LANES           = 32,
    parameter int TAG_W           = 32,
    parameter int WORD_INDEX_W    = 7,
    parameter int COUNTER_W       = 32,
    parameter int TOKEN_ID_W      = 8,
    parameter int LANE_ID_W       = (LANES <= 1) ? 1 : $clog2(LANES)
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         start_valid,
    output logic                         start_ready,
    input  logic [TAG_W-1:0]             start_tag,

    input  logic                         word_valid,
    output logic                         word_ready,
    input  logic [63:0]                  word_data,
    input  logic [WORD_INDEX_W-1:0]      word_index,
    input  logic                         word_last,

    output logic                         direct_valid,
    input  logic                         direct_ready,
    output logic [8:0]                   direct_gate_code,
    output logic [LANE_ID_W-1:0]         direct_lane_id,
    output logic [TOKEN_ID_W-1:0]        direct_token_id,
    output logic                         direct_head_last,

    output logic                         done_valid,
    input  logic                         done_ready,
    output logic [TAG_W-1:0]             done_tag,
    output logic                         done_error,

    output logic                         protocol_error,
    output logic [COUNTER_W-1:0]         count_heads,
    output logic [COUNTER_W-1:0]         count_records,
    output logic [COUNTER_W-1:0]         count_kzero_records,
    output logic [COUNTER_W-1:0]         count_direct_events,
    output logic [COUNTER_W-1:0]         count_input_stall_cycles,
    output logic [COUNTER_W-1:0]         count_output_stall_cycles
);

    localparam int RECORD_BITS = LANES + 9;
    localparam int RAW_BITS = TOKENS * RECORD_BITS;
    localparam int WORDS_PER_HEAD = (RAW_BITS + 63) / 64;
    localparam int LAST_WORD_BITS = RAW_BITS - (WORDS_PER_HEAD - 1) * 64;

    logic active_q;
    logic done_valid_q;
    logic session_error_q;
    logic [TAG_W-1:0] tag_q;
    logic [127:0] reservoir_q;
    logic [7:0] reservoir_bits_q;
    logic [7:0] records_consumed_q;
    logic [WORD_INDEX_W-1:0] expected_word_index_q;
    logic input_last_seen_q;
    logic [LANES-1:0] pending_k_q;
    logic [8:0] pending_gate_q;
    logic [TOKEN_ID_W-1:0] pending_token_q;

    logic consume_record;
    logic [40:0] record_comb;
    logic [LANES-1:0] record_k_comb;
    logic [8:0] record_gate_comb;
    logic [127:0] reservoir_after_record;
    logic [7:0] bits_after_record;
    logic [6:0] input_valid_bits;
    logic word_fire;
    logic direct_fire;
    logic done_fire;
    logic [LANES-1:0] selected_lane_mask;
    logic selected_lane_found;
    logic input_word_contract_ok;
    logic all_records_after_consume;
    logic input_last_after_fire;

    assign start_ready = !active_q && !done_valid_q;
    assign done_valid = done_valid_q;
    assign done_tag = tag_q;
    assign done_error = session_error_q;
    assign done_fire = done_valid && done_ready;

    assign consume_record = active_q && pending_k_q == '0 &&
                            32'(reservoir_bits_q) >= RECORD_BITS &&
                            32'(records_consumed_q) < TOKENS;
    assign record_comb = reservoir_q[40:0];
    assign record_k_comb = record_comb[31:0];
    assign record_gate_comb = record_comb[40:32];
    assign reservoir_after_record = consume_record ?
                                    (reservoir_q >> RECORD_BITS) : reservoir_q;
    assign bits_after_record = reservoir_bits_q -
                               (consume_record ? 8'(RECORD_BITS) : 8'(0));
    assign input_valid_bits = word_last ? 7'(LAST_WORD_BITS) : 7'd64;
    assign input_word_contract_ok =
        word_index == expected_word_index_q &&
        word_last == (32'(word_index) == WORDS_PER_HEAD - 1);
    assign all_records_after_consume =
        32'(records_consumed_q) + 1 == TOKENS;
    assign input_last_after_fire = input_last_seen_q ||
                                   (word_fire && word_last);

    assign word_ready = active_q && !input_last_seen_q &&
                        bits_after_record <= 64 &&
                        !(direct_valid && !direct_ready);
    assign word_fire = word_valid && word_ready;

    always_comb begin
        selected_lane_mask = '0;
        selected_lane_found = 1'b0;
        direct_lane_id = '0;
        for (int lane = 0; lane < LANES; lane = lane + 1) begin
            if (!selected_lane_found && pending_k_q[lane]) begin
                selected_lane_found = 1'b1;
                selected_lane_mask[lane] = 1'b1;
                direct_lane_id = LANE_ID_W'(lane);
            end
        end
    end

    assign direct_valid = active_q && pending_k_q != '0;
    assign direct_fire = direct_valid && direct_ready;
    assign direct_gate_code = pending_gate_q;
    assign direct_token_id = pending_token_q;
    assign direct_head_last = direct_valid &&
                              pending_k_q == selected_lane_mask &&
                              32'(records_consumed_q) == TOKENS;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            active_q <= 1'b0;
            done_valid_q <= 1'b0;
            session_error_q <= 1'b0;
            tag_q <= '0;
            reservoir_q <= '0;
            reservoir_bits_q <= '0;
            records_consumed_q <= '0;
            expected_word_index_q <= '0;
            input_last_seen_q <= 1'b0;
            pending_k_q <= '0;
            pending_gate_q <= '0;
            pending_token_q <= '0;
            protocol_error <= 1'b0;
            count_heads <= '0;
            count_records <= '0;
            count_kzero_records <= '0;
            count_direct_events <= '0;
            count_input_stall_cycles <= '0;
            count_output_stall_cycles <= '0;
        end else begin
            if (start_valid && start_ready) begin
                active_q <= 1'b1;
                session_error_q <= 1'b0;
                tag_q <= start_tag;
                reservoir_q <= '0;
                reservoir_bits_q <= '0;
                records_consumed_q <= '0;
                expected_word_index_q <= '0;
                input_last_seen_q <= 1'b0;
                pending_k_q <= '0;
                pending_gate_q <= '0;
                pending_token_q <= '0;
            end

            if (active_q) begin
                reservoir_q <= reservoir_after_record;
                reservoir_bits_q <= bits_after_record;

                if (word_fire) begin
                    reservoir_q <= reservoir_after_record |
                        (128'(word_data) << 32'(bits_after_record));
                    reservoir_bits_q <= bits_after_record + input_valid_bits;
                    expected_word_index_q <= expected_word_index_q + 1'b1;
                    input_last_seen_q <= word_last;
                    if (!input_word_contract_ok) begin
                        protocol_error <= 1'b1;
                        session_error_q <= 1'b1;
                    end
                end

                if (consume_record) begin
                    records_consumed_q <= records_consumed_q + 1'b1;
                    pending_k_q <= record_k_comb;
                    pending_gate_q <= record_gate_comb;
                    pending_token_q <= TOKEN_ID_W'(records_consumed_q);
                    count_records <= count_records + 1'b1;
                    if (record_k_comb == '0) begin
                        count_kzero_records <= count_kzero_records + 1'b1;
                        if (all_records_after_consume && input_last_after_fire &&
                            bits_after_record == 0) begin
                            active_q <= 1'b0;
                            done_valid_q <= 1'b1;
                            count_heads <= count_heads + 1'b1;
                        end
                    end
                end

                if (direct_fire) begin
                    pending_k_q <= pending_k_q & ~selected_lane_mask;
                    count_direct_events <= count_direct_events + 1'b1;
                    if (direct_head_last) begin
                        if (!input_last_seen_q || reservoir_bits_q != 0) begin
                            protocol_error <= 1'b1;
                            session_error_q <= 1'b1;
                        end
                        active_q <= 1'b0;
                        done_valid_q <= 1'b1;
                        count_heads <= count_heads + 1'b1;
                    end
                end
            end

            if (done_fire) begin
                done_valid_q <= 1'b0;
            end
            if (word_valid && !word_ready) begin
                count_input_stall_cycles <= count_input_stall_cycles + 1'b1;
            end
            if (direct_valid && !direct_ready) begin
                count_output_stall_cycles <= count_output_stall_cycles + 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
