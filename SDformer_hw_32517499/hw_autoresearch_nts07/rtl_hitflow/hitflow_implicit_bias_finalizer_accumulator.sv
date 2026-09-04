`timescale 1ns/1ps
`default_nettype none

// Product-only accumulator with an implicit-bias final drain. Bias is added on
// the output path and is never written back to the accumulator memory.
module hitflow_implicit_bias_finalizer_accumulator #(
    parameter int TOKENS       = 162,
    parameter int BANKS        = 2,
    parameter int PRODUCT_W    = 17,
    parameter int ACC_W        = 32,
    parameter int OUT_TILE     = 8,
    parameter int TAG_W        = 32,
    parameter int COUNTER_W    = 32,
    parameter int TOKEN_ID_W   = (TOKENS <= 1) ? 1 : $clog2(TOKENS)
) (
    input  logic                              clk_core,
    input  logic                              rst_core,
    input  logic                              flush,

    input  logic                              group_start_valid,
    output logic                              group_start_ready,
    input  logic [TAG_W-1:0]                  group_start_tag,

    input  logic [BANKS-1:0]                  update_valid,
    output logic [BANKS-1:0]                  update_ready,
    input  logic [(BANKS*TOKEN_ID_W)-1:0]     update_token_ids,
    input  logic [TAG_W-1:0]                  update_tag,
    input  logic [(OUT_TILE*PRODUCT_W)-1:0]   update_values,

    input  logic                              finalize_start_valid,
    output logic                              finalize_start_ready,
    input  logic [TAG_W-1:0]                  finalize_start_tag,
    input  logic [(OUT_TILE*ACC_W)-1:0]       finalize_bias_values,

    output logic [BANKS-1:0]                  final_valid,
    input  logic [BANKS-1:0]                  final_ready,
    output logic [(BANKS*TOKEN_ID_W)-1:0]     final_token_ids,
    output logic [TAG_W-1:0]                  final_tag,
    output logic [(BANKS*OUT_TILE*ACC_W)-1:0] final_values,

    output logic                              finalize_done_valid,
    input  logic                              finalize_done_ready,
    output logic [TAG_W-1:0]                  finalize_done_tag,
    output logic                              protocol_error,
    output logic                              accumulator_overflow,

    output logic [COUNTER_W-1:0]              count_updates,
    output logic [COUNTER_W-1:0]              count_product_writes,
    output logic [COUNTER_W-1:0]              count_final_reads,
    output logic [COUNTER_W-1:0]              count_final_emits,
    output logic [COUNTER_W-1:0]              count_update_stall_cycles,
    output logic [COUNTER_W-1:0]              count_final_stall_cycles
);

    localparam int BANK_DEPTH = (TOKENS + BANKS - 1) / BANKS;
    localparam int BANK_ADDR_W = (BANK_DEPTH <= 1) ? 1 : $clog2(BANK_DEPTH);
    localparam BANKS_LOOP = BANKS;
    localparam OUT_TILE_LOOP = OUT_TILE;

    logic group_active_q;
    logic finalizing_q;
    logic [TAG_W-1:0] group_tag_q;
    logic [(OUT_TILE*ACC_W)-1:0] resident_bias_q;
    logic [BANKS-1:0] bank_busy;
    logic [BANKS-1:0] drain_done;
    logic [BANKS-1:0] drain_read_pending;
    logic [BANKS-1:0] bank_update_fire;
    logic [BANKS-1:0] bank_write_fire;
    logic [BANKS-1:0] bank_final_read_fire;
    logic [BANKS-1:0] bank_final_emit_fire;
    logic [BANKS-1:0] bank_overflow;
    logic [BANKS-1:0] update_protocol_ok;
    logic [COUNTER_W-1:0] update_fire_count;
    logic [COUNTER_W-1:0] write_fire_count;
    logic [COUNTER_W-1:0] final_read_count;
    logic [COUNTER_W-1:0] final_emit_count;
    logic group_start_fire;
    logic finalize_start_fire;
    logic finalize_done_fire;

    assign group_start_ready = !flush && !group_active_q && !finalizing_q &&
                               (bank_busy == '0) &&
                               (drain_read_pending == '0);
    assign group_start_fire = group_start_valid && group_start_ready;
    assign finalize_start_ready = !flush && group_active_q && !finalizing_q &&
                                  (bank_busy == '0) &&
                                  (finalize_start_tag == group_tag_q);
    assign finalize_start_fire = finalize_start_valid && finalize_start_ready;
    assign finalize_done_valid = !flush && finalizing_q && (&drain_done) &&
                                 (drain_read_pending == '0);
    assign finalize_done_fire = finalize_done_valid && finalize_done_ready;
    assign finalize_done_tag = group_tag_q;
    assign final_tag = group_tag_q;

    always_comb begin
        update_fire_count = '0;
        write_fire_count = '0;
        final_read_count = '0;
        final_emit_count = '0;
        for (int bank = 0; bank < BANKS_LOOP; bank = bank + 1) begin
            if (bank_update_fire[bank])
                update_fire_count = update_fire_count + 1'b1;
            if (bank_write_fire[bank])
                write_fire_count = write_fire_count + 1'b1;
            if (bank_final_read_fire[bank])
                final_read_count = final_read_count + 1'b1;
            if (bank_final_emit_fire[bank])
                final_emit_count = final_emit_count + 1'b1;
        end
    end

    for (genvar bank = 0; bank < BANKS_LOOP; bank = bank + 1) begin : g_bank
        localparam int BANK_TOKENS = (TOKENS + BANKS - 1 - bank) / BANKS;
        logic [(OUT_TILE*ACC_W)-1:0] acc_mem [0:BANK_DEPTH-1];
        logic [BANK_DEPTH-1:0] value_valid_q;
        logic update_busy_q;
        logic [BANK_ADDR_W-1:0] update_address_q;
        logic [(OUT_TILE*ACC_W)-1:0] update_addend_q;
        logic [(OUT_TILE*ACC_W)-1:0] shared_addend_vector;
        logic [(OUT_TILE*ACC_W)-1:0] shared_sum_vector;
        logic shared_lane_overflow;
        logic [TOKEN_ID_W-1:0] input_token;
        logic [BANK_ADDR_W-1:0] input_address;
        logic input_token_in_range;
        logic input_matches_bank;

        logic [BANK_ADDR_W:0] drain_address_q;
        logic drain_done_q;
        logic drain_read_pending_q;
        logic [TOKEN_ID_W-1:0] drain_read_token_q;
        logic drain_read_retire;
        logic drain_read_slot_available;
        logic drain_issue;
        logic [(OUT_TILE*ACC_W)-1:0] shared_read_data_q;
        logic shared_read_was_valid_q;
        logic shared_read_fire;
        logic [BANK_ADDR_W-1:0] shared_read_address;

        assign input_token =
            update_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W];
        assign input_address = BANK_ADDR_W'(32'(input_token) / BANKS);
        assign input_token_in_range = 32'(input_token) < 32'(TOKENS);
        assign input_matches_bank = (32'(input_token) % BANKS) == bank;
        assign update_protocol_ok[bank] = !flush && group_active_q &&
                                          !finalizing_q &&
                                          (update_tag == group_tag_q) &&
                                          input_token_in_range &&
                                          input_matches_bank;
        assign update_ready[bank] = !update_busy_q &&
                                    update_protocol_ok[bank];
        assign bank_update_fire[bank] = update_valid[bank] &&
                                        update_ready[bank];
        assign bank_write_fire[bank] = !flush && update_busy_q;
        assign bank_busy[bank] = update_busy_q;

        assign final_valid[bank] = !flush && drain_read_pending_q &&
                                   !shared_lane_overflow;
        assign final_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W] =
            drain_read_token_q;
        assign final_values[(bank*OUT_TILE*ACC_W) +: (OUT_TILE*ACC_W)] =
            shared_sum_vector;
        assign bank_final_emit_fire[bank] = final_valid[bank] &&
                                             final_ready[bank];
        assign drain_read_pending[bank] = drain_read_pending_q;
        assign drain_done[bank] = drain_done_q;
        assign drain_read_retire = drain_read_pending_q &&
                                   (shared_lane_overflow || final_ready[bank]);
        assign drain_read_slot_available = !drain_read_pending_q ||
                                           drain_read_retire;
        assign drain_issue = !flush && finalizing_q && !drain_done_q &&
                             drain_read_slot_available;
        assign bank_final_read_fire[bank] = drain_issue;
        assign shared_read_fire = bank_update_fire[bank] || drain_issue;
        assign shared_read_address = bank_update_fire[bank] ? input_address :
                                     drain_address_q[BANK_ADDR_W-1:0];

        always_comb begin
            shared_addend_vector = finalizing_q ? resident_bias_q :
                                    update_addend_q;
            shared_sum_vector = '0;
            shared_lane_overflow = 1'b0;
            for (int lane = 0; lane < OUT_TILE_LOOP; lane = lane + 1) begin
                logic signed [ACC_W-1:0] old_value;
                logic signed [ACC_W-1:0] addend_value;
                logic signed [ACC_W-1:0] sum_value;
                old_value = shared_read_was_valid_q ?
                    $signed(shared_read_data_q[(lane*ACC_W) +: ACC_W]) : '0;
                addend_value =
                    $signed(shared_addend_vector[(lane*ACC_W) +: ACC_W]);
                sum_value = old_value + addend_value;
                shared_sum_vector[(lane*ACC_W) +: ACC_W] = sum_value;
                if ((old_value[ACC_W-1] == addend_value[ACC_W-1]) &&
                    (sum_value[ACC_W-1] != old_value[ACC_W-1]))
                    shared_lane_overflow = 1'b1;
            end
        end

        assign bank_overflow[bank] =
            (bank_write_fire[bank] && shared_lane_overflow) ||
            (drain_read_retire && shared_lane_overflow);

        always_ff @(posedge clk_core) begin
            if (rst_core || flush) begin
                value_valid_q <= '0;
                update_busy_q <= 1'b0;
                update_address_q <= '0;
                update_addend_q <= '0;
                shared_read_data_q <= '0;
                shared_read_was_valid_q <= 1'b0;
                drain_address_q <= '0;
                drain_done_q <= 1'b0;
                drain_read_pending_q <= 1'b0;
                drain_read_token_q <= '0;
            end else if (group_start_fire) begin
                value_valid_q <= '0;
                update_busy_q <= 1'b0;
                drain_address_q <= '0;
                drain_done_q <= 1'b0;
                drain_read_pending_q <= 1'b0;
            end else begin
                if (bank_write_fire[bank]) begin
                    acc_mem[update_address_q] <= shared_sum_vector;
                    value_valid_q[update_address_q] <= 1'b1;
                    update_busy_q <= 1'b0;
                end
                if (bank_update_fire[bank]) begin
                    update_address_q <= input_address;
                    for (int lane = 0; lane < OUT_TILE_LOOP;
                         lane = lane + 1) begin
                        update_addend_q[(lane*ACC_W) +: ACC_W] <= {
                            {(ACC_W-PRODUCT_W){
                                update_values[(lane*PRODUCT_W)+PRODUCT_W-1]
                            }},
                            update_values[(lane*PRODUCT_W) +: PRODUCT_W]
                        };
                    end
                    update_busy_q <= 1'b1;
                end
                if (shared_read_fire) begin
                    shared_read_data_q <= acc_mem[shared_read_address];
                    shared_read_was_valid_q <=
                        value_valid_q[shared_read_address];
                end

                if (finalize_start_fire) begin
                    drain_address_q <= '0;
                    drain_done_q <= (BANK_TOKENS == 0);
                    drain_read_pending_q <= 1'b0;
                end else if (finalizing_q) begin
                    if (drain_issue) begin
                        drain_read_pending_q <= 1'b1;
                        drain_read_token_q <= TOKEN_ID_W'(
                            32'(drain_address_q) * BANKS + bank);
                        if (32'(drain_address_q) == BANK_TOKENS - 1) begin
                            drain_done_q <= 1'b1;
                        end else begin
                            drain_address_q <= drain_address_q + 1'b1;
                        end
                    end else if (drain_read_retire) begin
                        drain_read_pending_q <= 1'b0;
                    end
                end
            end
        end
    end

    assign protocol_error = !flush && (
        |(update_valid & ~update_protocol_ok) ||
        (finalize_start_valid && group_active_q &&
         (finalize_start_tag != group_tag_q))
    );

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            group_active_q <= 1'b0;
            finalizing_q <= 1'b0;
            group_tag_q <= '0;
            resident_bias_q <= '0;
            accumulator_overflow <= 1'b0;
            count_updates <= '0;
            count_product_writes <= '0;
            count_final_reads <= '0;
            count_final_emits <= '0;
            count_update_stall_cycles <= '0;
            count_final_stall_cycles <= '0;
        end else if (flush) begin
            group_active_q <= 1'b0;
            finalizing_q <= 1'b0;
            group_tag_q <= '0;
            resident_bias_q <= '0;
            accumulator_overflow <= 1'b0;
        end else begin
            if (group_start_fire) begin
                group_active_q <= 1'b1;
                finalizing_q <= 1'b0;
                group_tag_q <= group_start_tag;
                accumulator_overflow <= 1'b0;
            end
            if (finalize_start_fire) begin
                finalizing_q <= 1'b1;
                resident_bias_q <= finalize_bias_values;
            end
            if (finalize_done_fire) begin
                group_active_q <= 1'b0;
                finalizing_q <= 1'b0;
            end
            if (bank_overflow != '0)
                accumulator_overflow <= 1'b1;
            count_updates <= count_updates + update_fire_count;
            count_product_writes <= count_product_writes + write_fire_count;
            count_final_reads <= count_final_reads + final_read_count;
            count_final_emits <= count_final_emits + final_emit_count;
            if ((update_valid & ~update_ready & update_protocol_ok) != '0)
                count_update_stall_cycles <= count_update_stall_cycles + 1'b1;
            if ((final_valid & ~final_ready) != '0)
                count_final_stall_cycles <= count_final_stall_cycles + 1'b1;
        end
    end

endmodule

`default_nettype wire
