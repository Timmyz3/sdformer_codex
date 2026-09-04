`timescale 1ns/1ps
`default_nettype none

module hitflow_banked_accumulator #(
    parameter int TOKENS       = 162,
    parameter int BANKS        = 2,
    parameter int PRODUCT_W    = 17,
    parameter int ACC_W        = 32,
    parameter int OUT_TILE     = 8,
    parameter int TAG_W        = 32,
    parameter int COUNTER_W    = 32,
    parameter int TOKEN_ID_W   = (TOKENS <= 1) ? 1 : $clog2(TOKENS)
) (
    input  logic                            clk_core,
    input  logic                            rst_core,
    input  logic                            flush,

    input  logic                            group_start_valid,
    output logic                            group_start_ready,
    input  logic [TAG_W-1:0]                group_start_tag,

    input  logic [BANKS-1:0]                update_valid,
    output logic [BANKS-1:0]                update_ready,
    input  logic [(BANKS*TOKEN_ID_W)-1:0]   update_token_ids,
    input  logic [TAG_W-1:0]                update_tag,
    input  logic                            update_is_bias,
    input  logic [(OUT_TILE*PRODUCT_W)-1:0] update_values,
    input  logic [(OUT_TILE*ACC_W)-1:0]     update_bias_values,

    output logic [BANKS-1:0]                final_valid,
    input  logic [BANKS-1:0]                final_ready,
    output logic [(BANKS*TOKEN_ID_W)-1:0]   final_token_ids,
    output logic [TAG_W-1:0]                final_tag,
    output logic [(BANKS*OUT_TILE*ACC_W)-1:0] final_values,

    input  logic                            group_finish_valid,
    output logic                            group_finish_ready,
    output logic [TAG_W-1:0]                group_finish_tag,
    output logic                            protocol_error,
    output logic                            accumulator_overflow,

    output logic [COUNTER_W-1:0]            count_updates,
    output logic [COUNTER_W-1:0]            count_writes,
    output logic [COUNTER_W-1:0]            count_bias_commits,
    output logic [COUNTER_W-1:0]            count_bank_stall_cycles,
    output logic [COUNTER_W-1:0]            count_final_stall_cycles
);

    localparam int BANK_DEPTH = (TOKENS + BANKS - 1) / BANKS;
    localparam int BANK_ADDR_W = (BANK_DEPTH <= 1) ? 1 : $clog2(BANK_DEPTH);
    localparam BANKS_LOOP = BANKS;
    localparam OUT_TILE_LOOP = OUT_TILE;

    logic group_active_q;
    logic [TAG_W-1:0] group_tag_q;
    logic [COUNTER_W-1:0] bias_commits_in_group_q;
    logic [BANKS-1:0] bank_busy;
    logic [BANKS-1:0] bank_update_fire;
    logic [BANKS-1:0] bank_write_fire;
    logic [BANKS-1:0] bank_bias_write_fire;
    logic [BANKS-1:0] bank_overflow;
    logic [BANKS-1:0] update_protocol_ok;
    logic [COUNTER_W-1:0] update_fire_count;
    logic [COUNTER_W-1:0] write_fire_count;
    logic [COUNTER_W-1:0] bias_fire_count;
    logic group_start_fire;
    logic group_finish_fire;

    assign group_start_ready = !flush && !group_active_q && (bank_busy == '0);
    assign group_start_fire = group_start_valid && group_start_ready;
    assign group_finish_ready = !flush && group_active_q && (bank_busy == '0) &&
                                (bias_commits_in_group_q == TOKENS);
    assign group_finish_fire = group_finish_valid && group_finish_ready;
    assign group_finish_tag = group_tag_q;
    assign final_tag = group_tag_q;
    assign protocol_error = !flush && |(update_valid & ~update_protocol_ok);

    always_comb begin
        update_fire_count = '0;
        write_fire_count = '0;
        bias_fire_count = '0;
        for (int bank = 32'd0; bank < BANKS_LOOP; bank = bank + 32'd1) begin
            if (bank_update_fire[bank]) begin
                update_fire_count = update_fire_count + 1'b1;
            end
            if (bank_write_fire[bank]) begin
                write_fire_count = write_fire_count + 1'b1;
            end
            if (bank_bias_write_fire[bank]) begin
                bias_fire_count = bias_fire_count + 1'b1;
            end
        end
    end

    for (genvar bank = 32'd0;
         bank < BANKS_LOOP;
         bank = bank + 32'd1) begin : g_acc_bank
        logic [(OUT_TILE*ACC_W)-1:0] acc_mem [0:BANK_DEPTH-1];
        logic [BANK_DEPTH-1:0] value_valid_q;
        logic [BANK_DEPTH-1:0] bias_committed_q;
        logic busy_q;
        logic [TOKEN_ID_W-1:0] token_q;
        logic [BANK_ADDR_W-1:0] address_q;
        logic is_bias_q;
        logic [(OUT_TILE*ACC_W)-1:0] read_data_q;
        logic read_was_valid_q;
        logic [(OUT_TILE*ACC_W)-1:0] addend_q;
        logic [(OUT_TILE*ACC_W)-1:0] sum_vector;
        logic lane_overflow;
        logic [TOKEN_ID_W-1:0] input_token;
        logic [BANK_ADDR_W-1:0] input_address;
        logic input_token_in_range;
        logic input_matches_bank;
        logic input_bias_fresh;
        logic commit_allowed;

        assign input_token =
            update_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W];
        assign input_address = BANK_ADDR_W'(32'(input_token) / BANKS);
        // TOKEN_ID_W=$clog2(TOKENS) cannot represent TOKENS when TOKENS is 2^n.
        assign input_token_in_range = 32'(input_token) < 32'(TOKENS);
        assign input_matches_bank = (32'(input_token) % BANKS) == bank;
        always_comb begin
            if (input_token_in_range) begin
                input_bias_fresh = !bias_committed_q[input_address];
            end else begin
                input_bias_fresh = 1'b0;
            end
        end
        assign update_protocol_ok[bank] = !flush && group_active_q &&
                                          (update_tag == group_tag_q) &&
                                          input_token_in_range &&
                                          input_matches_bank &&
                                          (!update_is_bias || input_bias_fresh);
        assign update_ready[bank] = !busy_q && update_protocol_ok[bank];
        assign bank_update_fire[bank] = update_valid[bank] &&
                                        update_ready[bank];
        assign bank_busy[bank] = busy_q;
        // Overflow is consumed locally as an error commit. It must never wait
        // for or handshake with the external final channel.
        assign commit_allowed = !flush && busy_q &&
            (!is_bias_q || lane_overflow || final_ready[bank]);
        assign bank_write_fire[bank] = commit_allowed;
        assign bank_bias_write_fire[bank] = commit_allowed && is_bias_q;
        assign final_valid[bank] = !flush && busy_q && is_bias_q &&
                                   !lane_overflow;
        assign final_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W] = token_q;
        assign final_values[(bank*OUT_TILE*ACC_W) +: (OUT_TILE*ACC_W)] =
            sum_vector;
        assign bank_overflow[bank] = commit_allowed && lane_overflow;

        always_comb begin
            sum_vector = '0;
            lane_overflow = 1'b0;
            for (int lane = 32'd0;
                 lane < OUT_TILE_LOOP;
                 lane = lane + 32'd1) begin
                logic signed [ACC_W-1:0] old_value;
                logic signed [ACC_W-1:0] addend_value;
                logic signed [ACC_W-1:0] sum_value;
                if (read_was_valid_q) begin
                    old_value = $signed(read_data_q[(lane*ACC_W) +: ACC_W]);
                end else begin
                    old_value = '0;
                end
                addend_value = $signed(addend_q[(lane*ACC_W) +: ACC_W]);
                sum_value = old_value + addend_value;
                sum_vector[(lane*ACC_W) +: ACC_W] = sum_value;
                if ((old_value[ACC_W-1] == addend_value[ACC_W-1]) &&
                    (sum_value[ACC_W-1] != old_value[ACC_W-1])) begin
                    lane_overflow = 1'b1;
                end
            end
        end

        always_ff @(posedge clk_core) begin
            if (rst_core) begin
                value_valid_q <= '0;
                bias_committed_q <= '0;
                busy_q <= 1'b0;
                token_q <= '0;
                address_q <= '0;
                is_bias_q <= 1'b0;
                read_data_q <= '0;
                read_was_valid_q <= 1'b0;
                addend_q <= '0;
            end else if (flush) begin
                value_valid_q <= '0;
                bias_committed_q <= '0;
                busy_q <= 1'b0;
                token_q <= '0;
                address_q <= '0;
                is_bias_q <= 1'b0;
                read_data_q <= '0;
                read_was_valid_q <= 1'b0;
                addend_q <= '0;
            end else if (group_start_fire) begin
                value_valid_q <= '0;
                bias_committed_q <= '0;
                busy_q <= 1'b0;
            end else begin
                if (commit_allowed) begin
                    acc_mem[address_q] <= sum_vector;
                    value_valid_q[address_q] <= 1'b1;
                    if (is_bias_q) begin
                        bias_committed_q[address_q] <= 1'b1;
                    end
                    busy_q <= 1'b0;
                end

                if (bank_update_fire[bank]) begin
                    token_q <= input_token;
                    address_q <= input_address;
                    is_bias_q <= update_is_bias;
                    read_data_q <= acc_mem[input_address];
                    read_was_valid_q <= value_valid_q[input_address];
                    for (int lane = 32'd0;
                         lane < OUT_TILE_LOOP;
                         lane = lane + 32'd1) begin
                        if (update_is_bias) begin
                            addend_q[(lane*ACC_W) +: ACC_W] <=
                                update_bias_values[(lane*ACC_W) +: ACC_W];
                        end else begin
                            addend_q[(lane*ACC_W) +: ACC_W] <= {
                                {(ACC_W-PRODUCT_W){
                                    update_values[(lane*PRODUCT_W)+PRODUCT_W-1]
                                }},
                                update_values[(lane*PRODUCT_W) +: PRODUCT_W]
                            };
                        end
                    end
                    busy_q <= 1'b1;
                end
            end
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            group_active_q             <= 1'b0;
            group_tag_q                <= '0;
            bias_commits_in_group_q    <= '0;
            accumulator_overflow       <= 1'b0;
            count_updates              <= '0;
            count_writes               <= '0;
            count_bias_commits         <= '0;
            count_bank_stall_cycles    <= '0;
            count_final_stall_cycles   <= '0;
        end else if (flush) begin
            group_active_q          <= 1'b0;
            group_tag_q             <= '0;
            bias_commits_in_group_q <= '0;
            accumulator_overflow    <= 1'b0;
        end else begin
            if (group_start_fire) begin
                group_active_q          <= 1'b1;
                group_tag_q             <= group_start_tag;
                bias_commits_in_group_q <= '0;
            end else if (group_finish_fire) begin
                group_active_q <= 1'b0;
            end

            if (!group_start_fire) begin
                bias_commits_in_group_q <= bias_commits_in_group_q +
                                           bias_fire_count;
            end
            count_updates <= count_updates + update_fire_count;
            count_writes <= count_writes + write_fire_count;
            count_bias_commits <= count_bias_commits + bias_fire_count;
            if ((update_valid & ~update_ready & update_protocol_ok) != '0) begin
                count_bank_stall_cycles <= count_bank_stall_cycles + 1'b1;
            end
            if ((final_valid & ~final_ready) != '0) begin
                count_final_stall_cycles <= count_final_stall_cycles + 1'b1;
            end
            if (bank_overflow != '0) begin
                accumulator_overflow <= 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
