`timescale 1ns/1ps
`default_nettype none

// M54 standalone K1..K4 union-source engine with sixteen finite contexts.
// Each accepted request reads the lowest remaining union row in every bank
// once, carries a strict response tag and full group metadata through a real
// 16-entry FIFO, and independently add/subtract/bypasses all destinations.
module qfit_k4_parent_delta_p8_l96_ctx16 #(
    parameter int TILE_BITS = 256,
    parameter int BANKS = 8,
    parameter int LANES = 96,
    parameter int CONTEXTS = 16,
    parameter int MAX_K = 4,
    parameter int META_DEPTH = 16,
    parameter int COMPLETE_DEPTH = 16,
    parameter int TAG_W = 48,
    parameter int RESPONSE_TAG_W = 16,
    parameter int W_W = 8,
    parameter int ACC_W = 19,
    parameter int BANK_ADDR_W = $clog2(TILE_BITS/BANKS),
    parameter int COUNT_W = $clog2(TILE_BITS+1),
    parameter int CONTEXT_W = $clog2(CONTEXTS),
    parameter int GROUP_COUNT_W = $clog2(MAX_K+1),
    parameter int META_PTR_W = $clog2(META_DEPTH),
    parameter int META_COUNT_W = $clog2(META_DEPTH+1),
    parameter int COMPLETE_PTR_W = $clog2(COMPLETE_DEPTH),
    parameter int COMPLETE_COUNT_W = $clog2(COMPLETE_DEPTH+1)
) (
    input  logic                                  clk_core,
    input  logic                                  rst_core,

    input  logic                                  command_valid,
    output logic                                  command_ready,
    input  logic [TAG_W-1:0]                      command_tag,
    input  logic [TILE_BITS-1:0]                  command_add_bits,
    input  logic [TILE_BITS-1:0]                  command_subtract_bits,
    input  logic [LANES*ACC_W-1:0]                command_seed_acc,
    output logic                                  command_accept,
    output logic [CONTEXT_W-1:0]                  command_accept_context,

    input  logic                                  launch_valid,
    output logic                                  launch_ready,
    input  logic [GROUP_COUNT_W-1:0]              launch_context_count,
    input  logic [MAX_K*CONTEXT_W-1:0]            launch_contexts,
    output logic                                  launch_accept,

    output logic                                  weight_request_valid,
    input  logic                                  weight_request_ready,
    output logic [RESPONSE_TAG_W-1:0]             weight_request_tag,
    output logic [GROUP_COUNT_W-1:0]              weight_request_context_count,
    output logic [MAX_K*CONTEXT_W-1:0]            weight_request_contexts,
    output logic [BANKS-1:0]                      weight_request_bank_valid,
    output logic [BANKS*BANK_ADDR_W-1:0]          weight_request_bank_addr,
    output logic [MAX_K*BANKS-1:0]                weight_request_context_valid,
    output logic [MAX_K*BANKS-1:0]                weight_request_context_subtract,
    output logic                                  weight_request_last,
    output logic                                  request_accept,

    input  logic                                  weight_response_valid,
    output logic                                  weight_response_ready,
    input  logic [RESPONSE_TAG_W-1:0]             weight_response_tag,
    input  logic [GROUP_COUNT_W-1:0]              weight_response_context_count,
    input  logic [MAX_K*CONTEXT_W-1:0]            weight_response_contexts,
    input  logic [BANKS-1:0]                      weight_response_bank_valid,
    input  logic [BANKS*LANES*W_W-1:0]           weight_response_data,
    output logic                                  response_accept,

    output logic                                  output_valid,
    input  logic                                  output_ready,
    output logic [TAG_W-1:0]                      output_tag,
    output logic [COUNT_W-1:0]                    output_source_count,
    output logic [LANES*ACC_W-1:0]                output_acc,
    output logic                                  output_accept,

    output logic                                  protocol_error,
    output logic                                  busy,
    output logic [CONTEXT_W:0]                    context_occupancy,
    output logic [META_COUNT_W-1:0]               response_metadata_occupancy,
    output logic [COMPLETE_COUNT_W-1:0]           complete_occupancy,
    output logic                                  group_active
);
    logic context_allocated_q [0:CONTEXTS-1];
    logic context_launched_q [0:CONTEXTS-1];
    logic [TAG_W-1:0] context_tag_q [0:CONTEXTS-1];
    logic [TILE_BITS-1:0] context_add_q [0:CONTEXTS-1];
    logic [TILE_BITS-1:0] context_subtract_q [0:CONTEXTS-1];
    logic [COUNT_W-1:0] context_source_count_q [0:CONTEXTS-1];
    logic signed [ACC_W-1:0] context_acc_q [0:CONTEXTS-1][0:LANES-1];

    logic free_found;
    logic [CONTEXT_W-1:0] free_context;
    logic [CONTEXTS-1:0] context_allocated_vector;
    logic [CONTEXTS-1:0] context_launched_vector;
    logic active_q;
    logic [GROUP_COUNT_W-1:0] active_count_q;
    logic [CONTEXT_W-1:0] active_context_q [0:MAX_K-1];

    logic launch_legal;
    logic [TILE_BITS-1:0] launch_union;
    logic launch_zero;
    logic [CONTEXT_W-1:0] launch_context [0:MAX_K-1];

    logic [BANKS-1:0] selected_bank_valid;
    logic [BANKS*BANK_ADDR_W-1:0] selected_bank_addr;
    logic [BANKS-1:0] selected_context_valid [0:MAX_K-1];
    logic [BANKS-1:0] selected_context_subtract [0:MAX_K-1];
    logic [TILE_BITS-1:0] selected_mask [0:MAX_K-1];
    logic [TILE_BITS-1:0] remaining_union_after;

    logic [RESPONSE_TAG_W-1:0] request_sequence_q;
    logic [RESPONSE_TAG_W-1:0] meta_tag_q [0:META_DEPTH-1];
    logic [GROUP_COUNT_W-1:0] meta_count_contexts_q [0:META_DEPTH-1];
    logic [MAX_K*CONTEXT_W-1:0] meta_contexts_q [0:META_DEPTH-1];
    logic [BANKS-1:0] meta_bank_valid_q [0:META_DEPTH-1];
    logic [MAX_K*BANKS-1:0] meta_context_valid_q [0:META_DEPTH-1];
    logic [MAX_K*BANKS-1:0] meta_context_subtract_q [0:META_DEPTH-1];
    logic meta_last_q [0:META_DEPTH-1];
    logic [META_PTR_W-1:0] meta_head_q;
    logic [META_PTR_W-1:0] meta_tail_q;
    logic [META_COUNT_W-1:0] meta_count_q;
    logic meta_credit;

    logic [RESPONSE_TAG_W-1:0] expected_tag;
    logic [GROUP_COUNT_W-1:0] expected_count;
    logic [MAX_K*CONTEXT_W-1:0] expected_contexts;
    logic [CONTEXT_W-1:0] expected_context [0:MAX_K-1];
    logic [BANKS-1:0] expected_bank_valid;
    logic [BANKS-1:0] expected_context_valid [0:MAX_K-1];
    logic [BANKS-1:0] expected_context_subtract [0:MAX_K-1];
    logic expected_last;
    logic response_contract_valid;

    logic signed [W_W:0] response_term [0:MAX_K-1][0:LANES-1][0:BANKS-1];
    logic signed [W_W+3:0] response_total [0:MAX_K-1][0:LANES-1];
    logic signed [ACC_W-1:0] response_sum [0:MAX_K-1][0:LANES-1];
    logic signed [ACC_W:0] response_acc_wide [0:MAX_K-1][0:LANES-1];
    logic response_acc_overflow;

    logic [TAG_W-1:0] complete_tag_q [0:COMPLETE_DEPTH-1];
    logic [COUNT_W-1:0] complete_source_count_q [0:COMPLETE_DEPTH-1];
    logic [LANES*ACC_W-1:0] complete_acc_q [0:COMPLETE_DEPTH-1];
    logic [COMPLETE_PTR_W-1:0] complete_head_q;
    logic [COMPLETE_PTR_W-1:0] complete_tail_q;
    logic [COMPLETE_COUNT_W-1:0] complete_count_q;
    logic [COMPLETE_COUNT_W:0] complete_credits;
    logic final_response_success;
    logic zero_launch_success;
    logic [GROUP_COUNT_W-1:0] complete_push_count;
    logic faulted_q;

    function automatic logic [COUNT_W-1:0] popcount_banks(
        input logic [BANKS-1:0] value
    );
        logic [COUNT_W-1:0] count;
        begin
            count = '0;
            for (int bank = 0; bank < BANKS; bank++)
                count = count + COUNT_W'(value[bank]);
            popcount_banks = count;
        end
    endfunction

`ifndef SYNTHESIS
    initial begin
        if (TILE_BITS != 256 || BANKS != 8 || LANES != 96
                || CONTEXTS != 16 || MAX_K != 4 || META_DEPTH != 16
                || COMPLETE_DEPTH != 16 || W_W != 8 || ACC_W != 19
                || CONTEXT_W != 4)
            $fatal(1, "M54 frozen K4-C16 P8-L96 geometry drift");
    end
`endif

    always_comb begin
        free_found = 1'b0;
        free_context = '0;
        context_occupancy = '0;
        for (int ctx = 0; ctx < CONTEXTS; ctx++) begin
            context_allocated_vector[ctx] = context_allocated_q[ctx];
            context_launched_vector[ctx] = context_launched_q[ctx];
            context_occupancy = context_occupancy
                + (context_allocated_q[ctx] ? 1'b1 : 1'b0);
            if (!free_found && !context_allocated_q[ctx]) begin
                free_found = 1'b1;
                free_context = CONTEXT_W'(ctx);
            end
        end
    end
    assign command_ready = !faulted_q && free_found;
    assign command_accept = command_valid && command_ready;
    assign command_accept_context = free_context;

    always_comb begin
        for (int slot = 0; slot < MAX_K; slot++)
            launch_context[slot] = launch_contexts[
                slot*CONTEXT_W +: CONTEXT_W];
        launch_legal = launch_context_count >= 1
            && launch_context_count <= MAX_K;
        launch_union = '0;
        for (int slot = 0; slot < MAX_K; slot++) begin
            if (slot < launch_context_count) begin
                launch_legal = launch_legal
                    && context_allocated_q[launch_context[slot]]
                    && !context_launched_q[launch_context[slot]];
                launch_union = launch_union
                    | context_add_q[launch_context[slot]]
                    | context_subtract_q[launch_context[slot]];
                for (int prior = 0; prior < MAX_K; prior++)
                    if (prior < slot && prior < launch_context_count)
                        launch_legal = launch_legal
                            && launch_context[slot] != launch_context[prior];
            end
        end
        launch_zero = launch_legal && launch_union == '0;
    end

    assign output_valid = !faulted_q && complete_count_q != 0;
    assign output_accept = output_valid && output_ready;
    assign output_tag = complete_count_q != 0
        ? complete_tag_q[complete_head_q] : '0;
    assign output_source_count = complete_count_q != 0
        ? complete_source_count_q[complete_head_q] : '0;
    assign output_acc = complete_count_q != 0
        ? complete_acc_q[complete_head_q] : '0;
    assign complete_credits = COMPLETE_DEPTH - complete_count_q
        + (output_accept ? 1'b1 : 1'b0);
    assign launch_ready = !faulted_q && !active_q
        && (!launch_zero || complete_credits >= launch_context_count);
    assign launch_accept = launch_valid && launch_ready;

    always_comb begin : select_lowest_union_row_per_bank
        selected_bank_valid = '0;
        selected_bank_addr = '0;
        remaining_union_after = '0;
        for (int slot = 0; slot < MAX_K; slot++) begin
            selected_context_valid[slot] = '0;
            selected_context_subtract[slot] = '0;
            selected_mask[slot] = '0;
        end
        for (int bank = 0; bank < BANKS; bank++) begin
            logic found;
            found = 1'b0;
            for (int row = 0; row < TILE_BITS/BANKS; row++) begin
                int source;
                logic any_context;
                source = row * BANKS + bank;
                any_context = 1'b0;
                for (int slot = 0; slot < MAX_K; slot++)
                    if (slot < active_count_q)
                        any_context = any_context
                            || context_add_q[active_context_q[slot]][source]
                            || context_subtract_q[active_context_q[slot]][source];
                if (!found && active_q && any_context) begin
                    found = 1'b1;
                    selected_bank_valid[bank] = 1'b1;
                    selected_bank_addr[bank*BANK_ADDR_W +: BANK_ADDR_W]
                        = BANK_ADDR_W'(row);
                    for (int slot = 0; slot < MAX_K; slot++) begin
                        if (slot < active_count_q) begin
                            selected_context_valid[slot][bank]
                                = context_add_q[active_context_q[slot]][source]
                                    || context_subtract_q[
                                        active_context_q[slot]][source];
                            selected_context_subtract[slot][bank]
                                = context_subtract_q[
                                    active_context_q[slot]][source];
                            selected_mask[slot][source]
                                = selected_context_valid[slot][bank];
                        end
                    end
                end
            end
        end
        for (int slot = 0; slot < MAX_K; slot++)
            if (slot < active_count_q)
                remaining_union_after = remaining_union_after
                    | ((context_add_q[active_context_q[slot]]
                        | context_subtract_q[active_context_q[slot]])
                       & ~selected_mask[slot]);
    end

    assign meta_credit = meta_count_q < META_DEPTH || response_accept;
    assign weight_request_valid = !faulted_q && active_q
        && |selected_bank_valid && meta_credit;
    assign request_accept = weight_request_valid && weight_request_ready;
    assign weight_request_tag = request_sequence_q;
    assign weight_request_context_count = active_count_q;
    assign weight_request_bank_valid = selected_bank_valid;
    assign weight_request_bank_addr = selected_bank_addr;
    assign weight_request_last = weight_request_valid
        && remaining_union_after == '0;
    always_comb begin
        weight_request_contexts = '0;
        weight_request_context_valid = '0;
        weight_request_context_subtract = '0;
        for (int slot = 0; slot < MAX_K; slot++) begin
            weight_request_contexts[slot*CONTEXT_W +: CONTEXT_W]
                = active_context_q[slot];
            weight_request_context_valid[slot*BANKS +: BANKS]
                = selected_context_valid[slot];
            weight_request_context_subtract[slot*BANKS +: BANKS]
                = selected_context_subtract[slot];
        end
    end

    assign expected_tag = meta_count_q != 0
        ? meta_tag_q[meta_head_q] : '0;
    assign expected_count = meta_count_q != 0
        ? meta_count_contexts_q[meta_head_q] : '0;
    assign expected_contexts = meta_count_q != 0
        ? meta_contexts_q[meta_head_q] : '0;
    assign expected_bank_valid = meta_count_q != 0
        ? meta_bank_valid_q[meta_head_q] : '0;
    assign expected_last = meta_count_q != 0
        ? meta_last_q[meta_head_q] : 1'b0;
    always_comb begin
        for (int slot = 0; slot < MAX_K; slot++) begin
            expected_context[slot] = expected_contexts[
                slot*CONTEXT_W +: CONTEXT_W];
            expected_context_valid[slot] = meta_count_q != 0
                ? meta_context_valid_q[meta_head_q][slot*BANKS +: BANKS]
                : '0;
            expected_context_subtract[slot] = meta_count_q != 0
                ? meta_context_subtract_q[meta_head_q][slot*BANKS +: BANKS]
                : '0;
        end
    end
    assign response_contract_valid =
        weight_response_tag == expected_tag
        && weight_response_context_count == expected_count
        && weight_response_contexts == expected_contexts
        && weight_response_bank_valid == expected_bank_valid;
    assign weight_response_ready = !faulted_q && meta_count_q != 0
        && (!expected_last || complete_credits >= expected_count);
    assign response_accept = weight_response_valid && weight_response_ready;

    always_comb begin : independently_sum_all_destinations
        response_acc_overflow = 1'b0;
        for (int slot = 0; slot < MAX_K; slot++) begin
            for (int lane = 0; lane < LANES; lane++) begin
                response_total[slot][lane] = '0;
                for (int bank = 0; bank < BANKS; bank++) begin
                    logic signed [W_W-1:0] raw_weight;
                    raw_weight = weight_response_data[
                        (bank*LANES + lane)*W_W +: W_W];
                    if (slot < expected_count
                            && expected_context_valid[slot][bank]) begin
                        if (expected_context_subtract[slot][bank])
                            response_term[slot][lane][bank]
                                = -{{1{raw_weight[W_W-1]}}, raw_weight};
                        else
                            response_term[slot][lane][bank]
                                = {{1{raw_weight[W_W-1]}}, raw_weight};
                    end else response_term[slot][lane][bank] = '0;
                    response_total[slot][lane]
                        = response_total[slot][lane]
                            + {{3{response_term[slot][lane][bank][W_W]}},
                               response_term[slot][lane][bank]};
                end
                response_sum[slot][lane]
                    = {{(ACC_W-(W_W+4)){
                            response_total[slot][lane][W_W+3]}},
                       response_total[slot][lane]};
                response_acc_wide[slot][lane]
                    = {{1{context_acc_q[expected_context[slot]][lane][ACC_W-1]}},
                        context_acc_q[expected_context[slot]][lane]}
                        + {{1{response_sum[slot][lane][ACC_W-1]}},
                           response_sum[slot][lane]};
                if (slot < expected_count
                        && response_acc_wide[slot][lane][ACC_W:ACC_W-1]
                            != 2'b00
                        && response_acc_wide[slot][lane][ACC_W:ACC_W-1]
                            != 2'b11)
                    response_acc_overflow = 1'b1;
            end
        end
    end

    assign final_response_success = response_accept
        && response_contract_valid && !response_acc_overflow && expected_last;
    assign zero_launch_success = launch_accept && launch_legal && launch_zero;
    assign complete_push_count = final_response_success ? expected_count
        : zero_launch_success ? launch_context_count : '0;
    assign response_metadata_occupancy = meta_count_q;
    assign complete_occupancy = complete_count_q;
    assign group_active = active_q;
    assign protocol_error = faulted_q;
    assign busy = context_occupancy != 0 || meta_count_q != 0
        || complete_count_q != 0 || active_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            faulted_q <= 1'b0;
            active_q <= 1'b0;
            active_count_q <= '0;
            request_sequence_q <= '0;
            meta_head_q <= '0;
            meta_tail_q <= '0;
            meta_count_q <= '0;
            complete_head_q <= '0;
            complete_tail_q <= '0;
            complete_count_q <= '0;
            for (int slot = 0; slot < MAX_K; slot++)
                active_context_q[slot] <= '0;
            for (int ctx = 0; ctx < CONTEXTS; ctx++) begin
                context_allocated_q[ctx] <= 1'b0;
                context_launched_q[ctx] <= 1'b0;
                context_tag_q[ctx] <= '0;
                context_add_q[ctx] <= '0;
                context_subtract_q[ctx] <= '0;
                context_source_count_q[ctx] <= '0;
                for (int lane = 0; lane < LANES; lane++)
                    context_acc_q[ctx][lane] <= '0;
            end
        end else begin
            if (command_accept) begin
                if (|(command_add_bits & command_subtract_bits)) begin
                    faulted_q <= 1'b1;
                end else begin
                    context_allocated_q[free_context] <= 1'b1;
                    context_launched_q[free_context] <= 1'b0;
                    context_tag_q[free_context] <= command_tag;
                    context_add_q[free_context] <= command_add_bits;
                    context_subtract_q[free_context] <= command_subtract_bits;
                    context_source_count_q[free_context] <= '0;
                    for (int lane = 0; lane < LANES; lane++)
                        context_acc_q[free_context][lane]
                            <= command_seed_acc[lane*ACC_W +: ACC_W];
                end
            end

            if (launch_accept) begin
                if (!launch_legal) begin
                    faulted_q <= 1'b1;
                end else if (!launch_zero) begin
                    active_q <= 1'b1;
                    active_count_q <= launch_context_count;
                    for (int slot = 0; slot < MAX_K; slot++) begin
                        active_context_q[slot] <= launch_context[slot];
                        if (slot < launch_context_count)
                            context_launched_q[launch_context[slot]] <= 1'b1;
                    end
                end
            end

            if (request_accept) begin
                request_sequence_q <= request_sequence_q + 1'b1;
                for (int slot = 0; slot < MAX_K; slot++) begin
                    if (slot < active_count_q) begin
                        context_add_q[active_context_q[slot]]
                            <= context_add_q[active_context_q[slot]]
                                & ~selected_mask[slot];
                        context_subtract_q[active_context_q[slot]]
                            <= context_subtract_q[active_context_q[slot]]
                                & ~selected_mask[slot];
                        context_source_count_q[active_context_q[slot]]
                            <= context_source_count_q[active_context_q[slot]]
                                + popcount_banks(selected_context_valid[slot]);
                    end
                end
                meta_tag_q[meta_tail_q] <= request_sequence_q;
                meta_count_contexts_q[meta_tail_q] <= active_count_q;
                meta_contexts_q[meta_tail_q] <= weight_request_contexts;
                meta_bank_valid_q[meta_tail_q] <= selected_bank_valid;
                meta_context_valid_q[meta_tail_q]
                    <= weight_request_context_valid;
                meta_context_subtract_q[meta_tail_q]
                    <= weight_request_context_subtract;
                meta_last_q[meta_tail_q] <= remaining_union_after == '0;
                meta_tail_q <= meta_tail_q + 1'b1;
            end

            if (response_accept) begin
                if (!response_contract_valid || response_acc_overflow) begin
                    faulted_q <= 1'b1;
                end else begin
                    for (int slot = 0; slot < MAX_K; slot++)
                        if (slot < expected_count)
                            for (int lane = 0; lane < LANES; lane++)
                                context_acc_q[expected_context[slot]][lane]
                                    <= response_acc_wide[slot][lane][ACC_W-1:0];
                    if (expected_last)
                        active_q <= 1'b0;
                end
                meta_head_q <= meta_head_q + 1'b1;
            end
            if (weight_response_valid && meta_count_q == 0)
                faulted_q <= 1'b1;

            case ({request_accept, response_accept})
                2'b10: meta_count_q <= meta_count_q + 1'b1;
                2'b01: meta_count_q <= meta_count_q - 1'b1;
                default: meta_count_q <= meta_count_q;
            endcase

            if (final_response_success) begin
                for (int slot = 0; slot < MAX_K; slot++) begin
                    if (slot < expected_count) begin
                        complete_tag_q[complete_tail_q
                            + COMPLETE_PTR_W'(slot)]
                            <= context_tag_q[expected_context[slot]];
                        complete_source_count_q[complete_tail_q
                            + COMPLETE_PTR_W'(slot)]
                            <= context_source_count_q[expected_context[slot]];
                        for (int lane = 0; lane < LANES; lane++)
                            complete_acc_q[complete_tail_q
                                + COMPLETE_PTR_W'(slot)]
                                [lane*ACC_W +: ACC_W]
                                <= response_acc_wide[slot][lane][ACC_W-1:0];
                        context_allocated_q[expected_context[slot]] <= 1'b0;
                        context_launched_q[expected_context[slot]] <= 1'b0;
                    end
                end
            end else if (zero_launch_success) begin
                for (int slot = 0; slot < MAX_K; slot++) begin
                    if (slot < launch_context_count) begin
                        complete_tag_q[complete_tail_q
                            + COMPLETE_PTR_W'(slot)]
                            <= context_tag_q[launch_context[slot]];
                        complete_source_count_q[complete_tail_q
                            + COMPLETE_PTR_W'(slot)] <= '0;
                        for (int lane = 0; lane < LANES; lane++)
                            complete_acc_q[complete_tail_q
                                + COMPLETE_PTR_W'(slot)]
                                [lane*ACC_W +: ACC_W]
                                <= context_acc_q[launch_context[slot]][lane];
                        context_allocated_q[launch_context[slot]] <= 1'b0;
                        context_launched_q[launch_context[slot]] <= 1'b0;
                    end
                end
            end

            if (complete_push_count != 0)
                complete_tail_q <= complete_tail_q + complete_push_count;
            if (output_accept)
                complete_head_q <= complete_head_q + 1'b1;
            case ({complete_push_count, output_accept})
                {3'd0,1'b0}: complete_count_q <= complete_count_q;
                {3'd0,1'b1}: complete_count_q <= complete_count_q - 1'b1;
                {3'd1,1'b0}: complete_count_q <= complete_count_q + 1;
                {3'd1,1'b1}: complete_count_q <= complete_count_q;
                {3'd2,1'b0}: complete_count_q <= complete_count_q + 2;
                {3'd2,1'b1}: complete_count_q <= complete_count_q + 1;
                {3'd3,1'b0}: complete_count_q <= complete_count_q + 3;
                {3'd3,1'b1}: complete_count_q <= complete_count_q + 2;
                {3'd4,1'b0}: complete_count_q <= complete_count_q + 4;
                {3'd4,1'b1}: complete_count_q <= complete_count_q + 3;
                default: complete_count_q <= complete_count_q;
            endcase
        end
    end
endmodule

`default_nettype wire
