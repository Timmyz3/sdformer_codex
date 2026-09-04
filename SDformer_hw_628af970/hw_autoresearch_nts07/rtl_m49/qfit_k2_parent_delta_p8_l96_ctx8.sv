`timescale 1ns/1ps
`default_nettype none

// M49 standalone dual-destination parent-delta source engine.
//
// Two explicitly launched destination contexts consume the union of their
// signed source masks.  For each bank and cycle the lowest remaining union
// row is read exactly once; the returned row independently updates zero, one,
// or two signed-19x96 accumulators.  A final K2 response atomically enqueues
// two complete vectors and releases both contexts in that same cycle.
module qfit_k2_parent_delta_p8_l96_ctx8 #(
    parameter int TILE_BITS = 256,
    parameter int BANKS = 8,
    parameter int LANES = 96,
    parameter int CONTEXTS = 8,
    parameter int META_DEPTH = 16,
    parameter int COMPLETE_DEPTH = 16,
    parameter int TAG_W = 48,
    parameter int W_W = 8,
    parameter int ACC_W = 19,
    parameter int BANK_ADDR_W = $clog2(TILE_BITS/BANKS),
    parameter int COUNT_W = $clog2(TILE_BITS+1),
    parameter int CONTEXT_W = $clog2(CONTEXTS),
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
    input  logic [CONTEXT_W-1:0]                  launch_context0,
    input  logic                                  launch_context1_valid,
    input  logic [CONTEXT_W-1:0]                  launch_context1,
    output logic                                  launch_accept,

    output logic                                  weight_request_valid,
    input  logic                                  weight_request_ready,
    output logic [BANKS-1:0]                      weight_request_bank_valid,
    output logic [BANKS*BANK_ADDR_W-1:0]          weight_request_bank_addr,
    output logic [CONTEXT_W-1:0]                  weight_request_context0,
    output logic                                  weight_request_context1_valid,
    output logic [CONTEXT_W-1:0]                  weight_request_context1,
    output logic [BANKS-1:0]                      weight_request_context0_valid,
    output logic [BANKS-1:0]                      weight_request_context0_subtract,
    output logic [BANKS-1:0]                      weight_request_context1_valid_by_bank,
    output logic [BANKS-1:0]                      weight_request_context1_subtract,
    output logic                                  weight_request_last,
    output logic                                  request_accept,

    input  logic                                  weight_response_valid,
    output logic                                  weight_response_ready,
    input  logic [CONTEXT_W-1:0]                  weight_response_context0,
    input  logic                                  weight_response_context1_valid,
    input  logic [CONTEXT_W-1:0]                  weight_response_context1,
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
    logic active_q;
    logic [CONTEXT_W-1:0] active_context0_q;
    logic active_context1_valid_q;
    logic [CONTEXT_W-1:0] active_context1_q;

    logic launch_legal;
    logic [TILE_BITS-1:0] launch_union;
    logic launch_zero;
    logic [1:0] launch_entry_count;

    logic [BANKS-1:0] selected_bank_valid;
    logic [BANKS*BANK_ADDR_W-1:0] selected_bank_addr;
    logic [BANKS-1:0] selected_context0_valid;
    logic [BANKS-1:0] selected_context0_subtract;
    logic [BANKS-1:0] selected_context1_valid;
    logic [BANKS-1:0] selected_context1_subtract;
    logic [TILE_BITS-1:0] selected_mask0;
    logic [TILE_BITS-1:0] selected_mask1;
    logic [TILE_BITS-1:0] remaining_union_after;

    logic [CONTEXT_W-1:0] meta_context0_q [0:META_DEPTH-1];
    logic meta_context1_valid_q [0:META_DEPTH-1];
    logic [CONTEXT_W-1:0] meta_context1_q [0:META_DEPTH-1];
    logic [BANKS-1:0] meta_bank_valid_q [0:META_DEPTH-1];
    logic [BANKS-1:0] meta_context0_valid_q [0:META_DEPTH-1];
    logic [BANKS-1:0] meta_context0_subtract_q [0:META_DEPTH-1];
    logic [BANKS-1:0] meta_context1_valid_by_bank_q [0:META_DEPTH-1];
    logic [BANKS-1:0] meta_context1_subtract_q [0:META_DEPTH-1];
    logic meta_last_q [0:META_DEPTH-1];
    logic [META_PTR_W-1:0] meta_head_q;
    logic [META_PTR_W-1:0] meta_tail_q;
    logic [META_COUNT_W-1:0] meta_count_q;
    logic meta_credit;

    logic [CONTEXT_W-1:0] expected_context0;
    logic expected_context1_valid;
    logic [CONTEXT_W-1:0] expected_context1;
    logic [BANKS-1:0] expected_bank_valid;
    logic [BANKS-1:0] expected_context0_valid;
    logic [BANKS-1:0] expected_context0_subtract;
    logic [BANKS-1:0] expected_context1_valid_by_bank;
    logic [BANKS-1:0] expected_context1_subtract;
    logic expected_last;
    logic response_contract_valid;

    logic signed [W_W:0] response_term0 [0:LANES-1][0:BANKS-1];
    logic signed [W_W:0] response_term1 [0:LANES-1][0:BANKS-1];
    logic signed [W_W+3:0] response_total0 [0:LANES-1];
    logic signed [W_W+3:0] response_total1 [0:LANES-1];
    logic signed [ACC_W-1:0] response_sum0 [0:LANES-1];
    logic signed [ACC_W-1:0] response_sum1 [0:LANES-1];
    logic signed [ACC_W:0] response_acc_wide0 [0:LANES-1];
    logic signed [ACC_W:0] response_acc_wide1 [0:LANES-1];
    logic response_acc_overflow;

    logic [TAG_W-1:0] complete_tag_q [0:COMPLETE_DEPTH-1];
    logic [COUNT_W-1:0] complete_source_count_q [0:COMPLETE_DEPTH-1];
    logic [LANES*ACC_W-1:0] complete_acc_q [0:COMPLETE_DEPTH-1];
    logic [COMPLETE_PTR_W-1:0] complete_head_q;
    logic [COMPLETE_PTR_W-1:0] complete_tail_q;
    logic [COMPLETE_COUNT_W-1:0] complete_count_q;
    logic [COMPLETE_COUNT_W:0] complete_credits;
    logic [1:0] final_entry_count;
    logic last_response_has_credit;
    logic final_response_success;
    logic zero_launch_success;
    logic [1:0] complete_push_count;
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
                || CONTEXTS != 8 || META_DEPTH != 16
                || COMPLETE_DEPTH != 16 || W_W != 8 || ACC_W != 19)
            $fatal(1, "M49 frozen K2-CTX8 P8-L96 geometry drift");
    end
`endif

    always_comb begin
        free_found = 1'b0;
        free_context = '0;
        context_occupancy = '0;
        for (int ctx = 0; ctx < CONTEXTS; ctx++) begin
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
        launch_legal = context_allocated_q[launch_context0]
            && !context_launched_q[launch_context0];
        if (launch_context1_valid)
            launch_legal = launch_legal
                && launch_context1 != launch_context0
                && context_allocated_q[launch_context1]
                && !context_launched_q[launch_context1];
        launch_union = context_add_q[launch_context0]
            | context_subtract_q[launch_context0];
        if (launch_context1_valid)
            launch_union = launch_union | context_add_q[launch_context1]
                | context_subtract_q[launch_context1];
        launch_zero = launch_legal && launch_union == '0;
        launch_entry_count = launch_context1_valid ? 2 : 1;
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
        && (!launch_zero || complete_credits >= launch_entry_count);
    assign launch_accept = launch_valid && launch_ready;

    always_comb begin : select_lowest_union_row_per_bank
        selected_bank_valid = '0;
        selected_bank_addr = '0;
        selected_context0_valid = '0;
        selected_context0_subtract = '0;
        selected_context1_valid = '0;
        selected_context1_subtract = '0;
        selected_mask0 = '0;
        selected_mask1 = '0;
        for (int bank = 0; bank < BANKS; bank++) begin
            logic found;
            found = 1'b0;
            for (int row = 0; row < TILE_BITS/BANKS; row++) begin
                int source;
                source = row * BANKS + bank;
                if (!found && active_q
                        && (context_add_q[active_context0_q][source]
                            || context_subtract_q[active_context0_q][source]
                            || (active_context1_valid_q
                                && (context_add_q[active_context1_q][source]
                                    || context_subtract_q[active_context1_q][source])))) begin
                    found = 1'b1;
                    selected_bank_valid[bank] = 1'b1;
                    selected_bank_addr[bank*BANK_ADDR_W +: BANK_ADDR_W]
                        = BANK_ADDR_W'(row);
                    selected_context0_valid[bank]
                        = context_add_q[active_context0_q][source]
                            || context_subtract_q[active_context0_q][source];
                    selected_context0_subtract[bank]
                        = context_subtract_q[active_context0_q][source];
                    selected_mask0[source] = selected_context0_valid[bank];
                    if (active_context1_valid_q) begin
                        selected_context1_valid[bank]
                            = context_add_q[active_context1_q][source]
                                || context_subtract_q[active_context1_q][source];
                        selected_context1_subtract[bank]
                            = context_subtract_q[active_context1_q][source];
                        selected_mask1[source] = selected_context1_valid[bank];
                    end
                end
            end
        end
        remaining_union_after =
            ((context_add_q[active_context0_q]
              | context_subtract_q[active_context0_q]) & ~selected_mask0);
        if (active_context1_valid_q)
            remaining_union_after = remaining_union_after
                | ((context_add_q[active_context1_q]
                    | context_subtract_q[active_context1_q]) & ~selected_mask1);
    end

    assign meta_credit = meta_count_q < META_DEPTH || response_accept;
    assign weight_request_valid = !faulted_q && active_q
        && |selected_bank_valid && meta_credit;
    assign request_accept = weight_request_valid && weight_request_ready;
    assign weight_request_bank_valid = selected_bank_valid;
    assign weight_request_bank_addr = selected_bank_addr;
    assign weight_request_context0 = active_context0_q;
    assign weight_request_context1_valid = active_context1_valid_q;
    assign weight_request_context1 = active_context1_q;
    assign weight_request_context0_valid = selected_context0_valid;
    assign weight_request_context0_subtract = selected_context0_subtract;
    assign weight_request_context1_valid_by_bank = selected_context1_valid;
    assign weight_request_context1_subtract = selected_context1_subtract;
    assign weight_request_last = weight_request_valid
        && remaining_union_after == '0;

    assign expected_context0 = meta_count_q != 0
        ? meta_context0_q[meta_head_q] : '0;
    assign expected_context1_valid = meta_count_q != 0
        ? meta_context1_valid_q[meta_head_q] : 1'b0;
    assign expected_context1 = meta_count_q != 0
        ? meta_context1_q[meta_head_q] : '0;
    assign expected_bank_valid = meta_count_q != 0
        ? meta_bank_valid_q[meta_head_q] : '0;
    assign expected_context0_valid = meta_count_q != 0
        ? meta_context0_valid_q[meta_head_q] : '0;
    assign expected_context0_subtract = meta_count_q != 0
        ? meta_context0_subtract_q[meta_head_q] : '0;
    assign expected_context1_valid_by_bank
        = meta_count_q != 0
            ? meta_context1_valid_by_bank_q[meta_head_q] : '0;
    assign expected_context1_subtract
        = meta_count_q != 0
            ? meta_context1_subtract_q[meta_head_q] : '0;
    assign expected_last = meta_count_q != 0
        ? meta_last_q[meta_head_q] : 1'b0;
    assign response_contract_valid =
        weight_response_context0 == expected_context0
        && weight_response_context1_valid == expected_context1_valid
        && (!expected_context1_valid
            || weight_response_context1 == expected_context1)
        && weight_response_bank_valid == expected_bank_valid;
    assign final_entry_count = expected_context1_valid ? 2 : 1;
    assign last_response_has_credit = !expected_last
        || complete_credits >= final_entry_count;
    assign weight_response_ready = !faulted_q && meta_count_q != 0
        && last_response_has_credit;
    assign response_accept = weight_response_valid && weight_response_ready;

    always_comb begin : sum_response_for_both_destinations
        response_acc_overflow = 1'b0;
        for (int lane = 0; lane < LANES; lane++) begin
            response_total0[lane] = '0;
            response_total1[lane] = '0;
            for (int bank = 0; bank < BANKS; bank++) begin
                logic signed [W_W-1:0] raw_weight;
                raw_weight = weight_response_data[
                    (bank*LANES + lane)*W_W +: W_W];
                if (expected_context0_valid[bank]) begin
                    if (expected_context0_subtract[bank])
                        response_term0[lane][bank]
                            = -{{1{raw_weight[W_W-1]}}, raw_weight};
                    else
                        response_term0[lane][bank]
                            = {{1{raw_weight[W_W-1]}}, raw_weight};
                end else response_term0[lane][bank] = '0;
                if (expected_context1_valid_by_bank[bank]) begin
                    if (expected_context1_subtract[bank])
                        response_term1[lane][bank]
                            = -{{1{raw_weight[W_W-1]}}, raw_weight};
                    else
                        response_term1[lane][bank]
                            = {{1{raw_weight[W_W-1]}}, raw_weight};
                end else response_term1[lane][bank] = '0;
                response_total0[lane] = response_total0[lane]
                    + {{3{response_term0[lane][bank][W_W]}},
                       response_term0[lane][bank]};
                response_total1[lane] = response_total1[lane]
                    + {{3{response_term1[lane][bank][W_W]}},
                       response_term1[lane][bank]};
            end
            response_sum0[lane]
                = {{(ACC_W-(W_W+4)){response_total0[lane][W_W+3]}},
                    response_total0[lane]};
            response_sum1[lane]
                = {{(ACC_W-(W_W+4)){response_total1[lane][W_W+3]}},
                    response_total1[lane]};
            response_acc_wide0[lane]
                = {{1{context_acc_q[expected_context0][lane][ACC_W-1]}},
                    context_acc_q[expected_context0][lane]}
                    + {{1{response_sum0[lane][ACC_W-1]}}, response_sum0[lane]};
            response_acc_wide1[lane]
                = {{1{context_acc_q[expected_context1][lane][ACC_W-1]}},
                    context_acc_q[expected_context1][lane]}
                    + {{1{response_sum1[lane][ACC_W-1]}}, response_sum1[lane]};
            if (response_acc_wide0[lane][ACC_W:ACC_W-1] != 2'b00
                    && response_acc_wide0[lane][ACC_W:ACC_W-1] != 2'b11)
                response_acc_overflow = 1'b1;
            if (expected_context1_valid
                    && response_acc_wide1[lane][ACC_W:ACC_W-1] != 2'b00
                    && response_acc_wide1[lane][ACC_W:ACC_W-1] != 2'b11)
                response_acc_overflow = 1'b1;
        end
    end

    assign final_response_success = response_accept
        && response_contract_valid && !response_acc_overflow && expected_last;
    assign zero_launch_success = launch_accept && launch_legal && launch_zero;
    assign complete_push_count = final_response_success ? final_entry_count
        : zero_launch_success ? launch_entry_count : 0;

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
            active_context0_q <= '0;
            active_context1_valid_q <= 1'b0;
            active_context1_q <= '0;
            meta_head_q <= '0;
            meta_tail_q <= '0;
            meta_count_q <= '0;
            complete_head_q <= '0;
            complete_tail_q <= '0;
            complete_count_q <= '0;
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
                    active_context0_q <= launch_context0;
                    active_context1_valid_q <= launch_context1_valid;
                    active_context1_q <= launch_context1;
                    context_launched_q[launch_context0] <= 1'b1;
                    if (launch_context1_valid)
                        context_launched_q[launch_context1] <= 1'b1;
                end
            end

            if (request_accept) begin
                context_add_q[active_context0_q]
                    <= context_add_q[active_context0_q] & ~selected_mask0;
                context_subtract_q[active_context0_q]
                    <= context_subtract_q[active_context0_q] & ~selected_mask0;
                context_source_count_q[active_context0_q]
                    <= context_source_count_q[active_context0_q]
                        + popcount_banks(selected_context0_valid);
                if (active_context1_valid_q) begin
                    context_add_q[active_context1_q]
                        <= context_add_q[active_context1_q] & ~selected_mask1;
                    context_subtract_q[active_context1_q]
                        <= context_subtract_q[active_context1_q] & ~selected_mask1;
                    context_source_count_q[active_context1_q]
                        <= context_source_count_q[active_context1_q]
                            + popcount_banks(selected_context1_valid);
                end
                meta_context0_q[meta_tail_q] <= active_context0_q;
                meta_context1_valid_q[meta_tail_q]
                    <= active_context1_valid_q;
                meta_context1_q[meta_tail_q] <= active_context1_q;
                meta_bank_valid_q[meta_tail_q] <= selected_bank_valid;
                meta_context0_valid_q[meta_tail_q] <= selected_context0_valid;
                meta_context0_subtract_q[meta_tail_q]
                    <= selected_context0_subtract;
                meta_context1_valid_by_bank_q[meta_tail_q]
                    <= selected_context1_valid;
                meta_context1_subtract_q[meta_tail_q]
                    <= selected_context1_subtract;
                meta_last_q[meta_tail_q] <= remaining_union_after == '0;
                meta_tail_q <= meta_tail_q + 1'b1;
            end

            if (response_accept) begin
                if (!response_contract_valid || response_acc_overflow) begin
                    faulted_q <= 1'b1;
                end else begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        context_acc_q[expected_context0][lane]
                            <= response_acc_wide0[lane][ACC_W-1:0];
                        if (expected_context1_valid)
                            context_acc_q[expected_context1][lane]
                                <= response_acc_wide1[lane][ACC_W-1:0];
                    end
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

            // One or two complete entries are written atomically.  For a
            // final response the FIFO payload includes that response's sum.
            if (final_response_success) begin
                complete_tag_q[complete_tail_q]
                    <= context_tag_q[expected_context0];
                complete_source_count_q[complete_tail_q]
                    <= context_source_count_q[expected_context0];
                for (int lane = 0; lane < LANES; lane++)
                    complete_acc_q[complete_tail_q][lane*ACC_W +: ACC_W]
                        <= response_acc_wide0[lane][ACC_W-1:0];
                context_allocated_q[expected_context0] <= 1'b0;
                context_launched_q[expected_context0] <= 1'b0;
                if (expected_context1_valid) begin
                    complete_tag_q[complete_tail_q + 1'b1]
                        <= context_tag_q[expected_context1];
                    complete_source_count_q[complete_tail_q + 1'b1]
                        <= context_source_count_q[expected_context1];
                    for (int lane = 0; lane < LANES; lane++)
                        complete_acc_q[complete_tail_q + 1'b1][lane*ACC_W +: ACC_W]
                            <= response_acc_wide1[lane][ACC_W-1:0];
                    context_allocated_q[expected_context1] <= 1'b0;
                    context_launched_q[expected_context1] <= 1'b0;
                end
            end else if (zero_launch_success) begin
                complete_tag_q[complete_tail_q]
                    <= context_tag_q[launch_context0];
                complete_source_count_q[complete_tail_q] <= '0;
                for (int lane = 0; lane < LANES; lane++)
                    complete_acc_q[complete_tail_q][lane*ACC_W +: ACC_W]
                        <= context_acc_q[launch_context0][lane];
                context_allocated_q[launch_context0] <= 1'b0;
                context_launched_q[launch_context0] <= 1'b0;
                if (launch_context1_valid) begin
                    complete_tag_q[complete_tail_q + 1'b1]
                        <= context_tag_q[launch_context1];
                    complete_source_count_q[complete_tail_q + 1'b1] <= '0;
                    for (int lane = 0; lane < LANES; lane++)
                        complete_acc_q[complete_tail_q + 1'b1][lane*ACC_W +: ACC_W]
                            <= context_acc_q[launch_context1][lane];
                    context_allocated_q[launch_context1] <= 1'b0;
                    context_launched_q[launch_context1] <= 1'b0;
                end
            end

            if (complete_push_count != 0)
                complete_tail_q <= complete_tail_q + complete_push_count;
            if (output_accept)
                complete_head_q <= complete_head_q + 1'b1;
            case ({complete_push_count, output_accept})
                3'b010: complete_count_q <= complete_count_q + 1;
                3'b100: complete_count_q <= complete_count_q + 2;
                3'b001: complete_count_q <= complete_count_q - 1;
                3'b011: complete_count_q <= complete_count_q;
                3'b101: complete_count_q <= complete_count_q + 1;
                default: complete_count_q <= complete_count_q;
            endcase
        end
    end
endmodule

`default_nettype wire
