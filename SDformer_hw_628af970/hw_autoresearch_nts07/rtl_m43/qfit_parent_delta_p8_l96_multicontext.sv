`timescale 1ns/1ps
`default_nettype none

// M43 signed parent-delta source engine.
//
// Four accumulator contexts decouple tile-row commands from the one-cycle
// banked weight response path.  Each request selects at most one add/subtract
// source from each of eight banks and broadcasts eight signed-INT8 weight rows
// across 96 output lanes.  Memory ownership and parent readiness are external;
// command_seed_acc must be the exact selected tile-parent partial output.
module qfit_parent_delta_p8_l96_multicontext #(
    parameter int TILE_BITS = 256,
    parameter int ISSUE_WIDTH = 8,
    parameter int OUT_LANES = 96,
    parameter int CONTEXTS = 4,
    parameter int META_DEPTH = 16,
    parameter int TAG_W = 48,
    parameter int W_W = 8,
    parameter int ACC_W = 19,
    parameter int INDEX_W = $clog2(TILE_BITS),
    parameter int BANK_ADDR_W = $clog2(TILE_BITS/ISSUE_WIDTH),
    parameter int COUNT_W = $clog2(TILE_BITS+1),
    parameter int CONTEXT_W = $clog2(CONTEXTS),
    parameter int META_PTR_W = $clog2(META_DEPTH),
    parameter int META_COUNT_W = $clog2(META_DEPTH+1)
) (
    input  logic                                  clk_core,
    input  logic                                  rst_core,

    input  logic                                  command_valid,
    output logic                                  command_ready,
    input  logic [TAG_W-1:0]                      command_tag,
    input  logic [TILE_BITS-1:0]                  command_add_bits,
    input  logic [TILE_BITS-1:0]                  command_subtract_bits,
    input  logic [OUT_LANES*ACC_W-1:0]            command_seed_acc,

    output logic                                  weight_request_valid,
    input  logic                                  weight_request_ready,
    output logic [ISSUE_WIDTH-1:0]                weight_request_bank_valid,
    output logic [ISSUE_WIDTH*BANK_ADDR_W-1:0]    weight_request_bank_addr,
    output logic [ISSUE_WIDTH-1:0]                weight_request_bank_subtract,
    output logic                                  weight_request_last,
    output logic [CONTEXT_W-1:0]                  weight_request_context,

    input  logic                                  weight_response_valid,
    output logic                                  weight_response_ready,
    input  logic [CONTEXT_W-1:0]                  weight_response_context,
    input  logic [ISSUE_WIDTH-1:0]                weight_response_bank_valid,
    input  logic [ISSUE_WIDTH*OUT_LANES*W_W-1:0] weight_response_data,

    output logic                                  output_valid,
    input  logic                                  output_ready,
    output logic [TAG_W-1:0]                      output_tag,
    output logic [COUNT_W-1:0]                    output_source_count,
    output logic [OUT_LANES*ACC_W-1:0]            output_acc,

    output logic                                  protocol_error,
    output logic                                  busy,
    output logic [CONTEXT_W:0]                    context_occupancy,
    output logic [META_COUNT_W-1:0]               response_metadata_occupancy,
    output logic                                  command_accept,
    output logic                                  request_accept,
    output logic                                  response_accept,
    output logic                                  output_accept
);
    logic context_allocated_q [0:CONTEXTS-1];
    logic context_issued_all_q [0:CONTEXTS-1];
    logic context_done_q [0:CONTEXTS-1];
    logic [TAG_W-1:0] context_tag_q [0:CONTEXTS-1];
    logic [TILE_BITS-1:0] context_add_q [0:CONTEXTS-1];
    logic [TILE_BITS-1:0] context_subtract_q [0:CONTEXTS-1];
    logic [COUNT_W-1:0] context_source_count_q [0:CONTEXTS-1];
    logic signed [ACC_W-1:0] context_acc_q [0:CONTEXTS-1][0:OUT_LANES-1];

    logic free_found;
    logic [CONTEXT_W-1:0] free_context;
    logic output_found;
    logic [CONTEXT_W-1:0] output_context;
    logic output_lock_valid_q;
    logic [CONTEXT_W-1:0] output_lock_context_q;
    logic [CONTEXT_W-1:0] issue_rr_q;
    logic issue_found;
    logic [CONTEXT_W-1:0] issue_context;
    logic issue_lock_valid_q;
    logic [CONTEXT_W-1:0] issue_lock_context_q;

    logic [ISSUE_WIDTH-1:0] selected_bank_valid_by_context [0:CONTEXTS-1];
    logic [ISSUE_WIDTH-1:0] selected_bank_subtract_by_context [0:CONTEXTS-1];
    logic [ISSUE_WIDTH*BANK_ADDR_W-1:0]
        selected_bank_addr_by_context [0:CONTEXTS-1];
    logic [TILE_BITS-1:0] selected_mask_by_context [0:CONTEXTS-1];
    logic [TILE_BITS-1:0] remaining_after_by_context [0:CONTEXTS-1];
    logic [COUNT_W-1:0] selected_count_by_context [0:CONTEXTS-1];

    logic [CONTEXT_W-1:0] meta_context_q [0:META_DEPTH-1];
    logic [ISSUE_WIDTH-1:0] meta_bank_valid_q [0:META_DEPTH-1];
    logic [ISSUE_WIDTH-1:0] meta_bank_subtract_q [0:META_DEPTH-1];
    logic meta_last_q [0:META_DEPTH-1];
    logic [META_PTR_W-1:0] meta_head_q;
    logic [META_PTR_W-1:0] meta_tail_q;
    logic [META_COUNT_W-1:0] meta_count_q;
    logic meta_credit;
    logic [CONTEXT_W-1:0] response_context;
    logic [ISSUE_WIDTH-1:0] response_expected_bank_valid;
    logic [ISSUE_WIDTH-1:0] response_expected_bank_subtract;
    logic response_expected_last;
    logic response_contract_valid;
    logic signed [W_W:0] response_term [0:OUT_LANES-1][0:ISSUE_WIDTH-1];
    logic signed [W_W+1:0] response_pair [0:OUT_LANES-1][0:3];
    logic signed [W_W+2:0] response_quad [0:OUT_LANES-1][0:1];
    logic signed [W_W+3:0] response_total [0:OUT_LANES-1];
    logic signed [ACC_W-1:0] response_sum [0:OUT_LANES-1];
    logic signed [ACC_W:0] response_acc_wide [0:OUT_LANES-1];
    logic response_acc_overflow;
    logic faulted_q;

    function automatic logic [COUNT_W-1:0] popcount_banks(
        input logic [ISSUE_WIDTH-1:0] value
    );
        logic [COUNT_W-1:0] count;
        begin
            count = '0;
            for (int bank = 0; bank < ISSUE_WIDTH; bank = bank + 1)
                count = count + COUNT_W'(value[bank]);
            popcount_banks = count;
        end
    endfunction

`ifndef SYNTHESIS
    initial begin
        if (TILE_BITS != 256 || ISSUE_WIDTH != 8 || OUT_LANES != 96
                || CONTEXTS != 4 || W_W != 8 || ACC_W != 19)
            $fatal(1, "M43 frozen P8-L96 geometry drift");
        if (TILE_BITS % ISSUE_WIDTH != 0)
            $fatal(1, "M43 tile must divide evenly across source banks");
        if ((CONTEXTS & (CONTEXTS-1)) != 0
                || (META_DEPTH & (META_DEPTH-1)) != 0)
            $fatal(1, "M43 context and metadata depths must be powers of two");
    end
`endif

    always_comb begin : select_each_context
        for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
            selected_bank_valid_by_context[ctx] = '0;
            selected_bank_subtract_by_context[ctx] = '0;
            selected_bank_addr_by_context[ctx] = '0;
            selected_mask_by_context[ctx] = '0;
            for (int bank = 0; bank < ISSUE_WIDTH; bank = bank + 1) begin
                logic found;
                found = 1'b0;
                for (int position = bank; position < TILE_BITS;
                     position = position + ISSUE_WIDTH) begin
                    if (!found && (context_add_q[ctx][position]
                            || context_subtract_q[ctx][position])) begin
                        found = 1'b1;
                        selected_bank_valid_by_context[ctx][bank] = 1'b1;
                        selected_bank_subtract_by_context[ctx][bank]
                            = context_subtract_q[ctx][position];
                        selected_bank_addr_by_context[ctx][
                            bank*BANK_ADDR_W +: BANK_ADDR_W]
                            = BANK_ADDR_W'(position / ISSUE_WIDTH);
                        selected_mask_by_context[ctx][position] = 1'b1;
                    end
                end
            end
            remaining_after_by_context[ctx]
                = (context_add_q[ctx] | context_subtract_q[ctx])
                    & ~selected_mask_by_context[ctx];
            selected_count_by_context[ctx]
                = popcount_banks(selected_bank_valid_by_context[ctx]);
        end
    end

    always_comb begin : arbitrate_contexts
        free_found = 1'b0;
        free_context = '0;
        output_found = output_lock_valid_q;
        output_context = output_lock_context_q;
        for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
            if (!free_found && !context_allocated_q[ctx]) begin
                free_found = 1'b1;
                free_context = CONTEXT_W'(ctx);
            end
            if (!output_found && context_allocated_q[ctx]
                    && context_done_q[ctx]) begin
                output_found = 1'b1;
                output_context = CONTEXT_W'(ctx);
            end
        end

        issue_found = issue_lock_valid_q;
        issue_context = issue_lock_context_q;
        for (int offset = 0; offset < CONTEXTS; offset = offset + 1) begin
            logic [CONTEXT_W-1:0] candidate;
            candidate = issue_rr_q + CONTEXT_W'(offset);
            if (!issue_found && context_allocated_q[candidate]
                    && !context_issued_all_q[candidate]
                    && |selected_bank_valid_by_context[candidate]) begin
                issue_found = 1'b1;
                issue_context = candidate;
            end
        end
    end

    assign meta_credit = meta_count_q < META_DEPTH || response_accept;
    assign command_ready = !faulted_q && free_found;
    assign command_accept = command_valid && command_ready;
    assign weight_request_valid = !faulted_q && issue_found && meta_credit;
    assign request_accept = weight_request_valid && weight_request_ready;
    assign weight_request_bank_valid = issue_found
        ? selected_bank_valid_by_context[issue_context] : '0;
    assign weight_request_bank_subtract = issue_found
        ? selected_bank_subtract_by_context[issue_context] : '0;
    assign weight_request_bank_addr = issue_found
        ? selected_bank_addr_by_context[issue_context] : '0;
    assign weight_request_last = weight_request_valid
        && remaining_after_by_context[issue_context] == '0;
    assign weight_request_context = issue_context;

    assign response_context = meta_context_q[meta_head_q];
    assign response_expected_bank_valid = meta_bank_valid_q[meta_head_q];
    assign response_expected_bank_subtract = meta_bank_subtract_q[meta_head_q];
    assign response_expected_last = meta_last_q[meta_head_q];
    assign weight_response_ready = !faulted_q && meta_count_q != 0;
    assign response_accept = weight_response_valid && weight_response_ready;
    assign response_contract_valid
        = weight_response_context == response_context
            && weight_response_bank_valid == response_expected_bank_valid;

    always_comb begin : sum_weight_response
        response_acc_overflow = 1'b0;
        for (int lane = 0; lane < OUT_LANES; lane = lane + 1) begin
            for (int bank = 0; bank < ISSUE_WIDTH; bank = bank + 1) begin
                logic signed [W_W-1:0] raw_weight;
                raw_weight = weight_response_data[
                    (bank*OUT_LANES + lane)*W_W +: W_W];
                if (response_expected_bank_valid[bank]) begin
                    if (response_expected_bank_subtract[bank])
                        response_term[lane][bank]
                            = -{{1{raw_weight[W_W-1]}}, raw_weight};
                    else
                        response_term[lane][bank]
                            = {{1{raw_weight[W_W-1]}}, raw_weight};
                end else begin
                    response_term[lane][bank] = '0;
                end
            end
            // The narrow balanced tree is exact for all eight signed-INT8
            // terms, including subtracting -128.  It avoids seven serial
            // ACC_W-wide additions on every output lane.
            for (int pair = 0; pair < 4; pair = pair + 1) begin
                response_pair[lane][pair]
                    = {{1{response_term[lane][pair*2][W_W]}},
                        response_term[lane][pair*2]}
                        + {{1{response_term[lane][pair*2+1][W_W]}},
                           response_term[lane][pair*2+1]};
            end
            for (int quad = 0; quad < 2; quad = quad + 1)
                response_quad[lane][quad]
                    = {{1{response_pair[lane][quad*2][W_W+1]}},
                        response_pair[lane][quad*2]}
                        + {{1{response_pair[lane][quad*2+1][W_W+1]}},
                           response_pair[lane][quad*2+1]};
            response_total[lane]
                = {{1{response_quad[lane][0][W_W+2]}},
                    response_quad[lane][0]}
                    + {{1{response_quad[lane][1][W_W+2]}},
                       response_quad[lane][1]};
            response_sum[lane]
                = {{(ACC_W-(W_W+4)){response_total[lane][W_W+3]}},
                    response_total[lane]};
            response_acc_wide[lane]
                = {{1{context_acc_q[response_context][lane][ACC_W-1]}},
                    context_acc_q[response_context][lane]}
                    + {{1{response_sum[lane][ACC_W-1]}}, response_sum[lane]};
            if (response_acc_wide[lane][ACC_W:ACC_W-1] != 2'b00
                    && response_acc_wide[lane][ACC_W:ACC_W-1] != 2'b11)
                response_acc_overflow = 1'b1;
        end
    end

    assign output_valid = !faulted_q && output_found;
    assign output_accept = output_valid && output_ready;
    assign output_tag = context_tag_q[output_context];
    assign output_source_count = context_source_count_q[output_context];
    generate
        for (genvar lane = 0; lane < OUT_LANES; lane = lane + 1) begin : g_output
            assign output_acc[lane*ACC_W +: ACC_W]
                = context_acc_q[output_context][lane];
        end
    endgenerate

    always_comb begin : count_contexts
        context_occupancy = '0;
        for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1)
            context_occupancy = context_occupancy
                + (context_allocated_q[ctx] ? 1'b1 : 1'b0);
    end
    assign response_metadata_occupancy = meta_count_q;
    assign busy = context_occupancy != 0 || meta_count_q != 0;
    assign protocol_error = faulted_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            faulted_q <= 1'b0;
            output_lock_valid_q <= 1'b0;
            output_lock_context_q <= '0;
            issue_lock_valid_q <= 1'b0;
            issue_lock_context_q <= '0;
            issue_rr_q <= '0;
            meta_head_q <= '0;
            meta_tail_q <= '0;
            meta_count_q <= '0;
            for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
                context_allocated_q[ctx] <= 1'b0;
                context_issued_all_q[ctx] <= 1'b0;
                context_done_q[ctx] <= 1'b0;
                context_tag_q[ctx] <= '0;
                context_add_q[ctx] <= '0;
                context_subtract_q[ctx] <= '0;
                context_source_count_q[ctx] <= '0;
                for (int lane = 0; lane < OUT_LANES; lane = lane + 1)
                    context_acc_q[ctx][lane] <= '0;
            end
        end else begin
            if (!output_lock_valid_q && output_valid && !output_ready) begin
                output_lock_valid_q <= 1'b1;
                output_lock_context_q <= output_context;
            end else if (output_lock_valid_q && output_accept) begin
                output_lock_valid_q <= 1'b0;
            end

            if (!issue_lock_valid_q && weight_request_valid
                    && !weight_request_ready) begin
                issue_lock_valid_q <= 1'b1;
                issue_lock_context_q <= issue_context;
            end else if (issue_lock_valid_q && request_accept) begin
                issue_lock_valid_q <= 1'b0;
            end

            if (command_accept) begin
                if (|(command_add_bits & command_subtract_bits)) begin
                    faulted_q <= 1'b1;
                end else begin
                    context_allocated_q[free_context] <= 1'b1;
                    context_issued_all_q[free_context]
                        <= (command_add_bits | command_subtract_bits) == '0;
                    context_done_q[free_context]
                        <= (command_add_bits | command_subtract_bits) == '0;
                    context_tag_q[free_context] <= command_tag;
                    context_add_q[free_context] <= command_add_bits;
                    context_subtract_q[free_context] <= command_subtract_bits;
                    context_source_count_q[free_context] <= '0;
                    for (int lane = 0; lane < OUT_LANES; lane = lane + 1)
                        context_acc_q[free_context][lane]
                            <= command_seed_acc[lane*ACC_W +: ACC_W];
                end
            end

            if (request_accept) begin
                context_add_q[issue_context]
                    <= context_add_q[issue_context]
                        & ~selected_mask_by_context[issue_context];
                context_subtract_q[issue_context]
                    <= context_subtract_q[issue_context]
                        & ~selected_mask_by_context[issue_context];
                context_source_count_q[issue_context]
                    <= context_source_count_q[issue_context]
                        + selected_count_by_context[issue_context];
                if (remaining_after_by_context[issue_context] == '0)
                    context_issued_all_q[issue_context] <= 1'b1;
                issue_rr_q <= issue_context + 1'b1;
                meta_context_q[meta_tail_q] <= issue_context;
                meta_bank_valid_q[meta_tail_q]
                    <= selected_bank_valid_by_context[issue_context];
                meta_bank_subtract_q[meta_tail_q]
                    <= selected_bank_subtract_by_context[issue_context];
                meta_last_q[meta_tail_q]
                    <= remaining_after_by_context[issue_context] == '0;
                meta_tail_q <= meta_tail_q + 1'b1;
            end

            if (response_accept) begin
                if (!response_contract_valid || response_acc_overflow) begin
                    faulted_q <= 1'b1;
                end else begin
                    for (int lane = 0; lane < OUT_LANES; lane = lane + 1)
                        context_acc_q[response_context][lane]
                            <= context_acc_q[response_context][lane]
                                + response_sum[lane];
                    if (response_expected_last)
                        context_done_q[response_context] <= 1'b1;
                end
                meta_head_q <= meta_head_q + 1'b1;
            end

            case ({request_accept, response_accept})
                2'b10: meta_count_q <= meta_count_q + 1'b1;
                2'b01: meta_count_q <= meta_count_q - 1'b1;
                default: meta_count_q <= meta_count_q;
            endcase

            if (weight_response_valid && meta_count_q == 0)
                faulted_q <= 1'b1;

            if (output_accept) begin
                context_allocated_q[output_context] <= 1'b0;
                context_issued_all_q[output_context] <= 1'b0;
                context_done_q[output_context] <= 1'b0;
                context_add_q[output_context] <= '0;
                context_subtract_q[output_context] <= '0;
            end
        end
    end
endmodule

`default_nettype wire
