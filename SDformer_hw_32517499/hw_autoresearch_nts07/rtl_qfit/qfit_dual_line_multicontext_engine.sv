`timescale 1ns/1ps
`default_nettype none

// Bank-decoupled Local/Motion engine for contexts sharing one exact weight
// object.  Each weight bank independently chooses a resident context, which
// fills otherwise idle banks without merging operator/group/source-chunk/
// output-lane identities.  Motion supplies the same selected-source bitmap as
// Local, plus a subset marking columns that must be subtracted.
//
// Weight storage is external: every asserted bank consumes one synchronous
// OUT_LANES x W_W word.  command_object_tag must encode the full physical
// weight identity/base selected by the system scheduler.
module qfit_dual_line_multicontext_engine #(
    parameter int TILE_BITS = 256,
    parameter int ISSUE_WIDTH = 16,
    parameter int CONTEXTS = 4,
    parameter int OUT_LANES = 96,
    parameter int TAG_W = 32,
    parameter int OBJECT_W = 64,
    parameter int W_W = 8,
    parameter int ACC_W = 32,
    parameter int INDEX_W = (TILE_BITS <= 1) ? 1 : $clog2(TILE_BITS),
    parameter int BANK_BITS = (ISSUE_WIDTH <= 1) ? 0 : $clog2(ISSUE_WIDTH),
    parameter int BANK_ADDR_W = INDEX_W - BANK_BITS,
    parameter int CTX_W = (CONTEXTS <= 1) ? 1 : $clog2(CONTEXTS),
    parameter int COUNT_W = $clog2(TILE_BITS + 1)
) (
    input  logic                                  clk_core,
    input  logic                                  rst_core,

    input  logic                                  command_valid,
    output logic                                  command_ready,
    input  logic [TAG_W-1:0]                      command_tag,
    input  logic [OBJECT_W-1:0]                   command_object_tag,
    input  logic                                  command_batch_last,
    input  logic                                  command_use_motion,
    input  logic [TILE_BITS-1:0]                  command_source_bits,
    input  logic [TILE_BITS-1:0]                  command_negative_bits,
    input  logic [OUT_LANES*ACC_W-1:0]            command_seed_acc,

    output logic                                  weight_request_valid,
    input  logic                                  weight_request_ready,
    output logic [OBJECT_W-1:0]                   weight_request_object_tag,
    output logic [ISSUE_WIDTH-1:0]                weight_request_bank_valid,
    output logic [ISSUE_WIDTH*BANK_ADDR_W-1:0]    weight_request_bank_addr,
    output logic [ISSUE_WIDTH*CTX_W-1:0]          weight_request_bank_context,
    output logic [ISSUE_WIDTH-1:0]                weight_request_bank_negative,

    input  logic                                  weight_response_valid,
    output logic                                  weight_response_ready,
    input  logic [ISSUE_WIDTH-1:0]                weight_response_bank_valid,
    input  logic [ISSUE_WIDTH*OUT_LANES*W_W-1:0] weight_response_data,

    output logic                                  output_valid,
    input  logic                                  output_ready,
    output logic [TAG_W-1:0]                      output_tag,
    output logic [OBJECT_W-1:0]                   output_object_tag,
    output logic                                  output_use_motion,
    output logic [COUNT_W-1:0]                    output_source_count,
    output logic [OUT_LANES*ACC_W-1:0]            output_acc,

    output logic [CONTEXTS-1:0]                   context_active,
    output logic                                  protocol_error
);
    logic object_valid_q;
    logic batch_sealed_q;
    logic [OBJECT_W-1:0] object_tag_q;
    logic [CONTEXTS-1:0] active_q;
    logic [CONTEXTS-1:0] done_q;
    logic [TAG_W-1:0] tag_q [0:CONTEXTS-1];
    logic use_motion_q [0:CONTEXTS-1];
    logic [TILE_BITS-1:0] remaining_q [0:CONTEXTS-1];
    logic [TILE_BITS-1:0] negative_q [0:CONTEXTS-1];
    logic [COUNT_W-1:0] source_count_q [0:CONTEXTS-1];
    logic signed [ACC_W-1:0] acc_q [0:CONTEXTS-1][0:OUT_LANES-1];

    logic pending_q;
    logic [ISSUE_WIDTH-1:0] pending_bank_valid_q;
    logic [ISSUE_WIDTH*CTX_W-1:0] pending_bank_context_q;
    logic [ISSUE_WIDTH-1:0] pending_bank_negative_q;
    logic [CONTEXTS-1:0] pending_context_last_q;
    logic faulted_q;

    logic [CONTEXTS-1:0] free_context;
    logic free_context_valid;
    logic [CTX_W-1:0] command_context;
    logic object_match;
    logic command_fire;
    logic request_fire;
    logic response_fire;
    logic response_contract_valid;
    logic can_issue_request;
    logic protocol_violation;

    logic [TILE_BITS-1:0] selection_mask [0:CONTEXTS-1];
    logic [TILE_BITS-1:0] remaining_after_request [0:CONTEXTS-1];
    logic [CONTEXTS-1:0] request_context_last;
    logic [COUNT_W-1:0] request_context_sources [0:CONTEXTS-1];
    logic selected_valid;

    logic output_context_valid;
    logic [CTX_W-1:0] output_context;
    logic [CONTEXTS-1:0] output_onehot;
    logic output_lock_q;
    logic [CTX_W-1:0] output_context_q;
    logic output_fire;
    logic signed [ACC_W-1:0] response_sum [0:CONTEXTS-1][0:OUT_LANES-1];

    function automatic logic bank_has_source(
        input logic [TILE_BITS-1:0] value,
        input integer bank
    );
        logic found;
        begin
            found = 1'b0;
            for (int source = bank; source < TILE_BITS; source = source + ISSUE_WIDTH)
                found = found | value[source];
            bank_has_source = found;
        end
    endfunction

    function automatic logic [BANK_ADDR_W-1:0] first_bank_address(
        input logic [TILE_BITS-1:0] value,
        input integer bank
    );
        logic found;
        logic [BANK_ADDR_W-1:0] address;
        begin
            found = 1'b0;
            address = '0;
            for (int source = bank; source < TILE_BITS; source = source + ISSUE_WIDTH) begin
                if (!found && value[source]) begin
                    found = 1'b1;
                    address = BANK_ADDR_W'(source >> BANK_BITS);
                end
            end
            first_bank_address = address;
        end
    endfunction

    function automatic logic signed [ACC_W-1:0] extend_weight(
        input logic [W_W-1:0] value
    );
        extend_weight = {{(ACC_W-W_W){value[W_W-1]}}, value};
    endfunction

    initial begin
        if (ACC_W < W_W)
            $error("ACC_W must be at least W_W");
        if (TILE_BITS % ISSUE_WIDTH != 0)
            $error("TILE_BITS must be divisible by ISSUE_WIDTH");
        if (ISSUE_WIDTH < 1 || (ISSUE_WIDTH & (ISSUE_WIDTH - 1)) != 0)
            $error("ISSUE_WIDTH must be a positive power of two");
        if (CONTEXTS < 2)
            $error("multicontext engine requires at least two contexts");
    end

    always_comb begin
        free_context = ~(active_q | done_q);
        free_context_valid = 1'b0;
        command_context = '0;
        for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
            if (!free_context_valid && free_context[ctx]) begin
                free_context_valid = 1'b1;
                command_context = CTX_W'(ctx);
            end
        end
    end

    assign object_match = !object_valid_q || command_object_tag == object_tag_q;
    assign command_ready = free_context_valid && object_match
        && !batch_sealed_q && !faulted_q;
    assign command_fire = command_valid && command_ready;

    always_comb begin
        for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1)
            selection_mask[ctx] = '0;
        weight_request_bank_valid = '0;
        weight_request_bank_addr = '0;
        weight_request_bank_context = '0;
        weight_request_bank_negative = '0;

        for (int bank = 0; bank < ISSUE_WIDTH; bank = bank + 1) begin : select_bank
            logic context_found;
            context_found = 1'b0;
            for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
                if (!context_found && active_q[ctx]
                        && bank_has_source(remaining_q[ctx], bank)) begin : select_context
                    logic [BANK_ADDR_W-1:0] address;
                    integer source;
                    context_found = 1'b1;
                    address = first_bank_address(remaining_q[ctx], bank);
                    source = $unsigned(address) * ISSUE_WIDTH + bank;
                    weight_request_bank_valid[bank] = 1'b1;
                    weight_request_bank_addr[bank*BANK_ADDR_W +: BANK_ADDR_W] = address;
                    weight_request_bank_context[bank*CTX_W +: CTX_W] = CTX_W'(ctx);
                    weight_request_bank_negative[bank] = negative_q[ctx][source];
                    selection_mask[ctx][source] = 1'b1;
                end
            end
        end

        selected_valid = |weight_request_bank_valid;
        request_context_last = '0;
        for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
            remaining_after_request[ctx] = remaining_q[ctx] & ~selection_mask[ctx];
            request_context_sources[ctx] = '0;
            for (int bank = 0; bank < ISSUE_WIDTH; bank = bank + 1) begin
                if (weight_request_bank_valid[bank]
                        && weight_request_bank_context[bank*CTX_W +: CTX_W] == CTX_W'(ctx))
                    request_context_sources[ctx]
                        = request_context_sources[ctx] + COUNT_W'(1);
            end
            request_context_last[ctx] = active_q[ctx]
                && selection_mask[ctx] != '0
                && remaining_after_request[ctx] == '0;
        end
    end

    assign response_contract_valid = weight_response_bank_valid == pending_bank_valid_q;
    assign weight_response_ready = pending_q;
    assign response_fire = weight_response_valid && weight_response_ready;
    assign can_issue_request = !pending_q
        || (weight_response_valid && response_contract_valid);
    assign weight_request_valid = object_valid_q && batch_sealed_q && selected_valid
        && can_issue_request && !faulted_q;
    assign weight_request_object_tag = object_tag_q;
    assign request_fire = weight_request_valid && weight_request_ready;

    always_comb begin
        for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
            for (int lane = 0; lane < OUT_LANES; lane = lane + 1) begin
                response_sum[ctx][lane] = '0;
                for (int bank = 0; bank < ISSUE_WIDTH; bank = bank + 1) begin
                    if (pending_bank_valid_q[bank]
                            && pending_bank_context_q[bank*CTX_W +: CTX_W] == CTX_W'(ctx)) begin
                        if (pending_bank_negative_q[bank])
                            response_sum[ctx][lane] = response_sum[ctx][lane]
                                - extend_weight(weight_response_data[
                                    (bank*OUT_LANES + lane)*W_W +: W_W
                                ]);
                        else
                            response_sum[ctx][lane] = response_sum[ctx][lane]
                                + extend_weight(weight_response_data[
                                    (bank*OUT_LANES + lane)*W_W +: W_W
                                ]);
                    end
                end
            end
        end
    end

    always_comb begin
        output_context_valid = 1'b0;
        output_context = '0;
        output_onehot = '0;
        if (output_lock_q) begin
            // Once valid has met downstream backpressure, retain the exact
            // context until its handshake.  A newly completed lower-index
            // context must not pre-empt or mutate the visible payload.
            output_context_valid = done_q[output_context_q];
            output_context = output_context_q;
            output_onehot[output_context_q] = done_q[output_context_q];
        end else begin
            for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
                if (!output_context_valid && done_q[ctx]) begin
                    output_context_valid = 1'b1;
                    output_context = CTX_W'(ctx);
                    output_onehot[ctx] = 1'b1;
                end
            end
        end
        output_tag = tag_q[output_context];
        output_object_tag = object_tag_q;
        output_use_motion = use_motion_q[output_context];
        output_source_count = source_count_q[output_context];
        output_acc = '0;
        for (int lane = 0; lane < OUT_LANES; lane = lane + 1)
            output_acc[lane*ACC_W +: ACC_W] = acc_q[output_context][lane];
    end

    assign output_valid = batch_sealed_q && output_context_valid && !faulted_q;
    assign output_fire = output_valid && output_ready;
    assign context_active = active_q | done_q;
    assign protocol_error = faulted_q;
    assign protocol_violation =
        (command_fire && ((command_negative_bits & ~command_source_bits) != '0))
        || (command_fire && !command_batch_last && $countones(free_context) == 1)
        || (weight_response_valid && !pending_q)
        || (weight_response_valid && pending_q && !response_contract_valid);

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            object_valid_q <= 1'b0;
            batch_sealed_q <= 1'b0;
            object_tag_q <= '0;
            active_q <= '0;
            done_q <= '0;
            pending_q <= 1'b0;
            pending_bank_valid_q <= '0;
            pending_bank_context_q <= '0;
            pending_bank_negative_q <= '0;
            pending_context_last_q <= '0;
            output_lock_q <= 1'b0;
            output_context_q <= '0;
            faulted_q <= 1'b0;
            for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
                tag_q[ctx] <= '0;
                use_motion_q[ctx] <= 1'b0;
                remaining_q[ctx] <= '0;
                negative_q[ctx] <= '0;
                source_count_q[ctx] <= '0;
                for (int lane = 0; lane < OUT_LANES; lane = lane + 1)
                    acc_q[ctx][lane] <= '0;
            end
        end else if (protocol_violation) begin
            object_valid_q <= 1'b0;
            batch_sealed_q <= 1'b0;
            active_q <= '0;
            done_q <= '0;
            pending_q <= 1'b0;
            pending_bank_valid_q <= '0;
            pending_context_last_q <= '0;
            output_lock_q <= 1'b0;
            faulted_q <= 1'b1;
        end else begin
            if (output_valid && !output_ready) begin
                output_lock_q <= 1'b1;
                output_context_q <= output_context;
            end else if (output_fire) begin
                output_lock_q <= 1'b0;
            end

            if (command_fire) begin
                object_valid_q <= 1'b1;
                if (command_batch_last)
                    batch_sealed_q <= 1'b1;
                if (!object_valid_q)
                    object_tag_q <= command_object_tag;
                tag_q[command_context] <= command_tag;
                use_motion_q[command_context] <= command_use_motion;
                remaining_q[command_context] <= command_source_bits;
                negative_q[command_context] <= command_negative_bits;
                source_count_q[command_context] <= '0;
                active_q[command_context] <= command_source_bits != '0;
                done_q[command_context] <= command_source_bits == '0;
                for (int lane = 0; lane < OUT_LANES; lane = lane + 1)
                    acc_q[command_context][lane]
                        <= command_seed_acc[lane*ACC_W +: ACC_W];
            end

            if (request_fire) begin
                pending_q <= 1'b1;
                pending_bank_valid_q <= weight_request_bank_valid;
                pending_bank_context_q <= weight_request_bank_context;
                pending_bank_negative_q <= weight_request_bank_negative;
                pending_context_last_q <= request_context_last;
                for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
                    // A newly accepted command is not visible to this cycle's
                    // selector.  Preserve its capture when command and issue
                    // handshakes happen together on different contexts.
                    if (!command_fire || command_context != CTX_W'(ctx)) begin
                        remaining_q[ctx] <= remaining_after_request[ctx];
                        source_count_q[ctx] <= source_count_q[ctx]
                            + request_context_sources[ctx];
                    end
                end
            end

            if (response_fire) begin
                for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
                    if (!command_fire || command_context != CTX_W'(ctx)) begin
                        for (int lane = 0; lane < OUT_LANES; lane = lane + 1)
                            acc_q[ctx][lane] <= acc_q[ctx][lane]
                                + response_sum[ctx][lane];
                    end
                    if (pending_context_last_q[ctx]) begin
                        active_q[ctx] <= 1'b0;
                        done_q[ctx] <= 1'b1;
                    end
                end
                if (!request_fire) begin
                    pending_q <= 1'b0;
                    pending_bank_valid_q <= '0;
                    pending_bank_context_q <= '0;
                    pending_bank_negative_q <= '0;
                    pending_context_last_q <= '0;
                end
            end

            if (output_fire) begin
                done_q[output_context] <= 1'b0;
                if (active_q == '0 && (done_q & ~output_onehot) == '0 && !command_fire)
                    object_valid_q <= 1'b0;
                if (active_q == '0 && (done_q & ~output_onehot) == '0 && !command_fire)
                    batch_sealed_q <= 1'b0;
            end
        end
    end
endmodule

`default_nettype wire
