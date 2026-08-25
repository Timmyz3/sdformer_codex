`timescale 1ns/1ps
`default_nettype none

// M227: one 96-output FC1 slice with eight raw activation contexts.
//
// A 256-bit scanner owns 384-bit presence/sign masks for each context.  The
// union bitmap walks only unique source channels.  One 768-bit INT8 weight
// vector is requested per unique source and held while up to FANOUT contexts
// update their private Acc19 state each cycle.  F1/F2/F4 therefore keep the
// same masks, eight contexts, weight port and result geometry; only the number
// of replicated accumulator write ports changes.
module m227_fc1_k8_masked_held_weight_slice #(
    parameter int FANOUT = 1,
    parameter int TAG_BITS = 24,
    parameter int EPOCH_BITS = 16,
    parameter int CHANNELS = 384,
    parameter int CONTEXTS = 8,
    parameter int LANES = 96,
    parameter int ACC_BITS = 19
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         begin_valid,
    output logic                         begin_ready,
    input  logic [TAG_BITS-1:0]          begin_tag,
    input  logic [EPOCH_BITS-1:0]        begin_epoch,
    output logic                         begin_accept,

    input  logic                         scan_valid,
    output logic                         scan_ready,
    input  logic [2:0]                   scan_context,
    input  logic                         scan_beat,
    input  logic [255:0]                 scan_presence,
    input  logic [255:0]                 scan_sign,
    output logic                         scan_accept,

    input  logic                         scan_done_valid,
    output logic                         scan_done_ready,
    output logic                         scan_done_accept,

    output logic                         weight_req_valid,
    input  logic                         weight_req_ready,
    output logic [TAG_BITS-1:0]          weight_req_tag,
    output logic [EPOCH_BITS-1:0]        weight_req_epoch,
    output logic [8:0]                   weight_req_source,
    output logic                         weight_req_accept,

    input  logic                         weight_rsp_valid,
    output logic                         weight_rsp_ready,
    input  logic [TAG_BITS-1:0]          weight_rsp_tag,
    input  logic [EPOCH_BITS-1:0]        weight_rsp_epoch,
    input  logic [8:0]                   weight_rsp_source,
    input  logic [LANES*8-1:0]           weight_rsp_data,
    output logic                         weight_rsp_accept,

    output logic                         result_valid,
    input  logic                         result_ready,
    output logic [TAG_BITS-1:0]          result_tag,
    output logic [EPOCH_BITS-1:0]        result_epoch,
    output logic [2:0]                   result_context,
    output logic [LANES*ACC_BITS-1:0]    result_accumulator,
    output logic                         result_last,
    output logic                         result_accept,

    output logic                         done_valid,
    input  logic                         done_ready,
    output logic [TAG_BITS-1:0]          done_tag,
    output logic [EPOCH_BITS-1:0]        done_epoch,
    output logic                         done_accept,

    output logic                         protocol_error,
    output logic                         numeric_overflow,
    output logic                         busy,
    output logic [4:0]                   debug_scan_count,
    output logic [9:0]                   debug_unique_sources,
    output logic [11:0]                  debug_context_updates,
    output logic [9:0]                   debug_weight_reads
);
    localparam bit PARAMETERS_LEGAL =
        (FANOUT == 1 || FANOUT == 2 || FANOUT == 4)
        && TAG_BITS == 24 && EPOCH_BITS == 16 && CHANNELS == 384
        && CONTEXTS == 8 && LANES == 96 && ACC_BITS == 19;

    typedef enum logic [2:0] {
        ST_IDLE, ST_SCAN, ST_REQUEST, ST_WAIT, ST_REPLAY, ST_DRAIN, ST_DONE
    } state_t;

    state_t state_q;
    logic fault_q, overflow_q;
    logic [TAG_BITS-1:0] tag_q;
    logic [EPOCH_BITS-1:0] epoch_q;
    logic [15:0] scan_seen_q;
    logic [4:0] scan_count_q;
    logic [CHANNELS-1:0] presence_q [0:CONTEXTS-1];
    logic [CHANNELS-1:0] sign_q [0:CONTEXTS-1];
    logic [CHANNELS-1:0] remaining_sources_q;
    logic [8:0] current_source_q;
    logic [7:0] replay_pending_q;
    logic [LANES*8-1:0] held_weight_q;
    logic signed [ACC_BITS-1:0] accumulator_q
        [0:CONTEXTS-1][0:LANES-1];
    logic [2:0] drain_context_q;
    logic [9:0] unique_sources_q, weight_reads_q;
    logic [11:0] context_updates_q;

    logic scan_shape_legal, illegal_begin, illegal_scan;
    logic illegal_scan_done, illegal_response;
    logic [7:0] selected_context_valid;
    logic [2:0] selected_context [0:3];
    logic [7:0] replay_pending_after;
    logic [CHANNELS-1:0] remaining_without_current;
    logic next_source_found;
    logic [8:0] next_source;
    logic overflow_this_cycle;

    function automatic logic [8:0] first_source(
        input logic [CHANNELS-1:0] bitmap
    );
        logic found;
        logic [8:0] index_value;
        begin
            found = 1'b0;
            index_value = '0;
            for (int index = 0; index < CHANNELS; index++) begin
                if (!found && bitmap[index]) begin
                    found = 1'b1;
                    index_value = index[8:0];
                end
            end
            return index_value;
        end
    endfunction

    generate
        if (!PARAMETERS_LEGAL) begin : g_illegal_parameters
            initial $fatal(1, "M227 frozen geometry/FANOUT drift");
        end
    endgenerate

    always_comb begin : replay_select
        logic [7:0] work;
        work = replay_pending_q;
        selected_context_valid = '0;
        for (int slot = 0; slot < 4; slot++)
            selected_context[slot] = '0;
        for (int slot = 0; slot < FANOUT; slot++) begin
            logic found;
            found = 1'b0;
            for (int ctx = 0; ctx < CONTEXTS; ctx++) begin
                if (!found && work[ctx]) begin
                    found = 1'b1;
                    selected_context_valid[slot] = 1'b1;
                    selected_context[slot] = ctx[2:0];
                    work[ctx] = 1'b0;
                end
            end
        end
        replay_pending_after = work;
    end

    always_comb begin : next_source_select
        remaining_without_current = remaining_sources_q;
        if (current_source_q < CHANNELS)
            remaining_without_current[current_source_q] = 1'b0;
        next_source_found = |remaining_without_current;
        next_source = first_source(remaining_without_current);
    end

    always_comb begin : interface_control
        scan_shape_legal = state_q == ST_SCAN && scan_context < CONTEXTS
            && !scan_seen_q[{scan_context,scan_beat}]
            && !(scan_beat && (|scan_presence[255:128]
                               || |scan_sign[255:128]))
            && !(|(scan_sign & ~scan_presence));
        illegal_begin = begin_valid && state_q != ST_IDLE;
        illegal_scan = scan_valid && !scan_shape_legal;
        illegal_scan_done = scan_done_valid
            && !(state_q == ST_SCAN && scan_count_q == 16);
        illegal_response = weight_rsp_valid
            && !(state_q == ST_WAIT
                && weight_rsp_tag == tag_q
                && weight_rsp_epoch == epoch_q
                && weight_rsp_source == current_source_q);

        begin_ready = state_q == ST_IDLE && !fault_q && !overflow_q;
        begin_accept = begin_valid && begin_ready;
        scan_ready = scan_shape_legal && !fault_q && !overflow_q;
        scan_accept = scan_valid && scan_ready;
        scan_done_ready = state_q == ST_SCAN && scan_count_q == 16
            && !fault_q && !overflow_q;
        scan_done_accept = scan_done_valid && scan_done_ready;

        weight_req_valid = state_q == ST_REQUEST
            && !fault_q && !overflow_q;
        weight_req_tag = weight_req_valid ? tag_q : '0;
        weight_req_epoch = weight_req_valid ? epoch_q : '0;
        weight_req_source = weight_req_valid ? current_source_q : '0;
        weight_req_accept = weight_req_valid && weight_req_ready;
        weight_rsp_ready = state_q == ST_WAIT && !fault_q && !overflow_q
            && !illegal_response;
        weight_rsp_accept = weight_rsp_valid && weight_rsp_ready;

        result_valid = state_q == ST_DRAIN && !fault_q && !overflow_q;
        result_tag = result_valid ? tag_q : '0;
        result_epoch = result_valid ? epoch_q : '0;
        result_context = result_valid ? drain_context_q : '0;
        result_accumulator = '0;
        if (result_valid) begin
            for (int lane = 0; lane < LANES; lane++)
                result_accumulator[lane*ACC_BITS +: ACC_BITS]
                    = accumulator_q[drain_context_q][lane];
        end
        result_last = result_valid && drain_context_q == CONTEXTS-1;
        result_accept = result_valid && result_ready;

        done_valid = state_q == ST_DONE && !fault_q && !overflow_q;
        done_tag = done_valid ? tag_q : '0;
        done_epoch = done_valid ? epoch_q : '0;
        done_accept = done_valid && done_ready;
        protocol_error = fault_q || illegal_begin || illegal_scan
            || illegal_scan_done || illegal_response;
        numeric_overflow = overflow_q || overflow_this_cycle;
        busy = state_q != ST_IDLE;
        debug_scan_count = scan_count_q;
        debug_unique_sources = unique_sources_q;
        debug_context_updates = context_updates_q;
        debug_weight_reads = weight_reads_q;
    end

    always_comb begin : overflow_audit
        overflow_this_cycle = 1'b0;
        if (state_q == ST_REPLAY) begin
            for (int slot = 0; slot < FANOUT; slot++) begin
                if (selected_context_valid[slot]) begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        logic signed [ACC_BITS:0] extended_sum;
                        logic signed [ACC_BITS-1:0] signed_weight;
                        signed_weight = {{(ACC_BITS-8){
                            held_weight_q[lane*8+7]}},
                            held_weight_q[lane*8 +: 8]};
                        if (sign_q[selected_context[slot]][current_source_q])
                            extended_sum = $signed(
                                accumulator_q[selected_context[slot]][lane])
                                - $signed(signed_weight);
                        else
                            extended_sum = $signed(
                                accumulator_q[selected_context[slot]][lane])
                                + $signed(signed_weight);
                        if (extended_sum[ACC_BITS]
                                != extended_sum[ACC_BITS-1])
                            overflow_this_cycle = 1'b1;
                    end
                end
            end
        end
    end

    always_ff @(posedge clk_core) begin : state_update
        if (rst_core) begin
            state_q <= ST_IDLE;
            fault_q <= 1'b0;
            overflow_q <= 1'b0;
            tag_q <= '0;
            epoch_q <= '0;
            scan_seen_q <= '0;
            scan_count_q <= '0;
            remaining_sources_q <= '0;
            current_source_q <= '0;
            replay_pending_q <= '0;
            held_weight_q <= '0;
            drain_context_q <= '0;
            unique_sources_q <= '0;
            context_updates_q <= '0;
            weight_reads_q <= '0;
            for (int ctx = 0; ctx < CONTEXTS; ctx++) begin
                presence_q[ctx] <= '0;
                sign_q[ctx] <= '0;
                for (int lane = 0; lane < LANES; lane++)
                    accumulator_q[ctx][lane] <= '0;
            end
        end else begin
            if (illegal_begin || illegal_scan || illegal_scan_done
                    || illegal_response)
                fault_q <= 1'b1;
            if (overflow_this_cycle)
                overflow_q <= 1'b1;

            if (!protocol_error && !numeric_overflow) begin
                case (state_q)
                    ST_IDLE: if (begin_accept) begin
                        state_q <= ST_SCAN;
                        tag_q <= begin_tag;
                        epoch_q <= begin_epoch;
                        scan_seen_q <= '0;
                        scan_count_q <= '0;
                        remaining_sources_q <= '0;
                        current_source_q <= '0;
                        replay_pending_q <= '0;
                        drain_context_q <= '0;
                        unique_sources_q <= '0;
                        context_updates_q <= '0;
                        weight_reads_q <= '0;
                        for (int ctx = 0; ctx < CONTEXTS;
                                ctx++) begin
                            presence_q[ctx] <= '0;
                            sign_q[ctx] <= '0;
                            for (int lane = 0; lane < LANES; lane++)
                                accumulator_q[ctx][lane] <= '0;
                        end
                    end

                    ST_SCAN: begin
                        if (scan_accept) begin
                            scan_seen_q[{scan_context,scan_beat}] <= 1'b1;
                            scan_count_q <= scan_count_q + 1'b1;
                            if (!scan_beat) begin
                                presence_q[scan_context][255:0]
                                    <= scan_presence;
                                sign_q[scan_context][255:0] <= scan_sign;
                            end else begin
                                presence_q[scan_context][383:256]
                                    <= scan_presence[127:0];
                                sign_q[scan_context][383:256]
                                    <= scan_sign[127:0];
                            end
                        end
                        if (scan_done_accept) begin
                            logic [CHANNELS-1:0] union_now;
                            union_now = '0;
                            for (int ctx = 0; ctx < CONTEXTS;
                                    ctx++)
                                union_now |= presence_q[ctx];
                            remaining_sources_q <= union_now;
                            if (|union_now) begin
                                current_source_q <= first_source(union_now);
                                state_q <= ST_REQUEST;
                            end else begin
                                drain_context_q <= '0;
                                state_q <= ST_DRAIN;
                            end
                        end
                    end

                    ST_REQUEST: if (weight_req_accept)
                        state_q <= ST_WAIT;

                    ST_WAIT: if (weight_rsp_accept) begin
                        logic [7:0] context_mask;
                        context_mask = '0;
                        for (int ctx = 0; ctx < CONTEXTS;
                                ctx++)
                            context_mask[ctx]
                                = presence_q[ctx][current_source_q];
                        held_weight_q <= weight_rsp_data;
                        replay_pending_q <= context_mask;
                        unique_sources_q <= unique_sources_q + 1'b1;
                        weight_reads_q <= weight_reads_q + 1'b1;
                        state_q <= ST_REPLAY;
                    end

                    ST_REPLAY: begin
                        for (int slot = 0; slot < FANOUT; slot++) begin
                            if (selected_context_valid[slot]) begin
                                for (int lane = 0; lane < LANES; lane++) begin
                                    logic signed [ACC_BITS-1:0]
                                        signed_weight;
                                    signed_weight = {{(ACC_BITS-8){
                                        held_weight_q[lane*8+7]}},
                                        held_weight_q[lane*8 +: 8]};
                                    if (sign_q[selected_context[slot]]
                                              [current_source_q])
                                        accumulator_q[selected_context[slot]]
                                            [lane] <= $signed(accumulator_q
                                            [selected_context[slot]][lane])
                                            - $signed(signed_weight);
                                    else
                                        accumulator_q[selected_context[slot]]
                                            [lane] <= $signed(accumulator_q
                                            [selected_context[slot]][lane])
                                            + $signed(signed_weight);
                                end
                                context_updates_q <= context_updates_q +
                                    (selected_context_valid[0]
                                     + selected_context_valid[1]
                                     + selected_context_valid[2]
                                     + selected_context_valid[3]);
                            end
                        end
                        replay_pending_q <= replay_pending_after;
                        if (replay_pending_after == 0) begin
                            remaining_sources_q
                                <= remaining_without_current;
                            if (next_source_found) begin
                                current_source_q <= next_source;
                                state_q <= ST_REQUEST;
                            end else begin
                                drain_context_q <= '0;
                                state_q <= ST_DRAIN;
                            end
                        end
                    end

                    ST_DRAIN: if (result_accept) begin
                        if (drain_context_q == CONTEXTS-1)
                            state_q <= ST_DONE;
                        else
                            drain_context_q <= drain_context_q + 1'b1;
                    end

                    ST_DONE: if (done_accept)
                        state_q <= ST_IDLE;
                    default: state_q <= ST_IDLE;
                endcase
            end
        end
    end
endmodule

`default_nettype wire
