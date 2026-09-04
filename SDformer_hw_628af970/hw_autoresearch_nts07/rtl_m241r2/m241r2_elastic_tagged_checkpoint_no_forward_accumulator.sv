`timescale 1ns/1ps
`default_nettype none

// M241r2: latency-elastic, identity-tagged four-bank INT8-to-Acc19 island.
//
// Relative to immutable M241 r1, both macro read paths use explicit
// request/response valid-ready channels.  Responses echo the complete
// sequence/operator/partition/window/checkpoint-epoch/payload/order identity;
// a stale or aliased response is never accepted.  Context open additionally
// requires an exact loader-binding identity.
//
// The accumulator retains lazy-valid state and a runtime overflow guard.  An
// overflow token produces an explicit abort, never a successful commit, and
// accepted younger s0/s1 tokens are quarantined.  Same-address serialization
// remains an interlock; there is no accumulator forwarding-data payload.
module m241r2_elastic_tagged_checkpoint_no_forward_accumulator #(
    parameter int LANES = 8,
    parameter int ROWS = 384,
    parameter int ACC_BITS = 19,
    parameter int SEQUENCE_BITS = 32,
    parameter int PARTITION_BITS = 9,
    parameter int WINDOW_BITS = 16,
    parameter int EPOCH_BITS = 16,
    parameter int PAYLOAD_BITS = 32,
    parameter int ORDER_BITS = 16,
    parameter int ROW_BITS = $clog2(ROWS),
    parameter int ACC_ADDR_BITS = $clog2(2 * ROWS)
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         loader_binding_valid,
    input  logic [1:0]                   loader_binding_operator,
    input  logic [PARTITION_BITS-1:0]    loader_binding_partition,
    input  logic [EPOCH_BITS-1:0]        loader_binding_weight_epoch,
    input  logic [PAYLOAD_BITS-1:0]      loader_binding_payload_id,

    input  logic                         context_open_valid,
    output logic                         context_open_ready,
    input  logic [SEQUENCE_BITS-1:0]     context_open_sequence,
    input  logic [1:0]                   context_open_operator,
    input  logic [PARTITION_BITS-1:0]    context_open_partition,
    input  logic [WINDOW_BITS-1:0]       context_open_window,
    input  logic [EPOCH_BITS-1:0]        context_open_weight_epoch,
    input  logic [PAYLOAD_BITS-1:0]      context_open_payload_id,
    output logic                         context_open_accept,

    input  logic                         descriptor_valid,
    output logic                         descriptor_ready,
    input  logic [SEQUENCE_BITS-1:0]     descriptor_sequence,
    input  logic [1:0]                   descriptor_operator,
    input  logic [PARTITION_BITS-1:0]    descriptor_partition,
    input  logic [WINDOW_BITS-1:0]       descriptor_window,
    input  logic [EPOCH_BITS-1:0]        descriptor_weight_epoch,
    input  logic [PAYLOAD_BITS-1:0]      descriptor_payload_id,
    input  logic [ORDER_BITS-1:0]        descriptor_order,
    input  logic [ROW_BITS-1:0]          descriptor_row,
    input  logic [3:0]                   descriptor_source,
    input  logic [3:0]                   descriptor_destination_valid,
    input  logic [2:0]                   descriptor_destination [0:3],
    input  logic [3:0]                   descriptor_negate,
    input  logic                         descriptor_last,
    output logic                         descriptor_accept,

    input  logic                         context_close_valid,
    output logic                         context_close_ready,
    output logic                         context_close_accept,
    output logic                         window_done,

    output logic                         weight_req_valid,
    input  logic                         weight_req_ready,
    output logic                         weight_req_accept,
    output logic [3:0]                   weight_req_bank_valid,
    output logic [4:0]                   weight_req_addr [0:3],
    output logic [SEQUENCE_BITS-1:0]     weight_req_sequence,
    output logic [1:0]                   weight_req_operator,
    output logic [PARTITION_BITS-1:0]    weight_req_partition,
    output logic [WINDOW_BITS-1:0]       weight_req_window,
    output logic [EPOCH_BITS-1:0]        weight_req_weight_epoch,
    output logic [PAYLOAD_BITS-1:0]      weight_req_payload_id,
    output logic [ORDER_BITS-1:0]        weight_req_order,
    output logic [3:0]                   weight_req_source,
    output logic                         weight_req_half,

    input  logic                         weight_rsp_valid,
    output logic                         weight_rsp_ready,
    output logic                         weight_rsp_accept,
    input  logic [3:0]                   weight_rsp_bank_valid,
    input  logic [SEQUENCE_BITS-1:0]     weight_rsp_sequence,
    input  logic [1:0]                   weight_rsp_operator,
    input  logic [PARTITION_BITS-1:0]    weight_rsp_partition,
    input  logic [WINDOW_BITS-1:0]       weight_rsp_window,
    input  logic [EPOCH_BITS-1:0]        weight_rsp_weight_epoch,
    input  logic [PAYLOAD_BITS-1:0]      weight_rsp_payload_id,
    input  logic [ORDER_BITS-1:0]        weight_rsp_order,
    input  logic [3:0]                   weight_rsp_source,
    input  logic                         weight_rsp_half,
    input  logic signed [7:0]            weight_rsp_data [0:3][0:LANES-1],

    output logic [3:0]                   weight_cache_hit,
    output logic [3:0]                   weight_cache_miss,

    output logic                         acc_req_valid,
    input  logic                         acc_req_ready,
    output logic                         acc_req_accept,
    output logic [3:0]                   acc_req_bank_valid,
    output logic [ACC_ADDR_BITS-1:0]     acc_req_addr [0:3],
    output logic [SEQUENCE_BITS-1:0]     acc_req_sequence,
    output logic [WINDOW_BITS-1:0]       acc_req_window,
    output logic [EPOCH_BITS-1:0]        acc_req_weight_epoch,
    output logic [PAYLOAD_BITS-1:0]      acc_req_payload_id,
    output logic [ORDER_BITS-1:0]        acc_req_order,

    input  logic                         acc_rsp_valid,
    output logic                         acc_rsp_ready,
    output logic                         acc_rsp_accept,
    input  logic [3:0]                   acc_rsp_bank_valid,
    input  logic [SEQUENCE_BITS-1:0]     acc_rsp_sequence,
    input  logic [WINDOW_BITS-1:0]       acc_rsp_window,
    input  logic [EPOCH_BITS-1:0]        acc_rsp_weight_epoch,
    input  logic [PAYLOAD_BITS-1:0]      acc_rsp_payload_id,
    input  logic [ORDER_BITS-1:0]        acc_rsp_order,
    input  logic signed [ACC_BITS-1:0]   acc_rsp_data [0:3][0:LANES-1],

    output logic [3:0]                   acc_wr_en,
    output logic [ACC_ADDR_BITS-1:0]     acc_wr_addr [0:3],
    output logic signed [ACC_BITS-1:0]   acc_wr_data [0:3][0:LANES-1],

    output logic                         commit_valid,
    input  logic                         commit_ready,
    output logic                         commit_accept,
    output logic [ORDER_BITS-1:0]        commit_order,
    output logic [WINDOW_BITS-1:0]       commit_window,
    output logic [ROW_BITS-1:0]          commit_row,
    output logic [3:0]                   commit_bank_valid,
    output logic [2:0]                   commit_destination [0:3],
    output logic                         commit_last,

    output logic                         abort_valid,
    input  logic                         abort_ready,
    output logic                         abort_accept,
    output logic [ORDER_BITS-1:0]        abort_order,
    output logic [WINDOW_BITS-1:0]       abort_window,
    output logic [1:0]                   abort_discarded_tokens,
    output logic                         context_abort,

    output logic                         rmw_alias_stall,
    output logic [ORDER_BITS-1:0]        next_descriptor_order,
    output logic                         context_active,
    output logic                         protocol_error,
    output logic                         overflow_error,
    output logic                         busy
);
    localparam int ACC_DEPTH = 2 * ROWS;

    logic protocol_fault_q;
    logic overflow_fault_q;
    logic context_active_q;
    logic [SEQUENCE_BITS-1:0] context_sequence_q;
    logic [1:0] context_operator_q;
    logic [PARTITION_BITS-1:0] context_partition_q;
    logic [WINDOW_BITS-1:0] context_window_q;
    logic [EPOCH_BITS-1:0] context_weight_epoch_q;
    logic [PAYLOAD_BITS-1:0] context_payload_id_q;
    logic [ORDER_BITS-1:0] expected_order_q;
    logic last_seen_q;
    logic [1:0] overflow_discarded_q;

    logic [ACC_DEPTH-1:0] address_valid_q [0:3];

    logic [3:0] cache_valid_q;
    logic [SEQUENCE_BITS-1:0] cache_sequence_q [0:3];
    logic [1:0] cache_operator_q [0:3];
    logic [PARTITION_BITS-1:0] cache_partition_q [0:3];
    logic [WINDOW_BITS-1:0] cache_window_q [0:3];
    logic [EPOCH_BITS-1:0] cache_epoch_q [0:3];
    logic [PAYLOAD_BITS-1:0] cache_payload_q [0:3];
    logic [3:0] cache_source_q [0:3];
    logic cache_half_q [0:3];
    logic signed [7:0] cache_data_q [0:3][0:LANES-1];

    logic s0_valid_q;
    logic [SEQUENCE_BITS-1:0] s0_sequence_q;
    logic [1:0] s0_operator_q;
    logic [PARTITION_BITS-1:0] s0_partition_q;
    logic [WINDOW_BITS-1:0] s0_window_q;
    logic [EPOCH_BITS-1:0] s0_epoch_q;
    logic [PAYLOAD_BITS-1:0] s0_payload_q;
    logic [ORDER_BITS-1:0] s0_order_q;
    logic [ROW_BITS-1:0] s0_row_q;
    logic [3:0] s0_source_q;
    logic s0_half_q;
    logic [3:0] s0_bank_valid_q;
    logic [2:0] s0_destination_q [0:3];
    logic [3:0] s0_negate_q;
    logic s0_last_q;
    logic [3:0] s0_miss_q;
    logic [3:0] s0_use_hit_q;
    logic signed [7:0] s0_hit_data_q [0:3][0:LANES-1];
    logic s0_req_sent_q;

    logic s1_valid_q;
    logic [SEQUENCE_BITS-1:0] s1_sequence_q;
    logic [WINDOW_BITS-1:0] s1_window_q;
    logic [EPOCH_BITS-1:0] s1_epoch_q;
    logic [PAYLOAD_BITS-1:0] s1_payload_q;
    logic [ORDER_BITS-1:0] s1_order_q;
    logic [ROW_BITS-1:0] s1_row_q;
    logic [3:0] s1_bank_valid_q;
    logic [2:0] s1_destination_q [0:3];
    logic signed [8:0] s1_delta_q [0:3][0:LANES-1];
    logic s1_last_q;
    logic s1_req_sent_q;
    logic [3:0] s1_req_bank_valid_q;

    logic s2_valid_q;
    logic [WINDOW_BITS-1:0] s2_window_q;
    logic [ORDER_BITS-1:0] s2_order_q;
    logic [ROW_BITS-1:0] s2_row_q;
    logic [3:0] s2_bank_valid_q;
    logic [2:0] s2_destination_q [0:3];
    logic [ACC_ADDR_BITS-1:0] s2_addr_q [0:3];
    logic signed [ACC_BITS-1:0] s2_sum_q [0:3][0:LANES-1];
    logic s2_overflow_q;
    logic s2_last_q;

    logic loader_binding_match;
    logic illegal_shape;
    logic bank_conflict;
    logic mixed_half;
    logic row_out_of_range;
    logic context_mismatch;
    logic order_mismatch;
    logic request_collision;
    logic illegal_request;
    logic incoming_half;
    logic [3:0] incoming_bank_valid;
    logic [2:0] incoming_destination [0:3];
    logic [3:0] incoming_negate;
    logic [3:0] resident_hit;
    logic [3:0] inflight_hit;
    logic [3:0] effective_hit;
    logic signed [7:0] effective_hit_data [0:3][0:LANES-1];

    logic weight_rsp_tag_match;
    logic weight_rsp_mismatch;
    logic s0_hit_only;
    logic s0_to_s1;
    logic s0_capacity;
    logic s1_alias;
    logic [3:0] s1_base_mask;
    logic acc_rsp_tag_match;
    logic acc_rsp_mismatch;
    logic s1_to_s2;
    logic s1_capacity;
    logic s2_capacity;
    logic signed [ACC_BITS-1:0] candidate_sum [0:3][0:LANES-1];
    logic candidate_overflow;
    logic [1:0] current_younger_count;

`ifndef SYNTHESIS
    initial begin
        if (ACC_BITS != 19 || SEQUENCE_BITS != 32
                || PARTITION_BITS != 9 || WINDOW_BITS != 16
                || EPOCH_BITS != 16 || PAYLOAD_BITS != 32
                || ORDER_BITS != 16 || ROWS < 2 || ROWS > 512
                || LANES < 1)
            $fatal(1, "M241r2 unsupported geometry");
    end
`endif

    function automatic logic [ACC_ADDR_BITS-1:0] dense_acc_addr(
        input logic half,
        input logic [ROW_BITS-1:0] row_value);
        logic [ACC_ADDR_BITS:0] expanded;
        begin
            expanded = row_value;
            if (half)
                expanded = expanded + ROWS;
            dense_acc_addr = expanded[ACC_ADDR_BITS-1:0];
        end
    endfunction

    always_comb begin : input_normalize_and_audit
        loader_binding_match = loader_binding_valid
            && loader_binding_operator == context_open_operator
            && loader_binding_partition == context_open_partition
            && loader_binding_weight_epoch == context_open_weight_epoch
            && loader_binding_payload_id == context_open_payload_id;
        case (descriptor_destination_valid)
            4'b0001, 4'b0011, 4'b0111, 4'b1111:
                illegal_shape = 1'b0;
            default:
                illegal_shape = 1'b1;
        endcase
        incoming_half = descriptor_destination[0][2];
        bank_conflict = 1'b0;
        mixed_half = 1'b0;
        incoming_bank_valid = '0;
        incoming_negate = '0;
        for (int bank = 0; bank < 4; bank++)
            incoming_destination[bank] = '0;
        for (int later = 0; later < 4; later++) begin
            if (descriptor_destination_valid[later]) begin
                if (descriptor_destination[later][2] != incoming_half)
                    mixed_half = 1'b1;
                incoming_bank_valid[
                    descriptor_destination[later][1:0]] = 1'b1;
                incoming_destination[
                    descriptor_destination[later][1:0]] =
                        descriptor_destination[later];
                incoming_negate[
                    descriptor_destination[later][1:0]] =
                        descriptor_negate[later];
            end
            for (int earlier = 0; earlier < later; earlier++) begin
                if (descriptor_destination_valid[later]
                        && descriptor_destination_valid[earlier]
                        && descriptor_destination[later][1:0]
                           == descriptor_destination[earlier][1:0])
                    bank_conflict = 1'b1;
            end
        end
        row_out_of_range = descriptor_row >= ROWS;
        context_mismatch = descriptor_sequence != context_sequence_q
                         || descriptor_operator != context_operator_q
                         || descriptor_partition != context_partition_q
                         || descriptor_window != context_window_q
                         || descriptor_weight_epoch
                            != context_weight_epoch_q
                         || descriptor_payload_id != context_payload_id_q;
        order_mismatch = descriptor_order != expected_order_q;
        request_collision = (context_open_valid && descriptor_valid)
                          || (context_open_valid && context_close_valid)
                          || (descriptor_valid && context_close_valid);
        illegal_request = request_collision
            || (context_open_valid
                && (!loader_binding_match || context_active_q || s0_valid_q
                    || s1_valid_q || s2_valid_q))
            || (descriptor_valid
                && (!context_active_q || last_seen_q || illegal_shape
                    || bank_conflict || mixed_half || row_out_of_range
                    || context_mismatch || order_mismatch))
            || (context_close_valid
                && (!context_active_q || !last_seen_q || s0_valid_q
                    || s1_valid_q || s2_valid_q || descriptor_valid));
    end

    always_comb begin : response_identity
        weight_rsp_tag_match = s0_valid_q && s0_req_sent_q
            && weight_rsp_bank_valid == s0_miss_q
            && weight_rsp_sequence == s0_sequence_q
            && weight_rsp_operator == s0_operator_q
            && weight_rsp_partition == s0_partition_q
            && weight_rsp_window == s0_window_q
            && weight_rsp_weight_epoch == s0_epoch_q
            && weight_rsp_payload_id == s0_payload_q
            && weight_rsp_order == s0_order_q
            && weight_rsp_source == s0_source_q
            && weight_rsp_half == s0_half_q;
        weight_rsp_mismatch = weight_rsp_valid && !weight_rsp_tag_match;

        acc_rsp_tag_match = s1_valid_q && s1_req_sent_q
            && acc_rsp_bank_valid == s1_req_bank_valid_q
            && acc_rsp_sequence == s1_sequence_q
            && acc_rsp_window == s1_window_q
            && acc_rsp_weight_epoch == s1_epoch_q
            && acc_rsp_payload_id == s1_payload_q
            && acc_rsp_order == s1_order_q;
        acc_rsp_mismatch = acc_rsp_valid && !acc_rsp_tag_match;
    end

    always_comb begin : accumulator_flow
        s2_capacity = !s2_valid_q || commit_accept;
        s1_alias = 1'b0;
        if (s1_valid_q && s2_valid_q) begin
            for (int bank = 0; bank < 4; bank++) begin
                if (s1_bank_valid_q[bank] && s2_bank_valid_q[bank]
                        && dense_acc_addr(
                            s1_destination_q[bank][2], s1_row_q)
                           == s2_addr_q[bank])
                    s1_alias = 1'b1;
            end
        end
        for (int bank = 0; bank < 4; bank++) begin
            s1_base_mask[bank] = s1_valid_q && s1_bank_valid_q[bank]
                && address_valid_q[bank][dense_acc_addr(
                    s1_destination_q[bank][2], s1_row_q)];
        end
        acc_rsp_ready = !rst_core && s1_valid_q && s1_req_sent_q
                      && !s1_alias && s2_capacity
                      && (!acc_rsp_valid || acc_rsp_tag_match);
        acc_rsp_accept = acc_rsp_valid && acc_rsp_ready;
        s1_to_s2 = s1_valid_q && !s1_alias && s2_capacity
                 && (((|s1_base_mask) == 1'b0 && !s1_req_sent_q)
                     || acc_rsp_accept);
        s1_capacity = !s1_valid_q || s1_to_s2;
    end

    always_comb begin : weight_flow_and_cache_lookup
        weight_rsp_ready = !rst_core && s0_valid_q && s0_req_sent_q
                         && s1_capacity
                         && (!weight_rsp_valid || weight_rsp_tag_match);
        weight_rsp_accept = weight_rsp_valid && weight_rsp_ready;
        s0_hit_only = s0_valid_q && !(|s0_miss_q);
        s0_to_s1 = s0_valid_q && s1_capacity
                 && (s0_hit_only || weight_rsp_accept);
        s0_capacity = !s0_valid_q || s0_to_s1;

        resident_hit = '0;
        inflight_hit = '0;
        effective_hit = '0;
        for (int bank = 0; bank < 4; bank++) begin
            resident_hit[bank] = incoming_bank_valid[bank]
                && cache_valid_q[bank]
                && cache_sequence_q[bank] == descriptor_sequence
                && cache_operator_q[bank] == descriptor_operator
                && cache_partition_q[bank] == descriptor_partition
                && cache_window_q[bank] == descriptor_window
                && cache_epoch_q[bank] == descriptor_weight_epoch
                && cache_payload_q[bank] == descriptor_payload_id
                && cache_source_q[bank] == descriptor_source
                && cache_half_q[bank] == incoming_half;
            inflight_hit[bank] = incoming_bank_valid[bank]
                && s0_to_s1 && s0_bank_valid_q[bank]
                && s0_sequence_q == descriptor_sequence
                && s0_operator_q == descriptor_operator
                && s0_partition_q == descriptor_partition
                && s0_window_q == descriptor_window
                && s0_epoch_q == descriptor_weight_epoch
                && s0_payload_q == descriptor_payload_id
                && s0_source_q == descriptor_source
                && s0_half_q == incoming_half;
            effective_hit[bank] = resident_hit[bank] || inflight_hit[bank];
            for (int lane = 0; lane < LANES; lane++) begin
                if (resident_hit[bank])
                    effective_hit_data[bank][lane] =
                        cache_data_q[bank][lane];
                else if (inflight_hit[bank] && s0_use_hit_q[bank])
                    effective_hit_data[bank][lane] =
                        s0_hit_data_q[bank][lane];
                else
                    effective_hit_data[bank][lane] =
                        weight_rsp_data[bank][lane];
            end
        end
    end

    always_comb begin : macro_requests
        weight_req_valid = !rst_core && s0_valid_q && |s0_miss_q
                         && !s0_req_sent_q;
        weight_req_accept = weight_req_valid && weight_req_ready;
        weight_req_bank_valid = s0_miss_q;
        weight_req_sequence = s0_sequence_q;
        weight_req_operator = s0_operator_q;
        weight_req_partition = s0_partition_q;
        weight_req_window = s0_window_q;
        weight_req_weight_epoch = s0_epoch_q;
        weight_req_payload_id = s0_payload_q;
        weight_req_order = s0_order_q;
        weight_req_source = s0_source_q;
        weight_req_half = s0_half_q;
        for (int bank = 0; bank < 4; bank++)
            weight_req_addr[bank] = {s0_half_q, s0_source_q};

        acc_req_valid = !rst_core && s1_valid_q && !s1_alias
                      && |s1_base_mask && !s1_req_sent_q;
        acc_req_accept = acc_req_valid && acc_req_ready;
        acc_req_bank_valid = s1_base_mask;
        acc_req_sequence = s1_sequence_q;
        acc_req_window = s1_window_q;
        acc_req_weight_epoch = s1_epoch_q;
        acc_req_payload_id = s1_payload_q;
        acc_req_order = s1_order_q;
        for (int bank = 0; bank < 4; bank++) begin
            acc_req_addr[bank] = dense_acc_addr(
                s1_destination_q[bank][2], s1_row_q);
        end
    end

    always_comb begin : candidate_arithmetic
        candidate_overflow = 1'b0;
        for (int bank = 0; bank < 4; bank++) begin
            for (int lane = 0; lane < LANES; lane++) begin
                logic signed [ACC_BITS:0] base_ext;
                logic signed [ACC_BITS:0] delta_ext;
                logic signed [ACC_BITS:0] total;
                if (s1_req_sent_q && s1_req_bank_valid_q[bank])
                    base_ext = {acc_rsp_data[bank][lane][ACC_BITS-1],
                                acc_rsp_data[bank][lane]};
                else
                    base_ext = '0;
                delta_ext = {{(ACC_BITS + 1 - 9)
                              {s1_delta_q[bank][lane][8]}},
                             s1_delta_q[bank][lane]};
                total = base_ext + delta_ext;
                candidate_sum[bank][lane] = total[ACC_BITS-1:0];
                if (s1_to_s2 && s1_bank_valid_q[bank]
                        && total[ACC_BITS] != total[ACC_BITS-1])
                    candidate_overflow = 1'b1;
            end
        end
    end

    always_comb begin : outputs
        weight_cache_hit = descriptor_accept
                         ? (incoming_bank_valid & effective_hit) : '0;
        weight_cache_miss = descriptor_accept
                          ? (incoming_bank_valid & ~effective_hit) : '0;
        commit_valid = !rst_core && s2_valid_q && !s2_overflow_q;
        commit_accept = commit_valid && commit_ready;
        abort_valid = !rst_core && s2_valid_q && s2_overflow_q;
        abort_accept = abort_valid && abort_ready;
        commit_order = s2_order_q;
        commit_window = s2_window_q;
        commit_row = s2_row_q;
        commit_bank_valid = s2_bank_valid_q;
        commit_destination = s2_destination_q;
        commit_last = s2_last_q;
        abort_order = s2_order_q;
        abort_window = s2_window_q;
        current_younger_count = {1'b0, s0_valid_q}
                              + {1'b0, s1_valid_q};
        abort_discarded_tokens = overflow_fault_q
                               ? overflow_discarded_q
                               : current_younger_count;
        for (int bank = 0; bank < 4; bank++) begin
            acc_wr_en[bank] = commit_accept && s2_bank_valid_q[bank];
            acc_wr_addr[bank] = s2_addr_q[bank];
            for (int lane = 0; lane < LANES; lane++)
                acc_wr_data[bank][lane] = s2_sum_q[bank][lane];
        end
        rmw_alias_stall = !rst_core && s1_valid_q && s1_alias;
        next_descriptor_order = expected_order_q;
        context_active = context_active_q;
        protocol_error = !rst_core && (protocol_fault_q || illegal_request
                                       || weight_rsp_mismatch
                                       || acc_rsp_mismatch);
        overflow_error = !rst_core && (overflow_fault_q
                                       || (s2_valid_q && s2_overflow_q));
        busy = context_active_q || s0_valid_q || s1_valid_q || s2_valid_q;
    end

    assign context_open_ready = !rst_core && !protocol_fault_q
                              && !overflow_fault_q && !context_active_q
                              && !s0_valid_q && !s1_valid_q && !s2_valid_q
                              && loader_binding_match && !descriptor_valid
                              && !context_close_valid && !weight_rsp_valid
                              && !acc_rsp_valid;
    assign descriptor_ready = !rst_core && !protocol_fault_q
                            && !overflow_fault_q
                            && !(s2_valid_q && s2_overflow_q)
                            && context_active_q && !last_seen_q
                            && !illegal_shape && !bank_conflict && !mixed_half
                            && !row_out_of_range && !context_mismatch
                            && !order_mismatch && s0_capacity
                            && !context_open_valid && !context_close_valid;
    assign context_close_ready = !rst_core && !protocol_fault_q
                               && !overflow_fault_q && context_active_q
                               && last_seen_q && !s0_valid_q && !s1_valid_q
                               && !s2_valid_q && !context_open_valid
                               && !descriptor_valid && !weight_rsp_valid
                               && !acc_rsp_valid;
    assign context_open_accept = context_open_valid && context_open_ready;
    assign descriptor_accept = descriptor_valid && descriptor_ready;
    assign context_close_accept = context_close_valid && context_close_ready;

    always_ff @(posedge clk_core) begin : state_update
        if (rst_core) begin
            protocol_fault_q <= 1'b0;
            overflow_fault_q <= 1'b0;
            context_active_q <= 1'b0;
            context_sequence_q <= '0;
            context_operator_q <= '0;
            context_partition_q <= '0;
            context_window_q <= '0;
            context_weight_epoch_q <= '0;
            context_payload_id_q <= '0;
            expected_order_q <= '0;
            last_seen_q <= 1'b0;
            overflow_discarded_q <= '0;
            cache_valid_q <= '0;
            s0_valid_q <= 1'b0;
            s0_sequence_q <= '0;
            s0_operator_q <= '0;
            s0_partition_q <= '0;
            s0_window_q <= '0;
            s0_epoch_q <= '0;
            s0_payload_q <= '0;
            s0_order_q <= '0;
            s0_row_q <= '0;
            s0_source_q <= '0;
            s0_half_q <= 1'b0;
            s0_bank_valid_q <= '0;
            s0_negate_q <= '0;
            s0_last_q <= 1'b0;
            s0_miss_q <= '0;
            s0_use_hit_q <= '0;
            s0_req_sent_q <= 1'b0;
            s1_valid_q <= 1'b0;
            s1_sequence_q <= '0;
            s1_window_q <= '0;
            s1_epoch_q <= '0;
            s1_payload_q <= '0;
            s1_order_q <= '0;
            s1_row_q <= '0;
            s1_bank_valid_q <= '0;
            s1_last_q <= 1'b0;
            s1_req_sent_q <= 1'b0;
            s1_req_bank_valid_q <= '0;
            s2_valid_q <= 1'b0;
            s2_window_q <= '0;
            s2_order_q <= '0;
            s2_row_q <= '0;
            s2_bank_valid_q <= '0;
            s2_overflow_q <= 1'b0;
            s2_last_q <= 1'b0;
            window_done <= 1'b0;
            context_abort <= 1'b0;
            for (int bank = 0; bank < 4; bank++) begin
                address_valid_q[bank] <= '0;
                cache_sequence_q[bank] <= '0;
                cache_operator_q[bank] <= '0;
                cache_partition_q[bank] <= '0;
                cache_window_q[bank] <= '0;
                cache_epoch_q[bank] <= '0;
                cache_payload_q[bank] <= '0;
                cache_source_q[bank] <= '0;
                cache_half_q[bank] <= 1'b0;
                s0_destination_q[bank] <= '0;
                s1_destination_q[bank] <= '0;
                s2_destination_q[bank] <= '0;
                s2_addr_q[bank] <= '0;
                for (int lane = 0; lane < LANES; lane++) begin
                    cache_data_q[bank][lane] <= '0;
                    s0_hit_data_q[bank][lane] <= '0;
                    s1_delta_q[bank][lane] <= '0;
                    s2_sum_q[bank][lane] <= '0;
                end
            end
        end else begin
            window_done <= 1'b0;
            context_abort <= 1'b0;
            if (illegal_request || weight_rsp_mismatch || acc_rsp_mismatch)
                protocol_fault_q <= 1'b1;

            // Overflow is an abort, not a successful commit.  Once visible,
            // accepted younger tokens are counted and quarantined exactly once.
            if (s2_valid_q && s2_overflow_q) begin
                if (!overflow_fault_q) begin
                    overflow_fault_q <= 1'b1;
                    overflow_discarded_q <= current_younger_count;
                    s0_valid_q <= 1'b0;
                    s1_valid_q <= 1'b0;
                end
                if (abort_accept) begin
                    s2_valid_q <= 1'b0;
                    context_abort <= 1'b1;
                end
            end else begin
                if (commit_accept) begin
                    for (int bank = 0; bank < 4; bank++) begin
                        if (s2_bank_valid_q[bank])
                            address_valid_q[bank][s2_addr_q[bank]] <= 1'b1;
                    end
                end

                if (s2_capacity) begin
                    s2_valid_q <= s1_to_s2;
                    if (s1_to_s2) begin
                        s2_window_q <= s1_window_q;
                        s2_order_q <= s1_order_q;
                        s2_row_q <= s1_row_q;
                        s2_bank_valid_q <= s1_bank_valid_q;
                        s2_overflow_q <= candidate_overflow;
                        s2_last_q <= s1_last_q;
                        for (int bank = 0; bank < 4; bank++) begin
                            s2_destination_q[bank]
                                <= s1_destination_q[bank];
                            s2_addr_q[bank] <= dense_acc_addr(
                                s1_destination_q[bank][2], s1_row_q);
                            for (int lane = 0; lane < LANES; lane++)
                                s2_sum_q[bank][lane]
                                    <= candidate_sum[bank][lane];
                        end
                    end
                end

                if (s1_capacity) begin
                    s1_valid_q <= s0_to_s1;
                    s1_req_sent_q <= 1'b0;
                    s1_req_bank_valid_q <= '0;
                    if (s0_to_s1) begin
                        s1_sequence_q <= s0_sequence_q;
                        s1_window_q <= s0_window_q;
                        s1_epoch_q <= s0_epoch_q;
                        s1_payload_q <= s0_payload_q;
                        s1_order_q <= s0_order_q;
                        s1_row_q <= s0_row_q;
                        s1_bank_valid_q <= s0_bank_valid_q;
                        s1_last_q <= s0_last_q;
                        for (int bank = 0; bank < 4; bank++) begin
                            logic signed [8:0] widened;
                            s1_destination_q[bank]
                                <= s0_destination_q[bank];
                            if (s0_bank_valid_q[bank]) begin
                                cache_valid_q[bank] <= 1'b1;
                                cache_sequence_q[bank] <= s0_sequence_q;
                                cache_operator_q[bank] <= s0_operator_q;
                                cache_partition_q[bank] <= s0_partition_q;
                                cache_window_q[bank] <= s0_window_q;
                                cache_epoch_q[bank] <= s0_epoch_q;
                                cache_payload_q[bank] <= s0_payload_q;
                                cache_source_q[bank] <= s0_source_q;
                                cache_half_q[bank] <= s0_half_q;
                            end
                            for (int lane = 0; lane < LANES; lane++) begin
                                logic signed [7:0] selected_weight;
                                selected_weight = s0_use_hit_q[bank]
                                    ? s0_hit_data_q[bank][lane]
                                    : weight_rsp_data[bank][lane];
                                if (s0_bank_valid_q[bank])
                                    cache_data_q[bank][lane]
                                        <= selected_weight;
                                widened = {selected_weight[7],
                                           selected_weight};
                                s1_delta_q[bank][lane]
                                    <= s0_negate_q[bank]
                                       ? -widened : widened;
                            end
                        end
                    end
                end else if (acc_req_accept) begin
                    s1_req_sent_q <= 1'b1;
                    s1_req_bank_valid_q <= s1_base_mask;
                end

                if (s0_capacity) begin
                    s0_valid_q <= descriptor_accept;
                    s0_req_sent_q <= 1'b0;
                    if (descriptor_accept) begin
                        s0_sequence_q <= descriptor_sequence;
                        s0_operator_q <= descriptor_operator;
                        s0_partition_q <= descriptor_partition;
                        s0_window_q <= descriptor_window;
                        s0_epoch_q <= descriptor_weight_epoch;
                        s0_payload_q <= descriptor_payload_id;
                        s0_order_q <= descriptor_order;
                        s0_row_q <= descriptor_row;
                        s0_source_q <= descriptor_source;
                        s0_half_q <= incoming_half;
                        s0_bank_valid_q <= incoming_bank_valid;
                        s0_negate_q <= incoming_negate;
                        s0_last_q <= descriptor_last;
                        s0_miss_q <= incoming_bank_valid & ~effective_hit;
                        s0_use_hit_q <= incoming_bank_valid & effective_hit;
                        for (int bank = 0; bank < 4; bank++) begin
                            s0_destination_q[bank]
                                <= incoming_destination[bank];
                            for (int lane = 0; lane < LANES; lane++)
                                s0_hit_data_q[bank][lane]
                                    <= effective_hit_data[bank][lane];
                        end
                    end
                end else if (weight_req_accept) begin
                    s0_req_sent_q <= 1'b1;
                end

                if (context_open_accept) begin
                    context_active_q <= 1'b1;
                    context_sequence_q <= context_open_sequence;
                    context_operator_q <= context_open_operator;
                    context_partition_q <= context_open_partition;
                    context_window_q <= context_open_window;
                    context_weight_epoch_q <= context_open_weight_epoch;
                    context_payload_id_q <= context_open_payload_id;
                    expected_order_q <= '0;
                    last_seen_q <= 1'b0;
                    cache_valid_q <= '0;
                    for (int bank = 0; bank < 4; bank++)
                        address_valid_q[bank] <= '0;
                end
                if (descriptor_accept) begin
                    expected_order_q <= expected_order_q + 1'b1;
                    if (descriptor_last)
                        last_seen_q <= 1'b1;
                end
                if (context_close_accept) begin
                    context_active_q <= 1'b0;
                    window_done <= 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
