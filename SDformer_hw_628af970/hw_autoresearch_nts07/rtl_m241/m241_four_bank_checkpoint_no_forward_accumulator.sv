`timescale 1ns/1ps
`default_nettype none

// M241: checkpoint-tagged four-bank INT8 weight-cache to Acc19 accumulator.
//
// This standalone island consumes the source-major, destination-half ordered
// descriptor stream selected by M152/M157.  Four independent resident weight
// banks supply one signed-INT8 vector per active destination bank.  A small
// bank-local cache retains {partition, epoch, source, destination-half}; an
// accepted descriptor can therefore reuse an in-flight or resident vector.
//
// Accumulator addresses are dense: half*ROWS + row.  A one-entry RMW stage and
// an explicit same-address interlock make the result independent of SRAM
// read-during-write behavior without carrying M155's forwarding-data payload.
// Lazy-valid state and the signed19 overflow guard remain present.  SRAMs,
// checkpoint loading, the upstream descriptorizer and final commit scan are
// explicit external cuts.
module m241_four_bank_checkpoint_no_forward_accumulator #(
    parameter int LANES = 8,
    parameter int ROWS = 384,
    parameter int ACC_BITS = 19,
    parameter int SEQUENCE_BITS = 32,
    parameter int PARTITION_BITS = 9,
    parameter int EPOCH_BITS = 16,
    parameter int ORDER_BITS = 16,
    parameter int ROW_BITS = $clog2(ROWS),
    parameter int ACC_ADDR_BITS = $clog2(2 * ROWS)
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         context_open_valid,
    output logic                         context_open_ready,
    input  logic [SEQUENCE_BITS-1:0]     context_open_sequence,
    input  logic [1:0]                   context_open_operator,
    input  logic [PARTITION_BITS-1:0]    context_open_partition,
    input  logic [EPOCH_BITS-1:0]        context_open_weight_epoch,
    output logic                         context_open_accept,

    input  logic                         descriptor_valid,
    output logic                         descriptor_ready,
    input  logic [SEQUENCE_BITS-1:0]     descriptor_sequence,
    input  logic [1:0]                   descriptor_operator,
    input  logic [PARTITION_BITS-1:0]    descriptor_partition,
    input  logic [EPOCH_BITS-1:0]        descriptor_weight_epoch,
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

    output logic [3:0]                   weight_rd_en,
    output logic [4:0]                   weight_rd_addr [0:3],
    input  logic signed [7:0]            weight_rd_data [0:3][0:LANES-1],
    output logic [3:0]                   weight_cache_hit,
    output logic [3:0]                   weight_cache_miss,

    output logic [3:0]                   acc_rd_en,
    output logic [ACC_ADDR_BITS-1:0]     acc_rd_addr [0:3],
    input  logic signed [ACC_BITS-1:0]   acc_rd_data [0:3][0:LANES-1],
    output logic [3:0]                   acc_wr_en,
    output logic [ACC_ADDR_BITS-1:0]     acc_wr_addr [0:3],
    output logic signed [ACC_BITS-1:0]   acc_wr_data [0:3][0:LANES-1],

    output logic                         commit_valid,
    input  logic                         commit_ready,
    output logic                         commit_accept,
    output logic [ORDER_BITS-1:0]        commit_order,
    output logic [ROW_BITS-1:0]          commit_row,
    output logic [3:0]                   commit_bank_valid,
    output logic [2:0]                   commit_destination [0:3],
    output logic                         commit_last,

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
    logic [EPOCH_BITS-1:0] context_weight_epoch_q;
    logic [ORDER_BITS-1:0] expected_order_q;
    logic last_seen_q;

    logic [ACC_DEPTH-1:0] address_valid_q [0:3];

    logic [3:0] cache_valid_q;
    logic [3:0] cache_source_q [0:3];
    logic cache_half_q [0:3];
    logic [PARTITION_BITS-1:0] cache_partition_q [0:3];
    logic [EPOCH_BITS-1:0] cache_epoch_q [0:3];
    logic signed [7:0] cache_data_q [0:3][0:LANES-1];

    logic s0_valid_q;
    logic [PARTITION_BITS-1:0] s0_partition_q;
    logic [EPOCH_BITS-1:0] s0_epoch_q;
    logic [ORDER_BITS-1:0] s0_order_q;
    logic [ROW_BITS-1:0] s0_row_q;
    logic [3:0] s0_source_q;
    logic s0_half_q;
    logic [3:0] s0_bank_valid_q;
    logic [2:0] s0_destination_q [0:3];
    logic [3:0] s0_negate_q;
    logic s0_last_q;
    logic [3:0] s0_macro_miss_q;
    logic [3:0] s0_use_hit_data_q;
    logic signed [7:0] s0_hit_data_q [0:3][0:LANES-1];

    logic s1_valid_q;
    logic [ORDER_BITS-1:0] s1_order_q;
    logic [ROW_BITS-1:0] s1_row_q;
    logic [3:0] s1_bank_valid_q;
    logic [2:0] s1_destination_q [0:3];
    logic signed [8:0] s1_delta_q [0:3][0:LANES-1];
    logic s1_last_q;

    logic s2_valid_q;
    logic [ORDER_BITS-1:0] s2_order_q;
    logic [ROW_BITS-1:0] s2_row_q;
    logic [3:0] s2_bank_valid_q;
    logic [2:0] s2_destination_q [0:3];
    logic [ACC_ADDR_BITS-1:0] s2_addr_q [0:3];
    logic [3:0] s2_base_valid_q;
    logic signed [8:0] s2_delta_q [0:3][0:LANES-1];
    logic s2_last_q;

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
    logic s2_capacity;
    logic s1_hazard;
    logic s1_capacity;
    logic s0_capacity;
    logic s1_issue;
    logic s2_overflow_any;
    logic signed [ACC_BITS:0] write_sum [0:3][0:LANES-1];

`ifndef SYNTHESIS
    initial begin
        if (ACC_BITS != 19 || SEQUENCE_BITS != 32
                || PARTITION_BITS != 9 || EPOCH_BITS != 16
                || ORDER_BITS != 16 || ROWS < 2 || ROWS > 512
                || LANES < 1)
            $fatal(1, "M241 unsupported geometry");
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

    always_comb begin : normalize_and_audit
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
                         || descriptor_weight_epoch
                            != context_weight_epoch_q;
        order_mismatch = descriptor_order != expected_order_q;
        request_collision = (context_open_valid && descriptor_valid)
                          || (context_open_valid && context_close_valid)
                          || (descriptor_valid && context_close_valid);
        illegal_request = request_collision
            || (context_open_valid
                && (context_active_q || s0_valid_q || s1_valid_q
                    || s2_valid_q))
            || (descriptor_valid
                && (!context_active_q || last_seen_q || illegal_shape
                    || bank_conflict || mixed_half || row_out_of_range
                    || context_mismatch || order_mismatch))
            || (context_close_valid
                && (!context_active_q || !last_seen_q || s0_valid_q
                    || s1_valid_q || s2_valid_q || descriptor_valid));
    end

    always_comb begin : cache_lookup
        resident_hit = '0;
        inflight_hit = '0;
        effective_hit = '0;
        for (int bank = 0; bank < 4; bank++) begin
            resident_hit[bank] = incoming_bank_valid[bank]
                && cache_valid_q[bank]
                && cache_source_q[bank] == descriptor_source
                && cache_half_q[bank] == incoming_half
                && cache_partition_q[bank] == descriptor_partition
                && cache_epoch_q[bank] == descriptor_weight_epoch;
            inflight_hit[bank] = incoming_bank_valid[bank]
                && s0_valid_q && s0_bank_valid_q[bank]
                && s0_source_q == descriptor_source
                && s0_half_q == incoming_half
                && s0_partition_q == descriptor_partition
                && s0_epoch_q == descriptor_weight_epoch;
            effective_hit[bank] = resident_hit[bank] || inflight_hit[bank];
            for (int lane = 0; lane < LANES; lane++) begin
                if (resident_hit[bank])
                    effective_hit_data[bank][lane] =
                        cache_data_q[bank][lane];
                else if (inflight_hit[bank] && s0_use_hit_data_q[bank])
                    effective_hit_data[bank][lane] =
                        s0_hit_data_q[bank][lane];
                else
                    effective_hit_data[bank][lane] =
                        weight_rd_data[bank][lane];
            end
        end
    end

    always_comb begin : pipeline_control
        s2_capacity = !s2_valid_q || commit_ready;
        s1_hazard = 1'b0;
        if (s1_valid_q && s2_valid_q) begin
            for (int bank = 0; bank < 4; bank++) begin
                if (s1_bank_valid_q[bank] && s2_bank_valid_q[bank]
                        && dense_acc_addr(
                            s1_destination_q[bank][2], s1_row_q)
                           == s2_addr_q[bank])
                    s1_hazard = 1'b1;
            end
        end
        s1_issue = s1_valid_q && s2_capacity && !s1_hazard;
        s1_capacity = !s1_valid_q || (s2_capacity && !s1_hazard);
        s0_capacity = !s0_valid_q || s1_capacity;
    end

    always_comb begin : macro_ports_and_sum
        weight_rd_en = '0;
        weight_cache_hit = '0;
        weight_cache_miss = '0;
        for (int bank = 0; bank < 4; bank++) begin
            weight_rd_addr[bank] = {incoming_half, descriptor_source};
            if (descriptor_accept && incoming_bank_valid[bank]) begin
                weight_cache_hit[bank] = effective_hit[bank];
                weight_cache_miss[bank] = !effective_hit[bank];
                weight_rd_en[bank] = !effective_hit[bank];
            end

            acc_rd_en[bank] = s1_issue && s1_bank_valid_q[bank]
                && address_valid_q[bank][dense_acc_addr(
                    s1_destination_q[bank][2], s1_row_q)];
            acc_rd_addr[bank] = dense_acc_addr(
                s1_destination_q[bank][2], s1_row_q);
            acc_wr_addr[bank] = s2_addr_q[bank];
        end

        s2_overflow_any = 1'b0;
        for (int bank = 0; bank < 4; bank++) begin
            for (int lane = 0; lane < LANES; lane++) begin
                logic signed [ACC_BITS:0] base_ext;
                logic signed [ACC_BITS:0] delta_ext;
                if (s2_base_valid_q[bank])
                    base_ext = {acc_rd_data[bank][lane][ACC_BITS-1],
                                acc_rd_data[bank][lane]};
                else
                    base_ext = '0;
                delta_ext = {{(ACC_BITS + 1 - 9)
                              {s2_delta_q[bank][lane][8]}},
                             s2_delta_q[bank][lane]};
                write_sum[bank][lane] = base_ext + delta_ext;
                if (s2_valid_q && s2_bank_valid_q[bank]
                        && write_sum[bank][lane][ACC_BITS]
                           != write_sum[bank][lane][ACC_BITS-1])
                    s2_overflow_any = 1'b1;
                acc_wr_data[bank][lane] =
                    write_sum[bank][lane][ACC_BITS-1:0];
            end
        end
        for (int bank = 0; bank < 4; bank++) begin
            acc_wr_en[bank] = commit_accept && !s2_overflow_any
                            && s2_bank_valid_q[bank];
        end
    end

    assign context_open_ready = !rst_core && !protocol_fault_q
                              && !overflow_fault_q && !context_active_q
                              && !s0_valid_q && !s1_valid_q && !s2_valid_q
                              && !descriptor_valid && !context_close_valid;
    assign descriptor_ready = !rst_core && !protocol_fault_q
                            && !overflow_fault_q && !s2_overflow_any
                            && context_active_q && !last_seen_q
                            && !illegal_shape && !bank_conflict && !mixed_half
                            && !row_out_of_range && !context_mismatch
                            && !order_mismatch && s0_capacity
                            && !context_open_valid && !context_close_valid;
    assign context_close_ready = !rst_core && !protocol_fault_q
                               && !overflow_fault_q && context_active_q
                               && last_seen_q && !s0_valid_q && !s1_valid_q
                               && !s2_valid_q && !context_open_valid
                               && !descriptor_valid;
    assign context_open_accept = context_open_valid && context_open_ready;
    assign descriptor_accept = descriptor_valid && descriptor_ready;
    assign context_close_accept = context_close_valid && context_close_ready;
    assign commit_valid = !rst_core && s2_valid_q;
    assign commit_accept = commit_valid && commit_ready;
    assign commit_order = s2_order_q;
    assign commit_row = s2_row_q;
    assign commit_bank_valid = s2_bank_valid_q;
    assign commit_destination = s2_destination_q;
    assign commit_last = s2_last_q;
    assign rmw_alias_stall = !rst_core && s1_valid_q && s1_hazard;
    assign next_descriptor_order = expected_order_q;
    assign context_active = context_active_q;
    assign protocol_error = !rst_core && (protocol_fault_q
                                          || illegal_request);
    assign overflow_error = !rst_core && (overflow_fault_q
                                          || (s2_valid_q
                                              && s2_overflow_any));
    assign busy = context_active_q || s0_valid_q || s1_valid_q || s2_valid_q;

    always_ff @(posedge clk_core) begin : state_update
        if (rst_core) begin
            protocol_fault_q <= 1'b0;
            overflow_fault_q <= 1'b0;
            context_active_q <= 1'b0;
            context_sequence_q <= '0;
            context_operator_q <= '0;
            context_partition_q <= '0;
            context_weight_epoch_q <= '0;
            expected_order_q <= '0;
            last_seen_q <= 1'b0;
            cache_valid_q <= '0;
            s0_valid_q <= 1'b0;
            s0_partition_q <= '0;
            s0_epoch_q <= '0;
            s0_order_q <= '0;
            s0_row_q <= '0;
            s0_source_q <= '0;
            s0_half_q <= 1'b0;
            s0_bank_valid_q <= '0;
            s0_negate_q <= '0;
            s0_last_q <= 1'b0;
            s0_macro_miss_q <= '0;
            s0_use_hit_data_q <= '0;
            s1_valid_q <= 1'b0;
            s1_order_q <= '0;
            s1_row_q <= '0;
            s1_bank_valid_q <= '0;
            s1_last_q <= 1'b0;
            s2_valid_q <= 1'b0;
            s2_order_q <= '0;
            s2_row_q <= '0;
            s2_bank_valid_q <= '0;
            s2_base_valid_q <= '0;
            s2_last_q <= 1'b0;
            window_done <= 1'b0;
            for (int bank = 0; bank < 4; bank++) begin
                address_valid_q[bank] <= '0;
                cache_source_q[bank] <= '0;
                cache_half_q[bank] <= 1'b0;
                cache_partition_q[bank] <= '0;
                cache_epoch_q[bank] <= '0;
                s0_destination_q[bank] <= '0;
                s1_destination_q[bank] <= '0;
                s2_destination_q[bank] <= '0;
                s2_addr_q[bank] <= '0;
                for (int lane = 0; lane < LANES; lane++) begin
                    cache_data_q[bank][lane] <= '0;
                    s0_hit_data_q[bank][lane] <= '0;
                    s1_delta_q[bank][lane] <= '0;
                    s2_delta_q[bank][lane] <= '0;
                end
            end
        end else begin
            window_done <= 1'b0;
            if (illegal_request)
                protocol_fault_q <= 1'b1;

            // Numeric overflow is the oldest token.  Suppress all four writes
            // and quarantine already accepted younger tokens; reset is the
            // only recovery.  Protocol faults instead drain older tokens.
            if (s2_valid_q && s2_overflow_any) begin
                overflow_fault_q <= 1'b1;
                s0_valid_q <= 1'b0;
                s1_valid_q <= 1'b0;
                if (commit_ready)
                    s2_valid_q <= 1'b0;
            end else begin
                if (commit_accept) begin
                    for (int bank = 0; bank < 4; bank++) begin
                        if (s2_bank_valid_q[bank])
                            address_valid_q[bank][s2_addr_q[bank]] <= 1'b1;
                    end
                end

                if (s2_capacity) begin
                    s2_valid_q <= s1_valid_q && !s1_hazard;
                    if (s1_valid_q && !s1_hazard) begin
                        s2_order_q <= s1_order_q;
                        s2_row_q <= s1_row_q;
                        s2_bank_valid_q <= s1_bank_valid_q;
                        s2_last_q <= s1_last_q;
                        for (int bank = 0; bank < 4; bank++) begin
                            s2_destination_q[bank]
                                <= s1_destination_q[bank];
                            s2_addr_q[bank] <= dense_acc_addr(
                                s1_destination_q[bank][2], s1_row_q);
                            s2_base_valid_q[bank]
                                <= address_valid_q[bank][dense_acc_addr(
                                    s1_destination_q[bank][2], s1_row_q)];
                            for (int lane = 0; lane < LANES; lane++)
                                s2_delta_q[bank][lane]
                                    <= s1_delta_q[bank][lane];
                        end
                    end
                end

                if (s1_capacity) begin
                    s1_valid_q <= s0_valid_q;
                    if (s0_valid_q) begin
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
                                cache_source_q[bank] <= s0_source_q;
                                cache_half_q[bank] <= s0_half_q;
                                cache_partition_q[bank] <= s0_partition_q;
                                cache_epoch_q[bank] <= s0_epoch_q;
                            end
                            for (int lane = 0; lane < LANES; lane++) begin
                                logic signed [7:0] selected_weight;
                                selected_weight = s0_use_hit_data_q[bank]
                                    ? s0_hit_data_q[bank][lane]
                                    : weight_rd_data[bank][lane];
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
                end

                if (s0_capacity) begin
                    s0_valid_q <= descriptor_accept;
                    if (descriptor_accept) begin
                        s0_partition_q <= descriptor_partition;
                        s0_epoch_q <= descriptor_weight_epoch;
                        s0_order_q <= descriptor_order;
                        s0_row_q <= descriptor_row;
                        s0_source_q <= descriptor_source;
                        s0_half_q <= incoming_half;
                        s0_bank_valid_q <= incoming_bank_valid;
                        s0_negate_q <= incoming_negate;
                        s0_last_q <= descriptor_last;
                        s0_macro_miss_q <= incoming_bank_valid
                                         & ~effective_hit;
                        s0_use_hit_data_q <= incoming_bank_valid
                                           & effective_hit;
                        for (int bank = 0; bank < 4; bank++) begin
                            s0_destination_q[bank]
                                <= incoming_destination[bank];
                            for (int lane = 0; lane < LANES; lane++)
                                s0_hit_data_q[bank][lane]
                                    <= effective_hit_data[bank][lane];
                        end
                    end
                end

                if (context_open_accept) begin
                    context_active_q <= 1'b1;
                    context_sequence_q <= context_open_sequence;
                    context_operator_q <= context_open_operator;
                    context_partition_q <= context_open_partition;
                    context_weight_epoch_q <= context_open_weight_epoch;
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
