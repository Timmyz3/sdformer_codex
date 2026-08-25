`timescale 1ns/1ps
`default_nettype none

// Four-bank frozen-weight supplier for cross-destination K4 descriptors.
//
// Unlike M151, this block never assumes that two destination blocks share a
// weight vector.  A resident partition contains 8 destinations x 16 sources x
// 96 signed-INT8 lanes (98,304 bits), split by destination[1:0] into four
// 32x768-bit banks.  A legal descriptor contains at most one destination from
// each bank and therefore issues as many as four independent vector reads in
// one cycle.  destination[2] selects the low/high destination in a bank.
//
// The SRAM arrays and their loading are explicit external cuts.  rd_data is a
// one-cycle synchronous response to rd_en/rd_addr.  One request register plus
// one elastic result register sustains descriptor II=1 when the consumer is
// ready and drains accepted traffic even after a fail-closed protocol fault.
module m154_four_bank_destination_vector_supplier #(
    parameter int LANES = 96,
    parameter int SEQUENCE_BITS = 32,
    parameter int PARTITION_BITS = 9
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         context_open_valid,
    output logic                         context_open_ready,
    input  logic [SEQUENCE_BITS-1:0]     context_open_sequence,
    input  logic [1:0]                   context_open_operator,
    input  logic [PARTITION_BITS-1:0]    context_open_partition,
    output logic                         context_open_accept,

    input  logic                         descriptor_valid,
    output logic                         descriptor_ready,
    input  logic [SEQUENCE_BITS-1:0]     descriptor_sequence,
    input  logic [1:0]                   descriptor_operator,
    input  logic [PARTITION_BITS-1:0]    descriptor_partition,
    input  logic [8:0]                   descriptor_row,
    input  logic [3:0]                   descriptor_source,
    input  logic [3:0]                   descriptor_destination_valid,
    input  logic [2:0]                   descriptor_destination [0:3],
    input  logic [3:0]                   descriptor_negate,
    input  logic                         descriptor_last,
    output logic                         descriptor_accept,

    input  logic                         context_close_valid,
    output logic                         context_close_ready,
    output logic                         context_close_accept,

    output logic [3:0]                   bank_rd_en,
    output logic [4:0]                   bank_rd_addr [0:3],
    input  logic signed [7:0]            bank_rd_data [0:3][0:LANES-1],

    output logic                         result_valid,
    input  logic                         result_ready,
    output logic [SEQUENCE_BITS-1:0]     result_sequence,
    output logic [1:0]                   result_operator,
    output logic [PARTITION_BITS-1:0]    result_partition,
    output logic [8:0]                   result_row,
    output logic [3:0]                   result_source,
    output logic [3:0]                   result_destination_valid,
    output logic [2:0]                   result_destination [0:3],
    output logic [3:0]                   result_negate,
    output logic                         result_last,
    output logic signed [7:0]            result_vector [0:3][0:LANES-1],
    output logic                         result_accept,

    output logic                         context_active,
    output logic                         protocol_error,
    output logic                         busy
);
    logic fault_q;
    logic context_active_q;
    logic [SEQUENCE_BITS-1:0] context_sequence_q;
    logic [1:0] context_operator_q;
    logic [PARTITION_BITS-1:0] context_partition_q;

    logic request_valid_q;
    logic [SEQUENCE_BITS-1:0] request_sequence_q;
    logic [1:0] request_operator_q;
    logic [PARTITION_BITS-1:0] request_partition_q;
    logic [8:0] request_row_q;
    logic [3:0] request_source_q;
    logic [3:0] request_destination_valid_q;
    logic [2:0] request_destination_q [0:3];
    logic [1:0] request_bank_q [0:3];
    logic [3:0] request_negate_q;
    logic request_last_q;

    logic result_valid_q;
    logic [SEQUENCE_BITS-1:0] result_sequence_q;
    logic [1:0] result_operator_q;
    logic [PARTITION_BITS-1:0] result_partition_q;
    logic [8:0] result_row_q;
    logic [3:0] result_source_q;
    logic [3:0] result_destination_valid_q;
    logic [2:0] result_destination_q [0:3];
    logic [3:0] result_negate_q;
    logic result_last_q;
    logic signed [7:0] result_vector_q [0:3][0:LANES-1];

    logic illegal_shape;
    logic bank_conflict;
    logic context_mismatch;
    logic request_collision;
    logic illegal_request;
    logic result_capacity;
    logic request_capacity;

`ifndef SYNTHESIS
    initial begin
        if (LANES != 96 || SEQUENCE_BITS != 32 || PARTITION_BITS != 9)
            $fatal(1, "M154 production geometry drift");
    end
`endif

    always_comb begin : protocol_audit
        case (descriptor_destination_valid)
            4'b0001, 4'b0011, 4'b0111, 4'b1111:
                illegal_shape = 1'b0;
            default:
                illegal_shape = 1'b1;
        endcase
        bank_conflict = 1'b0;
        for (int later = 1; later < 4; later++) begin
            for (int earlier = 0; earlier < later; earlier++) begin
                if (descriptor_destination_valid[later]
                        && descriptor_destination_valid[earlier]
                        && descriptor_destination[later][1:0]
                           == descriptor_destination[earlier][1:0])
                    bank_conflict = 1'b1;
            end
        end
        context_mismatch = descriptor_sequence != context_sequence_q
                         || descriptor_operator != context_operator_q
                         || descriptor_partition != context_partition_q;
        request_collision = (context_open_valid && descriptor_valid)
                          || (context_open_valid && context_close_valid)
                          || (descriptor_valid && context_close_valid);
        illegal_request = request_collision
            || (context_open_valid
                && (context_active_q || request_valid_q || result_valid_q))
            || (descriptor_valid
                && (!context_active_q || illegal_shape || bank_conflict
                    || context_mismatch))
            || (context_close_valid
                && (!context_active_q || request_valid_q || result_valid_q));
    end

    assign result_capacity = !result_valid_q || result_ready;
    assign request_capacity = !request_valid_q || result_capacity;
    assign context_open_ready = !rst_core && !fault_q
                              && !context_active_q
                              && !request_valid_q && !result_valid_q
                              && !descriptor_valid && !context_close_valid;
    assign descriptor_ready = !rst_core && !fault_q && context_active_q
                            && !illegal_shape && !bank_conflict
                            && !context_mismatch && request_capacity
                            && !context_open_valid && !context_close_valid;
    assign context_close_ready = !rst_core && !fault_q && context_active_q
                               && !request_valid_q && !result_valid_q
                               && !context_open_valid && !descriptor_valid;
    assign context_open_accept = context_open_valid && context_open_ready;
    assign descriptor_accept = descriptor_valid && descriptor_ready;
    assign context_close_accept = context_close_valid && context_close_ready;

    always_comb begin : bank_request_route
        bank_rd_en = '0;
        for (int bank = 0; bank < 4; bank++)
            bank_rd_addr[bank] = '0;
        if (descriptor_accept) begin
            for (int tuple = 0; tuple < 4; tuple++) begin
                if (descriptor_destination_valid[tuple]) begin
                    bank_rd_en[descriptor_destination[tuple][1:0]] = 1'b1;
                    bank_rd_addr[descriptor_destination[tuple][1:0]] = {
                        descriptor_destination[tuple][2],
                        descriptor_source};
                end
            end
        end
    end

    assign result_valid = !rst_core && result_valid_q;
    assign result_sequence = result_sequence_q;
    assign result_operator = result_operator_q;
    assign result_partition = result_partition_q;
    assign result_row = result_row_q;
    assign result_source = result_source_q;
    assign result_destination_valid = result_destination_valid_q;
    assign result_destination = result_destination_q;
    assign result_negate = result_negate_q;
    assign result_last = result_last_q;
    assign result_vector = result_vector_q;
    assign result_accept = result_valid && result_ready;
    assign context_active = context_active_q;
    assign protocol_error = !rst_core && (fault_q || illegal_request);
    assign busy = context_active_q || request_valid_q || result_valid_q;

    always_ff @(posedge clk_core) begin : state_update
        if (rst_core) begin
            fault_q <= 1'b0;
            context_active_q <= 1'b0;
            context_sequence_q <= '0;
            context_operator_q <= '0;
            context_partition_q <= '0;
            request_valid_q <= 1'b0;
            request_sequence_q <= '0;
            request_operator_q <= '0;
            request_partition_q <= '0;
            request_row_q <= '0;
            request_source_q <= '0;
            request_destination_valid_q <= '0;
            request_negate_q <= '0;
            request_last_q <= 1'b0;
            result_valid_q <= 1'b0;
            result_sequence_q <= '0;
            result_operator_q <= '0;
            result_partition_q <= '0;
            result_row_q <= '0;
            result_source_q <= '0;
            result_destination_valid_q <= '0;
            result_negate_q <= '0;
            result_last_q <= 1'b0;
            for (int tuple = 0; tuple < 4; tuple++) begin
                request_destination_q[tuple] <= '0;
                request_bank_q[tuple] <= '0;
                result_destination_q[tuple] <= '0;
                for (int lane = 0; lane < LANES; lane++)
                    result_vector_q[tuple][lane] <= '0;
            end
        end else begin
            if (illegal_request)
                fault_q <= 1'b1;

            // Accepted requests and results already in flight remain lossless
            // after a malformed younger request raises the sticky fault.
            if (result_capacity) begin
                result_valid_q <= request_valid_q;
                if (request_valid_q) begin
                    result_sequence_q <= request_sequence_q;
                    result_operator_q <= request_operator_q;
                    result_partition_q <= request_partition_q;
                    result_row_q <= request_row_q;
                    result_source_q <= request_source_q;
                    result_destination_valid_q
                        <= request_destination_valid_q;
                    result_negate_q <= request_negate_q;
                    result_last_q <= request_last_q;
                    for (int tuple = 0; tuple < 4; tuple++) begin
                        result_destination_q[tuple]
                            <= request_destination_q[tuple];
                        for (int lane = 0; lane < LANES; lane++) begin
                            result_vector_q[tuple][lane]
                                <= request_destination_valid_q[tuple]
                                ? bank_rd_data[request_bank_q[tuple]][lane]
                                : 8'sd0;
                        end
                    end
                end
            end

            if (request_capacity) begin
                request_valid_q <= descriptor_accept;
                if (descriptor_accept) begin
                    request_sequence_q <= descriptor_sequence;
                    request_operator_q <= descriptor_operator;
                    request_partition_q <= descriptor_partition;
                    request_row_q <= descriptor_row;
                    request_source_q <= descriptor_source;
                    request_destination_valid_q
                        <= descriptor_destination_valid;
                    request_negate_q <= descriptor_negate;
                    request_last_q <= descriptor_last;
                    for (int tuple = 0; tuple < 4; tuple++) begin
                        request_destination_q[tuple]
                            <= descriptor_destination[tuple];
                        request_bank_q[tuple]
                            <= descriptor_destination[tuple][1:0];
                    end
                end
            end

            if (context_open_accept) begin
                context_active_q <= 1'b1;
                context_sequence_q <= context_open_sequence;
                context_operator_q <= context_open_operator;
                context_partition_q <= context_open_partition;
            end
            if (context_close_accept)
                context_active_q <= 1'b0;
        end
    end
endmodule

`default_nettype wire
