`timescale 1ns/1ps
`default_nettype none

// Four-tuple destination conflict resolver for the M148 mosaic stream.
//
// Every accepted tuple carries one signed 96-lane contribution vector.  The
// first occurrence of each destination owns one result port; all later tuples
// with that destination are folded into the same vector.  An explicit 9-bit
// signed-negate stage preserves -(-128) == +128.  Two 10-bit pair sums and one
// 11-bit final sum cover the exact four-term range [-512, +512].
//
// The result register is a one-entry elastic stage: latency is one accepted
// clock and the initiation interval is one when the consumer is ready.  This
// standalone island assumes all four contribution vectors are already
// available.  Weight/PWP storage, SRAM ports, and accumulator commit are out
// of scope and must be priced separately.
module m149_destination_conflict_resolved_k4_combiner #(
    parameter int LANES = 96,
    parameter int SEQUENCE_BITS = 32,
    parameter int ROW_BITS = 9
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         descriptor_valid,
    output logic                         descriptor_ready,
    input  logic [SEQUENCE_BITS-1:0]     descriptor_sequence,
    input  logic [ROW_BITS-1:0]          descriptor_row,
    input  logic                         descriptor_last,
    input  logic [3:0]                   descriptor_tuple_valid,
    input  logic [2:0]                   tuple_destination [0:3],
    input  logic                         tuple_negate [0:3],
    input  logic signed [7:0]            tuple_vector [0:3][0:LANES-1],
    output logic                         descriptor_accept,

    output logic                         result_valid,
    input  logic                         result_ready,
    output logic [SEQUENCE_BITS-1:0]     result_sequence,
    output logic [ROW_BITS-1:0]          result_row,
    output logic                         result_last,
    output logic [3:0]                   result_group_valid,
    output logic [2:0]                   result_destination [0:3],
    output logic signed [10:0]           result_vector [0:3][0:LANES-1],
    output logic                         result_accept,

    output logic                         protocol_error,
    output logic                         busy
);
    logic fault_q;
    logic result_valid_q;
    logic [SEQUENCE_BITS-1:0] result_sequence_q;
    logic [ROW_BITS-1:0] result_row_q;
    logic result_last_q;
    logic [3:0] result_group_valid_q;
    logic [2:0] result_destination_q [0:3];
    logic signed [10:0] result_vector_q [0:3][0:LANES-1];

    logic illegal_descriptor;
    logic output_capacity;
    logic [3:0] combined_group_valid;
    logic [2:0] combined_destination [0:3];
    logic signed [10:0] combined_vector [0:3][0:LANES-1];

`ifndef SYNTHESIS
    initial begin
        if (LANES != 96 || SEQUENCE_BITS != 32 || ROW_BITS != 9)
            $fatal(1, "M149 production geometry drift");
    end
`endif

    always_comb begin : descriptor_legality
        illegal_descriptor = 1'b0;
        if (descriptor_valid) begin
            case (descriptor_tuple_valid)
                4'b0001, 4'b0011, 4'b0111, 4'b1111:
                    illegal_descriptor = 1'b0;
                default:
                    illegal_descriptor = 1'b1;
            endcase
        end
    end

    assign result_valid = !rst_core && !fault_q && result_valid_q;
    assign result_sequence = result_sequence_q;
    assign result_row = result_row_q;
    assign result_last = result_last_q;
    assign result_group_valid = result_group_valid_q;
    assign result_destination = result_destination_q;
    assign result_vector = result_vector_q;
    assign result_accept = result_valid && result_ready;
    assign output_capacity = !result_valid_q || result_ready;
    assign descriptor_ready = !rst_core && !fault_q
                            && !illegal_descriptor && output_capacity;
    assign descriptor_accept = descriptor_valid && descriptor_ready;
    assign protocol_error = !rst_core && (fault_q || illegal_descriptor);
    assign busy = result_valid_q;

    always_comb begin : exact_destination_fold
        logic signed [8:0] signed_operand [0:3][0:LANES-1];

        combined_group_valid = '0;
        for (int tuple = 0; tuple < 4; tuple++) begin
            combined_destination[tuple] = tuple_destination[tuple];
            combined_group_valid[tuple] = descriptor_tuple_valid[tuple];
            for (int earlier = 0; earlier < tuple; earlier++) begin
                if (descriptor_tuple_valid[earlier]
                        && tuple_destination[earlier]
                           == tuple_destination[tuple])
                    combined_group_valid[tuple] = 1'b0;
            end
        end

        for (int tuple = 0; tuple < 4; tuple++) begin
            for (int lane = 0; lane < LANES; lane++) begin
                logic signed [8:0] extended_value;
                extended_value = {tuple_vector[tuple][lane][7],
                                  tuple_vector[tuple][lane]};
                signed_operand[tuple][lane] = tuple_negate[tuple]
                    ? -extended_value : extended_value;
            end
        end

        for (int owner = 0; owner < 4; owner++) begin
            for (int lane = 0; lane < LANES; lane++) begin
                logic signed [9:0] pair01;
                logic signed [9:0] pair23;
                logic signed [8:0] term0;
                logic signed [8:0] term1;
                logic signed [8:0] term2;
                logic signed [8:0] term3;

                term0 = (descriptor_tuple_valid[0]
                         && tuple_destination[0]
                            == tuple_destination[owner])
                    ? signed_operand[0][lane] : 9'sd0;
                term1 = (descriptor_tuple_valid[1]
                         && tuple_destination[1]
                            == tuple_destination[owner])
                    ? signed_operand[1][lane] : 9'sd0;
                term2 = (descriptor_tuple_valid[2]
                         && tuple_destination[2]
                            == tuple_destination[owner])
                    ? signed_operand[2][lane] : 9'sd0;
                term3 = (descriptor_tuple_valid[3]
                         && tuple_destination[3]
                            == tuple_destination[owner])
                    ? signed_operand[3][lane] : 9'sd0;
                pair01 = term0 + term1;
                pair23 = term2 + term3;
                combined_vector[owner][lane]
                    = $signed({pair01[9], pair01})
                    + $signed({pair23[9], pair23});
                if (!combined_group_valid[owner])
                    combined_vector[owner][lane] = 11'sd0;
            end
        end
    end

    always_ff @(posedge clk_core) begin : elastic_result_stage
        if (rst_core) begin
            fault_q <= 1'b0;
            result_valid_q <= 1'b0;
            result_sequence_q <= '0;
            result_row_q <= '0;
            result_last_q <= 1'b0;
            result_group_valid_q <= '0;
            for (int group = 0; group < 4; group++) begin
                result_destination_q[group] <= '0;
                for (int lane = 0; lane < LANES; lane++)
                    result_vector_q[group][lane] <= '0;
            end
        end else begin
            if (illegal_descriptor) begin
                fault_q <= 1'b1;
                result_valid_q <= 1'b0;
            end else if (!fault_q && output_capacity) begin
                result_valid_q <= descriptor_accept;
                if (descriptor_accept) begin
                    result_sequence_q <= descriptor_sequence;
                    result_row_q <= descriptor_row;
                    result_last_q <= descriptor_last;
                    result_group_valid_q <= combined_group_valid;
                    for (int group = 0; group < 4; group++) begin
                        result_destination_q[group]
                            <= combined_destination[group];
                        for (int lane = 0; lane < LANES; lane++)
                            result_vector_q[group][lane]
                                <= combined_vector[group][lane];
                    end
                end
            end
        end
    end
endmodule

`default_nettype wire
