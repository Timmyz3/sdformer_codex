`timescale 1ns/1ps
`default_nettype none

// Ping-pong source-resident frontend for source-stationary K4 descriptors.
//
// A reconstructed signed-11 96-lane source vector is loaded once into either
// of two resident slots.  Each accepted descriptor then broadcasts that one
// vector to up to four distinct destination IDs, with independent negate bits.
// The second slot permits the next source to be loaded while the current source
// emits its two typical four-destination descriptors.  The vector is carried
// once across the output interface; four accumulator write ports and per-port
// signed application are explicit downstream cuts.
module m151_dual_buffer_source_resident_k4_multicast_frontend #(
    parameter int LANES = 96,
    parameter int SEQUENCE_BITS = 32,
    parameter int ROW_BITS = 9
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         load_valid,
    output logic                         load_ready,
    input  logic [SEQUENCE_BITS-1:0]     load_sequence,
    input  logic [ROW_BITS-1:0]          load_row,
    input  logic [3:0]                   load_source,
    input  logic signed [10:0]           load_vector [0:LANES-1],
    output logic                         load_accept,

    input  logic                         descriptor_valid,
    output logic                         descriptor_ready,
    input  logic [SEQUENCE_BITS-1:0]     descriptor_sequence,
    input  logic [ROW_BITS-1:0]          descriptor_row,
    input  logic [3:0]                   descriptor_source,
    input  logic [3:0]                   descriptor_destination_valid,
    input  logic [2:0]                   descriptor_destination [0:3],
    input  logic [3:0]                   descriptor_negate,
    input  logic                         descriptor_last_for_source,
    output logic                         descriptor_accept,

    output logic                         multicast_valid,
    input  logic                         multicast_ready,
    output logic [SEQUENCE_BITS-1:0]     multicast_sequence,
    output logic [ROW_BITS-1:0]          multicast_row,
    output logic [3:0]                   multicast_source,
    output logic [3:0]                   multicast_destination_valid,
    output logic [2:0]                   multicast_destination [0:3],
    output logic [3:0]                   multicast_negate,
    output logic                         multicast_last_for_source,
    output logic signed [10:0]           multicast_vector [0:LANES-1],
    output logic                         multicast_accept,

    output logic                         release_valid,
    output logic [SEQUENCE_BITS-1:0]     release_sequence,
    output logic [ROW_BITS-1:0]          release_row,
    output logic [3:0]                   release_source,

    output logic [1:0]                   resident_valid,
    output logic [1:0]                   resident_retiring,
    output logic                         protocol_error,
    output logic                         busy
);
    logic [1:0] slot_valid_q;
    logic [1:0] slot_retiring_q;
    logic [SEQUENCE_BITS-1:0] slot_sequence_q [0:1];
    logic [ROW_BITS-1:0] slot_row_q [0:1];
    logic [3:0] slot_source_q [0:1];
    logic signed [10:0] slot_vector_q [0:1][0:LANES-1];

    logic multicast_valid_q;
    logic multicast_slot_q;
    logic [SEQUENCE_BITS-1:0] multicast_sequence_q;
    logic [ROW_BITS-1:0] multicast_row_q;
    logic [3:0] multicast_source_q;
    logic [3:0] multicast_destination_valid_q;
    logic [2:0] multicast_destination_q [0:3];
    logic [3:0] multicast_negate_q;
    logic multicast_last_for_source_q;
    logic signed [10:0] multicast_vector_q [0:LANES-1];
    logic fault_q;

    logic load_duplicate;
    logic free_slot_valid;
    logic free_slot;
    logic descriptor_slot_hit [0:1];
    logic descriptor_match_valid;
    logic descriptor_match_slot;
    logic illegal_destination_shape;
    logic duplicate_destination;
    logic illegal_descriptor;
    logic illegal_request;
    logic output_capacity;

`ifndef SYNTHESIS
    initial begin
        if (LANES != 96 || SEQUENCE_BITS != 32 || ROW_BITS != 9)
            $fatal(1, "M151 production geometry drift");
    end
`endif

    always_comb begin : identity_and_protocol_audit
        load_duplicate = 1'b0;
        for (int slot = 0; slot < 2; slot++) begin
            if (slot_valid_q[slot]
                    && load_sequence == slot_sequence_q[slot]
                    && load_row == slot_row_q[slot]
                    && load_source == slot_source_q[slot])
                load_duplicate = 1'b1;
        end
        free_slot_valid = !slot_valid_q[0] || !slot_valid_q[1];
        free_slot = slot_valid_q[0] && !slot_valid_q[1];

        for (int slot = 0; slot < 2; slot++) begin
            descriptor_slot_hit[slot] = slot_valid_q[slot]
                && !slot_retiring_q[slot]
                && descriptor_sequence == slot_sequence_q[slot]
                && descriptor_row == slot_row_q[slot]
                && descriptor_source == slot_source_q[slot];
        end
        descriptor_match_valid = descriptor_slot_hit[0]
                               || descriptor_slot_hit[1];
        descriptor_match_slot = descriptor_slot_hit[1];

        case (descriptor_destination_valid)
            4'b0001, 4'b0011, 4'b0111, 4'b1111:
                illegal_destination_shape = 1'b0;
            default:
                illegal_destination_shape = 1'b1;
        endcase
        duplicate_destination = 1'b0;
        for (int later = 1; later < 4; later++) begin
            for (int earlier = 0; earlier < later; earlier++) begin
                if (descriptor_destination_valid[later]
                        && descriptor_destination_valid[earlier]
                        && descriptor_destination[later]
                           == descriptor_destination[earlier])
                    duplicate_destination = 1'b1;
            end
        end
        illegal_descriptor = descriptor_valid
            && (!descriptor_match_valid || illegal_destination_shape
                || duplicate_destination);
        illegal_request = (load_valid && load_duplicate)
                       || illegal_descriptor;
    end

    assign output_capacity = !multicast_valid_q || multicast_ready;
    assign protocol_error = !rst_core && (fault_q || illegal_request);
    assign load_ready = !rst_core && !protocol_error && free_slot_valid;
    assign descriptor_ready = !rst_core && !protocol_error
                            && descriptor_match_valid && output_capacity;
    assign load_accept = load_valid && load_ready;
    assign descriptor_accept = descriptor_valid && descriptor_ready;

    assign multicast_valid = !rst_core && !protocol_error
                           && multicast_valid_q;
    assign multicast_sequence = multicast_sequence_q;
    assign multicast_row = multicast_row_q;
    assign multicast_source = multicast_source_q;
    assign multicast_destination_valid = multicast_destination_valid_q;
    assign multicast_destination = multicast_destination_q;
    assign multicast_negate = multicast_negate_q;
    assign multicast_last_for_source = multicast_last_for_source_q;
    assign multicast_vector = multicast_vector_q;
    assign multicast_accept = multicast_valid && multicast_ready;

    assign release_valid = multicast_accept
                         && multicast_last_for_source_q;
    assign release_sequence = multicast_sequence_q;
    assign release_row = multicast_row_q;
    assign release_source = multicast_source_q;
    assign resident_valid = slot_valid_q;
    assign resident_retiring = slot_retiring_q;
    assign busy = (|slot_valid_q) || multicast_valid_q;

    always_ff @(posedge clk_core) begin : resident_and_output_state
        if (rst_core) begin
            slot_valid_q <= '0;
            slot_retiring_q <= '0;
            multicast_valid_q <= 1'b0;
            multicast_slot_q <= 1'b0;
            multicast_sequence_q <= '0;
            multicast_row_q <= '0;
            multicast_source_q <= '0;
            multicast_destination_valid_q <= '0;
            multicast_negate_q <= '0;
            multicast_last_for_source_q <= 1'b0;
            fault_q <= 1'b0;
            for (int slot = 0; slot < 2; slot++) begin
                slot_sequence_q[slot] <= '0;
                slot_row_q[slot] <= '0;
                slot_source_q[slot] <= '0;
                for (int lane = 0; lane < LANES; lane++)
                    slot_vector_q[slot][lane] <= '0;
            end
            for (int destination = 0; destination < 4; destination++)
                multicast_destination_q[destination] <= '0;
            for (int lane = 0; lane < LANES; lane++)
                multicast_vector_q[lane] <= '0;
        end else begin
            if (illegal_request) begin
                fault_q <= 1'b1;
                multicast_valid_q <= 1'b0;
            end else if (!fault_q) begin
                if (multicast_accept && multicast_last_for_source_q) begin
                    slot_valid_q[multicast_slot_q] <= 1'b0;
                    slot_retiring_q[multicast_slot_q] <= 1'b0;
                end

                if (load_accept) begin
                    slot_valid_q[free_slot] <= 1'b1;
                    slot_retiring_q[free_slot] <= 1'b0;
                    slot_sequence_q[free_slot] <= load_sequence;
                    slot_row_q[free_slot] <= load_row;
                    slot_source_q[free_slot] <= load_source;
                    for (int lane = 0; lane < LANES; lane++)
                        slot_vector_q[free_slot][lane] <= load_vector[lane];
                end

                if (output_capacity) begin
                    multicast_valid_q <= descriptor_accept;
                    if (descriptor_accept) begin
                        multicast_slot_q <= descriptor_match_slot;
                        multicast_sequence_q <= descriptor_sequence;
                        multicast_row_q <= descriptor_row;
                        multicast_source_q <= descriptor_source;
                        multicast_destination_valid_q
                            <= descriptor_destination_valid;
                        multicast_negate_q <= descriptor_negate;
                        multicast_last_for_source_q
                            <= descriptor_last_for_source;
                        for (int destination = 0; destination < 4;
                                destination++)
                            multicast_destination_q[destination]
                                <= descriptor_destination[destination];
                        for (int lane = 0; lane < LANES; lane++)
                            multicast_vector_q[lane]
                                <= slot_vector_q[descriptor_match_slot][lane];
                        if (descriptor_last_for_source)
                            slot_retiring_q[descriptor_match_slot] <= 1'b1;
                    end
                end
            end
        end
    end
endmodule

`default_nettype wire
