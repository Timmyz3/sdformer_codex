`timescale 1ns/1ps
`default_nettype none

// Destination-tagged cross-block K4 event packer.
//
// One raw row contains eight 16-bit source masks.  Instead of independently
// padding each destination block to K4, this block emits the first four set
// bits in canonical linear order {destination_block, source}.  Therefore one
// row produces exactly ceil(total_popcount/4) descriptors while preserving
// every tuple.  The first descriptor is fall-through so a one-descriptor or
// zero row consumes one accepted cycle, matching the M147 producer contract.
// Numeric update arithmetic, weight storage, and destination accumulators are
// outside this standalone module.
module m148_destination_tagged_mosaic_k4_packer #(
    parameter int DESTINATIONS = 8,
    parameter int SOURCES = 16,
    parameter int MASK_BITS = DESTINATIONS * SOURCES,
    parameter int SEQUENCE_BITS = 32,
    parameter int ROW_BITS = 9
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         row_valid,
    output logic                         row_ready,
    input  logic [SEQUENCE_BITS-1:0]     row_sequence,
    input  logic [ROW_BITS-1:0]          row_id,
    input  logic [MASK_BITS-1:0]         row_event_mask,
    output logic                         row_accept,

    output logic                         descriptor_valid,
    input  logic                         descriptor_ready,
    output logic [SEQUENCE_BITS-1:0]     descriptor_sequence,
    output logic [ROW_BITS-1:0]          descriptor_row,
    output logic [1:0]                   descriptor_count_m1,
    output logic [2:0]                   descriptor_destination [0:3],
    output logic [3:0]                   descriptor_source [0:3],
    output logic [3:0]                   descriptor_tuple_valid,
    output logic                         descriptor_last,
    output logic                         descriptor_accept,

    output logic                         done_valid,
    output logic [SEQUENCE_BITS-1:0]     done_sequence,
    output logic [ROW_BITS-1:0]          done_row,

    output logic                         observed_active,
    output logic [MASK_BITS-1:0]         observed_remaining_mask,
    output logic [7:0]                   observed_work_popcount,
    output logic [SEQUENCE_BITS-1:0]     observed_next_sequence,
    output logic                         protocol_error,
    output logic                         busy
);
    logic active_q;
    logic [MASK_BITS-1:0] remaining_q;
    logic [SEQUENCE_BITS-1:0] active_sequence_q;
    logic [ROW_BITS-1:0] active_row_q;
    logic [SEQUENCE_BITS-1:0] next_sequence_q;
    logic fault_q;

    logic [MASK_BITS-1:0] work_mask;
    logic [MASK_BITS-1:0] mask_after_picks;
    logic [MASK_BITS-1:0] selection_mask;
    logic [31:0] candidate_valid;
    logic [2:0] candidate_destination [0:31];
    logic [3:0] candidate_source [0:31];
    logic [DESTINATIONS-1:0] block_has_fifth;
    logic more_than_four;
    logic illegal_row;
    logic quarantine;
    logic row_capacity;

`ifndef SYNTHESIS
    initial begin
        if (DESTINATIONS != 8 || SOURCES != 16 || MASK_BITS != 128
                || SEQUENCE_BITS != 32 || ROW_BITS != 9)
            $fatal(1, "M148 production geometry drift");
    end
`endif

    assign illegal_row = row_valid && row_sequence != next_sequence_q;
    assign quarantine = fault_q || illegal_row;
    assign protocol_error = !rst_core && quarantine;
    assign row_capacity = !rst_core && !fault_q && !active_q;
    assign row_ready = row_capacity
                     && (!row_valid
                         || (!illegal_row
                             && ((row_event_mask == 0)
                                 || descriptor_ready)));
    assign row_accept = row_valid && row_ready;

    assign work_mask = active_q ? remaining_q : row_event_mask;
    assign descriptor_valid = !rst_core && !quarantine
                            && (active_q || row_valid)
                            && work_mask != 0;
    assign descriptor_accept = descriptor_valid && descriptor_ready;
    assign descriptor_sequence = active_q
        ? active_sequence_q : row_sequence;
    assign descriptor_row = active_q ? active_row_q : row_id;

    always_comb begin : hierarchical_canonical_first_four
        logic [31:0] candidate_scan;

        // Stage 1: each destination contributes at most its first four source
        // candidates.  A global first-four result can never require a fifth
        // candidate from any one destination.
        candidate_valid = '0;
        block_has_fifth = '0;
        for (int candidate = 0; candidate < 32; candidate++) begin
            candidate_destination[candidate] = '0;
            candidate_source[candidate] = '0;
        end
        for (int destination = 0; destination < DESTINATIONS;
                destination++) begin
            logic [15:0] block_scan;
            block_scan = work_mask[destination * SOURCES +: SOURCES];
            for (int local_pick = 0; local_pick < 4; local_pick++) begin
                logic found;
                int candidate;
                found = 1'b0;
                candidate = destination * 4 + local_pick;
                for (int source = 0; source < SOURCES; source++) begin
                    if (!found && block_scan[source]) begin
                        found = 1'b1;
                        candidate_valid[candidate] = 1'b1;
                        candidate_destination[candidate]
                            = destination[2:0];
                        candidate_source[candidate] = source[3:0];
                        block_scan[source] = 1'b0;
                    end
                end
            end
            block_has_fifth[destination] = |block_scan;
        end

        // Stage 2: merge the 32 already ordered local candidates.  This is a
        // much shallower cone than four serial scans across all 128 raw bits.
        candidate_scan = candidate_valid;
        descriptor_tuple_valid = '0;
        descriptor_count_m1 = '0;
        observed_work_popcount = '0;
        for (int linear = 0; linear < MASK_BITS; linear++)
            observed_work_popcount = observed_work_popcount
                                   + work_mask[linear];
        for (int pick = 0; pick < 4; pick++) begin
            logic found;
            found = 1'b0;
            descriptor_destination[pick] = '0;
            descriptor_source[pick] = '0;
            for (int candidate = 0; candidate < 32; candidate++) begin
                if (!found && candidate_scan[candidate]) begin
                    found = 1'b1;
                    descriptor_tuple_valid[pick] = 1'b1;
                    descriptor_destination[pick]
                        = candidate_destination[candidate];
                    descriptor_source[pick] = candidate_source[candidate];
                    candidate_scan[candidate] = 1'b0;
                end
            end
        end
        more_than_four = (|candidate_scan) || (|block_has_fifth);
        selection_mask = '0;
        for (int linear = 0; linear < MASK_BITS; linear++) begin
            for (int pick = 0; pick < 4; pick++) begin
                if (descriptor_tuple_valid[pick]
                        && linear[6:0]
                           == {descriptor_destination[pick],
                               descriptor_source[pick]})
                    selection_mask[linear] = 1'b1;
            end
        end
        mask_after_picks = work_mask & ~selection_mask;
        case (descriptor_tuple_valid)
            4'b0001: descriptor_count_m1 = 2'd0;
            4'b0011: descriptor_count_m1 = 2'd1;
            4'b0111: descriptor_count_m1 = 2'd2;
            default: descriptor_count_m1 = 2'd3;
        endcase
    end

    // Last depends only on whether a fifth candidate exists.  Keep the
    // selection-mask writeback cone out of the visible completion path.
    assign descriptor_last = descriptor_valid && !more_than_four;
    assign done_valid = (row_accept && row_event_mask == 0)
                      || (descriptor_accept && descriptor_last);
    // The active selector alone identifies both legal completion cases.  Do
    // not feed descriptor_last into the identity mux: that would put the full
    // four-pick priority cone on completion data for no semantic reason.
    assign done_sequence = active_q ? active_sequence_q : row_sequence;
    assign done_row = active_q ? active_row_q : row_id;
    assign observed_active = active_q;
    assign observed_remaining_mask = remaining_q;
    assign observed_next_sequence = next_sequence_q;
    assign busy = active_q;

    always_ff @(posedge clk_core) begin : packer_state
        if (rst_core) begin
            active_q <= 1'b0;
            remaining_q <= '0;
            active_sequence_q <= '0;
            active_row_q <= '0;
            next_sequence_q <= '0;
            fault_q <= 1'b0;
        end else begin
            if (illegal_row)
                fault_q <= 1'b1;
            if (!quarantine) begin
                if (row_accept)
                    next_sequence_q <= next_sequence_q + 1'b1;
                if (descriptor_accept) begin
                    if (descriptor_last) begin
                        active_q <= 1'b0;
                        remaining_q <= '0;
                    end else begin
                        active_q <= 1'b1;
                        remaining_q <= mask_after_picks;
                        if (!active_q) begin
                            active_sequence_q <= row_sequence;
                            active_row_q <= row_id;
                        end
                    end
                end else if (row_accept && row_event_mask == 0) begin
                    active_q <= 1'b0;
                    remaining_q <= '0;
                end
            end
        end
    end
endmodule

`default_nettype wire
