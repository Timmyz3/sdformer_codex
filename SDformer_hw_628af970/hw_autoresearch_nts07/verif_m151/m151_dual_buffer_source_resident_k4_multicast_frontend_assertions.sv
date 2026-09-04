`timescale 1ns/1ps
`default_nettype none

module m151_dual_buffer_source_resident_k4_multicast_frontend_assertions #(
    parameter int LANES = 96,
    parameter int SEQUENCE_BITS = 32,
    parameter int ROW_BITS = 9
) (
    input logic                         clk_core,
    input logic                         rst_core,
    input logic                         load_valid,
    input logic                         load_ready,
    input logic [SEQUENCE_BITS-1:0]     load_sequence,
    input logic [ROW_BITS-1:0]          load_row,
    input logic [3:0]                   load_source,
    input logic signed [10:0]           load_vector [0:LANES-1],
    input logic                         load_accept,
    input logic                         descriptor_valid,
    input logic                         descriptor_ready,
    input logic [SEQUENCE_BITS-1:0]     descriptor_sequence,
    input logic [ROW_BITS-1:0]          descriptor_row,
    input logic [3:0]                   descriptor_source,
    input logic [3:0]                   descriptor_destination_valid,
    input logic [2:0]                   descriptor_destination [0:3],
    input logic [3:0]                   descriptor_negate,
    input logic                         descriptor_last_for_source,
    input logic                         descriptor_accept,
    input logic                         multicast_valid,
    input logic                         multicast_ready,
    input logic [SEQUENCE_BITS-1:0]     multicast_sequence,
    input logic [ROW_BITS-1:0]          multicast_row,
    input logic [3:0]                   multicast_source,
    input logic [3:0]                   multicast_destination_valid,
    input logic [2:0]                   multicast_destination [0:3],
    input logic [3:0]                   multicast_negate,
    input logic                         multicast_last_for_source,
    input logic signed [10:0]           multicast_vector [0:LANES-1],
    input logic                         multicast_accept,
    input logic                         release_valid,
    input logic [SEQUENCE_BITS-1:0]     release_sequence,
    input logic [ROW_BITS-1:0]          release_row,
    input logic [3:0]                   release_source,
    input logic [1:0]                   resident_valid,
    input logic [1:0]                   resident_retiring,
    input logic                         protocol_error,
    input logic                         busy
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_load_accept_definition:
        assert property (load_accept == (load_valid && load_ready));
    ap_descriptor_accept_definition:
        assert property (descriptor_accept
                         == (descriptor_valid && descriptor_ready));
    ap_multicast_accept_definition:
        assert property (multicast_accept
                         == (multicast_valid && multicast_ready));
    ap_load_metadata_stable_under_stall:
        assert property (load_valid && !load_ready && !protocol_error
            |=> load_valid
                && $stable({load_sequence, load_row, load_source}));
    ap_descriptor_metadata_stable_under_stall:
        assert property (descriptor_valid && !descriptor_ready
                         && !protocol_error
            |=> descriptor_valid
                && $stable({descriptor_sequence, descriptor_row,
                            descriptor_source,
                            descriptor_destination_valid,
                            descriptor_destination[0],
                            descriptor_destination[1],
                            descriptor_destination[2],
                            descriptor_destination[3], descriptor_negate,
                            descriptor_last_for_source}));
    ap_multicast_metadata_stable_under_stall:
        assert property (multicast_valid && !multicast_ready
            |=> multicast_valid
                && $stable({multicast_sequence, multicast_row,
                            multicast_source,
                            multicast_destination_valid,
                            multicast_destination[0],
                            multicast_destination[1],
                            multicast_destination[2],
                            multicast_destination[3], multicast_negate,
                            multicast_last_for_source}));
    ap_multicast_follows_descriptor:
        assert property (descriptor_accept
            |=> multicast_valid
                && multicast_sequence == $past(descriptor_sequence)
                && multicast_row == $past(descriptor_row)
                && multicast_source == $past(descriptor_source)
                && multicast_destination_valid
                   == $past(descriptor_destination_valid)
                && multicast_negate == $past(descriptor_negate)
                && multicast_last_for_source
                   == $past(descriptor_last_for_source));
    ap_release_definition:
        assert property (release_valid
            == (multicast_accept && multicast_last_for_source));
    ap_release_identity:
        assert property (release_valid
            |-> release_sequence == multicast_sequence
                && release_row == multicast_row
                && release_source == multicast_source);
    ap_retiring_subset:
        assert property ((resident_retiring & ~resident_valid) == 0);
    ap_protocol_error_sticky:
        assert property (protocol_error |=> protocol_error);
    ap_busy_if_resident:
        assert property ((|resident_valid) |-> busy);

    cp_both_slots_resident:
        cover property (resident_valid == 2'b11);
    cp_overlap_load_descriptor:
        cover property (load_accept && descriptor_accept);
    cp_full_four_destination:
        cover property (multicast_accept
                        && multicast_destination_valid == 4'b1111);
    cp_tail_one_destination:
        cover property (multicast_accept
                        && multicast_destination_valid == 4'b0001);
    cp_tail_two_destination:
        cover property (multicast_accept
                        && multicast_destination_valid == 4'b0011);
    cp_tail_three_destination:
        cover property (multicast_accept
                        && multicast_destination_valid == 4'b0111);
    cp_multicast_stall:
        cover property (multicast_valid && !multicast_ready
                        ##1 multicast_valid && multicast_ready);
    cp_back_to_back_descriptor:
        cover property (descriptor_accept ##1 descriptor_accept);
    cp_release_and_other_slot_live:
        cover property (release_valid && $countones(resident_valid) == 2);

    generate
        for (genvar lane = 0; lane < LANES; lane++) begin : g_lane
            ap_load_lane_stable_under_stall:
                assert property (load_valid && !load_ready && !protocol_error
                    |=> $stable(load_vector[lane]));
            ap_multicast_lane_stable_under_stall:
                assert property (multicast_valid && !multicast_ready
                    |=> $stable(multicast_vector[lane]));
        end
    endgenerate
endmodule

`default_nettype wire
