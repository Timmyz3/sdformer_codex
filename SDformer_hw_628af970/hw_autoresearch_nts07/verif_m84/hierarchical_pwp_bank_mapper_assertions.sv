`timescale 1ns/1ps
`default_nettype none

module hierarchical_pwp_bank_mapper_assertions #(
    parameter int OFFSET_W = 13,
    parameter int ROW_W = 10
) (
    input logic clk_core,
    input logic rst_core,
    input logic sample_valid,
    input logic descriptor_valid,
    input logic descriptor_escape,
    input logic [3:0] descriptor_width,
    input logic [2:0] descriptor_beats,
    input logic [OFFSET_W-1:0] start_word,
    input logic [2:0] beat_index,
    input logic beat_index_valid,
    input logic [8*ROW_W-1:0] bank_row_addresses
);
    ap_legal_width: assert property (@(posedge clk_core)
        disable iff (rst_core) sample_valid && descriptor_valid
        |-> descriptor_width inside {8, 9, 10, 11, 12});
    ap_escape_has_no_payload_beat: assert property (@(posedge clk_core)
        disable iff (rst_core) sample_valid && descriptor_escape
        |-> descriptor_width == 12 && descriptor_beats == 0
            && !beat_index_valid);
    ap_regular_beat_range: assert property (@(posedge clk_core)
        disable iff (rst_core) sample_valid && descriptor_valid
            && !descriptor_escape
        |-> beat_index_valid == (beat_index < descriptor_beats));
    ap_rows_are_adjacent: assert property (@(posedge clk_core)
        disable iff (rst_core) sample_valid && beat_index_valid
        |-> (bank_row_addresses[0*ROW_W +: ROW_W]
                == (start_word + beat_index*8) / 8
             || bank_row_addresses[0*ROW_W +: ROW_W]
                == (start_word + beat_index*8) / 8 + 1));
    ap_invalid_code_blocks_reads: assert property (@(posedge clk_core)
        disable iff (rst_core) sample_valid && !descriptor_valid
        |-> !beat_index_valid);

    cp_width8: cover property (@(posedge clk_core)
        sample_valid && descriptor_valid && descriptor_width == 8);
    cp_width9: cover property (@(posedge clk_core)
        sample_valid && descriptor_valid && descriptor_width == 9);
    cp_width10: cover property (@(posedge clk_core)
        sample_valid && descriptor_valid && descriptor_width == 10);
    cp_width11: cover property (@(posedge clk_core)
        sample_valid && descriptor_valid && descriptor_width == 11);
    cp_escape: cover property (@(posedge clk_core)
        sample_valid && descriptor_escape);
    cp_cross_row: cover property (@(posedge clk_core)
        sample_valid && beat_index_valid && start_word[2:0] != 0);
    cp_invalid_code: cover property (@(posedge clk_core)
        sample_valid && !descriptor_valid);
endmodule

`default_nettype wire
