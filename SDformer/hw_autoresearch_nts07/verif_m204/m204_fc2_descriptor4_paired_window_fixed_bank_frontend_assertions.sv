`timescale 1ns/1ps
`default_nettype none

module m204_fc2_descriptor4_paired_window_fixed_bank_frontend_assertions #(
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12
) (
    input logic clk_core, input logic rst_core,
    input logic descriptor_valid, input logic descriptor_ready,
    input logic descriptor_accept, input logic [2:0] descriptor_count,
    input logic [3:0] descriptor_window_last,
    input logic group_valid, input logic group_ready, input logic group_accept,
    input logic [TAG_BITS-1:0] group_tag,
    input logic [2:0] group_output_block,
    input logic [3:0] group_source_count,
    input logic [7:0] group_bank_valid,
    input logic [CHANNEL_BITS-1:0] group_source_channel [0:7],
    input logic token_done_accept, input logic protocol_error,
    input logic pair_has_two, input logic pair_available,
    input logic upstream_done_accept
);
`ifdef SVA_RUNTIME_ENABLED
    function automatic logic [3:0] popcount8(input logic [7:0] value);
        logic [3:0] count;
        begin
            count = 0;
            for (int bit_index = 0; bit_index < 8; bit_index++)
                count = count + value[bit_index];
            return count;
        end
    endfunction
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_group_accept_definition:
        assert property (group_accept == (group_valid && group_ready));
    ap_descriptor_accept_definition:
        assert property (descriptor_accept
            == (descriptor_valid && descriptor_ready));
    ap_group_mask_count:
        assert property (group_valid
            |-> group_source_count == popcount8(group_bank_valid));
    ap_group_nonempty:
        assert property (group_valid |-> group_bank_valid != 0);
    ap_group_stable_under_stall:
        assert property (group_valid && !group_ready
            |=> group_valid && $stable(group_tag)
                && $stable(group_output_block)
                && $stable(group_source_count)
                && $stable(group_bank_valid)
                && $stable(group_source_channel));
    ap_fault_sticky:
        assert property ($past(protocol_error) |-> protocol_error);
    ap_descriptor_last_is_final_lane:
        assert property (descriptor_accept && descriptor_window_last != 0
            |-> (descriptor_count == 1 && descriptor_window_last == 4'b0001)
             || (descriptor_count == 2 && descriptor_window_last == 4'b0010)
             || (descriptor_count == 3 && descriptor_window_last == 4'b0100)
             || (descriptor_count == 4 && descriptor_window_last == 4'b1000));

    cp_descriptor4: cover property (descriptor_accept && descriptor_count == 4);
    cp_paired_window: cover property (pair_available && pair_has_two);
    cp_odd_tail: cover property (pair_available && !pair_has_two);
    cp_group_stall: cover property (group_valid && !group_ready);
    cp_group_accept: cover property (group_accept);
    cp_upstream_done: cover property (upstream_done_accept);
    cp_token_done: cover property (token_done_accept);
    cp_protocol_attack: cover property (protocol_error);
`endif
endmodule

bind m204_fc2_descriptor4_paired_window_fixed_bank_frontend
    m204_fc2_descriptor4_paired_window_fixed_bank_frontend_assertions sva (.*);

`default_nettype wire
