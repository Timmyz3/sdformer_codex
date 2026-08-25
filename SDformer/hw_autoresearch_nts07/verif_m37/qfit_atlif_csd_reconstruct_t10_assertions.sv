`timescale 1ns/1ps
`default_nettype none

module qfit_atlif_csd_reconstruct_t10_assertions #(
    parameter int TAG_W = 48,
    parameter int FIFO_DEPTH = 16,
    localparam int FIFO_COUNT_W = $clog2(FIFO_DEPTH+1)
) (
    input logic clk_core,
    input logic rst_core,
    input logic config_valid,
    input logic config_ready,
    input logic [(30*8)-1:0] config_left_factor,
    input logic [(30*4)-1:0] config_term_valid,
    input logic [(30*4)-1:0] config_term_negative,
    input logic [(30*4*3)-1:0] config_term_shift,
    input logic [(10*24)-1:0] config_bias,
    input logic signed [23:0] config_threshold,
    input logic descriptor_legal,
    input logic config_loaded,
    input logic config_release_valid,
    input logic config_release_ready,
    input logic input_valid,
    input logic input_ready,
    input logic [TAG_W-1:0] input_tag,
    input logic [(48*8)-1:0] input_intermediate,
    input logic result_valid,
    input logic result_ready,
    input logic [TAG_W-1:0] result_tag,
    input logic [2:0] result_beat,
    input logic [47:0] result_valid_bits,
    input logic [47:0] result_bits,
    input logic done,
    input logic [TAG_W-1:0] done_tag,
    input logic protocol_error,
    input logic busy,
    input logic arithmetic_active,
    input logic [2:0] phase_cycle,
    input logic phase4_chain_accept,
    input logic [1:0] input_buffer_occupancy,
    input logic [FIFO_COUNT_W-1:0] result_fifo_occupancy,
    input logic result_fifo_push,
    input logic result_fifo_pop,
    input logic [2:0] result_fifo_push_beat,
    input logic [TAG_W-1:0] result_fifo_push_tag,
    input logic input_accept,
    input logic input_accept_bank,
    input logic active_compute_bank,
    input logic [TAG_W-1:0] arithmetic_tag,
    input logic uses_integer_multiplier
);
    logic [1:0] owned_bank_valid_q;
    logic [TAG_W-1:0] owned_bank_tag_q [0:1];
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    initial $display("M37_SVA_BOUND=1");

    assert property (!uses_integer_multiplier);
    assert property (config_valid && !config_ready |=>
        $stable({config_valid, config_left_factor, config_term_valid,
                 config_term_negative, config_term_shift, config_bias,
                 config_threshold}));
    assert property (config_valid && config_ready && descriptor_legal
        |=> config_loaded && !protocol_error);
    assert property (config_valid && config_ready && !descriptor_legal
        |=> protocol_error && !config_loaded);
    assert property (config_release_valid && !config_release_ready
        |=> $stable(config_release_valid));
    assert property (config_release_ready |-> !busy && !input_valid);

    assert property (input_valid && !input_ready |=>
        $stable({input_valid, input_tag, input_intermediate}));
    assert property (input_valid && input_ready |-> config_loaded);
    assert property (input_buffer_occupancy <= 2);

    assert property (result_valid && !result_ready |=>
        $stable({result_valid, result_tag, result_beat,
                 result_valid_bits, result_bits}));
    assert property (result_valid |-> result_beat <= 4
        && result_valid_bits == {{16{1'b0}}, {32{1'b1}}}
        && result_bits[47:32] == 0);
    assert property (result_fifo_occupancy <= FIFO_DEPTH);
    assert property (arithmetic_active |-> phase_cycle <= 4);
    assert property (phase4_chain_accept |-> arithmetic_active
        && phase_cycle == 4 && input_valid && input_ready);
    assert property (done |-> !$isunknown(done_tag));
    assert property (done |-> $past(result_fifo_push
        && result_fifo_push_beat == 4)
        && done_tag == $past(result_fifo_push_tag));
    assert property (result_fifo_push && result_fifo_push_beat == 4
        |=> done && done_tag == $past(result_fifo_push_tag));
    assert property ($past(!rst_core)
        && $past(result_fifo_push && !result_fifo_pop)
        |-> result_fifo_occupancy == $past(result_fifo_occupancy)+1'b1);
    assert property ($past(!rst_core)
        && $past(!result_fifo_push && result_fifo_pop)
        |-> result_fifo_occupancy == $past(result_fifo_occupancy)-1'b1);
    assert property ($past(!rst_core)
        && $past(result_fifo_push == result_fifo_pop)
        |-> result_fifo_occupancy == $past(result_fifo_occupancy));
    assert property (protocol_error |=> protocol_error);

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            owned_bank_valid_q <= '0;
            owned_bank_tag_q[0] <= '0;
            owned_bank_tag_q[1] <= '0;
        end else begin
            if (arithmetic_active) begin
                assert (owned_bank_valid_q[active_compute_bank])
                    else $error("M37 active bank lacks accepted owner");
                assert (arithmetic_tag
                        == owned_bank_tag_q[active_compute_bank])
                    else $error("M37 active bank tag ownership mismatch");
                if (phase_cycle == 4)
                    owned_bank_valid_q[active_compute_bank] <= 1'b0;
            end
            if (input_accept) begin
                owned_bank_valid_q[input_accept_bank] <= 1'b1;
                owned_bank_tag_q[input_accept_bank] <= input_tag;
            end
        end
    end

    cover property (phase4_chain_accept);
    cover property (input_buffer_occupancy == 2);
    cover property (result_fifo_occupancy == FIFO_DEPTH);
    cover property ((result_valid && !result_ready)[*8]);
    cover property (result_fifo_occupancy == FIFO_DEPTH
        && result_valid && result_ready);
    cover property (config_valid && config_ready && !descriptor_legal
        ##1 protocol_error && !config_loaded);
    cover property (done && $past(result_fifo_push
        && result_fifo_push_beat == 4));
    cover property (config_release_valid && !config_release_ready
        && busy && input_valid && result_fifo_occupancy != 0);
endmodule

bind qfit_atlif_csd_reconstruct_t10
    qfit_atlif_csd_reconstruct_t10_assertions #(
        .TAG_W(TAG_W), .FIFO_DEPTH(FIFO_DEPTH)
    ) m37_assertions (.*);

`default_nettype wire
