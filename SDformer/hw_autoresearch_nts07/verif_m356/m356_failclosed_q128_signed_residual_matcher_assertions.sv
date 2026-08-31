module m356_failclosed_q128_signed_residual_matcher_assertions (
    input logic        core_clk,
    input logic        reset_n,
    input logic        cfg_valid,
    input logic        cfg_ready,
    input logic        cfg_active,
    input logic        cfg_protocol_error,
    input logic        in_valid,
    input logic        in_ready,
    input logic        out_valid,
    input logic        out_ready,
    input logic [15:0] out_original_pattern,
    input logic [15:0] out_best_center,
    input logic [6:0]  out_best_center_id,
    input logic [4:0]  out_best_distance,
    input logic [4:0]  out_population,
    input logic        out_use_pwp,
    input logic        out_fallback_bit_sparse,
    input logic [15:0] out_plus_mask,
    input logic [15:0] out_minus_mask
);

`ifdef SVA_RUNTIME_ENABLED
    function automatic logic [4:0] popcount16(input logic [15:0] value);
        integer bit_index;
        logic [4:0] count;
        begin
            count = '0;
            for (bit_index = 0; bit_index < 16; bit_index = bit_index + 1)
                count = count + value[bit_index];
            popcount16 = count;
        end
    endfunction

    ap_protocol_error_fail_closed: assert property (
        @(posedge core_clk) disable iff (!reset_n)
        cfg_protocol_error |-> !cfg_ready && !cfg_active && !in_ready);

    ap_protocol_error_sticky: assert property (
        @(posedge core_clk) disable iff (!reset_n)
        cfg_protocol_error |=> cfg_protocol_error);

    ap_protocol_error_blocks_configuration: assert property (
        @(posedge core_clk) disable iff (!reset_n)
        cfg_protocol_error && cfg_valid |-> !cfg_ready);

    ap_input_requires_active: assert property (
        @(posedge core_clk) disable iff (!reset_n)
        in_valid && in_ready |-> cfg_active && !cfg_protocol_error && !cfg_valid);

    ap_stall_stable: assert property (
        @(posedge core_clk) disable iff (!reset_n)
        out_valid && !out_ready |=> out_valid &&
        $stable({out_original_pattern, out_best_center,
                 out_best_center_id, out_best_distance, out_population,
                 out_use_pwp, out_fallback_bit_sparse,
                 out_plus_mask, out_minus_mask}));

    ap_flag_partition: assert property (
        @(posedge core_clk) disable iff (!reset_n)
        out_valid |-> out_use_pwp ^ out_fallback_bit_sparse);

    ap_distance_matches_payload: assert property (
        @(posedge core_clk) disable iff (!reset_n)
        out_valid |-> out_best_distance ==
            popcount16(out_original_pattern ^ out_best_center));

    ap_use_threshold: assert property (
        @(posedge core_clk) disable iff (!reset_n)
        out_valid && out_use_pwp |->
            ({1'b0, out_best_distance} + 6'd1) < {1'b0, out_population});

    ap_fallback_threshold: assert property (
        @(posedge core_clk) disable iff (!reset_n)
        out_valid && out_fallback_bit_sparse |->
            ({1'b0, out_best_distance} + 6'd1) >= {1'b0, out_population});

    ap_signed_residual_masks: assert property (
        @(posedge core_clk) disable iff (!reset_n)
        out_valid && out_use_pwp |->
            out_plus_mask == (out_original_pattern & ~out_best_center) &&
            out_minus_mask == (out_best_center & ~out_original_pattern) &&
            (out_plus_mask & out_minus_mask) == 0);

    ap_fallback_masks: assert property (
        @(posedge core_clk) disable iff (!reset_n)
        out_valid && out_fallback_bit_sparse |->
            out_plus_mask == out_original_pattern && out_minus_mask == 0);

    cp_config: cover property (@(posedge core_clk) disable iff (!reset_n)
        cfg_valid && cfg_ready ##1 cfg_active);
    cp_use_pwp: cover property (@(posedge core_clk) disable iff (!reset_n)
        out_valid && out_ready && out_use_pwp);
    cp_fallback: cover property (@(posedge core_clk) disable iff (!reset_n)
        out_valid && out_ready && out_fallback_bit_sparse);
    cp_positive_signed_residual: cover property (
        @(posedge core_clk) disable iff (!reset_n)
        out_valid && out_ready && out_use_pwp &&
        out_plus_mask != 0 && out_minus_mask != 0);
    cp_output_stall: cover property (@(posedge core_clk) disable iff (!reset_n)
        out_valid && !out_ready ##1 out_valid && out_ready);
`endif

endmodule
