module m364_banked_q128_independent_hammer_assertions (
    input logic          core_clk,
    input logic          reset_n,
    input logic          cfg_valid,
    input logic          cfg_ready,
    input logic          cfg_active,
    input logic          cfg_protocol_error,
    input logic          in_valid,
    input logic          in_ready,
    input logic          out_valid,
    input logic          out_ready,
    input logic [15:0]   out_original_pattern,
    input logic [15:0]   out_best_center,
    input logic [6:0]    out_best_center_id,
    input logic [4:0]    out_best_distance,
    input logic [4:0]    out_population,
    input logic          out_use_pwp,
    input logic          out_fallback_bit_sparse,
    input logic [15:0]   out_plus_mask,
    input logic [15:0]   out_minus_mask,
    input logic          stage0_valid,
    input logic          stage1_valid,
    input logic          stage2_valid,
    input logic [3:0]    cfg_next_group,
    input logic [2047:0] catalog_flat
);

`ifdef SVA_RUNTIME_ENABLED
    logic pipeline_empty;
    assign pipeline_empty = !stage0_valid && !stage1_valid &&
                            !stage2_valid && !out_valid;

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

    ap_cfg_handshake_only_when_empty: assert property (
        @(posedge core_clk) disable iff (!reset_n)
        cfg_valid && cfg_ready |-> pipeline_empty);

    ap_nonempty_blocks_cfg_ready: assert property (
        @(posedge core_clk) disable iff (!reset_n)
        !pipeline_empty |-> !cfg_ready);

    ap_cfg_request_blocks_new_input: assert property (
        @(posedge core_clk) disable iff (!reset_n)
        cfg_valid |-> !in_ready);

    ap_error_quarantine: assert property (
        @(posedge core_clk) disable iff (!reset_n)
        cfg_protocol_error |-> !cfg_ready && !cfg_active && !in_ready);

    ap_error_sticky_and_internal_state_frozen: assert property (
        @(posedge core_clk) disable iff (!reset_n)
        cfg_protocol_error |=> cfg_protocol_error &&
        $stable({cfg_active, cfg_next_group, catalog_flat}));

    ap_error_pipeline_empty: assert property (
        @(posedge core_clk) disable iff (!reset_n)
        cfg_protocol_error |-> pipeline_empty);

    ap_output_stall_payload_stable: assert property (
        @(posedge core_clk) disable iff (!reset_n)
        out_valid && !out_ready |=> out_valid &&
        $stable({out_original_pattern, out_best_center,
                 out_best_center_id, out_best_distance, out_population,
                 out_use_pwp, out_fallback_bit_sparse,
                 out_plus_mask, out_minus_mask}));

    ap_output_distance_self_consistent: assert property (
        @(posedge core_clk) disable iff (!reset_n)
        out_valid |-> out_best_distance ==
            popcount16(out_original_pattern ^ out_best_center));

    ap_output_flag_partition: assert property (
        @(posedge core_clk) disable iff (!reset_n)
        out_valid |-> out_use_pwp ^ out_fallback_bit_sparse);

    ap_output_signed_masks: assert property (
        @(posedge core_clk) disable iff (!reset_n)
        out_valid && out_use_pwp |->
        out_plus_mask == (out_original_pattern & ~out_best_center) &&
        out_minus_mask == (out_best_center & ~out_original_pattern) &&
        !(|(out_plus_mask & out_minus_mask)));

    ap_output_fallback_masks: assert property (
        @(posedge core_clk) disable iff (!reset_n)
        out_valid && out_fallback_bit_sparse |->
        out_plus_mask == out_original_pattern && out_minus_mask == 0);

    ap_reset_clears_pipeline: assert property (
        @(posedge core_clk)
        !reset_n |-> !stage0_valid && !stage1_valid &&
        !stage2_valid && !out_valid && !cfg_active &&
        !cfg_protocol_error && cfg_next_group == 0);

    cp_all_four_elastic_slots_full: cover property (
        @(posedge core_clk) disable iff (!reset_n)
        stage0_valid && stage1_valid && stage2_valid && out_valid);

    cp_long_output_stall: cover property (
        @(posedge core_clk) disable iff (!reset_n)
        out_valid && !out_ready [*16]);

    cp_bubble_refill: cover property (
        @(posedge core_clk) disable iff (!reset_n)
        in_valid && in_ready ##1 !in_valid ##1 in_valid && in_ready);

    cp_deferred_configuration: cover property (
        @(posedge core_clk) disable iff (!reset_n)
        cfg_valid && !cfg_ready ##1 cfg_valid && !cfg_ready [*2]
        ##[1:32] cfg_valid && cfg_ready);

    cp_sticky_error_freeze: cover property (
        @(posedge core_clk) disable iff (!reset_n)
        cfg_protocol_error [*8]);
`endif

endmodule
