module m311_near_match16_tau01_assertions (
    input logic        core_clk,
    input logic        reset_n,
    input logic        in_ready,
    input logic        out_valid,
    input logic        out_ready,
    input logic [15:0] out_original_pattern,
    input logic [15:0] out_selected_pattern,
    input logic [4:0]  out_selected_distance,
    input logic [4:0]  out_population,
    input logic [1:0]  out_tau,
    input logic        out_snapped,
    input logic        out_exact_hit,
    input logic        out_positive_distance
);

`ifdef SVA_RUNTIME_ENABLED
    ap_ready_equation: assert property (@(posedge core_clk) disable iff (!reset_n)
        in_ready == (!out_valid || out_ready));
    ap_stall_stable: assert property (@(posedge core_clk) disable iff (!reset_n)
        out_valid && !out_ready |=> out_valid &&
        $stable({out_original_pattern, out_selected_pattern,
                 out_selected_distance, out_population, out_tau,
                 out_snapped, out_exact_hit, out_positive_distance}));
    ap_flag_partition: assert property (@(posedge core_clk) disable iff (!reset_n)
        out_valid |-> (out_snapped ==
                       (out_exact_hit || out_positive_distance)) &&
                      !(out_exact_hit && out_positive_distance));
    ap_exact_semantics: assert property (@(posedge core_clk) disable iff (!reset_n)
        out_valid && out_exact_hit |-> out_selected_distance == 0 &&
        out_selected_pattern == out_original_pattern);
    ap_positive_semantics: assert property (@(posedge core_clk) disable iff (!reset_n)
        out_valid && out_positive_distance |-> out_selected_distance > 0 &&
        out_selected_distance <= out_tau);
    ap_guard_semantics: assert property (@(posedge core_clk) disable iff (!reset_n)
        out_valid && out_snapped |-> out_population >= 2);
    ap_tau0_exact_subset: assert property (@(posedge core_clk) disable iff (!reset_n)
        out_valid && out_tau == 0 |-> !out_positive_distance);
    ap_unsnapped_identity: assert property (@(posedge core_clk) disable iff (!reset_n)
        out_valid && !out_snapped |->
        out_selected_pattern == out_original_pattern);

    cp_stall: cover property (@(posedge core_clk) disable iff (!reset_n)
        out_valid && !out_ready ##1 out_valid && out_ready);
    cp_exact: cover property (@(posedge core_clk) disable iff (!reset_n)
        out_valid && out_ready && out_exact_hit);
    cp_positive: cover property (@(posedge core_clk) disable iff (!reset_n)
        out_valid && out_ready && out_positive_distance);
    cp_tau0: cover property (@(posedge core_clk) disable iff (!reset_n)
        out_valid && out_ready && out_tau == 0);
    cp_guard: cover property (@(posedge core_clk) disable iff (!reset_n)
        out_valid && out_ready && out_population < 2 && !out_snapped);
    cp_distance_reject: cover property (@(posedge core_clk) disable iff (!reset_n)
        out_valid && out_ready && out_population >= 2 &&
        out_selected_distance > out_tau && !out_snapped);
`endif

endmodule
