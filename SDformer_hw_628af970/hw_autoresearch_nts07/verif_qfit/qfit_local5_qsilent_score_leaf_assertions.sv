`timescale 1ns/1ps
`default_nettype none

module qfit_local5_qsilent_score_leaf_assertions #(
    parameter bit ENABLE_QSILENT = 1'b1,
    parameter int TAG_W = 16,
    parameter int SCORE_W = 16,
    parameter int GATE_W = 9
) (
    input logic clk_core,
    input logic rst_core,
    input logic out_valid,
    input logic out_ready,
    input logic [TAG_W-1:0] out_tag,
    input logic [5*SCORE_W-1:0] out_score_q7,
    input logic [5*GATE_W-1:0] out_gate_q17,
    input logic take_qsilent,
    input logic take_identk,
    input logic take_fast,
    input logic in_valid,
    input logic in_ready,
    input logic [TAG_W-1:0] in_tag,
    input logic base_in_valid,
    input logic [31:0] in_q,
    input logic [5*32-1:0] in_k,
    input logic [4:0] in_valid_mask,
    input logic emit_fast,
    input logic emit_base,
    input logic issue_fire,
    input logic retire_fire,
    input logic retire_head,
    input logic [1:0] retire_count_q
);
    logic [TAG_W-1:0] expected_tag_q [0:1];
    logic expected_wr_q;
    logic expected_rd_q;
    logic [1:0] expected_count_q;
    property p_output_stable_under_backpressure;
        @(posedge clk_core) disable iff (rst_core)
            out_valid && !out_ready
            |=> out_valid
                && $stable(out_tag)
                && $stable(out_score_q7)
                && $stable(out_gate_q17);
    endproperty

    assert property (p_output_stable_under_backpressure);

    property p_fast_steals_base;
        @(posedge clk_core) disable iff (rst_core)
        in_valid && in_ready && take_fast |-> !base_in_valid;
    endproperty
    assert property (p_fast_steals_base);

    property p_identk_fire_all_k_equal;
        @(posedge clk_core) disable iff (rst_core)
        in_valid && in_ready && take_identk |-> ident_k_equal(in_k, in_valid_mask);
    endproperty
    assert property (p_identk_fire_all_k_equal);

    property p_identk_fire_qnz;
        @(posedge clk_core) disable iff (rst_core)
        in_valid && in_ready && take_identk |-> (in_q != 32'd0);
    endproperty
    assert property (p_identk_fire_qnz);

    // Leftover (Q!=0 and not all valid K identical) must stay on the residual leaf.
    property p_leftover_not_fast;
        @(posedge clk_core) disable iff (rst_core)
        in_valid && in_ready && (in_q != 32'd0)
            && !ident_k_equal(in_k, in_valid_mask)
            |-> !take_fast;
    endproperty
    assert property (p_leftover_not_fast);

    property p_emit_mutex;
        @(posedge clk_core) disable iff (rst_core)
        !(emit_fast && emit_base);
    endproperty
    assert property (p_emit_mutex);

    property p_fast_only_at_retire_head;
        @(posedge clk_core) disable iff (rst_core)
        emit_fast |-> (retire_count_q != 2'd0) && retire_head;
    endproperty
    assert property (p_fast_only_at_retire_head);

    property p_base_only_at_retire_head;
        @(posedge clk_core) disable iff (rst_core)
        emit_base |-> (retire_count_q != 2'd0) && !retire_head;
    endproperty
    assert property (p_base_only_at_retire_head);

    always_ff @(posedge clk_core) begin
        if (rst_core || !ENABLE_QSILENT) begin
            expected_tag_q[0] <= '0;
            expected_tag_q[1] <= '0;
            expected_wr_q <= 1'b0;
            expected_rd_q <= 1'b0;
            expected_count_q <= 2'd0;
        end else begin
            assert (expected_count_q == retire_count_q)
                else $error("Q-silent issue/retire count mismatch");
            if (retire_fire) begin
                assert (expected_count_q != 2'd0)
                    else $error("Q-silent retired with empty issue queue");
                assert (out_tag == expected_tag_q[expected_rd_q])
                    else $error("Q-silent issue order mismatch");
                expected_rd_q <= expected_rd_q + 1'b1;
            end
            if (issue_fire) begin
                assert (expected_count_q < 2'd2)
                    else $error("Q-silent issue queue overflow");
                expected_tag_q[expected_wr_q] <= in_tag;
                expected_wr_q <= expected_wr_q + 1'b1;
            end
            unique case ({issue_fire, retire_fire})
                2'b10: expected_count_q <= expected_count_q + 2'd1;
                2'b01: expected_count_q <= expected_count_q - 2'd1;
                default: ;
            endcase
        end
    end

    function automatic logic ident_k_equal(
        input logic [5*32-1:0] k_bus,
        input logic [4:0] valid
    );
        logic [31:0] ref_k;
        logic seen;
        seen = 1'b0;
        ref_k = '0;
        ident_k_equal = 1'b1;
        for (int cand = 0; cand < 5; cand = cand + 1) begin
            if (valid[cand]) begin
                if (!seen) begin
                    ref_k = k_bus[cand*32 +: 32];
                    seen = 1'b1;
                end else if (k_bus[cand*32 +: 32] != ref_k)
                    ident_k_equal = 1'b0;
            end
        end
        if (!seen)
            ident_k_equal = 1'b0;
    endfunction
endmodule

bind qfit_local5_qsilent_score_leaf qfit_local5_qsilent_score_leaf_assertions #(
    .ENABLE_QSILENT(ENABLE_QSILENT),
    .TAG_W(TAG_W),
    .SCORE_W(SCORE_W),
    .GATE_W(GATE_W)
) u_qfit_local5_qsilent_score_leaf_assertions (
    .clk_core(clk_core),
    .rst_core(rst_core),
    .out_valid(out_valid),
    .out_ready(out_ready),
    .out_tag(out_tag),
    .out_score_q7(out_score_q7),
    .out_gate_q17(out_gate_q17),
    .take_qsilent(take_qsilent),
    .take_identk(take_identk),
    .take_fast(take_fast),
    .in_valid(in_valid),
    .in_ready(in_ready),
    .in_tag(in_tag),
    .base_in_valid(base_in_valid),
    .in_q(in_q),
    .in_k(in_k),
    .in_valid_mask(in_valid_mask),
    .emit_fast(emit_fast),
    .emit_base(emit_base),
    .issue_fire(issue_fire),
    .retire_fire(retire_fire),
    .retire_head(retire_head),
    .retire_count_q(retire_count_q)
);

`default_nettype wire
