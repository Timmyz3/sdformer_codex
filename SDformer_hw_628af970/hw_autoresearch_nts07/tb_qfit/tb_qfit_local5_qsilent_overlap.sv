`timescale 1ns/1ps
`default_nettype none

// Residual then Q==0 must stay in-order, and the silent dest may be accepted
// while the XOR walk is still busy.
module tb_qfit_local5_qsilent_overlap;
    logic clk = 1'b0;
    logic rst;
    logic in_valid;
    logic in_ready;
    logic [15:0] in_tag;
    logic [31:0] in_q;
    logic [159:0] in_k;
    logic [4:0] in_mask;
    logic out_valid;
    logic out_ready;
    logic [15:0] out_tag;
    logic [79:0] out_score;
    logic [31:0] overlap;
    integer got;
    integer wait_c;

    always #1 clk = ~clk;

    qfit_local5_qsilent_score_leaf #(
        .ENABLE_QSILENT(1'b1),
        .ENABLE_IDENTK(1'b1),
        .ARCH_QFSA(1'b1),
        .PIPE_COMPACTOR(1'b1),
        .XBF_BANKED(1'b1),
        .USE_THRESHOLD_ROUTE(1'b1),
        .ROUTE_THRESHOLD(8),
        .USE_BANK_PRESSURE_ROUTE(1'b1),
        .BANK_PRESSURE_THRESHOLD(2)
    ) dut (
        .clk_core(clk),
        .rst_core(rst),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_tag(in_tag),
        .in_q(in_q),
        .in_k(in_k),
        .in_valid_mask(in_mask),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_tag(out_tag),
        .out_score_q7(out_score),
        .out_gate_q17(),
        .out_k_self(),
        .out_valid_mask(),
        .perf_service_cycles(),
        .perf_route_direct_mask(),
        .perf_qsilent_rows(),
        .perf_identk_rows(),
        .perf_overlap_accepts(overlap)
    );

    initial begin
        rst = 1'b1;
        in_valid = 1'b0;
        out_ready = 1'b0;
        in_tag = '0;
        in_q = '0;
        in_k = '0;
        in_mask = 5'b11111;
        got = 0;
        repeat (4) @(negedge clk);
        rst = 1'b0;

        // Leftover residual: Q!=0 and not all K equal.
        in_tag = 16'd1;
        in_q = 32'hffff_0000;
        in_k = {32'h00ff_0000, 32'h00ff_00ff, 32'hffff_0000,
            32'h0000_ffff, 32'haaaa_5555};
        in_mask = 5'b11111;
        in_valid = 1'b1;
        wait_c = 0;
        @(posedge clk);
        while (!in_ready && wait_c < 8000) begin
            wait_c = wait_c + 1;
            @(posedge clk);
        end
        if (!in_ready)
            $fatal(1, "residual accept timeout");
        @(negedge clk);
        // Next dest is Q==0; accept it while residual is still walking.
        in_tag = 16'd2;
        in_q = 32'd0;
        in_k = {32'h1, 32'h3, 32'h7, 32'hf, 32'h1f};
        wait_c = 0;
        @(posedge clk);
        while (!in_ready && wait_c < 200) begin
            wait_c = wait_c + 1;
            @(posedge clk);
        end
        if (!in_ready)
            $fatal(1, "silent dest blocked by residual walk");
        @(negedge clk);
        in_valid = 1'b0;
        out_ready = 1'b1;
        wait_c = 0;
        while (got < 2 && wait_c < 8000) begin
            @(posedge clk);
            wait_c = wait_c + 1;
            if (out_valid) begin
                got = got + 1;
                if (got == 1 && out_tag != 16'd1)
                    $fatal(1, "order broke first tag=%0d", out_tag);
                if (got == 2 && out_tag != 16'd2)
                    $fatal(1, "order broke second tag=%0d", out_tag);
            end
        end
        if (got != 2)
            $fatal(1, "under-consumed %0d", got);
        if (overlap == 0)
            $fatal(1, "overlap accept did not fire");
        $display("QSILENT_OVERLAP order=ok overlap=%0d", overlap);
        $display("PASS tb_qfit_local5_qsilent_overlap");
        $finish;
    end
endmodule

`default_nettype wire
