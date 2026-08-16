`timescale 1ns/1ps
`default_nettype none

// Mixed residual/Q-silent/ident-K traffic with randomized output
// backpressure. Monotonic tags form an independent issue-order scoreboard.
module tb_qfit_local5_qsilent_random_backpressure;
    localparam int TAG_W = 16;
    localparam int TRANSACTIONS = 512;

    logic clk_core = 1'b0;
    logic rst_core;
    logic in_valid;
    logic in_ready;
    logic [TAG_W-1:0] in_tag;
    logic [31:0] in_q;
    logic [5*32-1:0] in_k;
    logic [4:0] in_valid_mask;
    logic out_valid;
    logic out_ready;
    logic [TAG_W-1:0] out_tag;
    logic [5*16-1:0] out_score_q7;
    logic [5*9-1:0] out_gate_q17;
    logic [31:0] out_k_self;
    logic [4:0] out_valid_mask;
    logic [31:0] overlap_accepts;
    logic [15:0] ready_lfsr_q;

    integer seed;
    integer issued;
    integer retired;
    integer stalled_outputs;
    integer cycles;

    always #1 clk_core = ~clk_core;

    always_comb begin
        in_tag = TAG_W'(issued);
        in_valid_mask = 5'b11111;
        unique case (issued % 3)
            0: begin
                in_q = 32'hffff_0000 ^ 32'(issued * 17);
                in_k = {
                    32'h00ff_0000 ^ 32'(issued * 3),
                    32'h00ff_00ff ^ 32'(issued * 5),
                    32'hffff_0000 ^ 32'(issued * 7),
                    32'h0000_ffff ^ 32'(issued * 11),
                    32'haaaa_5555 ^ 32'(issued * 13)
                };
            end
            1: begin
                in_q = 32'd0;
                in_k = {
                    32'h0000_00ff ^ 32'(issued),
                    32'h0000_000f ^ 32'(issued * 3),
                    32'hffff_ffff ^ 32'(issued * 5),
                    32'h0000_0001 ^ 32'(issued * 7),
                    32'h0000_0000 ^ 32'(issued * 11)
                };
            end
            default: begin
                in_q = 32'h1234_5678 ^ 32'(issued * 29);
                in_k = {5{32'h0f0f_0f0f ^ 32'(issued * 31)}};
            end
        endcase
    end

    assign in_valid = !rst_core && (issued < TRANSACTIONS);
    assign out_ready = ready_lfsr_q[0] || ready_lfsr_q[3];

    qfit_local5_qsilent_score_leaf #(
        .ENABLE_QSILENT(1'b1),
        .ENABLE_IDENTK(1'b1),
        .ENABLE_OVERLAP(1'b1),
        .ARCH_QFSA(1'b1),
        .PIPE_COMPACTOR(1'b1),
        .XBF_BANKED(1'b1),
        .USE_THRESHOLD_ROUTE(1'b1),
        .ROUTE_THRESHOLD(8),
        .USE_BANK_PRESSURE_ROUTE(1'b1),
        .BANK_PRESSURE_THRESHOLD(2),
        .TAG_W(TAG_W)
    ) dut (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_tag(in_tag),
        .in_q(in_q),
        .in_k(in_k),
        .in_valid_mask(in_valid_mask),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_tag(out_tag),
        .out_score_q7(out_score_q7),
        .out_gate_q17(out_gate_q17),
        .out_k_self(out_k_self),
        .out_valid_mask(out_valid_mask),
        .perf_service_cycles(),
        .perf_route_direct_mask(),
        .perf_qsilent_rows(),
        .perf_identk_rows(),
        .perf_overlap_accepts(overlap_accepts)
    );

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            ready_lfsr_q <= 16'(seed);
            issued <= 0;
            retired <= 0;
            stalled_outputs <= 0;
            cycles <= 0;
        end else begin
            ready_lfsr_q <= {
                ready_lfsr_q[14:0],
                ready_lfsr_q[15] ^ ready_lfsr_q[13]
                    ^ ready_lfsr_q[12] ^ ready_lfsr_q[10]
            };
            cycles <= cycles + 1;
            if (in_valid && in_ready)
                issued <= issued + 1;
            if (out_valid && !out_ready)
                stalled_outputs <= stalled_outputs + 1;
            if (out_valid && out_ready) begin
                if (out_tag !== TAG_W'(retired))
                    $fatal(1,
                        "random backpressure order mismatch got=%0d expected=%0d",
                        out_tag, retired);
                retired <= retired + 1;
            end
        end
    end

    initial begin
        seed = 16'h1;
        void'($value$plusargs("SEED=%d", seed));
        if (seed == 0)
            $fatal(1, "SEED must be nonzero");
        rst_core = 1'b1;
        repeat (5) @(negedge clk_core);
        rst_core = 1'b0;

        wait (retired == TRANSACTIONS);
        @(negedge clk_core);
        if (issued != TRANSACTIONS)
            $fatal(1, "random backpressure issue count=%0d", issued);
        if (stalled_outputs == 0)
            $fatal(1, "random backpressure generated no output stall");
        if (overlap_accepts == 0)
            $fatal(1, "random backpressure exercised no overlap");
        $display(
            "QSILENT_RANDOM_BP seed=%0d issued=%0d retired=%0d stalls=%0d overlap=%0d cycles=%0d",
            seed, issued, retired, stalled_outputs, overlap_accepts, cycles
        );
        $display("PASS tb_qfit_local5_qsilent_random_backpressure");
        $finish;
    end

    initial begin
        repeat (2_000_000) @(posedge clk_core);
        $fatal(1, "random backpressure timeout issued=%0d retired=%0d", issued, retired);
    end
endmodule

`default_nettype wire
