`timescale 1ns/1ps
`default_nettype none

// Miter the Q-silent leaf against the sealed residual leaf.
module tb_qfit_local5_qsilent_score_leaf;
    localparam int TAG_W = 16;
    localparam int NVEC = 64;

    logic clk_core = 1'b0;
    logic rst_core;
    logic in_valid;
    logic in_ready_base;
    logic in_ready_fast;
    logic [TAG_W-1:0] in_tag;
    logic [31:0] in_q;
    logic [159:0] in_k;
    logic [4:0] in_valid_mask;
    logic out_ready;

    logic base_valid;
    logic fast_valid;
    logic [TAG_W-1:0] base_tag;
    logic [TAG_W-1:0] fast_tag;
    logic [79:0] base_score;
    logic [79:0] fast_score;
    logic [44:0] base_gate;
    logic [44:0] fast_gate;
    logic [31:0] base_kself;
    logic [31:0] fast_kself;
    logic [4:0] base_mask;
    logic [4:0] fast_mask;
    logic [15:0] base_service;
    logic [15:0] fast_service;
    logic [3:0] base_route;
    logic [3:0] fast_route;
    logic [31:0] fast_hits;
    logic [31:0] identk_hits;

    integer i;
    integer checked;
    integer base_cycles;
    integer fast_cycles;
    integer q0_rows;
    integer identk_rows;

    always #1 clk_core = ~clk_core;

    qfit_local5_score_leaf #(
        .ARCH_QFSA(1'b1),
        .PIPE_COMPACTOR(1'b1),
        .XBF_BANKED(1'b1),
        .USE_THRESHOLD_ROUTE(1'b1),
        .ROUTE_THRESHOLD(8),
        .USE_BANK_PRESSURE_ROUTE(1'b1),
        .BANK_PRESSURE_THRESHOLD(2)
    ) u_base (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .in_valid(in_valid && in_ready_base && in_ready_fast),
        .in_ready(in_ready_base),
        .in_tag(in_tag),
        .in_q(in_q),
        .in_k(in_k),
        .in_valid_mask(in_valid_mask),
        .out_valid(base_valid),
        .out_ready(out_ready && fast_valid),
        .out_tag(base_tag),
        .out_score_q7(base_score),
        .out_gate_q17(base_gate),
        .out_k_self(base_kself),
        .out_valid_mask(base_mask),
        .perf_service_cycles(base_service),
        .perf_route_direct_mask(base_route)
    );

    qfit_local5_qsilent_score_leaf #(
        .ENABLE_QSILENT(1'b1),
        .ARCH_QFSA(1'b1),
        .PIPE_COMPACTOR(1'b1),
        .XBF_BANKED(1'b1),
        .USE_THRESHOLD_ROUTE(1'b1),
        .ROUTE_THRESHOLD(8),
        .USE_BANK_PRESSURE_ROUTE(1'b1),
        .BANK_PRESSURE_THRESHOLD(2)
    ) u_fast (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .in_valid(in_valid && in_ready_base && in_ready_fast),
        .in_ready(in_ready_fast),
        .in_tag(in_tag),
        .in_q(in_q),
        .in_k(in_k),
        .in_valid_mask(in_valid_mask),
        .out_valid(fast_valid),
        .out_ready(out_ready && base_valid),
        .out_tag(fast_tag),
        .out_score_q7(fast_score),
        .out_gate_q17(fast_gate),
        .out_k_self(fast_kself),
        .out_valid_mask(fast_mask),
        .perf_service_cycles(fast_service),
        .perf_route_direct_mask(fast_route),
        .perf_qsilent_rows(fast_hits),
        .perf_identk_rows(identk_hits),
        .perf_overlap_accepts()
    );

    task automatic apply_vec(
        input integer tag,
        input logic [31:0] q_bits,
        input logic [159:0] k_bits,
        input logic [4:0] mask
    );
        begin
            @(negedge clk_core);
            in_tag = TAG_W'(tag);
            in_q = q_bits;
            in_k = k_bits;
            in_valid_mask = mask;
            in_valid = 1'b1;
            do @(posedge clk_core); while (!(in_ready_base && in_ready_fast));
            @(negedge clk_core);
            in_valid = 1'b0;
        end
    endtask

    initial begin
        rst_core = 1'b1;
        in_valid = 1'b0;
        out_ready = 1'b1;
        in_tag = '0;
        in_q = '0;
        in_k = '0;
        in_valid_mask = 5'b11111;
        checked = 0;
        base_cycles = 0;
        fast_cycles = 0;
        q0_rows = 0;
        identk_rows = 0;
        repeat (4) @(posedge clk_core);
        rst_core = 1'b0;

        // Q==0 uniform empty K.
        apply_vec(1, 32'd0, 160'd0, 5'b11111);
        q0_rows = q0_rows + 1;
        // Q==0 mixed popcounts.
        apply_vec(2, 32'd0, {32'h0000_00ff, 32'h0000_000f, 32'hffff_ffff,
            32'h0000_0001, 32'h0000_0000}, 5'b11101);
        q0_rows = q0_rows + 1;
        // Q!=0 residual path must still match.
        apply_vec(3, 32'hffff_0000, {32'h00ff_0000, 32'h00ff_00ff,
            32'hffff_0000, 32'h0000_ffff, 32'haaaa_5555}, 5'b11111);
        // Q==0 with holes in the valid mask.
        apply_vec(4, 32'd0, {32'h1, 32'h3, 32'h7, 32'hf, 32'h1f}, 5'b10101);
        q0_rows = q0_rows + 1;
        // Q!=0 identical-K broadcast (all five and with a hole).
        apply_vec(5, 32'h00ff_00ff,
            {32'h0f0f_0f0f, 32'h0f0f_0f0f, 32'h0f0f_0f0f, 32'h0f0f_0f0f, 32'h0f0f_0f0f},
            5'b11111);
        identk_rows = identk_rows + 1;
        apply_vec(6, 32'h1234_5678,
            {32'h1111_0000, 32'h1111_0000, 32'h0, 32'h1111_0000, 32'h1111_0000},
            5'b11011);
        identk_rows = identk_rows + 1;

        for (i = 0; i < NVEC; i = i + 1) begin
            logic [31:0] q_bits;
            logic [159:0] k_bits;
            logic [4:0] mask;
            q_bits = (i[1:0] == 2'b00) ? 32'd0 : (32'h1111_1111 << i[3:0]);
            k_bits = {32'(i*13), 32'(i*17+1), 32'(i*19), 32'(~i), 32'(i*i)};
            mask = 5'(i[4:0] | 5'b00001);
            apply_vec(16 + i, q_bits, k_bits, mask);
            if (q_bits == 32'd0)
                q0_rows = q0_rows + 1;
            else begin
                begin : ident_check
                    integer c;
                    integer seen;
                    logic [31:0] refk;
                    logic ok;
                    seen = 0;
                    ok = 1;
                    refk = 32'd0;
                    for (c = 0; c < 5; c = c + 1) begin
                        if (mask[c]) begin
                            if (seen == 0) begin
                                refk = k_bits[c*32 +: 32];
                                seen = 1;
                            end else if (k_bits[c*32 +: 32] != refk)
                                ok = 0;
                        end
                    end
                    if (seen && ok)
                        identk_rows = identk_rows + 1;
                end
            end
        end

        begin : wait_drain
            integer guard;
            guard = 0;
            while (checked < (6 + NVEC)) begin
                @(posedge clk_core);
                guard = guard + 1;
                if (guard > 20000)
                    $fatal(1, "qsilent miter drain timeout checked=%0d", checked);
            end
        end
        if (checked != (6 + NVEC))
            $fatal(1, "qsilent miter under-consumed checked=%0d", checked);
        if (fast_hits != 32'(q0_rows))
            $fatal(1, "qsilent hit count %0d != %0d", fast_hits, q0_rows);
        if (identk_hits != 32'(identk_rows))
            $fatal(1, "identk hit count %0d != %0d", identk_hits, identk_rows);
        $display("QSILENT_MITER checked=%0d q0_rows=%0d identk_rows=%0d base_cycles=%0d fast_cycles=%0d",
            checked, q0_rows, identk_rows, base_cycles, fast_cycles);
        $display("PASS tb_qfit_local5_qsilent_score_leaf");
        $finish;
    end

    always @(posedge clk_core) begin
        if (!rst_core) begin
            if (base_valid)
                base_cycles = base_cycles + 1;
            if (fast_valid)
                fast_cycles = fast_cycles + 1;
            if (base_valid && fast_valid && out_ready) begin
                if (base_tag !== fast_tag
                    || base_score !== fast_score
                    || base_gate !== fast_gate
                    || base_kself !== fast_kself
                    || base_mask !== fast_mask)
                    $fatal(1, "qsilent miter mismatch tag=%0d", base_tag);
                checked = checked + 1;
            end
        end
    end
endmodule

`default_nettype wire
