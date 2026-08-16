`timescale 1ns/1ps
`default_nettype none

module tb_qfit_local5_score_leaf;

    logic clk_core;
    logic rst_core;
    logic in_valid;
    logic [15:0] in_tag;
    logic [31:0] in_q;
    logic [159:0] in_k;
    logic [4:0] in_valid_mask;
    logic out_ready;

    logic ready_w1;
    logic ready_q1;
    logic ready_q2;
    logic ready_xb;
    logic ready_xbp;
    logic ready_phase;
    logic valid_w1;
    logic valid_q1;
    logic valid_q2;
    logic valid_xb;
    logic valid_xbp;
    logic valid_phase;
    logic [15:0] tag_w1;
    logic [15:0] tag_q1;
    logic [15:0] tag_q2;
    logic [15:0] tag_xb;
    logic [15:0] tag_xbp;
    logic [15:0] tag_phase;
    logic [79:0] score_w1;
    logic [79:0] score_q1;
    logic [79:0] score_q2;
    logic [79:0] score_xb;
    logic [79:0] score_xbp;
    logic [79:0] score_phase;
    logic [44:0] gate_w1;
    logic [44:0] gate_q1;
    logic [44:0] gate_q2;
    logic [44:0] gate_xb;
    logic [44:0] gate_xbp;
    logic [44:0] gate_phase;
    logic [15:0] cycles_w1;
    logic [15:0] cycles_q1;
    logic [15:0] cycles_q2;
    logic [15:0] cycles_xb;
    logic [15:0] cycles_xbp;
    logic [15:0] cycles_phase;
    logic [3:0] route_w1;
    logic [3:0] route_q1;
    logic [3:0] route_q2;
    logic [3:0] route_xb;
    logic [3:0] route_xbp;
    logic [3:0] route_phase;

    logic [31:0] ref_k [0:4];
    logic [79:0] ref_score;
    logic [44:0] ref_gate;

    always #5 clk_core = ~clk_core;

    always_comb begin
        for (int cand = 0; cand < 5; cand = cand + 1)
            ref_k[cand] = in_k[cand*32 +: 32];
    end

    local5_stencil_token u_ref (
        .q_bits(in_q),
        .k_bits(ref_k),
        .valid(in_valid_mask),
        .score_q7(ref_score),
        .gate_q17(ref_gate)
    );

    qfit_local5_score_leaf #(
        .ARCH_QFSA(1'b0),
        .PIPE_COMPACTOR(1'b0)
    ) u_w1 (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .in_valid(in_valid),
        .in_ready(ready_w1),
        .in_tag(in_tag),
        .in_q(in_q),
        .in_k(in_k),
        .in_valid_mask(in_valid_mask),
        .out_valid(valid_w1),
        .out_ready(out_ready),
        .out_tag(tag_w1),
        .out_score_q7(score_w1),
        .out_gate_q17(gate_w1),
        .perf_service_cycles(cycles_w1),
        .perf_route_direct_mask(route_w1)
    );

    qfit_local5_score_leaf #(
        .ARCH_QFSA(1'b1),
        .PIPE_COMPACTOR(1'b0)
    ) u_q1 (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .in_valid(in_valid),
        .in_ready(ready_q1),
        .in_tag(in_tag),
        .in_q(in_q),
        .in_k(in_k),
        .in_valid_mask(in_valid_mask),
        .out_valid(valid_q1),
        .out_ready(out_ready),
        .out_tag(tag_q1),
        .out_score_q7(score_q1),
        .out_gate_q17(gate_q1),
        .perf_service_cycles(cycles_q1),
        .perf_route_direct_mask(route_q1)
    );

    qfit_local5_score_leaf #(
        .ARCH_QFSA(1'b1),
        .PIPE_COMPACTOR(1'b1)
    ) u_q2 (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .in_valid(in_valid),
        .in_ready(ready_q2),
        .in_tag(in_tag),
        .in_q(in_q),
        .in_k(in_k),
        .in_valid_mask(in_valid_mask),
        .out_valid(valid_q2),
        .out_ready(out_ready),
        .out_tag(tag_q2),
        .out_score_q7(score_q2),
        .out_gate_q17(gate_q2),
        .perf_service_cycles(cycles_q2),
        .perf_route_direct_mask(route_q2)
    );

    qfit_local5_score_leaf #(
        .ARCH_QFSA(1'b1),
        .PIPE_COMPACTOR(1'b1),
        .XBF_BANKED(1'b1),
        .USE_THRESHOLD_ROUTE(1'b1),
        .ROUTE_THRESHOLD(8)
    ) u_xb (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .in_valid(in_valid),
        .in_ready(ready_xb),
        .in_tag(in_tag),
        .in_q(in_q),
        .in_k(in_k),
        .in_valid_mask(in_valid_mask),
        .out_valid(valid_xb),
        .out_ready(out_ready),
        .out_tag(tag_xb),
        .out_score_q7(score_xb),
        .out_gate_q17(gate_xb),
        .perf_service_cycles(cycles_xb),
        .perf_route_direct_mask(route_xb)
    );

    qfit_local5_score_leaf #(
        .ARCH_QFSA(1'b1),
        .PIPE_COMPACTOR(1'b1),
        .XBF_BANKED(1'b1),
        .USE_THRESHOLD_ROUTE(1'b1),
        .ROUTE_THRESHOLD(8),
        .USE_BANK_PRESSURE_ROUTE(1'b1),
        .BANK_PRESSURE_THRESHOLD(2)
    ) u_xbp (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .in_valid(in_valid),
        .in_ready(ready_xbp),
        .in_tag(in_tag),
        .in_q(in_q),
        .in_k(in_k),
        .in_valid_mask(in_valid_mask),
        .out_valid(valid_xbp),
        .out_ready(out_ready),
        .out_tag(tag_xbp),
        .out_score_q7(score_xbp),
        .out_gate_q17(gate_xbp),
        .perf_service_cycles(cycles_xbp),
        .perf_route_direct_mask(route_xbp)
    );

    qfit_local5_score_leaf #(
        .ARCH_QFSA(1'b1),
        .PIPE_COMPACTOR(1'b1),
        .XBF_BANKED(1'b1),
        .ARCH_PHASE_RESIDUAL(1'b1),
        .USE_THRESHOLD_ROUTE(1'b1),
        .ROUTE_THRESHOLD(8),
        .USE_BANK_PRESSURE_ROUTE(1'b1),
        .BANK_PRESSURE_THRESHOLD(2)
    ) u_phase (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .in_valid(in_valid),
        .in_ready(ready_phase),
        .in_tag(in_tag),
        .in_q(in_q),
        .in_k(in_k),
        .in_valid_mask(in_valid_mask),
        .out_valid(valid_phase),
        .out_ready(out_ready),
        .out_tag(tag_phase),
        .out_score_q7(score_phase),
        .out_gate_q17(gate_phase),
        .perf_service_cycles(cycles_phase),
        .perf_route_direct_mask(route_phase)
    );

    task automatic check_outputs;
        integer ref_delta;
        integer phase_delta;
        begin
            if (
                tag_w1 !== in_tag
                || tag_q1 !== in_tag
                || tag_q2 !== in_tag
                || tag_xb !== in_tag
                || tag_xbp !== in_tag
                || tag_phase !== in_tag
            )
                $fatal(1, "tag mismatch");
            for (int cand = 0; cand < 5; cand = cand + 1) begin
                if (in_valid_mask[cand]) begin
                    if (
                        score_w1[cand*16 +: 16]
                        !== ref_score[cand*16 +: 16]
                    )
                        $fatal(1, "4xW1 score mismatch cand=%0d", cand);
                    if (
                        score_q1[cand*16 +: 16]
                        !== ref_score[cand*16 +: 16]
                    )
                        $fatal(1, "QFSA-1C score mismatch cand=%0d", cand);
                    if (
                        score_q2[cand*16 +: 16]
                        !== ref_score[cand*16 +: 16]
                    )
                        $fatal(1, "QFSA-2C score mismatch cand=%0d", cand);
                    if (
                        score_xb[cand*16 +: 16]
                        !== ref_score[cand*16 +: 16]
                    )
                        $fatal(1, "XBF-QFSA score mismatch cand=%0d", cand);
                    if (
                        score_xbp[cand*16 +: 16]
                        !== ref_score[cand*16 +: 16]
                    )
                        $fatal(
                            1,
                            "XBF-DBDR score mismatch cand=%0d",
                            cand
                        );
                    ref_delta = $signed(ref_score[cand*16 +: 16])
                              - $signed(ref_score[0 +: 16]);
                    phase_delta = $signed(score_phase[cand*16 +: 16])
                                - $signed(score_phase[0 +: 16]);
                    if (phase_delta != ref_delta)
                        $fatal(
                            1,
                            "phase residual score-difference mismatch cand=%0d ref=%0d got=%0d",
                            cand,
                            ref_delta,
                            phase_delta
                        );
                end
            end
            if (gate_w1 !== ref_gate)
                $fatal(1, "4xW1 gate mismatch");
            if (gate_q1 !== ref_gate)
                $fatal(1, "QFSA-1C gate mismatch");
            if (gate_q2 !== ref_gate)
                $fatal(1, "QFSA-2C gate mismatch");
            if (gate_xb !== ref_gate)
                $fatal(1, "XBF-QFSA gate mismatch");
            if (gate_xbp !== ref_gate)
                $fatal(1, "XBF-DBDR gate mismatch");
            if (gate_phase !== ref_gate)
                $fatal(
                    1,
                    "phase residual gate mismatch ref=%h got=%h ref_score=%h phase_score=%h",
                    ref_gate,
                    gate_phase,
                    ref_score,
                    score_phase
                );
            if (cycles_phase !== cycles_xbp)
                $fatal(
                    1,
                    "phase residual cycle mismatch phase=%0d xbp=%0d",
                    cycles_phase,
                    cycles_xbp
                );
        end
    endtask

    task automatic run_case(
        input logic [31:0] q_value,
        input logic [159:0] k_value,
        input logic [4:0] valid_value,
        input bit expect_pooling_gain,
        input logic check_route,
        input logic [3:0] expected_route
    );
        int timeout;
        logic [79:0] held_score_w1;
        logic [79:0] held_score_q1;
        logic [79:0] held_score_q2;
        logic [79:0] held_score_xb;
        logic [79:0] held_score_xbp;
        logic [79:0] held_score_phase;
        logic [44:0] held_gate_w1;
        logic [44:0] held_gate_q1;
        logic [44:0] held_gate_q2;
        logic [44:0] held_gate_xb;
        logic [44:0] held_gate_xbp;
        logic [44:0] held_gate_phase;
        begin
            while (!(
                ready_w1
                && ready_q1
                && ready_q2
                && ready_xb
                && ready_xbp
                && ready_phase
            ))
                @(posedge clk_core);
            @(negedge clk_core);
            in_q = q_value;
            in_k = k_value;
            in_valid_mask = valid_value | 5'b00001;
            in_tag = in_tag + 16'd1;
            in_valid = 1'b1;
            @(negedge clk_core);
            in_valid = 1'b0;

            timeout = 0;
            while (!(
                valid_w1
                && valid_q1
                && valid_q2
                && valid_xb
                && valid_xbp
                && valid_phase
            )) begin
                @(posedge clk_core);
                timeout = timeout + 1;
                if (timeout > 200)
                    $fatal(1, "timeout waiting outputs");
            end
            @(negedge clk_core);
            check_outputs();
            if (expect_pooling_gain && !(cycles_q1 < cycles_w1))
                $fatal(
                    1,
                    "expected QFSA pooling gain qfsa=%0d w1=%0d",
                    cycles_q1,
                    cycles_w1
                );
            if (
                check_route
                && (
                    route_w1 !== expected_route
                    || route_q1 !== expected_route
                    || route_q2 !== expected_route
                    || route_xb !== expected_route
                    || route_xbp !== expected_route
                    || route_phase !== expected_route
                )
            )
                $fatal(
                    1,
                    "route mismatch exp=%x w1=%x q1=%x q2=%x xb=%x xbp=%x phase=%x",
                    expected_route,
                    route_w1,
                    route_q1,
                    route_q2,
                    route_xb,
                    route_xbp,
                    route_phase
                );

            held_score_w1 = score_w1;
            held_score_q1 = score_q1;
            held_score_q2 = score_q2;
            held_score_xb = score_xb;
            held_score_xbp = score_xbp;
            held_score_phase = score_phase;
            held_gate_w1 = gate_w1;
            held_gate_q1 = gate_q1;
            held_gate_q2 = gate_q2;
            held_gate_xb = gate_xb;
            held_gate_xbp = gate_xbp;
            held_gate_phase = gate_phase;
            repeat ($urandom_range(0, 4)) begin
                @(posedge clk_core);
                if (
                    score_w1 !== held_score_w1
                    || score_q1 !== held_score_q1
                    || score_q2 !== held_score_q2
                    || score_xb !== held_score_xb
                    || score_xbp !== held_score_xbp
                    || score_phase !== held_score_phase
                    || gate_w1 !== held_gate_w1
                    || gate_q1 !== held_gate_q1
                    || gate_q2 !== held_gate_q2
                    || gate_xb !== held_gate_xb
                    || gate_xbp !== held_gate_xbp
                    || gate_phase !== held_gate_phase
                )
                    $fatal(1, "output changed under backpressure");
            end
            @(negedge clk_core);
            out_ready = 1'b1;
            @(negedge clk_core);
            out_ready = 1'b0;
        end
    endtask

    initial begin
        logic [159:0] directed_k;
        clk_core = 1'b0;
        rst_core = 1'b1;
        in_valid = 1'b0;
        in_tag = 16'd0;
        in_q = '0;
        in_k = '0;
        in_valid_mask = '0;
        out_ready = 1'b0;
        repeat (4) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        directed_k = '0;
        directed_k[0*32 +: 32] = 32'd0;
        directed_k[1*32 +: 32] = 32'h00000003;
        directed_k[2*32 +: 32] = 32'h0000000c;
        directed_k[3*32 +: 32] = 32'd0;
        directed_k[4*32 +: 32] = 32'd0;
        run_case(
            32'hffffffff,
            directed_k,
            5'b11111,
            1'b1,
            1'b0,
            4'd0
        );

        directed_k = '0;
        directed_k[0*32 +: 32] = 32'd0;
        directed_k[1*32 +: 32] = 32'h11111111;
        directed_k[2*32 +: 32] = 32'h22222222;
        directed_k[3*32 +: 32] = 32'h44444444;
        directed_k[4*32 +: 32] = 32'h88888888;
        run_case(
            32'h5a5aa5a5,
            directed_k,
            5'b11111,
            1'b0,
            1'b0,
            4'd0
        );
        if (route_xb !== 4'h0 || cycles_xb !== 16'd33)
            $fatal(
                1,
                "T8 hot-bank oracle mismatch route=%x cycles=%0d",
                route_xb,
                cycles_xb
            );
        if (route_xbp !== 4'hf || cycles_xbp !== 16'd4)
            $fatal(
                1,
                "DBDR hot-bank oracle mismatch route=%x cycles=%0d",
                route_xbp,
                cycles_xbp
            );

        directed_k = '0;
        directed_k[1*32 +: 32] = 32'h00000011;
        directed_k[2*32 +: 32] = 32'h00000022;
        directed_k[3*32 +: 32] = 32'h00000044;
        directed_k[4*32 +: 32] = 32'h00000088;
        run_case(
            32'h0f0ff0f0,
            directed_k,
            5'b11111,
            1'b0,
            1'b0,
            4'd0
        );
        if (route_xbp !== 4'h0 || cycles_xbp !== 16'd9)
            $fatal(
                1,
                "DBDR B2 bound mismatch route=%x cycles=%0d",
                route_xbp,
                cycles_xbp
            );

        directed_k = '0;
        directed_k[1*32 +: 32] = 32'h00000fff;
        directed_k[2*32 +: 32] = 32'h0f000000;
        directed_k[3*32 +: 32] = 32'hffffffff;
        directed_k[4*32 +: 32] = 32'hffffffff;
        run_case(
            32'h33cc55aa,
            directed_k,
            5'b00111,
            1'b0,
            1'b0,
            4'd0
        );
        if (route_xbp !== 4'h1 || cycles_xbp !== 16'd2)
            $fatal(
                1,
                "DBDR mixed-path mismatch route=%x cycles=%0d",
                route_xbp,
                cycles_xbp
            );

        for (int route = 0; route < 16; route = route + 1) begin
            directed_k = '0;
            for (int dir = 0; dir < 4; dir = dir + 1) begin
                directed_k[(dir+1)*32 +: 32] = route[dir]
                    ? 32'hffffffff : 32'd0;
            end
            run_case(
                32'h13579bdf,
                directed_k,
                5'b11111,
                1'b0,
                1'b1,
                4'(route)
            );
        end

        for (int test = 0; test < 300; test = test + 1) begin
            logic [159:0] random_k;
            for (int cand = 0; cand < 5; cand = cand + 1)
                random_k[cand*32 +: 32] = $urandom;
            run_case(
                $urandom,
                random_k,
                {4'($urandom_range(0, 15)), 1'b1},
                1'b0,
                1'b0,
                4'd0
            );
        end

        $display("PASS tb_qfit_local5_score_leaf");
        $finish;
    end

endmodule

`default_nettype wire
