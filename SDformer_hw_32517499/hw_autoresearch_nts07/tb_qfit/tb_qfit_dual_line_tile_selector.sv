`timescale 1ns/1ps
`default_nettype none

module tb_qfit_dual_line_tile_selector;
    localparam int TAG_W = 24;
    localparam int COUNT_W = 16;
    localparam int REQUESTS = 20000;

    logic clk_core, rst_core;
    logic request_valid, request_ready;
    logic [TAG_W-1:0] request_tag;
    logic [COUNT_W-1:0] request_valid_bits;
    logic [COUNT_W-1:0] request_current_nonzero;
    logic [COUNT_W-1:0] request_positive_transitions;
    logic [COUNT_W-1:0] request_negative_transitions;
    logic request_prior_state_valid;
    logic request_sequence_boundary;
    logic request_force_refresh;
    logic decision_valid, decision_ready;
    logic [TAG_W-1:0] decision_tag;
    logic decision_use_motion, decision_seed_previous;
    logic [COUNT_W:0] decision_work_count;
    logic [COUNT_W:0] decision_local_work_count;
    logic [COUNT_W:0] decision_transition_work_count;
    logic decision_force_local, decision_counts_legal;
    logic protocol_error;
    logic [31:0] perf_decisions, perf_local_decisions, perf_motion_decisions;
    logic [31:0] perf_local_work, perf_transition_work, perf_selected_work;

    logic [31:0] lfsr_q;
    logic expected_valid_q;
    logic [TAG_W-1:0] expected_tag_q;
    logic expected_motion_q, expected_force_local_q, expected_legal_q;
    logic [COUNT_W:0] expected_local_q, expected_transition_q, expected_work_q;
    integer issued, retired, motion_count, forced_count, illegal_count;
    integer cycles;

    qfit_dual_line_tile_selector u_dut (
        .clk_core(clk_core), .rst_core(rst_core),
        .request_valid(request_valid), .request_ready(request_ready),
        .request_tag(request_tag), .request_valid_bits(request_valid_bits),
        .request_current_nonzero(request_current_nonzero),
        .request_positive_transitions(request_positive_transitions),
        .request_negative_transitions(request_negative_transitions),
        .request_prior_state_valid(request_prior_state_valid),
        .request_sequence_boundary(request_sequence_boundary),
        .request_force_refresh(request_force_refresh),
        .decision_valid(decision_valid), .decision_ready(decision_ready),
        .decision_tag(decision_tag), .decision_use_motion(decision_use_motion),
        .decision_seed_previous(decision_seed_previous),
        .decision_work_count(decision_work_count),
        .decision_local_work_count(decision_local_work_count),
        .decision_transition_work_count(decision_transition_work_count),
        .decision_force_local(decision_force_local),
        .decision_counts_legal(decision_counts_legal),
        .protocol_error(protocol_error), .perf_decisions(perf_decisions),
        .perf_local_decisions(perf_local_decisions),
        .perf_motion_decisions(perf_motion_decisions),
        .perf_local_work(perf_local_work),
        .perf_transition_work(perf_transition_work),
        .perf_selected_work(perf_selected_work)
    );

    always #5 clk_core = ~clk_core;

    always_comb begin
        request_valid = issued < REQUESTS && lfsr_q[0];
        request_tag = TAG_W'(issued);
        request_valid_bits = 16'd512;
        request_current_nonzero = {7'd0, lfsr_q[8:0]};
        request_positive_transitions = {8'd0, lfsr_q[16:9]};
        request_negative_transitions = {8'd0, lfsr_q[24:17]};
        request_prior_state_valid = lfsr_q[25];
        request_sequence_boundary = lfsr_q[29:26] == 4'h0;
        request_force_refresh = lfsr_q[31:26] == 6'h3f;
        // Exercise the fail-safe path with counts above the tile size.
        if (lfsr_q[12:0] == 13'h1a5)
            request_current_nonzero = 16'd513;
        decision_ready = lfsr_q[3] || lfsr_q[7];
    end

    always @(posedge clk_core) begin
        logic [COUNT_W:0] transition;
        logic legal;
        logic force_local;
        logic motion;
        if (rst_core) begin
            lfsr_q <= 32'hc001d00d;
            expected_valid_q <= 1'b0;
            expected_tag_q <= '0;
            expected_motion_q <= 1'b0;
            expected_force_local_q <= 1'b0;
            expected_legal_q <= 1'b1;
            expected_local_q <= '0;
            expected_transition_q <= '0;
            expected_work_q <= '0;
            issued <= 0;
            retired <= 0;
            motion_count <= 0;
            forced_count <= 0;
            illegal_count <= 0;
            cycles <= 0;
        end else begin
            lfsr_q <= {lfsr_q[30:0],
                lfsr_q[31] ^ lfsr_q[21] ^ lfsr_q[1] ^ lfsr_q[0]};
            cycles <= cycles + 1;

            if (decision_valid && decision_ready) begin
                if (!expected_valid_q)
                    $fatal(1, "decision retired without expected request");
                if (decision_tag !== expected_tag_q
                    || decision_use_motion !== expected_motion_q
                    || decision_seed_previous !== expected_motion_q
                    || decision_force_local !== expected_force_local_q
                    || decision_counts_legal !== expected_legal_q
                    || decision_local_work_count !== expected_local_q
                    || decision_transition_work_count !== expected_transition_q
                    || decision_work_count !== expected_work_q)
                    $fatal(1, "decision mismatch tag=%0d", decision_tag);
                retired <= retired + 1;
                if (decision_use_motion)
                    motion_count <= motion_count + 1;
                if (decision_force_local)
                    forced_count <= forced_count + 1;
                if (!decision_counts_legal)
                    illegal_count <= illegal_count + 1;
                expected_valid_q <= 1'b0;
            end

            if (request_valid && request_ready) begin
                transition = {1'b0, request_positive_transitions}
                           + {1'b0, request_negative_transitions};
                legal = {1'b0, request_current_nonzero}
                        <= {1'b0, request_valid_bits}
                     && transition <= {1'b0, request_valid_bits};
                force_local = !request_prior_state_valid
                           || request_sequence_boundary
                           || request_force_refresh || !legal;
                motion = !force_local
                      && transition < {1'b0, request_current_nonzero};
                expected_valid_q <= 1'b1;
                expected_tag_q <= request_tag;
                expected_motion_q <= motion;
                expected_force_local_q <= force_local;
                expected_legal_q <= legal;
                expected_local_q <= {1'b0, request_current_nonzero};
                expected_transition_q <= transition;
                expected_work_q <= motion
                    ? transition : {1'b0, request_current_nonzero};
                issued <= issued + 1;
            end

            if (cycles > 200000)
                $fatal(1, "timeout issued=%0d retired=%0d", issued, retired);
            if (retired == REQUESTS) begin
                if (perf_decisions != REQUESTS
                    || perf_local_decisions + perf_motion_decisions != REQUESTS)
                    $fatal(1, "performance counter mismatch");
                if (!protocol_error || illegal_count == 0)
                    $fatal(1, "illegal-count fail-safe was not exercised");
                $display("PASS dual-line selector requests=%0d cycles=%0d local=%0d motion=%0d forced=%0d illegal=%0d selected_work=%0d",
                    retired, cycles, perf_local_decisions, perf_motion_decisions,
                    forced_count, illegal_count, perf_selected_work);
                $finish;
            end
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        repeat (8) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
    end
endmodule

`default_nettype wire
