`timescale 1ns/1ps
`default_nettype none

module tb_qfit_temporal_destination_commit_engine;
    localparam int CONTEXTS = 4;
    localparam int LANE_TILES = 8;
    localparam int LANES = 16;
    localparam int ACC_W = 32;
    localparam int TAG_W = 32;
    localparam int EPOCH_W = 16;
    localparam int DOMAIN_W = 8;
    localparam int STEP_W = 4;
    localparam int LEN_W = 4;
    localparam int CTX_W = 2;
    localparam int LANE_TILE_W = 3;

    logic clk_core = 1'b0;
    logic rst_core;
    logic [DOMAIN_W-1:0] active_domain;
    logic commit_valid;
    logic commit_ready;
    logic [CTX_W-1:0] commit_context;
    logic [LANE_TILE_W-1:0] commit_lane_tile;
    logic [EPOCH_W-1:0] commit_epoch;
    logic [DOMAIN_W-1:0] commit_domain;
    logic [STEP_W-1:0] commit_temporal_step;
    logic [LEN_W-1:0] commit_temporal_length;
    logic commit_temporal_first;
    logic commit_temporal_last;
    logic commit_use_motion;
    logic [TAG_W-1:0] commit_tag;
    logic [(LANES*ACC_W)-1:0] commit_acc;
    logic abort_valid;
    logic abort_ready;
    logic [CTX_W-1:0] abort_context;
    logic [LANE_TILE_W-1:0] abort_lane_tile;
    logic [EPOCH_W-1:0] abort_epoch;
    logic [DOMAIN_W-1:0] abort_domain;
    logic [TAG_W-1:0] abort_tag;
    logic abort_error;
    logic output_valid;
    logic output_ready;
    logic [CTX_W-1:0] output_context;
    logic [LANE_TILE_W-1:0] output_lane_tile;
    logic [EPOCH_W-1:0] output_epoch;
    logic [DOMAIN_W-1:0] output_domain;
    logic [STEP_W-1:0] output_temporal_step;
    logic [LEN_W-1:0] output_temporal_length;
    logic output_temporal_first;
    logic output_temporal_last;
    logic output_used_motion;
    logic [TAG_W-1:0] output_tag;
    logic [(LANES*ACC_W)-1:0] output_current_acc;
    logic protocol_error;

    logic signed [ACC_W-1:0] oracle_state [0:CONTEXTS-1][0:LANE_TILES-1][0:LANES-1];
    logic [EPOCH_W-1:0] oracle_epoch [0:CONTEXTS-1][0:LANE_TILES-1];
    logic [STEP_W-1:0] oracle_next_step [0:CONTEXTS-1][0:LANE_TILES-1];
    logic [LEN_W-1:0] oracle_length [0:CONTEXTS-1][0:LANE_TILES-1];
    logic [TAG_W-1:0] oracle_tag [0:CONTEXTS-1][0:LANE_TILES-1];
    logic oracle_initialized [0:CONTEXTS-1][0:LANE_TILES-1];
    logic oracle_valid [0:CONTEXTS-1][0:LANE_TILES-1];
    logic oracle_open [0:CONTEXTS-1][0:LANE_TILES-1];

    integer legal_commits;
    integer local_commits;
    integer motion_commits;
    integer output_stalls;
    integer sampled_protocol_errors;
    integer sampled_abort_errors;
    integer rejected_inputs;
    integer accepted_aborts;
    integer rejected_aborts;
    integer reset_blocked_inputs;
    logic auto_consume;

    always #1 clk_core = ~clk_core;
    always @(posedge clk_core) begin
        if (!rst_core && protocol_error)
            sampled_protocol_errors = sampled_protocol_errors + 1;
        if (!rst_core && abort_error)
            sampled_abort_errors = sampled_abort_errors + 1;
    end

    qfit_temporal_destination_commit_engine #(
        .CONTEXTS(CONTEXTS), .LANE_TILES(LANE_TILES), .LANES(LANES),
        .ACC_W(ACC_W), .TAG_W(TAG_W), .EPOCH_W(EPOCH_W),
        .DOMAIN_W(DOMAIN_W),
        .STEP_W(STEP_W), .LEN_W(LEN_W)
    ) dut (.*);

    task automatic consume_output(input integer holds);
        begin
            for (int hold = 0; hold < holds; hold++) begin
                @(posedge clk_core);
                #0.1;
                if (!output_valid) $fatal(1, "output dropped under backpressure");
                output_stalls = output_stalls + 1;
            end
            @(negedge clk_core);
            output_ready = 1'b1;
            @(posedge clk_core);
            #0.1;
            output_ready = 1'b0;
            if (output_valid) $fatal(1, "output did not retire");
        end
    endtask

    task automatic drive_commit(
        input integer ctx_value,
        input integer tile_value,
        input integer epoch_value,
        input integer step_value,
        input integer length_value,
        input logic first,
        input logic last,
        input logic use_motion,
        input integer tag_value,
        input integer value_base
    );
        logic signed [ACC_W-1:0] input_value;
        begin
            commit_context = CTX_W'(ctx_value);
            commit_lane_tile = LANE_TILE_W'(tile_value);
            commit_epoch = EPOCH_W'(epoch_value);
            commit_domain = active_domain;
            commit_temporal_step = STEP_W'(step_value);
            commit_temporal_length = LEN_W'(length_value);
            commit_temporal_first = first;
            commit_temporal_last = last;
            commit_use_motion = use_motion;
            commit_tag = TAG_W'(tag_value);
            for (int lane = 0; lane < LANES; lane++) begin
                input_value = ACC_W'(value_base + lane * 17 -
                    ((lane & 1) ? 31 : 0));
                commit_acc[(lane*ACC_W) +: ACC_W] = input_value;
            end
        end
    endtask

    task automatic legal_commit(
        input integer ctx_value,
        input integer tile_value,
        input integer epoch_value,
        input integer step_value,
        input integer length_value,
        input logic use_motion,
        input integer tag_value,
        input integer value_base
    );
        logic signed [ACC_W-1:0] input_value;
        logic signed [ACC_W-1:0] expected [0:LANES-1];
        logic first;
        logic last;
        begin
            if (output_valid)
                $fatal(1, "testbench attempted commit with pending output");
            first = step_value == 0;
            last = step_value == length_value - 1;
            @(negedge clk_core);
            drive_commit(ctx_value, tile_value, epoch_value, step_value,
                length_value, first, last, use_motion, tag_value, value_base);
            for (int lane = 0; lane < LANES; lane++) begin
                input_value = commit_acc[(lane*ACC_W) +: ACC_W];
                expected[lane] = use_motion ?
                    (oracle_state[ctx_value][tile_value][lane] + input_value) :
                    input_value;
            end
            commit_valid = 1'b1;
            #0.1;
            if (!commit_ready || protocol_error)
                $fatal(1, "legal commit rejected ctx=%0d tile=%0d epoch=%0h step=%0d len=%0d motion=%0b",
                       ctx_value, tile_value, epoch_value, step_value,
                       length_value, use_motion);
            @(posedge clk_core);
            #0.1;
            commit_valid = 1'b0;
            if (!output_valid || output_context != ctx_value ||
                output_lane_tile != tile_value || output_epoch != epoch_value ||
                output_domain != active_domain ||
                output_temporal_step != step_value ||
                output_temporal_length != length_value ||
                output_temporal_first != first || output_temporal_last != last ||
                output_used_motion != use_motion || output_tag != tag_value)
                $fatal(1, "commit output metadata mismatch");
            for (int lane = 0; lane < LANES; lane++) begin
                if ($signed(output_current_acc[(lane*ACC_W) +: ACC_W]) !==
                    expected[lane])
                    $fatal(1, "commit value mismatch lane=%0d got=%0d expected=%0d",
                           lane, $signed(output_current_acc[(lane*ACC_W) +: ACC_W]),
                           expected[lane]);
                oracle_state[ctx_value][tile_value][lane] = expected[lane];
            end
            oracle_initialized[ctx_value][tile_value] = 1'b1;
            oracle_valid[ctx_value][tile_value] = 1'b1;
            oracle_open[ctx_value][tile_value] = !last;
            oracle_epoch[ctx_value][tile_value] = EPOCH_W'(epoch_value);
            oracle_next_step[ctx_value][tile_value] = STEP_W'(step_value + 1);
            oracle_length[ctx_value][tile_value] = LEN_W'(length_value);
            oracle_tag[ctx_value][tile_value] = TAG_W'(tag_value);
            legal_commits = legal_commits + 1;
            if (use_motion) motion_commits = motion_commits + 1;
            else local_commits = local_commits + 1;
            if (auto_consume)
                consume_output((tag_value + ctx_value + tile_value) % 4);
        end
    endtask

    task automatic reject_commit(
        input integer ctx_value,
        input integer tile_value,
        input integer epoch_value,
        input integer step_value,
        input integer length_value,
        input logic first,
        input logic last,
        input logic use_motion,
        input integer tag_value
    );
        begin
            if (output_valid)
                $fatal(1, "ordinary reject task requires no pending output");
            @(negedge clk_core);
            drive_commit(ctx_value, tile_value, epoch_value, step_value,
                length_value, first, last, use_motion, tag_value, -77);
            commit_valid = 1'b1;
            #0.1;
            if (!protocol_error || commit_ready)
                $fatal(1, "illegal commit was not rejected tag=%0h", tag_value);
            @(posedge clk_core);
            #0.1;
            commit_valid = 1'b0;
            if (output_valid) $fatal(1, "illegal commit produced output");
            rejected_inputs = rejected_inputs + 1;
        end
    endtask

    task automatic legal_abort(
        input integer ctx_value,
        input integer tile_value,
        input integer epoch_value,
        input integer tag_value
    );
        begin
            if (output_valid) $fatal(1, "abort requires no pending output");
            @(negedge clk_core);
            abort_context = CTX_W'(ctx_value);
            abort_lane_tile = LANE_TILE_W'(tile_value);
            abort_epoch = EPOCH_W'(epoch_value);
            abort_domain = active_domain;
            abort_tag = TAG_W'(tag_value);
            abort_valid = 1'b1;
            #0.1;
            if (!abort_ready || abort_error) $fatal(1, "legal abort rejected");
            @(posedge clk_core);
            #0.1;
            abort_valid = 1'b0;
            oracle_valid[ctx_value][tile_value] = 1'b0;
            oracle_open[ctx_value][tile_value] = 1'b0;
            accepted_aborts = accepted_aborts + 1;
        end
    endtask

    task automatic reject_abort(
        input integer ctx_value,
        input integer tile_value,
        input integer epoch_value,
        input integer tag_value
    );
        begin
            @(negedge clk_core);
            abort_context = CTX_W'(ctx_value);
            abort_lane_tile = LANE_TILE_W'(tile_value);
            abort_epoch = EPOCH_W'(epoch_value);
            abort_domain = active_domain;
            abort_tag = TAG_W'(tag_value);
            abort_valid = 1'b1;
            #0.1;
            if (!abort_error || abort_ready) $fatal(1, "illegal abort accepted");
            @(posedge clk_core);
            #0.1;
            abort_valid = 1'b0;
            rejected_aborts = rejected_aborts + 1;
        end
    endtask

    task automatic clear_oracle_metadata;
        for (int ctx_i = 0; ctx_i < CONTEXTS; ctx_i++) begin
            for (int tile_i = 0; tile_i < LANE_TILES; tile_i++) begin
                oracle_initialized[ctx_i][tile_i] = 1'b0;
                oracle_valid[ctx_i][tile_i] = 1'b0;
                oracle_open[ctx_i][tile_i] = 1'b0;
                oracle_epoch[ctx_i][tile_i] = '0;
                oracle_next_step[ctx_i][tile_i] = '0;
                oracle_length[ctx_i][tile_i] = '0;
                oracle_tag[ctx_i][tile_i] = '0;
            end
        end
    endtask

    initial begin
`ifdef SIMULATOR_VCS
        $display("SIMULATOR=Synopsys VCS");
`else
        $fatal(1, "M8 regression requires Synopsys VCS identity");
`endif
`ifdef SVA_RUNTIME_ENABLED
        $display("ASSERTIONS=enabled");
`else
        $fatal(1, "M8 regression requires bound SVA");
`endif
        rst_core = 1'b1;
        active_domain = 8'h01;
        commit_valid = 1'b0;
        commit_context = '0;
        commit_lane_tile = '0;
        commit_epoch = '0;
        commit_domain = '0;
        commit_temporal_step = '0;
        commit_temporal_length = '0;
        commit_temporal_first = 1'b0;
        commit_temporal_last = 1'b0;
        commit_use_motion = 1'b0;
        commit_tag = '0;
        commit_acc = '0;
        abort_valid = 1'b0;
        abort_context = '0;
        abort_lane_tile = '0;
        abort_epoch = '0;
        abort_domain = '0;
        abort_tag = '0;
        output_ready = 1'b0;
        legal_commits = 0;
        local_commits = 0;
        motion_commits = 0;
        output_stalls = 0;
        sampled_protocol_errors = 0;
        sampled_abort_errors = 0;
        rejected_inputs = 0;
        accepted_aborts = 0;
        rejected_aborts = 0;
        reset_blocked_inputs = 0;
        auto_consume = 1'b1;
        clear_oracle_metadata();

        // Reset must never advertise a handshake or a protocol error.
        @(negedge clk_core);
        drive_commit(0, 0, 1, 0, 2, 1'b1, 1'b0, 1'b0, 32'h1000, 1);
        commit_valid = 1'b1;
        #0.1;
        if (commit_ready || protocol_error || abort_ready || abort_error)
            $fatal(1, "reset did not quiesce handshakes/errors");
        @(posedge clk_core);
        #0.1;
        if (output_valid) $fatal(1, "reset-period commit produced output");
        commit_valid = 1'b0;
        reset_blocked_inputs = reset_blocked_inputs + 1;
        @(negedge clk_core);
        rst_core = 1'b0;

        // Step, length, tag, generation, and open/closed admission.
        reject_commit(0, 0, 1, 1, 2, 1'b0, 1'b1, 1'b0, 32'h1001);
        reject_commit(0, 0, 1, 0, 2, 1'b1, 1'b0, 1'b1, 32'h1002);
        reject_commit(0, 0, 1, 0, 2, 1'b1, 1'b1, 1'b0, 32'h1003);
        reject_commit(0, 0, 1, 0, 3, 1'b1, 1'b0, 1'b0, 32'h1004);
        legal_commit(0, 0, 1, 0, 2, 1'b0, 32'h1010, -1000);
        reject_commit(0, 0, 2, 1, 2, 1'b0, 1'b1, 1'b1, 32'h1010);
        reject_commit(0, 0, 2, 0, 2, 1'b1, 1'b0, 1'b0, 32'h1012);
        reject_commit(0, 0, 1, 0, 2, 1'b0, 1'b0, 1'b1, 32'h1010);
        reject_commit(0, 0, 1, 2, 2, 1'b0, 1'b0, 1'b1, 32'h1010);
        reject_commit(0, 0, 1, 1, 2, 1'b0, 1'b1, 1'b1, 32'hbad0);
        legal_commit(0, 0, 1, 1, 2, 1'b1, 32'h1010, 37);
        reject_commit(0, 0, 1, 1, 2, 1'b0, 1'b1, 1'b1, 32'h1010);
        reject_commit(0, 0, 1, 0, 2, 1'b1, 1'b0, 1'b0, 32'h1010);
        reject_commit(0, 0, 0, 0, 2, 1'b1, 1'b0, 1'b0, 32'h1011);

        // Reset during an open sequence invalidates the resident generation domain.
        legal_commit(1, 1, 9, 0, 10, 1'b0, 32'h1020, 77);
        @(negedge clk_core); rst_core = 1'b1; active_domain = 8'h02;
        @(posedge clk_core); #0.1;
        @(negedge clk_core); rst_core = 1'b0;
        clear_oracle_metadata();
        // A pre-reset queued first is rejected even though its destination is
        // empty after reset, because its reset-domain nonce is stale.
        @(negedge clk_core);
        drive_commit(1, 7, 16'h1, 0, 2, 1'b1, 1'b0, 1'b0, 32'h1021, 5);
        commit_domain = 8'h01;
        commit_valid = 1'b1;
        #0.1;
        if (!protocol_error || commit_ready)
            $fatal(1, "pre-reset domain transaction was accepted");
        @(posedge clk_core); #0.1;
        commit_valid = 1'b0;
        rejected_inputs = rejected_inputs + 1;
        reject_commit(1, 1, 9, 1, 10, 1'b0, 1'b0, 1'b1, 32'h1020);

        // Abort recovers one stranded entry while retaining its epoch watermark.
        legal_commit(2, 2, 16'h10, 0, 10, 1'b0, 32'h1030, 111);
        reject_commit(2, 2, 16'h10, 1, 10, 1'b0, 1'b1, 1'b1, 32'h1030);
        reject_commit(2, 2, 16'h10, 1, 2, 1'b0, 1'b0, 1'b1, 32'h1030);
        legal_commit(2, 2, 16'h10, 1, 10, 1'b1, 32'h1030, -29);
        reject_abort(2, 2, 16'h11, 32'h1030);
        legal_abort(2, 2, 16'h10, 32'h1030);
        reject_commit(2, 2, 16'h10, 2, 10, 1'b0, 1'b0, 1'b1, 32'h1030);
        reject_commit(2, 2, 16'h10, 0, 2, 1'b1, 1'b0, 1'b0, 32'h1030);
        legal_commit(2, 2, 16'h11, 0, 2, 1'b0, 32'h1031, 19);
        legal_commit(2, 2, 16'h11, 1, 2, 1'b1, 32'h1031, -4);

        // Serial freshness permits epoch wrap but rejects stale ABA traffic.
        legal_commit(3, 3, 16'hfffe, 0, 2, 1'b0, 32'h1040, 1);
        legal_commit(3, 3, 16'hfffe, 1, 2, 1'b1, 32'h1040, 2);
        legal_commit(3, 3, 16'hffff, 0, 2, 1'b0, 32'h1041, 3);
        legal_commit(3, 3, 16'hffff, 1, 2, 1'b1, 32'h1041, 4);
        legal_commit(3, 3, 16'h0000, 0, 2, 1'b0, 32'h1042, 5);
        legal_commit(3, 3, 16'h0000, 1, 2, 1'b1, 32'h1042, 6);
        legal_commit(3, 3, 16'h7fff, 0, 2, 1'b0, 32'h1043, 7);
        legal_commit(3, 3, 16'h7fff, 1, 2, 1'b1, 32'h1043, 8);
        // Exactly half the serial space is ambiguous and must fail closed.
        reject_commit(3, 3, 16'hffff, 0, 2, 1'b1, 1'b0, 1'b0, 32'h1043);
        reject_commit(3, 3, 16'h7ffe, 0, 2, 1'b1, 1'b0, 1'b0, 32'h1044);

        // Invalid traffic remains visible while an output is backpressured.
        auto_consume = 1'b0;
        legal_commit(0, 4, 16'h20, 0, 2, 1'b0, 32'h1050, 100);
        // Abort is allowed to terminate future work only after an irrevocable
        // pending output has retired.
        @(negedge clk_core);
        abort_context = 0;
        abort_lane_tile = 4;
        abort_epoch = 16'h20;
        abort_domain = active_domain;
        abort_tag = 32'h1050;
        abort_valid = 1'b1;
        #0.1;
        if (!abort_error || abort_ready || !output_valid)
            $fatal(1, "same-sequence abort crossed a stalled output");
        @(posedge clk_core); #0.1;
        abort_valid = 1'b0;
        rejected_aborts = rejected_aborts + 1;
        @(negedge clk_core);
        drive_commit(0, 4, 16'h20, 0, 2, 1'b0, 1'b0, 1'b1, 32'h1050, -1);
        commit_valid = 1'b1;
        #0.1;
        if (!protocol_error || commit_ready || !output_valid)
            $fatal(1, "illegal stalled request was masked");
        @(posedge clk_core); #0.1;
        commit_valid = 1'b0;
        rejected_inputs = rejected_inputs + 1;
        consume_output(2);
        auto_consume = 1'b1;
        legal_commit(0, 4, 16'h20, 1, 2, 1'b1, 32'h1050, 9);

        // Hold a legal input stable, then retire the old output and accept the
        // continuation in the same cycle.
        auto_consume = 1'b0;
        legal_commit(1, 5, 16'h30, 0, 2, 1'b0, 32'h1060, 200);
        @(negedge clk_core);
        drive_commit(1, 5, 16'h30, 1, 2, 1'b0, 1'b1, 1'b1, 32'h1060, -17);
        commit_valid = 1'b1;
        #0.1;
        if (commit_ready || protocol_error)
            $fatal(1, "legal input did not wait cleanly for output space");
        @(posedge clk_core); #0.1;
        if (!output_valid) $fatal(1, "old output lost during input stall");
        @(negedge clk_core);
        output_ready = 1'b1;
        #0.1;
        if (!commit_ready || protocol_error)
            $fatal(1, "same-cycle retire/replace handshake failed");
        @(posedge clk_core); #0.1;
        commit_valid = 1'b0;
        output_ready = 1'b0;
        if (!output_valid || output_temporal_step != 1 ||
            output_tag != 32'h1060 || !output_used_motion)
            $fatal(1, "replacement output metadata mismatch");
        for (int lane = 0; lane < LANES; lane++) begin
            logic signed [ACC_W-1:0] delta;
            logic signed [ACC_W-1:0] expected;
            delta = commit_acc[(lane*ACC_W) +: ACC_W];
            expected = oracle_state[1][5][lane] + delta;
            if ($signed(output_current_acc[(lane*ACC_W) +: ACC_W]) !== expected)
                $fatal(1, "same-cycle replacement value mismatch lane=%0d", lane);
            oracle_state[1][5][lane] = expected;
        end
        oracle_open[1][5] = 1'b0;
        oracle_next_step[1][5] = 2;
        legal_commits = legal_commits + 1;
        motion_commits = motion_commits + 1;
        consume_output(1);
        auto_consume = 1'b1;

        // Leave three independent entries open together, then close them in a
        // different order to exercise real multi-entry interleaving.
        legal_commit(0, 6, 16'h40, 0, 2, 1'b0, 32'h1400, 10);
        legal_commit(1, 6, 16'h41, 0, 2, 1'b0, 32'h1401, 20);
        legal_commit(2, 6, 16'h42, 0, 2, 1'b0, 32'h1402, 30);
        legal_commit(2, 6, 16'h42, 1, 2, 1'b1, 32'h1402, -3);
        legal_commit(0, 6, 16'h40, 1, 2, 1'b1, 32'h1400, -1);
        legal_commit(1, 6, 16'h41, 1, 2, 1'b1, 32'h1401, -2);

        // Deterministic interleaved T2/T10 Local/Motion sequences.
        for (int seq_i = 0; seq_i < 64; seq_i++) begin
            int ctx_value;
            int tile_value;
            int epoch_value;
            int temporal_steps;
            int sequence_tag;
            ctx_value = seq_i % CONTEXTS;
            tile_value = (seq_i * 3) % LANE_TILES;
            epoch_value = 16'h100 + seq_i;
            temporal_steps = (seq_i & 1) ? 2 : 10;
            sequence_tag = 32'h2000 + seq_i;
            for (int step_i = 0; step_i < temporal_steps; step_i++) begin
                logic motion;
                integer base;
                motion = (step_i != 0) && (((seq_i + step_i) % 3) != 0);
                if (seq_i == 63 && step_i == temporal_steps - 1)
                    base = 32'h7fff_ff00;
                else
                    base = (seq_i * 100003) - (step_i * 70001) - 2000000;
                legal_commit(ctx_value, tile_value, epoch_value, step_i,
                    temporal_steps, motion, sequence_tag, base);
            end
        end

        $display("TEMPORAL_COMMIT_RESULT legal=%0d local=%0d motion=%0d rejected=%0d protocol_errors=%0d abort=%0d abort_rejected=%0d reset_blocked=%0d output_stalls=%0d",
                 legal_commits, local_commits, motion_commits, rejected_inputs,
                 sampled_protocol_errors, accepted_aborts, rejected_aborts,
                 reset_blocked_inputs, output_stalls);
        if (legal_commits != 409 || local_commits != 183 || motion_commits != 226 ||
            rejected_inputs != 21 || sampled_protocol_errors != 21 ||
            accepted_aborts != 1 || rejected_aborts != 2 ||
            sampled_abort_errors != 2 || reset_blocked_inputs != 1 ||
            output_stalls <= 0)
            $fatal(1, "M8.2 coverage/accounting mismatch");
        $display("PASS: Synopsys VCS M8.2 reset-fenced Local/Motion temporal destination commit exact");
        $finish;
    end
endmodule

`default_nettype wire
