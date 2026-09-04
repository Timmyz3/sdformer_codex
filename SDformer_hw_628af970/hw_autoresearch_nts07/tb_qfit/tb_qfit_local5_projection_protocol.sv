`timescale 1ns/1ps
`default_nettype none

module tb_qfit_local5_projection_protocol;
    localparam int HEIGHT = 2;
    localparam int WIDTH = 2;
    localparam int TIME_PLANES = 2;
    localparam int HEAD_DIM = 32;
    localparam int OUT_DIM = 1;
    localparam int TOKENS = HEIGHT * WIDTH;
    localparam int TOTAL = TIME_PLANES * TOKENS;
    localparam int Y_W = 1;
    localparam int X_W = 1;
    localparam int PLANE_W = 1;
    localparam int LANE_W = 5;
    localparam int OUT_W = 1;
    localparam int GATE_W = 9;
    localparam int W_W = 8;
    localparam int ACC_W = 32;

    logic clk_core;
    logic rst_core;
    logic weight_valid;
    logic weight_ready;
    logic [LANE_W-1:0] weight_lane;
    logic [OUT_W-1:0] weight_out;
    logic signed [W_W-1:0] weight_data;
    logic weight_last;
    logic weight_context_release;
    logic weight_context_release_ready;
    logic projection_start;
    logic projection_accumulate;
    logic projection_start_ready;
    logic projection_close;
    logic term_issue_enable;
    logic projection_close_ready;
    logic projection_busy;
    logic projection_done;
    logic stream_idle;
    logic plane_start;
    logic [PLANE_W-1:0] plane_id;
    logic plane_start_ready;
    logic in_valid;
    logic in_ready;
    logic [Y_W-1:0] in_y;
    logic [X_W-1:0] in_x;
    logic [HEAD_DIM-1:0] in_q;
    logic [5*HEAD_DIM-1:0] in_k;
    logic [4:0] in_valid_mask;
    logic read_valid;
    logic read_ready;
    logic [PLANE_W-1:0] read_plane;
    logic [Y_W-1:0] read_y;
    logic [X_W-1:0] read_x;
    logic [OUT_W-1:0] read_out;
    logic read_data_valid;
    logic signed [ACC_W-1:0] read_data;
    logic protocol_error;
    logic [31:0] perf_descriptors;
    logic [31:0] perf_product_terms;
    logic [31:0] perf_destination_updates;
    logic [31:0] perf_relation_stalls;
    integer last_pause_remaining;
    integer descriptor_last_pause_count;
    integer run_last_pause_count;
    logic current_last_paused;
    logic observe_descriptor_fault;
    integer descriptor_fault_bank_writes;

    qfit_local5_projection_tile #(
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .HEAD_DIM(HEAD_DIM),
        .OUT_DIM(OUT_DIM),
        .BACKEND_KIND(0)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    always @(posedge clk_core) begin
        if (rst_core) begin
            descriptor_fault_bank_writes <= 0;
        end else if (observe_descriptor_fault) begin
            for (int bank = 0; bank < 5; bank = bank + 1) begin
                if (
                    dut.g_tcfm5_backend.u_projection.term_commit
                    && dut.g_tcfm5_backend.u_projection.bank_write_enable[bank]
                )
                    descriptor_fault_bank_writes
                        <= descriptor_fault_bank_writes + 1;
            end
        end
    end

    // Pause every descriptor-final term for three issue opportunities. This
    // includes the final term of the complete run and exercises close/drain.
    always @(negedge clk_core) begin
        if (rst_core) begin
            term_issue_enable = 1'b1;
            last_pause_remaining = 0;
            descriptor_last_pause_count = 0;
            run_last_pause_count = 0;
            current_last_paused = 1'b0;
        end else if (last_pause_remaining != 0) begin
            term_issue_enable = 1'b0;
            last_pause_remaining = last_pause_remaining - 1;
        end else if (
            dut.term_valid
            && dut.backend_term_ready
            && dut.term_descriptor_last
            && !current_last_paused
        ) begin
            term_issue_enable = 1'b0;
            last_pause_remaining = 2;
            descriptor_last_pause_count = descriptor_last_pause_count + 1;
            if (dut.term_run_last)
                run_last_pause_count = run_last_pause_count + 1;
            current_last_paused = 1'b1;
        end else begin
            term_issue_enable = 1'b1;
            if (!dut.term_valid || !dut.term_descriptor_last)
                current_last_paused = 1'b0;
        end
    end

    initial begin : watchdog
        repeat (200000) @(posedge clk_core);
        $fatal(
            1,
            "protocol TB timeout active=%0b planes=%0d tokens=%0d desc=%0d",
            dut.run_active_q,
            dut.planes_completed_q,
            dut.plane_tokens_q,
            dut.run_descriptors_q
        );
    end

    task automatic clear_inputs;
        weight_valid = 1'b0;
        weight_lane = '0;
        weight_out = '0;
        weight_data = '0;
        weight_last = 1'b0;
        weight_context_release = 1'b0;
        projection_start = 1'b0;
        projection_accumulate = 1'b0;
        projection_close = 1'b0;
        plane_start = 1'b0;
        plane_id = '0;
        in_valid = 1'b0;
        in_y = '0;
        in_x = '0;
        in_q = '0;
        in_k = '0;
        in_valid_mask = '0;
        read_valid = 1'b0;
        read_plane = '0;
        read_y = '0;
        read_x = '0;
        read_out = '0;
        observe_descriptor_fault = 1'b0;
    endtask

    task automatic load_weights;
        for (int lane = 0; lane < HEAD_DIM; lane = lane + 1) begin
            @(negedge clk_core);
            weight_lane = LANE_W'(lane);
            weight_out = '0;
            weight_data = W_W'(lane + 1);
            weight_last = lane == HEAD_DIM - 1;
            weight_valid = 1'b1;
            @(posedge clk_core);
            if (!weight_ready)
                $fatal(1, "weight port not ready");
            @(negedge clk_core);
            weight_valid = 1'b0;
            weight_last = 1'b0;
        end
        wait (projection_start_ready);
    endtask

    task automatic reset_and_load;
        rst_core = 1'b1;
        clear_inputs();
        repeat (4) @(negedge clk_core);
        rst_core = 1'b0;
        load_weights();
    endtask

    task automatic request_illegal_early_weight_last;
        rst_core = 1'b1;
        clear_inputs();
        repeat (4) @(negedge clk_core);
        rst_core = 1'b0;
        @(negedge clk_core);
        weight_lane = '0;
        weight_out = '0;
        weight_data = 8'sd1;
        weight_last = 1'b1;
        weight_valid = 1'b1;
        #1;
        if (weight_ready)
            $fatal(1, "early weight_last unexpectedly ready");
        @(negedge clk_core);
        weight_valid = 1'b0;
        weight_last = 1'b0;
        if (!protocol_error || projection_start_ready)
            $fatal(1, "incomplete weight transaction was not fail-closed");
    endtask

    task automatic request_illegal_duplicate_weight;
        rst_core = 1'b1;
        clear_inputs();
        repeat (4) @(negedge clk_core);
        rst_core = 1'b0;
        @(negedge clk_core);
        weight_lane = '0;
        weight_out = '0;
        weight_data = 8'sd1;
        weight_last = 1'b0;
        weight_valid = 1'b1;
        @(posedge clk_core);
        if (!weight_ready)
            $fatal(1, "first weight write was not accepted");
        @(negedge clk_core);
        weight_valid = 1'b0;
        @(negedge clk_core);
        weight_valid = 1'b1;
        #1;
        if (weight_ready)
            $fatal(1, "duplicate weight unexpectedly ready");
        @(negedge clk_core);
        weight_valid = 1'b0;
        if (!protocol_error || projection_start_ready)
            $fatal(1, "duplicate weight was not fail-closed");
    endtask

    task automatic start_run;
        wait (projection_start_ready);
        @(negedge clk_core);
        projection_start = 1'b1;
        @(negedge clk_core);
        projection_start = 1'b0;
        wait (projection_busy);
    endtask

    task automatic expect_illegal_close;
        @(negedge clk_core);
        if (projection_close_ready)
            $fatal(1, "incomplete transaction unexpectedly close-ready");
        projection_close = 1'b1;
        @(negedge clk_core);
        projection_close = 1'b0;
        if (!protocol_error || projection_done)
            $fatal(1, "illegal close was not fail-closed");
    endtask

    task automatic request_plane(input int p);
        @(negedge clk_core);
        plane_id = PLANE_W'(p);
        wait (plane_start_ready);
        plane_start = 1'b1;
        @(negedge clk_core);
        plane_start = 1'b0;
    endtask

    task automatic request_illegal_plane(input int p);
        @(negedge clk_core);
        plane_id = PLANE_W'(p);
        #1;
        if (plane_start_ready)
            $fatal(1, "illegal plane unexpectedly ready p=%0d", p);
        plane_start = 1'b1;
        @(negedge clk_core);
        plane_start = 1'b0;
        if (!protocol_error)
            $fatal(1, "illegal plane request was not reported p=%0d", p);
    endtask

    task automatic drive_token(
        input int y,
        input int x,
        input bit active_k
    );
        logic handshake;
        @(negedge clk_core);
        in_y = Y_W'(y);
        in_x = X_W'(x);
        in_q = active_k
            ? {HEAD_DIM{1'b1}}
            : 32'h5a5a_0000 ^ 32'(y * WIDTH + x);
        in_k = '0;
        in_valid_mask = 5'b00001;
        if (y > 0)
            in_valid_mask[1] = 1'b1;
        if (y < HEIGHT - 1)
            in_valid_mask[2] = 1'b1;
        if (x > 0)
            in_valid_mask[3] = 1'b1;
        if (x < WIDTH - 1)
            in_valid_mask[4] = 1'b1;
        if (active_k) begin
            for (int role = 0; role < 5; role = role + 1) begin
                if (in_valid_mask[role])
                    in_k[role*HEAD_DIM +: HEAD_DIM] = {HEAD_DIM{1'b1}};
            end
        end
        in_valid = 1'b1;
        do begin
            @(posedge clk_core);
            handshake = in_ready;
            @(negedge clk_core);
        end while (!handshake);
        in_valid = 1'b0;
    endtask

    task automatic drive_full_plane(input int p, input bit active_k);
        request_plane(p);
        for (int y = 0; y < HEIGHT; y = y + 1)
            for (int x = 0; x < WIDTH; x = x + 1)
                drive_token(y, x, active_k);
        wait (dut.local_plane_start_ready);
    endtask

    task automatic request_illegal_token(input int y, input int x);
        @(negedge clk_core);
        in_y = Y_W'(y);
        in_x = X_W'(x);
        in_valid = 1'b1;
        #1;
        if (in_ready)
            $fatal(1, "wrong-raster token unexpectedly ready");
        @(negedge clk_core);
        in_valid = 1'b0;
        if (!protocol_error)
            $fatal(1, "wrong-raster token was not reported");
    endtask

    task automatic inject_duplicate_descriptor;
        wait (perf_descriptors == TOKENS);
        wait (dut.builder_descriptor_ready && !dut.descriptor_valid);
        @(negedge clk_core);
        force dut.descriptor_valid = 1'b1;
        force dut.descriptor_source_id = '0;
        force dut.descriptor_y = '0;
        force dut.descriptor_x = '0;
        force dut.descriptor_k = {HEAD_DIM{1'b1}};
        force dut.descriptor_incoming_gates = {
            9'd32, 9'd32, 9'd32, 9'd32, 9'd32
        };
        // Source 0 is the north-west corner: keep the relation descriptor
        // boundary-legal so this negative isolates source-id duplication.
        force dut.descriptor_valid_mask = 5'b0_1011;
        observe_descriptor_fault = 1'b1;
        repeat (20) @(posedge clk_core);
        observe_descriptor_fault = 1'b0;
        if (
            !protocol_error
            || perf_descriptors != TOKENS
            || perf_product_terms != 0
            || perf_destination_updates != 0
            || descriptor_fault_bank_writes != 0
        )
            $fatal(
                1,
                "duplicate descriptor was not fail-closed error=%0b desc=%0d terms=%0d updates=%0d writes=%0d",
                protocol_error,
                perf_descriptors,
                perf_product_terms,
                perf_destination_updates,
                descriptor_fault_bank_writes
            );

        // A rejected ready/valid item must remain stable until recovery. Reset
        // the transaction before releasing the forced producer signals.
        @(negedge clk_core);
        rst_core = 1'b1;
        clear_inputs();
        @(posedge clk_core);
        @(negedge clk_core);
        release dut.descriptor_valid;
        release dut.descriptor_source_id;
        release dut.descriptor_y;
        release dut.descriptor_x;
        release dut.descriptor_k;
        release dut.descriptor_incoming_gates;
        release dut.descriptor_valid_mask;
    endtask

    task automatic check_active_accumulators;
        localparam int EXPECTED = 3 * 32 * 528;
        for (int p = 0; p < TIME_PLANES; p = p + 1) begin
            for (int y = 0; y < HEIGHT; y = y + 1) begin
                for (int x = 0; x < WIDTH; x = x + 1) begin
                    @(negedge clk_core);
                    read_plane = PLANE_W'(p);
                    read_y = Y_W'(y);
                    read_x = X_W'(x);
                    read_out = '0;
                    read_valid = 1'b1;
                    @(posedge clk_core);
                    if (!read_ready)
                        $fatal(1, "active accumulator read not ready");
                    @(negedge clk_core);
                    read_valid = 1'b0;
                    wait (read_data_valid);
                    #1;
                    if (read_data !== ACC_W'(EXPECTED))
                        $fatal(
                            1,
                            "active accumulator mismatch p=%0d y=%0d x=%0d got=%0d exp=%0d",
                            p,
                            y,
                            x,
                            read_data,
                            EXPECTED
                        );
                end
            end
        end
    endtask

    task automatic close_complete_run;
        wait (stream_idle && projection_close_ready);
        @(negedge clk_core);
        projection_close = 1'b1;
        @(negedge clk_core);
        projection_close = 1'b0;
        wait (projection_done);
        if (protocol_error || perf_descriptors != TOTAL)
            $fatal(1, "complete transaction did not retire cleanly");
        while (!projection_start_ready)
            @(negedge clk_core);
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        clear_inputs();

        // A lone early weight_last cannot authorize execution.
        request_illegal_early_weight_last();

        // A duplicate address cannot advance the weight coverage ledger.
        request_illegal_duplicate_weight();

        // Zero-plane close must fail closed.
        reset_and_load();
        start_run();
        expect_illegal_close();

        // A half plane cannot be committed.
        reset_and_load();
        start_run();
        request_plane(0);
        drive_token(0, 0, 1'b0);
        expect_illegal_close();

        // Plane order is part of the transaction contract.
        reset_and_load();
        start_run();
        request_illegal_plane(1);

        // A completed plane cannot be submitted again.
        reset_and_load();
        start_run();
        drive_full_plane(0, 1'b0);
        request_illegal_plane(0);

        // A start command while active must not restart or clear state.
        reset_and_load();
        start_run();
        @(negedge clk_core);
        projection_start = 1'b1;
        @(negedge clk_core);
        projection_start = 1'b0;
        if (!protocol_error || !projection_busy)
            $fatal(1, "busy start was not fail-closed");

        // Token coordinates are a raster-order transaction contract.
        reset_and_load();
        start_run();
        request_plane(0);
        request_illegal_token(0, 1);

        // The descriptor uniqueness ledger rejects a repeated source even
        // after its complete plane has retired normally.
        reset_and_load();
        start_run();
        drive_full_plane(0, 1'b0);
        inject_duplicate_descriptor();

        // Two nonzero runs reuse weights, clear Acc, pause every descriptor
        // final term, and independently read back every accumulator.
        reset_and_load();
        descriptor_last_pause_count = 0;
        run_last_pause_count = 0;
        for (int run = 0; run < 2; run = run + 1) begin
            start_run();
            drive_full_plane(0, 1'b1);
            drive_full_plane(1, 1'b1);
            close_complete_run();
            if (perf_product_terms != 256 || perf_destination_updates != 768)
                $fatal(1, "nonzero run work counters are not run-scoped");
            check_active_accumulators();
        end
        if (descriptor_last_pause_count != 2 * TOTAL)
            $fatal(1, "not every descriptor-final term was paused");
        if (run_last_pause_count != 2)
            $fatal(1, "complete-run final term was not paused twice");

        $display(
            "PASS Local5 adversarial protocol negative=9 consecutive_nonzero_runs=2 descriptors_per_run=%0d descriptor_last_pauses=%0d run_last_pauses=%0d terms_per_run=256 updates_per_run=768 descriptor_fault_terms=0 descriptor_fault_updates=0 descriptor_fault_bank_writes=0 acc32_mismatch=0",
            TOTAL,
            descriptor_last_pause_count,
            run_last_pause_count
        );
        $finish;
    end
endmodule

`default_nettype wire
