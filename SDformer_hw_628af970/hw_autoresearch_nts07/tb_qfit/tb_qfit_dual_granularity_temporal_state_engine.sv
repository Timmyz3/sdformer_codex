`timescale 1ns/1ps
`default_nettype none

module tb_qfit_dual_granularity_temporal_state_engine;
    localparam int CONTEXTS = 2;
`ifdef QFIT_TSMC28_FULL_DEPTH
    // The vendor macro adapter is fixed at 128 rows.  Two contexts therefore
    // use 64 base tiles while preserving the same directed protocol traffic.
    localparam int BASE_TILES = 64;
`else
    localparam int BASE_TILES = 3;
`endif
    localparam int BANKS = 6;
    localparam int LANES_PER_BANK = 16;
    localparam int ACC_W = 32;
    localparam int TAG_W = 32;
    localparam int EPOCH_W = 16;
    localparam int DOMAIN_W = 32;
    localparam int STEP_W = 4;
    localparam int LEN_W = 4;
    localparam int CTX_W = 1;
    localparam int BASE_TILE_W = (BASE_TILES <= 1) ? 1 : $clog2(BASE_TILES);
    localparam int BANK_W = 3;
    localparam int BANK_ACC_BITS = LANES_PER_BANK*ACC_W;
    localparam int WIDE_ACC_BITS = BANKS*BANK_ACC_BITS;

    logic clk_core;
    logic por_core;
    logic rst_core;
    logic [DOMAIN_W-1:0] active_domain;
    logic domain_fence_ready;
    logic domain_fence_error;
    logic wide_valid;
    logic wide_ready;
    logic [CTX_W-1:0] wide_context;
    logic [BASE_TILE_W-1:0] wide_base_tile;
    logic [EPOCH_W-1:0] wide_epoch;
    logic [DOMAIN_W-1:0] wide_domain;
    logic [STEP_W-1:0] wide_temporal_step;
    logic [LEN_W-1:0] wide_temporal_length;
    logic wide_temporal_first;
    logic wide_temporal_last;
    logic wide_use_motion;
    logic [TAG_W-1:0] wide_tag;
    logic [WIDE_ACC_BITS-1:0] wide_acc;
    logic narrow_valid;
    logic narrow_ready;
    logic [CTX_W-1:0] narrow_context;
    logic [BASE_TILE_W-1:0] narrow_base_tile;
    logic [BANK_W-1:0] narrow_bank;
    logic [EPOCH_W-1:0] narrow_epoch;
    logic [DOMAIN_W-1:0] narrow_domain;
    logic [STEP_W-1:0] narrow_temporal_step;
    logic [LEN_W-1:0] narrow_temporal_length;
    logic narrow_temporal_first;
    logic narrow_temporal_last;
    logic narrow_use_motion;
    logic [TAG_W-1:0] narrow_tag;
    logic [BANK_ACC_BITS-1:0] narrow_acc;
    logic abort_valid;
    logic abort_ready;
    logic [CTX_W-1:0] abort_context;
    logic [BASE_TILE_W-1:0] abort_base_tile;
    logic [BANKS-1:0] abort_bank_mask;
    logic [EPOCH_W-1:0] abort_epoch;
    logic [DOMAIN_W-1:0] abort_domain;
    logic [TAG_W-1:0] abort_tag;
    logic abort_error;
    logic output_valid;
    logic output_ready;
    logic output_is_wide;
    logic [CTX_W-1:0] output_context;
    logic [BASE_TILE_W-1:0] output_base_tile;
    logic [BANKS-1:0] output_bank_mask;
    logic [EPOCH_W-1:0] output_epoch;
    logic [DOMAIN_W-1:0] output_domain;
    logic [STEP_W-1:0] output_temporal_step;
    logic [LEN_W-1:0] output_temporal_length;
    logic output_temporal_first;
    logic output_temporal_last;
    logic output_used_motion;
    logic [TAG_W-1:0] output_tag;
    logic [WIDE_ACC_BITS-1:0] output_current_acc;
    logic rmw_busy;
    logic wide_protocol_error;
    logic narrow_protocol_error;

    integer signed expected_state
        [0:CONTEXTS-1][0:BASE_TILES-1][0:BANKS-1][0:LANES_PER_BANK-1];
    integer wide_accepts;
    integer narrow_accepts;
    integer wide_local;
    integer wide_motion;
    integer narrow_local;
    integer narrow_motion;
    integer abort_accepts;
    integer wide_errors;
    integer narrow_errors;
    integer rmw_stalls;
    integer reset_block_checks;
    integer domain_fault_checks;
    integer full_depth_rows_checked;
    logic [DOMAIN_W-1:0] saved_domain;

    qfit_dual_granularity_temporal_state_engine #(
        .CONTEXTS(CONTEXTS), .BASE_TILES(BASE_TILES), .BANKS(BANKS),
        .LANES_PER_BANK(LANES_PER_BANK), .ACC_W(ACC_W), .TAG_W(TAG_W),
        .EPOCH_W(EPOCH_W), .DOMAIN_W(DOMAIN_W), .STEP_W(STEP_W),
        .LEN_W(LEN_W)
    ) dut (.*);

    always #1.5 clk_core = ~clk_core;

    task automatic set_wide_data(input integer signed seed);
        integer signed value;
        begin
            for (int bank = 0; bank < BANKS; bank = bank + 1)
                for (int lane = 0; lane < LANES_PER_BANK; lane = lane + 1) begin
                    value = seed + bank*37 - lane*3;
                    wide_acc[((bank*LANES_PER_BANK+lane)*ACC_W) +: ACC_W] = value;
                end
        end
    endtask

    task automatic set_narrow_data(input integer signed seed);
        integer signed value;
        begin
            for (int lane = 0; lane < LANES_PER_BANK; lane = lane + 1) begin
                value = seed - lane*5;
                narrow_acc[(lane*ACC_W) +: ACC_W] = value;
            end
        end
    endtask

    task automatic check_wide(
        input int ctx, input int tile, input bit motion,
        input integer signed seed
    );
        integer signed input_value;
        integer signed expected_value;
        begin
            if (!output_valid || !output_is_wide ||
                    output_bank_mask !== {BANKS{1'b1}})
                $fatal(1, "M9.1 wide output shape mismatch");
            for (int bank = 0; bank < BANKS; bank = bank + 1)
                for (int lane = 0; lane < LANES_PER_BANK; lane = lane + 1) begin
                    input_value = seed + bank*37 - lane*3;
                    expected_value = motion ?
                        expected_state[ctx][tile][bank][lane] + input_value :
                        input_value;
                    expected_state[ctx][tile][bank][lane] = expected_value;
                    if ($signed(output_current_acc[
                            ((bank*LANES_PER_BANK+lane)*ACC_W) +: ACC_W]) !==
                            expected_value)
                        $fatal(1, "M9.1 wide value mismatch bank=%0d lane=%0d",
                            bank, lane);
                end
        end
    endtask

    task automatic check_narrow(
        input int ctx, input int tile, input int bank, input bit motion,
        input integer signed seed
    );
        integer signed input_value;
        integer signed expected_value;
        begin
            if (!output_valid || output_is_wide ||
                    output_bank_mask !== (BANKS'(1'b1) << bank))
                $fatal(1, "M9.1 narrow output shape mismatch bank=%0d", bank);
            for (int lane = 0; lane < LANES_PER_BANK; lane = lane + 1) begin
                input_value = seed - lane*5;
                expected_value = motion ?
                    expected_state[ctx][tile][bank][lane] + input_value :
                    input_value;
                expected_state[ctx][tile][bank][lane] = expected_value;
                if ($signed(output_current_acc[
                        ((bank*LANES_PER_BANK+lane)*ACC_W) +: ACC_W]) !==
                        expected_value)
                    $fatal(1, "M9.1 narrow value mismatch bank=%0d lane=%0d",
                        bank, lane);
            end
        end
    endtask

    task automatic drive_wide(
        input int ctx, input int tile, input int epoch, input int step,
        input int length, input bit first, input bit last, input bit motion,
        input int tag, input integer signed seed
    );
        integer wait_cycles;
        begin
            @(negedge clk_core);
            wide_context = CTX_W'(ctx);
            wide_base_tile = BASE_TILE_W'(tile);
            wide_epoch = EPOCH_W'(epoch);
            wide_domain = active_domain;
            wide_temporal_step = STEP_W'(step);
            wide_temporal_length = LEN_W'(length);
            wide_temporal_first = first;
            wide_temporal_last = last;
            wide_use_motion = motion;
            wide_tag = TAG_W'(tag);
            set_wide_data(seed);
            wide_valid = 1'b1;
            wait_cycles = 0;
            do begin
                @(posedge clk_core);
                wait_cycles = wait_cycles + 1;
                // drive_wide is exclusively a legal-traffic helper.  Report
                // an admission failure immediately; retain the bounded wait
                // below to distinguish genuine legal backpressure/deadlock.
                if (wide_protocol_error)
                    $fatal(1, "M9.1 wide protocol rejection ctx=%0d tile=%0d epoch=%0d step=%0d open_mask=%b",
                        ctx, tile, epoch, step,
                        {dut.sequence_open_q[ctx][tile][5],
                         dut.sequence_open_q[ctx][tile][4],
                         dut.sequence_open_q[ctx][tile][3],
                         dut.sequence_open_q[ctx][tile][2],
                         dut.sequence_open_q[ctx][tile][1],
                         dut.sequence_open_q[ctx][tile][0]});
                if (wait_cycles > 64)
                    $fatal(1, "M9.1 wide-ready timeout ctx=%0d tile=%0d epoch=%0d step=%0d",
                        ctx, tile, epoch, step);
            end while (!wide_ready);
            @(negedge clk_core);
            wide_valid = 1'b0;
            wait_cycles = 0;
            while (!output_valid) begin
                @(negedge clk_core);
                wait_cycles = wait_cycles + 1;
                if (wait_cycles > 64)
                    $fatal(1, "M9.1 wide-output timeout ctx=%0d tile=%0d epoch=%0d step=%0d",
                        ctx, tile, epoch, step);
            end
            check_wide(ctx, tile, motion, seed);
            wide_accepts = wide_accepts + 1;
            if (motion) wide_motion = wide_motion + 1;
            else wide_local = wide_local + 1;
        end
    endtask

    task automatic drive_narrow(
        input int ctx, input int tile, input int bank, input int epoch,
        input int step, input int length, input bit first, input bit last,
        input bit motion, input int tag, input integer signed seed
    );
        integer wait_cycles;
        begin
            @(negedge clk_core);
            narrow_context = CTX_W'(ctx);
            narrow_base_tile = BASE_TILE_W'(tile);
            narrow_bank = BANK_W'(bank);
            narrow_epoch = EPOCH_W'(epoch);
            narrow_domain = active_domain;
            narrow_temporal_step = STEP_W'(step);
            narrow_temporal_length = LEN_W'(length);
            narrow_temporal_first = first;
            narrow_temporal_last = last;
            narrow_use_motion = motion;
            narrow_tag = TAG_W'(tag);
            set_narrow_data(seed);
            narrow_valid = 1'b1;
            wait_cycles = 0;
            do begin
                @(posedge clk_core);
                wait_cycles = wait_cycles + 1;
                if (narrow_protocol_error)
                    $fatal(1, "M9.1 narrow protocol rejection ctx=%0d tile=%0d bank=%0d epoch=%0d step=%0d open=%b",
                        ctx, tile, bank, epoch, step,
                        dut.sequence_open_q[ctx][tile][bank]);
                if (wait_cycles > 64)
                    $fatal(1, "M9.1 narrow-ready timeout ctx=%0d tile=%0d bank=%0d epoch=%0d step=%0d",
                        ctx, tile, bank, epoch, step);
            end while (!narrow_ready);
            @(negedge clk_core);
            narrow_valid = 1'b0;
            wait_cycles = 0;
            while (!output_valid) begin
                @(negedge clk_core);
                wait_cycles = wait_cycles + 1;
                if (wait_cycles > 64)
                    $fatal(1, "M9.1 narrow-output timeout ctx=%0d tile=%0d bank=%0d epoch=%0d step=%0d",
                        ctx, tile, bank, epoch, step);
            end
            check_narrow(ctx, tile, bank, motion, seed);
            narrow_accepts = narrow_accepts + 1;
            if (motion) narrow_motion = narrow_motion + 1;
            else narrow_local = narrow_local + 1;
        end
    endtask

    task automatic clear_expected;
        begin
            for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1)
                for (int tile = 0; tile < BASE_TILES; tile = tile + 1)
                    for (int bank = 0; bank < BANKS; bank = bank + 1)
                        for (int lane = 0; lane < LANES_PER_BANK; lane = lane + 1)
                            expected_state[ctx][tile][bank][lane] = 0;
        end
    endtask

    initial begin
`ifdef VCS
        $display("SIMULATOR=Synopsys VCS");
        $display("ASSERTIONS=enabled");
`else
        $fatal(1, "M9.1 verification requires Synopsys VCS");
`endif
        clk_core = 1'b0;
        por_core = 1'b1;
        rst_core = 1'b0;
        active_domain = 32'h20260821;
        wide_valid = 1'b0;
        narrow_valid = 1'b0;
        abort_valid = 1'b0;
        output_ready = 1'b1;
        wide_context = '0;
        wide_base_tile = '0;
        wide_epoch = '0;
        wide_domain = '0;
        wide_temporal_step = '0;
        wide_temporal_length = '0;
        wide_temporal_first = 1'b0;
        wide_temporal_last = 1'b0;
        wide_use_motion = 1'b0;
        wide_tag = '0;
        wide_acc = '0;
        narrow_context = '0;
        narrow_base_tile = '0;
        narrow_bank = '0;
        narrow_epoch = '0;
        narrow_domain = '0;
        narrow_temporal_step = '0;
        narrow_temporal_length = '0;
        narrow_temporal_first = 1'b0;
        narrow_temporal_last = 1'b0;
        narrow_use_motion = 1'b0;
        narrow_tag = '0;
        narrow_acc = '0;
        abort_context = '0;
        abort_base_tile = '0;
        abort_bank_mask = '0;
        abort_epoch = '0;
        abort_domain = '0;
        abort_tag = '0;
        wide_accepts = 0;
        narrow_accepts = 0;
        wide_local = 0;
        wide_motion = 0;
        narrow_local = 0;
        narrow_motion = 0;
        abort_accepts = 0;
        wide_errors = 0;
        narrow_errors = 0;
        rmw_stalls = 0;
        reset_block_checks = 0;
        domain_fault_checks = 0;
        full_depth_rows_checked = 0;
        clear_expected();

        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        por_core = 1'b0;
        // POR release must arm the first observed domain without relying on a
        // second reset source.
        wait (domain_fence_ready);

        // An uncoordinated runtime domain change is sticky even if the input
        // returns to its old value.  Recovery requires functional reset plus
        // presentation of a genuinely new generation domain.
        saved_domain = active_domain;
        @(negedge clk_core);
        active_domain = saved_domain + 1'b1;
        #0;
        if (domain_fence_ready || !domain_fence_error)
            $fatal(1, "M9.1 runtime domain jump was not detected");
        @(posedge clk_core);
        @(negedge clk_core);
        active_domain = saved_domain;
        @(posedge clk_core);
        @(negedge clk_core);
        if (domain_fence_ready || !domain_fence_error)
            $fatal(1, "M9.1 runtime domain fault was not sticky");
        rst_core = 1'b1;
        active_domain = saved_domain + DOMAIN_W'(2);
        @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        wait (domain_fence_ready);
        domain_fault_checks = domain_fault_checks + 1;

        // A legal valid can already be high when reset arrives between clock
        // edges.  Combinational ready and the SRAM write command must close
        // immediately, before the reset edge clears registered state.
        @(negedge clk_core);
        wide_context = 0;
        wide_base_tile = 0;
        wide_epoch = 1;
        wide_domain = active_domain;
        wide_temporal_step = 0;
        wide_temporal_length = 2;
        wide_temporal_first = 1'b1;
        wide_temporal_last = 1'b0;
        wide_use_motion = 1'b0;
        wide_tag = 'h0bad0001;
        set_wide_data(13);
        wide_valid = 1'b1;
        #0.5 rst_core = 1'b1;
        #0;
        if (wide_ready || dut.bank_write_enable != '0)
            $fatal(1, "M9.1 valid-during-reset escaped ready/write gate");
        @(posedge clk_core);
        if (wide_ready || output_valid || dut.bank_write_enable != '0)
            $fatal(1, "M9.1 reset edge accepted or retired a request");
        reset_block_checks = reset_block_checks + 1;
        @(negedge clk_core);
        wide_valid = 1'b0;
        rst_core = 1'b0;
        repeat (2) @(posedge clk_core);
        if (domain_fence_ready || !domain_fence_error)
            $fatal(1, "M9.1 same-domain reset did not retain watermark");
        @(negedge clk_core);
        active_domain = active_domain + 1'b1;
        wait (domain_fence_ready);

        // Cancel a Motion transaction after its synchronous read but before
        // writeback.  The stale delta must not touch SRAM or produce output.
        drive_narrow(1, 2, 0, 1, 0, 2, 1, 0, 0, 'h0bad0002, 31);
        @(negedge clk_core);
        narrow_context = 1;
        narrow_base_tile = 2;
        narrow_bank = 0;
        narrow_epoch = 1;
        narrow_domain = active_domain;
        narrow_temporal_step = 1;
        narrow_temporal_length = 2;
        narrow_temporal_first = 1'b0;
        narrow_temporal_last = 1'b1;
        narrow_use_motion = 1'b1;
        narrow_tag = 'h0bad0002;
        set_narrow_data(-9);
        narrow_valid = 1'b1;
        do @(posedge clk_core); while (!narrow_ready);
        @(negedge clk_core);
        narrow_valid = 1'b0;
        if (!rmw_busy)
            $fatal(1, "M9.1 reset-mid-RMW test missed its read reservation");
        rst_core = 1'b1;
        #0;
        if (dut.rmw_commit || dut.bank_write_enable != '0)
            $fatal(1, "M9.1 reset-mid-RMW allowed stale writeback");
        @(posedge clk_core);
        // Sample after the reset edge's nonblocking assignments have retired.
        @(negedge clk_core);
        if (rmw_busy || output_valid || dut.bank_write_enable != '0)
            $fatal(1, "M9.1 reset-mid-RMW did not cancel transaction");
        reset_block_checks = reset_block_checks + 1;
        rst_core = 1'b0;
        active_domain = active_domain + 1'b1;
        wait (domain_fence_ready);

        // A cancelled non-first request cannot recreate state in the new
        // domain.  Only a Local first transaction may rebuild the row.
        narrow_domain = active_domain;
        narrow_epoch = 2;
        narrow_tag = 'h0bad0003;
        narrow_valid = 1'b1;
        @(posedge clk_core);
        if (narrow_ready || !narrow_protocol_error)
            $fatal(1, "M9.1 cancelled Motion survived reset/domain fence");
        @(negedge clk_core);
        narrow_valid = 1'b0;
        drive_narrow(1, 2, 0, 2, 0, 2, 1, 0, 0, 'h0bad0003, 41);
        reset_block_checks = reset_block_checks + 1;

        // The directed reset prelude is deliberately outside the published
        // throughput ledger below.
        wide_accepts = 0;
        narrow_accepts = 0;
        wide_local = 0;
        wide_motion = 0;
        narrow_local = 0;
        narrow_motion = 0;
        wide_errors = 0;
        narrow_errors = 0;
        clear_expected();

`ifdef QFIT_TSMC28_FULL_DEPTH
        // The reset prelude deliberately leaves context 1 / tile 2 / bank 0
        // open after proving that a Local first can rebuild cancelled state.
        // Start the physical-memory sweep behind its own deployment-equivalent
        // reset/domain fence so the sweep does not violate that live sequence.
        // SRAM contents are not reset; each Local below overwrites its row.
        @(negedge clk_core);
        rst_core = 1'b1;
        active_domain = active_domain + 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        wait (domain_fence_ready);

        // Phase A writes a unique Local vector to every physical row before
        // any readback.  Phase B then performs the closing Motion RMW for all
        // 128 rows.  Delaying every read until after every write makes any
        // cross-row address alias observable; an interleaved write/read pair
        // would cover addresses but could miss aliasing between later rows.
        for (int physical_row = 0; physical_row < CONTEXTS*BASE_TILES;
                physical_row = physical_row + 1) begin
            int sweep_context;
            int sweep_tile;
            sweep_context = physical_row / BASE_TILES;
            sweep_tile = physical_row % BASE_TILES;
            drive_wide(
                sweep_context, sweep_tile, 100 + physical_row, 0, 2,
                1, 0, 0, 'h81000000 + physical_row, 1000 + physical_row*17
            );
            full_depth_rows_checked = full_depth_rows_checked + 1;
            if ((full_depth_rows_checked % 16) == 0)
                $display("M9_1_FULL_DEPTH_PROGRESS rows=%0d", full_depth_rows_checked);
        end
        $display("M9_1_FULL_DEPTH_PHASE phase=local rows=%0d", full_depth_rows_checked);

        for (int physical_row = 0; physical_row < CONTEXTS*BASE_TILES;
                physical_row = physical_row + 1) begin
            int sweep_context;
            int sweep_tile;
            sweep_context = physical_row / BASE_TILES;
            sweep_tile = physical_row % BASE_TILES;
            drive_wide(
                sweep_context, sweep_tile, 100 + physical_row, 1, 2,
                0, 1, 1, 'h81000000 + physical_row, -300 - physical_row*11
            );
        end
        $display("M9_1_FULL_DEPTH_PHASE phase=motion rows=%0d", full_depth_rows_checked);
        if (full_depth_rows_checked != 128)
            $fatal(1, "M9.1 full-depth address sweep cardinality mismatch");
        $display("M9_1_FULL_DEPTH rows=%0d local_writes=%0d motion_rmws=%0d bank_lane_value_checks=%0d",
            full_depth_rows_checked, full_depth_rows_checked,
            full_depth_rows_checked,
            full_depth_rows_checked*2*BANKS*LANES_PER_BANK);

        // Retire sweep state behind the same reset/domain-generation fence
        // used by deployment before starting the published protocol ledger.
        @(negedge clk_core);
        rst_core = 1'b1;
        active_domain = active_domain + 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        wait (domain_fence_ready);
        clear_expected();
        wide_accepts = 0;
        narrow_accepts = 0;
        wide_local = 0;
        wide_motion = 0;
        narrow_local = 0;
        narrow_motion = 0;
`endif

        // Cross-granularity direction one: wide Local step zero, six narrow
        // Motion step ones, then wide Motion steps two through nine.
        drive_wide(0, 0, 1, 0, 10, 1, 0, 0, 'h1001, 100);

        // Both ports are legal.  The previous grant was wide, so round-robin
        // must choose narrow.  The RMW waits behind the stalled Local output.
        output_ready = 1'b0;
        wide_context = 0;
        wide_base_tile = 0;
        wide_epoch = 1;
        wide_domain = active_domain;
        wide_temporal_step = 1;
        wide_temporal_length = 10;
        wide_temporal_first = 1'b0;
        wide_temporal_last = 1'b0;
        wide_use_motion = 1'b1;
        wide_tag = 'h1001;
        set_wide_data(-20);
        wide_valid = 1'b1;
        narrow_context = 0;
        narrow_base_tile = 0;
        narrow_bank = 0;
        narrow_epoch = 1;
        narrow_domain = active_domain;
        narrow_temporal_step = 1;
        narrow_temporal_length = 10;
        narrow_temporal_first = 1'b0;
        narrow_temporal_last = 1'b0;
        narrow_use_motion = 1'b1;
        narrow_tag = 'h1001;
        set_narrow_data(-7);
        narrow_valid = 1'b1;
        @(posedge clk_core);
        if (!narrow_ready || wide_ready)
            $fatal(1, "M9.1 round-robin did not choose narrow after wide");
        @(negedge clk_core);
        wide_valid = 1'b0;
        narrow_valid = 1'b0;
        repeat (3) begin
            @(posedge clk_core);
            if (!rmw_busy) $fatal(1, "M9.1 RMW did not hold under backpressure");
            rmw_stalls = rmw_stalls + 1;
        end
        @(negedge clk_core);
        output_ready = 1'b1;
        while (!output_valid || !output_used_motion) @(negedge clk_core);
        check_narrow(0, 0, 0, 1'b1, -7);
        narrow_accepts = narrow_accepts + 1;
        narrow_motion = narrow_motion + 1;

        for (int bank = 1; bank < BANKS; bank = bank + 1)
            drive_narrow(0, 0, bank, 1, 1, 10, 0, 0, 1, 'h1001, -7-bank);
        for (int step = 2; step < 10; step = step + 1)
            drive_wide(0, 0, 1, step, 10, 0, step == 9, 1,
                'h1001, -20-step);

        // Cross-granularity direction two: six narrow Local banks create one
        // coherent row.  Both continuations are legal; after a narrow grant,
        // round-robin must choose the all-bank wide request.
        for (int bank = 0; bank < BANKS; bank = bank + 1)
            drive_narrow(1, 1, bank, 2, 0, 2, 1, 0, 0,
                'h2002, 200+bank);
        @(negedge clk_core);
        wide_context = 1;
        wide_base_tile = 1;
        wide_epoch = 2;
        wide_domain = active_domain;
        wide_temporal_step = 1;
        wide_temporal_length = 2;
        wide_temporal_first = 1'b0;
        wide_temporal_last = 1'b1;
        wide_use_motion = 1'b1;
        wide_tag = 'h2002;
        set_wide_data(-11);
        wide_valid = 1'b1;
        narrow_context = 1;
        narrow_base_tile = 1;
        narrow_bank = 0;
        narrow_epoch = 2;
        narrow_domain = active_domain;
        narrow_temporal_step = 1;
        narrow_temporal_length = 2;
        narrow_temporal_first = 1'b0;
        narrow_temporal_last = 1'b1;
        narrow_use_motion = 1'b1;
        narrow_tag = 'h2002;
        set_narrow_data(-3);
        narrow_valid = 1'b1;
        @(posedge clk_core);
        if (!wide_ready || narrow_ready)
            $fatal(1, "M9.1 round-robin did not choose wide after narrow");
        @(negedge clk_core);
        wide_valid = 1'b0;
        narrow_valid = 1'b0;
        while (!output_valid || !output_used_motion) @(negedge clk_core);
        check_wide(1, 1, 1'b1, -11);
        wide_accepts = wide_accepts + 1;
        wide_motion = wide_motion + 1;

        // Abort one stranded narrow sequence, preserve its watermark, reject
        // same-epoch replay, then accept a fresh generation.
        drive_narrow(0, 2, 0, 5, 0, 10, 1, 0, 0, 'h5005, 55);
        @(negedge clk_core);
        abort_context = 0;
        abort_base_tile = 2;
        abort_bank_mask = 6'b000001;
        abort_epoch = 5;
        abort_domain = active_domain;
        abort_tag = 'h5005;
        abort_valid = 1'b1;
        do @(posedge clk_core); while (!abort_ready);
        @(negedge clk_core);
        abort_valid = 1'b0;
        abort_accepts = abort_accepts + 1;

        narrow_context = 0;
        narrow_base_tile = 2;
        narrow_bank = 0;
        narrow_epoch = 5;
        narrow_domain = active_domain;
        narrow_temporal_step = 0;
        narrow_temporal_length = 10;
        narrow_temporal_first = 1'b1;
        narrow_temporal_last = 1'b0;
        narrow_use_motion = 1'b0;
        narrow_tag = 'h5006;
        set_narrow_data(56);
        narrow_valid = 1'b1;
        @(posedge clk_core);
        if (narrow_ready || !narrow_protocol_error)
            $fatal(1, "M9.1 same-epoch replay passed abort watermark");
        @(negedge clk_core);
        narrow_valid = 1'b0;
        narrow_errors = narrow_errors + 1;
        drive_narrow(0, 2, 0, 6, 0, 10, 1, 0, 0, 'h6006, 66);

        // Functional reset with an unchanged domain must reject queued stale
        // work until the external reset controller presents a new domain.
        @(negedge clk_core);
        rst_core = 1'b1;
        repeat (2) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        repeat (2) @(posedge clk_core);
        if (domain_fence_ready || !domain_fence_error)
            $fatal(1, "M9.1 unchanged reset domain was not fenced");
        @(negedge clk_core);
        wide_context = 0;
        wide_base_tile = 0;
        wide_epoch = 7;
        wide_domain = active_domain;
        wide_temporal_step = 0;
        wide_temporal_length = 2;
        wide_temporal_first = 1'b1;
        wide_temporal_last = 1'b0;
        wide_use_motion = 1'b0;
        wide_tag = 'h7007;
        set_wide_data(70);
        wide_valid = 1'b1;
        @(posedge clk_core);
        if (wide_ready || !wide_protocol_error)
            $fatal(1, "M9.1 stale pre-reset work was admitted");
        @(negedge clk_core);
        wide_valid = 1'b0;
        wide_errors = wide_errors + 1;
        active_domain = active_domain + 1'b1;
        wait (domain_fence_ready);
        clear_expected();
        drive_wide(0, 0, 7, 0, 2, 1, 0, 0, 'h7007, 70);
        drive_wide(0, 0, 7, 1, 2, 0, 1, 1, 'h7007, -7);

        repeat (4) @(posedge clk_core);
        $display("M9_1_RESULT wide=%0d narrow=%0d wide_local=%0d wide_motion=%0d narrow_local=%0d narrow_motion=%0d abort=%0d wide_errors=%0d narrow_errors=%0d rmw_stalls=%0d reset_block_checks=%0d domain_fault_checks=%0d",
            wide_accepts, narrow_accepts, wide_local, wide_motion,
            narrow_local, narrow_motion, abort_accepts, wide_errors,
            narrow_errors, rmw_stalls, reset_block_checks,
            domain_fault_checks);
`ifdef QFIT_TSMC28_FULL_DEPTH
        if (full_depth_rows_checked != 128)
            $fatal(1, "M9.1 full-depth coverage ledger mismatch");
`endif
        if (wide_accepts != 12 || narrow_accepts != 14 ||
                wide_local != 2 || wide_motion != 10 ||
                narrow_local != 8 || narrow_motion != 6 ||
                abort_accepts != 1 || wide_errors != 1 ||
                narrow_errors != 1 || rmw_stalls != 3 ||
                reset_block_checks != 3 || domain_fault_checks != 1)
            $fatal(1, "M9.1 coverage ledger mismatch");
        $display("PASS: Synopsys VCS M9.1 SRAM-realistic atomic Local/Motion shared temporal state exact");
        $finish;
    end

`ifdef QFIT_TSMC28_FULL_DEPTH
    // A bounded wall in simulation time prevents a failed vendor-macro
    // handshake from consuming an unbounded amount of VCS runtime.
    initial begin
        #100000;
        $fatal(1, "M9.1 full-depth simulation timeout");
    end
`endif
endmodule

`default_nettype wire
