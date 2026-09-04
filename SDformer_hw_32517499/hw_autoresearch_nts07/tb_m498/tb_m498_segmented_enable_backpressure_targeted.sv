`timescale 1ns/1ps
`default_nettype none
module tb_m498_segmented_enable_backpressure_targeted;
    localparam int LANES = 96;
    localparam int ROW_BITS = 6;
    localparam int ROWS = 1 << ROW_BITS;

    logic clk_core, reset_n;
    logic prefetch_valid, prefetch_ready;
    logic [ROW_BITS-1:0] prefetch_parent_id;
    logic scratch_read_enable;
    logic [ROW_BITS-1:0] scratch_read_address;
    logic [LANES*12-1:0] scratch_read_data;
    logic issue_valid, issue_ready, issue_first, issue_last;
    logic [ROW_BITS-1:0] issue_row_id;
    logic issue_parent_valid;
    logic [ROW_BITS-1:0] issue_parent_id;
    logic [LANES*12-1:0] issue_residual_data;
    logic [LANES*19-1:0] issue_psum_prior;
    logic scratch_write_enable;
    logic [ROW_BITS-1:0] scratch_write_address;
    logic [LANES*12-1:0] scratch_write_data;
    logic psum_write_valid, psum_write_ready;
    logic [ROW_BITS-1:0] psum_write_address;
    logic [LANES*19-1:0] psum_write_data;
    logic row_complete, protocol_error, row_active;
    logic [1:0] parent_queue_occupancy;
    logic parent_queue_full;
    logic debug_forward_event, debug_scratch_read_event;
    logic debug_read_response_event, debug_dual_enqueue_event;
    logic debug_overflow_block_event, debug_stalled_raw_prefetch_event;

    logic [LANES*12-1:0] scratch_mem [0:ROWS-1];
    logic [ROW_BITS-1:0] scratch_read_address_q;
    integer stalled_cycles, reads_seen, forward_seen, writes_seen;
    integer child_value_checks, stale_mismatches;

    m498_segmented_enable_backpressure_safe_parent_queue_pipeline #(
        .LANES(LANES), .ROW_BITS(ROW_BITS)
    ) dut (.*);
    m476r2_backpressure_safe_assertions #(
        .LANES(LANES), .ROW_BITS(ROW_BITS)
    ) sva (.*);

    always #1.5 clk_core = ~clk_core;
    assign scratch_read_data = scratch_mem[scratch_read_address_q];
    always @(posedge clk_core) begin
        if (scratch_read_enable)
            scratch_read_address_q <= scratch_read_address;
        if (scratch_write_enable)
            scratch_mem[scratch_write_address] <= scratch_write_data;
    end

    task automatic clear_drives;
        begin
            prefetch_valid = 0;
            prefetch_parent_id = 0;
            issue_valid = 0;
            issue_row_id = 0;
            issue_first = 0;
            issue_last = 0;
            issue_parent_valid = 0;
            issue_parent_id = 0;
            issue_residual_data = 0;
            issue_psum_prior = 0;
        end
    endtask

    task automatic drive_all_lanes(input integer residual, input integer psum);
        integer lane;
        begin
            issue_residual_data = '0;
            issue_psum_prior = '0;
            for (lane = 0; lane < LANES; lane = lane + 1) begin
                issue_residual_data[lane*12 +: 12] = residual[11:0];
                issue_psum_prior[lane*19 +: 19] = psum[18:0];
            end
        end
    endtask

    always @(posedge clk_core) begin : monitor
        integer lane, got;
        if (reset_n) begin
            if (protocol_error)
                $fatal(1, "unexpected M476r2 protocol error");
            if (debug_stalled_raw_prefetch_event)
                stalled_cycles = stalled_cycles + 1;
            if (scratch_read_enable)
                reads_seen = reads_seen + 1;
            if (debug_forward_event)
                forward_seen = forward_seen + 1;
            if (scratch_write_enable) begin
                writes_seen = writes_seen + 1;
                if (scratch_write_address == 1) begin
                    for (lane = 0; lane < LANES; lane = lane + 1) begin
                        got = $signed(scratch_write_data[lane*12 +: 12]);
                        if (got != 1)
                            $fatal(1, "row1 new-value write mismatch lane=%0d got=%0d",
                                lane, got);
                    end
                end
                if (scratch_write_address == 2) begin
                    for (lane = 0; lane < LANES; lane = lane + 1) begin
                        got = $signed(scratch_write_data[lane*12 +: 12]);
                        child_value_checks = child_value_checks + 1;
                        if (got != 1)
                            stale_mismatches = stale_mismatches + 1;
                    end
                end
            end
        end
    end

    initial begin : test
        integer lane;
        clk_core = 0;
        reset_n = 0;
        psum_write_ready = 1;
        scratch_read_address_q = 0;
        stalled_cycles = 0;
        reads_seen = 0;
        forward_seen = 0;
        writes_seen = 0;
        child_value_checks = 0;
        stale_mismatches = 0;
        clear_drives();
        for (integer row = 0; row < ROWS; row = row + 1)
            scratch_mem[row] = '0;
        // Old row1 is deliberately 5; the stalled final issue will replace it
        // with 1.  Any premature normal read makes child row2 equal 5.
        for (lane = 0; lane < LANES; lane = lane + 1)
            scratch_mem[1][lane*12 +: 12] = 12'sd5;

        repeat (4) @(posedge clk_core);
        @(negedge clk_core);
        reset_n = 1;

        issue_valid = 1;
        issue_row_id = 1;
        issue_first = 1;
        issue_last = 1;
        issue_parent_valid = 0;
        issue_parent_id = 0;
        drive_all_lanes(1, 0);
        prefetch_valid = 1;
        prefetch_parent_id = 1;
        psum_write_ready = 0;

        repeat (3) begin
            @(posedge clk_core);
            if (issue_ready || prefetch_ready || scratch_read_enable
                    || scratch_write_enable)
                $fatal(1, "stalled RAW request escaped guard");
        end

        @(negedge clk_core);
        psum_write_ready = 1;
        #0;
        if (!issue_ready || !prefetch_ready || !debug_forward_event)
            $fatal(1, "release did not forward the new final value");
        @(posedge clk_core);
        @(negedge clk_core);
        issue_valid = 0;
        prefetch_valid = 0;

        if (parent_queue_occupancy != 1 || reads_seen != 0
                || forward_seen != 1)
            $fatal(1, "release transport mismatch occupancy=%0d reads=%0d fwd=%0d",
                parent_queue_occupancy, reads_seen, forward_seen);

        // Consume the queued row1 as an exact parent.  Correct r2 produces 1;
        // frozen r1's stale-read bug produces the preloaded old value 5.
        issue_valid = 1;
        issue_row_id = 2;
        issue_first = 1;
        issue_last = 1;
        issue_parent_valid = 1;
        issue_parent_id = 1;
        drive_all_lanes(0, 0);
        #0;
        if (!issue_ready)
            $fatal(1, "child exact-parent issue not ready");
        @(posedge clk_core);
        @(negedge clk_core);
        issue_valid = 0;

        repeat (2) @(posedge clk_core);
        if (stalled_cycles != 3 || reads_seen != 0 || forward_seen != 1
                || writes_seen != 2 || child_value_checks != LANES
                || stale_mismatches != 0)
            $fatal(1, "M476r2 evidence mismatch stalled=%0d reads=%0d fwd=%0d writes=%0d checks=%0d stale=%0d",
                stalled_cycles, reads_seen, forward_seen, writes_seen,
                child_value_checks, stale_mismatches);

        $display("PASS M498 segmented-enable stalled_raw_guard stalled=%0d reads=%0d forward=%0d writes=%0d child_checks=%0d stale_mismatches=%0d old=5 new=1",
            stalled_cycles, reads_seen, forward_seen, writes_seen,
            child_value_checks, stale_mismatches);
        $finish;
    end

    initial begin
        #10000;
        $fatal(1, "M479 lane-local timeout");
    end
endmodule
`default_nettype wire
