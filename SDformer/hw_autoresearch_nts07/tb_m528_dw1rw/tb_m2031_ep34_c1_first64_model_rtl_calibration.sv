`timescale 1ns/1ps
`default_nettype none

// One real ep34 64-row tile used only to calibrate the frozen C1 CPU
// recurrence against the RTL service-event counters.  It does not turn the
// 51.84M-row CPU cycle ratio into an RTL speedup or a system result.
module tb_m2031_ep34_c1_first64_model_rtl_calibration;
    localparam string FIXTURE = "/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/tb_m528_dw1rw/fixtures/m2031_ep34_c1_first64_support16.memh";

    logic clk_core, reset_n;
    logic prep_valid, prep_ready, prep_task_start, prep_task_last;
    logic [15:0] prep_epoch, prep_mask;
    logic [5:0] prep_row_id;
    logic [3:0] prep_reserved;
    logic issue_request_valid, issue_request_first, issue_request_last;
    logic issue_request_source_valid, issue_request_parent_valid;
    logic [15:0] issue_request_epoch;
    logic [5:0] issue_request_row_id, issue_request_parent_id;
    logic [3:0] issue_request_source_index;
    logic issue_data_valid, issue_data_ready;
    logic [1151:0] issue_residual_data;
    logic [1823:0] issue_psum_prior;
    logic psum_write_valid, psum_write_ready;
    logic [5:0] psum_write_address;
    logic [1823:0] psum_write_data;
    logic row_complete_valid, row_complete_ready;
    logic [5:0] row_complete_id;
    logic task_done_valid;
    logic [15:0] task_done_epoch;
    logic protocol_error, preprocess_busy, execute_busy;
    logic active_directory_bank;
    logic [1:0] parent_queue_occupancy;
    logic [2:0] parent_reserved_occupancy;
    logic [63:0] debug_parent_live_bitmap, debug_written_bitmap;
    logic debug_scratch_read_event, debug_scratch_write_event;
    logic debug_forward_event, debug_read_response_event;
    logic debug_dual_enqueue_event, debug_dead_write_elision_event;
    logic debug_deadline_hold_event, debug_overflow_block_event;
    logic debug_stalled_raw_event;
    logic [63:0] count_issue_accepts, count_parent_edges;
    logic [63:0] count_dead_write_elisions, count_macro_reads;
    logic [63:0] count_macro_writes, count_forwards;
    logic [63:0] count_deadline_holds, count_issue_stalls;
    logic [63:0] count_psum_commits, count_row_completions;

    logic [31:0] fixture_word [0:63];
    logic [15:0] fixture_mask [0:63];
    integer commit_checks;
    integer execute_cycles;
    logic execute_seen;

    m528_dead_write_only_1rw_product_capture_island_r2 dut (.*);

    initial clk_core = 1'b0;
    always #1.5 clk_core = ~clk_core;

    function automatic logic signed [11:0] source_value12(
        input integer source, input integer lane
    );
        logic signed [7:0] value8;
        integer raw;
        begin
            raw = ((source * 7 + lane * 3) % 17) - 8;
            value8 = raw;
            source_value12 = {{4{value8[7]}}, value8};
        end
    endfunction

    function automatic integer expected_lane(
        input logic [15:0] mask, input integer lane
    );
        integer value;
        begin
            value = 0;
            for (integer source = 0; source < 16; source = source + 1)
                if (mask[source])
                    value = value + $signed(source_value12(source, lane));
            return value;
        end
    endfunction

    task automatic clear_prep;
        begin
            prep_valid = 1'b0;
            prep_task_start = 1'b0;
            prep_task_last = 1'b0;
            prep_epoch = '0;
            prep_row_id = '0;
            prep_mask = '0;
            prep_reserved = '0;
        end
    endtask

    task automatic reset_dut;
        begin
            clear_prep();
            reset_n = 1'b0;
            repeat (5) @(posedge clk_core);
            @(negedge clk_core);
            reset_n = 1'b1;
            repeat (2) @(posedge clk_core);
        end
    endtask

    task automatic load_real_tile;
        begin
            for (integer row = 0; row < 64; row = row + 1) begin
                @(negedge clk_core);
                prep_valid = 1'b1;
                prep_task_start = (row == 0);
                prep_task_last = (row == 63);
                prep_epoch = 16'd34;
                prep_row_id = row[5:0];
                prep_mask = fixture_mask[row];
                prep_reserved = 4'b0;
                while (!prep_ready) @(negedge clk_core);
                @(posedge clk_core);
            end
            @(negedge clk_core);
            clear_prep();
        end
    endtask

    always_comb begin
        issue_data_valid = issue_request_valid;
        issue_residual_data = '0;
        issue_psum_prior = '0;
        for (integer lane = 0; lane < 96; lane = lane + 1)
            if (issue_request_source_valid)
                issue_residual_data[lane*12 +: 12] =
                    source_value12(issue_request_source_index, lane);
    end

    always_ff @(posedge clk_core or negedge reset_n) begin
        if (!reset_n) begin
            commit_checks <= 0;
            execute_cycles <= 0;
            execute_seen <= 1'b0;
        end else begin
            if (execute_busy) begin
                execute_cycles <= execute_cycles + 1;
                execute_seen <= 1'b1;
            end
            if (protocol_error)
                $fatal(1, "real ep34 tile raised protocol_error");
            if (psum_write_valid && psum_write_ready) begin
                if (!row_complete_valid || !row_complete_ready
                        || row_complete_id != psum_write_address)
                    $fatal(1, "non-atomic completion row=%0d", psum_write_address);
                for (integer lane = 0; lane < 96; lane = lane + 1)
                    if ($signed(psum_write_data[lane*19 +: 19])
                            !== expected_lane(fixture_mask[psum_write_address], lane))
                        $fatal(1, "numeric mismatch row=%0d lane=%0d got=%0d expected=%0d",
                            psum_write_address, lane,
                            $signed(psum_write_data[lane*19 +: 19]),
                            expected_lane(fixture_mask[psum_write_address], lane));
                commit_checks <= commit_checks + 1;
            end
        end
    end

    initial begin
        reset_n = 1'b0;
        psum_write_ready = 1'b1;
        row_complete_ready = 1'b1;
        clear_prep();
        $readmemh(FIXTURE, fixture_word);
        for (integer row = 0; row < 64; row = row + 1)
            fixture_mask[row] = fixture_word[row][15:0];

        reset_dut();
        load_real_tile();
        while (!(task_done_valid && task_done_epoch == 16'd34)) begin
            @(posedge clk_core);
            if (execute_cycles > 2000)
                $fatal(1, "real ep34 tile timeout");
        end
        @(negedge clk_core);

        if (!execute_seen || commit_checks != 64)
            $fatal(1, "coverage mismatch execute=%0d commits=%0d",
                execute_seen, commit_checks);
        if (count_issue_accepts != 64'd196
                || count_parent_edges != 64'd58
                || count_dead_write_elisions != 64'd31
                || count_macro_reads != 64'd54
                || count_macro_writes != 64'd33
                || count_forwards != 64'd4
                || count_deadline_holds != 64'd6
                || count_issue_stalls != 64'd14
                || count_psum_commits != 64'd64
                || count_row_completions != 64'd64)
            $fatal(1, "model/RTL counter mismatch issue=%0d edges=%0d elide=%0d read=%0d write=%0d fwd=%0d hold=%0d stall=%0d psum=%0d row=%0d",
                count_issue_accepts, count_parent_edges,
                count_dead_write_elisions, count_macro_reads,
                count_macro_writes, count_forwards, count_deadline_holds,
                count_issue_stalls, count_psum_commits,
                count_row_completions);

        $display("PASS_M2031_EP34_C1_FIRST64_MODEL_RTL_CALIBRATION rows=64 active=64 input_nnz=565 residual_nnz=192 exact_parent_rows=4 issue=196 parent_edges=58 dead_elisions=31 macro_reads=54 macro_writes=33 forwards=4 deadline_holds=6 stalls=14 psum_commits=64 row_completions=64 numeric_commits=64 rtl_cycle_speedup=false full_network=false system_speedup=false");
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "global watchdog expired");
    end
endmodule

`default_nettype wire
