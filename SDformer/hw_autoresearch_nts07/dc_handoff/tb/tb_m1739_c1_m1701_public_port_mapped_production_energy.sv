`timescale 1ns/1ps
`default_nettype none

// Public-port-only mapped-gate ep34-density-conditioned directed component
// activity workload for the M1701 C1 island.  Only the support-density strata
// come from ep34; residual and psum data are synthetic.  This is not a
// captured-sample, representative-workload, full-network or
// production-inference energy denominator.
//
// This testbench deliberately contains no force/release statement and never
// reads a hierarchical DUT signal.  The only hierarchical path in the whole
// campaign is the UCLI SAIF scope, which observes (but cannot drive) dut.
// A complete 64-row task is loaded through the public prep interface.  Source
// payloads are generated solely from public issue_request_* outputs, and every
// architectural psum commit is checked against an independent sum of the
// original 16-bit row mask.
module tb_m1739_c1_m1701_public_port_mapped_production_energy;
    logic clk_core;
    logic reset_n;

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

    logic [15:0] stimulus_masks [0:63];
    logic [63:0] committed_rows;
    integer error_count;
    integer observed_commits;
    integer measurement_cycles;
    logic measurement_open;
    localparam logic [15:0] TEST_EPOCH = 16'h1739;

    m935_m912_three_stage_exact_parent_match_product_capture_island dut (
        .clk_core(clk_core), .reset_n(reset_n),
        .prep_valid(prep_valid), .prep_ready(prep_ready),
        .prep_task_start(prep_task_start), .prep_task_last(prep_task_last),
        .prep_epoch(prep_epoch), .prep_row_id(prep_row_id),
        .prep_mask(prep_mask), .prep_reserved(prep_reserved),
        .issue_request_valid(issue_request_valid),
        .issue_request_epoch(issue_request_epoch),
        .issue_request_row_id(issue_request_row_id),
        .issue_request_first(issue_request_first),
        .issue_request_last(issue_request_last),
        .issue_request_source_valid(issue_request_source_valid),
        .issue_request_source_index(issue_request_source_index),
        .issue_request_parent_valid(issue_request_parent_valid),
        .issue_request_parent_id(issue_request_parent_id),
        .issue_data_valid(issue_data_valid),
        .issue_data_ready(issue_data_ready),
        .issue_residual_data(issue_residual_data),
        .issue_psum_prior(issue_psum_prior),
        .psum_write_valid(psum_write_valid),
        .psum_write_ready(psum_write_ready),
        .psum_write_address(psum_write_address),
        .psum_write_data(psum_write_data),
        .row_complete_valid(row_complete_valid),
        .row_complete_ready(row_complete_ready),
        .row_complete_id(row_complete_id),
        .task_done_valid(task_done_valid), .task_done_epoch(task_done_epoch),
        .protocol_error(protocol_error), .preprocess_busy(preprocess_busy),
        .execute_busy(execute_busy),
        .active_directory_bank(active_directory_bank),
        .parent_queue_occupancy(parent_queue_occupancy),
        .parent_reserved_occupancy(parent_reserved_occupancy),
        .debug_parent_live_bitmap(debug_parent_live_bitmap),
        .debug_written_bitmap(debug_written_bitmap),
        .debug_scratch_read_event(debug_scratch_read_event),
        .debug_scratch_write_event(debug_scratch_write_event),
        .debug_forward_event(debug_forward_event),
        .debug_read_response_event(debug_read_response_event),
        .debug_dual_enqueue_event(debug_dual_enqueue_event),
        .debug_dead_write_elision_event(debug_dead_write_elision_event),
        .debug_deadline_hold_event(debug_deadline_hold_event),
        .debug_overflow_block_event(debug_overflow_block_event),
        .debug_stalled_raw_event(debug_stalled_raw_event),
        .count_issue_accepts(count_issue_accepts),
        .count_parent_edges(count_parent_edges),
        .count_dead_write_elisions(count_dead_write_elisions),
        .count_macro_reads(count_macro_reads),
        .count_macro_writes(count_macro_writes),
        .count_forwards(count_forwards),
        .count_deadline_holds(count_deadline_holds),
        .count_issue_stalls(count_issue_stalls),
        .count_psum_commits(count_psum_commits),
        .count_row_completions(count_row_completions));

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
        integer result;
        begin
            result = 0;
            for (integer source = 0; source < 16; source = source + 1)
                if (mask[source])
                    result = result + $signed(source_value12(source, lane));
            return result;
        end
    endfunction

    task automatic initialize_masks;
        integer support;
        integer offset;
        logic [31:0] doubled;
        begin
            // M1590's frozen ep34 support ledger has 25,304,213 active rows.
            // Its active-only p25/p50/p75 support popcounts are exactly 1/2/4
            // (ledger SHA is bound by the M1739 source contract).  This
            // directed task covers those three observed density tiers without
            // claiming their empirical frequency or replaying captured data.
            for (integer row = 0; row < 64; row = row + 1) begin
                case (row % 3)
                    0: support = 1;
                    1: support = 2;
                    default: support = 4;
                endcase
                offset = (row / 3) % 16;
                doubled = {16'b0, ((17'h1ffff >> (16 - support)) & 16'hffff)};
                doubled = doubled | (doubled << 16);
                doubled = doubled >> offset;
                stimulus_masks[row] = doubled[15:0];
            end
        end
    endtask

    task automatic clear_prep;
        begin
            prep_valid = 1'b0;
            prep_task_start = 1'b0;
            prep_task_last = 1'b0;
            prep_epoch = '0;
            prep_row_id = '0;
            prep_mask = '0;
            prep_reserved = 4'b0;
        end
    endtask

    task automatic load_public_task;
        begin
            for (integer row = 0; row < 64; row = row + 1) begin
                @(negedge clk_core);
                prep_valid = 1'b1;
                prep_task_start = (row == 0);
                prep_task_last = (row == 63);
                prep_epoch = TEST_EPOCH;
                prep_row_id = row[5:0];
                prep_mask = stimulus_masks[row];
                prep_reserved = 4'b0;
                if (row == 0) begin
                    measurement_open = 1'b1;
                    measurement_cycles = 0;
                    $display("M1739_SAIF_WINDOW_START epoch=%0d", TEST_EPOCH);
                    if ($test$plusargs("M1739_UCLI_SAIF")) $stop;
                end
                while (!prep_ready) @(negedge clk_core);
                @(posedge clk_core);
            end
            @(negedge clk_core);
            clear_prep();
        end
    endtask

    // Payload generation consumes only the public request protocol.  Parent
    // reconstruction remains wholly inside the mapped DUT.
    always_comb begin
        issue_data_valid = issue_request_valid;
        issue_residual_data = '0;
        issue_psum_prior = '0;
        if (issue_request_source_valid)
            for (integer lane = 0; lane < 96; lane = lane + 1)
                issue_residual_data[lane*12 +: 12] =
                    source_value12(issue_request_source_index, lane);
    end

    always @(posedge clk_core or negedge reset_n) begin
        integer got_value;
        integer want_value;
        if (!reset_n) begin
            committed_rows = '0;
            observed_commits = 0;
            error_count = 0;
            measurement_cycles = 0;
        end else begin
            if (measurement_open)
                measurement_cycles = measurement_cycles + 1;
            if (protocol_error) begin
                error_count = error_count + 1;
                $error("M1739 public protocol_error asserted");
            end
            if (psum_write_valid && psum_write_ready) begin
                if (!row_complete_valid || !row_complete_ready
                        || row_complete_id !== psum_write_address) begin
                    error_count = error_count + 1;
                    $error("M1739 architectural commit handshake mismatch");
                end
                if (committed_rows[psum_write_address]) begin
                    error_count = error_count + 1;
                    $error("M1739 duplicate row commit row=%0d",
                        psum_write_address);
                end
                committed_rows[psum_write_address] = 1'b1;
                observed_commits = observed_commits + 1;
                for (integer lane = 0; lane < 96; lane = lane + 1) begin
                    got_value = $signed(psum_write_data[lane*19 +: 19]);
                    want_value = expected_lane(
                        stimulus_masks[psum_write_address], lane);
                    if (got_value != want_value) begin
                        error_count = error_count + 1;
                        $error("M1739 psum mismatch row=%0d lane=%0d got=%0d want=%0d",
                            psum_write_address, lane, got_value, want_value);
                    end
                end
            end
        end
    end

    initial begin
        reset_n = 1'b0;
        psum_write_ready = 1'b1;
        row_complete_ready = 1'b1;
        measurement_open = 1'b0;
        measurement_cycles = 0;
        error_count = 0;
        observed_commits = 0;
        committed_rows = '0;
        clear_prep();
        initialize_masks();
        repeat (8) @(posedge clk_core);
        @(negedge clk_core);
        reset_n = 1'b1;
        repeat (3) @(posedge clk_core);

        load_public_task();
        fork
            begin : done_wait
                wait (task_done_valid && task_done_epoch == TEST_EPOCH);
            end
            begin : watchdog
                repeat (30000) @(posedge clk_core);
                $fatal(1, "M1739 task timeout");
            end
        join_any
        disable fork;
        @(negedge clk_core);
        measurement_open = 1'b0;

        if (error_count != 0 || committed_rows !== 64'hffff_ffff_ffff_ffff
                || observed_commits != 64 || count_psum_commits != 64
                || count_row_completions != 64 || count_parent_edges == 0
                || (count_macro_reads + count_forwards) == 0
                || protocol_error !== 1'b0)
            $fatal(1, "M1739 public-port scoreboard failed errors=%0d commits=%0d rows=%h issue=%0d parents=%0d reads=%0d writes=%0d forwards=%0d",
                error_count, observed_commits, committed_rows,
                count_issue_accepts, count_parent_edges, count_macro_reads,
                count_macro_writes, count_forwards);

        $display("M1739_PUBLIC_COUNTERS cycles=%0d issue_accepts=%0d parent_edges=%0d macro_reads=%0d macro_writes=%0d forwards=%0d dead_write_elisions=%0d psum_commits=%0d row_completions=%0d",
            measurement_cycles, count_issue_accepts, count_parent_edges,
            count_macro_reads, count_macro_writes, count_forwards,
            count_dead_write_elisions, count_psum_commits,
            count_row_completions);
        $display("PASS_M1739_C1_M1701_PUBLIC_PORT_MAPPED_DIRECTED_COMPONENT_ACTIVITY");
        $display("M1739_SAIF_WINDOW_STOP cycles=%0d", measurement_cycles);
        if ($test$plusargs("M1739_UCLI_SAIF")) $stop;
        $finish;
    end
endmodule

`default_nettype wire
