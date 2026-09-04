`timescale 1ns/1ps
`default_nettype none

// M1270/R13 source-only real-M935 integrated protocol harness.
// The sole stimulus path is public prep plus external weight/psum services.
// No parent or child issue-request signal is procedurally overridden.  Frozen
// M528/M935/M1162 and R3 SVA are reused unchanged.  This file is not a launch,
// timing, cycle-speedup, PPA, energy, system, or headline authorization.
module tb_m1270r13_m1162_real_m935_protocol_unit_delay_r13;
    localparam integer PREP_WAIT_LIMIT = 2000;
    localparam integer ISSUE_WAIT_LIMIT = 2000;
    localparam integer SERVICE_WAIT_LIMIT = 100;

    logic clk_core = 1'b0;
    logic reset_n = 1'b0;
    always #1.5 clk_core = ~clk_core;

    logic prep_valid, prep_ready, prep_task_start, prep_task_last;
    logic [15:0] prep_epoch, prep_mask;
    logic [5:0] prep_row_id;
    logic [3:0] prep_reserved;
    logic issue_request_valid, issue_request_first, issue_request_last;
    logic issue_request_source_valid, issue_request_parent_valid;
    logic [15:0] issue_request_epoch;
    logic [5:0] issue_request_row_id, issue_request_parent_id;
    logic [3:0] issue_request_source_index;

    logic weight_req_valid, weight_req_ready;
    logic [15:0] weight_req_epoch;
    logic [5:0] weight_req_row;
    logic [3:0] weight_req_source;
    logic weight_req_source_valid;
    logic weight_rsp_valid, weight_rsp_ready;
    logic [1151:0] weight_data;
    logic psum_req_valid, psum_req_ready;
    logic [15:0] psum_req_epoch;
    logic [5:0] psum_req_addr;
    logic psum_rsp_valid, psum_rsp_ready;
    logic [1823:0] psum_data;

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

    logic request_hold_attack_mode;
    logic weight_service_attack_mode, psum_service_attack_mode;
    logic weight_service_fault, psum_service_fault;
    integer cycle_count;
    integer weight_fire_count, psum_fire_count, response_accept_count;
    integer row_complete_count, task_done_count;
    integer first_response_cycle, second_response_cycle;
    integer oracle_count;
    integer first_beat_count, nonfirst_beat_count, join_hold_count;
    integer normal_retired_beats, normal_response_base;

    m1162_m935_c1_common_charge_protocol_boundary dut (
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
        .weight_read_request_valid(weight_req_valid),
        .weight_read_request_ready(weight_req_ready),
        .weight_read_request_epoch(weight_req_epoch),
        .weight_read_request_row_id(weight_req_row),
        .weight_read_request_source_index(weight_req_source),
        .weight_read_request_source_valid(weight_req_source_valid),
        .weight_read_response_valid(weight_rsp_valid),
        .weight_read_response_ready(weight_rsp_ready),
        .weight_product_residual_data(weight_data),
        .psum_read_request_valid(psum_req_valid),
        .psum_read_request_ready(psum_req_ready),
        .psum_read_request_epoch(psum_req_epoch),
        .psum_read_request_address(psum_req_addr),
        .psum_read_response_valid(psum_rsp_valid),
        .psum_read_response_ready(psum_rsp_ready),
        .psum_read_response_data(psum_data),
        .psum_write_valid(psum_write_valid),
        .psum_write_ready(psum_write_ready),
        .psum_write_address(psum_write_address),
        .psum_write_data(psum_write_data),
        .row_complete_valid(row_complete_valid),
        .row_complete_ready(row_complete_ready),
        .row_complete_id(row_complete_id),
        .task_done_valid(task_done_valid), .task_done_epoch(task_done_epoch),
        .protocol_error(protocol_error),
        .preprocess_busy(preprocess_busy), .execute_busy(execute_busy),
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
        .count_row_completions(count_row_completions)
    );

    m1168r3_m1162_common_charge_protocol_assertions_r3 u_protocol_sva (
        .clk_core(clk_core), .reset_n(reset_n),
        .request_hold_attack_mode(request_hold_attack_mode),
        .weight_service_attack_mode(weight_service_attack_mode),
        .psum_service_attack_mode(psum_service_attack_mode),
        .issue_request_valid(issue_request_valid),
        .issue_request_epoch(issue_request_epoch),
        .issue_request_row_id(issue_request_row_id),
        .issue_request_first(issue_request_first),
        .issue_request_last(issue_request_last),
        .issue_request_source_valid(issue_request_source_valid),
        .issue_request_source_index(issue_request_source_index),
        .issue_request_parent_valid(issue_request_parent_valid),
        .issue_request_parent_id(issue_request_parent_id),
        .weight_request_valid(weight_req_valid),
        .weight_request_ready(weight_req_ready),
        .weight_request_epoch(weight_req_epoch),
        .weight_request_row_id(weight_req_row),
        .weight_request_source_index(weight_req_source),
        .weight_request_source_valid(weight_req_source_valid),
        .weight_response_valid(weight_rsp_valid),
        .weight_response_ready(weight_rsp_ready),
        .weight_response_data(weight_data),
        .psum_request_valid(psum_req_valid),
        .psum_request_ready(psum_req_ready),
        .psum_request_epoch(psum_req_epoch),
        .psum_request_address(psum_req_addr),
        .psum_response_valid(psum_rsp_valid),
        .psum_response_ready(psum_rsp_ready),
        .psum_response_data(psum_data),
        .request_active(dut.request_active_q),
        .weight_request_accepted(dut.weight_request_accepted_q),
        .psum_request_accepted(dut.psum_request_accepted_q),
        .request_first(dut.request_first_q),
        .core_issue_data_valid(dut.core_issue_data_valid),
        .core_issue_data_ready(dut.core_issue_data_ready),
        .response_accept(dut.response_accept_w),
        .boundary_fault(dut.boundary_fault_q)
    );

    m1168r3_service_assumption_checker u_service_checker (
        .clk_core(clk_core), .reset_n(reset_n),
        .weight_response_valid(weight_rsp_valid),
        .weight_response_ready(weight_rsp_ready),
        .weight_response_data(weight_data),
        .psum_response_valid(psum_rsp_valid),
        .psum_response_ready(psum_rsp_ready),
        .psum_response_data(psum_data),
        .weight_service_fault(weight_service_fault),
        .psum_service_fault(psum_service_fault)
    );

    always @(posedge clk_core) begin
        cycle_count = cycle_count + 1;
        if (reset_n && weight_req_valid && weight_req_ready)
            weight_fire_count = weight_fire_count + 1;
        if (reset_n && psum_req_valid && psum_req_ready)
            psum_fire_count = psum_fire_count + 1;
        if (reset_n && dut.response_accept_w) begin
            response_accept_count = response_accept_count + 1;
            if (response_accept_count == normal_response_base + 1)
                first_response_cycle = cycle_count;
            if (response_accept_count == normal_response_base + 2)
                second_response_cycle = cycle_count;
        end
        if (reset_n && row_complete_valid && row_complete_ready)
            row_complete_count = row_complete_count + 1;
        if (reset_n && task_done_valid)
            task_done_count = task_done_count + 1;
    end

    // Every dynamic oracle passes all relevant raw operands through this one
    // printer before it may fail.  A log can therefore distinguish count,
    // tuple, response-join, fault, and completion causes without a waveform.
    task automatic oracle(
        input string site,
        input logic condition,
        input integer beat,
        input integer expected_first,
        input integer weight_delta,
        input integer psum_delta,
        input integer response_delta
    );
        begin
            oracle_count = oracle_count + 1;
            $display("ORACLE_M1270R13 site=%s pass=%0d beat=%0d expected_first=%0d weight_delta=%0d psum_delta=%0d response_delta=%0d cycle=%0d issue_vfl=%0d%0d%0d source=%0d weight_vr=%0d%0d psum_vr=%0d%0d rsp_vr=%0d%0d/%0d%0d request_active=%0d accepted_wp=%0d%0d boundary_fault=%0d core_fault=%0d m935_fault=%0d row_done=%0d task_done=%0d",
                site, condition, beat, expected_first, weight_delta,
                psum_delta, response_delta, cycle_count,
                issue_request_valid, issue_request_first, issue_request_last,
                issue_request_source_index,
                weight_req_valid, weight_req_ready,
                psum_req_valid, psum_req_ready,
                weight_rsp_valid, weight_rsp_ready,
                psum_rsp_valid, psum_rsp_ready,
                dut.request_active_q, dut.weight_request_accepted_q,
                dut.psum_request_accepted_q, dut.boundary_fault_q,
                dut.core_protocol_error, dut.u_frozen_m935.fault_q,
                row_complete_count, task_done_count);
            $fflush();
            if (condition !== 1'b1)
                $fatal(1, "M1270R13 oracle failed site=%s", site);
        end
    endtask

    task automatic clear_public_drivers;
        begin
            prep_valid = 1'b0;
            prep_task_start = 1'b0;
            prep_task_last = 1'b0;
            prep_epoch = '0;
            prep_row_id = '0;
            prep_mask = '0;
            prep_reserved = '0;
            weight_req_ready = 1'b0;
            psum_req_ready = 1'b0;
            weight_rsp_valid = 1'b0;
            psum_rsp_valid = 1'b0;
            weight_data = '0;
            psum_data = '0;
            psum_write_ready = 1'b1;
            row_complete_ready = 1'b1;
            request_hold_attack_mode = 1'b0;
            weight_service_attack_mode = 1'b0;
            psum_service_attack_mode = 1'b0;
        end
    endtask

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            clear_public_drivers();
            reset_n = 1'b0;
            repeat (3) @(posedge clk_core);
            #1ps;
            oracle("reset_state",
                !dut.request_active_q && !dut.weight_request_accepted_q
                    && !dut.psum_request_accepted_q
                    && !dut.boundary_fault_q && !dut.core_protocol_error
                    && !dut.u_frozen_m935.fault_q,
                -1, -1, 0, 0, 0);
            @(negedge clk_core);
            reset_n = 1'b1;
            @(posedge clk_core); #1ps;
            oracle("reset_release",
                !protocol_error && !weight_service_fault
                    && !psum_service_fault,
                -1, -1, 0, 0, 0);
        end
    endtask

    // Frozen normal workload shape: one 64-row task, row zero mask 0003,
    // yielding exactly two real M935 source beats (first then non-first).
    task automatic load_normal_task(input logic [15:0] epoch);
        integer watchdog;
        begin
            for (integer row = 0; row < 64; row = row + 1) begin
                @(negedge clk_core);
                prep_valid = 1'b1;
                prep_task_start = (row == 0);
                prep_task_last = (row == 63);
                prep_epoch = epoch;
                prep_row_id = row[5:0];
                prep_mask = (row == 0) ? 16'h0003 : 16'h0000;
                prep_reserved = 4'b0;
                watchdog = 0;
                while (!prep_ready && watchdog < PREP_WAIT_LIMIT) begin
                    @(negedge clk_core);
                    watchdog = watchdog + 1;
                end
                oracle("prep_ready", prep_ready, row, -1,
                    watchdog, 0, 0);
                @(posedge clk_core);
            end
            @(negedge clk_core);
            prep_valid = 1'b0;
            prep_task_start = 1'b0;
            prep_task_last = 1'b0;
        end
    endtask

    task automatic serve_real_m935_beat(
        input logic expect_first,
        input integer beat_index
    );
        integer watchdog, w0, p0, response0;
        logic [3:0] served_source;
        begin
            #1ps;
            oracle("beat_entry",
                normal_retired_beats == beat_index
                    && response_accept_count
                        == normal_response_base + beat_index
                    && !weight_rsp_valid && !psum_rsp_valid,
                beat_index, expect_first,
                normal_retired_beats,
                response_accept_count - normal_response_base, 0);

            watchdog = 0;
            while (!weight_req_valid && watchdog < ISSUE_WAIT_LIMIT) begin
                @(posedge clk_core); #1ps;
                watchdog = watchdog + 1;
            end
            oracle("real_issue_tuple",
                weight_req_valid && issue_request_valid
                    && issue_request_first == expect_first
                    && issue_request_source_valid
                    && issue_request_source_index == beat_index[3:0]
                    && issue_request_last == (beat_index == 1)
                    && !issue_request_parent_valid,
                beat_index, expect_first, watchdog,
                issue_request_source_index, issue_request_last);
            served_source = issue_request_source_index;

            w0 = weight_fire_count;
            p0 = psum_fire_count;
            @(negedge clk_core);
            weight_req_ready = 1'b1;
            psum_req_ready = 1'b1;
            watchdog = 0;
            while ((weight_fire_count != w0 + 1
                    || psum_fire_count != p0 + expect_first)
                    && watchdog < SERVICE_WAIT_LIMIT) begin
                @(posedge clk_core); #1ps;
                watchdog = watchdog + 1;
                oracle("request_no_overshoot",
                    weight_fire_count <= w0 + 1
                        && psum_fire_count <= p0 + expect_first,
                    beat_index, expect_first,
                    weight_fire_count - w0, psum_fire_count - p0, 0);
            end
            oracle("request_exact_count",
                weight_fire_count == w0 + 1
                    && psum_fire_count == p0 + expect_first,
                beat_index, expect_first,
                weight_fire_count - w0, psum_fire_count - p0, 0);
            if (expect_first)
                first_beat_count = first_beat_count + 1;
            else
                nonfirst_beat_count = nonfirst_beat_count + 1;

            @(negedge clk_core);
            weight_req_ready = 1'b0;
            psum_req_ready = 1'b0;
            #1ps;
            oracle("request_retired",
                !weight_req_ready && !psum_req_ready
                    && weight_fire_count == w0 + 1
                    && psum_fire_count == p0 + expect_first,
                beat_index, expect_first,
                weight_fire_count - w0, psum_fire_count - p0, 0);

            response0 = response_accept_count;
            weight_rsp_valid = 1'b1;
            weight_data = '0;
            psum_rsp_valid = 1'b0;
            psum_data = '0;

            if (expect_first) begin
                // Two sampled cycles of response skew prove the atomic join:
                // weight alone is held and neither service response is taken.
                repeat (2) begin
                    @(posedge clk_core); #1ps;
                    oracle("first_weight_only_join_hold",
                        !weight_rsp_ready && !psum_rsp_ready
                            && response_accept_count == response0
                            && weight_rsp_valid && !psum_rsp_valid
                            && weight_data === '0 && psum_data === '0,
                        beat_index, expect_first,
                        weight_fire_count - w0, psum_fire_count - p0,
                        response_accept_count - response0);
                    join_hold_count = join_hold_count + 1;
                end
                @(negedge clk_core);
                psum_rsp_valid = 1'b1;
            end

            watchdog = 0;
            while (response_accept_count != response0 + 1
                    && watchdog < SERVICE_WAIT_LIMIT) begin
                @(posedge clk_core); #1ps;
                watchdog = watchdog + 1;
                oracle("response_stable_until_accept",
                    response_accept_count <= response0 + 1
                        && weight_rsp_valid && weight_data === '0
                        && psum_rsp_valid === expect_first
                        && psum_data === '0
                        && (!expect_first || psum_rsp_ready == weight_rsp_ready)
                        && (expect_first || !psum_rsp_ready),
                    beat_index, expect_first,
                    weight_fire_count - w0, psum_fire_count - p0,
                    response_accept_count - response0);
            end
            oracle("response_exact_accept",
                response_accept_count == response0 + 1,
                beat_index, expect_first,
                weight_fire_count - w0, psum_fire_count - p0,
                response_accept_count - response0);

            @(negedge clk_core);
            weight_rsp_valid = 1'b0;
            psum_rsp_valid = 1'b0;
            #1ps;
            oracle("response_retired",
                !weight_rsp_valid && !psum_rsp_valid
                    && response_accept_count == response0 + 1,
                beat_index, expect_first,
                weight_fire_count - w0, psum_fire_count - p0,
                response_accept_count - response0);
            normal_retired_beats = normal_retired_beats + 1;

            @(posedge clk_core); #1ps;
            oracle("wrapper_retired_or_next_real_tuple",
                response_accept_count
                        == normal_response_base + normal_retired_beats
                    && !weight_rsp_valid && !psum_rsp_valid
                    && !$isunknown(dut.request_active_q)
                    && (!dut.request_active_q
                        || (issue_request_valid
                            && !$isunknown({issue_request_first,
                                issue_request_source_index,
                                dut.request_first_q,
                                dut.request_source_index_q,
                                dut.weight_request_accepted_q,
                                dut.psum_request_accepted_q})
                            && dut.weight_request_accepted_q == 1'b0
                            && dut.psum_request_accepted_q
                                == !dut.request_first_q
                            && dut.request_first_q == issue_request_first
                            && dut.request_source_index_q
                                == issue_request_source_index
                            && dut.request_source_index_q
                                != served_source)),
                beat_index, expect_first,
                weight_fire_count - w0, psum_fire_count - p0,
                response_accept_count - normal_response_base);
        end
    endtask

    task automatic real_m935_completion;
        integer issue0, row0, done0, watchdog;
        begin
            reset_dut();
            oracle("attack_masks_clear",
                !request_hold_attack_mode && !weight_service_attack_mode
                    && !psum_service_attack_mode,
                -1, -1, 0, 0, 0);
            issue0 = count_issue_accepts;
            row0 = row_complete_count;
            done0 = task_done_count;
            normal_response_base = response_accept_count;
            normal_retired_beats = 0;
            load_normal_task(16'h9001);
            serve_real_m935_beat(1'b1, 0);
            serve_real_m935_beat(1'b0, 1);

            watchdog = 0;
            while (task_done_count == done0 && watchdog < ISSUE_WAIT_LIMIT) begin
                @(posedge clk_core); #1ps;
                watchdog = watchdog + 1;
            end
            oracle("row_task_completion",
                task_done_count == done0 + 1
                    && row_complete_count == row0 + 1
                    && count_issue_accepts == issue0 + 2
                    && count_psum_commits == 1
                    && count_row_completions == 1
                    && task_done_epoch == 16'h9001
                    && !protocol_error && !dut.boundary_fault_q
                    && !dut.core_protocol_error
                    && !dut.u_frozen_m935.fault_q
                    && !weight_service_fault && !psum_service_fault,
                2, -1, count_issue_accepts - issue0,
                row_complete_count - row0, task_done_count - done0);
            oracle("ii_ge_2",
                first_response_cycle >= 0
                    && second_response_cycle - first_response_cycle >= 2,
                2, -1, first_response_cycle, second_response_cycle,
                second_response_cycle - first_response_cycle);
        end
    endtask

    initial begin
        cycle_count = 0;
        weight_fire_count = 0;
        psum_fire_count = 0;
        response_accept_count = 0;
        row_complete_count = 0;
        task_done_count = 0;
        first_response_cycle = -1;
        second_response_cycle = -1;
        oracle_count = 0;
        first_beat_count = 0;
        nonfirst_beat_count = 0;
        join_hold_count = 0;
        normal_retired_beats = 0;
        normal_response_base = 0;
        clear_public_drivers();

        $display("PHASE_M1270R13_REAL_M935_INTEGRATED_ENTER"); $fflush();
        real_m935_completion();
        $display("PHASE_M1270R13_REAL_M935_INTEGRATED_COMPLETE"); $fflush();
        oracle("coverage_minima",
            first_beat_count == 1 && nonfirst_beat_count == 1
                && join_hold_count == 2 && normal_retired_beats == 2
                && oracle_count >= 80,
            2, -1, first_beat_count, nonfirst_beat_count,
            join_hold_count);

        $display("COVERAGE_M1270R13_REAL_M935 first_beats=%0d nonfirst_beats=%0d join_hold_cycles=%0d issue_accepts=%0d psum_reads=%0d row_completions=%0d task_completions=%0d response_cycle_gap=%0d oracle_records=%0d parent_issue_override=0 child_issue_override=0",
            first_beat_count, nonfirst_beat_count, join_hold_count,
            count_issue_accepts, psum_fire_count, row_complete_count,
            task_done_count, second_response_cycle - first_response_cycle,
            oracle_count);
        $display("PASS_M1270R13_REAL_M935_INTEGRATED_PROTOCOL_SOURCE_CANDIDATE real_m935=true parent_issue_override=0 child_issue_override=0 first_beats=1 nonfirst_beats=1 weight_requests=2 psum_requests=1 response_join_hold_cycles=2 ii_ge_2=true row_completions=1 task_completions=1 boundary_fault=0 core_fault=0 m935_fault=0 every_oracle_operands=true zero_sva_failures_required=true functional_vcs=false timing_verified=false cycles_measured=false speedup=false ppa=false energy=false system_speedup=false headline=false");
        $finish;
    end
endmodule

`default_nettype wire
