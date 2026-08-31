`timescale 1ns/1ps
`default_nettype none

// Additive M1168R3 negative-test-isolation repair.  R2 compiled, elaborated and
// linked but an intentional request attack reached a normal hold assertion and
// a service mutation was sampled on an ambiguous boundary.  R3 keeps all legal
// assertions enabled, adds narrow per-class attack windows, and samples the
// independent service checker after NBA completion.  This source is inert until
// a fresh different-author hammer and separately sealed release authorize R3.
module tb_m1168r3_m1162_common_charge_protocol_unit_delay_r3;
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
    integer last_response_accept_cycle;
    integer cov_weight_first, cov_psum_first, cov_weight_rsp_first;
    integer cov_psum_rsp_first, cov_long_request_stall;
    integer cov_long_response_backpressure, cov_nonfirst, cov_ii2;
    integer cov_reset_partial, cov_reset_complete, cov_reset_skew;
    integer cov_unsolicited_weight, cov_unsolicited_psum;
    integer cov_same_cycle_early, cov_duplicate_response;
    integer cov_cancel, cov_tuple_mutation, cov_nonfirst_psum;
    integer cov_weight_payload_mutation, cov_psum_valid_drop;
    integer cov_no_duplicate_request, cov_random_transactions;
    integer cov_normal_issue, cov_normal_row, cov_normal_task;
    integer cov_legal_masks_clear, cov_request_attack_windows;
    integer cov_weight_service_attack_windows, cov_psum_service_attack_windows;
    integer unsigned prng_q;

    // LRM-legal procedural-force staging.  These variables have static module
    // lifetime; none is an automatic task formal.  Every call assigns all five
    // fields before the hierarchical DUT force statements below.  Calls are
    // sequential and release_request ends the prior force before reuse.
    logic force_stage_first_q, force_stage_last_q;
    logic [15:0] force_stage_epoch_q;
    logic [5:0] force_stage_row_q;
    logic [3:0] force_stage_source_q;

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
            last_response_accept_cycle = cycle_count;
        end
        if (reset_n && row_complete_valid && row_complete_ready)
            row_complete_count = row_complete_count + 1;
        if (reset_n && task_done_valid)
            task_done_count = task_done_count + 1;
    end

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

    task automatic force_request(
        input logic first,
        input logic last,
        input logic [15:0] epoch,
        input logic [5:0] row,
        input logic [3:0] source
    );
        begin
            force_stage_first_q = first;
            force_stage_last_q = last;
            force_stage_epoch_q = epoch;
            force_stage_row_q = row;
            force_stage_source_q = source;
            force dut.issue_request_valid = 1'b1;
            force dut.issue_request_epoch = force_stage_epoch_q;
            force dut.issue_request_row_id = force_stage_row_q;
            force dut.issue_request_first = force_stage_first_q;
            force dut.issue_request_last = force_stage_last_q;
            force dut.issue_request_source_valid = 1'b1;
            force dut.issue_request_source_index = force_stage_source_q;
            force dut.issue_request_parent_valid = 1'b0;
            force dut.issue_request_parent_id = 6'b0;
            force dut.core_issue_data_ready = 1'b1;
        end
    endtask

    task automatic release_request;
        begin
            release dut.issue_request_valid;
            release dut.issue_request_epoch;
            release dut.issue_request_row_id;
            release dut.issue_request_first;
            release dut.issue_request_last;
            release dut.issue_request_source_valid;
            release dut.issue_request_source_index;
            release dut.issue_request_parent_valid;
            release dut.issue_request_parent_id;
            release dut.core_issue_data_ready;
        end
    endtask

    task automatic reset_dut;
        begin
            release_request();
            @(negedge clk_core);
            clear_public_drivers();
            reset_n = 1'b0;
            repeat (3) @(posedge clk_core);
            #1ps;
            if (dut.request_active_q || dut.weight_request_accepted_q
                    || dut.psum_request_accepted_q || dut.boundary_fault_q)
                $fatal(1, "reset failed to clear protocol state");
            @(negedge clk_core);
            reset_n = 1'b1;
            @(posedge clk_core);
            #1ps;
            if (protocol_error || weight_service_fault || psum_service_fault)
                $fatal(1, "reset release retained a sticky fault");
        end
    endtask

    // Every legal directed/random/frozen-M935 transaction calls this task after
    // reset.  Thus a PASS proves all three negative-test masks were low, rather
    // than merely assuming the masks did not leak out of an attack window.
    task automatic require_legal_masks_clear(input integer legal_case_id);
        begin
            #1ps;
            if (request_hold_attack_mode || weight_service_attack_mode
                    || psum_service_attack_mode)
                $fatal(1, "legal case %0d entered with an attack mask", legal_case_id);
            cov_legal_masks_clear = cov_legal_masks_clear + 1;
        end
    endtask

    task automatic require_sticky_protocol_fault(input integer attack_id);
        begin
            #1ps;
            if (!protocol_error || !dut.boundary_fault_q)
                $fatal(1, "attack %0d did not raise boundary sticky fault",
                    attack_id);
            @(posedge clk_core);
            #1ps;
            if (!protocol_error || !dut.boundary_fault_q)
                $fatal(1, "attack %0d boundary fault was not sticky", attack_id);
        end
    endtask

    task automatic complete_first_response;
        begin
            @(negedge clk_core);
            weight_rsp_valid = 1'b1;
            psum_rsp_valid = 1'b1;
            @(posedge clk_core);
            #1ps;
            @(negedge clk_core);
            weight_rsp_valid = 1'b0;
            psum_rsp_valid = 1'b0;
        end
    endtask

    task automatic directed_weight_first;
        integer w0, p0;
        begin
            reset_dut();
            require_legal_masks_clear(1);
            w0 = weight_fire_count;
            p0 = psum_fire_count;
            @(negedge clk_core);
            force_request(1'b1, 1'b0, 16'h1101, 6'd9, 4'd3);
            weight_req_ready = 1'b1;
            psum_req_ready = 1'b0;
            @(posedge clk_core); #1ps;
            if (weight_fire_count != w0 + 1 || psum_fire_count != p0)
                $fatal(1, "weight-first initial fire mismatch");
            cov_weight_first = cov_weight_first + 1;
            @(negedge clk_core);
            weight_rsp_valid = 1'b1;
            weight_data = {96{12'h031}};
            repeat (5) begin
                @(posedge clk_core); #1ps;
                if (weight_req_valid || !psum_req_valid || weight_rsp_ready)
                    $fatal(1, "weight-first suppression/join failure");
                cov_long_request_stall = cov_long_request_stall + 1;
            end
            @(negedge clk_core);
            psum_req_ready = 1'b1;
            @(posedge clk_core); #1ps;
            if (psum_fire_count != p0 + 1 || weight_fire_count != w0 + 1)
                $fatal(1, "weight-first peer fire or duplicate mismatch");
            cov_no_duplicate_request = cov_no_duplicate_request + 1;
            @(negedge clk_core);
            psum_rsp_valid = 1'b1;
            psum_data = {96{19'h00001}};
            @(posedge clk_core); #1ps;
            cov_weight_rsp_first = cov_weight_rsp_first + 1;
            @(negedge clk_core);
            weight_rsp_valid = 1'b0;
            psum_rsp_valid = 1'b0;
            release_request();
        end
    endtask

    task automatic directed_psum_first_and_backpressure;
        integer w0, p0;
        begin
            reset_dut();
            require_legal_masks_clear(2);
            w0 = weight_fire_count;
            p0 = psum_fire_count;
            @(negedge clk_core);
            force_request(1'b1, 1'b0, 16'h2202, 6'd10, 4'd4);
            weight_req_ready = 1'b0;
            psum_req_ready = 1'b1;
            @(posedge clk_core); #1ps;
            cov_psum_first = cov_psum_first + 1;
            @(negedge clk_core);
            psum_rsp_valid = 1'b1;
            psum_data = {96{19'h00002}};
            repeat (5) begin
                @(posedge clk_core); #1ps;
                if (!weight_req_valid || psum_req_valid || psum_rsp_ready)
                    $fatal(1, "psum-first suppression/join failure");
            end
            @(negedge clk_core);
            weight_req_ready = 1'b1;
            @(posedge clk_core); #1ps;
            if (weight_fire_count != w0 + 1 || psum_fire_count != p0 + 1)
                $fatal(1, "psum-first request counts mismatch");
            @(negedge clk_core);
            weight_rsp_valid = 1'b1;
            weight_data = {96{12'h012}};
            force dut.core_issue_data_ready = 1'b0;
            repeat (5) begin
                @(posedge clk_core); #1ps;
                if (weight_rsp_ready || psum_rsp_ready)
                    $fatal(1, "response consumed under core backpressure");
                if (weight_data != {96{12'h012}}
                        || psum_data != {96{19'h00002}})
                    $fatal(1, "held response payload changed");
                cov_long_response_backpressure =
                    cov_long_response_backpressure + 1;
            end
            force dut.core_issue_data_ready = 1'b1;
            @(posedge clk_core); #1ps;
            cov_psum_rsp_first = cov_psum_rsp_first + 1;
            @(negedge clk_core);
            weight_rsp_valid = 1'b0;
            psum_rsp_valid = 1'b0;
            release_request();
        end
    endtask

    task automatic directed_nonfirst;
        integer w0, p0;
        begin
            reset_dut();
            require_legal_masks_clear(3);
            w0 = weight_fire_count;
            p0 = psum_fire_count;
            @(negedge clk_core);
            force_request(1'b0, 1'b1, 16'h3303, 6'd11, 4'd7);
            weight_req_ready = 1'b1;
            psum_req_ready = 1'b1;
            @(posedge clk_core); #1ps;
            if (weight_fire_count != w0 + 1 || psum_fire_count != p0
                    || psum_req_valid)
                $fatal(1, "non-first beat issued a psum request");
            @(negedge clk_core);
            weight_rsp_valid = 1'b1;
            weight_data = {96{12'h003}};
            @(posedge clk_core); #1ps;
            if (psum_rsp_ready)
                $fatal(1, "non-first beat consumed psum response");
            cov_nonfirst = cov_nonfirst + 1;
            @(negedge clk_core);
            weight_rsp_valid = 1'b0;
            release_request();
        end
    endtask

    task automatic directed_ii2;
        integer first_accept_cycle, second_accept_cycle;
        begin
            reset_dut();
            require_legal_masks_clear(4);
            @(negedge clk_core);
            force_request(1'b1, 1'b0, 16'h4404, 6'd12, 4'd1);
            weight_req_ready = 1'b1;
            psum_req_ready = 1'b1;
            @(posedge clk_core); #1ps;
            @(negedge clk_core);
            weight_rsp_valid = 1'b1;
            psum_rsp_valid = 1'b1;
            @(posedge clk_core); #1ps;
            first_accept_cycle = last_response_accept_cycle;
            @(negedge clk_core);
            weight_rsp_valid = 1'b0;
            psum_rsp_valid = 1'b0;
            force dut.issue_request_epoch = 16'h4405;
            force dut.issue_request_row_id = 6'd13;
            force dut.issue_request_source_index = 4'd2;
            @(posedge clk_core); #1ps;
            @(negedge clk_core);
            weight_rsp_valid = 1'b1;
            psum_rsp_valid = 1'b1;
            @(posedge clk_core); #1ps;
            second_accept_cycle = last_response_accept_cycle;
            if (second_accept_cycle - first_accept_cycle != 2)
                $fatal(1, "zero-stall response II=%0d expected=2",
                    second_accept_cycle - first_accept_cycle);
            cov_ii2 = cov_ii2 + 1;
            @(negedge clk_core);
            weight_rsp_valid = 1'b0;
            psum_rsp_valid = 1'b0;
            release_request();
        end
    endtask

    task automatic reset_pending_cases;
        begin
            // Request-partial reset.
            reset_dut();
            @(negedge clk_core);
            force_request(1'b1, 1'b0, 16'h5101, 6'd21, 4'd1);
            weight_req_ready = 1'b1;
            psum_req_ready = 1'b0;
            @(posedge clk_core); #1ps;
            reset_n = 1'b0; #1ps;
            if (dut.request_active_q) $fatal(1, "partial reset did not cancel");
            cov_reset_partial = cov_reset_partial + 1;

            // Request-complete, no-response reset.
            release_request();
            reset_dut();
            @(negedge clk_core);
            force_request(1'b1, 1'b0, 16'h5102, 6'd22, 4'd2);
            weight_req_ready = 1'b1;
            psum_req_ready = 1'b1;
            @(posedge clk_core); #1ps;
            if (!dut.weight_request_accepted_q || !dut.psum_request_accepted_q)
                $fatal(1, "request-complete reset precondition absent");
            reset_n = 1'b0; #1ps;
            if (dut.request_active_q) $fatal(1, "complete reset did not cancel");
            cov_reset_complete = cov_reset_complete + 1;

            // One held response after both accepts.
            release_request();
            reset_dut();
            @(negedge clk_core);
            force_request(1'b1, 1'b0, 16'h5103, 6'd23, 4'd3);
            weight_req_ready = 1'b1;
            psum_req_ready = 1'b1;
            @(posedge clk_core); #1ps;
            @(negedge clk_core);
            weight_rsp_valid = 1'b1;
            @(posedge clk_core); #1ps;
            if (weight_rsp_ready) $fatal(1, "skew response was consumed alone");
            reset_n = 1'b0; #1ps;
            if (dut.request_active_q) $fatal(1, "skew reset did not cancel");
            weight_rsp_valid = 1'b0;
            cov_reset_skew = cov_reset_skew + 1;
            release_request();
        end
    endtask

    task automatic sticky_fault_attacks;
        begin
            // Unsolicited weight response.
            reset_dut();
            @(negedge clk_core); weight_rsp_valid = 1'b1;
            @(posedge clk_core); require_sticky_protocol_fault(1);
            cov_unsolicited_weight = cov_unsolicited_weight + 1;

            // Unsolicited psum response.
            reset_dut();
            @(negedge clk_core); psum_rsp_valid = 1'b1;
            @(posedge clk_core); require_sticky_protocol_fault(2);
            cov_unsolicited_psum = cov_unsolicited_psum + 1;

            // Same-cycle request/response violates minimum service latency.
            reset_dut();
            @(negedge clk_core);
            force_request(1'b1, 1'b0, 16'h6203, 6'd24, 4'd4);
            weight_req_ready = 1'b1; psum_req_ready = 1'b1;
            weight_rsp_valid = 1'b1;
            @(posedge clk_core); require_sticky_protocol_fault(3);
            cov_same_cycle_early = cov_same_cycle_early + 1;

            // Duplicate response remains asserted after legal consumption.
            release_request();
            reset_dut();
            @(negedge clk_core);
            force_request(1'b1, 1'b0, 16'h6204, 6'd25, 4'd5);
            weight_req_ready = 1'b1; psum_req_ready = 1'b1;
            @(posedge clk_core); #1ps;
            @(negedge clk_core);
            weight_rsp_valid = 1'b1; psum_rsp_valid = 1'b1;
            @(posedge clk_core); #1ps;
            // Keep the old response valid into the next transaction edge.
            @(posedge clk_core); require_sticky_protocol_fault(4);
            cov_duplicate_response = cov_duplicate_response + 1;

            // M935 held-request cancellation.
            release_request();
            reset_dut();
            @(negedge clk_core);
            force_request(1'b1, 1'b0, 16'h6205, 6'd26, 4'd6);
            weight_req_ready = 1'b1; psum_req_ready = 1'b0;
            @(posedge clk_core); #1ps;
            @(negedge clk_core);
            request_hold_attack_mode = 1'b1;
            force dut.issue_request_valid = 1'b0;
            @(posedge clk_core); require_sticky_protocol_fault(5);
            request_hold_attack_mode = 1'b0;
            cov_request_attack_windows = cov_request_attack_windows + 1;
            cov_cancel = cov_cancel + 1;

            // M935 held tuple mutation.
            release_request();
            reset_dut();
            @(negedge clk_core);
            force_request(1'b1, 1'b0, 16'h6206, 6'd27, 4'd7);
            weight_req_ready = 1'b0; psum_req_ready = 1'b0;
            @(posedge clk_core); #1ps;
            @(negedge clk_core);
            request_hold_attack_mode = 1'b1;
            force dut.issue_request_epoch = 16'hdead;
            @(posedge clk_core); require_sticky_protocol_fault(6);
            request_hold_attack_mode = 1'b0;
            cov_request_attack_windows = cov_request_attack_windows + 1;
            cov_tuple_mutation = cov_tuple_mutation + 1;

            // Psum response on a non-first transaction.
            release_request();
            reset_dut();
            @(negedge clk_core);
            force_request(1'b0, 1'b1, 16'h6207, 6'd28, 4'd8);
            weight_req_ready = 1'b1;
            @(posedge clk_core); #1ps;
            @(negedge clk_core); psum_rsp_valid = 1'b1;
            @(posedge clk_core); require_sticky_protocol_fault(7);
            cov_nonfirst_psum = cov_nonfirst_psum + 1;
            release_request();
        end
    endtask

    task automatic service_assumption_attacks;
        begin
            // Held weight payload mutation.  DUT consumption is blocked; the
            // independent service checker, not protocol_error, must latch it.
            reset_dut();
            @(negedge clk_core);
            force_request(1'b1, 1'b0, 16'h7301, 6'd31, 4'd1);
            weight_req_ready = 1'b1; psum_req_ready = 1'b1;
            @(posedge clk_core); #1ps;
            @(negedge clk_core);
            weight_rsp_valid = 1'b1; psum_rsp_valid = 1'b1;
            weight_data = {96{12'h011}}; psum_data = '0;
            force dut.core_issue_data_ready = 1'b0;
            weight_service_attack_mode = 1'b1;
            cov_weight_service_attack_windows =
                cov_weight_service_attack_windows + 1;
            @(posedge clk_core); #1ps;
            @(negedge clk_core); weight_data[11:0] = 12'h012;
            // Detection occurs on the next posedge.  Sample on the following
            // negedge so all checker NBA updates and UNIT_DELAY DUT activity
            // are complete; this removes R2's ambiguous same-edge sampling.
            @(posedge clk_core);
            @(negedge clk_core);
            if (!weight_service_fault || psum_service_fault || protocol_error)
                $fatal(1, "weight service mutation boundary misclassified weight=%0d psum=%0d protocol=%0d",
                    weight_service_fault, psum_service_fault, protocol_error);
            weight_service_attack_mode = 1'b0;
            cov_weight_payload_mutation = cov_weight_payload_mutation + 1;

            // Held psum valid drop is another service-side assumption attack.
            release_request();
            reset_dut();
            @(negedge clk_core);
            force_request(1'b1, 1'b0, 16'h7302, 6'd32, 4'd2);
            weight_req_ready = 1'b1; psum_req_ready = 1'b1;
            @(posedge clk_core); #1ps;
            @(negedge clk_core);
            weight_rsp_valid = 1'b1; psum_rsp_valid = 1'b1;
            weight_data = '0; psum_data = {96{19'h00003}};
            force dut.core_issue_data_ready = 1'b0;
            psum_service_attack_mode = 1'b1;
            cov_psum_service_attack_windows =
                cov_psum_service_attack_windows + 1;
            @(posedge clk_core); #1ps;
            @(negedge clk_core); psum_rsp_valid = 1'b0;
            @(posedge clk_core);
            @(negedge clk_core);
            if (!psum_service_fault || weight_service_fault || protocol_error)
                $fatal(1, "psum valid-drop boundary misclassified weight=%0d psum=%0d protocol=%0d",
                    weight_service_fault, psum_service_fault, protocol_error);
            psum_service_attack_mode = 1'b0;
            cov_psum_valid_drop = cov_psum_valid_drop + 1;
            release_request();
        end
    endtask

    task automatic random_legal_transaction(input integer index);
        integer w0, p0, stall_w, stall_p, hold_cycles;
        logic first;
        begin
            reset_dut();
            require_legal_masks_clear(100 + index);
            prng_q = prng_q * 32'd1664525 + 32'd1013904223;
            first = prng_q[0];
            stall_w = 1 + prng_q[3:1];
            stall_p = first ? (1 + prng_q[6:4]) : 0;
            hold_cycles = 1 + prng_q[9:7];
            w0 = weight_fire_count;
            p0 = psum_fire_count;
            @(negedge clk_core);
            force_request(first, 1'b0, 16'h8000 + index,
                index[5:0], prng_q[13:10]);
            weight_req_ready = 1'b0;
            psum_req_ready = 1'b0;
            fork
                begin
                    repeat (stall_w) @(negedge clk_core);
                    weight_req_ready = 1'b1;
                end
                begin
                    if (first) begin
                        repeat (stall_p) @(negedge clk_core);
                        psum_req_ready = 1'b1;
                    end
                end
            join
            wait (weight_fire_count == w0 + 1);
            if (first) wait (psum_fire_count == p0 + 1);
            @(negedge clk_core);
            if (index[0]) begin
                weight_rsp_valid = 1'b1;
                repeat (1 + prng_q[11:10]) @(negedge clk_core);
                if (first) psum_rsp_valid = 1'b1;
            end else begin
                if (first) psum_rsp_valid = 1'b1;
                repeat (1 + prng_q[11:10]) @(negedge clk_core);
                weight_rsp_valid = 1'b1;
            end
            force dut.core_issue_data_ready = 1'b0;
            repeat (hold_cycles) @(posedge clk_core);
            force dut.core_issue_data_ready = 1'b1;
            wait (dut.response_accept_w);
            @(posedge clk_core); #1ps;
            @(negedge clk_core);
            weight_rsp_valid = 1'b0;
            psum_rsp_valid = 1'b0;
            if (weight_fire_count != w0 + 1
                    || psum_fire_count != p0 + first)
                $fatal(1, "random transaction %0d duplicated request", index);
            cov_random_transactions = cov_random_transactions + 1;
            cov_no_duplicate_request = cov_no_duplicate_request + 1;
            release_request();
        end
    endtask

    task automatic load_normal_task(input logic [15:0] epoch);
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
                while (!prep_ready) @(negedge clk_core);
                @(posedge clk_core);
            end
            @(negedge clk_core);
            prep_valid = 1'b0;
            prep_task_start = 1'b0;
            prep_task_last = 1'b0;
        end
    endtask

    task automatic serve_normal_beat(input logic expect_first);
        integer watchdog;
        begin
            watchdog = 0;
            while (!weight_req_valid && watchdog < 2000) begin
                @(posedge clk_core);
                watchdog = watchdog + 1;
            end
            if (!weight_req_valid)
                $fatal(1, "normal M935 issue request timeout");
            if (issue_request_first != expect_first)
                $fatal(1, "normal M935 first flag got=%0d expected=%0d",
                    issue_request_first, expect_first);
            @(negedge clk_core);
            weight_req_ready = 1'b1;
            psum_req_ready = 1'b1;
            @(posedge clk_core); #1ps;
            @(negedge clk_core);
            weight_rsp_valid = 1'b1;
            weight_data = '0;
            psum_rsp_valid = expect_first;
            psum_data = '0;
            watchdog = 0;
            while (!dut.response_accept_w && watchdog < 100) begin
                @(posedge clk_core);
                watchdog = watchdog + 1;
            end
            if (!dut.response_accept_w)
                $fatal(1, "normal M935 response timeout");
            @(posedge clk_core); #1ps;
            @(negedge clk_core);
            weight_rsp_valid = 1'b0;
            psum_rsp_valid = 1'b0;
            cov_normal_issue = cov_normal_issue + 1;
        end
    endtask

    task automatic normal_m935_completion;
        integer issue0, row0, done0, watchdog;
        begin
            reset_dut();
            require_legal_masks_clear(200);
            issue0 = count_issue_accepts;
            row0 = row_complete_count;
            done0 = task_done_count;
            load_normal_task(16'h9001);
            serve_normal_beat(1'b1);
            serve_normal_beat(1'b0);
            watchdog = 0;
            while (task_done_count == done0 && watchdog < 2000) begin
                @(posedge clk_core);
                watchdog = watchdog + 1;
            end
            #1ps;
            if (task_done_count != done0 + 1
                    || row_complete_count != row0 + 1
                    || count_issue_accepts != issue0 + 2
                    || task_done_epoch != 16'h9001 || protocol_error)
                $fatal(1, "normal frozen-M935 completion mismatch issue=%0d row=%0d done=%0d err=%0d",
                    count_issue_accepts - issue0,
                    row_complete_count - row0,
                    task_done_count - done0, protocol_error);
            cov_normal_row = cov_normal_row + 1;
            cov_normal_task = cov_normal_task + 1;
        end
    endtask

    initial begin
        cycle_count = 0;
        weight_fire_count = 0;
        psum_fire_count = 0;
        response_accept_count = 0;
        row_complete_count = 0;
        task_done_count = 0;
        last_response_accept_cycle = -100;
        cov_weight_first = 0; cov_psum_first = 0;
        cov_weight_rsp_first = 0; cov_psum_rsp_first = 0;
        cov_long_request_stall = 0; cov_long_response_backpressure = 0;
        cov_nonfirst = 0; cov_ii2 = 0;
        cov_reset_partial = 0; cov_reset_complete = 0; cov_reset_skew = 0;
        cov_unsolicited_weight = 0; cov_unsolicited_psum = 0;
        cov_same_cycle_early = 0; cov_duplicate_response = 0;
        cov_cancel = 0; cov_tuple_mutation = 0; cov_nonfirst_psum = 0;
        cov_weight_payload_mutation = 0; cov_psum_valid_drop = 0;
        cov_no_duplicate_request = 0; cov_random_transactions = 0;
        cov_normal_issue = 0; cov_normal_row = 0; cov_normal_task = 0;
        cov_legal_masks_clear = 0; cov_request_attack_windows = 0;
        cov_weight_service_attack_windows = 0;
        cov_psum_service_attack_windows = 0;
        prng_q = 32'h1168_2026;
        clear_public_drivers();

        directed_weight_first();
        directed_psum_first_and_backpressure();
        directed_nonfirst();
        directed_ii2();
        reset_pending_cases();
        sticky_fault_attacks();
        service_assumption_attacks();
        for (integer test_index = 0; test_index < 24;
                test_index = test_index + 1)
            random_legal_transaction(test_index);
        normal_m935_completion();

        if (cov_weight_first != 1 || cov_psum_first != 1
                || cov_weight_rsp_first != 1 || cov_psum_rsp_first != 1
                || cov_long_request_stall < 5
                || cov_long_response_backpressure < 5
                || cov_nonfirst != 1 || cov_ii2 != 1
                || cov_reset_partial != 1 || cov_reset_complete != 1
                || cov_reset_skew != 1 || cov_unsolicited_weight != 1
                || cov_unsolicited_psum != 1 || cov_same_cycle_early != 1
                || cov_duplicate_response != 1 || cov_cancel != 1
                || cov_tuple_mutation != 1 || cov_nonfirst_psum != 1
                || cov_weight_payload_mutation != 1
                || cov_psum_valid_drop != 1
                || cov_no_duplicate_request < 25
                || cov_random_transactions != 24
                || cov_normal_issue != 2 || cov_normal_row != 1
                || cov_normal_task != 1 || cov_legal_masks_clear != 29
                || cov_request_attack_windows != 2
                || cov_weight_service_attack_windows != 1
                || cov_psum_service_attack_windows != 1)
            $fatal(1, "M1168R3 required coverage minima missing");

        $display("COVERAGE_M1168R3_PROTOCOL weight_first=%0d psum_first=%0d weight_rsp_first=%0d psum_rsp_first=%0d long_request=%0d long_response=%0d nonfirst=%0d ii2=%0d no_duplicate_request=%0d random=%0d legal_masks_clear=%0d",
            cov_weight_first, cov_psum_first, cov_weight_rsp_first,
            cov_psum_rsp_first, cov_long_request_stall,
            cov_long_response_backpressure, cov_nonfirst, cov_ii2,
            cov_no_duplicate_request, cov_random_transactions,
            cov_legal_masks_clear);
        $display("COVERAGE_M1168R3_RESETS_ATTACKS reset_partial=%0d reset_complete=%0d reset_skew=%0d unsolicited_weight=%0d unsolicited_psum=%0d same_cycle=%0d duplicate_response=%0d cancel=%0d tuple_mutation=%0d nonfirst_psum=%0d request_attack_windows=%0d",
            cov_reset_partial, cov_reset_complete, cov_reset_skew,
            cov_unsolicited_weight, cov_unsolicited_psum,
            cov_same_cycle_early, cov_duplicate_response, cov_cancel,
            cov_tuple_mutation, cov_nonfirst_psum,
            cov_request_attack_windows);
        $display("COVERAGE_M1168R3_SERVICE_ASSUMPTIONS weight_payload_mutation=%0d psum_valid_drop=%0d weight_windows=%0d psum_windows=%0d independent_checker=1 race_free_negedge_sample=1 dut_fault_claim=0",
            cov_weight_payload_mutation, cov_psum_valid_drop,
            cov_weight_service_attack_windows,
            cov_psum_service_attack_windows);
        $display("COVERAGE_M1168R3_FROZEN_M935 normal_issues=%0d normal_rows=%0d normal_tasks=%0d epoch=36865",
            cov_normal_issue, cov_normal_row, cov_normal_task);
        $display("PASS_M1168R3_M1162_COMMON_CHARGE_PROTOCOL_UNIT_DELAY_CANDIDATE directed_random=24 protocol_attacks=7 service_assumption_attacks=2 request_attack_windows=2 legal_masks_clear=29 reset_states=3 ii=2 normal_m935_rows=1 normal_m935_tasks=1 functional_vcs_only=true timing_verified=false cycles_measured=false speedup=false ppa=false energy=false system_speedup=false headline=false");
        $finish;
    end
endmodule

`default_nettype wire
