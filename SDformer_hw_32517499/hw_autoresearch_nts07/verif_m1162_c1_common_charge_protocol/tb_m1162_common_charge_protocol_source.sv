`timescale 1ns/1ps
`default_nettype none

// Source-only directed TB.  M1162 does not compile or run this file.
// Future VCS must compile it with frozen M935 and the foundry SRAM model.
module tb_m1162_common_charge_protocol_source;
    logic clk_core = 1'b0;
    logic reset_n = 1'b0;
    always #1.5 clk_core = ~clk_core;

    logic weight_req_ready, weight_rsp_valid;
    logic psum_req_ready, psum_rsp_valid;
    logic psum_write_ready, row_complete_ready;
    logic weight_req_valid, weight_rsp_ready;
    logic psum_req_valid, psum_rsp_ready;
    logic protocol_error;
    logic [15:0] weight_req_epoch, psum_req_epoch;
    logic [5:0] weight_req_row, psum_req_addr;
    logic [3:0] weight_req_source;
    logic weight_req_source_valid;
    logic [1151:0] weight_data;
    logic [1823:0] psum_data;

    int weight_fires;
    int psum_fires;
    int partial_weight_first;
    int partial_psum_first;
    int skew_weight_response;
    int skew_psum_response;
    int long_request_stall;
    int long_response_backpressure;
    int reset_pending;
    int early_response;
    int spurious_response;
    int request_cancellation;
    int tuple_mutation;
    int sticky_error_seen;

    m1162_m935_c1_common_charge_protocol_boundary dut (
        .clk_core(clk_core), .reset_n(reset_n),
        .prep_valid(1'b0), .prep_ready(), .prep_task_start(1'b0),
        .prep_task_last(1'b0), .prep_epoch('0), .prep_row_id('0),
        .prep_mask('0), .prep_reserved('0),
        .issue_request_valid(), .issue_request_epoch(),
        .issue_request_row_id(), .issue_request_first(),
        .issue_request_last(), .issue_request_source_valid(),
        .issue_request_source_index(), .issue_request_parent_valid(),
        .issue_request_parent_id(),
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
        .psum_write_valid(), .psum_write_ready(psum_write_ready),
        .psum_write_address(), .psum_write_data(),
        .row_complete_valid(), .row_complete_ready(row_complete_ready),
        .row_complete_id(), .task_done_valid(), .task_done_epoch(),
        .protocol_error(protocol_error), .preprocess_busy(), .execute_busy(),
        .active_directory_bank(), .parent_queue_occupancy(),
        .parent_reserved_occupancy(), .debug_parent_live_bitmap(),
        .debug_written_bitmap(), .debug_scratch_read_event(),
        .debug_scratch_write_event(), .debug_forward_event(),
        .debug_read_response_event(), .debug_dual_enqueue_event(),
        .debug_dead_write_elision_event(), .debug_deadline_hold_event(),
        .debug_overflow_block_event(), .debug_stalled_raw_event(),
        .count_issue_accepts(), .count_parent_edges(),
        .count_dead_write_elisions(), .count_macro_reads(),
        .count_macro_writes(), .count_forwards(), .count_deadline_holds(),
        .count_issue_stalls(), .count_psum_commits(),
        .count_row_completions()
    );

    // Freeze M935's producer side for protocol-only attacks.  The actual M935
    // regression is a separate future VCS gate.
    task automatic drive_request(input logic first, input logic [15:0] epoch,
                                 input logic [5:0] row);
        force dut.issue_request_valid = 1'b1;
        force dut.issue_request_epoch = epoch;
        force dut.issue_request_row_id = row;
        force dut.issue_request_first = first;
        force dut.issue_request_last = 1'b0;
        force dut.issue_request_source_valid = 1'b1;
        force dut.issue_request_source_index = 4'd3;
        force dut.issue_request_parent_valid = 1'b1;
        force dut.issue_request_parent_id = 6'd7;
        force dut.core_issue_data_ready = 1'b1;
    endtask

    task automatic release_request;
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
    endtask

    task automatic do_reset;
        reset_n = 1'b0;
        weight_req_ready = 1'b0;
        psum_req_ready = 1'b0;
        weight_rsp_valid = 1'b0;
        psum_rsp_valid = 1'b0;
        repeat (2) @(posedge clk_core);
        reset_n = 1'b1;
        @(posedge clk_core);
        if (protocol_error) $fatal(1, "sticky error did not clear on reset");
    endtask

    always @(posedge clk_core) begin
        if (reset_n && weight_req_valid && weight_req_ready) weight_fires++;
        if (reset_n && psum_req_valid && psum_req_ready) psum_fires++;
    end

    // Request payload stability under backpressure.
    ap_weight_hold: assert property (@(posedge clk_core) disable iff (!reset_n)
        weight_req_valid && !weight_req_ready |=>
        weight_req_valid && $stable({weight_req_epoch, weight_req_row,
                                     weight_req_source,
                                     weight_req_source_valid}));
    ap_psum_hold: assert property (@(posedge clk_core) disable iff (!reset_n)
        psum_req_valid && !psum_req_ready |=>
        psum_req_valid && $stable({psum_req_epoch, psum_req_addr}));
    ap_no_lone_weight_response_consume: assert property (
        @(posedge clk_core) disable iff (!reset_n)
        dut.request_active_q && dut.request_first_q
        && weight_rsp_valid && !psum_rsp_valid |-> !weight_rsp_ready);
    ap_no_lone_psum_response_consume: assert property (
        @(posedge clk_core) disable iff (!reset_n)
        dut.request_active_q && dut.request_first_q
        && psum_rsp_valid && !weight_rsp_valid |-> !psum_rsp_ready);

    initial begin
        weight_data = '0;
        psum_data = '0;
        psum_write_ready = 1'b1;
        row_complete_ready = 1'b1;
        weight_fires = 0;
        psum_fires = 0;
        partial_weight_first = 0;
        partial_psum_first = 0;
        skew_weight_response = 0;
        skew_psum_response = 0;
        long_request_stall = 0;
        long_response_backpressure = 0;
        reset_pending = 0;
        early_response = 0;
        spurious_response = 0;
        request_cancellation = 0;
        tuple_mutation = 0;
        sticky_error_seen = 0;

        // Weight accepts first; psum stalls.  Weight response may wait early
        // relative to the peer but is legal after its own accepted request.
        do_reset();
        drive_request(1'b1, 16'h1001, 6'd9);
        weight_req_ready = 1'b1;
        psum_req_ready = 1'b0;
        @(posedge clk_core);
        partial_weight_first++;
        weight_rsp_valid = 1'b1;
        repeat (4) begin
            @(posedge clk_core);
            if (weight_rsp_ready) $fatal(1, "lone weight response consumed");
            long_request_stall++;
        end
        psum_req_ready = 1'b1;
        @(posedge clk_core);
        psum_rsp_valid = 1'b1;
        skew_weight_response++;
        @(posedge clk_core);
        if (weight_fires != 1 || psum_fires != 1)
            $fatal(1, "duplicate or missing request fire");
        weight_rsp_valid = 1'b0;
        psum_rsp_valid = 1'b0;
        release_request();

        // Psum accepts first, then its response waits through weight stalls.
        do_reset();
        weight_fires = 0;
        psum_fires = 0;
        drive_request(1'b1, 16'h2002, 6'd10);
        weight_req_ready = 1'b0;
        psum_req_ready = 1'b1;
        @(posedge clk_core);
        partial_psum_first++;
        psum_rsp_valid = 1'b1;
        repeat (4) @(posedge clk_core);
        weight_req_ready = 1'b1;
        @(posedge clk_core);
        weight_rsp_valid = 1'b1;
        skew_psum_response++;
        force dut.core_issue_data_ready = 1'b0;
        repeat (4) begin
            @(posedge clk_core);
            if (weight_rsp_ready || psum_rsp_ready)
                $fatal(1, "response consumed under core backpressure");
            long_response_backpressure++;
        end
        force dut.core_issue_data_ready = 1'b1;
        @(posedge clk_core);
        weight_rsp_valid = 1'b0;
        psum_rsp_valid = 1'b0;
        if (weight_fires != 1 || psum_fires != 1)
            $fatal(1, "partial-order duplicate fire");
        release_request();

        // Reset cancels partial request state.
        do_reset();
        drive_request(1'b1, 16'h3003, 6'd11);
        weight_req_ready = 1'b1;
        psum_req_ready = 1'b0;
        @(posedge clk_core);
        reset_pending++;
        reset_n = 1'b0;
        @(posedge clk_core);
        if (dut.request_active_q) $fatal(1, "reset did not cancel request");
        reset_n = 1'b1;
        force dut.issue_request_valid = 1'b0;
        weight_rsp_valid = 1'b1;
        @(posedge clk_core);
        spurious_response++;
        weight_rsp_valid = 1'b0;
        if (!protocol_error) $fatal(1, "spurious response not sticky");
        sticky_error_seen++;

        // Same-cycle response is early under the frozen >=1-cycle response
        // latency, even if its request also accepts in that cycle.
        release_request();
        do_reset();
        drive_request(1'b1, 16'h4004, 6'd12);
        weight_req_ready = 1'b1;
        psum_req_ready = 1'b1;
        weight_rsp_valid = 1'b1;
        @(posedge clk_core);
        weight_rsp_valid = 1'b0;
        early_response++;
        if (!protocol_error) $fatal(1, "early response not sticky");

        // Held-request cancellation and tuple mutation are separate attacks.
        release_request();
        do_reset();
        drive_request(1'b1, 16'h5005, 6'd13);
        weight_req_ready = 1'b1;
        psum_req_ready = 1'b0;
        @(posedge clk_core);
        force dut.issue_request_valid = 1'b0;
        @(posedge clk_core);
        request_cancellation++;
        if (!protocol_error) $fatal(1, "request cancellation not sticky");

        release_request();
        do_reset();
        drive_request(1'b1, 16'h6006, 6'd14);
        weight_req_ready = 1'b0;
        psum_req_ready = 1'b0;
        @(posedge clk_core);
        force dut.issue_request_epoch = 16'hdead;
        @(posedge clk_core);
        tuple_mutation++;
        if (!protocol_error) $fatal(1, "tuple mutation not sticky");

        if (!(partial_weight_first && partial_psum_first
              && skew_weight_response && skew_psum_response
              && long_request_stall >= 4 && long_response_backpressure >= 4
              && reset_pending && early_response && spurious_response
              && request_cancellation && tuple_mutation
              && sticky_error_seen))
            $fatal(1, "required cover population missing");
        $display("PASS_M1162_SOURCE_DIRECTED_PLAN weight_first=%0d psum_first=%0d long_req=%0d long_rsp=%0d reset=%0d early=%0d spurious=%0d cancel=%0d mutate=%0d",
                 partial_weight_first, partial_psum_first, long_request_stall,
                 long_response_backpressure, reset_pending, early_response,
                 spurious_response, request_cancellation, tuple_mutation);
        release_request();
        $finish;
    end
endmodule

`default_nettype wire
