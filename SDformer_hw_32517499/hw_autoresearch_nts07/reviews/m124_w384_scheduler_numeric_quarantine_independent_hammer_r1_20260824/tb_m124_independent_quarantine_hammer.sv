`timescale 1ns/1ps
`default_nettype none

module tb_m124_independent_quarantine_hammer;
    localparam int LANES = 96;
    localparam int WIN_ROWS = 384;
    localparam int BLOCKS = 8;

    logic clk_core = 1'b0;
    logic rst_core = 1'b1;
    logic accumulator_window_start_valid = 1'b0;
    logic accumulator_window_start_ready, accumulator_window_start_accept;
    logic event_valid = 1'b0;
    logic event_ready;
    logic [3:0] event_source;
    logic [2:0] event_block;
    logic [8:0] event_row_offset;
    logic event_negate;
    logic [11:0] window_base_row;
    logic [15:0] window_context;
    logic event_accept;
    logic descriptor_close_valid = 1'b0;
    logic descriptor_close_ready;
    logic descriptor_close_accept;
    logic weight_prefetch_valid, weight_prefetch_ready;
    logic [3:0] weight_prefetch_source;
    logic [2:0] weight_prefetch_block;
    logic [15:0] weight_prefetch_context;
    logic weight_prefetch_accept;
    logic weight_rd_en;
    logic [6:0] weight_rd_key;
    logic [1:0] weight_rd_beat;
    logic [255:0] weight_rd_data;
    logic accumulator_window_end_valid = 1'b0;
    logic accumulator_window_end_ready, accumulator_window_end_accept;
    logic commit_valid, commit_ready;
    logic [2:0] commit_block;
    logic [8:0] commit_row;
    logic [1823:0] commit_data;
    logic commit_last, accumulator_window_done;
    logic lane_mem_rd_en;
    logic [11:0] lane_mem_rd_addr;
    logic [18:0] lane_mem_rd_data [0:LANES-1];
    logic lane_mem_wr_en;
    logic [11:0] lane_mem_wr_addr;
    logic [18:0] lane_mem_wr_data [0:LANES-1];
    logic descriptor_done, descriptor_done_empty;
    logic [11:0] descriptor_done_base_row;
    logic [15:0] descriptor_done_context;
    logic observed_service_valid, observed_service_ready;
    logic observed_service_accept, observed_numeric_service_accept;
    logic observed_service_is_event;
    logic [3:0] observed_service_source;
    logic [2:0] observed_service_block;
    logic [1:0] observed_service_load_beat;
    logic [8:0] observed_service_row_offset;
    logic observed_service_negate, observed_service_last_for_key;
    logic mapped_update_accept, tail_bypass_available;
    logic scheduler_protocol_error, numeric_protocol_error;
    logic protocol_error, busy;

    integer cycle_count;
    integer event_accept_count, close_accept_count;
    integer service_accept_count, numeric_service_accept_count;
    integer mapped_update_count, descriptor_done_count;
    integer lane_write_count, lane_write_under_fault_count;
    integer post_fault_public_leak_count, accept_mismatch_count;
    bit fault_monitor_enable;

    m124_w384_scheduler_numeric_quarantine_island dut (
        .clk_core(clk_core), .rst_core(rst_core),
        .accumulator_window_start_valid(accumulator_window_start_valid),
        .accumulator_window_start_ready(accumulator_window_start_ready),
        .accumulator_window_start_accept(accumulator_window_start_accept),
        .event_valid(event_valid), .event_ready(event_ready),
        .event_source(event_source), .event_block(event_block),
        .event_row_offset(event_row_offset), .event_negate(event_negate),
        .window_base_row(window_base_row), .window_context(window_context),
        .event_accept(event_accept),
        .descriptor_close_valid(descriptor_close_valid),
        .descriptor_close_ready(descriptor_close_ready),
        .descriptor_close_accept(descriptor_close_accept),
        .weight_prefetch_valid(weight_prefetch_valid),
        .weight_prefetch_ready(weight_prefetch_ready),
        .weight_prefetch_source(weight_prefetch_source),
        .weight_prefetch_block(weight_prefetch_block),
        .weight_prefetch_context(weight_prefetch_context),
        .weight_prefetch_accept(weight_prefetch_accept),
        .weight_rd_en(weight_rd_en), .weight_rd_key(weight_rd_key),
        .weight_rd_beat(weight_rd_beat), .weight_rd_data(weight_rd_data),
        .accumulator_window_end_valid(accumulator_window_end_valid),
        .accumulator_window_end_ready(accumulator_window_end_ready),
        .accumulator_window_end_accept(accumulator_window_end_accept),
        .commit_valid(commit_valid), .commit_ready(commit_ready),
        .commit_block(commit_block), .commit_row(commit_row),
        .commit_data(commit_data), .commit_last(commit_last),
        .accumulator_window_done(accumulator_window_done),
        .lane_mem_rd_en(lane_mem_rd_en), .lane_mem_rd_addr(lane_mem_rd_addr),
        .lane_mem_rd_data(lane_mem_rd_data), .lane_mem_wr_en(lane_mem_wr_en),
        .lane_mem_wr_addr(lane_mem_wr_addr), .lane_mem_wr_data(lane_mem_wr_data),
        .descriptor_done(descriptor_done),
        .descriptor_done_empty(descriptor_done_empty),
        .descriptor_done_base_row(descriptor_done_base_row),
        .descriptor_done_context(descriptor_done_context),
        .observed_service_valid(observed_service_valid),
        .observed_service_ready(observed_service_ready),
        .observed_service_accept(observed_service_accept),
        .observed_numeric_service_accept(observed_numeric_service_accept),
        .observed_service_is_event(observed_service_is_event),
        .observed_service_source(observed_service_source),
        .observed_service_block(observed_service_block),
        .observed_service_load_beat(observed_service_load_beat),
        .observed_service_row_offset(observed_service_row_offset),
        .observed_service_negate(observed_service_negate),
        .observed_service_last_for_key(observed_service_last_for_key),
        .mapped_update_accept(mapped_update_accept),
        .tail_bypass_available(tail_bypass_available),
        .scheduler_protocol_error(scheduler_protocol_error),
        .numeric_protocol_error(numeric_protocol_error),
        .protocol_error(protocol_error), .busy(busy)
    );

    always #1 clk_core = ~clk_core;

    initial begin : global_watchdog
        #100000;
        $fatal(1, "M124 independent global watchdog");
    end

    function automatic integer signed weight_value(input int key, input int lane);
        weight_value = ((key * 37 + lane * 29) & 8'hff) - 128;
    endfunction

    function automatic [255:0] weight_beat(input int key, input int beat);
        logic [255:0] result;
        integer signed value;
        begin
            result = '0;
            for (int byte_index = 0; byte_index < 32; byte_index++) begin
                value = weight_value(key, beat * 32 + byte_index);
                result[byte_index * 8 +: 8] = value[7:0];
            end
            weight_beat = result;
        end
    endfunction

    always @(posedge clk_core) begin : memory_models
        if (weight_rd_en)
            weight_rd_data <= weight_beat(weight_rd_key, weight_rd_beat);
        for (int lane = 0; lane < LANES; lane++) begin
            if (lane_mem_rd_en)
                lane_mem_rd_data[lane] <= '0;
        end
    end

    always @(posedge clk_core) begin : independent_monitors
        if (rst_core) begin
            cycle_count <= 0;
            event_accept_count <= 0;
            close_accept_count <= 0;
            service_accept_count <= 0;
            numeric_service_accept_count <= 0;
            mapped_update_count <= 0;
            descriptor_done_count <= 0;
            lane_write_count <= 0;
            lane_write_under_fault_count <= 0;
            post_fault_public_leak_count <= 0;
            accept_mismatch_count <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (event_accept) event_accept_count <= event_accept_count + 1;
            if (descriptor_close_accept) close_accept_count <= close_accept_count + 1;
            if (observed_service_accept) service_accept_count <= service_accept_count + 1;
            if (observed_numeric_service_accept)
                numeric_service_accept_count <= numeric_service_accept_count + 1;
            if (mapped_update_accept) mapped_update_count <= mapped_update_count + 1;
            if (descriptor_done) descriptor_done_count <= descriptor_done_count + 1;
            if (lane_mem_wr_en) lane_write_count <= lane_write_count + 1;
            if (lane_mem_wr_en && protocol_error)
                lane_write_under_fault_count <= lane_write_under_fault_count + 1;

            if (observed_service_accept !== observed_numeric_service_accept
                    || observed_service_accept
                       !== (observed_service_valid && observed_service_ready))
                accept_mismatch_count <= accept_mismatch_count + 1;

            if (fault_monitor_enable && protocol_error
                    && (accumulator_window_start_ready
                        || accumulator_window_start_accept
                        || accumulator_window_end_ready
                        || accumulator_window_end_accept
                        || event_ready || event_accept
                        || descriptor_close_ready || descriptor_close_accept
                        || observed_service_valid || observed_service_ready
                        || observed_service_accept
                        || observed_numeric_service_accept
                        || weight_prefetch_valid || weight_prefetch_accept
                        || weight_rd_en || mapped_update_accept
                        || commit_valid || accumulator_window_done
                        || descriptor_done))
                post_fault_public_leak_count <= post_fault_public_leak_count + 1;
        end
    end

    task automatic clear_memories;
        begin
            weight_rd_data = '0;
            for (int lane = 0; lane < LANES; lane++)
                lane_mem_rd_data[lane] = '0;
        end
    endtask

    task automatic reset_all;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            accumulator_window_start_valid = 1'b0;
            event_valid = 1'b0;
            descriptor_close_valid = 1'b0;
            accumulator_window_end_valid = 1'b0;
            commit_ready = 1'b0;
            weight_prefetch_ready = 1'b1;
            fault_monitor_enable = 1'b0;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
            repeat (2) @(posedge clk_core);
        end
    endtask

    task automatic start_window;
        int attempts;
        begin
            @(negedge clk_core);
            accumulator_window_start_valid = 1'b1;
            attempts = 0;
            do begin
                @(posedge clk_core);
                attempts = attempts + 1;
                if (attempts > 20)
                    $fatal(1, "M124 start watchdog ready=%0d accept=%0d top_err=%0d sched_err=%0d numeric_err=%0d quarantine=%0d raw_ready=%0d active=%0d mapper_busy=%0d",
                           accumulator_window_start_ready,
                           accumulator_window_start_accept, protocol_error,
                           scheduler_protocol_error, numeric_protocol_error,
                           dut.quarantine_q, dut.raw_window_start_ready,
                           dut.accumulator_window_active,
                           dut.numeric_mapper_busy);
            end while (!accumulator_window_start_accept);
            @(negedge clk_core);
            accumulator_window_start_valid = 1'b0;
        end
    endtask

    task automatic send_event(
        input int source, input int block_id, input int row,
        input bit negate, input int base, input int context_value
    );
        begin
            @(negedge clk_core);
            event_source = source[3:0];
            event_block = block_id[2:0];
            event_row_offset = row[8:0];
            event_negate = negate;
            window_base_row = base[11:0];
            window_context = context_value[15:0];
            event_valid = 1'b1;
            do @(posedge clk_core); while (!event_accept);
            @(negedge clk_core);
            event_valid = 1'b0;
        end
    endtask

    task automatic close_descriptor(input int base, input int context_value);
        begin
            @(negedge clk_core);
            window_base_row = base[11:0];
            window_context = context_value[15:0];
            descriptor_close_valid = 1'b1;
            do @(posedge clk_core); while (!descriptor_close_accept);
            @(negedge clk_core);
            descriptor_close_valid = 1'b0;
        end
    endtask

    task automatic wait_update_and_done;
        int start_cycle;
        begin
            start_cycle = cycle_count;
            while (mapped_update_count < 1 || descriptor_done_count < 1) begin
                @(posedge clk_core);
                if (cycle_count - start_cycle > 100)
                    $fatal(1, "M124 independent drain watchdog update=%0d done=%0d err=%0d",
                           mapped_update_count, descriptor_done_count,
                           protocol_error);
            end
            repeat (2) @(posedge clk_core);
        end
    endtask

    task automatic drive_invalid_row_fault(input bit with_collision);
        begin
            event_source = 0;
            event_block = 0;
            event_row_offset = 9'd400;
            event_negate = 1'b0;
            window_base_row = 12'd300;
            window_context = 16'h3333;
            event_valid = 1'b1;
            descriptor_close_valid = with_collision;
        end
    endtask

    task automatic check_public_quarantine(input string tag);
        begin
            if (!protocol_error
                    || accumulator_window_start_ready
                    || accumulator_window_start_accept
                    || accumulator_window_end_ready
                    || accumulator_window_end_accept
                    || event_ready || event_accept
                    || descriptor_close_ready || descriptor_close_accept
                    || observed_service_valid || observed_service_ready
                    || observed_service_accept
                    || observed_numeric_service_accept
                    || weight_prefetch_valid || weight_prefetch_accept
                    || weight_rd_en || mapped_update_accept
                    || commit_valid || accumulator_window_done
                    || descriptor_done)
                $fatal(1, "%s public quarantine failure", tag);
        end
    endtask

    initial begin
        integer start_cycle;
        integer lane_writes_before_fault;
        integer update_accepts_before_fault;
        bit raw_service_opportunity;
        bit raw_prefetch_opportunity;
        bit raw_commit_opportunity;

        clk_core = 1'b0;
        rst_core = 1'b1;
        accumulator_window_start_valid = 1'b0;
        event_valid = 1'b0;
        event_source = '0;
        event_block = '0;
        event_row_offset = '0;
        event_negate = 1'b0;
        window_base_row = '0;
        window_context = '0;
        descriptor_close_valid = 1'b0;
        weight_prefetch_ready = 1'b1;
        weight_rd_data = '0;
        accumulator_window_end_valid = 1'b0;
        commit_ready = 1'b0;
        fault_monitor_enable = 1'b0;
        clear_memories();

        // Exact reproduction of the M121 independent-hammer P0 sequence:
        // one accepted update, row=400 scheduler fault, then end/commit probe.
        reset_all();
        start_window();
        send_event(0, 0, 1, 0, 200, 16'h2222);
        close_descriptor(200, 16'h2222);
        wait_update_and_done();
        @(negedge clk_core);
        drive_invalid_row_fault(1'b0);
        #0.1;
        if (!scheduler_protocol_error || numeric_protocol_error)
            $fatal(1, "M124 M121-P0 reproduction fault classification mismatch");
        check_public_quarantine("M121-P0 fault cycle");
        @(posedge clk_core);
        @(negedge clk_core);
        event_valid = 1'b0;
        accumulator_window_end_valid = 1'b1;
        commit_ready = 1'b1;
        fault_monitor_enable = 1'b1;
        repeat (8) begin
            @(posedge clk_core);
            check_public_quarantine("M121-P0 post-fault end/commit");
        end
        if (post_fault_public_leak_count != 0)
            $fatal(1, "M124 M121-P0 closure leaked count=%0d",
                   post_fault_public_leak_count);
        $display("CLOSURE m121_scheduler_fault_end_commit_p0 exact_row400_sequence=1 end_ready=0 end_accept=0 commit_valid=0 sticky_cycles=8");

        // Single-domain scheduler collision with a semantically legal numeric
        // end request. The cross-domain double-collision testcase is selected
        // by +CROSS_FAULT_LOOP because it does not advance simulation time on
        // the frozen RTL.
        $display("STAGE scheduler_collision_end_and_continuous_valid");
        reset_all();
        start_window();
        @(negedge clk_core);
        accumulator_window_end_valid = 1'b1;
        commit_ready = 1'b1;
        drive_invalid_row_fault(1'b1);
        #0.1;
        if (!scheduler_protocol_error || numeric_protocol_error)
            $fatal(1, "M124 same-cycle scheduler collision classification mismatch");
        check_public_quarantine("scheduler collision same cycle");
        fault_monitor_enable = 1'b1;
        repeat (6) begin
            @(posedge clk_core);
            check_public_quarantine("continuous valid quarantine");
        end
        if (post_fault_public_leak_count != 0)
            $fatal(1, "M124 continuously-held valid leak count=%0d",
                   post_fault_public_leak_count);
        $display("CLOSURE same_cycle_scheduler_fault_end_event_close_and_continuous_valid end_accept=0 ingress_accept=0 probe_cycles=6");

        // Repeat from an idle numeric state with a legal start request, so
        // start and end suppression are independently covered without the
        // cross-domain double-collision oscillation.
        reset_all();
        @(negedge clk_core);
        accumulator_window_start_valid = 1'b1;
        drive_invalid_row_fault(1'b1);
        #0.1;
        if (!scheduler_protocol_error || numeric_protocol_error)
            $fatal(1, "M124 same-cycle scheduler/start classification mismatch");
        check_public_quarantine("scheduler collision same-cycle start");
        repeat (3) @(posedge clk_core);
        $display("CLOSURE same_cycle_scheduler_fault_start start_ready=0 start_accept=0 ingress_accept=0 sticky_cycles=3");

        // Reset is the sole recovery path. A fresh start must accept after it.
        $display("STAGE reset_recovery");
        reset_all();
        start_window();
        #0.1;
        if (protocol_error || !dut.accumulator_window_active)
            $fatal(1, "M124 reset recovery failed");
        $display("CLOSURE reset_only_recovery start_accept_after_reset=1 protocol_error=0");

        // A prefetch opportunity must be suppressed in the exact scheduler
        // fault cycle. Hold prefetch back so the opportunity is observable.
        $display("STAGE prefetch_fault");
        reset_all();
        start_window();
        weight_prefetch_ready = 1'b0;
        send_event(0, 0, 4, 0, 500, 16'h5555);
        close_descriptor(500, 16'h5555);
        start_cycle = cycle_count;
        while (!dut.raw_weight_prefetch_valid) begin
            @(negedge clk_core);
            if (cycle_count - start_cycle > 40)
                $fatal(1, "M124 raw prefetch opportunity watchdog");
        end
        raw_prefetch_opportunity = dut.raw_weight_prefetch_valid;
        drive_invalid_row_fault(1'b0);
        #0.1;
        check_public_quarantine("same-cycle prefetch fault");
        if (!raw_prefetch_opportunity || weight_prefetch_valid
                || weight_prefetch_accept)
            $fatal(1, "M124 same-cycle prefetch suppression mismatch");
        $display("CLOSURE same_cycle_prefetch_fault raw_opportunity_before_fault=1 public_prefetch_valid=0 public_prefetch_accept=0");

        // The same attack is repeated on a visible load-service opportunity.
        // Before fault, that token drives a weight read; after the combinational
        // scheduler fault, service/accept/read are all suppressed.
        $display("STAGE service_weight_fault");
        reset_all();
        start_window();
        send_event(0, 0, 5, 0, 600, 16'h6666);
        close_descriptor(600, 16'h6666);
        start_cycle = cycle_count;
        while (!(dut.service_valid && !dut.service_is_event
                 && dut.numeric_service_ready && weight_rd_en)) begin
            @(negedge clk_core);
            if (cycle_count - start_cycle > 60)
                $fatal(1, "M124 raw service/weight opportunity watchdog");
        end
        raw_service_opportunity = dut.service_valid && weight_rd_en;
        drive_invalid_row_fault(1'b0);
        #0.1;
        check_public_quarantine("same-cycle service/weight fault");
        if (!raw_service_opportunity || observed_service_accept || weight_rd_en)
            $fatal(1, "M124 same-cycle service/weight suppression mismatch");
        $display("CLOSURE same_cycle_service_weight_fault pre_fault_load_opportunity=1 service_accept=0 numeric_accept=0 weight_rd_en=0");

        // Hold a raw commit under backpressure, then fault the scheduler. The
        // raw numeric state may remain pending, but the public commit vanishes
        // combinationally in the fault cycle and stays quarantined.
        $display("STAGE commit_fault");
        reset_all();
        start_window();
        @(negedge clk_core);
        accumulator_window_end_valid = 1'b1;
        do @(posedge clk_core); while (!accumulator_window_end_accept);
        @(negedge clk_core);
        accumulator_window_end_valid = 1'b0;
        commit_ready = 1'b0;
        start_cycle = cycle_count;
        while (!(dut.raw_commit_valid && commit_valid)) begin
            @(negedge clk_core);
            if (cycle_count - start_cycle > 20)
                $fatal(1, "M124 raw commit opportunity watchdog");
        end
        raw_commit_opportunity = dut.raw_commit_valid;
        drive_invalid_row_fault(1'b0);
        #0.1;
        check_public_quarantine("same-cycle commit fault");
        if (!raw_commit_opportunity || !dut.raw_commit_valid || commit_valid)
            $fatal(1, "M124 same-cycle commit suppression mismatch");
        repeat (3) @(posedge clk_core);
        if (commit_valid)
            $fatal(1, "M124 post-fault commit escaped");
        $display("CLOSURE same_cycle_commit_fault raw_commit_pending=1 public_commit_valid=0 post_fault_commit_valid=0");

        // Accept one event into the numeric pipeline, then fault before the
        // already accepted vector write retires. The internal lane write is
        // allowed to drain; no mapped accept/commit may escape quarantine.
        $display("STAGE older_write_drain");
        reset_all();
        start_window();
        send_event(0, 0, 7, 0, 700, 16'h7777);
        close_descriptor(700, 16'h7777);
        start_cycle = cycle_count;
        while (!dut.raw_mapped_update_accept) begin
            @(negedge clk_core);
            if (cycle_count - start_cycle > 80)
                $fatal(1, "M124 raw mapped-update opportunity watchdog");
        end
        update_accepts_before_fault = mapped_update_count;
        lane_writes_before_fault = lane_write_count;
        drive_invalid_row_fault(1'b0);
        #0.1;
        if (!protocol_error || !dut.raw_mapped_update_accept
                || mapped_update_accept)
            $fatal(1, "M124 older update quarantine shape mismatch raw=%0d public=%0d",
                   dut.raw_mapped_update_accept, mapped_update_accept);
        repeat (4) @(posedge clk_core);
        if (lane_write_count <= lane_writes_before_fault
                || lane_write_under_fault_count < 1
                || mapped_update_count != update_accepts_before_fault
                || commit_valid)
            $fatal(1, "M124 older write drain mismatch writes=%0d prior=%0d under_fault=%0d mapped=%0d prior_mapped=%0d commit=%0d",
                   lane_write_count, lane_writes_before_fault,
                   lane_write_under_fault_count, mapped_update_count,
                   update_accepts_before_fault, commit_valid);
        $display("CLOSURE older_accepted_update_lane_write_drain raw_update_accept_fault_cycle=1 public_mapped_accept=0 lane_write_under_fault=1 commit_valid=0");

        // Pure numeric lifecycle collision: start and end together while the
        // scheduler is clean. Composite quarantine must trigger in that cycle.
        $display("STAGE numeric_fault");
        reset_all();
        @(negedge clk_core);
        accumulator_window_start_valid = 1'b1;
        accumulator_window_end_valid = 1'b1;
        commit_ready = 1'b1;
        #0.1;
        if (scheduler_protocol_error || !numeric_protocol_error)
            $fatal(1, "M124 numeric collision classification mismatch sched=%0d numeric=%0d",
                   scheduler_protocol_error, numeric_protocol_error);
        check_public_quarantine("numeric collision same cycle");
        repeat (5) begin
            @(posedge clk_core);
            check_public_quarantine("numeric fault sticky quarantine");
        end
        $display("CLOSURE numeric_fault_same_cycle_and_sticky scheduler_error=0 numeric_error=1 start_accept=0 end_accept=0 sticky_cycles=5");

        if (accept_mismatch_count != 0)
            $fatal(1, "M124 accept observation mismatch count=%0d",
                   accept_mismatch_count);
        $display("CLOSURE accept_observation_consistency scheduler_numeric_mismatches=0 valid_ready_mismatches=0");
        $display("PASS M124 independent quarantine hammer m121_p0_closed=1 single_domain_same_cycle_start=1 single_domain_same_cycle_end=1 same_cycle_prefetch=1 same_cycle_service_weight=1 same_cycle_commit=1 older_lane_write_drain=1 numeric_fault=1 continuous_valid=1 reset_recovery=1 accept_mismatch=0 m123_instantiated=true weight_response_valid=false descriptor_retry_dedup=false production_modified=false");
        if ($test$plusargs("CROSS_FAULT_LOOP")) begin
            reset_all();
            @(negedge clk_core);
            // Hold one side while arming the four external valids, then release
            // it. This prevents the reproducer itself from entering the loop
            // before the evidence marker is flushed.
            force dut.scheduler_protocol_error = 1'b1;
            accumulator_window_start_valid = 1'b1;
            accumulator_window_end_valid = 1'b1;
            event_valid = 1'b1;
            descriptor_close_valid = 1'b1;
            #0.1;
            $display("ARMED M124 cross_fault_comb_loop scheduler_event_close_collision=1 numeric_start_end_collision=1 forced_scheduler_error_before_release=1");
            $fflush();
            release dut.scheduler_protocol_error;
            #0.1;
            $display("UNEXPECTED_ADVANCE M124 cross_fault_comb_loop");
        end
        $finish;
    end
endmodule

`default_nettype wire
