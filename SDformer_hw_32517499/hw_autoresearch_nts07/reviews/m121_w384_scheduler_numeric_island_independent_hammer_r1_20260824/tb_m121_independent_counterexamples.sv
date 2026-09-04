`timescale 1ns/1ps
`default_nettype none

module tb_m121_independent_counterexamples;
    localparam int LANES = 96;
    localparam int WIN_ROWS = 384;
    localparam int BLOCKS = 8;

    logic clk_core, rst_core;
    logic accumulator_window_start_valid;
    logic accumulator_window_start_ready, accumulator_window_start_accept;
    logic event_valid, event_ready;
    logic [3:0] event_source;
    logic [2:0] event_block;
    logic [8:0] event_row_offset;
    logic event_negate;
    logic [11:0] window_base_row;
    logic [15:0] window_context;
    logic event_accept;
    logic descriptor_close_valid, descriptor_close_ready;
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
    logic accumulator_window_end_valid;
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

    logic [18:0] lane_memory [0:LANES-1][0:BLOCKS*WIN_ROWS-1];
    logic [255:0] delayed_response_q;
    integer memory_latency_mode;
    integer event_accept_count, close_accept_count, service_count;
    integer load_count, service_event_count, update_count, descriptor_done_count;
    integer weight_read_count, commit_valid_under_fault_count;
    integer cycle_count;
    logic [18:0] last_update_lane0, last_update_lane95;

    m121_w384_scheduler_numeric_island dut (
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
        if (weight_rd_en) begin
            delayed_response_q <= weight_beat(weight_rd_key, weight_rd_beat);
            if (memory_latency_mode == 1)
                weight_rd_data <= weight_beat(weight_rd_key, weight_rd_beat);
            else
                weight_rd_data <= delayed_response_q;
        end
        for (int lane = 0; lane < LANES; lane++) begin
            if (lane_mem_rd_en)
                lane_mem_rd_data[lane] <= lane_memory[lane][lane_mem_rd_addr];
            if (lane_mem_wr_en)
                lane_memory[lane][lane_mem_wr_addr] <= lane_mem_wr_data[lane];
        end
    end

    always @(posedge clk_core) begin : counters
        if (rst_core) begin
            event_accept_count <= 0;
            close_accept_count <= 0;
            service_count <= 0;
            load_count <= 0;
            service_event_count <= 0;
            update_count <= 0;
            descriptor_done_count <= 0;
            weight_read_count <= 0;
            commit_valid_under_fault_count <= 0;
            cycle_count <= 0;
            last_update_lane0 <= '0;
            last_update_lane95 <= '0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (event_accept) event_accept_count <= event_accept_count + 1;
            if (descriptor_close_accept) close_accept_count <= close_accept_count + 1;
            if (observed_service_accept) begin
                service_count <= service_count + 1;
                if (observed_service_is_event)
                    service_event_count <= service_event_count + 1;
                else
                    load_count <= load_count + 1;
            end
            if (weight_rd_en) weight_read_count <= weight_read_count + 1;
            if (mapped_update_accept) begin
                update_count <= update_count + 1;
                last_update_lane0 <= dut.numeric_island.mapper_update_delta[0 +: 19];
                last_update_lane95 <= dut.numeric_island.mapper_update_delta[95*19 +: 19];
            end
            if (descriptor_done) descriptor_done_count <= descriptor_done_count + 1;
            if (protocol_error && commit_valid)
                commit_valid_under_fault_count <= commit_valid_under_fault_count + 1;
        end
    end

    task automatic reset_all(input int latency_mode);
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            accumulator_window_start_valid = 1'b0;
            event_valid = 1'b0;
            descriptor_close_valid = 1'b0;
            accumulator_window_end_valid = 1'b0;
            commit_ready = 1'b0;
            memory_latency_mode = latency_mode;
            weight_rd_data = '0;
            delayed_response_q = '0;
            for (int lane = 0; lane < LANES; lane++) begin
                lane_mem_rd_data[lane] = '0;
                for (int address = 0; address < BLOCKS*WIN_ROWS; address++)
                    lane_memory[lane][address] = '0;
            end
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
            repeat (2) @(posedge clk_core);
        end
    endtask

    task automatic start_window;
        begin
            @(negedge clk_core);
            accumulator_window_start_valid = 1'b1;
            do @(posedge clk_core); while (!accumulator_window_start_accept);
            @(negedge clk_core);
            accumulator_window_start_valid = 1'b0;
        end
    endtask

    task automatic send_event(
        input int source, input int block_id, input int row,
        input bit negate, input int base, input int context,
        input int extra_hold_cycles
    );
        begin
            @(negedge clk_core);
            event_source = source[3:0];
            event_block = block_id[2:0];
            event_row_offset = row[8:0];
            event_negate = negate;
            window_base_row = base[11:0];
            window_context = context[15:0];
            event_valid = 1'b1;
            do @(posedge clk_core); while (!event_accept);
            repeat (extra_hold_cycles) @(posedge clk_core);
            @(negedge clk_core);
            event_valid = 1'b0;
        end
    endtask

    task automatic close_descriptor(input int base, input int context);
        begin
            @(negedge clk_core);
            window_base_row = base[11:0];
            window_context = context[15:0];
            descriptor_close_valid = 1'b1;
            do @(posedge clk_core); while (!descriptor_close_accept);
            @(negedge clk_core);
            descriptor_close_valid = 1'b0;
        end
    endtask

    task automatic wait_counts(input int updates, input int dones);
        int start_cycle;
        begin
            start_cycle = cycle_count;
            while (update_count < updates || descriptor_done_count < dones) begin
                @(posedge clk_core);
                if (cycle_count - start_cycle > 2000)
                    $fatal(1, "M121 independent watchdog updates=%0d/%0d done=%0d/%0d err=%0d",
                           update_count, updates, descriptor_done_count, dones,
                           protocol_error);
            end
            repeat (3) @(posedge clk_core);
        end
    endtask

    task automatic end_window_accept;
        begin
            @(negedge clk_core);
            accumulator_window_end_valid = 1'b1;
            do @(posedge clk_core); while (!accumulator_window_end_accept);
            @(negedge clk_core);
            accumulator_window_end_valid = 1'b0;
        end
    endtask

    initial begin
        integer signed expected0, expected95;
        integer start_cycle;
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
        memory_latency_mode = 1;
        delayed_response_q = '0;

        // Exact held valid is grace-suppressed, but a whole descriptor replay
        // after a sampled-low gap is a second legal descriptor with no ID dedup.
        reset_all(1);
        start_window();
        send_event(0, 0, 0, 0, 100, 16'h1111, 3);
        if (event_accept_count != 1 || protocol_error)
            $fatal(1, "held-valid grace failure accepts=%0d err=%0d",
                   event_accept_count, protocol_error);
        close_descriptor(100, 16'h1111);
        wait_counts(1, 1);
        send_event(0, 0, 0, 0, 100, 16'h1111, 0);
        close_descriptor(100, 16'h1111);
        wait_counts(2, 2);
        if (event_accept_count != 2 || close_accept_count != 2
                || service_count != 8 || load_count != 6
                || service_event_count != 2 || weight_read_count != 6
                || update_count != 2 || protocol_error)
            $fatal(1, "descriptor replay counterexample shape mismatch");
        $display("COUNTEREXAMPLE descriptor_replay_accepted accepts=2 closes=2 loads=6 events=2 updates=2 protocol_error=0 same_base=100 same_context=1111");

        // A scheduler-side sticky fault is combined at the top, but does not
        // gate numeric window_end or commit.  Previously accumulated partial
        // data can therefore be externally consumed while top protocol_error=1.
        reset_all(1);
        start_window();
        send_event(0, 0, 1, 0, 200, 16'h2222, 0);
        close_descriptor(200, 16'h2222);
        wait_counts(1, 1);
        @(negedge clk_core);
        event_source = 0;
        event_block = 0;
        event_row_offset = 9'd400;
        event_negate = 0;
        window_base_row = 12'd300;
        window_context = 16'h3333;
        event_valid = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        event_valid = 1'b0;
        if (!protocol_error || !scheduler_protocol_error
                || numeric_protocol_error)
            $fatal(1, "scheduler fault classification mismatch top=%0d sched=%0d numeric=%0d",
                   protocol_error, scheduler_protocol_error,
                   numeric_protocol_error);
        if (!accumulator_window_end_ready)
            $fatal(1, "scheduler fault unexpectedly gated numeric end ready");
        end_window_accept();
        start_cycle = cycle_count;
        while (!commit_valid) begin
            @(posedge clk_core);
            if (cycle_count - start_cycle > 20)
                $fatal(1, "commit did not escape combined fault");
        end
        if (!protocol_error || !commit_valid)
            $fatal(1, "fault/commit counterexample missing");
        repeat (2) @(posedge clk_core);
        if (commit_valid_under_fault_count == 0)
            $fatal(1, "commit-under-fault counter did not increment");
        $display("COUNTEREXAMPLE scheduler_fault_commit_escape end_accept=1 commit_valid_under_top_error=1 scheduler_error=1 numeric_error=0");

        // The same token counts remain legal with a two-cycle response model,
        // but the fixed-one-cycle tail interpretation maps stale beat data.
        reset_all(2);
        start_window();
        send_event(0, 0, 2, 0, 400, 16'h4444, 0);
        close_descriptor(400, 16'h4444);
        wait_counts(1, 1);
        expected0 = weight_value(0, 0);
        expected95 = weight_value(0, 95);
        if (service_count != 4 || load_count != 3
                || service_event_count != 1 || weight_read_count != 3
                || update_count != 1 || protocol_error)
            $fatal(1, "two-cycle latency counter shape mismatch");
        if ($signed(last_update_lane0) === expected0
                && $signed(last_update_lane95) === expected95)
            $fatal(1, "two-cycle latency unexpectedly preserved payload");
        $display("COUNTEREXAMPLE delayed_weight_response counters_ok=1 protocol_error=0 loads=3 events=1 updates=1 got_lane0=%0d expected_lane0=%0d got_lane95=%0d expected_lane95=%0d",
                 $signed(last_update_lane0), expected0,
                 $signed(last_update_lane95), expected95);

        $display("PASS M121 independent counterexamples held_valid_grace=1 whole_descriptor_replay_accepted=1 scheduler_fault_commit_escape=1 delayed_response_data_corruption=1");
        $finish;
    end
endmodule

`default_nettype wire
