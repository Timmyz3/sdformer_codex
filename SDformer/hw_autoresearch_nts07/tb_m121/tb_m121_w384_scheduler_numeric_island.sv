`timescale 1ns/1ps
`default_nettype none

module tb_m121_w384_scheduler_numeric_island;
    localparam int LANES = 96;
    localparam int ACC_BITS = 19;
    localparam int WIN_ROWS = 384;
    localparam int BLOCKS = 8;
    localparam int KEYS = 128;
    localparam int DESCRIPTORS = 2;
    localparam int EVENTS_PER_DESCRIPTOR = KEYS * WIN_ROWS;

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
    logic observed_service_accept, observed_service_is_event;
    logic observed_numeric_service_accept;
    logic [3:0] observed_service_source;
    logic [2:0] observed_service_block;
    logic [1:0] observed_service_load_beat;
    logic [8:0] observed_service_row_offset;
    logic observed_service_negate, observed_service_last_for_key;
    logic mapped_update_accept, tail_bypass_available;
    logic scheduler_protocol_error, numeric_protocol_error;
    logic protocol_error, busy;

    logic [18:0] lane_memory [0:LANES-1][0:BLOCKS*WIN_ROWS-1];
    integer signed reference [0:BLOCKS-1][0:WIN_ROWS-1][0:LANES-1];

    int cycle_count;
    int ingress_events;
    int close_accepts;
    int prefetch_accepts;
    int service_tokens;
    int service_loads;
    int service_events;
    int weight_reads;
    int tail_bypass_hits;
    int zero_bubble_key_transitions;
    int downstream_backpressure_cycles;
    int mapped_updates;
    int mapped_update_ii1_pairs;
    int lane_memory_writes;
    int lane_rw_overlap_cycles;
    int descriptor_dones;
    int commit_accepts;
    int commit_lane_checks;
    int commit_stalls;
    int protocol_attacks;
    int expected_commit_block;
    int expected_commit_row;
    bit previous_load2_accept;
    bit previous_nonfinal_last_event_accept;
    bit previous_mapped_update_accept;
    bit positive_phase;

    m121_w384_scheduler_numeric_island dut (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .accumulator_window_start_valid(accumulator_window_start_valid),
        .accumulator_window_start_ready(accumulator_window_start_ready),
        .accumulator_window_start_accept(accumulator_window_start_accept),
        .event_valid(event_valid),
        .event_ready(event_ready),
        .event_source(event_source),
        .event_block(event_block),
        .event_row_offset(event_row_offset),
        .event_negate(event_negate),
        .window_base_row(window_base_row),
        .window_context(window_context),
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
        .weight_rd_en(weight_rd_en),
        .weight_rd_key(weight_rd_key),
        .weight_rd_beat(weight_rd_beat),
        .weight_rd_data(weight_rd_data),
        .accumulator_window_end_valid(accumulator_window_end_valid),
        .accumulator_window_end_ready(accumulator_window_end_ready),
        .accumulator_window_end_accept(accumulator_window_end_accept),
        .commit_valid(commit_valid),
        .commit_ready(commit_ready),
        .commit_block(commit_block),
        .commit_row(commit_row),
        .commit_data(commit_data),
        .commit_last(commit_last),
        .accumulator_window_done(accumulator_window_done),
        .lane_mem_rd_en(lane_mem_rd_en),
        .lane_mem_rd_addr(lane_mem_rd_addr),
        .lane_mem_rd_data(lane_mem_rd_data),
        .lane_mem_wr_en(lane_mem_wr_en),
        .lane_mem_wr_addr(lane_mem_wr_addr),
        .lane_mem_wr_data(lane_mem_wr_data),
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
        .protocol_error(protocol_error),
        .busy(busy)
    );

    m121_w384_scheduler_numeric_island_assertions checks (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .event_valid(event_valid),
        .event_ready(event_ready),
        .event_accept(event_accept),
        .descriptor_close_valid(descriptor_close_valid),
        .descriptor_close_ready(descriptor_close_ready),
        .descriptor_close_accept(descriptor_close_accept),
        .observed_service_valid(observed_service_valid),
        .observed_service_ready(observed_service_ready),
        .observed_service_accept(observed_service_accept),
        .observed_numeric_service_accept(observed_numeric_service_accept),
        .observed_service_is_event(observed_service_is_event),
        .observed_service_source(observed_service_source),
        .observed_service_block(observed_service_block),
        .observed_service_load_beat(observed_service_load_beat),
        .observed_service_last_for_key(observed_service_last_for_key),
        .weight_rd_en(weight_rd_en),
        .mapped_update_accept(mapped_update_accept),
        .tail_bypass_available(tail_bypass_available),
        .commit_valid(commit_valid),
        .commit_ready(commit_ready),
        .commit_block(commit_block),
        .commit_row(commit_row),
        .commit_data(commit_data),
        .commit_last(commit_last),
        .accumulator_window_done(accumulator_window_done),
        .descriptor_done(descriptor_done),
        .lane_mem_rd_en(lane_mem_rd_en),
        .lane_mem_wr_en(lane_mem_wr_en),
        .scheduler_protocol_error(scheduler_protocol_error),
        .numeric_protocol_error(numeric_protocol_error),
        .protocol_error(protocol_error)
    );

    always #1 clk_core = ~clk_core;

    function automatic integer signed weight_value(
        input int key,
        input int lane
    );
        weight_value = ((key * 37 + lane * 29) & 8'hff) - 128;
    endfunction

    function automatic int descriptor_base(input int descriptor_index);
        descriptor_base = descriptor_index == 0 ? 100 : 600;
    endfunction

    function automatic int descriptor_context(input int descriptor_index);
        descriptor_context = descriptor_index == 0 ? 16'h1111 : 16'h2222;
    endfunction

    function automatic bit descriptor_negate(
        input int descriptor_index,
        input int key,
        input int row
    );
        descriptor_negate = ((descriptor_index + key + row) & 1) != 0;
    endfunction

    always @(posedge clk_core) begin : weight_memory_model
        logic [255:0] response;
        integer signed value;
        if (weight_rd_en) begin
            response = '0;
            for (int byte_index = 0; byte_index < 32; byte_index++) begin
                value = weight_value(weight_rd_key,
                    weight_rd_beat * 32 + byte_index);
                response[byte_index * 8 +: 8] = value[7:0];
            end
            weight_rd_data <= response;
        end
    end

    always @(posedge clk_core) begin : lane_memory_model
        for (int lane = 0; lane < LANES; lane++) begin
            if (lane_mem_rd_en)
                lane_mem_rd_data[lane]
                    <= lane_memory[lane][lane_mem_rd_addr];
            if (lane_mem_wr_en)
                lane_memory[lane][lane_mem_wr_addr]
                    <= lane_mem_wr_data[lane];
        end
        if (lane_mem_wr_en)
            lane_memory_writes <= lane_memory_writes + 1;
    end

    always @(posedge clk_core) begin : scoreboard
        integer signed value;
        if (rst_core) begin
            previous_load2_accept <= 1'b0;
            previous_nonfinal_last_event_accept <= 1'b0;
            previous_mapped_update_accept <= 1'b0;
            commit_ready <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;
            commit_ready <= ((cycle_count % 13) != 4)
                         && ((cycle_count % 31) != 9);
            if (event_accept)
                ingress_events <= ingress_events + 1;
            if (descriptor_close_accept)
                close_accepts <= close_accepts + 1;
            if (weight_prefetch_accept)
                prefetch_accepts <= prefetch_accepts + 1;
            if (observed_service_valid && !observed_service_ready)
                downstream_backpressure_cycles
                    <= downstream_backpressure_cycles + 1;
            if (observed_service_accept) begin
                service_tokens <= service_tokens + 1;
                if (observed_service_is_event) begin
                    service_events <= service_events + 1;
                    if (previous_load2_accept) begin
                        if (!tail_bypass_available)
                            $fatal(1, "M121 first event missing tail bypass");
                        tail_bypass_hits <= tail_bypass_hits + 1;
                    end
                    for (int lane = 0; lane < LANES; lane++) begin
                        value = weight_value(
                            {observed_service_source,
                             observed_service_block}, lane);
                        if (observed_service_negate)
                            value = -value;
                        reference[observed_service_block]
                                 [observed_service_row_offset][lane]
                            = reference[observed_service_block]
                                       [observed_service_row_offset][lane]
                            + value;
                    end
                end else begin
                    service_loads <= service_loads + 1;
                end
            end
            if (weight_rd_en)
                weight_reads <= weight_reads + 1;

            if (previous_nonfinal_last_event_accept) begin
                if (!observed_service_accept || observed_service_is_event
                        || observed_service_load_beat != 0)
                    $fatal(1, "M121 key transition inserted a downstream bubble");
                zero_bubble_key_transitions
                    <= zero_bubble_key_transitions + 1;
            end
            previous_nonfinal_last_event_accept
                <= observed_service_accept && observed_service_is_event
                && observed_service_last_for_key
                && {observed_service_source, observed_service_block} < 127;
            previous_load2_accept <= observed_service_accept
                                  && !observed_service_is_event
                                  && observed_service_load_beat == 2;

            if (mapped_update_accept) begin
                mapped_updates <= mapped_updates + 1;
                if (previous_mapped_update_accept)
                    mapped_update_ii1_pairs
                        <= mapped_update_ii1_pairs + 1;
            end
            previous_mapped_update_accept <= mapped_update_accept;
            if (lane_mem_rd_en && lane_mem_wr_en)
                lane_rw_overlap_cycles <= lane_rw_overlap_cycles + 1;

            if (descriptor_done) begin
                if (descriptor_done_empty
                        || descriptor_done_base_row
                           !== descriptor_base(descriptor_dones)[11:0]
                        || descriptor_done_context
                           !== descriptor_context(descriptor_dones)[15:0])
                    $fatal(1, "M121 descriptor_done identity mismatch index=%0d",
                           descriptor_dones);
                descriptor_dones <= descriptor_dones + 1;
            end
            if (commit_valid && !commit_ready)
                commit_stalls <= commit_stalls + 1;
            if (commit_valid && commit_ready) begin
                if (commit_block !== expected_commit_block[2:0]
                        || commit_row !== expected_commit_row[8:0])
                    $fatal(1, "M121 commit order mismatch");
                for (int lane = 0; lane < LANES; lane++) begin
                    if ($signed(commit_data[lane * ACC_BITS +: ACC_BITS])
                            !== reference[expected_commit_block]
                                          [expected_commit_row][lane])
                        $fatal(1, "M121 commit numeric mismatch block=%0d row=%0d lane=%0d got=%0d expected=%0d",
                               expected_commit_block, expected_commit_row,
                               lane,
                               $signed(commit_data[
                                   lane * ACC_BITS +: ACC_BITS]),
                               reference[expected_commit_block]
                                        [expected_commit_row][lane]);
                end
                commit_lane_checks <= commit_lane_checks + LANES;
                commit_accepts <= commit_accepts + 1;
                if (expected_commit_row == WIN_ROWS-1) begin
                    expected_commit_row <= 0;
                    if (expected_commit_block == BLOCKS-1)
                        expected_commit_block <= 0;
                    else
                        expected_commit_block <= expected_commit_block + 1;
                end else begin
                    expected_commit_row <= expected_commit_row + 1;
                end
            end
            if (positive_phase && protocol_error)
                $fatal(1, "M121 unexpected protocol error cycle=%0d sched=%0d numeric=%0d",
                       cycle_count, scheduler_protocol_error,
                       numeric_protocol_error);
        end
    end

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            accumulator_window_start_valid = 1'b0;
            event_valid = 1'b0;
            descriptor_close_valid = 1'b0;
            accumulator_window_end_valid = 1'b0;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
            repeat (2) @(posedge clk_core);
        end
    endtask

    task automatic clear_reference;
        begin
            for (int block_id = 0; block_id < BLOCKS; block_id++)
                for (int row = 0; row < WIN_ROWS; row++)
                    for (int lane = 0; lane < LANES; lane++)
                        reference[block_id][row][lane] = 0;
        end
    endtask

    task automatic start_accumulator_window;
        begin
            @(negedge clk_core);
            accumulator_window_start_valid = 1'b1;
            do @(posedge clk_core);
                while (!accumulator_window_start_accept);
            @(negedge clk_core);
            accumulator_window_start_valid = 1'b0;
        end
    endtask

    task automatic fill_full_descriptor(input int descriptor_index);
        begin
            for (int key = 0; key < KEYS; key++) begin
                for (int row = 0; row < WIN_ROWS; row++) begin
                    @(negedge clk_core);
                    event_valid = 1'b1;
                    event_source = key[6:3];
                    event_block = key[2:0];
                    event_row_offset = row[8:0];
                    event_negate = descriptor_negate(
                        descriptor_index, key, row);
                    window_base_row
                        = descriptor_base(descriptor_index)[11:0];
                    window_context
                        = descriptor_context(descriptor_index)[15:0];
                    do @(posedge clk_core); while (!event_accept);
                end
            end
            @(negedge clk_core);
            event_valid = 1'b0;
        end
    endtask

    task automatic close_descriptor(input int descriptor_index);
        begin
            @(negedge clk_core);
            window_base_row = descriptor_base(descriptor_index)[11:0];
            window_context = descriptor_context(descriptor_index)[15:0];
            descriptor_close_valid = 1'b1;
            do @(posedge clk_core); while (!descriptor_close_accept);
            @(negedge clk_core);
            descriptor_close_valid = 1'b0;
        end
    endtask

    task automatic wait_numeric_drain;
        int start_cycle;
        begin
            start_cycle = cycle_count;
            while (descriptor_dones != DESCRIPTORS
                    || mapped_updates != service_events
                    || observed_service_valid
                    || dut.scheduler_busy || dut.numeric_mapper_busy) begin
                @(posedge clk_core);
                if (cycle_count - start_cycle > 150000)
                    $fatal(1, "M121 service drain watchdog done=%0d events=%0d updates=%0d valid=%0d sched_busy=%0d map_busy=%0d",
                           descriptor_dones, service_events, mapped_updates,
                           observed_service_valid, dut.scheduler_busy,
                           dut.numeric_mapper_busy);
            end
        end
    endtask

    task automatic end_accumulator_window;
        begin
            @(negedge clk_core);
            accumulator_window_end_valid = 1'b1;
            do @(posedge clk_core); while (!accumulator_window_end_accept);
            @(negedge clk_core);
            accumulator_window_end_valid = 1'b0;
        end
    endtask

    task automatic wait_commit_done;
        int start_cycle;
        begin
            start_cycle = cycle_count;
            while (!accumulator_window_done) begin
                @(posedge clk_core);
                if (cycle_count - start_cycle > 10000)
                    $fatal(1, "M121 commit watchdog commits=%0d",
                           commit_accepts);
            end
            @(posedge clk_core);
        end
    endtask

    initial begin
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
        weight_rd_data = 'x;
        accumulator_window_end_valid = 1'b0;
        commit_ready = 1'b0;
        cycle_count = 0;
        ingress_events = 0;
        close_accepts = 0;
        prefetch_accepts = 0;
        service_tokens = 0;
        service_loads = 0;
        service_events = 0;
        weight_reads = 0;
        tail_bypass_hits = 0;
        zero_bubble_key_transitions = 0;
        downstream_backpressure_cycles = 0;
        mapped_updates = 0;
        mapped_update_ii1_pairs = 0;
        lane_memory_writes = 0;
        lane_rw_overlap_cycles = 0;
        descriptor_dones = 0;
        commit_accepts = 0;
        commit_lane_checks = 0;
        commit_stalls = 0;
        protocol_attacks = 0;
        expected_commit_block = 0;
        expected_commit_row = 0;
        previous_load2_accept = 1'b0;
        previous_nonfinal_last_event_accept = 1'b0;
        previous_mapped_update_accept = 1'b0;
        positive_phase = 1'b1;
        for (int lane = 0; lane < LANES; lane++) begin
            lane_mem_rd_data[lane] = 'x;
            for (int address = 0; address < BLOCKS * WIN_ROWS; address++)
                lane_memory[lane][address] = 'x;
        end
        clear_reference();

        reset_dut();
        start_accumulator_window();
        fill_full_descriptor(0);
        close_descriptor(0);
        fill_full_descriptor(1);
        close_descriptor(1);
        wait_numeric_drain();
        end_accumulator_window();
        wait_commit_done();
        repeat (3) @(posedge clk_core);

        if (ingress_events != DESCRIPTORS * EVENTS_PER_DESCRIPTOR
                || close_accepts != DESCRIPTORS
                || prefetch_accepts != DESCRIPTORS * KEYS
                || service_tokens != 99072 || service_loads != 768
                || service_events != 98304 || weight_reads != 768
                || tail_bypass_hits != 256
                || zero_bubble_key_transitions != 254
                || downstream_backpressure_cycles != 0
                || mapped_updates != 98304
                || mapped_update_ii1_pairs != 98048
                || lane_memory_writes != 98304
                || lane_rw_overlap_cycles != 98048
                || descriptor_dones != DESCRIPTORS
                || commit_accepts != BLOCKS * WIN_ROWS
                || commit_lane_checks != BLOCKS * WIN_ROWS * LANES
                || commit_stalls == 0)
            $fatal(1, "M121 conservation mismatch ingress=%0d close=%0d prefetch=%0d tokens=%0d loads=%0d events=%0d reads=%0d tail=%0d zero=%0d downstream_stall=%0d updates=%0d ii1=%0d writes=%0d overlap=%0d done=%0d commits=%0d lanes=%0d stalls=%0d",
                   ingress_events, close_accepts, prefetch_accepts,
                   service_tokens, service_loads, service_events,
                   weight_reads, tail_bypass_hits,
                   zero_bubble_key_transitions,
                   downstream_backpressure_cycles, mapped_updates,
                   mapped_update_ii1_pairs, lane_memory_writes,
                   lane_rw_overlap_cycles, descriptor_dones,
                   commit_accepts, commit_lane_checks, commit_stalls);

        // After reset, a descriptor is allowed to fill, but its first service
        // token must fail closed if no accumulator window was started.
        positive_phase = 1'b0;
        reset_dut();
        @(negedge clk_core);
        event_valid = 1'b1;
        event_source = 0;
        event_block = 0;
        event_row_offset = 0;
        event_negate = 1'b0;
        window_base_row = 12'd900;
        window_context = 16'h3333;
        do @(posedge clk_core); while (!event_accept);
        @(negedge clk_core);
        event_valid = 1'b0;
        descriptor_close_valid = 1'b1;
        do @(posedge clk_core); while (!descriptor_close_accept);
        @(negedge clk_core);
        descriptor_close_valid = 1'b0;
        while (!protocol_error)
            @(posedge clk_core);
        if (!numeric_protocol_error || scheduler_protocol_error
                || weight_rd_en)
            $fatal(1, "M121 missing-window attack classification mismatch");
        protocol_attacks = protocol_attacks + 1;
        repeat (2) @(posedge clk_core);
        if (!protocol_error)
            $fatal(1, "M121 numeric fault not sticky");

        $display("PASS M121 W384 scheduler numeric island VCS descriptors=2 ingress_events=98304 active_keys=256 prefetches=256 service_tokens=99072 weight_loads=768 service_events=98304 weight_reads=768 tail_bypass_first_events=256 zero_bubble_key_transitions=254 downstream_backpressure_cycles=0 mapped_updates=98304 mapped_ii1_pairs=98048 accumulator_writes=98304 lane_rw_overlap=98048 descriptor_done=2 commits=3072 commit_lane_checks=294912 commit_stalls=%0d protocol_attacks=1 weight_port_bits=256 weight_read_latency=1 accumulator_lanes=96 accumulator_bits=19 accumulator_bytes=700416 directed_end_to_end_service_island=true heldout_trace_replay=false foundry_sram_macro=false module_cycle_projection=2.53546204172554 physical_speedup=false system_speedup=false headline=false",
                 commit_stalls);
        $finish;
    end
endmodule

`default_nettype wire
