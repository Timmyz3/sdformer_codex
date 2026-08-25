`timescale 1ns/1ps
`default_nettype none

module tb_m120_pwp_tail_mapper_signed19_accumulator_island;
    localparam int LANES = 96;
    localparam int ACC_BITS = 19;
    localparam int WIN_ROWS = 384;
    localparam int BLOCKS = 8;
    localparam int COMMIT_VECTORS = WIN_ROWS * BLOCKS;

    logic clk_core, rst_core;
    logic window_start_valid, window_start_ready, window_start_accept;
    logic service_valid, service_ready, service_is_event;
    logic [3:0] service_source;
    logic [2:0] service_block;
    logic [1:0] service_load_beat;
    logic [8:0] service_row_offset;
    logic service_negate, service_last_for_key, service_accept;
    logic weight_rd_en;
    logic [6:0] weight_rd_key;
    logic [1:0] weight_rd_beat;
    logic [255:0] weight_rd_data;
    logic window_end_valid, window_end_ready, window_end_accept;
    logic commit_valid, commit_ready;
    logic [2:0] commit_block;
    logic [8:0] commit_row;
    logic [1823:0] commit_data;
    logic commit_last, window_done;
    logic lane_mem_rd_en;
    logic [11:0] lane_mem_rd_addr;
    logic [18:0] lane_mem_rd_data [0:LANES-1];
    logic lane_mem_wr_en;
    logic [11:0] lane_mem_wr_addr;
    logic [18:0] lane_mem_wr_data [0:LANES-1];
    logic mapped_update_accept, tail_bypass_available, mapper_busy;
    logic accumulator_window_active, protocol_error, busy;

    logic [18:0] lane_memory [0:LANES-1][0:BLOCKS*WIN_ROWS-1];
    integer signed reference [0:BLOCKS-1][0:WIN_ROWS-1][0:LANES-1];

    int cycle_count;
    int start_accepts;
    int end_accepts;
    int accepted_loads;
    int weight_reads;
    int accepted_events;
    int mapped_updates;
    int lane_memory_writes;
    int mapped_update_ii1_pairs;
    int lane_rw_overlap_cycles;
    int tail_bypass_hits;
    int negate_events;
    int commit_accepts;
    int commit_stalls;
    int commit_lane_checks;
    int completed_windows;
    int protocol_attacks;
    int expected_commit_block;
    int expected_commit_row;
    bit previous_load2_accept;
    bit previous_mapped_update_accept;
    bit positive_phase;

    m120_pwp_tail_mapper_signed19_accumulator_island dut (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start_valid(window_start_valid),
        .window_start_ready(window_start_ready),
        .window_start_accept(window_start_accept),
        .service_valid(service_valid),
        .service_ready(service_ready),
        .service_is_event(service_is_event),
        .service_source(service_source),
        .service_block(service_block),
        .service_load_beat(service_load_beat),
        .service_row_offset(service_row_offset),
        .service_negate(service_negate),
        .service_last_for_key(service_last_for_key),
        .service_accept(service_accept),
        .weight_rd_en(weight_rd_en),
        .weight_rd_key(weight_rd_key),
        .weight_rd_beat(weight_rd_beat),
        .weight_rd_data(weight_rd_data),
        .window_end_valid(window_end_valid),
        .window_end_ready(window_end_ready),
        .window_end_accept(window_end_accept),
        .commit_valid(commit_valid),
        .commit_ready(commit_ready),
        .commit_block(commit_block),
        .commit_row(commit_row),
        .commit_data(commit_data),
        .commit_last(commit_last),
        .window_done(window_done),
        .lane_mem_rd_en(lane_mem_rd_en),
        .lane_mem_rd_addr(lane_mem_rd_addr),
        .lane_mem_rd_data(lane_mem_rd_data),
        .lane_mem_wr_en(lane_mem_wr_en),
        .lane_mem_wr_addr(lane_mem_wr_addr),
        .lane_mem_wr_data(lane_mem_wr_data),
        .mapped_update_accept(mapped_update_accept),
        .tail_bypass_available(tail_bypass_available),
        .mapper_busy(mapper_busy),
        .accumulator_window_active(accumulator_window_active),
        .protocol_error(protocol_error),
        .busy(busy)
    );

    m120_pwp_tail_mapper_signed19_accumulator_island_assertions checks (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start_valid(window_start_valid),
        .window_start_ready(window_start_ready),
        .window_start_accept(window_start_accept),
        .service_valid(service_valid),
        .service_ready(service_ready),
        .service_is_event(service_is_event),
        .service_load_beat(service_load_beat),
        .service_accept(service_accept),
        .weight_rd_en(weight_rd_en),
        .window_end_valid(window_end_valid),
        .window_end_ready(window_end_ready),
        .window_end_accept(window_end_accept),
        .mapped_update_accept(mapped_update_accept),
        .tail_bypass_available(tail_bypass_available),
        .commit_valid(commit_valid),
        .commit_ready(commit_ready),
        .commit_block(commit_block),
        .commit_row(commit_row),
        .commit_data(commit_data),
        .commit_last(commit_last),
        .window_done(window_done),
        .lane_mem_rd_en(lane_mem_rd_en),
        .lane_mem_rd_addr(lane_mem_rd_addr),
        .lane_mem_wr_en(lane_mem_wr_en),
        .lane_mem_wr_addr(lane_mem_wr_addr),
        .mapper_busy(mapper_busy),
        .accumulator_window_active(accumulator_window_active),
        .protocol_error(protocol_error),
        .busy(busy)
    );

    always #1 clk_core = ~clk_core;

    function automatic integer signed weight_value(
        input int key,
        input int lane
    );
        weight_value = ((key * 37 + lane * 29) & 8'hff) - 128;
    endfunction

    function automatic int event_row(input int key, input int event_index);
        event_row = (key * 7 + event_index * 31) % WIN_ROWS;
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
            previous_mapped_update_accept <= 1'b0;
            commit_ready <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;
            commit_ready <= ((cycle_count % 13) != 4)
                         && ((cycle_count % 31) != 9);
            if (window_start_accept)
                start_accepts <= start_accepts + 1;
            if (window_end_accept)
                end_accepts <= end_accepts + 1;
            if (service_accept && !service_is_event)
                accepted_loads <= accepted_loads + 1;
            if (weight_rd_en)
                weight_reads <= weight_reads + 1;
            if (service_accept && service_is_event) begin
                accepted_events <= accepted_events + 1;
                if (service_negate)
                    negate_events <= negate_events + 1;
                if (previous_load2_accept) begin
                    if (!tail_bypass_available)
                        $fatal(1, "M120 first event missing tail bypass");
                    tail_bypass_hits <= tail_bypass_hits + 1;
                end
                for (int lane = 0; lane < LANES; lane++) begin
                    value = weight_value(
                        {service_source, service_block}, lane);
                    if (service_negate)
                        value = -value;
                    reference[service_block][service_row_offset][lane]
                        = reference[service_block]
                                   [service_row_offset][lane] + value;
                end
            end
            previous_load2_accept <= service_accept && !service_is_event
                                  && service_load_beat == 2;

            if (mapped_update_accept) begin
                mapped_updates <= mapped_updates + 1;
                if (previous_mapped_update_accept)
                    mapped_update_ii1_pairs
                        <= mapped_update_ii1_pairs + 1;
            end
            previous_mapped_update_accept <= mapped_update_accept;
            if (lane_mem_rd_en && lane_mem_wr_en)
                lane_rw_overlap_cycles <= lane_rw_overlap_cycles + 1;

            if (commit_valid && !commit_ready)
                commit_stalls <= commit_stalls + 1;
            if (commit_valid && commit_ready) begin
                if (commit_block !== expected_commit_block[2:0]
                        || commit_row !== expected_commit_row[8:0])
                    $fatal(1, "M120 commit order mismatch expected=%0d/%0d got=%0d/%0d",
                           expected_commit_block, expected_commit_row,
                           commit_block, commit_row);
                for (int lane = 0; lane < LANES; lane++) begin
                    if ($signed(commit_data[lane * ACC_BITS +: ACC_BITS])
                            !== reference[expected_commit_block]
                                          [expected_commit_row][lane])
                        $fatal(1, "M120 commit numeric mismatch block=%0d row=%0d lane=%0d got=%0d expected=%0d",
                               expected_commit_block, expected_commit_row,
                               lane,
                               $signed(commit_data[
                                   lane * ACC_BITS +: ACC_BITS]),
                               reference[expected_commit_block]
                                        [expected_commit_row][lane]);
                end
                commit_lane_checks <= commit_lane_checks + LANES;
                commit_accepts <= commit_accepts + 1;
                if (commit_last !== (expected_commit_block == BLOCKS-1
                                     && expected_commit_row == WIN_ROWS-1))
                    $fatal(1, "M120 commit_last mismatch");
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
            if (window_done)
                completed_windows <= completed_windows + 1;
            if (positive_phase && protocol_error)
                $fatal(1, "M120 unexpected protocol error cycle=%0d",
                       cycle_count);
        end
    end

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            window_start_valid = 1'b0;
            service_valid = 1'b0;
            window_end_valid = 1'b0;
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

    task automatic start_window;
        begin
            clear_reference();
            @(negedge clk_core);
            window_start_valid = 1'b1;
            do @(posedge clk_core); while (!window_start_accept);
            @(negedge clk_core);
            window_start_valid = 1'b0;
        end
    endtask

    task automatic drive_load(input int key, input int beat);
        begin
            @(negedge clk_core);
            service_valid = 1'b1;
            service_is_event = 1'b0;
            service_source = key[6:3];
            service_block = key[2:0];
            service_load_beat = beat[1:0];
            service_row_offset = '0;
            service_negate = 1'b0;
            service_last_for_key = 1'b0;
            do @(posedge clk_core); while (!service_accept);
        end
    endtask

    task automatic drive_event(
        input int window_index,
        input int key,
        input int event_index
    );
        begin
            @(negedge clk_core);
            service_valid = 1'b1;
            service_is_event = 1'b1;
            service_source = key[6:3];
            service_block = key[2:0];
            service_load_beat = '0;
            service_row_offset = event_row(key, event_index)[8:0];
            service_negate = ((window_index + key + event_index) & 1) != 0;
            service_last_for_key = event_index == 3;
            do @(posedge clk_core); while (!service_accept);
        end
    endtask

    task automatic stop_service_and_wait_updates;
        int start_cycle;
        begin
            @(negedge clk_core);
            service_valid = 1'b0;
            start_cycle = cycle_count;
            while (mapper_busy || mapped_updates != accepted_events) begin
                @(posedge clk_core);
                if (cycle_count - start_cycle > 1000)
                    $fatal(1, "M120 mapper drain watchdog events=%0d updates=%0d busy=%0d",
                           accepted_events, mapped_updates, mapper_busy);
            end
        end
    endtask

    task automatic end_window;
        begin
            @(negedge clk_core);
            window_end_valid = 1'b1;
            do @(posedge clk_core); while (!window_end_accept);
            @(negedge clk_core);
            window_end_valid = 1'b0;
        end
    endtask

    task automatic wait_window_done;
        int start_cycle;
        begin
            start_cycle = cycle_count;
            while (!window_done) begin
                @(posedge clk_core);
                if (cycle_count - start_cycle > 10000)
                    $fatal(1, "M120 commit watchdog commits=%0d", commit_accepts);
            end
            @(posedge clk_core);
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        window_start_valid = 1'b0;
        service_valid = 1'b0;
        service_is_event = 1'b0;
        service_source = '0;
        service_block = '0;
        service_load_beat = '0;
        service_row_offset = '0;
        service_negate = 1'b0;
        service_last_for_key = 1'b0;
        weight_rd_data = 'x;
        window_end_valid = 1'b0;
        commit_ready = 1'b0;
        cycle_count = 0;
        start_accepts = 0;
        end_accepts = 0;
        accepted_loads = 0;
        weight_reads = 0;
        accepted_events = 0;
        mapped_updates = 0;
        lane_memory_writes = 0;
        mapped_update_ii1_pairs = 0;
        lane_rw_overlap_cycles = 0;
        tail_bypass_hits = 0;
        negate_events = 0;
        commit_accepts = 0;
        commit_stalls = 0;
        commit_lane_checks = 0;
        completed_windows = 0;
        protocol_attacks = 0;
        expected_commit_block = 0;
        expected_commit_row = 0;
        previous_load2_accept = 1'b0;
        previous_mapped_update_accept = 1'b0;
        positive_phase = 1'b1;
        for (int lane = 0; lane < LANES; lane++) begin
            lane_mem_rd_data[lane] = 'x;
            for (int address = 0; address < BLOCKS * WIN_ROWS; address++)
                lane_memory[lane][address] = 'x;
        end

        reset_dut();
        for (int window_index = 0; window_index < 2; window_index++) begin
            start_window();
            for (int key = 0; key < 128; key++) begin
                drive_load(key, 0);
                drive_load(key, 1);
                drive_load(key, 2);
                for (int event_index = 0; event_index < 4; event_index++)
                    drive_event(window_index, key, event_index);
            end
            stop_service_and_wait_updates();
            end_window();
            wait_window_done();
        end

        repeat (3) @(posedge clk_core);
        if (start_accepts != 2 || end_accepts != 2
                || accepted_loads != 768 || weight_reads != 768
                || accepted_events != 1024 || mapped_updates != 1024
                || lane_memory_writes != 1024
                || mapped_update_ii1_pairs != 768
                || lane_rw_overlap_cycles != 768
                || tail_bypass_hits != 256 || negate_events != 512
                || commit_accepts != 2 * COMMIT_VECTORS
                || commit_lane_checks != 2 * COMMIT_VECTORS * LANES
                || completed_windows != 2 || commit_stalls == 0)
            $fatal(1, "M120 conservation mismatch starts=%0d ends=%0d loads=%0d reads=%0d events=%0d updates=%0d writes=%0d ii1=%0d overlap=%0d tail=%0d negate=%0d commits=%0d lane_checks=%0d windows=%0d stalls=%0d",
                   start_accepts, end_accepts, accepted_loads, weight_reads,
                   accepted_events, mapped_updates, lane_memory_writes,
                   mapped_update_ii1_pairs, lane_rw_overlap_cycles,
                   tail_bypass_hits, negate_events, commit_accepts,
                   commit_lane_checks, completed_windows, commit_stalls);

        positive_phase = 1'b0;
        reset_dut();
        @(negedge clk_core);
        service_valid = 1'b1;
        service_is_event = 1'b0;
        service_source = 0;
        service_block = 0;
        service_load_beat = 0;
        @(posedge clk_core);
        if (!protocol_error || service_ready || service_accept || weight_rd_en)
            $fatal(1, "M120 service outside window did not fail closed");
        protocol_attacks = protocol_attacks + 1;
        @(negedge clk_core);
        service_valid = 1'b0;
        repeat (2) @(posedge clk_core);
        if (!protocol_error)
            $fatal(1, "M120 wrapper fault not sticky");

        $display("PASS M120 integrated PWP tail mapper signed19 accumulator island VCS windows=2 groups=256 weight_loads=768 weight_reads=768 events=1024 mapped_updates=1024 accumulator_writes=1024 mapped_ii1_pairs=768 lane_rw_overlap=768 tail_bypass_first_events=256 negate_events=512 commits=6144 commit_lane_checks=589824 commit_stalls=%0d protocol_attacks=1 weight_port_bits=256 weight_read_latency=1 accumulator_lanes=96 accumulator_bits=19 accumulator_bytes=700416 exact_once_directed=true m117_scheduler_integrated=false heldout_trace_replay=false foundry_sram_macro=false scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false",
                 commit_stalls);
        $finish;
    end
endmodule

`default_nettype wire
