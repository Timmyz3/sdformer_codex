`timescale 1ns/1ps
`default_nettype none

module tb_m120_with_m123_integrated_hammer;
    localparam int LANES = 96;
    localparam int ACC_BITS = 19;
    localparam int VECTOR_BITS = 1824;
    localparam int WIN_ROWS = 384;
    localparam int BLOCKS = 8;
    localparam int DEPTH = BLOCKS * WIN_ROWS;

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
    logic [VECTOR_BITS-1:0] commit_data;
    logic commit_last, window_done;
    logic lane_mem_rd_en;
    logic [11:0] lane_mem_rd_addr;
    logic [18:0] lane_mem_rd_data [0:LANES-1];
    logic lane_mem_wr_en;
    logic [11:0] lane_mem_wr_addr;
    logic [18:0] lane_mem_wr_data [0:LANES-1];
    logic mapped_update_accept, tail_bypass_available, mapper_busy;
    logic accumulator_window_active, protocol_error, busy;

    logic [18:0] lane_memory [0:LANES-1][0:DEPTH-1];
    integer signed reference [0:BLOCKS-1][0:WIN_ROWS-1][0:LANES-1];
    logic prior_event_accept, prior_event_negate;
    logic [2:0] prior_event_block;
    logic [8:0] prior_event_row;
    logic [6:0] prior_event_key;
    logic prior_mapped_accept;
    logic [11:0] prior_mapped_address;
    logic prior_load2_accept;
    logic prior_commit_stall, prior_commit_last_accept;
    logic [2:0] stalled_commit_block;
    logic [8:0] stalled_commit_row;
    logic [VECTOR_BITS-1:0] stalled_commit_data;
    logic stalled_commit_last;
    logic positive_phase, automatic_commit_ready;

    integer cycle_count;
    integer total_loads, total_weight_reads, total_events;
    integer total_mapped_updates, total_writes;
    integer positive_loads, positive_weight_reads, positive_events;
    integer positive_updates, positive_writes;
    integer positive_tail_hits, positive_negated_events;
    integer positive_ii1_pairs, positive_rw_overlap;
    integer mapper_lane_checks, commit_lane_checks;
    integer commit_accepts, commit_stalls, stall_releases;
    integer completed_windows, expected_commit_block, expected_commit_row;
    integer address_zero_writes, address_last_writes;
    integer malformed_beat_attacks, malformed_key_attacks;
    integer early_end_attacks, older_update_drain_checks;
    integer same_address_events_accepted, same_address_updates_written;
    integer duplicate_events_accepted, duplicate_updates_written;
    integer reset_events_accepted, reset_updates_written;

    m120_pwp_tail_mapper_signed19_accumulator_island dut (
        .clk_core(clk_core), .rst_core(rst_core),
        .window_start_valid(window_start_valid),
        .window_start_ready(window_start_ready),
        .window_start_accept(window_start_accept),
        .service_valid(service_valid), .service_ready(service_ready),
        .service_is_event(service_is_event),
        .service_source(service_source), .service_block(service_block),
        .service_load_beat(service_load_beat),
        .service_row_offset(service_row_offset),
        .service_negate(service_negate),
        .service_last_for_key(service_last_for_key),
        .service_accept(service_accept), .weight_rd_en(weight_rd_en),
        .weight_rd_key(weight_rd_key), .weight_rd_beat(weight_rd_beat),
        .weight_rd_data(weight_rd_data),
        .window_end_valid(window_end_valid),
        .window_end_ready(window_end_ready),
        .window_end_accept(window_end_accept),
        .commit_valid(commit_valid), .commit_ready(commit_ready),
        .commit_block(commit_block), .commit_row(commit_row),
        .commit_data(commit_data), .commit_last(commit_last),
        .window_done(window_done), .lane_mem_rd_en(lane_mem_rd_en),
        .lane_mem_rd_addr(lane_mem_rd_addr),
        .lane_mem_rd_data(lane_mem_rd_data),
        .lane_mem_wr_en(lane_mem_wr_en),
        .lane_mem_wr_addr(lane_mem_wr_addr),
        .lane_mem_wr_data(lane_mem_wr_data),
        .mapped_update_accept(mapped_update_accept),
        .tail_bypass_available(tail_bypass_available),
        .mapper_busy(mapper_busy),
        .accumulator_window_active(accumulator_window_active),
        .protocol_error(protocol_error), .busy(busy)
    );

    // Reuse the frozen production SVA on the positive legal campaign.  The
    // negative campaigns deliberately violate its next-cycle exact-once
    // property to establish the admitted boundary, so they are masked here
    // and checked by the independent scoreboard below.
    m120_pwp_tail_mapper_signed19_accumulator_island_assertions checks (
        .clk_core(clk_core), .rst_core(rst_core || !positive_phase),
        .window_start_valid(window_start_valid),
        .window_start_ready(window_start_ready),
        .window_start_accept(window_start_accept),
        .service_valid(service_valid), .service_ready(service_ready),
        .service_is_event(service_is_event),
        .service_load_beat(service_load_beat),
        .service_accept(service_accept), .weight_rd_en(weight_rd_en),
        .window_end_valid(window_end_valid),
        .window_end_ready(window_end_ready),
        .window_end_accept(window_end_accept),
        .mapped_update_accept(mapped_update_accept),
        .tail_bypass_available(tail_bypass_available),
        .commit_valid(commit_valid), .commit_ready(commit_ready),
        .commit_block(commit_block), .commit_row(commit_row),
        .commit_data(commit_data), .commit_last(commit_last),
        .window_done(window_done), .lane_mem_rd_en(lane_mem_rd_en),
        .lane_mem_rd_addr(lane_mem_rd_addr),
        .lane_mem_wr_en(lane_mem_wr_en),
        .lane_mem_wr_addr(lane_mem_wr_addr), .mapper_busy(mapper_busy),
        .accumulator_window_active(accumulator_window_active),
        .protocol_error(protocol_error), .busy(busy)
    );

    always #1 clk_core = ~clk_core;

    function automatic integer signed weight_value(
        input integer key,
        input integer lane
    );
        begin
            case (lane)
                0: weight_value = -128;
                1: weight_value = 127;
                2: weight_value = -1;
                3: weight_value = 1;
                default: weight_value = ((key * 37 + lane * 29) & 8'hff)
                                              - 128;
            endcase
        end
    endfunction

    function automatic integer positive_event_row(
        input integer key,
        input integer event_index
    );
        if (key == 127 && event_index == 3)
            positive_event_row = 383;
        else
            positive_event_row = (key * 7 + event_index * 31) % WIN_ROWS;
    endfunction

    function automatic logic [11:0] flat_addr(
        input integer block,
        input integer row
    );
        flat_addr = block * WIN_ROWS + row;
    endfunction

    always @(posedge clk_core) begin : synchronous_weight_memory
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

    always @(posedge clk_core) begin : synchronous_lane_memory
        for (int lane = 0; lane < LANES; lane++) begin
            if (lane_mem_rd_en)
                lane_mem_rd_data[lane]
                    <= lane_memory[lane][lane_mem_rd_addr];
            if (lane_mem_wr_en)
                lane_memory[lane][lane_mem_wr_addr]
                    <= lane_mem_wr_data[lane];
        end
    end

    always @(negedge clk_core) begin
        if (automatic_commit_ready && !rst_core)
            commit_ready = ((cycle_count % 7) != 2)
                         && ((cycle_count % 19) != 5)
                         && ((cycle_count % 43) != 11);
    end

    always @(posedge clk_core) begin : independent_monitors
        integer signed expected;
        if (rst_core) begin
            prior_event_accept <= 1'b0;
            prior_mapped_accept <= 1'b0;
            prior_load2_accept <= 1'b0;
            prior_commit_stall <= 1'b0;
            prior_commit_last_accept <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;

            if (service_accept && !service_is_event) begin
                total_loads <= total_loads + 1;
                if (positive_phase)
                    positive_loads <= positive_loads + 1;
            end
            if (weight_rd_en) begin
                total_weight_reads <= total_weight_reads + 1;
                if (positive_phase)
                    positive_weight_reads <= positive_weight_reads + 1;
            end
            if (service_accept && service_is_event) begin
                total_events <= total_events + 1;
                if (positive_phase) begin
                    positive_events <= positive_events + 1;
                    if (service_negate)
                        positive_negated_events <= positive_negated_events + 1;
                    if (prior_load2_accept) begin
                        if (!tail_bypass_available)
                            $fatal(1, "M120 hammer load2 tail bypass missing");
                        positive_tail_hits <= positive_tail_hits + 1;
                    end
                    for (int lane = 0; lane < LANES; lane++) begin
                        expected = weight_value(
                            {service_source, service_block}, lane);
                        if (service_negate)
                            expected = -expected;
                        reference[service_block][service_row_offset][lane]
                            = reference[service_block]
                                       [service_row_offset][lane] + expected;
                    end
                end
                prior_event_block <= service_block;
                prior_event_row <= service_row_offset;
                prior_event_key <= {service_source, service_block};
                prior_event_negate <= service_negate;
            end
            prior_event_accept <= service_accept && service_is_event;
            prior_load2_accept <= service_accept && !service_is_event
                                && service_load_beat == 2;

            if (mapped_update_accept) begin
                total_mapped_updates <= total_mapped_updates + 1;
                if (positive_phase) begin
                    if (!prior_event_accept)
                        $fatal(1, "M120 hammer mapped update without unique prior event");
                    if (dut.mapper_update_block !== prior_event_block
                            || dut.mapper_update_row !== prior_event_row)
                        $fatal(1, "M120 hammer mapped address mismatch");
                    for (int lane = 0; lane < LANES; lane++) begin
                        expected = weight_value(prior_event_key, lane);
                        if (prior_event_negate)
                            expected = -expected;
                        if ($signed(dut.mapper_update_delta[
                                lane * ACC_BITS +: ACC_BITS]) !== expected)
                            $fatal(1, "M120 hammer mapper numeric mismatch lane=%0d got=%0d expected=%0d",
                                   lane,
                                   $signed(dut.mapper_update_delta[
                                       lane * ACC_BITS +: ACC_BITS]),
                                   expected);
                        mapper_lane_checks = mapper_lane_checks + 1;
                    end
                    positive_updates <= positive_updates + 1;
                    if (prior_mapped_accept)
                        positive_ii1_pairs <= positive_ii1_pairs + 1;
                end
                prior_mapped_address <= flat_addr(
                    dut.mapper_update_block, dut.mapper_update_row);
            end
            if (lane_mem_wr_en) begin
                total_writes <= total_writes + 1;
                if (lane_mem_wr_addr !== prior_mapped_address)
                    $fatal(1, "M120 hammer accumulator write lacks prior mapped update");
                if (positive_phase)
                    positive_writes <= positive_writes + 1;
                if (lane_mem_wr_addr == 0)
                    address_zero_writes <= address_zero_writes + 1;
                if (lane_mem_wr_addr == DEPTH-1)
                    address_last_writes <= address_last_writes + 1;
            end
            if (positive_phase && lane_mem_rd_en && lane_mem_wr_en)
                positive_rw_overlap <= positive_rw_overlap + 1;
            prior_mapped_accept <= mapped_update_accept;

            if (prior_commit_stall) begin
                if (!commit_valid || commit_block !== stalled_commit_block
                        || commit_row !== stalled_commit_row
                        || commit_data !== stalled_commit_data
                        || commit_last !== stalled_commit_last)
                    $fatal(1, "M120 hammer commit changed under backpressure");
                if (commit_ready)
                    stall_releases <= stall_releases + 1;
            end
            if (commit_valid && !commit_ready) begin
                stalled_commit_block <= commit_block;
                stalled_commit_row <= commit_row;
                stalled_commit_data <= commit_data;
                stalled_commit_last <= commit_last;
                if (positive_phase)
                    commit_stalls <= commit_stalls + 1;
            end
            prior_commit_stall <= commit_valid && !commit_ready;

            if (positive_phase && commit_valid && commit_ready) begin
                if (commit_block !== expected_commit_block[2:0]
                        || commit_row !== expected_commit_row[8:0])
                    $fatal(1, "M120 hammer commit order mismatch");
                for (int lane = 0; lane < LANES; lane++) begin
                    if ($signed(commit_data[lane * ACC_BITS +: ACC_BITS])
                            !== reference[expected_commit_block]
                                          [expected_commit_row][lane])
                        $fatal(1, "M120 hammer commit numeric mismatch block=%0d row=%0d lane=%0d got=%0d expected=%0d",
                               expected_commit_block, expected_commit_row, lane,
                               $signed(commit_data[
                                   lane * ACC_BITS +: ACC_BITS]),
                               reference[expected_commit_block]
                                        [expected_commit_row][lane]);
                    commit_lane_checks = commit_lane_checks + 1;
                end
                commit_accepts <= commit_accepts + 1;
                if (commit_last !== (expected_commit_block == BLOCKS-1
                                     && expected_commit_row == WIN_ROWS-1))
                    $fatal(1, "M120 hammer commit_last mismatch");
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
            if (window_done) begin
                if (!prior_commit_last_accept)
                    $fatal(1, "M120 hammer window_done without prior last accept");
                completed_windows <= completed_windows + 1;
            end
            prior_commit_last_accept <= commit_valid && commit_ready
                                      && commit_last;

            if (positive_phase && protocol_error)
                $fatal(1, "M120 hammer unexpected positive protocol_error");
        end
    end

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            window_start_valid = 1'b0;
            service_valid = 1'b0;
            window_end_valid = 1'b0;
            commit_ready = 1'b0;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
            repeat (2) @(posedge clk_core);
        end
    endtask

    task automatic clear_reference;
        for (int block = 0; block < BLOCKS; block++)
            for (int row = 0; row < WIN_ROWS; row++)
                for (int lane = 0; lane < LANES; lane++)
                    reference[block][row][lane] = 0;
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

    task automatic drive_load(input integer key, input integer beat);
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
        input integer key,
        input integer row,
        input logic negate,
        input logic last
    );
        begin
            @(negedge clk_core);
            service_valid = 1'b1;
            service_is_event = 1'b1;
            service_source = key[6:3];
            service_block = key[2:0];
            service_load_beat = '0;
            service_row_offset = row[8:0];
            service_negate = negate;
            service_last_for_key = last;
            do @(posedge clk_core); while (!service_accept);
        end
    endtask

    task automatic stop_service_and_drain;
        integer start_cycle;
        begin
            @(negedge clk_core);
            service_valid = 1'b0;
            start_cycle = cycle_count;
            while (mapper_busy || total_mapped_updates != total_events) begin
                @(posedge clk_core);
                if (cycle_count - start_cycle > 2000)
                    $fatal(1, "M120 hammer mapper drain watchdog");
            end
            repeat (2) @(posedge clk_core);
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
        integer start_cycle;
        begin
            start_cycle = cycle_count;
            while (!window_done) begin
                @(posedge clk_core);
                if (cycle_count - start_cycle > 12000)
                    $fatal(1, "M120 hammer commit watchdog");
            end
            @(posedge clk_core);
        end
    endtask

    task automatic check_sticky_fault;
        begin
            repeat (3) @(posedge clk_core);
            if (!protocol_error || window_start_ready || service_ready
                    || window_end_ready)
                $fatal(1, "M120 hammer fault not sticky/quarantined");
        end
    endtask

    initial begin
        integer base_events, base_updates, base_writes;
        integer signed expected_same;
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
        positive_phase = 1'b1;
        automatic_commit_ready = 1'b1;
        cycle_count = 0;
        total_loads = 0;
        total_weight_reads = 0;
        total_events = 0;
        total_mapped_updates = 0;
        total_writes = 0;
        positive_loads = 0;
        positive_weight_reads = 0;
        positive_events = 0;
        positive_updates = 0;
        positive_writes = 0;
        positive_tail_hits = 0;
        positive_negated_events = 0;
        positive_ii1_pairs = 0;
        positive_rw_overlap = 0;
        mapper_lane_checks = 0;
        commit_lane_checks = 0;
        commit_accepts = 0;
        commit_stalls = 0;
        stall_releases = 0;
        completed_windows = 0;
        expected_commit_block = 0;
        expected_commit_row = 0;
        address_zero_writes = 0;
        address_last_writes = 0;
        malformed_beat_attacks = 0;
        malformed_key_attacks = 0;
        early_end_attacks = 0;
        older_update_drain_checks = 0;
        same_address_events_accepted = 0;
        same_address_updates_written = 0;
        duplicate_events_accepted = 0;
        duplicate_updates_written = 0;
        reset_events_accepted = 0;
        reset_updates_written = 0;
        prior_event_accept = 1'b0;
        prior_mapped_accept = 1'b0;
        prior_load2_accept = 1'b0;
        prior_commit_stall = 1'b0;
        prior_commit_last_accept = 1'b0;
        prior_mapped_address = '0;
        for (int lane = 0; lane < LANES; lane++) begin
            lane_mem_rd_data[lane] = 'x;
            for (int address = 0; address < DEPTH; address++)
                lane_memory[lane][address] = 'x;
        end

        // Positive closure campaign: 1024 accepted events, updates and writes.
        reset_dut();
        for (int window_index = 0; window_index < 2; window_index++) begin
            start_window();
            if (window_index == 1) begin
                if ($signed(lane_memory[0][0]) !== -128
                        || $signed(lane_memory[0][DEPTH-1]) !== -128)
                    $fatal(1, "M120 hammer lazy clear swept physical data");
            end
            for (int key = 0; key < 128; key++) begin
                drive_load(key, 0);
                drive_load(key, 1);
                drive_load(key, 2);
                if (window_index == 0 && key == 0
                        && (positive_updates != 0 || positive_writes != 0))
                    $fatal(1, "M120 hammer load token caused update/write");
                for (int event_index = 0; event_index < 4; event_index++)
                    drive_event(key, positive_event_row(key, event_index),
                                ((window_index + key + event_index) & 1) != 0,
                                event_index == 3);
            end
            stop_service_and_drain();
            end_window();
            wait_window_done();
        end
        repeat (3) @(posedge clk_core);
        if (positive_loads != 768 || positive_weight_reads != 768
                || positive_events != 1024 || positive_updates != 1024
                || positive_writes != 1024 || positive_tail_hits != 256
                || positive_negated_events != 512
                || positive_ii1_pairs != 768
                || positive_rw_overlap != 768
                || mapper_lane_checks != 98304
                || commit_accepts != 6144
                || commit_lane_checks != 589824
                || completed_windows != 2 || commit_stalls < 100
                || stall_releases == 0 || address_zero_writes == 0
                || address_last_writes == 0)
            $fatal(1, "M120 hammer positive conservation mismatch loads=%0d reads=%0d events=%0d updates=%0d writes=%0d tail=%0d neg=%0d ii1=%0d rw=%0d maplanes=%0d commits=%0d commitlanes=%0d windows=%0d stalls=%0d releases=%0d addr=%0d/%0d",
                   positive_loads, positive_weight_reads, positive_events,
                   positive_updates, positive_writes, positive_tail_hits,
                   positive_negated_events, positive_ii1_pairs,
                   positive_rw_overlap, mapper_lane_checks,
                   commit_accepts, commit_lane_checks, completed_windows,
                   commit_stalls, stall_releases,
                   address_zero_writes, address_last_writes);

        positive_phase = 1'b0;
        automatic_commit_ready = 1'b0;

        // Malformed load beat: beat2 cannot replace expected beat1.
        reset_dut();
        start_window();
        drive_load(3, 0);
        @(negedge clk_core);
        service_valid = 1'b1;
        service_is_event = 1'b0;
        service_source = 0;
        service_block = 3;
        service_load_beat = 2;
        @(posedge clk_core);
        if (!protocol_error || service_accept || weight_rd_en)
            $fatal(1, "M120 hammer malformed load beat not rejected");
        malformed_beat_attacks = malformed_beat_attacks + 1;
        @(negedge clk_core);
        service_valid = 1'b0;
        check_sticky_fault();

        // Malformed key: beat1 must match the key accepted by beat0.
        reset_dut();
        start_window();
        drive_load(4, 0);
        @(negedge clk_core);
        service_valid = 1'b1;
        service_is_event = 1'b0;
        service_source = 0;
        service_block = 5;
        service_load_beat = 1;
        @(posedge clk_core);
        if (!protocol_error || service_accept || weight_rd_en)
            $fatal(1, "M120 hammer malformed load key not rejected");
        malformed_key_attacks = malformed_key_attacks + 1;
        @(negedge clk_core);
        service_valid = 1'b0;
        check_sticky_fault();

        // Premature end must fail, but the event accepted one cycle earlier
        // must still be accepted by M118 and written.
        reset_dut();
        start_window();
        drive_load(6, 0);
        drive_load(6, 1);
        drive_load(6, 2);
        base_updates = total_mapped_updates;
        base_writes = total_writes;
        drive_event(6, 99, 1'b0, 1'b0);
        @(negedge clk_core);
        service_valid = 1'b0;
        window_end_valid = 1'b1;
        @(posedge clk_core);
        if (!protocol_error || window_end_accept || !mapped_update_accept)
            $fatal(1, "M120 hammer premature end/drain behavior mismatch");
        early_end_attacks = early_end_attacks + 1;
        @(negedge clk_core);
        window_end_valid = 1'b0;
        @(posedge clk_core);
        if (!lane_mem_wr_en)
            $fatal(1, "M120 hammer older mapped update did not drain write");
        older_update_drain_checks = older_update_drain_checks + 1;
        repeat (2) @(posedge clk_core);
        if (total_mapped_updates - base_updates != 1
                || total_writes - base_writes != 1)
            $fatal(1, "M120 hammer older update conservation mismatch");

        // Replay the exact M120 legal-shaped consecutive same-address event
        // counterexample with the review-only M123 substitution. Both service
        // accepts must map and write, with the exact doubled lane sum.
        reset_dut();
        start_window();
        drive_load(8, 0);
        drive_load(8, 1);
        drive_load(8, 2);
        base_events = total_events;
        base_updates = total_mapped_updates;
        base_writes = total_writes;
        drive_event(8, 44, 1'b0, 1'b0);
        drive_event(8, 44, 1'b0, 1'b1);
        @(negedge clk_core);
        service_valid = 1'b0;
        repeat (4) @(posedge clk_core);
        same_address_events_accepted = total_events - base_events;
        same_address_updates_written = total_writes - base_writes;
        if (protocol_error || same_address_events_accepted != 2
                || total_mapped_updates - base_updates != 2
                || same_address_updates_written != 2)
            $fatal(1, "M123 integrated M120 same-address closure mismatch events=%0d updates=%0d writes=%0d fault=%0d",
                   same_address_events_accepted,
                   total_mapped_updates - base_updates,
                   same_address_updates_written, protocol_error);
        for (int lane = 0; lane < LANES; lane++) begin
            expected_same = 2 * weight_value(8, lane);
            if ($signed(lane_memory[lane][flat_addr(0, 44)])
                    !== expected_same)
                $fatal(1, "M123 integrated M120 same-address numeric mismatch lane=%0d got=%0d expected=%0d",
                       lane,
                       $signed(lane_memory[lane][flat_addr(0, 44)]),
                       expected_same);
        end

        // A retry separated by another address is not detected and is applied
        // twice, demonstrating that M120 has no transaction identity/dedup.
        reset_dut();
        start_window();
        drive_load(9, 0);
        drive_load(9, 1);
        drive_load(9, 2);
        base_events = total_events;
        base_updates = total_mapped_updates;
        base_writes = total_writes;
        drive_event(9, 50, 1'b0, 1'b0);
        drive_event(9, 51, 1'b0, 1'b0);
        drive_event(9, 50, 1'b0, 1'b1);
        @(negedge clk_core);
        service_valid = 1'b0;
        while (mapper_busy)
            @(posedge clk_core);
        repeat (3) @(posedge clk_core);
        duplicate_events_accepted = total_events - base_events;
        duplicate_updates_written = total_writes - base_writes;
        if (protocol_error || duplicate_events_accepted != 3
                || total_mapped_updates - base_updates != 3
                || duplicate_updates_written != 3
                || $signed(lane_memory[0][flat_addr(1, 50)]) !== -256)
            $fatal(1, "M120 hammer retry/dedup finding mismatch events=%0d updates=%0d writes=%0d value=%0d fault=%0d",
                   duplicate_events_accepted,
                   total_mapped_updates - base_updates,
                   duplicate_updates_written,
                   $signed(lane_memory[0][flat_addr(1, 50)]),
                   protocol_error);

        // Reset after event acceptance flushes the pending mapped update.
        reset_dut();
        start_window();
        drive_load(10, 0);
        drive_load(10, 1);
        drive_load(10, 2);
        base_events = total_events;
        base_updates = total_mapped_updates;
        base_writes = total_writes;
        drive_event(10, 77, 1'b0, 1'b1);
        @(negedge clk_core);
        service_valid = 1'b0;
        rst_core = 1'b1;
        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        repeat (3) @(posedge clk_core);
        reset_events_accepted = total_events - base_events;
        reset_updates_written = total_writes - base_writes;
        if (reset_events_accepted != 1
                || total_mapped_updates - base_updates != 0
                || reset_updates_written != 0)
            $fatal(1, "M120 hammer reset-loss finding mismatch events=%0d updates=%0d writes=%0d",
                   reset_events_accepted,
                   total_mapped_updates - base_updates,
                   reset_updates_written);

        if (malformed_beat_attacks != 1 || malformed_key_attacks != 1
                || early_end_attacks != 1
                || older_update_drain_checks != 1
                || same_address_events_accepted != 2
                || same_address_updates_written != 2
                || duplicate_events_accepted != 3
                || duplicate_updates_written != 3
                || reset_events_accepted != 1
                || reset_updates_written != 0)
            $fatal(1, "M120 hammer attack counter mismatch");

        $display("PASS M123 integrated M120 counterexample commercial_vcs=true review_only_substitution=true positive_loads=768 positive_weight_reads=768 positive_events=1024 positive_updates=1024 positive_writes=1024 positive_ii1_pairs=768 positive_rw_overlap=768 mapper_lane_checks=98304 tail_bypass_hits=256 negate_events=512 commits=6144 commit_lane_checks=589824 commit_stalls=%0d stall_releases=%0d lazy_clear_windows=2 address_minmax=true int8_endpoints=true malformed_beat_attacks=1 malformed_key_attacks=1 early_end_attacks=1 older_update_drain_checks=1 same_address_events_accepted=2 same_address_mapped_updates=2 same_address_updates_written=2 same_address_lane_checks=96 same_address_accept_then_loss_closed=true retry_events_accepted=3 retry_updates_written=3 retry_dedup_absent=true reset_events_accepted=1 reset_updates_written=0 reset_exact_once_undefined=true accumulator_bytes=700416 combined_bytes=725416 directed_legal_and_same_address_exact_once=true heldout_duplicate_retry_reset_exact_once=false foundry_sram_macro=false scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false",
                 commit_stalls, stall_releases);
        $finish;
    end
endmodule

`default_nettype wire
