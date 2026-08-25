`timescale 1ns/1ps
`default_nettype none

module tb_m119_pwp_weight_tail_bypass_mapper;
    localparam int LANES = 96;
    localparam int ACC_BITS = 19;
    localparam int DELTA_BITS = LANES * ACC_BITS;

    logic clk_core, rst_core;
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
    logic update_valid, update_ready;
    logic [2:0] update_block;
    logic [8:0] update_row;
    logic [DELTA_BITS-1:0] update_delta;
    logic update_accept;
    logic payload_active, tail_bypass_available, protocol_error, busy;

    typedef struct packed {
        logic [6:0] key;
        logic [2:0] block_id;
        logic [8:0] row_id;
        logic negate;
    } expected_update_t;
    expected_update_t expected_q[$];
    expected_update_t expected_item;

    int cycle_count;
    int accepted_loads;
    int accepted_events;
    int accepted_updates;
    int weight_reads;
    int lane_checks;
    int tail_bypass_hits;
    int event_ii1_pairs;
    int update_stall_cycles;
    int negate_events;
    int protocol_attacks;
    bit previous_event_accept;
    bit previous_load2_accept;
    bit positive_phase;

    m119_pwp_weight_tail_bypass_mapper dut (
        .clk_core(clk_core),
        .rst_core(rst_core),
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
        .update_valid(update_valid),
        .update_ready(update_ready),
        .update_block(update_block),
        .update_row(update_row),
        .update_delta(update_delta),
        .update_accept(update_accept),
        .payload_active(payload_active),
        .tail_bypass_available(tail_bypass_available),
        .protocol_error(protocol_error),
        .busy(busy)
    );

    m119_pwp_weight_tail_bypass_mapper_assertions checks (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .service_valid(service_valid),
        .service_ready(service_ready),
        .service_is_event(service_is_event),
        .service_load_beat(service_load_beat),
        .service_accept(service_accept),
        .weight_rd_en(weight_rd_en),
        .weight_rd_key(weight_rd_key),
        .weight_rd_beat(weight_rd_beat),
        .update_valid(update_valid),
        .update_ready(update_ready),
        .update_block(update_block),
        .update_row(update_row),
        .update_delta(update_delta),
        .update_accept(update_accept),
        .payload_active(payload_active),
        .tail_bypass_available(tail_bypass_available),
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

    // Fixed one-cycle synchronous 256-bit weight response.  Each beat maps 32
    // consecutive signed INT8 output lanes.
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

    always @(posedge clk_core) begin : scoreboard
        integer signed expected_value;
        if (rst_core) begin
            previous_event_accept <= 1'b0;
            previous_load2_accept <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (service_accept && !service_is_event)
                accepted_loads <= accepted_loads + 1;
            if (weight_rd_en)
                weight_reads <= weight_reads + 1;
            if (service_accept && service_is_event) begin
                expected_item.key = {service_source, service_block};
                expected_item.block_id = service_block;
                expected_item.row_id = service_row_offset;
                expected_item.negate = service_negate;
                expected_q.push_back(expected_item);
                accepted_events <= accepted_events + 1;
                if (service_negate)
                    negate_events <= negate_events + 1;
                if (previous_event_accept)
                    event_ii1_pairs <= event_ii1_pairs + 1;
                if (previous_load2_accept) begin
                    if (!tail_bypass_available)
                        $fatal(1, "M119 first event lacked tail bypass key=%0d",
                               {service_source, service_block});
                    tail_bypass_hits <= tail_bypass_hits + 1;
                end
            end
            previous_event_accept <= service_accept && service_is_event;
            previous_load2_accept <= service_accept && !service_is_event
                                  && service_load_beat == 2;

            if (update_valid && !update_ready)
                update_stall_cycles <= update_stall_cycles + 1;
            if (update_accept) begin
                if (expected_q.size() == 0)
                    $fatal(1, "M119 update without accepted event");
                expected_item = expected_q.pop_front();
                if (update_block !== expected_item.block_id
                        || update_row !== expected_item.row_id)
                    $fatal(1, "M119 update identity mismatch");
                for (int lane = 0; lane < LANES; lane++) begin
                    expected_value = weight_value(expected_item.key, lane);
                    if (expected_item.negate)
                        expected_value = -expected_value;
                    if ($signed(update_delta[lane * ACC_BITS +: ACC_BITS])
                            !== expected_value)
                        $fatal(1, "M119 numeric mismatch key=%0d row=%0d lane=%0d got=%0d expected=%0d",
                               expected_item.key, expected_item.row_id, lane,
                               $signed(update_delta[
                                   lane * ACC_BITS +: ACC_BITS]),
                               expected_value);
                end
                lane_checks <= lane_checks + LANES;
                accepted_updates <= accepted_updates + 1;
            end
            if (positive_phase && protocol_error)
                $fatal(1, "M119 unexpected protocol error cycle=%0d",
                       cycle_count);
        end
    end

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            service_valid = 1'b0;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
            repeat (2) @(posedge clk_core);
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
        input int key,
        input int row,
        input bit negate,
        input bit last_for_key
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
            service_last_for_key = last_for_key;
            do @(posedge clk_core); while (!service_accept);
        end
    endtask

    task automatic stop_service;
        begin
            @(negedge clk_core);
            service_valid = 1'b0;
        end
    endtask

    task automatic wait_updates_drained;
        int start_cycle;
        begin
            start_cycle = cycle_count;
            while (expected_q.size() != 0 || update_valid) begin
                @(posedge clk_core);
                if (cycle_count - start_cycle > 1000)
                    $fatal(1, "M119 drain watchdog q=%0d valid=%0d",
                           expected_q.size(), update_valid);
            end
            @(posedge clk_core);
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        service_valid = 1'b0;
        service_is_event = 1'b0;
        service_source = '0;
        service_block = '0;
        service_load_beat = '0;
        service_row_offset = '0;
        service_negate = 1'b0;
        service_last_for_key = 1'b0;
        weight_rd_data = 'x;
        update_ready = 1'b1;
        cycle_count = 0;
        accepted_loads = 0;
        accepted_events = 0;
        accepted_updates = 0;
        weight_reads = 0;
        lane_checks = 0;
        tail_bypass_hits = 0;
        event_ii1_pairs = 0;
        update_stall_cycles = 0;
        negate_events = 0;
        protocol_attacks = 0;
        previous_event_accept = 1'b0;
        previous_load2_accept = 1'b0;
        positive_phase = 1'b1;

        reset_dut();
        for (int key = 0; key < 128; key++) begin
            drive_load(key, 0);
            drive_load(key, 1);
            drive_load(key, 2);
            for (int event_index = 0; event_index < 4; event_index++)
                drive_event(key, event_index,
                    ((key + event_index) & 1) != 0,
                    event_index == 3);
        end
        stop_service();
        wait_updates_drained();

        // Capture one tail-bypassed event while the output is empty, then
        // hold the accumulator side for exactly three full cycles.
        drive_load(7, 0);
        drive_load(7, 1);
        drive_load(7, 2);
        update_ready = 1'b0;
        drive_event(7, 77, 1'b1, 1'b1);
        stop_service();
        repeat (3) @(posedge clk_core);
        if (!update_valid)
            $fatal(1, "M119 stalled update disappeared");
        @(negedge clk_core);
        update_ready = 1'b1;
        wait_updates_drained();

        if (accepted_loads != 387 || weight_reads != 387
                || accepted_events != 513 || accepted_updates != 513
                || lane_checks != 49248 || tail_bypass_hits != 129
                || event_ii1_pairs != 384 || update_stall_cycles != 3
                || negate_events != 257)
            $fatal(1, "M119 conservation mismatch loads=%0d reads=%0d events=%0d updates=%0d lanes=%0d tail=%0d ii1=%0d stalls=%0d negate=%0d",
                   accepted_loads, weight_reads, accepted_events,
                   accepted_updates, lane_checks, tail_bypass_hits,
                   event_ii1_pairs, update_stall_cycles, negate_events);

        // Wrong first beat is rejected immediately and becomes sticky.  It
        // cannot create a weight read or an update.
        positive_phase = 1'b0;
        reset_dut();
        @(negedge clk_core);
        service_valid = 1'b1;
        service_is_event = 1'b0;
        service_source = 0;
        service_block = 0;
        service_load_beat = 1;
        @(posedge clk_core);
        if (!protocol_error || service_ready || service_accept || weight_rd_en)
            $fatal(1, "M119 malformed beat did not fail closed");
        protocol_attacks = protocol_attacks + 1;
        @(negedge clk_core);
        service_valid = 1'b0;
        repeat (2) @(posedge clk_core);
        if (!protocol_error)
            $fatal(1, "M119 protocol fault not sticky");

        $display("PASS M119 PWP weight tail-bypass mapper VCS groups=129 weight_loads=387 weight_reads=387 events=513 updates=513 lane_checks=49248 tail_bypass_first_events=129 event_ii1_pairs=384 update_stalls=3 negate_events=257 protocol_attacks=1 weight_port_bits=256 weight_beats=3 weight_payload_bits=768 lanes=96 acc_bits=19 delta_bits=1824 fixed_read_latency=1 tail_bypass=true exact_once_directed=true accumulator_integrated=false foundry_sram_macro=false scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false");
        $finish;
    end
endmodule

`default_nettype wire
