`timescale 1ns/1ps
`default_nettype none

module tb_m131_independent_hammer;
    localparam int SOURCES = 16;
    localparam int LANES = 96;
    localparam int ACC_BITS = 19;
    localparam int UPDATE_BITS = LANES * ACC_BITS;

    logic clk_core, rst_core;
    logic weight_fill_valid, weight_fill_ready, weight_fill_accept;
    logic [2:0] weight_fill_block;
    logic [3:0] weight_fill_source;
    logic [1:0] weight_fill_beat;
    logic [255:0] weight_fill_data;
    logic group_valid, group_ready, group_accept;
    logic [2:0] group_block;
    logic [8:0] group_row;
    logic [1:0] group_source_count_m1;
    logic [3:0] group_source [0:3];
    logic [3:0] group_negate;
    logic group_last;
    logic update_valid, update_ready, update_accept;
    logic [2:0] update_block;
    logic [8:0] update_row;
    logic [UPDATE_BITS-1:0] update_delta;
    logic [15:0] update_selected_mask;
    logic update_last;
    logic done_valid;
    logic [2:0] done_block;
    logic [8:0] done_row;
    logic [15:0] observed_cache_valid;
    logic observed_resident_block_valid;
    logic [2:0] observed_resident_block;
    logic observed_pair_pipeline_valid;
    logic observed_row_stream_open;
    logic protocol_error, busy;

    typedef struct packed {
        logic [2:0] block_id;
        logic [8:0] row_id;
        logic [1:0] count_m1;
        logic [15:0] source_ids;
        logic [3:0] negate;
        logic last;
    } descriptor35_t;

    typedef struct packed {
        logic [2:0] block_id;
        logic [8:0] row_id;
        logic [15:0] selected_mask;
        logic last;
        logic [UPDATE_BITS-1:0] delta;
    } expected_update_t;

    expected_update_t expected_q[$];
    int cycle_count, group_accepts, update_accepts, source_checks, lane_checks;
    int k1, k2, k3, k4, plus512, minus512;
    int cross_group_intervals, cross_update_intervals;
    int last_cross_group_cycle, last_cross_update_cycle;
    int done_checks, done_overlap_checks, done_tag_checks;
    int output_stall_cycles, group_stall_cycles, max_output_stall;
    int current_output_stall, long_stall_replace;
    int idle_payload_ready_checks, open_row_idle_payload_ready_checks;
    int within_duplicate_attacks, within_descending_attacks;
    int cross_repeat_attacks, cross_backtrack_attacks;
    int row_identity_attacks, dirty_source_attacks, dirty_negate_attacks;
    int nonlast_source15_attacks, cache_miss_attacks, block_attacks;
    int reset_checks, reset_aborted_descriptors;
    int gapped_partition_descriptors_accepted;
    bit cross_phase;

    m131_synthesis_safe_compact_canonical_k4_row_fold dut (.*);

    m131_synthesis_safe_compact_canonical_k4_row_fold_assertions sva (
        .clk_core, .rst_core, .weight_fill_accept, .weight_fill_ready,
        .group_valid, .group_ready, .group_accept, .group_block, .group_row,
        .group_source_count_m1, .group_source, .group_negate, .group_last,
        .update_valid, .update_ready, .update_accept, .update_block,
        .update_row, .update_delta, .update_selected_mask, .update_last,
        .done_valid, .done_block, .done_row, .protocol_error
    );

    initial clk_core = 1'b0;
    always #1 clk_core = ~clk_core;

    function automatic integer signed model_weight(
        input int source,
        input int lane
    );
        int raw;
        begin
            if (lane == 0 && source < 4)
                model_weight = -128;
            else if (lane == 1 && source < 4)
                model_weight = 127;
            else begin
                raw = (source * 43 + lane * 29 + source * lane * 7) & 8'hff;
                model_weight = raw - 128;
            end
        end
    endfunction

    function automatic descriptor35_t make_descriptor(
        input int block_id,
        input int row_id,
        input int count,
        input int id0, input int id1, input int id2, input int id3,
        input logic [3:0] negate,
        input bit last
    );
        descriptor35_t result;
        int ids [0:3];
        begin
            if (count < 1 || count > 4)
                $fatal(1, "M131 review descriptor count out of range");
            ids[0] = id0; ids[1] = id1; ids[2] = id2; ids[3] = id3;
            result = '0;
            result.block_id = block_id[2:0];
            result.row_id = row_id[8:0];
            result.count_m1 = count - 1;
            result.negate = negate;
            result.last = last;
            for (int pick = 0; pick < 4; pick++)
                result.source_ids[pick * 4 +: 4] = ids[pick][3:0];
            return result;
        end
    endfunction

    function automatic logic [15:0] descriptor_mask(input descriptor35_t desc);
        logic [15:0] result;
        begin
            result = '0;
            for (int pick = 0; pick < 4; pick++)
                if (pick <= desc.count_m1)
                    result[desc.source_ids[pick * 4 +: 4]] = 1'b1;
            return result;
        end
    endfunction

    function automatic logic [UPDATE_BITS-1:0]
        descriptor_delta(input descriptor35_t desc);
        logic [UPDATE_BITS-1:0] result;
        integer signed sum;
        int source;
        begin
            result = '0;
            for (int lane = 0; lane < LANES; lane++) begin
                sum = 0;
                for (int pick = 0; pick < 4; pick++) begin
                    if (pick <= desc.count_m1) begin
                        source = desc.source_ids[pick * 4 +: 4];
                        sum += desc.negate[pick]
                             ? -model_weight(source, lane)
                             : model_weight(source, lane);
                    end
                end
                result[lane * ACC_BITS +: ACC_BITS] = sum[ACC_BITS-1:0];
            end
            return result;
        end
    endfunction

    task automatic clear_inputs;
        begin
            weight_fill_valid = 0;
            weight_fill_block = 0;
            weight_fill_source = 0;
            weight_fill_beat = 0;
            weight_fill_data = 0;
            group_valid = 0;
            group_block = 0;
            group_row = 0;
            group_source_count_m1 = 0;
            group_negate = 0;
            group_last = 0;
            for (int pick = 0; pick < 4; pick++)
                group_source[pick] = 0;
        end
    endtask

    task automatic clear_group;
        begin
            group_valid = 0;
            group_block = 0;
            group_row = 0;
            group_source_count_m1 = 0;
            group_negate = 0;
            group_last = 0;
            for (int pick = 0; pick < 4; pick++)
                group_source[pick] = 0;
        end
    endtask

    task automatic drive_descriptor_ports(input descriptor35_t desc);
        begin
            group_block = desc.block_id;
            group_row = desc.row_id;
            group_source_count_m1 = desc.count_m1;
            group_negate = desc.negate;
            group_last = desc.last;
            for (int pick = 0; pick < 4; pick++)
                group_source[pick] = desc.source_ids[pick * 4 +: 4];
        end
    endtask

    task automatic apply_reset;
        begin
            @(negedge clk_core);
            rst_core = 1;
            update_ready = 0;
            cross_phase = 0;
            clear_inputs();
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 0;
            update_ready = 1;
            repeat (2) @(posedge clk_core);
            if (protocol_error || busy || observed_cache_valid != 0
                    || observed_resident_block_valid
                    || observed_pair_pipeline_valid
                    || observed_row_stream_open)
                $fatal(1, "M131 reset did not clear architectural state");
        end
    endtask

    task automatic fill_source(input int source);
        logic [255:0] payload;
        int value;
        begin
            for (int beat = 0; beat < 3; beat++) begin
                payload = 0;
                for (int item = 0; item < 32; item++) begin
                    value = model_weight(source, beat * 32 + item);
                    payload[item * 8 +: 8] = value[7:0];
                end
                @(negedge clk_core);
                weight_fill_valid = 1;
                weight_fill_block = 3;
                weight_fill_source = source[3:0];
                weight_fill_beat = beat[1:0];
                weight_fill_data = payload;
                do @(posedge clk_core); while (!weight_fill_accept);
                if (protocol_error)
                    $fatal(1, "M131 unexpected fill protocol error");
            end
            @(negedge clk_core);
            weight_fill_valid = 0;
        end
    endtask

    task automatic fill_all;
        begin
            for (int source = 0; source < SOURCES; source++)
                fill_source(source);
            if (observed_cache_valid != 16'hffff
                    || !observed_resident_block_valid
                    || observed_resident_block != 3)
                $fatal(1, "M131 full cache identity mismatch");
        end
    endtask

    task automatic send_descriptor(input descriptor35_t desc);
        begin
            @(negedge clk_core);
            group_valid = 1;
            drive_descriptor_ports(desc);
            do @(posedge clk_core); while (!group_accept);
            if (protocol_error)
                $fatal(1, "M131 legal descriptor raised protocol error");
        end
    endtask

    task automatic stop_and_drain;
        int watchdog;
        begin
            @(negedge clk_core);
            clear_group();
            watchdog = 0;
            while (expected_q.size() != 0 || update_valid) begin
                @(posedge clk_core);
                watchdog++;
                if (watchdog > 1000)
                    $fatal(1, "M131 review drain timeout");
            end
            repeat (2) @(posedge clk_core);
        end
    endtask

    task automatic expect_fault(input descriptor35_t desc, input int kind);
        begin
            @(negedge clk_core);
            group_valid = 1;
            drive_descriptor_ports(desc);
            #1ps;
            if (!protocol_error || group_ready || group_accept || update_valid
                    || done_valid)
                $fatal(1, "M131 descriptor attack kind=%0d not fail closed", kind);
            @(posedge clk_core);
            @(negedge clk_core);
            clear_group();
            #1ps;
            if (!protocol_error)
                $fatal(1, "M131 descriptor attack kind=%0d not sticky", kind);
            case (kind)
                0: within_duplicate_attacks++;
                1: within_descending_attacks++;
                2: cross_repeat_attacks++;
                3: cross_backtrack_attacks++;
                4: row_identity_attacks++;
                5: dirty_source_attacks++;
                6: dirty_negate_attacks++;
                7: nonlast_source15_attacks++;
                8: cache_miss_attacks++;
                default: block_attacks++;
            endcase
        end
    endtask

    always @(posedge clk_core) begin : scoreboard
        expected_update_t accepted;
        descriptor35_t live;
        cycle_count++;
        if (rst_core) begin
            expected_q.delete();
            current_output_stall = 0;
        end else begin
            if (update_valid) begin
                if (expected_q.size() == 0)
                    $fatal(1, "M131 visible update with empty scoreboard");
                if ({update_block, update_row, update_selected_mask, update_last}
                        !== {expected_q[0].block_id, expected_q[0].row_id,
                             expected_q[0].selected_mask, expected_q[0].last})
                    $fatal(1, "M131 update identity mismatch cycle=%0d", cycle_count);
                if (update_delta !== expected_q[0].delta)
                    $fatal(1, "M131 update numeric mismatch cycle=%0d", cycle_count);
                if (!update_ready) begin
                    output_stall_cycles++;
                    current_output_stall++;
                    if (current_output_stall > max_output_stall)
                        max_output_stall = current_output_stall;
                end else begin
                    current_output_stall = 0;
                end
            end else begin
                current_output_stall = 0;
            end

            if (done_valid) begin
                done_checks++;
                if (!update_accept || !update_last
                        || {done_block, done_row} !== {update_block, update_row})
                    $fatal(1, "M131 tagged done mismatch cycle=%0d", cycle_count);
                done_tag_checks++;
                if (group_accept
                        && {done_block, done_row} != {group_block, group_row}) begin
                    done_overlap_checks++;
                    if ({done_block, done_row}
                            !== {expected_q[0].block_id, expected_q[0].row_id})
                        $fatal(1, "M131 overlapped done lost prior row tag");
                end
            end

            if (update_accept) begin
                if (expected_q.size() == 0)
                    $fatal(1, "M131 accepted update with empty scoreboard");
                if ($signed(update_delta[0 +: ACC_BITS]) == 512)
                    plus512++;
                if ($signed(update_delta[0 +: ACC_BITS]) == -512)
                    minus512++;
                expected_q.pop_front();
                update_accepts++;
                lane_checks += LANES;
                if (cross_phase) begin
                    if (last_cross_update_cycle != 0) begin
                        if (cycle_count - last_cross_update_cycle != 1)
                            $fatal(1, "M131 cross-row update II drift");
                        cross_update_intervals++;
                    end
                    last_cross_update_cycle = cycle_count;
                end
            end

            if (group_accept) begin
                live.block_id = group_block;
                live.row_id = group_row;
                live.count_m1 = group_source_count_m1;
                live.negate = group_negate;
                live.last = group_last;
                for (int pick = 0; pick < 4; pick++)
                    live.source_ids[pick * 4 +: 4] = group_source[pick];
                accepted.block_id = live.block_id;
                accepted.row_id = live.row_id;
                accepted.selected_mask = descriptor_mask(live);
                accepted.last = live.last;
                accepted.delta = descriptor_delta(live);
                expected_q.push_back(accepted);
                group_accepts++;
                source_checks += live.count_m1 + 1;
                case (live.count_m1)
                    0: k1++;
                    1: k2++;
                    2: k3++;
                    default: k4++;
                endcase
                if (cross_phase) begin
                    if (last_cross_group_cycle != 0) begin
                        if (cycle_count - last_cross_group_cycle != 1)
                            $fatal(1, "M131 cross-row descriptor II drift");
                        cross_group_intervals++;
                    end
                    last_cross_group_cycle = cycle_count;
                end
            end
        end
    end

    initial begin : attack_sequence
        descriptor35_t desc;
        descriptor35_t held;
        int watchdog;

        if ($bits(descriptor35_t) != 35)
            $fatal(1, "M131 descriptor packed width is not 35 bits");
        rst_core = 1;
        update_ready = 0;
        cross_phase = 0;
        cycle_count = 0;
        group_accepts = 0; update_accepts = 0;
        source_checks = 0; lane_checks = 0;
        k1 = 0; k2 = 0; k3 = 0; k4 = 0;
        plus512 = 0; minus512 = 0;
        cross_group_intervals = 0; cross_update_intervals = 0;
        last_cross_group_cycle = 0; last_cross_update_cycle = 0;
        done_checks = 0; done_overlap_checks = 0; done_tag_checks = 0;
        output_stall_cycles = 0; group_stall_cycles = 0;
        max_output_stall = 0; current_output_stall = 0;
        long_stall_replace = 0;
        idle_payload_ready_checks = 0;
        open_row_idle_payload_ready_checks = 0;
        within_duplicate_attacks = 0; within_descending_attacks = 0;
        cross_repeat_attacks = 0; cross_backtrack_attacks = 0;
        row_identity_attacks = 0; dirty_source_attacks = 0;
        dirty_negate_attacks = 0; nonlast_source15_attacks = 0;
        cache_miss_attacks = 0; block_attacks = 0;
        reset_checks = 0; reset_aborted_descriptors = 0;
        gapped_partition_descriptors_accepted = 0;
        clear_inputs();

        apply_reset();
        fill_all();

        // When valid is low, ready must remain capacity-only while every
        // semantic field is changed to values that would be illegal if valid.
        @(negedge clk_core);
        group_valid = 0;
        for (int probe = 0; probe < 16; probe++) begin
            group_block = probe[2:0];
            group_row = (9'h1ff - probe);
            group_source_count_m1 = probe[1:0];
            group_source[0] = 15 - probe[3:0];
            group_source[1] = probe[3:0];
            group_source[2] = probe[3:0];
            group_source[3] = ~probe[3:0];
            group_negate = probe[3:0];
            group_last = probe[0];
            #1ps;
            if (!group_ready || protocol_error || group_accept)
                $fatal(1, "M131 idle payload influenced ready probe=%0d", probe);
            idle_payload_ready_checks++;
        end
        clear_group();

        // K1-K4 and both signed four-source boundaries.
        send_descriptor(make_descriptor(3, 10, 1, 15, 0, 0, 0, 0, 1));
        send_descriptor(make_descriptor(3, 11, 2, 1, 9, 0, 0, 2'b01, 1));
        send_descriptor(make_descriptor(3, 12, 3, 2, 7, 14, 0, 3'b101, 1));
        send_descriptor(make_descriptor(3, 13, 4, 0, 1, 2, 3, 4'b0000, 1));
        send_descriptor(make_descriptor(3, 14, 4, 0, 1, 2, 3, 4'b1111, 1));
        stop_and_drain();

        // Strictly increasing same-row descriptors are accepted. The gaps
        // prove stream-local monotonicity is not complete partition losslessness.
        send_descriptor(make_descriptor(3, 20, 2, 0, 2, 0, 0, 0, 0));
        gapped_partition_descriptors_accepted++;
        stop_and_drain();
        if (!observed_row_stream_open)
            $fatal(1, "M131 row stream did not remain open");
        @(negedge clk_core);
        group_valid = 0;
        group_block = 7;
        group_row = 9'h1ff;
        group_source_count_m1 = 3;
        group_source[0] = 15;
        group_source[1] = 1;
        group_source[2] = 1;
        group_source[3] = 0;
        group_negate = 4'hf;
        group_last = 0;
        #1ps;
        if (!group_ready || protocol_error)
            $fatal(1, "M131 open-row idle payload influenced ready");
        open_row_idle_payload_ready_checks++;
        send_descriptor(make_descriptor(3, 20, 3, 4, 7, 9, 0, 3'b010, 0));
        gapped_partition_descriptors_accepted++;
        send_descriptor(make_descriptor(3, 20, 2, 12, 15, 0, 0, 2'b01, 1));
        gapped_partition_descriptors_accepted++;
        stop_and_drain();
        if (observed_row_stream_open)
            $fatal(1, "M131 last descriptor did not close row stream");

        // 96 independent rows: descriptor accepts, updates, and tagged done
        // all overlap at II1 after pipeline fill, with prior-row done tags.
        cross_phase = 1;
        last_cross_group_cycle = 0;
        last_cross_update_cycle = 0;
        for (int row = 0; row < 96; row++)
            send_descriptor(make_descriptor(3, 100 + row, 4,
                                            0, 1, 2, 3,
                                            row[3:0], 1));
        stop_and_drain();
        cross_phase = 0;

        // Hold one output and the next valid descriptor for 73 cycles, then
        // require same-cycle output retirement and input replacement.
        @(negedge clk_core);
        update_ready = 0;
        group_valid = 1;
        desc = make_descriptor(3, 210, 4, 4, 5, 6, 7, 4'b0101, 1);
        drive_descriptor_ports(desc);
        update_ready = 1;
        do @(posedge clk_core); while (!group_accept);
        @(negedge clk_core);
        update_ready = 0;
        held = make_descriptor(3, 211, 3, 8, 10, 15, 0, 3'b011, 1);
        drive_descriptor_ports(held);
        for (int stall = 0; stall < 73; stall++) begin
            @(posedge clk_core);
            if (group_ready || group_accept || !update_valid || protocol_error
                    || done_valid)
                $fatal(1, "M131 long ready-valid stall failed at %0d", stall);
            group_stall_cycles++;
            @(negedge clk_core);
            if ({group_block, group_row, group_source_count_m1,
                 group_negate, group_last}
                    !== {held.block_id, held.row_id, held.count_m1,
                         held.negate, held.last})
                $fatal(1, "M131 held descriptor metadata changed under stall");
            for (int pick = 0; pick < 4; pick++)
                if (group_source[pick]
                        !== held.source_ids[pick * 4 +: 4])
                    $fatal(1, "M131 held descriptor ID changed under stall");
        end
        update_ready = 1;
        @(posedge clk_core);
        if (!update_accept || !group_accept || !done_valid
                || {done_block, done_row} != {3'd3, 9'd210})
            $fatal(1, "M131 same-cycle stall replacement/tagged done failed");
        long_stall_replace++;
        stop_and_drain();

        // Reset while one output and the next input descriptor are stalled.
        @(negedge clk_core);
        update_ready = 1;
        group_valid = 1;
        drive_descriptor_ports(make_descriptor(3, 220, 2,
                                               0, 1, 0, 0, 0, 1));
        do @(posedge clk_core); while (!group_accept);
        @(negedge clk_core);
        update_ready = 0;
        drive_descriptor_ports(make_descriptor(3, 221, 2,
                                               2, 3, 0, 0, 0, 1));
        #1ps;
        if (!update_valid || group_ready || done_valid)
            $fatal(1, "M131 reset setup did not create dual stall");
        reset_aborted_descriptors += expected_q.size();
        rst_core = 1;
        weight_fill_valid = 1;
        #1ps;
        if (weight_fill_ready || weight_fill_accept || group_ready
                || group_accept || update_valid || update_accept
                || done_valid || protocol_error)
            $fatal(1, "M131 reset timing isolation failed");
        reset_checks++;
        repeat (2) @(posedge clk_core);
        @(negedge clk_core);
        clear_inputs();
        rst_core = 0;
        update_ready = 1;
        repeat (2) @(posedge clk_core);
        if (busy || protocol_error || observed_cache_valid != 0
                || observed_row_stream_open)
            $fatal(1, "M131 reset failed to flush stalled state");

        // Negative campaign: every violation must fail closed and sticky.
        apply_reset(); fill_all();
        expect_fault(make_descriptor(3, 300, 2, 4, 4, 0, 0, 0, 1), 0);

        apply_reset(); fill_all();
        expect_fault(make_descriptor(3, 301, 3, 1, 5, 3, 0, 0, 1), 1);

        apply_reset(); fill_all();
        send_descriptor(make_descriptor(3, 302, 2, 2, 3, 0, 0, 0, 0));
        stop_and_drain();
        expect_fault(make_descriptor(3, 302, 2, 3, 4, 0, 0, 0, 1), 2);

        apply_reset(); fill_all();
        send_descriptor(make_descriptor(3, 303, 2, 5, 6, 0, 0, 0, 0));
        stop_and_drain();
        expect_fault(make_descriptor(3, 303, 2, 4, 7, 0, 0, 0, 1), 3);

        apply_reset(); fill_all();
        send_descriptor(make_descriptor(3, 304, 2, 0, 1, 0, 0, 0, 0));
        stop_and_drain();
        expect_fault(make_descriptor(3, 305, 2, 2, 3, 0, 0, 0, 1), 4);

        apply_reset(); fill_all();
        desc = make_descriptor(3, 306, 1, 2, 0, 0, 0, 0, 1);
        desc.source_ids[4 +: 4] = 7;
        expect_fault(desc, 5);

        apply_reset(); fill_all();
        desc = make_descriptor(3, 307, 2, 1, 3, 0, 0, 0, 1);
        desc.negate[3] = 1;
        expect_fault(desc, 6);

        apply_reset(); fill_all();
        expect_fault(make_descriptor(3, 308, 2, 14, 15, 0, 0, 0, 0), 7);

        apply_reset();
        fill_source(0); fill_source(1); fill_source(2); fill_source(3);
        expect_fault(make_descriptor(3, 309, 1, 8, 0, 0, 0, 0, 1), 8);

        apply_reset(); fill_all();
        expect_fault(make_descriptor(4, 310, 1, 0, 0, 0, 0, 0, 1), 9);

        if (group_accepts != update_accepts + reset_aborted_descriptors
                || lane_checks != update_accepts * LANES
                || k1 == 0 || k2 == 0 || k3 == 0 || k4 == 0
                || plus512 == 0 || minus512 == 0
                || cross_group_intervals != 95
                || cross_update_intervals != 95
                || done_checks != done_tag_checks || done_overlap_checks < 95
                || group_stall_cycles != 73 || max_output_stall < 73
                || long_stall_replace != 1
                || idle_payload_ready_checks != 16
                || open_row_idle_payload_ready_checks != 1
                || within_duplicate_attacks != 1
                || within_descending_attacks != 1
                || cross_repeat_attacks != 1 || cross_backtrack_attacks != 1
                || row_identity_attacks != 1 || dirty_source_attacks != 1
                || dirty_negate_attacks != 1
                || nonlast_source15_attacks != 1
                || cache_miss_attacks != 1 || block_attacks != 1
                || reset_checks != 1
                || gapped_partition_descriptors_accepted != 3)
            $fatal(1, "M131 aggregate mismatch groups=%0d updates=%0d K=%0d/%0d/%0d/%0d cross=%0d/%0d done=%0d/%0d stalls=%0d/%0d",
                   group_accepts, update_accepts, k1, k2, k3, k4,
                   cross_group_intervals, cross_update_intervals,
                   done_checks, done_overlap_checks,
                   group_stall_cycles, max_output_stall);

        $display("PASS M131 independent hammer descriptor_bits=35 groups=%0d updates=%0d reset_aborted_descriptors=%0d sources=%0d lanes=%0d k1=%0d k2=%0d k3=%0d k4=%0d cross_group_ii1_intervals=%0d cross_update_ii1_intervals=%0d done=%0d done_tags=%0d done_overlap_next_row=%0d output_stall_cycles=%0d max_output_stall=%0d group_stall_cycles=%0d long_stall_replace=%0d plus512=%0d minus512=%0d idle_payload_ready_checks=%0d open_row_idle_payload_ready_checks=%0d within_duplicate_attacks=%0d within_descending_attacks=%0d cross_repeat_attacks=%0d cross_backtrack_attacks=%0d row_identity_attacks=%0d dirty_source_attacks=%0d dirty_negate_attacks=%0d nonlast_source15_attacks=%0d cache_miss_attacks=%0d block_attacks=%0d reset_checks=%0d gapped_partition_descriptors_accepted=%0d internal_ready_valid_loop_observed=false predecessor_negative_index_present=false complete_row_partition_losslessness=false descriptor_producer_implemented=false descriptor_payload_bits_only=true dc_frequency_improvement=false physical_speedup=false system_speedup=false headline=false",
                 group_accepts, update_accepts, reset_aborted_descriptors,
                 source_checks, lane_checks, k1, k2, k3, k4,
                 cross_group_intervals, cross_update_intervals,
                 done_checks, done_tag_checks, done_overlap_checks,
                 output_stall_cycles, max_output_stall, group_stall_cycles,
                 long_stall_replace, plus512, minus512,
                 idle_payload_ready_checks, open_row_idle_payload_ready_checks,
                 within_duplicate_attacks, within_descending_attacks,
                 cross_repeat_attacks, cross_backtrack_attacks,
                 row_identity_attacks, dirty_source_attacks,
                 dirty_negate_attacks, nonlast_source15_attacks,
                 cache_miss_attacks, block_attacks, reset_checks,
                 gapped_partition_descriptors_accepted);
        $finish;
    end

    initial begin
        #500000;
        $fatal(1, "M131 independent VCS timeout");
    end
endmodule

`default_nettype wire
