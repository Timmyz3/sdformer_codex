`timescale 1ns/1ps
`default_nettype none

module tb_m128_independent_hammer;
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
    logic [3:0] group_source_valid;
    logic [3:0] group_source [0:3];
    logic [3:0] group_negate;
    logic [15:0] group_selected_mask;
    logic group_last;
    logic update_valid, update_ready, update_accept;
    logic [2:0] update_block;
    logic [8:0] update_row;
    logic [UPDATE_BITS-1:0] update_delta;
    logic [15:0] update_selected_mask;
    logic update_last, row_done;
    logic [15:0] observed_cache_valid;
    logic observed_resident_block_valid;
    logic [2:0] observed_resident_block;
    logic observed_pair_pipeline_valid, protocol_error, busy;

    typedef struct packed {
        logic [2:0] block_id;
        logic [8:0] row_id;
        logic [3:0] source_valid;
        logic [15:0] source_ids;
        logic [3:0] negate;
        logic [15:0] selected_mask;
        logic last;
    } descriptor53_t;

    descriptor53_t expected_q[$];
    bit cross_phase;
    int cycle_count, group_accepts, update_accepts, lane_checks;
    int source_checks, row_done_checks, stall_cycles, group_stall_cycles;
    int maximum_output_stall, current_output_stall;
    int k1, k2, k3, k4, plus512, minus512;
    int cross_group_intervals, cross_update_intervals;
    int last_cross_group_cycle, last_cross_update_cycle;
    int long_stall_replacement_checks, reset_checks;
    int reset_aborted_descriptors;
    int duplicate_attacks, dirty_source_attacks, dirty_negate_attacks;
    int mask_attacks, cache_miss_attacks, block_attacks;
    int fill_collision_attacks, empty_attacks;
    int noncanonical_order_accepts, holey_valid_accepts;
    int cross_descriptor_duplicate_accepts;
    int semantic_ready_probes, done_overlap_next_row;
    bit expected_row_done;
    logic [8:0] expected_done_row;

    m128_descriptor_streamed_k4_row_fold dut (.*);

    m128_descriptor_streamed_k4_row_fold_assertions sva (
        .clk_core, .rst_core, .weight_fill_accept, .weight_fill_ready,
        .group_valid, .group_ready, .group_accept, .group_source_valid,
        .update_valid, .update_ready, .update_accept, .update_block,
        .update_row, .update_delta, .update_selected_mask, .update_last,
        .row_done, .protocol_error
    );

    initial clk_core = 0;
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
                raw = (source * 47 + lane * 31 + source * lane * 5) & 8'hff;
                model_weight = raw - 128;
            end
        end
    endfunction

    function automatic descriptor53_t make_descriptor(
        input int block_id,
        input int row_id,
        input logic [3:0] valid,
        input int id0, input int id1, input int id2, input int id3,
        input logic [3:0] negate,
        input bit last
    );
        descriptor53_t result;
        int ids [0:3];
        begin
            ids[0] = id0; ids[1] = id1; ids[2] = id2; ids[3] = id3;
            result = '0;
            result.block_id = block_id[2:0];
            result.row_id = row_id[8:0];
            result.source_valid = valid;
            result.negate = negate;
            result.last = last;
            for (int pick = 0; pick < 4; pick++) begin
                result.source_ids[pick * 4 +: 4] = ids[pick][3:0];
                if (valid[pick])
                    result.selected_mask[ids[pick]] = 1'b1;
            end
            return result;
        end
    endfunction

    function automatic logic [UPDATE_BITS-1:0]
        expected_delta(input descriptor53_t desc);
        logic [UPDATE_BITS-1:0] result;
        integer signed sum;
        int source;
        begin
            result = '0;
            for (int lane = 0; lane < LANES; lane++) begin
                sum = 0;
                for (int pick = 0; pick < 4; pick++) begin
                    if (desc.source_valid[pick]) begin
                        source = desc.source_ids[pick * 4 +: 4];
                        sum = sum + (desc.negate[pick]
                              ? -model_weight(source, lane)
                              : model_weight(source, lane));
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
            group_source_valid = 0;
            group_negate = 0;
            group_selected_mask = 0;
            group_last = 0;
            for (int pick = 0; pick < 4; pick++)
                group_source[pick] = 0;
        end
    endtask

    task automatic drive_descriptor_ports(input descriptor53_t desc);
        begin
            group_block = desc.block_id;
            group_row = desc.row_id;
            group_source_valid = desc.source_valid;
            group_negate = desc.negate;
            group_selected_mask = desc.selected_mask;
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
                    || observed_pair_pipeline_valid)
                $fatal(1, "M128 reset did not clear architectural state");
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
                $fatal(1, "M128 full cache identity mismatch");
        end
    endtask

    task automatic send_descriptor(input descriptor53_t desc);
        begin
            @(negedge clk_core);
            group_valid = 1;
            drive_descriptor_ports(desc);
            do @(posedge clk_core); while (!group_accept);
            if (protocol_error)
                $fatal(1, "M128 legal descriptor raised protocol error");
        end
    endtask

    task automatic stop_and_drain;
        int watchdog;
        begin
            @(negedge clk_core);
            group_valid = 0;
            watchdog = 0;
            while (expected_q.size() != 0 || update_valid) begin
                @(posedge clk_core);
                watchdog++;
                if (watchdog > 1000)
                    $fatal(1, "M128 drain timeout");
            end
            repeat (2) @(posedge clk_core);
        end
    endtask

    task automatic expect_descriptor_fault(
        input descriptor53_t desc,
        input int kind
    );
        begin
            @(negedge clk_core);
            group_valid = 1;
            drive_descriptor_ports(desc);
            #1ps;
            if (!protocol_error || group_ready || group_accept || update_valid)
                $fatal(1, "M128 descriptor attack kind=%0d not fail closed", kind);
            @(posedge clk_core);
            @(negedge clk_core);
            group_valid = 0;
            repeat (2) @(posedge clk_core);
            if (!protocol_error)
                $fatal(1, "M128 descriptor attack kind=%0d not sticky", kind);
            case (kind)
                0: duplicate_attacks++;
                1: dirty_source_attacks++;
                2: dirty_negate_attacks++;
                3: mask_attacks++;
                4: cache_miss_attacks++;
                5: block_attacks++;
                default: empty_attacks++;
            endcase
        end
    endtask

    always @(posedge clk_core) begin : scoreboard
        descriptor53_t front;
        descriptor53_t accepted;
        cycle_count++;
        if (rst_core) begin
            expected_q.delete();
            expected_row_done = 0;
            current_output_stall = 0;
        end else begin
            if (row_done !== expected_row_done)
                $fatal(1, "M128 row_done timing mismatch cycle=%0d", cycle_count);
            if (row_done) begin
                row_done_checks++;
                if (update_valid && update_row != expected_done_row)
                    done_overlap_next_row++;
            end
            if (update_valid) begin
                if (expected_q.size() == 0)
                    $fatal(1, "M128 visible update with empty scoreboard");
                front = expected_q[0];
                if ({update_block, update_row, update_selected_mask, update_last}
                    !== {front.block_id, front.row_id,
                         front.selected_mask, front.last})
                    $fatal(1, "M128 update identity mismatch cycle=%0d", cycle_count);
                if (update_delta !== expected_delta(front))
                    $fatal(1, "M128 update numeric mismatch cycle=%0d", cycle_count);
                if (!update_ready) begin
                    stall_cycles++;
                    current_output_stall++;
                    if (current_output_stall > maximum_output_stall)
                        maximum_output_stall = current_output_stall;
                end else begin
                    current_output_stall = 0;
                end
            end else begin
                current_output_stall = 0;
            end
            if (update_accept) begin
                front = expected_q.pop_front();
                update_accepts++;
                lane_checks += LANES;
                if ($signed(update_delta[0 +: ACC_BITS]) == 512)
                    plus512++;
                if ($signed(update_delta[0 +: ACC_BITS]) == -512)
                    minus512++;
                if (cross_phase) begin
                    if (last_cross_update_cycle != 0) begin
                        if (cycle_count - last_cross_update_cycle != 1)
                            $fatal(1, "M128 cross-row update II drift");
                        cross_update_intervals++;
                    end
                    last_cross_update_cycle = cycle_count;
                end
                expected_row_done = update_last;
                expected_done_row = update_row;
            end else begin
                expected_row_done = 0;
            end
            if (group_accept) begin
                accepted.block_id = group_block;
                accepted.row_id = group_row;
                accepted.source_valid = group_source_valid;
                accepted.negate = group_negate;
                accepted.selected_mask = group_selected_mask;
                accepted.last = group_last;
                for (int pick = 0; pick < 4; pick++)
                    accepted.source_ids[pick * 4 +: 4] = group_source[pick];
                expected_q.push_back(accepted);
                group_accepts++;
                source_checks += $countones(group_source_valid);
                case ($countones(group_source_valid))
                    1: k1++;
                    2: k2++;
                    3: k3++;
                    4: k4++;
                    default: $fatal(1, "M128 accepted empty descriptor");
                endcase
                if (cross_phase) begin
                    if (last_cross_group_cycle != 0) begin
                        if (cycle_count - last_cross_group_cycle != 1)
                            $fatal(1, "M128 cross-row group II drift");
                        cross_group_intervals++;
                    end
                    last_cross_group_cycle = cycle_count;
                end
            end
        end
    end

    initial begin
        descriptor53_t desc;
        descriptor53_t held;
        logic [15:0] original_mask;
        logic [3:0] original_ids [0:3];
        int watchdog;

        if ($bits(descriptor53_t) != 53)
            $fatal(1, "M128 descriptor packed width is not 53 bits");
        clk_core = 0;
        rst_core = 1;
        update_ready = 0;
        cross_phase = 0;
        cycle_count = 0;
        group_accepts = 0;
        update_accepts = 0;
        lane_checks = 0;
        source_checks = 0;
        row_done_checks = 0;
        stall_cycles = 0;
        group_stall_cycles = 0;
        maximum_output_stall = 0;
        current_output_stall = 0;
        k1 = 0; k2 = 0; k3 = 0; k4 = 0;
        plus512 = 0; minus512 = 0;
        cross_group_intervals = 0;
        cross_update_intervals = 0;
        last_cross_group_cycle = 0;
        last_cross_update_cycle = 0;
        long_stall_replacement_checks = 0;
        reset_checks = 0;
        reset_aborted_descriptors = 0;
        duplicate_attacks = 0;
        dirty_source_attacks = 0;
        dirty_negate_attacks = 0;
        mask_attacks = 0;
        cache_miss_attacks = 0;
        block_attacks = 0;
        fill_collision_attacks = 0;
        empty_attacks = 0;
        noncanonical_order_accepts = 0;
        holey_valid_accepts = 0;
        cross_descriptor_duplicate_accepts = 0;
        semantic_ready_probes = 0;
        done_overlap_next_row = 0;
        expected_row_done = 0;
        clear_inputs();

        apply_reset();
        fill_all();

        // Ready is payload-semantic even when valid is low. This settles
        // without an internal loop, but is not a payload-independent ready.
        @(negedge clk_core);
        desc = make_descriptor(3, 1, 4'b0001, 0, 0, 0, 0, 0, 1);
        drive_descriptor_ports(desc);
        group_valid = 0;
        #1ps;
        if (!group_ready || protocol_error)
            $fatal(1, "M128 legal semantic-ready probe failed");
        semantic_ready_probes++;
        group_selected_mask = 16'h0002;
        #1ps;
        if (group_ready || protocol_error)
            $fatal(1, "M128 invalid semantic-ready probe failed");
        semantic_ready_probes++;
        clear_inputs();

        // Directed K1-K4 and both signed boundaries.
        send_descriptor(make_descriptor(3, 10, 4'b0001, 15, 0, 0, 0, 0, 1));
        send_descriptor(make_descriptor(3, 11, 4'b0011, 1, 9, 0, 0, 2'b01, 1));
        send_descriptor(make_descriptor(3, 12, 4'b0111, 2, 7, 14, 0, 3'b101, 1));
        send_descriptor(make_descriptor(3, 13, 4'b1111, 0, 1, 2, 3, 4'b0000, 1));
        send_descriptor(make_descriptor(3, 14, 4'b1111, 0, 1, 2, 3, 4'b1111, 1));
        stop_and_drain();

        // 128 independent rows: both descriptor accepts and updates are II1.
        cross_phase = 1;
        last_cross_group_cycle = 0;
        last_cross_update_cycle = 0;
        for (int row = 0; row < 128; row++)
            send_descriptor(make_descriptor(3, 100 + row, 4'b1111,
                                            0, 1, 2, 3,
                                            row[3:0], 1));
        stop_and_drain();
        cross_phase = 0;

        // Populate the output stage, hold the next descriptor stable for 97
        // cycles, then replace the retiring entry in one accepted cycle.
        @(negedge clk_core);
        update_ready = 0;
        group_valid = 1;
        desc = make_descriptor(3, 300, 4'b1111, 4, 5, 6, 7, 4'b0101, 1);
        drive_descriptor_ports(desc);
        do @(posedge clk_core); while (!group_accept);
        @(negedge clk_core);
        held = make_descriptor(3, 301, 4'b0111, 8, 10, 15, 0, 3'b011, 1);
        drive_descriptor_ports(held);
        for (int stall = 0; stall < 97; stall++) begin
            @(posedge clk_core);
            if (group_ready || group_accept || !update_valid || protocol_error)
                $fatal(1, "M128 long ready-valid stall failed at %0d", stall);
            group_stall_cycles++;
            @(negedge clk_core);
            if ({group_block, group_row, group_source_valid,
                 group_negate, group_selected_mask, group_last}
                !== {held.block_id, held.row_id, held.source_valid,
                     held.negate, held.selected_mask, held.last})
                $fatal(1, "M128 held descriptor changed under stall");
            for (int pick = 0; pick < 4; pick++)
                if (group_source[pick]
                        !== held.source_ids[pick * 4 +: 4])
                    $fatal(1, "M128 held ID changed under stall");
        end
        update_ready = 1;
        @(posedge clk_core);
        if (!update_accept || !group_accept)
            $fatal(1, "M128 same-cycle stall replacement failed");
        long_stall_replacement_checks++;
        stop_and_drain();

        // Locally distinct but noncanonical descriptors are accepted. These
        // probes establish that order/packing belongs to the external producer.
        desc = make_descriptor(3, 320, 4'b1111, 7, 2, 12, 1, 4'b0101, 1);
        send_descriptor(desc);
        noncanonical_order_accepts++;
        desc = make_descriptor(3, 321, 4'b0101, 2, 0, 8, 0, 4'b0101, 1);
        send_descriptor(desc);
        holey_valid_accepts++;
        // The core has no cross-descriptor per-row conservation ledger.
        desc = make_descriptor(3, 322, 4'b0001, 5, 0, 0, 0, 0, 0);
        send_descriptor(desc);
        desc = make_descriptor(3, 322, 4'b0001, 5, 0, 0, 0, 0, 1);
        send_descriptor(desc);
        cross_descriptor_duplicate_accepts += 2;
        stop_and_drain();

        // Reset while one output and one input descriptor are stalled.
        @(negedge clk_core);
        update_ready = 0;
        group_valid = 1;
        drive_descriptor_ports(make_descriptor(3, 330, 4'b0011,
                                               0, 1, 0, 0, 0, 1));
        do @(posedge clk_core); while (!group_accept);
        @(negedge clk_core);
        drive_descriptor_ports(make_descriptor(3, 331, 4'b0011,
                                               2, 3, 0, 0, 0, 1));
        #1ps;
        if (!update_valid || group_ready)
            $fatal(1, "M128 reset setup did not create dual stall");
        reset_aborted_descriptors += expected_q.size();
        rst_core = 1;
        weight_fill_valid = 1;
        #1ps;
        if (weight_fill_ready || weight_fill_accept || group_ready
                || group_accept || update_valid || update_accept
                || row_done || protocol_error)
            $fatal(1, "M128 reset timing isolation failed");
        reset_checks++;
        repeat (2) @(posedge clk_core);
        @(negedge clk_core);
        clear_inputs();
        rst_core = 0;
        update_ready = 1;
        repeat (2) @(posedge clk_core);
        if (busy || protocol_error || observed_cache_valid != 0)
            $fatal(1, "M128 reset failed to flush stalled state");

        // Negative descriptor audit campaign; reset/refill only required IDs.
        apply_reset(); fill_source(0); fill_source(1); fill_source(2); fill_source(3);
        desc = make_descriptor(3, 400, 4'b0011, 1, 1, 0, 0, 0, 1);
        expect_descriptor_fault(desc, 0);

        apply_reset(); fill_source(0); fill_source(1); fill_source(2); fill_source(3);
        desc = make_descriptor(3, 401, 4'b0001, 1, 0, 0, 0, 0, 1);
        desc.source_ids[4 +: 4] = 4'd9;
        expect_descriptor_fault(desc, 1);

        apply_reset(); fill_source(0); fill_source(1); fill_source(2); fill_source(3);
        desc = make_descriptor(3, 402, 4'b0001, 1, 0, 0, 0, 0, 1);
        desc.negate[2] = 1;
        expect_descriptor_fault(desc, 2);

        apply_reset(); fill_source(0); fill_source(1); fill_source(2); fill_source(3);
        desc = make_descriptor(3, 403, 4'b0011, 0, 2, 0, 0, 0, 1);
        desc.selected_mask = 16'h0001;
        expect_descriptor_fault(desc, 3);

        apply_reset(); fill_source(0); fill_source(1); fill_source(2); fill_source(3);
        desc = make_descriptor(3, 404, 4'b0001, 8, 0, 0, 0, 0, 1);
        expect_descriptor_fault(desc, 4);

        apply_reset(); fill_source(0); fill_source(1); fill_source(2); fill_source(3);
        desc = make_descriptor(4, 405, 4'b0001, 0, 0, 0, 0, 0, 1);
        expect_descriptor_fault(desc, 5);

        apply_reset(); fill_source(0); fill_source(1); fill_source(2); fill_source(3);
        desc = make_descriptor(3, 406, 4'b0000, 0, 0, 0, 0, 0, 1);
        expect_descriptor_fault(desc, 6);

        apply_reset(); fill_source(0); fill_source(1); fill_source(2); fill_source(3);
        @(negedge clk_core);
        group_valid = 1;
        drive_descriptor_ports(make_descriptor(3, 407, 4'b0001,
                                               0, 0, 0, 0, 0, 1));
        weight_fill_valid = 1;
        weight_fill_block = 3;
        weight_fill_source = 1;
        weight_fill_beat = 0;
        weight_fill_data = 0;
        #1ps;
        if (!protocol_error || group_ready || group_accept
                || weight_fill_ready || weight_fill_accept || update_valid)
            $fatal(1, "M128 fill/group collision not fail closed");
        fill_collision_attacks++;
        @(posedge clk_core);
        @(negedge clk_core);
        clear_inputs();
        repeat (2) @(posedge clk_core);
        if (!protocol_error)
            $fatal(1, "M128 fill collision fault not sticky");

        if (group_accepts != update_accepts + reset_aborted_descriptors
                || lane_checks != update_accepts * LANES
                || source_checks < 500 || k1 == 0 || k2 == 0
                || k3 == 0 || k4 == 0 || plus512 == 0 || minus512 == 0
                || cross_group_intervals != 127
                || cross_update_intervals != 127
                || group_stall_cycles != 97
                || maximum_output_stall < 97
                || long_stall_replacement_checks != 1
                || reset_checks != 1
                || duplicate_attacks != 1 || dirty_source_attacks != 1
                || dirty_negate_attacks != 1 || mask_attacks != 1
                || cache_miss_attacks != 1 || block_attacks != 1
                || empty_attacks != 1 || fill_collision_attacks != 1
                || noncanonical_order_accepts != 1
                || holey_valid_accepts != 1
                || cross_descriptor_duplicate_accepts != 2
                || semantic_ready_probes != 2
                || done_overlap_next_row == 0)
            $fatal(1, "M128 aggregate mismatch groups=%0d updates=%0d sources=%0d lanes=%0d K=%0d/%0d/%0d/%0d cross=%0d/%0d stalls=%0d/%0d done_overlap=%0d",
                   group_accepts, update_accepts, source_checks, lane_checks,
                   k1, k2, k3, k4, cross_group_intervals,
                   cross_update_intervals, group_stall_cycles,
                   maximum_output_stall, done_overlap_next_row);

        $display("PASS M128 independent hammer descriptor_bits=53 groups=%0d updates=%0d reset_aborted_descriptors=%0d sources=%0d lanes=%0d k1=%0d k2=%0d k3=%0d k4=%0d cross_group_ii1_intervals=%0d cross_update_ii1_intervals=%0d output_stall_cycles=%0d max_output_stall=%0d group_stall_cycles=%0d long_stall_replace=%0d plus512=%0d minus512=%0d row_done_checks=%0d row_done_overlap_next_row=%0d semantic_ready_probes=%0d duplicate_attacks=%0d dirty_source_attacks=%0d dirty_negate_attacks=%0d mask_attacks=%0d cache_miss_attacks=%0d block_attacks=%0d empty_attacks=%0d fill_collision_attacks=%0d reset_checks=%0d noncanonical_order_accepts=%0d holey_valid_accepts=%0d cross_descriptor_duplicate_accepts=%0d internal_combinational_loop_observed=false canonical_order_enforced=false valid_left_packing_enforced=false cross_descriptor_source_conservation_enforced=false descriptor_predecode_external=true descriptor_predecode_cost_modeled=false descriptor_bandwidth_accounted=false dc_frequency_improvement=false physical_speedup=false system_speedup=false headline=false",
                 group_accepts, update_accepts, reset_aborted_descriptors,
                 source_checks, lane_checks,
                 k1, k2, k3, k4, cross_group_intervals,
                 cross_update_intervals, stall_cycles, maximum_output_stall,
                 group_stall_cycles, long_stall_replacement_checks,
                 plus512, minus512, row_done_checks,
                 done_overlap_next_row, semantic_ready_probes,
                 duplicate_attacks, dirty_source_attacks,
                 dirty_negate_attacks, mask_attacks, cache_miss_attacks,
                 block_attacks, empty_attacks, fill_collision_attacks,
                 reset_checks, noncanonical_order_accepts,
                 holey_valid_accepts, cross_descriptor_duplicate_accepts);
        $finish;
    end

    initial begin
        #500000;
        $fatal(1, "M128 independent VCS timeout");
    end
endmodule

`default_nettype wire
