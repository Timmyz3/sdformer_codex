`timescale 1ns/1ps
`default_nettype none

module tb_m131_synthesis_safe_compact_canonical_k4_row_fold;
    localparam int SOURCES = 16;
    localparam int LANES = 96;
    localparam int ACC_BITS = 19;
    localparam int UPDATE_BITS = LANES * ACC_BITS;

    logic clk_core;
    logic rst_core;
    logic weight_fill_valid;
    logic weight_fill_ready;
    logic [2:0] weight_fill_block;
    logic [3:0] weight_fill_source;
    logic [1:0] weight_fill_beat;
    logic [255:0] weight_fill_data;
    logic weight_fill_accept;
    logic group_valid;
    logic group_ready;
    logic [2:0] group_block;
    logic [8:0] group_row;
    logic [1:0] group_source_count_m1;
    logic [3:0] group_source [0:3];
    logic [3:0] group_negate;
    logic group_last;
    logic group_accept;
    logic update_valid;
    logic update_ready;
    logic [2:0] update_block;
    logic [8:0] update_row;
    logic [UPDATE_BITS-1:0] update_delta;
    logic [15:0] update_selected_mask;
    logic update_last;
    logic update_accept;
    logic done_valid;
    logic [2:0] done_block;
    logic [8:0] done_row;
    logic [15:0] observed_cache_valid;
    logic observed_resident_block_valid;
    logic [2:0] observed_resident_block;
    logic observed_pair_pipeline_valid;
    logic observed_row_stream_open;
    logic protocol_error;
    logic busy;

    logic signed [7:0] weight_model [0:SOURCES-1][0:LANES-1];
    bit force_update_ready;
    bit pattern_stall_enable;
    bit force_update_stall;
    bit cross_row_phase;
    int unsigned cycle_count;
    int unsigned group_accept_count;
    int unsigned update_accept_count;
    int unsigned source_contribution_count;
    int unsigned lane_check_count;
    int unsigned done_count;
    int unsigned done_overlap_next_group_count;
    int unsigned stall_cycle_count;
    int unsigned long_stall_cycles;
    int unsigned cross_row_update_count;
    int unsigned cross_row_ii1_count;
    int unsigned last_cross_row_update_cycle;
    int unsigned plus512_checks;
    int unsigned protocol_attacks;
    int unsigned reset_attacks;
    int unsigned idle_payload_toggle_checks;

    typedef struct packed {
        logic [2:0] block_id;
        logic [8:0] row_id;
        logic [15:0] selected_mask;
        logic last;
        logic [UPDATE_BITS-1:0] delta;
    } expected_update_t;
    expected_update_t expected_q[$];

    m131_synthesis_safe_compact_canonical_k4_row_fold dut (.*);

    m131_synthesis_safe_compact_canonical_k4_row_fold_assertions sva (
        .clk_core,
        .rst_core,
        .weight_fill_accept,
        .weight_fill_ready,
        .group_valid,
        .group_ready,
        .group_accept,
        .group_block,
        .group_row,
        .group_source_count_m1,
        .group_source,
        .group_negate,
        .group_last,
        .update_valid,
        .update_ready,
        .update_accept,
        .update_block,
        .update_row,
        .update_delta,
        .update_selected_mask,
        .update_last,
        .done_valid,
        .done_block,
        .done_row,
        .protocol_error
    );

    initial clk_core = 1'b0;
    always #1.5 clk_core = ~clk_core;

    function automatic logic [15:0] current_group_mask();
        logic [15:0] result;
        result = '0;
        for (int pick = 0; pick <= group_source_count_m1; pick++)
            result[group_source[pick]] = 1'b1;
        return result;
    endfunction

    function automatic logic [UPDATE_BITS-1:0]
        expected_current_group_delta();
        logic [UPDATE_BITS-1:0] result;
        integer signed lane_sum;
        integer signed weight_value;
        result = '0;
        for (int lane = 0; lane < LANES; lane++) begin
            lane_sum = 0;
            for (int pick = 0; pick <= group_source_count_m1; pick++) begin
                weight_value = $signed(
                    weight_model[group_source[pick]][lane]);
                if (group_negate[pick])
                    lane_sum = lane_sum - weight_value;
                else
                    lane_sum = lane_sum + weight_value;
            end
            result[lane * ACC_BITS +: ACC_BITS] = lane_sum[ACC_BITS-1:0];
        end
        return result;
    endfunction

    task automatic clear_inputs;
        weight_fill_valid = 1'b0;
        weight_fill_block = '0;
        weight_fill_source = '0;
        weight_fill_beat = '0;
        weight_fill_data = '0;
        group_valid = 1'b0;
        group_block = '0;
        group_row = '0;
        group_source_count_m1 = '0;
        group_negate = '0;
        group_last = 1'b0;
        for (int pick = 0; pick < 4; pick++)
            group_source[pick] = '0;
    endtask

    task automatic clear_group;
        group_valid = 1'b0;
        group_block = '0;
        group_row = '0;
        group_source_count_m1 = '0;
        group_negate = '0;
        group_last = 1'b0;
        for (int pick = 0; pick < 4; pick++)
            group_source[pick] = '0;
    endtask

    task automatic apply_reset(input int cycles);
        @(negedge clk_core);
        rst_core = 1'b1;
        clear_inputs();
        repeat (cycles) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
    endtask

    task automatic fill_one_source(input int source_id);
        for (int beat = 0; beat < 3; beat++) begin
            @(negedge clk_core);
            weight_fill_valid = 1'b1;
            weight_fill_block = 3'd3;
            weight_fill_source = source_id[3:0];
            weight_fill_beat = beat[1:0];
            weight_fill_data = '0;
            for (int lane_in_beat = 0; lane_in_beat < 32;
                    lane_in_beat++) begin
                weight_fill_data[lane_in_beat * 8 +: 8]
                    = weight_model[source_id][beat * 32 + lane_in_beat];
            end
            do begin
                @(posedge clk_core);
                if (protocol_error)
                    $fatal(1, "unexpected protocol error during fill");
            end while (!weight_fill_accept);
        end
        @(negedge clk_core);
        weight_fill_valid = 1'b0;
    endtask

    task automatic fill_all_sources;
        for (int source = 0; source < SOURCES; source++)
            fill_one_source(source);
        if (observed_cache_valid !== 16'hffff
                || !observed_resident_block_valid
                || observed_resident_block != 3)
            $fatal(1, "cache identity did not close after fill");
    endtask

    task automatic drive_group(
        input int row_id,
        input int source_base,
        input int source_count,
        input bit last,
        input bit force_plus512
    );
        if (source_count < 1 || source_count > 4
                || source_base < 0
                || source_base + source_count > SOURCES)
            $fatal(1, "invalid canonical group task arguments");
        @(negedge clk_core);
        group_valid = 1'b1;
        group_block = 3'd3;
        group_row = row_id[8:0];
        group_source_count_m1 = (source_count - 1);
        group_negate = '0;
        group_last = last;
        for (int pick = 0; pick < 4; pick++) begin
            group_source[pick] = '0;
            if (pick < source_count) begin
                group_source[pick] = source_base + pick;
                group_negate[pick] = force_plus512
                    ? 1'b1 : ((row_id + source_base + pick) % 3 == 0);
            end
        end
        do begin
            @(posedge clk_core);
            if (protocol_error)
                $fatal(1, "unexpected protocol error during row %0d",
                       row_id);
        end while (!group_accept);
    endtask

    task automatic reset_and_refill;
        apply_reset(3);
        if (protocol_error || busy)
            $fatal(1, "reset did not clear compact fold island");
        fill_all_sources();
    endtask

    always @(negedge clk_core) begin
        if (rst_core || force_update_stall)
            update_ready <= 1'b0;
        else if (force_update_ready)
            update_ready <= 1'b1;
        else if (pattern_stall_enable)
            update_ready <= ((cycle_count % 5) != 0);
        else
            update_ready <= 1'b1;
    end

    always @(posedge clk_core) begin : scoreboard
        expected_update_t accepted;
        cycle_count++;

        if (!rst_core && update_valid) begin
            if (expected_q.size() == 0)
                $fatal(1, "update visible with empty scoreboard");
            if (update_block !== expected_q[0].block_id
                    || update_row !== expected_q[0].row_id
                    || update_selected_mask !== expected_q[0].selected_mask
                    || update_last !== expected_q[0].last
                    || update_delta !== expected_q[0].delta)
                $fatal(1, "compact descriptor update mismatch at cycle %0d",
                       cycle_count);
            if (!update_ready)
                stall_cycle_count++;
        end

        if (!rst_core && done_valid) begin
            if (!update_accept || !update_last
                    || done_block !== update_block
                    || done_row !== update_row)
                $fatal(1, "tagged done mismatch at cycle %0d", cycle_count);
            done_count++;
            if (group_accept
                    && {group_block, group_row} != {done_block, done_row})
                done_overlap_next_group_count++;
        end

        if (!rst_core && update_accept) begin
            if (expected_q.size() == 0)
                $fatal(1, "accepted update with empty scoreboard");
            if ($signed(update_delta[0 +: ACC_BITS]) == 512)
                plus512_checks++;
            expected_q.pop_front();
            update_accept_count++;
            lane_check_count += LANES;
            if (cross_row_phase && update_last) begin
                if (cross_row_update_count != 0) begin
                    if (cycle_count - last_cross_row_update_cycle != 1)
                        $fatal(1, "cross-row compact descriptor II drift: %0d",
                               cycle_count - last_cross_row_update_cycle);
                    cross_row_ii1_count++;
                end
                last_cross_row_update_cycle = cycle_count;
                cross_row_update_count++;
            end
        end

        if (!rst_core && group_accept) begin
            accepted.block_id = group_block;
            accepted.row_id = group_row;
            accepted.selected_mask = current_group_mask();
            accepted.last = group_last;
            accepted.delta = expected_current_group_delta();
            expected_q.push_back(accepted);
            group_accept_count++;
            source_contribution_count += group_source_count_m1 + 1;
        end
    end

    initial begin : test_sequence
        int source_count;
        int source_base;
        rst_core = 1'b1;
        update_ready = 1'b0;
        force_update_ready = 1'b0;
        pattern_stall_enable = 1'b0;
        force_update_stall = 1'b0;
        cross_row_phase = 1'b0;
        cycle_count = 0;
        group_accept_count = 0;
        update_accept_count = 0;
        source_contribution_count = 0;
        lane_check_count = 0;
        done_count = 0;
        done_overlap_next_group_count = 0;
        stall_cycle_count = 0;
        long_stall_cycles = 0;
        cross_row_update_count = 0;
        cross_row_ii1_count = 0;
        last_cross_row_update_cycle = 0;
        plus512_checks = 0;
        protocol_attacks = 0;
        reset_attacks = 0;
        idle_payload_toggle_checks = 0;
        clear_inputs();

        for (int source = 0; source < SOURCES; source++) begin
            for (int lane = 0; lane < LANES; lane++) begin
                weight_model[source][lane]
                    = ((source * 31 + lane * 17) % 256) - 128;
            end
        end
        for (int source = 0; source < 4; source++)
            weight_model[source][0] = -128;

        apply_reset(3);
        force_update_ready = 1'b1;
        fill_all_sources();

        // Idle ready is capacity-only: invalid payload toggles cannot lower it
        // while group_valid is low.
        @(negedge clk_core);
        group_valid = 1'b0;
        group_block = 3'd7;
        group_row = 9'h1ff;
        group_source_count_m1 = 2'd3;
        group_source[0] = 4'd14;
        group_source[1] = 4'd2;
        group_source[2] = 4'd2;
        group_source[3] = 4'd1;
        group_negate = 4'hf;
        group_last = 1'b0;
        #1ps;
        if (!group_ready || protocol_error)
            $fatal(1, "idle group_ready depends on semantic payload");
        idle_payload_toggle_checks++;

        // 64 single-descriptor rows.  Once the first update appears, all
        // accepted updates and tagged completions must be adjacent.
        cross_row_phase = 1'b1;
        drive_group(0, 0, 4, 1'b1, 1'b1);
        for (int row = 1; row < 64; row++)
            drive_group(row, row % 13, 4, 1'b1, 1'b0);
        @(negedge clk_core);
        clear_group();
        wait (update_accept_count == group_accept_count);
        @(posedge clk_core);
        cross_row_phase = 1'b0;
        if (cross_row_update_count != 64 || cross_row_ii1_count != 63)
            $fatal(1, "cross-row II=1 coverage mismatch %0d/%0d",
                   cross_row_update_count, cross_row_ii1_count);
        if (done_overlap_next_group_count < 63)
            $fatal(1, "tagged done did not overlap next-row groups");

        // Canonical K1-K4 traffic, including multi-descriptor rows, under a
        // deterministic one-in-five output stall pattern.
        force_update_ready = 1'b0;
        pattern_stall_enable = 1'b1;
        for (int row = 64; row < 192; row++) begin
            source_count = 1 + (row % 4);
            if ((row % 3) == 0) begin
                drive_group(row, 0, source_count, 1'b0, 1'b0);
                drive_group(row, source_count,
                            1 + ((row + 1) % 4), 1'b1, 1'b0);
            end else begin
                source_base = (row * 5) % (17 - source_count);
                drive_group(row, source_base, source_count, 1'b1, 1'b0);
            end
        end
        @(negedge clk_core);
        clear_group();
        wait (update_accept_count == group_accept_count);

        // Explicit 17-cycle output stall proves full payload stability.
        pattern_stall_enable = 1'b0;
        force_update_ready = 1'b1;
        drive_group(192, 4, 4, 1'b1, 1'b0);
        force_update_ready = 1'b0;
        force_update_stall = 1'b1;
        @(negedge clk_core);
        clear_group();
        repeat (17) begin
            @(posedge clk_core);
            if (!update_valid)
                $fatal(1, "long-stall update disappeared");
            long_stall_cycles++;
        end
        @(negedge clk_core);
        force_update_stall = 1'b0;
        force_update_ready = 1'b1;
        wait (update_accept_count == group_accept_count);
        repeat (2) @(posedge clk_core);
        if (expected_q.size() != 0 || update_valid
                || observed_row_stream_open)
            $fatal(1, "compact descriptor pipe did not drain");
        if (plus512_checks == 0)
            $fatal(1, "+512 signed boundary was not observed");

        // Unsorted IDs fail closed.
        @(negedge clk_core);
        group_valid = 1'b1;
        group_block = 3'd3;
        group_row = 9'd400;
        group_source_count_m1 = 2'd1;
        group_source[0] = 4'd6;
        group_source[1] = 4'd5;
        group_source[2] = '0;
        group_source[3] = '0;
        group_negate = '0;
        group_last = 1'b1;
        #1ps;
        if (!protocol_error || group_ready || group_accept || update_valid)
            $fatal(1, "unsorted compact descriptor was not quarantined");
        protocol_attacks++;
        @(posedge clk_core);
        reset_and_refill();

        // A source repeated across two descriptors of one row fails closed.
        drive_group(401, 2, 2, 1'b0, 1'b0);
        @(negedge clk_core);
        clear_group();
        wait (update_accept_count == group_accept_count);
        @(negedge clk_core);
        group_valid = 1'b1;
        group_block = 3'd3;
        group_row = 9'd401;
        group_source_count_m1 = 2'd1;
        group_source[0] = 4'd3;
        group_source[1] = 4'd4;
        group_source[2] = '0;
        group_source[3] = '0;
        group_negate = '0;
        group_last = 1'b1;
        #1ps;
        if (!protocol_error || group_ready || group_accept)
            $fatal(1, "cross-descriptor duplicate was not quarantined");
        protocol_attacks++;
        @(posedge clk_core);
        reset_and_refill();

        // An open row cannot silently change identity.
        drive_group(402, 0, 2, 1'b0, 1'b0);
        @(negedge clk_core);
        clear_group();
        wait (update_accept_count == group_accept_count);
        @(negedge clk_core);
        group_valid = 1'b1;
        group_block = 3'd3;
        group_row = 9'd403;
        group_source_count_m1 = 2'd1;
        group_source[0] = 4'd2;
        group_source[1] = 4'd3;
        group_source[2] = '0;
        group_source[3] = '0;
        group_negate = '0;
        group_last = 1'b1;
        #1ps;
        if (!protocol_error || group_ready || group_accept)
            $fatal(1, "open-row identity drift was not quarantined");
        protocol_attacks++;
        @(posedge clk_core);
        reset_and_refill();

        // Dirty padding fails closed, then reset must quiesce all visible
        // handshakes and clear the sticky fault without a phantom token.
        @(negedge clk_core);
        group_valid = 1'b1;
        group_block = 3'd3;
        group_row = 9'd404;
        group_source_count_m1 = 2'd0;
        group_source[0] = 4'd2;
        group_source[1] = 4'd7;
        group_source[2] = '0;
        group_source[3] = '0;
        group_negate = '0;
        group_last = 1'b1;
        #1ps;
        if (!protocol_error || group_ready || group_accept)
            $fatal(1, "dirty compact padding was not quarantined");
        protocol_attacks++;
        @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b1;
        #1ps;
        if (protocol_error || weight_fill_accept || group_accept
                || update_valid || done_valid)
            $fatal(1, "reset did not quiesce compact fold island");
        reset_attacks++;
        clear_inputs();
        repeat (2) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        repeat (2) @(posedge clk_core);
        if (protocol_error || busy || expected_q.size() != 0)
            $fatal(1, "final reset did not clear compact fold island");
        if (group_accept_count != update_accept_count)
            $fatal(1, "accepted group/update conservation mismatch");

        $display("PASS M131 compact canonical K4 row fold VCS groups=%0d updates=%0d sources=%0d lanes=%0d done=%0d done_overlap=%0d stalls=%0d long_stall=%0d cross_row_updates=%0d cross_row_ii1=%0d plus512=%0d protocol_attacks=%0d reset_attacks=%0d idle_payload=%0d descriptor_bits=35 producer_implemented=false physical_speedup=false system_speedup=false headline=false",
                 group_accept_count, update_accept_count,
                 source_contribution_count, lane_check_count,
                 done_count, done_overlap_next_group_count,
                 stall_cycle_count, long_stall_cycles,
                 cross_row_update_count, cross_row_ii1_count,
                 plus512_checks, protocol_attacks, reset_attacks,
                 idle_payload_toggle_checks);
        $finish;
    end

    initial begin
        #500000;
        $fatal(1, "M131 directed VCS timeout");
    end
endmodule

`default_nettype wire

