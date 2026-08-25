`timescale 1ns/1ps
`default_nettype none

module tb_m128_descriptor_streamed_k4_row_fold;
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
    logic [3:0] group_source_valid;
    logic [3:0] group_source [0:3];
    logic [3:0] group_negate;
    logic [15:0] group_selected_mask;
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
    logic row_done;
    logic [15:0] observed_cache_valid;
    logic observed_resident_block_valid;
    logic [2:0] observed_resident_block;
    logic observed_pair_pipeline_valid;
    logic protocol_error;
    logic busy;

    logic signed [7:0] weight_model [0:SOURCES-1][0:LANES-1];
    bit force_update_ready;
    bit random_stall_enable;
    bit cross_row_phase;
    int unsigned cycle_count;
    int unsigned group_accept_count;
    int unsigned update_accept_count;
    int unsigned source_contribution_count;
    int unsigned lane_check_count;
    int unsigned row_done_count;
    int unsigned stall_cycle_count;
    int unsigned cross_row_update_count;
    int unsigned cross_row_ii1_count;
    int unsigned last_cross_row_update_cycle;
    int unsigned plus512_checks;
    int unsigned protocol_attacks;
    int unsigned reset_attacks;

    typedef struct packed {
        logic [2:0] block_id;
        logic [8:0] row_id;
        logic [15:0] selected_mask;
        logic last;
        logic [UPDATE_BITS-1:0] delta;
    } expected_update_t;
    expected_update_t expected_q[$];

    m128_descriptor_streamed_k4_row_fold dut (.*);

    m128_descriptor_streamed_k4_row_fold_assertions sva (
        .clk_core,
        .rst_core,
        .weight_fill_accept,
        .weight_fill_ready,
        .group_valid,
        .group_ready,
        .group_accept,
        .group_source_valid,
        .update_valid,
        .update_ready,
        .update_accept,
        .update_block,
        .update_row,
        .update_delta,
        .update_selected_mask,
        .update_last,
        .row_done,
        .protocol_error
    );

    initial clk_core = 1'b0;
    always #1.5 clk_core = ~clk_core;

    function automatic logic [UPDATE_BITS-1:0]
        expected_current_group_delta();
        logic [UPDATE_BITS-1:0] result;
        integer signed lane_sum;
        integer signed weight_value;
        result = '0;
        for (int lane = 0; lane < LANES; lane++) begin
            lane_sum = 0;
            for (int pick = 0; pick < 4; pick++) begin
                if (group_source_valid[pick]) begin
                    weight_value = $signed(
                        weight_model[group_source[pick]][lane]);
                    if (group_negate[pick])
                        lane_sum = lane_sum - weight_value;
                    else
                        lane_sum = lane_sum + weight_value;
                end
            end
            result[lane * ACC_BITS +: ACC_BITS] = lane_sum[ACC_BITS-1:0];
        end
        return result;
    endfunction

    function automatic int count_valid_sources(input logic [3:0] valid);
        int count;
        count = 0;
        for (int pick = 0; pick < 4; pick++)
            count += valid[pick];
        return count;
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
        group_source_valid = '0;
        group_negate = '0;
        group_selected_mask = '0;
        group_last = 1'b0;
        for (int pick = 0; pick < 4; pick++)
            group_source[pick] = '0;
    endtask

    task automatic apply_reset(input int cycles);
        @(negedge clk_core);
        rst_core = 1'b1;
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

    task automatic drive_group(
        input int descriptor_id,
        input int source_count,
        input bit last,
        input bit force_plus512
    );
        int source_id;
        @(negedge clk_core);
        group_valid = 1'b1;
        group_block = 3'd3;
        group_row = descriptor_id[8:0];
        group_source_valid = '0;
        group_negate = '0;
        group_selected_mask = '0;
        group_last = last;
        for (int pick = 0; pick < 4; pick++) begin
            group_source[pick] = '0;
            if (pick < source_count) begin
                source_id = force_plus512 ? pick
                    : ((descriptor_id * 3 + pick * 5) % SOURCES);
                group_source_valid[pick] = 1'b1;
                group_source[pick] = source_id[3:0];
                group_selected_mask[source_id] = 1'b1;
                group_negate[pick] = force_plus512
                    ? 1'b1 : ((descriptor_id + pick) % 3 == 0);
            end
        end
        do begin
            @(posedge clk_core);
            if (protocol_error)
                $fatal(1, "unexpected protocol error during descriptor %0d",
                       descriptor_id);
        end while (!group_accept);
    endtask

    always @(negedge clk_core) begin
        if (rst_core)
            update_ready <= 1'b0;
        else if (force_update_ready)
            update_ready <= 1'b1;
        else if (random_stall_enable)
            update_ready <= ($urandom_range(0, 4) != 0);
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
                $fatal(1, "descriptor update mismatch at cycle %0d",
                       cycle_count);
            if (!update_ready)
                stall_cycle_count++;
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
                        $fatal(1, "cross-row descriptor II drift: %0d",
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
            accepted.selected_mask = group_selected_mask;
            accepted.last = group_last;
            accepted.delta = expected_current_group_delta();
            expected_q.push_back(accepted);
            group_accept_count++;
            source_contribution_count
                += count_valid_sources(group_source_valid);
        end

        if (!rst_core && row_done)
            row_done_count++;
    end

    initial begin : test_sequence
        rst_core = 1'b1;
        update_ready = 1'b0;
        force_update_ready = 1'b0;
        random_stall_enable = 1'b0;
        cross_row_phase = 1'b0;
        cycle_count = 0;
        group_accept_count = 0;
        update_accept_count = 0;
        source_contribution_count = 0;
        lane_check_count = 0;
        row_done_count = 0;
        stall_cycle_count = 0;
        cross_row_update_count = 0;
        cross_row_ii1_count = 0;
        last_cross_row_update_cycle = 0;
        plus512_checks = 0;
        protocol_attacks = 0;
        reset_attacks = 0;
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
        for (int source = 0; source < SOURCES; source++)
            fill_one_source(source);

        if (observed_cache_valid !== 16'hffff
                || !observed_resident_block_valid
                || observed_resident_block != 3)
            $fatal(1, "cache identity did not close after fill");

        // One boundary group followed by 64 single-group rows.  Once the
        // first output appears, every accepted update must be one cycle apart.
        cross_row_phase = 1'b1;
        drive_group(0, 4, 1'b1, 1'b1);
        for (int descriptor = 1; descriptor < 64; descriptor++)
            drive_group(descriptor, 4, 1'b1, 1'b0);
        @(negedge clk_core);
        group_valid = 1'b0;
        wait (update_accept_count == group_accept_count);
        @(posedge clk_core);
        cross_row_phase = 1'b0;
        if (cross_row_update_count != 64 || cross_row_ii1_count != 63)
            $fatal(1, "cross-row II=1 coverage mismatch %0d/%0d",
                   cross_row_update_count, cross_row_ii1_count);

        // Mixed K1-K4 descriptors with randomized output stalls.
        force_update_ready = 1'b0;
        random_stall_enable = 1'b1;
        for (int descriptor = 64; descriptor < 384; descriptor++)
            drive_group(descriptor, 1 + (descriptor % 4),
                        (descriptor % 3) == 0, 1'b0);
        @(negedge clk_core);
        group_valid = 1'b0;
        wait (update_accept_count == group_accept_count);
        repeat (3) @(posedge clk_core);
        if (expected_q.size() != 0 || update_valid)
            $fatal(1, "descriptor pipe did not drain");
        if (plus512_checks == 0)
            $fatal(1, "+512 signed boundary was not observed");

        // Duplicate-source descriptor must fail closed without acceptance.
        random_stall_enable = 1'b0;
        force_update_ready = 1'b1;
        @(negedge clk_core);
        group_valid = 1'b1;
        group_block = 3'd3;
        group_row = 9'd7;
        group_source_valid = 4'b0011;
        group_source[0] = 4'd3;
        group_source[1] = 4'd3;
        group_source[2] = '0;
        group_source[3] = '0;
        group_negate = '0;
        group_selected_mask = 16'h0008;
        group_last = 1'b1;
        #1ps;
        if (!protocol_error || group_ready || group_accept || update_valid)
            $fatal(1, "duplicate descriptor was not quarantined");
        protocol_attacks++;
        @(posedge clk_core);

        // Reset while the illegal request is held must quiesce every visible
        // handshake and clear the sticky quarantine without a phantom accept.
        @(negedge clk_core);
        rst_core = 1'b1;
        #1ps;
        if (protocol_error || weight_fill_accept || group_accept
                || update_valid || row_done)
            $fatal(1, "reset did not quiesce descriptor island");
        reset_attacks++;
        repeat (2) @(posedge clk_core);
        @(negedge clk_core);
        group_valid = 1'b0;
        rst_core = 1'b0;
        repeat (2) @(posedge clk_core);
        if (protocol_error || busy)
            $fatal(1, "reset did not clear descriptor island");

        $display("PASS M128 descriptor-streamed K4 row fold VCS groups=%0d updates=%0d sources=%0d lanes=%0d rows_done=%0d stalls=%0d cross_row_updates=%0d cross_row_ii1=%0d plus512=%0d protocol_attacks=%0d reset_attacks=%0d cache_bytes=1536 descriptor_predecode_external=true physical_speedup=false system_speedup=false headline=false",
                 group_accept_count, update_accept_count,
                 source_contribution_count, lane_check_count,
                 row_done_count, stall_cycle_count,
                 cross_row_update_count, cross_row_ii1_count,
                 plus512_checks, protocol_attacks, reset_attacks);
        $finish;
    end

    initial begin
        #300000;
        $fatal(1, "M128 directed VCS timeout");
    end
endmodule

`default_nettype wire
