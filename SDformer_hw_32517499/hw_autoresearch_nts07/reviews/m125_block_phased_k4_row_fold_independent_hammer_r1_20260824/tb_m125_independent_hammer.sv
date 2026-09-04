`timescale 1ns/1ps
`default_nettype none

module tb_m125_independent_hammer;
    localparam int SOURCES = 16;
    localparam int LANES = 96;
    localparam int ACC_BITS = 19;

    logic clk_core, rst_core;
    logic weight_fill_valid, weight_fill_ready;
    logic [2:0] weight_fill_block;
    logic [3:0] weight_fill_source;
    logic [1:0] weight_fill_beat;
    logic [255:0] weight_fill_data;
    logic weight_fill_accept;
    logic row_valid, row_ready;
    logic [2:0] row_block;
    logic [8:0] row_offset;
    logic [15:0] row_source_mask, row_negate_mask;
    logic row_accept;
    logic update_valid, update_ready;
    logic [2:0] update_block;
    logic [8:0] update_row;
    logic [1823:0] update_delta;
    logic [15:0] update_selected_mask;
    logic update_accept, row_done;
    logic [15:0] observed_remaining_mask, observed_cache_valid;
    logic observed_resident_block_valid;
    logic [2:0] observed_resident_block;
    logic protocol_error, busy;

    int cycle_count;
    int rows_checked;
    int updates_checked;
    int sources_checked;
    int lanes_checked;
    int stalls_checked;
    int full_k4_checked;
    int tail_checked;
    int same_row_replays;
    int cache_identity_attacks;
    int block_identity_attacks;
    int fill_sequence_attacks;
    int block_transition_checks;
    int plus512_checks;
    int minus512_checks;
    int reset_fill_phantom;
    int reset_row_phantom;
    int reset_update_phantom;
    int reset_update_visible;
    bit manual_ready;
    bit manual_ready_value;

    m125_block_phased_k4_row_fold dut (.*);

    m125_block_phased_k4_row_fold_assertions checks (
        .clk_core(clk_core), .rst_core(rst_core),
        .weight_fill_valid(weight_fill_valid),
        .weight_fill_ready(weight_fill_ready),
        .weight_fill_accept(weight_fill_accept),
        .row_valid(row_valid), .row_ready(row_ready),
        .row_accept(row_accept), .update_valid(update_valid),
        .update_ready(update_ready), .update_accept(update_accept),
        .update_block(update_block), .update_row(update_row),
        .update_delta(update_delta),
        .update_selected_mask(update_selected_mask),
        .observed_remaining_mask(observed_remaining_mask),
        .row_done(row_done), .protocol_error(protocol_error)
    );

    always #1 clk_core = ~clk_core;

    always @(posedge clk_core) begin
        if (rst_core)
            cycle_count <= 0;
        else
            cycle_count <= cycle_count + 1;
    end

    always_comb begin
        if (manual_ready)
            update_ready = manual_ready_value;
        else
            update_ready = ((cycle_count % 5) != 1)
                         && ((cycle_count % 11) != 7);
    end

    function automatic integer signed model_weight(
        input int block_id,
        input int source,
        input int lane
    );
        int raw;
        begin
            if (lane == 0 && source < 4)
                model_weight = -128;
            else if (lane == 1 && source < 4)
                model_weight = 127;
            else if (lane == 2 && source < 4)
                model_weight = -128;
            else if (lane == 3 && source < 4)
                model_weight = 127;
            else begin
                raw = (block_id * 53 + source * 37 + lane * 29) & 8'hff;
                model_weight = raw - 128;
            end
        end
    endfunction

    function automatic logic [15:0] oracle_lowest4(input logic [15:0] mask);
        logic [15:0] remaining;
        int picked;
        begin
            oracle_lowest4 = 0;
            remaining = mask;
            picked = 0;
            for (int source = 0; source < SOURCES; source++) begin
                if (remaining[source] && picked < 4) begin
                    oracle_lowest4[source] = 1'b1;
                    picked = picked + 1;
                end
            end
        end
    endfunction

    task automatic drive_idle;
        begin
            weight_fill_valid = 1'b0;
            weight_fill_block = 0;
            weight_fill_source = 0;
            weight_fill_beat = 0;
            weight_fill_data = 0;
            row_valid = 1'b0;
            row_block = 0;
            row_offset = 0;
            row_source_mask = 0;
            row_negate_mask = 0;
        end
    endtask

    task automatic clean_reset;
        begin
            @(negedge clk_core);
            drive_idle();
            manual_ready = 1'b1;
            manual_ready_value = 1'b0;
            rst_core = 1'b1;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
            repeat (2) @(posedge clk_core);
            manual_ready = 1'b0;
            if (protocol_error || observed_resident_block_valid
                    || observed_cache_valid != 0 || observed_remaining_mask != 0)
                $fatal(1, "M125 reset did not clear state");
        end
    endtask

    task automatic fill_source(input int block_id, input int source);
        logic [255:0] payload;
        int value;
        begin
            for (int beat = 0; beat < 3; beat++) begin
                payload = 0;
                for (int item = 0; item < 32; item++) begin
                    value = model_weight(block_id, source, beat * 32 + item);
                    payload[item * 8 +: 8] = value[7:0];
                end
                @(negedge clk_core);
                weight_fill_valid = 1'b1;
                weight_fill_block = block_id[2:0];
                weight_fill_source = source[3:0];
                weight_fill_beat = beat[1:0];
                weight_fill_data = payload;
                do @(posedge clk_core); while (!weight_fill_accept);
            end
            @(negedge clk_core);
            weight_fill_valid = 1'b0;
        end
    endtask

    task automatic fill_block(input int block_id);
        begin
            for (int source = 0; source < SOURCES; source++)
                fill_source(block_id, source);
            if (!observed_resident_block_valid
                    || observed_resident_block !== block_id[2:0]
                    || observed_cache_valid !== 16'hffff)
                $fatal(1, "M125 full cache identity mismatch block=%0d valid=%h",
                       block_id, observed_cache_valid);
        end
    endtask

    task automatic check_row(
        input int block_id,
        input int row_id,
        input logic [15:0] initial_mask,
        input logic [15:0] negate_mask,
        input int forced_stall_cycles
    );
        logic [15:0] remaining;
        logic [15:0] consumed;
        logic [15:0] expected_selected;
        logic [1823:0] stalled_delta;
        logic [15:0] stalled_selected;
        logic [2:0] stalled_block;
        logic [8:0] stalled_row;
        int expected_value;
        int update_count;
        int stall_left;
        int watchdog;
        bit stall_snapshot_valid;
        begin
            remaining = initial_mask;
            consumed = 0;
            update_count = 0;
            stall_left = forced_stall_cycles;
            watchdog = 0;
            stall_snapshot_valid = 0;

            @(negedge clk_core);
            row_valid = 1'b1;
            row_block = block_id[2:0];
            row_offset = row_id[8:0];
            row_source_mask = initial_mask;
            row_negate_mask = negate_mask;
            do @(posedge clk_core); while (!row_accept);
            @(negedge clk_core);
            row_valid = 1'b0;
            if (forced_stall_cycles > 0) begin
                manual_ready = 1'b1;
                manual_ready_value = 1'b0;
            end

            while (!row_done) begin
                @(posedge clk_core);
                watchdog = watchdog + 1;
                if (watchdog > 200)
                    $fatal(1, "M125 independent row watchdog row=%0d", row_id);
                if (update_valid) begin
                    expected_selected = oracle_lowest4(remaining);
                    if (update_selected_mask !== expected_selected)
                        $fatal(1, "M125 canonical lowest4 mismatch row=%0d got=%h expected=%h remaining=%h",
                               row_id, update_selected_mask, expected_selected, remaining);
                    if ((update_selected_mask & consumed) != 0)
                        $fatal(1, "M125 duplicate source row=%0d selected=%h consumed=%h",
                               row_id, update_selected_mask, consumed);
                    if (update_block !== block_id[2:0]
                            || update_row !== row_id[8:0])
                        $fatal(1, "M125 update identity mismatch");
                    for (int lane = 0; lane < LANES; lane++) begin
                        expected_value = 0;
                        for (int source = 0; source < SOURCES; source++) begin
                            if (expected_selected[source]) begin
                                if (negate_mask[source])
                                    expected_value = expected_value
                                        - model_weight(block_id, source, lane);
                                else
                                    expected_value = expected_value
                                        + model_weight(block_id, source, lane);
                            end
                        end
                        if ($signed(update_delta[lane * ACC_BITS +: ACC_BITS])
                                !== expected_value)
                            $fatal(1, "M125 numeric mismatch row=%0d lane=%0d got=%0d expected=%0d sel=%h neg=%h",
                                   row_id, lane,
                                   $signed(update_delta[lane * ACC_BITS +: ACC_BITS]),
                                   expected_value, expected_selected, negate_mask);
                    end
                    if (!update_ready) begin
                        stalls_checked = stalls_checked + 1;
                        if (!stall_snapshot_valid) begin
                            stalled_delta = update_delta;
                            stalled_selected = update_selected_mask;
                            stalled_block = update_block;
                            stalled_row = update_row;
                            stall_snapshot_valid = 1'b1;
                        end else if (update_delta !== stalled_delta
                                || update_selected_mask !== stalled_selected
                                || update_block !== stalled_block
                                || update_row !== stalled_row)
                            $fatal(1, "M125 output changed under stall row=%0d", row_id);
                    end
                end
                if (manual_ready && stall_left > 0) begin
                    stall_left = stall_left - 1;
                    if (stall_left == 0)
                        manual_ready <= 1'b0;
                end
                if (update_accept) begin
                    if (stall_snapshot_valid
                            && (update_delta !== stalled_delta
                                || update_selected_mask !== stalled_selected
                                || update_block !== stalled_block
                                || update_row !== stalled_row))
                        $fatal(1, "M125 stalled transaction changed at release");
                    consumed = consumed | update_selected_mask;
                    remaining = remaining & ~update_selected_mask;
                    sources_checked = sources_checked
                                    + $countones(update_selected_mask);
                    lanes_checked = lanes_checked + LANES;
                    updates_checked = updates_checked + 1;
                    update_count = update_count + 1;
                    if ($countones(update_selected_mask) == 4)
                        full_k4_checked = full_k4_checked + 1;
                    else
                        tail_checked = tail_checked + 1;
                    if (row_id == 100 && update_selected_mask == 16'h000f) begin
                        if ($signed(update_delta[0 +: ACC_BITS]) !== 512)
                            $fatal(1, "M125 +512 signed11 boundary mismatch");
                        plus512_checks = plus512_checks + 1;
                    end
                    if (row_id == 102 && update_selected_mask == 16'h000f) begin
                        if ($signed(update_delta[2 * ACC_BITS +: ACC_BITS]) !== -512)
                            $fatal(1, "M125 -512 signed11 boundary mismatch");
                        minus512_checks = minus512_checks + 1;
                    end
                    stall_snapshot_valid = 1'b0;
                end
            end
            manual_ready = 1'b0;
            if (consumed !== initial_mask || remaining != 0
                    || observed_remaining_mask != 0
                    || update_count != (($countones(initial_mask) + 3) / 4))
                $fatal(1, "M125 source conservation mismatch row=%0d initial=%h consumed=%h remaining=%h updates=%0d",
                       row_id, initial_mask, consumed, remaining, update_count);
            rows_checked = rows_checked + 1;
            @(posedge clk_core);
        end
    endtask

    task automatic expect_fault_then_reset(input int kind);
        begin
            @(negedge clk_core);
            if (kind == 0) begin
                row_valid = 1'b1;
                row_block = observed_resident_block;
                row_offset = 9;
                row_source_mask = 16'h8000;
                row_negate_mask = 0;
                cache_identity_attacks = cache_identity_attacks + 1;
            end else if (kind == 1) begin
                row_valid = 1'b1;
                row_block = observed_resident_block + 1'b1;
                row_offset = 10;
                row_source_mask = 16'h0001;
                row_negate_mask = 0;
                block_identity_attacks = block_identity_attacks + 1;
            end else begin
                weight_fill_valid = 1'b1;
                weight_fill_block = observed_resident_block;
                weight_fill_source = 0;
                weight_fill_beat = 2;
                weight_fill_data = 0;
                fill_sequence_attacks = fill_sequence_attacks + 1;
            end
            @(posedge clk_core);
            if (!protocol_error || row_accept || weight_fill_accept || update_valid)
                $fatal(1, "M125 fault kind=%0d not fail-closed", kind);
            @(negedge clk_core);
            drive_idle();
            repeat (2) @(posedge clk_core);
            if (!protocol_error)
                $fatal(1, "M125 fault kind=%0d not sticky", kind);
            clean_reset();
        end
    endtask

    task automatic probe_reset_phantoms;
        begin
            clean_reset();
            @(negedge clk_core);
            rst_core = 1'b1;
            weight_fill_valid = 1'b1;
            weight_fill_block = 2;
            weight_fill_source = 1;
            weight_fill_beat = 0;
            weight_fill_data = 'h1234;
            @(posedge clk_core);
            if (weight_fill_accept)
                reset_fill_phantom = reset_fill_phantom + 1;
            @(negedge clk_core);
            drive_idle();
            rst_core = 1'b0;
            repeat (2) @(posedge clk_core);
            if (observed_cache_valid != 0 || observed_resident_block_valid)
                $fatal(1, "M125 reset fill phantom changed architectural state");

            fill_block(5);
            @(negedge clk_core);
            row_valid = 1'b1;
            row_block = 5;
            row_offset = 222;
            row_source_mask = 16'h000f;
            row_negate_mask = 0;
            rst_core = 1'b1;
            @(posedge clk_core);
            if (row_accept)
                reset_row_phantom = reset_row_phantom + 1;
            @(negedge clk_core);
            drive_idle();
            rst_core = 1'b0;
            repeat (2) @(posedge clk_core);
            if (observed_remaining_mask != 0 || row_done)
                $fatal(1, "M125 reset row phantom changed architectural state");

            fill_block(6);
            manual_ready = 1'b1;
            manual_ready_value = 1'b0;
            @(negedge clk_core);
            row_valid = 1'b1;
            row_block = 6;
            row_offset = 223;
            row_source_mask = 16'h00f0;
            row_negate_mask = 16'h0050;
            do @(posedge clk_core); while (!row_accept);
            @(negedge clk_core);
            row_valid = 1'b0;
            if (!update_valid)
                @(posedge clk_core);
            @(negedge clk_core);
            manual_ready_value = 1'b1;
            rst_core = 1'b1;
            if (update_valid)
                reset_update_visible = reset_update_visible + 1;
            @(posedge clk_core);
            if (update_accept)
                reset_update_phantom = reset_update_phantom + 1;
            @(negedge clk_core);
            drive_idle();
            manual_ready_value = 1'b0;
            rst_core = 1'b0;
            repeat (2) @(posedge clk_core);
            manual_ready = 1'b0;
            if (observed_remaining_mask != 0 || row_done)
                $fatal(1, "M125 reset update phantom changed architectural state");
        end
    endtask

    initial begin
        clk_core = 0;
        rst_core = 1;
        manual_ready = 1;
        manual_ready_value = 0;
        cycle_count = 0;
        rows_checked = 0;
        updates_checked = 0;
        sources_checked = 0;
        lanes_checked = 0;
        stalls_checked = 0;
        full_k4_checked = 0;
        tail_checked = 0;
        same_row_replays = 0;
        cache_identity_attacks = 0;
        block_identity_attacks = 0;
        fill_sequence_attacks = 0;
        block_transition_checks = 0;
        plus512_checks = 0;
        minus512_checks = 0;
        reset_fill_phantom = 0;
        reset_row_phantom = 0;
        reset_update_phantom = 0;
        reset_update_visible = 0;
        drive_idle();

        clean_reset();
        fill_block(3);
        check_row(3, 100, 16'h000f, 16'h000f, 5);
        check_row(3, 101, 16'h9185, 16'h8104, 3);
        check_row(3, 222, 16'hffff, 16'ha55a, 7);
        check_row(3, 222, 16'h8421, 16'h8020, 4);
        same_row_replays = same_row_replays + 1;
        check_row(3, 511, 16'h8000, 16'h8000, 2);
        check_row(3, 0, 16'h0000, 16'h0000, 0);
        check_row(3, 102, 16'h000f, 16'h0000, 1);
        check_row(3, 103, 16'h0007, 16'h0005, 1);
        check_row(3, 104, 16'h0003, 16'h0002, 1);

        fill_source(4, 0);
        if (!observed_resident_block_valid || observed_resident_block != 4
                || observed_cache_valid != 16'h0001)
            $fatal(1, "M125 block transition did not invalidate old cache valid=%h block=%0d",
                   observed_cache_valid, observed_resident_block);
        block_transition_checks = block_transition_checks + 1;
        expect_fault_then_reset(0);
        fill_source(4, 0);
        expect_fault_then_reset(1);
        fill_source(4, 0);
        expect_fault_then_reset(2);

        probe_reset_phantoms();

        if (rows_checked != 9 || updates_checked != 12
                || sources_checked != 40 || lanes_checked != 1152
                || stalls_checked == 0 || full_k4_checked != 8
                || tail_checked != 4 || same_row_replays != 1
                || cache_identity_attacks != 1
                || block_identity_attacks != 1
                || fill_sequence_attacks != 1
                || block_transition_checks != 1
                || plus512_checks != 1 || minus512_checks != 1)
            $fatal(1, "M125 independent aggregate mismatch rows=%0d updates=%0d sources=%0d lanes=%0d stalls=%0d full=%0d tail=%0d",
                   rows_checked, updates_checked, sources_checked,
                   lanes_checked, stalls_checked, full_k4_checked, tail_checked);
        if (reset_fill_phantom != 1 || reset_row_phantom != 1
                || reset_update_visible != 1 || reset_update_phantom != 1)
            $fatal(1, "M125 reset finding unexpectedly absent fill=%0d row=%0d visible=%0d update=%0d",
                   reset_fill_phantom, reset_row_phantom,
                   reset_update_visible, reset_update_phantom);

        $display("PASS M125 independent hammer rows=%0d updates=%0d sources=%0d lanes=%0d stalls=%0d full_k4=%0d tails=%0d same_row_replays=%0d cache_identity_attacks=%0d block_identity_attacks=%0d fill_sequence_attacks=%0d block_transition_checks=%0d plus512=%0d minus512=%0d reset_fill_phantom=%0d reset_row_phantom=%0d reset_update_visible=%0d reset_update_phantom=%0d reset_quiescence=false canonical_lowest4=true source_conservation=true cache_bytes=1536_logical logical_read_bits_per_update=3072 projection_3p1725=cycle_only physical_speedup=false system_speedup=false",
                 rows_checked, updates_checked, sources_checked,
                 lanes_checked, stalls_checked, full_k4_checked, tail_checked,
                 same_row_replays, cache_identity_attacks,
                 block_identity_attacks, fill_sequence_attacks,
                 block_transition_checks,
                 plus512_checks, minus512_checks, reset_fill_phantom,
                 reset_row_phantom, reset_update_visible,
                 reset_update_phantom);
        $finish;
    end
endmodule

`default_nettype wire
