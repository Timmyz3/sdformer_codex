`timescale 1ns/1ps
`default_nettype none

module tb_m125_block_phased_k4_row_fold;
    localparam int SOURCES = 16;
    localparam int LANES = 96;
    localparam int ACC_BITS = 19;
    localparam int ROWS = 66;

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
    int fill_accepts;
    int row_accepts;
    int update_accepts;
    int selected_source_checks;
    int numeric_lane_checks;
    int row_done_count;
    int update_stall_cycles;
    int full_k4_updates;
    int tail_updates;
    int same_row_update_pairs;
    int negated_minus128_contributions;
    int plus512_checks;
    bit previous_update_accept;
    logic [2:0] previous_update_block;
    logic [8:0] previous_update_row;
    bit positive_phase;

    m125_block_phased_k4_row_fold dut (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .weight_fill_valid(weight_fill_valid),
        .weight_fill_ready(weight_fill_ready),
        .weight_fill_block(weight_fill_block),
        .weight_fill_source(weight_fill_source),
        .weight_fill_beat(weight_fill_beat),
        .weight_fill_data(weight_fill_data),
        .weight_fill_accept(weight_fill_accept),
        .row_valid(row_valid),
        .row_ready(row_ready),
        .row_block(row_block),
        .row_offset(row_offset),
        .row_source_mask(row_source_mask),
        .row_negate_mask(row_negate_mask),
        .row_accept(row_accept),
        .update_valid(update_valid),
        .update_ready(update_ready),
        .update_block(update_block),
        .update_row(update_row),
        .update_delta(update_delta),
        .update_selected_mask(update_selected_mask),
        .update_accept(update_accept),
        .row_done(row_done),
        .observed_remaining_mask(observed_remaining_mask),
        .observed_cache_valid(observed_cache_valid),
        .observed_resident_block_valid(observed_resident_block_valid),
        .observed_resident_block(observed_resident_block),
        .protocol_error(protocol_error),
        .busy(busy)
    );

    m125_block_phased_k4_row_fold_assertions checks (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .weight_fill_valid(weight_fill_valid),
        .weight_fill_ready(weight_fill_ready),
        .weight_fill_accept(weight_fill_accept),
        .row_valid(row_valid),
        .row_ready(row_ready),
        .row_accept(row_accept),
        .update_valid(update_valid),
        .update_ready(update_ready),
        .update_accept(update_accept),
        .update_block(update_block),
        .update_row(update_row),
        .update_delta(update_delta),
        .update_selected_mask(update_selected_mask),
        .observed_remaining_mask(observed_remaining_mask),
        .row_done(row_done),
        .protocol_error(protocol_error)
    );

    always #1 clk_core = ~clk_core;

    function automatic integer signed weight_value(
        input int source,
        input int lane
    );
        if (lane == 0 && source < 4)
            weight_value = -128;
        else
            weight_value = ((source * 37 + lane * 29) & 8'hff) - 128;
    endfunction

    function automatic logic [15:0] expected_select4(input logic [15:0] mask);
        logic [15:0] remaining;
        logic found;
        begin
            expected_select4 = '0;
            remaining = mask;
            for (int pick = 0; pick < 4; pick++) begin
                found = 1'b0;
                for (int source = 0; source < SOURCES; source++) begin
                    if (!found && remaining[source]) begin
                        expected_select4[source] = 1'b1;
                        remaining[source] = 1'b0;
                        found = 1'b1;
                    end
                end
            end
        end
    endfunction

    function automatic logic [15:0] row_mask_for(input int row);
        logic [31:0] mixed;
        begin
            if (row == 0)
                row_mask_for = 16'h0000;
            else if (row == 1)
                row_mask_for = 16'hffff;
            else if (row == 2)
                row_mask_for = 16'h000f;
            else begin
                mixed = row * 32'h00009e37;
                row_mask_for = mixed[15:0]
                             ^ (16'ha5a5 >> (row % 5));
                if (row_mask_for == 0)
                    row_mask_for = 16'h0001;
            end
        end
    endfunction

    function automatic logic [15:0] negate_mask_for(
        input int row,
        input logic [15:0] mask
    );
        logic [31:0] mixed;
        begin
            if (row == 1)
                negate_mask_for = 16'h000f;
            else if (row == 2)
                negate_mask_for = 16'h0000;
            else begin
                mixed = row * 32'h00001357;
                negate_mask_for = mask & (mixed[15:0] ^ 16'h5a5a);
            end
        end
    endfunction

    always @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
            update_ready <= 1'b0;
            previous_update_accept <= 1'b0;
            previous_update_block <= '0;
            previous_update_row <= '0;
        end else begin
            cycle_count <= cycle_count + 1;
            update_ready <= ((cycle_count % 7) != 2)
                         && ((cycle_count % 19) != 6);
            if (weight_fill_accept)
                fill_accepts <= fill_accepts + 1;
            if (row_accept)
                row_accepts <= row_accepts + 1;
            if (update_valid && !update_ready)
                update_stall_cycles <= update_stall_cycles + 1;
            if (update_accept) begin
                update_accepts <= update_accepts + 1;
                selected_source_checks <= selected_source_checks
                                        + $countones(update_selected_mask);
                numeric_lane_checks <= numeric_lane_checks + LANES;
                if ($countones(update_selected_mask) == 4)
                    full_k4_updates <= full_k4_updates + 1;
                else
                    tail_updates <= tail_updates + 1;
                if (previous_update_accept
                        && previous_update_block == update_block
                        && previous_update_row == update_row)
                    same_row_update_pairs <= same_row_update_pairs + 1;
                previous_update_block <= update_block;
                previous_update_row <= update_row;
            end
            previous_update_accept <= update_accept;
            if (row_done)
                row_done_count <= row_done_count + 1;
            if (positive_phase && protocol_error)
                $fatal(1, "M125 unexpected protocol_error cycle=%0d", cycle_count);
        end
    end

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            weight_fill_valid = 1'b0;
            row_valid = 1'b0;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
            repeat (2) @(posedge clk_core);
        end
    endtask

    task automatic fill_one_source(input int block, input int source);
        logic [255:0] payload;
        integer signed value;
        begin
            for (int beat = 0; beat < 3; beat++) begin
                payload = '0;
                for (int item = 0; item < 32; item++) begin
                    value = weight_value(source, beat * 32 + item);
                    payload[item * 8 +: 8] = value[7:0];
                end
                @(negedge clk_core);
                weight_fill_valid = 1'b1;
                weight_fill_block = block[2:0];
                weight_fill_source = source[3:0];
                weight_fill_beat = beat[1:0];
                weight_fill_data = payload;
                do @(posedge clk_core); while (!weight_fill_accept);
            end
            @(negedge clk_core);
            weight_fill_valid = 1'b0;
        end
    endtask

    task automatic send_and_check_row(
        input int block,
        input int row,
        input logic [15:0] mask,
        input logic [15:0] negate
    );
        logic [15:0] remaining;
        logic [15:0] expected_selected;
        integer signed expected_delta;
        integer signed expected_total [0:LANES-1];
        integer signed observed_total [0:LANES-1];
        int start_cycle;
        begin
            remaining = mask;
            for (int lane = 0; lane < LANES; lane++) begin
                expected_total[lane] = 0;
                observed_total[lane] = 0;
                for (int source = 0; source < SOURCES; source++) begin
                    if (mask[source]) begin
                        expected_delta = weight_value(source, lane);
                        if (negate[source])
                            expected_delta = -expected_delta;
                        expected_total[lane]
                            = expected_total[lane] + expected_delta;
                    end
                end
            end

            @(negedge clk_core);
            row_valid = 1'b1;
            row_block = block[2:0];
            row_offset = row[8:0];
            row_source_mask = mask;
            row_negate_mask = negate;
            do @(posedge clk_core); while (!row_accept);
            @(negedge clk_core);
            row_valid = 1'b0;
            start_cycle = cycle_count;
            while (!row_done) begin
                @(posedge clk_core);
                if (update_accept) begin
                    expected_selected = expected_select4(remaining);
                    if (update_selected_mask !== expected_selected
                            || update_block !== block[2:0]
                            || update_row !== row[8:0])
                        $fatal(1, "M125 selection/address mismatch row=%0d got=%h expected=%h",
                               row, update_selected_mask,
                               expected_selected);
                    for (int lane = 0; lane < LANES; lane++) begin
                        expected_delta = 0;
                        for (int source = 0; source < SOURCES; source++) begin
                            if (expected_selected[source]) begin
                                expected_delta = expected_delta
                                    + (negate[source]
                                       ? -weight_value(source, lane)
                                       : weight_value(source, lane));
                            end
                        end
                        if ($signed(update_delta[lane * ACC_BITS +: ACC_BITS])
                                !== expected_delta)
                            $fatal(1, "M125 delta mismatch row=%0d lane=%0d got=%0d expected=%0d selected=%h",
                                   row, lane,
                                   $signed(update_delta[lane * ACC_BITS +: ACC_BITS]),
                                   expected_delta, expected_selected);
                        observed_total[lane]
                            = observed_total[lane]
                            + $signed(update_delta[lane * ACC_BITS +: ACC_BITS]);
                    end
                    if (expected_selected[0] && negate[0]
                            && weight_value(0, 0) == -128)
                        negated_minus128_contributions
                            = negated_minus128_contributions + 1;
                    if (row == 1 && expected_selected == 16'h000f) begin
                        if ($signed(update_delta[0 +: ACC_BITS]) != 512)
                            $fatal(1, "M125 generic signed11 +512 boundary failed");
                        plus512_checks = plus512_checks + 1;
                    end
                    remaining = remaining & ~expected_selected;
                end
                if (cycle_count - start_cycle > 200)
                    $fatal(1, "M125 row watchdog row=%0d remaining=%h", row, remaining);
            end
            if (remaining != 0 || observed_remaining_mask != 0)
                $fatal(1, "M125 row did not clear mask row=%0d remaining=%h dut=%h",
                       row, remaining, observed_remaining_mask);
            for (int lane = 0; lane < LANES; lane++) begin
                if (observed_total[lane] != expected_total[lane])
                    $fatal(1, "M125 row total mismatch row=%0d lane=%0d got=%0d expected=%0d",
                           row, lane, observed_total[lane],
                           expected_total[lane]);
            end
            @(posedge clk_core);
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        weight_fill_valid = 1'b0;
        weight_fill_block = '0;
        weight_fill_source = '0;
        weight_fill_beat = '0;
        weight_fill_data = '0;
        row_valid = 1'b0;
        row_block = '0;
        row_offset = '0;
        row_source_mask = '0;
        row_negate_mask = '0;
        update_ready = 1'b0;
        cycle_count = 0;
        fill_accepts = 0;
        row_accepts = 0;
        update_accepts = 0;
        selected_source_checks = 0;
        numeric_lane_checks = 0;
        row_done_count = 0;
        update_stall_cycles = 0;
        full_k4_updates = 0;
        tail_updates = 0;
        same_row_update_pairs = 0;
        negated_minus128_contributions = 0;
        plus512_checks = 0;
        previous_update_accept = 1'b0;
        previous_update_block = '0;
        previous_update_row = '0;
        positive_phase = 1'b1;

        reset_dut();
        for (int source = 0; source < SOURCES; source++)
            fill_one_source(3, source);
        if (!observed_resident_block_valid || observed_resident_block != 3
                || observed_cache_valid != 16'hffff)
            $fatal(1, "M125 cache fill identity mismatch");
        for (int row = 0; row < ROWS; row++) begin
            logic [15:0] mask;
            logic [15:0] negate;
            mask = row_mask_for(row);
            negate = negate_mask_for(row, mask);
            send_and_check_row(3, row, mask, negate);
        end
        repeat (3) @(posedge clk_core);
        if (fill_accepts != 48 || row_accepts != ROWS
                || row_done_count != ROWS || update_accepts == 0
                || selected_source_checks == 0 || numeric_lane_checks == 0
                || full_k4_updates == 0 || tail_updates == 0
                || same_row_update_pairs == 0 || update_stall_cycles == 0
                || negated_minus128_contributions == 0
                || plus512_checks != 1 || protocol_error)
            $fatal(1, "M125 positive conservation/coverage mismatch fills=%0d rows=%0d done=%0d updates=%0d selected=%0d lanes=%0d full=%0d tail=%0d same=%0d stalls=%0d negmin=%0d plus512=%0d",
                   fill_accepts, row_accepts, row_done_count,
                   update_accepts, selected_source_checks,
                   numeric_lane_checks, full_k4_updates, tail_updates,
                   same_row_update_pairs, update_stall_cycles,
                   negated_minus128_contributions, plus512_checks);

        positive_phase = 1'b0;
        reset_dut();
        fill_one_source(4, 0);
        @(negedge clk_core);
        row_valid = 1'b1;
        row_block = 4;
        row_offset = 9;
        row_source_mask = 16'h0002;
        row_negate_mask = 0;
        @(posedge clk_core);
        if (row_ready || row_accept || !protocol_error || update_valid)
            $fatal(1, "M125 invalid cache-source attack not quarantined");
        @(negedge clk_core);
        row_valid = 1'b0;
        repeat (2) @(posedge clk_core);
        if (!protocol_error)
            $fatal(1, "M125 protocol fault not sticky");

        $display("PASS M125 block-phased K4 row fold VCS fills=%0d rows=%0d row_done=%0d updates=%0d selected_sources=%0d numeric_lane_checks=%0d full_k4_updates=%0d tail_updates=%0d same_row_update_pairs=%0d update_stalls=%0d negated_minus128_contributions=%0d plus512_checks=%0d cache_bytes=1536 resident_blocks=1 logical_read_bits_per_update=3072 generic_fold_bits=11 accumulator_delta_bits=19 canonical_select_clear=true fixed8_service_island_projection=3.1725369008459166 projection_only=true m123_integrated=false foundry_weight_macro=false physical_speedup=false system_speedup=false headline=false",
                 fill_accepts, row_accepts, row_done_count,
                 update_accepts, selected_source_checks,
                 numeric_lane_checks, full_k4_updates, tail_updates,
                 same_row_update_pairs, update_stall_cycles,
                 negated_minus128_contributions, plus512_checks);
        $finish;
    end
endmodule

`default_nettype wire
