`timescale 1ns/1ps
`default_nettype none

module tb_m127_block_phased_pipelined_k4_row_fold;
    localparam int SOURCES = 16;
    localparam int LANES = 96;
    localparam int ACC_BITS = 19;

    logic clk_core, rst_core;
    logic weight_fill_valid;
    logic [2:0] weight_fill_block;
    logic [3:0] weight_fill_source;
    logic [1:0] weight_fill_beat;
    logic [255:0] weight_fill_data;
    logic row_valid;
    logic [2:0] row_block;
    logic [8:0] row_offset;
    logic [15:0] row_source_mask, row_negate_mask;
    logic update_ready;

    logic d_fill_ready, d_fill_accept, d_row_ready, d_row_accept;
    logic d_update_valid, d_update_accept, d_row_done;
    logic [2:0] d_update_block;
    logic [8:0] d_update_row;
    logic [1823:0] d_update_delta;
    logic [15:0] d_update_selected_mask, d_remaining, d_cache_valid;
    logic d_resident_valid, d_pipe_valid, d_protocol_error, d_busy;
    logic [2:0] d_resident_block;

    logic r_fill_ready, r_fill_accept, r_row_ready, r_row_accept;
    logic r_update_valid, r_update_accept, r_row_done;
    logic [2:0] r_update_block;
    logic [8:0] r_update_row;
    logic [1823:0] r_update_delta;
    logic [15:0] r_update_selected_mask, r_remaining, r_cache_valid;
    logic r_resident_valid, r_protocol_error, r_busy;
    logic [2:0] r_resident_block;

    int cycle_count;
    int fill_accepts;
    int row_accepts;
    int row_dones;
    int update_accepts;
    int selected_sources;
    int numeric_lane_checks;
    int full_k4_updates;
    int tail_updates;
    int ii1_update_pairs;
    int update_stall_cycles;
    int plus512_checks;
    int cycle_exact_checks;
    int reset_attacks;
    int protocol_attacks;
    bit previous_update_accept;
    bit positive_phase;
    bit force_update_ready;

    m127_block_phased_pipelined_k4_row_fold dut (
        .clk_core(clk_core), .rst_core(rst_core),
        .weight_fill_valid(weight_fill_valid),
        .weight_fill_ready(d_fill_ready),
        .weight_fill_block(weight_fill_block),
        .weight_fill_source(weight_fill_source),
        .weight_fill_beat(weight_fill_beat),
        .weight_fill_data(weight_fill_data),
        .weight_fill_accept(d_fill_accept),
        .row_valid(row_valid), .row_ready(d_row_ready),
        .row_block(row_block), .row_offset(row_offset),
        .row_source_mask(row_source_mask),
        .row_negate_mask(row_negate_mask),
        .row_accept(d_row_accept),
        .update_valid(d_update_valid), .update_ready(update_ready),
        .update_block(d_update_block), .update_row(d_update_row),
        .update_delta(d_update_delta),
        .update_selected_mask(d_update_selected_mask),
        .update_accept(d_update_accept), .row_done(d_row_done),
        .observed_remaining_mask(d_remaining),
        .observed_cache_valid(d_cache_valid),
        .observed_resident_block_valid(d_resident_valid),
        .observed_resident_block(d_resident_block),
        .observed_pair_pipeline_valid(d_pipe_valid),
        .protocol_error(d_protocol_error), .busy(d_busy)
    );

    m125_block_phased_k4_row_fold reference (
        .clk_core(clk_core), .rst_core(rst_core),
        .weight_fill_valid(weight_fill_valid),
        .weight_fill_ready(r_fill_ready),
        .weight_fill_block(weight_fill_block),
        .weight_fill_source(weight_fill_source),
        .weight_fill_beat(weight_fill_beat),
        .weight_fill_data(weight_fill_data),
        .weight_fill_accept(r_fill_accept),
        .row_valid(row_valid), .row_ready(r_row_ready),
        .row_block(row_block), .row_offset(row_offset),
        .row_source_mask(row_source_mask),
        .row_negate_mask(row_negate_mask),
        .row_accept(r_row_accept),
        .update_valid(r_update_valid), .update_ready(update_ready),
        .update_block(r_update_block), .update_row(r_update_row),
        .update_delta(r_update_delta),
        .update_selected_mask(r_update_selected_mask),
        .update_accept(r_update_accept), .row_done(r_row_done),
        .observed_remaining_mask(r_remaining),
        .observed_cache_valid(r_cache_valid),
        .observed_resident_block_valid(r_resident_valid),
        .observed_resident_block(r_resident_block),
        .protocol_error(r_protocol_error), .busy(r_busy)
    );

    m127_block_phased_pipelined_k4_row_fold_assertions checks (
        .clk_core(clk_core), .rst_core(rst_core),
        .weight_fill_valid(weight_fill_valid),
        .weight_fill_ready(d_fill_ready),
        .weight_fill_accept(d_fill_accept),
        .row_valid(row_valid), .row_ready(d_row_ready),
        .row_accept(d_row_accept),
        .update_valid(d_update_valid), .update_ready(update_ready),
        .update_accept(d_update_accept),
        .update_block(d_update_block), .update_row(d_update_row),
        .update_delta(d_update_delta),
        .update_selected_mask(d_update_selected_mask),
        .observed_remaining_mask(d_remaining),
        .observed_pair_pipeline_valid(d_pipe_valid),
        .row_done(d_row_done), .protocol_error(d_protocol_error)
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
                row_mask_for = 16'h000f;
            else if (row == 2)
                row_mask_for = 16'hffff;
            else if (row == 3)
                row_mask_for = 16'h0001;
            else if (row == 4)
                row_mask_for = 16'h0003;
            else if (row == 5)
                row_mask_for = 16'h0007;
            else begin
                mixed = row * 32'h9e3779b9;
                row_mask_for = mixed[15:0] ^ mixed[31:16];
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
            else begin
                mixed = row * 32'h27d4eb2d;
                negate_mask_for = mask & (mixed[15:0] ^ 16'h5a5a);
            end
        end
    endfunction

    always @(posedge clk_core) begin
        if (rst_core) begin
            update_ready <= 1'b0;
            previous_update_accept <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;
            update_ready <= force_update_ready
                         || (((cycle_count % 7) != 2)
                             && ((cycle_count % 19) != 6));
            if (positive_phase) begin
                cycle_exact_checks <= cycle_exact_checks + 1;
                if ({d_fill_ready, d_fill_accept, d_row_ready, d_row_accept,
                     d_update_valid, d_update_accept,
                     d_row_done, d_remaining, d_cache_valid,
                     d_resident_valid, d_resident_block,
                     d_protocol_error, d_busy}
                    !==
                    {r_fill_ready, r_fill_accept, r_row_ready, r_row_accept,
                     r_update_valid, r_update_accept,
                     r_row_done, r_remaining, r_cache_valid,
                     r_resident_valid, r_resident_block,
                     r_protocol_error, r_busy})
                    begin
                        $display("M127/M125 status d fill=%b/%b row=%b/%b upd=%b/%b done=%b rem=%h busy=%b err=%b",
                                 d_fill_ready, d_fill_accept,
                                 d_row_ready, d_row_accept,
                                 d_update_valid, d_update_accept,
                                 d_row_done, d_remaining, d_busy,
                                 d_protocol_error);
                        $display("M127/M125 status r fill=%b/%b row=%b/%b upd=%b/%b done=%b rem=%h busy=%b err=%b",
                                 r_fill_ready, r_fill_accept,
                                 r_row_ready, r_row_accept,
                                 r_update_valid, r_update_accept,
                                 r_row_done, r_remaining, r_busy,
                                 r_protocol_error);
                        $display("M127/M125 update d block=%0d row=%0d sel=%h lane0=%0d; r block=%0d row=%0d sel=%h lane0=%0d",
                                 d_update_block, d_update_row,
                                 d_update_selected_mask,
                                 $signed(d_update_delta[0 +: ACC_BITS]),
                                 r_update_block, r_update_row,
                                 r_update_selected_mask,
                                 $signed(r_update_delta[0 +: ACC_BITS]));
                        $fatal(1, "M127/M125 cycle-exact mismatch cycle=%0d",
                               cycle_count);
                    end
                if (d_update_valid
                        && {d_update_block, d_update_row,
                            d_update_delta, d_update_selected_mask}
                           !==
                           {r_update_block, r_update_row,
                            r_update_delta, r_update_selected_mask})
                    $fatal(1, "M127/M125 valid-update mismatch cycle=%0d",
                           cycle_count);
                if (d_protocol_error)
                    $fatal(1, "M127 unexpected positive protocol_error");
            end
            if (d_fill_accept)
                fill_accepts <= fill_accepts + 1;
            if (d_row_accept)
                row_accepts <= row_accepts + 1;
            if (d_row_done)
                row_dones <= row_dones + 1;
            if (d_update_valid && !update_ready)
                update_stall_cycles <= update_stall_cycles + 1;
            if (d_update_accept) begin
                update_accepts <= update_accepts + 1;
                selected_sources <= selected_sources
                                    + $countones(d_update_selected_mask);
                numeric_lane_checks <= numeric_lane_checks + LANES;
                if ($countones(d_update_selected_mask) == 4)
                    full_k4_updates <= full_k4_updates + 1;
                else
                    tail_updates <= tail_updates + 1;
                if (previous_update_accept)
                    ii1_update_pairs <= ii1_update_pairs + 1;
            end
            previous_update_accept <= d_update_accept;
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
                do @(posedge clk_core); while (!d_fill_accept);
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
        logic [15:0] selected;
        integer signed expected_delta;
        int watchdog;
        begin
            @(negedge clk_core);
            row_valid = 1'b1;
            row_block = block[2:0];
            row_offset = row[8:0];
            row_source_mask = mask;
            row_negate_mask = negate;
            do @(posedge clk_core); while (!d_row_accept);
            @(negedge clk_core);
            row_valid = 1'b0;
            remaining = mask;
            watchdog = 0;
            while (!d_row_done) begin
                @(posedge clk_core);
                watchdog = watchdog + 1;
                if (d_update_accept) begin
                    selected = expected_select4(remaining);
                    if (d_update_selected_mask !== selected
                            || d_update_block !== block[2:0]
                            || d_update_row !== row[8:0])
                        $fatal(1, "M127 identity mismatch block=%0d row=%0d",
                               block, row);
                    for (int lane = 0; lane < LANES; lane++) begin
                        expected_delta = 0;
                        for (int source = 0; source < SOURCES; source++)
                            if (selected[source])
                                expected_delta = expected_delta
                                    + (negate[source]
                                       ? -weight_value(source, lane)
                                       : weight_value(source, lane));
                        if ($signed(d_update_delta[
                                lane * ACC_BITS +: ACC_BITS])
                                !== expected_delta)
                            $fatal(1, "M127 numeric mismatch row=%0d lane=%0d",
                                   row, lane);
                    end
                    if (row == 1 && selected == 16'h000f) begin
                        if ($signed(d_update_delta[0 +: ACC_BITS]) != 512)
                            $fatal(1, "M127 +512 boundary mismatch");
                        plus512_checks = plus512_checks + 1;
                    end
                    remaining = remaining & ~selected;
                end
                if (watchdog > 200)
                    $fatal(1, "M127 row watchdog row=%0d remaining=%h",
                           row, remaining);
            end
            if (remaining != 0 || d_remaining != 0)
                $fatal(1, "M127 source conservation mismatch row=%0d",
                       row);
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
        row_dones = 0;
        update_accepts = 0;
        selected_sources = 0;
        numeric_lane_checks = 0;
        full_k4_updates = 0;
        tail_updates = 0;
        ii1_update_pairs = 0;
        update_stall_cycles = 0;
        plus512_checks = 0;
        cycle_exact_checks = 0;
        reset_attacks = 0;
        protocol_attacks = 0;
        previous_update_accept = 1'b0;
        positive_phase = 1'b1;
        force_update_ready = 1'b0;

        reset_dut();
        for (int block = 3; block <= 4; block++) begin
            for (int source = 0; source < SOURCES; source++)
                fill_one_source(block, source);
            for (int row = 0; row < 40; row++) begin
                logic [15:0] mask;
                logic [15:0] negate;
                mask = row_mask_for(row);
                negate = negate_mask_for(row, mask);
                if (block == 3 && row == 2)
                    force_update_ready = 1'b1;
                send_and_check_row(block, row, mask, negate);
                force_update_ready = 1'b0;
            end
        end
        repeat (3) @(posedge clk_core);
        if (fill_accepts != 96 || row_accepts != 80 || row_dones != 80
                || update_accepts == 0 || selected_sources == 0
                || numeric_lane_checks == 0 || full_k4_updates == 0
                || tail_updates == 0 || ii1_update_pairs == 0
                || update_stall_cycles == 0 || plus512_checks != 2
                || cycle_exact_checks == 0 || d_protocol_error)
            $fatal(1, "M127 positive coverage mismatch fills=%0d rows=%0d done=%0d updates=%0d selected=%0d lanes=%0d full=%0d tail=%0d ii1=%0d stalls=%0d plus512=%0d exact=%0d",
                   fill_accepts, row_accepts, row_dones,
                   update_accepts, selected_sources, numeric_lane_checks,
                   full_k4_updates, tail_updates, ii1_update_pairs,
                   update_stall_cycles, plus512_checks,
                   cycle_exact_checks);

        positive_phase = 1'b0;
        @(negedge clk_core);
        rst_core = 1'b1;
        weight_fill_valid = 1'b1;
        weight_fill_beat = 0;
        row_valid = 1'b1;
        #0.1;
        if (d_fill_ready || d_fill_accept || d_row_ready || d_row_accept
                || d_update_valid || d_update_accept || d_row_done
                || d_protocol_error)
            $fatal(1, "M127 reset isolation failed");
        repeat (2) @(posedge clk_core);
        reset_attacks = reset_attacks + 1;

        @(negedge clk_core);
        weight_fill_valid = 1'b0;
        row_valid = 1'b0;
        rst_core = 1'b0;
        repeat (2) @(posedge clk_core);
        fill_one_source(5, 0);
        @(negedge clk_core);
        row_valid = 1'b1;
        row_block = 5;
        row_offset = 9;
        row_source_mask = 16'h0002;
        row_negate_mask = 0;
        @(posedge clk_core);
        if (d_row_ready || d_row_accept || !d_protocol_error
                || d_update_valid)
            $fatal(1, "M127 invalid-cache attack not quarantined");
        @(negedge clk_core);
        row_valid = 1'b0;
        repeat (2) @(posedge clk_core);
        if (!d_protocol_error)
            $fatal(1, "M127 protocol fault not sticky");
        protocol_attacks = protocol_attacks + 1;

        $display("PASS M127 pipelined K4 row fold VCS fills=%0d rows=%0d row_done=%0d updates=%0d selected_sources=%0d numeric_lane_checks=%0d full_k4_updates=%0d tail_updates=%0d ii1_update_pairs=%0d update_stalls=%0d plus512_checks=%0d cycle_exact_checks=%0d reset_attacks=%0d protocol_attacks=%0d pair_pipeline_bits=1920 first_group_extra_cycles=0 m125_cycle_exact_positive=true reset_isolation=true cache_bytes=1536 foundry_weight_macro=false physical_speedup=false system_speedup=false headline=false",
                 fill_accepts, row_accepts, row_dones,
                 update_accepts, selected_sources, numeric_lane_checks,
                 full_k4_updates, tail_updates, ii1_update_pairs,
                 update_stall_cycles, plus512_checks,
                 cycle_exact_checks, reset_attacks, protocol_attacks);
        $finish;
    end
endmodule

`default_nettype wire
