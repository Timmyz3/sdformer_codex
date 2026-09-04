`timescale 1ns/1ps
`default_nettype none

module tb_m102_bit_sparse_weight_stream;
    localparam int ROW_W = 10;
    localparam int LANES = 96;
    localparam int OUT_W = 12;
    localparam int ROWS_PER_BANK = 384;
    localparam int QUEUE_DEPTH = 512;

    logic clk_core, rst_core;
    logic lookup_valid, lookup_ready;
    logic [3:0] lookup_source;
    logic [2:0] lookup_block;
    logic [1:0] lookup_beat;
    logic [31:0] lookup_tag;
    logic [255:0] bank_words;
    logic [8*ROW_W-1:0] bank_row_addresses;
    logic output_valid, output_ready, output_escape, output_accept;
    logic [31:0] output_tag;
    logic [3:0] output_width;
    logic [LANES*OUT_W-1:0] output_values;
    logic protocol_error, busy;
    logic request_violation, request_fault, m82_output_valid;
    logic accepted_grace_match;

    logic [31:0] bank_mem [0:7][0:ROWS_PER_BANK-1];
    integer expected_source_q [0:QUEUE_DEPTH-1];
    integer expected_block_q [0:QUEUE_DEPTH-1];
    integer expected_tag_q [0:QUEUE_DEPTH-1];
    integer queue_read_ptr, queue_write_ptr, queue_count;

    integer cycle_count, accepted_beats, accepted_starts, output_count;
    integer previous_start_cycle, ii_checks, output_stall_cycles;
    integer protocol_attacks, reset_recoveries, signed_boundary_outputs;
    integer same_cycle_release_attacks;
    integer accepted_grace_holds;
    logic ii_check_enable, random_backpressure_enable;
    logic [15:0] ready_lfsr;

    m102_bit_sparse_weight_stream dut (.*);
    m102_bit_sparse_weight_stream_assertions m102_sva (.*);
    assign request_violation = dut.request_violation;
    assign request_fault = dut.request_fault_q;
    assign m82_output_valid = dut.m82_output_valid;
    assign accepted_grace_match = dut.accepted_grace_match;

    always #1.5 clk_core = ~clk_core;

    function automatic integer signed expected_weight(
        input integer source_value,
        input integer block_value,
        input integer lane_value
    );
        integer raw;
        begin
            if (lane_value == 0) begin
                expected_weight = -128;
            end else if (lane_value == 1) begin
                expected_weight = 127;
            end else begin
                raw = (source_value * 61 + block_value * 29
                       + lane_value * 17 + 11) & 8'hff;
                expected_weight = raw - 128;
            end
        end
    endfunction

    always_comb begin : port_cut_weight_memory
        bank_words = '0;
        for (int bank = 0; bank < 8; bank++) begin
            if (bank_row_addresses[bank*ROW_W +: ROW_W] < ROWS_PER_BANK)
                bank_words[bank*32 +: 32] = bank_mem[bank][
                    bank_row_addresses[bank*ROW_W +: ROW_W]];
        end
    end

    always @(negedge clk_core) begin
        if (random_backpressure_enable && !rst_core) begin
            ready_lfsr = {ready_lfsr[14:0],
                          ready_lfsr[15] ^ ready_lfsr[13]
                          ^ ready_lfsr[12] ^ ready_lfsr[10]};
            output_ready = ready_lfsr[0] || ready_lfsr[3];
        end
    end

    always @(posedge clk_core) begin : independent_scoreboard
        integer expected_row;
        cycle_count = cycle_count + 1;
        if (rst_core) begin
            queue_read_ptr = 0;
            queue_write_ptr = 0;
            queue_count = 0;
        end else begin
            if (output_valid) begin
                if (queue_count == 0)
                    $fatal(1, "M102 output without queued vector tag=%0d",
                           output_tag);
                if (output_tag !== expected_tag_q[queue_read_ptr])
                    $fatal(1, "M102 output tag mismatch got=%0d expected=%0d",
                           output_tag, expected_tag_q[queue_read_ptr]);
                if (output_width !== 8 || output_escape)
                    $fatal(1, "M102 output format width=%0d escape=%0d",
                           output_width, output_escape);
                for (int lane = 0; lane < LANES; lane++) begin
                    if ($signed(output_values[lane*OUT_W +: OUT_W])
                            !== expected_weight(
                                expected_source_q[queue_read_ptr],
                                expected_block_q[queue_read_ptr], lane))
                        $fatal(1, "M102 signed lane mismatch tag=%0d source=%0d block=%0d lane=%0d got=%0d expected=%0d",
                               output_tag,
                               expected_source_q[queue_read_ptr],
                               expected_block_q[queue_read_ptr], lane,
                               $signed(output_values[lane*OUT_W +: OUT_W]),
                               expected_weight(
                                   expected_source_q[queue_read_ptr],
                                   expected_block_q[queue_read_ptr], lane));
                end
            end
            if (output_valid && !output_ready)
                output_stall_cycles = output_stall_cycles + 1;
            if (output_accept) begin
                if ($signed(output_values[0*OUT_W +: OUT_W]) != -128
                        || $signed(output_values[1*OUT_W +: OUT_W]) != 127)
                    $fatal(1, "M102 signed boundary mismatch tag=%0d lane0=%0d lane1=%0d",
                           output_tag,
                           $signed(output_values[0*OUT_W +: OUT_W]),
                           $signed(output_values[1*OUT_W +: OUT_W]));
                signed_boundary_outputs = signed_boundary_outputs + 1;
                output_count = output_count + 1;
                queue_read_ptr = (queue_read_ptr + 1) % QUEUE_DEPTH;
                queue_count = queue_count - 1;
            end

            if (lookup_valid && lookup_ready) begin
                accepted_beats = accepted_beats + 1;
                expected_row = ((lookup_source * 8 + lookup_block) * 24
                                + lookup_beat * 8) / 8;
                for (int bank = 0; bank < 8; bank++) begin
                    if (bank_row_addresses[bank*ROW_W +: ROW_W]
                            !== expected_row)
                        $fatal(1, "M102 bank row mismatch source=%0d block=%0d beat=%0d bank=%0d got=%0d expected=%0d",
                               lookup_source, lookup_block, lookup_beat,
                               bank,
                               bank_row_addresses[bank*ROW_W +: ROW_W],
                               expected_row);
                end
                if (lookup_beat == 0) begin
                    if (queue_count >= QUEUE_DEPTH)
                        $fatal(1, "M102 expected queue overflow");
                    expected_source_q[queue_write_ptr] = lookup_source;
                    expected_block_q[queue_write_ptr] = lookup_block;
                    expected_tag_q[queue_write_ptr] = lookup_tag;
                    queue_write_ptr = (queue_write_ptr + 1) % QUEUE_DEPTH;
                    queue_count = queue_count + 1;
                    accepted_starts = accepted_starts + 1;
                    if (ii_check_enable) begin
                        if (previous_start_cycle >= 0) begin
                            if (cycle_count - previous_start_cycle != 3)
                                $fatal(1, "M102 fixed8 II mismatch got=%0d expected=3 tag=%0d",
                                       cycle_count - previous_start_cycle,
                                       lookup_tag);
                            ii_checks = ii_checks + 1;
                        end
                        previous_start_cycle = cycle_count;
                    end
                end
            end
        end
    end

    task automatic set_lookup(
        input integer source_value,
        input integer block_value,
        input integer beat_value,
        input integer tag_value
    );
        begin
            lookup_source = source_value[3:0];
            lookup_block = block_value[2:0];
            lookup_beat = beat_value[1:0];
            lookup_tag = tag_value[31:0];
        end
    endtask

    task automatic drive_accepted_beat(
        input integer source_value,
        input integer block_value,
        input integer beat_value,
        input integer tag_value
    );
        begin
            @(negedge clk_core);
            set_lookup(source_value, block_value, beat_value, tag_value);
            lookup_valid = 1'b1;
            do @(posedge clk_core); while (!lookup_ready);
            #0.01;
            lookup_valid = 1'b0;
        end
    endtask

    task automatic drive_vector(
        input integer source_value,
        input integer block_value,
        input integer tag_value
    );
        begin
            drive_accepted_beat(source_value, block_value, 0, tag_value);
            drive_accepted_beat(source_value, block_value, 1, tag_value);
            drive_accepted_beat(source_value, block_value, 2, tag_value);
        end
    endtask

    task automatic drive_vector_hold_final_request(
        input integer source_value,
        input integer block_value,
        input integer tag_value
    );
        logic [LANES*OUT_W-1:0] held_values;
        begin
            drive_accepted_beat(source_value, block_value, 0, tag_value);
            drive_accepted_beat(source_value, block_value, 1, tag_value);
            @(negedge clk_core);
            output_ready = 1'b0;
            set_lookup(source_value, block_value, 2, tag_value);
            lookup_valid = 1'b1;
            do @(posedge clk_core); while (!lookup_ready);
            #0.1;
            held_values = output_values;
            if (!accepted_grace_match || request_violation || protocol_error
                    || lookup_ready || !output_valid || !m82_output_valid)
                $fatal(1, "M102 accepted final request grace failed after transfer grace=%0d violation=%0d fault=%0d ready=%0d output=%0d m82=%0d",
                       accepted_grace_match, request_violation,
                       protocol_error, lookup_ready, output_valid,
                       m82_output_valid);
            // Keep the exact accepted request asserted for a complete extra
            // cycle.  It must neither be accepted twice nor hide the result.
            @(posedge clk_core); #0.1;
            if (!accepted_grace_match || request_violation || protocol_error
                    || lookup_ready || !output_valid
                    || output_values !== held_values)
                $fatal(1, "M102 accepted request grace was not stable");
            @(negedge clk_core);
            lookup_valid = 1'b0;
            output_ready = 1'b1;
            accepted_grace_holds = accepted_grace_holds + 1;
        end
    endtask

    task automatic stop_lookup;
        begin
            @(negedge clk_core);
            lookup_valid = 1'b0;
        end
    endtask

    task automatic wait_for_drain;
        begin
            stop_lookup();
            wait (queue_count == 0 && !busy && !output_valid);
            @(posedge clk_core);
        end
    endtask

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            random_backpressure_enable = 1'b0;
            rst_core = 1'b1;
            lookup_valid = 1'b0;
            output_ready = 1'b1;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
            @(posedge clk_core);
            #0.1;
            if (protocol_error || busy || output_valid)
                $fatal(1, "M102 reset recovery failed error=%0d busy=%0d output=%0d",
                       protocol_error, busy, output_valid);
            reset_recoveries = reset_recoveries + 1;
        end
    endtask

    task automatic present_invalid_request(
        input integer source_value,
        input integer block_value,
        input integer beat_value,
        input integer tag_value
    );
        begin
            @(negedge clk_core);
            set_lookup(source_value, block_value, beat_value, tag_value);
            lookup_valid = 1'b1;
            #0.1;
            if (lookup_ready)
                $fatal(1, "M102 invalid request advertised ready source=%0d block=%0d beat=%0d tag=%0d",
                       source_value, block_value, beat_value, tag_value);
            @(posedge clk_core);
            @(negedge clk_core);
            lookup_valid = 1'b0;
            #0.1;
            if (!protocol_error)
                $fatal(1, "M102 invalid request did not fault source=%0d block=%0d beat=%0d tag=%0d",
                       source_value, block_value, beat_value, tag_value);
            repeat (2) begin
                @(posedge clk_core);
                #0.1;
                if (!protocol_error || lookup_ready || output_valid
                        || output_accept)
                    $fatal(1, "M102 fault not sticky/fail-closed");
            end
            protocol_attacks = protocol_attacks + 1;
        end
    endtask

    task automatic attack_stalled_output_same_cycle_invalid;
        begin
            reset_dut();
            @(negedge clk_core);
            output_ready = 1'b0;
            drive_vector(3, 4, 32'h4f00);
            stop_lookup();
            wait (output_valid && dut.m82_output_valid);
            @(negedge clk_core);
            set_lookup(3, 4, 1, 32'h4f01);
            lookup_valid = 1'b1;
            output_ready = 1'b1;
            #0.1;
            if (!request_violation || !protocol_error || lookup_ready
                    || output_valid || output_accept
                    || !dut.m82_output_valid)
                $fatal(1, "M102 baseline same-cycle quarantine failed before edge violation=%0d fault=%0d ready=%0d valid=%0d accept=%0d m82=%0d",
                       request_violation, protocol_error, lookup_ready,
                       output_valid, output_accept, dut.m82_output_valid);
            @(posedge clk_core); #0.1;
            if (!request_fault || !protocol_error || output_valid
                    || output_accept || !dut.m82_output_valid)
                $fatal(1, "M102 baseline same-cycle quarantine failed at edge");
            @(negedge clk_core);
            lookup_valid = 1'b0;
            repeat (2) begin
                @(posedge clk_core); #0.1;
                if (!request_fault || !protocol_error || output_valid
                        || output_accept || !dut.m82_output_valid)
                    $fatal(1, "M102 baseline buffered fault was not sticky");
            end
            protocol_attacks = protocol_attacks + 1;
            same_cycle_release_attacks = same_cycle_release_attacks + 1;
        end
    endtask

    initial begin : initialize_weight_image
        integer logical_word;
        integer value;
        for (int bank = 0; bank < 8; bank++)
            for (int row = 0; row < ROWS_PER_BANK; row++)
                bank_mem[bank][row] = '0;
        for (int source = 0; source < 16; source++) begin
            for (int block = 0; block < 8; block++) begin
                for (int lane = 0; lane < LANES; lane++) begin
                    logical_word = (source * 8 + block) * 24 + lane / 4;
                    value = expected_weight(source, block, lane);
                    bank_mem[logical_word & 7][logical_word >> 3]
                            [(lane % 4)*8 +: 8] = value[7:0];
                end
            end
        end
    end

    initial begin : watchdog
        #1000000;
        $fatal(1, "M102 watchdog timeout beats=%0d starts=%0d outputs=%0d attacks=%0d",
               accepted_beats, accepted_starts, output_count,
               protocol_attacks);
    end

    initial begin : directed_test
        logic [LANES*OUT_W-1:0] held_output;
        logic [31:0] held_tag;

        clk_core = 1'b0;
        rst_core = 1'b1;
        lookup_valid = 1'b0;
        lookup_source = '0;
        lookup_block = '0;
        lookup_beat = '0;
        lookup_tag = '0;
        output_ready = 1'b1;
        ii_check_enable = 1'b0;
        random_backpressure_enable = 1'b0;
        ready_lfsr = 16'h1d3f;
        cycle_count = 0;
        accepted_beats = 0;
        accepted_starts = 0;
        output_count = 0;
        previous_start_cycle = -1;
        ii_checks = 0;
        output_stall_cycles = 0;
        protocol_attacks = 0;
        reset_recoveries = 0;
        signed_boundary_outputs = 0;
        same_cycle_release_attacks = 0;
        accepted_grace_holds = 0;
        queue_read_ptr = 0;
        queue_write_ptr = 0;
        queue_count = 0;

        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        // Boundary source/block identities and a continuous always-ready stream.
        ii_check_enable = 1'b1;
        previous_start_cycle = -1;
        for (int vector_id = 0; vector_id < 24; vector_id++)
            drive_vector(vector_id % 16, (vector_id * 5) % 8,
                         32'h1000 + vector_id);
        ii_check_enable = 1'b0;
        wait_for_drain();
        if (ii_checks != 23)
            $fatal(1, "M102 II=3 coverage mismatch got=%0d expected=23",
                   ii_checks);

        // Deterministic pseudo-random output backpressure over all identities.
        random_backpressure_enable = 1'b1;
        for (int vector_id = 0; vector_id < 64; vector_id++)
            drive_vector((vector_id * 7) % 16, (vector_id * 3) % 8,
                         32'h2000 + vector_id);
        stop_lookup();
        wait (queue_count == 0 && !busy && !output_valid);
        @(negedge clk_core);
        random_backpressure_enable = 1'b0;
        output_ready = 1'b1;
        if (output_stall_cycles == 0)
            $fatal(1, "M102 random backpressure produced no output stall");

        // Directed stalled output: payload/tag must remain stable and a legal
        // next start must be backpressured without being treated as a fault.
        @(negedge clk_core);
        output_ready = 1'b0;
        drive_vector(15, 7, 32'h3000);
        stop_lookup();
        wait (output_valid);
        @(negedge clk_core);
        held_output = output_values;
        held_tag = output_tag;
        set_lookup(0, 0, 0, 32'h3001);
        lookup_valid = 1'b1;
        repeat (4) begin
            @(posedge clk_core);
            #0.1;
            if (!output_valid || output_values !== held_output
                    || output_tag !== held_tag || lookup_ready
                    || protocol_error)
                $fatal(1, "M102 directed backpressure stability failure");
        end
        @(negedge clk_core);
        lookup_valid = 1'b0;
        output_ready = 1'b1;
        wait (queue_count == 0 && !busy && !output_valid);

        // Attack 1: continuation while idle.
        reset_dut();
        present_invalid_request(0, 0, 1, 32'h4000);

        // Attack 2: final beat while idle.
        reset_dut();
        present_invalid_request(15, 7, 2, 32'h4001);

        // Attack 3: beat order restarts at zero after an accepted first beat.
        reset_dut();
        drive_accepted_beat(2, 3, 0, 32'h4002);
        // Make the valid-low boundary observable at an active edge before
        // presenting the same identity as a genuinely new illegal request.
        @(posedge clk_core); #0.1;
        present_invalid_request(2, 3, 0, 32'h4002);

        // Attack 4: source identity mutates on beat one.
        reset_dut();
        drive_accepted_beat(4, 5, 0, 32'h4003);
        present_invalid_request(5, 5, 1, 32'h4003);

        // Attack 5: output-block identity mutates on beat one.
        reset_dut();
        drive_accepted_beat(6, 1, 0, 32'h4004);
        present_invalid_request(6, 2, 1, 32'h4004);

        // Attack 6: tag identity mutates on beat one.
        reset_dut();
        drive_accepted_beat(8, 6, 0, 32'h4005);
        present_invalid_request(8, 6, 1, 32'h4006);

        // Attack 7: an invalid request and ready release in the same cycle
        // cannot retire an older stalled M82 output.
        attack_stalled_output_same_cycle_invalid();

        // Final reset plus a boundary vector proves functional recovery.
        reset_dut();
        drive_vector_hold_final_request(15, 7, 32'h5000);
        wait_for_drain();

        if (accepted_beats != 277 || accepted_starts != 95
                || output_count != 90 || signed_boundary_outputs != 90
                || ii_checks != 23 || protocol_attacks != 7
                || same_cycle_release_attacks != 1
                || accepted_grace_holds != 1
                || reset_recoveries != 8 || output_stall_cycles == 0
                || protocol_error)
            $fatal(1, "M102 coverage mismatch beats=%0d starts=%0d outputs=%0d signed=%0d ii=%0d attacks=%0d same_cycle=%0d grace=%0d resets=%0d stalls=%0d error=%0d",
                   accepted_beats, accepted_starts, output_count,
                   signed_boundary_outputs, ii_checks, protocol_attacks,
                   same_cycle_release_attacks, accepted_grace_holds,
                   reset_recoveries,
                   output_stall_cycles, protocol_error);

        $display("PASS M102 bit-sparse-r4 baseline vectors=90 beats=277 starts=95 ii3_checks=23 lanes=96 signed_min=-128 signed_max=127 stalls=%0d attacks=7 same_cycle_release_attacks=1 accepted_grace_holds=1 resets=8 precompacted=true macros=0",
                 output_stall_cycles);
        $finish;
    end
endmodule

`default_nettype wire
