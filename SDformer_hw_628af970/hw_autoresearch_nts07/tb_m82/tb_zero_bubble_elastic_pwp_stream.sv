`timescale 1ns/1ps
`default_nettype none

module tb_zero_bubble_elastic_pwp_stream;
    localparam int LANES = 96;
    localparam int OUT_W = 12;
    localparam int BUFFER_W = 1280;

    logic clk_core, rst_core;
    logic beat_valid, beat_ready, beat_start, beat_last, beat_accept;
    logic [3:0] beat_width;
    logic [31:0] beat_tag;
    logic [255:0] beat_data;
    logic output_valid, output_ready, output_escape, output_accept;
    logic [31:0] output_tag;
    logic [3:0] output_width;
    logic [LANES*OUT_W-1:0] output_values;
    logic protocol_error, collecting, busy;

    logic [BUFFER_W-1:0] packed_bits;
    integer cycle_count, accepted_beats, accepted_starts;
    integer regular_outputs, escape_outputs, protocol_attacks;
    integer previous_start_cycle, previous_transaction_beats;
    integer minimum_ii_checks;
    logic [31:0] last_counted_tag;

    zero_bubble_elastic_pwp_stream dut (.*);
    zero_bubble_elastic_pwp_stream_assertions m82_sva (.*);

    always #1.5 clk_core = ~clk_core;
    always @(posedge clk_core) cycle_count <= cycle_count + 1;

    initial begin
        #20000;
        $fatal(1, "M82 watchdog timeout regular=%0d escape=%0d starts=%0d attacks=%0d",
               regular_outputs, escape_outputs, accepted_starts,
               protocol_attacks);
    end

    function automatic integer signed expected_lane(
        input integer txid, input integer lane, input integer width
    );
        integer raw;
        begin
            raw = (txid * 17 + lane * 13 + width * 7) % (1 << width);
            if (raw >= (1 << (width - 1))) raw = raw - (1 << width);
            expected_lane = raw;
        end
    endfunction

    task automatic pack_transaction(input integer width, input integer txid);
        integer value;
        begin
            packed_bits = '0;
            for (int lane = 0; lane < LANES; lane++) begin
                value = expected_lane(txid, lane, width);
                case (width)
                    8: packed_bits[lane*8 +: 8] = 8'(value);
                    9: packed_bits[lane*9 +: 9] = 9'(value);
                    10: packed_bits[lane*10 +: 10] = 10'(value);
                    11: packed_bits[lane*11 +: 11] = 11'(value);
                    default: $fatal(1, "M82 invalid pack width");
                endcase
            end
        end
    endtask

    task automatic drive_beat(
        input logic start_value,
        input logic last_value,
        input integer width_value,
        input integer tag_value,
        input logic [255:0] data_value
    );
        begin
            @(negedge clk_core);
            beat_start = start_value;
            beat_last = last_value;
            beat_width = width_value[3:0];
            beat_tag = tag_value[31:0];
            beat_data = data_value;
            beat_valid = 1'b1;
            do @(posedge clk_core); while (!beat_accept);
            accepted_beats++;
            if (start_value) begin
                accepted_starts++;
                if (previous_start_cycle >= 0) begin
                    if (cycle_count - previous_start_cycle
                            != previous_transaction_beats)
                        $fatal(1, "M82 transaction bubble got_ii=%0d expected=%0d tag=%0d",
                               cycle_count - previous_start_cycle,
                               previous_transaction_beats, tag_value);
                    minimum_ii_checks++;
                end
                previous_start_cycle = cycle_count;
                previous_transaction_beats = (width_value == 12)
                    ? 1 : (LANES * width_value + 255) / 256;
            end
        end
    endtask

    task automatic drive_regular(input integer width, input integer txid);
        integer beats;
        begin
            pack_transaction(width, txid);
            beats = (LANES * width + 255) / 256;
            for (int beat = 0; beat < beats; beat++)
                drive_beat(beat == 0, beat == beats-1,
                           beat == 0 ? width : 0,
                           beat == 0 ? txid : 0,
                           packed_bits[beat*256 +: 256]);
        end
    endtask

    task automatic drive_escape(input integer txid);
        begin
            drive_beat(1'b1, 1'b1, 12, txid, '0);
        end
    endtask

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            beat_valid = 1'b0;
            output_ready = 1'b1;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core); rst_core = 1'b0;
        end
    endtask

    always @(posedge clk_core) begin
        #1;
        if (output_valid) begin
            if (output_escape) begin
                if (output_width !== 12 || output_values !== '0)
                    $fatal(1, "M82 malformed escape output");
            end else begin
                for (int lane = 0; lane < LANES; lane++)
                    if ($signed(output_values[lane*OUT_W +: OUT_W])
                            !== expected_lane(output_tag, lane, output_width))
                        $fatal(1, "M82 lane mismatch tag=%0d width=%0d lane=%0d got=%0d expected=%0d",
                               output_tag, output_width, lane,
                               $signed(output_values[lane*OUT_W +: OUT_W]),
                               expected_lane(output_tag, lane, output_width));
            end
        end
        if (output_valid && output_tag != last_counted_tag) begin
            if (output_escape) escape_outputs++;
            else regular_outputs++;
            last_counted_tag = output_tag;
        end
    end

    initial begin
        logic [1151:0] held_output;
        clk_core = 1'b0;
        rst_core = 1'b1;
        beat_valid = 1'b0;
        beat_start = 1'b0;
        beat_last = 1'b0;
        beat_width = '0;
        beat_tag = '0;
        beat_data = '0;
        output_ready = 1'b1;
        cycle_count = 0;
        accepted_beats = 0;
        accepted_starts = 0;
        regular_outputs = 0;
        escape_outputs = 0;
        protocol_attacks = 0;
        previous_start_cycle = -1;
        previous_transaction_beats = 0;
        minimum_ii_checks = 0;
        last_counted_tag = '1;
        repeat (5) @(posedge clk_core);
        @(negedge clk_core); rst_core = 1'b0;

        // No idle cycles are inserted between any of these transactions.
        for (int txid = 0; txid < 128; txid++)
            drive_regular(8 + (txid % 4), txid);
        for (int txid = 128; txid < 136; txid++)
            drive_escape(txid);
        @(negedge clk_core); beat_valid = 1'b0;
        $display("M82 legal producer complete regular=%0d escape=%0d starts=%0d ii=%0d",
                 regular_outputs, escape_outputs, accepted_starts,
                 minimum_ii_checks);
        wait (regular_outputs == 128 && escape_outputs == 8);
        $display("M82 legal consumer complete");
        do @(posedge clk_core); while (output_valid);

        // Directed output backpressure; input must stop and output stay stable.
        @(negedge clk_core); output_ready = 1'b0;
        previous_start_cycle = -1;
        drive_regular(11, 200);
        @(negedge clk_core); beat_valid = 1'b0;
        $display("M82 stall producer complete output_valid=%0d", output_valid);
        do @(posedge clk_core); while (!output_valid);
        #1; held_output = output_values;
        repeat (3) begin
            @(posedge clk_core); #1;
            if (!output_valid || output_values !== held_output || beat_ready)
                $fatal(1, "M82 backpressure stability/readiness failure");
        end
        @(negedge clk_core); output_ready = 1'b1;
        do @(posedge clk_core); while (!output_accept);

        reset_dut();
        // Attack 1: continuation without a start descriptor.
        previous_start_cycle = -1;
        drive_beat(1'b0, 1'b0, 0, 0, '0);
        @(negedge clk_core); beat_valid = 1'b0;
        @(posedge clk_core); #1;
        if (!protocol_error) $fatal(1, "M82 orphan continuation accepted");
        protocol_attacks++;

        reset_dut();
        // Attack 2: a regular first beat cannot also be final.
        previous_start_cycle = -1;
        drive_beat(1'b1, 1'b1, 9, 901, '0);
        @(negedge clk_core); beat_valid = 1'b0;
        @(posedge clk_core); #1;
        if (!protocol_error) $fatal(1, "M82 premature last accepted");
        protocol_attacks++;

        reset_dut();
        // Attack 3: nonzero padding above the 9-bit payload.
        previous_start_cycle = -1;
        packed_bits = '0;
        packed_bits[900] = 1'b1;
        for (int beat = 0; beat < 4; beat++)
            drive_beat(beat == 0, beat == 3,
                       beat == 0 ? 9 : 0,
                       beat == 0 ? 902 : 0,
                       packed_bits[beat*256 +: 256]);
        @(negedge clk_core); beat_valid = 1'b0;
        @(posedge clk_core); #1;
        if (!protocol_error) $fatal(1, "M82 nonzero padding accepted");
        protocol_attacks++;

        if (accepted_starts != 139 || minimum_ii_checks != 135
                || regular_outputs != 129 || escape_outputs != 8
                || protocol_attacks != 3)
            $fatal(1, "M82 coverage counter mismatch starts=%0d ii=%0d regular=%0d escape=%0d attacks=%0d",
                   accepted_starts, minimum_ii_checks, regular_outputs,
                   escape_outputs, protocol_attacks);
        $display("PASS M82 zero-bubble regular=129 escapes=8 starts=139 ii_checks=135 stalls=1 lanes=96 protocol_attacks=3 service=3,4,4,5");
        $finish;
    end
endmodule

`default_nettype wire
