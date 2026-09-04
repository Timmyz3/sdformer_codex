`timescale 1ns/1ps
`default_nettype none

module tb_m79_independent_hammer;
    localparam int LANES = 96;
    localparam int OUT_W = 12;
    localparam int BUFFER_W = 1280;

    logic clk_core, rst_core;
    logic command_valid, command_ready, command_accept;
    logic [31:0] command_tag;
    logic [3:0] command_width;
    logic beat_valid, beat_ready, beat_last, beat_accept;
    logic [255:0] beat_data;
    logic output_valid, output_ready, output_escape, output_accept;
    logic [31:0] output_tag;
    logic [3:0] output_width;
    logic [LANES*OUT_W-1:0] output_values;
    logic protocol_error, busy;

    logic [BUFFER_W-1:0] packed_bits;
    integer signed golden [0:LANES-1];
    integer checks;

    precision_elastic_pwp_beat_assembler dut (.*);

    always #1.5 clk_core = ~clk_core;

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            command_valid = 1'b0;
            beat_valid = 1'b0;
            beat_last = 1'b0;
            beat_data = '0;
            output_ready = 1'b1;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
        end
    endtask

    task automatic make_extreme_payload(input integer width);
        integer raw;
        begin
            packed_bits = '0;
            for (int lane = 0; lane < LANES; lane++) begin
                case (lane)
                    0: raw = -(1 << (width - 1));
                    1: raw =  (1 << (width - 1)) - 1;
                    2: raw = -1;
                    3: raw = 0;
                    default: begin
                        raw = (lane * 29 + width * 31) % (1 << width);
                        if (raw >= (1 << (width - 1))) raw -= (1 << width);
                    end
                endcase
                golden[lane] = raw;
                case (width)
                    8: packed_bits[lane*8 +: 8] = 8'(raw);
                    9: packed_bits[lane*9 +: 9] = 9'(raw);
                    10: packed_bits[lane*10 +: 10] = 10'(raw);
                    11: packed_bits[lane*11 +: 11] = 11'(raw);
                    default: $fatal(1, "independent invalid payload width");
                endcase
            end
        end
    endtask

    task automatic check_output(input integer width, input integer tag);
        begin
            #0.1;
            if (!output_valid || output_tag !== tag[31:0]
                    || output_width !== width[3:0] || output_escape)
                $fatal(1, "independent metadata mismatch width=%0d", width);
            for (int lane = 0; lane < LANES; lane++) begin
                if ($signed(output_values[lane*OUT_W +: OUT_W]) !== golden[lane])
                    $fatal(1, "independent signed unpack mismatch width=%0d lane=%0d got=%0d expected=%0d",
                           width, lane,
                           $signed(output_values[lane*OUT_W +: OUT_W]),
                           golden[lane]);
            end
            checks++;
        end
    endtask

    // Measure the minimum command-to-command interval with output_ready held
    // high.  The second command is presented on the first negedge after the
    // first transaction's final accepted beat.
    task automatic check_back_to_back_interval(input integer width,
                                                input integer expected_beats);
        realtime first_accept_time, second_accept_time;
        begin
            reset_dut();
            make_extreme_payload(width);

            @(negedge clk_core);
            command_tag = 32'h1000 + width;
            command_width = width[3:0];
            command_valid = 1'b1;
            do @(posedge clk_core); while (!command_accept);
            first_accept_time = $realtime;
            @(negedge clk_core);
            command_valid = 1'b0;

            for (int beat = 0; beat < expected_beats; beat++) begin
                beat_data = packed_bits[beat*256 +: 256];
                beat_last = (beat == expected_beats - 1);
                beat_valid = 1'b1;
                do @(posedge clk_core); while (!beat_accept);
                @(negedge clk_core);
            end
            beat_valid = 1'b0;
            beat_last = 1'b0;
            check_output(width, 32'h1000 + width);

            command_tag = 32'h2000 + width;
            command_width = width[3:0];
            command_valid = 1'b1;
            do @(posedge clk_core); while (!command_accept);
            second_accept_time = $realtime;
            if ((second_accept_time - first_accept_time) !=
                    3.0 * (expected_beats + 1))
                $fatal(1, "independent II mismatch width=%0d got_ns=%0.3f expected_cycles=%0d",
                       width, second_accept_time - first_accept_time,
                       expected_beats + 1);
            $display("M79_INDEPENDENT_II width=%0d beats=%0d command_ii_cycles=%0d",
                     width, expected_beats, expected_beats + 1);
            @(negedge clk_core);
            command_valid = 1'b0;
            checks++;
        end
    endtask

    task automatic attack_padding(input integer width,
                                  input integer expected_beats);
        begin
            reset_dut();
            packed_bits = '0;
            packed_bits[LANES*width] = 1'b1;
            @(negedge clk_core);
            command_tag = 32'h3000 + width;
            command_width = width[3:0];
            command_valid = 1'b1;
            do @(posedge clk_core); while (!command_accept);
            @(negedge clk_core);
            command_valid = 1'b0;
            for (int beat = 0; beat < expected_beats; beat++) begin
                beat_data = packed_bits[beat*256 +: 256];
                beat_last = (beat == expected_beats - 1);
                beat_valid = 1'b1;
                do @(posedge clk_core); while (!beat_accept);
                @(negedge clk_core);
            end
            beat_valid = 1'b0;
            beat_last = 1'b0;
            @(posedge clk_core); #0.1;
            if (!protocol_error || output_valid)
                $fatal(1, "independent padding attack not fail-closed width=%0d", width);
            checks++;
        end
    endtask

    task automatic attack_missing_final_last;
        begin
            reset_dut();
            @(negedge clk_core);
            command_tag = 32'h4000;
            command_width = 4'd10;
            command_valid = 1'b1;
            do @(posedge clk_core); while (!command_accept);
            @(negedge clk_core);
            command_valid = 1'b0;
            for (int beat = 0; beat < 4; beat++) begin
                beat_data = '0;
                beat_last = 1'b0;
                beat_valid = 1'b1;
                do @(posedge clk_core); while (!beat_accept);
                @(negedge clk_core);
            end
            beat_valid = 1'b0;
            @(posedge clk_core); #0.1;
            if (!protocol_error || output_valid)
                $fatal(1, "independent missing-final-last attack not fail-closed");
            checks++;
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        command_valid = 1'b0;
        command_tag = '0;
        command_width = '0;
        beat_valid = 1'b0;
        beat_last = 1'b0;
        beat_data = '0;
        output_ready = 1'b1;
        checks = 0;

        check_back_to_back_interval(8, 3);
        check_back_to_back_interval(9, 4);
        check_back_to_back_interval(10, 4);
        check_back_to_back_interval(11, 5);
        attack_padding(9, 4);
        attack_padding(10, 4);
        attack_padding(11, 5);
        attack_missing_final_last();

        if (checks != 12) $fatal(1, "independent check count mismatch %0d", checks);
        $display("PASS M79 independent hammer checks=%0d signed_extremes=4 widths=4 padding_attacks=3 missing_last=1",
                 checks);
        $finish;
    end
endmodule

`default_nettype wire
