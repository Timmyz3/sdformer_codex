`timescale 1ns/1ps
`default_nettype none

module tb_m82_independent_hammer;
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
    integer signed golden [0:LANES-1];
    integer normal_outputs, escape_outputs, ii_checks;
    integer stall_cycles, protocol_attacks;
    integer last_width, last_service;
    realtime last_start_time;
    logic have_last_start;

    zero_bubble_elastic_pwp_stream dut (.*);
    zero_bubble_elastic_pwp_stream_assertions independent_sva (.*);

    always #1.5 clk_core = ~clk_core;

    initial begin
        #30000;
        $fatal(1, "M82 independent watchdog");
    end

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            beat_valid = 1'b0;
            beat_start = 1'b0;
            beat_last = 1'b0;
            beat_width = '0;
            beat_tag = '0;
            beat_data = '0;
            output_ready = 1'b1;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
            have_last_start = 1'b0;
        end
    endtask

    task automatic make_extreme_payload(input integer width,
                                        input integer seed);
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
                        raw = (seed * 37 + lane * 29 + width * 31)
                            % (1 << width);
                        if (raw >= (1 << (width - 1))) raw -= (1 << width);
                    end
                endcase
                golden[lane] = raw;
                case (width)
                    8: packed_bits[lane*8 +: 8] = 8'(raw);
                    9: packed_bits[lane*9 +: 9] = 9'(raw);
                    10: packed_bits[lane*10 +: 10] = 10'(raw);
                    11: packed_bits[lane*11 +: 11] = 11'(raw);
                    default: $fatal(1, "M82 independent invalid width");
                endcase
            end
        end
    endtask

    task automatic note_start(input integer width,
                              input integer service);
        realtime observed;
        begin
            if (have_last_start) begin
                observed = ($realtime - last_start_time) / 3.0;
                if (observed != last_service)
                    $fatal(1, "M82 independent II mismatch previous_width=%0d observed=%0.3f expected=%0d",
                           last_width, observed, last_service);
                $display("M82_INDEPENDENT_II previous_width=%0d observed_cycles=%0d",
                         last_width, last_service);
                ii_checks++;
            end
            have_last_start = 1'b1;
            last_start_time = $realtime;
            last_width = width;
            last_service = service;
        end
    endtask

    task automatic drive_beat(
        input logic start_value,
        input logic last_value,
        input integer width_value,
        input integer tag_value,
        input logic [255:0] data_value,
        input integer service_value
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
            if (start_value) note_start(width_value, service_value);
        end
    endtask

    task automatic check_regular_output(input integer width,
                                        input integer tag);
        begin
            #0.1;
            if (!output_valid || output_escape || output_width !== width[3:0]
                    || output_tag !== tag[31:0])
                $fatal(1, "M82 independent regular metadata mismatch width=%0d tag=%0d",
                       width, tag);
            for (int lane = 0; lane < LANES; lane++) begin
                if ($signed(output_values[lane*OUT_W +: OUT_W]) !== golden[lane])
                    $fatal(1, "M82 independent signed mismatch width=%0d tag=%0d lane=%0d got=%0d expected=%0d",
                           width, tag, lane,
                           $signed(output_values[lane*OUT_W +: OUT_W]),
                           golden[lane]);
            end
            normal_outputs++;
        end
    endtask

    task automatic drive_regular(input integer width,
                                 input integer tag);
        integer beats;
        begin
            make_extreme_payload(width, tag);
            beats = (LANES * width + 255) / 256;
            for (int beat = 0; beat < beats; beat++) begin
                drive_beat(beat == 0, beat == beats - 1,
                           beat == 0 ? width : 0,
                           beat == 0 ? tag : 0,
                           packed_bits[beat*256 +: 256], beats);
            end
            check_regular_output(width, tag);
        end
    endtask

    task automatic drive_escape(input integer tag);
        begin
            drive_beat(1'b1, 1'b1, 12, tag, '0, 1);
            #0.1;
            if (!output_valid || !output_escape || output_width !== 12
                    || output_tag !== tag[31:0] || output_values !== '0)
                $fatal(1, "M82 independent escape mismatch tag=%0d", tag);
            escape_outputs++;
        end
    endtask

    task automatic check_stalled_boundary;
        logic [1151:0] held_values;
        logic [255:0] first_data;
        integer second_tag;
        begin
            reset_dut();
            output_ready = 1'b0;
            drive_regular(8, 500);
            held_values = output_values;
            second_tag = 501;
            make_extreme_payload(9, second_tag);
            first_data = packed_bits[255:0];

            @(negedge clk_core);
            beat_start = 1'b1;
            beat_last = 1'b0;
            beat_width = 4'd9;
            beat_tag = second_tag;
            beat_data = first_data;
            beat_valid = 1'b1;
            repeat (3) begin
                @(posedge clk_core); #0.1;
                if (beat_ready || beat_accept || !output_valid
                        || output_values !== held_values)
                    $fatal(1, "M82 independent stall hold failure");
                stall_cycles++;
            end
            @(negedge clk_core);
            output_ready = 1'b1;
            @(posedge clk_core); #0.1;
            if (!beat_accept)
                $fatal(1, "M82 independent stalled start did not resume");
            if (output_valid)
                $fatal(1, "M82 independent previous output did not retire");

            for (int beat = 1; beat < 4; beat++) begin
                drive_beat(1'b0, beat == 3, 0, 0,
                           packed_bits[beat*256 +: 256], 4);
            end
            check_regular_output(9, second_tag);
            @(negedge clk_core);
            beat_valid = 1'b0;
        end
    endtask

    task automatic attack_start_mid_transaction;
        begin
            reset_dut();
            make_extreme_payload(10, 600);
            drive_beat(1'b1, 1'b0, 10, 600, packed_bits[255:0], 4);
            // The second start is intentionally illegal and is excluded from
            // the legal-stream initiation-interval scoreboard.
            have_last_start = 1'b0;
            drive_beat(1'b1, 1'b0, 10, 601, packed_bits[511:256], 4);
            #0.1;
            if (!protocol_error || output_valid)
                $fatal(1, "M82 independent mid-transaction start not rejected");
            protocol_attacks++;
            @(posedge clk_core); #0.1;
            if (!protocol_error) $fatal(1, "M82 independent start fault not sticky");
        end
    endtask

    task automatic attack_last(input logic premature);
        begin
            reset_dut();
            make_extreme_payload(11, 610 + premature);
            drive_beat(1'b1, 1'b0, 11, 610 + premature,
                       packed_bits[255:0], 5);
            if (premature) begin
                drive_beat(1'b0, 1'b1, 0, 0, packed_bits[511:256], 5);
            end else begin
                for (int beat = 1; beat < 5; beat++)
                    drive_beat(1'b0, 1'b0, 0, 0,
                               packed_bits[beat*256 +: 256], 5);
            end
            #0.1;
            if (!protocol_error || output_valid)
                $fatal(1, "M82 independent last attack not rejected premature=%0d",
                       premature);
            protocol_attacks++;
            @(posedge clk_core); #0.1;
            if (!protocol_error) $fatal(1, "M82 independent last fault not sticky");
        end
    endtask

    task automatic attack_padding(input integer width,
                                  input integer beats,
                                  input integer tag);
        begin
            reset_dut();
            packed_bits = '0;
            packed_bits[LANES*width] = 1'b1;
            for (int beat = 0; beat < beats; beat++)
                drive_beat(beat == 0, beat == beats-1,
                           beat == 0 ? width : 0,
                           beat == 0 ? tag : 0,
                           packed_bits[beat*256 +: 256], beats);
            #0.1;
            if (!protocol_error || output_valid)
                $fatal(1, "M82 independent padding attack not rejected width=%0d",
                       width);
            protocol_attacks++;
            @(posedge clk_core); #0.1;
            if (!protocol_error) $fatal(1, "M82 independent padding fault not sticky");
        end
    endtask

    task automatic attack_escape(input logic bad_last,
                                 input logic bad_data);
        logic [255:0] value;
        begin
            reset_dut();
            value = '0;
            value[0] = bad_data;
            drive_beat(1'b1, !bad_last, 12, 700 + bad_data,
                       value, 1);
            #0.1;
            if (!protocol_error || output_valid)
                $fatal(1, "M82 independent malformed escape not rejected");
            protocol_attacks++;
            @(posedge clk_core); #0.1;
            if (!protocol_error) $fatal(1, "M82 independent escape fault not sticky");
        end
    endtask

    initial begin
        integer width_sequence [0:11];
        clk_core = 1'b0;
        rst_core = 1'b1;
        beat_valid = 1'b0;
        beat_start = 1'b0;
        beat_last = 1'b0;
        beat_width = '0;
        beat_tag = '0;
        beat_data = '0;
        output_ready = 1'b1;
        normal_outputs = 0;
        escape_outputs = 0;
        ii_checks = 0;
        stall_cycles = 0;
        protocol_attacks = 0;
        have_last_start = 1'b0;
        width_sequence[0] = 8;
        width_sequence[1] = 11;
        width_sequence[2] = 9;
        width_sequence[3] = 10;
        width_sequence[4] = 12;
        width_sequence[5] = 12;
        width_sequence[6] = 8;
        width_sequence[7] = 12;
        width_sequence[8] = 11;
        width_sequence[9] = 10;
        width_sequence[10] = 9;
        width_sequence[11] = 8;

        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        for (int index = 0; index < 12; index++) begin
            if (width_sequence[index] == 12)
                drive_escape(100 + index);
            else
                drive_regular(width_sequence[index], 100 + index);
        end
        @(negedge clk_core);
        beat_valid = 1'b0;

        check_stalled_boundary();
        attack_start_mid_transaction();
        attack_last(1'b1);
        attack_last(1'b0);
        attack_padding(9, 4, 620);
        attack_padding(10, 4, 621);
        attack_padding(11, 5, 622);
        attack_escape(1'b1, 1'b0);
        attack_escape(1'b0, 1'b1);

        if (normal_outputs != 11 || escape_outputs != 3 || ii_checks != 11
                || stall_cycles != 3 || protocol_attacks != 8)
            $fatal(1, "M82 independent coverage mismatch normal=%0d escape=%0d ii=%0d stall=%0d attacks=%0d",
                   normal_outputs, escape_outputs, ii_checks, stall_cycles,
                   protocol_attacks);
        $display("PASS M82 independent hammer normal=11 escapes=3 mixed_ii=11 stall_cycles=3 attacks=8 signed_extremes=8,9,10,11 service=3,4,4,5 escape_service=1");
        $finish;
    end
endmodule

`default_nettype wire
