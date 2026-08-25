`timescale 1ns/1ps
`default_nettype none

module tb_precision_elastic_pwp_beat_assembler;
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
    integer transactions_checked, beats_accepted, escapes_checked;
    integer stalls_checked, protocol_attacks;

    precision_elastic_pwp_beat_assembler dut (.*);
    precision_elastic_pwp_beat_assembler_assertions m79_sva (.*);

    always #1.5 clk_core = ~clk_core;

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            command_valid = 1'b0;
            beat_valid = 1'b0;
            beat_last = 1'b0;
            output_ready = 1'b1;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core); rst_core = 1'b0;
        end
    endtask

    task automatic pack_values(input integer width, input integer txid);
        integer raw;
        begin
            packed_bits = '0;
            for (int lane = 0; lane < LANES; lane++) begin
                raw = (txid * 17 + lane * 13 + width * 7) % (1 << width);
                if (raw >= (1 << (width - 1))) raw = raw - (1 << width);
                golden[lane] = raw;
                case (width)
                    8: packed_bits[lane*8 +: 8] = 8'(raw);
                    9: packed_bits[lane*9 +: 9] = 9'(raw);
                    10: packed_bits[lane*10 +: 10] = 10'(raw);
                    11: packed_bits[lane*11 +: 11] = 11'(raw);
                    default: $fatal(1, "M79 invalid TB pack width");
                endcase
            end
        end
    endtask

    task automatic send_command(input integer width, input integer txid);
        begin
            @(negedge clk_core);
            command_tag = txid;
            command_width = width[3:0];
            command_valid = 1'b1;
            do @(posedge clk_core); while (!command_accept);
            @(negedge clk_core); command_valid = 1'b0;
        end
    endtask

    task automatic send_regular(input integer width, input integer txid,
                                input logic stall_output);
        integer beats;
        logic [LANES*OUT_W-1:0] held_values;
        begin
            pack_values(width, txid);
            send_command(width, txid);
            beats = (LANES * width + 255) / 256;
            for (int beat = 0; beat < beats; beat++) begin
                if (((txid + beat) % 5) == 0) @(posedge clk_core);
                @(negedge clk_core);
                beat_data = packed_bits[beat*256 +: 256];
                beat_last = (beat == beats - 1);
                beat_valid = 1'b1;
                do @(posedge clk_core); while (!beat_accept);
                beats_accepted++;
                @(negedge clk_core);
                beat_valid = 1'b0;
                beat_last = 1'b0;
            end
            output_ready = 1'b0;
            do @(posedge clk_core); while (!output_valid);
            #1;
            if (output_tag !== txid[31:0] || output_width !== width[3:0]
                    || output_escape)
                $fatal(1, "M79 regular metadata mismatch tx=%0d", txid);
            for (int lane = 0; lane < LANES; lane++)
                if ($signed(output_values[lane*OUT_W +: OUT_W]) !== golden[lane])
                    $fatal(1, "M79 sign extension mismatch tx=%0d width=%0d lane=%0d got=%0d expected=%0d",
                           txid, width, lane,
                           $signed(output_values[lane*OUT_W +: OUT_W]),
                           golden[lane]);
            if (stall_output) begin
                held_values = output_values;
                repeat (3) begin
                    @(posedge clk_core); #1;
                    if (!output_valid || output_values !== held_values)
                        $fatal(1, "M79 output changed under stall");
                end
                stalls_checked++;
            end
            @(negedge clk_core); output_ready = 1'b1;
            do @(posedge clk_core); while (!output_accept);
            transactions_checked++;
        end
    endtask

    task automatic send_escape(input integer txid);
        begin
            send_command(12, txid);
            output_ready = 1'b0;
            do @(posedge clk_core); while (!output_valid);
            #1;
            if (!output_escape || output_width !== 12 || output_values !== '0
                    || output_tag !== txid[31:0])
                $fatal(1, "M79 escape mismatch tx=%0d", txid);
            @(negedge clk_core); output_ready = 1'b1;
            do @(posedge clk_core); while (!output_accept);
            transactions_checked++;
            escapes_checked++;
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
        transactions_checked = 0;
        beats_accepted = 0;
        escapes_checked = 0;
        stalls_checked = 0;
        protocol_attacks = 0;
        repeat (5) @(posedge clk_core);
        @(negedge clk_core); rst_core = 1'b0;

        for (int txid = 0; txid < 128; txid++)
            send_regular(8 + (txid % 4), txid, (txid % 11) == 0);
        for (int txid = 128; txid < 136; txid++)
            send_escape(txid);

        // Attack 1: premature beat_last must fail closed.
        send_command(9, 900);
        @(negedge clk_core);
        beat_data = '0;
        beat_last = 1'b1;
        beat_valid = 1'b1;
        do @(posedge clk_core); while (!beat_accept);
        @(negedge clk_core); beat_valid = 1'b0; beat_last = 1'b0;
        @(posedge clk_core); #1;
        if (!protocol_error) $fatal(1, "M79 premature-last attack accepted");
        protocol_attacks++;

        reset_dut();
        // Attack 2: nonzero padding above a 9-bit payload must fail closed.
        packed_bits = '0;
        packed_bits[900] = 1'b1;
        send_command(9, 901);
        for (int beat = 0; beat < 4; beat++) begin
            @(negedge clk_core);
            beat_data = packed_bits[beat*256 +: 256];
            beat_last = (beat == 3);
            beat_valid = 1'b1;
            do @(posedge clk_core); while (!beat_accept);
            @(negedge clk_core); beat_valid = 1'b0; beat_last = 1'b0;
        end
        @(posedge clk_core); #1;
        if (!protocol_error) $fatal(1, "M79 nonzero-padding attack accepted");
        protocol_attacks++;

        if (transactions_checked != 136 || beats_accepted != 512
                || escapes_checked != 8 || stalls_checked != 12
                || protocol_attacks != 2)
            $fatal(1, "M79 coverage counter mismatch");
        $display("PASS M79 directed transactions=%0d beats=%0d escapes=%0d stalls=%0d lanes=%0d protocol_attacks=%0d widths=8,9,10,11,12",
                 transactions_checked, beats_accepted, escapes_checked,
                 stalls_checked, LANES, protocol_attacks);
        $finish;
    end
endmodule

`default_nettype wire
