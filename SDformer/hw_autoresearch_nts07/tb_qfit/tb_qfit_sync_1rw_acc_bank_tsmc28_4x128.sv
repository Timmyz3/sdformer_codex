`timescale 1ns/1ps
`default_nettype none

module tb_qfit_sync_1rw_acc_bank_tsmc28_4x128;
    logic clk_core = 1'b0;
    logic enable = 1'b0;
    logic write_enable = 1'b0;
    logic [6:0] address = '0;
    logic [511:0] write_data = '0;
    logic [511:0] read_data;
    logic [511:0] expected [0:127];
    logic [511:0] last_read;

    qfit_sync_1rw_acc_bank dut (.*);

    always #1 clk_core = ~clk_core;

    function automatic logic [511:0] pattern(input int unsigned row);
        logic [511:0] value;
        begin
            for (int word = 0; word < 16; word = word + 1)
                value[word*32 +: 32] = 32'h9e37_79b9 * (row + 1) ^
                                       32'h0101_0101 * word;
            return value;
        end
    endfunction

    task automatic drive_write(input int unsigned row);
        @(negedge clk_core);
        enable = 1'b1;
        write_enable = 1'b1;
        address = row[6:0];
        write_data = expected[row];
        @(posedge clk_core);
        #0.1;
    endtask

    task automatic drive_read_check(input int unsigned row);
        @(negedge clk_core);
        enable = 1'b1;
        write_enable = 1'b0;
        address = row[6:0];
        @(posedge clk_core);
        #0.1;
        if (read_data !== expected[row])
            $fatal(1, "macro-bank mismatch row=%0d got=%h expected=%h",
                   row, read_data, expected[row]);
        last_read = expected[row];
    endtask

    task automatic drive_write_check_hold(
        input int unsigned row,
        input logic [511:0] value
    );
        @(negedge clk_core);
        enable = 1'b1;
        write_enable = 1'b1;
        address = row[6:0];
        write_data = value;
        @(posedge clk_core);
        #0.1;
        if (read_data !== last_read)
            $fatal(1, "write cycle changed held output row=%0d", row);
        expected[row] = value;
    endtask

    task automatic drive_disabled_check_hold(
        input int unsigned row,
        input logic write_mode,
        input logic [511:0] value
    );
        @(negedge clk_core);
        enable = 1'b0;
        write_enable = write_mode;
        address = row[6:0];
        write_data = value;
        @(posedge clk_core);
        #0.1;
        if (read_data !== last_read)
            $fatal(1, "disabled macro-bank output did not hold row=%0d", row);
    endtask

    initial begin
        for (int row = 0; row < 128; row = row + 1)
            expected[row] = pattern(row);
        repeat (2) @(posedge clk_core);
        for (int row = 0; row < 128; row = row + 1)
            drive_write(row);
        for (int row = 127; row >= 0; row = row - 1)
            drive_read_check(row);

        // Disabled cycles must ignore a changing address, write control, and
        // data while preserving the last registered read value.
        drive_disabled_check_hold(127, 1'b1, ~expected[127]);
        drive_disabled_check_hold(63, 1'b0, expected[1]);

        // Exercise same-address R->W->R and require the vendor model's
        // documented write-cycle output hold behavior.
        drive_read_check(127);
        drive_write_check_hold(127, ~expected[127]);
        drive_read_check(127);

        // Deterministic mixed traffic covers disabled cycles, reads, writes,
        // address changes, and both same/different-address transitions.
        for (int txn = 0; txn < 512; txn = txn + 1) begin
            int unsigned row;
            int unsigned op;
            logic [511:0] value;
            row = (txn * 37 + 13) & 127;
            op = (txn * 17 + 5) % 7;
            value = pattern(row) ^ {16{32'h6d2b_79f5 * (txn + 1)}};
            if (op == 0)
                drive_disabled_check_hold(row, txn[0], value);
            else if (op <= 3)
                drive_write_check_hold(row, value);
            else
                drive_read_check(row);
        end

        // A final illegal control probe must fail noisy instead of silently
        // aliasing a legal macro operation.  This intentionally ends the test
        // because the vendor model may poison its internal contents.
        @(negedge clk_core);
        enable = 1'bx;
        write_enable = 1'b0;
        address = 7'd9;
        @(posedge clk_core);
        #0.1;
        if (!$isunknown(read_data))
            $fatal(1, "unknown enable did not propagate to macro output");

        $display("PASS_TSMC28_4X128_LOGICAL_128X512_BANK rows=128 slices=4 mixed_transactions=512 illegal_control_fail_noisy=1");
        $finish;
    end
endmodule

`default_nettype wire
