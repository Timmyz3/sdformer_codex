`timescale 1ns/1ps
`default_nettype none

module tb_qfit_tcfm5_acc_bank_1r1w;
    localparam int DEPTH = 4;
    localparam int OUT_DIM = 2;
    localparam int ACC_W = 16;
    localparam int ADDR_W = 2;
    localparam int VEC_W = OUT_DIM * ACC_W;

    logic clk_core;
    logic rst_core;
    logic clear_valid;
    logic [ADDR_W-1:0] clear_addr;
    logic update_valid;
    logic [ADDR_W-1:0] update_addr;
    logic [VEC_W-1:0] update_delta;
    logic update_idle;
    logic read_valid;
    logic [ADDR_W-1:0] read_addr;
    logic read_data_valid;
    logic [VEC_W-1:0] read_data;

    qfit_tcfm5_acc_bank #(
        .DEPTH(DEPTH),
        .OUT_DIM(OUT_DIM),
        .ACC_W(ACC_W)
    ) dut (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .clear_valid(clear_valid),
        .clear_addr(clear_addr),
        .update_valid(update_valid),
        .update_addr(update_addr),
        .update_delta(update_delta),
        .update_idle(update_idle),
        .read_valid(read_valid),
        .read_addr(read_addr),
        .read_data_valid(read_data_valid),
        .read_data(read_data)
    );

    always #5 clk_core = ~clk_core;

    function automatic logic [VEC_W-1:0] vec(
        input integer lane0,
        input integer lane1
    );
        vec = {ACC_W'(lane1), ACC_W'(lane0)};
    endfunction

    task automatic issue_update(
        input logic [ADDR_W-1:0] addr,
        input logic [VEC_W-1:0] delta
    );
        @(negedge clk_core);
        update_valid = 1'b1;
        update_addr = addr;
        update_delta = delta;
    endtask

    task automatic stop_update;
        @(negedge clk_core);
        update_valid = 1'b0;
        update_addr = '0;
        update_delta = '0;
    endtask

    task automatic check_read(
        input logic [ADDR_W-1:0] addr,
        input logic [VEC_W-1:0] expected
    );
        @(negedge clk_core);
        read_valid = 1'b1;
        read_addr = addr;
        @(posedge clk_core);
        #1;
        if (!read_data_valid || read_data !== expected)
            $fatal(
                1,
                "read mismatch addr=%0d expected=%h got_valid=%0b got=%h",
                addr,
                expected,
                read_data_valid,
                read_data
            );
        @(negedge clk_core);
        read_valid = 1'b0;
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        clear_valid = 1'b0;
        clear_addr = '0;
        update_valid = 1'b0;
        update_addr = '0;
        update_delta = '0;
        read_valid = 1'b0;
        read_addr = '0;

        repeat (3) @(posedge clk_core);
        rst_core = 1'b0;

        for (int addr = 0; addr < DEPTH; addr = addr + 1) begin
            @(negedge clk_core);
            clear_valid = 1'b1;
            clear_addr = ADDR_W'(addr);
        end
        @(negedge clk_core);
        clear_valid = 1'b0;

        // A/A/A stresses consecutive read-during-write forwarding.
        issue_update(2'd0, vec(1, 10));
        issue_update(2'd0, vec(2, 20));
        issue_update(2'd0, vec(3, 30));

        // A/B/A stresses a one-cycle address gap around a pending write.
        issue_update(2'd1, vec(4, 40));
        issue_update(2'd2, vec(5, 50));
        issue_update(2'd1, vec(6, 60));
        stop_update();

        wait (update_idle);
        @(posedge clk_core);
        check_read(2'd0, vec(6, 60));
        check_read(2'd1, vec(10, 100));
        check_read(2'd2, vec(5, 50));
        check_read(2'd3, vec(0, 0));

        $display("PASS qfit_tcfm5_acc_bank_1r1w");
        $finish;
    end
endmodule

`default_nettype wire
