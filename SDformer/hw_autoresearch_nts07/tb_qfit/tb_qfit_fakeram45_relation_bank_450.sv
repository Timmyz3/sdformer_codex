`timescale 1ns/1ps
`default_nettype none

module tb_qfit_fakeram45_relation_bank_450;
    logic clk_core = 1'b0;
    logic rst_core = 1'b1;
    logic write_valid;
    logic [8:0] write_addr;
    logic [31:0] write_data32;
    logic [9:0] write_data10;
    logic read_valid;
    logic [8:0] read_addr;
    logic generic_valid32;
    logic macro_valid32;
    logic [31:0] generic_data32;
    logic [31:0] macro_data32;
    logic generic_valid10;
    logic macro_valid10;
    logic [9:0] generic_data10;
    logic [9:0] macro_data10;
    integer responses;

    always #5 clk_core = ~clk_core;

    qfit_sync_relation_bank #(
        .DEPTH(450), .DATA_W(32), .READ_LATENCY(1)
    ) u_generic32 (
        .clk_core(clk_core), .rst_core(rst_core),
        .write_valid(write_valid), .write_addr(write_addr),
        .write_data(write_data32), .read_valid(read_valid),
        .read_addr(read_addr), .read_data_valid(generic_valid32),
        .read_data(generic_data32)
    );

    qfit_fakeram45_relation_bank_450 #(.DATA_W(32)) u_macro32 (
        .clk_core(clk_core), .rst_core(rst_core),
        .write_valid(write_valid), .write_addr(write_addr),
        .write_data(write_data32), .read_valid(read_valid),
        .read_addr(read_addr), .read_data_valid(macro_valid32),
        .read_data(macro_data32)
    );

    qfit_sync_relation_bank #(
        .DEPTH(450), .DATA_W(10), .READ_LATENCY(1)
    ) u_generic10 (
        .clk_core(clk_core), .rst_core(rst_core),
        .write_valid(write_valid), .write_addr(write_addr),
        .write_data(write_data10), .read_valid(read_valid),
        .read_addr(read_addr), .read_data_valid(generic_valid10),
        .read_data(generic_data10)
    );

    qfit_fakeram45_relation_bank_450 #(.DATA_W(10)) u_macro10 (
        .clk_core(clk_core), .rst_core(rst_core),
        .write_valid(write_valid), .write_addr(write_addr),
        .write_data(write_data10), .read_valid(read_valid),
        .read_addr(read_addr), .read_data_valid(macro_valid10),
        .read_data(macro_data10)
    );

    always @(negedge clk_core) begin
        if (!rst_core) begin
            if (
                generic_valid32 !== macro_valid32
                || generic_valid10 !== macro_valid10
            )
                $fatal(1, "read valid mismatch");
            if (generic_valid32) begin
                if (generic_data32 !== macro_data32)
                    $fatal(1, "32-bit data mismatch");
                if (generic_data10 !== macro_data10)
                    $fatal(1, "10-bit data mismatch");
                responses = responses + 1;
            end
        end
    end

    initial begin
        write_valid = 1'b0;
        write_addr = '0;
        write_data32 = '0;
        write_data10 = '0;
        read_valid = 1'b0;
        read_addr = '0;
        responses = 0;

        repeat (4) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        for (integer address = 0; address < 450; address++) begin
            @(negedge clk_core);
            write_valid = 1'b1;
            write_addr = 9'(address);
            write_data32 = 32'h5a5a0000 ^ (32'(address) * 32'h10204081);
            write_data10 = 10'((address * 37) ^ (address >> 2));
        end
        @(negedge clk_core);
        write_valid = 1'b0;

        for (integer request = 0; request < 450; request++) begin
            @(negedge clk_core);
            read_valid = 1'b1;
            read_addr = 9'((request * 137) % 450);
        end
        @(negedge clk_core);
        read_valid = 1'b0;
        repeat (4) @(negedge clk_core);

        if (responses != 450)
            $fatal(1, "response count mismatch: %0d", responses);
        $display("PASS fakeram45 relation bank: 450 reads, 32b/10b exact");
        $finish;
    end
endmodule

`default_nettype wire
