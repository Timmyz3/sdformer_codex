`timescale 1ns/1ps
`default_nettype none

module tb_h67_active_descriptor_store_macro_miter;
    localparam int DEPTH = 450;
    localparam int DATA_W = 20;
    localparam int ADDR_W = $clog2(DEPTH);

    logic clk_core;
    logic rst_core;
    logic window_start;
    logic [1:0] write_count;
    logic [ADDR_W-1:0] write0_addr;
    logic [DATA_W-1:0] write0_data;
    logic [ADDR_W-1:0] write1_addr;
    logic [DATA_W-1:0] write1_data;
    logic read_req_valid;
    logic [ADDR_W-1:0] read_req_addr;
    logic read_resp_ready;

    logic behavior_read_req_ready;
    logic behavior_read_resp_valid;
    logic [ADDR_W-1:0] behavior_read_resp_addr;
    logic [DATA_W-1:0] behavior_read_resp_data;
    logic behavior_protocol_error;
    logic macro_read_req_ready;
    logic macro_read_resp_valid;
    logic [ADDR_W-1:0] macro_read_resp_addr;
    logic [DATA_W-1:0] macro_read_resp_data;
    logic macro_protocol_error;

    logic [DATA_W-1:0] golden [0:DEPTH-1];
    integer seed;
    integer cycle_count;
    integer read_count;
    integer index;

    h67_banked_active_descriptor_store #(
        .DEPTH(DEPTH), .DATA_W(DATA_W), .MEMORY_IMPL(0)
    ) u_behavior (
        .clk_core, .rst_core, .window_start,
        .write_count, .write0_addr, .write0_data, .write1_addr, .write1_data,
        .read_req_valid, .read_req_ready(behavior_read_req_ready), .read_req_addr,
        .read_resp_valid(behavior_read_resp_valid), .read_resp_ready,
        .read_resp_addr(behavior_read_resp_addr),
        .read_resp_data(behavior_read_resp_data),
        .protocol_error(behavior_protocol_error)
    );

    h67_banked_active_descriptor_store #(
        .DEPTH(DEPTH), .DATA_W(DATA_W), .MEMORY_IMPL(1)
    ) u_macro (
        .clk_core, .rst_core, .window_start,
        .write_count, .write0_addr, .write0_data, .write1_addr, .write1_data,
        .read_req_valid, .read_req_ready(macro_read_req_ready), .read_req_addr,
        .read_resp_valid(macro_read_resp_valid), .read_resp_ready,
        .read_resp_addr(macro_read_resp_addr), .read_resp_data(macro_read_resp_data),
        .protocol_error(macro_protocol_error)
    );

    always #5 clk_core = ~clk_core;

    task automatic idle_cycle;
        begin
            @(negedge clk_core);
            write_count = 0;
            read_req_valid = 0;
        end
    endtask

    task automatic write_single(
        input logic [ADDR_W-1:0] addr,
        input logic [DATA_W-1:0] data
    );
        begin
            @(negedge clk_core);
            write_count = 1;
            write0_addr = addr;
            write0_data = data;
            write1_addr = '0;
            write1_data = '0;
            read_req_valid = 0;
            golden[addr] = data;
            @(posedge clk_core);
            #1;
            if (addr[0]) begin
                if (u_macro.g_fakeram45.u_bank1.mem[addr >> 1][31:DATA_W]
                    !== {(32-DATA_W){data[0]}})
                    $fatal(1, "bank1 padding mismatch addr=%0d", addr);
            end else begin
                if (u_macro.g_fakeram45.u_bank0.mem[addr >> 1][31:DATA_W]
                    !== {(32-DATA_W){data[0]}})
                    $fatal(1, "bank0 padding mismatch addr=%0d", addr);
            end
        end
    endtask

    task automatic write_pair(
        input logic [ADDR_W-1:0] even_addr,
        input logic [DATA_W-1:0] even_data,
        input logic [DATA_W-1:0] odd_data
    );
        begin
            @(negedge clk_core);
            write_count = 2;
            write0_addr = even_addr;
            write0_data = even_data;
            write1_addr = even_addr + 1'b1;
            write1_data = odd_data;
            read_req_valid = 0;
            golden[even_addr] = even_data;
            golden[even_addr + 1'b1] = odd_data;
            @(posedge clk_core);
            #1;
            if (u_macro.g_fakeram45.u_bank0.mem[even_addr >> 1][DATA_W-1:0]
                !== even_data)
                $fatal(1, "bank0 pair write mismatch addr=%0d", even_addr);
            if (u_macro.g_fakeram45.u_bank1.mem[even_addr >> 1][DATA_W-1:0]
                !== odd_data)
                $fatal(1, "bank1 pair write mismatch addr=%0d", even_addr + 1);
        end
    endtask

    task automatic read_one(input logic [ADDR_W-1:0] addr);
        begin
            @(negedge clk_core);
            write_count = 0;
            read_req_valid = 1;
            read_req_addr = addr;
            while (!behavior_read_req_ready || !macro_read_req_ready)
                @(negedge clk_core);
            @(posedge clk_core);
            #1;
            read_req_valid = 0;
            while (!behavior_read_resp_valid || !macro_read_resp_valid) begin
                read_resp_ready = $urandom(seed) & 1;
                @(posedge clk_core);
                #1;
            end
            if (behavior_read_resp_addr !== addr || macro_read_resp_addr !== addr)
                $fatal(1, "response address mismatch expected=%0d behavior=%0d macro=%0d",
                       addr, behavior_read_resp_addr, macro_read_resp_addr);
            if (behavior_read_resp_data !== golden[addr]
                || macro_read_resp_data !== golden[addr])
                $fatal(1, "response data mismatch addr=%0d expected=%h behavior=%h macro=%h",
                       addr, golden[addr], behavior_read_resp_data, macro_read_resp_data);
            read_count = read_count + 1;
            read_resp_ready = 1;
            @(posedge clk_core);
            #1;
        end
    endtask

    always @(negedge clk_core) begin
        cycle_count = cycle_count + 1;
        if (behavior_read_req_ready !== macro_read_req_ready)
            $fatal(1, "read ready mismatch cycle=%0d", cycle_count);
        if (behavior_read_resp_valid !== macro_read_resp_valid)
            $fatal(1, "read valid mismatch cycle=%0d", cycle_count);
        if (behavior_protocol_error !== macro_protocol_error)
            $fatal(1, "protocol error mismatch cycle=%0d", cycle_count);
        if (behavior_read_resp_valid && macro_read_resp_valid) begin
            if (behavior_read_resp_addr !== macro_read_resp_addr
                || behavior_read_resp_data !== macro_read_resp_data)
                $fatal(1, "macro/behavior response mismatch cycle=%0d", cycle_count);
        end
    end

    initial begin
        clk_core = 0;
        rst_core = 1;
        window_start = 0;
        write_count = 0;
        write0_addr = 0;
        write0_data = 0;
        write1_addr = 0;
        write1_data = 0;
        read_req_valid = 0;
        read_req_addr = 0;
        read_resp_ready = 1;
        seed = 32'h67a45020;
        cycle_count = 0;
        read_count = 0;
        for (index = 0; index < DEPTH; index = index + 1)
            golden[index] = '0;

        repeat (4) @(posedge clk_core);
        rst_core = 0;
        idle_cycle();

        // 覆盖单写、双bank同拍写、边界地址和padding的0/1两种模式。
        write_pair(0, 20'h00000, 20'hfffff);
        write_single(448, 20'ha5a5a);
        write_single(449, 20'h5a5a5);
        for (index = 2; index < 130; index = index + 2)
            write_pair(index[ADDR_W-1:0], $urandom(seed), $urandom(seed));
        for (index = 0; index < 130; index = index + 1)
            read_one(index[ADDR_W-1:0]);
        read_one(448);
        read_one(449);

        if (behavior_protocol_error || macro_protocol_error)
            $fatal(1, "unexpected protocol error");
        $display("PASS tb_h67_active_descriptor_store_macro_miter reads=%0d cycles=%0d",
                 read_count, cycle_count);
        $finish;
    end
endmodule

`default_nettype wire
