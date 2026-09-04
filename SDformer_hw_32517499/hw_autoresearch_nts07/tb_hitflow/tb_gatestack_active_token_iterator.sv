`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_active_token_iterator;
    localparam int TOKENS = 162;
    logic clk_core;
    logic rst_core;
    logic load_valid;
    logic load_ready;
    logic [31:0] load_tag;
    logic [TOKENS-1:0] load_active_token_mask;
    logic token_valid;
    logic token_ready;
    logic [31:0] token_tag;
    logic [7:0] token_id;
    logic token_last;
    logic done_valid;
    logic done_ready;
    logic [31:0] done_tag;
    logic [31:0] count_loads;
    logic [31:0] count_tokens;
    logic [31:0] count_stall_cycles;
    logic [31:0] prng_q;

    gatestack_active_token_iterator dut (.*);
    always #5 clk_core <= ~clk_core;

    function automatic int first_set(input logic [TOKENS-1:0] mask);
        first_set = -1;
        for (int index = 0; index < TOKENS; index = index + 1) begin
            if ((first_set < 0) && mask[index]) first_set = index;
        end
    endfunction

    task automatic run_mask(input logic [TOKENS-1:0] mask, input logic [31:0] tag);
        logic [TOKENS-1:0] expected;
        int index;
        int timeout;
        begin
            expected = mask;
            while (!load_ready) @(posedge clk_core);
            @(negedge clk_core);
            load_valid = 1'b1;
            load_tag = tag;
            load_active_token_mask = mask;
            @(posedge clk_core);
            @(negedge clk_core);
            load_valid = 1'b0;
            timeout = 0;
            while (!done_valid) begin
                @(negedge clk_core);
                prng_q = {prng_q[30:0], prng_q[31] ^ prng_q[21] ^ prng_q[1] ^ prng_q[0]};
                token_ready = prng_q[0] | prng_q[2];
                @(posedge clk_core);
                if (token_valid && token_ready) begin
                    index = first_set(expected);
                    if (index < 0 || token_id != index || token_tag != tag)
                        $fatal(1, "active-token iterator mismatch");
                    expected[index] = 1'b0;
                    if (token_last != (expected == '0))
                        $fatal(1, "active-token last mismatch");
                end
                timeout = timeout + 1;
                if (timeout > 5000) $fatal(1, "active-token timeout");
            end
            if (expected != '0 || done_tag != tag) $fatal(1, "active-token early done");
            @(negedge clk_core);
            token_ready = 1'b0;
            done_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            done_ready = 1'b0;
        end
    endtask

    initial begin
        logic [TOKENS-1:0] mask;
        clk_core = 1'b0;
        rst_core = 1'b1;
        load_valid = 1'b0;
        load_tag = '0;
        load_active_token_mask = '0;
        token_ready = 1'b0;
        done_ready = 1'b0;
        prng_q = 32'h725c_a11e;
        repeat (4) @(posedge clk_core);
        rst_core = 1'b0;
        run_mask('0, 32'h3000);
        run_mask({TOKENS{1'b1}}, 32'h3001);
        for (int trial = 0; trial < 100; trial = trial + 1) begin
            mask = '0;
            for (int index = 0; index < TOKENS; index = index + 1) begin
                prng_q = {prng_q[30:0], prng_q[31] ^ prng_q[21] ^ prng_q[1] ^ prng_q[0]};
                mask[index] = prng_q[0] && prng_q[4];
            end
            run_mask(mask, 32'h3100 + trial);
        end
        if (count_loads != 102) $fatal(1, "active-token load count mismatch");
        $display("PASS: active-token iterator loads=%0d tokens=%0d stalls=%0d",
                 count_loads, count_tokens, count_stall_cycles);
        $finish;
    end
endmodule

`default_nettype wire
