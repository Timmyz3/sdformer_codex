`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_obi_iterator;
    localparam int SLOTS = 4;
    localparam int LANES = 32;
    localparam int MASK_W = SLOTS * LANES;

    logic clk_core;
    logic rst_core;
    logic load_valid;
    logic load_ready;
    logic [31:0] load_tag;
    logic [MASK_W-1:0] load_occupied_mask;
    logic entry_valid;
    logic entry_ready;
    logic [31:0] entry_tag;
    logic [1:0] entry_slot_id;
    logic [4:0] entry_lane_id;
    logic entry_last;
    logic done_valid;
    logic done_ready;
    logic [31:0] done_tag;
    logic [31:0] count_loads;
    logic [31:0] count_entries;
    logic [31:0] count_entry_stall_cycles;

    logic [31:0] prng_q;

    gatestack_obi_iterator #(
        .SLOTS(SLOTS),
        .LANES(LANES)
    ) dut (.*);

    always #5 clk_core <= ~clk_core;

    function automatic int first_set(input logic [MASK_W-1:0] mask);
        first_set = -1;
        for (int index = 0; index < MASK_W; index = index + 1) begin
            if ((first_set < 0) && mask[index]) begin
                first_set = index;
            end
        end
    endfunction

    task automatic run_mask(
        input logic [MASK_W-1:0] mask,
        input logic [31:0] tag
    );
        logic [MASK_W-1:0] expected;
        int expected_index;
        int cycles;
        begin
            expected = mask;
            while (!load_ready) @(posedge clk_core);
            @(negedge clk_core);
            load_valid = 1'b1;
            load_tag = tag;
            load_occupied_mask = mask;
            @(posedge clk_core);
            @(negedge clk_core);
            load_valid = 1'b0;

            cycles = 0;
            while (!done_valid) begin
                @(negedge clk_core);
                prng_q = {prng_q[30:0], prng_q[31] ^ prng_q[21] ^ prng_q[1] ^ prng_q[0]};
                entry_ready = prng_q[0] | prng_q[3];
                done_ready = 1'b0;
                @(posedge clk_core);
                if (entry_valid && entry_ready) begin
                    expected_index = first_set(expected);
                    if (expected_index < 0) begin
                        $fatal(1, "OBI emitted an entry for empty expected mask");
                    end
                    if ((entry_slot_id * LANES + 32'(entry_lane_id)) != expected_index) begin
                        $fatal(1, "OBI order mismatch expected=%0d got=%0d",
                               expected_index,
                               entry_slot_id * LANES + 32'(entry_lane_id));
                    end
                    if (entry_tag != tag) begin
                        $fatal(1, "OBI tag mismatch");
                    end
                    expected[expected_index] = 1'b0;
                    if (entry_last != (expected == '0)) begin
                        $fatal(1, "OBI entry_last mismatch");
                    end
                end
                cycles = cycles + 1;
                if (cycles > 5000) begin
                    $fatal(1, "OBI timeout");
                end
            end
            if (expected != '0 || done_tag != tag) begin
                $fatal(1, "OBI done before all entries retired");
            end
            @(negedge clk_core);
            entry_ready = 1'b0;
            done_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            done_ready = 1'b0;
        end
    endtask

    initial begin
        logic [MASK_W-1:0] mask;
        clk_core = 1'b0;
        rst_core = 1'b1;
        load_valid = 1'b0;
        load_tag = '0;
        load_occupied_mask = '0;
        entry_ready = 1'b0;
        done_ready = 1'b0;
        prng_q = 32'h1ace_b00c;
        repeat (4) @(posedge clk_core);
        rst_core = 1'b0;

        run_mask('0, 32'h1000);
        mask = '0;
        mask[127] = 1'b1;
        run_mask(mask, 32'h1001);
        mask = '0;
        mask[0] = 1'b1;
        mask[31] = 1'b1;
        mask[32] = 1'b1;
        mask[63] = 1'b1;
        mask[95] = 1'b1;
        mask[127] = 1'b1;
        run_mask(mask, 32'h1002);
        run_mask({MASK_W{1'b1}}, 32'h1003);

        for (int trial = 0; trial < 100; trial = trial + 1) begin
            mask = '0;
            for (int index = 0; index < MASK_W; index = index + 1) begin
                prng_q = {prng_q[30:0], prng_q[31] ^ prng_q[21] ^ prng_q[1] ^ prng_q[0]};
                mask[index] = prng_q[0] && prng_q[5];
            end
            run_mask(mask, 32'h2000 + trial);
        end

        if (count_loads != 104) begin
            $fatal(1, "OBI load counter mismatch: %0d", count_loads);
        end
        $display("PASS: GateStack OBI iterator loads=%0d entries=%0d stalls=%0d",
                 count_loads, count_entries, count_entry_stall_cycles);
        $finish;
    end

endmodule

`default_nettype wire
