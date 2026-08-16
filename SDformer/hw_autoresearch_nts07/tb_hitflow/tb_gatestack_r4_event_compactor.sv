`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_r4_event_compactor;
    localparam int LANES = 32;
`ifdef TB_WAYS
    localparam int WAYS = `TB_WAYS;
`else
    localparam int WAYS = 4;
`endif
    localparam int COUNT_W = $clog2(WAYS + 1);
    logic clk_core;
    logic rst_core;
    logic token_valid;
    logic token_ready;
    logic [31:0] token_tag;
    logic [7:0] token_id;
    logic [1:0] token_slot_id;
    logic [LANES-1:0] token_k_bits;
    logic event_valid;
    logic event_ready;
    logic [31:0] event_tag;
    logic [7:0] event_token_id;
    logic [1:0] event_slot_id;
    logic [WAYS-1:0] event_lane_valid;
    logic [WAYS*5-1:0] event_lane_ids;
    logic [COUNT_W-1:0] event_count;
    logic event_last_for_token;
    logic [31:0] count_tokens;
    logic [31:0] count_events;
    logic [31:0] count_event_stall_cycles;
    logic [31:0] prng_q;
    int expected_total_events;

    gatestack_event_compactor #(.WAYS(WAYS)) dut (.*);
    always #5 clk_core <= ~clk_core;

    function automatic int first_set(input logic [LANES-1:0] mask);
        first_set = -1;
        for (int index = 0; index < LANES; index = index + 1) begin
            if ((first_set < 0) && mask[index]) first_set = index;
        end
    endfunction

    task automatic run_token(
        input logic [LANES-1:0] mask,
        input logic [7:0] id,
        input logic [1:0] slot,
        input logic [31:0] tag
    );
        logic [LANES-1:0] expected;
        int lane;
        int emitted;
        int timeout;
        begin
            expected = mask;
            while (!token_ready) @(posedge clk_core);
            @(negedge clk_core);
            token_valid = 1'b1;
            token_k_bits = mask;
            token_id = id;
            token_slot_id = slot;
            token_tag = tag;
            @(posedge clk_core);
            @(negedge clk_core);
            token_valid = 1'b0;
            timeout = 0;
            while (expected != '0) begin
                prng_q = {prng_q[30:0], prng_q[31] ^ prng_q[21] ^ prng_q[1] ^ prng_q[0]};
                event_ready = prng_q[0] | prng_q[3];
                @(posedge clk_core);
                if (event_valid && event_ready) begin
                    if (event_tag != tag || event_token_id != id || event_slot_id != slot)
                        $fatal(1, "R4 event metadata mismatch");
                    emitted = 0;
                    for (int way = 0; way < WAYS; way = way + 1) begin
                        if (event_lane_valid[way]) begin
                            lane = first_set(expected);
                            if (lane < 0 || 32'(event_lane_ids[way*5 +: 5]) != lane)
                                $fatal(1, "R4 lane order mismatch");
                            expected[lane] = 1'b0;
                            emitted = emitted + 1;
                        end
                    end
                    if (32'(event_count) != emitted ||
                        event_last_for_token != (expected == '0))
                        $fatal(1, "R4 event count/last mismatch");
                    expected_total_events = expected_total_events + emitted;
                end
                @(negedge clk_core);
                timeout = timeout + 1;
                if (timeout > 1000) $fatal(1, "R4 timeout");
            end
            event_ready = 1'b0;
            while (!token_ready) @(posedge clk_core);
        end
    endtask

    initial begin
        logic [LANES-1:0] mask;
        clk_core = 1'b0;
        rst_core = 1'b1;
        token_valid = 1'b0;
        token_tag = '0;
        token_id = '0;
        token_slot_id = '0;
        token_k_bits = '0;
        event_ready = 1'b0;
        prng_q = 32'h4e71_c0de;
        expected_total_events = 0;
        repeat (4) @(posedge clk_core);
        rst_core = 1'b0;
        run_token('0, 8'd0, 2'd0, 32'h4000);
        run_token({LANES{1'b1}}, 8'd1, 2'd3, 32'h4001);
        for (int trial = 0; trial < 500; trial = trial + 1) begin
            mask = '0;
            for (int lane = 0; lane < LANES; lane = lane + 1) begin
                prng_q = {prng_q[30:0], prng_q[31] ^ prng_q[21] ^ prng_q[1] ^ prng_q[0]};
                mask[lane] = prng_q[0] && prng_q[2];
            end
            run_token(mask, 8'(trial), 2'(trial), 32'h4100 + trial);
        end
        if (count_tokens != 502 || count_events != expected_total_events)
            $fatal(1, "R4 counters mismatch tokens=%0d events=%0d expected=%0d",
                   count_tokens, count_events, expected_total_events);
        $display("PASS: R%0d event compactor tokens=%0d events=%0d stalls=%0d",
                 WAYS, count_tokens, count_events, count_event_stall_cycles);
        $finish;
    end
endmodule

`default_nettype wire
