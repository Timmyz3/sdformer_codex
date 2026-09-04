`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_ipd_cache_fill_adapter;
    logic clk_core, rst_core;
    logic begin_valid, begin_ready;
    logic [2:0] begin_head_id;
    logic [15:0] begin_tag;
    logic [7:0] begin_term_count;
    logic begin_cache_allowed;
    logic entry_valid, entry_ready;
    logic [8:0] entry_gate_code;
    logic [4:0] entry_lane_id;
    logic [7:0] entry_destination_count;
    logic entry_last;
    logic cache_begin_valid, cache_begin_ready;
    logic [2:0] cache_begin_head_id;
    logic [15:0] cache_begin_tag;
    logic [7:0] cache_begin_term_count;
    logic cache_begin_cacheable;
    logic cache_entry_valid, cache_entry_ready;
    logic [8:0] cache_entry_gate_code;
    logic [4:0] cache_entry_lane_id;
    logic [7:0] cache_entry_destination_count;
    logic cache_entry_last, session_active, protocol_error;
    logic [31:0] count_cacheable_fills, count_bypass_fills;
    int cache_entries;

    gatestack_ipd_cache_fill_adapter #(
        .TAG_W(16), .HEAD_ID_W(3)
    ) dut (.*);
    always #5 clk_core <= ~clk_core;

    always @(posedge clk_core) begin
        if (rst_core) begin
            cache_entries <= 0;
        end else if (cache_entry_valid && cache_entry_ready) begin
            cache_entries <= cache_entries + 1;
        end
    end

    task automatic start_fill(
        input logic [2:0] head_id,
        input logic [15:0] tag_value,
        input logic [7:0] terms,
        input logic cacheable,
        input logic cache_allowed
    );
        begin
            @(negedge clk_core);
            begin_head_id = head_id;
            begin_tag = tag_value;
            begin_term_count = terms;
            begin_cache_allowed = cache_allowed;
            cache_begin_cacheable = cacheable;
            begin_valid = 1'b1;
            if (cache_allowed) begin
                repeat (2) @(posedge clk_core);
                if (!cache_begin_valid || cache_begin_head_id != head_id ||
                    cache_begin_tag != tag_value ||
                    cache_begin_term_count != terms)
                    $fatal(1, "fill begin metadata mismatch");
                @(negedge clk_core);
                cache_begin_ready = 1'b1;
                do @(posedge clk_core); while (!begin_ready);
            end else begin
                #1;
                if (!begin_ready || cache_begin_valid)
                    $fatal(1, "non-IPD fill did not bypass cache");
                @(posedge clk_core);
            end
            @(negedge clk_core);
            begin_valid = 1'b0;
            cache_begin_ready = 1'b0;
        end
    endtask

    task automatic send_entry(
        input int index_value,
        input logic last_value,
        input logic stall_cache
    );
        begin
            @(negedge clk_core);
            entry_gate_code = 9'(index_value + 1);
            entry_lane_id = 5'(index_value);
            entry_destination_count = 8'(index_value + 2);
            entry_last = last_value;
            entry_valid = 1'b1;
            cache_entry_ready = !stall_cache;
            if (stall_cache) begin
                repeat (2) @(posedge clk_core);
                if (entry_ready || !cache_entry_valid)
                    $fatal(1, "cacheable entry ignored cache backpressure");
                @(negedge clk_core);
                cache_entry_ready = 1'b1;
            end
            do @(posedge clk_core); while (!entry_ready);
            if (cache_entry_valid &&
                (cache_entry_gate_code != entry_gate_code ||
                 cache_entry_lane_id != entry_lane_id ||
                 cache_entry_destination_count !=
                    entry_destination_count ||
                 cache_entry_last != entry_last))
                $fatal(1, "fill entry metadata mismatch");
            @(negedge clk_core);
            entry_valid = 1'b0;
            entry_last = 1'b0;
            cache_entry_ready = 1'b0;
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        begin_valid = 1'b0;
        begin_head_id = '0;
        begin_tag = '0;
        begin_term_count = '0;
        begin_cache_allowed = 1'b1;
        entry_valid = 1'b0;
        entry_gate_code = '0;
        entry_lane_id = '0;
        entry_destination_count = '0;
        entry_last = 1'b0;
        cache_begin_ready = 1'b0;
        cache_begin_cacheable = 1'b0;
        cache_entry_ready = 1'b0;
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;

        start_fill(3'd1, 16'ha101, 8'd2, 1'b1, 1'b1);
        send_entry(0, 1'b0, 1'b1);
        send_entry(1, 1'b1, 1'b0);
        if (session_active)
            $fatal(1, "cacheable fill did not retire");

        // Bypass mode must consume entries even while cache entry ready is 0.
        start_fill(3'd2, 16'ha102, 8'd3, 1'b0, 1'b1);
        for (int index = 0; index < 3; index = index + 1) begin
            @(negedge clk_core);
            entry_gate_code = 9'(index + 4);
            entry_lane_id = 5'(index + 1);
            entry_destination_count = 8'(index + 1);
            entry_last = index == 2;
            entry_valid = 1'b1;
            cache_entry_ready = 1'b0;
            do @(posedge clk_core); while (!entry_ready);
            if (cache_entry_valid)
                $fatal(1, "bypass fill wrote cache entry");
            @(negedge clk_core);
            entry_valid = 1'b0;
        end

        // FADC descriptor taps are drained locally without a cache transaction.
        start_fill(3'd4, 16'ha104, 8'd1, 1'b1, 1'b0);
        send_entry(0, 1'b1, 1'b0);

        start_fill(3'd3, 16'ha103, 8'd0, 1'b1, 1'b1);
        repeat (3) @(posedge clk_core);
        if (session_active || protocol_error || count_cacheable_fills != 2 ||
            count_bypass_fills != 2 || cache_entries != 2)
            $fatal(1, "IPD fill adapter counters mismatch");
        $display("PASS: IPD auto-fill cacheable=%0d bypass=%0d entries=%0d",
                 count_cacheable_fills, count_bypass_fills, cache_entries);
        $finish;
    end

    initial begin
        repeat (2000) @(posedge clk_core);
        $fatal(1, "IPD cache fill adapter timeout");
    end
endmodule

`default_nettype wire
