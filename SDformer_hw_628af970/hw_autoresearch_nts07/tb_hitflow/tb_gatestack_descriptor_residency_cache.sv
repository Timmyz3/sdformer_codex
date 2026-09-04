`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_descriptor_residency_cache;
    localparam int CONTEXTS = 2;
    localparam int HEADS = 3;
    localparam int CACHE_TERMS = 8;

    logic clk_core;
    logic rst_core;
    logic fill_begin_valid;
    logic fill_begin_ready;
    logic fill_context_id;
    logic [1:0] fill_head_id;
    logic [31:0] fill_tag;
    logic [7:0] fill_term_count;
    logic fill_begin_cacheable;
    logic fill_entry_valid;
    logic fill_entry_ready;
    logic [8:0] fill_gate_code;
    logic [4:0] fill_lane_id;
    logic [7:0] fill_destination_count;
    logic fill_entry_last;
    logic lookup_valid;
    logic lookup_ready;
    logic lookup_context_id;
    logic [1:0] lookup_head_id;
    logic [31:0] lookup_expected_tag;
    logic lookup_meta_valid;
    logic lookup_meta_ready;
    logic lookup_hit;
    logic [31:0] lookup_tag;
    logic [7:0] lookup_term_count;
    logic lookup_entry_valid;
    logic lookup_entry_ready;
    logic [8:0] lookup_gate_code;
    logic [4:0] lookup_lane_id;
    logic [7:0] lookup_destination_count;
    logic [2:0] lookup_term_index;
    logic lookup_entry_last;
    logic release_valid;
    logic release_ready;
    logic release_context_id;
    logic [1:0] release_head_id;
    logic [31:0] release_expected_tag;
    logic [5:0] cache_valid_flat;
    logic protocol_error;
    logic [31:0] count_cached_heads;
    logic [31:0] count_bypass_heads;
    logic [31:0] count_lookup_hits;
    logic [31:0] count_lookup_misses;
    logic [31:0] count_releases;
    logic [31:0] count_release_noops;
    logic [31:0] count_release_tag_mismatches;

    gatestack_descriptor_residency_cache #(
        .CONTEXTS(CONTEXTS), .HEADS(HEADS), .CACHE_TERMS(CACHE_TERMS)
    ) dut (.*);
    always #5 clk_core <= ~clk_core;

    task automatic begin_fill(
        input logic context_id,
        input logic [1:0] head_id,
        input logic [31:0] tag,
        input logic [7:0] terms
    );
        begin
            @(negedge clk_core);
            fill_context_id = context_id;
            fill_head_id = head_id;
            fill_tag = tag;
            fill_term_count = terms;
            fill_begin_valid = 1'b1;
            #1;
            if (fill_begin_cacheable != (32'(terms) <= CACHE_TERMS)) begin
                $fatal(1, "fill cacheable boundary mismatch");
            end
            do @(posedge clk_core); while (!fill_begin_ready);
            @(negedge clk_core);
            fill_begin_valid = 1'b0;
        end
    endtask

    task automatic fill_cache(
        input logic context_id,
        input logic [1:0] head_id,
        input logic [31:0] tag,
        input int terms
    );
        begin
            begin_fill(context_id, head_id, tag, 8'(terms));
            if (terms <= CACHE_TERMS) begin
                for (int index = 0; index < terms; index = index + 1) begin
                    @(negedge clk_core);
                    fill_entry_valid = 1'b1;
                    fill_gate_code = 9'(index + 1);
                    fill_lane_id = 5'(index % 32);
                    fill_destination_count = 8'(index + 2);
                    fill_entry_last = index == terms - 1;
                    do @(posedge clk_core); while (!fill_entry_ready);
                    @(negedge clk_core);
                    fill_entry_valid = 1'b0;
                    fill_entry_last = 1'b0;
                end
            end
        end
    endtask

    task automatic lookup_cache(
        input logic context_id,
        input logic [1:0] head_id,
        input logic expected_hit,
        input logic [31:0] expected_tag,
        input int expected_terms
    );
        int entries;
        int cycles;
        begin
            entries = 0;
            cycles = 0;
            @(negedge clk_core);
            lookup_context_id = context_id;
            lookup_head_id = head_id;
            lookup_expected_tag = expected_tag;
            lookup_valid = 1'b1;
            do @(posedge clk_core); while (!lookup_ready);
            @(negedge clk_core);
            lookup_valid = 1'b0;
            repeat (2) @(posedge clk_core);
            if (!lookup_meta_valid || lookup_hit != expected_hit ||
                (expected_hit && (lookup_tag != expected_tag ||
                 lookup_term_count != 8'(expected_terms)))) begin
                $fatal(1, "lookup metadata mismatch");
            end
            @(negedge clk_core);
            lookup_meta_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            lookup_meta_ready = 1'b0;
            while (entries < expected_terms) begin
                @(negedge clk_core);
                lookup_entry_ready = (cycles % 3) != 1;
                @(posedge clk_core);
                if (lookup_entry_valid && lookup_entry_ready) begin
                    if (lookup_gate_code != 9'(entries + 1) ||
                        lookup_lane_id != 5'(entries % 32) ||
                        lookup_destination_count != 8'(entries + 2) ||
                        lookup_term_index != 3'(entries) ||
                        lookup_entry_last != (entries == expected_terms - 1)) begin
                        $fatal(1, "lookup entry mismatch %0d", entries);
                    end
                    entries = entries + 1;
                end
                cycles = cycles + 1;
            end
            @(negedge clk_core);
            lookup_entry_ready = 1'b0;
        end
    endtask

    task automatic release_cache(
        input logic context_id,
        input logic [1:0] head_id,
        input logic [31:0] expected_tag
    );
        begin
            @(negedge clk_core);
            release_context_id = context_id;
            release_head_id = head_id;
            release_expected_tag = expected_tag;
            release_valid = 1'b1;
            do @(posedge clk_core); while (!release_ready);
            @(negedge clk_core);
            release_valid = 1'b0;
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        fill_begin_valid = 1'b0;
        fill_context_id = '0;
        fill_head_id = '0;
        fill_tag = '0;
        fill_term_count = '0;
        fill_entry_valid = 1'b0;
        fill_gate_code = '0;
        fill_lane_id = '0;
        fill_destination_count = '0;
        fill_entry_last = 1'b0;
        lookup_valid = 1'b0;
        lookup_context_id = '0;
        lookup_head_id = '0;
        lookup_expected_tag = '0;
        lookup_meta_ready = 1'b0;
        lookup_entry_ready = 1'b0;
        release_valid = 1'b0;
        release_context_id = '0;
        release_head_id = '0;
        release_expected_tag = '0;
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;

        fill_cache(0, 0, 32'hca00_0000, 3);
        if (!cache_valid_flat[0]) $fatal(1, "cache fill failed");
        fork
            lookup_cache(0, 0, 1'b1, 32'hca00_0000, 3);
            fill_cache(1, 2, 32'hca01_0002, 8);
        join
        lookup_cache(1, 2, 1'b1, 32'hca01_0002, 8);
        lookup_cache(0, 0, 1'b0, 32'hbad0_0000, 0);

        // Depth+1 bypasses without consuming entry payload.
        begin_fill(0, 1, 32'hca00_0001, 9);
        lookup_cache(0, 1, 1'b0, 32'd0, 0);

        release_cache(0, 0, 32'hca00_0000);
        fill_cache(0, 0, 32'hca10_0000, 2);
        // A delayed release for the old payload must not evict the refill.
        release_cache(0, 0, 32'hca00_0000);
        if (!cache_valid_flat[0])
            $fatal(1, "stale release evicted newer cache line");
        lookup_cache(0, 0, 1'b1, 32'hca10_0000, 2);
        release_cache(0, 0, 32'hca10_0000);
        release_cache(1, 2, 32'hca01_0002);
        release_cache(0, 1, 32'hca00_0001);
        if (cache_valid_flat != 0 || !protocol_error ||
            count_cached_heads != 3 || count_bypass_heads != 1 ||
            count_lookup_hits != 3 || count_lookup_misses != 2 ||
            count_releases != 3 || count_release_noops != 1 ||
            count_release_tag_mismatches != 1) begin
            $fatal(1, "cache counters/release mismatch");
        end
        $display("PASS: descriptor residency cache cached=%0d bypass=%0d hits=%0d misses=%0d releases=%0d",
                 count_cached_heads, count_bypass_heads, count_lookup_hits,
                 count_lookup_misses, count_releases);
        $finish;
    end

    initial begin
        repeat (20000) @(posedge clk_core);
        $fatal(1, "descriptor cache TB timeout");
    end

endmodule

`default_nettype wire
