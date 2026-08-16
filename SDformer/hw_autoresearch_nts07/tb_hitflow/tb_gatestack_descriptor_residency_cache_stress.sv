`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_descriptor_residency_cache_stress;
    localparam int CONTEXTS = 2;
    localparam int HEADS = 4;
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
    logic [7:0] cache_valid_flat;
    logic protocol_error;
    logic [31:0] count_cached_heads;
    logic [31:0] count_bypass_heads;
    logic [31:0] count_lookup_hits;
    logic [31:0] count_lookup_misses;
    logic [31:0] count_releases;
    logic [31:0] count_release_noops;
    logic [31:0] count_release_tag_mismatches;

    logic model_valid [0:CONTEXTS-1][0:HEADS-1];
    logic [31:0] model_tag [0:CONTEXTS-1][0:HEADS-1];
    int model_terms [0:CONTEXTS-1][0:HEADS-1];
    logic [31:0] prng_q;
    int expected_cached;
    int expected_bypass;
    int expected_hits;
    int expected_misses;
    int expected_releases;
    int expected_noops;
    int expected_mismatches;

    gatestack_descriptor_residency_cache #(
        .CONTEXTS(CONTEXTS),
        .HEADS(HEADS),
        .CACHE_TERMS(CACHE_TERMS)
    ) dut (.*);

    always #5 clk_core <= ~clk_core;

    task automatic advance_prng;
        begin
            prng_q = {prng_q[30:0],
                prng_q[31] ^ prng_q[21] ^ prng_q[1] ^ prng_q[0]};
        end
    endtask

    task automatic fill_slot(
        input logic context_id,
        input logic [1:0] head_id,
        input logic [31:0] tag,
        input int terms
    );
        begin
            if (model_valid[context_id][head_id])
                $fatal(1, "test attempted fill of valid slot");
            @(negedge clk_core);
            fill_begin_valid = 1'b1;
            fill_context_id = 1'(context_id);
            fill_head_id = 2'(head_id);
            fill_tag = tag;
            fill_term_count = 8'(terms);
            do @(posedge clk_core); while (!fill_begin_ready);
            if (fill_begin_cacheable != (terms <= CACHE_TERMS))
                $fatal(1, "cacheable decision mismatch");
            @(negedge clk_core);
            fill_begin_valid = 1'b0;

            if (terms <= CACHE_TERMS) begin
                expected_cached = expected_cached + 1;
                for (int index = 0; index < terms; index = index + 1) begin
                    advance_prng();
                    repeat (32'(prng_q[1:0])) @(posedge clk_core);
                    @(negedge clk_core);
                    fill_entry_valid = 1'b1;
                    fill_gate_code = 9'((tag + index) & 32'h1ff);
                    fill_lane_id = 5'(index % 32);
                    fill_destination_count = 8'(1 + (index % 162));
                    fill_entry_last = index == terms - 1;
                    do @(posedge clk_core); while (!fill_entry_ready);
                    @(negedge clk_core);
                    fill_entry_valid = 1'b0;
                end
                model_valid[context_id][head_id] = 1'b1;
                model_tag[context_id][head_id] = tag;
                model_terms[context_id][head_id] = terms;
            end else begin
                expected_bypass = expected_bypass + 1;
            end
        end
    endtask

    task automatic lookup_slot(
        input int context_id,
        input int head_id,
        input logic [31:0] expected_tag
    );
        logic expected_hit;
        int terms;
        int stalls;
        begin
            expected_hit = model_valid[context_id][head_id] &&
                           model_tag[context_id][head_id] == expected_tag;
            terms = model_valid[context_id][head_id] ?
                    model_terms[context_id][head_id] : 0;
            @(negedge clk_core);
            lookup_valid = 1'b1;
            lookup_context_id = 1'(context_id);
            lookup_head_id = 2'(head_id);
            lookup_expected_tag = expected_tag;
            do @(posedge clk_core); while (!lookup_ready);
            @(negedge clk_core);
            lookup_valid = 1'b0;

            advance_prng();
            stalls = 1 + 32'(prng_q[2:0]);
            repeat (stalls) @(posedge clk_core);
            if (!lookup_meta_valid || lookup_hit != expected_hit ||
                lookup_term_count != 8'(terms) ||
                lookup_tag != (model_valid[context_id][head_id] ?
                               model_tag[context_id][head_id] : 32'd0))
                $fatal(1, "lookup metadata mismatch c=%0d h=%0d", context_id, head_id);
            @(negedge clk_core);
            lookup_meta_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            lookup_meta_ready = 1'b0;

            if (expected_hit) expected_hits = expected_hits + 1;
            else expected_misses = expected_misses + 1;

            if (expected_hit && terms != 0) begin
                for (int index = 0; index < terms; index = index + 1) begin
                    advance_prng();
                    repeat (32'(prng_q[1:0])) @(posedge clk_core);
                    if (!lookup_entry_valid || lookup_term_index != 3'(index) ||
                        lookup_gate_code != 9'((model_tag[context_id][head_id] + index) & 32'h1ff) ||
                        lookup_lane_id != 5'(index % 32) ||
                        lookup_destination_count != 8'(1 + (index % 162)) ||
                        lookup_entry_last != (index == terms - 1))
                        $fatal(1, "lookup entry mismatch c=%0d h=%0d index=%0d",
                               context_id, head_id, index);
                    @(negedge clk_core);
                    lookup_entry_ready = 1'b1;
                    @(posedge clk_core);
                    @(negedge clk_core);
                    lookup_entry_ready = 1'b0;
                end
            end
        end
    endtask

    task automatic release_slot(
        input logic context_id,
        input logic [1:0] head_id,
        input logic [31:0] expected_tag
    );
        begin
            @(negedge clk_core);
            release_valid = 1'b1;
            release_context_id = 1'(context_id);
            release_head_id = 2'(head_id);
            release_expected_tag = expected_tag;
            do @(posedge clk_core); while (!release_ready);
            @(negedge clk_core);
            release_valid = 1'b0;
            if (!model_valid[context_id][head_id]) begin
                expected_noops = expected_noops + 1;
            end else if (model_tag[context_id][head_id] == expected_tag) begin
                expected_releases = expected_releases + 1;
                model_valid[context_id][head_id] = 1'b0;
            end else begin
                expected_mismatches = expected_mismatches + 1;
            end
        end
    endtask

    initial begin
        int context_id;
        int head_id;
        int operation;
        int terms;
        logic [31:0] next_tag;
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
        prng_q = 32'hca7e_5715;
        expected_cached = 0;
        expected_bypass = 0;
        expected_hits = 0;
        expected_misses = 0;
        expected_releases = 0;
        expected_noops = 0;
        expected_mismatches = 0;
        for (int c = 0; c < CONTEXTS; c = c + 1)
            for (int h = 0; h < HEADS; h = h + 1) begin
                model_valid[c][h] = 1'b0;
                model_tag[c][h] = '0;
                model_terms[c][h] = 0;
            end
        repeat (4) @(posedge clk_core);
        rst_core = 1'b0;

        // Directed stale-release-after-refill sequence.
        fill_slot(0, 0, 32'h5100_0001, 4);
        release_slot(0, 0, 32'h5100_0001);
        fill_slot(0, 0, 32'h5100_0002, 6);
        release_slot(0, 0, 32'h5100_0001);
        lookup_slot(0, 0, 32'h5100_0002);
        release_slot(0, 0, 32'h5100_0002);

        for (int trial = 0; trial < 2000; trial = trial + 1) begin
            advance_prng();
            context_id = 32'(prng_q[0]);
            head_id = 32'(prng_q[2:1]);
            operation = 32'(prng_q[5:3]) % 4;
            next_tag = 32'h5200_0000 + trial;
            if (operation == 0 && !model_valid[context_id][head_id]) begin
                advance_prng();
                terms = 32'(prng_q[3:0]) % 10;
                fill_slot(1'(context_id), 2'(head_id), next_tag, terms);
            end else if (operation == 1) begin
                lookup_slot(context_id, head_id,
                    model_valid[context_id][head_id] ?
                    model_tag[context_id][head_id] : next_tag);
            end else if (operation == 2) begin
                release_slot(1'(context_id), 2'(head_id),
                    model_valid[context_id][head_id] ?
                    model_tag[context_id][head_id] : next_tag);
            end else if (model_valid[context_id][head_id]) begin
                release_slot(1'(context_id), 2'(head_id),
                             model_tag[context_id][head_id] ^ 32'h1);
                lookup_slot(context_id, head_id,
                            model_tag[context_id][head_id]);
            end else begin
                lookup_slot(context_id, head_id, next_tag);
            end
        end

        if (count_cached_heads != expected_cached ||
            count_bypass_heads != expected_bypass ||
            count_lookup_hits != expected_hits ||
            count_lookup_misses != expected_misses ||
            count_releases != expected_releases ||
            count_release_noops != expected_noops ||
            count_release_tag_mismatches != expected_mismatches ||
            !protocol_error) begin
            $fatal(1,
                "stress counters mismatch cached=%0d/%0d bypass=%0d/%0d hit=%0d/%0d miss=%0d/%0d rel=%0d/%0d noop=%0d/%0d stale=%0d/%0d err=%0d",
                count_cached_heads, expected_cached,
                count_bypass_heads, expected_bypass,
                count_lookup_hits, expected_hits,
                count_lookup_misses, expected_misses,
                count_releases, expected_releases,
                count_release_noops, expected_noops,
                count_release_tag_mismatches, expected_mismatches,
                protocol_error);
        end
        for (int c = 0; c < CONTEXTS; c = c + 1)
            for (int h = 0; h < HEADS; h = h + 1)
                if (cache_valid_flat[c * HEADS + h] != model_valid[c][h])
                    $fatal(1, "final shadow-state mismatch c=%0d h=%0d", c, h);

        $display(
            "PASS: descriptor cache stress ops=2006 cached=%0d bypass=%0d hits=%0d misses=%0d releases=%0d noops=%0d stale=%0d",
            count_cached_heads, count_bypass_heads, count_lookup_hits,
            count_lookup_misses, count_releases, count_release_noops,
            count_release_tag_mismatches);
        $finish;
    end

endmodule

`default_nettype wire
