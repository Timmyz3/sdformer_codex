`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_replay_plan_builder #(
    parameter int DUT_CSR_FORMAT = 2
);
    logic clk_core, rst_core;
    logic request_valid, request_ready, request_context_id;
    logic [4:0] request_head_id;
    logic [31:0] request_execution_tag;
    logic [5:0] request_head_index;
    logic [9:0] request_input_channel_base;
    logic [7:0] request_output_tile;
    logic request_last_head, request_last_output_tile;
    logic slot_inspect_valid, slot_inspect_ready, slot_inspect_context_id;
    logic [4:0] slot_inspect_head_id;
    logic slot_meta_valid, slot_meta_ready, slot_meta_exists;
    logic [31:0] slot_meta_tag;
    logic slot_meta_mode_is_csr;
    logic [1:0] slot_meta_format;
    logic [15:0] slot_meta_payload_bits, slot_meta_word_count;
    logic cache_lookup_valid, cache_lookup_ready, cache_lookup_context_id;
    logic [4:0] cache_lookup_head_id;
    logic [31:0] cache_lookup_expected_tag;
    logic cache_meta_valid, cache_meta_ready, cache_meta_hit;
    logic [31:0] cache_meta_tag;
    logic [7:0] cache_meta_term_count;
    logic plan_valid, plan_ready, plan_context_id;
    logic [4:0] plan_head_id;
    logic [31:0] plan_payload_tag, plan_execution_tag;
    logic [1:0] plan_route;
    logic [1:0] plan_format;
    logic [5:0] plan_head_index;
    logic [9:0] plan_input_channel_base;
    logic [7:0] plan_output_tile;
    logic plan_last_head, plan_last_output_tile, plan_cache_owned;
    logic plan_slot_replay_required;
    logic [6:0] plan_replay_start_word;
    logic [7:0] plan_resident_term_count;
    logic [12:0] plan_resident_event_count;
    logic reject_valid, reject_ready;
    logic [31:0] reject_payload_tag;
    logic [31:0] reject_execution_tag;
    logic protocol_error;
    logic [31:0] count_requests, count_resident_plans;
    logic [31:0] count_ipd_plans, count_fadc_plans;
    logic [31:0] count_raw_plans, count_rejects;

    gatestack_replay_plan_builder #(
        .CSR_FORMAT_FADC24(DUT_CSR_FORMAT)
    ) dut (.*);
    always #5 clk_core <= ~clk_core;

    task automatic send_request(input int id);
        begin
            @(negedge clk_core);
            request_context_id = id[0];
            request_head_id = 5'(id);
            request_execution_tag = 32'he500_0000 + 32'(id);
            request_head_index = 6'(id);
            request_input_channel_base = 10'(id * 32);
            request_output_tile = 8'(id + 2);
            request_last_head = id == 3;
            request_last_output_tile = id == 2;
            request_valid = 1'b1;
            do @(posedge clk_core); while (!request_ready);
            @(negedge clk_core);
            request_valid = 1'b0;
        end
    endtask

    task automatic send_slot_meta(
        input logic exists,
        input logic csr,
        input logic [1:0] format_value,
        input logic [31:0] tag,
        input logic [15:0] payload_bits,
        input logic [15:0] words
    );
        begin
            while (!slot_inspect_valid) @(posedge clk_core);
            if (slot_inspect_context_id != request_context_id ||
                slot_inspect_head_id != request_head_id)
                $fatal(1, "slot inspect identity mismatch");
            repeat (2) @(posedge clk_core);
            @(negedge clk_core);
            slot_inspect_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            slot_inspect_ready = 1'b0;
            slot_meta_exists = exists;
            slot_meta_mode_is_csr = csr;
            slot_meta_format = format_value;
            slot_meta_tag = tag;
            slot_meta_payload_bits = payload_bits;
            slot_meta_word_count = words;
            slot_meta_valid = 1'b1;
            do @(posedge clk_core); while (!slot_meta_ready);
            @(negedge clk_core);
            slot_meta_valid = 1'b0;
        end
    endtask

    task automatic send_cache_meta(
        input logic hit,
        input logic [31:0] tag,
        input logic [7:0] terms
    );
        begin
            while (!cache_lookup_valid) @(posedge clk_core);
            if (cache_lookup_context_id != request_context_id ||
                cache_lookup_head_id != request_head_id ||
                cache_lookup_expected_tag != slot_meta_tag)
                $fatal(1, "cache lookup identity mismatch");
            repeat (2) @(posedge clk_core);
            @(negedge clk_core);
            cache_lookup_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            cache_lookup_ready = 1'b0;
            cache_meta_hit = hit;
            cache_meta_tag = tag;
            cache_meta_term_count = terms;
            cache_meta_valid = 1'b1;
            do @(posedge clk_core); while (!cache_meta_ready);
            @(negedge clk_core);
            cache_meta_valid = 1'b0;
        end
    endtask

    task automatic accept_plan(
        input int id,
        input logic [1:0] route,
        input logic [1:0] format_value,
        input logic [31:0] payload_tag,
        input logic cache_owned,
        input logic slot_required,
        input logic [6:0] start_word,
        input logic [7:0] terms,
        input logic [12:0] events
    );
        begin
            while (!plan_valid) @(posedge clk_core);
            if (plan_context_id != id[0] || plan_head_id != 5'(id) ||
                plan_payload_tag != payload_tag ||
                plan_execution_tag != 32'he500_0000 + 32'(id) ||
                plan_route != route || plan_format != format_value ||
                plan_head_index != 6'(id) ||
                plan_input_channel_base != 10'(id * 32) ||
                plan_output_tile != 8'(id + 2) ||
                plan_last_head != (id == 3) ||
                plan_last_output_tile != (id == 2) ||
                plan_cache_owned != cache_owned ||
                plan_slot_replay_required != slot_required ||
                plan_replay_start_word != start_word ||
                plan_resident_term_count != terms ||
                plan_resident_event_count != events)
                $fatal(1, "plan mismatch id=%0d", id);
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            plan_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            plan_ready = 1'b0;
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        request_valid = 1'b0;
        request_context_id = 1'b0;
        request_head_id = '0;
        request_execution_tag = '0;
        request_head_index = '0;
        request_input_channel_base = '0;
        request_output_tile = '0;
        request_last_head = 1'b0;
        request_last_output_tile = 1'b0;
        slot_inspect_ready = 1'b0;
        slot_meta_valid = 1'b0;
        slot_meta_exists = 1'b0;
        slot_meta_tag = '0;
        slot_meta_mode_is_csr = 1'b0;
        slot_meta_format = 2'd0;
        slot_meta_payload_bits = '0;
        slot_meta_word_count = '0;
        cache_lookup_ready = 1'b0;
        cache_meta_valid = 1'b0;
        cache_meta_hit = 1'b0;
        cache_meta_tag = '0;
        cache_meta_term_count = '0;
        plan_ready = 1'b0;
        reject_ready = 1'b0;
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;

        send_request(0);
        send_slot_meta(1'b1, 1'b1, 2'd1, 32'hd500_0000, 296, 5);
        send_cache_meta(1'b1, 32'hd500_0000, 3);
        accept_plan(0, 2'd0, 2'd1, 32'hd500_0000,
                    1'b1, 1'b1, 4, 3, 5);

        send_request(1);
        send_slot_meta(1'b1, 1'b1, 2'd1, 32'hd500_0001, 296, 5);
        send_cache_meta(1'b0, 32'h0, 0);
        accept_plan(1, 2'd1, 2'd1, 32'hd500_0001,
                    1'b0, 1'b1, 0, 0, 0);

        send_request(2);
        send_slot_meta(1'b1, 1'b0, 2'd0, 32'hd500_0002, 6642, 104);
        accept_plan(2, 2'd2, 2'd0, 32'hd500_0002,
                    1'b0, 1'b1, 0, 0, 0);

        send_request(3);
        send_slot_meta(1'b0, 1'b0, 2'd0, 32'h0, 0, 0);
        while (!reject_valid) @(posedge clk_core);
        if (reject_payload_tag != 0 ||
            reject_execution_tag != 32'he500_0003)
            $fatal(1, "reject tag mismatch");
        repeat (2) @(posedge clk_core);
        @(negedge clk_core);
        reject_ready = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        reject_ready = 1'b0;

        // FADC is exact CSR replay but never performs an IPD cache lookup.
        send_request(4);
        send_slot_meta(1'b1, 1'b1, 2'd2, 32'hd500_0004, 512, 8);
        if (DUT_CSR_FORMAT == 2) begin
            accept_plan(4, 2'd1, 2'd2, 32'hd500_0004,
                        1'b0, 1'b1, 0, 0, 0);
        end else begin
            // A statically configured IPD decoder must reject FADC metadata
            // without issuing a descriptor-cache lookup.
            repeat (8) begin
                @(posedge clk_core);
                if (cache_lookup_valid)
                    $fatal(1, "unsupported FADC issued cache lookup");
            end
            if (!reject_valid || reject_payload_tag != 32'hd500_0004 ||
                reject_execution_tag != 32'he500_0004)
                $fatal(1, "unsupported FADC was not bounded-rejected");
            @(negedge clk_core);
            reject_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            reject_ready = 1'b0;
        end

        repeat (2) @(posedge clk_core);
        if (!protocol_error || count_requests != 5 ||
            count_resident_plans != 1 || count_ipd_plans != 1 ||
            count_fadc_plans != (DUT_CSR_FORMAT == 2 ? 1 : 0) ||
            count_raw_plans != 1 ||
            count_rejects != (DUT_CSR_FORMAT == 2 ? 1 : 2))
            $fatal(1, "plan builder counters mismatch");
        $display("PASS: replay plans resident=%0d ipd=%0d fadc=%0d raw=%0d reject=%0d",
                 count_resident_plans, count_ipd_plans, count_fadc_plans,
                 count_raw_plans, count_rejects);
        $finish;
    end

    initial begin
        repeat (10000) @(posedge clk_core);
        $fatal(1, "replay plan builder timeout");
    end
endmodule

`default_nettype wire
