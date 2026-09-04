`timescale 1ns/1ps
`default_nettype none

module tb_m492_hammer_adapter_edges;
    localparam int TAG_BITS = 24;
    localparam int CHANNEL_BITS = 12;
    localparam int EPOCH_BITS = 16;
    localparam int GENERATION_BITS = 32;
    localparam int LANES = 16;

    logic clk_core = 0;
    logic rst_core;
    always #1.5 clk_core = ~clk_core;

    logic core_req_valid, core_req_ready, core_req_accept;
    logic [EPOCH_BITS-1:0] core_req_epoch;
    logic [2:0] core_req_slot;
    logic [GENERATION_BITS-1:0] core_req_generation;
    logic [TAG_BITS-1:0] core_req_tag;
    logic [2:0] core_req_output_block, core_req_slice;
    logic [3:0] core_req_source_count;
    logic [7:0] core_req_bank_valid;
    logic [CHANNEL_BITS-1:0] core_req_source_channel [0:7];

    logic [7:0] bank_req_valid, bank_req_ready, bank_req_accept;
    logic [EPOCH_BITS-1:0] bank_req_epoch [0:7];
    logic [2:0] bank_req_slot [0:7];
    logic [GENERATION_BITS-1:0] bank_req_generation [0:7];
    logic [TAG_BITS-1:0] bank_req_tag [0:7];
    logic [2:0] bank_req_output_block [0:7], bank_req_slice [0:7];
    logic [CHANNEL_BITS-1:0] bank_req_source_channel [0:7];

    logic [7:0] bank_rsp_valid, bank_rsp_ready, bank_rsp_accept;
    logic [EPOCH_BITS-1:0] bank_rsp_epoch [0:7];
    logic [2:0] bank_rsp_slot [0:7];
    logic [GENERATION_BITS-1:0] bank_rsp_generation [0:7];
    logic [TAG_BITS-1:0] bank_rsp_tag [0:7];
    logic signed [7:0] bank_rsp_weight [0:7][0:LANES-1];

    logic core_rsp_valid, core_rsp_ready, core_rsp_accept;
    logic [EPOCH_BITS-1:0] core_rsp_epoch;
    logic [2:0] core_rsp_slot;
    logic [GENERATION_BITS-1:0] core_rsp_generation;
    logic [TAG_BITS-1:0] core_rsp_tag;
    logic [7:0] core_rsp_bank_valid;
    logic signed [7:0] core_rsp_weight [0:7][0:LANES-1];
    logic protocol_error, stale_response_seen, busy;
    logic [3:0] debug_live_slots;
    logic [31:0] debug_bundle_request_count, debug_bank_request_count;
    logic [31:0] debug_bank_response_count, debug_bundle_response_count;

    m490_fc2_bundle_to_8bank_cutthrough_adapter dut (.*);

    task automatic set_request(
        input logic [GENERATION_BITS-1:0] generation,
        input logic [TAG_BITS-1:0] tag,
        input logic [7:0] mask);
        begin
            core_req_epoch = 16'h12;
            core_req_slot = 0;
            core_req_generation = generation;
            core_req_tag = tag;
            core_req_output_block = 1;
            core_req_slice = 2;
            core_req_bank_valid = mask;
            core_req_source_count = $countones(mask);
            for (int bank = 0; bank < 8; bank++)
                core_req_source_channel[bank] = 12'(8*11 + bank);
        end
    endtask

    task automatic set_response(
        input logic [GENERATION_BITS-1:0] generation,
        input logic [TAG_BITS-1:0] tag,
        input logic [7:0] mask,
        input integer base);
        begin
            bank_rsp_valid = mask;
            for (int bank = 0; bank < 8; bank++) begin
                bank_rsp_epoch[bank] = 16'h12;
                bank_rsp_slot[bank] = 0;
                bank_rsp_generation[bank] = generation;
                bank_rsp_tag[bank] = tag;
                for (int lane = 0; lane < LANES; lane++)
                    bank_rsp_weight[bank][lane] = base + bank + lane;
            end
        end
    endtask

    initial begin
        rst_core = 1;
        core_req_valid = 0;
        core_rsp_ready = 0;
        bank_req_ready = 0;
        bank_rsp_valid = 0;
        set_request(1, 24'h100001, 8'hff);
        set_response(1, 24'h100001, 0, 0);
        repeat (3) @(negedge clk_core);
        rst_core = 0;

        // Unequal ready: accept the atomic core request once, distribute four
        // physical banks now, retain the other four, and never duplicate.
        @(negedge clk_core);
        set_request(1, 24'h100001, 8'hff);
        core_req_valid = 1;
        bank_req_ready = 8'h0f;
        #0.1;
        if (!core_req_accept || bank_req_accept !== 8'h0f)
            $fatal(1, "partial first distribution failed");
        @(posedge clk_core);
        @(negedge clk_core);
        core_req_valid = 0;
        bank_req_ready = 0;
        #0.1;
        if (bank_req_valid !== 8'hf0 || bank_req_accept != 0)
            $fatal(1, "pending-bank hold failed");
        @(posedge clk_core);
        @(negedge clk_core);
        bank_req_ready = 8'hf0;
        #0.1;
        if (bank_req_valid !== 8'hf0 || bank_req_accept !== 8'hf0)
            $fatal(1, "partial second distribution failed");
        @(posedge clk_core);

        // Complete through the bypass while the core is stalled. The exact
        // visible payload must remain stable after bank inputs disappear.
        @(negedge clk_core);
        bank_req_ready = 0;
        core_rsp_ready = 0;
        set_response(1, 24'h100001, 8'hff, 10);
        #0.1;
        if (!core_rsp_valid || core_rsp_accept
                || core_rsp_weight[7][15] !== 8'(10+7+15))
            $fatal(1, "cutthrough stall presentation failed");
        @(posedge clk_core);
        @(negedge clk_core);
        bank_rsp_valid = 0;
        for (int bank = 0; bank < 8; bank++)
            for (int lane = 0; lane < LANES; lane++)
                bank_rsp_weight[bank][lane] = -8'sd99;
        #0.1;
        if (!core_rsp_valid || core_rsp_tag !== 24'h100001
                || core_rsp_weight[7][15] !== 8'(10+7+15))
            $fatal(1, "stored stall payload changed");
        repeat (2) @(posedge clk_core);

        // Retire the old response and allocate the exact slot to a new
        // generation on the same edge. Old response state must not survive.
        @(negedge clk_core);
        set_request(2, 24'h200002, 8'h03);
        core_req_valid = 1;
        core_rsp_ready = 1;
        bank_req_ready = 8'h03;
        #0.1;
        if (!core_rsp_accept || !core_req_accept
                || core_rsp_slot != core_req_slot)
            $fatal(1, "same-cycle retire/reuse failed");
        @(posedge clk_core);
        @(negedge clk_core);
        core_req_valid = 0;
        bank_req_ready = 0;
        #0.1;
        if (protocol_error || debug_live_slots != 1 || core_rsp_valid)
            $fatal(1, "replacement slot state corrupt");

        set_response(2, 24'h200002, 8'h03, 40);
        #0.1;
        if (!core_rsp_accept || core_rsp_tag !== 24'h200002
                || core_rsp_bank_valid !== 8'h03
                || core_rsp_weight[0][0] !== 8'd40
                || core_rsp_weight[1][15] !== 8'd56)
            $fatal(1, "replacement payload contaminated");
        @(posedge clk_core);
        @(negedge clk_core);
        bank_rsp_valid = 0;
        #0.1;
        if (protocol_error || stale_response_seen || debug_live_slots != 0
                || debug_bundle_request_count != 2
                || debug_bank_request_count != 10
                || debug_bank_response_count != 10
                || debug_bundle_response_count != 2)
            $fatal(1, "conservation/final state failed");

        $display("PASS M492 HAMMER adapter partial-ready/stall/reuse numeric exact");
        $finish;
    end
endmodule

`default_nettype wire
