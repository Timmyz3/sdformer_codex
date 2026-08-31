`timescale 1ns/1ps
`default_nettype none

// M803/C2 R16 unit-level adversarial campaign for the channel-split K8 bundle adapter.  This TB
// drives only legal ports: no hierarchical state writes, force, deposit, bind
// writes, or simulation-only RTL hooks.  The full M519 K1/K8/K1x8 workload TB
// remains a separate runner phase and protects the r2 numeric/cycle contract.
module tb_m803_c2_r16_channel_split_adapter_attacks;
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
    logic [2:0] bank_req_output_block [0:7];
    logic [2:0] bank_req_slice [0:7];
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
    logic [31:0] debug_bundle_request_count;
    logic [31:0] debug_bank_request_count;
    logic [31:0] debug_bank_response_count;
    logic [31:0] debug_bundle_response_count;

    int errors = 0;
    int attack_classes = 0;
    int reset_cases = 0;
    int legal_response_on_request_fault = 0;
    int request_side_effect_violations = 0;
    int response_side_effect_violations = 0;
    int sticky_quiescent_checks = 0;
    int normal_requests = 0;
    int normal_responses = 0;
    int same_cycle_reuse_cases = 0;

    m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter dut (.*);

    function automatic logic [3:0] popcount8(input logic [7:0] value);
        logic [3:0] count;
        begin
            count = 0;
            for (int bank = 0; bank < 8; bank++)
                count = count + value[bank];
            return count;
        end
    endfunction

    task automatic expect_true(input logic condition, input string message);
        if (!condition) begin
            errors++;
            $error("M803 R16 attack TB: %s", message);
        end
    endtask

    task automatic clear_request;
        core_req_valid = 0;
        core_req_epoch = 0;
        core_req_slot = 0;
        core_req_generation = 0;
        core_req_tag = 0;
        core_req_output_block = 0;
        core_req_slice = 0;
        core_req_source_count = 0;
        core_req_bank_valid = 0;
        for (int bank = 0; bank < 8; bank++)
            core_req_source_channel[bank] = bank;
    endtask

    task automatic clear_response;
        bank_rsp_valid = 0;
        for (int bank = 0; bank < 8; bank++) begin
            bank_rsp_epoch[bank] = 0;
            bank_rsp_slot[bank] = 0;
            bank_rsp_generation[bank] = 0;
            bank_rsp_tag[bank] = 0;
            for (int lane = 0; lane < LANES; lane++)
                bank_rsp_weight[bank][lane] = 0;
        end
    endtask

    task automatic set_legal_request(
        input logic [2:0] slot,
        input logic [7:0] mask,
        input logic [GENERATION_BITS-1:0] generation,
        input logic [TAG_BITS-1:0] tag,
        input logic [2:0] output_block,
        input logic [2:0] slice
    );
        core_req_valid = 1;
        core_req_epoch = 16'h0051;
        core_req_slot = slot;
        core_req_generation = generation;
        core_req_tag = tag;
        core_req_output_block = output_block;
        core_req_slice = slice;
        core_req_source_count = popcount8(mask);
        core_req_bank_valid = mask;
        for (int bank = 0; bank < 8; bank++)
            core_req_source_channel[bank] = CHANNEL_BITS'(bank + 8);
    endtask

    task automatic set_legal_response(
        input logic [2:0] slot,
        input logic [7:0] mask,
        input logic [GENERATION_BITS-1:0] generation,
        input logic [TAG_BITS-1:0] tag,
        input logic signed [7:0] base_weight
    );
        bank_rsp_valid = mask;
        for (int bank = 0; bank < 8; bank++) begin
            bank_rsp_epoch[bank] = 16'h0051;
            bank_rsp_slot[bank] = slot;
            bank_rsp_generation[bank] = generation;
            bank_rsp_tag[bank] = tag;
            for (int lane = 0; lane < LANES; lane++)
                bank_rsp_weight[bank][lane]
                    = base_weight + $signed(bank) - $signed(lane);
        end
    endtask

    task automatic reset_dut;
        @(negedge clk_core);
        rst_core = 1;
        clear_request();
        clear_response();
        bank_req_ready = 8'hff;
        core_rsp_ready = 1;
        repeat (2) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 0;
        #0.1;
        expect_true(!protocol_error && !stale_response_seen,
            "reset must clear sticky protocol/stale state");
        expect_true(debug_live_slots == 0
            && debug_bundle_request_count == 0
            && debug_bank_request_count == 0
            && debug_bank_response_count == 0
            && debug_bundle_response_count == 0,
            "reset must clear all request/response ledgers");
        reset_cases++;
    endtask

    task automatic issue_legal_request(
        input logic [2:0] slot,
        input logic [7:0] mask,
        input logic [GENERATION_BITS-1:0] generation,
        input logic [TAG_BITS-1:0] tag
    );
        @(negedge clk_core);
        bank_req_ready = 8'hff;
        set_legal_request(slot, mask, generation, tag, 3'd1, 3'd2);
        #0.1;
        expect_true(core_req_accept,
            "legal request must be accepted on the normal path");
        expect_true(bank_req_accept == mask,
            "all ready banks must accept the legal request");
        @(posedge clk_core);
        @(negedge clk_core);
        clear_request();
        #0.1;
        expect_true(debug_live_slots == 1,
            "legal request must own exactly one response slot");
        expect_true(debug_bundle_request_count == 1
            && debug_bank_request_count == popcount8(mask),
            "normal request ledgers must count accepted traffic");
        normal_requests++;
    endtask

    task automatic check_sticky_quiescence;
        logic [31:0] req_before, bank_req_before;
        logic [31:0] bank_rsp_before, bundle_rsp_before;
        req_before = debug_bundle_request_count;
        bank_req_before = debug_bank_request_count;
        bank_rsp_before = debug_bank_response_count;
        bundle_rsp_before = debug_bundle_response_count;
        @(negedge clk_core);
        clear_request();
        clear_response();
        bank_req_ready = 8'hff;
        core_rsp_ready = 1;
        #0.1;
        expect_true(protocol_error,
            "an attack must latch sticky fault by the following cycle");
        expect_true(!core_req_accept && bank_req_accept == 0
            && bank_rsp_accept == 0 && !core_rsp_accept,
            "sticky fault must quiesce all four accept ledgers");
        @(posedge clk_core);
        @(negedge clk_core);
        #0.1;
        expect_true(debug_bundle_request_count == req_before
            && debug_bank_request_count == bank_req_before
            && debug_bank_response_count == bank_rsp_before
            && debug_bundle_response_count == bundle_rsp_before,
            "sticky-fault cycle must have zero new external side effects");
        sticky_quiescent_checks++;
    endtask

    task automatic request_only_attack_and_check(input int attack_id);
        logic [31:0] req_before, bank_before;
        req_before = debug_bundle_request_count;
        bank_before = debug_bank_request_count;
        #0.1;
        expect_true(protocol_error && !core_req_accept
            && bank_req_valid == 0 && bank_req_accept == 0,
            "malformed request must be rejected with zero bank issue");
        @(posedge clk_core);
        @(negedge clk_core);
        clear_request();
        #0.1;
        if (debug_bundle_request_count != req_before
                || debug_bank_request_count != bank_before)
            request_side_effect_violations++;
        expect_true(debug_bundle_request_count == req_before
            && debug_bank_request_count == bank_before,
            "malformed request must not mutate request ledgers");
        attack_classes++;
        check_sticky_quiescence();
    endtask

    initial begin : campaign
        rst_core = 1;
        clear_request();
        clear_response();
        bank_req_ready = 8'hff;
        core_rsp_ready = 1;

        // A1: final legal response and malformed request on the same edge.
        // The request has zero side effects; the independently owned response
        // retires exactly once through the cut-through path.
        reset_dut();
        issue_legal_request(3'd0, 8'ha5, 32'h100, 24'h510001);
        @(negedge clk_core);
        set_legal_response(3'd0, 8'ha5, 32'h100, 24'h510001, 8'sd17);
        set_legal_request(3'd1, 8'h00, 32'h101, 24'h510002, 3'd1, 3'd2);
        core_req_source_count = 0;
        #0.1;
        expect_true(protocol_error && !core_req_accept
            && bank_req_accept == 0,
            "same-edge malformed request must be rejected");
        expect_true(bank_rsp_accept == 8'ha5 && core_rsp_accept,
            "legal multi-bank cut-through response must survive request fault");
        expect_true(core_rsp_slot == 0 && core_rsp_tag == 24'h510001
            && core_rsp_bank_valid == 8'ha5
            && core_rsp_weight[0][0] == 8'sd17
            && core_rsp_weight[7][0] == 8'sd24,
            "surviving response identity/payload must remain exact");
        @(posedge clk_core);
        @(negedge clk_core);
        clear_request();
        clear_response();
        #0.1;
        expect_true(debug_live_slots == 0
            && debug_bundle_response_count == 1
            && debug_bank_response_count == 4,
            "surviving multi-bank response must retire once from both ledgers");
        legal_response_on_request_fault++;
        normal_responses++;
        attack_classes++;
        check_sticky_quiescence();

        // A2: source_count/mask mismatch.
        reset_dut();
        @(negedge clk_core);
        set_legal_request(3'd0, 8'h03, 32'h200, 24'h520001, 3'd1, 3'd2);
        core_req_source_count = 1;
        request_only_attack_and_check(2);

        // A3: zero bank mask.
        reset_dut();
        @(negedge clk_core);
        set_legal_request(3'd0, 8'h00, 32'h300, 24'h530001, 3'd1, 3'd2);
        core_req_source_count = 0;
        request_only_attack_and_check(3);

        // A4: active source channel does not map to its bank.
        reset_dut();
        @(negedge clk_core);
        set_legal_request(3'd0, 8'h04, 32'h400, 24'h540001, 3'd1, 3'd2);
        core_req_source_channel[2] = 12'h00b;
        request_only_attack_and_check(4);

        // A5: source_count exceeds the architectural maximum.
        reset_dut();
        @(negedge clk_core);
        set_legal_request(3'd0, 8'h01, 32'h500, 24'h550001, 3'd7, 3'd2);
        core_req_output_block = 3'b111;
        core_req_source_count = 9;
        request_only_attack_and_check(5);

        // A6: slice range violation.
        reset_dut();
        @(negedge clk_core);
        set_legal_request(3'd0, 8'h01, 32'h600, 24'h560001, 3'd1, 3'd6);
        request_only_attack_and_check(6);

        // A7: stale/illegal bank response and a legal request on the same edge.
        // Both channels close because the response identity is unowned.
        reset_dut();
        @(negedge clk_core);
        set_legal_request(3'd1, 8'h02, 32'h701, 24'h570002, 3'd1, 3'd2);
        set_legal_response(3'd0, 8'h01, 32'h700, 24'h570001, -8'sd3);
        #0.1;
        expect_true(protocol_error && stale_response_seen,
            "stale bank response must raise protocol/stale flags");
        expect_true(!core_req_accept && bank_req_accept == 0
            && bank_rsp_accept == 0 && !core_rsp_accept,
            "illegal response must close response and request channels");
        @(posedge clk_core);
        @(negedge clk_core);
        clear_request();
        clear_response();
        #0.1;
        expect_true(debug_bundle_request_count == 0
            && debug_bank_request_count == 0
            && debug_bank_response_count == 0
            && debug_bundle_response_count == 0,
            "illegal response edge must mutate no traffic ledger");
        if (debug_bank_response_count != 0
                || debug_bundle_response_count != 0)
            response_side_effect_violations++;
        attack_classes++;
        check_sticky_quiescence();

        // A8: malformed request during a partially accepted pending bank fanout.
        reset_dut();
        @(negedge clk_core);
        bank_req_ready = 8'h01;
        set_legal_request(3'd0, 8'h03, 32'h800, 24'h580001, 3'd1, 3'd2);
        #0.1;
        expect_true(core_req_accept && bank_req_accept == 8'h01,
            "partial fanout must accept the bundle and first bank only");
        @(posedge clk_core);
        @(negedge clk_core);
        clear_request();
        bank_req_ready = 8'h02;
        #0.1;
        expect_true(bank_req_valid == 8'h02,
            "remaining bank beat must be pending before attack");
        set_legal_request(3'd1, 8'h00, 32'h801, 24'h580002, 3'd1, 3'd2);
        core_req_source_count = 0;
        #0.1;
        expect_true(protocol_error && bank_req_valid == 0
            && bank_req_accept == 0,
            "pending drain must stop immediately on malformed request");
        @(posedge clk_core);
        @(negedge clk_core);
        clear_request();
        #0.1;
        expect_true(debug_bank_request_count == 1
            && debug_bundle_request_count == 1,
            "attack must neither drop-count nor duplicate pending bank beat");
        attack_classes++;
        check_sticky_quiescence();

        // A9: response is backpressured, then a malformed request arrives.
        // The final legal payload is stable on the attack edge, but cannot
        // retire without ready and is poisoned by sticky fault afterwards.
        reset_dut();
        issue_legal_request(3'd0, 8'h01, 32'h900, 24'h590001);
        @(negedge clk_core);
        core_rsp_ready = 0;
        set_legal_response(3'd0, 8'h01, 32'h900, 24'h590001, 8'sd29);
        #0.1;
        expect_true(core_rsp_valid && !core_rsp_accept
            && bank_rsp_accept == 8'h01,
            "legal final bank beat must enter response hold under backpressure");
        @(posedge clk_core);
        @(negedge clk_core);
        clear_response();
        set_legal_request(3'd1, 8'h00, 32'h901, 24'h590002, 3'd1, 3'd2);
        core_req_source_count = 0;
        #0.1;
        expect_true(protocol_error && core_rsp_valid && !core_rsp_accept,
            "held response must remain visible and stable on request-fault edge");
        expect_true(core_rsp_slot == 0 && core_rsp_tag == 24'h590001
            && core_rsp_weight[0][0] == 8'sd29,
            "backpressured held response payload must be exact");
        @(posedge clk_core);
        @(negedge clk_core);
        clear_request();
        core_rsp_ready = 1;
        #0.1;
        expect_true(protocol_error && !core_rsp_valid,
            "sticky fault must poison an unaccepted held response");
        attack_classes++;
        check_sticky_quiescence();

        // A10: the same held-response state may retire once if ready is high on
        // the malformed-request edge; request traffic remains rejected.
        reset_dut();
        issue_legal_request(3'd0, 8'h01, 32'ha00, 24'h5a0001);
        @(negedge clk_core);
        core_rsp_ready = 0;
        set_legal_response(3'd0, 8'h01, 32'ha00, 24'h5a0001, -8'sd41);
        #0.1;
        expect_true(core_rsp_valid && bank_rsp_accept == 8'h01,
            "held-response setup must accept the final bank beat");
        @(posedge clk_core);
        @(negedge clk_core);
        clear_response();
        core_rsp_ready = 1;
        set_legal_request(3'd1, 8'h00, 32'ha01, 24'h5a0002, 3'd1, 3'd2);
        core_req_source_count = 0;
        #0.1;
        expect_true(protocol_error && core_rsp_accept
            && !core_req_accept && bank_req_accept == 0,
            "held legal response must retire once despite request attack");
        @(posedge clk_core);
        @(negedge clk_core);
        clear_request();
        #0.1;
        expect_true(debug_live_slots == 0
            && debug_bundle_response_count == 1,
            "held response retirement must clear ownership exactly once");
        legal_response_on_request_fault++;
        normal_responses++;
        attack_classes++;
        check_sticky_quiescence();

        // L1: legal same-cycle retirement and exact-slot reuse. This is the
        // performance behavior retained from M490 and intentionally absent from
        // M499. The old generation must retire while the new generation remains
        // live and subsequently returns an exact multi-bank payload.
        reset_dut();
        issue_legal_request(3'd3, 8'ha5, 32'hd00, 24'h5d0001);
        @(negedge clk_core);
        set_legal_response(3'd3, 8'ha5, 32'hd00, 24'h5d0001, 8'sd11);
        set_legal_request(3'd3, 8'h3c, 32'hd01, 24'h5d0002, 3'd2, 3'd4);
        #0.1;
        expect_true(!protocol_error && core_rsp_accept && core_req_accept,
            "legal same-slot response/request must both accept in one cycle");
        expect_true(bank_rsp_accept == 8'ha5 && bank_req_accept == 8'h3c,
            "same-slot reuse must preserve both multi-bank accept ledgers");
        expect_true(core_rsp_generation == 32'hd00
            && core_rsp_tag == 24'h5d0001
            && core_rsp_weight[7][0] == 8'sd18,
            "retiring generation identity and payload must remain exact");
        @(posedge clk_core);
        @(negedge clk_core);
        clear_request();
        clear_response();
        #0.1;
        expect_true(!protocol_error && debug_live_slots == 1
            && debug_bundle_request_count == 2
            && debug_bundle_response_count == 1
            && debug_bank_request_count == 8
            && debug_bank_response_count == 4,
            "new same-slot generation must remain the sole live owner");
        set_legal_response(3'd3, 8'h3c, 32'hd01, 24'h5d0002, -8'sd9);
        #0.1;
        expect_true(core_rsp_accept && bank_rsp_accept == 8'h3c
            && core_rsp_generation == 32'hd01
            && core_rsp_tag == 24'h5d0002
            && core_rsp_weight[5][0] == -8'sd4,
            "new generation response must retire exactly once");
        @(posedge clk_core);
        @(negedge clk_core);
        clear_response();
        #0.1;
        expect_true(debug_live_slots == 0
            && debug_bundle_request_count == 2
            && debug_bundle_response_count == 2
            && debug_bank_request_count == debug_bank_response_count,
            "same-cycle reuse must close all bundle/bank conservation ledgers");
        normal_requests++;
        normal_responses += 2;
        same_cycle_reuse_cases++;

        // A11/A12: sticky containment was checked after every attack; reset is
        // itself a directed recovery class and must restore a clean normal path.
        reset_dut();
        attack_classes++;
        issue_legal_request(3'd2, 8'h10, 32'hc00, 24'h5c0001);
        @(negedge clk_core);
        set_legal_response(3'd2, 8'h10, 32'hc00, 24'h5c0001, 8'sd7);
        #0.1;
        expect_true(core_rsp_accept && bank_rsp_accept == 8'h10
            && !protocol_error,
            "post-reset normal response must remain zero-latency cut-through");
        @(posedge clk_core);
        @(negedge clk_core);
        clear_response();
        #0.1;
        expect_true(debug_bundle_response_count == 1
            && debug_live_slots == 0 && !protocol_error,
            "post-reset response must retire normally");
        normal_responses++;
        attack_classes++;

        expect_true(request_side_effect_violations == 0,
            "all request attacks must have zero request-side effects");
        expect_true(response_side_effect_violations == 0,
            "all response attacks must have zero illegal response effects");
        expect_true(attack_classes >= 12,
            "at least twelve directed attack/recovery classes are required");
        expect_true(legal_response_on_request_fault == 2,
            "cut-through and held legal response/request-fault cases required");
        expect_true(same_cycle_reuse_cases == 1,
            "one exact legal same-cycle slot-reuse regression is required");
        expect_true(sticky_quiescent_checks >= 10,
            "sticky fault quiescence must be checked after every fault class");

        if (errors == 0) begin
            $display("PASS M803 R16 channel-split cutthrough adapter VCS attack_classes=%0d reset_cases=%0d legal_response_on_request_fault=%0d same_cycle_reuse_cases=%0d sticky_quiescent_checks=%0d normal_requests=%0d normal_responses=%0d request_side_effect_violations=%0d response_side_effect_violations=%0d", attack_classes, reset_cases, legal_response_on_request_fault, same_cycle_reuse_cases, sticky_quiescent_checks, normal_requests, normal_responses, request_side_effect_violations, response_side_effect_violations);
            $finish;
        end
        $fatal(1, "FAIL M803 R16 channel-split adapter errors=%0d", errors);
    end

    initial begin : watchdog
        repeat (2000) @(posedge clk_core);
        $fatal(1, "M803 R16 attack TB watchdog");
    end
endmodule

`default_nettype wire

