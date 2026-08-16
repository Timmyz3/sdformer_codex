`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_replay_launch_control;
    logic clk_core, rst_core, launch_valid, launch_ready;
    logic launch_context_id;
    logic [4:0] launch_head_id;
    logic launch_done_valid, launch_done_ready;
    logic [1:0] launch_done_route;
    logic [31:0] launch_done_tag;
    logic launch_done_error;
    logic slot_inspect_valid, slot_inspect_ready;
    logic slot_inspect_context_id;
    logic [4:0] slot_inspect_head_id;
    logic slot_meta_valid, slot_meta_ready, slot_meta_exists;
    logic [31:0] slot_meta_tag;
    logic slot_meta_mode_is_csr;
    logic [15:0] slot_meta_payload_bits, slot_meta_word_count;
    logic cache_lookup_valid, cache_lookup_ready;
    logic cache_lookup_context_id;
    logic [4:0] cache_lookup_head_id;
    logic [31:0] cache_lookup_expected_tag;
    logic cache_meta_valid, cache_meta_ready, cache_meta_hit;
    logic [31:0] cache_meta_tag;
    logic [7:0] cache_meta_term_count;
    logic slot_replay_begin_valid, slot_replay_begin_ready;
    logic slot_replay_context_id;
    logic [4:0] slot_replay_head_id;
    logic [6:0] slot_replay_start_word;
    logic resident_start_valid, resident_start_ready;
    logic [31:0] resident_start_tag;
    logic [7:0] resident_start_term_count;
    logic [12:0] resident_start_event_count;
    logic ipd_start_valid, ipd_start_ready;
    logic raw_start_valid, raw_start_ready;
    logic [31:0] raw_start_tag;
    logic route_start_valid, route_start_ready;
    logic [1:0] route_start_select;
    logic protocol_error;
    logic [31:0] count_launches, count_resident_launches;
    logic [31:0] count_ipd_launches, count_raw_launches;
    logic [31:0] count_launch_errors;
    int observed_slot_replays, observed_resident_starts;
    int observed_ipd_starts, observed_raw_starts, observed_route_starts;
    logic [6:0] observed_last_start_word;

    gatestack_replay_launch_control dut (.*);
    always #5 clk_core <= ~clk_core;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            observed_slot_replays <= 0;
            observed_resident_starts <= 0;
            observed_ipd_starts <= 0;
            observed_raw_starts <= 0;
            observed_route_starts <= 0;
            observed_last_start_word <= '0;
        end else begin
            if (slot_replay_begin_valid && slot_replay_begin_ready) begin
                if (slot_replay_context_id != launch_context_id ||
                    slot_replay_head_id != launch_head_id)
                    $fatal(1,"slot replay identity mismatch");
                observed_slot_replays <= observed_slot_replays + 1;
                observed_last_start_word <= slot_replay_start_word;
            end
            if (resident_start_valid && resident_start_ready)
                observed_resident_starts <= observed_resident_starts + 1;
            if (ipd_start_valid && ipd_start_ready)
                observed_ipd_starts <= observed_ipd_starts + 1;
            if (raw_start_valid && raw_start_ready)
                observed_raw_starts <= observed_raw_starts + 1;
            if (route_start_valid && route_start_ready) begin
                if (route_start_select > 2)
                    $fatal(1,"route select out of range");
                observed_route_starts <= observed_route_starts + 1;
            end
        end
    end

    task automatic run_launch(input int scenario, input int head);
        logic [31:0] tag;
        logic expected_error;
        logic [1:0] expected_route;
        int replay_before, resident_before, ipd_before, raw_before, route_before;
        begin
            tag = 32'h7100_0000 + 32'(scenario);
            expected_error = scenario == 4;
            expected_route = (scenario == 1) ? 2'd1 :
                             (scenario == 2) ? 2'd2 : 2'd0;
            replay_before = observed_slot_replays;
            resident_before = observed_resident_starts;
            ipd_before = observed_ipd_starts;
            raw_before = observed_raw_starts;
            route_before = observed_route_starts;
            @(negedge clk_core);
            launch_context_id = 1'(head & 1);
            launch_head_id = 5'(head);
            launch_valid = 1'b1;
            do @(posedge clk_core); while (!launch_ready);
            @(negedge clk_core); launch_valid = 1'b0;
            wait (slot_inspect_valid);
            if (slot_inspect_context_id != 1'(head & 1) ||
                slot_inspect_head_id != 5'(head))
                $fatal(1,"inspect request mismatch");
            @(posedge clk_core); @(negedge clk_core);
            slot_meta_exists = scenario != 4;
            slot_meta_tag = tag;
            slot_meta_mode_is_csr = scenario != 2;
            if (scenario == 2) begin
                slot_meta_payload_bits = 16'd6642;
                slot_meta_word_count = 16'd104;
            end else if (scenario == 3) begin
                slot_meta_payload_bits = 16'd128;
                slot_meta_word_count = 16'd2;
            end else begin
                slot_meta_payload_bits = 16'd344;
                slot_meta_word_count = 16'd6;
            end
            slot_meta_valid = 1'b1;
            do @(posedge clk_core); while (!slot_meta_ready);
            @(negedge clk_core); slot_meta_valid = 1'b0;
            if (scenario != 2 && scenario != 4) begin
                wait (cache_lookup_valid);
                if (cache_lookup_context_id != 1'(head & 1) ||
                    cache_lookup_head_id != 5'(head) ||
                    cache_lookup_expected_tag != tag)
                    $fatal(1,"cache lookup contract mismatch");
                @(posedge clk_core); @(negedge clk_core);
                cache_meta_hit = scenario != 1;
                cache_meta_tag = tag;
                cache_meta_term_count = scenario == 3 ? 8'd0 : 8'd3;
                cache_meta_valid = 1'b1;
                do @(posedge clk_core); while (!cache_meta_ready);
                @(negedge clk_core); cache_meta_valid = 1'b0;
            end
            wait (launch_done_valid);
            if (launch_done_error != expected_error ||
                launch_done_route != expected_route ||
                (!expected_error && launch_done_tag != tag))
                $fatal(1,"launch done mismatch scenario=%0d",scenario);
            @(negedge clk_core); launch_done_ready = 1'b1;
            @(posedge clk_core); @(negedge clk_core); launch_done_ready = 1'b0;
            if (!expected_error && observed_route_starts != route_before + 1)
                $fatal(1,"route start count mismatch");
            if (scenario == 0) begin
                if (observed_resident_starts != resident_before + 1 ||
                    observed_slot_replays != replay_before + 1 ||
                    observed_last_start_word != 7'd4 ||
                    resident_start_tag != tag ||
                    resident_start_term_count != 3 ||
                    resident_start_event_count != 11)
                    $fatal(1,"resident launch mismatch");
            end else if (scenario == 1) begin
                if (observed_ipd_starts != ipd_before + 1 ||
                    observed_slot_replays != replay_before + 1 ||
                    observed_last_start_word != 0)
                    $fatal(1,"IPD launch mismatch");
            end else if (scenario == 2) begin
                if (observed_raw_starts != raw_before + 1 ||
                    observed_slot_replays != replay_before + 1 ||
                    observed_last_start_word != 0 || raw_start_tag != tag)
                    $fatal(1,"RAW launch mismatch");
            end else if (scenario == 3) begin
                if (observed_resident_starts != resident_before + 1 ||
                    observed_slot_replays != replay_before ||
                    resident_start_term_count != 0 ||
                    resident_start_event_count != 0)
                    $fatal(1,"zero resident launch mismatch");
            end else if (observed_route_starts != route_before ||
                         observed_slot_replays != replay_before) begin
                $fatal(1,"error launch emitted work");
            end
        end
    endtask

    initial begin
        clk_core=0; rst_core=1; launch_valid=0; launch_context_id=0;
        launch_head_id=0; launch_done_ready=0; slot_inspect_ready=1;
        slot_meta_valid=0; slot_meta_exists=0; slot_meta_tag=0;
        slot_meta_mode_is_csr=0; slot_meta_payload_bits=0;
        slot_meta_word_count=0; cache_lookup_ready=1; cache_meta_valid=0;
        cache_meta_hit=0; cache_meta_tag=0; cache_meta_term_count=0;
        slot_replay_begin_ready=1; resident_start_ready=1;
        ipd_start_ready=1; raw_start_ready=1; route_start_ready=1;
        repeat(5) @(posedge clk_core); rst_core=0;
        run_launch(0,0);
        run_launch(1,1);
        run_launch(2,2);
        run_launch(3,3);
        run_launch(4,4);
        if (!protocol_error || count_launches!=5 ||
            count_resident_launches!=2 || count_ipd_launches!=1 ||
            count_raw_launches!=1 || count_launch_errors!=1)
            $fatal(1,"launch control counters mismatch");
        $display("PASS: replay launch control launches=%0d resident=%0d ipd=%0d raw=%0d errors=%0d",
                 count_launches,count_resident_launches,count_ipd_launches,
                 count_raw_launches,count_launch_errors);
        $finish;
    end
    initial begin repeat(5000) @(posedge clk_core); $fatal(1,"launch control timeout"); end
endmodule

`default_nettype wire
