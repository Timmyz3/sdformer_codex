`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_replay_atomic_commit;
    logic clk_core, rst_core;
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
    logic projection_commit_pulse, projection_reserve_ready;
    logic projection_context_id;
    logic [4:0] projection_head_id;
    logic [31:0] projection_payload_tag, projection_execution_tag;
    logic [1:0] projection_route;
    logic [1:0] projection_format;
    logic [5:0] projection_head_index;
    logic [9:0] projection_input_channel_base;
    logic [7:0] projection_output_tile;
    logic projection_last_head;
    logic [7:0] projection_resident_term_count;
    logic [12:0] projection_resident_event_count;
    logic slot_commit_pulse, slot_reserve_ready, slot_context_id;
    logic [4:0] slot_head_id;
    logic [31:0] slot_payload_tag;
    logic [6:0] slot_replay_start_word;
    logic lifecycle_commit_pulse, lifecycle_reserve_ready;
    logic lifecycle_context_id;
    logic [4:0] lifecycle_head_id;
    logic [31:0] lifecycle_payload_tag, lifecycle_execution_tag;
    logic lifecycle_cache_owned, lifecycle_last_output_tile;
    logic reject_valid, reject_ready;
    logic [31:0] reject_execution_tag;
    logic commit_pulse;
    logic [31:0] commit_execution_tag;
    logic protocol_error;
    logic [31:0] count_commits, count_rejects;
    int projection_fires, slot_fires, lifecycle_fires, reject_fires;

    gatestack_replay_atomic_commit dut (.*);

    always #5 clk_core <= ~clk_core;

    always @(posedge clk_core) begin
        if (rst_core) begin
            projection_fires <= 0;
            slot_fires <= 0;
            lifecycle_fires <= 0;
            reject_fires <= 0;
        end else begin
            if (projection_commit_pulse)
                projection_fires <= projection_fires + 1;
            if (slot_commit_pulse)
                slot_fires <= slot_fires + 1;
            if (lifecycle_commit_pulse)
                lifecycle_fires <= lifecycle_fires + 1;
            if (reject_valid && reject_ready)
                reject_fires <= reject_fires + 1;
        end
    end

    task automatic check_metadata;
        begin
            if (projection_context_id != plan_context_id ||
                projection_head_id != plan_head_id ||
                projection_payload_tag != plan_payload_tag ||
                projection_execution_tag != plan_execution_tag ||
                projection_route != plan_route ||
                projection_format != plan_format ||
                projection_head_index != plan_head_index ||
                projection_input_channel_base != plan_input_channel_base ||
                projection_output_tile != plan_output_tile ||
                projection_last_head != plan_last_head ||
                projection_resident_term_count !=
                    plan_resident_term_count ||
                projection_resident_event_count !=
                    plan_resident_event_count ||
                slot_context_id != plan_context_id ||
                slot_head_id != plan_head_id ||
                slot_payload_tag != plan_payload_tag ||
                slot_replay_start_word != plan_replay_start_word ||
                lifecycle_context_id != plan_context_id ||
                lifecycle_head_id != plan_head_id ||
                lifecycle_payload_tag != plan_payload_tag ||
                lifecycle_execution_tag != plan_execution_tag ||
                lifecycle_cache_owned != plan_cache_owned ||
                lifecycle_last_output_tile != plan_last_output_tile ||
                reject_execution_tag != plan_execution_tag)
                $fatal(1, "atomic commit metadata mismatch");
        end
    endtask

    task automatic drive_common;
        begin
            plan_context_id = 1'b1;
            plan_head_id = 5'd7;
            plan_payload_tag = 32'hc400_1007;
            plan_execution_tag = 32'hc4a3_0702;
            plan_head_index = 6'd7;
            plan_input_channel_base = 10'd224;
            plan_output_tile = 8'd3;
            plan_last_head = 1'b0;
            plan_last_output_tile = 1'b0;
            plan_replay_start_word = 7'd13;
            plan_format = 2'd1;
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        plan_valid = 1'b0;
        plan_context_id = '0;
        plan_head_id = '0;
        plan_payload_tag = '0;
        plan_execution_tag = '0;
        plan_route = '0;
        plan_format = 2'd0;
        plan_head_index = '0;
        plan_input_channel_base = '0;
        plan_output_tile = '0;
        plan_last_head = 1'b0;
        plan_last_output_tile = 1'b0;
        plan_cache_owned = 1'b0;
        plan_slot_replay_required = 1'b0;
        plan_replay_start_word = '0;
        plan_resident_term_count = '0;
        plan_resident_event_count = '0;
        projection_reserve_ready = 1'b0;
        slot_reserve_ready = 1'b0;
        lifecycle_reserve_ready = 1'b0;
        reject_ready = 1'b0;
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;

        // Resident plan with payload must acquire all three resources atomically.
        @(negedge clk_core);
        drive_common();
        plan_route = 2'd0;
        plan_cache_owned = 1'b1;
        plan_slot_replay_required = 1'b1;
        plan_resident_term_count = 8'd3;
        plan_resident_event_count = 13'd9;
        plan_replay_start_word = 7'd4;
        plan_valid = 1'b1;
        lifecycle_reserve_ready = 1'b1;
        slot_reserve_ready = 1'b1;
        repeat (2) @(posedge clk_core);
        if (plan_ready || projection_fires != 0 || slot_fires != 0 ||
            lifecycle_fires != 0)
            $fatal(1, "partial commit with projection blocked");
        check_metadata();

        @(negedge clk_core);
        projection_reserve_ready = 1'b1;
        lifecycle_reserve_ready = 1'b0;
        repeat (2) @(posedge clk_core);
        if (plan_ready || projection_fires != 0 || slot_fires != 0 ||
            lifecycle_fires != 0)
            $fatal(1, "partial commit with lifecycle blocked");

        @(negedge clk_core);
        lifecycle_reserve_ready = 1'b1;
        slot_reserve_ready = 1'b0;
        repeat (2) @(posedge clk_core);
        if (plan_ready || projection_fires != 0 || slot_fires != 0 ||
            lifecycle_fires != 0)
            $fatal(1, "partial commit with slot blocked");

        @(negedge clk_core);
        slot_reserve_ready = 1'b1;
        do @(posedge clk_core); while (!plan_ready);
        @(negedge clk_core);
        plan_valid = 1'b0;
        plan_execution_tag = 32'hdead_beef;
        if (!commit_pulse || commit_execution_tag != 32'hc4a3_0702)
            $fatal(1, "commit pulse/tag capture mismatch");
        repeat (2) @(posedge clk_core);
        if (projection_fires != 1 || slot_fires != 1 ||
            lifecycle_fires != 1 || count_commits != 1)
            $fatal(1, "three-resource atomic commit failed");

        // Empty resident plan needs projection+lifecycle but no slot replay.
        @(negedge clk_core);
        drive_common();
        plan_execution_tag = 32'hc4a4_0702;
        plan_route = 2'd0;
        plan_cache_owned = 1'b1;
        plan_slot_replay_required = 1'b0;
        plan_resident_term_count = '0;
        plan_resident_event_count = '0;
        plan_replay_start_word = 7'd2;
        slot_reserve_ready = 1'b0;
        plan_valid = 1'b1;
        do @(posedge clk_core); while (!plan_ready);
        @(negedge clk_core);
        plan_valid = 1'b0;
        repeat (2) @(posedge clk_core);
        if (projection_fires != 2 || slot_fires != 1 ||
            lifecycle_fires != 2 || count_commits != 2)
            $fatal(1, "slotless resident commit failed");

        // RAW without slot replay is malformed and must reject before acquire.
        @(negedge clk_core);
        drive_common();
        plan_execution_tag = 32'hc4ff_0702;
        plan_route = 2'd2;
        plan_format = 2'd0;
        plan_cache_owned = 1'b0;
        plan_slot_replay_required = 1'b0;
        plan_resident_term_count = '0;
        plan_resident_event_count = '0;
        plan_replay_start_word = '0;
        reject_ready = 1'b0;
        plan_valid = 1'b1;
        repeat (2) @(posedge clk_core);
        if (!reject_valid || plan_ready || projection_commit_pulse ||
            slot_commit_pulse || lifecycle_commit_pulse)
            $fatal(1, "malformed plan acquired a resource");
        @(negedge clk_core);
        reject_ready = 1'b1;
        do @(posedge clk_core); while (!plan_ready);
        @(negedge clk_core);
        plan_valid = 1'b0;
        reject_ready = 1'b0;
        repeat (2) @(posedge clk_core);
        if (!protocol_error || count_rejects != 1 || reject_fires != 1 ||
            projection_fires != 2 || slot_fires != 1 ||
            lifecycle_fires != 2)
            $fatal(1, "malformed plan rejection failed");

        $display("PASS: atomic commits=%0d rejects=%0d projection=%0d slot=%0d lifecycle=%0d",
                 count_commits, count_rejects, projection_fires,
                 slot_fires, lifecycle_fires);
        $finish;
    end

    initial begin
        repeat (10000) @(posedge clk_core);
        $fatal(1, "replay atomic commit timeout");
    end
endmodule

`default_nettype wire
