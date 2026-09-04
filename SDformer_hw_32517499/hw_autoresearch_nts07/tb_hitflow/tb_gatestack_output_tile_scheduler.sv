`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_output_tile_scheduler;
    localparam int HEADS = 3;
    localparam int LANES = 32;
    localparam int HEAD_ID_W = 2;
    logic clk_core, rst_core;
    logic group_valid, group_ready, group_context_id;
    logic [31:0] group_tag;
    logic [2:0] group_head_count;
    logic [7:0] group_first_output_tile, group_output_tile_count;
    logic tile_start_valid, tile_start_ready;
    logic [31:0] tile_start_tag;
    logic [7:0] tile_start_output_tile;
    logic [2:0] tile_start_head_count;
    logic head_issue_valid, head_issue_ready;
    logic head_issue_context_id;
    logic [31:0] head_issue_tag;
    logic [HEAD_ID_W-1:0] head_issue_head_id;
    logic [2:0] head_issue_head_index;
    logic [9:0] head_issue_input_channel_base;
    logic [7:0] head_issue_output_tile;
    logic head_issue_last_head, head_issue_last_output_tile;
    logic head_done_valid, head_done_ready;
    logic [31:0] head_done_tag;
    logic [HEAD_ID_W-1:0] head_done_head_id;
    logic head_done_error;
    logic tile_done_valid, tile_done_ready;
    logic [31:0] tile_done_tag;
    logic tile_done_error;
    logic group_done_valid, group_done_ready;
    logic [31:0] group_done_tag;
    logic group_done_error, protocol_error;
    logic [31:0] count_groups, count_tile_starts;
    logic [31:0] count_head_issues, count_group_errors;
    int observed_tiles, observed_heads;

    gatestack_output_tile_scheduler #(
        .HEADS(HEADS), .LANES(LANES), .HEAD_COUNT_W(3),
        .HEAD_ID_W(HEAD_ID_W)
    ) dut (.*);

    always #5 clk_core <= ~clk_core;

    task automatic start_group;
        begin
            @(negedge clk_core);
            group_context_id = 1'b1;
            group_tag = 32'h9100_0042;
            group_head_count = 3;
            group_first_output_tile = 7;
            group_output_tile_count = 3;
            group_valid = 1'b1;
            do @(posedge clk_core); while (!group_ready);
            @(negedge clk_core);
            group_valid = 1'b0;
        end
    endtask

    task automatic accept_tile_start(input logic [7:0] expected_tile);
        logic [31:0] expected_exec_tag;
        begin
            expected_exec_tag = group_tag + 32'(expected_tile) - 32'd7;
            while (!tile_start_valid) @(posedge clk_core);
            if (tile_start_tag != expected_exec_tag ||
                tile_start_output_tile != expected_tile ||
                tile_start_head_count != 3)
                $fatal(1, "tile start mismatch");
            repeat (2) @(posedge clk_core);
            @(negedge clk_core);
            tile_start_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            tile_start_ready = 1'b0;
            observed_tiles = observed_tiles + 1;
        end
    endtask

    task automatic service_head(input logic [7:0] expected_tile,
                                input int expected_head,
                                input logic expected_last_tile);
        logic [31:0] expected_exec_tag;
        begin
            expected_exec_tag = group_tag + 32'(expected_tile) - 32'd7;
            while (!head_issue_valid) @(posedge clk_core);
            if (head_issue_context_id != 1'b1 ||
                head_issue_tag != expected_exec_tag ||
                head_issue_head_id != HEAD_ID_W'(expected_head) ||
                head_issue_head_index != 3'(expected_head) ||
                head_issue_input_channel_base != 10'(expected_head * LANES) ||
                head_issue_output_tile != expected_tile ||
                head_issue_last_head != (expected_head == HEADS - 1) ||
                head_issue_last_output_tile != expected_last_tile)
                $fatal(1, "head issue mismatch tile=%0d head=%0d",
                       expected_tile, expected_head);
            repeat (expected_head + 1) @(posedge clk_core);
            @(negedge clk_core);
            head_issue_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            head_issue_ready = 1'b0;
            repeat (2) @(posedge clk_core);
            @(negedge clk_core);
            head_done_tag = expected_exec_tag;
            head_done_head_id = HEAD_ID_W'(expected_head);
            head_done_error = 1'b0;
            head_done_valid = 1'b1;
            do @(posedge clk_core); while (!head_done_ready);
            @(negedge clk_core);
            head_done_valid = 1'b0;
            observed_heads = observed_heads + 1;
        end
    endtask

    task automatic finish_tile(input logic [7:0] expected_tile);
        begin
            while (!tile_done_ready) @(posedge clk_core);
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            tile_done_tag = group_tag + 32'(expected_tile) - 32'd7;
            tile_done_error = 1'b0;
            tile_done_valid = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            tile_done_valid = 1'b0;
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        group_valid = 1'b0;
        group_context_id = '0;
        group_tag = '0;
        group_head_count = '0;
        group_first_output_tile = '0;
        group_output_tile_count = '0;
        tile_start_ready = 1'b0;
        head_issue_ready = 1'b0;
        head_done_valid = 1'b0;
        head_done_tag = '0;
        head_done_head_id = '0;
        head_done_error = 1'b0;
        tile_done_valid = 1'b0;
        tile_done_tag = '0;
        tile_done_error = 1'b0;
        group_done_ready = 1'b0;
        observed_tiles = 0;
        observed_heads = 0;
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;

        start_group();
        for (int tile = 7; tile < 10; tile = tile + 1) begin
            accept_tile_start(8'(tile));
            for (int head = 0; head < HEADS; head = head + 1)
                service_head(8'(tile), head, tile == 9);
            finish_tile(8'(tile));
        end

        while (!group_done_valid) @(posedge clk_core);
        if (group_done_tag != group_tag || group_done_error)
            $fatal(1, "group done mismatch");
        repeat (2) @(posedge clk_core);
        @(negedge clk_core);
        group_done_ready = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        group_done_ready = 1'b0;
        if (protocol_error || observed_tiles != 3 || observed_heads != 9 ||
            count_groups != 1 || count_tile_starts != 3 ||
            count_head_issues != 9 || count_group_errors != 0)
            $fatal(1, "scheduler counters or error mismatch");

        // A malformed zero-tile group must be rejected before acquisition.
        @(negedge clk_core);
        group_output_tile_count = '0;
        group_valid = 1'b1;
        @(posedge clk_core);
        if (group_ready) $fatal(1, "illegal group was admitted");
        @(negedge clk_core);
        group_valid = 1'b0;
        repeat (2) @(posedge clk_core);
        if (!protocol_error || count_groups != 1)
            $fatal(1, "illegal group rejection was not sticky");
        $display("PASS: output-tile scheduler tiles=%0d heads=%0d illegal_reject=1",
                 observed_tiles, observed_heads);
        $finish;
    end

    initial begin
        repeat (10000) @(posedge clk_core);
        $fatal(1, "output-tile scheduler timeout");
    end
endmodule

`default_nettype wire
