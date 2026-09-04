`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_output_tile_scheduler_stage_sweep;
    localparam int HEADS = 24;
    localparam int HEAD_ID_W = 5;
    logic clk_core, rst_core;
    logic group_valid, group_ready, group_context_id;
    logic [31:0] group_tag;
    logic [5:0] group_head_count;
    logic [7:0] group_first_output_tile, group_output_tile_count;
    logic tile_start_valid, tile_start_ready;
    logic [31:0] tile_start_tag;
    logic [7:0] tile_start_output_tile;
    logic [5:0] tile_start_head_count;
    logic head_issue_valid, head_issue_ready, head_issue_context_id;
    logic [31:0] head_issue_tag;
    logic [HEAD_ID_W-1:0] head_issue_head_id;
    logic [5:0] head_issue_head_index;
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

    gatestack_output_tile_scheduler #(
        .HEADS(HEADS), .LANES(32), .HEAD_COUNT_W(6),
        .HEAD_ID_W(HEAD_ID_W)
    ) dut (.*);

    always #5 clk_core <= ~clk_core;

    task automatic run_stage(input int stage, input int heads);
        logic [31:0] expected_tag;
        begin
            expected_tag = 32'hb300_0000 + 32'(stage);
            @(negedge clk_core);
            group_context_id = stage[0];
            group_tag = expected_tag;
            group_head_count = 6'(heads);
            group_first_output_tile = '0;
            group_output_tile_count = 8'(heads);
            group_valid = 1'b1;
            do @(posedge clk_core); while (!group_ready);
            @(negedge clk_core);
            group_valid = 1'b0;

            for (int tile = 0; tile < heads; tile = tile + 1) begin
                while (!tile_start_valid) @(posedge clk_core);
                if (tile_start_tag != expected_tag + 32'(tile) ||
                    tile_start_output_tile != 8'(tile) ||
                    tile_start_head_count != 6'(heads))
                    $fatal(1, "stage tile mismatch stage=%0d tile=%0d",
                           stage, tile);
                repeat ((stage + tile) % 3) @(posedge clk_core);
                @(negedge clk_core);
                tile_start_ready = 1'b1;
                @(posedge clk_core);
                @(negedge clk_core);
                tile_start_ready = 1'b0;

                for (int head = 0; head < heads; head = head + 1) begin
                    while (!head_issue_valid) @(posedge clk_core);
                    if (head_issue_context_id != stage[0] ||
                        head_issue_tag != expected_tag + 32'(tile) ||
                        head_issue_head_id != HEAD_ID_W'(head) ||
                        head_issue_head_index != 6'(head) ||
                        head_issue_input_channel_base != 10'(head * 32) ||
                        head_issue_output_tile != 8'(tile) ||
                        head_issue_last_head != (head == heads - 1) ||
                        head_issue_last_output_tile != (tile == heads - 1))
                        $fatal(1, "stage head mismatch s=%0d t=%0d h=%0d",
                               stage, tile, head);
                    repeat ((stage + tile + head) % 4) @(posedge clk_core);
                    @(negedge clk_core);
                    head_issue_ready = 1'b1;
                    @(posedge clk_core);
                    @(negedge clk_core);
                    head_issue_ready = 1'b0;
                    repeat ((head % 3) + 1) @(posedge clk_core);
                    @(negedge clk_core);
                    head_done_tag = expected_tag + 32'(tile);
                    head_done_head_id = HEAD_ID_W'(head);
                    head_done_error = 1'b0;
                    head_done_valid = 1'b1;
                    do @(posedge clk_core); while (!head_done_ready);
                    @(negedge clk_core);
                    head_done_valid = 1'b0;
                end

                while (!tile_done_ready) @(posedge clk_core);
                repeat (tile % 3) @(posedge clk_core);
                @(negedge clk_core);
                tile_done_tag = expected_tag + 32'(tile);
                tile_done_error = 1'b0;
                tile_done_valid = 1'b1;
                @(posedge clk_core);
                @(negedge clk_core);
                tile_done_valid = 1'b0;
            end

            while (!group_done_valid) @(posedge clk_core);
            if (group_done_tag != expected_tag || group_done_error)
                $fatal(1, "stage group completion mismatch");
            @(negedge clk_core);
            group_done_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            group_done_ready = 1'b0;
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        group_valid = 1'b0;
        group_context_id = 1'b0;
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
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;

        run_stage(0, 3);
        run_stage(1, 6);
        run_stage(2, 12);
        run_stage(3, 24);
        if (protocol_error || count_groups != 4 ||
            count_tile_starts != 45 || count_head_issues != 765 ||
            count_group_errors != 0)
            $fatal(1, "stage sweep counters mismatch");
        $display("PASS: H67 stage sweep groups=4 tiles=%0d head_issues=%0d",
                 count_tile_starts, count_head_issues);
        $finish;
    end

    initial begin
        repeat (300000) @(posedge clk_core);
        $fatal(1, "output-tile stage sweep timeout");
    end
endmodule

`default_nettype wire
