`timescale 1ns/1ps
`default_nettype none

module tb_qfit_relation_memo_tile_controller;
    logic clk_core;
    logic rst_core;
    logic tile_start;
    logic tile_ready;
    logic tile_prefer_replay;
    logic [4:0] tile_head_index;
    logic tile_done;
    logic fallback_taken;
    logic use_replay;
    logic replay_start;
    logic replay_cmd_ready;
    logic [4:0] replay_head_index;
    logic replay_done;
    logic replay_miss;
    logic recompute_request;
    logic recompute_grant;
    logic head_start;
    logic head_ready;
    logic [4:0] head_index;
    logic head_done;
    logic descriptor_stream_idle;
    logic projection_start;
    logic projection_close;
    logic projection_close_ready;
    logic projection_done;
    logic protocol_error;

    qfit_relation_memo_tile_controller dut (.*);

    always #5 clk_core = ~clk_core;

    task automatic start_tile(
        input int head,
        input bit prefer_replay
    );
        while (!tile_ready)
            @(negedge clk_core);
        tile_head_index = 5'(head);
        tile_prefer_replay = prefer_replay;
        tile_start = 1'b1;
        @(negedge clk_core);
        tile_start = 1'b0;
    endtask

    task automatic grant_recompute;
        while (!recompute_request)
            @(negedge clk_core);
        repeat (3) @(negedge clk_core);
        recompute_grant = 1'b1;
        @(negedge clk_core);
        recompute_grant = 1'b0;
        while (!head_start)
            @(negedge clk_core);
        head_done = 1'b1;
        @(negedge clk_core);
        head_done = 1'b0;
    endtask

    task automatic finish_projection;
        while (!projection_close)
            @(negedge clk_core);
        projection_done = 1'b1;
        @(negedge clk_core);
        projection_done = 1'b0;
        while (!tile_done)
            @(negedge clk_core);
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        tile_start = 1'b0;
        tile_prefer_replay = 1'b0;
        tile_head_index = '0;
        replay_cmd_ready = 1'b1;
        replay_done = 1'b0;
        replay_miss = 1'b0;
        recompute_grant = 1'b0;
        head_ready = 1'b1;
        head_done = 1'b0;
        descriptor_stream_idle = 1'b1;
        projection_close_ready = 1'b1;
        projection_done = 1'b0;
        repeat (4) @(negedge clk_core);
        rst_core = 1'b0;

        start_tile(2, 1'b0);
        grant_recompute();
        finish_projection();
        if (fallback_taken)
            $fatal(1, "unexpected fallback on direct recompute");

        start_tile(3, 1'b1);
        while (!replay_start)
            @(negedge clk_core);
        replay_done = 1'b1;
        replay_miss = 1'b0;
        @(negedge clk_core);
        replay_done = 1'b0;
        finish_projection();
        if (fallback_taken)
            $fatal(1, "unexpected fallback on replay hit");

        start_tile(4, 1'b1);
        while (!replay_start)
            @(negedge clk_core);
        replay_done = 1'b1;
        replay_miss = 1'b1;
        @(negedge clk_core);
        replay_done = 1'b0;
        replay_miss = 1'b0;
        grant_recompute();
        finish_projection();
        if (!fallback_taken)
            $fatal(1, "fallback was not recorded");
        if (protocol_error)
            $fatal(1, "unexpected protocol error");
        $display(
            "PASS relation memo tile controller replay-hit/miss/fallback head=%0d",
            head_index
        );
        $finish;
    end

    logic [4:0] unused_replay_head;
    logic unused_projection_start;
    assign unused_replay_head = replay_head_index;
    assign unused_projection_start = projection_start;
endmodule

`default_nettype wire
