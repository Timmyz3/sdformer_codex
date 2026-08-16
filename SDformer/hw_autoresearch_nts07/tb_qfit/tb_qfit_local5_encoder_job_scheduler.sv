`timescale 1ns/1ps
`default_nettype none

module tb_qfit_local5_encoder_job_scheduler;
    localparam int TAG_W = 24;
    localparam int WINDOW_W = 9;
    localparam int HEAD_W = 5;
    localparam int OUTPUT_TILE_W = 5;

    logic clk_core, rst_core, start_frame, frame_busy, frame_done;
    logic tile_start_valid, tile_start_ready;
    logic [TAG_W-1:0] tile_start_tag;
    logic [1:0] tile_start_stage;
    logic [2:0] tile_start_block;
    logic [WINDOW_W-1:0] tile_start_window;
    logic [OUTPUT_TILE_W-1:0] tile_start_output_tile;
    logic [5:0] tile_start_head_count;
    logic head_job_valid, head_job_ready;
    logic [TAG_W-1:0] head_job_tag;
    logic [1:0] head_job_stage;
    logic [2:0] head_job_block;
    logic [WINDOW_W-1:0] head_job_window;
    logic [HEAD_W-1:0] head_job_input_head;
    logic [9:0] head_job_input_channel_base;
    logic [OUTPUT_TILE_W-1:0] head_job_output_tile;
    logic head_job_decode_required, head_job_cache_release;
    logic head_job_last_input_head, head_job_last_output_tile;
    logic head_done_valid, head_done_ready;
    logic [TAG_W-1:0] head_done_tag;
    logic [HEAD_W-1:0] head_done_input_head;
    logic head_done_error;
    logic tile_done_valid, tile_done_ready;
    logic [TAG_W-1:0] tile_done_tag;
    logic tile_done_error;
    logic protocol_error;
    logic [31:0] perf_window_groups, perf_output_tiles, perf_head_replays;
    logic [31:0] perf_decode_intent_jobs, perf_release_intent_jobs;

    logic [15:0] lfsr_q;
    logic head_pending_q, tile_pending_q;
    logic [2:0] head_delay_q, tile_delay_q;
    logic [TAG_W-1:0] pending_head_tag_q, pending_tile_tag_q;
    logic [HEAD_W-1:0] pending_head_id_q;

    integer expected_descriptor;
    integer expected_window;
    integer expected_group;
    integer expected_tile;
    integer expected_head;
    integer seen_groups;
    integer seen_tiles;
    integer seen_heads;
    integer seen_decodes;
    integer seen_releases;
    integer done_pulses;
    integer stall_seed;
    integer elapsed_cycles;

    qfit_local5_encoder_job_scheduler dut (.*);

    always #5 clk_core = ~clk_core;

    function automatic integer descriptor_stage(input integer descriptor);
        if (descriptor < 2) descriptor_stage = 0;
        else if (descriptor < 4) descriptor_stage = 1;
        else if (descriptor < 10) descriptor_stage = 2;
        else descriptor_stage = 3;
    endfunction

    function automatic integer descriptor_block(input integer descriptor);
        case (descriptor)
            0, 2, 4, 10: descriptor_block = 0;
            1, 3, 5, 11: descriptor_block = 1;
            default: descriptor_block = descriptor - 4;
        endcase
    endfunction

    function automatic integer descriptor_heads(input integer descriptor);
        case (descriptor_stage(descriptor))
            0: descriptor_heads = 3;
            1: descriptor_heads = 6;
            2: descriptor_heads = 12;
            default: descriptor_heads = 24;
        endcase
    endfunction

    function automatic integer descriptor_windows(input integer descriptor);
        case (descriptor_stage(descriptor))
            0: descriptor_windows = 440;
            1: descriptor_windows = 120;
            2: descriptor_windows = 30;
            default: descriptor_windows = 10;
        endcase
    endfunction

    task automatic advance_group;
        begin
            expected_group = expected_group + 1;
            expected_window = expected_window + 1;
            if (expected_window == descriptor_windows(expected_descriptor)) begin
                expected_window = 0;
                expected_descriptor = expected_descriptor + 1;
            end
        end
    endtask

    assign tile_start_ready = !tile_pending_q && lfsr_q[0];
    assign head_job_ready = !head_pending_q && lfsr_q[1];
    assign head_done_valid = head_pending_q && head_delay_q == 0;
    assign head_done_tag = pending_head_tag_q;
    assign head_done_input_head = pending_head_id_q;
    assign head_done_error = 1'b0;
    assign tile_done_valid = tile_pending_q && tile_delay_q == 0;
    assign tile_done_tag = pending_tile_tag_q;
    assign tile_done_error = 1'b0;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            lfsr_q <= stall_seed[15:0];
            head_pending_q <= 1'b0;
            tile_pending_q <= 1'b0;
            head_delay_q <= '0;
            tile_delay_q <= '0;
            pending_head_tag_q <= '0;
            pending_head_id_q <= '0;
            pending_tile_tag_q <= '0;
        end else begin
            lfsr_q <= {lfsr_q[14:0], lfsr_q[15] ^ lfsr_q[13] ^ lfsr_q[12] ^ lfsr_q[10]};
            if (head_pending_q && head_delay_q != 0)
                head_delay_q <= head_delay_q - 1'b1;
            if (tile_pending_q && tile_delay_q != 0)
                tile_delay_q <= tile_delay_q - 1'b1;

            if (head_job_valid && head_job_ready) begin
                head_pending_q <= 1'b1;
                head_delay_q <= 3'd1 + {1'b0, lfsr_q[4:3]};
                pending_head_tag_q <= head_job_tag;
                pending_head_id_q <= head_job_input_head;
                if (head_job_last_input_head) begin
                    tile_pending_q <= 1'b1;
                    tile_delay_q <= 3'd1 + {1'b0, lfsr_q[7:6]};
                    pending_tile_tag_q <= head_job_tag;
                end
            end
            if (head_done_valid && head_done_ready)
                head_pending_q <= 1'b0;
            if (tile_done_valid && tile_done_ready)
                tile_pending_q <= 1'b0;
        end
    end

    always @(posedge clk_core) begin
        integer heads;
        integer stage;
        integer block;
        if (!rst_core) begin
            elapsed_cycles = elapsed_cycles + 1;
            heads = descriptor_heads(expected_descriptor);
            stage = descriptor_stage(expected_descriptor);
            block = descriptor_block(expected_descriptor);
            if (tile_start_valid && tile_start_ready) begin
                if (tile_start_stage != 2'(stage)
                    || tile_start_block != 3'(block)
                    || tile_start_window != WINDOW_W'(expected_window)
                    || tile_start_output_tile != OUTPUT_TILE_W'(expected_tile)
                    || tile_start_head_count != 6'(heads)
                    || tile_start_tag != TAG_W'(expected_group * 32 + expected_tile))
                    $fatal(1, "tile identity mismatch g=%0d t=%0d", expected_group, expected_tile);
                seen_tiles = seen_tiles + 1;
            end
            if (head_job_valid && head_job_ready) begin
                if (head_job_stage != 2'(stage)
                    || head_job_block != 3'(block)
                    || head_job_window != WINDOW_W'(expected_window)
                    || head_job_output_tile != OUTPUT_TILE_W'(expected_tile)
                    || head_job_input_head != HEAD_W'(expected_head)
                    || head_job_input_channel_base != 10'(expected_head * 32)
                    || head_job_tag != TAG_W'(expected_group * 32 + expected_tile)
                    || head_job_decode_required != (expected_tile == 0)
                    || head_job_cache_release != (expected_tile == heads - 1)
                    || head_job_last_input_head != (expected_head == heads - 1)
                    || head_job_last_output_tile != (expected_tile == heads - 1))
                    $fatal(1, "head identity mismatch g=%0d t=%0d h=%0d", expected_group, expected_tile, expected_head);
                seen_heads = seen_heads + 1;
                if (head_job_decode_required)
                    seen_decodes = seen_decodes + 1;
                if (head_job_cache_release)
                    seen_releases = seen_releases + 1;
                expected_head = expected_head + 1;
                if (expected_head == heads)
                    expected_head = 0;
            end
            if (tile_done_valid && tile_done_ready) begin
                if (tile_done_tag != TAG_W'(expected_group * 32 + expected_tile))
                    $fatal(1, "tile completion tag mismatch");
                expected_tile = expected_tile + 1;
                if (expected_tile == heads) begin
                    expected_tile = 0;
                    seen_groups = seen_groups + 1;
                    advance_group();
                end
            end
            if (frame_done)
                done_pulses = done_pulses + 1;
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        start_frame = 1'b0;
        expected_descriptor = 0;
        expected_window = 0;
        expected_group = 0;
        expected_tile = 0;
        expected_head = 0;
        seen_groups = 0;
        seen_tiles = 0;
        seen_heads = 0;
        seen_decodes = 0;
        seen_releases = 0;
        done_pulses = 0;
        elapsed_cycles = 0;
        stall_seed = 1;
        if (!$value$plusargs("STALL_SEED=%d", stall_seed))
            stall_seed = 1;
        if ((stall_seed & 16'hffff) == 0)
            stall_seed = 1;
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;
        @(negedge clk_core);
        start_frame = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        start_frame = 1'b0;

        wait (frame_done);
        @(posedge clk_core);
        #1;
        if (protocol_error || seen_groups != 1320 || seen_tiles != 6720
            || seen_heads != 54000 || seen_decodes != 6720
            || seen_releases != 6720 || done_pulses != 1
            || perf_window_groups != 1320 || perf_output_tiles != 6720
            || perf_head_replays != 54000
            || perf_decode_intent_jobs != 6720
            || perf_release_intent_jobs != 6720 || expected_group != 1320)
            $fatal(1, "frame ledger mismatch groups=%0d tiles=%0d heads=%0d decode=%0d release=%0d perf=%0d/%0d/%0d/%0d/%0d error=%0d",
                   seen_groups, seen_tiles, seen_heads, seen_decodes, seen_releases,
                   perf_window_groups, perf_output_tiles, perf_head_replays,
                   perf_decode_intent_jobs, perf_release_intent_jobs,
                   protocol_error);
        $display("PASS Local5 encoder scheduler seed=%0d cycles=%0d groups=%0d tiles=%0d replays=%0d decode=%0d release=%0d",
                 stall_seed, elapsed_cycles, seen_groups, seen_tiles, seen_heads,
                 seen_decodes, seen_releases);
        $finish;
    end

    initial begin
        repeat (2000000) @(posedge clk_core);
        $fatal(1, "Local5 encoder scheduler timeout");
    end
endmodule

`default_nettype wire
