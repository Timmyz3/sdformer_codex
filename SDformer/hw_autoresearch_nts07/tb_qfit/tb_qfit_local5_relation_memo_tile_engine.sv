`timescale 1ns/1ps
`default_nettype none

module tb_qfit_local5_relation_memo_tile_engine;
    localparam int HEIGHT = 15;
    localparam int WIDTH = 15;
    localparam int TIME_PLANES = 2;
    localparam int TOKENS = HEIGHT * WIDTH;
    localparam int OUT_DIM = 4;
    localparam int MAX_HEADS = 24;
    localparam int HEAD_W = $clog2(MAX_HEADS);
    localparam int PTR_W = $clog2(513);

    logic clk_core;
    logic rst_core;
    logic window_start;
    logic tile_start;
    logic tile_ready;
    logic tile_prefer_replay;
    logic [HEAD_W-1:0] tile_head_index;
    logic tile_done;
    logic fallback_taken;
    logic recompute_request;
    logic recompute_grant;
    logic plane_start;
    logic plane_id;
    logic in_valid;
    logic in_ready;
    logic [3:0] in_y;
    logic [3:0] in_x;
    logic [4:0] in_candidate_valid;
    logic [31:0] in_k_self;
    logic [44:0] in_direction_gates;
    logic plane_idle;
    logic weight_valid;
    logic weight_ready;
    logic [4:0] weight_lane;
    logic [1:0] weight_out;
    logic signed [7:0] weight_data;
    logic weight_last;
    logic read_valid;
    logic read_ready;
    logic read_plane;
    logic [3:0] read_y;
    logic [3:0] read_x;
    logic [1:0] read_out;
    logic read_data_valid;
    logic signed [31:0] read_data;
    logic descriptor_valid;
    logic descriptor_ready;
    logic [8:0] descriptor_source_id;
    logic [3:0] descriptor_y;
    logic [3:0] descriptor_x;
    logic [31:0] descriptor_k;
    logic [44:0] descriptor_gates;
    logic [4:0] descriptor_valid_mask;
    logic descriptor_last;
    logic descriptor_stream_idle;
    logic head_done;
    logic head_resident;
    logic head_critical;
    logic head_overflow;
    logic [31:0] head_service_cycles;
    logic [PTR_W-1:0] head_record_count;
    logic protocol_error;
    logic [31:0] perf_speculative_writes;
    logic [31:0] perf_discarded_writes;
    logic [31:0] perf_committed_records;
    logic [31:0] perf_replay_reads;
    logic [31:0] perf_capacity_misses;
    logic [31:0] perf_descriptors;
    logic [31:0] perf_product_terms;
    logic [31:0] perf_destination_updates;

    integer weight_ref [0:31][0:OUT_DIM-1];
    integer ref_acc [0:1][0:TIME_PLANES-1][0:HEIGHT-1][0:WIDTH-1][0:OUT_DIM-1];
    integer prior_acc [0:1][0:TIME_PLANES-1][0:HEIGHT-1][0:WIDTH-1][0:OUT_DIM-1];
    int recompute_count;
    int tile_descriptor_count;
    int current_head;

    qfit_local5_relation_memo_tile_engine dut (.*);

    always #5 clk_core = ~clk_core;

    function automatic logic [4:0] candidate_mask(
        input int p,
        input int y,
        input int x
    );
        logic [4:0] mask;
        mask = 5'b00001;
        if (y > 0)
            mask[1] = 1'b1;
        if (y < HEIGHT - 1)
            mask[2] = 1'b1;
        if (x > 0)
            mask[3] = 1'b1;
        if (x < WIDTH - 1)
            mask[4] = 1'b1;
        if (p == 1 && y == 7 && x == 7)
            mask[0] = 1'b0;
        if (p == 1 && y == 8 && x == 7)
            mask[1] = 1'b0;
        if (p == 1 && y == 7 && x == 8)
            mask[3] = 1'b0;
        return mask;
    endfunction

    function automatic logic source_is_active(
        input int head,
        input int p,
        input int y,
        input int x
    );
        int source_id;
        source_id = p * TOKENS + y * WIDTH + x;
        if (head == 0)
            source_is_active = source_id < 10
                || (
                    source_id >= TOKENS + 7 * WIDTH + 3
                    && source_id < TOKENS + 7 * WIDTH + 13
                );
        else
            source_is_active = 1'b1;
    endfunction

    task automatic load_weights;
        for (int lane = 0; lane < 32; lane = lane + 1) begin
            for (int out = 0; out < OUT_DIM; out = out + 1) begin
                while (!weight_ready)
                    @(negedge clk_core);
                weight_ref[lane][out] = ((lane * 3 + out * 2) % 11) - 5;
                weight_lane = 5'(lane);
                weight_out = 2'(out);
                weight_data = 8'(weight_ref[lane][out]);
                weight_last = lane == 31 && out == OUT_DIM - 1;
                weight_valid = 1'b1;
                @(negedge clk_core);
            end
        end
        weight_valid = 1'b0;
        weight_last = 1'b0;
    endtask

    task automatic build_raw_references;
        int sy;
        int sx;
        logic [4:0] mask;
        for (int head = 0; head < 2; head = head + 1) begin
            for (int p = 0; p < TIME_PLANES; p = p + 1) begin
                for (int y = 0; y < HEIGHT; y = y + 1) begin
                    for (int x = 0; x < WIDTH; x = x + 1) begin
                        mask = candidate_mask(p, y, x);
                        for (int role = 0; role < 5; role = role + 1) begin
                            if (mask[role]) begin
                                sy = y;
                                sx = x;
                                case (role)
                                    1: sy = y - 1;
                                    2: sy = y + 1;
                                    3: sx = x - 1;
                                    4: sx = x + 1;
                                    default: begin
                                        sy = y;
                                        sx = x;
                                    end
                                endcase
                                if (source_is_active(head, p, sy, sx))
                                    for (
                                        int out = 0;
                                        out < OUT_DIM;
                                        out = out + 1
                                    )
                                        ref_acc[head][p][y][x][out] =
                                            ref_acc[head][p][y][x][out]
                                            + 7 * weight_ref[0][out];
                            end
                        end
                    end
                end
            end
        end
    endtask

    task automatic drive_plane(input int head, input int p);
        int accepted;
        bit fire;
        while (!plane_idle)
            @(negedge clk_core);
        plane_id = p[0];
        plane_start = 1'b1;
        @(negedge clk_core);
        plane_start = 1'b0;
        accepted = 0;
        in_valid = 1'b1;
        while (accepted < TOKENS) begin
            in_y = 4'(accepted / WIDTH);
            in_x = 4'(accepted % WIDTH);
            in_candidate_valid = candidate_mask(
                p,
                accepted / WIDTH,
                accepted % WIDTH
            );
            in_k_self = source_is_active(
                head,
                p,
                accepted / WIDTH,
                accepted % WIDTH
            ) ? 32'h0000_0001 : '0;
            in_direction_gates = {5{9'd7}};
            @(posedge clk_core);
            fire = in_ready;
            @(negedge clk_core);
            if (fire)
                accepted = accepted + 1;
        end
        in_valid = 1'b0;
        while (!plane_idle)
            @(negedge clk_core);
    endtask

    task automatic issue_tile(
        input int head,
        input bit prefer_replay,
        input bit expect_recompute,
        input bit expect_fallback
    );
        while (!tile_ready)
            @(negedge clk_core);
        current_head = head;
        tile_descriptor_count = 0;
        tile_head_index = HEAD_W'(head);
        tile_prefer_replay = prefer_replay;
        tile_start = 1'b1;
        @(negedge clk_core);
        tile_start = 1'b0;
        if (expect_recompute) begin
            while (!recompute_request)
                @(negedge clk_core);
            recompute_count = recompute_count + 1;
            recompute_grant = 1'b1;
            @(negedge clk_core);
            recompute_grant = 1'b0;
            for (int p = 0; p < TIME_PLANES; p = p + 1)
                drive_plane(head, p);
        end
        while (!tile_done)
            @(negedge clk_core);
        if (fallback_taken != expect_fallback)
            $fatal(1, "fallback decision mismatch head=%0d", head);
        if (expect_recompute && tile_descriptor_count != TIME_PLANES * TOKENS)
            $fatal(1, "recompute descriptor count mismatch head=%0d", head);
        if (!expect_recompute && head == 0 && tile_descriptor_count != 20)
            $fatal(1, "resident replay descriptor count mismatch");
    endtask

    task automatic read_and_check(
        input int head,
        input bit compare_prior
    );
        bit fire;
        for (int p = 0; p < TIME_PLANES; p = p + 1) begin
            for (int y = 0; y < HEIGHT; y = y + 1) begin
                for (int x = 0; x < WIDTH; x = x + 1) begin
                    for (int out = 0; out < OUT_DIM; out = out + 1) begin
                        while (!read_ready)
                            @(negedge clk_core);
                        read_plane = p[0];
                        read_y = 4'(y);
                        read_x = 4'(x);
                        read_out = 2'(out);
                        read_valid = 1'b1;
                        @(posedge clk_core);
                        fire = read_ready;
                        @(negedge clk_core);
                        read_valid = 1'b0;
                        if (!fire)
                            $fatal(1, "read handshake lost");
                        while (!read_data_valid)
                            @(negedge clk_core);
                        if ($signed(read_data) != ref_acc[head][p][y][x][out])
                            $fatal(
                                1,
                                "raw-input Acc32 mismatch head=%0d p=%0d y=%0d x=%0d out=%0d got=%0d exp=%0d",
                                head,
                                p,
                                y,
                                x,
                                out,
                                $signed(read_data),
                                ref_acc[head][p][y][x][out]
                            );
                        if (compare_prior) begin
                            if (
                                $signed(read_data)
                                != prior_acc[head][p][y][x][out]
                            )
                                $fatal(1, "first/reuse Acc32 mismatch head=%0d", head);
                        end else begin
                            prior_acc[head][p][y][x][out]
                                = $signed(read_data);
                        end
                    end
                end
            end
        end
    endtask

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            tile_descriptor_count <= 0;
        end else if (descriptor_valid && descriptor_ready) begin
            tile_descriptor_count <= tile_descriptor_count + 1;
            if (descriptor_k != 0 && descriptor_gates == 0)
                $fatal(1, "active descriptor lost gates head=%0d", current_head);
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        window_start = 1'b0;
        tile_start = 1'b0;
        tile_prefer_replay = 1'b0;
        tile_head_index = '0;
        recompute_grant = 1'b0;
        plane_start = 1'b0;
        plane_id = 1'b0;
        in_valid = 1'b0;
        in_y = '0;
        in_x = '0;
        in_candidate_valid = '0;
        in_k_self = '0;
        in_direction_gates = '0;
        weight_valid = 1'b0;
        weight_lane = '0;
        weight_out = '0;
        weight_data = '0;
        weight_last = 1'b0;
        read_valid = 1'b0;
        read_plane = 1'b0;
        read_y = '0;
        read_x = '0;
        read_out = '0;
        recompute_count = 0;
        tile_descriptor_count = 0;
        current_head = 0;
        for (int head = 0; head < 2; head = head + 1)
            for (int p = 0; p < TIME_PLANES; p = p + 1)
                for (int y = 0; y < HEIGHT; y = y + 1)
                    for (int x = 0; x < WIDTH; x = x + 1)
                        for (int out = 0; out < OUT_DIM; out = out + 1) begin
                            ref_acc[head][p][y][x][out] = 0;
                            prior_acc[head][p][y][x][out] = 0;
                        end
        repeat (5) @(negedge clk_core);
        rst_core = 1'b0;
        load_weights();
        build_raw_references();
        window_start = 1'b1;
        @(negedge clk_core);
        window_start = 1'b0;

        issue_tile(0, 1'b0, 1'b1, 1'b0);
        if (!head_resident || !head_critical || head_record_count != 20)
            $fatal(1, "sparse head admission mismatch");
        read_and_check(0, 1'b0);

        issue_tile(0, 1'b1, 1'b0, 1'b0);
        read_and_check(0, 1'b1);

        issue_tile(1, 1'b0, 1'b1, 1'b0);
        if (head_resident || head_critical || head_service_cycles != 465)
            $fatal(1, "dense head admission mismatch");
        read_and_check(1, 1'b0);

        issue_tile(1, 1'b1, 1'b1, 1'b1);
        read_and_check(1, 1'b1);

        if (recompute_count != 3)
            $fatal(1, "recompute request count mismatch");
        if (protocol_error)
            $fatal(1, "unexpected protocol error");
        if (perf_committed_records != 20 || perf_replay_reads != 20)
            $fatal(1, "resident counter mismatch");
        if (perf_discarded_writes != 900)
            $fatal(1, "fallback rollback count mismatch");
        if (perf_capacity_misses != 0)
            $fatal(1, "unexpected capacity miss");
        $display(
            "PASS mixed resident/replay/nonresident/fallback tiles=4 recompute=%0d acc32=%0d",
            recompute_count,
            4 * TIME_PLANES * TOKENS * OUT_DIM
        );
        $finish;
    end
endmodule

`default_nettype wire
