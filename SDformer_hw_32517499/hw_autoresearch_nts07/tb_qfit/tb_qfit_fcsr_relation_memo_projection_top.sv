`timescale 1ns/1ps
`default_nettype none

module tb_qfit_fcsr_relation_memo_projection_top;
    localparam int HEIGHT = 15;
    localparam int WIDTH = 15;
    localparam int TIME_PLANES = 2;
    localparam int TOKENS = HEIGHT * WIDTH;
    localparam int TOTAL = TOKENS * TIME_PLANES;
    localparam int OUT_DIM = 4;
    localparam int MAX_HEADS = 24;
    localparam int HEAD_W = $clog2(MAX_HEADS);
    localparam int PTR_W = $clog2(513);

    logic clk_core;
    logic rst_core;
    logic window_start;
    logic head_start;
    logic head_ready;
    logic [HEAD_W-1:0] head_index;
    logic head_done;
    logic head_resident;
    logic head_critical;
    logic head_overflow;
    logic [31:0] head_service_cycles;
    logic [PTR_W-1:0] head_record_count;
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
    logic use_replay;
    logic replay_start;
    logic replay_cmd_ready;
    logic [HEAD_W-1:0] replay_head_index;
    logic replay_done;
    logic replay_miss;
    logic weight_valid;
    logic weight_ready;
    logic [4:0] weight_lane;
    logic [1:0] weight_out;
    logic signed [7:0] weight_data;
    logic weight_last;
    logic weight_context_release;
    logic weight_context_release_ready;
    logic projection_start;
    logic projection_accumulate;
    logic projection_close;
    logic projection_close_ready;
    logic projection_busy;
    logic projection_done;
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
    integer ref_acc [0:TIME_PLANES-1][0:HEIGHT-1][0:WIDTH-1][0:OUT_DIM-1];
    integer live_acc [0:TIME_PLANES-1][0:HEIGHT-1][0:WIDTH-1][0:OUT_DIM-1];
    integer active_source [0:31];
    logic [31:0] active_k [0:31];
    logic [44:0] active_gates [0:31];
    logic [4:0] active_mask [0:31];
    int live_descriptor_count;
    int active_descriptor_count;
    int replay_descriptor_count;
    bit head_done_seen;
    bit replay_done_seen;

    qfit_fcsr_relation_memo_projection_top dut (.*);

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
        input int p,
        input int y,
        input int x
    );
        int source_id;
        source_id = p * TOKENS + y * WIDTH + x;
        source_is_active = source_id < 10
            || (
                source_id >= TOKENS + 7 * WIDTH + 3
                && source_id < TOKENS + 7 * WIDTH + 13
            );
    endfunction

    function automatic logic [8:0] gate_value(
        input int p,
        input int y,
        input int x,
        input int role
    );
        gate_value = 9'd7;
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
        weight_context_release = 1'b0;
    endtask

    task automatic drive_plane(input int p);
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
                p,
                accepted / WIDTH,
                accepted % WIDTH
            ) ? 32'h0000_0001 : '0;
            for (int role = 0; role < 5; role = role + 1)
                in_direction_gates[role*9 +: 9] = gate_value(
                    p,
                    accepted / WIDTH,
                    accepted % WIDTH,
                    role
                );
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

    task automatic close_projection;
        while (!projection_close_ready)
            @(negedge clk_core);
        projection_close = 1'b1;
        @(negedge clk_core);
        projection_close = 1'b0;
        while (!projection_done)
            @(negedge clk_core);
    endtask

    task automatic read_and_check(input bit save_live);
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
                        if ($signed(read_data) != ref_acc[p][y][x][out])
                            $fatal(
                                1,
                                "Acc32 reference mismatch mode=%0d p=%0d y=%0d x=%0d out=%0d got=%0d exp=%0d",
                                save_live,
                                p,
                                y,
                                x,
                                out,
                                $signed(read_data),
                                ref_acc[p][y][x][out]
                            );
                        if (save_live)
                            live_acc[p][y][x][out] = $signed(read_data);
                        else if (
                            $signed(read_data) != live_acc[p][y][x][out]
                        )
                            $fatal(1, "live/replay Acc32 mismatch");
                    end
                end
            end
        end
    endtask

    task automatic build_reference_from_raw_inputs;
        int sy;
        int sx;
        int gate;
        logic [4:0] mask;
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
                            if (source_is_active(p, sy, sx)) begin
                                gate = gate_value(p, y, x, role);
                                for (
                                    int out = 0;
                                    out < OUT_DIM;
                                    out = out + 1
                                )
                                    ref_acc[p][y][x][out] =
                                        ref_acc[p][y][x][out]
                                        + gate * weight_ref[0][out];
                            end
                        end
                    end
                end
            end
        end
    endtask

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            live_descriptor_count <= 0;
            active_descriptor_count <= 0;
            replay_descriptor_count <= 0;
            head_done_seen <= 1'b0;
            replay_done_seen <= 1'b0;
        end else begin
            if (head_done)
                head_done_seen <= 1'b1;
            if (replay_done)
                replay_done_seen <= 1'b1;
            if (descriptor_valid && descriptor_ready) begin
                if (!use_replay) begin
                    live_descriptor_count <= live_descriptor_count + 1;
                    if (descriptor_k != 0) begin
                        active_source[active_descriptor_count]
                            <= descriptor_source_id;
                        active_k[active_descriptor_count] <= descriptor_k;
                        active_gates[active_descriptor_count]
                            <= descriptor_gates;
                        active_mask[active_descriptor_count]
                            <= descriptor_valid_mask;
                        active_descriptor_count
                            <= active_descriptor_count + 1;
                    end
                end else begin
                    if (
                        descriptor_source_id
                        != 9'(active_source[replay_descriptor_count])
                        || descriptor_k
                           != active_k[replay_descriptor_count]
                        || descriptor_gates
                           != active_gates[replay_descriptor_count]
                        || descriptor_valid_mask
                           != active_mask[replay_descriptor_count]
                    )
                        $fatal(1, "replay descriptor mismatch index=%0d", replay_descriptor_count);
                    replay_descriptor_count <= replay_descriptor_count + 1;
                end
            end
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        window_start = 1'b0;
        head_start = 1'b0;
        head_index = '0;
        plane_start = 1'b0;
        plane_id = 1'b0;
        in_valid = 1'b0;
        in_y = '0;
        in_x = '0;
        in_candidate_valid = '0;
        in_k_self = '0;
        in_direction_gates = '0;
        use_replay = 1'b0;
        replay_start = 1'b0;
        replay_head_index = '0;
        weight_valid = 1'b0;
        weight_lane = '0;
        weight_out = '0;
        weight_data = '0;
        weight_last = 1'b0;
        projection_start = 1'b0;
        projection_accumulate = 1'b0;
        projection_close = 1'b0;
        read_valid = 1'b0;
        read_plane = 1'b0;
        read_y = '0;
        read_x = '0;
        read_out = '0;
        for (int p = 0; p < TIME_PLANES; p = p + 1)
            for (int y = 0; y < HEIGHT; y = y + 1)
                for (int x = 0; x < WIDTH; x = x + 1)
                    for (int out = 0; out < OUT_DIM; out = out + 1) begin
                        ref_acc[p][y][x][out] = 0;
                        live_acc[p][y][x][out] = 0;
                    end
        repeat (5) @(negedge clk_core);
        rst_core = 1'b0;
        load_weights();
        build_reference_from_raw_inputs();

        window_start = 1'b1;
        @(negedge clk_core);
        window_start = 1'b0;
        projection_start = 1'b1;
        @(negedge clk_core);
        projection_start = 1'b0;
        while (!head_ready)
            @(negedge clk_core);
        head_index = '0;
        head_start = 1'b1;
        @(negedge clk_core);
        head_start = 1'b0;
        for (int p = 0; p < TIME_PLANES; p = p + 1)
            drive_plane(p);
        while (!head_done_seen || !descriptor_stream_idle)
            @(negedge clk_core);
        if (!head_resident || !head_critical || head_overflow)
            $fatal(1, "resident admission mismatch");
        if (head_service_cycles != 35 || head_record_count != 20)
            $fatal(1, "resident service contract mismatch");
        if (live_descriptor_count != TOTAL || active_descriptor_count != 20)
            $fatal(1, "live descriptor count mismatch");
        close_projection();
        read_and_check(1'b1);

        use_replay = 1'b1;
        projection_start = 1'b1;
        @(negedge clk_core);
        projection_start = 1'b0;
        while (!replay_cmd_ready)
            @(negedge clk_core);
        replay_head_index = '0;
        replay_start = 1'b1;
        @(negedge clk_core);
        replay_start = 1'b0;
        while (!replay_done_seen || !descriptor_stream_idle)
            @(negedge clk_core);
        if (replay_miss)
            $fatal(1, "unexpected replay miss");
        if (replay_descriptor_count != 20)
            $fatal(1, "replay descriptor count mismatch");
        close_projection();
        read_and_check(1'b0);

        if (protocol_error)
            $fatal(1, "unexpected protocol error");
        if (perf_committed_records != 20 || perf_replay_reads != 20)
            $fatal(1, "vault performance counter mismatch");
        if (perf_discarded_writes != 0 || perf_capacity_misses != 0)
            $fatal(1, "unexpected vault discard");
        $display(
            "PASS FCSR-vault-term-TCFM5 Acc32 miter live_desc=%0d replay_desc=%0d acc32=%0d",
            live_descriptor_count,
            replay_descriptor_count,
            TOTAL * OUT_DIM
        );
        $finish;
    end
endmodule

`default_nettype wire
