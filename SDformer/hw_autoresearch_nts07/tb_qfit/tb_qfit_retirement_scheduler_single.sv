`timescale 1ns/1ps
`default_nettype none

module tb_qfit_retirement_scheduler_single #(
    parameter int MODE = 0,
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2
);
    localparam int TOKENS = HEIGHT * WIDTH;
    localparam int TOTAL = TOKENS * TIME_PLANES;
    localparam int Y_W = $clog2(HEIGHT);
    localparam int X_W = $clog2(WIDTH);
    localparam int PLANE_W = $clog2(TIME_PLANES);
    localparam int SOURCE_ID_W = $clog2(TOTAL);

    logic clk_core;
    logic rst_core;
    logic plane_start;
    logic [PLANE_W-1:0] plane_id;
    logic in_valid;
    logic in_ready;
    logic [Y_W-1:0] in_y;
    logic [X_W-1:0] in_x;
    logic [4:0] in_candidate_valid;
    logic retire_valid;
    logic retire_ready;
    logic [SOURCE_ID_W-1:0] retire_source_id;
    logic [Y_W-1:0] retire_y;
    logic [X_W-1:0] retire_x;
    logic [31:0] perf_producer_stalls;
    logic [2:0] perf_max_pending;

    int expected_sequence [0:TOTAL-1];
    bit retired_seen [0:TOTAL-1];
    int expected_count;
    int retired_count;
    int plane_cycles [0:TIME_PLANES-1];
    int cycle_count;

    qfit_retirement_scheduler #(
        .MODE(MODE),
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES)
    ) dut (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .plane_start(plane_start),
        .plane_id(plane_id),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_y(in_y),
        .in_x(in_x),
        .in_candidate_valid(in_candidate_valid),
        .retire_valid(retire_valid),
        .retire_ready(retire_ready),
        .retire_source_id(retire_source_id),
        .retire_y(retire_y),
        .retire_x(retire_x),
        .perf_producer_stalls(perf_producer_stalls),
        .perf_max_pending(perf_max_pending)
    );

    always #5 clk_core = ~clk_core;

    function automatic logic [4:0] candidate_mask(
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
        return mask;
    endfunction

    task automatic build_expected;
        expected_count = 0;
        for (int p = 0; p < TIME_PLANES; p = p + 1) begin
            if (MODE == 2) begin
                for (int sid = 0; sid < TOKENS; sid = sid + 1) begin
                    expected_sequence[expected_count] = p * TOKENS + sid;
                    expected_count = expected_count + 1;
                end
            end else begin
                for (int y = 0; y < HEIGHT; y = y + 1) begin
                    for (int x = 0; x < WIDTH; x = x + 1) begin
                        if (y > 0) begin
                            expected_sequence[expected_count] =
                                p * TOKENS + (y - 1) * WIDTH + x;
                            expected_count = expected_count + 1;
                        end
                        if (y == HEIGHT - 1 && x > 0) begin
                            expected_sequence[expected_count] =
                                p * TOKENS + y * WIDTH + x - 1;
                            expected_count = expected_count + 1;
                        end
                        if (
                            y == HEIGHT - 1
                            && x == WIDTH - 1
                        ) begin
                            expected_sequence[expected_count] =
                                p * TOKENS + y * WIDTH + x;
                            expected_count = expected_count + 1;
                        end
                    end
                end
            end
        end
        if (expected_count != TOTAL)
            $fatal(1, "expected count=%0d total=%0d", expected_count, TOTAL);
    endtask

    task automatic drive_plane(input int p);
        int accepted;
        int start_cycle;
        int watchdog;
        logic input_handshake;

        @(negedge clk_core);
        plane_id = PLANE_W'(p);
        plane_start = 1'b1;
        in_valid = 1'b0;
        @(negedge clk_core);
        plane_start = 1'b0;
        start_cycle = cycle_count;

        accepted = 0;
        in_y = '0;
        in_x = '0;
        in_candidate_valid = candidate_mask(0, 0);
        in_valid = 1'b1;
        while (accepted < TOKENS) begin
            @(posedge clk_core);
            input_handshake = in_ready;
            @(negedge clk_core);
            if (input_handshake) begin
                accepted = accepted + 1;
                if (accepted < TOKENS) begin
                    in_y = Y_W'(accepted / WIDTH);
                    in_x = X_W'(accepted % WIDTH);
                    in_candidate_valid = candidate_mask(
                        accepted / WIDTH,
                        accepted % WIDTH
                    );
                end else begin
                    in_valid = 1'b0;
                end
            end
        end

        watchdog = 0;
        while (retired_count < (p + 1) * TOKENS) begin
            @(negedge clk_core);
            watchdog = watchdog + 1;
            if (watchdog > TOTAL * 8)
                $fatal(
                    1,
                    "mode%0d plane%0d drain timeout retired=%0d",
                    MODE,
                    p,
                    retired_count
                );
        end
        plane_cycles[p] = cycle_count - start_cycle;
        repeat (2) @(negedge clk_core);
    endtask

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
            retire_ready <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (plane_id == 0)
                retire_ready <= 1'b1;
            else
                retire_ready <= ($urandom_range(0, 3) != 0);

            if (retire_valid && retire_ready) begin
                int sid;
                sid = int'(retire_source_id);
                if (sid < 0 || sid >= TOTAL)
                    $fatal(1, "out-of-range id=%0d", sid);
                if (retired_seen[sid])
                    $fatal(1, "duplicate id=%0d", sid);
                if (sid != expected_sequence[retired_count])
                    $fatal(
                        1,
                        "mode%0d sequence mismatch index=%0d got=%0d expected=%0d",
                        MODE,
                        retired_count,
                        sid,
                        expected_sequence[retired_count]
                    );
                if (
                    sid % TOKENS
                    != int'(retire_y) * WIDTH + int'(retire_x)
                )
                    $fatal(
                        1,
                        "coordinate mismatch id=%0d y=%0d x=%0d",
                        sid,
                        retire_y,
                        retire_x
                    );
                retired_seen[sid] <= 1'b1;
                retired_count <= retired_count + 1;
            end
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        plane_start = 1'b0;
        plane_id = '0;
        in_valid = 1'b0;
        in_y = '0;
        in_x = '0;
        in_candidate_valid = '0;
        retire_ready = 1'b0;
        retired_count = 0;
        cycle_count = 0;
        for (int sid = 0; sid < TOTAL; sid = sid + 1)
            retired_seen[sid] = 1'b0;
        build_expected();

        repeat (4) @(negedge clk_core);
        rst_core = 1'b0;

        for (int p = 0; p < TIME_PLANES; p = p + 1)
            drive_plane(p);

        if (retired_count != TOTAL)
            $fatal(1, "retired=%0d total=%0d", retired_count, TOTAL);
        for (int sid = 0; sid < TOTAL; sid = sid + 1)
            if (!retired_seen[sid])
                $fatal(1, "missing id=%0d", sid);
        if (perf_max_pending > 3'd2)
            $fatal(1, "pending bound=%0d", perf_max_pending);

        $display(
            "PASS mode=%0d H=%0d W=%0d plane0_cycles=%0d plane1_cycles=%0d stalls=%0d max_pending=%0d",
            MODE,
            HEIGHT,
            WIDTH,
            plane_cycles[0],
            plane_cycles[1],
            perf_producer_stalls,
            perf_max_pending
        );
        $finish;
    end
endmodule

`default_nettype wire
