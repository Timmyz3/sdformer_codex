`timescale 1ns/1ps
`default_nettype none

module tb_qfit_retirement_scheduler;
    localparam int HEIGHT = 3;
    localparam int WIDTH = 4;
    localparam int TIME_PLANES = 2;
    localparam int TOKENS = HEIGHT * WIDTH;
    localparam int TOTAL = TOKENS * TIME_PLANES;
    localparam int Y_W = $clog2(HEIGHT);
    localparam int X_W = $clog2(WIDTH);
    localparam int PLANE_W = 1;
    localparam int SOURCE_ID_W = $clog2(TOTAL);

    logic clk_core;
    logic rst_core;
    logic plane_start [0:2];
    logic [PLANE_W-1:0] plane_id [0:2];
    logic source_valid [0:2];
    logic in_ready [0:2];
    logic [Y_W-1:0] in_y [0:2];
    logic [X_W-1:0] in_x [0:2];
    logic [4:0] in_candidate_valid [0:2];
    logic retire_valid [0:2];
    logic retire_ready [0:2];
    logic [SOURCE_ID_W-1:0] retire_source_id [0:2];
    logic [Y_W-1:0] retire_y [0:2];
    logic [X_W-1:0] retire_x [0:2];
    logic [31:0] perf_stalls [0:2];
    logic [2:0] perf_max_pending [0:2];

    int retired_count [0:2];
    int retired_sequence [0:2][0:TOTAL-1];
    bit retired_seen [0:2][0:TOTAL-1];
    int plane_cycles [0:2][0:TIME_PLANES-1];
    int cycle_count;

    for (genvar mode = 0; mode < 3; mode = mode + 1) begin : g_dut
        qfit_retirement_scheduler #(
            .MODE(mode),
            .HEIGHT(HEIGHT),
            .WIDTH(WIDTH),
            .TIME_PLANES(TIME_PLANES)
        ) dut (
            .clk_core(clk_core),
            .rst_core(rst_core),
            .plane_start(plane_start[mode]),
            .plane_id(plane_id[mode]),
            .in_valid(source_valid[mode]),
            .in_ready(in_ready[mode]),
            .in_y(in_y[mode]),
            .in_x(in_x[mode]),
            .in_candidate_valid(in_candidate_valid[mode]),
            .retire_valid(retire_valid[mode]),
            .retire_ready(retire_ready[mode]),
            .retire_source_id(retire_source_id[mode]),
            .retire_y(retire_y[mode]),
            .retire_x(retire_x[mode]),
            .perf_producer_stalls(perf_stalls[mode]),
            .perf_max_pending(perf_max_pending[mode])
        );
    end

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

    task automatic send_pixel(
        input int mode,
        input int y,
        input int x
    );
        @(negedge clk_core);
        in_y[mode] = Y_W'(y);
        in_x[mode] = X_W'(x);
        in_candidate_valid[mode] = candidate_mask(y, x);
        source_valid[mode] = 1'b1;
        do begin
            @(negedge clk_core);
        end while (!in_ready[mode]);
        source_valid[mode] = 1'b0;
    endtask

    task automatic drive_mode(input int mode);
        int start_cycle;
        int watchdog;
        for (int p = 0; p < TIME_PLANES; p = p + 1) begin
            while (retired_count[mode] != p * TOKENS)
                @(negedge clk_core);

            @(negedge clk_core);
            plane_id[mode] = PLANE_W'(p);
            plane_start[mode] = 1'b1;
            source_valid[mode] = 1'b0;
            @(negedge clk_core);
            plane_start[mode] = 1'b0;
            start_cycle = cycle_count;

            for (int y = 0; y < HEIGHT; y = y + 1)
                for (int x = 0; x < WIDTH; x = x + 1)
                    send_pixel(mode, y, x);

            watchdog = 0;
            while (retired_count[mode] < (p + 1) * TOKENS) begin
                @(negedge clk_core);
                watchdog = watchdog + 1;
                if (watchdog > 500) begin
                    if (mode == 2)
                        $display(
                            "stripe debug p=%0d retired=%0d valid=%0b ready=%0b context=%0d head=%0b tail=%0b x=%0d",
                            p,
                            retired_count[mode],
                            retire_valid[mode],
                            retire_ready[mode],
                            g_dut[2].dut.stripe_context_count_q,
                            g_dut[2].dut.stripe_head_q,
                            g_dut[2].dut.stripe_tail_q,
                            g_dut[2].dut.stripe_x_q
                        );
                    $fatal(
                        1,
                        "mode%0d plane%0d retirement timeout count=%0d",
                        mode,
                        p,
                        retired_count[mode]
                    );
                end
            end
            plane_cycles[mode][p] = cycle_count - start_cycle;
            repeat (2) @(negedge clk_core);
        end
    endtask

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
            for (int mode = 0; mode < 3; mode = mode + 1)
                retire_ready[mode] <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;
            for (int mode = 0; mode < 3; mode = mode + 1) begin
                if (plane_id[mode] == 0)
                    retire_ready[mode] <= 1'b1;
                else
                    retire_ready[mode]
                        <= ($urandom_range(0, 3) != 0);

                if (retire_valid[mode] && retire_ready[mode]) begin
                    int sid;
                    sid = int'(retire_source_id[mode]);
                    if (sid < 0 || sid >= TOTAL)
                        $fatal(1, "mode%0d out-of-range id=%0d", mode, sid);
                    if (retired_seen[mode][sid])
                        $fatal(1, "mode%0d duplicate id=%0d", mode, sid);
                    retired_seen[mode][sid] <= 1'b1;
                    retired_sequence[mode][retired_count[mode]] <= sid;
                    retired_count[mode] <= retired_count[mode] + 1;
                end
            end
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        cycle_count = 0;
        for (int mode = 0; mode < 3; mode = mode + 1) begin
            plane_start[mode] = 1'b0;
            source_valid[mode] = 1'b0;
            retire_ready[mode] = 1'b0;
            plane_id[mode] = '0;
            in_y[mode] = '0;
            in_x[mode] = '0;
            in_candidate_valid[mode] = '0;
            retired_count[mode] = 0;
            for (int sid = 0; sid < TOTAL; sid = sid + 1)
                retired_seen[mode][sid] = 1'b0;
        end

        repeat (4) @(negedge clk_core);
        rst_core = 1'b0;

        fork
            drive_mode(0);
            drive_mode(1);
            drive_mode(2);
        join

        for (int mode = 0; mode < 3; mode = mode + 1) begin
            if (retired_count[mode] != TOTAL)
                $fatal(
                    1,
                    "mode%0d count=%0d expected=%0d",
                    mode,
                    retired_count[mode],
                    TOTAL
                );
            for (int sid = 0; sid < TOTAL; sid = sid + 1)
                if (!retired_seen[mode][sid])
                    $fatal(1, "mode%0d missing id=%0d", mode, sid);
        end

        for (int index = 0; index < TOTAL; index = index + 1)
            if (
                retired_sequence[0][index]
                != retired_sequence[1][index]
            )
                $fatal(
                    1,
                    "FCSR/dynamic mismatch index=%0d fcsr=%0d dynamic=%0d",
                    index,
                    retired_sequence[0][index],
                    retired_sequence[1][index]
                );

        if (perf_max_pending[0] != 3'd2)
            $fatal(1, "FCSR max_pending=%0d", perf_max_pending[0]);
        if (perf_max_pending[1] != 3'd2)
            $fatal(1, "dynamic max_pending=%0d", perf_max_pending[1]);
        if (perf_max_pending[2] > 3'd2)
            $fatal(1, "stripe max_pending=%0d", perf_max_pending[2]);

        $display(
            "PASS qfit_retirement_scheduler plane0_cycles fcsr=%0d dynamic=%0d stripe=%0d plane1_cycles=%0d/%0d/%0d stalls=%0d/%0d/%0d",
            plane_cycles[0][0],
            plane_cycles[1][0],
            plane_cycles[2][0],
            plane_cycles[0][1],
            plane_cycles[1][1],
            plane_cycles[2][1],
            perf_stalls[0],
            perf_stalls[1],
            perf_stalls[2]
        );
        $finish;
    end
endmodule

`default_nettype wire
