`timescale 1ns/1ps
`default_nettype none

module tb_qfit_banked_dynamic_retirement_miter #(
    parameter int HEIGHT = 5,
    parameter int WIDTH = 7,
    parameter int DILATION = 1
);
    localparam int TIME_PLANES = 2;
    localparam int TOKENS = HEIGHT * WIDTH;
    localparam int TOTAL = TOKENS * TIME_PLANES;
    localparam int Y_W = $clog2(HEIGHT);
    localparam int X_W = $clog2(WIDTH);
    localparam int SOURCE_ID_W = $clog2(TOTAL);

    logic clk_core;
    logic rst_core;
    logic plane_start;
    logic plane_id;
    logic in_valid;
    logic [Y_W-1:0] in_y;
    logic [X_W-1:0] in_x;
    logic [4:0] in_candidate_valid;
    logic [4:0] compiled_candidate_valid;
    logic retire_ready;

    logic ref_in_ready;
    logic ref_retire_valid;
    logic [SOURCE_ID_W-1:0] ref_retire_source_id;
    logic [Y_W-1:0] ref_retire_y;
    logic [X_W-1:0] ref_retire_x;
    logic ref_plane_idle;
    logic [31:0] ref_perf_stalls;
    logic [2:0] ref_perf_pending;

    logic bank_in_ready;
    logic bank_retire_valid;
    logic [SOURCE_ID_W-1:0] bank_retire_source_id;
    logic [Y_W-1:0] bank_retire_y;
    logic [X_W-1:0] bank_retire_x;
    logic bank_plane_idle;
    logic [31:0] bank_perf_stalls;
    logic [2:0] bank_perf_pending;

    logic compiled_in_ready;
    logic compiled_retire_valid;
    logic [SOURCE_ID_W-1:0] compiled_retire_source_id;
    logic [Y_W-1:0] compiled_retire_y;
    logic [X_W-1:0] compiled_retire_x;
    logic compiled_plane_idle;
    logic [31:0] compiled_perf_stalls;
    logic [2:0] compiled_perf_pending;

    bit seen [0:TOTAL-1];
    int seed;
    int retire_count;
    int expected_retire_count;
    int long_stall_remaining;
    bit long_stall_started;

    qfit_retirement_scheduler #(
        .MODE(1),
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .DILATION(DILATION)
    ) u_reference (
        .clk_core,
        .rst_core,
        .plane_start,
        .plane_id,
        .in_valid,
        .in_ready(ref_in_ready),
        .in_y,
        .in_x,
        .in_candidate_valid,
        .retire_valid(ref_retire_valid),
        .retire_ready,
        .retire_source_id(ref_retire_source_id),
        .retire_y(ref_retire_y),
        .retire_x(ref_retire_x),
        .plane_idle(ref_plane_idle),
        .perf_producer_stalls(ref_perf_stalls),
        .perf_max_pending(ref_perf_pending)
    );

    qfit_banked_dynamic_retirement_scheduler #(
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .DILATION(DILATION)
    ) u_banked (
        .clk_core,
        .rst_core,
        .plane_start,
        .plane_id,
        .in_valid,
        .in_ready(bank_in_ready),
        .in_y,
        .in_x,
        .in_candidate_valid,
        .retire_valid(bank_retire_valid),
        .retire_ready,
        .retire_source_id(bank_retire_source_id),
        .retire_y(bank_retire_y),
        .retire_x(bank_retire_x),
        .plane_idle(bank_plane_idle),
        .perf_producer_stalls(bank_perf_stalls),
        .perf_max_pending(bank_perf_pending)
    );

    qfit_retirement_scheduler #(
        .MODE(0),
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .DILATION(DILATION),
        .FILTER_FCSR_EVENTS(1'b1)
    ) u_compiled_static (
        .clk_core,
        .rst_core,
        .plane_start,
        .plane_id,
        .in_valid,
        .in_ready(compiled_in_ready),
        .in_y,
        .in_x,
        .in_candidate_valid(compiled_candidate_valid),
        .retire_valid(compiled_retire_valid),
        .retire_ready,
        .retire_source_id(compiled_retire_source_id),
        .retire_y(compiled_retire_y),
        .retire_x(compiled_retire_x),
        .plane_idle(compiled_plane_idle),
        .perf_producer_stalls(compiled_perf_stalls),
        .perf_max_pending(compiled_perf_pending)
    );

    always #5 clk_core = ~clk_core;

    function automatic bit source_active(input int p, input int y, input int x);
        int sid;
        sid = p * TOKENS + y * WIDTH + x;
        if (seed == 99)
            // For DILATION=2, y=0 and y=10 reuse the same five-row ring
            // address and bank while being separated by two generations.
            source_active = x == 3 && (y == 0 || y == 10);
        else
            source_active = ((sid * 13 + seed * 5 + y * 3 + x) % 7) != 0;
    endfunction

    function automatic logic [4:0] candidate_mask(
        input int p,
        input int y,
        input int x
    );
        logic [4:0] mask;
        mask = '0;
        mask[0] = source_active(p, y, x);
        if (y >= DILATION)
            mask[1] = source_active(p, y - DILATION, x);
        if (y < HEIGHT - DILATION)
            mask[2] = source_active(p, y + DILATION, x);
        if (x >= DILATION)
            mask[3] = source_active(p, y, x - DILATION);
        if (x < WIDTH - DILATION)
            mask[4] = source_active(p, y, x + DILATION);
        return mask;
    endfunction

    function automatic logic [4:0] compiled_mask(
        input int p,
        input int y,
        input int x
    );
        logic [4:0] mask;
        mask = '0;
        if (y >= DILATION)
            mask[0] = source_active(p, y - DILATION, x);
        if (x >= DILATION)
            mask[1] = source_active(p, y, x - DILATION);
        mask[2] = source_active(p, y, x);
        return mask;
    endfunction

    task automatic drive_plane(input int p);
        int accepted;
        logic handshake;
        while (!ref_plane_idle || !bank_plane_idle || !compiled_plane_idle)
            @(negedge clk_core);
        @(negedge clk_core);
        plane_id = p[0];
        plane_start = 1'b1;
        in_valid = 1'b0;
        @(negedge clk_core);
        plane_start = 1'b0;
        accepted = 0;
        while (accepted < TOKENS) begin
            in_valid = ($urandom_range(0, 4) != 0);
            in_y = Y_W'(accepted / WIDTH);
            in_x = X_W'(accepted % WIDTH);
            in_candidate_valid = candidate_mask(
                p,
                accepted / WIDTH,
                accepted % WIDTH
            );
            compiled_candidate_valid = compiled_mask(
                p,
                accepted / WIDTH,
                accepted % WIDTH
            );
            @(posedge clk_core);
            handshake = in_valid && ref_in_ready;
            @(negedge clk_core);
            if (handshake)
                accepted = accepted + 1;
        end
        in_valid = 1'b0;
        while (!ref_plane_idle || !bank_plane_idle || !compiled_plane_idle)
            @(negedge clk_core);
    endtask

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            retire_ready <= 1'b0;
            long_stall_remaining <= 0;
            long_stall_started <= 1'b0;
        end else begin
            if (ref_retire_valid && !long_stall_started) begin
                long_stall_started <= 1'b1;
                long_stall_remaining <= 25;
                retire_ready <= 1'b0;
            end else if (long_stall_remaining > 0) begin
                long_stall_remaining <= long_stall_remaining - 1;
                retire_ready <= 1'b0;
            end else begin
                retire_ready <= ($urandom_range(0, 5) != 0);
            end
        end
    end

    always_ff @(posedge clk_core) begin
        if (!rst_core) begin
            if (ref_in_ready !== bank_in_ready)
                $fatal(1, "in_ready mismatch ref=%b bank=%b", ref_in_ready, bank_in_ready);
            if (ref_in_ready !== compiled_in_ready)
                $fatal(1, "in_ready mismatch ref=%b compiled=%b", ref_in_ready, compiled_in_ready);
            if (ref_retire_valid !== bank_retire_valid)
                $fatal(1, "retire_valid mismatch ref=%b bank=%b", ref_retire_valid, bank_retire_valid);
            if (ref_retire_valid !== compiled_retire_valid)
                $fatal(1, "retire_valid mismatch ref=%b compiled=%b", ref_retire_valid, compiled_retire_valid);
            if (ref_retire_valid) begin
                if (
                    ref_retire_source_id !== bank_retire_source_id
                    || ref_retire_y !== bank_retire_y
                    || ref_retire_x !== bank_retire_x
                )
                    $fatal(
                        1,
                        "retire payload mismatch ref=%0d/%0d/%0d bank=%0d/%0d/%0d",
                        ref_retire_source_id,
                        ref_retire_y,
                        ref_retire_x,
                        bank_retire_source_id,
                        bank_retire_y,
                        bank_retire_x
                    );
                if (
                    ref_retire_source_id !== compiled_retire_source_id
                    || ref_retire_y !== compiled_retire_y
                    || ref_retire_x !== compiled_retire_x
                )
                    $fatal(
                        1,
                        "retire payload mismatch ref=%0d/%0d/%0d compiled=%0d/%0d/%0d",
                        ref_retire_source_id,
                        ref_retire_y,
                        ref_retire_x,
                        compiled_retire_source_id,
                        compiled_retire_y,
                        compiled_retire_x
                    );
            end
            if (ref_plane_idle !== bank_plane_idle)
                $fatal(1, "plane_idle mismatch ref=%b bank=%b", ref_plane_idle, bank_plane_idle);
            if (ref_plane_idle !== compiled_plane_idle)
                $fatal(1, "plane_idle mismatch ref=%b compiled=%b", ref_plane_idle, compiled_plane_idle);
            if (ref_retire_valid && retire_ready) begin
                int sid;
                sid = ref_retire_source_id;
                if (seen[sid])
                    $fatal(1, "duplicate retirement sid=%0d", sid);
                if (!source_active(sid / TOKENS, (sid % TOKENS) / WIDTH, sid % WIDTH))
                    $fatal(1, "inactive source retired sid=%0d", sid);
                seen[sid] <= 1'b1;
                retire_count <= retire_count + 1;
            end
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        plane_start = 1'b0;
        plane_id = 1'b0;
        in_valid = 1'b0;
        in_y = '0;
        in_x = '0;
        in_candidate_valid = '0;
        compiled_candidate_valid = '0;
        retire_ready = 1'b0;
        retire_count = 0;
        expected_retire_count = 0;
        long_stall_remaining = 0;
        long_stall_started = 1'b0;
        seed = 1;
        void'($value$plusargs("SEED=%d", seed));
        for (int sid = 0; sid < TOTAL; sid = sid + 1)
            seen[sid] = 1'b0;
        for (int p = 0; p < TIME_PLANES; p = p + 1)
            for (int y = 0; y < HEIGHT; y = y + 1)
                for (int x = 0; x < WIDTH; x = x + 1)
                    expected_retire_count += source_active(p, y, x);

        repeat (4) @(negedge clk_core);
        rst_core = 1'b0;
        drive_plane(0);
        drive_plane(1);
        repeat (4) @(negedge clk_core);
        if (retire_count != expected_retire_count)
            $fatal(
                1,
                "retirement population mismatch got=%0d expected=%0d",
                retire_count,
                expected_retire_count
            );
        $display(
            "PASS dilation_miter d=%0d seed=%0d retire=%0d stalls=%0d pending=%0d",
            DILATION,
            seed,
            retire_count,
            ref_perf_stalls,
            ref_perf_pending
        );
        $finish;
    end

    initial begin
        repeat (20000) @(posedge clk_core);
        $fatal(1, "timeout");
    end

endmodule

`default_nettype wire
