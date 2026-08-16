`timescale 1ns/1ps
`default_nettype none

module tb_qfit_dual_color_active_source_index;
    localparam int HEIGHT = 5;
    localparam int WIDTH = 5;
    localparam int TIME_PLANES = 1;
    localparam int SOURCE_ID_W = $clog2(HEIGHT * WIDTH * TIME_PLANES);

    logic clk_core;
    logic rst_core;
    logic build_start;
    logic build_seal;
    logic build_active;
    logic build_done;
    logic in_valid;
    logic in_ready;
    logic in_plane;
    logic [2:0] in_destination_y;
    logic [2:0] in_destination_x;
    logic [4:0] in_active_candidate_mask;
    logic out_valid;
    logic out_ready;
    logic [SOURCE_ID_W-1:0] out_source_id;
    logic out_source_plane;
    logic [2:0] out_source_y;
    logic [2:0] out_source_x;
    logic out_last;
    logic protocol_error;
    logic [31:0] perf_input_candidates;
    logic [31:0] perf_unique_sources;
    logic [31:0] perf_duplicate_sets;
    logic [31:0] perf_bank_conflicts;

    logic [HEIGHT*WIDTH-1:0] seen;
    integer output_count;
    integer cycle_count;

    qfit_dual_color_active_source_index #(
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    task automatic send_destination(input integer y, input integer x);
        logic [4:0] mask;
        begin
            mask = 5'b00001;
            if (y > 0)          mask[1] = 1'b1;
            if (y < HEIGHT - 1) mask[2] = 1'b1;
            if (x > 0)          mask[3] = 1'b1;
            if (x < WIDTH - 1)  mask[4] = 1'b1;
            @(negedge clk_core);
            in_destination_y = y[2:0];
            in_destination_x = x[2:0];
            in_active_candidate_mask = mask;
            in_valid = 1'b1;
            while (!in_ready) @(negedge clk_core);
            @(negedge clk_core);
            in_valid = 1'b0;
            in_active_candidate_mask = '0;
        end
    endtask

    always @(negedge clk_core) begin
        if (rst_core) begin
            out_ready <= 1'b0;
            cycle_count <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            out_ready <= (cycle_count % 4) != 1;
        end
    end

    // 仅在真实传输边沿采样，避免ready驱动与scoreboard之间的调度竞态。
    always @(posedge clk_core) begin
        if (!rst_core && out_valid && out_ready) begin
            if (out_source_plane != 0)
                $fatal(1, "plane mismatch");
            if (integer'(out_source_y) >= HEIGHT || integer'(out_source_x) >= WIDTH)
                $fatal(1, "coordinate out of range");
            if (
                integer'(out_source_id)
                != integer'(out_source_y) * WIDTH + integer'(out_source_x)
            )
                $fatal(1, "source id decode mismatch");
            if (seen[out_source_id])
                $fatal(1, "duplicate source emission id=%0d", out_source_id);
            seen[out_source_id] = 1'b1;
            output_count = output_count + 1;
            if (out_last != (output_count == HEIGHT * WIDTH))
                $fatal(1, "last mismatch count=%0d", output_count);
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        build_start = 1'b0;
        build_seal = 1'b0;
        in_valid = 1'b0;
        in_plane = 1'b0;
        in_destination_y = '0;
        in_destination_x = '0;
        in_active_candidate_mask = '0;
        out_ready = 1'b0;
        seen = '0;
        output_count = 0;
        cycle_count = 0;
        repeat (3) @(negedge clk_core);
        rst_core = 1'b0;

        @(negedge clk_core);
        build_start = 1'b1;
        @(negedge clk_core);
        build_start = 1'b0;
        for (int y = 0; y < HEIGHT; y = y + 1)
            for (int x = 0; x < WIDTH; x = x + 1)
                send_destination(y, x);

        @(negedge clk_core);
        build_seal = 1'b1;
        @(negedge clk_core);
        build_seal = 1'b0;
        wait (build_done);
        repeat (2) @(negedge clk_core);

        if (protocol_error || perf_bank_conflicts != 0)
            $fatal(1, "unexpected topology protocol error/conflict");
        if (output_count != HEIGHT * WIDTH || seen != {HEIGHT*WIDTH{1'b1}})
            $fatal(1, "active source set mismatch count=%0d", output_count);
        if (
            perf_input_candidates != 105
            || perf_unique_sources != 25
            || perf_duplicate_sets != 80
        )
            $fatal(
                1,
                "counter mismatch input=%0d unique=%0d duplicate=%0d",
                perf_input_candidates,
                perf_unique_sources,
                perf_duplicate_sets
            );
        $display(
            "PASS dual-color active source input=%0d unique=%0d duplicate=%0d",
            perf_input_candidates,
            perf_unique_sources,
            perf_duplicate_sets
        );
        $finish;
    end
endmodule

`default_nettype wire
