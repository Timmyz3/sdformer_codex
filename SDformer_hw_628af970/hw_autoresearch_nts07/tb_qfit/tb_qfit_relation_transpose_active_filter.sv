`timescale 1ns/1ps
`default_nettype none

module tb_qfit_relation_transpose_active_filter;
    localparam int HEIGHT = 3;
    localparam int WIDTH = 3;
    localparam int TIME_PLANES = 2;
    localparam int TOKENS = HEIGHT * WIDTH;
    localparam int K_W = 16;
    localparam int GATE_W = 9;
    localparam int Y_W = $clog2(HEIGHT);
    localparam int X_W = $clog2(WIDTH);
    localparam int SOURCE_ID_W = $clog2(TOKENS * TIME_PLANES);

    logic clk_core = 1'b0;
    logic rst_core;
    logic plane_start;
    logic plane_id;
    logic in_valid;
    logic in_ready;
    logic [Y_W-1:0] in_y;
    logic [X_W-1:0] in_x;
    logic [4:0] in_candidate_valid;
    logic [K_W-1:0] in_k_self;
    logic [5*GATE_W-1:0] in_direction_gates;
    logic descriptor_valid;
    logic descriptor_ready;
    logic [SOURCE_ID_W-1:0] descriptor_source_id;
    logic [Y_W-1:0] descriptor_y;
    logic [X_W-1:0] descriptor_x;
    logic [K_W-1:0] descriptor_k;
    logic [5*GATE_W-1:0] descriptor_incoming_gates;
    logic [4:0] descriptor_valid_mask;
    logic plane_idle;
    logic [31:0] perf_producer_stalls;
    logic [2:0] perf_max_pending;
    int descriptor_count;
    int expected_sid [0:2];

    qfit_relation_transpose_leaf #(
        .SCHED_MODE(0),
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .K_W(K_W),
        .GATE_W(GATE_W),
        .SKIP_ZERO_K(1'b1)
    ) dut (
        .debug_read_pending(),
        .debug_k_read_data_valid(),
        .*
    );

    always #5 clk_core = ~clk_core;

    function automatic logic [4:0] candidate_mask(input int y, input int x);
        logic [4:0] mask;
        mask = 5'b00001;
        if (y > 0) mask[1] = 1'b1;
        if (y < HEIGHT - 1) mask[2] = 1'b1;
        if (x > 0) mask[3] = 1'b1;
        if (x < WIDTH - 1) mask[4] = 1'b1;
        return mask;
    endfunction

    function automatic logic [K_W-1:0] k_value(
        input int plane, input int y, input int x
    );
        k_value = '0;
        if (plane == 0 && y == 1 && x == 0)
            k_value = 16'h0101;
        if (plane == 0 && y == 2 && x == 1)
            k_value = 16'h0201;
        if (plane == 0 && y == 2 && x == 2)
            k_value = 16'h0202;
    endfunction

    task automatic drive_plane(input int plane);
        int accepted;
        bit fire;
        wait (plane_idle);
        @(negedge clk_core);
        plane_id = plane[0];
        plane_start = 1'b1;
        @(negedge clk_core);
        plane_start = 1'b0;
        accepted = 0;
        in_valid = 1'b1;
        while (accepted < TOKENS) begin
            in_y = Y_W'(accepted / WIDTH);
            in_x = X_W'(accepted % WIDTH);
            in_candidate_valid = candidate_mask(accepted / WIDTH, accepted % WIDTH);
            in_k_self = k_value(plane, accepted / WIDTH, accepted % WIDTH);
            for (int role = 0; role < 5; role = role + 1)
                in_direction_gates[role*GATE_W +: GATE_W] =
                    GATE_W'(1 + plane * 64 + accepted * 5 + role);
            @(posedge clk_core);
            fire = in_ready;
            @(negedge clk_core);
            if (fire) accepted = accepted + 1;
        end
        in_valid = 1'b0;
        wait (plane_idle);
        repeat (2) @(negedge clk_core);
    endtask

    always_ff @(posedge clk_core) begin
        if (!rst_core && descriptor_valid && descriptor_ready) begin
            if (descriptor_count >= 3)
                $fatal(1, "unexpected descriptor sid=%0d", descriptor_source_id);
            if (descriptor_source_id != SOURCE_ID_W'(expected_sid[descriptor_count]))
                $fatal(1, "descriptor order mismatch index=%0d got=%0d expected=%0d",
                    descriptor_count, descriptor_source_id, expected_sid[descriptor_count]);
            if (descriptor_k == '0)
                $fatal(1, "zero-K descriptor escaped filter");
            descriptor_count <= descriptor_count + 1;
        end
    end

    initial begin
        rst_core = 1'b1;
        plane_start = 1'b0;
        plane_id = 1'b0;
        in_valid = 1'b0;
        in_y = '0;
        in_x = '0;
        in_candidate_valid = '0;
        in_k_self = '0;
        in_direction_gates = '0;
        descriptor_ready = 1'b1;
        descriptor_count = 0;
        expected_sid[0] = 3;
        expected_sid[1] = 7;
        expected_sid[2] = 8;
        repeat (4) @(negedge clk_core);
        rst_core = 1'b0;
        drive_plane(0);
        if (descriptor_count != 3)
            $fatal(1, "mixed plane descriptor count=%0d expected=3", descriptor_count);
        drive_plane(1);
        if (descriptor_count != 3)
            $fatal(1, "all-zero plane leaked descriptor count=%0d", descriptor_count);
        $display("PASS active-filter boundary descriptors=%0d last_self=1 cross_row_0_to_1=1 empty_plane=1",
            descriptor_count);
        $finish;
    end

    initial begin
        repeat (5000) @(posedge clk_core);
        $fatal(1, "active-filter boundary timeout");
    end
endmodule

`default_nettype wire
