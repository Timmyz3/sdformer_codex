`timescale 1ns/1ps
`default_nettype none

module tb_qfit_relation_transpose_perf #(
    parameter int SCHED_MODE = 0,
    parameter int STRIPE_RING_ROWS = 4
);
    localparam int HEIGHT = 15;
    localparam int WIDTH = 15;
    localparam int TOKENS = HEIGHT * WIDTH;
    localparam int Y_W = $clog2(HEIGHT);
    localparam int X_W = $clog2(WIDTH);
    localparam int SOURCE_ID_W = $clog2(TOKENS);
    localparam int GATE_W = 9;

    logic clk_core;
    logic rst_core;
    logic plane_start;
    logic plane_id;
    logic in_valid;
    logic in_ready;
    logic [Y_W-1:0] in_y;
    logic [X_W-1:0] in_x;
    logic [4:0] in_candidate_valid;
    logic [31:0] in_k_self;
    logic [5*GATE_W-1:0] in_direction_gates;
    logic descriptor_valid;
    logic descriptor_ready;
    logic [SOURCE_ID_W-1:0] descriptor_source_id;
    logic [Y_W-1:0] descriptor_y;
    logic [X_W-1:0] descriptor_x;
    logic [31:0] descriptor_k;
    logic [5*GATE_W-1:0] descriptor_incoming_gates;
    logic [4:0] descriptor_valid_mask;
    logic plane_idle;
    logic [31:0] perf_producer_stalls;
    logic [2:0] perf_max_pending;
    int cycle_count;
    int start_cycle;
    int descriptor_count;

    qfit_relation_transpose_leaf #(
        .SCHED_MODE(SCHED_MODE),
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(1),
        .STRIPE_RING_ROWS(STRIPE_RING_ROWS)
    ) dut (
        .debug_read_pending(),
        .debug_k_read_data_valid(),
        .*
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

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
            descriptor_count <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (descriptor_valid && descriptor_ready) begin
                if (
                    descriptor_source_id
                    != descriptor_y * WIDTH + descriptor_x
                )
                    $fatal(
                        1,
                        "descriptor identity mismatch id=%0d y=%0d x=%0d",
                        descriptor_source_id,
                        descriptor_y,
                        descriptor_x
                    );
                descriptor_count <= descriptor_count + 1;
            end
        end
    end

    initial begin
        logic handshake;
        clk_core = 1'b0;
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
        cycle_count = 0;
        descriptor_count = 0;
        repeat (4) @(negedge clk_core);
        rst_core = 1'b0;
        @(negedge clk_core);
        if (!plane_idle)
            $fatal(1, "not idle before plane start");
        plane_start = 1'b1;
        @(negedge clk_core);
        plane_start = 1'b0;
        start_cycle = cycle_count;

        for (int index = 0; index < TOKENS; index = index + 1) begin
            in_y = Y_W'(index / WIDTH);
            in_x = X_W'(index % WIDTH);
            in_candidate_valid = candidate_mask(
                index / WIDTH,
                index % WIDTH
            );
            in_k_self = 32'(index + 1);
            for (int role = 0; role < 5; role = role + 1)
                in_direction_gates[role*GATE_W +: GATE_W] =
                    GATE_W'(role * 32 + index % 31 + 1);
            in_valid = 1'b1;
            do begin
                @(posedge clk_core);
                handshake = in_ready;
                @(negedge clk_core);
            end while (!handshake);
            in_valid = 1'b0;
        end

        while (descriptor_count < TOKENS)
            @(negedge clk_core);
        if (!plane_idle)
            @(negedge clk_core);
        $display(
            "PASS qfit_relation_perf mode=%0d cycles=%0d stalls=%0d",
            SCHED_MODE,
            cycle_count - start_cycle,
            perf_producer_stalls
        );
        $finish;
    end
endmodule

`default_nettype wire
