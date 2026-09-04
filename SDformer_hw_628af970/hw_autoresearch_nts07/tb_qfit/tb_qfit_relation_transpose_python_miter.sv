`timescale 1ns/1ps
`default_nettype none

module tb_qfit_relation_transpose_python_miter;
    localparam int HEIGHT = 15;
    localparam int WIDTH = 15;
    localparam int TIME_PLANES = 2;
    localparam int TOKENS = HEIGHT * WIDTH;
    localparam int TOTAL = TOKENS * TIME_PLANES;
    localparam int K_W = 32;
    localparam int GATE_W = 9;
    localparam int Y_W = $clog2(HEIGHT);
    localparam int X_W = $clog2(WIDTH);
    localparam int SOURCE_ID_W = $clog2(TOTAL);

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

    logic [4:0] input_valid_mem [0:TOTAL-1];
    logic [K_W-1:0] input_k_mem [0:TOTAL-1];
    logic [5*GATE_W-1:0] input_gates_mem [0:TOTAL-1];
    logic [K_W-1:0] expected_k_mem [0:TOTAL-1];
    logic [5*GATE_W-1:0] expected_gates_mem [0:TOTAL-1];
    logic [4:0] expected_mask_mem [0:TOTAL-1];
    bit seen [0:TOTAL-1];
    integer observed;
    integer output_stalls;
    string vector_dir;

    always #1 clk_core = ~clk_core;

    qfit_relation_transpose_leaf #(
        .SCHED_MODE(0),
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .K_W(K_W),
        .GATE_W(GATE_W)
    ) dut (
        .debug_read_pending(),
        .debug_k_read_data_valid(),
        .*
    );

    task automatic drive_plane(input int plane);
        int accepted;
        int global_index;
        logic accepted_fire;
        begin
            while (!plane_idle)
                @(negedge clk_core);
            plane_id = plane[0];
            plane_start = 1'b1;
            @(negedge clk_core);
            plane_start = 1'b0;
            accepted = 0;
            in_valid = 1'b1;
            while (accepted < TOKENS) begin
                global_index = plane * TOKENS + accepted;
                in_y = Y_W'(accepted / WIDTH);
                in_x = X_W'(accepted % WIDTH);
                in_candidate_valid = input_valid_mem[global_index];
                in_k_self = input_k_mem[global_index];
                in_direction_gates = input_gates_mem[global_index];
                @(posedge clk_core);
                accepted_fire = in_ready;
                @(negedge clk_core);
                if (accepted_fire)
                    accepted = accepted + 1;
            end
            in_valid = 1'b0;
            while (observed < (plane + 1) * TOKENS)
                @(negedge clk_core);
        end
    endtask

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            descriptor_ready <= 1'b0;
            observed <= 0;
            output_stalls <= 0;
        end else begin
            descriptor_ready <= ($urandom_range(0, 4) != 0);
            if (descriptor_valid && !descriptor_ready)
                output_stalls <= output_stalls + 1;
            if (descriptor_valid && descriptor_ready) begin
                int sid;
                sid = descriptor_source_id;
                if (sid < 0 || sid >= TOTAL)
                    $fatal(1, "source id越界: %0d", sid);
                if (seen[sid])
                    $fatal(1, "source id重复: %0d", sid);
                if (descriptor_y != Y_W'((sid % TOKENS) / WIDTH))
                    $fatal(1, "source y不一致: %0d", sid);
                if (descriptor_x != X_W'((sid % TOKENS) % WIDTH))
                    $fatal(1, "source x不一致: %0d", sid);
                if (descriptor_k !== expected_k_mem[sid])
                    $fatal(1, "K mismatch sid=%0d", sid);
                if (
                    descriptor_incoming_gates
                    !== expected_gates_mem[sid]
                )
                    $fatal(1, "gate mismatch sid=%0d", sid);
                if (descriptor_valid_mask !== expected_mask_mem[sid])
                    $fatal(1, "mask mismatch sid=%0d", sid);
                seen[sid] <= 1'b1;
                observed <= observed + 1;
            end
        end
    end

    initial begin
        if (!$value$plusargs("VECTOR_DIR=%s", vector_dir))
            vector_dir = "tb_qfit/vectors/local5_relation_t450";
        $readmemh({vector_dir, "/input_valid.memh"}, input_valid_mem);
        $readmemh({vector_dir, "/input_k.memh"}, input_k_mem);
        $readmemh({vector_dir, "/input_gates.memh"}, input_gates_mem);
        $readmemh({vector_dir, "/expected_k.memh"}, expected_k_mem);
        $readmemh(
            {vector_dir, "/expected_gates.memh"},
            expected_gates_mem
        );
        $readmemh(
            {vector_dir, "/expected_mask.memh"},
            expected_mask_mem
        );
        rst_core = 1'b1;
        plane_start = 1'b0;
        plane_id = 1'b0;
        in_valid = 1'b0;
        in_y = '0;
        in_x = '0;
        in_candidate_valid = '0;
        in_k_self = '0;
        in_direction_gates = '0;
        descriptor_ready = 1'b0;
        observed = 0;
        output_stalls = 0;
        for (int sid = 0; sid < TOTAL; sid = sid + 1)
            seen[sid] = 1'b0;
        repeat (5) @(negedge clk_core);
        rst_core = 1'b0;
        for (int plane = 0; plane < TIME_PLANES; plane = plane + 1)
            drive_plane(plane);
        for (int sid = 0; sid < TOTAL; sid = sid + 1)
            if (!seen[sid])
                $fatal(1, "source id缺失: %0d", sid);
        if (output_stalls == 0)
            $fatal(1, "未覆盖descriptor反压");
        $display(
            "PASS Python/RTL relation miter descriptors=%0d stalls=%0d",
            observed,
            output_stalls
        );
        $finish;
    end

    initial begin
        repeat (200000) @(posedge clk_core);
        $fatal(1, "Python/RTL relation miter超时");
    end
endmodule

`default_nettype wire
