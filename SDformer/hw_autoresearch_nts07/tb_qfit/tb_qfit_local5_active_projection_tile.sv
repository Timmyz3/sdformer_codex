`timescale 1ns/1ps
`default_nettype none

module tb_qfit_local5_active_projection_tile #(
    parameter int BACKEND_KIND = 0,
    parameter int RELATION_READ_LATENCY = 1
);
    localparam int HEIGHT = 3;
    localparam int WIDTH = 6;
    localparam int TIME_PLANES = 2;
    localparam int HEAD_DIM = 32;
    localparam int OUT_DIM = 2;
    localparam int GATE_W = 9;
    localparam int W_W = 8;
    localparam int ACC_W = 32;
    localparam int Y_W = $clog2(HEIGHT);
    localparam int X_W = $clog2(WIDTH);
    localparam int PLANE_W = 1;
    localparam int LANE_W = $clog2(HEAD_DIM);
    localparam int OUT_W = $clog2(OUT_DIM);
    localparam int TOTAL = HEIGHT * WIDTH * TIME_PLANES;

    logic clk_core;
    logic rst_core;
    logic weight_valid;
    logic weight_ready;
    logic [LANE_W-1:0] weight_lane;
    logic [OUT_W-1:0] weight_out;
    logic signed [W_W-1:0] weight_data;
    logic weight_last;
    logic weight_context_release;
    logic weight_context_release_ready;
    logic projection_start;
    logic projection_accumulate;
    logic projection_close;
    logic projection_close_ready;
    logic projection_busy;
    logic projection_done;
    logic relation_start;
    logic relation_seal;
    logic relation_active;
    logic relation_done;
    logic relation_valid;
    logic relation_ready;
    logic [PLANE_W-1:0] relation_plane;
    logic [Y_W-1:0] relation_destination_y;
    logic [X_W-1:0] relation_destination_x;
    logic [4:0] relation_candidate_valid;
    logic [4:0] relation_active_candidate_mask;
    logic [HEAD_DIM-1:0] relation_k_self;
    logic [5*GATE_W-1:0] relation_direction_gates;
    logic read_valid;
    logic read_ready;
    logic [PLANE_W-1:0] read_plane;
    logic [Y_W-1:0] read_y;
    logic [X_W-1:0] read_x;
    logic [OUT_W-1:0] read_out;
    logic read_data_valid;
    logic signed [ACC_W-1:0] read_data;
    logic protocol_error;
    logic [31:0] perf_relation_writes;
    logic [31:0] perf_active_source_reads;
    logic [31:0] perf_dense_reads_avoided;
    logic [31:0] perf_memory_wait_cycles;
    logic [31:0] perf_descriptors;
    logic [31:0] perf_product_terms;
    logic [31:0] perf_destination_updates;

    integer expected [0:TIME_PLANES-1][0:HEIGHT-1][0:WIDTH-1][0:OUT_DIM-1];
    integer run_cycles;
    integer expected_active_sources;
    integer expected_updates;

    qfit_local5_active_projection_tile #(
        .HEIGHT(HEIGHT), .WIDTH(WIDTH), .TIME_PLANES(TIME_PLANES),
        .HEAD_DIM(HEAD_DIM), .OUT_DIM(OUT_DIM), .GATE_W(GATE_W),
        .W_W(W_W), .ACC_W(ACC_W), .BACKEND_KIND(BACKEND_KIND),
        .RELATION_READ_LATENCY(RELATION_READ_LATENCY)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    function automatic integer source_id(
        input integer p, input integer y, input integer x
    );
        source_id = p * HEIGHT * WIDTH + y * WIDTH + x;
    endfunction

    function automatic logic [HEAD_DIM-1:0] source_k(
        input integer p, input integer y, input integer x
    );
        integer sid;
        logic [HEAD_DIM-1:0] value;
        begin
            sid = source_id(p, y, x);
            value = '0;
            if ((sid % 4) != 0) begin
                value[(sid * 3 + 1) % HEAD_DIM] = 1'b1;
                value[(sid * 7 + 5) % HEAD_DIM] = 1'b1;
            end
            source_k = value;
        end
    endfunction

    function automatic integer source_gate(
        input integer p, input integer y, input integer x
    );
        source_gate = 64 + 32 * (source_id(p, y, x) % 3);
    endfunction

    function automatic integer weight_value(
        input integer lane, input integer out
    );
        weight_value = (lane % 5 + 1) * (out == 0 ? 1 : -2);
    endfunction

    task automatic role_source(
        input integer dy,
        input integer dx,
        input integer role,
        output integer sy,
        output integer sx,
        output bit valid
    );
        begin
            sy = dy;
            sx = dx;
            case (role)
                1: sy = dy - 1;
                2: sy = dy + 1;
                3: sx = dx - 1;
                4: sx = dx + 1;
                default: begin end
            endcase
            valid = sy >= 0 && sy < HEIGHT && sx >= 0 && sx < WIDTH;
        end
    endtask

    task automatic load_weight(
        input integer lane, input integer out, input bit last
    );
        begin
            @(negedge clk_core);
            weight_lane = LANE_W'(lane);
            weight_out = OUT_W'(out);
            weight_data = W_W'(weight_value(lane, out));
            weight_last = last;
            weight_valid = 1'b1;
            do @(posedge clk_core); while (!weight_ready);
            @(negedge clk_core);
            weight_valid = 1'b0;
            weight_last = 1'b0;
        end
    endtask

    task automatic prepare_relation(
        input integer p, input integer y, input integer x
    );
        integer sy;
        integer sx;
        integer gate;
        bit valid;
        logic [HEAD_DIM-1:0] candidate_k;
        begin
            relation_plane = PLANE_W'(p);
            relation_destination_y = Y_W'(y);
            relation_destination_x = X_W'(x);
            relation_candidate_valid = '0;
            relation_active_candidate_mask = '0;
            relation_direction_gates = '0;
            relation_k_self = source_k(p, y, x);
            for (integer role = 0; role < 5; role = role + 1) begin
                role_source(y, x, role, sy, sx, valid);
                if (valid) begin
                    candidate_k = source_k(p, sy, sx);
                    gate = source_gate(p, sy, sx);
                    relation_candidate_valid[role] = 1'b1;
                    relation_direction_gates[role*GATE_W +: GATE_W]
                        = GATE_W'(gate);
                    relation_active_candidate_mask[role]
                        = candidate_k != '0 && gate != 0;
                end
            end
        end
    endtask

    task automatic accumulate_reference(
        input integer p, input integer y, input integer x
    );
        integer sy;
        integer sx;
        integer gate;
        bit valid;
        logic [HEAD_DIM-1:0] candidate_k;
        begin
            for (integer role = 0; role < 5; role = role + 1) begin
                role_source(y, x, role, sy, sx, valid);
                if (valid) begin
                    candidate_k = source_k(p, sy, sx);
                    gate = source_gate(p, sy, sx);
                    for (integer lane = 0; lane < HEAD_DIM; lane = lane + 1)
                        if (candidate_k[lane]) begin
                            for (integer out = 0; out < OUT_DIM; out = out + 1)
                                expected[p][y][x][out] =
                                    expected[p][y][x][out]
                                    + gate * weight_value(lane, out);
                            expected_updates = expected_updates + 1;
                        end
                end
            end
        end
    endtask

    task automatic send_relation(
        input integer p, input integer y, input integer x
    );
        bit accepted;
        begin
            prepare_relation(p, y, x);
            relation_valid = 1'b1;
            do begin
                @(posedge clk_core);
                accepted = relation_ready;
                @(negedge clk_core);
            end while (!accepted);
            accumulate_reference(p, y, x);
            relation_valid = 1'b0;
        end
    endtask

    task automatic check_acc(
        input integer p, input integer y, input integer x, input integer out
    );
        begin
            @(negedge clk_core);
            read_plane = PLANE_W'(p);
            read_y = Y_W'(y);
            read_x = X_W'(x);
            read_out = OUT_W'(out);
            read_valid = 1'b1;
            do @(posedge clk_core); while (!read_ready);
            @(negedge clk_core);
            read_valid = 1'b0;
            wait (read_data_valid);
            #1;
            if ($signed(read_data) !== expected[p][y][x][out])
                $fatal(1,
                    "acc mismatch backend=%0d p=%0d y=%0d x=%0d out=%0d got=%0d expected=%0d",
                    BACKEND_KIND, p, y, x, out, $signed(read_data),
                    expected[p][y][x][out]);
        end
    endtask

    always_ff @(posedge clk_core) begin
        if (rst_core)
            run_cycles <= 0;
        else if (projection_busy)
            run_cycles <= run_cycles + 1;
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        weight_valid = 1'b0;
        weight_lane = '0;
        weight_out = '0;
        weight_data = '0;
        weight_last = 1'b0;
        weight_context_release = 1'b0;
        projection_start = 1'b0;
        projection_accumulate = 1'b0;
        projection_close = 1'b0;
        relation_start = 1'b0;
        relation_seal = 1'b0;
        relation_valid = 1'b0;
        relation_plane = '0;
        relation_destination_y = '0;
        relation_destination_x = '0;
        relation_candidate_valid = '0;
        relation_active_candidate_mask = '0;
        relation_k_self = '0;
        relation_direction_gates = '0;
        read_valid = 1'b0;
        read_plane = '0;
        read_y = '0;
        read_x = '0;
        read_out = '0;
        run_cycles = 0;
        expected_active_sources = 0;
        expected_updates = 0;
        for (integer p = 0; p < TIME_PLANES; p = p + 1)
            for (integer y = 0; y < HEIGHT; y = y + 1)
                for (integer x = 0; x < WIDTH; x = x + 1) begin
                    if (source_k(p, y, x) != '0)
                        expected_active_sources = expected_active_sources + 1;
                    for (integer out = 0; out < OUT_DIM; out = out + 1)
                        expected[p][y][x][out] = 0;
                end

        repeat (4) @(negedge clk_core);
        rst_core = 1'b0;
        for (integer lane = 0; lane < HEAD_DIM; lane = lane + 1)
            for (integer out = 0; out < OUT_DIM; out = out + 1)
                load_weight(lane, out,
                    lane == HEAD_DIM - 1 && out == OUT_DIM - 1);

        @(negedge clk_core);
        projection_start = 1'b1;
        relation_start = 1'b1;
        @(negedge clk_core);
        projection_start = 1'b0;
        relation_start = 1'b0;
        for (integer p = 0; p < TIME_PLANES; p = p + 1)
            for (integer y = 0; y < HEIGHT; y = y + 1)
                for (integer x = 0; x < WIDTH; x = x + 1)
                    send_relation(p, y, x);

        @(negedge clk_core);
        relation_seal = 1'b1;
        @(negedge clk_core);
        relation_seal = 1'b0;
        wait (projection_close_ready);
        @(negedge clk_core);
        projection_close = 1'b1;
        @(negedge clk_core);
        projection_close = 1'b0;
        wait (projection_done);

        for (integer p = 0; p < TIME_PLANES; p = p + 1)
            for (integer y = 0; y < HEIGHT; y = y + 1)
                for (integer x = 0; x < WIDTH; x = x + 1)
                    for (integer out = 0; out < OUT_DIM; out = out + 1)
                        check_acc(p, y, x, out);

        if (protocol_error)
            $fatal(1, "unexpected protocol error");
        if (perf_relation_writes != TOTAL)
            $fatal(1, "relation writes got=%0d expected=%0d",
                perf_relation_writes, TOTAL);
        if (perf_active_source_reads != expected_active_sources)
            $fatal(1, "active reads got=%0d expected=%0d",
                perf_active_source_reads, expected_active_sources);
        if (perf_dense_reads_avoided != TOTAL - expected_active_sources)
            $fatal(1, "dense reads avoided mismatch");
        if (perf_descriptors != expected_active_sources)
            $fatal(1, "descriptor count mismatch");
        if (perf_destination_updates != expected_updates)
            $fatal(1, "update count got=%0d expected=%0d",
                perf_destination_updates, expected_updates);
        $display(
            "PASS active_projection backend=%0d latency=%0d cycles=%0d writes=%0d active_sources=%0d avoided=%0d memory_wait=%0d terms=%0d updates=%0d",
            BACKEND_KIND, RELATION_READ_LATENCY, run_cycles,
            perf_relation_writes, perf_active_source_reads,
            perf_dense_reads_avoided,
            perf_memory_wait_cycles,
            perf_product_terms, perf_destination_updates);
        $finish;
    end
endmodule

`default_nettype wire
