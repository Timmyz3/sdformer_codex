`timescale 1ns/1ps
`default_nettype none

module tb_qfit_dual_color_relation_frontier;
    localparam int HEIGHT = 5;
    localparam int WIDTH = 5;
    localparam int TIME_PLANES = 1;
    localparam int K_W = 32;
    localparam int GATE_W = 9;
    localparam int SOURCE_ID_W = $clog2(HEIGHT * WIDTH);

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
    logic [4:0] in_candidate_valid;
    logic [4:0] in_active_candidate_mask;
    logic [K_W-1:0] in_k_self;
    logic [5*GATE_W-1:0] in_direction_gates;
    logic descriptor_valid;
    logic descriptor_ready;
    logic [SOURCE_ID_W-1:0] descriptor_source_id;
    logic descriptor_plane;
    logic [2:0] descriptor_y;
    logic [2:0] descriptor_x;
    logic [K_W-1:0] descriptor_k;
    logic [5*GATE_W-1:0] descriptor_incoming_gates;
    logic [4:0] descriptor_valid_mask;
    logic descriptor_last;
    logic protocol_error;
    logic [31:0] perf_relation_writes;
    logic [31:0] perf_source_reads;
    logic [31:0] perf_dense_reads_avoided;
    logic [HEIGHT*WIDTH-1:0] seen;
    integer output_count;
    integer cycle_count;

    qfit_dual_color_relation_frontier #(
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .K_W(K_W),
        .GATE_W(GATE_W)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    function automatic logic [GATE_W-1:0] gate_value(
        input integer destination_id,
        input integer role
    );
        gate_value = GATE_W'(destination_id * 7 + role + 1);
    endfunction

    task automatic send_destination(input integer y, input integer x);
        integer destination_id;
        integer source_x [0:4];
        integer source_y [0:4];
        integer source_id;
        logic [4:0] valid_mask;
        logic [4:0] active_mask;
        logic [5*GATE_W-1:0] gates;
        begin
            destination_id = y * WIDTH + x;
            source_x[0] = x;
            source_x[1] = x;
            source_x[2] = x;
            source_x[3] = x;
            source_x[4] = x;
            source_y[0] = y;
            source_y[1] = y;
            source_y[2] = y;
            source_y[3] = y;
            source_y[4] = y;
            source_x[3] = x - 1;
            source_x[4] = x + 1;
            source_y[1] = y - 1;
            source_y[2] = y + 1;
            valid_mask = '0;
            active_mask = '0;
            gates = '0;
            for (integer role = 0; role < 5; role = role + 1) begin
                if (
                    source_x[role] >= 0 && source_x[role] < WIDTH
                    && source_y[role] >= 0 && source_y[role] < HEIGHT
                ) begin
                    valid_mask[role] = 1'b1;
                    source_id = source_y[role] * WIDTH + source_x[role];
                    active_mask[role] = (source_id % 3) == 0;
                    gates[role*GATE_W +: GATE_W]
                        = gate_value(destination_id, role);
                end
            end
            @(negedge clk_core);
            while (!in_ready) @(negedge clk_core);
            in_destination_y = y[2:0];
            in_destination_x = x[2:0];
            in_candidate_valid = valid_mask;
            in_active_candidate_mask = active_mask;
            in_k_self = ((destination_id % 3) == 0)
                ? (32'b1 << (destination_id % 32)) : '0;
            in_direction_gates = gates;
            in_valid = 1'b1;
            @(negedge clk_core);
            in_valid = 1'b0;
        end
    endtask

    always @(negedge clk_core) begin
        if (rst_core) begin
            descriptor_ready <= 1'b0;
            cycle_count <= 0;
        end else begin
            descriptor_ready <= (cycle_count % 5) != 2;
            cycle_count <= cycle_count + 1;
        end
    end

    always @(posedge clk_core) begin : scoreboard
        integer source_id;
        integer y;
        integer x;
        integer destination_id;
        logic [4:0] expected_mask;
        logic [5*GATE_W-1:0] expected_gates;
        if (!rst_core && descriptor_valid && descriptor_ready) begin
            source_id = descriptor_source_id;
            y = descriptor_y;
            x = descriptor_x;
            if (descriptor_plane != 0 || source_id != y * WIDTH + x)
                $fatal(1, "source coordinate mismatch");
            if ((source_id % 3) != 0)
                $fatal(1, "inactive source emitted id=%0d", source_id);
            if (seen[source_id])
                $fatal(1, "duplicate descriptor id=%0d", source_id);
            if (descriptor_k != (32'b1 << (source_id % 32)))
                $fatal(1, "K mismatch id=%0d", source_id);

            expected_mask = '0;
            expected_gates = '0;
            destination_id = y * WIDTH + x;
            expected_mask[0] = 1'b1;
            expected_gates[0*GATE_W +: GATE_W]
                = gate_value(destination_id, 0);
            if (y < HEIGHT - 1) begin
                destination_id = (y + 1) * WIDTH + x;
                expected_mask[1] = 1'b1;
                expected_gates[1*GATE_W +: GATE_W]
                    = gate_value(destination_id, 1);
            end
            if (y > 0) begin
                destination_id = (y - 1) * WIDTH + x;
                expected_mask[2] = 1'b1;
                expected_gates[2*GATE_W +: GATE_W]
                    = gate_value(destination_id, 2);
            end
            if (x < WIDTH - 1) begin
                destination_id = y * WIDTH + x + 1;
                expected_mask[3] = 1'b1;
                expected_gates[3*GATE_W +: GATE_W]
                    = gate_value(destination_id, 3);
            end
            if (x > 0) begin
                destination_id = y * WIDTH + x - 1;
                expected_mask[4] = 1'b1;
                expected_gates[4*GATE_W +: GATE_W]
                    = gate_value(destination_id, 4);
            end
            if (
                descriptor_valid_mask != expected_mask
                || descriptor_incoming_gates != expected_gates
            )
                $fatal(1, "relation transpose mismatch id=%0d", source_id);
            seen[source_id] = 1'b1;
            output_count = output_count + 1;
            if (descriptor_last != (output_count == 9))
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
        in_candidate_valid = '0;
        in_active_candidate_mask = '0;
        in_k_self = '0;
        in_direction_gates = '0;
        descriptor_ready = 1'b0;
        seen = '0;
        output_count = 0;
        cycle_count = 0;
        repeat (3) @(negedge clk_core);
        rst_core = 1'b0;
        @(negedge clk_core);
        build_start = 1'b1;
        @(negedge clk_core);
        build_start = 1'b0;
        for (integer y = 0; y < HEIGHT; y = y + 1)
            for (integer x = 0; x < WIDTH; x = x + 1)
                send_destination(y, x);
        @(negedge clk_core);
        build_seal = 1'b1;
        @(negedge clk_core);
        build_seal = 1'b0;
        wait (build_done);
        repeat (2) @(negedge clk_core);
        if (protocol_error)
            $fatal(1, "unexpected protocol error");
        if (
            output_count != 9
            || perf_relation_writes != 25
            || perf_source_reads != 9
            || perf_dense_reads_avoided != 16
        )
            $fatal(
                1,
                "counter mismatch output=%0d writes=%0d reads=%0d avoided=%0d",
                output_count,
                perf_relation_writes,
                perf_source_reads,
                perf_dense_reads_avoided
            );
        $display(
            "PASS dual-color relation frontier writes=%0d reads=%0d avoided=%0d",
            perf_relation_writes,
            perf_source_reads,
            perf_dense_reads_avoided
        );
        $finish;
    end
endmodule

`default_nettype wire
