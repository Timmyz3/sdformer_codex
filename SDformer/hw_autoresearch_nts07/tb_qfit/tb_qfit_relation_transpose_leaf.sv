`timescale 1ns/1ps
`default_nettype none

module tb_qfit_relation_transpose_leaf #(
    parameter int SCHED_MODE = 0,
    parameter int STRIPE_RING_ROWS = 4
);
    localparam int HEIGHT = 5;
    localparam int WIDTH = 4;
    localparam int TIME_PLANES = 2;
    localparam int TOKENS = HEIGHT * WIDTH;
    localparam int TOTAL = TOKENS * TIME_PLANES;
    localparam int Y_W = $clog2(HEIGHT);
    localparam int X_W = $clog2(WIDTH);
    localparam int SOURCE_ID_W = $clog2(TOTAL);
    localparam int K_W = 16;
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
    bit seen [0:TOTAL-1];
    int descriptor_count;
    int stall_remaining;
    bit long_stall_started;

    qfit_relation_transpose_leaf #(
        .SCHED_MODE(SCHED_MODE),
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .K_W(K_W),
        .GATE_W(GATE_W),
        .STRIPE_RING_ROWS(STRIPE_RING_ROWS)
    ) dut (
        .debug_read_pending(),
        .debug_k_read_data_valid(),
        .*
    );

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
        // FCSR must preserve non-geometric runtime candidate invalidation.
        if (SCHED_MODE == 0 && p == 1) begin
            if (y == 1 && x == 1)
                mask[0] = 1'b0;
            if (y == 2 && x == 1)
                mask[1] = 1'b0;
            if (y == 1 && x == 2)
                mask[3] = 1'b0;
        end
        return mask;
    endfunction

    function automatic logic [GATE_W-1:0] gate_value(
        input int p,
        input int y,
        input int x,
        input int role
    );
        if (
            p == 1
            && y == HEIGHT - 1
            && x == WIDTH - 1
            && role == 0
        )
            gate_value = GATE_W'(256);
        else
            gate_value =
                GATE_W'(1 + p * 80 + y * 20 + x * 5 + role);
    endfunction

    function automatic logic [K_W-1:0] k_value(
        input int p,
        input int y,
        input int x
    );
        k_value = K_W'(16'h4000 + p * TOKENS + y * WIDTH + x);
    endfunction

    task automatic drive_plane(input int p);
        int accepted;
        logic handshake;
        @(negedge clk_core);
        plane_id = p[0];
        plane_start = 1'b1;
        in_valid = 1'b0;
        @(negedge clk_core);
        plane_start = 1'b0;
        accepted = 0;
        in_valid = 1'b1;
        while (accepted < TOKENS) begin
            in_y = Y_W'(accepted / WIDTH);
            in_x = X_W'(accepted % WIDTH);
            in_candidate_valid = candidate_mask(
                p,
                accepted / WIDTH,
                accepted % WIDTH
            );
            in_k_self = k_value(
                p,
                accepted / WIDTH,
                accepted % WIDTH
            );
            for (int role = 0; role < 5; role = role + 1)
                in_direction_gates[role*GATE_W +: GATE_W] =
                    gate_value(
                        p,
                        accepted / WIDTH,
                        accepted % WIDTH,
                        role
                    );
            @(posedge clk_core);
            handshake = in_ready;
            @(negedge clk_core);
            if (handshake)
                accepted = accepted + 1;
        end
        in_valid = 1'b0;
        while (descriptor_count < (p + 1) * TOKENS)
            @(negedge clk_core);
        repeat (2) @(negedge clk_core);
    endtask

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            descriptor_ready <= 1'b0;
            stall_remaining <= 0;
            long_stall_started <= 1'b0;
        end else begin
            if (
                descriptor_valid
                && !long_stall_started
            ) begin
                long_stall_started <= 1'b1;
                stall_remaining <= 20;
                descriptor_ready <= 1'b0;
            end else if (stall_remaining > 0) begin
                stall_remaining <= stall_remaining - 1;
                descriptor_ready <= 1'b0;
            end else begin
                descriptor_ready <= ($urandom_range(0, 4) != 0);
            end
            if (descriptor_valid && descriptor_ready) begin
                int sid;
                int p;
                int y;
                int x;
                logic [4:0] expected_mask;
                logic [4:0] neighbor_mask;
                logic [GATE_W-1:0] expected_gate;
                sid = descriptor_source_id;
                p = sid / TOKENS;
                y = descriptor_y;
                x = descriptor_x;
                if (sid != p * TOKENS + y * WIDTH + x)
                    $fatal(1, "source coordinate mismatch sid=%0d", sid);
                if (seen[sid])
                    $fatal(1, "duplicate sid=%0d", sid);
                if (descriptor_k != k_value(p, y, x))
                    $fatal(1, "K mismatch sid=%0d", sid);
                expected_mask = '0;
                neighbor_mask = candidate_mask(p, y, x);
                expected_mask[0] = neighbor_mask[0];
                if (y < HEIGHT - 1) begin
                    neighbor_mask = candidate_mask(p, y + 1, x);
                    expected_mask[1] = neighbor_mask[1];
                end
                if (y > 0) begin
                    neighbor_mask = candidate_mask(p, y - 1, x);
                    expected_mask[2] = neighbor_mask[2];
                end
                if (x < WIDTH - 1) begin
                    neighbor_mask = candidate_mask(p, y, x + 1);
                    expected_mask[3] = neighbor_mask[3];
                end
                if (x > 0) begin
                    neighbor_mask = candidate_mask(p, y, x - 1);
                    expected_mask[4] = neighbor_mask[4];
                end
                if (descriptor_valid_mask != expected_mask)
                    $fatal(
                        1,
                        "mask mismatch sid=%0d got=%b exp=%b",
                        sid,
                        descriptor_valid_mask,
                        expected_mask
                    );
                for (int role = 0; role < 5; role = role + 1) begin
                    expected_gate = '0;
                    case (role)
                        0: expected_gate = gate_value(p, y, x, 0);
                        1: if (y < HEIGHT - 1)
                            expected_gate = gate_value(p, y + 1, x, 1);
                        2: if (y > 0)
                            expected_gate = gate_value(p, y - 1, x, 2);
                        3: if (x < WIDTH - 1)
                            expected_gate = gate_value(p, y, x + 1, 3);
                        4: if (x > 0)
                            expected_gate = gate_value(p, y, x - 1, 4);
                    endcase
                    if (
                        descriptor_valid_mask[role]
                        && descriptor_incoming_gates[
                            role*GATE_W +: GATE_W
                        ] != expected_gate
                    )
                        $fatal(
                            1,
                            "gate mismatch sid=%0d role=%0d got=%0d exp=%0d",
                            sid,
                            role,
                            descriptor_incoming_gates[
                                role*GATE_W +: GATE_W
                            ],
                            expected_gate
                        );
                end
                seen[sid] <= 1'b1;
                descriptor_count <= descriptor_count + 1;
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
        in_k_self = '0;
        in_direction_gates = '0;
        descriptor_ready = 1'b0;
        descriptor_count = 0;
        stall_remaining = 0;
        long_stall_started = 1'b0;
        for (int sid = 0; sid < TOTAL; sid = sid + 1)
            seen[sid] = 1'b0;
        repeat (4) @(negedge clk_core);
        rst_core = 1'b0;
        for (int p = 0; p < TIME_PLANES; p = p + 1)
            drive_plane(p);
        for (int sid = 0; sid < TOTAL; sid = sid + 1)
            if (!seen[sid])
                $fatal(1, "missing sid=%0d", sid);
        $display(
            "PASS qfit_relation_transpose mode=%0d descriptors=%0d stalls=%0d max_pending=%0d",
            SCHED_MODE,
            descriptor_count,
            perf_producer_stalls,
            perf_max_pending
        );
        $finish;
    end
endmodule

`default_nettype wire
