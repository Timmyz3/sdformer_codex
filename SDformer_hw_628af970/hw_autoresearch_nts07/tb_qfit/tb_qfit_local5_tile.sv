`timescale 1ns/1ps
`default_nettype none

module tb_qfit_local5_tile;
    localparam int HEIGHT = 3;
    localparam int WIDTH = 4;
    localparam int TIME_PLANES = 2;
    localparam int TOKENS = HEIGHT * WIDTH;
    localparam int TOTAL = TOKENS * TIME_PLANES;
    localparam int Y_W = $clog2(HEIGHT);
    localparam int X_W = $clog2(WIDTH);
    localparam int SOURCE_ID_W = $clog2(TOTAL);
    localparam int GATE_W = 9;

    logic clk_core;
    logic rst_core;
    logic plane_start;
    logic plane_id;
    logic plane_start_ready;
    logic in_valid;
    logic in_ready;
    logic [Y_W-1:0] in_y;
    logic [X_W-1:0] in_x;
    logic [31:0] in_q;
    logic [159:0] in_k;
    logic [4:0] in_valid_mask;
    logic descriptor_valid;
    logic descriptor_ready;
    logic [SOURCE_ID_W-1:0] descriptor_source_id;
    logic [Y_W-1:0] descriptor_y;
    logic [X_W-1:0] descriptor_x;
    logic [31:0] descriptor_k;
    logic [5*GATE_W-1:0] descriptor_incoming_gates;
    logic [4:0] descriptor_valid_mask;
    logic [15:0] perf_score_service_cycles;
    logic [3:0] perf_score_direct_mask;
    logic [31:0] perf_relation_stalls;
    logic [2:0] perf_relation_max_pending;

    logic [31:0] ref_k [0:4];
    logic [79:0] ref_score;
    logic [5*GATE_W-1:0] ref_gate;
    logic [5*GATE_W-1:0] expected_gate [0:TOTAL-1];
    logic [4:0] expected_valid [0:TOTAL-1];
    bit seen [0:TOTAL-1];
    int descriptor_count;

    qfit_local5_tile #(
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES)
    ) dut (.*);

    local5_stencil_token u_reference (
        .q_bits(in_q),
        .k_bits(ref_k),
        .valid(in_valid_mask),
        .score_q7(ref_score),
        .gate_q17(ref_gate)
    );

    always #5 clk_core = ~clk_core;

    function automatic logic [31:0] source_k(
        input int p,
        input int y,
        input int x
    );
        logic [31:0] base;
        base = 32'h1357_9bdf ^ (32'(p) << 29);
        source_k = {base[30:0], base[31]}
                 ^ (32'h0001_0101 * (y * WIDTH + x + 1));
    endfunction

    function automatic logic [31:0] query_bits(
        input int p,
        input int y,
        input int x
    );
        query_bits = 32'ha5c3_5a3c
                   ^ (32'h0102_0408 * (p * TOKENS + y * WIDTH + x));
    endfunction

    task automatic prepare_destination(
        input int p,
        input int y,
        input int x
    );
        int sy;
        int sx;
        in_y = Y_W'(y);
        in_x = X_W'(x);
        in_q = query_bits(p, y, x);
        in_valid_mask = 5'b00001;
        for (int role = 0; role < 5; role = role + 1) begin
            sy = y;
            sx = x;
            case (role)
                1: sy = y - 1;
                2: sy = y + 1;
                3: sx = x - 1;
                4: sx = x + 1;
            endcase
            if (
                sy >= 0
                && sy < HEIGHT
                && sx >= 0
                && sx < WIDTH
            ) begin
                in_valid_mask[role] = 1'b1;
                in_k[role*32 +: 32] = source_k(p, sy, sx);
            end else begin
                in_k[role*32 +: 32] = '0;
            end
        end
        if (p == 1) begin
            if (y == 1 && x == 1)
                in_valid_mask[0] = 1'b0;
            if (y == 2 && x == 1)
                in_valid_mask[1] = 1'b0;
            if (y == 1 && x == 2)
                in_valid_mask[3] = 1'b0;
        end
    endtask

    task automatic drive_plane(input int p);
        logic handshake;
        @(negedge clk_core);
        if (!plane_start_ready)
            $fatal(1, "plane start requested while tile is not idle");
        plane_id = p[0];
        plane_start = 1'b1;
        in_valid = 1'b0;
        @(negedge clk_core);
        plane_start = 1'b0;

        for (int index = 0; index < TOKENS; index = index + 1) begin
            prepare_destination(p, index / WIDTH, index % WIDTH);
            in_valid = 1'b1;
            do begin
                @(posedge clk_core);
                handshake = in_ready;
                @(negedge clk_core);
            end while (!handshake);
            expected_gate[p*TOKENS + index] = ref_gate;
            expected_valid[p*TOKENS + index] = in_valid_mask;
            in_valid = 1'b0;
            if (p == 0 && index == 2) begin
                // A next-plane valid may arrive early and remain asserted.
                // It must not clear the active score/relation transaction.
                plane_id = 1'b1;
                plane_start = 1'b1;
                repeat (3) begin
                    @(negedge clk_core);
                    if (plane_start_ready)
                        $fatal(1, "early plane start became ready");
                end
                plane_start = 1'b0;
                plane_id = p[0];
            end
        end

        while (descriptor_count < (p + 1) * TOKENS)
            @(negedge clk_core);
        repeat (2) @(negedge clk_core);
    endtask

    always_comb begin
        for (int role = 0; role < 5; role = role + 1)
            ref_k[role] = in_k[role*32 +: 32];
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            descriptor_ready <= 1'b0;
        end else begin
            descriptor_ready <= ($urandom_range(0, 4) != 0);
            if (descriptor_valid && descriptor_ready) begin
                int sid;
                int p;
                int y;
                int x;
                int dest_id;
                logic [4:0] expected_mask;
                logic [GATE_W-1:0] expected_value;
                sid = descriptor_source_id;
                p = sid / TOKENS;
                y = descriptor_y;
                x = descriptor_x;
                if (sid != p * TOKENS + y * WIDTH + x)
                    $fatal(1, "descriptor coordinate mismatch sid=%0d", sid);
                if (seen[sid])
                    $fatal(1, "duplicate descriptor sid=%0d", sid);
                if (descriptor_k != source_k(p, y, x))
                    $fatal(1, "descriptor K mismatch sid=%0d", sid);

                expected_mask = '0;
                expected_mask[0] = expected_valid[
                    p*TOKENS + y*WIDTH + x
                ][0];
                if (y < HEIGHT - 1)
                    expected_mask[1] = expected_valid[
                        p*TOKENS + (y+1)*WIDTH + x
                    ][1];
                if (y > 0)
                    expected_mask[2] = expected_valid[
                        p*TOKENS + (y-1)*WIDTH + x
                    ][2];
                if (x < WIDTH - 1)
                    expected_mask[3] = expected_valid[
                        p*TOKENS + y*WIDTH + x+1
                    ][3];
                if (x > 0)
                    expected_mask[4] = expected_valid[
                        p*TOKENS + y*WIDTH + x-1
                    ][4];
                if (descriptor_valid_mask != expected_mask)
                    $fatal(1, "descriptor mask mismatch sid=%0d", sid);

                for (int role = 0; role < 5; role = role + 1) begin
                    dest_id = y * WIDTH + x;
                    case (role)
                        1: dest_id = (y + 1) * WIDTH + x;
                        2: dest_id = (y - 1) * WIDTH + x;
                        3: dest_id = y * WIDTH + x + 1;
                        4: dest_id = y * WIDTH + x - 1;
                    endcase
                    expected_value = '0;
                    if (expected_mask[role])
                        expected_value = expected_gate[
                            p*TOKENS + dest_id
                        ][role*GATE_W +: GATE_W];
                    if (
                        descriptor_valid_mask[role]
                        && descriptor_incoming_gates[
                            role*GATE_W +: GATE_W
                        ] != expected_value
                    )
                        $fatal(
                            1,
                            "integrated gate mismatch sid=%0d role=%0d got=%0d exp=%0d",
                            sid,
                            role,
                            descriptor_incoming_gates[
                                role*GATE_W +: GATE_W
                            ],
                            expected_value
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
        in_q = '0;
        in_k = '0;
        in_valid_mask = '0;
        descriptor_ready = 1'b0;
        descriptor_count = 0;
        for (int sid = 0; sid < TOTAL; sid = sid + 1)
            seen[sid] = 1'b0;
        for (int sid = 0; sid < TOTAL; sid = sid + 1)
            expected_valid[sid] = '0;
        repeat (4) @(negedge clk_core);
        rst_core = 1'b0;
        for (int p = 0; p < TIME_PLANES; p = p + 1)
            drive_plane(p);
        for (int sid = 0; sid < TOTAL; sid = sid + 1)
            if (!seen[sid])
                $fatal(1, "missing descriptor sid=%0d", sid);
        $display(
            "PASS qfit_local5_tile descriptors=%0d relation_stalls=%0d max_pending=%0d",
            descriptor_count,
            perf_relation_stalls,
            perf_relation_max_pending
        );
        $finish;
    end
endmodule

`default_nettype wire
