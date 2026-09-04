`timescale 1ns/1ps
`default_nettype none

module tb_qfit_local5_projection_tile #(
    parameter int BACKEND_KIND = 0,
    parameter int HEIGHT = 3,
    parameter int WIDTH = 6,
    parameter int TIME_PLANES = 2,
    parameter int HEAD_DIM = 32,
    parameter int OUT_DIM = 2
);
    localparam int TOKENS = HEIGHT * WIDTH;
    localparam int TOTAL = TOKENS * TIME_PLANES;
    localparam int Y_W = $clog2(HEIGHT);
    localparam int X_W = $clog2(WIDTH);
    localparam int PLANE_W = 1;
    localparam int LANE_W = $clog2(HEAD_DIM);
    localparam int OUT_W = $clog2(OUT_DIM);
    localparam int GATE_W = 9;
    localparam int W_W = 8;
    localparam int ACC_W = 32;

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
    logic projection_start_ready;
    logic projection_close;
    logic term_issue_enable;
    logic projection_close_ready;
    logic projection_busy;
    logic projection_done;
    logic stream_idle;
    logic plane_start;
    logic [PLANE_W-1:0] plane_id;
    logic plane_start_ready;
    logic in_valid;
    logic in_ready;
    logic [Y_W-1:0] in_y;
    logic [X_W-1:0] in_x;
    logic [HEAD_DIM-1:0] in_q;
    logic [5*HEAD_DIM-1:0] in_k;
    logic [4:0] in_valid_mask;
    logic read_valid;
    logic read_ready;
    logic [PLANE_W-1:0] read_plane;
    logic [Y_W-1:0] read_y;
    logic [X_W-1:0] read_x;
    logic [OUT_W-1:0] read_out;
    logic read_data_valid;
    logic signed [ACC_W-1:0] read_data;
    logic protocol_error;
    logic [31:0] perf_descriptors;
    logic [31:0] perf_product_terms;
    logic [31:0] perf_destination_updates;
    logic [31:0] perf_relation_stalls;

    logic [31:0] ref_k [0:4];
    logic [79:0] ref_score;
    logic [5*GATE_W-1:0] ref_gate;
    integer signed expected
        [0:TIME_PLANES-1][0:HEIGHT-1][0:WIDTH-1][0:OUT_DIM-1];
    integer baseline_linear5_cycles;
    integer baseline_affine4_cycles;
    integer baseline_single_bank_cycles;
    integer tile_run_cycles;
    integer unique_lane_gate_products;
    logic product_key_seen
        [0:HEAD_DIM-1][0:(1 << GATE_W)-1];
    logic lane1_valid [0:HEAD_DIM-1];
    logic [GATE_W-1:0] lane1_gate [0:HEAD_DIM-1];
    integer lane1_hits;
    integer lane1_misses;
    logic lane2_valid [0:HEAD_DIM-1][0:1];
    logic [GATE_W-1:0] lane2_gate [0:HEAD_DIM-1][0:1];
    logic lane2_mru [0:HEAD_DIM-1];
    integer lane2_hits;
    integer lane2_misses;
    logic dm16_valid [0:15];
    integer dm16_tag [0:15];
    integer dm16_hits;
    integer dm16_misses;
    integer term_trace_fd;
    integer term_trace_seq;
    string term_trace_path;
    integer term_stall_seed;
    integer term_stall_mode;
    integer term_issue_stall_cycles;
    integer term_boundary_block_hits;
    integer directed_hold_remaining;
    logic first_term_hold_started;
    logic final_term_hold_started;
    logic final_term_hold_done;
    integer busy_half_cycles;
    integer acc32_mismatch_count;
    integer python_acc32_mismatch_count;
    integer python_fullchain_enabled;
    string python_inputs_path;
    string python_expected_path;
    logic [31:0] python_q
        [0:TIME_PLANES-1][0:HEIGHT-1][0:WIDTH-1];
    logic [31:0] python_k
        [0:TIME_PLANES-1][0:HEIGHT-1][0:WIDTH-1][0:4];
    logic [4:0] python_valid_mask
        [0:TIME_PLANES-1][0:HEIGHT-1][0:WIDTH-1];
    integer signed python_expected
        [0:TIME_PLANES-1][0:HEIGHT-1][0:WIDTH-1][0:OUT_DIM-1];
    logic python_input_seen
        [0:TIME_PLANES-1][0:HEIGHT-1][0:WIDTH-1];
    logic python_expected_seen
        [0:TIME_PLANES-1][0:HEIGHT-1][0:WIDTH-1][0:OUT_DIM-1];
    logic [15:0] term_stall_lfsr;

    qfit_local5_projection_tile #(
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .HEAD_DIM(HEAD_DIM),
        .OUT_DIM(OUT_DIM),
        .BACKEND_KIND(BACKEND_KIND)
    ) dut (.*);

    local5_stencil_token u_reference (
        .q_bits(in_q),
        .k_bits(ref_k),
        .valid(in_valid_mask),
        .score_q7(ref_score),
        .gate_q17(ref_gate)
    );

    always #5 clk_core = ~clk_core;

    // Drive the test-only issue gate on the falling edge so a detected held
    // term is blocked before the DUT's next active edge.
    always @(negedge clk_core) begin
        if (rst_core) begin
            term_stall_lfsr <= term_stall_seed == 0
                ? 16'h1 : term_stall_seed[15:0];
            term_issue_enable <= 1'b1;
            term_issue_stall_cycles <= 0;
            term_boundary_block_hits <= 0;
            directed_hold_remaining <= 0;
            first_term_hold_started <= 1'b0;
            final_term_hold_started <= 1'b0;
            final_term_hold_done <= 1'b0;
            busy_half_cycles <= 0;
        end else begin
            if (projection_busy)
                busy_half_cycles <= busy_half_cycles + 1;

            case (term_stall_mode)
                1: begin
                    term_stall_lfsr <= {
                        term_stall_lfsr[14:0],
                        term_stall_lfsr[15]
                        ^ term_stall_lfsr[13]
                        ^ term_stall_lfsr[12]
                        ^ term_stall_lfsr[10]
                    };
                    term_issue_enable <=
                        term_stall_lfsr[2:0] != 3'b000;
                end
                2: begin
                    if (!first_term_hold_started && dut.term_valid) begin
                        first_term_hold_started <= 1'b1;
                        directed_hold_remaining <= 64;
                        term_issue_enable <= 1'b0;
                    end else if (directed_hold_remaining > 0) begin
                        directed_hold_remaining <= directed_hold_remaining - 1;
                        term_issue_enable <= 1'b0;
                    end else begin
                        term_issue_enable <= 1'b1;
                    end
                end
                3: begin
                    // Repeating 32-cycle stop/32-cycle release burst.
                    term_issue_enable <= !busy_half_cycles[5];
                end
                4: begin
                    if (
                        !final_term_hold_started
                        && dut.term_valid
                        && dut.term_descriptor_last
                        && 32'(dut.term_source_id) == TOTAL - 1
                    ) begin
                        final_term_hold_started <= 1'b1;
                        directed_hold_remaining <= 32;
                        term_issue_enable <= 1'b0;
                    end else if (directed_hold_remaining > 0) begin
                        directed_hold_remaining <= directed_hold_remaining - 1;
                        term_issue_enable <= 1'b0;
                        if (directed_hold_remaining == 1)
                            final_term_hold_done <= 1'b1;
                    end else begin
                        term_issue_enable <= 1'b1;
                    end
                end
                default: term_issue_enable <= 1'b1;
            endcase

            if (projection_busy && !term_issue_enable)
                term_issue_stall_cycles <= term_issue_stall_cycles + 1;
            if (
                dut.term_valid
                && dut.backend_term_ready
                && !term_issue_enable
            )
                term_boundary_block_hits <= term_boundary_block_hits + 1;
        end
    end

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

    function automatic int signed weight_value(
        input int lane,
        input int out
    );
        weight_value = (
            ((lane + 1) * 37 + (out + 1) * 53 + lane * out * 11) % 127
        ) - 63;
    endfunction

    task automatic load_python_fullchain_oracle;
        integer input_fd;
        integer expected_fd;
        integer rc;
        integer input_rows;
        integer expected_rows;
        integer rp;
        integer ry;
        integer rx;
        integer ro;
        integer signed rvalue;
        logic [31:0] rq;
        logic [31:0] rk0;
        logic [31:0] rk1;
        logic [31:0] rk2;
        logic [31:0] rk3;
        logic [31:0] rk4;
        logic [4:0] rmask;

        python_fullchain_enabled = 0;
        input_rows = 0;
        expected_rows = 0;
        for (int p = 0; p < TIME_PLANES; p = p + 1)
            for (int y = 0; y < HEIGHT; y = y + 1)
                for (int x = 0; x < WIDTH; x = x + 1) begin
                    python_input_seen[p][y][x] = 1'b0;
                    for (int out = 0; out < OUT_DIM; out = out + 1)
                        python_expected_seen[p][y][x][out] = 1'b0;
                end

        if (
            $value$plusargs("PY_INPUTS=%s", python_inputs_path)
            != $value$plusargs("PY_EXPECTED=%s", python_expected_path)
        )
            $fatal(1, "PY_INPUTS and PY_EXPECTED must be supplied together");
        if (python_inputs_path != "" || python_expected_path != "") begin
            input_fd = $fopen(python_inputs_path, "r");
            expected_fd = $fopen(python_expected_path, "r");
            if (input_fd == 0 || expected_fd == 0)
                $fatal(1, "cannot open independent Python fullchain oracle");

            while (!$feof(input_fd)) begin
                rc = $fscanf(
                    input_fd,
                    "%d %d %d %h %h %h %h %h %h %h\n",
                    rp, ry, rx, rq, rk0, rk1, rk2, rk3, rk4, rmask
                );
                if (rc == 10) begin
                    if (
                        rp < 0 || rp >= TIME_PLANES
                        || ry < 0 || ry >= HEIGHT
                        || rx < 0 || rx >= WIDTH
                        || python_input_seen[rp][ry][rx]
                    )
                        $fatal(1, "invalid or duplicate Python input row");
                    python_input_seen[rp][ry][rx] = 1'b1;
                    python_q[rp][ry][rx] = rq;
                    python_k[rp][ry][rx][0] = rk0;
                    python_k[rp][ry][rx][1] = rk1;
                    python_k[rp][ry][rx][2] = rk2;
                    python_k[rp][ry][rx][3] = rk3;
                    python_k[rp][ry][rx][4] = rk4;
                    python_valid_mask[rp][ry][rx] = rmask;
                    input_rows = input_rows + 1;
                end
            end
            while (!$feof(expected_fd)) begin
                rc = $fscanf(
                    expected_fd,
                    "%d %d %d %d %d\n",
                    rp, ry, rx, ro, rvalue
                );
                if (rc == 5) begin
                    if (
                        rp < 0 || rp >= TIME_PLANES
                        || ry < 0 || ry >= HEIGHT
                        || rx < 0 || rx >= WIDTH
                        || ro < 0 || ro >= OUT_DIM
                        || python_expected_seen[rp][ry][rx][ro]
                    )
                        $fatal(1, "invalid or duplicate Python expected row");
                    python_expected_seen[rp][ry][rx][ro] = 1'b1;
                    python_expected[rp][ry][rx][ro] = rvalue;
                    expected_rows = expected_rows + 1;
                end
            end
            $fclose(input_fd);
            $fclose(expected_fd);
            if (input_rows != TOTAL || expected_rows != TOTAL * OUT_DIM)
                $fatal(
                    1,
                    "incomplete Python oracle inputs=%0d expected=%0d",
                    input_rows,
                    expected_rows
                );
            python_fullchain_enabled = 1;
        end
    endtask

    task automatic load_weight(
        input int lane,
        input int out,
        input bit last
    );
        @(negedge clk_core);
        weight_lane = LANE_W'(lane);
        weight_out = OUT_W'(out);
        weight_data = W_W'(weight_value(lane, out));
        weight_last = last;
        weight_valid = 1'b1;
        @(posedge clk_core);
        if (!weight_ready)
            $fatal(1, "weight port not ready");
        @(negedge clk_core);
        weight_valid = 1'b0;
        weight_last = 1'b0;
        weight_context_release = 1'b0;
    endtask

    task automatic prepare_destination(
        input int p,
        input int y,
        input int x
    );
        int sy;
        int sx;
        in_y = Y_W'(y);
        in_x = X_W'(x);
        if (python_fullchain_enabled) begin
            in_q = python_q[p][y][x];
            in_valid_mask = python_valid_mask[p][y][x];
            in_k = '0;
            for (int role = 0; role < 5; role = role + 1)
                in_k[role*HEAD_DIM +: HEAD_DIM] =
                    python_k[p][y][x][role];
        end else begin
            in_q = query_bits(p, y, x);
            in_valid_mask = 5'b00001;
            in_k = '0;
            for (int role = 0; role < 5; role = role + 1) begin
                sy = y;
                sx = x;
                case (role)
                    1: sy = y - 1;
                    2: sy = y + 1;
                    3: sx = x - 1;
                    4: sx = x + 1;
                    default: begin end
                endcase
                if (
                    sy >= 0 && sy < HEIGHT
                    && sx >= 0 && sx < WIDTH
                ) begin
                    in_valid_mask[role] = 1'b1;
                    in_k[role*HEAD_DIM +: HEAD_DIM] =
                        source_k(p, sy, sx);
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
        end
    endtask

    task automatic accumulate_reference(
        input int p,
        input int y,
        input int x
    );
        int sy;
        int sx;
        logic [HEAD_DIM-1:0] source_bits;
        logic [GATE_W-1:0] gate;
        for (int role = 0; role < 5; role = role + 1) begin
            sy = y;
            sx = x;
            case (role)
                1: sy = y - 1;
                2: sy = y + 1;
                3: sx = x - 1;
                4: sx = x + 1;
                default: begin end
            endcase
            if (in_valid_mask[role]) begin
                source_bits = in_k[role*HEAD_DIM +: HEAD_DIM];
                gate = ref_gate[role*GATE_W +: GATE_W];
                for (int lane = 0; lane < HEAD_DIM; lane = lane + 1)
                    if (source_bits[lane])
                        for (
                            int out = 0;
                            out < OUT_DIM;
                            out = out + 1
                        )
                            expected[p][y][x][out] =
                                expected[p][y][x][out]
                                + gate * weight_value(lane, out);
            end
        end
    endtask

    task automatic drive_plane(input int p);
        logic handshake;
        @(negedge clk_core);
        plane_id = PLANE_W'(p);
        wait (plane_start_ready);
        @(negedge clk_core);
        plane_start = 1'b1;
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
            accumulate_reference(p, index / WIDTH, index % WIDTH);
            in_valid = 1'b0;
        end
        wait (dut.local_plane_start_ready);
    endtask

    task automatic check_acc(
        input int p,
        input int y,
        input int x,
        input int out
    );
        integer signed oracle_expected;
        @(negedge clk_core);
        read_plane = PLANE_W'(p);
        read_y = Y_W'(y);
        read_x = X_W'(x);
        read_out = OUT_W'(out);
        read_valid = 1'b1;
        @(posedge clk_core);
        if (!read_ready)
            $fatal(1, "read not ready");
        @(negedge clk_core);
        read_valid = 1'b0;
        wait (read_data_valid);
        #1;
        oracle_expected = python_fullchain_enabled
            ? python_expected[p][y][x][out]
            : expected[p][y][x][out];
        if (read_data !== ACC_W'(oracle_expected)) begin
            acc32_mismatch_count = acc32_mismatch_count + 1;
            if (python_fullchain_enabled)
                python_acc32_mismatch_count =
                    python_acc32_mismatch_count + 1;
            $fatal(
                1,
                "projection mismatch p=%0d y=%0d x=%0d out=%0d got=%0d exp=%0d",
                p, y, x, out, read_data, oracle_expected
            );
        end
        if (
            python_fullchain_enabled
            && expected[p][y][x][out]
               != python_expected[p][y][x][out]
        )
            $fatal(
                1,
                "SV/Python oracle disagreement p=%0d y=%0d x=%0d out=%0d sv=%0d py=%0d",
                p, y, x, out,
                expected[p][y][x][out],
                python_expected[p][y][x][out]
            );
        @(negedge clk_core);
    endtask

    always_comb begin
        for (int role = 0; role < 5; role = role + 1)
            ref_k[role] = in_k[role*HEAD_DIM +: HEAD_DIM];
    end

    // Equal-capacity/equal-port analytical baseline. Linear-5 uses the same
    // five one-update-per-cycle 1R1W banks but maps by raster token ID instead
    // of the Local5 topology coloring. Affine-4 uses the best topology-only
    // four-bank formula from the deployment-window DSE. Conflicting terms are
    // replayed until every selected destination retires.
    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            baseline_linear5_cycles <= 0;
            baseline_affine4_cycles <= 0;
            baseline_single_bank_cycles <= 0;
            tile_run_cycles <= 0;
            unique_lane_gate_products <= 0;
            lane1_hits <= 0;
            lane1_misses <= 0;
            lane2_hits <= 0;
            lane2_misses <= 0;
            dm16_hits <= 0;
            dm16_misses <= 0;
            for (int lane = 0; lane < HEAD_DIM; lane = lane + 1)
                for (int gate = 0; gate < (1 << GATE_W); gate = gate + 1) begin
                    product_key_seen[lane][gate] = 1'b0;
                    if (gate < 2) begin
                        lane2_valid[lane][gate] = 1'b0;
                        lane2_gate[lane][gate] = '0;
                    end
                    if (gate == 0) begin
                        lane1_valid[lane] = 1'b0;
                        lane1_gate[lane] = '0;
                        lane2_mru[lane] = 1'b0;
                    end
                end
            for (int entry = 0; entry < 16; entry = entry + 1) begin
                dm16_valid[entry] = 1'b0;
                dm16_tag[entry] = 0;
            end
        end else begin
            if (projection_busy)
                tile_run_cycles <= tile_run_cycles + 1;
            if (dut.term_valid && dut.term_ready) begin
                integer bank_count [0:4];
                integer affine4_count [0:3];
                integer max_count;
                integer affine4_max;
                integer product_key;
                integer dm16_index;
                integer sy;
                integer sx;
                integer dy;
                integer dx;
                integer linear_id;
                integer update_count;
                if (BACKEND_KIND == 0 && term_trace_fd != 0) begin
                    $fdisplay(
                        term_trace_fd,
                        "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d",
                        term_trace_seq,
                        dut.term_source_plane,
                        dut.term_source_y,
                        dut.term_source_x,
                        dut.term_lane,
                        dut.term_gate,
                        dut.term_destination_mask,
                        1'b0
                    );
                    term_trace_seq <= term_trace_seq + 1;
                end
                if (
                    !product_key_seen[
                        dut.term_lane
                    ][dut.term_gate]
                ) begin
                    product_key_seen[
                        dut.term_lane
                    ][dut.term_gate] <= 1'b1;
                    unique_lane_gate_products <=
                        unique_lane_gate_products + 1;
                end
                if (
                    lane1_valid[dut.term_lane]
                    && lane1_gate[dut.term_lane] == dut.term_gate
                ) begin
                    lane1_hits <= lane1_hits + 1;
                end else begin
                    lane1_misses <= lane1_misses + 1;
                    lane1_valid[dut.term_lane] <= 1'b1;
                    lane1_gate[dut.term_lane] <= dut.term_gate;
                end
                if (
                    lane2_valid[dut.term_lane][0]
                    && lane2_gate[dut.term_lane][0] == dut.term_gate
                ) begin
                    lane2_hits <= lane2_hits + 1;
                    lane2_mru[dut.term_lane] <= 1'b0;
                end else if (
                    lane2_valid[dut.term_lane][1]
                    && lane2_gate[dut.term_lane][1] == dut.term_gate
                ) begin
                    lane2_hits <= lane2_hits + 1;
                    lane2_mru[dut.term_lane] <= 1'b1;
                end else begin
                    lane2_misses <= lane2_misses + 1;
                    if (!lane2_valid[dut.term_lane][0]) begin
                        lane2_valid[dut.term_lane][0] <= 1'b1;
                        lane2_gate[dut.term_lane][0] <= dut.term_gate;
                        lane2_mru[dut.term_lane] <= 1'b0;
                    end else if (!lane2_valid[dut.term_lane][1]) begin
                        lane2_valid[dut.term_lane][1] <= 1'b1;
                        lane2_gate[dut.term_lane][1] <= dut.term_gate;
                        lane2_mru[dut.term_lane] <= 1'b1;
                    end else if (lane2_mru[dut.term_lane]) begin
                        lane2_gate[dut.term_lane][0] <= dut.term_gate;
                        lane2_mru[dut.term_lane] <= 1'b0;
                    end else begin
                        lane2_gate[dut.term_lane][1] <= dut.term_gate;
                        lane2_mru[dut.term_lane] <= 1'b1;
                    end
                end
                product_key =
                    dut.term_lane * (1 << GATE_W) + dut.term_gate;
                dm16_index = product_key & 15;
                if (
                    dm16_valid[dm16_index]
                    && dm16_tag[dm16_index] == product_key
                ) begin
                    dm16_hits <= dm16_hits + 1;
                end else begin
                    dm16_misses <= dm16_misses + 1;
                    dm16_valid[dm16_index] <= 1'b1;
                    dm16_tag[dm16_index] <= product_key;
                end
                for (int bank = 0; bank < 5; bank = bank + 1)
                    bank_count[bank] = 0;
                for (int bank = 0; bank < 4; bank = bank + 1)
                    affine4_count[bank] = 0;
                update_count = 0;
                sy = dut.term_source_y;
                sx = dut.term_source_x;
                for (int role = 0; role < 5; role = role + 1) begin
                    dy = sy;
                    dx = sx;
                    case (role)
                        1: dy = sy + 1;
                        2: dy = sy - 1;
                        3: dx = sx + 1;
                        4: dx = sx - 1;
                        default: begin end
                    endcase
                    if (dut.term_destination_mask[role]) begin
                        linear_id =
                            dut.term_source_plane * TOKENS
                            + dy * WIDTH + dx;
                        bank_count[linear_id % 5] =
                            bank_count[linear_id % 5] + 1;
                        affine4_count[((dx + 2*dy) % 4 + 4) % 4] =
                            affine4_count[((dx + 2*dy) % 4 + 4) % 4]
                            + 1;
                        update_count = update_count + 1;
                    end
                end
                max_count = 0;
                for (int bank = 0; bank < 5; bank = bank + 1)
                    if (bank_count[bank] > max_count)
                        max_count = bank_count[bank];
                affine4_max = 0;
                for (int bank = 0; bank < 4; bank = bank + 1)
                    if (affine4_count[bank] > affine4_max)
                        affine4_max = affine4_count[bank];
                baseline_linear5_cycles <=
                    baseline_linear5_cycles + max_count;
                baseline_affine4_cycles <=
                    baseline_affine4_cycles + affine4_max;
                baseline_single_bank_cycles <=
                    baseline_single_bank_cycles + update_count;
            end
        end
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
        plane_start = 1'b0;
        plane_id = '0;
        in_valid = 1'b0;
        in_y = '0;
        in_x = '0;
        in_q = '0;
        in_k = '0;
        in_valid_mask = '0;
        read_valid = 1'b0;
        read_plane = '0;
        read_y = '0;
        read_x = '0;
        read_out = '0;
        baseline_linear5_cycles = 0;
        baseline_single_bank_cycles = 0;
        tile_run_cycles = 0;
        unique_lane_gate_products = 0;
        term_trace_fd = 0;
        term_trace_seq = 0;
        term_stall_seed = 0;
        term_stall_mode = 0;
        acc32_mismatch_count = 0;
        python_acc32_mismatch_count = 0;
        python_inputs_path = "";
        python_expected_path = "";
        load_python_fullchain_oracle();
        if ($value$plusargs("TERM_STALL_SEED=%d", term_stall_seed)) begin
            if (term_stall_seed == 0)
                $fatal(1, "TERM_STALL_SEED must be nonzero");
            term_stall_mode = 1;
        end
        if ($value$plusargs("TERM_STALL_MODE=%d", term_stall_mode)) begin
            if (term_stall_mode < 0 || term_stall_mode > 4)
                $fatal(1, "TERM_STALL_MODE must be in [0,4]");
        end
        if (term_stall_mode == 1 && term_stall_seed == 0)
            $fatal(1, "LFSR stall mode requires TERM_STALL_SEED");
        if (
            BACKEND_KIND == 0
            && $value$plusargs("TERM_TRACE=%s", term_trace_path)
        ) begin
            term_trace_fd = $fopen(term_trace_path, "w");
            if (term_trace_fd == 0)
                $fatal(1, "cannot open term trace %s", term_trace_path);
            $fdisplay(
                term_trace_fd,
                "seq,plane,y,x,lane,gate,mask,window_last"
            );
        end
        for (int p = 0; p < TIME_PLANES; p = p + 1)
            for (int y = 0; y < HEIGHT; y = y + 1)
                for (int x = 0; x < WIDTH; x = x + 1)
                    for (int out = 0; out < OUT_DIM; out = out + 1)
                        expected[p][y][x][out] = 0;
        repeat (4) @(negedge clk_core);
        rst_core = 1'b0;

        // An early close is rejected and reported. projection_start below
        // begins a new transaction and clears this sticky protocol status.
        @(negedge clk_core);
        projection_close = 1'b1;
        @(negedge clk_core);
        projection_close = 1'b0;
        if (!protocol_error)
            $fatal(1, "early projection close was not reported");

        for (int lane = 0; lane < HEAD_DIM; lane = lane + 1)
            for (int out = 0; out < OUT_DIM; out = out + 1)
                load_weight(
                    lane,
                    out,
                    lane == HEAD_DIM - 1 && out == OUT_DIM - 1
                );
        wait (projection_start_ready);
        @(negedge clk_core);
        projection_start = 1'b1;
        @(negedge clk_core);
        projection_start = 1'b0;
        wait (projection_busy);
        if (protocol_error)
            $fatal(1, "projection start did not clear close error");

        drive_plane(0);
        drive_plane(1);
        wait (perf_descriptors == TOTAL && stream_idle);
        wait (projection_close_ready);
        repeat (2) @(negedge clk_core);
        projection_close = 1'b1;
        @(negedge clk_core);
        projection_close = 1'b0;
        wait (projection_done);

        for (int p = 0; p < TIME_PLANES; p = p + 1)
            for (int y = 0; y < HEIGHT; y = y + 1)
                for (int x = 0; x < WIDTH; x = x + 1)
                    for (int out = 0; out < OUT_DIM; out = out + 1)
                        check_acc(p, y, x, out);
        if (protocol_error)
            $fatal(1, "unexpected projection protocol error");
        if (perf_descriptors != TOTAL)
            $fatal(1, "descriptor count mismatch");
        if (
            perf_product_terms == 0
            || perf_destination_updates < perf_product_terms
        )
            $fatal(1, "invalid projection performance counters");
        if (term_stall_mode != 0 && term_boundary_block_hits == 0)
            $fatal(1, "fixed-seed backpressure did not hit a ready term");
        if (term_stall_mode == 2 && !first_term_hold_started)
            $fatal(1, "first-term full-stop mode was not exercised");
        if (
            term_stall_mode == 4
            && (!final_term_hold_started || !final_term_hold_done)
        )
            $fatal(1, "last-term directed stall was not completed");
        $display(
            "PASS qfit_local5_projection_tile backend=%0d tile_cycles=%0d descriptors=%0d terms=%0d updates=%0d stalls=%0d issue_seed=%0d issue_stall_cycles=%0d issue_block_hits=%0d lane_gate_products=%0d lane1_hits=%0d lane1_misses=%0d lane2_hits=%0d lane2_misses=%0d dm16_hits=%0d dm16_misses=%0d linear5_cycles=%0d affine4_cycles=%0d single_cycles=%0d acc32_mismatch=%0d python_fullchain_miter=%0d python_acc32_mismatch=%0d",
            BACKEND_KIND,
            tile_run_cycles,
            perf_descriptors,
            perf_product_terms,
            perf_destination_updates,
            perf_relation_stalls,
            term_stall_seed,
            term_issue_stall_cycles,
            term_boundary_block_hits,
            unique_lane_gate_products,
            lane1_hits,
            lane1_misses,
            lane2_hits,
            lane2_misses,
            dm16_hits,
            dm16_misses,
            baseline_linear5_cycles,
            baseline_affine4_cycles,
            baseline_single_bank_cycles,
            acc32_mismatch_count,
            python_fullchain_enabled,
            python_acc32_mismatch_count
        );
        $display(
            "LOCAL5_STALL_COVERAGE mode=%0d first_hold=%0d final_hold=%0d final_done=%0d",
            term_stall_mode,
            first_term_hold_started,
            final_term_hold_started,
            final_term_hold_done
        );
        if (term_trace_fd != 0)
            $fclose(term_trace_fd);
        $finish;
    end
endmodule

`default_nettype wire
