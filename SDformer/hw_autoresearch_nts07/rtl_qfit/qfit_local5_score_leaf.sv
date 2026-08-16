`timescale 1ns/1ps
`default_nettype none

// Local5 exact score leaf.
// ARCH_QFSA=0: four fixed W1 residual lanes.
// ARCH_QFSA=1: one cross-direction tagged W4 lane pool.
// PIPE_COMPACTOR=1 registers selected tags/lane IDs before delta reduction.
module qfit_local5_score_leaf #(
    parameter bit ARCH_QFSA = 1'b1,
    parameter bit PIPE_COMPACTOR = 1'b0,
    parameter bit XBF_BANKED = 1'b0,
    // Default-off exact alternative: remove the shared even Q7 translation.
    parameter bit ARCH_PHASE_RESIDUAL = 1'b0,
    parameter bit USE_THRESHOLD_ROUTE = 1'b0,
    parameter int ROUTE_THRESHOLD = 8,
    parameter bit USE_BANK_PRESSURE_ROUTE = 1'b0,
    parameter int BANK_PRESSURE_THRESHOLD = 2,
    parameter int TAG_W = 16,
    parameter int SCORE_W = 16,
    parameter int GATE_W = 9
) (
    input  logic                     clk_core,
    input  logic                     rst_core,
    input  logic                     in_valid,
    output logic                     in_ready,
    input  logic [TAG_W-1:0]         in_tag,
    input  logic [31:0]              in_q,
    input  logic [5*32-1:0]          in_k,
    input  logic [4:0]               in_valid_mask,
    output logic                     out_valid,
    input  logic                     out_ready,
    output logic [TAG_W-1:0]         out_tag,
    output logic [5*SCORE_W-1:0]     out_score_q7,
    output logic [5*GATE_W-1:0]      out_gate_q17,
    output logic [31:0]              out_k_self,
    output logic [4:0]               out_valid_mask,
    output logic [15:0]              perf_service_cycles,
    output logic [3:0]               perf_route_direct_mask
);

    typedef enum logic [2:0] {
        ST_IDLE    = 3'd0,
        ST_ROUTE   = 3'd1,
        ST_SERVICE = 3'd2,
        ST_RNE     = 3'd3,
        ST_OUT     = 3'd4
    } state_t;

    state_t state_q;
    logic [TAG_W-1:0] tag_q;
    logic [31:0] q_q;
    logic [31:0] k_q [0:4];
    logic [4:0] valid_q;
    logic signed [12:0] acc_q [0:4];
    logic signed [SCORE_W-1:0] anchor_base_q;
    logic signed [8:0] phase_score_q [0:4];
    logic [31:0] residual_q [0:3];
    logic [3:0] direct_q;
    logic signed [SCORE_W-1:0] score_q [0:4];
    logic [15:0] service_cycles_q;
    logic [3:0] route_mask_q;

    logic pipe_valid_q;
    logic [3:0] pipe_event_valid_q;
    logic [7:0] pipe_event_dir_q;
    logic [19:0] pipe_event_lane_q;

    logic [31:0] delta_mask [0:3];
    logic [5:0] delta_count [0:3];
    logic [3:0] route_direct_mask;
    logic [5:0] route_best_cost;

    logic [31:0] compact_next [0:3];
    logic [3:0] compact_event_valid;
    logic [7:0] compact_event_dir;
    logic [19:0] compact_event_lane;
    logic [31:0] xorbank_next [0:3];
    logic [3:0] xorbank_event_valid;
    logic [7:0] xorbank_event_dir;
    logic [19:0] xorbank_event_lane;
    logic [31:0] service_next [0:3];
    logic [3:0] service_event_valid;
    logic [7:0] service_event_dir;
    logic [19:0] service_event_lane;
    logic signed [9:0] compact_delta [0:3];
    logic signed [9:0] pipe_delta [0:3];

    logic [31:0] w1_next [0:3];
    logic signed [9:0] w1_delta [0:3];
    logic [3:0] w1_valid;
    logic [5:0] delta_bank_count [0:3][0:3];

    logic [4:0] score_valid;
    logic [5*SCORE_W-1:0] score_bus;
    logic [5*9-1:0] phase_score_bus;
    logic [5*GATE_W-1:0] gate_bus;

    function automatic logic [5:0] popcount32(
        input logic [31:0] bits
    );
        logic [5:0] count;
        count = '0;
        for (int lane = 0; lane < 32; lane = lane + 1)
            count = count + 6'(bits[lane]);
        popcount32 = count;
    endfunction

    function automatic logic [11:0] raw16(
        input logic [31:0] q_bits,
        input logic [31:0] k_bits
    );
        logic [11:0] value;
        value = '0;
        for (int lane = 0; lane < 32; lane = lane + 1) begin
            if (q_bits[lane] && k_bits[lane])
                value = value + 12'd64;
            else if (!q_bits[lane] && !k_bits[lane])
                value = value + 12'd1;
        end
        raw16 = value;
    endfunction

    function automatic logic signed [9:0] lane_delta(
        input logic q_bit,
        input logic old_k,
        input logic new_k
    );
        logic signed [9:0] old_value;
        logic signed [9:0] new_value;
        old_value = (q_bit && old_k) ? 10'sd64
                  : ((!q_bit && !old_k) ? 10'sd1 : 10'sd0);
        new_value = (q_bit && new_k) ? 10'sd64
                  : ((!q_bit && !new_k) ? 10'sd1 : 10'sd0);
        lane_delta = new_value - old_value;
    endfunction

    function automatic logic signed [SCORE_W-1:0] rne_q7(
        input logic signed [12:0] raw_value
    );
        logic [12:0] nonnegative;
        logic [8:0] quotient;
        logic [3:0] remainder;
        logic increment;
        nonnegative = raw_value[12] ? 13'd0 : raw_value;
        quotient = nonnegative[12:4];
        remainder = nonnegative[3:0];
        increment = (remainder > 4'd8)
                 || ((remainder == 4'd8) && quotient[0]);
        rne_q7 = $signed({7'b0, quotient})
               + SCORE_W'(increment);
    endfunction

    always_comb begin
        for (int dir = 0; dir < 4; dir = dir + 1) begin
            delta_mask[dir] = (k_q[0] ^ k_q[dir+1])
                            & {32{valid_q[dir+1]}};
            delta_count[dir] = popcount32(delta_mask[dir]);
            for (int bank = 0; bank < 4; bank = bank + 1) begin
                delta_bank_count[dir][bank] = '0;
                for (int lane = 0; lane < 32; lane = lane + 1) begin
                    if ((lane[1:0] ^ dir[1:0]) == bank[1:0])
                        delta_bank_count[dir][bank]
                            = delta_bank_count[dir][bank]
                            + 6'(delta_mask[dir][lane]);
                end
            end
        end
    end

    always_comb begin
        logic [5:0] best_cost;
        logic [5:0] direct_cycles;
        logic [7:0] residual_total;
        logic [5:0] residual_cycles;
        logic [5:0] residual_effective;
        logic [5:0] fixed_max;
        logic [6:0] bank_load [0:3];
        best_cost = 6'd63;
        route_direct_mask = '0;
        if (USE_THRESHOLD_ROUTE) begin
            best_cost = '0;
            for (int dir = 0; dir < 4; dir = dir + 1) begin
                logic [5:0] direction_max_bank;
                direction_max_bank = '0;
                for (int bank = 0; bank < 4; bank = bank + 1) begin
                    if (
                        delta_bank_count[dir][bank]
                        > direction_max_bank
                    )
                        direction_max_bank =
                            delta_bank_count[dir][bank];
                end
                route_direct_mask[dir] = valid_q[dir+1]
                    && (
                        (delta_count[dir] > 6'(ROUTE_THRESHOLD))
                        || (
                            USE_BANK_PRESSURE_ROUTE
                            && XBF_BANKED
                            && (
                                direction_max_bank
                                > 6'(BANK_PRESSURE_THRESHOLD)
                            )
                        )
                    );
                if (
                    route_direct_mask[dir]
                    || (
                        valid_q[dir+1]
                        && delta_count[dir] != 0
                    )
                )
                    best_cost = 6'd1;
            end
        end else begin
            for (int mask = 0; mask < 16; mask = mask + 1) begin
                direct_cycles = '0;
                residual_total = '0;
                fixed_max = '0;
                for (int bank = 0; bank < 4; bank = bank + 1)
                    bank_load[bank] = '0;
                for (int dir = 0; dir < 4; dir = dir + 1) begin
                    if (mask[dir] && valid_q[dir+1])
                        direct_cycles = direct_cycles + 6'd1;
                    if (!mask[dir]) begin
                        residual_total = residual_total
                                       + {2'b00, delta_count[dir]};
                        if (delta_count[dir] > fixed_max)
                            fixed_max = delta_count[dir];
                        for (int bank = 0; bank < 4; bank = bank + 1)
                            bank_load[bank] = bank_load[bank]
                                + {1'b0, delta_bank_count[dir][bank]};
                    end
                end
                if (ARCH_QFSA && XBF_BANKED) begin
                    residual_cycles = '0;
                    for (int bank = 0; bank < 4; bank = bank + 1) begin
                        if (bank_load[bank] > {1'b0, residual_cycles})
                            residual_cycles = 6'(bank_load[bank]);
                    end
                end else if (ARCH_QFSA)
                    residual_cycles = 6'(
                        (residual_total + 8'd3) >> 2
                    );
                else
                    residual_cycles = fixed_max;
                residual_effective = residual_cycles
                    + 6'(
                        ARCH_QFSA
                        && PIPE_COMPACTOR
                        && residual_cycles != 0
                    );
                if (
                    (
                        direct_cycles
                        > residual_effective
                        ? direct_cycles
                        : residual_effective
                    ) < best_cost
                ) begin
                    best_cost = (
                        direct_cycles
                        > residual_effective
                        ? direct_cycles
                        : residual_effective
                    );
                    route_direct_mask = 4'(mask);
                end
            end
        end
        route_best_cost = best_cost;
    end

    qfit_tagged_compactor4 u_compactor (
        .mask_n(residual_q[0]),
        .mask_s(residual_q[1]),
        .mask_e(residual_q[2]),
        .mask_w(residual_q[3]),
        .next_mask_n(compact_next[0]),
        .next_mask_s(compact_next[1]),
        .next_mask_e(compact_next[2]),
        .next_mask_w(compact_next[3]),
        .event_valid(compact_event_valid),
        .event_dir(compact_event_dir),
        .event_lane(compact_event_lane)
    );

    qfit_xorbank_compactor4 u_xorbank (
        .mask_n(residual_q[0]),
        .mask_s(residual_q[1]),
        .mask_e(residual_q[2]),
        .mask_w(residual_q[3]),
        .next_mask_n(xorbank_next[0]),
        .next_mask_s(xorbank_next[1]),
        .next_mask_e(xorbank_next[2]),
        .next_mask_w(xorbank_next[3]),
        .event_valid(xorbank_event_valid),
        .event_dir(xorbank_event_dir),
        .event_lane(xorbank_event_lane)
    );

    always_comb begin
        if (XBF_BANKED) begin
            for (int dir = 0; dir < 4; dir = dir + 1)
                service_next[dir] = xorbank_next[dir];
            service_event_valid = xorbank_event_valid;
            service_event_dir = xorbank_event_dir;
            service_event_lane = xorbank_event_lane;
        end else begin
            for (int dir = 0; dir < 4; dir = dir + 1)
                service_next[dir] = compact_next[dir];
            service_event_valid = compact_event_valid;
            service_event_dir = compact_event_dir;
            service_event_lane = compact_event_lane;
        end
    end

    always_comb begin
        for (int dir = 0; dir < 4; dir = dir + 1) begin
            compact_delta[dir] = '0;
            pipe_delta[dir] = '0;
            w1_next[dir] = residual_q[dir];
            w1_delta[dir] = '0;
            w1_valid[dir] = 1'b0;
        end
        for (int way = 0; way < 4; way = way + 1) begin
            logic [1:0] dir;
            logic [4:0] lane;
            dir = service_event_dir[way*2 +: 2];
            lane = service_event_lane[way*5 +: 5];
            if (service_event_valid[way]) begin
                compact_delta[dir] = compact_delta[dir]
                    + lane_delta(
                        q_q[lane],
                        k_q[0][lane],
                        k_q[dir+1][lane]
                    );
            end
            dir = pipe_event_dir_q[way*2 +: 2];
            lane = pipe_event_lane_q[way*5 +: 5];
            if (pipe_event_valid_q[way]) begin
                pipe_delta[dir] = pipe_delta[dir]
                    + lane_delta(
                        q_q[lane],
                        k_q[0][lane],
                        k_q[dir+1][lane]
                    );
            end
        end
        for (int dir = 0; dir < 4; dir = dir + 1) begin
            logic found;
            found = 1'b0;
            for (int lane = 0; lane < 32; lane = lane + 1) begin
                if (!found && residual_q[dir][lane]) begin
                    w1_valid[dir] = 1'b1;
                    w1_next[dir][lane] = 1'b0;
                    w1_delta[dir] = lane_delta(
                        q_q[lane],
                        k_q[0][lane],
                        k_q[dir+1][lane]
                    );
                    found = 1'b1;
                end
            end
        end
    end

    always_comb begin
        for (int cand = 0; cand < 5; cand = cand + 1) begin
            phase_score_bus[cand*9 +: 9] = phase_score_q[cand];
            if (ARCH_PHASE_RESIDUAL)
                score_bus[cand*SCORE_W +: SCORE_W] =
                    {{(SCORE_W-9){phase_score_q[cand][8]}}, phase_score_q[cand]};
            else
                score_bus[cand*SCORE_W +: SCORE_W] = score_q[cand];
        end
        score_valid = valid_q;
    end

    generate
        if (ARCH_PHASE_RESIDUAL) begin : g_phase_shiftmax
            local5_shiftmax5_q17 #(
                .N_CAND(5),
                .SCORE_W(9),
                .GATE_W(GATE_W)
            ) u_shiftmax (
                .score_q7(phase_score_bus),
                .valid(score_valid),
                .gate_q17(gate_bus)
            );
        end else begin : g_absolute_shiftmax
            local5_shiftmax5_q17 #(
                .N_CAND(5),
                .SCORE_W(SCORE_W),
                .GATE_W(GATE_W)
            ) u_shiftmax (
                .score_q7(score_bus),
                .valid(score_valid),
                .gate_q17(gate_bus)
            );
        end
    endgenerate

    assign in_ready = (state_q == ST_IDLE);
    assign out_valid = (state_q == ST_OUT);
    assign out_tag = tag_q;
    assign out_score_q7 = score_bus;
    assign out_gate_q17 = gate_bus;
    assign out_k_self = k_q[0];
    assign out_valid_mask = valid_q;
    assign perf_service_cycles = service_cycles_q;
    assign perf_route_direct_mask = route_mask_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            tag_q <= '0;
            q_q <= '0;
            valid_q <= '0;
            direct_q <= '0;
            anchor_base_q <= '0;
            pipe_valid_q <= 1'b0;
            pipe_event_valid_q <= '0;
            pipe_event_dir_q <= '0;
            pipe_event_lane_q <= '0;
            service_cycles_q <= '0;
            route_mask_q <= '0;
            for (int cand = 0; cand < 5; cand = cand + 1) begin
                k_q[cand] <= '0;
                acc_q[cand] <= '0;
                score_q[cand] <= '0;
                phase_score_q[cand] <= '0;
            end
            for (int dir = 0; dir < 4; dir = dir + 1)
                residual_q[dir] <= '0;
        end else begin
            case (state_q)
                ST_IDLE: begin
                    service_cycles_q <= '0;
                    if (in_valid) begin
                        tag_q <= in_tag;
                        q_q <= in_q;
                        valid_q <= in_valid_mask;
                        for (int cand = 0; cand < 5; cand = cand + 1)
                            k_q[cand] <= in_k[cand*32 +: 32];
                        state_q <= ST_ROUTE;
                    end
                end

                ST_ROUTE: begin
                    logic [11:0] anchor;
                    anchor = raw16(q_q, k_q[0]);
                    anchor_base_q <= SCORE_W'({anchor[11:5], 1'b0});
                    acc_q[0] <= $signed({1'b0, anchor});
                    for (int dir = 0; dir < 4; dir = dir + 1) begin
                        acc_q[dir+1] <= $signed({1'b0, anchor});
                        residual_q[dir] <= route_direct_mask[dir]
                            ? 32'd0 : delta_mask[dir];
                    end
                    direct_q <= route_direct_mask
                              & valid_q[4:1];
                    route_mask_q <= route_direct_mask
                                  & valid_q[4:1];
                    pipe_valid_q <= 1'b0;
                    if (
                        route_best_cost == 0
                        || valid_q[4:1] == 4'd0
                    )
                        state_q <= ST_RNE;
                    else
                        state_q <= ST_SERVICE;
                end

                ST_SERVICE: begin
                    logic [3:0] direct_after;
                    logic direct_found;
                    logic residual_after_any;
                    logic new_pipe_valid;
                    direct_after = direct_q;
                    direct_found = 1'b0;
                    for (int dir = 0; dir < 4; dir = dir + 1) begin
                        if (!direct_found && direct_q[dir]) begin
                            acc_q[dir+1] <= $signed(
                                {1'b0, raw16(q_q, k_q[dir+1])}
                            );
                            direct_after[dir] = 1'b0;
                            direct_found = 1'b1;
                        end
                    end
                    direct_q <= direct_after;

                    if (ARCH_QFSA) begin
                        if (PIPE_COMPACTOR && pipe_valid_q) begin
                            for (int dir = 0; dir < 4; dir = dir + 1) begin
                                if (pipe_delta[dir] != 0)
                                    acc_q[dir+1] <=
                                        acc_q[dir+1]
                                        + {{3{pipe_delta[dir][9]}},
                                           pipe_delta[dir]};
                            end
                        end
                        if (!PIPE_COMPACTOR) begin
                            for (int dir = 0; dir < 4; dir = dir + 1) begin
                                if (compact_delta[dir] != 0)
                                    acc_q[dir+1] <=
                                        acc_q[dir+1]
                                        + {{3{compact_delta[dir][9]}},
                                           compact_delta[dir]};
                            end
                        end
                        residual_after_any = 1'b0;
                        for (int dir = 0; dir < 4; dir = dir + 1) begin
                            residual_q[dir] <= service_next[dir];
                            residual_after_any = residual_after_any
                                               || (service_next[dir] != 0);
                        end
                        new_pipe_valid = PIPE_COMPACTOR
                                       && (service_event_valid != 0);
                        pipe_valid_q <= new_pipe_valid;
                        if (PIPE_COMPACTOR) begin
                            pipe_event_valid_q <= service_event_valid;
                            pipe_event_dir_q <= service_event_dir;
                            pipe_event_lane_q <= service_event_lane;
                        end
                    end else begin
                        residual_after_any = 1'b0;
                        new_pipe_valid = 1'b0;
                        for (int dir = 0; dir < 4; dir = dir + 1) begin
                            residual_q[dir] <= w1_next[dir];
                            if (w1_valid[dir])
                                acc_q[dir+1] <=
                                    acc_q[dir+1]
                                    + {{3{w1_delta[dir][9]}},
                                       w1_delta[dir]};
                            residual_after_any = residual_after_any
                                               || (w1_next[dir] != 0);
                        end
                    end
                    service_cycles_q <= service_cycles_q + 16'd1;
                    if (
                        direct_after == 0
                        && !residual_after_any
                        && !new_pipe_valid
                    )
                        state_q <= ST_RNE;
                end

                ST_RNE: begin
                    for (int cand = 0; cand < 5; cand = cand + 1) begin
                        if (ARCH_PHASE_RESIDUAL) begin
                            if (valid_q[cand]) begin
                                phase_score_q[cand] <= 9'(
                                    rne_q7(acc_q[cand]) - anchor_base_q
                                );
                            end else begin
                                phase_score_q[cand] <= -9'sd256;
                            end
                        end else begin
                            if (valid_q[cand])
                                score_q[cand] <= rne_q7(acc_q[cand]);
                            else
                                score_q[cand] <= SCORE_W'(-256);
                        end
                    end
                    state_q <= ST_OUT;
                end

                ST_OUT: begin
                    if (out_ready)
                        state_q <= ST_IDLE;
                end

                default: state_q <= ST_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
