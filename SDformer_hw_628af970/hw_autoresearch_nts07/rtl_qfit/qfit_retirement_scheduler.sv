`timescale 1ns/1ps
`default_nettype none

// Fixed-stencil source retirement scheduler.
// MODE=0: closed-form frontier (FCSR).
// MODE=1: dynamic per-source completion counters.
// MODE=2: nonblocking two-context row stripe.
module qfit_retirement_scheduler #(
    parameter int MODE = 0,
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int DILATION = 1,
    // In FCSR mode, bits 0/1/2 of in_candidate_valid may qualify the
    // north-row, bottom-row-left, and final-self retirement events.
    parameter bit FILTER_FCSR_EVENTS = 1'b0,
    parameter bit STRIPE_EARLY_FILL = 1'b0,
    parameter int STRIPE_RING_ROWS = 3,
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int PLANE_W =
        (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int SOURCE_ID_W =
        (HEIGHT * WIDTH * TIME_PLANES <= 1)
        ? 1 : $clog2(HEIGHT * WIDTH * TIME_PLANES)
) (
    input  logic                   clk_core,
    input  logic                   rst_core,
    input  logic                   plane_start,
    input  logic [PLANE_W-1:0]     plane_id,
    input  logic                   in_valid,
    output logic                   in_ready,
    input  logic [Y_W-1:0]         in_y,
    input  logic [X_W-1:0]         in_x,
    input  logic [4:0]             in_candidate_valid,
    output logic                   retire_valid,
    input  logic                   retire_ready,
    output logic [SOURCE_ID_W-1:0] retire_source_id,
    output logic [Y_W-1:0]         retire_y,
    output logic [X_W-1:0]         retire_x,
    output logic                   plane_idle,
    output logic [31:0]            perf_producer_stalls,
    output logic [2:0]             perf_max_pending
);

    localparam int TOKENS = HEIGHT * WIDTH;
    localparam int COUNTER_ROWS = 2 * DILATION + 1;
    localparam int COUNTER_ENTRIES = COUNTER_ROWS * WIDTH;
    localparam int MODE_FCSR = 0;
    localparam int MODE_DYNAMIC = 1;
    localparam int MODE_STRIPE = 2;

    logic [PLANE_W-1:0] plane_q;
    logic retire_valid_q;
    logic [SOURCE_ID_W-1:0] retire_source_q;
    logic [Y_W-1:0] retire_y_q;
    logic [X_W-1:0] retire_x_q;
    logic [31:0] stalls_q;
    logic [2:0] max_pending_q;
    logic output_slot_available;

    logic [COUNTER_ENTRIES*3-1:0] seen_flat_q;
    logic [SOURCE_ID_W-1:0] event_id [0:4];
    logic [Y_W-1:0] event_y [0:4];
    logic [X_W-1:0] event_x [0:4];
    logic [2:0] event_count;
    logic [SOURCE_ID_W-1:0] pending_id_q [0:3];
    logic [Y_W-1:0] pending_y_q [0:3];
    logic [X_W-1:0] pending_x_q [0:3];
    logic [2:0] pending_count_q;

`ifdef QFIT_USE_COMPILED_CROSS_R1
    logic [2:0] compiled_event_valid;
    logic [3*Y_W-1:0] compiled_event_y;
    logic [3*X_W-1:0] compiled_event_x;

    generated_cross_r1_retirement_rules #(
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .Y_W(Y_W),
        .X_W(X_W)
    ) u_compiled_cross_r1_rules (
        .in_valid(in_valid),
        .in_y(in_y),
        .in_x(in_x),
        .event_valid(compiled_event_valid),
        .event_y(compiled_event_y),
        .event_x(compiled_event_x)
    );
`elsif QFIT_USE_COMPILED_CROSS_R2
    logic [2:0] compiled_event_valid;
    logic [3*Y_W-1:0] compiled_event_y;
    logic [3*X_W-1:0] compiled_event_x;

    generated_cross_r2_retirement_rules #(
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .Y_W(Y_W),
        .X_W(X_W)
    ) u_compiled_cross_r2_rules (
        .in_valid(in_valid),
        .in_y(in_y),
        .in_x(in_x),
        .event_valid(compiled_event_valid),
        .event_y(compiled_event_y),
        .event_x(compiled_event_x)
    );
`endif

    logic [Y_W-1:0] stripe_row_q [0:1];
    logic stripe_head_q;
    logic stripe_tail_q;
    logic [1:0] stripe_context_count_q;
    logic [X_W-1:0] stripe_x_q;
    logic [1:0] stripe_release_count;
    logic [Y_W-1:0] stripe_release_row [0:1];
    logic stripe_emit;
    logic stripe_pop;
    logic stripe_row_start_block;
    logic [2:0] stripe_capacity;

    function automatic logic [SOURCE_ID_W-1:0] make_source_id(
        input logic [PLANE_W-1:0] p,
        input int y,
        input int x
    );
        int value;
        value = p * TOKENS + y * WIDTH + x;
        make_source_id = value;
    endfunction

    function automatic logic [2:0] expected_consumers(
        input int y,
        input int x
    );
        int count;
        count = 1;
        if (y >= DILATION)
            count = count + 1;
        if (y < HEIGHT - DILATION)
            count = count + 1;
        if (x >= DILATION)
            count = count + 1;
        if (x < WIDTH - DILATION)
            count = count + 1;
        expected_consumers = count;
    endfunction

    always_comb begin
        logic [4:0] completion;
        logic [SOURCE_ID_W-1:0] candidate_id [0:4];
        logic [2:0] candidate_seen [0:4];
        logic [$clog2(COUNTER_ENTRIES+1)-1:0]
            candidate_local [0:4];
        int sy [0:4];
        int sx [0:4];
        int role_order [0:4];

        sy[0] = in_y;
        sx[0] = in_x;
        sy[1] = in_y - DILATION;
        sx[1] = in_x;
        sy[2] = in_y + DILATION;
        sx[2] = in_x;
        sy[3] = in_y;
        sx[3] = in_x - DILATION;
        sy[4] = in_y;
        sx[4] = in_x + DILATION;
        role_order[0] = 1;
        role_order[1] = 3;
        role_order[2] = 0;
        role_order[3] = 2;
        role_order[4] = 4;

        completion = '0;
        for (int role = 0; role < 5; role = role + 1) begin
            candidate_id[role] = '0;
            candidate_local[role] = '0;
            candidate_seen[role] = '0;
            if (in_candidate_valid[role]) begin
                candidate_id[role] = make_source_id(
                    plane_q,
                    sy[role],
                    sx[role]
                );
                candidate_local[role] =
                    $clog2(COUNTER_ENTRIES+1)'(
                        (sy[role] % COUNTER_ROWS) * WIDTH + sx[role]
                    );
                candidate_seen[role] = seen_flat_q[
                    3 * candidate_local[role] +: 3
                ];
                if (
                    role == 2
                    && in_x == 0
                    && in_y < HEIGHT - DILATION
                )
                    candidate_seen[role] = '0;
                completion[role] = (
                    candidate_seen[role] + 3'd1
                    == expected_consumers(sy[role], sx[role])
                );
            end
        end

        event_count = '0;
        for (int slot = 0; slot < 5; slot = slot + 1)
            event_id[slot] = '0;
        for (int slot = 0; slot < 5; slot = slot + 1) begin
            event_y[slot] = '0;
            event_x[slot] = '0;
        end

        if (MODE == MODE_FCSR) begin
`ifdef QFIT_USE_COMPILED_CROSS_R1
            for (int slot = 0; slot < 3; slot = slot + 1) begin
                if (
                    compiled_event_valid[slot]
                    && (!FILTER_FCSR_EVENTS || in_candidate_valid[slot])
                ) begin
                    event_y[event_count] =
                        compiled_event_y[slot*Y_W +: Y_W];
                    event_x[event_count] =
                        compiled_event_x[slot*X_W +: X_W];
                    event_id[event_count] = make_source_id(
                        plane_q,
                        compiled_event_y[slot*Y_W +: Y_W],
                        compiled_event_x[slot*X_W +: X_W]
                    );
                    event_count = event_count + 3'd1;
                end
            end
`elsif QFIT_USE_COMPILED_CROSS_R2
            for (int slot = 0; slot < 3; slot = slot + 1) begin
                if (
                    compiled_event_valid[slot]
                    && (!FILTER_FCSR_EVENTS || in_candidate_valid[slot])
                ) begin
                    event_y[event_count] =
                        compiled_event_y[slot*Y_W +: Y_W];
                    event_x[event_count] =
                        compiled_event_x[slot*X_W +: X_W];
                    event_id[event_count] = make_source_id(
                        plane_q,
                        compiled_event_y[slot*Y_W +: Y_W],
                        compiled_event_x[slot*X_W +: X_W]
                    );
                    event_count = event_count + 3'd1;
                end
            end
`else
            if (
                in_y >= DILATION
                && (!FILTER_FCSR_EVENTS || in_candidate_valid[0])
            ) begin
                event_id[event_count] = make_source_id(
                    plane_q,
                    in_y - DILATION,
                    in_x
                );
                event_y[event_count] = Y_W'(in_y - DILATION);
                event_x[event_count] = in_x;
                event_count = event_count + 3'd1;
            end
            if (
                in_y >= HEIGHT - DILATION
                && in_x >= DILATION
                && (!FILTER_FCSR_EVENTS || in_candidate_valid[1])
            ) begin
                event_id[event_count] = make_source_id(
                    plane_q,
                    in_y,
                    in_x - DILATION
                );
                event_y[event_count] = in_y;
                event_x[event_count] = X_W'(in_x - DILATION);
                event_count = event_count + 3'd1;
            end
            if (
                in_y >= HEIGHT - DILATION
                && in_x >= WIDTH - DILATION
                && (!FILTER_FCSR_EVENTS || in_candidate_valid[2])
            ) begin
                event_id[event_count] = make_source_id(
                    plane_q,
                    in_y,
                    in_x
                );
                event_y[event_count] = in_y;
                event_x[event_count] = in_x;
                event_count = event_count + 3'd1;
            end
`endif
        end else if (MODE == MODE_DYNAMIC) begin
            for (int slot = 0; slot < 5; slot = slot + 1) begin
                if (completion[role_order[slot]]) begin
                    event_id[event_count] =
                        candidate_id[role_order[slot]];
                    event_y[event_count] =
                        sy[role_order[slot]];
                    event_x[event_count] =
                        sx[role_order[slot]];
                    event_count = event_count + 3'd1;
                end
            end
        end
    end

    always_comb begin
        stripe_release_count = '0;
        stripe_release_row[0] = '0;
        stripe_release_row[1] = '0;
        if (in_x == WIDTH - 1) begin
            if (in_y > 0) begin
                stripe_release_row[stripe_release_count[0]] =
                    in_y - 1'b1;
                stripe_release_count = stripe_release_count + 2'd1;
            end
            if (in_y == HEIGHT - 1) begin
                stripe_release_row[stripe_release_count[0]] = in_y;
                stripe_release_count = stripe_release_count + 2'd1;
            end
        end
    end

    assign output_slot_available = !retire_valid_q || retire_ready;
    assign stripe_emit = output_slot_available
                      && stripe_context_count_q != 0;
    assign stripe_pop = stripe_emit
                     && stripe_x_q == WIDTH - 1;
    assign stripe_row_start_block =
        (
            !STRIPE_EARLY_FILL
            && in_x == 0
            && stripe_context_count_q == 2
            && !stripe_pop
        )
        || (
            (
                stripe_context_count_q != 0
                || retire_valid_q
            )
            && 32'(in_y)
               >= 32'(
                    retire_valid_q
                    ? retire_y_q
                    : stripe_row_q[stripe_head_q]
                  )
                  + STRIPE_RING_ROWS
                  - (
                      (
                          retire_valid_q
                          ? retire_y_q
                          : stripe_row_q[stripe_head_q]
                      ) != 0
                    )
        );
    assign stripe_capacity = 3'd2
                           - {1'b0, stripe_context_count_q}
                           + {2'b00, stripe_pop};

    always_comb begin
        if (MODE == MODE_STRIPE) begin
            in_ready = !plane_start
                    && !stripe_row_start_block
                    && {1'b0, stripe_release_count}
                       <= stripe_capacity;
        end else begin
            in_ready = !plane_start
                    && pending_count_q == 0
                    && output_slot_available;
        end
    end

    assign retire_valid = retire_valid_q;
    assign retire_source_id = retire_source_q;
    assign retire_y = retire_y_q;
    assign retire_x = retire_x_q;
    assign perf_producer_stalls = stalls_q;
    assign perf_max_pending = max_pending_q;
    always_comb begin
        if (MODE == MODE_STRIPE)
            plane_idle = stripe_context_count_q == 0
                      && !retire_valid_q;
        else
            plane_idle = pending_count_q == 0
                      && !retire_valid_q;
    end

    generate
        if (MODE == MODE_STRIPE) begin : g_stripe
            always_ff @(posedge clk_core) begin
                if (rst_core || plane_start) begin
                    plane_q <= plane_id;
                    retire_valid_q <= 1'b0;
                    retire_source_q <= '0;
                    retire_y_q <= '0;
                    retire_x_q <= '0;
                    stalls_q <= '0;
                    max_pending_q <= '0;
                    stripe_row_q[0] <= '0;
                    stripe_row_q[1] <= '0;
                    stripe_head_q <= 1'b0;
                    stripe_tail_q <= 1'b0;
                    stripe_context_count_q <= '0;
                    stripe_x_q <= '0;
                end else begin
                    logic [1:0] pushes;
                    logic [2:0] next_context_count;
                    pushes = (in_valid && in_ready)
                           ? stripe_release_count : 2'd0;
                    next_context_count =
                        {1'b0, stripe_context_count_q}
                        - {2'b00, stripe_pop}
                        + {1'b0, pushes};

                    if (retire_valid_q && retire_ready)
                        retire_valid_q <= 1'b0;
                    if (stripe_emit) begin
                        retire_valid_q <= 1'b1;
                        retire_source_q <= make_source_id(
                            plane_q,
                            stripe_row_q[stripe_head_q],
                            stripe_x_q
                        );
                        retire_y_q <= stripe_row_q[stripe_head_q];
                        retire_x_q <= stripe_x_q;
                        if (stripe_pop) begin
                            stripe_head_q <= ~stripe_head_q;
                            stripe_x_q <= '0;
                        end else begin
                            stripe_x_q <= stripe_x_q + 1'b1;
                        end
                    end

                    if (pushes != 0) begin
                        stripe_row_q[stripe_tail_q]
                            <= stripe_release_row[0];
                        if (pushes == 2)
                            stripe_row_q[~stripe_tail_q]
                                <= stripe_release_row[1];
                        if (pushes == 1)
                            stripe_tail_q <= ~stripe_tail_q;
                    end
                    stripe_context_count_q
                        <= next_context_count[1:0];

                    if (in_valid && !in_ready)
                        stalls_q <= stalls_q + 32'd1;
                    if (
                        next_context_count
                        > max_pending_q
                    )
                        max_pending_q <= next_context_count;
                end
            end
        end else begin : g_frontier
            always_ff @(posedge clk_core) begin
                if (rst_core || plane_start) begin
                    plane_q <= plane_id;
                    retire_valid_q <= 1'b0;
                    retire_source_q <= '0;
                    retire_y_q <= '0;
                    retire_x_q <= '0;
                    stalls_q <= '0;
                    max_pending_q <= '0;
                    pending_count_q <= '0;
                    seen_flat_q <= '0;
                    stripe_row_q[0] <= '0;
                    stripe_row_q[1] <= '0;
                    stripe_head_q <= 1'b0;
                    stripe_tail_q <= 1'b0;
                    stripe_context_count_q <= '0;
                    stripe_x_q <= '0;
                    for (int slot = 0; slot < 4; slot = slot + 1)
                        pending_id_q[slot] <= '0;
                    for (int slot = 0; slot < 4; slot = slot + 1) begin
                        pending_y_q[slot] <= '0;
                        pending_x_q[slot] <= '0;
                    end
                end else begin
                    if (retire_valid_q && retire_ready)
                        retire_valid_q <= 1'b0;

                    if (
                        pending_count_q != 0
                        && output_slot_available
                    ) begin
                        retire_valid_q <= 1'b1;
                        retire_source_q <= pending_id_q[0];
                        retire_y_q <= pending_y_q[0];
                        retire_x_q <= pending_x_q[0];
                        for (int slot = 0; slot < 3; slot = slot + 1)
                            pending_id_q[slot]
                                <= pending_id_q[slot+1];
                        for (int slot = 0; slot < 3; slot = slot + 1) begin
                            pending_y_q[slot] <= pending_y_q[slot+1];
                            pending_x_q[slot] <= pending_x_q[slot+1];
                        end
                        pending_id_q[3] <= '0;
                        pending_y_q[3] <= '0;
                        pending_x_q[3] <= '0;
                        pending_count_q <= pending_count_q - 3'd1;
                    end else if (in_valid && in_ready) begin
                        if (event_count != 0) begin
                            retire_valid_q <= 1'b1;
                            retire_source_q <= event_id[0];
                            retire_y_q <= event_y[0];
                            retire_x_q <= event_x[0];
                            for (
                                int slot = 0;
                                slot < 4;
                                slot = slot + 1
                            ) begin
                                pending_id_q[slot]
                                    <= event_id[slot+1];
                                pending_y_q[slot]
                                    <= event_y[slot+1];
                                pending_x_q[slot]
                                    <= event_x[slot+1];
                            end
                            pending_count_q <= event_count - 3'd1;
                        end
                        if (MODE == MODE_DYNAMIC) begin
                            int sy;
                            int sx;
                            int local_source;
                            if (
                                in_x == 0
                                && in_y < HEIGHT - DILATION
                            )
                                seen_flat_q[
                                    3 * WIDTH
                                    * ((in_y + DILATION) % COUNTER_ROWS)
                                    +: 3 * WIDTH
                                ] <= '0;
                            for (
                                int role = 0;
                                role < 5;
                                role = role + 1
                            ) begin
                                sy = in_y;
                                sx = in_x;
                                case (role)
                                    1: sy = in_y - DILATION;
                                    2: sy = in_y + DILATION;
                                    3: sx = in_x - DILATION;
                                    4: sx = in_x + DILATION;
                                    default: begin end
                                endcase
                                if (in_candidate_valid[role]) begin
                                    local_source =
                                        (sy % COUNTER_ROWS) * WIDTH + sx;
                                    if (
                                        role == 2
                                        && in_x == 0
                                        && in_y < HEIGHT - DILATION
                                    )
                                        seen_flat_q[
                                            3 * local_source +: 3
                                        ] <= 3'd1;
                                    else
                                        seen_flat_q[
                                            3 * local_source +: 3
                                        ] <= seen_flat_q[
                                            3 * local_source +: 3
                                        ] + 3'd1;
                                end
                            end
                        end
                    end

                    if (in_valid && !in_ready)
                        stalls_q <= stalls_q + 32'd1;
                    if (pending_count_q > max_pending_q)
                        max_pending_q <= pending_count_q;
                    if (
                        in_valid
                        && in_ready
                        && event_count != 0
                        && event_count - 3'd1 > max_pending_q
                    )
                        max_pending_q <= event_count - 3'd1;
                end
            end
        end
    endgenerate

endmodule

`default_nettype wire
