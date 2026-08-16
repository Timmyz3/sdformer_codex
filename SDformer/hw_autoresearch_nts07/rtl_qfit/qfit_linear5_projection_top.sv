`timescale 1ns/1ps
`default_nettype none

// Linear-5 exact-replay projection backend.
//
// Tokens use raster banking:
//   global_id = plane * HEIGHT * WIDTH + y * WIDTH + x
//   bank      = global_id % 5
//   address   = global_id / 5
//
// One accepted term computes one packed product vector. At most one selected
// destination is issued to each bank on the acceptance cycle. Destinations
// that collide in a bank are retained with the product and replayed exactly.
module qfit_linear5_projection_top #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int HEAD_DIM = 32,
    parameter int OUT_DIM = 4,
    parameter int GATE_W = 9,
    parameter int W_W = 8,
    parameter int ACC_W = 32,
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int PLANE_W =
        (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int LANE_W =
        (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM),
    parameter int OUT_W =
        (OUT_DIM <= 1) ? 1 : $clog2(OUT_DIM)
) (
    input  logic                       clk_core,
    input  logic                       rst_core,

    input  logic                       weight_valid,
    output logic                       weight_ready,
    input  logic [LANE_W-1:0]          weight_lane,
    input  logic [OUT_W-1:0]           weight_out,
    input  logic signed [W_W-1:0]      weight_data,
    input  logic                       weight_last,

    input  logic                       run_start,
    output logic                       run_busy,
    output logic                       run_done,

    input  logic                       term_valid,
    output logic                       term_ready,
    input  logic [PLANE_W-1:0]         term_source_plane,
    input  logic [Y_W-1:0]             term_source_y,
    input  logic [X_W-1:0]             term_source_x,
    input  logic [LANE_W-1:0]          term_lane,
    input  logic [GATE_W-1:0]          term_gate,
    input  logic [4:0]                 term_destination_mask,
    input  logic                       term_window_last,
    input  logic                       window_close,
    output logic                       window_close_ready,

    input  logic                       read_valid,
    output logic                       read_ready,
    input  logic [PLANE_W-1:0]         read_plane,
    input  logic [Y_W-1:0]             read_y,
    input  logic [X_W-1:0]             read_x,
    input  logic [OUT_W-1:0]           read_out,
    output logic                       read_data_valid,
    output logic signed [ACC_W-1:0]    read_data,

    output logic                       protocol_error,
    output logic [31:0]                perf_product_terms,
    output logic [31:0]                perf_destination_updates
);
    localparam int TOTAL_TOKENS = TIME_PLANES * HEIGHT * WIDTH;
    localparam int BANK_DEPTH = (TOTAL_TOKENS + 4) / 5;
    localparam int BANK_ADDR_W =
        (BANK_DEPTH <= 1) ? 1 : $clog2(BANK_DEPTH);
    localparam int ACC_VEC_W = OUT_DIM * ACC_W;
    localparam int WEIGHT_VEC_W = OUT_DIM * W_W;

    typedef enum logic [2:0] {
        ST_LOAD = 3'd0,
        ST_CLEAR = 3'd1,
        ST_RUN = 3'd2,
        ST_DRAIN = 3'd3,
        ST_DONE = 3'd4
    } state_t;

    state_t state_q;
    logic [WEIGHT_VEC_W-1:0] weight_row_q [0:HEAD_DIM-1];
    logic [WEIGHT_VEC_W-1:0] selected_weight_row;
    logic weights_loaded_q;
    logic [BANK_ADDR_W-1:0] clear_addr_q;
    logic protocol_error_q;
    logic [31:0] perf_terms_q;
    logic [31:0] perf_updates_q;

    logic role_valid [0:4];
    logic [2:0] role_bank [0:4];
    logic [BANK_ADDR_W-1:0] role_addr [0:4];
    logic [4:0] live_valid_mask;
    logic [4:0] live_selected_mask;
    logic [4:0] live_remaining_mask;
    logic live_contract_valid;
    logic term_boundary_error;
    logic [ACC_VEC_W-1:0] live_product_vector;

    logic replay_valid_q;
    logic [4:0] replay_remaining_mask_q;
    logic [2:0] replay_bank_q [0:4];
    logic [BANK_ADDR_W-1:0] replay_addr_q [0:4];
    logic [ACC_VEC_W-1:0] replay_product_q;
    logic replay_last_q;
    logic close_pending_q;
    logic [4:0] replay_issue_mask;
    logic [4:0] replay_next_remaining;

    logic bank_update_valid [0:4];
    logic [BANK_ADDR_W-1:0] bank_update_addr [0:4];
    logic [ACC_VEC_W-1:0] bank_update_delta;
    logic [ACC_VEC_W-1:0] bank_read_data [0:4];
    logic bank_read_data_valid [0:4];
    logic bank_update_idle [0:4];
    logic all_banks_idle;

    logic [2:0] read_bank;
    logic [BANK_ADDR_W-1:0] read_addr;
    logic [2:0] read_bank_q;
    logic [OUT_W-1:0] read_out_q;
    logic read_fire;
    logic term_fire;
    logic window_close_fire;
    logic [31:0] live_issue_count;
    logic [31:0] replay_issue_count;

    function automatic logic [2:0] x_mod5_lut(
        input logic [X_W-1:0] x
    );
        logic [2:0] remainder;
        begin
            x_mod5_lut = '0;
            remainder = '0;
            for (integer col = 0; col < WIDTH; col = col + 1) begin
                if (x == X_W'(col))
                    x_mod5_lut = remainder;
                remainder = (remainder == 3'd4)
                    ? 3'd0
                    : remainder + 1'b1;
            end
        end
    endfunction

    function automatic logic [BANK_ADDR_W-1:0] x_div5_lut(
        input logic [X_W-1:0] x
    );
        logic [BANK_ADDR_W-1:0] quotient;
        logic [2:0] remainder;
        begin
            x_div5_lut = '0;
            quotient = '0;
            remainder = '0;
            for (integer col = 0; col < WIDTH; col = col + 1) begin
                if (x == X_W'(col))
                    x_div5_lut = quotient;
                if (remainder == 3'd4) begin
                    remainder = '0;
                    quotient = quotient + 1'b1;
                end else begin
                    remainder = remainder + 1'b1;
                end
            end
        end
    endfunction

    function automatic logic [2:0] row_base_mod5_lut(
        input logic [PLANE_W-1:0] plane,
        input logic [Y_W-1:0] y
    );
        begin
            row_base_mod5_lut = '0;
            for (integer p = 0; p < TIME_PLANES; p = p + 1)
                for (integer row = 0; row < HEIGHT; row = row + 1)
                    if (
                        plane == PLANE_W'(p)
                        && y == Y_W'(row)
                    )
                        row_base_mod5_lut = 3'(
                            (p * HEIGHT * WIDTH + row * WIDTH) % 5
                        );
        end
    endfunction

    function automatic logic [BANK_ADDR_W-1:0] row_base_div5_lut(
        input logic [PLANE_W-1:0] plane,
        input logic [Y_W-1:0] y
    );
        begin
            row_base_div5_lut = '0;
            for (integer p = 0; p < TIME_PLANES; p = p + 1)
                for (integer row = 0; row < HEIGHT; row = row + 1)
                    if (
                        plane == PLANE_W'(p)
                        && y == Y_W'(row)
                    )
                        row_base_div5_lut = BANK_ADDR_W'(
                            (p * HEIGHT * WIDTH + row * WIDTH) / 5
                        );
        end
    endfunction

    function automatic logic [2:0] raster_bank_lut(
        input logic [PLANE_W-1:0] plane,
        input logic [Y_W-1:0] y,
        input logic [X_W-1:0] x
    );
        logic [3:0] sum;
        begin
            sum = {1'b0, row_base_mod5_lut(plane, y)}
                + {1'b0, x_mod5_lut(x)};
            raster_bank_lut = (sum >= 4'd5)
                ? 3'(sum - 4'd5)
                : sum[2:0];
        end
    endfunction

    function automatic logic [BANK_ADDR_W-1:0] raster_addr_lut(
        input logic [PLANE_W-1:0] plane,
        input logic [Y_W-1:0] y,
        input logic [X_W-1:0] x
    );
        logic [3:0] sum;
        begin
            sum = {1'b0, row_base_mod5_lut(plane, y)}
                + {1'b0, x_mod5_lut(x)};
            raster_addr_lut =
                row_base_div5_lut(plane, y)
                + x_div5_lut(x)
                + BANK_ADDR_W'(sum >= 4'd5);
        end
    endfunction

    always_comb begin
        logic [Y_W-1:0] role_y [0:4];
        logic [X_W-1:0] role_x [0:4];
        logic bank_selected;

        for (integer role = 0; role < 5; role = role + 1) begin
            role_y[role] = term_source_y;
            role_x[role] = term_source_x;
            role_valid[role] = 1'b1;
        end
        role_valid[1] = 32'(term_source_y) < HEIGHT - 1;
        role_valid[2] = term_source_y != '0;
        role_valid[3] = 32'(term_source_x) < WIDTH - 1;
        role_valid[4] = term_source_x != '0;
        if (role_valid[1])
            role_y[1] = term_source_y + 1'b1;
        if (role_valid[2])
            role_y[2] = term_source_y - 1'b1;
        if (role_valid[3])
            role_x[3] = term_source_x + 1'b1;
        if (role_valid[4])
            role_x[4] = term_source_x - 1'b1;

        live_valid_mask = '0;
        for (integer role = 0; role < 5; role = role + 1) begin
            role_bank[role] = raster_bank_lut(
                term_source_plane,
                role_y[role],
                role_x[role]
            );
            role_addr[role] = raster_addr_lut(
                term_source_plane,
                role_y[role],
                role_x[role]
            );
            live_valid_mask[role] =
                role_valid[role] && term_destination_mask[role];
        end

        term_boundary_error = 1'b0;
        live_selected_mask = '0;
        for (integer role = 0; role < 5; role = role + 1)
            if (term_destination_mask[role] && !role_valid[role])
                term_boundary_error = 1'b1;
        for (integer bank = 0; bank < 5; bank = bank + 1) begin
            bank_selected = 1'b0;
            for (integer role = 0; role < 5; role = role + 1) begin
                if (
                    live_valid_mask[role]
                    && role_bank[role] == 3'(bank)
                    && !bank_selected
                ) begin
                    live_selected_mask[role] = 1'b1;
                    bank_selected = 1'b1;
                end
            end
        end
        live_remaining_mask =
            live_valid_mask & ~live_selected_mask;

        live_contract_valid =
            32'(term_source_plane) < TIME_PLANES
            && 32'(term_source_y) < HEIGHT
            && 32'(term_source_x) < WIDTH
            && 32'(term_lane) < HEAD_DIM
            && term_gate != '0
            && term_destination_mask != '0
            && !term_boundary_error;

        live_product_vector = '0;
        for (integer out = 0; out < OUT_DIM; out = out + 1)
            live_product_vector[out*ACC_W +: ACC_W] =
                ACC_W'(
                    signed'({1'b0, term_gate})
                    * signed'(selected_weight_row[out*W_W +: W_W])
                );

        replay_issue_mask = '0;
        for (integer bank = 0; bank < 5; bank = bank + 1) begin
            bank_selected = 1'b0;
            for (integer role = 0; role < 5; role = role + 1) begin
                if (
                    replay_remaining_mask_q[role]
                    && replay_bank_q[role] == 3'(bank)
                    && !bank_selected
                ) begin
                    replay_issue_mask[role] = 1'b1;
                    bank_selected = 1'b1;
                end
            end
        end
        replay_next_remaining =
            replay_remaining_mask_q & ~replay_issue_mask;

        for (integer bank = 0; bank < 5; bank = bank + 1) begin
            bank_update_valid[bank] = 1'b0;
            bank_update_addr[bank] = '0;
            if (state_q == ST_RUN && replay_valid_q) begin
                for (integer role = 0; role < 5; role = role + 1) begin
                    if (
                        replay_issue_mask[role]
                        && replay_bank_q[role] == 3'(bank)
                    ) begin
                        bank_update_valid[bank] = 1'b1;
                        bank_update_addr[bank] = replay_addr_q[role];
                    end
                end
            end else if (
                state_q == ST_RUN
                && term_fire
                && live_contract_valid
            ) begin
                for (integer role = 0; role < 5; role = role + 1) begin
                    if (
                        live_selected_mask[role]
                        && role_bank[role] == 3'(bank)
                    ) begin
                        bank_update_valid[bank] = 1'b1;
                        bank_update_addr[bank] = role_addr[role];
                    end
                end
            end
        end
        bank_update_delta =
            replay_valid_q ? replay_product_q : live_product_vector;

        live_issue_count = '0;
        replay_issue_count = '0;
        for (integer role = 0; role < 5; role = role + 1) begin
            if (live_selected_mask[role])
                live_issue_count = live_issue_count + 1'b1;
            if (replay_issue_mask[role])
                replay_issue_count = replay_issue_count + 1'b1;
        end

        read_bank = raster_bank_lut(read_plane, read_y, read_x);
        read_addr = raster_addr_lut(read_plane, read_y, read_x);
    end

    assign selected_weight_row = weight_row_q[term_lane];
    assign weight_ready = state_q == ST_LOAD;
    assign term_ready =
        state_q == ST_RUN
        && !replay_valid_q
        && !close_pending_q;
    assign term_fire = term_valid && term_ready;
    assign run_busy =
        state_q == ST_CLEAR
        || state_q == ST_RUN
        || state_q == ST_DRAIN;
    assign run_done = state_q == ST_DONE;
    assign window_close_ready =
        state_q == ST_RUN
        && !term_valid
        && !close_pending_q;
    assign window_close_fire =
        window_close && window_close_ready;
    assign read_ready =
        state_q == ST_DONE
        && 32'(read_plane) < TIME_PLANES
        && 32'(read_y) < HEIGHT
        && 32'(read_x) < WIDTH
        && 32'(read_out) < OUT_DIM;
    assign read_fire = read_valid && read_ready;
    assign read_data_valid =
        bank_read_data_valid[0]
        || bank_read_data_valid[1]
        || bank_read_data_valid[2]
        || bank_read_data_valid[3]
        || bank_read_data_valid[4];
    assign read_data = read_data_valid
        ? bank_read_data[read_bank_q][read_out_q*ACC_W +: ACC_W]
        : '0;
    assign protocol_error = protocol_error_q;
    assign perf_product_terms = perf_terms_q;
    assign perf_destination_updates = perf_updates_q;
    assign all_banks_idle =
        bank_update_idle[0]
        && bank_update_idle[1]
        && bank_update_idle[2]
        && bank_update_idle[3]
        && bank_update_idle[4];

    generate
        for (genvar bank = 0; bank < 5; bank = bank + 1) begin : g_acc
            qfit_tcfm5_acc_bank #(
                .DEPTH(BANK_DEPTH),
                .OUT_DIM(OUT_DIM),
                .ACC_W(ACC_W)
            ) u_acc_bank (
                .clk_core(clk_core),
                .rst_core(rst_core),
                .clear_valid(state_q == ST_CLEAR),
                .clear_addr(clear_addr_q),
                .update_valid(bank_update_valid[bank]),
                .update_addr(bank_update_addr[bank]),
                .update_delta(bank_update_delta),
                .update_idle(bank_update_idle[bank]),
                .read_valid(read_fire && read_bank == 3'(bank)),
                .read_addr(read_addr),
                .read_data_valid(bank_read_data_valid[bank]),
                .read_data(bank_read_data[bank])
            );
        end
    endgenerate

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_LOAD;
            weights_loaded_q <= 1'b0;
            clear_addr_q <= '0;
            protocol_error_q <= 1'b0;
            perf_terms_q <= '0;
            perf_updates_q <= '0;
            replay_valid_q <= 1'b0;
            replay_remaining_mask_q <= '0;
            replay_product_q <= '0;
            replay_last_q <= 1'b0;
            close_pending_q <= 1'b0;
            read_bank_q <= '0;
            read_out_q <= '0;
            for (int role = 0; role < 5; role = role + 1) begin
                replay_bank_q[role] <= '0;
                replay_addr_q[role] <= '0;
            end
        end else begin
            case (state_q)
                ST_LOAD: begin
                    if (weight_valid && weight_ready) begin
                        weight_row_q[weight_lane][weight_out*W_W +: W_W]
                            <= weight_data;
                        if (weight_last)
                            weights_loaded_q <= 1'b1;
                    end
                    if (run_start && weights_loaded_q) begin
                        clear_addr_q <= '0;
                        protocol_error_q <= 1'b0;
                        perf_terms_q <= '0;
                        perf_updates_q <= '0;
                        replay_valid_q <= 1'b0;
                        replay_remaining_mask_q <= '0;
                        replay_last_q <= 1'b0;
                        close_pending_q <= 1'b0;
                        state_q <= ST_CLEAR;
                    end
                end

                ST_CLEAR: begin
                    if (
                        clear_addr_q
                        == BANK_ADDR_W'(BANK_DEPTH - 1)
                    ) begin
                        state_q <= ST_RUN;
                    end else begin
                        clear_addr_q <= clear_addr_q + 1'b1;
                    end
                end

                ST_RUN: begin
                    if (window_close_fire)
                        close_pending_q <= 1'b1;

                    if (replay_valid_q) begin
                        perf_updates_q <=
                            perf_updates_q + replay_issue_count;
                        replay_remaining_mask_q <=
                            replay_next_remaining;
                        if (replay_next_remaining == '0) begin
                            replay_valid_q <= 1'b0;
                            if (
                                replay_last_q
                                || close_pending_q
                                || window_close_fire
                            ) begin
                                close_pending_q <= 1'b0;
                                state_q <= ST_DRAIN;
                            end
                        end
                    end else if (term_fire) begin
                        if (!live_contract_valid) begin
                            protocol_error_q <= 1'b1;
                            if (term_window_last)
                                state_q <= ST_DRAIN;
                        end else begin
                            perf_terms_q <= perf_terms_q + 1'b1;
                            perf_updates_q <=
                                perf_updates_q + live_issue_count;
                            if (live_remaining_mask != '0) begin
                                replay_valid_q <= 1'b1;
                                replay_remaining_mask_q <=
                                    live_remaining_mask;
                                replay_product_q <=
                                    live_product_vector;
                                replay_last_q <= term_window_last;
                                for (
                                    int role = 0;
                                    role < 5;
                                    role = role + 1
                                ) begin
                                    replay_bank_q[role] <= role_bank[role];
                                    replay_addr_q[role] <= role_addr[role];
                                end
                            end else if (term_window_last) begin
                                state_q <= ST_DRAIN;
                            end
                        end
                    end else if (window_close_fire) begin
                        close_pending_q <= 1'b0;
                        state_q <= ST_DRAIN;
                    end

                    if (window_close && !window_close_ready)
                        protocol_error_q <= 1'b1;
                end

                ST_DRAIN: begin
                    if (all_banks_idle)
                        state_q <= ST_DONE;
                end

                ST_DONE: begin
                    if (read_fire) begin
                        read_bank_q <= read_bank;
                        read_out_q <= read_out;
                    end
                    if (read_valid && !read_ready)
                        protocol_error_q <= 1'b1;
                    if (run_start && weights_loaded_q) begin
                        clear_addr_q <= '0;
                        protocol_error_q <= 1'b0;
                        perf_terms_q <= '0;
                        perf_updates_q <= '0;
                        replay_valid_q <= 1'b0;
                        replay_remaining_mask_q <= '0;
                        replay_last_q <= 1'b0;
                        close_pending_q <= 1'b0;
                        state_q <= ST_CLEAR;
                    end
                end

                default: state_q <= ST_LOAD;
            endcase
        end
    end
endmodule

`default_nettype wire
