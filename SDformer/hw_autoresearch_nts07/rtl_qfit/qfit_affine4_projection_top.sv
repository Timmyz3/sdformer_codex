`timescale 1ns/1ps
`default_nettype none

// Four-bank affine-colored exact-replay projection baseline.
//
// bank = (x + 2*y) mod 4
// addr = plane*HEIGHT*ceil(WIDTH/4)
//      + y*ceil(WIDTH/4) + floor(x/4)
//
// Self, west, and east occupy distinct banks. North and south share one bank.
// If both are selected, south is issued with the primary term and north is
// retained as an exact product/address replay for the following cycle.
module qfit_affine4_projection_top #(
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
    output logic [31:0]                perf_destination_updates,
    output logic [31:0]                perf_replay_updates
);
    localparam int BANKS = 4;
    localparam int X_GROUPS = (WIDTH + 3) / 4;
    localparam int PLANE_BANK_DEPTH = HEIGHT * X_GROUPS;
    localparam int BANK_DEPTH = TIME_PLANES * PLANE_BANK_DEPTH;
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
    logic [31:0] perf_replays_q;

    logic role_valid [0:4];
    logic [Y_W-1:0] role_y [0:4];
    logic [X_W-1:0] role_x [0:4];
    logic [1:0] role_bank [0:4];
    logic [BANK_ADDR_W-1:0] role_addr [0:4];
    logic [2:0] term_valid_update_count;
    logic term_boundary_error;
    logic term_conflict;
    logic term_contract_ok;
    logic term_fire;
    logic term_update_fire;
    logic [ACC_VEC_W-1:0] product_vector;

    logic [3:0] primary_update_vec;
    logic [BANK_ADDR_W-1:0] primary_update_addr [0:BANKS-1];
    logic [3:0] bank_update_valid_vec;
    logic [BANK_ADDR_W-1:0] bank_update_addr [0:BANKS-1];
    logic [ACC_VEC_W-1:0] bank_update_delta [0:BANKS-1];

    logic replay_valid_q;
    logic [1:0] replay_bank_q;
    logic [BANK_ADDR_W-1:0] replay_addr_q;
    logic [ACC_VEC_W-1:0] replay_product_q;
    logic replay_issue;

    logic [ACC_VEC_W-1:0] bank_read_data [0:BANKS-1];
    logic bank_read_data_valid [0:BANKS-1];
    logic bank_update_idle [0:BANKS-1];
    logic all_banks_idle;
    logic [1:0] read_bank;
    logic [BANK_ADDR_W-1:0] read_addr;
    logic [1:0] read_bank_q;
    logic [OUT_W-1:0] read_out_q;
    logic read_fire;
    logic window_close_fire;

    function automatic logic [1:0] affine_bank(
        input logic [Y_W-1:0] y,
        input logic [X_W-1:0] x
    );
        logic [31:0] affine_sum;
        begin
            affine_sum = 32'(x) + (32'(y) << 1);
            affine_bank = 2'(affine_sum % 32'd4);
        end
    endfunction

    function automatic logic [BANK_ADDR_W-1:0] x_group_lut(
        input logic [X_W-1:0] x
    );
        logic [BANK_ADDR_W-1:0] group;
        logic [1:0] color;
        begin
            x_group_lut = '0;
            group = '0;
            color = '0;
            for (integer col = 0; col < WIDTH; col = col + 1) begin
                if (x == X_W'(col))
                    x_group_lut = group;
                if (color == 2'd3) begin
                    color = '0;
                    group = group + 1'b1;
                end else begin
                    color = color + 1'b1;
                end
            end
        end
    endfunction

    function automatic logic [BANK_ADDR_W-1:0] bank_address(
        input logic [PLANE_W-1:0] plane,
        input logic [Y_W-1:0] y,
        input logic [X_W-1:0] x
    );
        logic [BANK_ADDR_W-1:0] plane_base;
        logic [BANK_ADDR_W-1:0] row_base;
        begin
            plane_base = '0;
            row_base = '0;
            for (integer p = 0; p < TIME_PLANES; p = p + 1)
                if (plane == PLANE_W'(p))
                    plane_base = BANK_ADDR_W'(p * PLANE_BANK_DEPTH);
            for (integer row = 0; row < HEIGHT; row = row + 1)
                if (y == Y_W'(row))
                    row_base = BANK_ADDR_W'(row * X_GROUPS);
            bank_address = plane_base + row_base + x_group_lut(x);
        end
    endfunction

    always_comb begin
        role_y[0] = term_source_y;
        role_y[1] = term_source_y;
        role_y[2] = term_source_y;
        role_y[3] = term_source_y;
        role_y[4] = term_source_y;
        role_x[0] = term_source_x;
        role_x[1] = term_source_x;
        role_x[2] = term_source_x;
        role_x[3] = term_source_x;
        role_x[4] = term_source_x;
        role_valid[0] = 1'b1;
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

        for (integer role = 0; role < 5; role = role + 1) begin
            role_bank[role] = affine_bank(role_y[role], role_x[role]);
            role_addr[role] = bank_address(
                term_source_plane,
                role_y[role],
                role_x[role]
            );
        end

        term_conflict =
            term_destination_mask[1]
            && role_valid[1]
            && term_destination_mask[2]
            && role_valid[2];
        term_valid_update_count = '0;
        term_boundary_error = 1'b0;
        primary_update_vec = '0;
        for (integer bank = 0; bank < BANKS; bank = bank + 1)
            primary_update_addr[bank] = '0;

        for (integer role = 0; role < 5; role = role + 1) begin
            if (term_destination_mask[role] && !role_valid[role])
                term_boundary_error = 1'b1;
            if (term_destination_mask[role] && role_valid[role]) begin
                term_valid_update_count =
                    term_valid_update_count + 1'b1;
                // South is primary; north is the sole possible replay.
                if (!(term_conflict && role == 2)) begin
                    primary_update_vec[role_bank[role]] = 1'b1;
                    primary_update_addr[role_bank[role]] =
                        role_addr[role];
                end
            end
        end

        product_vector = '0;
        for (integer out = 0; out < OUT_DIM; out = out + 1)
            product_vector[out*ACC_W +: ACC_W] =
                ACC_W'(
                    signed'({1'b0, term_gate})
                    * signed'(selected_weight_row[out*W_W +: W_W])
                );
    end

    assign selected_weight_row = weight_row_q[term_lane];
    assign term_contract_ok =
        32'(term_source_plane) < TIME_PLANES
        && 32'(term_source_y) < HEIGHT
        && 32'(term_source_x) < WIDTH
        && 32'(term_lane) < HEAD_DIM
        && term_gate != '0
        && term_destination_mask != '0
        && !term_boundary_error;
    assign term_ready = state_q == ST_RUN && !replay_valid_q;
    assign term_fire = term_valid && term_ready;
    assign term_update_fire = term_fire && term_contract_ok;
    assign replay_issue =
        replay_valid_q && (state_q == ST_RUN || state_q == ST_DRAIN);

    always_comb begin
        bank_update_valid_vec = '0;
        for (integer bank = 0; bank < BANKS; bank = bank + 1) begin
            bank_update_addr[bank] = '0;
            bank_update_delta[bank] = '0;
            if (replay_issue && replay_bank_q == 2'(bank)) begin
                bank_update_valid_vec[bank] = 1'b1;
                bank_update_addr[bank] = replay_addr_q;
                bank_update_delta[bank] = replay_product_q;
            end else if (
                term_update_fire
                && primary_update_vec[bank]
            ) begin
                bank_update_valid_vec[bank] = 1'b1;
                bank_update_addr[bank] = primary_update_addr[bank];
                bank_update_delta[bank] = product_vector;
            end
        end
    end

    assign weight_ready = state_q == ST_LOAD;
    assign run_busy =
        state_q == ST_CLEAR || state_q == ST_RUN || state_q == ST_DRAIN;
    assign run_done = state_q == ST_DONE;
    assign window_close_ready =
        state_q == ST_RUN && !term_valid && !replay_valid_q;
    assign window_close_fire = window_close && window_close_ready;

    assign read_bank = affine_bank(read_y, read_x);
    assign read_addr = bank_address(read_plane, read_y, read_x);
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
        || bank_read_data_valid[3];
    assign read_data = read_data_valid
        ? bank_read_data[read_bank_q][read_out_q*ACC_W +: ACC_W]
        : '0;

    assign protocol_error = protocol_error_q;
    assign perf_product_terms = perf_terms_q;
    assign perf_destination_updates = perf_updates_q;
    assign perf_replay_updates = perf_replays_q;
    assign all_banks_idle =
        bank_update_idle[0]
        && bank_update_idle[1]
        && bank_update_idle[2]
        && bank_update_idle[3];

    generate
        for (genvar bank = 0; bank < BANKS; bank = bank + 1) begin : g_acc
            qfit_tcfm5_acc_bank #(
                .DEPTH(BANK_DEPTH),
                .OUT_DIM(OUT_DIM),
                .ACC_W(ACC_W)
            ) u_acc_bank (
                .clk_core(clk_core),
                .rst_core(rst_core),
                .clear_valid(state_q == ST_CLEAR),
                .clear_addr(clear_addr_q),
                .update_valid(bank_update_valid_vec[bank]),
                .update_addr(bank_update_addr[bank]),
                .update_delta(bank_update_delta[bank]),
                .update_idle(bank_update_idle[bank]),
                .read_valid(read_fire && read_bank == 2'(bank)),
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
            perf_replays_q <= '0;
            replay_valid_q <= 1'b0;
            replay_bank_q <= '0;
            replay_addr_q <= '0;
            replay_product_q <= '0;
            read_bank_q <= '0;
            read_out_q <= '0;
        end else begin
            if (replay_issue)
                replay_valid_q <= 1'b0;

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
                        perf_replays_q <= '0;
                        replay_valid_q <= 1'b0;
                        state_q <= ST_CLEAR;
                    end
                end

                ST_CLEAR: begin
                    if (clear_addr_q == BANK_ADDR_W'(BANK_DEPTH - 1)) begin
                        state_q <= ST_RUN;
                    end else begin
                        clear_addr_q <= clear_addr_q + 1'b1;
                    end
                end

                ST_RUN: begin
                    if (term_fire) begin
                        if (!term_contract_ok) begin
                            protocol_error_q <= 1'b1;
                        end else begin
                            perf_terms_q <= perf_terms_q + 1'b1;
                            perf_updates_q <=
                                perf_updates_q
                                + 32'(term_valid_update_count);
                            if (term_conflict) begin
                                replay_valid_q <= 1'b1;
                                replay_bank_q <= role_bank[2];
                                replay_addr_q <= role_addr[2];
                                replay_product_q <= product_vector;
                                perf_replays_q <= perf_replays_q + 1'b1;
                            end
                        end
                        if (term_window_last)
                            state_q <= ST_DRAIN;
                    end else if (window_close_fire) begin
                        state_q <= ST_DRAIN;
                    end
                    if (window_close && !window_close_ready)
                        protocol_error_q <= 1'b1;
                end

                ST_DRAIN: begin
                    if (!replay_valid_q && all_banks_idle)
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
                        perf_replays_q <= '0;
                        replay_valid_q <= 1'b0;
                        state_q <= ST_CLEAR;
                    end
                end

                default: begin
                    state_q <= ST_LOAD;
                    replay_valid_q <= 1'b0;
                end
            endcase
        end
    end
endmodule

`default_nettype wire
