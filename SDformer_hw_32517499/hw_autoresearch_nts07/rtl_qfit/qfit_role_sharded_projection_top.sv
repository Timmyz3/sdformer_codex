`timescale 1ns/1ps
`default_nettype none

// Five role-local partial-accumulator banks.
//
// Each role owns a complete raster-addressed destination space. One term can
// therefore update all five legal roles in one cycle without an intra-term
// bank conflict. Readback synchronously reads all role banks and reduces the
// five signed partial vectors.
module qfit_role_sharded_projection_top #(
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
    localparam int ROLE_DEPTH = TIME_PLANES * HEIGHT * WIDTH;
    localparam int ROLE_ADDR_W =
        (ROLE_DEPTH <= 1) ? 1 : $clog2(ROLE_DEPTH);
    localparam int ACC_VEC_W = OUT_DIM * ACC_W;
    localparam int WEIGHT_VEC_W = OUT_DIM * W_W;
    localparam int REDUCE_W = ACC_W + 3;

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
    logic [ROLE_ADDR_W-1:0] clear_addr_q;
    logic protocol_error_q;
    logic [31:0] perf_terms_q;
    logic [31:0] perf_updates_q;

    logic role_valid [0:4];
    logic [ROLE_ADDR_W-1:0] role_addr [0:4];
    logic role_update_valid [0:4];
    logic [ACC_VEC_W-1:0] product_vector;
    logic term_contract_valid;
    logic term_boundary_error;
    logic [31:0] term_update_count;

    logic [ACC_VEC_W-1:0] role_read_data [0:4];
    logic role_read_data_valid [0:4];
    logic role_update_idle [0:4];
    logic all_roles_idle;
    logic [ROLE_ADDR_W-1:0] read_addr;
    logic [OUT_W-1:0] read_out_q;
    logic [ACC_VEC_W-1:0] reduced_vector;
    logic signed [REDUCE_W-1:0] reduction_lane [0:OUT_DIM-1];
    logic term_fire;
    logic read_fire;
    logic window_close_fire;

    function automatic logic [ROLE_ADDR_W-1:0] raster_address(
        input logic [PLANE_W-1:0] plane,
        input logic [Y_W-1:0] y,
        input logic [X_W-1:0] x
    );
        logic [ROLE_ADDR_W-1:0] plane_base;
        logic [ROLE_ADDR_W-1:0] row_base;
        begin
            plane_base = '0;
            row_base = '0;
            for (integer p = 0; p < TIME_PLANES; p = p + 1)
                if (plane == PLANE_W'(p))
                    plane_base =
                        ROLE_ADDR_W'(p * HEIGHT * WIDTH);
            for (integer row = 0; row < HEIGHT; row = row + 1)
                if (y == Y_W'(row))
                    row_base = ROLE_ADDR_W'(row * WIDTH);
            raster_address = plane_base + row_base + ROLE_ADDR_W'(x);
        end
    endfunction

    always_comb begin
        logic [Y_W-1:0] role_y [0:4];
        logic [X_W-1:0] role_x [0:4];

        for (integer role = 0; role < 5; role = role + 1) begin
            role_y[role] = term_source_y;
            role_x[role] = term_source_x;
            role_valid[role] = 1'b1;
        end
        role_valid[1] = 32'(term_source_y) < HEIGHT - 1;
        role_valid[2] = term_source_y != '0;
        // role3/4 are LEFT/RIGHT candidates. Source-major consumers are
        // therefore east/west of the source, respectively.
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

        for (integer role = 0; role < 5; role = role + 1)
            role_addr[role] = raster_address(
                term_source_plane,
                role_y[role],
                role_x[role]
            );

        term_boundary_error = 1'b0;
        for (integer role = 0; role < 5; role = role + 1)
            if (term_destination_mask[role] && !role_valid[role])
                term_boundary_error = 1'b1;

        term_contract_valid =
            32'(term_source_plane) < TIME_PLANES
            && 32'(term_source_y) < HEIGHT
            && 32'(term_source_x) < WIDTH
            && 32'(term_lane) < HEAD_DIM
            && term_gate != '0
            && term_destination_mask != '0
            && !term_boundary_error;

        product_vector = '0;
        for (integer out = 0; out < OUT_DIM; out = out + 1) begin
            product_vector[out*ACC_W +: ACC_W] =
                ACC_W'(
                    signed'({1'b0, term_gate})
                    * signed'(selected_weight_row[out*W_W +: W_W])
                );
        end

        term_update_count = '0;
        for (integer role = 0; role < 5; role = role + 1) begin
            role_update_valid[role] =
                state_q == ST_RUN
                && term_fire
                && term_contract_valid
                && term_destination_mask[role]
                && role_valid[role];
            if (
                term_destination_mask[role]
                && role_valid[role]
            )
                term_update_count = term_update_count + 1'b1;
        end

        read_addr = raster_address(read_plane, read_y, read_x);

        reduced_vector = '0;
        for (integer out = 0; out < OUT_DIM; out = out + 1) begin
            reduction_lane[out] = '0;
            for (integer role = 0; role < 5; role = role + 1) begin
                reduction_lane[out] =
                    reduction_lane[out]
                    + REDUCE_W'(
                        signed'(
                            role_read_data[role][
                                out*ACC_W +: ACC_W
                            ]
                        )
                    );
            end
            reduced_vector[out*ACC_W +: ACC_W] =
                reduction_lane[out][ACC_W-1:0];
        end
    end

    assign selected_weight_row = weight_row_q[term_lane];
    assign weight_ready = state_q == ST_LOAD;
    assign term_ready = state_q == ST_RUN;
    assign term_fire = term_valid && term_ready;
    assign run_busy =
        state_q == ST_CLEAR
        || state_q == ST_RUN
        || state_q == ST_DRAIN;
    assign run_done = state_q == ST_DONE;
    assign window_close_ready = state_q == ST_RUN && !term_valid;
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
        role_read_data_valid[0]
        && role_read_data_valid[1]
        && role_read_data_valid[2]
        && role_read_data_valid[3]
        && role_read_data_valid[4];
    assign read_data = read_data_valid
        ? reduced_vector[read_out_q*ACC_W +: ACC_W]
        : '0;
    assign protocol_error = protocol_error_q;
    assign perf_product_terms = perf_terms_q;
    assign perf_destination_updates = perf_updates_q;
    assign all_roles_idle =
        role_update_idle[0]
        && role_update_idle[1]
        && role_update_idle[2]
        && role_update_idle[3]
        && role_update_idle[4];

    generate
        for (genvar role = 0; role < 5; role = role + 1) begin : g_role
            qfit_tcfm5_acc_bank #(
                .DEPTH(ROLE_DEPTH),
                .OUT_DIM(OUT_DIM),
                .ACC_W(ACC_W)
            ) u_partial_acc (
                .clk_core(clk_core),
                .rst_core(rst_core),
                .clear_valid(state_q == ST_CLEAR),
                .clear_addr(clear_addr_q),
                .update_valid(role_update_valid[role]),
                .update_addr(role_addr[role]),
                .update_delta(product_vector),
                .update_idle(role_update_idle[role]),
                .read_valid(read_fire),
                .read_addr(read_addr),
                .read_data_valid(role_read_data_valid[role]),
                .read_data(role_read_data[role])
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
            read_out_q <= '0;
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
                        state_q <= ST_CLEAR;
                    end
                end

                ST_CLEAR: begin
                    if (
                        clear_addr_q
                        == ROLE_ADDR_W'(ROLE_DEPTH - 1)
                    ) begin
                        state_q <= ST_RUN;
                    end else begin
                        clear_addr_q <= clear_addr_q + 1'b1;
                    end
                end

                ST_RUN: begin
                    if (term_fire) begin
                        if (!term_contract_valid) begin
                            protocol_error_q <= 1'b1;
                        end else begin
                            perf_terms_q <= perf_terms_q + 1'b1;
                            perf_updates_q <=
                                perf_updates_q + term_update_count;
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
                    if (all_roles_idle)
                        state_q <= ST_DONE;
                end

                ST_DONE: begin
                    if (read_fire)
                        read_out_q <= read_out;
                    if (read_valid && !read_ready)
                        protocol_error_q <= 1'b1;
                    if (run_start && weights_loaded_q) begin
                        clear_addr_q <= '0;
                        protocol_error_q <= 1'b0;
                        perf_terms_q <= '0;
                        perf_updates_q <= '0;
                        state_q <= ST_CLEAR;
                    end
                end

                default: state_q <= ST_LOAD;
            endcase
        end
    end
endmodule

`default_nettype wire
