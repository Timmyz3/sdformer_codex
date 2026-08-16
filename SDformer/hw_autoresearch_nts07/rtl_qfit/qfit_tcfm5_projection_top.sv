`timescale 1ns/1ps
`default_nettype none

// Topology-Colored Five-bank Multicast (TCFM-5) projection backend.
//
// For bank(x,y) = (x + 2*y) mod 5, the five-point stencil
// Source consumers {self, down, up, right, left} map to five distinct banks.
// A source
// term therefore computes gate*W once and can update every selected
// destination in one cycle without an accumulator-bank conflict.
module qfit_tcfm5_projection_top #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int HEAD_DIM = 32,
    parameter int OUT_DIM = 4,
    parameter int GATE_W = 9,
    parameter int W_W = 8,
    parameter int ACC_W = 32,
    parameter bit ENABLE_VECTOR_READ = 1'b0,
    parameter bit USE_PRECOMPUTED_PRODUCT = 1'b0,
    // 0: inferred synchronous 1R1W bank; 1: legal single-port 1RW bank.
    parameter int ACC_BACKEND_KIND = 0,
    parameter int ACC_MEMORY_IMPL = 0,
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
    input  logic                       weight_context_release,
    output logic                       weight_context_release_ready,

    input  logic                       run_start,
    input  logic                       run_accumulate,
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
    input  logic [OUT_DIM*ACC_W-1:0]   term_product,
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

    input  logic                       vector_read_valid,
    output logic                       vector_read_ready,
    input  logic [PLANE_W-1:0]         vector_read_plane,
    input  logic [Y_W-1:0]             vector_read_y,
    input  logic [X_W-1:0]             vector_read_x,
    output logic                       vector_read_data_valid,
    output logic [OUT_DIM*ACC_W-1:0]   vector_read_data,

    output logic                       protocol_error,
    output logic [31:0]                perf_product_terms,
    output logic [31:0]                perf_destination_updates
);
    localparam int X_GROUPS = (WIDTH + 4) / 5;
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
    logic accumulator_initialized_q;
    logic [BANK_ADDR_W-1:0] clear_addr_q;
    logic protocol_error_q;
    logic [31:0] perf_terms_q;
    logic [31:0] perf_updates_q;

    logic [2:0] source_color;
    logic [2:0] source_x_color;
    logic [2:0] source_y_color;
    logic [2:0] read_x_color;
    logic [2:0] read_y_color;
    logic [2:0] bank_role [0:4];
    logic bank_write_enable [0:4];
    logic bank_role_valid [0:4];
    logic [BANK_ADDR_W-1:0] bank_write_addr [0:4];
    logic [2:0] read_bank;
    logic [BANK_ADDR_W-1:0] read_addr;
    logic [ACC_VEC_W-1:0] product_vector;
    logic [ACC_VEC_W-1:0] bank_read_data [0:4];
    logic bank_read_data_valid [0:4];
    logic bank_update_idle [0:4];
    logic bank_update_ready [0:4];
    logic bank_flush_ready [0:4];
    logic bank_flush_done [0:4];
    logic bank_read_ready [0:4];
    logic bank_protocol_error [0:4];
    logic [31:0] unused_bank_updates [0:4];
    logic [31:0] unused_bank_reads [0:4];
    logic [31:0] unused_bank_writes [0:4];
    logic [4:0] flush_seen_q;
    logic all_banks_idle;
    logic [2:0] read_bank_q;
    logic [OUT_W-1:0] read_out_q;
    logic read_vector_q;
    logic read_fire;
    logic vector_read_fire;
    logic combined_read_fire;
    logic vector_read_active;
    logic window_close_fire;
    logic term_fire;
    logic term_contract_valid;
    logic term_boundary_error;
    logic term_commit;
    logic bank_update_any;
    logic weight_context_release_fire;
    logic term_banks_ready;
    logic run_start_accepted;

    function automatic logic [2:0] x_color_lut(
        input logic [X_W-1:0] x
    );
        logic [2:0] color;
        begin
            x_color_lut = '0;
            color = '0;
            for (integer col = 0; col < WIDTH; col = col + 1) begin
                if (x == X_W'(col))
                    x_color_lut = color;
                color = (color == 3'd4) ? 3'd0 : color + 1'b1;
            end
        end
    endfunction

    function automatic logic [BANK_ADDR_W-1:0] x_group_lut(
        input logic [X_W-1:0] x
    );
        logic [BANK_ADDR_W-1:0] group;
        logic [2:0] color;
        begin
            x_group_lut = '0;
            group = '0;
            color = '0;
            for (integer col = 0; col < WIDTH; col = col + 1) begin
                if (x == X_W'(col))
                    x_group_lut = group;
                if (color == 3'd4) begin
                    color = '0;
                    group = group + 1'b1;
                end else begin
                    color = color + 1'b1;
                end
            end
        end
    endfunction

    function automatic logic [2:0] y_color_lut(
        input logic [Y_W-1:0] y
    );
        logic [2:0] color;
        begin
            y_color_lut = '0;
            color = '0;
            for (integer row = 0; row < HEIGHT; row = row + 1) begin
                if (y == Y_W'(row))
                    y_color_lut = color;
                if (color >= 3'd3)
                    color = color - 3'd3;
                else
                    color = color + 3'd2;
            end
        end
    endfunction

    function automatic logic [2:0] add_mod5(
        input logic [2:0] lhs,
        input logic [2:0] rhs
    );
        logic [3:0] sum;
        begin
            sum = {1'b0, lhs} + {1'b0, rhs};
            if (sum >= 4'd5)
                sum = sum - 4'd5;
            add_mod5 = sum[2:0];
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
            for (
                integer p = 0;
                p < TIME_PLANES;
                p = p + 1
            )
                if (plane == PLANE_W'(p))
                    plane_base =
                        BANK_ADDR_W'(p * PLANE_BANK_DEPTH);
            for (integer row = 0; row < HEIGHT; row = row + 1)
                if (y == Y_W'(row))
                    row_base = BANK_ADDR_W'(row * X_GROUPS);
            bank_address = plane_base + row_base + x_group_lut(x);
        end
    endfunction

    always_comb begin
        logic [Y_W-1:0] role_y [0:4];
        logic [X_W-1:0] role_x [0:4];
        logic role_valid [0:4];
        source_x_color = x_color_lut(term_source_x);
        source_y_color = y_color_lut(term_source_y);
        source_color = add_mod5(source_x_color, source_y_color);

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

        case (source_color)
            3'd0: begin
                bank_role[0] = 3'd0;
                bank_role[1] = 3'd3;
                bank_role[2] = 3'd1;
                bank_role[3] = 3'd2;
                bank_role[4] = 3'd4;
            end
            3'd1: begin
                bank_role[0] = 3'd4;
                bank_role[1] = 3'd0;
                bank_role[2] = 3'd3;
                bank_role[3] = 3'd1;
                bank_role[4] = 3'd2;
            end
            3'd2: begin
                bank_role[0] = 3'd2;
                bank_role[1] = 3'd4;
                bank_role[2] = 3'd0;
                bank_role[3] = 3'd3;
                bank_role[4] = 3'd1;
            end
            3'd3: begin
                bank_role[0] = 3'd1;
                bank_role[1] = 3'd2;
                bank_role[2] = 3'd4;
                bank_role[3] = 3'd0;
                bank_role[4] = 3'd3;
            end
            default: begin
                bank_role[0] = 3'd3;
                bank_role[1] = 3'd1;
                bank_role[2] = 3'd2;
                bank_role[3] = 3'd4;
                bank_role[4] = 3'd0;
            end
        endcase
        for (integer bank = 0; bank < 5; bank = bank + 1) begin
            bank_role_valid[bank] =
                role_valid[bank_role[bank]];
            bank_write_addr[bank] = bank_address(
                term_source_plane,
                role_y[bank_role[bank]],
                role_x[bank_role[bank]]
            );
            bank_write_enable[bank] =
                term_destination_mask[bank_role[bank]]
                && bank_role_valid[bank];
        end
        read_x_color = x_color_lut(
            ENABLE_VECTOR_READ ? vector_read_x : read_x
        );
        read_y_color = y_color_lut(
            ENABLE_VECTOR_READ ? vector_read_y : read_y
        );
        read_bank = add_mod5(read_x_color, read_y_color);
        read_addr = ENABLE_VECTOR_READ
            ? bank_address(vector_read_plane, vector_read_y, vector_read_x)
            : bank_address(read_plane, read_y, read_x);
        product_vector = term_product;
        if (!USE_PRECOMPUTED_PRODUCT) begin
            product_vector = '0;
            for (integer out = 0; out < OUT_DIM; out = out + 1)
                product_vector[out*ACC_W +: ACC_W] =
                    ACC_W'(
                        signed'({1'b0, term_gate})
                        * signed'(selected_weight_row[out*W_W +: W_W])
                    );
        end
    end

    assign selected_weight_row = weight_row_q[term_lane];
    assign weight_ready = state_q == ST_LOAD;
    assign weight_context_release_ready = state_q == ST_DONE
                                        && !read_valid
                                        && !vector_read_active
                                        && !read_data_valid
                                        && !vector_read_data_valid;
    assign weight_context_release_fire = weight_context_release
                                      && weight_context_release_ready;
    always_comb begin
        term_banks_ready = 1'b1;
        for (integer bank = 0; bank < 5; bank = bank + 1)
            if (bank_write_enable[bank] && !bank_update_ready[bank])
                term_banks_ready = 1'b0;
    end
    assign term_ready = state_q == ST_RUN && term_banks_ready;
    assign term_fire = term_valid && term_ready;
    assign term_contract_valid =
        32'(term_source_plane) < TIME_PLANES
        && 32'(term_source_y) < HEIGHT
        && 32'(term_source_x) < WIDTH
        && 32'(term_lane) < HEAD_DIM
        && term_gate != '0
        && term_destination_mask != '0;
    always_comb begin
        term_boundary_error = 1'b0;
        for (integer bank = 0; bank < 5; bank = bank + 1)
            if (
                term_destination_mask[bank_role[bank]]
                && !bank_role_valid[bank]
            )
                term_boundary_error = 1'b1;
    end
    assign term_commit =
        term_fire && term_contract_valid && !term_boundary_error;
    assign bank_update_any =
        term_commit
        && (
            bank_write_enable[0]
            || bank_write_enable[1]
            || bank_write_enable[2]
            || bank_write_enable[3]
            || bank_write_enable[4]
        );
    assign run_busy = state_q == ST_CLEAR
                   || state_q == ST_RUN
                   || state_q == ST_DRAIN;
    assign run_done = state_q == ST_DONE;
    assign vector_read_active = ENABLE_VECTOR_READ && vector_read_valid;
    assign read_ready = !ENABLE_VECTOR_READ && state_q == ST_DONE
                     && 32'(read_plane) < TIME_PLANES
                     && 32'(read_y) < HEIGHT
                     && 32'(read_x) < WIDTH
                     && 32'(read_out) < OUT_DIM
                     && bank_read_ready[read_bank];
    assign read_fire = read_valid && read_ready;
    assign vector_read_ready = ENABLE_VECTOR_READ && state_q == ST_DONE
                            && !read_valid
                            && 32'(vector_read_plane) < TIME_PLANES
                            && 32'(vector_read_y) < HEIGHT
                            && 32'(vector_read_x) < WIDTH
                            && bank_read_ready[read_bank];
    assign vector_read_fire = vector_read_active && vector_read_ready;
    assign combined_read_fire = read_fire || vector_read_fire;
    assign window_close_ready = state_q == ST_RUN && !term_valid;
    assign window_close_fire = window_close && window_close_ready;
    assign read_data_valid = !read_vector_q && (
        bank_read_data_valid[0]
        || bank_read_data_valid[1]
        || bank_read_data_valid[2]
        || bank_read_data_valid[3]
        || bank_read_data_valid[4]
    );
    assign read_data = read_data_valid
        ? bank_read_data[read_bank_q][read_out_q*ACC_W +: ACC_W]
        : '0;
    assign vector_read_data_valid = read_vector_q && (
        bank_read_data_valid[0]
        || bank_read_data_valid[1]
        || bank_read_data_valid[2]
        || bank_read_data_valid[3]
        || bank_read_data_valid[4]
    );
    assign vector_read_data = vector_read_data_valid
        ? bank_read_data[read_bank_q] : '0;
    assign protocol_error = protocol_error_q;
    assign perf_product_terms = perf_terms_q;
    assign perf_destination_updates = perf_updates_q;
    assign all_banks_idle = &flush_seen_q;
    assign run_start_accepted = run_start
        && (state_q == ST_LOAD || state_q == ST_DONE)
        && !weight_context_release_fire
        && weights_loaded_q
        && (!run_accumulate || accumulator_initialized_q);

    generate
        for (genvar bank = 0; bank < 5; bank = bank + 1) begin : g_acc
            if (ACC_BACKEND_KIND == 0) begin : g_1r1w
                qfit_tcfm5_acc_bank #(
                    .DEPTH(BANK_DEPTH),
                    .OUT_DIM(OUT_DIM),
                    .ACC_W(ACC_W)
                ) u_acc_bank (
                    .clk_core(clk_core),
                    .rst_core(rst_core),
                    .clear_valid(state_q == ST_CLEAR),
                    .clear_addr(clear_addr_q),
                    .update_valid(term_commit && bank_write_enable[bank]),
                    .update_addr(bank_write_addr[bank]),
                    .update_delta(product_vector),
                    .update_idle(bank_update_idle[bank]),
                    .read_valid(combined_read_fire && read_bank == bank),
                    .read_addr(read_addr),
                    .read_data_valid(bank_read_data_valid[bank]),
                    .read_data(bank_read_data[bank])
                );
                assign bank_update_ready[bank] = 1'b1;
                assign bank_flush_ready[bank] = 1'b1;
                assign bank_flush_done[bank] = bank_update_idle[bank];
                assign bank_read_ready[bank] = 1'b1;
                assign bank_protocol_error[bank] = 1'b0;
                assign unused_bank_updates[bank] = '0;
                assign unused_bank_reads[bank] = '0;
                assign unused_bank_writes[bank] = '0;
            end else begin : g_1rw
                qfit_direct_1rw_acc_bank #(
                    .DEPTH(BANK_DEPTH), .OUT_DIM(OUT_DIM), .ACC_W(ACC_W),
                    .MEMORY_IMPL(ACC_MEMORY_IMPL)
                ) u_acc_bank (
                    .clk_core(clk_core), .rst_core(rst_core),
                    .run_start(run_start_accepted),
                    .run_accumulate(run_accumulate),
                    .update_valid(term_commit && bank_write_enable[bank]),
                    .update_ready(bank_update_ready[bank]),
                    .update_addr(bank_write_addr[bank]),
                    .update_delta(product_vector),
                    .flush_valid(state_q == ST_DRAIN && !flush_seen_q[bank]),
                    .flush_ready(bank_flush_ready[bank]),
                    .flush_done(bank_flush_done[bank]),
                    .read_valid(combined_read_fire && read_bank == bank),
                    .read_ready(bank_read_ready[bank]),
                    .read_addr(read_addr),
                    .read_data_valid(bank_read_data_valid[bank]),
                    .read_data(bank_read_data[bank]),
                    .protocol_error(bank_protocol_error[bank]),
                    .perf_updates(unused_bank_updates[bank]),
                    .perf_sram_reads(unused_bank_reads[bank]),
                    .perf_sram_writes(unused_bank_writes[bank])
                );
                assign bank_update_idle[bank] = bank_update_ready[bank];
            end
        end
    endgenerate

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_LOAD;
            weights_loaded_q <= 1'b0;
            accumulator_initialized_q <= 1'b0;
            clear_addr_q <= '0;
            protocol_error_q <= 1'b0;
            perf_terms_q <= '0;
            perf_updates_q <= '0;
            flush_seen_q <= '0;
            read_bank_q <= '0;
            read_out_q <= '0;
            read_vector_q <= 1'b0;
        end else begin
            case (state_q)
                ST_LOAD: begin
                    if (weight_valid && weight_ready) begin
                        if (!USE_PRECOMPUTED_PRODUCT)
                            weight_row_q[weight_lane][weight_out*W_W +: W_W]
                                <= weight_data;
                        if (weight_last)
                            weights_loaded_q <= 1'b1;
                    end
                    if (!weight_context_release_fire && run_start) begin
                        if (!weights_loaded_q
                            || (run_accumulate
                                && !accumulator_initialized_q)) begin
                            protocol_error_q <= 1'b1;
                        end else begin
                            clear_addr_q <= '0;
                            protocol_error_q <= 1'b0;
                            perf_terms_q <= '0;
                            perf_updates_q <= '0;
                            flush_seen_q <= '0;
                            if (run_accumulate) begin
                                state_q <= ST_RUN;
                            end else if (ACC_BACKEND_KIND != 0) begin
                                accumulator_initialized_q <= 1'b1;
                                state_q <= ST_RUN;
                            end else begin
                                accumulator_initialized_q <= 1'b0;
                                state_q <= ST_CLEAR;
                            end
                        end
                    end
                end

                ST_CLEAR: begin
                    if (
                        clear_addr_q
                        == BANK_ADDR_W'(BANK_DEPTH - 1)
                    ) begin
                        accumulator_initialized_q <= 1'b1;
                        state_q <= ST_RUN;
                    end else begin
                        clear_addr_q <= clear_addr_q + 1'b1;
                    end
                end

                ST_RUN: begin
                    if (term_fire) begin
                        logic [31:0] update_count;
                        update_count = '0;
                        if (!term_commit) begin
                            protocol_error_q <= 1'b1;
                        end else begin
                            perf_terms_q <= perf_terms_q + 1'b1;
                            for (
                                int bank = 0;
                                bank < 5;
                                bank = bank + 1
                            ) begin
                                if (bank_write_enable[bank]) begin
                                    update_count = update_count + 1'b1;
                                end
                            end
                            perf_updates_q <=
                                perf_updates_q + update_count;
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
                    for (integer bank = 0; bank < 5; bank = bank + 1) begin
                        if (bank_flush_done[bank])
                            flush_seen_q[bank] <= 1'b1;
                        if (bank_protocol_error[bank])
                            protocol_error_q <= 1'b1;
                    end
                    if (&(flush_seen_q
                          | {bank_flush_done[4], bank_flush_done[3],
                             bank_flush_done[2], bank_flush_done[1],
                             bank_flush_done[0]}))
                        state_q <= ST_DONE;
                end

                ST_DONE: begin
                    if (weight_context_release_fire) begin
                        weights_loaded_q <= 1'b0;
                        state_q <= ST_LOAD;
                    end else if (combined_read_fire) begin
                        read_bank_q <= read_bank;
                        read_out_q <= vector_read_fire ? '0 : read_out;
                        read_vector_q <= vector_read_fire;
                    end
                    if (read_valid && !read_ready)
                        protocol_error_q <= 1'b1;
                    if (vector_read_active && !vector_read_ready)
                        protocol_error_q <= 1'b1;
                    if (run_start && !weight_context_release_fire) begin
                        if (!weights_loaded_q
                            || (run_accumulate
                                && !accumulator_initialized_q)) begin
                            protocol_error_q <= 1'b1;
                        end else begin
                            clear_addr_q <= '0;
                            protocol_error_q <= 1'b0;
                            perf_terms_q <= '0;
                            perf_updates_q <= '0;
                            flush_seen_q <= '0;
                            if (run_accumulate) begin
                                state_q <= ST_RUN;
                            end else if (ACC_BACKEND_KIND != 0) begin
                                accumulator_initialized_q <= 1'b1;
                                state_q <= ST_RUN;
                            end else begin
                                accumulator_initialized_q <= 1'b0;
                                state_q <= ST_CLEAR;
                            end
                        end
                    end
                end

                default: state_q <= ST_LOAD;
            endcase
            if (weight_context_release && !weight_context_release_ready)
                protocol_error_q <= 1'b1;
            for (integer bank = 0; bank < 5; bank = bank + 1)
                if (bank_protocol_error[bank])
                    protocol_error_q <= 1'b1;
        end
    end
endmodule

`default_nettype wire
