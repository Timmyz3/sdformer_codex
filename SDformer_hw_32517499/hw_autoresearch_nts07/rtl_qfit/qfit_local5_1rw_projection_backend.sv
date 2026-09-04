`timescale 1ns/1ps
`default_nettype none

// Fair Local5 five-bank backend over one shared 1RW contract.
// MODE=0: direct same-port RMW. MODE=1: geometry-ahead GASR-2C.
module qfit_local5_1rw_projection_backend #(
    parameter int MODE = 1,
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int HEAD_DIM = 32,
    parameter int OUT_DIM = 2,
    parameter int GATE_W = 9,
    parameter int W_W = 8,
    parameter int ACC_W = 32,
    parameter int ACC_MEMORY_IMPL = 0,
    parameter int SOURCE_ID_W = $clog2(HEIGHT * WIDTH * TIME_PLANES),
    parameter int Y_W = $clog2(HEIGHT),
    parameter int X_W = $clog2(WIDTH),
    parameter int PLANE_W = (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int LANE_W = $clog2(HEAD_DIM),
    parameter int OUT_W = (OUT_DIM <= 1) ? 1 : $clog2(OUT_DIM)
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

    input  logic                       geometry_valid,
    output logic                       geometry_ready,
    input  logic [SOURCE_ID_W-1:0]     geometry_source_id,
    input  logic [PLANE_W-1:0]         geometry_plane,
    input  logic [Y_W-1:0]             geometry_y,
    input  logic [X_W-1:0]             geometry_x,
    input  logic [4:0]                 geometry_role_mask,
    input  logic                       geometry_last,

    input  logic                       term_valid,
    output logic                       term_ready,
    input  logic [SOURCE_ID_W-1:0]     term_source_id,
    input  logic [PLANE_W-1:0]         term_source_plane,
    input  logic [Y_W-1:0]             term_source_y,
    input  logic [X_W-1:0]             term_source_x,
    input  logic [LANE_W-1:0]          term_lane,
    input  logic [GATE_W-1:0]          term_gate,
    input  logic [4:0]                 term_destination_mask,
    input  logic                       term_last,
    input  logic                       term_source_last,
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
    output logic [31:0]                perf_term_stall_cycles,
    output logic [31:0]                perf_sram_reads,
    output logic [31:0]                perf_sram_writes
);
    localparam int X_GROUPS = (WIDTH + 4) / 5;
    localparam int PLANE_BANK_DEPTH = HEIGHT * X_GROUPS;
    localparam int BANK_DEPTH = TIME_PLANES * PLANE_BANK_DEPTH;
    localparam int BANK_ADDR_W = $clog2(BANK_DEPTH);
    localparam int VEC_W = OUT_DIM * ACC_W;

    typedef enum logic [1:0] {
        ST_LOAD = 2'd0, ST_RUN = 2'd1, ST_FLUSH = 2'd2, ST_DONE = 2'd3
    } state_t;
    state_t state_q;

    logic signed [W_W-1:0] weight_q [0:HEAD_DIM-1][0:OUT_DIM-1];
    logic weights_loaded_q;
    logic current_source_valid_q;
    logic [SOURCE_ID_W-1:0] current_source_id_q;
    logic current_source_last_q;
    logic next_source_valid_q;
    logic [SOURCE_ID_W-1:0] next_source_id_q;
    logic next_source_last_q;
    logic next_bank_valid_q [0:4];
    logic [BANK_ADDR_W-1:0] next_bank_addr_q [0:4];
    logic [4:0] flush_seen_q;
    logic protocol_error_q;
    logic [31:0] terms_q;
    logic [31:0] updates_q;
    logic [31:0] stalls_q;

    logic geometry_bank_valid [0:4];
    logic [BANK_ADDR_W-1:0] geometry_bank_addr [0:4];
    logic term_bank_enable [0:4];
    logic [BANK_ADDR_W-1:0] term_bank_addr [0:4];
    logic [2:0] read_bank;
    logic [BANK_ADDR_W-1:0] read_addr;
    logic [VEC_W-1:0] product_vector;
    logic [VEC_W-1:0] bank_read_data [0:4];
    logic bank_read_data_valid [0:4];
    logic bank_update_ready [0:4];
    logic bank_prepare_ready [0:4];
    logic bank_activate_ready [0:4];
    logic bank_flush_ready [0:4];
    logic bank_flush_done [0:4];
    logic bank_read_ready [0:4];
    logic [31:0] bank_reads [0:4];
    logic [31:0] bank_writes [0:4];
    logic bank_protocol_error [0:4];
    logic [31:0] unused_bank_updates [0:4];
    logic [31:0] unused_prepare_hits [0:4];
    logic [31:0] unused_prepare_misses [0:4];
    logic [2:0] read_bank_q;
    logic [OUT_W-1:0] read_out_q;
    logic geometry_fire;
    logic term_fire;
    logic read_fire;
    logic next_activate_ready;
    logic term_updates_ready;
    logic source_boundary_ready;
    logic [4:0] geometry_bank_valid_packed;
    logic [4:0] unused_geometry_valid;
    logic [5*BANK_ADDR_W-1:0] geometry_bank_addr_packed;
    logic [4:0] unused_term_geometry_valid;
    logic [4:0] term_bank_enable_packed;
    logic [5*BANK_ADDR_W-1:0] term_bank_addr_packed;
    logic geometry_boundary_error;
    logic term_boundary_error;

    function automatic logic [BANK_ADDR_W-1:0] bank_address(
        input logic [PLANE_W-1:0] plane,
        input logic [Y_W-1:0] y,
        input logic [X_W-1:0] x
    );
        bank_address = BANK_ADDR_W'(
            32'(plane) * PLANE_BANK_DEPTH + 32'(y) * X_GROUPS + 32'(x) / 5
        );
    endfunction

    always_comb begin
        for (integer bank = 0; bank < 5; bank++) begin
            geometry_bank_valid[bank] = geometry_bank_valid_packed[bank];
            geometry_bank_addr[bank] = geometry_bank_addr_packed[
                bank*BANK_ADDR_W +: BANK_ADDR_W
            ];
            term_bank_addr[bank] = term_bank_addr_packed[
                bank*BANK_ADDR_W +: BANK_ADDR_W
            ];
            term_bank_enable[bank] = term_bank_enable_packed[bank];
        end

        read_bank = 3'((32'(read_x) + 2 * 32'(read_y)) % 5);
        read_addr = bank_address(read_plane, read_y, read_x);
        product_vector = '0;
        for (integer out = 0; out < OUT_DIM; out++)
            product_vector[out*ACC_W +: ACC_W] = ACC_W'(
                signed'({1'b0, term_gate})
                * signed'(weight_q[term_lane][out])
            );
    end

    qfit_local5_color_map #(
        .HEIGHT(HEIGHT), .WIDTH(WIDTH), .TIME_PLANES(TIME_PLANES),
        .Y_W(Y_W), .X_W(X_W), .PLANE_W(PLANE_W),
        .BANK_DEPTH(BANK_DEPTH), .BANK_ADDR_W(BANK_ADDR_W)
    ) u_geometry_map (
        .source_plane(geometry_plane), .source_y(geometry_y),
        .source_x(geometry_x), .role_mask(geometry_role_mask),
        .bank_geometry_valid(unused_geometry_valid),
        .bank_enable(geometry_bank_valid_packed),
        .bank_address_packed(geometry_bank_addr_packed),
        .boundary_error(geometry_boundary_error)
    );

    qfit_local5_color_map #(
        .HEIGHT(HEIGHT), .WIDTH(WIDTH), .TIME_PLANES(TIME_PLANES),
        .Y_W(Y_W), .X_W(X_W), .PLANE_W(PLANE_W),
        .BANK_DEPTH(BANK_DEPTH), .BANK_ADDR_W(BANK_ADDR_W)
    ) u_term_map (
        .source_plane(term_source_plane), .source_y(term_source_y),
        .source_x(term_source_x), .role_mask(term_destination_mask),
        .bank_geometry_valid(unused_term_geometry_valid),
        .bank_enable(term_bank_enable_packed),
        .bank_address_packed(term_bank_addr_packed),
        .boundary_error(term_boundary_error)
    );

    always_comb begin
        term_updates_ready = 1'b1;
        next_activate_ready = 1'b1;
        for (integer bank = 0; bank < 5; bank++) begin
            if (term_bank_enable[bank] && !bank_update_ready[bank])
                term_updates_ready = 1'b0;
            if (next_bank_valid_q[bank] && !bank_activate_ready[bank])
                next_activate_ready = 1'b0;
        end
        source_boundary_ready = MODE == 0 || !term_last || term_source_last
                              || (next_source_valid_q && next_activate_ready);
    end

    assign weight_ready = state_q == ST_LOAD;
    assign run_busy = state_q == ST_RUN || state_q == ST_FLUSH;
    assign run_done = state_q == ST_DONE;
    assign geometry_ready = MODE == 0
        ? state_q == ST_RUN
        : state_q == ST_RUN && !next_source_valid_q
          && (&{(!geometry_bank_valid[4] || bank_prepare_ready[4]),
                 (!geometry_bank_valid[3] || bank_prepare_ready[3]),
                 (!geometry_bank_valid[2] || bank_prepare_ready[2]),
                 (!geometry_bank_valid[1] || bank_prepare_ready[1]),
                 (!geometry_bank_valid[0] || bank_prepare_ready[0])})
          && (current_source_valid_q
              || (&{(!geometry_bank_valid[4] || bank_activate_ready[4]),
                     (!geometry_bank_valid[3] || bank_activate_ready[3]),
                     (!geometry_bank_valid[2] || bank_activate_ready[2]),
                     (!geometry_bank_valid[1] || bank_activate_ready[1]),
                     (!geometry_bank_valid[0] || bank_activate_ready[0])}));
    assign geometry_fire = geometry_valid && geometry_ready;
    assign term_ready = state_q == ST_RUN && term_updates_ready
                      && source_boundary_ready
                      && (MODE == 0
                          || (current_source_valid_q
                              && term_source_id == current_source_id_q));
    assign term_fire = term_valid && term_ready;
    assign window_close_ready = state_q == ST_RUN && !term_valid
                              && !next_source_valid_q;
    assign read_ready = state_q == ST_DONE
                      && 32'(read_plane) < TIME_PLANES
                      && 32'(read_y) < HEIGHT && 32'(read_x) < WIDTH
                      && 32'(read_out) < OUT_DIM
                      && bank_read_ready[read_bank];
    assign read_fire = read_valid && read_ready;
    assign read_data_valid = bank_read_data_valid[0]
                           || bank_read_data_valid[1]
                           || bank_read_data_valid[2]
                           || bank_read_data_valid[3]
                           || bank_read_data_valid[4];
    assign read_data = read_data_valid
        ? bank_read_data[read_bank_q][read_out_q*ACC_W +: ACC_W] : '0;
    assign protocol_error = protocol_error_q;
    assign perf_product_terms = terms_q;
    assign perf_destination_updates = updates_q;
    assign perf_term_stall_cycles = stalls_q;

    always_comb begin
        perf_sram_reads = '0;
        perf_sram_writes = '0;
        for (integer bank = 0; bank < 5; bank++) begin
            perf_sram_reads = perf_sram_reads + bank_reads[bank];
            perf_sram_writes = perf_sram_writes + bank_writes[bank];
        end
    end

    generate
        for (genvar bank = 0; bank < 5; bank++) begin : g_bank
            if (MODE == 0) begin : g_direct
                qfit_direct_1rw_acc_bank #(
                    .DEPTH(BANK_DEPTH), .OUT_DIM(OUT_DIM), .ACC_W(ACC_W),
                    .MEMORY_IMPL(ACC_MEMORY_IMPL)
                ) u_acc (
                    .clk_core(clk_core), .rst_core(rst_core),
                    .run_start(run_start), .run_accumulate(1'b0),
                    .update_valid(term_fire && term_bank_enable[bank]),
                    .update_ready(bank_update_ready[bank]),
                    .update_addr(term_bank_addr[bank]),
                    .update_delta(product_vector),
                    .flush_valid(state_q == ST_FLUSH && !flush_seen_q[bank]),
                    .flush_ready(bank_flush_ready[bank]),
                    .flush_done(bank_flush_done[bank]),
                    .read_valid(read_fire && read_bank == bank),
                    .read_ready(bank_read_ready[bank]), .read_addr(read_addr),
                    .read_data_valid(bank_read_data_valid[bank]),
                    .read_data(bank_read_data[bank]),
                    .protocol_error(bank_protocol_error[bank]),
                    .perf_updates(unused_bank_updates[bank]),
                    .perf_sram_reads(bank_reads[bank]),
                    .perf_sram_writes(bank_writes[bank])
                );
                assign bank_prepare_ready[bank] = 1'b1;
                assign bank_activate_ready[bank] = 1'b1;
                assign unused_prepare_hits[bank] = '0;
                assign unused_prepare_misses[bank] = '0;
            end else begin : g_gasr
                qfit_gasr2c_acc_bank #(
                    .DEPTH(BANK_DEPTH), .OUT_DIM(OUT_DIM), .ACC_W(ACC_W),
                    .MEMORY_IMPL(ACC_MEMORY_IMPL)
                ) u_acc (
                    .clk_core(clk_core), .rst_core(rst_core),
                    .run_start(run_start),
                    .prepare_valid(geometry_valid && state_q == ST_RUN
                                   && !next_source_valid_q
                                   && geometry_bank_valid[bank]),
                    .prepare_ready(bank_prepare_ready[bank]),
                    .prepare_addr(geometry_bank_addr[bank]),
                    .activate_valid(
                        (geometry_fire && !current_source_valid_q
                         && geometry_bank_valid[bank])
                        || (term_fire && term_last && !term_source_last
                            && next_bank_valid_q[bank])
                    ),
                    .activate_ready(bank_activate_ready[bank]),
                    .activate_addr(
                        current_source_valid_q
                        ? next_bank_addr_q[bank] : geometry_bank_addr[bank]
                    ),
                    .update_valid(term_fire && term_bank_enable[bank]),
                    .update_ready(bank_update_ready[bank]),
                    .update_addr(term_bank_addr[bank]),
                    .update_delta(product_vector),
                    .flush_valid(state_q == ST_FLUSH && !flush_seen_q[bank]),
                    .flush_ready(bank_flush_ready[bank]),
                    .flush_done(bank_flush_done[bank]),
                    .read_valid(read_fire && read_bank == bank),
                    .read_ready(bank_read_ready[bank]), .read_addr(read_addr),
                    .read_data_valid(bank_read_data_valid[bank]),
                    .read_data(bank_read_data[bank]),
                    .protocol_error(bank_protocol_error[bank]),
                    .perf_updates(unused_bank_updates[bank]),
                    .perf_prepare_hits(unused_prepare_hits[bank]),
                    .perf_prepare_misses(unused_prepare_misses[bank]),
                    .perf_sram_reads(bank_reads[bank]),
                    .perf_sram_writes(bank_writes[bank])
                );
            end
        end
    endgenerate

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_LOAD;
            weights_loaded_q <= 1'b0;
            current_source_valid_q <= 1'b0;
            current_source_id_q <= '0;
            current_source_last_q <= 1'b0;
            next_source_valid_q <= 1'b0;
            next_source_id_q <= '0;
            next_source_last_q <= 1'b0;
            flush_seen_q <= '0;
            protocol_error_q <= 1'b0;
            terms_q <= '0;
            updates_q <= '0;
            stalls_q <= '0;
            read_bank_q <= '0;
            read_out_q <= '0;
            for (integer bank = 0; bank < 5; bank++) begin
                next_bank_valid_q[bank] <= 1'b0;
                next_bank_addr_q[bank] <= '0;
            end
            for (integer lane = 0; lane < HEAD_DIM; lane++)
                for (integer out = 0; out < OUT_DIM; out++)
                    weight_q[lane][out] <= '0;
        end else begin
            if (weight_valid && weight_ready) begin
                weight_q[weight_lane][weight_out] <= weight_data;
                if (weight_last)
                    weights_loaded_q <= 1'b1;
            end
            if (term_valid && !term_ready && state_q == ST_RUN)
                stalls_q <= stalls_q + 1'b1;
            if (read_fire) begin
                read_bank_q <= read_bank;
                read_out_q <= read_out;
            end

            case (state_q)
                ST_LOAD: begin
                    if (run_start && weights_loaded_q) begin
                        state_q <= ST_RUN;
                        current_source_valid_q <= 1'b0;
                        next_source_valid_q <= 1'b0;
                        protocol_error_q <= 1'b0;
                        terms_q <= '0;
                        updates_q <= '0;
                        stalls_q <= '0;
                    end
                end
                ST_RUN: begin
                    if (geometry_fire && MODE != 0) begin
                        if (!current_source_valid_q) begin
                            current_source_valid_q <= 1'b1;
                            current_source_id_q <= geometry_source_id;
                            current_source_last_q <= geometry_last;
                        end else begin
                            next_source_valid_q <= 1'b1;
                            next_source_id_q <= geometry_source_id;
                            next_source_last_q <= geometry_last;
                            for (integer bank = 0; bank < 5; bank++) begin
                                next_bank_valid_q[bank]
                                    <= geometry_bank_valid[bank];
                                next_bank_addr_q[bank]
                                    <= geometry_bank_addr[bank];
                            end
                        end
                    end
                    if (term_fire) begin
                        logic [31:0] update_count;
                        update_count = '0;
                        terms_q <= terms_q + 1'b1;
                        if (term_boundary_error)
                            protocol_error_q <= 1'b1;
                        for (integer bank = 0; bank < 5; bank++)
                            if (term_bank_enable[bank])
                                update_count = update_count + 1'b1;
                        updates_q <= updates_q + update_count;
                        if (MODE != 0 && term_last) begin
                            if (term_source_last != current_source_last_q)
                                protocol_error_q <= 1'b1;
                            if (!term_source_last) begin
                                current_source_id_q <= next_source_id_q;
                                current_source_last_q <= next_source_last_q;
                                next_source_valid_q <= 1'b0;
                            end
                        end
                    end
                    for (integer bank = 0; bank < 5; bank++)
                        if (bank_protocol_error[bank])
                            protocol_error_q <= 1'b1;
                    if (window_close && window_close_ready) begin
                        state_q <= ST_FLUSH;
                        flush_seen_q <= '0;
                    end
                end
                ST_FLUSH: begin
                    for (integer bank = 0; bank < 5; bank++)
                        if (bank_flush_done[bank])
                            flush_seen_q[bank] <= 1'b1;
                    if (&(flush_seen_q
                          | {bank_flush_done[4], bank_flush_done[3],
                             bank_flush_done[2], bank_flush_done[1],
                             bank_flush_done[0]}))
                        state_q <= ST_DONE;
                end
                ST_DONE: begin
                    if (run_start && weights_loaded_q) begin
                        state_q <= ST_RUN;
                        current_source_valid_q <= 1'b0;
                        next_source_valid_q <= 1'b0;
                        protocol_error_q <= 1'b0;
                        terms_q <= '0;
                        updates_q <= '0;
                        stalls_q <= '0;
                    end
                end
                default: state_q <= ST_LOAD;
            endcase
        end
    end
endmodule

`default_nettype wire
