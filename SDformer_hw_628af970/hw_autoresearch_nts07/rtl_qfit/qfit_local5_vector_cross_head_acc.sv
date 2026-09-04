`timescale 1ns/1ps
`default_nettype none

// Strong materialization baseline: five legal 1RW vector banks preserve the
// TCFM5 color layout while accumulating complete 1024-bit rows across heads.
module qfit_local5_vector_cross_head_acc #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int OUT_DIM = 32,
    parameter int ACC_W = 32,
    parameter int MEMORY_IMPL = 0,
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int PLANE_W =
        (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES)
) (
    input  logic                         clk_core,
    input  logic                         rst_core,
    input  logic                         run_start,
    input  logic                         update_valid,
    output logic                         update_ready,
    input  logic [PLANE_W-1:0]           update_plane,
    input  logic [Y_W-1:0]               update_y,
    input  logic [X_W-1:0]               update_x,
    input  logic [OUT_DIM*ACC_W-1:0]     update_delta,
    input  logic                         flush_valid,
    output logic                         flush_ready,
    output logic                         flush_done,
    input  logic                         read_valid,
    output logic                         read_ready,
    input  logic [PLANE_W-1:0]           read_plane,
    input  logic [Y_W-1:0]               read_y,
    input  logic [X_W-1:0]               read_x,
    output logic                         read_data_valid,
    output logic [OUT_DIM*ACC_W-1:0]     read_data,
    output logic                         protocol_error
);
    localparam int X_GROUPS = (WIDTH + 4) / 5;
    localparam int PLANE_BANK_DEPTH = HEIGHT * X_GROUPS;
    localparam int BANK_DEPTH = TIME_PLANES * PLANE_BANK_DEPTH;
    localparam int BANK_ADDR_W =
        (BANK_DEPTH <= 1) ? 1 : $clog2(BANK_DEPTH);
    localparam int VEC_W = OUT_DIM * ACC_W;

    typedef enum logic [1:0] {ST_IDLE, ST_RUN, ST_FLUSH, ST_DONE} state_t;
    state_t state_q;
    logic [4:0] flush_seen_q;
    logic [2:0] update_bank;
    logic [2:0] read_bank;
    logic [2:0] read_bank_q;
    logic [BANK_ADDR_W-1:0] update_addr;
    logic [BANK_ADDR_W-1:0] read_addr;
    logic bank_update_ready [0:4];
    logic bank_flush_ready [0:4];
    logic bank_flush_done [0:4];
    logic bank_read_ready [0:4];
    logic bank_read_valid [0:4];
    logic [VEC_W-1:0] bank_read_data [0:4];
    logic bank_error [0:4];
    logic [31:0] unused_updates [0:4];
    logic [31:0] unused_reads [0:4];
    logic [31:0] unused_writes [0:4];
    logic update_fire;
    logic read_fire;
    logic all_flush_ready;

    function automatic logic [2:0] x_color_lut(input logic [X_W-1:0] x);
        logic [2:0] color;
        begin
            x_color_lut = '0;
            color = '0;
            for (integer col = 0; col < WIDTH; col++) begin
                if (x == X_W'(col)) x_color_lut = color;
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
            for (integer col = 0; col < WIDTH; col++) begin
                if (x == X_W'(col)) x_group_lut = group;
                if (color == 3'd4) begin
                    color = '0;
                    group = group + 1'b1;
                end else color = color + 1'b1;
            end
        end
    endfunction

    function automatic logic [2:0] y_color_lut(input logic [Y_W-1:0] y);
        logic [2:0] color;
        begin
            y_color_lut = '0;
            color = '0;
            for (integer row = 0; row < HEIGHT; row++) begin
                if (y == Y_W'(row)) y_color_lut = color;
                color = (color >= 3'd3) ? color - 3'd3 : color + 3'd2;
            end
        end
    endfunction

    function automatic logic [2:0] add_mod5(
        input logic [2:0] lhs, input logic [2:0] rhs
    );
        logic [3:0] sum;
        begin
            sum = {1'b0, lhs} + {1'b0, rhs};
            if (sum >= 4'd5) sum = sum - 4'd5;
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
            for (integer p = 0; p < TIME_PLANES; p++)
                if (plane == PLANE_W'(p))
                    plane_base = BANK_ADDR_W'(p * PLANE_BANK_DEPTH);
            for (integer row = 0; row < HEIGHT; row++)
                if (y == Y_W'(row))
                    row_base = BANK_ADDR_W'(row * X_GROUPS);
            bank_address = plane_base + row_base + x_group_lut(x);
        end
    endfunction

    always_comb begin
        update_bank = add_mod5(x_color_lut(update_x), y_color_lut(update_y));
        read_bank = add_mod5(x_color_lut(read_x), y_color_lut(read_y));
        update_addr = bank_address(update_plane, update_y, update_x);
        read_addr = bank_address(read_plane, read_y, read_x);
        all_flush_ready = 1'b1;
        for (integer bank = 0; bank < 5; bank++)
            if (!bank_flush_ready[bank]) all_flush_ready = 1'b0;
    end

    assign update_ready = state_q == ST_RUN
                       && 32'(update_plane) < TIME_PLANES
                       && 32'(update_y) < HEIGHT
                       && 32'(update_x) < WIDTH
                       && bank_update_ready[update_bank];
    assign update_fire = update_valid && update_ready;
    assign flush_ready = state_q == ST_RUN && !update_valid && all_flush_ready;
    assign read_ready = state_q == ST_DONE
                     && 32'(read_plane) < TIME_PLANES
                     && 32'(read_y) < HEIGHT
                     && 32'(read_x) < WIDTH
                     && bank_read_ready[read_bank];
    assign read_fire = read_valid && read_ready;
    assign read_data_valid = bank_read_valid[0] || bank_read_valid[1]
                          || bank_read_valid[2] || bank_read_valid[3]
                          || bank_read_valid[4];
    assign read_data = read_data_valid ? bank_read_data[read_bank_q] : '0;
    assign protocol_error = |{bank_error[0], bank_error[1], bank_error[2],
                              bank_error[3], bank_error[4]};

    generate
        for (genvar bank = 0; bank < 5; bank++) begin : g_bank
            qfit_direct_1rw_acc_bank #(
                .DEPTH(BANK_DEPTH), .OUT_DIM(OUT_DIM), .ACC_W(ACC_W),
                .MEMORY_IMPL(MEMORY_IMPL)
            ) u_bank (
                .clk_core(clk_core), .rst_core(rst_core),
                .run_start(run_start), .run_accumulate(1'b0),
                .update_valid(state_q == ST_RUN
                              && update_valid && update_bank == bank),
                .update_ready(bank_update_ready[bank]),
                .update_addr(update_addr), .update_delta(update_delta),
                .flush_valid(state_q == ST_RUN
                             && flush_valid && !update_valid),
                .flush_ready(bank_flush_ready[bank]),
                .flush_done(bank_flush_done[bank]),
                .read_valid(read_fire && read_bank == bank),
                .read_ready(bank_read_ready[bank]), .read_addr(read_addr),
                .read_data_valid(bank_read_valid[bank]),
                .read_data(bank_read_data[bank]),
                .protocol_error(bank_error[bank]),
                .perf_updates(unused_updates[bank]),
                .perf_sram_reads(unused_reads[bank]),
                .perf_sram_writes(unused_writes[bank])
            );
        end
    endgenerate

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            flush_seen_q <= '0;
            flush_done <= 1'b0;
            read_bank_q <= '0;
        end else begin
            flush_done <= 1'b0;
            if (run_start) begin
                state_q <= ST_RUN;
                flush_seen_q <= '0;
            end else begin
                case (state_q)
                    ST_RUN: if (flush_valid && flush_ready)
                        state_q <= ST_FLUSH;
                    ST_FLUSH: begin
                        for (integer bank = 0; bank < 5; bank++)
                            if (bank_flush_done[bank]) flush_seen_q[bank] <= 1'b1;
                        if (&(flush_seen_q
                              | {bank_flush_done[4], bank_flush_done[3],
                                 bank_flush_done[2], bank_flush_done[1],
                                 bank_flush_done[0]})) begin
                            flush_done <= 1'b1;
                            state_q <= ST_DONE;
                        end
                    end
                    ST_DONE: if (read_fire) read_bank_q <= read_bank;
                    default: begin end
                endcase
            end
        end
    end
endmodule

`default_nettype wire
