`timescale 1ns/1ps
`default_nettype none

// Synchronous six-bank relation transpose with two-entry atomic descriptor FIFO.
// Candidate role order is {RIGHT, LEFT, DOWN, UP, SELF}, with SELF in
// the least-significant slice. After relation transpose, the role bit is
// preserved while the matching consumer lies in the opposite direction.
module qfit_relation_transpose_leaf #(
    parameter int SCHED_MODE = 0,
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int K_W = 32,
    parameter int GATE_W = 9,
    // Optional exact retirement filter. Every raster token is still written,
    // while K==0 sources retire without issuing the six payload reads.
    parameter bit SKIP_ZERO_K = 1'b0,
    parameter int STRIPE_RING_ROWS = 4,
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int PLANE_W =
        (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int SOURCE_ID_W =
        (HEIGHT * WIDTH * TIME_PLANES <= 1)
        ? 1 : $clog2(HEIGHT * WIDTH * TIME_PLANES)
) (
    input  logic                       clk_core,
    input  logic                       rst_core,
    input  logic                       plane_start,
    input  logic [PLANE_W-1:0]         plane_id,
    input  logic                       in_valid,
    output logic                       in_ready,
    input  logic [Y_W-1:0]             in_y,
    input  logic [X_W-1:0]             in_x,
    input  logic [4:0]                 in_candidate_valid,
    input  logic [K_W-1:0]             in_k_self,
    input  logic [5*GATE_W-1:0]        in_direction_gates,
    output logic                       descriptor_valid,
    input  logic                       descriptor_ready,
    output logic [SOURCE_ID_W-1:0]     descriptor_source_id,
    output logic [Y_W-1:0]             descriptor_y,
    output logic [X_W-1:0]             descriptor_x,
    output logic [K_W-1:0]             descriptor_k,
    output logic [5*GATE_W-1:0]        descriptor_incoming_gates,
    output logic [4:0]                 descriptor_valid_mask,
    output logic                       plane_idle,
    output logic [31:0]                perf_producer_stalls,
    output logic [2:0]                 perf_max_pending,
    output logic                       debug_read_pending,
    output logic                       debug_k_read_data_valid
);
    localparam int RING_ROWS =
        (SCHED_MODE == 2) ? STRIPE_RING_ROWS : 3;
    localparam int RING_ENTRIES = RING_ROWS * WIDTH;
    localparam int RING_ADDR_W =
        (RING_ENTRIES <= 1) ? 1 : $clog2(RING_ENTRIES);
    localparam int ROW_SLOT_W =
        (RING_ROWS <= 1) ? 1 : $clog2(RING_ROWS);
    localparam int FIFO_DEPTH = 2;
    localparam int TOKENS = HEIGHT * WIDTH;
    localparam int TOKEN_COUNT_W =
        (TOKENS <= 1) ? 1 : $clog2(TOKENS + 1);

`ifndef SYNTHESIS
    initial begin
        if (SCHED_MODE == 4 && !SKIP_ZERO_K)
            $fatal(
                1,
                "compiled cross_r1 scheduler requires SKIP_ZERO_K=1"
            );
    end
`endif

    logic retire_valid;
    logic retire_ready;
    logic scheduler_in_ready;
    logic scheduler_plane_idle;
    logic [SOURCE_ID_W-1:0] retire_source_id;
    logic [Y_W-1:0] retire_y;
    logic [X_W-1:0] retire_x;

    logic write_fire;
    logic read_issue;
    logic read_response;
    logic fifo_pop;
    logic [2:0] reserved_after_pop;
    logic [RING_ADDR_W-1:0] write_addr;
    logic [RING_ADDR_W-1:0] source_addr;
    logic [RING_ADDR_W-1:0] north_dest_addr;
    logic [RING_ADDR_W-1:0] south_dest_addr;
    logic [RING_ADDR_W-1:0] east_dest_addr;
    logic [RING_ADDR_W-1:0] west_dest_addr;
    logic [ROW_SLOT_W-1:0] write_row_slot;
    logic [ROW_SLOT_W-1:0] source_row_slot;
    logic [ROW_SLOT_W-1:0] north_row_slot;
    logic [ROW_SLOT_W-1:0] south_row_slot;

    logic k_rd_valid;
    logic self_rd_valid;
    logic n_rd_valid;
    logic s_rd_valid;
    logic e_rd_valid;
    logic w_rd_valid;
    logic [K_W-1:0] k_rd_data;
    logic [GATE_W:0] self_rd_data;
    logic [GATE_W:0] n_rd_data;
    logic [GATE_W:0] s_rd_data;
    logic [GATE_W:0] e_rd_data;
    logic [GATE_W:0] w_rd_data;

    logic read_inflight_q;
    logic [RING_ENTRIES-1:0] k_active_q;
    logic retire_k_active;
    logic [SOURCE_ID_W-1:0] read_source_id_q;
    logic [Y_W-1:0] read_y_q;
    logic [X_W-1:0] read_x_q;
    logic [4:0] read_valid_mask_q;
    logic [5*GATE_W-1:0] response_gates;
    logic [4:0] response_valid_mask;
    logic plane_start_fire;
    logic [4:0] scheduler_candidate_valid;
    logic [2:0] fcsr_active_events;
    logic plane_active_q;
    logic plane_input_complete_q;
    logic [TOKEN_COUNT_W-1:0] accepted_tokens_q;
    logic datapath_drained;

    logic fifo_head_q;
    logic fifo_tail_q;
    logic [1:0] fifo_count_q;
    logic [SOURCE_ID_W-1:0] fifo_source_id [0:FIFO_DEPTH-1];
    logic [Y_W-1:0] fifo_y [0:FIFO_DEPTH-1];
    logic [X_W-1:0] fifo_x [0:FIFO_DEPTH-1];
    logic [K_W-1:0] fifo_k [0:FIFO_DEPTH-1];
    logic [5*GATE_W-1:0] fifo_gates [0:FIFO_DEPTH-1];
    logic [4:0] fifo_valid_mask [0:FIFO_DEPTH-1];

    function automatic logic [ROW_SLOT_W-1:0] row_slot_from_y(
        input logic [Y_W-1:0] y
    );
        row_slot_from_y = '0;
        for (integer row = 0; row < HEIGHT; row = row + 1) begin
            if (y == row)
                row_slot_from_y = row % RING_ROWS;
        end
    endfunction

    generate
        if (SCHED_MODE == 3) begin : g_banked_dynamic_scheduler
            qfit_banked_dynamic_retirement_scheduler #(
                .HEIGHT(HEIGHT),
                .WIDTH(WIDTH),
                .TIME_PLANES(TIME_PLANES)
            ) u_scheduler (
                .clk_core(clk_core), .rst_core(rst_core),
                .plane_start(plane_start_fire), .plane_id(plane_id),
                .in_valid(in_valid), .in_ready(scheduler_in_ready),
                .in_y(in_y), .in_x(in_x),
                .in_candidate_valid(scheduler_candidate_valid),
                .retire_valid(retire_valid), .retire_ready(retire_ready),
                .retire_source_id(retire_source_id),
                .retire_y(retire_y), .retire_x(retire_x),
                .plane_idle(scheduler_plane_idle),
                .perf_producer_stalls(perf_producer_stalls),
                .perf_max_pending(perf_max_pending)
            );
        end else if (SCHED_MODE == 4) begin : g_compiled_cross_r1_scheduler
            logic [1:0] compiled_max_pending;

            generated_cross_r1_retirement_scheduler #(
                .HEIGHT(HEIGHT),
                .WIDTH(WIDTH),
                .TIME_PLANES(TIME_PLANES)
            ) u_scheduler (
                .clk_core(clk_core), .rst_core(rst_core),
                .plane_start(plane_start_fire), .plane_id(plane_id),
                .in_valid(in_valid), .in_ready(scheduler_in_ready),
                .in_candidate_valid(scheduler_candidate_valid[2:0]),
                .in_y(in_y), .in_x(in_x),
                .retire_valid(retire_valid), .retire_ready(retire_ready),
                .retire_source_id(retire_source_id),
                .retire_y(retire_y), .retire_x(retire_x),
                .plane_idle(scheduler_plane_idle),
                .perf_producer_stalls(perf_producer_stalls),
                .perf_max_pending(compiled_max_pending)
            );

            always_comb begin
                perf_max_pending = {1'b0, compiled_max_pending};
            end
        end else begin : g_existing_scheduler
            qfit_retirement_scheduler #(
                .MODE(SCHED_MODE),
                .HEIGHT(HEIGHT),
                .WIDTH(WIDTH),
                .TIME_PLANES(TIME_PLANES),
                .FILTER_FCSR_EVENTS(SKIP_ZERO_K),
                .STRIPE_EARLY_FILL(
                    SCHED_MODE == 2 && STRIPE_RING_ROWS >= 4
                ),
                .STRIPE_RING_ROWS(STRIPE_RING_ROWS)
            ) u_scheduler (
                .clk_core(clk_core), .rst_core(rst_core),
                .plane_start(plane_start_fire), .plane_id(plane_id),
                .in_valid(in_valid), .in_ready(scheduler_in_ready),
                .in_y(in_y), .in_x(in_x),
                .in_candidate_valid(scheduler_candidate_valid),
                .retire_valid(retire_valid), .retire_ready(retire_ready),
                .retire_source_id(retire_source_id),
                .retire_y(retire_y), .retire_x(retire_x),
                .plane_idle(scheduler_plane_idle),
                .perf_producer_stalls(perf_producer_stalls),
                .perf_max_pending(perf_max_pending)
            );
        end
    endgenerate

    always_comb begin
        write_row_slot = row_slot_from_y(in_y);
        source_row_slot = row_slot_from_y(retire_y);
        north_row_slot = (source_row_slot == 0)
            ? RING_ROWS - 1 : source_row_slot - 1'b1;
        south_row_slot = (source_row_slot == RING_ROWS - 1)
            ? '0 : source_row_slot + 1'b1;
        write_addr = write_row_slot * WIDTH + in_x;
        source_addr = source_row_slot * WIDTH + retire_x;
        north_dest_addr = north_row_slot * WIDTH + retire_x;
        south_dest_addr = south_row_slot * WIDTH + retire_x;
        east_dest_addr = source_addr;
        west_dest_addr = source_addr;
        if (retire_x < WIDTH - 1)
            east_dest_addr =
                source_row_slot * WIDTH + retire_x + 1'b1;
        if (retire_x > 0)
            west_dest_addr =
                source_row_slot * WIDTH + retire_x - 1'b1;
    end

    // Closed-form retirement can discard K==0 sources before they enter the
    // scheduler pending queue. Bit order follows the three FCSR event clauses
    // in qfit_retirement_scheduler.
    always_comb begin
        fcsr_active_events = '0;
        if (in_y > 0)
            fcsr_active_events[0] = k_active_q[
                row_slot_from_y(in_y - 1'b1) * WIDTH + in_x
            ];
        if (in_y == HEIGHT - 1 && in_x > 0)
            fcsr_active_events[1] = k_active_q[
                row_slot_from_y(in_y) * WIDTH + in_x - 1'b1
            ];
        if (in_y == HEIGHT - 1 && in_x == WIDTH - 1)
            fcsr_active_events[2] = in_k_self != '0;
        scheduler_candidate_valid = SKIP_ZERO_K
            && (SCHED_MODE == 0 || SCHED_MODE == 4)
            ? {2'b00, fcsr_active_events}
            : in_candidate_valid;
    end

    assign in_ready = scheduler_in_ready
                   && plane_active_q
                   && !plane_input_complete_q;
    assign plane_start_fire = plane_start && plane_idle;
    assign write_fire = !rst_core
                     && !plane_start_fire
                     && in_valid
                     && in_ready;
    assign fifo_pop = fifo_count_q != 0 && descriptor_ready;
    assign reserved_after_pop =
        {1'b0, fifo_count_q}
        + {2'b00, read_inflight_q}
        - {2'b00, fifo_pop};
    // Retirement is registered by the scheduler.  A source that becomes
    // retire-valid from the current input has already updated k_active_q at
    // the same clock edge, so no combinational write-through is required.
    // Keeping readiness independent of write_fire also breaks the
    // scheduler-ready -> input-ready -> write-fire loop.
    assign retire_k_active = k_active_q[source_addr];
    assign retire_ready = SKIP_ZERO_K && !retire_k_active
                        ? 1'b1
                        : reserved_after_pop < FIFO_DEPTH;
    assign read_issue = retire_valid
                      && retire_ready
                      && (!SKIP_ZERO_K || retire_k_active);
    assign read_response = read_inflight_q
                         && k_rd_valid
                         && self_rd_valid
                         && n_rd_valid
                         && s_rd_valid
                         && e_rd_valid
                         && w_rd_valid;
    assign debug_read_pending = read_inflight_q;
    assign debug_k_read_data_valid = k_rd_valid;

    qfit_sync_1r1w_bank #(
        .DATA_W(K_W),
        .DEPTH(RING_ENTRIES)
    ) u_k_bank (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .wr_en(write_fire),
        .wr_addr(write_addr),
        .wr_data(in_k_self),
        .rd_en(read_issue),
        .rd_addr(source_addr),
        .rd_valid(k_rd_valid),
        .rd_data(k_rd_data)
    );

    qfit_sync_1r1w_bank #(
        .DATA_W(GATE_W + 1),
        .DEPTH(RING_ENTRIES)
    ) u_self_bank (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .wr_en(write_fire),
        .wr_addr(write_addr),
        .wr_data({
            in_candidate_valid[0],
            in_direction_gates[0*GATE_W +: GATE_W]
        }),
        .rd_en(read_issue),
        .rd_addr(source_addr),
        .rd_valid(self_rd_valid),
        .rd_data(self_rd_data)
    );

    qfit_sync_1r1w_bank #(
        .DATA_W(GATE_W + 1),
        .DEPTH(RING_ENTRIES)
    ) u_n_bank (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .wr_en(write_fire),
        .wr_addr(write_addr),
        .wr_data({
            in_candidate_valid[1],
            in_direction_gates[1*GATE_W +: GATE_W]
        }),
        .rd_en(read_issue),
        .rd_addr(south_dest_addr),
        .rd_valid(n_rd_valid),
        .rd_data(n_rd_data)
    );

    qfit_sync_1r1w_bank #(
        .DATA_W(GATE_W + 1),
        .DEPTH(RING_ENTRIES)
    ) u_s_bank (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .wr_en(write_fire),
        .wr_addr(write_addr),
        .wr_data({
            in_candidate_valid[2],
            in_direction_gates[2*GATE_W +: GATE_W]
        }),
        .rd_en(read_issue),
        .rd_addr(north_dest_addr),
        .rd_valid(s_rd_valid),
        .rd_data(s_rd_data)
    );

    qfit_sync_1r1w_bank #(
        .DATA_W(GATE_W + 1),
        .DEPTH(RING_ENTRIES)
    ) u_e_bank (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .wr_en(write_fire),
        .wr_addr(write_addr),
        .wr_data({
            in_candidate_valid[3],
            in_direction_gates[3*GATE_W +: GATE_W]
        }),
        .rd_en(read_issue),
        .rd_addr(east_dest_addr),
        .rd_valid(e_rd_valid),
        .rd_data(e_rd_data)
    );

    qfit_sync_1r1w_bank #(
        .DATA_W(GATE_W + 1),
        .DEPTH(RING_ENTRIES)
    ) u_w_bank (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .wr_en(write_fire),
        .wr_addr(write_addr),
        .wr_data({
            in_candidate_valid[4],
            in_direction_gates[4*GATE_W +: GATE_W]
        }),
        .rd_en(read_issue),
        .rd_addr(west_dest_addr),
        .rd_valid(w_rd_valid),
        .rd_data(w_rd_data)
    );

    always_comb begin
        response_gates = '0;
        response_valid_mask = '0;
        response_valid_mask[0] =
            read_valid_mask_q[0] && self_rd_data[GATE_W];
        response_valid_mask[1] =
            read_valid_mask_q[1] && n_rd_data[GATE_W];
        response_valid_mask[2] =
            read_valid_mask_q[2] && s_rd_data[GATE_W];
        response_valid_mask[3] =
            read_valid_mask_q[3] && e_rd_data[GATE_W];
        response_valid_mask[4] =
            read_valid_mask_q[4] && w_rd_data[GATE_W];
        if (response_valid_mask[0])
            response_gates[0*GATE_W +: GATE_W] =
                self_rd_data[GATE_W-1:0];
        if (response_valid_mask[1])
            response_gates[1*GATE_W +: GATE_W] =
                n_rd_data[GATE_W-1:0];
        if (response_valid_mask[2])
            response_gates[2*GATE_W +: GATE_W] =
                s_rd_data[GATE_W-1:0];
        if (response_valid_mask[3])
            response_gates[3*GATE_W +: GATE_W] =
                e_rd_data[GATE_W-1:0];
        if (response_valid_mask[4])
            response_gates[4*GATE_W +: GATE_W] =
                w_rd_data[GATE_W-1:0];
    end

    assign descriptor_valid = fifo_count_q != 0;
    assign descriptor_source_id = fifo_source_id[fifo_head_q];
    assign descriptor_y = fifo_y[fifo_head_q];
    assign descriptor_x = fifo_x[fifo_head_q];
    assign descriptor_k = fifo_k[fifo_head_q];
    assign descriptor_incoming_gates = fifo_gates[fifo_head_q];
    assign descriptor_valid_mask = fifo_valid_mask[fifo_head_q];
    assign datapath_drained = scheduler_plane_idle
                           && !read_inflight_q
                           && fifo_count_q == 0
                           && !k_rd_valid;
    assign plane_idle = !plane_active_q && datapath_drained;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            plane_active_q <= 1'b0;
            plane_input_complete_q <= 1'b0;
            accepted_tokens_q <= '0;
            read_inflight_q <= 1'b0;
            read_source_id_q <= '0;
            read_y_q <= '0;
            read_x_q <= '0;
            read_valid_mask_q <= '0;
            k_active_q <= '0;
            fifo_head_q <= 1'b0;
            fifo_tail_q <= 1'b0;
            fifo_count_q <= '0;
            for (int slot = 0; slot < FIFO_DEPTH; slot = slot + 1) begin
                fifo_source_id[slot] <= '0;
                fifo_y[slot] <= '0;
                fifo_x[slot] <= '0;
                fifo_k[slot] <= '0;
                fifo_gates[slot] <= '0;
                fifo_valid_mask[slot] <= '0;
            end
        end else if (plane_start_fire) begin
            plane_active_q <= 1'b1;
            plane_input_complete_q <= 1'b0;
            accepted_tokens_q <= '0;
            read_inflight_q <= 1'b0;
            read_source_id_q <= '0;
            read_y_q <= '0;
            read_x_q <= '0;
            read_valid_mask_q <= '0;
            k_active_q <= '0;
            fifo_head_q <= 1'b0;
            fifo_tail_q <= 1'b0;
            fifo_count_q <= '0;
            for (int slot = 0; slot < FIFO_DEPTH; slot = slot + 1) begin
                fifo_source_id[slot] <= '0;
                fifo_y[slot] <= '0;
                fifo_x[slot] <= '0;
                fifo_k[slot] <= '0;
                fifo_gates[slot] <= '0;
                fifo_valid_mask[slot] <= '0;
            end
        end else begin
            if (write_fire) begin
                k_active_q[write_addr] <= in_k_self != '0;
                accepted_tokens_q <= accepted_tokens_q + 1'b1;
                if (
                    accepted_tokens_q
                    == TOKEN_COUNT_W'(TOKENS - 1)
                )
                    plane_input_complete_q <= 1'b1;
            end
            if (
                plane_active_q
                && plane_input_complete_q
                && datapath_drained
            ) begin
                plane_active_q <= 1'b0;
                plane_input_complete_q <= 1'b0;
            end
            if (fifo_pop)
                fifo_head_q <= ~fifo_head_q;
            if (read_response) begin
                fifo_source_id[fifo_tail_q] <= read_source_id_q;
                fifo_y[fifo_tail_q] <= read_y_q;
                fifo_x[fifo_tail_q] <= read_x_q;
                fifo_k[fifo_tail_q] <= k_rd_data;
                fifo_gates[fifo_tail_q] <= response_gates;
                fifo_valid_mask[fifo_tail_q] <= response_valid_mask;
                fifo_tail_q <= ~fifo_tail_q;
            end
            case ({read_response, fifo_pop})
                2'b10: fifo_count_q <= fifo_count_q + 2'd1;
                2'b01: fifo_count_q <= fifo_count_q - 2'd1;
                default: begin end
            endcase

            if (read_response)
                read_inflight_q <= 1'b0;
            if (read_issue) begin
                read_inflight_q <= 1'b1;
                read_source_id_q <= retire_source_id;
                read_y_q <= retire_y;
                read_x_q <= retire_x;
                read_valid_mask_q[0] <= 1'b1;
                read_valid_mask_q[1] <= retire_y < HEIGHT - 1;
                read_valid_mask_q[2] <= retire_y > 0;
                read_valid_mask_q[3] <= retire_x < WIDTH - 1;
                read_valid_mask_q[4] <= retire_x > 0;
            end
        end
    end
endmodule

`default_nettype wire
