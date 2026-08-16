`timescale 1ns/1ps
`default_nettype none

// Small deterministic fixture for the passive EREP v4 calibration harness.
// It exercises the existing Direct online path and the existing TCFM5 1RW
// backend. It is not an EREP candidate datapath.
module tb_qfit_local5_erep_calibration_v4;
    localparam int HEIGHT = 3;
    localparam int WIDTH = 3;
    localparam int TIME_PLANES = 1;
    localparam int SOURCES = HEIGHT * WIDTH * TIME_PLANES;
    localparam int HEAD_DIM = 4;
    localparam int OUT_DIM = 2;
    localparam int GATE_W = 9;
    localparam int W_W = 8;
    localparam int ACC_W = 32;
    localparam int Y_W = $clog2(HEIGHT);
    localparam int X_W = $clog2(WIDTH);
    localparam int PLANE_W = 1;
    localparam int LANE_W = $clog2(HEAD_DIM);
    localparam int OUT_W = $clog2(OUT_DIM);
    localparam int BANK_DEPTH = TIME_PLANES * HEIGHT * ((WIDTH + 4) / 5);
    localparam int BANK_ADDR_W = $clog2(BANK_DEPTH);
    localparam int VEC_W = OUT_DIM * ACC_W;

    logic clk_core = 1'b0;
    logic rst_core;

    logic d_weight_valid;
    logic d_weight_ready;
    logic [LANE_W-1:0] d_weight_lane;
    logic [OUT_W-1:0] d_weight_out;
    logic signed [W_W-1:0] d_weight_data;
    logic d_weight_last;
    logic d_projection_start;
    logic d_projection_close;
    logic d_projection_close_ready;
    logic d_projection_busy;
    logic d_projection_done;
    logic d_relation_start;
    logic d_relation_seal;
    logic d_relation_active;
    logic d_relation_done;
    logic d_relation_valid;
    logic d_relation_ready;
    logic [PLANE_W-1:0] d_relation_plane;
    logic [Y_W-1:0] d_relation_destination_y;
    logic [X_W-1:0] d_relation_destination_x;
    logic [4:0] d_relation_candidate_valid;
    logic [4:0] d_relation_active_candidate_mask;
    logic [HEAD_DIM-1:0] d_relation_k_self;
    logic [5*GATE_W-1:0] d_relation_direction_gates;
    logic d_read_valid;
    logic d_read_ready;
    logic [PLANE_W-1:0] d_read_plane;
    logic [Y_W-1:0] d_read_y;
    logic [X_W-1:0] d_read_x;
    logic [OUT_W-1:0] d_read_out;
    logic d_read_data_valid;
    logic signed [ACC_W-1:0] d_read_data;
    logic d_protocol_error;
    logic [31:0] d_perf_relation_writes;
    logic [31:0] d_perf_active_source_reads;
    logic [31:0] d_perf_dense_reads_avoided;
    logic [31:0] d_perf_memory_wait_cycles;
    logic [31:0] d_perf_descriptors;
    logic [31:0] d_perf_product_terms;
    logic [31:0] d_perf_destination_updates;
    logic [31:0] d_perf_term_stall_cycles;
    logic [31:0] d_perf_sram_reads;
    logic [31:0] d_perf_sram_writes;

    logic t_weight_valid;
    logic t_weight_ready;
    logic [LANE_W-1:0] t_weight_lane;
    logic [OUT_W-1:0] t_weight_out;
    logic signed [W_W-1:0] t_weight_data;
    logic t_weight_last;
    logic t_weight_context_release;
    logic t_weight_context_release_ready;
    logic t_run_start;
    logic t_run_accumulate;
    logic t_run_busy;
    logic t_run_done;
    logic t_term_valid;
    logic t_term_ready;
    logic [PLANE_W-1:0] t_term_source_plane;
    logic [Y_W-1:0] t_term_source_y;
    logic [X_W-1:0] t_term_source_x;
    logic [LANE_W-1:0] t_term_lane;
    logic [GATE_W-1:0] t_term_gate;
    logic [4:0] t_term_destination_mask;
    logic t_term_window_last;
    logic t_window_close;
    logic t_window_close_ready;
    logic t_read_valid;
    logic t_read_ready;
    logic [PLANE_W-1:0] t_read_plane;
    logic [Y_W-1:0] t_read_y;
    logic [X_W-1:0] t_read_x;
    logic [OUT_W-1:0] t_read_out;
    logic t_read_data_valid;
    logic signed [ACC_W-1:0] t_read_data;
    logic t_vector_read_valid;
    logic t_vector_read_ready;
    logic [PLANE_W-1:0] t_vector_read_plane;
    logic [Y_W-1:0] t_vector_read_y;
    logic [X_W-1:0] t_vector_read_x;
    logic t_vector_read_data_valid;
    logic [VEC_W-1:0] t_vector_read_data;
    logic t_protocol_error;
    logic [31:0] t_perf_product_terms;
    logic [31:0] t_perf_destination_updates;

    logic ser_in_ready;
    logic [PLANE_W-1:0] ser_pending_plane_q;
    logic [Y_W-1:0] ser_pending_y_q;
    logic [X_W-1:0] ser_pending_x_q;
    logic ser_pending_last_q;
    logic ser_out_valid;
    logic ser_out_ready;
    logic [PLANE_W-1:0] ser_out_plane;
    logic [Y_W-1:0] ser_out_y;
    logic [X_W-1:0] ser_out_x;
    logic [OUT_W-1:0] ser_out_index;
    logic signed [ACC_W-1:0] ser_out_data;
    logic ser_out_last;

    logic [4:0] d_bank_update_enable;
    logic [4:0] d_bank_update_ready;
    logic [5*BANK_ADDR_W-1:0] d_bank_update_addr;
    logic [4:0] d_bank_command_valid;
    logic [4:0] d_bank_command_write;
    logic [5*BANK_ADDR_W-1:0] d_bank_command_addr;
    logic [5*VEC_W-1:0] d_bank_command_write_data;
    logic [4:0] t_bank_update_enable;
    logic [4:0] t_bank_update_ready;
    logic [5*BANK_ADDR_W-1:0] t_bank_update_addr;
    logic [4:0] t_bank_command_valid;
    logic [4:0] t_bank_command_write;
    logic [5*BANK_ADDR_W-1:0] t_bank_command_addr;
    logic [5*VEC_W-1:0] t_bank_command_write_data;

    integer expected_direct [0:SOURCES-1][0:OUT_DIM-1];
    integer expected_tcfm5 [0:SOURCES-1][0:OUT_DIM-1];
    integer serializer_checks;
    integer serializer_last_checks;

    qfit_local5_1rw_active_projection_tile #(
        .MODE(0),
        .GEOMETRY_SYNC_MODE(1),
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .HEAD_DIM(HEAD_DIM),
        .OUT_DIM(OUT_DIM),
        .GATE_W(GATE_W),
        .W_W(W_W),
        .ACC_W(ACC_W),
        .RELATION_READ_LATENCY(1),
        .RELATION_MEMORY_IMPL(0),
        .ACC_MEMORY_IMPL(0)
    ) u_active (
        .clk_core(clk_core), .rst_core(rst_core),
        .weight_valid(d_weight_valid), .weight_ready(d_weight_ready),
        .weight_lane(d_weight_lane), .weight_out(d_weight_out),
        .weight_data(d_weight_data), .weight_last(d_weight_last),
        .projection_start(d_projection_start),
        .projection_close(d_projection_close),
        .projection_close_ready(d_projection_close_ready),
        .projection_busy(d_projection_busy), .projection_done(d_projection_done),
        .relation_start(d_relation_start), .relation_seal(d_relation_seal),
        .relation_active(d_relation_active), .relation_done(d_relation_done),
        .relation_valid(d_relation_valid), .relation_ready(d_relation_ready),
        .relation_plane(d_relation_plane),
        .relation_destination_y(d_relation_destination_y),
        .relation_destination_x(d_relation_destination_x),
        .relation_candidate_valid(d_relation_candidate_valid),
        .relation_active_candidate_mask(d_relation_active_candidate_mask),
        .relation_k_self(d_relation_k_self),
        .relation_direction_gates(d_relation_direction_gates),
        .read_valid(d_read_valid), .read_ready(d_read_ready),
        .read_plane(d_read_plane), .read_y(d_read_y), .read_x(d_read_x),
        .read_out(d_read_out), .read_data_valid(d_read_data_valid),
        .read_data(d_read_data), .protocol_error(d_protocol_error),
        .perf_relation_writes(d_perf_relation_writes),
        .perf_active_source_reads(d_perf_active_source_reads),
        .perf_dense_reads_avoided(d_perf_dense_reads_avoided),
        .perf_memory_wait_cycles(d_perf_memory_wait_cycles),
        .perf_descriptors(d_perf_descriptors),
        .perf_product_terms(d_perf_product_terms),
        .perf_destination_updates(d_perf_destination_updates),
        .perf_term_stall_cycles(d_perf_term_stall_cycles),
        .perf_sram_reads(d_perf_sram_reads),
        .perf_sram_writes(d_perf_sram_writes)
    );

    qfit_tcfm5_projection_top #(
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .HEAD_DIM(HEAD_DIM),
        .OUT_DIM(OUT_DIM),
        .GATE_W(GATE_W),
        .W_W(W_W),
        .ACC_W(ACC_W),
        .ENABLE_VECTOR_READ(1'b1),
        .ACC_BACKEND_KIND(1),
        .ACC_MEMORY_IMPL(0)
    ) u_tcfm5 (
        .clk_core(clk_core), .rst_core(rst_core),
        .weight_valid(t_weight_valid), .weight_ready(t_weight_ready),
        .weight_lane(t_weight_lane), .weight_out(t_weight_out),
        .weight_data(t_weight_data), .weight_last(t_weight_last),
        .weight_context_release(t_weight_context_release),
        .weight_context_release_ready(t_weight_context_release_ready),
        .run_start(t_run_start), .run_accumulate(t_run_accumulate),
        .run_busy(t_run_busy), .run_done(t_run_done),
        .term_valid(t_term_valid), .term_ready(t_term_ready),
        .term_source_plane(t_term_source_plane),
        .term_source_y(t_term_source_y), .term_source_x(t_term_source_x),
        .term_lane(t_term_lane), .term_gate(t_term_gate),
        .term_destination_mask(t_term_destination_mask),
        .term_product('0),
        .term_window_last(t_term_window_last),
        .window_close(t_window_close),
        .window_close_ready(t_window_close_ready),
        .read_valid(t_read_valid), .read_ready(t_read_ready),
        .read_plane(t_read_plane), .read_y(t_read_y), .read_x(t_read_x),
        .read_out(t_read_out), .read_data_valid(t_read_data_valid),
        .read_data(t_read_data),
        .vector_read_valid(t_vector_read_valid),
        .vector_read_ready(t_vector_read_ready),
        .vector_read_plane(t_vector_read_plane),
        .vector_read_y(t_vector_read_y), .vector_read_x(t_vector_read_x),
        .vector_read_data_valid(t_vector_read_data_valid),
        .vector_read_data(t_vector_read_data),
        .protocol_error(t_protocol_error),
        .perf_product_terms(t_perf_product_terms),
        .perf_destination_updates(t_perf_destination_updates)
    );

    qfit_acc32_vector_serializer #(
        .HEIGHT(HEIGHT), .WIDTH(WIDTH), .TIME_PLANES(TIME_PLANES),
        .OUT_DIM(OUT_DIM), .ACC_W(ACC_W)
    ) u_serializer (
        .clk_core(clk_core), .rst_core(rst_core),
        .in_valid(t_vector_read_data_valid), .in_ready(ser_in_ready),
        .in_plane(ser_pending_plane_q), .in_y(ser_pending_y_q),
        .in_x(ser_pending_x_q), .in_data(t_vector_read_data),
        .in_last(ser_pending_last_q),
        .out_valid(ser_out_valid), .out_ready(ser_out_ready),
        .out_plane(ser_out_plane), .out_y(ser_out_y), .out_x(ser_out_x),
        .out_index(ser_out_index), .out_data(ser_out_data),
        .out_last(ser_out_last)
    );

    assign d_bank_update_enable = {
        u_active.u_backend.term_bank_enable[4],
        u_active.u_backend.term_bank_enable[3],
        u_active.u_backend.term_bank_enable[2],
        u_active.u_backend.term_bank_enable[1],
        u_active.u_backend.term_bank_enable[0]
    };
    assign d_bank_update_ready = {
        u_active.u_backend.bank_update_ready[4],
        u_active.u_backend.bank_update_ready[3],
        u_active.u_backend.bank_update_ready[2],
        u_active.u_backend.bank_update_ready[1],
        u_active.u_backend.bank_update_ready[0]
    };
    assign d_bank_update_addr = {
        u_active.u_backend.term_bank_addr[4],
        u_active.u_backend.term_bank_addr[3],
        u_active.u_backend.term_bank_addr[2],
        u_active.u_backend.term_bank_addr[1],
        u_active.u_backend.term_bank_addr[0]
    };
    assign d_bank_command_valid = {
        u_active.u_backend.g_bank[4].g_direct.u_acc.memory_command_valid,
        u_active.u_backend.g_bank[3].g_direct.u_acc.memory_command_valid,
        u_active.u_backend.g_bank[2].g_direct.u_acc.memory_command_valid,
        u_active.u_backend.g_bank[1].g_direct.u_acc.memory_command_valid,
        u_active.u_backend.g_bank[0].g_direct.u_acc.memory_command_valid
    };
    assign d_bank_command_write = {
        u_active.u_backend.g_bank[4].g_direct.u_acc.memory_command_write,
        u_active.u_backend.g_bank[3].g_direct.u_acc.memory_command_write,
        u_active.u_backend.g_bank[2].g_direct.u_acc.memory_command_write,
        u_active.u_backend.g_bank[1].g_direct.u_acc.memory_command_write,
        u_active.u_backend.g_bank[0].g_direct.u_acc.memory_command_write
    };
    assign d_bank_command_addr = {
        u_active.u_backend.g_bank[4].g_direct.u_acc.memory_command_addr,
        u_active.u_backend.g_bank[3].g_direct.u_acc.memory_command_addr,
        u_active.u_backend.g_bank[2].g_direct.u_acc.memory_command_addr,
        u_active.u_backend.g_bank[1].g_direct.u_acc.memory_command_addr,
        u_active.u_backend.g_bank[0].g_direct.u_acc.memory_command_addr
    };
    assign d_bank_command_write_data = {
        u_active.u_backend.g_bank[4].g_direct.u_acc.memory_command_write_data,
        u_active.u_backend.g_bank[3].g_direct.u_acc.memory_command_write_data,
        u_active.u_backend.g_bank[2].g_direct.u_acc.memory_command_write_data,
        u_active.u_backend.g_bank[1].g_direct.u_acc.memory_command_write_data,
        u_active.u_backend.g_bank[0].g_direct.u_acc.memory_command_write_data
    };

    assign t_bank_update_enable = {
        u_tcfm5.bank_write_enable[4], u_tcfm5.bank_write_enable[3],
        u_tcfm5.bank_write_enable[2], u_tcfm5.bank_write_enable[1],
        u_tcfm5.bank_write_enable[0]
    };
    assign t_bank_update_ready = {
        u_tcfm5.bank_update_ready[4], u_tcfm5.bank_update_ready[3],
        u_tcfm5.bank_update_ready[2], u_tcfm5.bank_update_ready[1],
        u_tcfm5.bank_update_ready[0]
    };
    assign t_bank_update_addr = {
        u_tcfm5.bank_write_addr[4], u_tcfm5.bank_write_addr[3],
        u_tcfm5.bank_write_addr[2], u_tcfm5.bank_write_addr[1],
        u_tcfm5.bank_write_addr[0]
    };
    assign t_bank_command_valid = {
        u_tcfm5.g_acc[4].g_1rw.u_acc_bank.memory_command_valid,
        u_tcfm5.g_acc[3].g_1rw.u_acc_bank.memory_command_valid,
        u_tcfm5.g_acc[2].g_1rw.u_acc_bank.memory_command_valid,
        u_tcfm5.g_acc[1].g_1rw.u_acc_bank.memory_command_valid,
        u_tcfm5.g_acc[0].g_1rw.u_acc_bank.memory_command_valid
    };
    assign t_bank_command_write = {
        u_tcfm5.g_acc[4].g_1rw.u_acc_bank.memory_command_write,
        u_tcfm5.g_acc[3].g_1rw.u_acc_bank.memory_command_write,
        u_tcfm5.g_acc[2].g_1rw.u_acc_bank.memory_command_write,
        u_tcfm5.g_acc[1].g_1rw.u_acc_bank.memory_command_write,
        u_tcfm5.g_acc[0].g_1rw.u_acc_bank.memory_command_write
    };
    assign t_bank_command_addr = {
        u_tcfm5.g_acc[4].g_1rw.u_acc_bank.memory_command_addr,
        u_tcfm5.g_acc[3].g_1rw.u_acc_bank.memory_command_addr,
        u_tcfm5.g_acc[2].g_1rw.u_acc_bank.memory_command_addr,
        u_tcfm5.g_acc[1].g_1rw.u_acc_bank.memory_command_addr,
        u_tcfm5.g_acc[0].g_1rw.u_acc_bank.memory_command_addr
    };
    assign t_bank_command_write_data = {
        u_tcfm5.g_acc[4].g_1rw.u_acc_bank.memory_command_write_data,
        u_tcfm5.g_acc[3].g_1rw.u_acc_bank.memory_command_write_data,
        u_tcfm5.g_acc[2].g_1rw.u_acc_bank.memory_command_write_data,
        u_tcfm5.g_acc[1].g_1rw.u_acc_bank.memory_command_write_data,
        u_tcfm5.g_acc[0].g_1rw.u_acc_bank.memory_command_write_data
    };

`ifndef QFIT_EREP_BIND_V4
    qfit_local5_erep_direct_monitor_v4 #(
        .HEIGHT(HEIGHT), .WIDTH(WIDTH), .TIME_PLANES(TIME_PLANES),
        .HEAD_DIM(HEAD_DIM), .OUT_DIM(OUT_DIM), .GATE_W(GATE_W),
        .W_W(W_W), .ACC_W(ACC_W)
    ) u_direct_monitor (
        .clk_core(clk_core), .rst_core(rst_core),
        .weight_valid(d_weight_valid), .weight_ready(d_weight_ready),
        .weight_lane(d_weight_lane), .weight_out(d_weight_out),
        .weight_data(d_weight_data), .weight_last(d_weight_last),
        .projection_start(d_projection_start),
        .projection_close(d_projection_close),
        .projection_close_ready(d_projection_close_ready),
        .projection_busy(d_projection_busy), .projection_done(d_projection_done),
        .relation_start(d_relation_start), .relation_seal(d_relation_seal),
        .relation_active(d_relation_active), .relation_done(d_relation_done),
        .relation_valid(d_relation_valid), .relation_ready(d_relation_ready),
        .relation_plane(d_relation_plane),
        .relation_destination_y(d_relation_destination_y),
        .relation_destination_x(d_relation_destination_x),
        .relation_candidate_valid(d_relation_candidate_valid),
        .relation_active_candidate_mask(d_relation_active_candidate_mask),
        .relation_k_self(d_relation_k_self),
        .relation_direction_gates(d_relation_direction_gates),
        .relation_read_fire(u_active.u_relation_frontier.read_issue),
        .relation_read_source_id(u_active.u_relation_frontier.index_source_id),
        .relation_read_plane(u_active.u_relation_frontier.index_source_plane),
        .relation_read_y(u_active.u_relation_frontier.index_source_y),
        .relation_read_x(u_active.u_relation_frontier.index_source_x),
        .relation_read_last(u_active.u_relation_frontier.index_out_last),
        .descriptor_valid(u_active.builder_descriptor_valid),
        .descriptor_ready(u_active.builder_descriptor_ready),
        .descriptor_source_id(u_active.descriptor_source_id),
        .descriptor_plane(u_active.descriptor_plane),
        .descriptor_y(u_active.descriptor_y),
        .descriptor_x(u_active.descriptor_x),
        .descriptor_k(u_active.descriptor_k),
        .descriptor_gates(u_active.descriptor_gates),
        .descriptor_mask(u_active.descriptor_mask),
        .descriptor_last(u_active.descriptor_last),
        .fifo_enqueue(u_active.u_builder_fifo.enqueue),
        .fifo_dequeue(u_active.u_builder_fifo.dequeue),
        .fifo_count(u_active.u_builder_fifo.count_q),
        .fifo_head_source_id(
            u_active.u_builder_fifo.source_id_q[
                u_active.u_builder_fifo.read_ptr_q
            ]
        ),
        .fifo_head_plane(u_active.u_builder_fifo.plane_q[
            u_active.u_builder_fifo.read_ptr_q]),
        .fifo_head_y(u_active.u_builder_fifo.y_q[
            u_active.u_builder_fifo.read_ptr_q]),
        .fifo_head_x(u_active.u_builder_fifo.x_q[
            u_active.u_builder_fifo.read_ptr_q]),
        .term_valid(u_active.term_valid), .term_ready(u_active.term_ready),
        .term_source_id(u_active.term_source_id),
        .term_source_plane(u_active.term_source_plane),
        .term_source_y(u_active.term_source_y),
        .term_source_x(u_active.term_source_x),
        .term_lane(u_active.term_lane), .term_gate(u_active.term_gate),
        .term_destination_mask(u_active.term_destination_mask),
        .term_last(u_active.term_last),
        .term_source_last(u_active.term_source_last),
        .bank_update_enable(d_bank_update_enable),
        .bank_update_ready(d_bank_update_ready),
        .bank_update_addr(d_bank_update_addr),
        .bank_update_delta(u_active.u_backend.product_vector),
        .bank_command_valid(d_bank_command_valid),
        .bank_command_write(d_bank_command_write),
        .bank_command_addr(d_bank_command_addr),
        .bank_command_write_data(d_bank_command_write_data),
        .read_valid(d_read_valid), .read_ready(d_read_ready),
        .read_plane(d_read_plane), .read_y(d_read_y), .read_x(d_read_x),
        .read_out(d_read_out), .read_data_valid(d_read_data_valid),
        .read_data(d_read_data), .protocol_error(d_protocol_error)
    );

    qfit_local5_erep_tcfm5_monitor_v4 #(
        .HEIGHT(HEIGHT), .WIDTH(WIDTH), .TIME_PLANES(TIME_PLANES),
        .HEAD_DIM(HEAD_DIM), .OUT_DIM(OUT_DIM), .GATE_W(GATE_W),
        .W_W(W_W), .ACC_W(ACC_W)
    ) u_tcfm5_monitor (
        .clk_core(clk_core), .rst_core(rst_core),
        .weight_valid(t_weight_valid), .weight_ready(t_weight_ready),
        .weight_lane(t_weight_lane), .weight_out(t_weight_out),
        .weight_data(t_weight_data), .weight_last(t_weight_last),
        .run_start(t_run_start),
        .run_start_accepted(u_tcfm5.run_start_accepted),
        .run_busy(t_run_busy), .run_done(t_run_done),
        .state(u_tcfm5.state_q), .term_valid(t_term_valid),
        .term_ready(t_term_ready), .term_commit(u_tcfm5.term_commit),
        .term_source_plane(t_term_source_plane),
        .term_source_y(t_term_source_y), .term_source_x(t_term_source_x),
        .term_lane(t_term_lane), .term_gate(t_term_gate),
        .term_destination_mask(t_term_destination_mask),
        .term_window_last(t_term_window_last),
        .window_close(t_window_close),
        .window_close_ready(t_window_close_ready),
        .bank_update_enable(t_bank_update_enable),
        .bank_update_ready(t_bank_update_ready),
        .bank_update_addr(t_bank_update_addr),
        .bank_update_delta(u_tcfm5.product_vector),
        .bank_command_valid(t_bank_command_valid),
        .bank_command_write(t_bank_command_write),
        .bank_command_addr(t_bank_command_addr),
        .bank_command_write_data(t_bank_command_write_data),
        .read_valid(t_read_valid), .read_ready(t_read_ready),
        .read_plane(t_read_plane), .read_y(t_read_y), .read_x(t_read_x),
        .read_out(t_read_out), .read_data_valid(t_read_data_valid),
        .read_data(t_read_data),
        .vector_read_valid(t_vector_read_valid),
        .vector_read_ready(t_vector_read_ready),
        .vector_read_plane(t_vector_read_plane),
        .vector_read_y(t_vector_read_y),
        .vector_read_x(t_vector_read_x),
        .vector_read_data_valid(t_vector_read_data_valid),
        .vector_read_data(t_vector_read_data),
        .protocol_error(t_protocol_error)
    );

    qfit_local5_erep_serializer_monitor_v4 #(
        .HEIGHT(HEIGHT), .WIDTH(WIDTH), .TIME_PLANES(TIME_PLANES),
        .OUT_DIM(OUT_DIM), .ACC_W(ACC_W)
    ) u_serializer_monitor (
        .clk_core(clk_core), .rst_core(rst_core),
        .in_valid(t_vector_read_data_valid), .in_ready(ser_in_ready),
        .in_plane(ser_pending_plane_q), .in_y(ser_pending_y_q),
        .in_x(ser_pending_x_q), .in_data(t_vector_read_data),
        .in_last(ser_pending_last_q),
        .out_valid(ser_out_valid), .out_ready(ser_out_ready),
        .out_plane(ser_out_plane), .out_y(ser_out_y), .out_x(ser_out_x),
        .out_index(ser_out_index), .out_data(ser_out_data),
        .out_last(ser_out_last)
    );
`endif

    always #1 clk_core = ~clk_core;

    function automatic integer weight_value(
        input integer lane,
        input integer out
    );
        weight_value = (lane + 1) * (out == 0 ? 1 : -1);
    endfunction

    function automatic logic [HEAD_DIM-1:0] direct_k(
        input integer source
    );
        case (source)
            0: direct_k = 4'b0011;
            1: direct_k = 4'b0010;
            2: direct_k = 4'b1100;
            3: direct_k = 4'b1001;
            4: direct_k = 4'b0100;
            default: direct_k = 4'b0111;
        endcase
    endfunction

    task automatic load_direct_weight(
        input integer lane,
        input integer out,
        input logic last
    );
        begin
            @(negedge clk_core);
            d_weight_lane = LANE_W'(lane);
            d_weight_out = OUT_W'(out);
            d_weight_data = W_W'(weight_value(lane, out));
            d_weight_last = last;
            d_weight_valid = 1'b1;
            do @(posedge clk_core); while (!d_weight_ready);
            @(negedge clk_core);
            d_weight_valid = 1'b0;
            d_weight_last = 1'b0;
        end
    endtask

    task automatic load_tcfm5_weight(
        input integer lane,
        input integer out,
        input logic last
    );
        begin
            @(negedge clk_core);
            t_weight_lane = LANE_W'(lane);
            t_weight_out = OUT_W'(out);
            t_weight_data = W_W'(weight_value(lane, out));
            t_weight_last = last;
            t_weight_valid = 1'b1;
            do @(posedge clk_core); while (!t_weight_ready);
            @(negedge clk_core);
            t_weight_valid = 1'b0;
            t_weight_last = 1'b0;
        end
    endtask

    task automatic send_direct_relation(input integer source);
        logic [HEAD_DIM-1:0] k_value;
        integer out;
        integer lane;
        begin
            k_value = direct_k(source);
            @(negedge clk_core);
            d_relation_plane = '0;
            d_relation_destination_y = Y_W'(source / WIDTH);
            d_relation_destination_x = X_W'(source % WIDTH);
            d_relation_candidate_valid = 5'b00001;
            d_relation_active_candidate_mask = 5'b00001;
            d_relation_k_self = k_value;
            d_relation_direction_gates = '0;
            d_relation_direction_gates[0 +: GATE_W] = GATE_W'(source + 1);
            d_relation_valid = 1'b1;
            do @(posedge clk_core); while (!d_relation_ready);
            @(negedge clk_core);
            d_relation_valid = 1'b0;

            for (out = 0; out < OUT_DIM; out = out + 1)
                for (lane = 0; lane < HEAD_DIM; lane = lane + 1)
                    if (k_value[lane])
                        expected_direct[source][out] =
                            expected_direct[source][out]
                            + (source + 1) * weight_value(lane, out);
        end
    endtask

    task automatic check_direct(
        input integer source,
        input integer out
    );
        begin
            @(negedge clk_core);
            d_read_plane = '0;
            d_read_y = Y_W'(source / WIDTH);
            d_read_x = X_W'(source % WIDTH);
            d_read_out = OUT_W'(out);
            d_read_valid = 1'b1;
            do @(posedge clk_core); while (!d_read_ready);
            @(negedge clk_core);
            d_read_valid = 1'b0;
            wait (d_read_data_valid);
            #1;
            if ($signed(d_read_data) != expected_direct[source][out])
                $fatal(1,
                    "Direct mismatch source=%0d out=%0d got=%0d expected=%0d",
                    source, out, $signed(d_read_data),
                    expected_direct[source][out]);
        end
    endtask

    function automatic integer role_source(
        input integer source_y,
        input integer source_x,
        input integer role
    );
        integer y;
        integer x;
        begin
            y = source_y;
            x = source_x;
            case (role)
                1: y = source_y + 1;
                2: y = source_y - 1;
                3: x = source_x + 1;
                4: x = source_x - 1;
                default: begin end
            endcase
            if (y < 0 || y >= HEIGHT || x < 0 || x >= WIDTH)
                role_source = -1;
            else
                role_source = y * WIDTH + x;
        end
    endfunction

    task automatic send_tcfm5_term(
        input integer source_y,
        input integer source_x,
        input integer lane,
        input integer gate,
        input logic [4:0] mask,
        input logic last
    );
        integer role;
        integer source;
        integer out;
        begin
            t_term_source_plane = '0;
            t_term_source_y = Y_W'(source_y);
            t_term_source_x = X_W'(source_x);
            t_term_lane = LANE_W'(lane);
            t_term_gate = GATE_W'(gate);
            t_term_destination_mask = mask;
            t_term_window_last = last;
            t_term_valid = 1'b1;
            do @(posedge clk_core); while (!t_term_ready);
            @(negedge clk_core);
            t_term_valid = 1'b0;
            t_term_window_last = 1'b0;

            for (role = 0; role < 5; role = role + 1) begin
                if (mask[role]) begin
                    source = role_source(source_y, source_x, role);
                    if (source < 0)
                        $fatal(1, "TCFM5 fixture generated an invalid role");
                    for (out = 0; out < OUT_DIM; out = out + 1)
                        expected_tcfm5[source][out] =
                            expected_tcfm5[source][out]
                            + gate * weight_value(lane, out);
                end
            end
        end
    endtask

    task automatic request_tcfm5_vector(
        input integer source,
        input logic last
    );
        integer checks_before;
        begin
            checks_before = serializer_checks;
            @(negedge clk_core);
            ser_out_ready = 1'b0;
            t_vector_read_plane = '0;
            t_vector_read_y = Y_W'(source / WIDTH);
            t_vector_read_x = X_W'(source % WIDTH);
            t_vector_read_valid = 1'b1;
            do @(posedge clk_core); while (!t_vector_read_ready);
            @(negedge clk_core);
            t_vector_read_valid = 1'b0;
            wait (t_vector_read_data_valid);
            if (!ser_in_ready)
                $fatal(1, "serializer was not ready for atomic vector response");
            @(posedge clk_core);
            wait (ser_out_valid);
            repeat (2) @(posedge clk_core);
            @(negedge clk_core);
            ser_out_ready = 1'b1;
            wait (serializer_checks == checks_before + OUT_DIM);
            if (last)
                wait (serializer_last_checks == 1);
        end
    endtask

    always @(posedge clk_core) begin
        if (rst_core) begin
            ser_pending_plane_q <= '0;
            ser_pending_y_q <= '0;
            ser_pending_x_q <= '0;
            ser_pending_last_q <= 1'b0;
        end else if (t_vector_read_valid && t_vector_read_ready) begin
            ser_pending_plane_q <= t_vector_read_plane;
            ser_pending_y_q <= t_vector_read_y;
            ser_pending_x_q <= t_vector_read_x;
            ser_pending_last_q <=
                integer'(t_vector_read_y) == HEIGHT - 1
                && integer'(t_vector_read_x) == WIDTH - 1;
        end
    end

    always @(posedge clk_core) begin
        integer source;
        if (!rst_core && ser_out_valid && ser_out_ready) begin
            source = integer'(ser_out_y) * WIDTH + integer'(ser_out_x);
            if ($signed(ser_out_data)
                != expected_tcfm5[source][integer'(ser_out_index)])
                $fatal(1,
                    "TCFM5 serializer mismatch source=%0d out=%0d got=%0d expected=%0d",
                    source, ser_out_index, $signed(ser_out_data),
                    expected_tcfm5[source][integer'(ser_out_index)]);
            serializer_checks <= serializer_checks + 1;
            if (ser_out_last)
                serializer_last_checks <= serializer_last_checks + 1;
        end
    end

    initial begin : stimulus
        integer source;
        integer lane;
        integer out;

        rst_core = 1'b1;
        d_weight_valid = 1'b0;
        d_weight_lane = '0;
        d_weight_out = '0;
        d_weight_data = '0;
        d_weight_last = 1'b0;
        d_projection_start = 1'b0;
        d_projection_close = 1'b0;
        d_relation_start = 1'b0;
        d_relation_seal = 1'b0;
        d_relation_valid = 1'b0;
        d_relation_plane = '0;
        d_relation_destination_y = '0;
        d_relation_destination_x = '0;
        d_relation_candidate_valid = '0;
        d_relation_active_candidate_mask = '0;
        d_relation_k_self = '0;
        d_relation_direction_gates = '0;
        d_read_valid = 1'b0;
        d_read_plane = '0;
        d_read_y = '0;
        d_read_x = '0;
        d_read_out = '0;

        t_weight_valid = 1'b0;
        t_weight_lane = '0;
        t_weight_out = '0;
        t_weight_data = '0;
        t_weight_last = 1'b0;
        t_weight_context_release = 1'b0;
        t_run_start = 1'b0;
        t_run_accumulate = 1'b0;
        t_term_valid = 1'b0;
        t_term_source_plane = '0;
        t_term_source_y = '0;
        t_term_source_x = '0;
        t_term_lane = '0;
        t_term_gate = '0;
        t_term_destination_mask = '0;
        t_term_window_last = 1'b0;
        t_window_close = 1'b0;
        t_read_valid = 1'b0;
        t_read_plane = '0;
        t_read_y = '0;
        t_read_x = '0;
        t_read_out = '0;
        t_vector_read_valid = 1'b0;
        t_vector_read_plane = '0;
        t_vector_read_y = '0;
        t_vector_read_x = '0;
        ser_out_ready = 1'b1;
        serializer_checks = 0;
        serializer_last_checks = 0;

        for (source = 0; source < SOURCES; source = source + 1)
            for (out = 0; out < OUT_DIM; out = out + 1) begin
                expected_direct[source][out] = 0;
                expected_tcfm5[source][out] = 0;
            end

        repeat (5) @(negedge clk_core);
        rst_core = 1'b0;

        for (lane = 0; lane < HEAD_DIM; lane = lane + 1)
            for (out = 0; out < OUT_DIM; out = out + 1)
                load_direct_weight(lane, out,
                    lane == HEAD_DIM - 1 && out == OUT_DIM - 1);
        for (lane = 0; lane < HEAD_DIM; lane = lane + 1)
            for (out = 0; out < OUT_DIM; out = out + 1)
                load_tcfm5_weight(lane, out,
                    lane == HEAD_DIM - 1 && out == OUT_DIM - 1);

        @(negedge clk_core);
        d_projection_start = 1'b1;
        d_relation_start = 1'b1;
        @(negedge clk_core);
        d_projection_start = 1'b0;
        d_relation_start = 1'b0;
        for (source = 0; source < SOURCES; source = source + 1)
            send_direct_relation(source);
        @(negedge clk_core);
        d_relation_seal = 1'b1;
        @(negedge clk_core);
        d_relation_seal = 1'b0;
        wait (d_projection_close_ready);
        @(negedge clk_core);
        d_projection_close = 1'b1;
        @(negedge clk_core);
        d_projection_close = 1'b0;
        wait (d_projection_done);
        for (source = 0; source < SOURCES; source = source + 1)
            for (out = 0; out < OUT_DIM; out = out + 1)
                check_direct(source, out);

        if (d_perf_relation_writes != SOURCES
            || d_perf_active_source_reads != SOURCES
            || d_perf_descriptors != SOURCES
            || d_perf_product_terms != 20
            || d_perf_destination_updates != 20)
            $fatal(1,
                "Direct counter mismatch relation=%0d active=%0d descriptors=%0d terms=%0d updates=%0d",
                d_perf_relation_writes, d_perf_active_source_reads,
                d_perf_descriptors, d_perf_product_terms,
                d_perf_destination_updates);

        @(negedge clk_core);
        t_run_start = 1'b1;
        @(negedge clk_core);
        t_run_start = 1'b0;
        wait (t_term_ready);
        @(negedge clk_core);
        send_tcfm5_term(0, 1, 0, 3, 5'b11011, 1'b0);
        send_tcfm5_term(1, 1, 1, 2, 5'b11101, 1'b0);
        send_tcfm5_term(0, 0, 2, 4, 5'b01011, 1'b1);
        wait (t_run_done);
        for (source = 0; source < SOURCES; source = source + 1)
            request_tcfm5_vector(source, source == SOURCES - 1);

        if (d_protocol_error || t_protocol_error)
            $fatal(1, "unexpected protocol error at fixture completion");
        if (t_perf_product_terms != 3 || t_perf_destination_updates != 11)
            $fatal(1,
                "TCFM5 counter mismatch terms=%0d updates=%0d",
                t_perf_product_terms, t_perf_destination_updates);
        if (serializer_checks != SOURCES * OUT_DIM)
            $fatal(1, "serializer output count mismatch got=%0d", serializer_checks);
        if (serializer_last_checks != 1)
            $fatal(1, "serializer last count mismatch got=%0d",
                serializer_last_checks);

        $display("PASS Local5 EREP calibration v4 direct_terms=%0d direct_updates=%0d tcfm5_terms=%0d tcfm5_updates=%0d serializer_outputs=%0d",
            d_perf_product_terms, d_perf_destination_updates,
            t_perf_product_terms, t_perf_destination_updates,
            serializer_checks);
        $finish;
    end

    initial begin
        repeat (100000) @(posedge clk_core);
        $fatal(1, "Local5 EREP calibration v4 timeout");
    end
endmodule

`default_nettype wire
