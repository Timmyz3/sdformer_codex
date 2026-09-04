`timescale 1ns/1ps
`default_nettype none

// FCSR relation descriptors are consumed live for the first output tile and
// replayed from the exact relation vault for later tiles. Both paths share the
// same source-major term builder and TCFM-5 accumulator backend.
module qfit_fcsr_relation_memo_projection_top #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int HEAD_DIM = 32,
    parameter int OUT_DIM = 4,
    parameter int GATE_W = 9,
    parameter int W_W = 8,
    parameter int ACC_W = 32,
    parameter bit ENABLE_VECTOR_READ = 1'b0,
    parameter int ACC_BACKEND_KIND = 0,
    parameter int ACC_MEMORY_IMPL = 0,
    parameter int MAX_HEADS = 24,
    parameter int SOURCE_ID_W =
        (HEIGHT * WIDTH * TIME_PLANES <= 1)
        ? 1 : $clog2(HEIGHT * WIDTH * TIME_PLANES),
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int PLANE_W =
        (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int HEAD_W = (MAX_HEADS <= 1) ? 1 : $clog2(MAX_HEADS),
    parameter int PTR_W = $clog2(513),
    parameter int LANE_W =
        (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM),
    parameter int OUT_W = (OUT_DIM <= 1) ? 1 : $clog2(OUT_DIM)
) (
    input  logic                       clk_core,
    input  logic                       rst_core,

    input  logic                       window_start,
    input  logic                       head_start,
    output logic                       head_ready,
    input  logic [HEAD_W-1:0]          head_index,
    output logic                       head_done,
    output logic                       head_resident,
    output logic                       head_critical,
    output logic                       head_overflow,
    output logic [31:0]                head_service_cycles,
    output logic [PTR_W-1:0]           head_record_count,

    input  logic                       plane_start,
    input  logic [PLANE_W-1:0]         plane_id,
    input  logic                       in_valid,
    output logic                       in_ready,
    input  logic [Y_W-1:0]             in_y,
    input  logic [X_W-1:0]             in_x,
    input  logic [4:0]                 in_candidate_valid,
    input  logic [HEAD_DIM-1:0]        in_k_self,
    input  logic [5*GATE_W-1:0]        in_direction_gates,
    output logic                       plane_idle,

    input  logic                       use_replay,
    input  logic                       replay_start,
    output logic                       replay_cmd_ready,
    input  logic [HEAD_W-1:0]          replay_head_index,
    output logic                       replay_done,
    output logic                       replay_miss,

    input  logic                       weight_valid,
    output logic                       weight_ready,
    input  logic [LANE_W-1:0]          weight_lane,
    input  logic [OUT_W-1:0]           weight_out,
    input  logic signed [W_W-1:0]      weight_data,
    input  logic                       weight_last,
    input  logic                       weight_context_release,
    output logic                       weight_context_release_ready,
    input  logic                       projection_start,
    input  logic                       projection_accumulate,
    input  logic                       projection_close,
    output logic                       projection_close_ready,
    output logic                       projection_busy,
    output logic                       projection_done,

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

    output logic                       descriptor_valid,
    output logic                       descriptor_ready,
    output logic [SOURCE_ID_W-1:0]     descriptor_source_id,
    output logic [Y_W-1:0]             descriptor_y,
    output logic [X_W-1:0]             descriptor_x,
    output logic [HEAD_DIM-1:0]        descriptor_k,
    output logic [5*GATE_W-1:0]        descriptor_gates,
    output logic [4:0]                 descriptor_valid_mask,
    output logic                       descriptor_last,
    output logic                       descriptor_stream_idle,

    output logic                       protocol_error,
    output logic [31:0]                perf_speculative_writes,
    output logic [31:0]                perf_discarded_writes,
    output logic [31:0]                perf_committed_records,
    output logic [31:0]                perf_replay_reads,
    output logic [31:0]                perf_capacity_misses,
    output logic [31:0]                perf_descriptors,
    output logic [31:0]                perf_product_terms,
    output logic [31:0]                perf_destination_updates
);
    localparam int TOKENS_PER_PLANE = HEIGHT * WIDTH;

    logic live_valid;
    logic live_ready;
    logic [SOURCE_ID_W-1:0] live_source_id;
    logic [Y_W-1:0] live_y;
    logic [X_W-1:0] live_x;
    logic [HEAD_DIM-1:0] live_k;
    logic [5*GATE_W-1:0] live_gates;
    logic [4:0] live_valid_mask;
    logic live_last;
    logic replay_valid;
    logic replay_ready;
    logic [SOURCE_ID_W-1:0] replay_source_id;
    logic [Y_W-1:0] replay_y;
    logic [X_W-1:0] replay_x;
    logic [HEAD_DIM-1:0] replay_k;
    logic [5*GATE_W-1:0] replay_gates;
    logic [4:0] replay_valid_mask;
    logic replay_last;
    logic memo_protocol_error;

    logic term_valid;
    logic term_ready;
    logic [SOURCE_ID_W-1:0] term_source_id;
    logic [Y_W-1:0] term_source_y;
    logic [X_W-1:0] term_source_x;
    logic [LANE_W-1:0] term_lane;
    logic [GATE_W-1:0] term_gate;
    logic [4:0] term_destination_mask;
    logic term_last;
    logic [PLANE_W-1:0] term_source_plane;
    logic backend_close_ready;
    logic backend_protocol_error;
    logic close_protocol_error_q;
    logic [31:0] unused_builder_terms;
    logic [31:0] unused_builder_updates;

    qfit_fcsr_relation_memo_top #(
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .HEAD_DIM(HEAD_DIM),
        .GATE_W(GATE_W),
        .MAX_HEADS(MAX_HEADS)
    ) u_relation_memo (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start(window_start),
        .head_start(head_start),
        .head_ready(head_ready),
        .head_index(head_index),
        .plane_start(plane_start),
        .plane_id(plane_id),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_y(in_y),
        .in_x(in_x),
        .in_candidate_valid(in_candidate_valid),
        .in_k_self(in_k_self),
        .in_direction_gates(in_direction_gates),
        .live_valid(live_valid),
        .live_ready(live_ready),
        .live_source_id(live_source_id),
        .live_y(live_y),
        .live_x(live_x),
        .live_k(live_k),
        .live_gates(live_gates),
        .live_valid_mask(live_valid_mask),
        .live_last(live_last),
        .head_done(head_done),
        .head_resident(head_resident),
        .head_critical(head_critical),
        .head_overflow(head_overflow),
        .head_service_cycles(head_service_cycles),
        .head_record_count(head_record_count),
        .replay_start(replay_start),
        .replay_cmd_ready(replay_cmd_ready),
        .replay_head_index(replay_head_index),
        .replay_valid(replay_valid),
        .replay_ready(replay_ready),
        .replay_source_id(replay_source_id),
        .replay_y(replay_y),
        .replay_x(replay_x),
        .replay_k(replay_k),
        .replay_gates(replay_gates),
        .replay_valid_mask(replay_valid_mask),
        .replay_last(replay_last),
        .replay_done(replay_done),
        .replay_miss(replay_miss),
        .plane_idle(plane_idle),
        .protocol_error(memo_protocol_error),
        .perf_speculative_writes(perf_speculative_writes),
        .perf_discarded_writes(perf_discarded_writes),
        .perf_committed_records(perf_committed_records),
        .perf_replay_reads(perf_replay_reads),
        .perf_capacity_misses(perf_capacity_misses)
    );

    assign descriptor_valid = use_replay ? replay_valid : live_valid;
    assign descriptor_source_id =
        use_replay ? replay_source_id : live_source_id;
    assign descriptor_y = use_replay ? replay_y : live_y;
    assign descriptor_x = use_replay ? replay_x : live_x;
    assign descriptor_k = use_replay ? replay_k : live_k;
    assign descriptor_gates = use_replay ? replay_gates : live_gates;
    assign descriptor_valid_mask =
        use_replay ? replay_valid_mask : live_valid_mask;
    assign descriptor_last = use_replay ? replay_last : live_last;
    assign live_ready = !use_replay && descriptor_ready;
    assign replay_ready = use_replay && descriptor_ready;

    qfit_source_multicast_term_builder #(
        .HEAD_DIM(HEAD_DIM),
        .GATE_W(GATE_W),
        .SOURCE_ID_W(SOURCE_ID_W),
        .Y_W(Y_W),
        .X_W(X_W)
    ) u_term_builder (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .descriptor_valid(descriptor_valid),
        .descriptor_ready(descriptor_ready),
        .descriptor_source_id(descriptor_source_id),
        .descriptor_y(descriptor_y),
        .descriptor_x(descriptor_x),
        .descriptor_k(descriptor_k),
        .descriptor_incoming_gates(descriptor_gates),
        .descriptor_valid_mask(descriptor_valid_mask),
        .term_valid(term_valid),
        .term_ready(term_ready),
        .term_source_id(term_source_id),
        .term_source_y(term_source_y),
        .term_source_x(term_source_x),
        .term_lane(term_lane),
        .term_gate(term_gate),
        .term_destination_mask(term_destination_mask),
        .term_last(term_last),
        .perf_descriptors(perf_descriptors),
        .perf_terms(unused_builder_terms),
        .perf_destination_updates(unused_builder_updates)
    );

    always_comb begin
        term_source_plane = '0;
        for (integer plane = 1; plane < TIME_PLANES; plane = plane + 1)
            if (32'(term_source_id) >= plane * TOKENS_PER_PLANE)
                term_source_plane = PLANE_W'(plane);
    end

    qfit_tcfm5_projection_top #(
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .HEAD_DIM(HEAD_DIM),
        .OUT_DIM(OUT_DIM),
        .GATE_W(GATE_W),
        .W_W(W_W),
        .ACC_W(ACC_W),
        .ENABLE_VECTOR_READ(ENABLE_VECTOR_READ),
        .ACC_BACKEND_KIND(ACC_BACKEND_KIND),
        .ACC_MEMORY_IMPL(ACC_MEMORY_IMPL)
    ) u_projection (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .weight_valid(weight_valid),
        .weight_ready(weight_ready),
        .weight_lane(weight_lane),
        .weight_out(weight_out),
        .weight_data(weight_data),
        .weight_last(weight_last),
        .weight_context_release(weight_context_release),
        .weight_context_release_ready(weight_context_release_ready),
        .run_start(projection_start),
        .run_accumulate(projection_accumulate),
        .run_busy(projection_busy),
        .run_done(projection_done),
        .term_valid(term_valid),
        .term_ready(term_ready),
        .term_source_plane(term_source_plane),
        .term_source_y(term_source_y),
        .term_source_x(term_source_x),
        .term_lane(term_lane),
        .term_gate(term_gate),
        .term_destination_mask(term_destination_mask),
        .term_product('0),
        .term_window_last(1'b0),
        .window_close(projection_close && projection_close_ready),
        .window_close_ready(backend_close_ready),
        .read_valid(read_valid),
        .read_ready(read_ready),
        .read_plane(read_plane),
        .read_y(read_y),
        .read_x(read_x),
        .read_out(read_out),
        .read_data_valid(read_data_valid),
        .read_data(read_data),
        .vector_read_valid(vector_read_valid),
        .vector_read_ready(vector_read_ready),
        .vector_read_plane(vector_read_plane),
        .vector_read_y(vector_read_y),
        .vector_read_x(vector_read_x),
        .vector_read_data_valid(vector_read_data_valid),
        .vector_read_data(vector_read_data),
        .protocol_error(backend_protocol_error),
        .perf_product_terms(perf_product_terms),
        .perf_destination_updates(perf_destination_updates)
    );

    assign descriptor_stream_idle = descriptor_ready
                                  && !descriptor_valid
                                  && !term_valid;
    assign projection_close_ready = descriptor_stream_idle
                                  && backend_close_ready;
    assign protocol_error = memo_protocol_error
                         || backend_protocol_error
                         || close_protocol_error_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            close_protocol_error_q <= 1'b0;
        end else begin
            if (projection_start)
                close_protocol_error_q <= 1'b0;
            if (projection_close && !projection_close_ready)
                close_protocol_error_q <= 1'b1;
            if (use_replay && live_valid)
                close_protocol_error_q <= 1'b1;
            if (!use_replay && replay_valid)
                close_protocol_error_q <= 1'b1;
        end
    end
endmodule

`default_nettype wire
