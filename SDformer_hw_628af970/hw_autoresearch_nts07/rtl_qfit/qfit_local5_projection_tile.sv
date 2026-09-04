`timescale 1ns/1ps
`default_nettype none

// End-to-end Local5 architecture slice:
// XBF-DBDR score -> FCSR-RX transpose -> gate-equivalence terms
// -> topology-colored conflict-free projection.
module qfit_local5_projection_tile #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int HEAD_DIM = 32,
    parameter int OUT_DIM = 4,
    parameter int GATE_W = 9,
    parameter int W_W = 8,
    parameter int ACC_W = 32,
    parameter bit ENABLE_VECTOR_READ = 1'b0,
    parameter int TAG_W = 16,
    parameter int ACC_BACKEND_KIND = 0,
    parameter int ACC_MEMORY_IMPL = 0,
    // 0: TCFM-5, 1: Affine-4, 2: Linear-5, 3: Role-Sharded.
    parameter int BACKEND_KIND = 0,
    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),
    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),
    parameter int PLANE_W =
        (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),
    parameter int LANE_W =
        (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM),
    parameter int OUT_W =
        (OUT_DIM <= 1) ? 1 : $clog2(OUT_DIM),
    parameter int SOURCE_ID_W =
        (HEIGHT * WIDTH * TIME_PLANES <= 1)
        ? 1 : $clog2(HEIGHT * WIDTH * TIME_PLANES)
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
    input  logic                       projection_start,
    input  logic                       projection_accumulate,
    output logic                       projection_start_ready,
    input  logic                       projection_close,
    // Scheduler-visible pause for the term boundary. Both valid and ready
    // are gated so a held builder term cannot be consumed by the backend.
    input  logic                       term_issue_enable,
    output logic                       projection_close_ready,
    output logic                       projection_busy,
    output logic                       projection_done,
    output logic                       stream_idle,

    input  logic                       plane_start,
    input  logic [PLANE_W-1:0]         plane_id,
    output logic                       plane_start_ready,
    input  logic                       in_valid,
    output logic                       in_ready,
    input  logic [Y_W-1:0]             in_y,
    input  logic [X_W-1:0]             in_x,
    input  logic [HEAD_DIM-1:0]        in_q,
    input  logic [5*HEAD_DIM-1:0]      in_k,
    input  logic [4:0]                 in_valid_mask,

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
    output logic [31:0]                perf_descriptors,
    output logic [31:0]                perf_product_terms,
    output logic [31:0]                perf_destination_updates,
    output logic [31:0]                perf_relation_stalls
);
    localparam int TOKENS_PER_PLANE = HEIGHT * WIDTH;
    localparam int TOTAL_TOKENS = TIME_PLANES * TOKENS_PER_PLANE;
    localparam int TOTAL_WEIGHTS = HEAD_DIM * OUT_DIM;
    localparam int PLANE_COUNT_W = $clog2(TIME_PLANES + 1);
    localparam int TOKEN_COUNT_W = $clog2(TOKENS_PER_PLANE + 1);
    localparam int DESCRIPTOR_COUNT_W = $clog2(TOTAL_TOKENS + 1);
    localparam int WEIGHT_COUNT_W = $clog2(TOTAL_WEIGHTS + 1);

    initial begin
        if (HEAD_DIM != 32)
            $error("qfit_local5_tile currently requires HEAD_DIM=32");
    end

    wire descriptor_valid;
    wire descriptor_ready;
    wire builder_descriptor_ready;
    wire [SOURCE_ID_W-1:0] descriptor_source_id;
    wire [Y_W-1:0] descriptor_y;
    wire [X_W-1:0] descriptor_x;
    wire [HEAD_DIM-1:0] descriptor_k;
    wire [5*GATE_W-1:0] descriptor_incoming_gates;
    wire [4:0] descriptor_valid_mask;
    logic [15:0] unused_score_cycles;
    logic [3:0] unused_score_direct_mask;
    logic [2:0] unused_relation_max_pending;
    logic local_plane_start_ready;
    logic local_in_ready;

    logic term_valid;
    logic term_ready;
    logic backend_term_ready;
    logic backend_weight_ready;
    logic backend_weight_valid;
    logic [SOURCE_ID_W-1:0] term_source_id;
    logic [Y_W-1:0] term_source_y;
    logic [X_W-1:0] term_source_x;
    logic [LANE_W-1:0] term_lane;
    logic [GATE_W-1:0] term_gate;
    logic [4:0] term_destination_mask;
    logic term_descriptor_last;
    logic term_run_last;
    logic [PLANE_W-1:0] term_source_plane;
    logic [31:0] unused_builder_terms;
    logic [31:0] unused_builder_updates;
    logic [31:0] unused_builder_descriptors;
    logic [31:0] backend_terms;
    logic [31:0] backend_updates;
    logic backend_protocol_error;
    logic backend_close_ready;
    logic [31:0] unused_backend_replays;
    logic run_protocol_error_q;
    logic weight_protocol_error_q;
    logic weights_loaded_q;
    logic [WEIGHT_COUNT_W-1:0] weight_count_q;
    logic [TOTAL_WEIGHTS-1:0] weight_seen_q;
    logic run_active_q;
    logic plane_open_q;
    logic [PLANE_COUNT_W-1:0] planes_completed_q;
    logic [TOKEN_COUNT_W-1:0] plane_tokens_q;
    logic [DESCRIPTOR_COUNT_W-1:0] run_descriptors_q;
    logic [TOTAL_TOKENS-1:0] descriptor_seen_q;
    logic [Y_W-1:0] expected_y_q;
    logic [X_W-1:0] expected_x_q;
    logic [PLANE_W-1:0] expected_plane_id;
    logic projection_start_fire;
    logic plane_start_fire;
    logic input_fire;
    logic descriptor_fire;
    logic descriptor_attempt;
    logic plane_request_legal;
    logic input_request_legal;
    logic descriptor_contract_valid;
    logic weight_request_legal;
    logic weight_fire;
    logic backend_weight_context_release_ready;
    logic weight_context_release_fire;

    always_comb begin
        term_source_plane = '0;
        for (
            integer plane = 1;
            plane < TIME_PLANES;
            plane = plane + 1
        ) begin
            if (
                32'(term_source_id)
                >= plane * TOKENS_PER_PLANE
            )
                term_source_plane = PLANE_W'(plane);
        end
    end

    assign expected_plane_id = PLANE_W'(planes_completed_q);
    assign projection_start_ready = weights_loaded_q
                                  && !weight_protocol_error_q
                                  && !run_active_q
                                  && !projection_busy
                                  && (backend_weight_ready || projection_done);
    assign projection_start_fire = projection_start
                                && projection_start_ready;
    assign plane_request_legal = run_active_q
                              && !plane_open_q
                              && 32'(planes_completed_q) < TIME_PLANES
                              && plane_id == expected_plane_id;
    assign plane_start_ready = plane_request_legal
                            && local_plane_start_ready;
    assign plane_start_fire = plane_start && plane_start_ready;
    assign input_request_legal = run_active_q
                              && plane_open_q
                              && in_y == expected_y_q
                              && in_x == expected_x_q;
    assign in_ready = input_request_legal && local_in_ready;
    assign input_fire = in_valid && in_ready;
    assign descriptor_ready = builder_descriptor_ready
                            && descriptor_contract_valid;
    assign descriptor_attempt = descriptor_valid
                              && builder_descriptor_ready;
    assign descriptor_fire = descriptor_valid && descriptor_ready;
    assign backend_weight_valid = weight_valid && weight_request_legal;
    assign weight_ready = backend_weight_ready && weight_request_legal;
    assign weight_fire = weight_valid && weight_ready;
    assign weight_context_release_ready = !run_active_q
        && projection_done
        && backend_weight_context_release_ready;
    assign weight_context_release_fire = weight_context_release
        && weight_context_release_ready;
    assign term_run_last = term_descriptor_last
                        && 32'(run_descriptors_q) == TOTAL_TOKENS;
    always_comb begin
        weight_request_legal = 1'b0;
        for (integer lane = 0; lane < HEAD_DIM; lane = lane + 1) begin
            for (integer out = 0; out < OUT_DIM; out = out + 1) begin
                if (
                    weight_lane == LANE_W'(lane)
                    && weight_out == OUT_W'(out)
                ) begin
                    weight_request_legal = !weights_loaded_q
                        && !weight_seen_q[lane * OUT_DIM + out]
                        && weight_last
                           == (
                               32'(weight_count_q)
                               == TOTAL_WEIGHTS - 1
                              );
                end
            end
        end
    end
    always_comb begin
        descriptor_contract_valid = 1'b0;
        for (integer sid = 0; sid < TOTAL_TOKENS; sid = sid + 1) begin
            if (descriptor_source_id == SOURCE_ID_W'(sid)) begin
                descriptor_contract_valid = !descriptor_seen_q[sid]
                    && descriptor_y
                       == Y_W'((sid % TOKENS_PER_PLANE) / WIDTH)
                    && descriptor_x
                       == X_W'((sid % TOKENS_PER_PLANE) % WIDTH);
            end
        end
    end

    qfit_local5_tile #(
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .TAG_W(TAG_W),
        .GATE_W(GATE_W)
    ) u_local5 (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .plane_start(plane_start_fire),
        .plane_id(expected_plane_id),
        .plane_start_ready(local_plane_start_ready),
        .in_valid(in_valid && input_request_legal),
        .in_ready(local_in_ready),
        .in_y(in_y),
        .in_x(in_x),
        .in_q(in_q),
        .in_k(in_k),
        .in_valid_mask(in_valid_mask),
        .descriptor_valid(descriptor_valid),
        .descriptor_ready(descriptor_ready),
        .descriptor_source_id(descriptor_source_id),
        .descriptor_y(descriptor_y),
        .descriptor_x(descriptor_x),
        .descriptor_k(descriptor_k),
        .descriptor_incoming_gates(descriptor_incoming_gates),
        .descriptor_valid_mask(descriptor_valid_mask),
        .perf_score_service_cycles(unused_score_cycles),
        .perf_score_direct_mask(unused_score_direct_mask),
        .perf_relation_stalls(perf_relation_stalls),
        .perf_relation_max_pending(unused_relation_max_pending)
    );

    qfit_source_multicast_term_builder #(
        .HEAD_DIM(HEAD_DIM),
        .GATE_W(GATE_W),
        .SOURCE_ID_W(SOURCE_ID_W),
        .Y_W(Y_W),
        .X_W(X_W)
    ) u_term_builder (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .descriptor_valid(
            descriptor_valid && descriptor_contract_valid
        ),
        .descriptor_ready(builder_descriptor_ready),
        .descriptor_source_id(descriptor_source_id),
        .descriptor_y(descriptor_y),
        .descriptor_x(descriptor_x),
        .descriptor_k(descriptor_k),
        .descriptor_incoming_gates(descriptor_incoming_gates),
        .descriptor_valid_mask(descriptor_valid_mask),
        .term_valid(term_valid),
        .term_ready(term_ready),
        .term_source_id(term_source_id),
        .term_source_y(term_source_y),
        .term_source_x(term_source_x),
        .term_lane(term_lane),
        .term_gate(term_gate),
        .term_destination_mask(term_destination_mask),
        .term_last(term_descriptor_last),
        .perf_descriptors(unused_builder_descriptors),
        .perf_terms(unused_builder_terms),
        .perf_destination_updates(unused_builder_updates)
    );

    generate
        if (BACKEND_KIND == 0) begin : g_tcfm5_backend
            assign unused_backend_replays = '0;
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
                .weight_valid(backend_weight_valid),
                .weight_ready(backend_weight_ready),
                .weight_lane(weight_lane),
                .weight_out(weight_out),
                .weight_data(weight_data),
                .weight_last(weight_last),
                .weight_context_release(weight_context_release_fire),
                .weight_context_release_ready(
                    backend_weight_context_release_ready
                ),
                .run_start(projection_start_fire),
                .run_accumulate(projection_accumulate),
                .run_busy(projection_busy),
                .run_done(projection_done),
                .term_valid(term_valid && term_issue_enable),
                .term_ready(backend_term_ready),
                .term_source_plane(term_source_plane),
                .term_source_y(term_source_y),
                .term_source_x(term_source_x),
                .term_lane(term_lane),
                .term_gate(term_gate),
                .term_destination_mask(term_destination_mask),
                .term_product('0),
                .term_window_last(1'b0),
                .window_close(
                    projection_close && projection_close_ready
                ),
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
                .perf_product_terms(backend_terms),
                .perf_destination_updates(backend_updates)
            );
        end else if (BACKEND_KIND == 1) begin : g_affine4_backend
            assign backend_weight_context_release_ready = 1'b0;
            assign vector_read_ready = 1'b0;
            assign vector_read_data_valid = 1'b0;
            assign vector_read_data = '0;
            qfit_affine4_projection_top #(
                .HEIGHT(HEIGHT),
                .WIDTH(WIDTH),
                .TIME_PLANES(TIME_PLANES),
                .HEAD_DIM(HEAD_DIM),
                .OUT_DIM(OUT_DIM),
                .GATE_W(GATE_W),
                .W_W(W_W),
                .ACC_W(ACC_W)
            ) u_projection (
                .clk_core(clk_core),
                .rst_core(rst_core),
                .weight_valid(backend_weight_valid),
                .weight_ready(backend_weight_ready),
                .weight_lane(weight_lane),
                .weight_out(weight_out),
                .weight_data(weight_data),
                .weight_last(weight_last),
                .run_start(projection_start_fire),
                .run_busy(projection_busy),
                .run_done(projection_done),
                .term_valid(term_valid && term_issue_enable),
                .term_ready(backend_term_ready),
                .term_source_plane(term_source_plane),
                .term_source_y(term_source_y),
                .term_source_x(term_source_x),
                .term_lane(term_lane),
                .term_gate(term_gate),
                .term_destination_mask(term_destination_mask),
                .term_window_last(1'b0),
                .window_close(
                    projection_close && projection_close_ready
                ),
                .window_close_ready(backend_close_ready),
                .read_valid(read_valid),
                .read_ready(read_ready),
                .read_plane(read_plane),
                .read_y(read_y),
                .read_x(read_x),
                .read_out(read_out),
                .read_data_valid(read_data_valid),
                .read_data(read_data),
                .protocol_error(backend_protocol_error),
                .perf_product_terms(backend_terms),
                .perf_destination_updates(backend_updates),
                .perf_replay_updates(unused_backend_replays)
            );
        end else if (BACKEND_KIND == 2) begin : g_linear5_backend
            assign backend_weight_context_release_ready = 1'b0;
            assign unused_backend_replays = '0;
            assign vector_read_ready = 1'b0;
            assign vector_read_data_valid = 1'b0;
            assign vector_read_data = '0;
            qfit_linear5_projection_top #(
                .HEIGHT(HEIGHT),
                .WIDTH(WIDTH),
                .TIME_PLANES(TIME_PLANES),
                .HEAD_DIM(HEAD_DIM),
                .OUT_DIM(OUT_DIM),
                .GATE_W(GATE_W),
                .W_W(W_W),
                .ACC_W(ACC_W)
            ) u_projection (
                .clk_core(clk_core),
                .rst_core(rst_core),
                .weight_valid(backend_weight_valid),
                .weight_ready(backend_weight_ready),
                .weight_lane(weight_lane),
                .weight_out(weight_out),
                .weight_data(weight_data),
                .weight_last(weight_last),
                .run_start(projection_start_fire),
                .run_busy(projection_busy),
                .run_done(projection_done),
                .term_valid(term_valid && term_issue_enable),
                .term_ready(backend_term_ready),
                .term_source_plane(term_source_plane),
                .term_source_y(term_source_y),
                .term_source_x(term_source_x),
                .term_lane(term_lane),
                .term_gate(term_gate),
                .term_destination_mask(term_destination_mask),
                .term_window_last(1'b0),
                .window_close(
                    projection_close && projection_close_ready
                ),
                .window_close_ready(backend_close_ready),
                .read_valid(read_valid),
                .read_ready(read_ready),
                .read_plane(read_plane),
                .read_y(read_y),
                .read_x(read_x),
                .read_out(read_out),
                .read_data_valid(read_data_valid),
                .read_data(read_data),
                .protocol_error(backend_protocol_error),
                .perf_product_terms(backend_terms),
                .perf_destination_updates(backend_updates)
            );
        end else begin : g_role_sharded_backend
            assign backend_weight_context_release_ready = 1'b0;
            assign unused_backend_replays = '0;
            assign vector_read_ready = 1'b0;
            assign vector_read_data_valid = 1'b0;
            assign vector_read_data = '0;
            qfit_role_sharded_projection_top #(
                .HEIGHT(HEIGHT),
                .WIDTH(WIDTH),
                .TIME_PLANES(TIME_PLANES),
                .HEAD_DIM(HEAD_DIM),
                .OUT_DIM(OUT_DIM),
                .GATE_W(GATE_W),
                .W_W(W_W),
                .ACC_W(ACC_W)
            ) u_projection (
                .clk_core(clk_core),
                .rst_core(rst_core),
                .weight_valid(backend_weight_valid),
                .weight_ready(backend_weight_ready),
                .weight_lane(weight_lane),
                .weight_out(weight_out),
                .weight_data(weight_data),
                .weight_last(weight_last),
                .run_start(projection_start_fire),
                .run_busy(projection_busy),
                .run_done(projection_done),
                .term_valid(term_valid && term_issue_enable),
                .term_ready(backend_term_ready),
                .term_source_plane(term_source_plane),
                .term_source_y(term_source_y),
                .term_source_x(term_source_x),
                .term_lane(term_lane),
                .term_gate(term_gate),
                .term_destination_mask(term_destination_mask),
                .term_window_last(1'b0),
                .window_close(
                    projection_close && projection_close_ready
                ),
                .window_close_ready(backend_close_ready),
                .read_valid(read_valid),
                .read_ready(read_ready),
                .read_plane(read_plane),
                .read_y(read_y),
                .read_x(read_x),
                .read_out(read_out),
                .read_data_valid(read_data_valid),
                .read_data(read_data),
                .protocol_error(backend_protocol_error),
                .perf_product_terms(backend_terms),
                .perf_destination_updates(backend_updates)
            );
        end
    endgenerate

    assign term_ready = backend_term_ready && term_issue_enable;
    assign perf_descriptors = 32'(run_descriptors_q);
    assign perf_product_terms = backend_terms;
    assign perf_destination_updates = backend_updates;
    assign stream_idle = run_active_q
                      && !plane_open_q
                      && 32'(planes_completed_q) == TIME_PLANES
                      && 32'(run_descriptors_q) == TOTAL_TOKENS
                      && builder_descriptor_ready
                      && !descriptor_valid
                      && !term_valid;
    assign projection_close_ready = stream_idle
                                  && backend_close_ready
                                  && !run_protocol_error_q
                                  && !weight_protocol_error_q
                                  && !backend_protocol_error;
    assign protocol_error = backend_protocol_error
                         || run_protocol_error_q
                         || weight_protocol_error_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            weights_loaded_q <= 1'b0;
            run_active_q <= 1'b0;
            plane_open_q <= 1'b0;
            planes_completed_q <= '0;
            plane_tokens_q <= '0;
            run_descriptors_q <= '0;
            descriptor_seen_q <= '0;
            expected_y_q <= '0;
            expected_x_q <= '0;
            run_protocol_error_q <= 1'b0;
            weight_protocol_error_q <= 1'b0;
            weight_count_q <= '0;
            weight_seen_q <= '0;
        end else begin
            if (weight_fire) begin
                weight_count_q <= weight_count_q + 1'b1;
                for (integer lane = 0; lane < HEAD_DIM; lane = lane + 1) begin
                    for (integer out = 0; out < OUT_DIM; out = out + 1) begin
                        if (
                            weight_lane == LANE_W'(lane)
                            && weight_out == OUT_W'(out)
                        )
                            weight_seen_q[lane * OUT_DIM + out] <= 1'b1;
                    end
                end
                if (weight_last)
                    weights_loaded_q <= 1'b1;
            end

            if (weight_context_release_fire) begin
                weights_loaded_q <= 1'b0;
                weight_count_q <= '0;
                weight_seen_q <= '0;
            end

            if (
                weight_valid
                && backend_weight_ready
                && !weight_request_legal
            )
                weight_protocol_error_q <= 1'b1;
            if (
                weight_context_release
                && !weight_context_release_ready
            )
                weight_protocol_error_q <= 1'b1;

            if (projection_start_fire) begin
                run_active_q <= 1'b1;
                plane_open_q <= 1'b0;
                planes_completed_q <= '0;
                plane_tokens_q <= '0;
                run_descriptors_q <= '0;
                descriptor_seen_q <= '0;
                expected_y_q <= '0;
                expected_x_q <= '0;
                run_protocol_error_q <= 1'b0;
            end else if (projection_done && run_active_q) begin
                run_active_q <= 1'b0;
            end

            if (plane_start_fire) begin
                plane_open_q <= 1'b1;
                plane_tokens_q <= '0;
                expected_y_q <= '0;
                expected_x_q <= '0;
            end

            if (input_fire) begin
                if (plane_tokens_q == TOKEN_COUNT_W'(TOKENS_PER_PLANE - 1)) begin
                    plane_tokens_q <= TOKEN_COUNT_W'(TOKENS_PER_PLANE);
                    plane_open_q <= 1'b0;
                    planes_completed_q <= planes_completed_q + 1'b1;
                end else begin
                    plane_tokens_q <= plane_tokens_q + 1'b1;
                    if (expected_x_q == X_W'(WIDTH - 1)) begin
                        expected_x_q <= '0;
                        expected_y_q <= expected_y_q + 1'b1;
                    end else begin
                        expected_x_q <= expected_x_q + 1'b1;
                    end
                end
            end

            if (descriptor_fire) begin
                descriptor_seen_q[descriptor_source_id] <= 1'b1;
                run_descriptors_q <= run_descriptors_q + 1'b1;
            end
            if (descriptor_attempt && !descriptor_contract_valid)
                run_protocol_error_q <= 1'b1;

            if (projection_start && !projection_start_ready)
                run_protocol_error_q <= 1'b1;
            if (projection_close && !projection_close_ready)
                run_protocol_error_q <= 1'b1;
            if (plane_start && !plane_request_legal)
                run_protocol_error_q <= 1'b1;
            if (in_valid && !input_request_legal)
                run_protocol_error_q <= 1'b1;
        end
    end
endmodule

`default_nettype wire
