`timescale 1ns/1ps
`default_nettype none

// SRAM-realistic Local5 relation frontier. K and each of the five directional
// gate/valid relations reside in independent synchronous 1R1W banks.
module qfit_dual_color_relation_frontier_sync #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int K_W = 32,
    parameter int GATE_W = 9,
    parameter int READ_LATENCY = 1,
    // 0: generic synchronous memory contract; 1: Nangate45 fakeram binding.
    parameter int RELATION_MEMORY_IMPL = 0,
    // Interface-compatible with the rolling sidecar. The sealed T450
    // implementation has no alternate scheduler and therefore accepts 0 only.
    parameter int ROLLING_SCHED_MODE = 0,
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
    input  logic                       build_start,
    input  logic                       build_seal,
    output logic                       build_active,
    output logic                       build_done,
    input  logic                       in_valid,
    output logic                       in_ready,
    input  logic [PLANE_W-1:0]         in_plane,
    input  logic [Y_W-1:0]             in_destination_y,
    input  logic [X_W-1:0]             in_destination_x,
    input  logic [4:0]                 in_candidate_valid,
    input  logic [4:0]                 in_active_candidate_mask,
    input  logic [K_W-1:0]             in_k_self,
    input  logic [5*GATE_W-1:0]        in_direction_gates,
    output logic                       descriptor_valid,
    input  logic                       descriptor_ready,
    output logic [SOURCE_ID_W-1:0]     descriptor_source_id,
    output logic [PLANE_W-1:0]         descriptor_plane,
    output logic [Y_W-1:0]             descriptor_y,
    output logic [X_W-1:0]             descriptor_x,
    output logic [K_W-1:0]             descriptor_k,
    output logic [5*GATE_W-1:0]        descriptor_incoming_gates,
    output logic [4:0]                 descriptor_valid_mask,
    output logic                       descriptor_last,

    // Geometry is exposed at relation-read issue time, before K/gate payload
    // returns. The request remains stable until geometry_ready is asserted.
    output logic                       geometry_valid,
    input  logic                       geometry_ready,
    output logic [SOURCE_ID_W-1:0]     geometry_source_id,
    output logic [PLANE_W-1:0]         geometry_plane,
    output logic [Y_W-1:0]             geometry_y,
    output logic [X_W-1:0]             geometry_x,
    output logic                       geometry_last,

    output logic                       protocol_error,
    output logic [31:0]                perf_relation_writes,
    output logic [31:0]                perf_source_reads,
    output logic [31:0]                perf_dense_reads_avoided,
    output logic [31:0]                perf_memory_wait_cycles
);
    localparam int TOKENS_PER_PLANE = HEIGHT * WIDTH;
    localparam int TOTAL_SOURCES = TOKENS_PER_PLANE * TIME_PLANES;
    localparam int ADDR_W = (TOTAL_SOURCES <= 1) ? 1 : $clog2(TOTAL_SOURCES);
    localparam int EXPECTED_WRITES_W = $clog2(TOTAL_SOURCES + 1);

    logic index_build_active;
    logic index_build_done;
    logic index_in_ready;
    logic index_out_valid;
    logic index_out_ready;
    logic [SOURCE_ID_W-1:0] index_source_id;
    logic [PLANE_W-1:0] index_source_plane;
    logic [Y_W-1:0] index_source_y;
    logic [X_W-1:0] index_source_x;
    logic index_out_last;
    logic index_protocol_error;
    logic [31:0] unused_input_candidates;
    logic [31:0] unused_unique_sources;
    logic [31:0] unused_duplicate_sets;
    logic [31:0] index_bank_conflicts;
    logic [31:0] unused_word_probes;

    logic write_fire;
    logic [ADDR_W-1:0] write_address;
    logic read_issue;
    logic [ADDR_W-1:0] k_read_address;
    logic [ADDR_W-1:0] role_read_address [0:4];
    logic [4:0] role_geometry_valid;
    logic k_read_data_valid;
    logic [K_W-1:0] k_read_data;
    logic role_read_data_valid [0:4];
    logic [GATE_W:0] role_read_data [0:4];

    logic read_pending_q;
    logic [SOURCE_ID_W-1:0] pending_source_id_q;
    logic [PLANE_W-1:0] pending_plane_q;
    logic [Y_W-1:0] pending_y_q;
    logic [X_W-1:0] pending_x_q;
    logic [4:0] pending_geometry_valid_q;
    logic pending_last_q;
    logic descriptor_valid_q;
    logic [SOURCE_ID_W-1:0] descriptor_source_id_q;
    logic [PLANE_W-1:0] descriptor_plane_q;
    logic [Y_W-1:0] descriptor_y_q;
    logic [X_W-1:0] descriptor_x_q;
    logic [K_W-1:0] descriptor_k_q;
    logic [5*GATE_W-1:0] descriptor_gates_q;
    logic [4:0] descriptor_mask_q;
    logic descriptor_last_q;
    logic [EXPECTED_WRITES_W-1:0] accepted_writes_q;
    logic protocol_error_q;
    logic [31:0] relation_writes_q;
    logic [31:0] source_reads_q;
    logic [31:0] memory_wait_cycles_q;

    initial begin
        if (ROLLING_SCHED_MODE != 0)
            $fatal(1, "sealed T450 frontier requires ROLLING_SCHED_MODE=0");
        if (
            RELATION_MEMORY_IMPL == 1
            && (
                TOTAL_SOURCES != 450
                || K_W != 32
                || GATE_W != 9
                || READ_LATENCY != 1
            )
        )
            $error(
                "fakeram45 binding requires T=450, K_W=32, GATE_W=9, latency=1"
            );
    end

    function automatic logic [ADDR_W-1:0] source_address(
        input integer plane, input integer y, input integer x
    );
        source_address = ADDR_W'(plane * TOKENS_PER_PLANE + y * WIDTH + x);
    endfunction

    qfit_dual_color_word_skipper_index #(
        .HEIGHT(HEIGHT), .WIDTH(WIDTH), .TIME_PLANES(TIME_PLANES)
    ) u_active_index (
        .clk_core(clk_core), .rst_core(rst_core),
        .build_start(build_start), .build_seal(build_seal),
        .build_active(index_build_active), .build_done(index_build_done),
        .in_valid(in_valid), .in_ready(index_in_ready),
        .in_plane(in_plane), .in_destination_y(in_destination_y),
        .in_destination_x(in_destination_x),
        .in_active_candidate_mask(in_active_candidate_mask),
        .out_valid(index_out_valid), .out_ready(index_out_ready),
        .out_source_id(index_source_id),
        .out_source_plane(index_source_plane), .out_source_y(index_source_y),
        .out_source_x(index_source_x), .out_last(index_out_last),
        .protocol_error(index_protocol_error),
        .perf_input_candidates(unused_input_candidates),
        .perf_unique_sources(unused_unique_sources),
        .perf_duplicate_sets(unused_duplicate_sets),
        .perf_bank_conflicts(index_bank_conflicts),
        .perf_word_probes(unused_word_probes)
    );

    assign in_ready = index_in_ready;
    assign write_fire = in_valid && in_ready;
    assign write_address = source_address(
        in_plane, in_destination_y, in_destination_x
    );
    assign geometry_valid = index_out_valid
                          && !read_pending_q
                          && (!descriptor_valid_q || descriptor_ready);
    assign geometry_source_id = index_source_id;
    assign geometry_plane = index_source_plane;
    assign geometry_y = index_source_y;
    assign geometry_x = index_source_x;
    assign geometry_last = index_out_last;
    assign index_out_ready = geometry_valid && geometry_ready;
    assign read_issue = index_out_valid && index_out_ready;
    assign k_read_address = source_address(
        index_source_plane, index_source_y, index_source_x
    );

    always_comb begin
        for (integer role = 0; role < 5; role = role + 1) begin
            role_read_address[role] = k_read_address;
            role_geometry_valid[role] = 1'b1;
        end
        if (index_source_y < HEIGHT - 1)
            role_read_address[1] = source_address(
                index_source_plane, index_source_y + 1, index_source_x
            );
        else
            role_geometry_valid[1] = 1'b0;
        if (index_source_y != 0)
            role_read_address[2] = source_address(
                index_source_plane, index_source_y - 1, index_source_x
            );
        else
            role_geometry_valid[2] = 1'b0;
        if (index_source_x < WIDTH - 1)
            role_read_address[3] = source_address(
                index_source_plane, index_source_y, index_source_x + 1
            );
        else
            role_geometry_valid[3] = 1'b0;
        if (index_source_x != 0)
            role_read_address[4] = source_address(
                index_source_plane, index_source_y, index_source_x - 1
            );
        else
            role_geometry_valid[4] = 1'b0;
    end

    generate
        if (RELATION_MEMORY_IMPL == 0) begin : g_generic_memory
            qfit_sync_relation_bank #(
                .DEPTH(TOTAL_SOURCES), .DATA_W(K_W),
                .READ_LATENCY(READ_LATENCY)
            ) u_k_bank (
                .clk_core(clk_core), .rst_core(rst_core),
                .write_valid(write_fire), .write_addr(write_address),
                .write_data(in_k_self), .read_valid(read_issue),
                .read_addr(k_read_address),
                .read_data_valid(k_read_data_valid), .read_data(k_read_data)
            );

            for (genvar role = 0; role < 5; role = role + 1) begin : g_role_bank
                qfit_sync_relation_bank #(
                    .DEPTH(TOTAL_SOURCES), .DATA_W(GATE_W + 1),
                    .READ_LATENCY(READ_LATENCY)
                ) u_relation_bank (
                    .clk_core(clk_core), .rst_core(rst_core),
                    .write_valid(write_fire), .write_addr(write_address),
                    .write_data({
                        in_candidate_valid[role],
                        in_direction_gates[role*GATE_W +: GATE_W]
                    }),
                    .read_valid(read_issue),
                    .read_addr(role_read_address[role]),
                    .read_data_valid(role_read_data_valid[role]),
                    .read_data(role_read_data[role])
                );
            end
        end else begin : g_fakeram45_memory
            qfit_fakeram45_relation_bank_450 #(
                .DATA_W(K_W), .ADDR_W(ADDR_W)
            ) u_k_bank (
                .clk_core(clk_core), .rst_core(rst_core),
                .write_valid(write_fire), .write_addr(write_address),
                .write_data(in_k_self), .read_valid(read_issue),
                .read_addr(k_read_address),
                .read_data_valid(k_read_data_valid), .read_data(k_read_data)
            );

            for (genvar role = 0; role < 5; role = role + 1) begin : g_role_bank
                qfit_fakeram45_relation_bank_450 #(
                    .DATA_W(GATE_W + 1), .ADDR_W(ADDR_W)
                ) u_relation_bank (
                    .clk_core(clk_core), .rst_core(rst_core),
                    .write_valid(write_fire), .write_addr(write_address),
                    .write_data({
                        in_candidate_valid[role],
                        in_direction_gates[role*GATE_W +: GATE_W]
                    }),
                    .read_valid(read_issue),
                    .read_addr(role_read_address[role]),
                    .read_data_valid(role_read_data_valid[role]),
                    .read_data(role_read_data[role])
                );
            end
        end
    endgenerate

    assign build_active = index_build_active || read_pending_q || descriptor_valid_q;
    assign build_done = index_build_done && !read_pending_q && !descriptor_valid_q;
    assign descriptor_valid = descriptor_valid_q;
    assign descriptor_source_id = descriptor_source_id_q;
    assign descriptor_plane = descriptor_plane_q;
    assign descriptor_y = descriptor_y_q;
    assign descriptor_x = descriptor_x_q;
    assign descriptor_k = descriptor_k_q;
    assign descriptor_incoming_gates = descriptor_gates_q;
    assign descriptor_valid_mask = descriptor_mask_q;
    assign descriptor_last = descriptor_last_q;
    assign protocol_error = protocol_error_q || index_protocol_error
                          || index_bank_conflicts != 0;
    assign perf_relation_writes = relation_writes_q;
    assign perf_source_reads = source_reads_q;
    assign perf_dense_reads_avoided = index_build_done
        ? 32'(TOTAL_SOURCES) - source_reads_q : '0;
    assign perf_memory_wait_cycles = memory_wait_cycles_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            read_pending_q <= 1'b0;
            pending_source_id_q <= '0;
            pending_plane_q <= '0;
            pending_y_q <= '0;
            pending_x_q <= '0;
            pending_geometry_valid_q <= '0;
            pending_last_q <= 1'b0;
            descriptor_valid_q <= 1'b0;
            descriptor_source_id_q <= '0;
            descriptor_plane_q <= '0;
            descriptor_y_q <= '0;
            descriptor_x_q <= '0;
            descriptor_k_q <= '0;
            descriptor_gates_q <= '0;
            descriptor_mask_q <= '0;
            descriptor_last_q <= 1'b0;
            accepted_writes_q <= '0;
            protocol_error_q <= 1'b0;
            relation_writes_q <= '0;
            source_reads_q <= '0;
            memory_wait_cycles_q <= '0;
        end else if (build_start) begin
            read_pending_q <= 1'b0;
            descriptor_valid_q <= 1'b0;
            accepted_writes_q <= '0;
            protocol_error_q <= 1'b0;
            relation_writes_q <= '0;
            source_reads_q <= '0;
            memory_wait_cycles_q <= '0;
        end else begin
            if (write_fire) begin
                accepted_writes_q <= accepted_writes_q + 1'b1;
                relation_writes_q <= relation_writes_q + 1'b1;
            end
            if (
                build_seal
                && accepted_writes_q != EXPECTED_WRITES_W'(TOTAL_SOURCES)
            )
                protocol_error_q <= 1'b1;
            if (descriptor_valid_q && descriptor_ready)
                descriptor_valid_q <= 1'b0;

            if (read_issue) begin
                read_pending_q <= 1'b1;
                pending_source_id_q <= index_source_id;
                pending_plane_q <= index_source_plane;
                pending_y_q <= index_source_y;
                pending_x_q <= index_source_x;
                pending_geometry_valid_q <= role_geometry_valid;
                pending_last_q <= index_out_last;
                source_reads_q <= source_reads_q + 1'b1;
            end
            if (read_pending_q)
                memory_wait_cycles_q <= memory_wait_cycles_q + 1'b1;

            if (k_read_data_valid) begin
                descriptor_valid_q <= 1'b1;
                descriptor_source_id_q <= pending_source_id_q;
                descriptor_plane_q <= pending_plane_q;
                descriptor_y_q <= pending_y_q;
                descriptor_x_q <= pending_x_q;
                descriptor_k_q <= k_read_data;
                descriptor_gates_q <= '0;
                descriptor_mask_q <= '0;
                descriptor_last_q <= pending_last_q;
                read_pending_q <= 1'b0;
                for (integer role = 0; role < 5; role = role + 1) begin
                    descriptor_gates_q[role*GATE_W +: GATE_W]
                        <= role_read_data[role][GATE_W-1:0];
                    descriptor_mask_q[role]
                        <= pending_geometry_valid_q[role]
                        && role_read_data[role][GATE_W];
                    if (!role_read_data_valid[role])
                        protocol_error_q <= 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
