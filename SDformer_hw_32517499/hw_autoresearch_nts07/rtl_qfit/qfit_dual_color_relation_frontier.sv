`timescale 1ns/1ps
`default_nettype none

// Local5双向五色关系前沿。
// score阶段按destination写入局部关系，同时用同一五色映射建立active-source集合；
// seal后仅对active source做关系转置读出，避免固定450-source扫描。
module qfit_dual_color_relation_frontier #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int K_W = 32,
    parameter int GATE_W = 9,
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

    output logic                       protocol_error,
    output logic [31:0]                perf_relation_writes,
    output logic [31:0]                perf_source_reads,
    output logic [31:0]                perf_dense_reads_avoided
);
    localparam int TOKENS_PER_PLANE = HEIGHT * WIDTH;
    localparam int TOTAL_SOURCES = TOKENS_PER_PLANE * TIME_PLANES;
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
    logic [31:0] index_unique_sources;
    logic [31:0] unused_duplicate_sets;
    logic [31:0] index_bank_conflicts;
    logic [31:0] unused_word_probes;

    logic [K_W-1:0] k_store [0:TOTAL_SOURCES-1];
    logic [GATE_W-1:0] gate_store [0:4][0:TOTAL_SOURCES-1];
    logic valid_store [0:4][0:TOTAL_SOURCES-1];
    logic [EXPECTED_WRITES_W-1:0] accepted_writes_q;
    logic descriptor_valid_q;
    logic [SOURCE_ID_W-1:0] descriptor_source_id_q;
    logic [PLANE_W-1:0] descriptor_plane_q;
    logic [Y_W-1:0] descriptor_y_q;
    logic [X_W-1:0] descriptor_x_q;
    logic [K_W-1:0] descriptor_k_q;
    logic [5*GATE_W-1:0] descriptor_gates_q;
    logic [4:0] descriptor_mask_q;
    logic descriptor_last_q;
    logic protocol_error_q;
    logic [31:0] relation_writes_q;
    logic [31:0] source_reads_q;

    function automatic integer source_address(
        input integer plane,
        input integer y,
        input integer x
    );
        source_address = plane * TOKENS_PER_PLANE + y * WIDTH + x;
    endfunction

    qfit_dual_color_word_skipper_index #(
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES)
    ) u_active_index (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .build_start(build_start),
        .build_seal(build_seal),
        .build_active(index_build_active),
        .build_done(index_build_done),
        .in_valid(in_valid),
        .in_ready(index_in_ready),
        .in_plane(in_plane),
        .in_destination_y(in_destination_y),
        .in_destination_x(in_destination_x),
        .in_active_candidate_mask(in_active_candidate_mask),
        .out_valid(index_out_valid),
        .out_ready(index_out_ready),
        .out_source_id(index_source_id),
        .out_source_plane(index_source_plane),
        .out_source_y(index_source_y),
        .out_source_x(index_source_x),
        .out_last(index_out_last),
        .protocol_error(index_protocol_error),
        .perf_input_candidates(unused_input_candidates),
        .perf_unique_sources(index_unique_sources),
        .perf_duplicate_sets(unused_duplicate_sets),
        .perf_bank_conflicts(index_bank_conflicts),
        .perf_word_probes(unused_word_probes)
    );

    assign in_ready = index_in_ready;
    assign index_out_ready = !descriptor_valid_q || descriptor_ready;
    assign build_active = index_build_active || descriptor_valid_q;
    assign build_done = index_build_done && !descriptor_valid_q;
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
    assign perf_dense_reads_avoided =
        index_build_done
        ? 32'(TOTAL_SOURCES) - source_reads_q
        : '0;

    always_ff @(posedge clk_core) begin : frontier_state
        integer write_address;
        integer read_address;
        integer neighbor_address;
        integer plane;
        integer y;
        integer x;
        integer role;
        if (rst_core) begin
            accepted_writes_q <= '0;
            descriptor_valid_q <= 1'b0;
            descriptor_source_id_q <= '0;
            descriptor_plane_q <= '0;
            descriptor_y_q <= '0;
            descriptor_x_q <= '0;
            descriptor_k_q <= '0;
            descriptor_gates_q <= '0;
            descriptor_mask_q <= '0;
            descriptor_last_q <= 1'b0;
            protocol_error_q <= 1'b0;
            relation_writes_q <= '0;
            source_reads_q <= '0;
        end else begin
            if (build_start) begin
                accepted_writes_q <= '0;
                descriptor_valid_q <= 1'b0;
                protocol_error_q <= 1'b0;
                relation_writes_q <= '0;
                source_reads_q <= '0;
            end else begin
                if (in_valid && in_ready) begin
                    write_address = source_address(
                        in_plane,
                        in_destination_y,
                        in_destination_x
                    );
                    if (
                        in_plane >= TIME_PLANES
                        || in_destination_y >= HEIGHT
                        || in_destination_x >= WIDTH
                        || write_address < 0
                        || write_address >= TOTAL_SOURCES
                    ) begin
                        protocol_error_q <= 1'b1;
                    end else begin
                        k_store[write_address] <= in_k_self;
                        for (role = 0; role < 5; role = role + 1) begin
                            gate_store[role][write_address]
                                <= in_direction_gates[role*GATE_W +: GATE_W];
                            valid_store[role][write_address]
                                <= in_candidate_valid[role];
                        end
                        accepted_writes_q <= accepted_writes_q + 1'b1;
                        relation_writes_q <= relation_writes_q + 1'b1;
                    end
                end

                if (
                    build_seal
                    && accepted_writes_q != EXPECTED_WRITES_W'(TOTAL_SOURCES)
                )
                    protocol_error_q <= 1'b1;

                if (descriptor_valid_q && descriptor_ready)
                    descriptor_valid_q <= 1'b0;

                if (index_out_valid && index_out_ready) begin
                    plane = index_source_plane;
                    y = index_source_y;
                    x = index_source_x;
                    read_address = source_address(plane, y, x);
                    descriptor_valid_q <= 1'b1;
                    descriptor_source_id_q <= index_source_id;
                    descriptor_plane_q <= index_source_plane;
                    descriptor_y_q <= index_source_y;
                    descriptor_x_q <= index_source_x;
                    descriptor_k_q <= k_store[read_address];
                    descriptor_gates_q <= '0;
                    descriptor_mask_q <= '0;
                    descriptor_last_q <= index_out_last;
                    source_reads_q <= source_reads_q + 1'b1;

                    // SELF。
                    descriptor_gates_q[0*GATE_W +: GATE_W]
                        <= gate_store[0][read_address];
                    descriptor_mask_q[0]
                        <= valid_store[0][read_address];
                    // UP source由南侧destination的UP候选引用。
                    if (y < HEIGHT - 1) begin
                        neighbor_address = source_address(plane, y + 1, x);
                        descriptor_gates_q[1*GATE_W +: GATE_W]
                            <= gate_store[1][neighbor_address];
                        descriptor_mask_q[1]
                            <= valid_store[1][neighbor_address];
                    end
                    // DOWN source由北侧destination的DOWN候选引用。
                    if (y > 0) begin
                        neighbor_address = source_address(plane, y - 1, x);
                        descriptor_gates_q[2*GATE_W +: GATE_W]
                            <= gate_store[2][neighbor_address];
                        descriptor_mask_q[2]
                            <= valid_store[2][neighbor_address];
                    end
                    // LEFT source由东侧destination的LEFT候选引用。
                    if (x < WIDTH - 1) begin
                        neighbor_address = source_address(plane, y, x + 1);
                        descriptor_gates_q[3*GATE_W +: GATE_W]
                            <= gate_store[3][neighbor_address];
                        descriptor_mask_q[3]
                            <= valid_store[3][neighbor_address];
                    end
                    // RIGHT source由西侧destination的RIGHT候选引用。
                    if (x > 0) begin
                        neighbor_address = source_address(plane, y, x - 1);
                        descriptor_gates_q[4*GATE_W +: GATE_W]
                            <= gate_store[4][neighbor_address];
                        descriptor_mask_q[4]
                            <= valid_store[4][neighbor_address];
                    end
                end
            end
        end
    end
endmodule

`default_nettype wire
