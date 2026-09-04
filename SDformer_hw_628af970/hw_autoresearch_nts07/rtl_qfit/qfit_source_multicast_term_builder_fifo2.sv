`timescale 1ns/1ps
`default_nettype none

// Two-entry descriptor decoupler in front of the source-major term builder.
// It lets the relation frontier issue/read the next source while the current
// source is still expanding into lane/gate terms.
module qfit_source_multicast_term_builder_fifo2 #(
    parameter int HEAD_DIM = 32,
    parameter int GATE_W = 9,
    parameter int SOURCE_ID_W = 9,
    parameter int PLANE_W = 1,
    parameter int Y_W = 4,
    parameter int X_W = 4,
    parameter int LANE_W = (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM)
) (
    input  logic                       clk_core,
    input  logic                       rst_core,

    input  logic                       descriptor_valid,
    output logic                       descriptor_ready,
    input  logic [SOURCE_ID_W-1:0]     descriptor_source_id,
    input  logic [PLANE_W-1:0]         descriptor_plane,
    input  logic [Y_W-1:0]             descriptor_y,
    input  logic [X_W-1:0]             descriptor_x,
    input  logic [HEAD_DIM-1:0]        descriptor_k,
    input  logic [5*GATE_W-1:0]        descriptor_incoming_gates,
    input  logic [4:0]                 descriptor_valid_mask,
    input  logic                       descriptor_last,

    output logic                       term_valid,
    input  logic                       term_ready,
    output logic [SOURCE_ID_W-1:0]     term_source_id,
    output logic [PLANE_W-1:0]         term_source_plane,
    output logic [Y_W-1:0]             term_source_y,
    output logic [X_W-1:0]             term_source_x,
    output logic [LANE_W-1:0]          term_lane,
    output logic [GATE_W-1:0]          term_gate,
    output logic [4:0]                 term_destination_mask,
    output logic                       term_last,
    output logic                       term_source_last,

    output logic                       pipeline_idle,
    output logic                       protocol_error,
    output logic [31:0]                perf_descriptors,
    output logic [31:0]                perf_terms,
    output logic [31:0]                perf_destination_updates
);
    logic [SOURCE_ID_W-1:0] source_id_q [0:1];
    logic [PLANE_W-1:0] plane_q [0:1];
    logic [Y_W-1:0] y_q [0:1];
    logic [X_W-1:0] x_q [0:1];
    logic [HEAD_DIM-1:0] k_q [0:1];
    logic [5*GATE_W-1:0] gates_q [0:1];
    logic [4:0] mask_q [0:1];
    logic last_q [0:1];
    logic write_ptr_q;
    logic read_ptr_q;
    logic [1:0] count_q;
    logic builder_descriptor_ready;
    logic builder_descriptor_valid;
    logic builder_term_last;
    logic active_source_last_q;
    logic [PLANE_W-1:0] active_source_plane_q;
    logic enqueue;
    logic dequeue;
    logic protocol_error_q;

    assign builder_descriptor_valid = count_q != 0;
    assign dequeue = builder_descriptor_valid && builder_descriptor_ready;
    assign descriptor_ready = count_q < 2 || dequeue;
    assign enqueue = descriptor_valid && descriptor_ready;
    assign term_last = builder_term_last;
    assign term_source_last = builder_term_last && active_source_last_q;
    assign pipeline_idle = count_q == 0
                         && builder_descriptor_ready
                         && !term_valid;
    assign protocol_error = protocol_error_q;

    qfit_source_multicast_term_builder #(
        .HEAD_DIM(HEAD_DIM),
        .GATE_W(GATE_W),
        .SOURCE_ID_W(SOURCE_ID_W),
        .Y_W(Y_W),
        .X_W(X_W)
    ) u_builder (
        .clk_core(clk_core), .rst_core(rst_core),
        .descriptor_valid(builder_descriptor_valid),
        .descriptor_ready(builder_descriptor_ready),
        .descriptor_source_id(source_id_q[read_ptr_q]),
        .descriptor_y(y_q[read_ptr_q]),
        .descriptor_x(x_q[read_ptr_q]),
        .descriptor_k(k_q[read_ptr_q]),
        .descriptor_incoming_gates(gates_q[read_ptr_q]),
        .descriptor_valid_mask(mask_q[read_ptr_q]),
        .term_valid(term_valid), .term_ready(term_ready),
        .term_source_id(term_source_id),
        .term_source_y(term_source_y), .term_source_x(term_source_x),
        .term_lane(term_lane), .term_gate(term_gate),
        .term_destination_mask(term_destination_mask),
        .term_last(builder_term_last),
        .perf_descriptors(perf_descriptors), .perf_terms(perf_terms),
        .perf_destination_updates(perf_destination_updates)
    );

    assign term_source_plane = active_source_plane_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            write_ptr_q <= 1'b0;
            read_ptr_q <= 1'b0;
            count_q <= '0;
            active_source_last_q <= 1'b0;
            active_source_plane_q <= '0;
            protocol_error_q <= 1'b0;
            for (integer slot = 0; slot < 2; slot++) begin
                source_id_q[slot] <= '0;
                plane_q[slot] <= '0;
                y_q[slot] <= '0;
                x_q[slot] <= '0;
                k_q[slot] <= '0;
                gates_q[slot] <= '0;
                mask_q[slot] <= '0;
                last_q[slot] <= 1'b0;
            end
        end else begin
            if (count_q > 2)
                protocol_error_q <= 1'b1;
            if (enqueue) begin
                source_id_q[write_ptr_q] <= descriptor_source_id;
                plane_q[write_ptr_q] <= descriptor_plane;
                y_q[write_ptr_q] <= descriptor_y;
                x_q[write_ptr_q] <= descriptor_x;
                k_q[write_ptr_q] <= descriptor_k;
                gates_q[write_ptr_q] <= descriptor_incoming_gates;
                mask_q[write_ptr_q] <= descriptor_valid_mask;
                last_q[write_ptr_q] <= descriptor_last;
                write_ptr_q <= !write_ptr_q;
            end
            if (dequeue) begin
                active_source_last_q <= last_q[read_ptr_q];
                active_source_plane_q <= plane_q[read_ptr_q];
                read_ptr_q <= !read_ptr_q;
            end
            case ({enqueue, dequeue})
                2'b10: count_q <= count_q + 1'b1;
                2'b01: count_q <= count_q - 1'b1;
                default: count_q <= count_q;
            endcase
        end
    end
endmodule

`default_nettype wire
