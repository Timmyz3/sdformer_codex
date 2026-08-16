`timescale 1ns/1ps
`default_nettype none

module qfit_relation_memo_tile_controller #(
    parameter int MAX_HEADS = 24,
    parameter int HEAD_W = (MAX_HEADS <= 1) ? 1 : $clog2(MAX_HEADS)
) (
    input  logic                       clk_core,
    input  logic                       rst_core,

    input  logic                       tile_start,
    output logic                       tile_ready,
    input  logic                       tile_prefer_replay,
    input  logic [HEAD_W-1:0]          tile_head_index,
    output logic                       tile_done,
    output logic                       fallback_taken,

    output logic                       use_replay,
    output logic                       replay_start,
    input  logic                       replay_cmd_ready,
    output logic [HEAD_W-1:0]          replay_head_index,
    input  logic                       replay_done,
    input  logic                       replay_miss,

    output logic                       recompute_request,
    input  logic                       recompute_grant,
    output logic                       head_start,
    input  logic                       head_ready,
    output logic [HEAD_W-1:0]          head_index,
    input  logic                       head_done,

    input  logic                       descriptor_stream_idle,
    output logic                       projection_start,
    output logic                       projection_close,
    input  logic                       projection_close_ready,
    input  logic                       projection_done,

    output logic                       protocol_error
);
    typedef enum logic [3:0] {
        ST_IDLE = 4'd0,
        ST_REPLAY_ISSUE = 4'd1,
        ST_REPLAY_WAIT = 4'd2,
        ST_RECOMPUTE_REQUEST = 4'd3,
        ST_RECOMPUTE_WAIT = 4'd4,
        ST_DESCRIPTOR_DRAIN = 4'd5,
        ST_PROJECTION_WAIT = 4'd6
    } state_t;

    state_t state_q;
    logic [HEAD_W-1:0] current_head_q;

    assign tile_ready = state_q == ST_IDLE;
    assign recompute_request = state_q == ST_RECOMPUTE_REQUEST;
    assign replay_head_index = current_head_q;
    assign head_index = current_head_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            current_head_q <= '0;
            use_replay <= 1'b0;
            replay_start <= 1'b0;
            head_start <= 1'b0;
            projection_start <= 1'b0;
            projection_close <= 1'b0;
            tile_done <= 1'b0;
            fallback_taken <= 1'b0;
            protocol_error <= 1'b0;
        end else begin
            replay_start <= 1'b0;
            head_start <= 1'b0;
            projection_start <= 1'b0;
            projection_close <= 1'b0;
            tile_done <= 1'b0;

            if (tile_start && !tile_ready)
                protocol_error <= 1'b1;

            case (state_q)
                ST_IDLE: begin
                    if (tile_start) begin
                        if (tile_head_index >= HEAD_W'(MAX_HEADS)) begin
                            protocol_error <= 1'b1;
                        end else begin
                            current_head_q <= tile_head_index;
                            fallback_taken <= 1'b0;
                            projection_start <= 1'b1;
                            if (tile_prefer_replay) begin
                                use_replay <= 1'b1;
                                state_q <= ST_REPLAY_ISSUE;
                            end else begin
                                use_replay <= 1'b0;
                                state_q <= ST_RECOMPUTE_REQUEST;
                            end
                        end
                    end
                end

                ST_REPLAY_ISSUE: begin
                    if (replay_cmd_ready) begin
                        replay_start <= 1'b1;
                        state_q <= ST_REPLAY_WAIT;
                    end
                end

                ST_REPLAY_WAIT: begin
                    if (replay_done) begin
                        if (replay_miss) begin
                            fallback_taken <= 1'b1;
                            use_replay <= 1'b0;
                            state_q <= ST_RECOMPUTE_REQUEST;
                        end else begin
                            state_q <= ST_DESCRIPTOR_DRAIN;
                        end
                    end
                end

                ST_RECOMPUTE_REQUEST: begin
                    if (recompute_grant && head_ready) begin
                        head_start <= 1'b1;
                        state_q <= ST_RECOMPUTE_WAIT;
                    end
                end

                ST_RECOMPUTE_WAIT: begin
                    if (head_done)
                        state_q <= ST_DESCRIPTOR_DRAIN;
                end

                ST_DESCRIPTOR_DRAIN: begin
                    if (
                        descriptor_stream_idle
                        && projection_close_ready
                    ) begin
                        projection_close <= 1'b1;
                        state_q <= ST_PROJECTION_WAIT;
                    end
                end

                ST_PROJECTION_WAIT: begin
                    if (projection_done) begin
                        tile_done <= 1'b1;
                        state_q <= ST_IDLE;
                    end
                end

                default: begin
                    protocol_error <= 1'b1;
                    state_q <= ST_IDLE;
                end
            endcase
        end
    end
endmodule

`default_nettype wire
