// Collect all T decisions before exposing any firing result to a consumer.
// Full-T recovery arithmetic is an explicit external service.
module atlif_threshold_group_commit #(
    parameter int B = 32,
    parameter int T = 10
) (
    input logic clk,
    input logic rst_n,
    input logic start_valid_i,
    output logic start_ready_o,
    input logic [31:0] context_base_i, // low four bits must be zero
    input logic [15:0] epoch_i,
    input logic [31:0] theta_payload_i,
    input logic packet_valid_i,
    output logic packet_ready_o,
    input logic [1:0] packet_status_i,
    input logic [B-1:0] packet_bits_i,
    input logic [31:0] packet_context_i,
    input logic [15:0] packet_epoch_i,
    input logic [31:0] packet_theta_i,
    output logic replay_valid_o,
    input logic replay_ready_i,
    output logic [T-1:0] replay_mask_o,
    input logic repair_valid_i,
    output logic repair_ready_o,
    input logic [T*B-1:0] repair_bits_i,
    input logic [31:0] repair_context_base_i,
    input logic [15:0] repair_epoch_i,
    input logic [31:0] repair_theta_i,
    output logic result_valid_o,
    input logic result_ready_i,
    output logic [1:0] result_status_o, // 0=whole group COMMIT, 2=ERROR
    output logic [T*B-1:0] result_bits_o,
    output logic [31:0] context_base_o,
    output logic [15:0] epoch_o,
    output logic [31:0] theta_payload_o
);
    localparam int TIW = (T > 1) ? $clog2(T) : 1;
    typedef enum logic [2:0] {IDLE, COLLECT, CHECK, REPLAY, REPAIR, RESULT} state_t;
    state_t state_q;
    logic [4:0] count_q;
    logic error_q;
    logic [T*B-1:0] bits_q;
    logic [T-1:0] replay_q;

    initial begin
        if (B < 2 || T < 1 || T > 16) $fatal(1, "Unsupported group B/T");
    end

    assign start_ready_o = state_q == IDLE;
    assign packet_ready_o = state_q == COLLECT;
    assign replay_valid_o = state_q == REPLAY;
    assign replay_mask_o = replay_q;
    assign repair_ready_o = state_q == REPAIR;
    assign result_valid_o = state_q == RESULT;
    assign result_status_o = error_q ? 2'd2 : 2'd0;
    // No provisional packet bits leak to a downstream consumer.
    assign result_bits_o = (state_q == RESULT && !error_q) ? bits_q : '0;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state_q <= IDLE;
            count_q <= '0;
            error_q <= 1'b0;
            bits_q <= '0;
            replay_q <= '0;
            context_base_o <= '0;
            epoch_o <= '0;
            theta_payload_o <= '0;
        end else begin
            case (state_q)
                IDLE: if (start_valid_i) begin
                    state_q <= COLLECT;
                    count_q <= '0;
                    bits_q <= '0;
                    replay_q <= '0;
                    error_q <= context_base_i[3:0] != 4'd0;
                    context_base_o <= context_base_i;
                    epoch_o <= epoch_i;
                    theta_payload_o <= theta_payload_i;
                end
                COLLECT: if (packet_valid_i) begin
                    if (packet_context_i != (context_base_o | {28'd0,count_q[3:0]}) ||
                        packet_epoch_i != epoch_o || packet_theta_i != theta_payload_o ||
                        packet_status_i >= 2'd2) error_q <= 1'b1;
                    if (packet_status_i == 2'd1) replay_q[count_q[TIW-1:0]] <= 1'b1;
                    if (packet_status_i == 2'd0)
                        bits_q[int'(count_q)*B +: B] <= packet_bits_i;
                    if (count_q == 5'(T-1)) state_q <= CHECK;
                    else count_q <= count_q + 1'b1;
                end
                CHECK: begin
                    if (error_q) state_q <= RESULT;
                    else if (|replay_q) state_q <= REPLAY;
                    else state_q <= RESULT;
                end
                REPLAY: if (replay_ready_i) state_q <= REPAIR;
                REPAIR: if (repair_valid_i) begin
                    error_q <= repair_context_base_i != context_base_o ||
                               repair_epoch_i != epoch_o ||
                               repair_theta_i != theta_payload_o;
                    bits_q <= repair_bits_i;
                    state_q <= RESULT;
                end
                RESULT: if (result_ready_i) state_q <= IDLE;
                default: state_q <= IDLE;
            endcase
        end
    end
endmodule
