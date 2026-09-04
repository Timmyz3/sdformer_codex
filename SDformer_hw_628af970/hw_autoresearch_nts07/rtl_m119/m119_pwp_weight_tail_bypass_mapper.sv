`timescale 1ns/1ps
`default_nettype none

// Three-beat INT8 weight assembler and signed19 PWP event mapper.
//
// A synchronous 256-bit weight port returns each accepted load beat one cycle
// later.  Beat2 therefore returns while the immediately following first event
// is visible.  The tail-bypass path combines that response with stored beats
// 0/1, avoiding an otherwise uncounted one-cycle bubble per active source key.
// Event updates use a one-entry elastic output and remain stable under
// accumulator backpressure.  This module is a standalone numeric service
// island; the weight macro and M117/M118 integration remain separate evidence.
module m119_pwp_weight_tail_bypass_mapper #(
    parameter int LANES = 96,
    parameter int WEIGHT_BITS = 8,
    parameter int ACC_BITS = 19,
    parameter int BEAT_BITS = 256,
    parameter int PAYLOAD_BITS = LANES * WEIGHT_BITS,
    parameter int DELTA_BITS = LANES * ACC_BITS
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         service_valid,
    output logic                         service_ready,
    input  logic                         service_is_event,
    input  logic [3:0]                   service_source,
    input  logic [2:0]                   service_block,
    input  logic [1:0]                   service_load_beat,
    input  logic [8:0]                   service_row_offset,
    input  logic                         service_negate,
    input  logic                         service_last_for_key,
    output logic                         service_accept,

    output logic                         weight_rd_en,
    output logic [6:0]                   weight_rd_key,
    output logic [1:0]                   weight_rd_beat,
    input  logic [BEAT_BITS-1:0]         weight_rd_data,

    output logic                         update_valid,
    input  logic                         update_ready,
    output logic [2:0]                   update_block,
    output logic [8:0]                   update_row,
    output logic [DELTA_BITS-1:0]        update_delta,
    output logic                         update_accept,

    output logic                         payload_active,
    output logic                         tail_bypass_available,
    output logic                         protocol_error,
    output logic                         busy
);
    logic request_fault_q;
    logic payload_active_q;
    logic event_phase_q;
    logic [1:0] expected_load_beat_q;
    logic [6:0] payload_key_q;
    logic [PAYLOAD_BITS-1:0] payload_q;
    logic [2:0] beat_valid_q;

    logic response_valid_q;
    logic [6:0] response_key_q;
    logic [1:0] response_beat_q;

    logic update_valid_q;
    logic [2:0] update_block_q;
    logic [8:0] update_row_q;
    logic [DELTA_BITS-1:0] update_delta_q;

    logic [6:0] service_key;
    logic load_shape_valid, event_shape_valid, token_shape_valid;
    logic output_slot_available;
    logic payload_ready_now;
    logic illegal_request;
    logic [PAYLOAD_BITS-1:0] complete_payload;
    logic [DELTA_BITS-1:0] mapped_delta;
    logic signed [WEIGHT_BITS-1:0] lane_weight [0:LANES-1];
    logic signed [ACC_BITS-1:0] lane_weight_ext [0:LANES-1];
    logic signed [ACC_BITS-1:0] lane_delta [0:LANES-1];

`ifndef SYNTHESIS
    initial begin
        if (LANES != 96 || WEIGHT_BITS != 8 || ACC_BITS != 19
                || BEAT_BITS != 256 || PAYLOAD_BITS != 768
                || DELTA_BITS != 1824)
            $fatal(1, "M119 production geometry drift");
    end
`endif

    assign service_key = {service_source, service_block};
    assign load_shape_valid = !service_is_event && !event_phase_q
                            && service_load_beat < 3
                            && service_load_beat == expected_load_beat_q
                            && ((expected_load_beat_q == 0
                                 && !payload_active_q)
                                || (expected_load_beat_q != 0
                                    && payload_active_q
                                    && service_key == payload_key_q));
    assign event_shape_valid = service_is_event && event_phase_q
                             && payload_active_q
                             && service_key == payload_key_q;
    assign token_shape_valid = load_shape_valid || event_shape_valid;
    assign illegal_request = service_valid && !token_shape_valid;
    assign protocol_error = request_fault_q || illegal_request;

    assign tail_bypass_available = beat_valid_q[0] && beat_valid_q[1]
                                 && !beat_valid_q[2]
                                 && response_valid_q
                                 && response_beat_q == 2
                                 && response_key_q == payload_key_q;
    assign payload_ready_now = &beat_valid_q || tail_bypass_available;
    assign output_slot_available = !update_valid_q || update_ready;
    assign service_ready = !protocol_error && token_shape_valid
                         && (!service_is_event
                             || (payload_ready_now
                                 && output_slot_available));
    assign service_accept = service_valid && service_ready;

    assign weight_rd_en = service_accept && !service_is_event;
    assign weight_rd_key = service_key;
    assign weight_rd_beat = service_load_beat;

    assign update_valid = update_valid_q;
    assign update_block = update_block_q;
    assign update_row = update_row_q;
    assign update_delta = update_delta_q;
    assign update_accept = update_valid && update_ready;
    assign payload_active = payload_active_q;
    assign busy = payload_active_q || response_valid_q || update_valid_q;

    always_comb begin : tail_bypass_and_signed_map
        complete_payload = payload_q;
        if (tail_bypass_available)
            complete_payload[2 * BEAT_BITS +: BEAT_BITS] = weight_rd_data;
        mapped_delta = '0;
        for (int lane = 0; lane < LANES; lane++) begin
            lane_weight[lane]
                = complete_payload[lane * WEIGHT_BITS +: WEIGHT_BITS];
            lane_weight_ext[lane]
                = {{(ACC_BITS-WEIGHT_BITS){lane_weight[lane][WEIGHT_BITS-1]}},
                   lane_weight[lane]};
            lane_delta[lane] = service_negate
                             ? -lane_weight_ext[lane] : lane_weight_ext[lane];
            mapped_delta[lane * ACC_BITS +: ACC_BITS] = lane_delta[lane];
        end
    end

    always_ff @(posedge clk_core) begin : state_update
        if (rst_core) begin
            request_fault_q <= 1'b0;
            payload_active_q <= 1'b0;
            event_phase_q <= 1'b0;
            expected_load_beat_q <= '0;
            payload_key_q <= '0;
            payload_q <= '0;
            beat_valid_q <= '0;
            response_valid_q <= 1'b0;
            response_key_q <= '0;
            response_beat_q <= '0;
            update_valid_q <= 1'b0;
            update_block_q <= '0;
            update_row_q <= '0;
            update_delta_q <= '0;
        end else begin
            if (illegal_request)
                request_fault_q <= 1'b1;

            response_valid_q <= weight_rd_en;
            if (weight_rd_en) begin
                response_key_q <= weight_rd_key;
                response_beat_q <= weight_rd_beat;
            end
            if (response_valid_q && payload_active_q
                    && response_key_q == payload_key_q
                    && response_beat_q < 3) begin
                payload_q[response_beat_q * BEAT_BITS +: BEAT_BITS]
                    <= weight_rd_data;
                beat_valid_q[response_beat_q] <= 1'b1;
            end

            if (update_accept)
                update_valid_q <= 1'b0;

            if (!request_fault_q && !illegal_request && service_accept) begin
                if (!service_is_event) begin
                    if (service_load_beat == 0) begin
                        payload_active_q <= 1'b1;
                        payload_key_q <= service_key;
                        payload_q <= '0;
                        beat_valid_q <= '0;
                    end
                    if (service_load_beat == 2) begin
                        expected_load_beat_q <= '0;
                        event_phase_q <= 1'b1;
                    end else begin
                        expected_load_beat_q
                            <= service_load_beat + 1'b1;
                    end
                end else begin
                    update_valid_q <= 1'b1;
                    update_block_q <= service_block;
                    update_row_q <= service_row_offset;
                    update_delta_q <= mapped_delta;
                    if (service_last_for_key) begin
                        payload_active_q <= 1'b0;
                        event_phase_q <= 1'b0;
                        expected_load_beat_q <= '0;
                        beat_valid_q <= '0;
                    end
                end
            end
        end
    end
endmodule

`default_nettype wire
