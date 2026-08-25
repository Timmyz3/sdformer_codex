`timescale 1ns/1ps
`default_nettype none

// Phase-local correction service primitive.  A 96xINT8 weight vector is
// loaded in three 256-bit slots and then held while one destination/sign
// descriptor is accepted per cycle.  Phase transpose, descriptor storage,
// SRAMs, accumulation, and bank-conflict scheduling are explicit port cuts.
module m104_held_weight_correction_broadcaster #(
    parameter int TAG_W = 32
) (
    input  logic                     clk_core,
    input  logic                     rst_core,

    input  logic                     load_valid,
    output logic                     load_ready,
    input  logic [3:0]               load_source,
    input  logic [2:0]               load_block,
    input  logic [1:0]               load_beat,
    input  logic [255:0]             load_data,
    output logic                     load_accept,

    input  logic                     event_valid,
    output logic                     event_ready,
    input  logic [3:0]               event_source,
    input  logic [2:0]               event_block,
    input  logic                     event_negate,
    input  logic                     event_last_for_key,
    input  logic [TAG_W-1:0]         event_tag,
    output logic                     event_accept,

    output logic                     output_valid,
    input  logic                     output_ready,
    output logic [TAG_W-1:0]         output_tag,
    output logic [3:0]               output_source,
    output logic [2:0]               output_block,
    output logic                     output_negate,
    output logic [96*12-1:0]         output_values,
    output logic                     output_accept,

    output logic                     held_valid,
    output logic                     collecting,
    output logic [1:0]               expected_load_beat,
    output logic                     protocol_error,
    output logic                     busy
);
    logic [767:0] held_weight_q;
    logic [3:0] held_source_q;
    logic [2:0] held_block_q;
    logic held_valid_q, collecting_q;
    logic [1:0] expected_load_beat_q;
    logic request_fault_q;
    logic accepted_event_grace_q;
    logic [3:0] accepted_event_grace_source_q;
    logic [2:0] accepted_event_grace_block_q;
    logic accepted_event_grace_negate_q;
    logic accepted_event_grace_last_q;
    logic [TAG_W-1:0] accepted_event_grace_tag_q;

    logic output_valid_q;
    logic [TAG_W-1:0] output_tag_q;
    logic [3:0] output_source_q;
    logic [2:0] output_block_q;
    logic output_negate_q;
    logic [96*12-1:0] output_values_q;

    logic load_identity_valid, event_identity_valid;
    logic accepted_event_grace_match;
    logic request_collision, illegal_request;

`ifndef SYNTHESIS
    initial begin
        if (TAG_W != 32)
            $fatal(1, "M104 frozen tag geometry drift");
    end
`endif

    always_comb begin : request_audit
        if (collecting_q) begin
            load_identity_valid = !held_valid_q
                                && load_source == held_source_q
                                && load_block == held_block_q
                                && load_beat == expected_load_beat_q;
        end else begin
            load_identity_valid = !held_valid_q && load_beat == 0;
        end
        event_identity_valid = held_valid_q
                            && event_source == held_source_q
                            && event_block == held_block_q;
        accepted_event_grace_match = accepted_event_grace_q
                                  && event_source
                                     == accepted_event_grace_source_q
                                  && event_block
                                     == accepted_event_grace_block_q
                                  && event_negate
                                     == accepted_event_grace_negate_q
                                  && event_last_for_key
                                     == accepted_event_grace_last_q
                                  && event_tag == accepted_event_grace_tag_q;
        request_collision = load_valid && event_valid;
        illegal_request = request_collision
                       || (load_valid && !load_identity_valid)
                       || (event_valid && !event_identity_valid
                                       && !accepted_event_grace_match);
    end

    assign protocol_error = request_fault_q || illegal_request;
    assign load_ready = !protocol_error && !event_valid
                      && load_identity_valid;
    assign event_ready = !protocol_error && !load_valid
                       && event_identity_valid
                       && !accepted_event_grace_match
                       && (!output_valid_q || output_ready);
    assign load_accept = load_valid && load_ready;
    assign event_accept = event_valid && event_ready;

    // Gate the registered output with both sticky and same-cycle faults.  An
    // invalid new request therefore cannot retire an older stalled result.
    assign output_valid = !protocol_error && output_valid_q;
    assign output_tag = output_valid ? output_tag_q : '0;
    assign output_source = output_valid ? output_source_q : '0;
    assign output_block = output_valid ? output_block_q : '0;
    assign output_negate = output_valid && output_negate_q;
    assign output_values = output_valid ? output_values_q : '0;
    assign output_accept = output_valid && output_ready;

    assign held_valid = held_valid_q;
    assign collecting = collecting_q;
    assign expected_load_beat = expected_load_beat_q;
    assign busy = collecting_q || held_valid_q || output_valid_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            held_weight_q <= '0;
            held_source_q <= '0;
            held_block_q <= '0;
            held_valid_q <= 1'b0;
            collecting_q <= 1'b0;
            expected_load_beat_q <= '0;
            request_fault_q <= 1'b0;
            accepted_event_grace_q <= 1'b0;
            accepted_event_grace_source_q <= '0;
            accepted_event_grace_block_q <= '0;
            accepted_event_grace_negate_q <= 1'b0;
            accepted_event_grace_last_q <= 1'b0;
            accepted_event_grace_tag_q <= '0;
            output_valid_q <= 1'b0;
            output_tag_q <= '0;
            output_source_q <= '0;
            output_block_q <= '0;
            output_negate_q <= 1'b0;
            output_values_q <= '0;
        end else begin
            if (!event_valid || !accepted_event_grace_match)
                accepted_event_grace_q <= 1'b0;
            if (illegal_request)
                request_fault_q <= 1'b1;

            if (!protocol_error) begin
                if (output_valid_q && output_ready)
                    output_valid_q <= 1'b0;

                if (load_accept) begin
                    held_weight_q[load_beat*256 +: 256] <= load_data;
                    if (!collecting_q) begin
                        held_source_q <= load_source;
                        held_block_q <= load_block;
                        collecting_q <= 1'b1;
                        expected_load_beat_q <= 2'd1;
                    end else if (load_beat == 2'd2) begin
                        collecting_q <= 1'b0;
                        expected_load_beat_q <= '0;
                        held_valid_q <= 1'b1;
                    end else begin
                        expected_load_beat_q <= expected_load_beat_q + 1'b1;
                    end
                end

                if (event_accept) begin
                    accepted_event_grace_q <= 1'b1;
                    accepted_event_grace_source_q <= event_source;
                    accepted_event_grace_block_q <= event_block;
                    accepted_event_grace_negate_q <= event_negate;
                    accepted_event_grace_last_q <= event_last_for_key;
                    accepted_event_grace_tag_q <= event_tag;
                    output_valid_q <= 1'b1;
                    output_tag_q <= event_tag;
                    output_source_q <= event_source;
                    output_block_q <= event_block;
                    output_negate_q <= event_negate;
                    for (int lane = 0; lane < 96; lane++) begin
                        if (event_negate)
                            output_values_q[lane*12 +: 12] <=
                                -$signed({{4{held_weight_q[lane*8+7]}},
                                          held_weight_q[lane*8 +: 8]});
                        else
                            output_values_q[lane*12 +: 12] <=
                                {{4{held_weight_q[lane*8+7]}},
                                  held_weight_q[lane*8 +: 8]};
                    end
                    if (event_last_for_key)
                        held_valid_q <= 1'b0;
                end
            end
        end
    end
endmodule

`default_nettype wire
