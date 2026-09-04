`timescale 1ns/1ps
`default_nettype none

// Beat assembler for M78's precision-elastic 96-lane signed PWP vectors.
// Widths 8--11 consume ceil(96*width/256) SRAM beats and are sign-extended to
// the canonical 12-bit accumulator interface.  Width 12 is the single-outlier
// escape token: no PWP data are consumed and the scheduler uses bit-sparse
// weights for that output block.
module precision_elastic_pwp_beat_assembler #(
    parameter int LANES = 96,
    parameter int BEAT_W = 256,
    parameter int MAX_BEATS = 5,
    parameter int OUT_W = 12,
    parameter int TAG_W = 32
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         command_valid,
    output logic                         command_ready,
    input  logic [TAG_W-1:0]             command_tag,
    input  logic [3:0]                   command_width,
    output logic                         command_accept,

    input  logic                         beat_valid,
    output logic                         beat_ready,
    input  logic                         beat_last,
    input  logic [BEAT_W-1:0]            beat_data,
    output logic                         beat_accept,

    output logic                         output_valid,
    input  logic                         output_ready,
    output logic [TAG_W-1:0]             output_tag,
    output logic [3:0]                   output_width,
    output logic                         output_escape,
    output logic [LANES*OUT_W-1:0]       output_values,
    output logic                         output_accept,

    output logic                         protocol_error,
    output logic                         busy
);
    localparam int BUFFER_W = BEAT_W * MAX_BEATS;

    logic collecting_q;
    logic faulted_q;
    logic [2:0] beat_index_q;
    logic [2:0] beats_needed_q;
    logic [3:0] active_width_q;
    logic [TAG_W-1:0] active_tag_q;
    logic [BUFFER_W-1:0] buffer_q;

    logic output_valid_q;
    logic [TAG_W-1:0] output_tag_q;
    logic [3:0] output_width_q;
    logic output_escape_q;
    logic [LANES*OUT_W-1:0] output_values_q;

    logic [BUFFER_W-1:0] buffer_with_beat;
    logic [LANES*OUT_W-1:0] assembled_values;
    logic expected_last;
    logic final_padding_zero;

`ifndef SYNTHESIS
    initial begin
        if (LANES != 96 || BEAT_W != 256 || MAX_BEATS != 5 || OUT_W != 12)
            $fatal(1, "M79 frozen 96-lane 256-bit-beat geometry drift");
    end
`endif

    assign output_valid = !faulted_q && output_valid_q;
    assign output_tag = output_valid_q ? output_tag_q : '0;
    assign output_width = output_valid_q ? output_width_q : '0;
    assign output_escape = output_valid_q && output_escape_q;
    assign output_values = output_valid_q ? output_values_q : '0;
    assign output_accept = output_valid && output_ready;
    assign command_ready = !faulted_q && !collecting_q
        && (!output_valid_q || output_ready);
    assign command_accept = command_valid && command_ready;
    assign beat_ready = !faulted_q && collecting_q && !output_valid_q;
    assign beat_accept = beat_valid && beat_ready;
    assign protocol_error = faulted_q;
    assign busy = collecting_q || output_valid_q;
    assign expected_last = (beat_index_q + 1'b1 == beats_needed_q);

    always_comb begin : assemble_current_beat
        buffer_with_beat = buffer_q;
        case (beat_index_q)
            3'd0: buffer_with_beat[0*BEAT_W +: BEAT_W] = beat_data;
            3'd1: buffer_with_beat[1*BEAT_W +: BEAT_W] = beat_data;
            3'd2: buffer_with_beat[2*BEAT_W +: BEAT_W] = beat_data;
            3'd3: buffer_with_beat[3*BEAT_W +: BEAT_W] = beat_data;
            3'd4: buffer_with_beat[4*BEAT_W +: BEAT_W] = beat_data;
            default: buffer_with_beat = '0;
        endcase
        assembled_values = '0;
        for (int lane = 0; lane < LANES; lane++) begin
            case (active_width_q)
                4'd8: assembled_values[lane*OUT_W +: OUT_W] = {
                    {4{buffer_with_beat[lane*8 + 7]}},
                    buffer_with_beat[lane*8 +: 8]};
                4'd9: assembled_values[lane*OUT_W +: OUT_W] = {
                    {3{buffer_with_beat[lane*9 + 8]}},
                    buffer_with_beat[lane*9 +: 9]};
                4'd10: assembled_values[lane*OUT_W +: OUT_W] = {
                    {2{buffer_with_beat[lane*10 + 9]}},
                    buffer_with_beat[lane*10 +: 10]};
                4'd11: assembled_values[lane*OUT_W +: OUT_W] = {
                    buffer_with_beat[lane*11 + 10],
                    buffer_with_beat[lane*11 +: 11]};
                default: assembled_values[lane*OUT_W +: OUT_W] = '0;
            endcase
        end
        case (active_width_q)
            4'd8: final_padding_zero = 1'b1;
            4'd9: final_padding_zero = !(|buffer_with_beat[1023:864]);
            4'd10: final_padding_zero = !(|buffer_with_beat[1023:960]);
            4'd11: final_padding_zero = !(|buffer_with_beat[1279:1056]);
            default: final_padding_zero = 1'b0;
        endcase
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            collecting_q <= 1'b0;
            faulted_q <= 1'b0;
            beat_index_q <= '0;
            beats_needed_q <= '0;
            active_width_q <= '0;
            active_tag_q <= '0;
            buffer_q <= '0;
            output_valid_q <= 1'b0;
            output_tag_q <= '0;
            output_width_q <= '0;
            output_escape_q <= 1'b0;
            output_values_q <= '0;
        end else begin
            if (output_accept)
                output_valid_q <= 1'b0;

            if (command_accept) begin
                active_tag_q <= command_tag;
                active_width_q <= command_width;
                beat_index_q <= '0;
                buffer_q <= '0;
                output_escape_q <= 1'b0;
                case (command_width)
                    4'd8: begin collecting_q <= 1'b1; beats_needed_q <= 3; end
                    4'd9: begin collecting_q <= 1'b1; beats_needed_q <= 4; end
                    4'd10: begin collecting_q <= 1'b1; beats_needed_q <= 4; end
                    4'd11: begin collecting_q <= 1'b1; beats_needed_q <= 5; end
                    4'd12: begin
                        collecting_q <= 1'b0;
                        beats_needed_q <= '0;
                        output_valid_q <= 1'b1;
                        output_tag_q <= command_tag;
                        output_width_q <= command_width;
                        output_escape_q <= 1'b1;
                        output_values_q <= '0;
                    end
                    default: begin
                        collecting_q <= 1'b0;
                        faulted_q <= 1'b1;
                    end
                endcase
            end

            if (beat_accept) begin
                buffer_q <= buffer_with_beat;
                if (beat_last != expected_last) begin
                    collecting_q <= 1'b0;
                    faulted_q <= 1'b1;
                end else if (expected_last) begin
                    collecting_q <= 1'b0;
                    if (!final_padding_zero) begin
                        faulted_q <= 1'b1;
                    end else begin
                        output_valid_q <= 1'b1;
                        output_tag_q <= active_tag_q;
                        output_width_q <= active_width_q;
                        output_escape_q <= 1'b0;
                        output_values_q <= assembled_values;
                    end
                end else begin
                    beat_index_q <= beat_index_q + 1'b1;
                end
            end

            if (beat_valid && !collecting_q && !command_accept)
                faulted_q <= 1'b1;
        end
    end
endmodule

`default_nettype wire
