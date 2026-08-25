`timescale 1ns/1ps
`default_nettype none

// M166: replay-side rank-3 left projection and binary ATLIF backend.
//
// Dynamic BN coefficients are constant for one completed hidden-channel
// group.  Software or a separate coefficient unit may therefore precompute
//
//   folded_left[t,r,l] = alpha[l] * left[t,r]
//   folded_bias[t,l]   = offset[l] * sum_r(left[t,r] * sum_tau(right[r,tau]))
//                        + temporal_bias[t] - center[t]
//
// in an admitted fixed-point format.  The tile replay kernel then needs only
// three signed 8x8 products per output.  One 96-product pool emits two time
// rows x 16 lanes per cycle, hence five uninterrupted service cycles per
// rank-state tile.  Thresholding happens before the output FIFO, so the dense
// reconstructed T x lane tensor is never materialized.
//
// This module intentionally does not generate reciprocal-square-root/BN
// coefficients, buffer a complete BN epoch, perform fc2, or prove that an
// INT8 folded coefficient format preserves network accuracy.
module m166_q8_prefolded_rank3_left_atlif_backend #(
    parameter int TAG_BITS = 16,
    parameter int INPUT_FIFO_DEPTH = 2,
    parameter int OUTPUT_FIFO_DEPTH = 16
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         config_valid,
    output logic                         config_ready,
    input  logic signed [7:0]            config_folded_left [0:9][0:2][0:15],
    input  logic signed [23:0]           config_folded_bias [0:9][0:15],
    input  logic signed [23:0]           config_threshold,
    output logic                         config_accept,

    input  logic                         rank_valid,
    output logic                         rank_ready,
    input  logic [TAG_BITS-1:0]          rank_tag,
    input  logic                         rank_channel_last,
    input  logic signed [7:0]            rank_data [0:2][0:15],
    output logic                         rank_accept,

    output logic                         event_valid,
    input  logic                         event_ready,
    output logic [TAG_BITS-1:0]          event_tag,
    output logic                         event_channel_last,
    output logic [2:0]                   event_beat,
    output logic [31:0]                  event_bits,
    output logic                         event_accept,

    output logic                         configured,
    output logic                         protocol_error,
    output logic                         busy
);
    localparam int INPUT_PTR_BITS = $clog2(INPUT_FIFO_DEPTH);
    localparam int OUTPUT_PTR_BITS = $clog2(OUTPUT_FIFO_DEPTH);
    localparam int OUTPUT_COUNT_BITS = $clog2(OUTPUT_FIFO_DEPTH + 1);
    localparam int PRODUCT_SLOTS = 96;

    logic fault_q;
    logic configured_q;
    logic signed [7:0] folded_left_q [0:9][0:2][0:15];
    logic signed [23:0] folded_bias_q [0:9][0:15];
    logic signed [23:0] threshold_q;

    logic signed [7:0] input_rank_mem
        [0:INPUT_FIFO_DEPTH-1][0:2][0:15];
    logic [TAG_BITS-1:0] input_tag_mem [0:INPUT_FIFO_DEPTH-1];
    logic input_last_mem [0:INPUT_FIFO_DEPTH-1];
    logic [INPUT_PTR_BITS-1:0] input_wr_ptr_q;
    logic [INPUT_PTR_BITS-1:0] input_rd_ptr_q;
    logic [INPUT_PTR_BITS:0] input_count_q;

    logic service_active_q;
    logic [2:0] service_phase_q;

    logic [TAG_BITS-1:0] output_tag_mem [0:OUTPUT_FIFO_DEPTH-1];
    logic output_last_mem [0:OUTPUT_FIFO_DEPTH-1];
    logic [2:0] output_beat_mem [0:OUTPUT_FIFO_DEPTH-1];
    logic [31:0] output_bits_mem [0:OUTPUT_FIFO_DEPTH-1];
    logic [OUTPUT_PTR_BITS-1:0] output_wr_ptr_q;
    logic [OUTPUT_PTR_BITS-1:0] output_rd_ptr_q;
    logic [OUTPUT_COUNT_BITS-1:0] output_count_q;

    logic signed [7:0] multiplier_a [0:PRODUCT_SLOTS-1];
    logic signed [7:0] multiplier_b [0:PRODUCT_SLOTS-1];
    wire signed [15:0] multiplier_product [0:PRODUCT_SLOTS-1];

    logic safe_config_boundary;
    logic illegal_config;
    logic illegal_rank;
    logic illegal_request;
    logic input_capacity;
    logic output_capacity_five;
    logic output_capacity_five_after_push;
    logic input_push;
    logic input_release;
    logic output_push;
    logic output_pop;
    logic next_input_available;
    logic [OUTPUT_COUNT_BITS:0] output_count_after_push;

`ifndef SYNTHESIS
    initial begin
        if (TAG_BITS != 16 || INPUT_FIFO_DEPTH != 2
                || OUTPUT_FIFO_DEPTH != 16)
            $fatal(1, "M166 production geometry drift");
        if ((1 << INPUT_PTR_BITS) != INPUT_FIFO_DEPTH
                || (1 << OUTPUT_PTR_BITS) != OUTPUT_FIFO_DEPTH)
            $fatal(1, "M166 FIFO depths must be powers of two");
    end
`endif

    assign output_pop = (output_count_q != 0) && event_ready;
    assign output_push = service_active_q;
    assign input_release = service_active_q && service_phase_q == 4;
    assign input_capacity = (input_count_q < INPUT_FIFO_DEPTH)
        || input_release;
    assign output_capacity_five
        = output_count_q <= OUTPUT_FIFO_DEPTH - 5;

    always_comb begin
        output_count_after_push = output_count_q;
        if (output_push)
            output_count_after_push = output_count_after_push + 1'b1;
        if (output_pop)
            output_count_after_push = output_count_after_push - 1'b1;
    end
    assign output_capacity_five_after_push
        = output_count_after_push <= OUTPUT_FIFO_DEPTH - 5;

    assign safe_config_boundary = !service_active_q
        && input_count_q == 0 && output_count_q == 0;
    assign illegal_config = config_valid && !safe_config_boundary;
    assign illegal_rank = rank_valid && !configured_q;
    assign illegal_request = (config_valid && rank_valid)
        || illegal_config || illegal_rank;

    assign config_ready = !fault_q && !rank_valid && safe_config_boundary;
    assign config_accept = config_valid && config_ready;
    assign rank_ready = !fault_q && !config_valid && !illegal_rank
        && input_capacity;
    assign rank_accept = rank_valid && rank_ready;
    assign input_push = rank_accept;

    assign event_valid = output_count_q != 0;
    assign event_accept = event_valid && event_ready;
    assign event_tag = output_tag_mem[output_rd_ptr_q];
    assign event_channel_last = output_last_mem[output_rd_ptr_q];
    assign event_beat = output_beat_mem[output_rd_ptr_q];
    assign event_bits = output_bits_mem[output_rd_ptr_q];

    assign configured = configured_q;
    assign protocol_error = fault_q || illegal_request;
    assign busy = service_active_q || input_count_q != 0
        || output_count_q != 0;
    assign next_input_available = (input_count_q > 1) || input_push;

    for (genvar product_slot = 0; product_slot < PRODUCT_SLOTS;
            product_slot++) begin : g_product
        assign multiplier_product[product_slot]
            = multiplier_a[product_slot] * multiplier_b[product_slot];
    end

    always_comb begin : select_multiplier_operands
        for (int slot = 0; slot < PRODUCT_SLOTS; slot++) begin
            multiplier_a[slot] = '0;
            multiplier_b[slot] = '0;
        end
        if (service_active_q) begin
            for (int row = 0; row < 2; row++) begin
                for (int lane = 0; lane < 16; lane++) begin
                    for (int rank = 0; rank < 3; rank++) begin
                        multiplier_a[((row*16+lane)*3)+rank]
                            = input_rank_mem[input_rd_ptr_q][rank][lane];
                        multiplier_b[((row*16+lane)*3)+rank]
                            = folded_left_q[(service_phase_q*2)+row]
                                [rank][lane];
                    end
                end
            end
        end
    end

    always_ff @(posedge clk_core) begin : state_update
        logic signed [25:0] reconstructed;
        logic [31:0] packed_events;
        if (rst_core) begin
            fault_q <= 1'b0;
            configured_q <= 1'b0;
            threshold_q <= '0;
            input_wr_ptr_q <= '0;
            input_rd_ptr_q <= '0;
            input_count_q <= '0;
            service_active_q <= 1'b0;
            service_phase_q <= '0;
            output_wr_ptr_q <= '0;
            output_rd_ptr_q <= '0;
            output_count_q <= '0;
        end else begin
            if (illegal_request)
                fault_q <= 1'b1;

            if (config_accept) begin
                configured_q <= 1'b1;
                threshold_q <= config_threshold;
                for (int time_index = 0; time_index < 10; time_index++) begin
                    for (int lane = 0; lane < 16; lane++) begin
                        folded_bias_q[time_index][lane]
                            <= config_folded_bias[time_index][lane];
                        for (int rank = 0; rank < 3; rank++)
                            folded_left_q[time_index][rank][lane]
                                <= config_folded_left[time_index][rank][lane];
                    end
                end
            end

            if (input_push) begin
                input_tag_mem[input_wr_ptr_q] <= rank_tag;
                input_last_mem[input_wr_ptr_q] <= rank_channel_last;
                for (int rank = 0; rank < 3; rank++)
                    for (int lane = 0; lane < 16; lane++)
                        input_rank_mem[input_wr_ptr_q][rank][lane]
                            <= rank_data[rank][lane];
                input_wr_ptr_q <= input_wr_ptr_q + 1'b1;
            end

            if (output_pop)
                output_rd_ptr_q <= output_rd_ptr_q + 1'b1;

            if (output_push) begin
                packed_events = '0;
                for (int row = 0; row < 2; row++) begin
                    for (int lane = 0; lane < 16; lane++) begin
                        reconstructed = {
                            {2{folded_bias_q[(service_phase_q*2)+row]
                                [lane][23]}},
                            folded_bias_q[(service_phase_q*2)+row][lane]
                        } + {
                            {10{multiplier_product[((row*16+lane)*3)][15]}},
                            multiplier_product[((row*16+lane)*3)]
                        } + {
                            {10{multiplier_product[((row*16+lane)*3)+1][15]}},
                            multiplier_product[((row*16+lane)*3)+1]
                        } + {
                            {10{multiplier_product[((row*16+lane)*3)+2][15]}},
                            multiplier_product[((row*16+lane)*3)+2]
                        };
                        packed_events[(row*16)+lane]
                            = reconstructed >= $signed({
                                {2{threshold_q[23]}}, threshold_q
                            });
                    end
                end
                output_tag_mem[output_wr_ptr_q]
                    <= input_tag_mem[input_rd_ptr_q];
                output_last_mem[output_wr_ptr_q]
                    <= input_last_mem[input_rd_ptr_q];
                output_beat_mem[output_wr_ptr_q] <= service_phase_q;
                output_bits_mem[output_wr_ptr_q] <= packed_events;
                output_wr_ptr_q <= output_wr_ptr_q + 1'b1;
            end

            case ({input_push, input_release})
                2'b10: input_count_q <= input_count_q + 1'b1;
                2'b01: input_count_q <= input_count_q - 1'b1;
                default: input_count_q <= input_count_q;
            endcase
            case ({output_push, output_pop})
                2'b10: output_count_q <= output_count_q + 1'b1;
                2'b01: output_count_q <= output_count_q - 1'b1;
                default: output_count_q <= output_count_q;
            endcase

            if (!service_active_q) begin
                service_phase_q <= '0;
                if (input_count_q != 0 && output_capacity_five)
                    service_active_q <= 1'b1;
            end else if (service_phase_q == 4) begin
                input_rd_ptr_q <= input_rd_ptr_q + 1'b1;
                service_phase_q <= '0;
                if (next_input_available && output_capacity_five_after_push)
                    service_active_q <= 1'b1;
                else
                    service_active_q <= 1'b0;
            end else begin
                service_phase_q <= service_phase_q + 1'b1;
            end
        end
    end
endmodule

`default_nettype wire
