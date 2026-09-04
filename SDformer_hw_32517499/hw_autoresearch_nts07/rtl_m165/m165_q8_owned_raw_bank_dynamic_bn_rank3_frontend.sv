`timescale 1ns/1ps
`default_nettype none

// M165: Q8 early-requant candidate front end for the H67 FFN path.
//
// One input tile contains T=10 samples for 16 hidden channels.  The stream
// supplies two time rows per accepted beat, so five accepted beats expose 32
// Q8 values/cycle.  In the same five cycles the block:
//   * computes exact sum and sum-of-squares for dynamic-BN coefficient work;
//   * uses 96 signed 8x8 products/cycle for a rank-3 right projection;
//   * queues the 48 raw projection states; and
//   * shares 16 programmable RNE/saturating requantizers across three cycles.
//
// Dynamic-BN mean/variance, reciprocal square root, affine application,
// ATLIF, the rank-3 left projection and fc2 are explicit downstream cuts.
// The factor row sums are carried with each packet so the downstream block can
// implement alpha*(R*x) + offset*(R*1) rather than silently dropping R*1.
//
// The datapath is specialized to the frozen H67 maximum BN1 population:
// 192,000 Q8 samples per hidden channel.  Exact worst-case bounds are signed
// 26 bits for sum, unsigned 32 bits for sumsq and unsigned 18 bits for count.
// A longer channel fails closed before any bounded state can overflow.
// M165 additionally keeps a raw FIFO entry owned until its third requant rank
// is committed.  The requantizer reads that owned bank in place, eliminating
// the 48 x 19-bit quant_raw copy while preserving a two-entry elastic FIFO.
module m165_q8_owned_raw_bank_dynamic_bn_rank3_frontend #(
    parameter int TAG_BITS = 16,
    parameter int RAW_FIFO_DEPTH = 2,
    parameter int OUTPUT_FIFO_DEPTH = 2
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         config_valid,
    output logic                         config_ready,
    input  logic signed [7:0]            config_factor [0:2][0:9],
    input  logic [4:0]                   config_requant_shift,
    output logic                         config_accept,

    input  logic                         tile_valid,
    output logic                         tile_ready,
    input  logic [TAG_BITS-1:0]          tile_tag,
    input  logic [2:0]                   tile_beat,
    input  logic                         tile_channel_start,
    input  logic                         tile_channel_last,
    input  logic signed [7:0]            tile_data [0:1][0:15],
    output logic                         tile_accept,

    output logic                         rank_valid,
    input  logic                         rank_ready,
    output logic [TAG_BITS-1:0]          rank_tag,
    output logic                         rank_channel_last,
    output logic signed [7:0]            rank_data [0:2][0:15],
    output logic signed [11:0]           rank_factor_sum [0:2],
    output logic                         rank_accept,

    output logic                         moment_valid,
    input  logic                         moment_ready,
    output logic [TAG_BITS-1:0]          moment_tag,
    output logic [17:0]                  moment_count,
    output logic signed [25:0]           moment_sum [0:15],
    output logic [31:0]                  moment_sumsq [0:15],
    output logic                         moment_accept,

    output logic                         configured,
    output logic                         channel_active,
    output logic                         protocol_error,
    output logic                         busy
);
    localparam int RAW_PTR_BITS = $clog2(RAW_FIFO_DEPTH);
    localparam int OUT_PTR_BITS = $clog2(OUTPUT_FIFO_DEPTH);
    localparam int MAX_SAMPLES_PER_LANE = 192000;

    logic fault_q;
    logic configured_q;
    logic signed [7:0] factor_q [0:2][0:9];
    logic signed [11:0] factor_sum_q [0:2];
    logic signed [11:0] config_factor_sum [0:2];
    logic [4:0] requant_shift_q;

    logic channel_active_q;
    logic [TAG_BITS-1:0] channel_tag_q;
    logic [2:0] beat_expected_q;
    logic tile_last_q;
    logic signed [18:0] projection_acc_q [0:2][0:15];
    logic signed [25:0] channel_sum_q [0:15];
    logic [31:0] channel_sumsq_q [0:15];
    logic [17:0] channel_count_q;

    logic signed [15:0] product_a [0:2][0:15];
    logic signed [15:0] product_b [0:2][0:15];
    logic signed [16:0] product_pair [0:2][0:15];
    logic signed [18:0] projection_next [0:2][0:15];
    logic [15:0] square_term [0:1][0:15];
    logic signed [8:0] beat_sum [0:15];
    logic [16:0] beat_sumsq [0:15];
    logic signed [25:0] channel_sum_next [0:15];
    logic [31:0] channel_sumsq_next [0:15];
    logic [17:0] channel_count_next;

    logic signed [18:0] raw_mem [0:RAW_FIFO_DEPTH-1][0:2][0:15];
    logic [TAG_BITS-1:0] raw_tag_mem [0:RAW_FIFO_DEPTH-1];
    logic raw_last_mem [0:RAW_FIFO_DEPTH-1];
    logic [RAW_PTR_BITS-1:0] raw_wr_ptr_q;
    logic [RAW_PTR_BITS-1:0] raw_rd_ptr_q;
    logic [RAW_PTR_BITS:0] raw_count_q;

    logic quant_busy_q;
    logic [1:0] quant_rank_q;
    logic [TAG_BITS-1:0] quant_tag_q;
    logic quant_last_q;

    logic [OUTPUT_FIFO_DEPTH-1:0][TAG_BITS-1:0] out_tag_mem;
    logic [OUTPUT_FIFO_DEPTH-1:0] out_last_mem;
    logic signed [7:0] out_data_mem
        [0:OUTPUT_FIFO_DEPTH-1][0:2][0:15];
    logic [OUT_PTR_BITS-1:0] out_wr_ptr_q;
    logic [OUT_PTR_BITS-1:0] out_rd_ptr_q;
    logic [OUT_PTR_BITS:0] out_count_q;

    logic moment_valid_q;
    logic [TAG_BITS-1:0] moment_tag_q;
    logic [17:0] moment_count_q;
    logic signed [25:0] moment_sum_q [0:15];
    logic [31:0] moment_sumsq_q [0:15];

    logic safe_config_boundary;
    logic illegal_config;
    logic illegal_tile;
    logic illegal_request;
    logic raw_capacity;
    logic output_capacity;
    logic moment_capacity;
    logic raw_push;
    logic raw_start;
    logic raw_release;
    logic output_push;
    logic output_pop;
    logic moment_push;
    logic moment_pop;

`ifndef SYNTHESIS
    initial begin
        if (TAG_BITS != 16 || RAW_FIFO_DEPTH != 2
                || OUTPUT_FIFO_DEPTH != 2)
            $fatal(1, "M165 production geometry drift");
    end
`endif

    function automatic logic signed [7:0] requant_rne_sat(
        input logic signed [18:0] value,
        input logic [4:0] shift
    );
        logic negative;
        logic [24:0] magnitude;
        logic [24:0] quotient;
        logic [24:0] remainder;
        logic [24:0] half;
        logic round_up;
        logic signed [25:0] rounded;
        begin
            negative = value[18];
            magnitude = negative
                ? $unsigned(-$signed({{6{value[18]}}, value}))
                : $unsigned({6'b0, value});
            quotient = magnitude;
            remainder = '0;
            half = '0;
            round_up = 1'b0;
            if (shift != 0) begin
                quotient = magnitude >> shift;
                remainder = magnitude
                    & ((25'd1 << shift) - 25'd1);
                half = 25'd1 << (shift - 1'b1);
                round_up = (remainder > half)
                    || ((remainder == half) && quotient[0]);
            end
            if (round_up)
                quotient = quotient + 1'b1;
            rounded = negative ? -$signed({1'b0, quotient})
                               :  $signed({1'b0, quotient});
            if (rounded > 26'sd127)
                requant_rne_sat = 8'sd127;
            else if (rounded < -26'sd128)
                requant_rne_sat = -8'sd128;
            else
                requant_rne_sat = rounded[7:0];
        end
    endfunction

    always_comb begin : arithmetic_front
        for (int row = 0; row < 2; row++) begin
            for (int lane = 0; lane < 16; lane++) begin
                square_term[row][lane]
                    = $signed(tile_data[row][lane])
                    * $signed(tile_data[row][lane]);
            end
        end

        // BN1 normalizes each hidden channel independently.  Each lane is a
        // hidden channel and receives two temporal samples per beat; never
        // reduce moments across lanes.  The 32-square issue width therefore
        // feeds 16 independent two-sample accumulators.
        for (int lane = 0; lane < 16; lane++) begin
            beat_sum[lane]
                = $signed({tile_data[0][lane][7], tile_data[0][lane]})
                + $signed({tile_data[1][lane][7], tile_data[1][lane]});
            beat_sumsq[lane]
                = {1'b0, square_term[0][lane]}
                + {1'b0, square_term[1][lane]};
            channel_sum_next[lane] = channel_sum_q[lane]
                + {{17{beat_sum[lane][8]}}, beat_sum[lane]};
            channel_sumsq_next[lane] = channel_sumsq_q[lane]
                + {{15{1'b0}}, beat_sumsq[lane]};
        end

        for (int rank = 0; rank < 3; rank++) begin
            for (int lane = 0; lane < 16; lane++) begin
                product_a[rank][lane]
                    = $signed(tile_data[0][lane])
                    * $signed(factor_q[rank][{beat_expected_q, 1'b0}]);
                product_b[rank][lane]
                    = $signed(tile_data[1][lane])
                    * $signed(factor_q[rank][{beat_expected_q, 1'b1}]);
                product_pair[rank][lane]
                    = $signed({product_a[rank][lane][15],
                               product_a[rank][lane]})
                    + $signed({product_b[rank][lane][15],
                               product_b[rank][lane]});
                if (beat_expected_q == 0)
                    projection_next[rank][lane]
                        = {{2{product_pair[rank][lane][16]}},
                           product_pair[rank][lane]};
                else
                    projection_next[rank][lane]
                        = projection_acc_q[rank][lane]
                        + {{2{product_pair[rank][lane][16]}},
                           product_pair[rank][lane]};
            end
        end

        channel_count_next = channel_count_q + 18'd2;
    end

    always_comb begin : config_row_sums
        for (int rank = 0; rank < 3; rank++) begin
            config_factor_sum[rank] = '0;
            for (int time_index = 0; time_index < 10; time_index++)
                config_factor_sum[rank] = config_factor_sum[rank]
                    + {{4{config_factor[rank][time_index][7]}},
                       config_factor[rank][time_index]};
        end
    end

    assign output_pop = (out_count_q != 0) && rank_ready;
    assign rank_valid = out_count_q != 0;
    assign rank_accept = rank_valid && rank_ready;
    assign output_capacity = (out_count_q < OUTPUT_FIFO_DEPTH) || output_pop;
    assign output_push = quant_busy_q && quant_rank_q == 2;

    assign raw_start = !quant_busy_q && raw_count_q != 0
        && output_capacity;
    // raw_count includes the bank currently owned by the requantizer.  It is
    // released only when rank 2 commits, never when service merely starts.
    assign raw_release = output_push;
    assign raw_capacity = (raw_count_q < RAW_FIFO_DEPTH) || raw_release;
    assign raw_push = tile_accept && beat_expected_q == 4;

    assign moment_pop = moment_valid_q && moment_ready;
    assign moment_capacity = !moment_valid_q || moment_ready;
    assign moment_push = tile_accept && beat_expected_q == 4 && tile_last_q;

    always_comb begin : protocol_guard
        safe_config_boundary = !channel_active_q && beat_expected_q == 0
            && raw_count_q == 0 && !quant_busy_q && out_count_q == 0
            && !moment_valid_q;
        illegal_config = config_valid
            && (!safe_config_boundary || config_requant_shift > 23);
        illegal_tile = 1'b0;
        if (tile_valid) begin
            if (!configured_q || tile_beat != beat_expected_q)
                illegal_tile = 1'b1;
            if (channel_count_q > MAX_SAMPLES_PER_LANE - 2)
                illegal_tile = 1'b1;
            if (beat_expected_q == 0) begin
                if (tile_channel_start != !channel_active_q)
                    illegal_tile = 1'b1;
                if (channel_active_q && tile_tag != channel_tag_q)
                    illegal_tile = 1'b1;
            end else begin
                if (tile_channel_start || tile_channel_last
                        || tile_tag != channel_tag_q)
                    illegal_tile = 1'b1;
            end
        end
        illegal_request = (config_valid && tile_valid)
            || illegal_config || illegal_tile;
    end

    assign config_ready = !fault_q && !tile_valid && safe_config_boundary;
    assign config_accept = config_valid && config_ready;
    assign tile_ready = !fault_q && !config_valid && !illegal_tile
        && ((beat_expected_q != 4) || raw_capacity)
        && ((beat_expected_q != 4) || !tile_last_q || moment_capacity);
    assign tile_accept = tile_valid && tile_ready;

    assign configured = configured_q;
    assign channel_active = channel_active_q;
    assign protocol_error = fault_q || illegal_request;
    assign busy = channel_active_q || beat_expected_q != 0
        || raw_count_q != 0 || quant_busy_q || out_count_q != 0
        || moment_valid_q;

    assign rank_tag = out_tag_mem[out_rd_ptr_q];
    assign rank_channel_last = out_last_mem[out_rd_ptr_q];
    generate
        for (genvar output_rank = 0; output_rank < 3; output_rank++) begin : g_output_rank
            assign rank_factor_sum[output_rank] = factor_sum_q[output_rank];
            for (genvar output_lane = 0; output_lane < 16; output_lane++) begin : g_output_lane
                assign rank_data[output_rank][output_lane]
                    = out_data_mem[out_rd_ptr_q][output_rank][output_lane];
            end
        end
    endgenerate

    assign moment_valid = moment_valid_q;
    assign moment_accept = moment_valid_q && moment_ready;
    assign moment_tag = moment_tag_q;
    assign moment_count = moment_count_q;
    generate
        for (genvar moment_lane = 0; moment_lane < 16;
                moment_lane++) begin : g_moment_lane
            assign moment_sum[moment_lane] = moment_sum_q[moment_lane];
            assign moment_sumsq[moment_lane] = moment_sumsq_q[moment_lane];
        end
    endgenerate

    always_ff @(posedge clk_core) begin : state_update
        if (rst_core) begin
            fault_q <= 1'b0;
            configured_q <= 1'b0;
            requant_shift_q <= '0;
            channel_active_q <= 1'b0;
            channel_tag_q <= '0;
            beat_expected_q <= '0;
            tile_last_q <= 1'b0;
            channel_count_q <= '0;
            raw_wr_ptr_q <= '0;
            raw_rd_ptr_q <= '0;
            raw_count_q <= '0;
            quant_busy_q <= 1'b0;
            quant_rank_q <= '0;
            quant_tag_q <= '0;
            quant_last_q <= 1'b0;
            out_wr_ptr_q <= '0;
            out_rd_ptr_q <= '0;
            out_count_q <= '0;
            moment_valid_q <= 1'b0;
            moment_tag_q <= '0;
            moment_count_q <= '0;
            for (int lane = 0; lane < 16; lane++) begin
                channel_sum_q[lane] <= '0;
                channel_sumsq_q[lane] <= '0;
                moment_sum_q[lane] <= '0;
                moment_sumsq_q[lane] <= '0;
            end
            for (int rank = 0; rank < 3; rank++) begin
                factor_sum_q[rank] <= '0;
                for (int time_index = 0; time_index < 10; time_index++)
                    factor_q[rank][time_index] <= '0;
                for (int lane = 0; lane < 16; lane++) begin
                    projection_acc_q[rank][lane] <= '0;
                end
            end
        end else begin
            if (illegal_request)
                fault_q <= 1'b1;

            if (config_accept) begin
                configured_q <= 1'b1;
                requant_shift_q <= config_requant_shift;
                for (int rank = 0; rank < 3; rank++) begin
                    factor_sum_q[rank] <= config_factor_sum[rank];
                    for (int time_index = 0; time_index < 10;
                            time_index++) begin
                        factor_q[rank][time_index]
                            <= config_factor[rank][time_index];
                    end
                end
            end

            if (tile_accept) begin
                for (int rank = 0; rank < 3; rank++) begin
                    for (int lane = 0; lane < 16; lane++)
                        projection_acc_q[rank][lane]
                            <= projection_next[rank][lane];
                end
                for (int lane = 0; lane < 16; lane++) begin
                    channel_sum_q[lane] <= channel_sum_next[lane];
                    channel_sumsq_q[lane] <= channel_sumsq_next[lane];
                end
                channel_count_q <= channel_count_next;

                if (beat_expected_q == 0) begin
                    tile_last_q <= tile_channel_last;
                    if (tile_channel_start) begin
                        channel_active_q <= 1'b1;
                        channel_tag_q <= tile_tag;
                    end
                end
                if (beat_expected_q == 4) begin
                    beat_expected_q <= 0;
                    if (tile_last_q)
                        channel_active_q <= 1'b0;
                end else begin
                    beat_expected_q <= beat_expected_q + 1'b1;
                end
            end

            if (raw_push) begin
                raw_tag_mem[raw_wr_ptr_q] <= channel_tag_q;
                raw_last_mem[raw_wr_ptr_q] <= tile_last_q;
                for (int rank = 0; rank < 3; rank++)
                    for (int lane = 0; lane < 16; lane++)
                        raw_mem[raw_wr_ptr_q][rank][lane]
                            <= projection_next[rank][lane];
                raw_wr_ptr_q <= raw_wr_ptr_q + 1'b1;
            end
            if (raw_start) begin
                quant_tag_q <= raw_tag_mem[raw_rd_ptr_q];
                quant_last_q <= raw_last_mem[raw_rd_ptr_q];
                quant_busy_q <= 1'b1;
                quant_rank_q <= 0;
            end else if (quant_busy_q) begin
                for (int lane = 0; lane < 16; lane++)
                    out_data_mem[out_wr_ptr_q][quant_rank_q][lane]
                        <= requant_rne_sat(
                            raw_mem[raw_rd_ptr_q][quant_rank_q][lane],
                            requant_shift_q);
                if (quant_rank_q == 2) begin
                    out_tag_mem[out_wr_ptr_q] <= quant_tag_q;
                    out_last_mem[out_wr_ptr_q] <= quant_last_q;
                    out_wr_ptr_q <= out_wr_ptr_q + 1'b1;
                    quant_busy_q <= 1'b0;
                    raw_rd_ptr_q <= raw_rd_ptr_q + 1'b1;
                end else begin
                    quant_rank_q <= quant_rank_q + 1'b1;
                end
            end

            case ({raw_push, raw_release})
                2'b10: raw_count_q <= raw_count_q + 1'b1;
                2'b01: raw_count_q <= raw_count_q - 1'b1;
                default: raw_count_q <= raw_count_q;
            endcase
            if (output_pop)
                out_rd_ptr_q <= out_rd_ptr_q + 1'b1;
            case ({output_push, output_pop})
                2'b10: out_count_q <= out_count_q + 1'b1;
                2'b01: out_count_q <= out_count_q - 1'b1;
                default: out_count_q <= out_count_q;
            endcase

            if (moment_pop)
                moment_valid_q <= 1'b0;
            if (moment_push) begin
                moment_valid_q <= 1'b1;
                moment_tag_q <= channel_tag_q;
                moment_count_q <= channel_count_next;
                for (int lane = 0; lane < 16; lane++) begin
                    moment_sum_q[lane] <= channel_sum_next[lane];
                    moment_sumsq_q[lane] <= channel_sumsq_next[lane];
                    // The completed channel is snapshotted above.  Clear the
                    // active bank now so the next legal channel needs no
                    // tile_channel_start mux on either wide accumulator.
                    channel_sum_q[lane] <= '0;
                    channel_sumsq_q[lane] <= '0;
                end
                channel_count_q <= '0;
            end
        end
    end
endmodule

`default_nettype wire
