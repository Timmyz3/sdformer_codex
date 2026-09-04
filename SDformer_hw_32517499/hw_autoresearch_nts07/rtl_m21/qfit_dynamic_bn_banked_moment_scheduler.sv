`timescale 1ns/1ps

// M21 banked raw-moment scheduling RTL.
//
// One 16-channel arithmetic slice serializes each accepted 96-channel packet
// into six updates.  Sum/sumsq state is resident per lane tile and channel
// slice, allowing packets from different lane tiles to be interleaved.  This
// block deliberately stops at exact raw moments: it does not implement mean,
// variance, rsqrt, affine BatchNorm, ATLIF, or any system-level speedup.
//
// Current timing caveat: every slice of a last packet publishes a registered
// result and waits for its retirement before the next slice updates.  Those
// six registered-result retirement bubbles are not represented in the current
// Python M21 DSE.  A future low-risk throughput extension is a small result
// queue (or six-slice snapshot queue) decoupling state update from retirement;
// that buffering and its done accounting are intentionally outside this RTL.
module qfit_dynamic_bn_banked_moment_scheduler #(
    parameter int IN_W = 32,
    parameter int TAG_W = 48,
    parameter int MAX_REDUCTION_POPULATION = 4194304,
    parameter int MAX_LANE_TILES = 16,
    localparam int SLICE_LANES = 16,
    localparam int PACKET_LANES = 96,
    localparam int SLICES = PACKET_LANES / SLICE_LANES,
    localparam int FIFO_DEPTH = 4,
    localparam int COUNT_W = $clog2(MAX_REDUCTION_POPULATION + 1),
    localparam int LANE_TILE_W = (MAX_LANE_TILES <= 2) ? 1 : $clog2(MAX_LANE_TILES),
    localparam int ACTIVE_TILES_W = $clog2(MAX_LANE_TILES + 1),
    localparam int POP_GROWTH_W =
        (MAX_REDUCTION_POPULATION <= 1) ? 0 : $clog2(MAX_REDUCTION_POPULATION),
    localparam int SUM_W = IN_W + POP_GROWTH_W,
    localparam int SQUARE_W = (2 * IN_W) - 1,
    localparam int SUMSQ_W = SQUARE_W + POP_GROWTH_W
) (
    input  logic                                  clk_core,
    input  logic                                  rst_core,

    input  logic                                  operator_start_valid,
    output logic                                  operator_start_ready,
    input  logic [COUNT_W-1:0]                    operator_reduction_population,
    input  logic [ACTIVE_TILES_W-1:0]             operator_active_lane_tiles,
    input  logic [TAG_W-1:0]                      operator_start_tag,
    output logic                                  operator_start_legal,
    output logic                                  operator_active,
    output logic [COUNT_W-1:0]                    active_reduction_population,
    output logic [ACTIVE_TILES_W-1:0]             active_lane_tiles,
    output logic [TAG_W-1:0]                      active_tag,

    input  logic                                  packet_valid,
    output logic                                  packet_ready,
    input  logic [LANE_TILE_W-1:0]                packet_lane_tile_id,
    input  logic                                  packet_first,
    input  logic                                  packet_last,
    input  logic [(PACKET_LANES*IN_W)-1:0]        packet_values,
    output logic                                  packet_legal,
    output logic [COUNT_W-1:0]                    packet_accepted_count,

    output logic                                  result_valid,
    input  logic                                  result_ready,
    output logic [TAG_W-1:0]                      result_tag,
    output logic [LANE_TILE_W-1:0]                result_lane_tile_id,
    output logic [$clog2(SLICES)-1:0]             result_slice_id,
    output logic [COUNT_W-1:0]                    result_count,
    output logic [(SLICE_LANES*SUM_W)-1:0]        result_sum,
    output logic [(SLICE_LANES*SUMSQ_W)-1:0]      result_sumsq,

    output logic                                  operator_done,
    output logic [TAG_W-1:0]                      operator_done_tag,
    output logic                                  protocol_error,
    output logic [$clog2(FIFO_DEPTH+1)-1:0]       fifo_level,
    output logic [$clog2(SLICES)-1:0]             serializer_slice
);
    localparam int FIFO_PTR_W = $clog2(FIFO_DEPTH);
    localparam int FIFO_COUNT_W = $clog2(FIFO_DEPTH + 1);
    localparam int SLICE_W = $clog2(SLICES);
    localparam int RESULT_TARGET_W = $clog2((MAX_LANE_TILES*SLICES) + 1);

    logic operator_active_q;
    logic [COUNT_W-1:0] population_q;
    logic [ACTIVE_TILES_W-1:0] active_tiles_q;
    logic [TAG_W-1:0] tag_q;
    logic protocol_error_q;
    logic operator_done_q;
    logic [TAG_W-1:0] operator_done_tag_q;

    logic [COUNT_W-1:0] lane_accepted_q [0:MAX_LANE_TILES-1];
    logic lane_final_enqueued_q [0:MAX_LANE_TILES-1];

    logic [(PACKET_LANES*IN_W)-1:0] fifo_values_q [0:FIFO_DEPTH-1];
    logic [LANE_TILE_W-1:0] fifo_lane_tile_q [0:FIFO_DEPTH-1];
    logic fifo_first_q [0:FIFO_DEPTH-1];
    logic fifo_last_q [0:FIFO_DEPTH-1];
    logic [FIFO_PTR_W-1:0] fifo_write_ptr_q;
    logic [FIFO_PTR_W-1:0] fifo_read_ptr_q;
    logic [FIFO_COUNT_W-1:0] fifo_count_q;

    logic signed [SUM_W-1:0] sum_bank_q
        [0:MAX_LANE_TILES-1][0:SLICES-1][0:SLICE_LANES-1];
    logic [SUMSQ_W-1:0] sumsq_bank_q
        [0:MAX_LANE_TILES-1][0:SLICES-1][0:SLICE_LANES-1];

    logic [SLICE_W-1:0] slice_q;
    logic result_valid_q;
    logic [TAG_W-1:0] result_tag_q;
    logic [LANE_TILE_W-1:0] result_lane_tile_q;
    logic [SLICE_W-1:0] result_slice_q;
    logic [COUNT_W-1:0] result_count_q;
    logic signed [SUM_W-1:0] result_sum_q [0:SLICE_LANES-1];
    logic [SUMSQ_W-1:0] result_sumsq_q [0:SLICE_LANES-1];
    logic [RESULT_TARGET_W-1:0] results_retired_q;

    logic operator_start_fire;
    logic packet_fire;
    logic enqueue_fire;
    logic process_slice;
    logic result_fire;
    logic dequeue_candidate;
    logic dequeue_fire;
    logic illegal_packet_fire;
    logic lane_id_in_range;
    logic [COUNT_W-1:0] selected_accepted_count;
    logic selected_lane_final;

    function automatic logic [SQUARE_W-1:0] exact_square(
        input logic signed [IN_W-1:0] value
    );
        logic [IN_W-1:0] magnitude;
        logic [(2*IN_W)-1:0] product;
        begin
            magnitude = value[IN_W-1]
                ? (~$unsigned(value) + 1'b1) : $unsigned(value);
            product = magnitude * magnitude;
            exact_square = product[SQUARE_W-1:0];
        end
    endfunction

`ifndef SYNTHESIS
    initial begin
        if (IN_W < 2)
            $fatal(1, "M21 IN_W must be at least two signed bits");
        if (MAX_REDUCTION_POPULATION < 1)
            $fatal(1, "M21 maximum reduction population must be positive");
        if (MAX_LANE_TILES < 1 || MAX_LANE_TILES > 16)
            $fatal(1, "M21 supports one through sixteen resident lane tiles");
        if (PACKET_LANES != 96 || SLICE_LANES != 16 || SLICES != 6)
            $fatal(1, "M21 freezes a 96-to-six-by-16 serializer");
        if (FIFO_DEPTH != 4)
            $fatal(1, "M21 freezes a four-packet payload FIFO");
    end
`endif

    assign operator_start_ready = !rst_core && !protocol_error_q
        && !operator_active_q && fifo_count_q == 0 && !result_valid_q;
    assign operator_start_fire = operator_start_valid && operator_start_ready;
    assign operator_start_legal = operator_start_ready
        && operator_reduction_population != 0
        && operator_reduction_population <= MAX_REDUCTION_POPULATION
        && operator_active_lane_tiles != 0
        && operator_active_lane_tiles <= MAX_LANE_TILES;

    always_comb begin
        lane_id_in_range = 1'b0;
        selected_accepted_count = '0;
        selected_lane_final = 1'b0;
        if (packet_lane_tile_id < MAX_LANE_TILES) begin
            selected_accepted_count = lane_accepted_q[packet_lane_tile_id];
            selected_lane_final = lane_final_enqueued_q[packet_lane_tile_id];
            lane_id_in_range = packet_lane_tile_id < active_tiles_q;
        end
    end

    // dequeue_candidate is deliberately independent of packet_valid/ready and
    // packet legality.  It therefore allows a full FIFO to advertise ready
    // without a combinational loop.  An illegal accepted packet subsequently
    // suppresses both the candidate dequeue and its arithmetic update.
    assign dequeue_candidate = !rst_core && !protocol_error_q
        && operator_active_q && fifo_count_q != 0
        && ((!result_valid_q && !fifo_last_q[fifo_read_ptr_q]
             && slice_q == SLICES-1)
            || (result_valid_q && result_ready
                && fifo_last_q[fifo_read_ptr_q] && slice_q == SLICES-1));
    assign packet_ready = !rst_core && !protocol_error_q && operator_active_q
        && (fifo_count_q < FIFO_DEPTH || dequeue_candidate);
    assign packet_fire = packet_valid && packet_ready;
    assign packet_accepted_count = selected_accepted_count;
    assign packet_legal = packet_ready && lane_id_in_range
        && !selected_lane_final
        && (packet_first == (selected_accepted_count == 0))
        && (packet_last == ((selected_accepted_count + 1'b1) == population_q));
    assign enqueue_fire = packet_fire && packet_legal;
    assign illegal_packet_fire = packet_fire && !packet_legal;

    // An illegal enqueue wins over all datapath progress in that cycle.  The
    // partially accumulated operator is thereafter recoverable only by reset.
    assign process_slice = operator_active_q && fifo_count_q != 0
        && !result_valid_q && !protocol_error_q
        && !illegal_packet_fire;
    // Exported valid and internal retirement are the same handshake contract.
    // A concurrently accepted illegal packet suppresses valid combinationally,
    // before the sticky error register updates, so no external observer can
    // consume a result that this fail-closed operator will never retire.
    assign result_fire = result_valid && result_ready;
    assign dequeue_fire = dequeue_candidate && !illegal_packet_fire;

    assign operator_active = operator_active_q;
    assign active_reduction_population = population_q;
    assign active_lane_tiles = active_tiles_q;
    assign active_tag = tag_q;
    assign protocol_error = protocol_error_q;
    assign operator_done = operator_done_q;
    assign operator_done_tag = operator_done_tag_q;
    assign fifo_level = fifo_count_q;
    assign serializer_slice = slice_q;

    assign result_valid = result_valid_q && !protocol_error_q
        && !illegal_packet_fire;
    assign result_tag = result_tag_q;
    assign result_lane_tile_id = result_lane_tile_q;
    assign result_slice_id = result_slice_q;
    assign result_count = result_count_q;
    generate
        for (genvar lane = 0; lane < SLICE_LANES; lane++) begin : gen_result_pack
            assign result_sum[(lane*SUM_W) +: SUM_W] = result_sum_q[lane];
            assign result_sumsq[(lane*SUMSQ_W) +: SUMSQ_W] = result_sumsq_q[lane];
        end
    endgenerate

    always_ff @(posedge clk_core) begin : scheduler_state
        if (rst_core) begin
            operator_active_q <= 1'b0;
            population_q <= '0;
            active_tiles_q <= '0;
            tag_q <= '0;
            protocol_error_q <= 1'b0;
            operator_done_q <= 1'b0;
            operator_done_tag_q <= '0;
            fifo_write_ptr_q <= '0;
            fifo_read_ptr_q <= '0;
            fifo_count_q <= '0;
            slice_q <= '0;
            result_valid_q <= 1'b0;
            result_tag_q <= '0;
            result_lane_tile_q <= '0;
            result_slice_q <= '0;
            result_count_q <= '0;
            results_retired_q <= '0;
            for (int tile = 0; tile < MAX_LANE_TILES; tile++) begin
                lane_accepted_q[tile] <= '0;
                lane_final_enqueued_q[tile] <= 1'b0;
            end
            for (int lane = 0; lane < SLICE_LANES; lane++) begin
                result_sum_q[lane] <= '0;
                result_sumsq_q[lane] <= '0;
            end
        end else begin
            operator_done_q <= 1'b0;

            if (operator_start_fire && !operator_start_legal) begin
                protocol_error_q <= 1'b1;
            end else if (operator_start_fire) begin
                operator_active_q <= 1'b1;
                population_q <= operator_reduction_population;
                active_tiles_q <= operator_active_lane_tiles;
                tag_q <= operator_start_tag;
                results_retired_q <= '0;
                slice_q <= '0;
                for (int tile = 0; tile < MAX_LANE_TILES; tile++) begin
                    lane_accepted_q[tile] <= '0;
                    lane_final_enqueued_q[tile] <= 1'b0;
                end
            end

            if (packet_fire && !packet_legal) begin
                protocol_error_q <= 1'b1;
            end else if (enqueue_fire) begin
                fifo_values_q[fifo_write_ptr_q] <= packet_values;
                fifo_lane_tile_q[fifo_write_ptr_q] <= packet_lane_tile_id;
                fifo_first_q[fifo_write_ptr_q] <= packet_first;
                fifo_last_q[fifo_write_ptr_q] <= packet_last;
                fifo_write_ptr_q <= fifo_write_ptr_q + 1'b1;
                lane_accepted_q[packet_lane_tile_id]
                    <= selected_accepted_count + 1'b1;
                if (packet_last)
                    lane_final_enqueued_q[packet_lane_tile_id] <= 1'b1;
            end

            if (process_slice) begin
                for (int lane = 0; lane < SLICE_LANES; lane++) begin : update_slice
                    logic signed [IN_W-1:0] lane_value;
                    logic signed [SUM_W-1:0] value_extended;
                    logic [SQUARE_W-1:0] lane_square;
                    logic [SUMSQ_W-1:0] square_extended;
                    logic signed [SUM_W-1:0] next_sum;
                    logic [SUMSQ_W-1:0] next_sumsq;

                    lane_value = $signed(fifo_values_q[fifo_read_ptr_q]
                        [((slice_q*SLICE_LANES + lane)*IN_W) +: IN_W]);
                    value_extended = {{(SUM_W-IN_W){lane_value[IN_W-1]}}, lane_value};
                    lane_square = exact_square(lane_value);
                    square_extended = {{(SUMSQ_W-SQUARE_W){1'b0}}, lane_square};
                    next_sum = fifo_first_q[fifo_read_ptr_q]
                        ? value_extended
                        : sum_bank_q[fifo_lane_tile_q[fifo_read_ptr_q]][slice_q][lane]
                            + value_extended;
                    next_sumsq = fifo_first_q[fifo_read_ptr_q]
                        ? square_extended
                        : sumsq_bank_q[fifo_lane_tile_q[fifo_read_ptr_q]][slice_q][lane]
                            + square_extended;
                    sum_bank_q[fifo_lane_tile_q[fifo_read_ptr_q]][slice_q][lane]
                        <= next_sum;
                    sumsq_bank_q[fifo_lane_tile_q[fifo_read_ptr_q]][slice_q][lane]
                        <= next_sumsq;
                    if (fifo_last_q[fifo_read_ptr_q]) begin
                        result_sum_q[lane] <= next_sum;
                        result_sumsq_q[lane] <= next_sumsq;
                    end
                end

                if (fifo_last_q[fifo_read_ptr_q]) begin
                    result_valid_q <= 1'b1;
                    result_tag_q <= tag_q;
                    result_lane_tile_q <= fifo_lane_tile_q[fifo_read_ptr_q];
                    result_slice_q <= slice_q;
                    result_count_q <= population_q;
                end else if (slice_q == SLICES-1) begin
                    slice_q <= '0;
                end else begin
                    slice_q <= slice_q + 1'b1;
                end
            end

            if (result_fire) begin
                result_valid_q <= 1'b0;
                results_retired_q <= results_retired_q + 1'b1;
                if (slice_q == SLICES-1)
                    slice_q <= '0;
                else
                    slice_q <= slice_q + 1'b1;

                if ((results_retired_q + 1'b1)
                    == (active_tiles_q * SLICES)) begin
                    operator_active_q <= 1'b0;
                    operator_done_q <= 1'b1;
                    operator_done_tag_q <= tag_q;
                    population_q <= '0;
                    active_tiles_q <= '0;
                end
            end

            case ({enqueue_fire, dequeue_fire})
                2'b10: fifo_count_q <= fifo_count_q + 1'b1;
                2'b01: begin
                    fifo_count_q <= fifo_count_q - 1'b1;
                    fifo_read_ptr_q <= fifo_read_ptr_q + 1'b1;
                end
                2'b11: fifo_read_ptr_q <= fifo_read_ptr_q + 1'b1;
                default: begin end
            endcase
        end
    end
endmodule
