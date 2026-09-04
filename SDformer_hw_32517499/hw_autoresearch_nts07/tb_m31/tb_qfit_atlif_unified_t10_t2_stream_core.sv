`timescale 1ns/1ps
`default_nettype none

module tb_qfit_atlif_unified_t10_t2_stream_core;
    localparam int TAG_W = 48;
    localparam int T = 10;
    localparam int RANK = 3;
    localparam int T10_LANES = 16;
    localparam int T2_LANES = 24;
    localparam int ACC_W = 24;
    localparam int T10_FIRST_TILES = 24;
    localparam int T10_DIRECTED_TILES = 4;
    localparam int T10_TILES = T10_FIRST_TILES + T10_DIRECTED_TILES;
    localparam int T2_FIRST_PACKETS = 64;
    localparam int T2_STALL_PACKETS = 24;
    localparam int T2_PACKETS = T2_FIRST_PACKETS + T2_STALL_PACKETS + 1;
    localparam int DIRECTED_SHIFT = 8;
    localparam integer signed DIRECTED_THRESHOLD = 12345;

    logic clk_core = 1'b0;
    logic rst_core = 1'b1;
    logic parameter_valid = 1'b0;
    logic parameter_ready;
    logic parameter_mode = 1'b0;
    logic [(RANK*T*8)-1:0] parameter_t10_right_factor = '0;
    logic [(T*RANK*8)-1:0] parameter_t10_left_factor = '0;
    logic [(T*ACC_W)-1:0] parameter_t10_bias = '0;
    logic signed [ACC_W-1:0] parameter_t10_threshold = '0;
    logic [4:0] parameter_t10_requant_shift = '0;
    logic [31:0] parameter_t2_weight = '0;
    logic [47:0] parameter_t2_bias = '0;
    logic signed [23:0] parameter_t2_threshold = '0;
    logic parameter_loaded;
    logic loaded_mode;
    logic parameter_release_valid = 1'b0;
    logic parameter_release_ready;
    logic input_valid = 1'b0;
    logic input_ready;
    logic [TAG_W-1:0] input_tag = '0;
    logic [2:0] input_beat = '0;
    logic [23:0] input_lane_valid = '0;
    logic [255:0] input_port0_values = '0;
    logic [255:0] input_port1_values = '0;
    logic result_valid;
    logic result_ready = 1'b0;
    logic result_mode;
    logic [TAG_W-1:0] result_tag;
    logic [2:0] result_beat;
    logic [47:0] result_valid_bits;
    logic [47:0] result_bits;
    logic done;
    logic done_mode;
    logic [TAG_W-1:0] done_tag;
    logic protocol_error;
    logic busy;
    logic arithmetic_active;
    logic [1:0] issue_kind;
    logic [2:0] phase_cycle;
    logic [95:0] multiplier_active_mask;
    logic [4:0] result_fifo_occupancy;

    integer signed t10_cfg_right [0:RANK*T-1];
    integer signed t10_cfg_left [0:T*RANK-1];
    integer signed t10_cfg_bias [0:T-1];
    integer signed t10_x [0:T10_TILES-1][0:T*T10_LANES-1];
    integer signed t10_reference_intermediate
        [0:T10_TILES-1][0:RANK*T10_LANES-1];
    bit t10_reference_bits [0:T10_TILES-1][0:T*T10_LANES-1];

    integer signed t2_cfg_weight [0:3];
    integer signed t2_cfg_bias [0:1];
    integer signed t2_cfg_threshold;
    integer signed t2_x0 [0:T2_PACKETS-1][0:T2_LANES-1];
    integer signed t2_x1 [0:T2_PACKETS-1][0:T2_LANES-1];
    bit t2_reference_bits [0:T2_PACKETS-1][0:47];

    integer cycle_count = 0;
    integer parameter_fires = 0;
    integer release_fires = 0;
    integer mode_sequence_index = 0;
    integer input_beats = 0;
    integer t2_input_packets = 0;
    integer t10_output_beats = 0;
    integer t2_output_packets = 0;
    integer t10_done_tiles = 0;
    integer t2_done_packets = 0;
    integer t10_tile_starts = 0;
    integer t10_stage1_cycles = 0;
    integer t10_stage2_cycles = 0;
    integer t2_issue_cycles = 0;
    integer arithmetic_cycles = 0;
    integer previous_t10_start = -1;
    integer previous_t2_accept = -1;
    integer t2_ii_matches = 0;
    integer maximum_fifo_occupancy = 0;
    integer full_fifo_cycles = 0;
    integer full_pop_push_cycles = 0;
    integer input_stall_cycles = 0;
    integer output_stall_cycles = 0;
    integer release_input_collisions = 0;
    integer t10_intermediate_checked = 0;
    integer tie_cases = 0;
    integer positive_intermediate_saturations = 0;
    integer negative_intermediate_saturations = 0;
    integer positive_output_saturations = 0;
    integer negative_output_saturations = 0;
    integer threshold_equal_cases = 0;
    integer threshold_below_cases = 0;
    integer t2_positive_saturations = 0;
    integer t2_negative_saturations = 0;
    integer t2_nondegenerate_one_bits = 0;
    integer t2_nondegenerate_zero_bits = 0;
    integer t10_credit_wait_cycles = 0;
    logic check_t10_ii = 1'b1;
    logic force_sink_stall = 1'b0;
    logic sparse_stalls_enable = 1'b1;

    always #5 clk_core = ~clk_core;

    qfit_atlif_unified_t10_t2_stream_core dut (.*);

    function automatic integer signed rne_q8(
        input longint signed value, input integer shift
    );
        longint unsigned magnitude;
        longint unsigned quotient;
        longint unsigned remainder;
        longint unsigned half;
        longint signed rounded;
        begin
            magnitude = value < 0 ? -value : value;
            quotient = magnitude >> shift;
            remainder = shift == 0 ? 0
                : magnitude & ((64'd1 << shift)-1);
            half = shift == 0 ? 0 : 64'd1 << (shift-1);
            if (shift != 0
                && (remainder > half || (remainder == half && quotient[0])))
                quotient = quotient + 1;
            rounded = value < 0 ? -$signed(quotient) : $signed(quotient);
            if (rounded > 127)
                rne_q8 = 127;
            else if (rounded < -128)
                rne_q8 = -128;
            else
                rne_q8 = rounded;
        end
    endfunction

    function automatic integer signed clamp_q24(input longint signed value);
        if (value > 8388607)
            clamp_q24 = 8388607;
        else if (value < -8388608)
            clamp_q24 = -8388608;
        else
            clamp_q24 = value;
    endfunction

    task automatic compute_t10_reference(
        input int first_tile,
        input int last_tile,
        input int shift,
        input integer signed threshold,
        input bit count_coverage
    );
        integer signed intermediate [0:RANK*T10_LANES-1];
        longint signed accumulator;
        longint unsigned magnitude;
        longint unsigned quotient;
        longint unsigned remainder;
        longint unsigned half;
        longint signed rounded_unclamped;
        integer signed saturated;
        for (int tile = first_tile; tile < last_tile; tile++) begin
            for (int rank_index = 0; rank_index < RANK; rank_index++) begin
                for (int lane = 0; lane < T10_LANES; lane++) begin
                    accumulator = 0;
                    for (int row = 0; row < T; row++)
                        accumulator += t10_x[tile][row*T10_LANES+lane]
                            * t10_cfg_right[rank_index*T+row];
                    magnitude = accumulator < 0 ? -accumulator : accumulator;
                    quotient = magnitude >> shift;
                    remainder = shift == 0 ? 0
                        : magnitude & ((64'd1 << shift)-1);
                    half = shift == 0 ? 0 : 64'd1 << (shift-1);
                    if (count_coverage && shift != 0 && remainder == half)
                        tie_cases = tie_cases + 1;
                    if (shift != 0 && (remainder > half
                        || (remainder == half && quotient[0])))
                        quotient = quotient + 1;
                    rounded_unclamped = accumulator < 0
                        ? -$signed(quotient) : $signed(quotient);
                    if (count_coverage && rounded_unclamped > 127)
                        positive_intermediate_saturations
                            = positive_intermediate_saturations + 1;
                    if (count_coverage && rounded_unclamped < -128)
                        negative_intermediate_saturations
                            = negative_intermediate_saturations + 1;
                    intermediate[rank_index*T10_LANES+lane]
                        = rne_q8(accumulator, shift);
                    t10_reference_intermediate[tile]
                        [rank_index*T10_LANES+lane]
                        = intermediate[rank_index*T10_LANES+lane];
                end
            end
            for (int row = 0; row < T; row++) begin
                for (int lane = 0; lane < T10_LANES; lane++) begin
                    accumulator = t10_cfg_bias[row];
                    for (int rank_index = 0; rank_index < RANK; rank_index++)
                        accumulator += intermediate[rank_index*T10_LANES+lane]
                            * t10_cfg_left[row*RANK+rank_index];
                    if (count_coverage && accumulator > 8388607)
                        positive_output_saturations
                            = positive_output_saturations + 1;
                    if (count_coverage && accumulator < -8388608)
                        negative_output_saturations
                            = negative_output_saturations + 1;
                    saturated = clamp_q24(accumulator);
                    if (count_coverage && saturated == threshold)
                        threshold_equal_cases = threshold_equal_cases + 1;
                    if (count_coverage && saturated == threshold-1)
                        threshold_below_cases = threshold_below_cases + 1;
                    t10_reference_bits[tile][row*T10_LANES+lane]
                        = saturated >= threshold;
                end
            end
        end
    endtask

    task automatic build_t10_normal;
        for (int index = 0; index < RANK*T; index++)
            t10_cfg_right[index] = (index*13 + 5) % 15 - 7;
        for (int index = 0; index < T*RANK; index++)
            t10_cfg_left[index] = (index*17 + 3) % 15 - 7;
        for (int row = 0; row < T; row++)
            t10_cfg_bias[row] = row*37 - 160;
        for (int tile = 0; tile < T10_FIRST_TILES; tile++)
            for (int index = 0; index < T*T10_LANES; index++)
                t10_x[tile][index] = (tile*31 + index*19 + 11) % 16 - 8;
        compute_t10_reference(0, T10_FIRST_TILES, 3, 0, 1'b0);
    endtask

    task automatic build_t10_directed;
        for (int index = 0; index < RANK*T; index++)
            t10_cfg_right[index] = 0;
        for (int index = 0; index < T*RANK; index++)
            t10_cfg_left[index] = 0;
        for (int row = 0; row < T; row++)
            t10_cfg_bias[row] = 0;
        for (int row = 0; row < 6; row++)
            t10_cfg_right[row] = 1;
        for (int row = 0; row < 3; row++)
            t10_cfg_right[T+row] = 127;
        t10_cfg_bias[0] = DIRECTED_THRESHOLD;
        t10_cfg_bias[1] = DIRECTED_THRESHOLD-1;
        t10_cfg_bias[2] = 8388607;
        t10_cfg_bias[3] = -8388608;
        t10_cfg_left[(2*RANK)+1] = 127;
        t10_cfg_left[(3*RANK)+1] = 127;
        for (int row = 4; row < T; row++)
            t10_cfg_left[row*RANK] = row[0] ? -1 : 1;
        for (int tile = T10_FIRST_TILES; tile < T10_TILES; tile++) begin
            for (int index = 0; index < T*T10_LANES; index++)
                t10_x[tile][index] = 0;
            for (int lane = 0; lane < T10_LANES; lane++) begin
                case (lane % 8)
                    0: begin t10_x[tile][lane] = 127;
                             t10_x[tile][T10_LANES+lane] = 1; end
                    1: begin t10_x[tile][lane] = 127;
                             t10_x[tile][T10_LANES+lane] = 127;
                             t10_x[tile][(2*T10_LANES)+lane] = 127;
                             t10_x[tile][(3*T10_LANES)+lane] = 3; end
                    2: t10_x[tile][lane] = -128;
                    3: begin t10_x[tile][lane] = -128;
                             t10_x[tile][T10_LANES+lane] = -128;
                             t10_x[tile][(2*T10_LANES)+lane] = -128; end
                    4: begin t10_x[tile][lane] = 127;
                             t10_x[tile][T10_LANES+lane] = 127;
                             t10_x[tile][(2*T10_LANES)+lane] = 127;
                             t10_x[tile][(3*T10_LANES)+lane] = 127;
                             t10_x[tile][(4*T10_LANES)+lane] = 127;
                             t10_x[tile][(5*T10_LANES)+lane] = 5; end
                    5: begin t10_x[tile][lane] = -128;
                             t10_x[tile][T10_LANES+lane] = -128;
                             t10_x[tile][(2*T10_LANES)+lane] = -128;
                             t10_x[tile][(3*T10_LANES)+lane] = -128;
                             t10_x[tile][(4*T10_LANES)+lane] = -128; end
                    6: begin t10_x[tile][lane] = 127;
                             t10_x[tile][T10_LANES+lane] = 127;
                             t10_x[tile][(2*T10_LANES)+lane] = 2; end
                    default: begin t10_x[tile][lane] = -128;
                                   t10_x[tile][T10_LANES+lane] = -128; end
                endcase
            end
        end
        compute_t10_reference(T10_FIRST_TILES, T10_TILES,
            DIRECTED_SHIFT, DIRECTED_THRESHOLD, 1'b1);
    endtask

    task automatic compute_t2_reference_range(
        input int first_packet, input int last_packet,
        input bit count_saturation, input bit count_diversity
    );
        longint signed sum0;
        longint signed sum1;
        integer signed saturated0;
        integer signed saturated1;
        bit reference0;
        bit reference1;
        for (int packet = first_packet; packet < last_packet; packet++) begin
            for (int lane = 0; lane < T2_LANES; lane++) begin
                sum0 = t2_cfg_bias[0]
                    + t2_x0[packet][lane]*t2_cfg_weight[0]
                    + t2_x1[packet][lane]*t2_cfg_weight[1];
                sum1 = t2_cfg_bias[1]
                    + t2_x0[packet][lane]*t2_cfg_weight[2]
                    + t2_x1[packet][lane]*t2_cfg_weight[3];
                if (count_saturation && sum0 > 8388607)
                    t2_positive_saturations = t2_positive_saturations + 1;
                if (count_saturation && sum1 < -8388608)
                    t2_negative_saturations = t2_negative_saturations + 1;
                saturated0 = clamp_q24(sum0);
                saturated1 = clamp_q24(sum1);
                reference0 = saturated0 >= t2_cfg_threshold;
                reference1 = saturated1 >= t2_cfg_threshold;
                t2_reference_bits[packet][lane] = reference0;
                t2_reference_bits[packet][T2_LANES+lane] = reference1;
                if (count_diversity) begin
                    t2_nondegenerate_one_bits = t2_nondegenerate_one_bits
                        + reference0 + reference1;
                    t2_nondegenerate_zero_bits = t2_nondegenerate_zero_bits
                        + (!reference0) + (!reference1);
                end
            end
        end
    endtask

    task automatic build_t2_nondegenerate;
        t2_cfg_weight[0] = 3;
        t2_cfg_weight[1] = -5;
        t2_cfg_weight[2] = 7;
        t2_cfg_weight[3] = 2;
        t2_cfg_bias[0] = 100;
        t2_cfg_bias[1] = -200;
        t2_cfg_threshold = 0;
        for (int packet = 0; packet < T2_FIRST_PACKETS; packet++) begin
            for (int lane = 0; lane < T2_LANES; lane++) begin
                t2_x0[packet][lane]
                    = (packet*37 + lane*19 + 7) % 256 - 128;
                t2_x1[packet][lane]
                    = (packet*23 + lane*31 + 11) % 256 - 128;
            end
        end
        compute_t2_reference_range(0, T2_FIRST_PACKETS, 1'b0, 1'b1);
    endtask

    task automatic build_t2_saturation;
        t2_cfg_weight[0] = 127;
        t2_cfg_weight[1] = 127;
        t2_cfg_weight[2] = 127;
        t2_cfg_weight[3] = 127;
        t2_cfg_bias[0] = 8388607;
        t2_cfg_bias[1] = -8388608;
        t2_cfg_threshold = -1;
        for (int packet = T2_FIRST_PACKETS;
             packet < T2_PACKETS; packet++) begin
            for (int lane = 0; lane < T2_LANES; lane++) begin
                t2_x0[packet][lane] = ((packet+lane) % 2) ? -128 : 127;
                t2_x1[packet][lane] = ((packet+lane) % 2) ? -128 : 127;
            end
        end
        compute_t2_reference_range(T2_FIRST_PACKETS, T2_PACKETS,
            1'b1, 1'b0);
    endtask

    task automatic load_t10(
        input int shift, input integer signed threshold
    );
        parameter_mode = 1'b0;
        for (int index = 0; index < RANK*T; index++)
            parameter_t10_right_factor[(index*8)+:8]
                = t10_cfg_right[index][7:0];
        for (int index = 0; index < T*RANK; index++)
            parameter_t10_left_factor[(index*8)+:8]
                = t10_cfg_left[index][7:0];
        for (int row = 0; row < T; row++)
            parameter_t10_bias[(row*ACC_W)+:ACC_W]
                = t10_cfg_bias[row][ACC_W-1:0];
        parameter_t10_threshold = threshold[ACC_W-1:0];
        parameter_t10_requant_shift = shift[4:0];
        @(negedge clk_core);
        parameter_valid = 1'b1;
        do @(posedge clk_core); while (!parameter_ready);
        @(negedge clk_core);
        parameter_valid = 1'b0;
    endtask

    task automatic load_t2;
        parameter_mode = 1'b1;
        for (int index = 0; index < 4; index++)
            parameter_t2_weight[(index*8)+:8]
                = t2_cfg_weight[index][7:0];
        for (int index = 0; index < 2; index++)
            parameter_t2_bias[(index*ACC_W)+:ACC_W]
                = t2_cfg_bias[index][ACC_W-1:0];
        parameter_t2_threshold = t2_cfg_threshold[ACC_W-1:0];
        @(negedge clk_core);
        parameter_valid = 1'b1;
        do @(posedge clk_core); while (!parameter_ready);
        @(negedge clk_core);
        parameter_valid = 1'b0;
    endtask

    task automatic release_context;
        @(negedge clk_core);
        parameter_release_valid = 1'b1;
        do @(posedge clk_core); while (!parameter_release_ready);
        @(negedge clk_core);
        parameter_release_valid = 1'b0;
    endtask

    task automatic send_t10_tile(input int tile);
        input_lane_valid = 24'h00ffff;
        input_port1_values = '0;
        for (int beat = 0; beat < 5; beat++) begin
            input_port0_values = '0;
            for (int index = 0; index < 32; index++)
                input_port0_values[(index*8)+:8]
                    = t10_x[tile][(beat*32)+index][7:0];
            input_tag = 48'h4d310a000000 + tile;
            input_beat = beat[2:0];
            input_valid = 1'b1;
            do @(posedge clk_core); while (!input_ready);
            @(negedge clk_core);
        end
        input_valid = 1'b0;
    endtask

    task automatic drive_t2_packet(input int packet);
        input_tag = 48'h4d3102000000 + packet;
        input_beat = '0;
        input_lane_valid = {T2_LANES{1'b1}};
        input_port0_values = '0;
        input_port1_values = '0;
        for (int lane = 0; lane < T2_LANES; lane++) begin
            input_port0_values[(lane*8)+:8] = t2_x0[packet][lane][7:0];
            input_port1_values[(lane*8)+:8] = t2_x1[packet][lane][7:0];
        end
    endtask

    task automatic send_t2_range(input int first_packet, input int last_packet);
        @(negedge clk_core);
        input_valid = 1'b1;
        for (int packet = first_packet; packet < last_packet; packet++) begin
            drive_t2_packet(packet);
            do @(posedge clk_core); while (!input_ready);
            @(negedge clk_core);
        end
        input_valid = 1'b0;
    endtask

    task automatic release_t2_with_input_collision(input int packet);
        @(negedge clk_core);
        parameter_release_valid = 1'b1;
        input_valid = 1'b1;
        drive_t2_packet(packet);
        #1;
        if (parameter_release_ready || !input_ready)
            $fatal(1, "M31 T2 input-priority release arbitration failed");
        @(posedge clk_core);
        @(negedge clk_core);
        input_valid = 1'b0;
        do @(posedge clk_core); while (!parameter_release_ready);
        @(negedge clk_core);
        parameter_release_valid = 1'b0;
        release_input_collisions = release_input_collisions + 1;
    endtask

    always @(posedge clk_core) begin
        if (!rst_core) begin
            cycle_count = cycle_count + 1;
            if (parameter_valid && parameter_ready) begin
                if ((mode_sequence_index == 0 && parameter_mode != 0)
                    || (mode_sequence_index inside {[1:2]}
                        && parameter_mode != 1)
                    || (mode_sequence_index == 3 && parameter_mode != 0))
                    $fatal(1, "M31 context mode order mismatch index=%0d mode=%0d",
                           mode_sequence_index, parameter_mode);
                mode_sequence_index = mode_sequence_index + 1;
                parameter_fires = parameter_fires + 1;
            end
            if (parameter_release_valid && parameter_release_ready)
                release_fires = release_fires + 1;
            if (input_valid && !input_ready)
                input_stall_cycles = input_stall_cycles + 1;
            if (input_valid && input_ready) begin
                input_beats = input_beats + 1;
                if (loaded_mode) begin
                    if (t2_input_packets < T2_FIRST_PACKETS) begin
                        if (previous_t2_accept >= 0) begin
                            if (cycle_count-previous_t2_accept != 1)
                                $fatal(1, "M31 T2 II drift got=%0d",
                                       cycle_count-previous_t2_accept);
                            t2_ii_matches = t2_ii_matches + 1;
                        end
                        previous_t2_accept = cycle_count;
                    end
                    t2_input_packets = t2_input_packets + 1;
                end
            end
            if (arithmetic_active) begin
                arithmetic_cycles = arithmetic_cycles + 1;
                if (multiplier_active_mask !== {96{1'b1}})
                    $fatal(1, "M31 active multiplier mask mismatch");
                case (issue_kind)
                    1: t10_stage1_cycles = t10_stage1_cycles + 1;
                    2: t10_stage2_cycles = t10_stage2_cycles + 1;
                    3: t2_issue_cycles = t2_issue_cycles + 1;
                    default: $fatal(1, "M31 active issue kind invalid");
                endcase
            end
            if (issue_kind == 1 && phase_cycle == 0) begin
                if (check_t10_ii && previous_t10_start >= 0
                    && cycle_count-previous_t10_start != 10)
                    $fatal(1, "M31 T10 II drift got=%0d expected=10",
                           cycle_count-previous_t10_start);
                previous_t10_start = cycle_count;
                t10_tile_starts = t10_tile_starts + 1;
            end
            if (result_valid && !result_ready)
                output_stall_cycles = output_stall_cycles + 1;
            if (result_fifo_occupancy > maximum_fifo_occupancy)
                maximum_fifo_occupancy = result_fifo_occupancy;
            if (result_fifo_occupancy == 16)
                full_fifo_cycles = full_fifo_cycles + 1;
            if (!loaded_mode && busy && issue_kind == 0
                && result_fifo_occupancy >= 12)
                t10_credit_wait_cycles = t10_credit_wait_cycles + 1;
            if (result_fifo_occupancy == 16 && result_valid && result_ready
                && input_valid && input_ready && loaded_mode)
                full_pop_push_cycles = full_pop_push_cycles + 1;
            if (result_valid && result_ready) begin
                if (!result_mode) begin
                    int tile;
                    tile = t10_output_beats / 5;
                    if (result_tag !== 48'h4d310a000000 + tile
                        || result_beat !== (t10_output_beats % 5)
                        || result_valid_bits
                            !== {{16{1'b0}}, {32{1'b1}}}
                        || result_bits[47:32] !== '0)
                        $fatal(1, "M31 T10 output identity mismatch beat=%0d",
                               t10_output_beats);
                    for (int index = 0; index < 32; index++)
                        if (result_bits[index]
                            !== t10_reference_bits[tile]
                                [(result_beat*32)+index])
                            $fatal(1, "M31 T10 arithmetic mismatch tile=%0d beat=%0d index=%0d",
                                   tile, result_beat, index);
                    t10_output_beats = t10_output_beats + 1;
                end else begin
                    if (result_tag !== 48'h4d3102000000 + t2_output_packets
                        || result_beat != 0
                        || result_valid_bits !== {48{1'b1}})
                        $fatal(1, "M31 T2 output identity mismatch packet=%0d",
                               t2_output_packets);
                    for (int index = 0; index < 48; index++)
                        if (result_bits[index]
                            !== t2_reference_bits[t2_output_packets][index])
                            $fatal(1, "M31 T2 arithmetic mismatch packet=%0d bit=%0d",
                                   t2_output_packets, index);
                    t2_output_packets = t2_output_packets + 1;
                end
            end
            if (done) begin
                if (!done_mode) begin
                    if (done_tag !== 48'h4d310a000000 + t10_done_tiles)
                        $fatal(1, "M31 T10 done order mismatch index=%0d",
                               t10_done_tiles);
                    t10_done_tiles = t10_done_tiles + 1;
                end else begin
                    if (done_tag !== 48'h4d3102000000 + t2_done_packets)
                        $fatal(1, "M31 T2 done order mismatch index=%0d",
                               t2_done_packets);
                    t2_done_packets = t2_done_packets + 1;
                end
            end
        end
    end

    always @(negedge clk_core) begin
        if (rst_core)
            result_ready <= 1'b0;
        else
            result_ready <= !force_sink_stall
                && !(sparse_stalls_enable
                    && ((cycle_count % 29) inside {[11:13]}));
    end

    always @(negedge clk_core) begin
        if (!rst_core && issue_kind == 2 && phase_cycle == 0) begin
            for (int index = 0; index < RANK*T10_LANES; index++) begin
                if ($signed(dut.t10_intermediate_q[index])
                    !== t10_reference_intermediate
                        [t10_intermediate_checked][index])
                    $fatal(1, "M31 T10 intermediate mismatch tile=%0d index=%0d got=%0d expected=%0d",
                           t10_intermediate_checked, index,
                           $signed(dut.t10_intermediate_q[index]),
                           t10_reference_intermediate
                               [t10_intermediate_checked][index]);
            end
            t10_intermediate_checked = t10_intermediate_checked + 1;
        end
    end

    initial begin
`ifdef SIMULATOR_VCS
        $display("SIMULATOR=Synopsys VCS");
`else
        $fatal(1, "M31 evidence requires Synopsys VCS");
`endif
`ifdef SVA_RUNTIME_ENABLED
        $display("ASSERTIONS=enabled");
`else
        $fatal(1, "M31 evidence requires enabled SVA");
`endif
        if ($bits(dut.u_mul_pool.product) != 96*16)
            $fatal(1, "M31 sole multiplier pool width mismatch");

        build_t10_normal();
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;

        load_t10(3, 0);
        for (int tile = 0; tile < T10_FIRST_TILES; tile++)
            send_t10_tile(tile);
        while (t10_output_beats < T10_FIRST_TILES*5
            || t10_done_tiles < T10_FIRST_TILES || busy)
            @(posedge clk_core);
        check_t10_ii = 1'b0;
        release_context();

        sparse_stalls_enable = 1'b0;
        build_t2_nondegenerate();
        load_t2();
        send_t2_range(0, T2_FIRST_PACKETS);
        while (t2_output_packets < T2_FIRST_PACKETS || busy)
            @(posedge clk_core);
        release_context();

        build_t2_saturation();
        load_t2();
        force_sink_stall = 1'b1;
        fork
            send_t2_range(T2_FIRST_PACKETS, T2_PACKETS-1);
            begin
                repeat (40) @(posedge clk_core);
                force_sink_stall = 1'b0;
            end
        join
        while (t2_output_packets < T2_PACKETS-1 || busy)
            @(posedge clk_core);
        release_t2_with_input_collision(T2_PACKETS-1);
        while (t2_output_packets < T2_PACKETS || busy)
            @(posedge clk_core);

        build_t10_directed();
        sparse_stalls_enable = 1'b1;
        load_t10(DIRECTED_SHIFT, DIRECTED_THRESHOLD);
        force_sink_stall = 1'b1;
        fork
            begin
                for (int tile = T10_FIRST_TILES; tile < T10_TILES; tile++)
                    send_t10_tile(tile);
            end
            begin
                repeat (100) @(posedge clk_core);
                force_sink_stall = 1'b0;
            end
        join
        while (t10_output_beats < T10_TILES*5
            || t10_done_tiles < T10_TILES || busy)
            @(posedge clk_core);
        release_context();
        repeat (3) @(posedge clk_core);

        if (protocol_error || parameter_loaded || busy)
            $fatal(1, "M31 did not finish cleanly");
        if (parameter_fires != 4 || release_fires != 4
            || mode_sequence_index != 4 || release_input_collisions != 1)
            $fatal(1, "M31 context accounting mismatch load=%0d release=%0d mode=%0d collision=%0d",
                   parameter_fires, release_fires, mode_sequence_index,
                   release_input_collisions);
        if (input_beats != T10_TILES*5 + T2_PACKETS
            || t2_input_packets != T2_PACKETS)
            $fatal(1, "M31 input accounting mismatch beats=%0d t2=%0d",
                   input_beats, t2_input_packets);
        if (t10_output_beats != T10_TILES*5
            || t2_output_packets != T2_PACKETS
            || t10_done_tiles != T10_TILES
            || t2_done_packets != T2_PACKETS)
            $fatal(1, "M31 output accounting mismatch t10=%0d/%0d t2=%0d/%0d",
                   t10_output_beats, t10_done_tiles,
                   t2_output_packets, t2_done_packets);
        if (t10_tile_starts != T10_TILES
            || t10_stage1_cycles != T10_TILES*5
            || t10_stage2_cycles != T10_TILES*5
            || t2_issue_cycles != T2_PACKETS
            || arithmetic_cycles != T10_TILES*10 + T2_PACKETS)
            $fatal(1, "M31 arithmetic accounting mismatch starts=%0d s1=%0d s2=%0d t2=%0d all=%0d",
                   t10_tile_starts, t10_stage1_cycles, t10_stage2_cycles,
                   t2_issue_cycles, arithmetic_cycles);
        if (t2_ii_matches != T2_FIRST_PACKETS-1
            || t10_intermediate_checked != T10_TILES)
            $fatal(1, "M31 miter/II accounting mismatch t2ii=%0d intermediate=%0d",
                   t2_ii_matches, t10_intermediate_checked);
        if (maximum_fifo_occupancy != 16 || full_fifo_cycles == 0
            || full_pop_push_cycles == 0 || input_stall_cycles == 0
            || output_stall_cycles == 0 || t10_credit_wait_cycles == 0)
            $fatal(1, "M31 FIFO coverage incomplete max=%0d full=%0d poppush=%0d install=%0d outstall=%0d t10wait=%0d",
                   maximum_fifo_occupancy, full_fifo_cycles,
                   full_pop_push_cycles, input_stall_cycles,
                   output_stall_cycles, t10_credit_wait_cycles);
        if (tie_cases < 8 || positive_intermediate_saturations == 0
            || negative_intermediate_saturations == 0
            || positive_output_saturations == 0
            || negative_output_saturations == 0
            || threshold_equal_cases == 0 || threshold_below_cases == 0
            || t2_positive_saturations == 0 || t2_negative_saturations == 0
            || t2_nondegenerate_one_bits == 0
            || t2_nondegenerate_zero_bits == 0)
            $fatal(1, "M31 directed coverage incomplete ties=%0d t10_mid=%0d/%0d t10_out=%0d/%0d threshold=%0d/%0d t2_sat=%0d/%0d t2_diversity=%0d/%0d",
                   tie_cases, positive_intermediate_saturations,
                   negative_intermediate_saturations,
                   positive_output_saturations, negative_output_saturations,
                   threshold_equal_cases, threshold_below_cases,
                   t2_positive_saturations, t2_negative_saturations,
                   t2_nondegenerate_one_bits,
                   t2_nondegenerate_zero_bits);
        $display("M31_PASS modes=T10_T2N_T2S_T10 sole_mul_pool=1 mul_slots=96 t10_tiles=%0d t10_arithmetic=%0d t10_ii=10 t10_credit_wait=%0d t2_packets=%0d t2_nondegenerate=%0d t2_ii=1 t2_ii_matches=%0d total_arithmetic=%0d max_fifo=%0d full_cycles=%0d full_pop_push=%0d stalls=%0d/%0d release_input_collisions=%0d t10_ties=%0d t10_mid_sat=%0d/%0d t10_out_sat=%0d/%0d threshold_eq_below=%0d/%0d t2_sat=%0d/%0d t2_diversity=%0d/%0d",
                 T10_TILES, T10_TILES*10, t10_credit_wait_cycles,
                 T2_PACKETS, T2_FIRST_PACKETS, t2_ii_matches,
                 arithmetic_cycles, maximum_fifo_occupancy,
                 full_fifo_cycles, full_pop_push_cycles,
                 input_stall_cycles, output_stall_cycles,
                 release_input_collisions, tie_cases,
                 positive_intermediate_saturations,
                 negative_intermediate_saturations,
                 positive_output_saturations, negative_output_saturations,
                 threshold_equal_cases, threshold_below_cases,
                 t2_positive_saturations, t2_negative_saturations,
                 t2_nondegenerate_one_bits,
                 t2_nondegenerate_zero_bits);
        $finish;
    end

    initial begin
        #300000;
        $fatal(1, "M31 timeout");
    end
endmodule

`default_nettype wire
