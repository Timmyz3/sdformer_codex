`timescale 1ns/1ps
`default_nettype none

module tb_m164_q8_bounded_dynamic_bn_rank3_frontend;
    localparam int TAG_BITS = 16;

    logic clk_core;
    logic rst_core;
    logic config_valid;
    logic config_ready;
    logic signed [7:0] config_factor [0:2][0:9];
    logic [4:0] config_requant_shift;
    logic config_accept;
    logic tile_valid;
    logic tile_ready;
    logic [15:0] tile_tag;
    logic [2:0] tile_beat;
    logic tile_channel_start;
    logic tile_channel_last;
    logic signed [7:0] tile_data [0:1][0:15];
    logic tile_accept;
    logic rank_valid;
    logic rank_ready;
    logic [15:0] rank_tag;
    logic rank_channel_last;
    logic signed [7:0] rank_data [0:2][0:15];
    logic signed [11:0] rank_factor_sum [0:2];
    logic rank_accept;
    logic moment_valid;
    logic moment_ready;
    logic [15:0] moment_tag;
    logic [17:0] moment_count;
    logic signed [25:0] moment_sum [0:15];
    logic [31:0] moment_sumsq [0:15];
    logic moment_accept;
    logic configured;
    logic channel_active;
    logic protocol_error;
    logic busy;

    logic force_rank_stall;
    logic force_moment_stall;
    int cycle_count;
    int input_beats;
    int rank_results;
    int moment_results;
    int rank_stall_cycles;
    int moment_stall_cycles;
    int input_gap_cycles;
    int protocol_attacks;
    int rne_half_even_checks;
    int saturation_checks;
    int shift23_checks;
    int factor_ref [0:2][0:9];
    int factor_sum_ref [0:2];
    int shift_ref;
    longint signed channel_sum_ref [0:15];
    longint signed channel_sumsq_ref [0:15];
    int channel_count_ref;

    typedef struct packed {
        logic [15:0] tag;
        logic last;
        logic [383:0] payload;
    } rank_expected_t;
    typedef struct packed {
        logic [15:0] tag;
        logic [17:0] count;
        logic [415:0] sum_payload;
        logic [511:0] sumsq_payload;
    } moment_expected_t;
    rank_expected_t rank_expected_q[$];
    moment_expected_t moment_expected_q[$];

    m164_q8_bounded_dynamic_bn_rank3_frontend dut (.*);
    m164_q8_bounded_dynamic_bn_rank3_frontend_assertions checks (.*);

    always #1.5 clk_core = ~clk_core;

    function automatic integer requant_ref(
        input longint signed value,
        input integer shift
    );
        longint unsigned magnitude;
        longint unsigned quotient;
        longint unsigned remainder;
        longint unsigned half;
        longint signed rounded;
        begin
            magnitude = value < 0 ? -value : value;
            quotient = magnitude;
            remainder = 0;
            half = 0;
            if (shift != 0) begin
                quotient = magnitude >> shift;
                remainder = magnitude & ((64'd1 << shift) - 1);
                half = 64'd1 << (shift - 1);
                if ((remainder > half)
                        || ((remainder == half) && quotient[0]))
                    quotient = quotient + 1;
            end
            rounded = value < 0 ? -quotient : quotient;
            if (rounded > 127)
                requant_ref = 127;
            else if (rounded < -128)
                requant_ref = -128;
            else
                requant_ref = rounded;
        end
    endfunction

    always @(negedge clk_core) begin
        if (rst_core) begin
            rank_ready <= 1'b0;
            moment_ready <= 1'b0;
        end else begin
            rank_ready <= !force_rank_stall
                && ((cycle_count % 13) != 2)
                && ((cycle_count % 17) != 6);
            moment_ready <= !force_moment_stall
                && ((cycle_count % 11) != 3)
                && ((cycle_count % 19) != 8);
        end
    end

    always @(posedge clk_core) begin : scoreboard
        rank_expected_t rank_expected;
        moment_expected_t moment_expected;
        if (rst_core) begin
            cycle_count <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (tile_accept)
                input_beats <= input_beats + 1;
            if (rank_valid && !rank_ready)
                rank_stall_cycles <= rank_stall_cycles + 1;
            if (moment_valid && !moment_ready)
                moment_stall_cycles <= moment_stall_cycles + 1;
            if (rank_accept) begin
                if (rank_expected_q.size() == 0)
                    $fatal(1, "M164 unexpected rank output");
                rank_expected = rank_expected_q.pop_front();
                if (rank_tag !== rank_expected.tag
                        || rank_channel_last !== rank_expected.last)
                    $fatal(1, "M164 rank metadata mismatch");
                for (int rank = 0; rank < 3; rank++) begin
                    if (rank_factor_sum[rank] !== factor_sum_ref[rank])
                        $fatal(1, "M164 factor row-sum mismatch rank=%0d",
                            rank);
                    for (int lane = 0; lane < 16; lane++) begin
                        if (rank_data[rank][lane]
                                !== $signed(rank_expected.payload[
                                    (rank * 16 + lane) * 8 +: 8]))
                            $fatal(1,
                                "M164 rank mismatch rank=%0d lane=%0d got=%0d expected=%0d",
                                rank, lane, rank_data[rank][lane],
                                $signed(rank_expected.payload[
                                    (rank * 16 + lane) * 8 +: 8]));
                    end
                end
                if (rank_tag == 16'h6314) begin
                    for (int rank = 0; rank < 3; rank++) begin
                        if (rank_data[rank][0] !== 8'sd0
                                || rank_data[rank][1] !== 8'sd2
                                || rank_data[rank][2] !== 8'sd0
                                || rank_data[rank][3] !== -8'sd2)
                            $fatal(1, "M164 explicit signed ties-to-even mismatch");
                    end
                    rne_half_even_checks <= rne_half_even_checks + 12;
                end
                if (rank_tag == 16'h6315) begin
                    for (int rank = 0; rank < 3; rank++) begin
                        if (rank_data[rank][0] !== 8'sd127
                                || rank_data[rank][1] !== -8'sd128)
                            $fatal(1, "M164 explicit saturation mismatch");
                    end
                    saturation_checks <= saturation_checks + 6;
                end
                if (rank_tag == 16'h6316) begin
                    for (int rank = 0; rank < 3; rank++) begin
                        if (rank_data[rank][0] !== 8'sd0)
                            $fatal(1, "M164 explicit shift23 mismatch");
                    end
                    shift23_checks <= shift23_checks + 3;
                end
                rank_results <= rank_results + 1;
            end
            if (moment_accept) begin
                if (moment_expected_q.size() == 0)
                    $fatal(1, "M164 unexpected moment output");
                moment_expected = moment_expected_q.pop_front();
                if (moment_tag !== moment_expected.tag
                        || moment_count !== moment_expected.count)
                    $fatal(1, "M164 moment metadata mismatch");
                for (int lane = 0; lane < 16; lane++) begin
                    if (moment_sum[lane]
                            !== $signed(moment_expected.sum_payload[
                                lane * 26 +: 26])
                            || moment_sumsq[lane]
                            !== moment_expected.sumsq_payload[
                                lane * 32 +: 32])
                        $fatal(1,
                            "M164 per-lane moment mismatch lane=%0d got sum=%0d sumsq=%0d expected sum=%0d sumsq=%0d",
                            lane, moment_sum[lane], moment_sumsq[lane],
                            $signed(moment_expected.sum_payload[
                                lane * 26 +: 26]),
                            moment_expected.sumsq_payload[
                                lane * 32 +: 32]);
                end
                moment_results <= moment_results + 1;
            end
            if (cycle_count > 200000)
                $fatal(1, "M164 watchdog timeout");
        end
    end

    task automatic reset_dut;
        begin
            rst_core = 1'b1;
            config_valid = 1'b0;
            tile_valid = 1'b0;
            force_rank_stall = 1'b0;
            force_moment_stall = 1'b0;
            repeat (4) @(negedge clk_core);
            rst_core = 1'b0;
        end
    endtask

    task automatic load_config;
        int value;
        begin
            shift_ref = 9;
            for (int rank = 0; rank < 3; rank++) begin
                factor_sum_ref[rank] = 0;
                for (int time_index = 0; time_index < 10;
                        time_index++) begin
                    if (rank == 0 && time_index == 0)
                        value = -128;
                    else if (rank == 2 && time_index == 9)
                        value = 127;
                    else
                        value = ((rank * 67 + time_index * 29 + 11)
                            % 63) - 31;
                    factor_ref[rank][time_index] = value;
                    factor_sum_ref[rank] += value;
                    config_factor[rank][time_index] = value;
                end
            end
            config_requant_shift = shift_ref;
            @(negedge clk_core);
            config_valid = 1'b1;
            do @(posedge clk_core); while (!config_accept);
            @(negedge clk_core);
            config_valid = 1'b0;
        end
    endtask

    task automatic load_directed_config(
        input int directed_shift,
        input int first_factor
    );
        begin
            shift_ref = directed_shift;
            for (int rank = 0; rank < 3; rank++) begin
                factor_sum_ref[rank] = first_factor;
                for (int time_index = 0; time_index < 10;
                        time_index++) begin
                    factor_ref[rank][time_index]
                        = time_index == 0 ? first_factor : 0;
                    config_factor[rank][time_index]
                        = time_index == 0 ? first_factor : 0;
                end
            end
            config_requant_shift = directed_shift;
            @(negedge clk_core);
            config_valid = 1'b1;
            do @(posedge clk_core); while (!config_accept);
            @(negedge clk_core);
            config_valid = 1'b0;
        end
    endtask

    task automatic send_tile(
        input int channel_index,
        input int tile_index,
        input logic start_flag,
        input logic last_flag,
        input int data_mode
    );
        int values [0:4][0:1][0:15];
        longint signed projection;
        rank_expected_t expected_rank;
        moment_expected_t expected_moment;
        int gap_cycles;
        begin
            expected_rank = '0;
            expected_moment = '0;
            expected_rank.tag = 16'h6300 + channel_index;
            expected_rank.last = last_flag;
            for (int beat = 0; beat < 5; beat++) begin
                for (int row = 0; row < 2; row++) begin
                    for (int lane = 0; lane < 16; lane++) begin
                        if (data_mode == 1)
                            case (lane)
                                0: values[beat][row][lane]
                                    = (beat == 0 && row == 0) ? 1 : 0;
                                1: values[beat][row][lane]
                                    = (beat == 0 && row == 0) ? 3 : 0;
                                2: values[beat][row][lane]
                                    = (beat == 0 && row == 0) ? -1 : 0;
                                3: values[beat][row][lane]
                                    = (beat == 0 && row == 0) ? -3 : 0;
                                default: values[beat][row][lane] = 0;
                            endcase
                        else if (data_mode == 2)
                            case (lane)
                                0: values[beat][row][lane]
                                    = (beat == 0 && row == 0) ? 127 : 0;
                                1: values[beat][row][lane]
                                    = (beat == 0 && row == 0) ? -128 : 0;
                                default: values[beat][row][lane] = 0;
                            endcase
                        else if (data_mode == 3)
                            values[beat][row][lane]
                                = (beat == 0 && row == 0 && lane == 0)
                                    ? 127 : 0;
                        else if (data_mode == 4)
                            values[beat][row][lane] = -128;
                        else if (beat == 0 && row == 0 && lane == 0)
                            values[beat][row][lane] = -128;
                        else if (beat == 0 && row == 1 && lane == 15)
                            values[beat][row][lane] = 127;
                        else if (((channel_index + tile_index + beat
                                    + row + lane) % 9) == 0)
                            values[beat][row][lane] = 0;
                        else
                            values[beat][row][lane]
                                = $urandom_range(0, 255) - 128;
                        channel_sum_ref[lane]
                            += values[beat][row][lane];
                        channel_sumsq_ref[lane]
                            += values[beat][row][lane]
                             * values[beat][row][lane];
                    end
                end
            end
            channel_count_ref += 10;
            for (int rank = 0; rank < 3; rank++) begin
                for (int lane = 0; lane < 16; lane++) begin
                    projection = 0;
                    for (int beat = 0; beat < 5; beat++) begin
                        projection += values[beat][0][lane]
                            * factor_ref[rank][beat * 2];
                        projection += values[beat][1][lane]
                            * factor_ref[rank][beat * 2 + 1];
                    end
                    expected_rank.payload[(rank * 16 + lane) * 8 +: 8]
                        = requant_ref(projection, shift_ref);
                end
            end
            rank_expected_q.push_back(expected_rank);
            if (last_flag) begin
                expected_moment.tag = expected_rank.tag;
                expected_moment.count = channel_count_ref;
                for (int lane = 0; lane < 16; lane++) begin
                    expected_moment.sum_payload[lane * 26 +: 26]
                        = channel_sum_ref[lane];
                    expected_moment.sumsq_payload[lane * 32 +: 32]
                        = channel_sumsq_ref[lane];
                end
                moment_expected_q.push_back(expected_moment);
            end

            for (int beat = 0; beat < 5; beat++) begin
                tile_tag = expected_rank.tag;
                tile_beat = beat;
                tile_channel_start = beat == 0 ? start_flag : 1'b0;
                tile_channel_last = beat == 0 ? last_flag : 1'b0;
                for (int row = 0; row < 2; row++)
                    for (int lane = 0; lane < 16; lane++)
                        tile_data[row][lane] = values[beat][row][lane];
                if (data_mode != 0
                        || (channel_index == 0 && tile_index == 0))
                    gap_cycles = 0;
                else
                    gap_cycles = $urandom_range(0, 2);
                input_gap_cycles += gap_cycles;
                repeat (gap_cycles) @(negedge clk_core);
                tile_valid = 1'b1;
                do @(posedge clk_core); while (!tile_accept);
                @(negedge clk_core);
                tile_valid = 1'b0;
            end
        end
    endtask

    task automatic wait_for_drain;
        begin
            wait (rank_expected_q.size() == 0
                && moment_expected_q.size() == 0
                && !rank_valid && !moment_valid && !busy);
            @(negedge clk_core);
        end
    endtask

    initial begin : test
        clk_core = 1'b0;
        rst_core = 1'b1;
        config_valid = 1'b0;
        config_requant_shift = '0;
        tile_valid = 1'b0;
        tile_tag = '0;
        tile_beat = '0;
        tile_channel_start = 1'b0;
        tile_channel_last = 1'b0;
        rank_ready = 1'b0;
        moment_ready = 1'b0;
        force_rank_stall = 1'b0;
        force_moment_stall = 1'b0;
        cycle_count = 0;
        input_beats = 0;
        rank_results = 0;
        moment_results = 0;
        rank_stall_cycles = 0;
        moment_stall_cycles = 0;
        input_gap_cycles = 0;
        protocol_attacks = 0;
        rne_half_even_checks = 0;
        saturation_checks = 0;
        shift23_checks = 0;
        channel_count_ref = 0;
        for (int lane = 0; lane < 16; lane++) begin
            channel_sum_ref[lane] = 0;
            channel_sumsq_ref[lane] = 0;
        end
        for (int row = 0; row < 2; row++)
            for (int lane = 0; lane < 16; lane++)
                tile_data[row][lane] = '0;
        for (int rank = 0; rank < 3; rank++)
            for (int time_index = 0; time_index < 10; time_index++)
                config_factor[rank][time_index] = '0;

        reset_dut();
        load_config();
        if (!configured)
            $fatal(1, "M164 configuration was not retained");

        for (int channel_index = 0; channel_index < 20;
                channel_index++) begin
            int tiles_in_channel;
            tiles_in_channel = 1 + (channel_index % 5);
            channel_count_ref = 0;
            for (int lane = 0; lane < 16; lane++) begin
                channel_sum_ref[lane] = 0;
                channel_sumsq_ref[lane] = 0;
            end
            for (int tile_index = 0; tile_index < tiles_in_channel;
                    tile_index++) begin
                send_tile(channel_index, tile_index,
                    tile_index == 0,
                    tile_index == tiles_in_channel - 1, 0);
            end
        end
        wait_for_drain();

        // Explicitly cover the RNE and saturation boundaries that were absent
        // from M163r2: positive/negative half ties with even/odd quotients,
        // shift zero saturation, and the maximum legal shift of 23.
        channel_count_ref = 0;
        for (int lane = 0; lane < 16; lane++) begin
            channel_sum_ref[lane] = 0;
            channel_sumsq_ref[lane] = 0;
        end
        load_directed_config(1, 1);
        send_tile(20, 0, 1'b1, 1'b1, 1);
        wait_for_drain();

        channel_count_ref = 0;
        for (int lane = 0; lane < 16; lane++) begin
            channel_sum_ref[lane] = 0;
            channel_sumsq_ref[lane] = 0;
        end
        load_directed_config(0, 2);
        send_tile(21, 0, 1'b1, 1'b1, 2);
        wait_for_drain();

        channel_count_ref = 0;
        for (int lane = 0; lane < 16; lane++) begin
            channel_sum_ref[lane] = 0;
            channel_sumsq_ref[lane] = 0;
        end
        load_directed_config(23, 127);
        send_tile(22, 0, 1'b1, 1'b1, 3);
        wait_for_drain();

        // Exercise the frozen H67 stage-0 population bound exactly: 19,200
        // spatial tiles x T10 = 192,000 samples for every hidden lane.  This
        // reaches the signed-sum, unsigned-sumsq and count width contracts.
        channel_count_ref = 0;
        for (int lane = 0; lane < 16; lane++) begin
            channel_sum_ref[lane] = 0;
            channel_sumsq_ref[lane] = 0;
        end
        for (int tile_index = 0; tile_index < 19200; tile_index++)
            send_tile(23, tile_index, tile_index == 0,
                tile_index == 19199, 4);
        wait_for_drain();

        // One final channel is held at both outputs while a younger malformed
        // beat attacks the idle input boundary.  Accepted results must survive
        // the sticky fail-closed transition and drain afterwards.
        force_rank_stall = 1'b1;
        force_moment_stall = 1'b1;
        channel_count_ref = 0;
        for (int lane = 0; lane < 16; lane++) begin
            channel_sum_ref[lane] = 0;
            channel_sumsq_ref[lane] = 0;
        end
        send_tile(24, 0, 1'b1, 1'b1, 0);
        wait (rank_valid && moment_valid);
        @(negedge clk_core);
        tile_tag = 16'hbad0;
        tile_beat = 0;
        tile_channel_start = 1'b0;
        tile_channel_last = 1'b0;
        tile_valid = 1'b1;
        #0.1;
        if (!protocol_error || tile_ready)
            $fatal(1, "M164 malformed idle beat did not fail closed");
        @(posedge clk_core);
        @(negedge clk_core);
        tile_valid = 1'b0;
        protocol_attacks = protocol_attacks + 1;
        force_rank_stall = 1'b0;
        force_moment_stall = 1'b0;
        wait (rank_expected_q.size() == 0
            && moment_expected_q.size() == 0
            && !rank_valid && !moment_valid);
        repeat (3) @(negedge clk_core);

        if (!protocol_error || config_ready || tile_ready)
            $fatal(1, "M164 sticky fail-close was not retained");
        if (input_beats != 96320 || rank_results != 19264
                || moment_results != 25 || protocol_attacks != 1)
            $fatal(1,
                "M164 count mismatch beats=%0d rank=%0d moments=%0d attacks=%0d",
                input_beats, rank_results, moment_results,
                protocol_attacks);
        if (rank_stall_cycles == 0 || moment_stall_cycles == 0)
            $fatal(1, "M164 output stalls were not exercised");
        if (rne_half_even_checks != 12 || saturation_checks != 6
                || shift23_checks != 3)
            $fatal(1,
                "M164 directed RNE coverage mismatch ties=%0d sat=%0d shift23=%0d",
                rne_half_even_checks, saturation_checks, shift23_checks);

        $display("PASS M164 bounded-width per-hidden-lane dynamic-BN rank3 frontend VCS channels=25 tiles=19264 input_beats=96320 q8_samples=3082240 signed_products=9246720 squares=3082240 rank_results=19264 moment_results=25 moment_state_lanes=16 max_samples_per_lane=192000 max_population_exercised=true exact_max_negative_sum=-24576000 exact_max_sumsq=3145728000 sum_bits=26 sumsq_bits=32 count_bits=18 projection_bits=19 moment_samples_per_lane_total=192650 explicit_rne_half_even_checks=12 explicit_saturation_checks=6 explicit_shift23_checks=3 rank_stall_cycles=%0d moment_stall_cycles=%0d input_gap_cycles=%0d protocol_attacks=1 product_slots=96 square_issue_lanes=32 requant_lanes=16 input_tile_ii_accepted_cycles=5 coefficient_generation=false atlif=false left_projection=false fc2=false network_accuracy=false physical_speedup=false system_speedup=false headline=false",
            rank_stall_cycles, moment_stall_cycles, input_gap_cycles);
        $finish;
    end
endmodule

`default_nettype wire
