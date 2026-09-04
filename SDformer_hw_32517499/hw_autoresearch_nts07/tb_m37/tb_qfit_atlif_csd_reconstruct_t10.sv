`timescale 1ns/1ps
`default_nettype none

module tb_qfit_atlif_csd_reconstruct_t10;
    localparam int TAG_W = 48;
    localparam int T = 10;
    localparam int RANK = 3;
    localparam int LANES = 16;
    localparam int TERMS = 4;
    localparam int COEFFICIENTS = T*RANK;
    localparam int INTERMEDIATES = RANK*LANES;
    localparam int PRODUCTS_PER_TILE = T*LANES*RANK;
    localparam int NOMINAL_TILES = 96;
    localparam int CHARACTERIZATION_GROUPS = 9;
    localparam int CHARACTERIZATION_TILES_PER_GROUP = 16;
    localparam int CHARACTERIZATION_TILES
        = CHARACTERIZATION_GROUPS*CHARACTERIZATION_TILES_PER_GROUP;
    localparam int THRESHOLD_CONFIGS = 5;
    localparam int TOTAL_TILES
        = NOMINAL_TILES+CHARACTERIZATION_TILES+THRESHOLD_CONFIGS;
    localparam int FULL_RATE_TILES = 72;
    localparam int LONG_STALL_CYCLES = 140;
    localparam int ILLEGAL_CLASSES = 7;
    localparam int ILLEGAL_TESTS = ILLEGAL_CLASSES*COEFFICIENTS;
    localparam logic [31:0] RANDOM_SEED = 32'h4d370203;
    localparam logic [TAG_W-1:0] TAG_BASE = 48'h4d3700000000;

    logic clk_core = 1'b0;
    logic rst_core = 1'b1;
    logic config_valid = 1'b0;
    logic config_ready;
    logic [(COEFFICIENTS*8)-1:0] config_left_factor = '0;
    logic [(COEFFICIENTS*TERMS)-1:0] config_term_valid = '0;
    logic [(COEFFICIENTS*TERMS)-1:0] config_term_negative = '0;
    logic [(COEFFICIENTS*TERMS*3)-1:0] config_term_shift = '0;
    logic [(T*24)-1:0] config_bias = '0;
    logic signed [23:0] config_threshold = '0;
    logic descriptor_legal;
    logic config_loaded;
    logic config_release_valid = 1'b0;
    logic config_release_ready;
    logic input_valid = 1'b0;
    logic input_ready;
    logic [TAG_W-1:0] input_tag = '0;
    logic [(INTERMEDIATES*8)-1:0] input_intermediate = '0;
    logic result_valid;
    logic result_ready = 1'b0;
    logic [TAG_W-1:0] result_tag;
    logic [2:0] result_beat;
    logic [47:0] result_valid_bits;
    logic [47:0] result_bits;
    logic done;
    logic [TAG_W-1:0] done_tag;
    logic protocol_error;
    logic busy;
    logic arithmetic_active;
    logic [2:0] phase_cycle;
    logic phase4_chain_accept;
    logic [1:0] input_buffer_occupancy;
    logic [4:0] result_fifo_occupancy;
    logic result_fifo_push;
    logic result_fifo_pop;
    logic [2:0] result_fifo_push_beat;
    logic [TAG_W-1:0] result_fifo_push_tag;
    logic input_accept;
    logic input_accept_bank;
    logic active_compute_bank;
    logic [TAG_W-1:0] arithmetic_tag;
    logic uses_integer_multiplier;

    integer signed coefficient [0:COEFFICIENTS-1];
    integer signed bias [0:T-1];
    integer signed threshold;
    integer signed tile_coefficient [0:TOTAL_TILES-1][0:COEFFICIENTS-1];
    integer signed vectors [0:TOTAL_TILES-1][0:INTERMEDIATES-1];
    integer signed expected_products
        [0:TOTAL_TILES-1][0:PRODUCTS_PER_TILE-1];
    bit expected_bits [0:TOTAL_TILES-1][0:(T*LANES)-1];
    bit [65535:0] dut_pair_seen = '0;
    bit [255:0] nominal_input_value_seen = '0;
    bit [2:0] legal_special_seen = '0;

    integer threshold_value [0:THRESHOLD_CONFIGS-1];
    integer threshold_equal_count [0:THRESHOLD_CONFIGS-1];
    integer threshold_just_below_raw_count [0:THRESHOLD_CONFIGS-1];
    integer threshold_positive_saturation_count [0:THRESHOLD_CONFIGS-1];
    integer threshold_negative_saturation_count [0:THRESHOLD_CONFIGS-1];
    integer illegal_class_count [0:ILLEGAL_CLASSES-1];

    integer cycle_count = 0;
    integer accepted_tiles = 0;
    integer nominal_accepted_tiles = 0;
    integer received_beats = 0;
    integer done_tiles = 0;
    integer previous_nominal_accept_cycle = -1;
    integer ii5_matches = 0;
    integer chain_accepts = 0;
    integer arithmetic_issues = 0;
    integer product_miter_checks = 0;
    integer output_bit_checks = 0;
    integer dut_unique_signed_input_coefficient_product_pairs = 0;
    integer unique_tile_payloads = 0;
    integer unique_expected_product_fingerprints = 0;
    integer unique_expected_bitmaps = 0;
    integer consecutive_identical_tile_payloads = 0;
    integer nominal_unique_signed_inputs = 0;
    integer maximum_fifo_occupancy = 0;
    integer fifo_full_cycles = 0;
    integer full_pop_push_cycles = 0;
    integer output_stall_cycles = 0;
    integer input_stall_cycles = 0;
    integer live_pin_perturbations = 0;
    integer config_loads = 0;
    integer config_releases = 0;
    integer release_reload_successes = 0;
    integer release_reject_cycles = 0;
    integer release_busy_rejects = 0;
    integer release_fifo_nonempty_rejects = 0;
    integer release_input_valid_rejects = 0;
    integer illegal_accepts = 0;
    integer illegal_rejections = 0;
    integer generic_positive_saturations = 0;
    integer generic_negative_saturations = 0;
    integer output_ones = 0;
    integer output_zeros = 0;
    integer done_with_fifo_pending = 0;
    logic [31:0] xorshift_state;
    integer vector_fd;
    bit force_long_stall = 1'b0;
    bit sparse_ready_enable = 1'b0;
    string vector_file;

    always #5 clk_core = ~clk_core;
    qfit_atlif_csd_reconstruct_t10 dut (.*);

    function automatic logic [31:0] xorshift32(input logic [31:0] state);
        logic [31:0] next;
        begin
            next = state;
            next = next ^ (next << 13);
            next = next ^ (next >> 17);
            next = next ^ (next << 5);
            xorshift32 = next;
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

    task automatic clear_config_buses;
        config_left_factor = '0;
        config_term_valid = '0;
        config_term_negative = '0;
        config_term_shift = '0;
        config_bias = '0;
        config_threshold = '0;
    endtask

    task automatic encode_naf(input int coefficient_index,
                              input integer signed value);
        integer sign_value;
        integer remaining;
        integer digit;
        integer shift;
        integer term;
        integer signed signed_digit;
        begin
            config_left_factor[(coefficient_index*8) +: 8] = value[7:0];
            sign_value = value < 0 ? -1 : 1;
            remaining = value < 0 ? -value : value;
            shift = 0;
            term = 0;
            while (remaining != 0) begin
                if (remaining & 1) begin
                    digit = 2 - (remaining & 3);
                    signed_digit = sign_value*digit;
                    if (term >= TERMS)
                        $fatal(1, "M37 TB NAF term overflow value=%0d", value);
                    config_term_valid[(coefficient_index*TERMS)+term] = 1'b1;
                    config_term_negative[(coefficient_index*TERMS)+term]
                        = signed_digit < 0;
                    config_term_shift[
                        (((coefficient_index*TERMS)+term)*3) +: 3]
                        = shift[2:0];
                    remaining = remaining - digit;
                    term = term + 1;
                end
                remaining = remaining >>> 1;
                shift = shift + 1;
            end
        end
    endtask

    task automatic materialize_config;
        clear_config_buses();
        for (int index = 0; index < COEFFICIENTS; index++)
            encode_naf(index, coefficient[index]);
        for (int row = 0; row < T; row++)
            config_bias[(row*24) +: 24] = bias[row][23:0];
        config_threshold = threshold[23:0];
        #1;
        if (!descriptor_legal)
            $fatal(1, "M37 legal configuration rejected");
    endtask

    task automatic build_nominal_config;
        threshold = 0;
        for (int index = 0; index < COEFFICIENTS; index++)
            coefficient[index] = ((index*73 + 19) % 256) - 128;
        for (int row = 0; row < T; row++)
            bias[row] = row*53 - 230;
        // Row zero is a real-DUT sign signature: seven lanes encode the tile
        // identity through H[rank0,lane] >= 0.  The remaining nine rows keep
        // the nondegenerate xorshift arithmetic workload.
        coefficient[0] = 1;
        coefficient[1] = 0;
        coefficient[2] = 0;
        bias[0] = 0;
        materialize_config();
    endtask

    task automatic build_characterization_config(input int group);
        integer encoded_value;
        threshold = 0;
        for (int index = 0; index < COEFFICIENTS; index++) begin
            encoded_value = group*COEFFICIENTS + index;
            if (encoded_value > 255)
                encoded_value = 255;
            coefficient[index] = encoded_value - 128;
        end
        for (int row = 0; row < T; row++)
            bias[row] = 0;
        materialize_config();
    endtask

    task automatic build_threshold_config(input int threshold_index);
        threshold = threshold_value[threshold_index];
        for (int index = 0; index < COEFFICIENTS; index++)
            coefficient[index] = 0;
        for (int row = 0; row < T; row++)
            bias[row] = 0;
        bias[0] = threshold;
        if (threshold == -8388608) begin
            bias[1] = -8388608;
            coefficient[4] = 1;
        end else begin
            bias[1] = threshold-1;
        end
        bias[2] = 8388607;
        coefficient[6] = -128;
        bias[3] = -8388608;
        coefficient[11] = 127;
        materialize_config();
    endtask

    task automatic write_current_config(input string label);
        $fwrite(vector_fd, "CONFIG %s threshold=%0d bias", label, threshold);
        for (int row = 0; row < T; row++)
            $fwrite(vector_fd, " %0d", bias[row]);
        $fwrite(vector_fd, " coeff");
        for (int index = 0; index < COEFFICIENTS; index++)
            $fwrite(vector_fd, " %0d", coefficient[index]);
        $fwrite(vector_fd, "\n");
    endtask

    task automatic prepare_reference(input int tile,
                                     input int threshold_index);
        longint signed sum;
        integer signed saturated;
        integer product_index;
        for (int index = 0; index < COEFFICIENTS; index++)
            tile_coefficient[tile][index] = coefficient[index];
        $fwrite(vector_fd, "TILE %03d", tile);
        for (int index = 0; index < INTERMEDIATES; index++)
            $fwrite(vector_fd, " %0d", vectors[tile][index]);
        $fwrite(vector_fd, "\n");
        for (int row = 0; row < T; row++) begin
            for (int lane = 0; lane < LANES; lane++) begin
                sum = bias[row];
                for (int rank_index = 0; rank_index < RANK;
                     rank_index++) begin
                    product_index = ((row*LANES)+lane)*RANK+rank_index;
                    expected_products[tile][product_index]
                        = vectors[tile][(rank_index*LANES)+lane]
                            * coefficient[(row*RANK)+rank_index];
                    sum += expected_products[tile][product_index];
                end
                if (sum > 8388607)
                    generic_positive_saturations += 1;
                if (sum < -8388608)
                    generic_negative_saturations += 1;
                saturated = clamp_q24(sum);
                expected_bits[tile][(row*LANES)+lane]
                    = saturated >= threshold;
                if (saturated >= threshold)
                    output_ones += 1;
                else
                    output_zeros += 1;
                if (threshold_index >= 0) begin
                    if (saturated == threshold)
                        threshold_equal_count[threshold_index] += 1;
                    if (sum == $signed(threshold)-1)
                        threshold_just_below_raw_count[threshold_index] += 1;
                    if (sum > 8388607)
                        threshold_positive_saturation_count[
                            threshold_index] += 1;
                    if (sum < -8388608)
                        threshold_negative_saturation_count[
                            threshold_index] += 1;
                end
            end
        end
    endtask

    task automatic build_nominal_vectors_and_audit;
        bit same_payload;
        bit same_arithmetic;
        bit same_bitmap;
        bit duplicate_payload;
        bit duplicate_arithmetic;
        bit duplicate_bitmap;
        xorshift_state = RANDOM_SEED;
        for (int tile = 0; tile < NOMINAL_TILES; tile++) begin
            for (int index = 0; index < INTERMEDIATES; index++) begin
                xorshift_state = xorshift32(xorshift_state);
                vectors[tile][index] = $signed(xorshift_state[7:0]);
                if (tile < 16)
                    vectors[tile][index]
                        = (tile*16 + (index%LANES)) - 128;
                if (index < 7)
                    vectors[tile][index]
                        = ((tile >> index) & 1) ? 1 : -1;
                nominal_input_value_seen[vectors[tile][index]+128] = 1'b1;
            end
            prepare_reference(tile, -1);
        end
        for (int tile = 0; tile < NOMINAL_TILES; tile++) begin
            duplicate_payload = 1'b0;
            duplicate_arithmetic = 1'b0;
            duplicate_bitmap = 1'b0;
            for (int previous = 0; previous < tile; previous++) begin
                same_payload = 1'b1;
                for (int index = 0; index < INTERMEDIATES; index++)
                    if (vectors[tile][index] != vectors[previous][index])
                        same_payload = 1'b0;
                same_arithmetic = 1'b1;
                for (int index = 0; index < PRODUCTS_PER_TILE; index++)
                    if (expected_products[tile][index]
                            != expected_products[previous][index])
                        same_arithmetic = 1'b0;
                same_bitmap = 1'b1;
                for (int index = 0; index < T*LANES; index++)
                    if (expected_bits[tile][index]
                            != expected_bits[previous][index])
                        same_bitmap = 1'b0;
                duplicate_payload |= same_payload;
                duplicate_arithmetic |= same_arithmetic;
                duplicate_bitmap |= same_bitmap;
                if (previous == tile-1 && same_payload)
                    consecutive_identical_tile_payloads += 1;
            end
            if (!duplicate_payload)
                unique_tile_payloads += 1;
            if (!duplicate_arithmetic)
                unique_expected_product_fingerprints += 1;
            if (!duplicate_bitmap)
                unique_expected_bitmaps += 1;
        end
        nominal_unique_signed_inputs = $countones(nominal_input_value_seen);
        if (unique_tile_payloads != NOMINAL_TILES
                || unique_expected_product_fingerprints != NOMINAL_TILES
                || unique_expected_bitmaps != NOMINAL_TILES
                || consecutive_identical_tile_payloads != 0
                || nominal_unique_signed_inputs != 256)
            $fatal(1, "M37 nominal uniqueness failed payload=%0d arithmetic=%0d bitmap=%0d consecutive=%0d values=%0d",
                   unique_tile_payloads,
                   unique_expected_product_fingerprints,
                   unique_expected_bitmaps,
                   consecutive_identical_tile_payloads,
                   nominal_unique_signed_inputs);
    endtask

    task automatic prepare_characterization_tiles(input int group,
                                                   input int first_tile);
        for (int offset = 0; offset < CHARACTERIZATION_TILES_PER_GROUP;
             offset++) begin
            for (int rank_index = 0; rank_index < RANK; rank_index++)
                for (int lane = 0; lane < LANES; lane++)
                    vectors[first_tile+offset][(rank_index*LANES)+lane]
                        = -128 + offset*16 + lane;
            prepare_reference(first_tile+offset, -1);
        end
    endtask

    task automatic prepare_threshold_tile(input int threshold_index,
                                          input int tile);
        for (int index = 0; index < INTERMEDIATES; index++)
            vectors[tile][index] = 0;
        for (int lane = 0; lane < LANES; lane++) begin
            vectors[tile][lane] = -128;
            vectors[tile][LANES+lane] = -1;
            vectors[tile][(2*LANES)+lane] = -128;
        end
        prepare_reference(tile, threshold_index);
    endtask

    task automatic pulse_reset;
        @(negedge clk_core);
        rst_core = 1'b1;
        input_valid = 1'b0;
        config_valid = 1'b0;
        config_release_valid = 1'b0;
        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
    endtask

    task automatic load_config;
        @(negedge clk_core);
        config_valid = 1'b1;
        do @(posedge clk_core); while (!config_ready);
        config_loads += 1;
        if (config_releases > 0)
            release_reload_successes += 1;
        for (int index = 0; index < COEFFICIENTS; index++) begin
            if (coefficient[index] == 0)
                legal_special_seen[0] = 1'b1;
            if (coefficient[index] == -128)
                legal_special_seen[1] = 1'b1;
            if (coefficient[index] == 127)
                legal_special_seen[2] = 1'b1;
        end
        @(negedge clk_core);
        config_valid = 1'b0;
        if (!config_loaded || protocol_error)
            $fatal(1, "M37 legal configuration load failed");
    endtask

    task automatic release_config;
        @(negedge clk_core);
        config_release_valid = 1'b1;
        do @(posedge clk_core); while (!config_release_ready);
        config_releases += 1;
        @(negedge clk_core);
        config_release_valid = 1'b0;
        if (config_loaded)
            $fatal(1, "M37 configuration release failed");
    endtask

    task automatic perturb_live_config(input int tile);
        config_left_factor[7:0] = tile[7:0];
        config_term_valid[3:0] = 4'b0101;
        config_term_negative[3:0] = tile[3:0];
        config_term_shift[11:0] = 12'hfff;
        config_bias[23:0] = 24'h5a0000 + tile;
        config_threshold = -24'sd765432;
        live_pin_perturbations += 1;
    endtask

    task automatic send_tile_range(input int first_tile,
                                   input int count,
                                   input bit perturb_config);
        @(negedge clk_core);
        input_valid = 1'b1;
        for (int tile = first_tile; tile < first_tile+count; tile++) begin
            input_tag = TAG_BASE + tile;
            for (int index = 0; index < INTERMEDIATES; index++)
                input_intermediate[(index*8) +: 8]
                    = vectors[tile][index][7:0];
            if (perturb_config)
                perturb_live_config(tile);
            do @(posedge clk_core); while (!input_ready);
            @(negedge clk_core);
        end
        input_valid = 1'b0;
    endtask

    task automatic long_stall_controller;
        wait (nominal_accepted_tiles >= FULL_RATE_TILES);
        @(negedge clk_core);
        force_long_stall = 1'b1;
        repeat (LONG_STALL_CYCLES) @(negedge clk_core);
        force_long_stall = 1'b0;
        sparse_ready_enable = 1'b1;
    endtask

    task automatic release_rejection_probe;
        wait (nominal_accepted_tiles >= 8);
        @(negedge clk_core);
        config_release_valid = 1'b1;
        // A refused release is a decoupled handshake request, not a pulse:
        // retain valid throughout busy/input/FIFO rejection and consume the
        // request only when the completely drained configuration can retire.
        do @(posedge clk_core); while (!config_release_ready);
        config_releases += 1;
        @(negedge clk_core);
        config_release_valid = 1'b0;
        if (config_loaded)
            $fatal(1, "M37 held release handshake failed");
    endtask

    task automatic build_zero_config;
        threshold = 0;
        for (int index = 0; index < COEFFICIENTS; index++)
            coefficient[index] = 0;
        for (int row = 0; row < T; row++)
            bias[row] = 0;
        materialize_config();
    endtask

    task automatic corrupt_descriptor(input int illegal_class,
                                      input int coefficient_index);
        integer base_term;
        begin
            base_term = coefficient_index*TERMS;
            case (illegal_class)
                0: begin
                    config_left_factor[(coefficient_index*8) +: 8] = 8'sd5;
                    config_term_valid[base_term +: 4] = 4'b0101;
                    config_term_shift[((base_term+0)*3) +: 3] = 3'd0;
                    config_term_shift[((base_term+2)*3) +: 3] = 3'd2;
                end
                1: config_term_negative[base_term] = 1'b1;
                2: config_term_shift[(base_term*3) +: 3] = 3'd1;
                3: begin
                    config_left_factor[(coefficient_index*8) +: 8] = 8'sd2;
                    config_term_valid[base_term +: 4] = 4'b0011;
                    config_term_shift[((base_term+0)*3) +: 3] = 3'd0;
                    config_term_shift[((base_term+1)*3) +: 3] = 3'd0;
                end
                4: begin
                    config_left_factor[(coefficient_index*8) +: 8] = 8'sd5;
                    config_term_valid[base_term +: 4] = 4'b0011;
                    config_term_shift[((base_term+0)*3) +: 3] = 3'd2;
                    config_term_shift[((base_term+1)*3) +: 3] = 3'd0;
                end
                5: begin
                    config_left_factor[(coefficient_index*8) +: 8] = 8'sd3;
                    config_term_valid[base_term +: 4] = 4'b0011;
                    config_term_shift[((base_term+0)*3) +: 3] = 3'd0;
                    config_term_shift[((base_term+1)*3) +: 3] = 3'd1;
                end
                default: begin
                    config_left_factor[(coefficient_index*8) +: 8] = 8'sd0;
                    config_term_valid[base_term +: 4] = 4'b0001;
                    config_term_shift[(base_term*3) +: 3] = 3'd2;
                end
            endcase
        end
    endtask

    task automatic run_illegal_matrix;
        for (int illegal_class = 0; illegal_class < ILLEGAL_CLASSES;
             illegal_class++) begin
            for (int coefficient_index = 0;
                 coefficient_index < COEFFICIENTS; coefficient_index++) begin
                pulse_reset();
                build_zero_config();
                corrupt_descriptor(illegal_class, coefficient_index);
                #1;
                if (descriptor_legal)
                    $fatal(1, "M37 illegal descriptor marked legal class=%0d coefficient=%0d",
                           illegal_class, coefficient_index);
                $fdisplay(vector_fd, "ILLEGAL class=%0d coefficient=%0d",
                          illegal_class, coefficient_index);
                @(negedge clk_core);
                config_valid = 1'b1;
                do @(posedge clk_core); while (!config_ready);
                illegal_accepts += 1;
                illegal_class_count[illegal_class] += 1;
                @(negedge clk_core);
                config_valid = 1'b0;
                @(posedge clk_core);
                if (!protocol_error || config_loaded)
                    $fatal(1, "M37 illegal descriptor failed to close class=%0d coefficient=%0d",
                           illegal_class, coefficient_index);
                illegal_rejections += 1;
            end
        end
    endtask

    always @(negedge clk_core) begin
        if (rst_core)
            result_ready <= 1'b0;
        else if (force_long_stall)
            result_ready <= 1'b0;
        else if (sparse_ready_enable)
            result_ready <= (cycle_count % 19) != 7;
        else
            result_ready <= 1'b1;
    end

    always @(posedge clk_core) begin
        if (!rst_core) begin
            cycle_count += 1;
            if (input_valid && !input_ready)
                input_stall_cycles += 1;
            if (input_valid && input_ready) begin
                int accepted_tile;
                accepted_tile = input_tag - TAG_BASE;
                if (accepted_tile < NOMINAL_TILES) begin
                    if (nominal_accepted_tiles >= 3
                            && nominal_accepted_tiles < FULL_RATE_TILES) begin
                        if (cycle_count-previous_nominal_accept_cycle != 5)
                            $fatal(1, "M37 nominal accept II drift tile=%0d got=%0d",
                                   accepted_tile,
                                   cycle_count-previous_nominal_accept_cycle);
                        ii5_matches += 1;
                    end
                    previous_nominal_accept_cycle = cycle_count;
                    nominal_accepted_tiles += 1;
                end
                accepted_tiles += 1;
            end
            if (phase4_chain_accept)
                chain_accepts += 1;
            if (arithmetic_active)
                arithmetic_issues += 1;
            if (result_fifo_occupancy > maximum_fifo_occupancy)
                maximum_fifo_occupancy = result_fifo_occupancy;
            if (result_fifo_occupancy == 16)
                fifo_full_cycles += 1;
            if (result_fifo_occupancy == 16 && result_fifo_push
                    && result_fifo_pop)
                full_pop_push_cycles += 1;
            if (result_valid && !result_ready)
                output_stall_cycles += 1;
            if (config_release_valid && !config_release_ready) begin
                release_reject_cycles += 1;
                if (busy)
                    release_busy_rejects += 1;
                if (result_fifo_occupancy != 0)
                    release_fifo_nonempty_rejects += 1;
                if (input_valid)
                    release_input_valid_rejects += 1;
            end
            if (done) begin
                if (done_tag != TAG_BASE + done_tiles)
                    $fatal(1, "M37 done ownership mismatch got=%h index=%0d",
                           done_tag, done_tiles);
                if (result_fifo_occupancy != 0)
                    done_with_fifo_pending += 1;
                done_tiles += 1;
            end
            if (result_valid && result_ready) begin
                int tile;
                int beat;
                tile = received_beats / 5;
                beat = received_beats % 5;
                if (result_tag != TAG_BASE + tile || result_beat != beat
                        || result_valid_bits
                            != {{16{1'b0}}, {32{1'b1}}}
                        || result_bits[47:32] != 0)
                    $fatal(1, "M37 output identity mismatch tile=%0d beat=%0d",
                           tile, beat);
                for (int index = 0; index < 32; index++) begin
                    if (result_bits[index]
                            !== expected_bits[tile][(beat*32)+index])
                        $fatal(1, "M37 bit miter mismatch tile=%0d beat=%0d index=%0d got=%0d expected=%0d",
                               tile, beat, index, result_bits[index],
                               expected_bits[tile][(beat*32)+index]);
                    output_bit_checks += 1;
                end
                received_beats += 1;
            end
        end
    end

    // This is the only 256x256 coverage ledger: every entry is set from a
    // product committed by the DUT, never from a standalone reference loop.
    always @(posedge clk_core) begin
        if (!rst_core && result_fifo_push) begin
            int tile;
            int row;
            int lane;
            int coefficient_index;
            int intermediate_index;
            int product_index;
            int pair_index;
            integer signed direct_product;
            tile = result_fifo_push_tag - TAG_BASE;
            for (int output_index = 0; output_index < 32; output_index++) begin
                row = (result_fifo_push_beat*2) + (output_index/LANES);
                lane = output_index % LANES;
                for (int rank_index = 0; rank_index < RANK; rank_index++) begin
                    coefficient_index = (row*RANK)+rank_index;
                    intermediate_index = (rank_index*LANES)+lane;
                    product_index = ((row*LANES)+lane)*RANK+rank_index;
                    direct_product = expected_products[tile][product_index];
                    if ($signed(dut.product_q[
                            (output_index*RANK)+rank_index])
                            !== direct_product)
                        $fatal(1, "M37 DUT product miter mismatch tile=%0d beat=%0d output=%0d rank=%0d got=%0d expected=%0d",
                               tile, result_fifo_push_beat, output_index,
                               rank_index, $signed(dut.product_q[
                                   (output_index*RANK)+rank_index]),
                               direct_product);
                    pair_index = (tile_coefficient[tile][coefficient_index]+128)
                        * 256 + vectors[tile][intermediate_index] + 128;
                    dut_pair_seen[pair_index] = 1'b1;
                    product_miter_checks += 1;
                end
            end
        end
    end

    initial begin
`ifdef SIMULATOR_VCS
        $display("SIMULATOR=Synopsys VCS");
`else
        $fatal(1, "M37 evidence requires Synopsys VCS");
`endif
`ifdef SVA_RUNTIME_ENABLED
        $display("ASSERTIONS=enabled");
`else
        $fatal(1, "M37 evidence requires enabled SVA");
`endif
        $display("M37_RANDOM_SEED=0x%08x", RANDOM_SEED);
        if (!$value$plusargs("M37_VECTOR_FILE=%s", vector_file))
            $fatal(1, "M37 vector file plusarg required");
        vector_fd = $fopen(vector_file, "w");
        if (vector_fd == 0)
            $fatal(1, "M37 cannot open vector file");
        $fdisplay(vector_fd, "seed=%08x total_tiles=%0d", RANDOM_SEED,
                  TOTAL_TILES);
        threshold_value[0] = -8388608;
        threshold_value[1] = -12345;
        threshold_value[2] = 0;
        threshold_value[3] = 12345;
        threshold_value[4] = 8388607;
        for (int threshold_index = 0; threshold_index < THRESHOLD_CONFIGS;
             threshold_index++) begin
            threshold_equal_count[threshold_index] = 0;
            threshold_just_below_raw_count[threshold_index] = 0;
            threshold_positive_saturation_count[threshold_index] = 0;
            threshold_negative_saturation_count[threshold_index] = 0;
        end
        for (int illegal_class = 0; illegal_class < ILLEGAL_CLASSES;
             illegal_class++)
            illegal_class_count[illegal_class] = 0;

        build_nominal_config();
        write_current_config("nominal_unique");
        build_nominal_vectors_and_audit();
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;
        load_config();
        fork
            send_tile_range(0, NOMINAL_TILES, 1'b1);
            long_stall_controller();
            release_rejection_probe();
        join
        while (received_beats < NOMINAL_TILES*5
                || done_tiles < NOMINAL_TILES || busy)
            @(posedge clk_core);
        sparse_ready_enable = 1'b0;

        for (int group = 0; group < CHARACTERIZATION_GROUPS; group++) begin
            int first_tile;
            first_tile = NOMINAL_TILES
                + group*CHARACTERIZATION_TILES_PER_GROUP;
            build_characterization_config(group);
            write_current_config($sformatf("dut_full_domain_group_%0d", group));
            prepare_characterization_tiles(group, first_tile);
            load_config();
            send_tile_range(first_tile,
                CHARACTERIZATION_TILES_PER_GROUP, 1'b0);
            while (received_beats < (first_tile
                    + CHARACTERIZATION_TILES_PER_GROUP)*5
                    || done_tiles < first_tile
                        + CHARACTERIZATION_TILES_PER_GROUP || busy)
                @(posedge clk_core);
            release_config();
        end

        for (int threshold_index = 0; threshold_index < THRESHOLD_CONFIGS;
             threshold_index++) begin
            int tile;
            tile = NOMINAL_TILES + CHARACTERIZATION_TILES + threshold_index;
            build_threshold_config(threshold_index);
            write_current_config($sformatf("threshold_%0d", threshold_index));
            prepare_threshold_tile(threshold_index, tile);
            load_config();
            send_tile_range(tile, 1, 1'b0);
            while (received_beats < (tile+1)*5 || done_tiles < tile+1 || busy)
                @(posedge clk_core);
            release_config();
        end

        dut_unique_signed_input_coefficient_product_pairs
            = $countones(dut_pair_seen);
        run_illegal_matrix();
        $fclose(vector_fd);
        repeat (3) @(posedge clk_core);

        if (accepted_tiles != TOTAL_TILES
                || received_beats != TOTAL_TILES*5
                || done_tiles != TOTAL_TILES
                || arithmetic_issues != TOTAL_TILES*5)
            $fatal(1, "M37 accounting mismatch accepted=%0d receive=%0d done=%0d issue=%0d",
                   accepted_tiles, received_beats, done_tiles,
                   arithmetic_issues);
        if (product_miter_checks != TOTAL_TILES*PRODUCTS_PER_TILE
                || output_bit_checks != TOTAL_TILES*T*LANES
                || dut_unique_signed_input_coefficient_product_pairs != 65536)
            $fatal(1, "M37 DUT coverage mismatch products=%0d bits=%0d unique_pairs=%0d",
                   product_miter_checks, output_bit_checks,
                   dut_unique_signed_input_coefficient_product_pairs);
        if (unique_tile_payloads != 96
                || unique_expected_product_fingerprints != 96
                || unique_expected_bitmaps != 96
                || consecutive_identical_tile_payloads != 0
                || nominal_unique_signed_inputs != 256)
            $fatal(1, "M37 frozen uniqueness drift payload=%0d arithmetic=%0d bitmap=%0d consecutive=%0d inputs=%0d",
                   unique_tile_payloads,
                   unique_expected_product_fingerprints,
                   unique_expected_bitmaps,
                   consecutive_identical_tile_payloads,
                   nominal_unique_signed_inputs);
        if (ii5_matches < 64 || chain_accepts < 64
                || maximum_fifo_occupancy != 16 || fifo_full_cycles == 0
                || full_pop_push_cycles == 0
                || output_stall_cycles < LONG_STALL_CYCLES
                || input_stall_cycles == 0)
            $fatal(1, "M37 flow coverage incomplete ii=%0d chain=%0d max=%0d full=%0d poppush=%0d stalls=%0d/%0d",
                   ii5_matches, chain_accepts, maximum_fifo_occupancy,
                   fifo_full_cycles, full_pop_push_cycles,
                   input_stall_cycles, output_stall_cycles);
        if (config_loads != 15 || config_releases != 15
                || release_reload_successes != 14
                || release_reject_cycles == 0 || release_busy_rejects == 0
                || release_fifo_nonempty_rejects == 0
                || release_input_valid_rejects == 0
                || live_pin_perturbations != NOMINAL_TILES
                || legal_special_seen != 3'b111)
            $fatal(1, "M37 config/release coverage incomplete load=%0d release=%0d reload=%0d reject=%0d busy=%0d fifo=%0d input=%0d perturb=%0d special=%b",
                   config_loads, config_releases, release_reload_successes,
                   release_reject_cycles, release_busy_rejects,
                   release_fifo_nonempty_rejects,
                   release_input_valid_rejects, live_pin_perturbations,
                   legal_special_seen);
        if (illegal_accepts != ILLEGAL_TESTS
                || illegal_rejections != ILLEGAL_TESTS)
            $fatal(1, "M37 illegal matrix count mismatch %0d/%0d",
                   illegal_accepts, illegal_rejections);
        for (int illegal_class = 0; illegal_class < ILLEGAL_CLASSES;
             illegal_class++)
            if (illegal_class_count[illegal_class] != COEFFICIENTS)
                $fatal(1, "M37 illegal class coverage mismatch class=%0d count=%0d",
                       illegal_class, illegal_class_count[illegal_class]);
        for (int threshold_index = 0; threshold_index < THRESHOLD_CONFIGS;
             threshold_index++) begin
            if (threshold_equal_count[threshold_index] < LANES
                    || threshold_just_below_raw_count[threshold_index] < LANES
                    || threshold_positive_saturation_count[threshold_index]
                        < LANES
                    || threshold_negative_saturation_count[threshold_index]
                        < LANES)
                $fatal(1, "M37 threshold coverage incomplete index=%0d threshold=%0d eq=%0d below=%0d sat=%0d/%0d",
                       threshold_index, threshold_value[threshold_index],
                       threshold_equal_count[threshold_index],
                       threshold_just_below_raw_count[threshold_index],
                       threshold_positive_saturation_count[threshold_index],
                       threshold_negative_saturation_count[threshold_index]);
        end
        if (generic_positive_saturations == 0
                || generic_negative_saturations == 0
                || output_ones == 0 || output_zeros == 0
                || done_with_fifo_pending == 0 || uses_integer_multiplier)
            $fatal(1, "M37 arithmetic diversity incomplete sat=%0d/%0d bits=%0d/%0d donepending=%0d multiplier=%0d",
                   generic_positive_saturations,
                   generic_negative_saturations, output_ones,
                   output_zeros, done_with_fifo_pending,
                   uses_integer_multiplier);

        $display("M37_PASS total_tiles=%0d nominal_tiles=96 dut_unique_signed_input_coefficient_product_pairs=%0d product_miters=%0d bit_miters=%0d arithmetic_issues=%0d no_data_multiplier=1",
                 TOTAL_TILES, dut_unique_signed_input_coefficient_product_pairs,
                 product_miter_checks, output_bit_checks,
                 arithmetic_issues);
        $display("M37_UNIQUENESS unique_tile_payloads=%0d unique_expected_product_fingerprints=%0d unique_expected_bitmaps=%0d consecutive_identical=%0d nominal_unique_signed_inputs=%0d",
                 unique_tile_payloads,
                 unique_expected_product_fingerprints,
                 unique_expected_bitmaps,
                 consecutive_identical_tile_payloads,
                 nominal_unique_signed_inputs);
        $display("M37_FLOW conditional_standalone_accept_ii5_matches=%0d phase4_chain_accepts=%0d max_fifo=%0d fifo_full_cycles=%0d full_pop_push=%0d stalls=%0d/%0d done_with_fifo_pending=%0d",
                 ii5_matches, chain_accepts,
                 maximum_fifo_occupancy, fifo_full_cycles,
                 full_pop_push_cycles, input_stall_cycles,
                 output_stall_cycles, done_with_fifo_pending);
        $display("M37_CONFIG config_load_release_reload=%0d/%0d/%0d release_reject_busy_fifo_input=%0d/%0d/%0d/%0d live_pin_perturbations=%0d legal_zero_min_max=1",
                 config_loads, config_releases,
                 release_reload_successes, release_reject_cycles,
                 release_busy_rejects, release_fifo_nonempty_rejects,
                 release_input_valid_rejects, live_pin_perturbations);
        $display("M37_ILLEGAL illegal_matrix=%0d/%0d illegal_classes=%0d,%0d,%0d,%0d,%0d,%0d,%0d",
                 illegal_accepts, illegal_rejections,
                 illegal_class_count[0], illegal_class_count[1],
                 illegal_class_count[2], illegal_class_count[3],
                 illegal_class_count[4], illegal_class_count[5],
                 illegal_class_count[6]);
        for (int threshold_index = 0; threshold_index < THRESHOLD_CONFIGS;
             threshold_index++)
            $display("M37_THRESHOLD index=%0d value=%0d equal=%0d just_below_raw=%0d positive_saturation=%0d negative_saturation=%0d",
                 threshold_index, threshold_value[threshold_index],
                 threshold_equal_count[threshold_index],
                 threshold_just_below_raw_count[threshold_index],
                 threshold_positive_saturation_count[threshold_index],
                 threshold_negative_saturation_count[threshold_index]);
        $display("M37_DIVERSITY generic_saturation=%0d/%0d diversity=%0d/%0d",
                 generic_positive_saturations,
                 generic_negative_saturations, output_ones,
                 output_zeros);
        $finish;
    end

    initial begin
        #3000000;
        $fatal(1, "M37 timeout");
    end
endmodule

`default_nettype wire
