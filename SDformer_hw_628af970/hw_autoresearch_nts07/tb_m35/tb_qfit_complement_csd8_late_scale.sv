`timescale 1ns/1ps
`default_nettype none

module tb_qfit_complement_csd8_late_scale;
    localparam int TAG_W = 48;
    localparam int EPOCH_W = 16;
    localparam int OUTPUTS = 8;
    localparam int TERMS = 4;
    localparam int CONFIGS = 10;
    localparam int PACKETS_PER_CONFIG = 512;
    localparam int PACKETS = CONFIGS*PACKETS_PER_CONFIG;
    localparam int FULL_BURST = 64;
    localparam integer unsigned RANDOM_SEED = 32'h4d350102;

    typedef enum logic [3:0] {
        LOAD, SEND, DRAIN, RELEASE,
        NEG_SUM, NEG_SUM_WAIT, RESET_ASSERT, NEG_SHIFT, NEG_SHIFT_WAIT,
        NEG_DONE, FINISH
    } state_t;
    state_t state = LOAD;

    logic clk_core = 1'b0;
    logic rst_core = 1'b1;
    logic config_valid = 1'b0;
    logic config_ready;
    logic [EPOCH_W-1:0] config_epoch = '0;
    logic [9:0] config_delta = '0;
    logic [TERMS-1:0] config_term_valid = '0;
    logic [TERMS-1:0] config_term_negative = '0;
    logic [3:0] config_term_shift [0:TERMS-1];
    logic config_loaded;
    logic [EPOCH_W-1:0] loaded_epoch;
    logic config_release_valid = 1'b0;
    logic config_release_ready;
    logic input_valid = 1'b0;
    logic input_ready;
    logic [TAG_W-1:0] input_tag = '0;
    logic [OUTPUTS-1:0] input_valid_bits = '0;
    logic signed [31:0] input_accumulator [0:OUTPUTS-1];
    logic output_valid;
    logic output_ready = 1'b0;
    logic [TAG_W-1:0] output_tag;
    logic [EPOCH_W-1:0] output_epoch;
    logic [OUTPUTS-1:0] output_valid_bits;
    logic signed [55:0] output_product [0:OUTPUTS-1];
    logic descriptor_legal;
    logic uses_integer_multiplier;
    logic busy;
    logic protocol_error;

    logic signed [31:0] vectors [0:PACKETS-1][0:OUTPUTS-1];
    logic [OUTPUTS-1:0] masks [0:PACKETS-1];
    longint signed expected [0:PACKETS-1][0:OUTPUTS-1];
    integer send_total = 0;
    integer receive_total = 0;
    integer accepted_in_config = 0;
    integer received_in_config = 0;
    integer config_index = 0;
    integer cycle_count = 0;
    integer stall_cycles = 0;
    integer consecutive_full_rate = 0;
    integer expected_valid_products = 0;
    integer valid_product_checks = 0;
    integer config_loads = 0;
    integer config_releases = 0;
    integer illegal_config_accepts = 0;
    integer illegal_config_rejections = 0;
    integer illegal_sum_rejections = 0;
    integer illegal_shift_rejections = 0;
    integer busy_release_rejects = 0;
    integer descriptor_pin_perturbations = 0;
    integer seed_work;
    integer seed_sink;
    integer vector_fd;
    logic previous_full_fire = 1'b0;
    logic [255:0] mask_seen = '0;
    string vector_file;

    always #5 clk_core = ~clk_core;
    qfit_complement_csd8_late_scale dut (.*);

    function automatic integer delta_for(input integer index);
        case (index)
            0: delta_for = 2; 1: delta_for = 15; 2: delta_for = 1;
            3: delta_for = 21; 4: delta_for = 110; 5: delta_for = 18;
            6: delta_for = 121; 7: delta_for = 144; 8: delta_for = 97;
            default: delta_for = 588;
        endcase
    endfunction

    task automatic drive_descriptor(input integer index);
        config_delta = delta_for(index);
        config_term_valid = '0;
        config_term_negative = '0;
        for (int term = 0; term < TERMS; term++)
            config_term_shift[term] = '0;
        case (index)
            0: begin config_term_valid=4'b0001; config_term_shift[0]=1; end
            1: begin config_term_valid=4'b0011; config_term_negative=4'b0001;
                config_term_shift[0]=0; config_term_shift[1]=4; end
            2: begin config_term_valid=4'b0001; config_term_shift[0]=0; end
            3: begin config_term_valid=4'b0111; config_term_shift[0]=0;
                config_term_shift[1]=2; config_term_shift[2]=4; end
            4: begin config_term_valid=4'b0111; config_term_negative=4'b0011;
                config_term_shift[0]=1; config_term_shift[1]=4;
                config_term_shift[2]=7; end
            5: begin config_term_valid=4'b0011; config_term_shift[0]=1;
                config_term_shift[1]=4; end
            6: begin config_term_valid=4'b0111; config_term_negative=4'b0010;
                config_term_shift[0]=0; config_term_shift[1]=3;
                config_term_shift[2]=7; end
            7: begin config_term_valid=4'b0011; config_term_shift[0]=4;
                config_term_shift[1]=7; end
            8: begin config_term_valid=4'b0111; config_term_negative=4'b0010;
                config_term_shift[0]=0; config_term_shift[1]=5;
                config_term_shift[2]=7; end
            default: begin config_term_valid=4'b1111;
                config_term_negative=4'b0001; config_term_shift[0]=2;
                config_term_shift[1]=4; config_term_shift[2]=6;
                config_term_shift[3]=9; end
        endcase
    endtask

    function automatic longint signed reference_product(
        input logic signed [31:0] accumulator,
        input integer delta
    );
        longint signed wide_accumulator;
        begin
            wide_accumulator = accumulator;
            reference_product = wide_accumulator * ((1 << 24)-delta);
        end
    endfunction

    task automatic build_vectors;
        if (!$value$plusargs("M35_VECTOR_FILE=%s", vector_file))
            $fatal(1, "M35 vector file plusarg required");
        vector_fd = $fopen(vector_file, "w");
        if (vector_fd == 0)
            $fatal(1, "M35 cannot open vector file");
        for (int packet = 0; packet < PACKETS; packet++) begin
            int offset;
            int descriptor;
            offset = packet % PACKETS_PER_CONFIG;
            descriptor = packet / PACKETS_PER_CONFIG;
            masks[packet] = offset < FULL_BURST ? 8'hff : offset[7:0];
            for (int output_index = 0; output_index < OUTPUTS; output_index++)
                vectors[packet][output_index] = $signed($urandom());
            if (offset == 0) begin
                vectors[packet][0] = 32'sh80000000;
                vectors[packet][1] = 32'sh7fffffff;
                vectors[packet][2] = -32'sd1;
                vectors[packet][3] = 32'sd0;
                vectors[packet][4] = 32'sd1;
            end
            expected_valid_products += $countones(masks[packet]);
            for (int output_index = 0; output_index < OUTPUTS; output_index++)
                expected[packet][output_index] = reference_product(
                    vectors[packet][output_index], delta_for(descriptor));
            $fdisplay(vector_fd,
                "%04d %03h %02h %08h %08h %08h %08h %08h %08h %08h %08h",
                packet, delta_for(descriptor), masks[packet],
                vectors[packet][0], vectors[packet][1], vectors[packet][2],
                vectors[packet][3], vectors[packet][4], vectors[packet][5],
                vectors[packet][6], vectors[packet][7]);
        end
        $fclose(vector_fd);
    endtask

    always @(negedge clk_core) begin
        config_valid <= 1'b0;
        config_release_valid <= 1'b0;
        input_valid <= 1'b0;
        if (rst_core) begin
            output_ready <= 1'b0;
            if (state == RESET_ASSERT) begin
                rst_core <= 1'b0;
                state = NEG_SHIFT;
            end
        end else begin
            output_ready <= (state == SEND && accepted_in_config < FULL_BURST)
                ? 1'b1 : ($urandom_range(0, 4) != 0);
            drive_descriptor(config_index);
            config_epoch <= config_index + 1;
            case (state)
                LOAD: config_valid <= 1'b1;
                SEND: begin
                    input_valid <= 1'b1;
                    // Exercise an early release request while work is live.
                    // The engine must refuse it without changing the epoch.
                    if ((accepted_in_config % 73) == 31)
                        config_release_valid <= 1'b1;
                    input_tag <= send_total;
                    input_valid_bits <= masks[send_total];
                    for (int output_index = 0; output_index < OUTPUTS;
                         output_index++)
                        input_accumulator[output_index] <=
                            vectors[send_total][output_index];
                    // Descriptor pins are allowed to move after the static
                    // configuration handshake; arithmetic must use the
                    // latched descriptor, not these live pins.
                    if ((accepted_in_config % 67) == 19) begin
                        config_delta = 10'd1023;
                        config_term_valid = 4'b0001;
                        config_term_negative = 4'b0000;
                        config_term_shift[0] = 4'd9;
                        descriptor_pin_perturbations += 1;
                    end
                end
                RELEASE: config_release_valid <= 1'b1;
                NEG_SUM: begin
                    // Independent negative A: every shift is in range, but
                    // the descriptor reconstructs two instead of delta one.
                    config_valid <= 1'b1;
                    config_delta = 10'd1;
                    config_term_valid = 4'b0001;
                    config_term_negative = 4'b0000;
                    config_term_shift[0] = 4'd1;
                end
                RESET_ASSERT: begin
                    // protocol_error is sticky by contract.  Reset the same
                    // DUT before testing the independent shift-range guard.
                    rst_core <= 1'b1;
                end
                NEG_SHIFT: begin
                    // Independent negative B: +2^10 - 1 reconstructs the
                    // declared delta 1023 exactly, but shift 10 is illegal.
                    config_valid <= 1'b1;
                    config_delta = 10'd1023;
                    config_term_valid = 4'b0011;
                    config_term_negative = 4'b0010;
                    config_term_shift[0] = 4'd10;
                    config_term_shift[1] = 4'd0;
                end
                default: begin end
            endcase
        end
    end

    always @(posedge clk_core) begin
        if (!rst_core) begin
            cycle_count += 1;
            if (config_valid && config_ready) begin
                if (state == NEG_SUM || state == NEG_SHIFT) begin
                    if (descriptor_legal)
                        $fatal(1, "M35 illegal descriptor was marked legal");
                    illegal_config_accepts += 1;
                    state = state == NEG_SUM ? NEG_SUM_WAIT : NEG_SHIFT_WAIT;
                end else begin
                    config_loads += 1;
                    accepted_in_config = 0;
                    received_in_config = 0;
                    state = SEND;
                end
            end
            if (input_valid && input_ready) begin
                mask_seen[input_valid_bits] = 1'b1;
                if (previous_full_fire && input_valid_bits == 8'hff)
                    consecutive_full_rate += 1;
                previous_full_fire = input_valid_bits == 8'hff;
                accepted_in_config += 1;
                send_total += 1;
                if (accepted_in_config == PACKETS_PER_CONFIG)
                    state = DRAIN;
            end else begin
                previous_full_fire = 1'b0;
            end
            if (output_valid && !output_ready)
                stall_cycles += 1;
            if (state == SEND && busy && config_release_valid
                    && !config_release_ready)
                busy_release_rejects += 1;
            if (output_valid && output_ready) begin
                if (output_tag != receive_total)
                    $fatal(1, "M35 output tag mismatch");
                if (output_epoch != (receive_total/PACKETS_PER_CONFIG)+1)
                    $fatal(1, "M35 config epoch mismatch");
                if (output_valid_bits != masks[receive_total])
                    $fatal(1, "M35 output mask mismatch");
                for (int output_index = 0; output_index < OUTPUTS;
                     output_index++) begin
                    if (output_valid_bits[output_index]
                        && $signed(output_product[output_index])
                            != expected[receive_total][output_index])
                        $fatal(1, "M35 exact complement product mismatch");
                    if (output_valid_bits[output_index])
                        valid_product_checks += 1;
                end
                receive_total += 1;
                received_in_config += 1;
                if (state == DRAIN
                        && received_in_config == PACKETS_PER_CONFIG)
                    state = RELEASE;
            end
            if (config_release_valid && config_release_ready) begin
                config_releases += 1;
                config_index += 1;
                state = config_index == CONFIGS ? NEG_SUM : LOAD;
            end
            if (uses_integer_multiplier)
                $fatal(1, "M35 multiplier-free guard fired");
            if (protocol_error && state != NEG_SUM_WAIT
                    && state != NEG_SHIFT_WAIT && state != RESET_ASSERT
                    && state != NEG_DONE && state != FINISH)
                $fatal(1, "M35 unexpected protocol error");
            if (state == NEG_DONE) begin
                state = FINISH;
            end else if (state == NEG_SUM_WAIT && protocol_error) begin
                if (config_loaded)
                    $fatal(1, "M35 illegal-sum descriptor changed loaded state");
                illegal_config_rejections += 1;
                illegal_sum_rejections += 1;
                state = RESET_ASSERT;
            end else if (state == NEG_SHIFT_WAIT && protocol_error) begin
                if (config_loaded)
                    $fatal(1, "M35 illegal-shift descriptor changed loaded state");
                illegal_config_rejections += 1;
                illegal_shift_rejections += 1;
                state = NEG_DONE;
            end
            if (state == FINISH) begin
                if (send_total != PACKETS || receive_total != PACKETS
                    || config_loads != CONFIGS || config_releases != CONFIGS
                    || valid_product_checks != expected_valid_products
                    || consecutive_full_rate < CONFIGS*(FULL_BURST-1)
                    || stall_cycles == 0 || mask_seen != {256{1'b1}}
                    || illegal_config_accepts != 2
                    || illegal_config_rejections != 2
                    || illegal_sum_rejections != 1
                    || illegal_shift_rejections != 1
                    || busy_release_rejects == 0
                    || descriptor_pin_perturbations == 0)
                    $fatal(1, "M35 coverage/accounting incomplete");
                $display("M35_PASS packets=%0d valid_products=%0d config_loads=%0d config_releases=%0d stalls=%0d consecutive_full_rate=%0d masks_all=1 illegal_accepts=%0d illegal_rejections=%0d illegal_sum_rejections=%0d illegal_shift_rejections=%0d busy_release_rejects=%0d descriptor_pin_perturbations=%0d",
                    PACKETS, valid_product_checks, config_loads,
                    config_releases, stall_cycles, consecutive_full_rate,
                    illegal_config_accepts, illegal_config_rejections,
                    illegal_sum_rejections, illegal_shift_rejections,
                    busy_release_rejects, descriptor_pin_perturbations);
                $display("M35_RANDOM_SEED=0x%08x", RANDOM_SEED);
`ifdef SIMULATOR_VCS
                $display("SIMULATOR=Synopsys VCS");
`else
                $fatal(1, "M35 requires Synopsys VCS");
`endif
`ifdef SVA_RUNTIME_ENABLED
                $display("ASSERTIONS=enabled");
`else
                $fatal(1, "M35 requires SVA");
`endif
                $finish;
            end
            if (cycle_count > 100000)
                $fatal(1, "M35 simulation timeout");
        end
    end

    initial begin
        for (int term = 0; term < TERMS; term++)
            config_term_shift[term] = '0;
        for (int output_index = 0; output_index < OUTPUTS; output_index++)
            input_accumulator[output_index] = '0;
        seed_work = RANDOM_SEED;
        seed_sink = $urandom(seed_work);
        build_vectors();
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;
    end
endmodule

`default_nettype wire
