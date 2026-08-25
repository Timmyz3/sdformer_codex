`timescale 1ns/1ps
`default_nettype none

module tb_qfit_threshold_late_scale_uq0p24_radix20x4;
    localparam int TAG_W = 48;
    localparam int OUTPUTS = 4;
    localparam int PACKETS = 2048;
    localparam int FULL_BURST = 256;
    localparam integer unsigned RANDOM_SEED = 32'h4d333402;

    logic clk_core = 1'b0;
    logic rst_core = 1'b1;
    logic input_valid = 1'b0;
    logic input_ready;
    logic [TAG_W-1:0] input_tag = '0;
    logic [OUTPUTS-1:0] input_valid_bits = '0;
    logic signed [31:0] input_accumulator [0:OUTPUTS-1];
    logic [23:0] input_threshold_uq0p24 = '0;
    logic output_valid;
    logic output_ready = 1'b0;
    logic [TAG_W-1:0] output_tag;
    logic [OUTPUTS-1:0] output_valid_bits;
    logic signed [55:0] output_product [0:OUTPUTS-1];
    logic [95:0] multiplier_active_mask;
    logic digit_residual_zero;
    logic recombination_fits_signed56;
    logic protocol_error;

    logic signed [31:0] vector_accumulator [0:PACKETS-1][0:OUTPUTS-1];
    logic [23:0] vector_threshold [0:PACKETS-1];
    logic [OUTPUTS-1:0] vector_valid_bits [0:PACKETS-1];
    longint signed expected_product [0:PACKETS-1][0:OUTPUTS-1];

    integer send_index = 0;
    integer receive_index = 0;
    integer cycle_count = 0;
    integer stalled_output_cycles = 0;
    integer consecutive_full_rate_packets = 0;
    integer digit_reconstruction_checks = 0;
    integer negative_uq_digit_checks = 0;
    integer min_times_max_checks = 0;
    integer valid_scalar_products = 0;
    integer expected_valid_scalar_products = 0;
    integer seed_work;
    integer seed_sink;
    integer vector_fd;
    logic previous_full_input_fire = 1'b0;
    logic [15:0] mask_seen = '0;
    string vector_file;

    always #5 clk_core = ~clk_core;

    qfit_threshold_late_scale_uq0p24_radix20x4 dut (.*);

    function automatic longint signed reference_product(
        input logic signed [31:0] accumulator,
        input logic [23:0] threshold
    );
        longint signed accumulator_wide;
        longint signed threshold_wide;
        begin
            accumulator_wide = accumulator;
            threshold_wide = $unsigned(threshold);
            reference_product = accumulator_wide * threshold_wide;
        end
    endfunction

    function automatic logic [23:0] checkpoint_threshold(input integer index);
        case (index)
            0: checkpoint_threshold = 24'hfffffe;
            1: checkpoint_threshold = 24'hfffff1;
            2: checkpoint_threshold = 24'hffffff;
            3: checkpoint_threshold = 24'hffffeb;
            4: checkpoint_threshold = 24'hffff92;
            5: checkpoint_threshold = 24'hffffee;
            6: checkpoint_threshold = 24'hffff87;
            7: checkpoint_threshold = 24'hffff70;
            8: checkpoint_threshold = 24'hffff9f;
            default: checkpoint_threshold = 24'hfffdb4;
        endcase
    endfunction

    task automatic build_vectors;
        if (!$value$plusargs("M33_UQ_VECTOR_FILE=%s", vector_file))
            $fatal(1, "M33b vector output path plusarg is required");
        vector_fd = $fopen(vector_file, "w");
        if (vector_fd == 0)
            $fatal(1, "M33b cannot open vector output file");

        for (int packet = 0; packet < PACKETS; packet++) begin
            vector_threshold[packet] = $urandom();
            vector_valid_bits[packet] = packet < FULL_BURST
                ? 4'hf : packet[3:0];
            for (int output_index = 0; output_index < OUTPUTS;
                 output_index++)
                vector_accumulator[packet][output_index] = $signed($urandom());
        end

        for (int packet = 0; packet < 10; packet++)
            vector_threshold[packet] = checkpoint_threshold(packet);
        vector_threshold[10] = 24'hffffff;
        vector_accumulator[10][0] = 32'sh80000000;
        vector_accumulator[10][1] = 32'sh7fffffff;
        vector_accumulator[10][2] = -32'sd1;
        vector_accumulator[10][3] = 32'sd0;
        vector_threshold[11] = 24'h000000;
        vector_threshold[12] = 24'h000001;
        vector_threshold[13] = 24'h00003f;
        vector_threshold[14] = 24'h000040;
        vector_threshold[15] = 24'h00007f;
        vector_threshold[16] = 24'h000080;

        for (int packet = 0; packet < PACKETS; packet++) begin
            expected_valid_scalar_products += $countones(
                vector_valid_bits[packet]
            );
            for (int output_index = 0; output_index < OUTPUTS;
                 output_index++)
                expected_product[packet][output_index] = reference_product(
                    vector_accumulator[packet][output_index],
                    vector_threshold[packet]
                );
            $fdisplay(vector_fd,
                "%04d %06h %01h %08h %08h %08h %08h",
                packet, vector_threshold[packet], vector_valid_bits[packet],
                vector_accumulator[packet][0], vector_accumulator[packet][1],
                vector_accumulator[packet][2], vector_accumulator[packet][3]);
        end
        $fclose(vector_fd);
    endtask

    task automatic check_digit_reconstruction;
        longint signed reconstructed;
        longint signed digit_wide;
        if (input_valid && input_ready) begin
            if (!digit_residual_zero)
                $fatal(1, "M33b nonzero final balanced-radix residual");
            reconstructed = 0;
            for (int digit_index = 0; digit_index < 4; digit_index++) begin
                digit_wide = $signed(dut.threshold_digit[digit_index]);
                reconstructed += digit_wide <<< (7*digit_index);
                if (dut.threshold_digit[digit_index] < 0)
                    negative_uq_digit_checks += 1;
            end
            if (reconstructed != $unsigned(input_threshold_uq0p24))
                $fatal(1, "M33b UQ threshold digit reconstruction mismatch");
            for (int output_index = 0; output_index < OUTPUTS;
                 output_index++) begin
                reconstructed = 0;
                for (int digit_index = 0; digit_index < 5; digit_index++) begin
                    digit_wide = $signed(dut.acc_digit[output_index][digit_index]);
                    reconstructed += digit_wide <<< (7*digit_index);
                end
                if (reconstructed != $signed(input_accumulator[output_index]))
                    $fatal(1, "M33b accumulator digit reconstruction mismatch");
                digit_reconstruction_checks += 1;
            end
        end
    endtask

    always @(negedge clk_core) begin
        if (rst_core) begin
            input_valid <= 1'b0;
            output_ready <= 1'b0;
        end else begin
            output_ready <= send_index < FULL_BURST
                ? 1'b1 : ($urandom_range(0, 4) != 0);
            input_valid <= send_index < PACKETS;
            if (send_index < PACKETS) begin
                input_tag <= send_index;
                input_valid_bits <= vector_valid_bits[send_index];
                input_threshold_uq0p24 <= vector_threshold[send_index];
                for (int output_index = 0; output_index < OUTPUTS;
                     output_index++)
                    input_accumulator[output_index]
                        <= vector_accumulator[send_index][output_index];
            end
        end
    end

    always @(posedge clk_core) begin
        if (!rst_core) begin
            cycle_count += 1;
            check_digit_reconstruction();
            if (input_valid && input_ready) begin
                mask_seen[input_valid_bits] = 1'b1;
                if (previous_full_input_fire && input_valid_bits == 4'hf)
                    consecutive_full_rate_packets += 1;
                previous_full_input_fire = input_valid_bits == 4'hf;
                send_index += 1;
            end else begin
                previous_full_input_fire = 1'b0;
            end
            if (output_valid && !output_ready)
                stalled_output_cycles += 1;
            if (output_valid && output_ready) begin
                if (output_tag != receive_index)
                    $fatal(1, "M33b output tag mismatch");
                if (output_valid_bits != vector_valid_bits[receive_index])
                    $fatal(1, "M33b output valid mask mismatch");
                for (int output_index = 0; output_index < OUTPUTS;
                     output_index++) begin
                    if (output_valid_bits[output_index]
                        && $signed(output_product[output_index])
                            != expected_product[receive_index][output_index])
                        $fatal(1, "M33b exact UQ product mismatch");
                    if (output_valid_bits[output_index])
                        valid_scalar_products += 1;
                end
                if (receive_index == 10) begin
                    if (output_product[0] != -56'sd36028794871480320
                        || output_product[1] != 56'sd36028794854703105)
                        $fatal(1, "M33b signed56 UQ extreme mismatch");
                    min_times_max_checks += 2;
                end
                receive_index += 1;
            end
            if (protocol_error)
                $fatal(1, "M33b protocol/arithmetic guard fired");
            if (receive_index == PACKETS) begin
                if (send_index != PACKETS || min_times_max_checks != 2
                    || stalled_output_cycles == 0
                    || consecutive_full_rate_packets < FULL_BURST-1
                    || valid_scalar_products != expected_valid_scalar_products
                    || digit_reconstruction_checks != PACKETS*OUTPUTS
                    || negative_uq_digit_checks == 0
                    || mask_seen != 16'hffff)
                    $fatal(1, "M33b coverage/accounting incomplete");
                $display("M33_UQ_PASS packets=%0d valid_scalar_products=%0d digit_reconstruction_checks=%0d negative_uq_digits=%0d stalls=%0d consecutive_full_rate=%0d masks=%04h",
                    PACKETS, valid_scalar_products,
                    digit_reconstruction_checks, negative_uq_digit_checks,
                    stalled_output_cycles, consecutive_full_rate_packets,
                    mask_seen);
                $display("M33_UQ_RANDOM_SEED=0x%08x", RANDOM_SEED);
`ifdef SIMULATOR_VCS
                $display("SIMULATOR=Synopsys VCS");
`else
                $fatal(1, "M33b requires Synopsys VCS");
`endif
`ifdef SVA_RUNTIME_ENABLED
                $display("ASSERTIONS=enabled");
`else
                $fatal(1, "M33b requires runtime SVA");
`endif
                $finish;
            end
            if (cycle_count > 100000)
                $fatal(1, "M33b simulation timeout");
        end
    end

    initial begin
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
