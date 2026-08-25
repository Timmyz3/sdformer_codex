`timescale 1ns/1ps
`default_nettype none

module tb_qfit_threshold_late_scale_radix20x4;
    localparam int TAG_W = 48;
    localparam int OUTPUTS = 4;
    localparam int PACKETS = 1200;

    logic clk_core = 1'b0;
    logic rst_core = 1'b1;
    logic input_valid = 1'b0;
    logic input_ready;
    logic [TAG_W-1:0] input_tag = '0;
    logic [OUTPUTS-1:0] input_valid_bits = '0;
    logic signed [31:0] input_accumulator [0:OUTPUTS-1];
    logic signed [23:0] input_threshold = '0;
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
    logic signed [23:0] vector_threshold [0:PACKETS-1];
    logic [OUTPUTS-1:0] vector_valid_bits [0:PACKETS-1];
    longint signed expected_product [0:PACKETS-1][0:OUTPUTS-1];

    integer send_index = 0;
    integer receive_index = 0;
    integer cycle_count = 0;
    integer stalled_output_cycles = 0;
    integer consecutive_full_rate_packets = 0;
    integer digit_reconstruction_checks = 0;
    integer min_min_checks = 0;
    integer positive_carry_checks = 0;
    integer valid_scalar_products = 0;
    integer expected_valid_scalar_products = 0;
    integer seed_work;
    integer seed_sink;
    logic previous_input_fire = 1'b0;
    localparam integer unsigned RANDOM_SEED = 32'h4d333302;

    always #5 clk_core = ~clk_core;

    qfit_threshold_late_scale_radix20x4 dut (.*);

    function automatic longint signed reference_product(
        input logic signed [31:0] accumulator,
        input logic signed [23:0] threshold
    );
        longint signed accumulator_wide;
        longint signed threshold_wide;
        begin
            accumulator_wide = accumulator;
            threshold_wide = threshold;
            reference_product = accumulator_wide * threshold_wide;
        end
    endfunction

    function automatic logic [3:0] tail_mask(input integer index);
        case (index % 4)
            0: tail_mask = 4'b0001;
            1: tail_mask = 4'b0011;
            2: tail_mask = 4'b0111;
            default: tail_mask = 4'b1111;
        endcase
    endfunction

    task automatic build_vectors;
        vector_accumulator[0][0] = 32'sh80000000;
        vector_accumulator[0][1] = 32'sh80000000;
        vector_accumulator[0][2] = 32'sh80000000;
        vector_accumulator[0][3] = 32'sh80000000;
        vector_threshold[0] = 24'sh800000;
        vector_valid_bits[0] = 4'hf;

        vector_accumulator[1][0] = 32'sh7fffffff;
        vector_accumulator[1][1] = 32'sh7fffffff;
        vector_accumulator[1][2] = 32'sh7fffffff;
        vector_accumulator[1][3] = 32'sh7fffffff;
        vector_threshold[1] = 24'sh7fffff;
        vector_valid_bits[1] = 4'hf;

        vector_accumulator[2][0] = 32'sh80000000;
        vector_accumulator[2][1] = 32'sh7fffffff;
        vector_accumulator[2][2] = -32'sd1;
        vector_accumulator[2][3] = 32'sd1;
        vector_threshold[2] = 24'sh7fffff;
        vector_valid_bits[2] = 4'hf;

        vector_accumulator[3][0] = 32'sd0;
        vector_accumulator[3][1] = 32'sd63;
        vector_accumulator[3][2] = -32'sd64;
        vector_accumulator[3][3] = 32'sd127;
        vector_threshold[3] = -24'sd64;
        vector_valid_bits[3] = 4'hf;

        for (int packet = 4; packet < PACKETS; packet++) begin
            vector_threshold[packet] = $signed($urandom());
            vector_valid_bits[packet] = packet < 64
                ? 4'hf : tail_mask(packet);
            for (int output_index = 0; output_index < OUTPUTS;
                 output_index++)
                vector_accumulator[packet][output_index] = $signed($urandom());
        end
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
        end
    endtask

    task automatic check_digit_reconstruction;
        longint signed reconstructed;
        longint signed digit_wide;
        if (input_valid && input_ready) begin
            if (!digit_residual_zero)
                $fatal(1, "M33 nonzero final balanced-radix residual");
            reconstructed = 0;
            for (int digit_index = 0; digit_index < 4; digit_index++) begin
                digit_wide = $signed(dut.threshold_digit[digit_index]);
                reconstructed += digit_wide <<< (7*digit_index);
            end
            if (reconstructed != $signed(input_threshold))
                $fatal(1, "M33 threshold digit reconstruction mismatch");
            for (int output_index = 0; output_index < OUTPUTS;
                 output_index++) begin
                reconstructed = 0;
                for (int digit_index = 0; digit_index < 5; digit_index++) begin
                    digit_wide = $signed(dut.acc_digit[output_index][digit_index]);
                    reconstructed += digit_wide <<< (7*digit_index);
                end
                if (reconstructed != $signed(input_accumulator[output_index]))
                    $fatal(1, "M33 accumulator digit reconstruction mismatch");
                digit_reconstruction_checks += 1;
            end
        end
    endtask

    always @(negedge clk_core) begin
        if (rst_core) begin
            input_valid <= 1'b0;
            output_ready <= 1'b0;
        end else begin
            output_ready <= send_index < 64
                ? 1'b1 : ($urandom_range(0, 4) != 0);
            input_valid <= send_index < PACKETS;
            if (send_index < PACKETS) begin
                input_tag <= send_index;
                input_valid_bits <= vector_valid_bits[send_index];
                input_threshold <= vector_threshold[send_index];
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
                if (previous_input_fire && input_valid_bits == 4'hf)
                    consecutive_full_rate_packets += 1;
                previous_input_fire = 1'b1;
                send_index += 1;
            end else begin
                previous_input_fire = 1'b0;
            end
            if (output_valid && !output_ready)
                stalled_output_cycles += 1;
            if (output_valid && output_ready) begin
                if (output_tag != receive_index)
                    $fatal(1, "M33 output tag mismatch");
                if (output_valid_bits != vector_valid_bits[receive_index])
                    $fatal(1, "M33 output valid mask mismatch");
                for (int output_index = 0; output_index < OUTPUTS;
                     output_index++) begin
                    if (output_valid_bits[output_index]
                        && $signed(output_product[output_index])
                            != expected_product[receive_index][output_index])
                        $fatal(1, "M33 exact signed product mismatch");
                    if (output_valid_bits[output_index])
                        valid_scalar_products += 1;
                end
                if (receive_index == 0) begin
                    for (int output_index = 0; output_index < OUTPUTS;
                         output_index++) begin
                        if (output_product[output_index]
                            != 56'sh40000000000000)
                            $fatal(1, "M33 min-times-min signed56 corner mismatch");
                        min_min_checks += 1;
                    end
                end
                if (receive_index == 1)
                    positive_carry_checks += 1;
                receive_index += 1;
            end
            if (protocol_error)
                $fatal(1, "M33 protocol/arithmetic guard fired");
            if (receive_index == PACKETS) begin
                if (send_index != PACKETS || min_min_checks != 4
                    || positive_carry_checks != 1
                    || stalled_output_cycles == 0
                    || consecutive_full_rate_packets < 63
                    || valid_scalar_products != expected_valid_scalar_products
                    || digit_reconstruction_checks != PACKETS*OUTPUTS)
                    $fatal(1, "M33 coverage/accounting incomplete");
                $display("M33_PASS packets=%0d valid_scalar_products=%0d digit_reconstruction_checks=%0d stalls=%0d consecutive_full_rate=%0d",
                    PACKETS, valid_scalar_products,
                    digit_reconstruction_checks, stalled_output_cycles,
                    consecutive_full_rate_packets);
                $display("M33_RANDOM_SEED=0x%08x", RANDOM_SEED);
`ifdef SIMULATOR_VCS
                $display("SIMULATOR=Synopsys VCS");
`else
                $fatal(1, "M33 requires Synopsys VCS");
`endif
`ifdef SVA_RUNTIME_ENABLED
                $display("ASSERTIONS=enabled");
`else
                $fatal(1, "M33 requires runtime SVA");
`endif
                $finish;
            end
            if (cycle_count > 100000)
                $fatal(1, "M33 simulation timeout");
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
