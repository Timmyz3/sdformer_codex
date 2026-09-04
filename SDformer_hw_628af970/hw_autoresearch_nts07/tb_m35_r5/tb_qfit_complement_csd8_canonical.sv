`timescale 1ns/1ps
`default_nettype none

module tb_qfit_complement_csd8_canonical;
    localparam int TAG_W = 48;
    localparam int EPOCH_W = 16;
    localparam int OUTPUTS = 8;
    localparam int CONFIGS = 10;
    localparam int PACKETS_PER_CONFIG = 1024;
    localparam int PACKETS = CONFIGS*PACKETS_PER_CONFIG;
    localparam int FULL_BURST = 128;
    localparam integer unsigned RANDOM_SEED = 32'h4d350105;

    typedef enum logic [4:0] {
        LOAD, SEND, DRAIN, RELEASE,
        RESET_TEST_LOAD, RESET_TEST_SEND, RESET_TEST_WAIT_STALL,
        RESET_TEST_ASSERT, RESET_TEST_CHECK,
        ILLEGAL_LOAD, ILLEGAL_WAIT, ILLEGAL_RESET_ASSERT,
        ILLEGAL_RESET_CHECK, ILLEGAL_FINAL_WAIT, FINISH
    } state_t;
    state_t state = LOAD;

    logic clk_core = 1'b0;
    logic rst_core = 1'b1;
    logic config_valid = 1'b0;
    logic config_ready;
    logic [EPOCH_W-1:0] config_epoch = '0;
    logic [3:0] config_descriptor_id = '0;
    logic config_loaded;
    logic [EPOCH_W-1:0] loaded_epoch;
    logic [3:0] loaded_descriptor_id;
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
    integer illegal_id = 10;
    integer cycle_count = 0;
    integer stall_cycles = 0;
    integer consecutive_full_rate = 0;
    integer expected_valid_products = 0;
    integer valid_product_checks = 0;
    integer all_lane_product_checks = 0;
    integer config_loads = 0;
    integer config_releases = 0;
    integer illegal_config_accepts = 0;
    integer illegal_config_rejections = 0;
    integer busy_release_rejects = 0;
    integer descriptor_id_perturbations = 0;
    integer reset_test_inputs = 0;
    integer reset_under_stall_events = 0;
    integer seed_work;
    integer seed_sink;
    integer trace_fd;
    logic previous_full_fire = 1'b0;
    logic [255:0] mask_seen = '0;
    string trace_file;

    always #5 clk_core = ~clk_core;
    qfit_complement_csd8_canonical dut (.*);

    function automatic integer delta_for(input integer index);
        case (index)
            0: delta_for = 2; 1: delta_for = 15; 2: delta_for = 1;
            3: delta_for = 21; 4: delta_for = 110; 5: delta_for = 18;
            6: delta_for = 121; 7: delta_for = 144; 8: delta_for = 97;
            default: delta_for = 588;
        endcase
    endfunction

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
        if (!$value$plusargs("M35_R5_TRACE_FILE=%s", trace_file))
            $fatal(1, "M35-r5 trace file plusarg required");
        trace_fd = $fopen(trace_file, "w");
        if (trace_fd == 0)
            $fatal(1, "M35-r5 cannot open handshake trace");
        $fdisplay(trace_fd,
            "kind,sequence,descriptor_id,epoch,tag,mask,lane0,lane1,lane2,lane3,lane4,lane5,lane6,lane7");
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
                vectors[packet][1] = 32'sh80000001;
                vectors[packet][2] = -32'sd1;
                vectors[packet][3] = 32'sd0;
                vectors[packet][4] = 32'sd1;
                vectors[packet][5] = 32'sh7ffffffe;
                vectors[packet][6] = 32'sh7fffffff;
                vectors[packet][7] = -32'sd129;
            end
            expected_valid_products += $countones(masks[packet]);
            for (int output_index = 0; output_index < OUTPUTS; output_index++)
                expected[packet][output_index] = reference_product(
                    vectors[packet][output_index], delta_for(descriptor));
        end
    endtask

    task automatic write_input_trace(input integer seq_index);
        $fdisplay(trace_fd,
            "I,%0d,%0d,%0d,%0h,%02h,%08h,%08h,%08h,%08h,%08h,%08h,%08h,%08h",
            seq_index, seq_index/PACKETS_PER_CONFIG,
            (seq_index/PACKETS_PER_CONFIG)+1, input_tag, input_valid_bits,
            input_accumulator[0], input_accumulator[1],
            input_accumulator[2], input_accumulator[3],
            input_accumulator[4], input_accumulator[5],
            input_accumulator[6], input_accumulator[7]);
    endtask

    task automatic write_output_trace(input integer seq_index);
        $fdisplay(trace_fd,
            "O,%0d,%0d,%0d,%0h,%02h,%014h,%014h,%014h,%014h,%014h,%014h,%014h,%014h",
            seq_index, seq_index/PACKETS_PER_CONFIG,
            output_epoch, output_tag, output_valid_bits,
            output_product[0], output_product[1],
            output_product[2], output_product[3],
            output_product[4], output_product[5],
            output_product[6], output_product[7]);
    endtask

    task automatic write_config_trace(
        input string kind,
        input integer candidate_id,
        input logic observed_flag
    );
        $fdisplay(trace_fd,
            "%s,%0d,%0d,%0d,0,%02h,0,0,0,0,0,0,0,0",
            kind, candidate_id, candidate_id,
            config_epoch, {7'b0, observed_flag});
    endtask

    always @(negedge clk_core) begin
        config_valid = 1'b0;
        config_release_valid = 1'b0;
        input_valid = 1'b0;
        if (rst_core) begin
            output_ready = 1'b0;
            if (state == RESET_TEST_ASSERT) begin
                rst_core = 1'b0;
                state = RESET_TEST_CHECK;
            end else if (state == ILLEGAL_RESET_ASSERT) begin
                rst_core = 1'b0;
                state = ILLEGAL_RESET_CHECK;
            end
        end else begin
            output_ready = 1'b1;
            config_descriptor_id = config_index[3:0];
            config_epoch = config_index + 1;
            case (state)
                LOAD: begin
                    config_valid = 1'b1;
                end
                SEND: begin
                    output_ready = (accepted_in_config < FULL_BURST)
                        ? 1'b1 : ($urandom_range(0, 4) != 0);
                    input_valid = 1'b1;
                    if ((accepted_in_config % 73) == 31)
                        config_release_valid = 1'b1;
                    input_tag = send_total;
                    input_valid_bits = masks[send_total];
                    for (int output_index = 0; output_index < OUTPUTS;
                         output_index++)
                        input_accumulator[output_index] =
                            vectors[send_total][output_index];
                    // Live candidate ID pins deliberately become illegal
                    // while config_valid=0.  Arithmetic must use the latched
                    // canonical row and must not alias 4'hA.
                    if ((accepted_in_config % 67) == 19) begin
                        config_descriptor_id = 4'hA;
                        descriptor_id_perturbations += 1;
                    end
                end
                DRAIN: begin
                    output_ready = 1'b1;
                end
                RELEASE: begin
                    config_release_valid = 1'b1;
                end
                RESET_TEST_LOAD: begin
                    config_descriptor_id = 4'd0;
                    config_epoch = 16'h6000;
                    config_valid = 1'b1;
                    output_ready = 1'b0;
                end
                RESET_TEST_SEND: begin
                    config_descriptor_id = 4'hA;
                    input_valid = 1'b1;
                    input_tag = 48'hfeed00000001;
                    input_valid_bits = 8'hff;
                    for (int output_index = 0; output_index < OUTPUTS;
                         output_index++)
                        input_accumulator[output_index] =
                            output_index == 0 ? 32'sh80000000
                                              : $signed(output_index);
                    output_ready = 1'b0;
                end
                RESET_TEST_WAIT_STALL: begin
                    config_descriptor_id = 4'hA;
                    output_ready = 1'b0;
                    if (output_valid) begin
                        rst_core = 1'b1;
                        reset_under_stall_events += 1;
                        state = RESET_TEST_ASSERT;
                    end
                end
                ILLEGAL_LOAD: begin
                    config_descriptor_id = illegal_id[3:0];
                    config_epoch = 16'h7000 + illegal_id;
                    config_valid = 1'b1;
                end
                ILLEGAL_WAIT: begin
                    // Preserve the attempted candidate identity in the
                    // externally replayable protocol-error trace.
                    config_descriptor_id = illegal_id[3:0];
                    config_epoch = 16'h7000 + illegal_id;
                end
                ILLEGAL_RESET_ASSERT: begin
                    rst_core = 1'b1;
                end
                default: begin end
            endcase
        end
    end

    always @(posedge clk_core) begin
        if (!rst_core) begin
            cycle_count += 1;
            if (config_valid && config_ready) begin
                if (state == LOAD) begin
                    if (!descriptor_legal
                            || config_descriptor_id != config_index[3:0])
                        $fatal(1, "M35-r5 legal descriptor decode mismatch");
                    write_config_trace("C", config_index, descriptor_legal);
                    config_loads += 1;
                    accepted_in_config = 0;
                    received_in_config = 0;
                    state = SEND;
                end else if (state == RESET_TEST_LOAD) begin
                    if (!descriptor_legal || config_descriptor_id != 4'd0)
                        $fatal(1, "M35-r5 reset test legal decode mismatch");
                    state = RESET_TEST_SEND;
                end else if (state == ILLEGAL_LOAD) begin
                    if (descriptor_legal || config_descriptor_id < 4'd10)
                        $fatal(1, "M35-r5 illegal ID alias accepted as legal");
                    write_config_trace("C", illegal_id, descriptor_legal);
                    illegal_config_accepts += 1;
                    state = ILLEGAL_WAIT;
                end else begin
                    $fatal(1, "M35-r5 unexpected configuration handshake");
                end
            end

            if (input_valid && input_ready) begin
                if (state == SEND) begin
                    write_input_trace(send_total);
                    mask_seen[input_valid_bits] = 1'b1;
                    if (previous_full_fire && input_valid_bits == 8'hff)
                        consecutive_full_rate += 1;
                    previous_full_fire = input_valid_bits == 8'hff;
                    accepted_in_config += 1;
                    send_total += 1;
                    if (accepted_in_config == PACKETS_PER_CONFIG)
                        state = DRAIN;
                end else if (state == RESET_TEST_SEND) begin
                    reset_test_inputs += 1;
                    state = RESET_TEST_WAIT_STALL;
                end else begin
                    $fatal(1, "M35-r5 unexpected input handshake");
                end
            end else begin
                previous_full_fire = 1'b0;
            end

            if (output_valid && !output_ready)
                stall_cycles += 1;
            if (state == SEND && busy && config_release_valid
                    && !config_release_ready)
                busy_release_rejects += 1;

            if (output_valid && output_ready) begin
                if (receive_total >= PACKETS)
                    $fatal(1, "M35-r5 unexpected output after main workload");
                if (output_tag != receive_total)
                    $fatal(1, "M35-r5 output tag/order mismatch");
                if (output_epoch != (receive_total/PACKETS_PER_CONFIG)+1)
                    $fatal(1, "M35-r5 output epoch mismatch");
                if (output_valid_bits != masks[receive_total])
                    $fatal(1, "M35-r5 output mask mismatch");
                write_output_trace(receive_total);
                for (int output_index = 0; output_index < OUTPUTS;
                     output_index++) begin
                    if ($signed(output_product[output_index])
                            != expected[receive_total][output_index])
                        $fatal(1,
                            "M35-r5 exact DUT signed56 product mismatch");
                    all_lane_product_checks += 1;
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
                if (state != RELEASE)
                    $fatal(1, "M35-r5 release accepted outside empty boundary");
                config_releases += 1;
                config_index += 1;
                state = config_index == CONFIGS ? RESET_TEST_LOAD : LOAD;
            end

            if (uses_integer_multiplier)
                $fatal(1, "M35-r5 multiplier-free guard fired");
            if (protocol_error && state != ILLEGAL_WAIT
                    && state != ILLEGAL_RESET_ASSERT
                    && state != ILLEGAL_FINAL_WAIT && state != FINISH)
                $fatal(1, "M35-r5 unexpected protocol error");

            if (state == RESET_TEST_CHECK) begin
                if (config_loaded || output_valid || busy || protocol_error)
                    $fatal(1, "M35-r5 reset-under-stall did not flush state");
                config_index = 0;
                state = ILLEGAL_LOAD;
            end else if (state == ILLEGAL_WAIT && protocol_error) begin
                if (config_loaded || output_valid || busy)
                    $fatal(1, "M35-r5 illegal ID changed architectural state");
                write_config_trace("E", illegal_id, protocol_error);
                illegal_config_rejections += 1;
                if (illegal_id == 15) begin
                    state = ILLEGAL_FINAL_WAIT;
                end else begin
                    state = ILLEGAL_RESET_ASSERT;
                end
            end else if (state == ILLEGAL_FINAL_WAIT) begin
                state = FINISH;
            end else if (state == ILLEGAL_RESET_CHECK) begin
                if (config_loaded || output_valid || busy || protocol_error)
                    $fatal(1, "M35-r5 illegal-ID reset recovery failed");
                illegal_id += 1;
                state = ILLEGAL_LOAD;
            end

            if (state == FINISH) begin
                $fclose(trace_fd);
                if (send_total != PACKETS || receive_total != PACKETS
                    || config_loads != CONFIGS || config_releases != CONFIGS
                    || all_lane_product_checks != PACKETS*OUTPUTS
                    || valid_product_checks != expected_valid_products
                    || consecutive_full_rate < CONFIGS*(FULL_BURST-1)
                    || stall_cycles == 0 || mask_seen != {256{1'b1}}
                    || illegal_config_accepts != 6
                    || illegal_config_rejections != 6
                    || busy_release_rejects == 0
                    || descriptor_id_perturbations == 0
                    || reset_test_inputs != 1
                    || reset_under_stall_events != 1)
                    $fatal(1, "M35-r5 coverage/accounting incomplete");
                $display("M35_R5_PASS packets=%0d all_lane_products=%0d valid_products=%0d config_loads=%0d config_releases=%0d legal_ids=10 illegal_ids=6 illegal_rejections=%0d stalls=%0d consecutive_full_rate=%0d masks_all=1 busy_release_rejects=%0d idA_pin_perturbations=%0d reset_under_stall=%0d mismatches=0",
                    PACKETS, all_lane_product_checks, valid_product_checks,
                    config_loads, config_releases,
                    illegal_config_rejections, stall_cycles,
                    consecutive_full_rate, busy_release_rejects,
                    descriptor_id_perturbations,
                    reset_under_stall_events);
                $display("M35_R5_RANDOM_SEED=0x%08x", RANDOM_SEED);
`ifdef SIMULATOR_VCS
                $display("M35_R5_SIMULATOR=Synopsys VCS");
`else
                $fatal(1, "M35-r5 requires Synopsys VCS");
`endif
`ifdef SVA_RUNTIME_ENABLED
                $display("M35_R5_ASSERTIONS=enabled");
`else
                $fatal(1, "M35-r5 requires SVA");
`endif
                $finish;
            end
            if (cycle_count > 200000)
                $fatal(1, "M35-r5 simulation timeout");
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
