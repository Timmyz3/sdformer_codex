`timescale 1ns/1ps
`default_nettype none

module tb_qfit_atlif_t2_packed24_stream_core;
    localparam int TAG_W = 48;
    localparam int LANES = 24;
    localparam int ACC_W = 24;
    localparam int FIRST_STREAM_PACKETS = 64;
    localparam int SECOND_STREAM_PACKETS = 24;
    localparam int COLLISION_PACKETS = 2;
    localparam int FIRST_CONTEXT_PACKETS = FIRST_STREAM_PACKETS + 1;
    localparam int PACKETS = FIRST_STREAM_PACKETS + SECOND_STREAM_PACKETS
        + COLLISION_PACKETS;

    logic clk_core = 1'b0;
    logic rst_core = 1'b1;
    logic parameter_valid = 1'b0;
    logic parameter_ready;
    logic [31:0] parameter_weight = '0;
    logic [47:0] parameter_bias = '0;
    logic signed [23:0] parameter_threshold = '0;
    logic parameter_loaded;
    logic parameter_release_valid = 1'b0;
    logic parameter_release_ready;
    logic input_valid = 1'b0;
    logic input_ready;
    logic [TAG_W-1:0] input_tag = '0;
    logic [LANES-1:0] input_lane_valid = {LANES{1'b1}};
    logic [255:0] input_t0_values = '0;
    logic [255:0] input_t1_values = '0;
    logic result_valid;
    logic result_ready = 1'b0;
    logic [TAG_W-1:0] result_tag;
    logic [LANES-1:0] result_lane_valid;
    logic [LANES-1:0] result_t0_bits;
    logic [LANES-1:0] result_t1_bits;
    logic done;
    logic [TAG_W-1:0] done_tag;
    logic protocol_error;
    logic busy;
    logic arithmetic_active;
    logic [95:0] multiplier_active_mask;
    logic [4:0] result_fifo_occupancy;

    integer signed cfg_weight [0:3];
    integer signed cfg_bias [0:1];
    integer signed cfg_threshold;
    integer signed packet_x0 [0:PACKETS-1][0:LANES-1];
    integer signed packet_x1 [0:PACKETS-1][0:LANES-1];
    bit reference_t0 [0:PACKETS-1][0:LANES-1];
    bit reference_t1 [0:PACKETS-1][0:LANES-1];

    integer cycle_count = 0;
    integer input_packets = 0;
    integer output_packets = 0;
    integer done_packets = 0;
    integer parameter_fires = 0;
    integer release_fires = 0;
    integer previous_first_accept = -1;
    integer first_ii_matches = 0;
    integer input_stall_cycles = 0;
    integer output_stall_cycles = 0;
    integer maximum_fifo_occupancy = 0;
    integer full_fifo_cycles = 0;
    integer full_pop_push_cycles = 0;
    integer positive_output_saturations = 0;
    integer negative_output_saturations = 0;
    integer threshold_equal_cases = 0;
    integer threshold_below_cases = 0;
    integer release_collision_checks = 0;
    logic force_sink_stall = 1'b0;

    always #5 clk_core = ~clk_core;
    qfit_atlif_t2_packed24_stream_core dut (.*);

    function automatic integer signed clamp_q24(input longint signed value);
        if (value > 8388607)
            clamp_q24 = 8388607;
        else if (value < -8388608)
            clamp_q24 = -8388608;
        else
            clamp_q24 = value;
    endfunction

    task automatic configure_context(input int mode);
        if (mode == 0) begin
            cfg_weight[0] = 3;
            cfg_weight[1] = -5;
            cfg_weight[2] = 7;
            cfg_weight[3] = 2;
            cfg_bias[0] = 100;
            cfg_bias[1] = -200;
            cfg_threshold = 0;
        end else begin
            cfg_weight[0] = 127;
            cfg_weight[1] = 127;
            cfg_weight[2] = 127;
            cfg_weight[3] = 127;
            cfg_bias[0] = 8388607;
            cfg_bias[1] = -8388608;
            cfg_threshold = -1;
        end
    endtask

    task automatic build_packets(
        input int first_packet, input int last_packet, input int mode
    );
        longint signed sum0;
        longint signed sum1;
        integer signed saturated0;
        integer signed saturated1;
        configure_context(mode);
        for (int packet = first_packet; packet < last_packet; packet++) begin
            for (int lane = 0; lane < LANES; lane++) begin
                if (mode == 0) begin
                    packet_x0[packet][lane]
                        = (packet*37 + lane*19 + 7) % 256 - 128;
                    packet_x1[packet][lane]
                        = (packet*23 + lane*31 + 11) % 256 - 128;
                end else begin
                    packet_x0[packet][lane] = lane[0] ? -128 : 127;
                    packet_x1[packet][lane] = lane[0] ? -128 : 127;
                end
                sum0 = cfg_bias[0]
                    + packet_x0[packet][lane] * cfg_weight[0]
                    + packet_x1[packet][lane] * cfg_weight[1];
                sum1 = cfg_bias[1]
                    + packet_x0[packet][lane] * cfg_weight[2]
                    + packet_x1[packet][lane] * cfg_weight[3];
                if (mode != 0 && sum0 > 8388607)
                    positive_output_saturations
                        = positive_output_saturations + 1;
                if (mode != 0 && sum1 < -8388608)
                    negative_output_saturations
                        = negative_output_saturations + 1;
                saturated0 = clamp_q24(sum0);
                saturated1 = clamp_q24(sum1);
                if (saturated0 == cfg_threshold)
                    threshold_equal_cases = threshold_equal_cases + 1;
                if (saturated1 == cfg_threshold-1)
                    threshold_below_cases = threshold_below_cases + 1;
                reference_t0[packet][lane] = saturated0 >= cfg_threshold;
                reference_t1[packet][lane] = saturated1 >= cfg_threshold;
            end
        end
    endtask

    task automatic load_context;
        for (int index = 0; index < 4; index++)
            parameter_weight[(index*8) +: 8] = cfg_weight[index][7:0];
        for (int index = 0; index < 2; index++)
            parameter_bias[(index*ACC_W) +: ACC_W]
                = cfg_bias[index][ACC_W-1:0];
        parameter_threshold = cfg_threshold[ACC_W-1:0];
        @(negedge clk_core);
        parameter_valid = 1'b1;
        do @(posedge clk_core); while (!parameter_ready);
        @(negedge clk_core);
        parameter_valid = 1'b0;
    endtask

    task automatic release_context_with_input_collision(input int packet);
        @(negedge clk_core);
        parameter_release_valid = 1'b1;
        input_valid = 1'b1;
        input_tag = 48'h4d3100000000 + packet;
        input_lane_valid = {LANES{1'b1}};
        input_t0_values = '0;
        input_t1_values = '0;
        for (int lane = 0; lane < LANES; lane++) begin
            input_t0_values[(lane*8) +: 8] = packet_x0[packet][lane][7:0];
            input_t1_values[(lane*8) +: 8] = packet_x1[packet][lane][7:0];
        end
        #1;
        if (parameter_release_ready || !input_ready)
            $fatal(1, "M30B input-priority release arbitration failed");
        @(posedge clk_core);
        @(negedge clk_core);
        input_valid = 1'b0;
        do @(posedge clk_core); while (!parameter_release_ready);
        @(negedge clk_core);
        parameter_release_valid = 1'b0;
        release_collision_checks = release_collision_checks + 1;
    endtask

    task automatic send_range(input int first_packet, input int last_packet);
        @(negedge clk_core);
        input_valid = 1'b1;
        input_lane_valid = {LANES{1'b1}};
        for (int packet = first_packet; packet < last_packet; packet++) begin
            input_tag = 48'h4d3100000000 + packet;
            input_t0_values = '0;
            input_t1_values = '0;
            for (int lane = 0; lane < LANES; lane++) begin
                input_t0_values[(lane*8) +: 8]
                    = packet_x0[packet][lane][7:0];
                input_t1_values[(lane*8) +: 8]
                    = packet_x1[packet][lane][7:0];
            end
            do @(posedge clk_core); while (!input_ready);
            @(negedge clk_core);
        end
        input_valid = 1'b0;
    endtask

    always @(posedge clk_core) begin
        if (!rst_core) begin
            cycle_count = cycle_count + 1;
            if (parameter_valid && parameter_ready)
                parameter_fires = parameter_fires + 1;
            if (parameter_release_valid && parameter_release_ready)
                release_fires = release_fires + 1;
            if (input_valid && !input_ready)
                input_stall_cycles = input_stall_cycles + 1;
            if (result_valid && !result_ready)
                output_stall_cycles = output_stall_cycles + 1;
            if (result_fifo_occupancy > maximum_fifo_occupancy)
                maximum_fifo_occupancy = result_fifo_occupancy;
            if (result_fifo_occupancy == 16)
                full_fifo_cycles = full_fifo_cycles + 1;
            if (result_fifo_occupancy == 16 && result_valid && result_ready
                && input_valid && input_ready)
                full_pop_push_cycles = full_pop_push_cycles + 1;
            if (input_valid && input_ready) begin
                if (!arithmetic_active
                    || multiplier_active_mask !== {96{1'b1}})
                    $fatal(1, "M30B did not use all 96 products on input fire");
                if (input_packets < FIRST_STREAM_PACKETS) begin
                    if (previous_first_accept >= 0) begin
                        if (cycle_count-previous_first_accept != 1)
                            $fatal(1, "M30B first-cohort II drift got=%0d",
                                   cycle_count-previous_first_accept);
                        first_ii_matches = first_ii_matches + 1;
                    end
                    previous_first_accept = cycle_count;
                end
                input_packets = input_packets + 1;
            end
            if (done) begin
                if (done_tag !== 48'h4d3100000000 + done_packets)
                    $fatal(1, "M30B done tag mismatch index=%0d", done_packets);
                done_packets = done_packets + 1;
            end
            if (result_valid && result_ready) begin
                if (result_tag !== 48'h4d3100000000 + output_packets
                    || result_lane_valid !== {LANES{1'b1}})
                    $fatal(1, "M30B output identity mismatch index=%0d",
                           output_packets);
                for (int lane = 0; lane < LANES; lane++) begin
                    if (result_t0_bits[lane]
                        !== reference_t0[output_packets][lane]
                        || result_t1_bits[lane]
                        !== reference_t1[output_packets][lane])
                        $fatal(1, "M30B arithmetic mismatch packet=%0d lane=%0d",
                               output_packets, lane);
                end
                output_packets = output_packets + 1;
            end
        end
    end

    always @(negedge clk_core) begin
        if (rst_core)
            result_ready <= 1'b0;
        else
            result_ready <= !force_sink_stall;
    end

    initial begin
`ifdef SIMULATOR_VCS
        $display("SIMULATOR=Synopsys VCS");
`else
        $fatal(1, "M30B evidence requires Synopsys VCS");
`endif
`ifdef SVA_RUNTIME_ENABLED
        $display("ASSERTIONS=enabled");
`else
        $fatal(1, "M30B evidence requires enabled SVA");
`endif
        build_packets(0, FIRST_CONTEXT_PACKETS, 0);
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;
        load_context();
        send_range(0, FIRST_STREAM_PACKETS);
        while (output_packets < FIRST_STREAM_PACKETS || busy)
            @(posedge clk_core);
        release_context_with_input_collision(FIRST_STREAM_PACKETS);

        build_packets(FIRST_CONTEXT_PACKETS, PACKETS, 1);
        load_context();
        force_sink_stall = 1'b1;
        fork
            send_range(FIRST_CONTEXT_PACKETS, PACKETS-1);
            begin
                repeat (40) @(posedge clk_core);
                force_sink_stall = 1'b0;
            end
        join
        while (output_packets < PACKETS-1 || busy)
            @(posedge clk_core);
        release_context_with_input_collision(PACKETS-1);
        while (output_packets < PACKETS || busy)
            @(posedge clk_core);
        repeat (3) @(posedge clk_core);

        if (protocol_error || parameter_loaded)
            $fatal(1, "M30B did not finish cleanly");
        if (parameter_fires != 2 || release_fires != 2
            || release_collision_checks != 2)
            $fatal(1, "M30B context accounting mismatch load=%0d release=%0d collision=%0d",
                   parameter_fires, release_fires, release_collision_checks);
        if (input_packets != PACKETS || output_packets != PACKETS
            || done_packets != PACKETS
            || first_ii_matches != FIRST_STREAM_PACKETS-1)
            $fatal(1, "M30B packet accounting mismatch input=%0d output=%0d done=%0d ii=%0d",
                   input_packets, output_packets, done_packets,
                   first_ii_matches);
        if (maximum_fifo_occupancy != 16 || full_fifo_cycles == 0
            || full_pop_push_cycles == 0 || input_stall_cycles == 0
            || output_stall_cycles == 0)
            $fatal(1, "M30B FIFO coverage incomplete max=%0d full=%0d poppush=%0d install=%0d outstall=%0d",
                   maximum_fifo_occupancy, full_fifo_cycles,
                   full_pop_push_cycles, input_stall_cycles,
                   output_stall_cycles);
        if (positive_output_saturations == 0
            || negative_output_saturations == 0)
            $fatal(1, "M30B directed saturation coverage incomplete");
        if (threshold_equal_cases == 0 || threshold_below_cases == 0)
            $fatal(1, "M30B directed threshold boundary coverage incomplete equality=%0d below=%0d",
                   threshold_equal_cases, threshold_below_cases);
        $display("M30B_PASS packets=%0d lanes_per_packet=24 products_per_packet=96 first_ii=1 ii_matches=%0d max_fifo=%0d full_cycles=%0d full_pop_push=%0d input_stalls=%0d output_stalls=%0d out_sat=%0d/%0d threshold_eq_below=%0d/%0d releases=%0d input_priority_collisions=%0d",
                 PACKETS, first_ii_matches, maximum_fifo_occupancy,
                 full_fifo_cycles, full_pop_push_cycles, input_stall_cycles,
                 output_stall_cycles, positive_output_saturations,
                 negative_output_saturations, threshold_equal_cases,
                 threshold_below_cases, release_fires,
                 release_collision_checks);
        $finish;
    end

    initial begin
        #200000;
        $fatal(1, "M30B timeout");
    end
endmodule

`default_nettype wire
