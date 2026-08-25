`timescale 1ns/1ps
`default_nettype none

module tb_m133_dualrow512_elastic_pwp_stream;
    localparam int LANES = 96;
    localparam int OUT_W = 12;
    localparam int OUT_BITS = LANES * OUT_W;

    logic clk_core;
    logic rst_core;
    logic beat_valid;
    logic beat_ready;
    logic beat_start;
    logic beat_last;
    logic [3:0] beat_width;
    logic [31:0] beat_tag;
    logic [511:0] beat_data;
    logic beat_accept;
    logic output_valid;
    logic output_ready;
    logic [31:0] output_tag;
    logic [3:0] output_width;
    logic output_escape;
    logic [OUT_BITS-1:0] output_values;
    logic output_accept;
    logic protocol_error;
    logic collecting;
    logic busy;

    bit force_output_ready;
    bit pattern_stall_enable;
    bit force_output_stall;
    bit ii_phase;
    bit attack_phase;
    int unsigned cycle_count;
    int unsigned beat_accept_count;
    int unsigned vector_start_count;
    int unsigned output_accept_count;
    int unsigned lane_check_count;
    int unsigned escape_count;
    int unsigned output_stall_cycles;
    int unsigned long_stall_cycles;
    int unsigned start_ii_checks;
    int unsigned last_start_cycle;
    int unsigned prior_start_beats;
    int unsigned boundary_vectors;
    int unsigned protocol_attacks;
    int unsigned stall_fault_overlap_attacks;
    int unsigned reset_attacks;
    int unsigned idle_payload_checks;

    typedef struct packed {
        logic [31:0] tag;
        logic [3:0] width;
        logic escape;
        logic [OUT_BITS-1:0] values;
    } expected_output_t;
    expected_output_t expected_q[$];

    m133_dualrow512_elastic_pwp_stream dut (.*);

    m133_dualrow512_elastic_pwp_stream_assertions sva (.*);

    initial clk_core = 1'b0;
    always #1.5 clk_core = ~clk_core;

    function automatic int beats_for_width(input int width);
        if (width == 11)
            return 3;
        if (width inside {8, 9, 10})
            return 2;
        return 1;
    endfunction

    task automatic clear_beat;
        beat_valid = 1'b0;
        beat_start = 1'b0;
        beat_last = 1'b0;
        beat_width = '0;
        beat_tag = '0;
        beat_data = '0;
    endtask

    task automatic apply_reset(input int cycles);
        @(negedge clk_core);
        rst_core = 1'b1;
        clear_beat();
        expected_q.delete();
        repeat (cycles) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
    endtask

    task automatic drive_vector(
        input int width,
        input int tag,
        input bit boundary
    );
        logic [1535:0] packed_payload;
        logic [OUT_BITS-1:0] expected_values;
        expected_output_t expected;
        integer signed value;
        int beats;
        packed_payload = '0;
        expected_values = '0;
        beats = beats_for_width(width);
        for (int lane = 0; lane < LANES; lane++) begin
            if (boundary)
                value = (lane & 1) ? ((1 << (width-1)) - 1)
                                   : -(1 << (width-1));
            else begin
                value = ((tag * 29 + lane * 17) % (1 << width));
                value = value - (1 << (width-1));
            end
            case (width)
                8: packed_payload[lane*8 +: 8] = value[7:0];
                9: packed_payload[lane*9 +: 9] = value[8:0];
                10: packed_payload[lane*10 +: 10] = value[9:0];
                default: packed_payload[lane*11 +: 11] = value[10:0];
            endcase
            expected_values[lane*OUT_W +: OUT_W] = value[OUT_W-1:0];
        end
        expected.tag = tag;
        expected.width = width;
        expected.escape = 1'b0;
        expected.values = expected_values;
        expected_q.push_back(expected);
        if (boundary)
            boundary_vectors++;

        for (int beat = 0; beat < beats; beat++) begin
            @(negedge clk_core);
            beat_valid = 1'b1;
            beat_start = (beat == 0);
            beat_last = (beat == beats - 1);
            beat_width = (beat == 0) ? width[3:0] : 4'd0;
            beat_tag = (beat == 0) ? tag : 0;
            beat_data = packed_payload[beat*512 +: 512];
            do begin
                @(posedge clk_core);
                if (protocol_error)
                    $fatal(1, "unexpected PWP protocol error width=%0d tag=%0d",
                           width, tag);
            end while (!beat_accept);
        end
    endtask

    task automatic drive_escape(input int tag);
        expected_output_t expected;
        expected.tag = tag;
        expected.width = 12;
        expected.escape = 1'b1;
        expected.values = '0;
        expected_q.push_back(expected);
        @(negedge clk_core);
        beat_valid = 1'b1;
        beat_start = 1'b1;
        beat_last = 1'b1;
        beat_width = 4'd12;
        beat_tag = tag;
        beat_data = '0;
        do begin
            @(posedge clk_core);
            if (protocol_error)
                $fatal(1, "unexpected escape protocol error");
        end while (!beat_accept);
    endtask

    always @(negedge clk_core) begin
        if (rst_core || force_output_stall)
            output_ready <= 1'b0;
        else if (force_output_ready)
            output_ready <= 1'b1;
        else if (pattern_stall_enable)
            output_ready <= ((cycle_count % 7) >= 2);
        else
            output_ready <= 1'b1;
    end

    always @(posedge clk_core) begin : scoreboard
        cycle_count++;
        if (!rst_core && beat_accept) begin
            beat_accept_count++;
            if (!attack_phase && beat_start) begin
                if (ii_phase && vector_start_count != 0) begin
                    if (cycle_count - last_start_cycle != prior_start_beats)
                        $fatal(1, "PWP start II mismatch got=%0d expected=%0d",
                               cycle_count - last_start_cycle,
                               prior_start_beats);
                    start_ii_checks++;
                end
                last_start_cycle = cycle_count;
                prior_start_beats = beats_for_width(beat_width);
                vector_start_count++;
            end
        end

        if (!rst_core && output_valid && !attack_phase) begin
            if (expected_q.size() == 0)
                $fatal(1, "PWP output visible with empty scoreboard");
            if (output_tag !== expected_q[0].tag
                    || output_width !== expected_q[0].width
                    || output_escape !== expected_q[0].escape
                    || output_values !== expected_q[0].values)
                $fatal(1, "PWP output mismatch cycle=%0d tag=%0d",
                       cycle_count, output_tag);
            if (!output_ready)
                output_stall_cycles++;
        end

        if (!rst_core && output_accept && !attack_phase) begin
            if (expected_q.size() == 0)
                $fatal(1, "PWP output accepted with empty scoreboard");
            if (output_escape)
                escape_count++;
            else
                lane_check_count += LANES;
            expected_q.pop_front();
            output_accept_count++;
        end
    end

    initial begin : test_sequence
        rst_core = 1'b1;
        output_ready = 1'b0;
        force_output_ready = 1'b0;
        pattern_stall_enable = 1'b0;
        force_output_stall = 1'b0;
        ii_phase = 1'b0;
        attack_phase = 1'b0;
        cycle_count = 0;
        beat_accept_count = 0;
        vector_start_count = 0;
        output_accept_count = 0;
        lane_check_count = 0;
        escape_count = 0;
        output_stall_cycles = 0;
        long_stall_cycles = 0;
        start_ii_checks = 0;
        last_start_cycle = 0;
        prior_start_beats = 0;
        boundary_vectors = 0;
        protocol_attacks = 0;
        stall_fault_overlap_attacks = 0;
        reset_attacks = 0;
        idle_payload_checks = 0;
        clear_beat();

        apply_reset(3);
        force_output_ready = 1'b1;

        // Ready must not depend on semantic payload while valid is low.
        @(negedge clk_core);
        beat_valid = 1'b0;
        beat_start = 1'b0;
        beat_last = 1'b1;
        beat_width = 4'd7;
        beat_tag = 32'hdeadbeef;
        beat_data = '1;
        #1ps;
        if (!beat_ready || protocol_error)
            $fatal(1, "idle PWP ready depends on payload");
        idle_payload_checks++;

        // Consecutive vectors prove exact 2/2/2/3 start initiation intervals.
        clear_beat();
        ii_phase = 1'b1;
        for (int vector = 0; vector < 64; vector++)
            drive_vector(8 + (vector % 4), vector, vector < 4);
        @(negedge clk_core);
        clear_beat();
        wait (output_accept_count == vector_start_count);
        @(posedge clk_core);
        ii_phase = 1'b0;
        if (start_ii_checks != 63)
            $fatal(1, "PWP exact start II checks mismatch %0d",
                   start_ii_checks);

        // Mixed traffic and escapes under deterministic backpressure.
        force_output_ready = 1'b0;
        pattern_stall_enable = 1'b1;
        for (int vector = 64; vector < 104; vector++) begin
            if ((vector % 10) == 0)
                drive_escape(vector);
            else
                drive_vector(8 + (vector % 4), vector, 1'b0);
        end
        @(negedge clk_core);
        clear_beat();
        wait (output_accept_count == vector_start_count);

        // Explicit 23-cycle output stall validates all output fields.
        pattern_stall_enable = 1'b0;
        force_output_ready = 1'b1;
        drive_vector(11, 104, 1'b1);
        force_output_ready = 1'b0;
        force_output_stall = 1'b1;
        @(negedge clk_core);
        clear_beat();
        repeat (23) begin
            @(posedge clk_core);
            if (!output_valid)
                $fatal(1, "long-stall PWP output disappeared");
            long_stall_cycles++;
        end
        @(negedge clk_core);
        force_output_stall = 1'b0;
        force_output_ready = 1'b1;
        wait (output_accept_count == vector_start_count);
        repeat (2) @(posedge clk_core);
        if (expected_q.size() != 0 || output_valid || busy)
            $fatal(1, "PWP positive stream did not drain");

        // Dirty final padding is a same-cycle visible, sticky fault.
        attack_phase = 1'b1;
        @(negedge clk_core);
        beat_valid = 1'b1;
        beat_start = 1'b1;
        beat_last = 1'b0;
        beat_width = 4'd8;
        beat_tag = 32'd200;
        beat_data = '0;
        @(posedge clk_core);
        if (!beat_accept)
            $fatal(1, "dirty-padding setup beat not accepted");
        @(negedge clk_core);
        beat_start = 1'b0;
        beat_last = 1'b1;
        beat_width = '0;
        beat_tag = '0;
        beat_data = '0;
        beat_data[400] = 1'b1;
        #1ps;
        if (!protocol_error || beat_ready || beat_accept || output_valid)
            $fatal(1, "dirty PWP padding was not quarantined");
        protocol_attacks++;
        @(posedge clk_core);
        apply_reset(3);
        attack_phase = 1'b0;

        // Buffered output must be quarantined on the same edge that an invalid
        // new request would otherwise release it.
        attack_phase = 1'b1;
        force_output_ready = 1'b0;
        force_output_stall = 1'b1;
        @(negedge clk_core);
        beat_valid = 1'b1;
        beat_start = 1'b1;
        beat_last = 1'b0;
        beat_width = 4'd8;
        beat_tag = 32'd201;
        beat_data = '0;
        @(posedge clk_core);
        if (!beat_accept)
            $fatal(1, "same-cycle attack start not accepted");
        @(negedge clk_core);
        beat_start = 1'b0;
        beat_last = 1'b1;
        beat_width = '0;
        beat_tag = '0;
        beat_data = '0;
        @(posedge clk_core);
        if (!beat_accept)
            $fatal(1, "same-cycle attack final not accepted");
        // Hold the completed output for a full sampled stall edge before the
        // malformed next request.  Legal-traffic stall stability and
        // fail-closed request quarantine must compose without an SVA conflict.
        @(negedge clk_core);
        clear_beat();
        @(posedge clk_core);
        if (!output_valid || output_ready)
            $fatal(1, "overlap attack did not establish stalled output");
        force_output_stall = 1'b0;
        force_output_ready = 1'b1;
        @(negedge clk_core);
        beat_valid = 1'b1;
        beat_start = 1'b0;
        beat_last = 1'b0;
        beat_width = 4'd9;
        beat_tag = 32'hbad;
        beat_data = '0;
        #1ps;
        if (!protocol_error || beat_ready || beat_accept
                || output_valid || output_accept)
            $fatal(1, "same-cycle release fault did not quarantine output");
        protocol_attacks++;
        stall_fault_overlap_attacks++;
        @(posedge clk_core);

        // Reset while faulted must quiesce and recover without phantom output.
        @(negedge clk_core);
        rst_core = 1'b1;
        #1ps;
        if (protocol_error || beat_accept || output_valid || output_accept)
            $fatal(1, "reset did not quiesce PWP stream");
        reset_attacks++;
        clear_beat();
        repeat (2) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        attack_phase = 1'b0;
        repeat (2) @(posedge clk_core);
        if (protocol_error || busy)
            $fatal(1, "reset did not clear PWP stream");

        $display("PASS M133r2 dualrow512 elastic PWP stream VCS vectors=%0d outputs=%0d beats=%0d lanes=%0d escapes=%0d ii_checks=%0d stalls=%0d long_stall=%0d boundaries=%0d protocol_attacks=%0d stall_fault_overlap=%0d reset_attacks=%0d idle_payload=%0d cycles_8_9_10_11=2_2_2_3 input_bits=512 bank_mapper=false macro=false physical_speedup=false system_speedup=false headline=false",
                 vector_start_count, output_accept_count,
                 beat_accept_count, lane_check_count, escape_count,
                 start_ii_checks, output_stall_cycles,
                 long_stall_cycles, boundary_vectors,
                 protocol_attacks, stall_fault_overlap_attacks,
                 reset_attacks, idle_payload_checks);
        $finish;
    end

    initial begin
        #500000;
        $fatal(1, "M133 directed VCS timeout");
    end
endmodule

`default_nettype wire
