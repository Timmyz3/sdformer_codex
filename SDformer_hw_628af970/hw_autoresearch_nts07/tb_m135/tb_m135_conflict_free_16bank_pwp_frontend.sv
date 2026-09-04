`timescale 1ns/1ps
`default_nettype none

module tb_m135_conflict_free_16bank_pwp_frontend;
    localparam int LANES = 96;
    localparam int OUT_W = 12;
    localparam int OUT_BITS = LANES*OUT_W;

    logic clk_core, rst_core;
    logic beat_valid, beat_ready, beat_start, beat_last;
    logic [3:0] beat_width;
    logic [31:0] beat_tag;
    logic [11:0] logical_base_word;
    logic [511:0] bank_words;
    logic [127:0] bank_row_addresses;
    logic [15:0] bank_use_mask;
    logic bank_conflict_free;
    logic beat_accept;
    logic output_valid, output_ready;
    logic [31:0] output_tag;
    logic [3:0] output_width;
    logic output_escape;
    logic [OUT_BITS-1:0] output_values;
    logic output_accept, protocol_error, busy;
    logic collecting;

    bit force_ready, pattern_stall, attack_phase, ii_phase;
    int unsigned cycle_count, vector_count, output_count, beat_count;
    int unsigned lane_checks, escape_count, ii_checks, last_start_cycle;
    int unsigned prior_start_beats, stall_cycles, row_crossings;
    int unsigned base_offset_hits[0:15];
    int unsigned invalid_base_attacks, reset_attacks;

    typedef struct packed {
        logic [31:0] tag;
        logic [3:0] width;
        logic escape;
        logic [OUT_BITS-1:0] values;
    } expected_t;
    expected_t expected_q[$];

    m135_conflict_free_16bank_pwp_frontend dut (.*);
    m135_conflict_free_16bank_pwp_frontend_assertions sva (.*);

    initial clk_core = 1'b0;
    always #1.5 clk_core = ~clk_core;

    function automatic int beats_for_width(input int width);
        if (width == 11) return 3;
        if (width inside {8,9,10}) return 2;
        return 1;
    endfunction

    task automatic clear_beat;
        beat_valid = 0;
        beat_start = 0;
        beat_last = 0;
        beat_width = 0;
        beat_tag = 0;
        logical_base_word = 0;
        bank_words = 0;
    endtask

    task automatic apply_reset(input int cycles);
        @(negedge clk_core);
        rst_core = 1;
        clear_beat();
        expected_q.delete();
        repeat (cycles) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 0;
    endtask

    task automatic drive_vector(input int width, input int tag,
                                input bit force_last_base);
        logic [1535:0] packed_payload;
        logic [OUT_BITS-1:0] expected_values;
        expected_t expected;
        integer signed value;
        int beats, base_word, physical_bank;
        packed_payload = 0;
        expected_values = 0;
        beats = beats_for_width(width);
        for (int lane = 0; lane < LANES; lane++) begin
            value = ((tag*31 + lane*19) % (1 << width))
                    - (1 << (width-1));
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
        expected.escape = 0;
        expected.values = expected_values;
        expected_q.push_back(expected);

        for (int beat = 0; beat < beats; beat++) begin
            if (force_last_base && beat == 0)
                base_word = 3664;
            else if (tag < 16 && beat == 0)
                base_word = 16 + tag;
            else
                base_word = (tag*53 + beat*197) % 3665;
            @(negedge clk_core);
            beat_valid = 1;
            beat_start = beat == 0;
            beat_last = beat == beats-1;
            beat_width = beat == 0 ? width[3:0] : 0;
            beat_tag = beat == 0 ? tag : 0;
            logical_base_word = base_word[11:0];
            bank_words = 0;
            #1ps;
            for (int word = 0; word < 16; word++) begin
                physical_bank = (base_word + word) & 15;
                bank_words[physical_bank*32 +: 32] =
                    packed_payload[beat*512 + word*32 +: 32];
            end
            do begin
                @(posedge clk_core);
                if (protocol_error)
                    $fatal(1, "M135 unexpected protocol error tag=%0d beat=%0d",
                           tag, beat);
            end while (!beat_accept);
            base_offset_hits[base_word & 15]++;
            if ((base_word & 15) != 0)
                row_crossings++;
        end
    endtask

    task automatic drive_escape(input int tag);
        expected_t expected;
        expected.tag = tag;
        expected.width = 12;
        expected.escape = 1;
        expected.values = 0;
        expected_q.push_back(expected);
        @(negedge clk_core);
        beat_valid = 1;
        beat_start = 1;
        beat_last = 1;
        beat_width = 12;
        beat_tag = tag;
        logical_base_word = 12'hfff;
        bank_words = '1;
        do @(posedge clk_core); while (!beat_accept);
        if (bank_row_addresses != 0)
            $fatal(1, "M135 escape touched bank address");
    endtask

    always @(negedge clk_core) begin
        if (rst_core) output_ready <= 0;
        else if (force_ready) output_ready <= 1;
        else if (pattern_stall) output_ready <= (cycle_count % 7) >= 2;
        else output_ready <= 1;
    end

    always @(posedge clk_core) begin
        cycle_count++;
        if (!rst_core && beat_accept) begin
            beat_count++;
            if (!attack_phase && beat_start) begin
                if (ii_phase && vector_count != 0) begin
                    if (cycle_count-last_start_cycle != prior_start_beats)
                        $fatal(1, "M135 II mismatch got=%0d expected=%0d",
                               cycle_count-last_start_cycle,
                               prior_start_beats);
                    ii_checks++;
                end
                last_start_cycle = cycle_count;
                prior_start_beats = beats_for_width(beat_width);
                vector_count++;
            end
        end
        if (!rst_core && output_valid && !attack_phase) begin
            if (expected_q.size() == 0)
                $fatal(1, "M135 output with empty scoreboard");
            if (output_tag !== expected_q[0].tag
                    || output_width !== expected_q[0].width
                    || output_escape !== expected_q[0].escape
                    || output_values !== expected_q[0].values)
                $fatal(1, "M135 output mismatch tag=%0d", output_tag);
            if (!output_ready) stall_cycles++;
        end
        if (!rst_core && output_accept && !attack_phase) begin
            if (output_escape) escape_count++;
            else lane_checks += LANES;
            expected_q.pop_front();
            output_count++;
        end
    end

    initial begin : test_sequence
        rst_core = 1;
        output_ready = 0;
        force_ready = 0;
        pattern_stall = 0;
        attack_phase = 0;
        ii_phase = 0;
        cycle_count = 0;
        vector_count = 0;
        output_count = 0;
        beat_count = 0;
        lane_checks = 0;
        escape_count = 0;
        ii_checks = 0;
        last_start_cycle = 0;
        prior_start_beats = 0;
        stall_cycles = 0;
        row_crossings = 0;
        invalid_base_attacks = 0;
        reset_attacks = 0;
        for (int bank = 0; bank < 16; bank++) base_offset_hits[bank] = 0;
        clear_beat();

        apply_reset(3);
        force_ready = 1;
        ii_phase = 1;
        for (int vector = 0; vector < 64; vector++)
            drive_vector(8+(vector%4), vector, vector == 0);
        @(negedge clk_core);
        clear_beat();
        wait (output_count == vector_count);
        ii_phase = 0;
        if (ii_checks != 63)
            $fatal(1, "M135 II count mismatch %0d", ii_checks);

        force_ready = 0;
        pattern_stall = 1;
        for (int vector = 64; vector < 96; vector++) begin
            if ((vector % 8) == 0) drive_escape(vector);
            else drive_vector(8+(vector%4), vector, 0);
        end
        @(negedge clk_core);
        clear_beat();
        wait (output_count == vector_count);
        pattern_stall = 0;
        force_ready = 1;
        repeat (2) @(posedge clk_core);

        for (int bank = 0; bank < 16; bank++)
            if (base_offset_hits[bank] == 0)
                $fatal(1, "M135 base offset unhit %0d", bank);

        // First illegal 16-word window must fail in the same cycle and stick.
        attack_phase = 1;
        @(negedge clk_core);
        beat_valid = 1;
        beat_start = 1;
        beat_last = 0;
        beat_width = 8;
        beat_tag = 200;
        logical_base_word = 3665;
        bank_words = '1;
        #1ps;
        if (!protocol_error || beat_ready || beat_accept
                || output_valid || bank_row_addresses != 0
                || bank_use_mask != 0 || bank_conflict_free)
            $fatal(1, "M135 invalid base not quarantined");
        invalid_base_attacks++;
        @(posedge clk_core);
        @(negedge clk_core);
        clear_beat();
        #1ps;
        if (!protocol_error || beat_ready || output_valid)
            $fatal(1, "M135 mapper fault not sticky");
        apply_reset(3);
        reset_attacks++;
        attack_phase = 0;
        repeat (2) @(posedge clk_core);
        if (protocol_error || busy)
            $fatal(1, "M135 reset did not recover");

        if (vector_count != 96 || output_count != 96
                || beat_count != 212 || lane_checks != 8832
                || escape_count != 4)
            $fatal(1, "M135 counter mismatch vectors=%0d outputs=%0d beats=%0d lanes=%0d escapes=%0d",
                   vector_count, output_count, beat_count,
                   lane_checks, escape_count);
        $display("PASS M135 conflict-free 16-bank PWP frontend VCS vectors=%0d outputs=%0d beats=%0d lanes=%0d escapes=%0d ii_checks=%0d stalls=%0d row_crossings=%0d base_offsets=16 invalid_base_attacks=%0d reset_attacks=%0d cycles_8_9_10_11=2_2_2_3 banks=16 service_bits=512 macro=false physical_speedup=false system_speedup=false headline=false",
                 vector_count, output_count, beat_count, lane_checks,
                 escape_count, ii_checks, stall_cycles, row_crossings,
                 invalid_base_attacks, reset_attacks);
        $finish;
    end

    initial begin
        #2000000;
        $fatal(1, "M135 directed VCS timeout");
    end
endmodule

`default_nettype wire
