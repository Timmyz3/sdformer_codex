`timescale 1ns/1ps
`default_nettype none

module tb_m99_phase_slack_guarded_wordpacked_pwp_stream;
    logic clk_core, rst_core;
    logic phase_load_valid;
    logic [591:0] phase_metadata;
    logic lookup_valid;
    logic [3:0] lookup_pattern;
    logic [2:0] lookup_block, lookup_beat;
    logic [31:0] lookup_tag;
    logic [255:0] bank_words;
    logic output_ready;

    logic ref_phase_load_ready, ref_phase_loaded, ref_metadata_error;
    logic ref_lookup_ready, ref_output_valid, ref_output_escape;
    logic [79:0] ref_bank_rows;
    logic [31:0] ref_output_tag;
    logic [3:0] ref_output_width;
    logic [1151:0] ref_output_values;
    logic ref_output_accept, ref_protocol_error, ref_busy;

    logic dut_phase_load_ready, dut_phase_loaded, dut_metadata_error;
    logic dut_lookup_ready, dut_output_valid, dut_output_escape;
    logic [79:0] dut_bank_rows;
    logic [31:0] dut_output_tag;
    logic [3:0] dut_output_width;
    logic [1151:0] dut_output_values;
    logic dut_output_accept, dut_protocol_error, dut_busy;

    integer cycle_count, legal_entries, legal_beats, stall_cycles;
    integer poison_attacks, early_lookup_attacks, simultaneous_attacks;
    integer loaded_priority_attacks;
    integer parser_cycles;
    logic compare_enable;

    guarded_wordpacked_pwp_stream reference (
        .clk_core(clk_core), .rst_core(rst_core),
        .phase_load_valid(phase_load_valid),
        .phase_load_ready(ref_phase_load_ready),
        .phase_metadata(phase_metadata),
        .phase_loaded(ref_phase_loaded),
        .metadata_error(ref_metadata_error),
        .lookup_valid(lookup_valid), .lookup_ready(ref_lookup_ready),
        .lookup_pattern(lookup_pattern), .lookup_block(lookup_block),
        .lookup_beat(lookup_beat), .lookup_tag(lookup_tag),
        .bank_words(bank_words), .bank_row_addresses(ref_bank_rows),
        .output_valid(ref_output_valid), .output_ready(output_ready),
        .output_tag(ref_output_tag), .output_width(ref_output_width),
        .output_escape(ref_output_escape),
        .output_values(ref_output_values),
        .output_accept(ref_output_accept),
        .protocol_error(ref_protocol_error), .busy(ref_busy)
    );

    phase_slack_guarded_wordpacked_pwp_stream dut (
        .clk_core(clk_core), .rst_core(rst_core),
        .phase_load_valid(phase_load_valid),
        .phase_load_ready(dut_phase_load_ready),
        .phase_metadata(phase_metadata),
        .phase_loaded(dut_phase_loaded),
        .metadata_error(dut_metadata_error),
        .lookup_valid(lookup_valid), .lookup_ready(dut_lookup_ready),
        .lookup_pattern(lookup_pattern), .lookup_block(lookup_block),
        .lookup_beat(lookup_beat), .lookup_tag(lookup_tag),
        .bank_words(bank_words), .bank_row_addresses(dut_bank_rows),
        .output_valid(dut_output_valid), .output_ready(output_ready),
        .output_tag(dut_output_tag), .output_width(dut_output_width),
        .output_escape(dut_output_escape),
        .output_values(dut_output_values),
        .output_accept(dut_output_accept),
        .protocol_error(dut_protocol_error), .busy(dut_busy)
    );

    phase_slack_guarded_wordpacked_pwp_stream_assertions dut_sva (
        .clk_core(clk_core), .rst_core(rst_core),
        .phase_load_valid(phase_load_valid),
        .phase_load_ready(dut_phase_load_ready),
        .phase_loaded(dut_phase_loaded),
        .metadata_error(dut_metadata_error),
        .lookup_valid(lookup_valid), .lookup_ready(dut_lookup_ready),
        .output_valid(dut_output_valid), .output_ready(output_ready),
        .output_tag(dut_output_tag), .output_width(dut_output_width),
        .output_escape(dut_output_escape),
        .output_values(dut_output_values),
        .output_accept(dut_output_accept),
        .protocol_error(dut_protocol_error), .busy(dut_busy),
        .parse_active(dut.parse_active_q),
        .parse_index(dut.parse_index_q),
        .parse_cursor(dut.parse_cursor_q),
        .parse_code(dut.parse_code),
        .parse_poison(dut.parse_poison_q),
        .captured_metadata(dut.metadata_q),
        .lookup_error(dut.lookup_error_q),
        .m82_beat_accept(dut.m82_beat_accept)
    );

    always #1.5 clk_core = ~clk_core;
    always @(posedge clk_core) cycle_count <= cycle_count + 1;

    always @(posedge clk_core) begin
        #1;
        if (compare_enable) begin
            if ({ref_lookup_ready, ref_bank_rows, ref_output_valid,
                 ref_output_tag, ref_output_width, ref_output_escape,
                 ref_output_values, ref_output_accept, ref_protocol_error,
                 ref_metadata_error, ref_busy}
                !==
                {dut_lookup_ready, dut_bank_rows, dut_output_valid,
                 dut_output_tag, dut_output_width, dut_output_escape,
                 dut_output_values, dut_output_accept, dut_protocol_error,
                 dut_metadata_error, dut_busy})
                $fatal(1, "M99 M85 differential mismatch cycle=%0d", cycle_count);
        end
    end

    initial begin
        #300000;
        $fatal(1, "M99 watchdog timeout cycle=%0d", cycle_count);
    end

    function automatic integer words_for_code(input integer code);
        case (code)
            0: words_for_code = 24;
            1: words_for_code = 27;
            2: words_for_code = 30;
            3: words_for_code = 33;
            4: words_for_code = 0;
            default: words_for_code = 0;
        endcase
    endfunction

    function automatic integer beats_for_code(input integer code);
        case (code)
            0: beats_for_code = 3;
            1, 2: beats_for_code = 4;
            3: beats_for_code = 5;
            4: beats_for_code = 1;
            default: beats_for_code = 0;
        endcase
    endfunction

    task automatic build_legal_metadata;
        integer cursor, code;
        begin
            phase_metadata = '0;
            cursor = 0;
            for (int pattern = 0; pattern < 16; pattern++) begin
                phase_metadata[384+pattern*13 +: 13] = cursor[12:0];
                for (int block = 0; block < 8; block++) begin
                    code = (pattern*8 + block) % 5;
                    phase_metadata[(pattern*8+block)*3 +: 3] = code[2:0];
                    cursor += words_for_code(code);
                end
            end
            if (cursor <= 0 || cursor > 3680)
                $fatal(1, "M99 generated illegal terminal=%0d", cursor);
        end
    endtask

    task automatic reset_pair;
        begin
            compare_enable = 1'b0;
            @(negedge clk_core);
            rst_core = 1'b1;
            phase_load_valid = 1'b0;
            lookup_valid = 1'b0;
            output_ready = 1'b1;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
        end
    endtask

    task automatic load_and_wait_for_parser(input logic expect_poison);
        integer waited;
        begin
            do @(posedge clk_core); while (!ref_phase_load_ready
                                           || !dut_phase_load_ready);
            @(negedge clk_core);
            phase_load_valid = 1'b1;
            @(posedge clk_core);
            #1;
            if (!ref_phase_loaded || dut_phase_loaded || !dut_busy)
                $fatal(1, "M99 load-accept state mismatch");
            @(negedge clk_core);
            phase_load_valid = 1'b0;
            waited = 0;
            while (!dut_phase_loaded) begin
                @(posedge clk_core);
                #1;
                waited++;
                if (!dut_phase_loaded && dut_phase_load_ready)
                    $fatal(1, "M99 accepted overlapping metadata parse");
                if (waited > 128)
                    $fatal(1, "M99 parser exceeded 128 cycles");
            end
            if (waited != 128)
                $fatal(1, "M99 parser latency got=%0d expected=128", waited);
            parser_cycles += waited;
            compare_enable = 1'b1;
            #1;
            if (dut_metadata_error !== expect_poison
                    || ref_metadata_error !== expect_poison)
                $fatal(1, "M99 poison mismatch expected=%0d", expect_poison);
        end
    endtask

    task automatic drive_entry(input integer entry, input integer code);
        integer beats;
        logic [1151:0] held_output;
        begin
            beats = beats_for_code(code);
            for (int beat = 0; beat < beats; beat++) begin
                @(negedge clk_core);
                lookup_valid = 1'b1;
                lookup_pattern = (entry / 8);
                lookup_block = (entry % 8);
                lookup_beat = beat;
                lookup_tag = beat == 0 ? entry + 1 : 0;
                for (int word = 0; word < 8; word++)
                    bank_words[word*32 +: 32] =
                        32'h9e370000 ^ ((entry+1) << 8) ^ (beat << 4) ^ word;
                do begin
                    @(posedge clk_core);
                    #1;
                    if (ref_lookup_ready !== dut_lookup_ready)
                        $fatal(1, "M99 ready mismatch entry=%0d beat=%0d",
                               entry, beat);
                end while (!dut_lookup_ready);
                legal_beats++;
            end
            legal_entries++;
            if ((entry % 31) == 0) begin
                @(negedge clk_core);
                lookup_valid = 1'b0;
                output_ready = 1'b0;
                held_output = dut_output_values;
                repeat (2) begin
                    @(posedge clk_core);
                    #1;
                    if (!dut_output_valid || dut_output_values !== held_output)
                        $fatal(1, "M99 output changed under stall entry=%0d", entry);
                    stall_cycles++;
                end
                @(negedge clk_core);
                output_ready = 1'b1;
            end
        end
    endtask

    initial begin
        integer code;
        clk_core = 0;
        rst_core = 1;
        phase_load_valid = 0;
        phase_metadata = '0;
        lookup_valid = 0;
        lookup_pattern = 0;
        lookup_block = 0;
        lookup_beat = 0;
        lookup_tag = 0;
        bank_words = 0;
        output_ready = 1;
        compare_enable = 0;
        cycle_count = 0;
        legal_entries = 0;
        legal_beats = 0;
        stall_cycles = 0;
        poison_attacks = 0;
        early_lookup_attacks = 0;
        simultaneous_attacks = 0;
        loaded_priority_attacks = 0;
        parser_cycles = 0;

        // A new phase and an old-phase lookup may not be accepted together.
        reset_pair();
        build_legal_metadata();
        @(negedge clk_core);
        phase_load_valid = 1'b1;
        lookup_valid = 1'b1;
        lookup_pattern = 0; lookup_block = 0; lookup_beat = 0;
        @(posedge clk_core); #1;
        if (dut_phase_load_ready || dut_lookup_ready || dut_busy
                || dut_phase_loaded)
            $fatal(1, "M99 simultaneous load/lookup was not rejected");
        simultaneous_attacks++;

        // Early use is sticky-faulted and cannot leak an unaudited lookup.
        reset_pair();
        build_legal_metadata();
        do @(posedge clk_core); while (!dut_phase_load_ready);
        @(negedge clk_core); phase_load_valid = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        phase_load_valid = 1'b0;
        lookup_valid = 1'b1;
        lookup_pattern = 0; lookup_block = 0; lookup_beat = 0;
        @(posedge clk_core); #1;
        if (dut_lookup_ready || !dut_protocol_error)
            $fatal(1, "M99 early lookup did not fail closed");
        early_lookup_attacks++;

        // With a clean old phase, simultaneous replacement and lookup gives
        // priority to the old-phase lookup but must not accept the load.
        reset_pair();
        build_legal_metadata();
        load_and_wait_for_parser(1'b0);
        compare_enable = 1'b0;
        @(negedge clk_core);
        phase_load_valid = 1'b1;
        lookup_valid = 1'b1;
        lookup_pattern = 0; lookup_block = 4; lookup_beat = 0;
        lookup_tag = 32'h99;
        bank_words = '0;
        @(posedge clk_core); #1;
        if (dut_phase_load_ready || !dut_lookup_ready
                || !dut_output_valid || !dut_output_escape
                || dut_output_tag != 32'h99 || dut_protocol_error)
            $fatal(1, "M99 loaded simultaneous request did not prioritize lookup");
        loaded_priority_attacks++;

        // Legal metadata and all 128 entries are differential-checked to M85.
        reset_pair();
        build_legal_metadata();
        load_and_wait_for_parser(1'b0);
        for (int entry = 0; entry < 128; entry++) begin
            code = phase_metadata[entry*3 +: 3];
            drive_entry(entry, code);
        end
        @(negedge clk_core); lookup_valid = 1'b0;
        do @(posedge clk_core); while (dut_busy || ref_busy);

        // The three frozen M85 poison classes must match after serial audit.
        for (int attack = 0; attack < 3; attack++) begin
            reset_pair();
            build_legal_metadata();
            if (attack == 0) phase_metadata[0 +: 3] = 5;
            if (attack == 1) phase_metadata[384+4*13 +: 13] ^= 1;
            if (attack == 2) phase_metadata[384+15*13 +: 13] = 8191;
            load_and_wait_for_parser(1'b1);
            lookup_valid = 1'b1;
            lookup_pattern = 0; lookup_block = 1; lookup_beat = 0;
            @(posedge clk_core); #1;
            if (dut_lookup_ready || ref_lookup_ready
                    || !dut_protocol_error || !ref_protocol_error)
                $fatal(1, "M99 poison attack accepted attack=%0d", attack);
            lookup_valid = 1'b0;
            poison_attacks++;
        end

        if (legal_entries != 128 || legal_beats != 436
                || stall_cycles != 10 || poison_attacks != 3
                || early_lookup_attacks != 1 || simultaneous_attacks != 1
                || loaded_priority_attacks != 1 || parser_cycles != 640)
            $fatal(1, "M99 coverage mismatch entries=%0d beats=%0d stalls=%0d poison=%0d early=%0d simultaneous=%0d loaded_priority=%0d parser=%0d",
                   legal_entries, legal_beats, stall_cycles, poison_attacks,
                   early_lookup_attacks, simultaneous_attacks,
                   loaded_priority_attacks, parser_cycles);
        $display("PASS M99 M85-differential entries=128 beats=436 parser_cycles=640 stalls=10 poison_attacks=3 early_lookup_attacks=1 simultaneous_unloaded_attacks=1 simultaneous_loaded_priority_attacks=1");
        $finish;
    end
endmodule

`default_nettype wire
