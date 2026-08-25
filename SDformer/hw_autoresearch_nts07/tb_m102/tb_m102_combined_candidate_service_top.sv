`timescale 1ns/1ps
`default_nettype none

module tb_m102_combined_candidate_service_top;
    logic clk_core, rst_core;
    logic phase_load_valid, phase_load_ready;
    logic [591:0] phase_metadata;
    logic phase_loaded, metadata_error;
    logic service_valid, service_ready;
    logic [1:0] service_kind;
    logic [3:0] service_pattern, service_source;
    logic [2:0] service_block, service_beat;
    logic service_negate;
    logic [31:0] service_tag;
    logic [255:0] bank_words;
    logic [79:0] bank_row_addresses;
    logic bank_select_pwp;
    logic output_valid, output_ready;
    logic [31:0] output_tag;
    logic [1:0] output_kind;
    logic [3:0] output_width;
    logic output_escape;
    logic [1151:0] output_values;
    logic output_accept, protocol_error, busy;

    integer cycle_count, parser_cycles, accepted_beats, accepted_vectors;
    integer pwp_vectors, correction_vectors, fallback_vectors, stall_cycles;
    integer protocol_attacks, continuation_attacks, metadata_attacks;
    integer fault_stall_attacks;
    integer accepted_grace_holds;
    integer ii3_observations;
    integer previous_start_cycle, previous_vector_beats;

    m102_combined_candidate_service_top dut (.*);

    m102_combined_candidate_service_assertions dut_sva (
        .clk_core(clk_core), .rst_core(rst_core),
        .phase_load_valid(phase_load_valid),
        .phase_load_ready(phase_load_ready), .phase_loaded(phase_loaded),
        .metadata_error(metadata_error), .service_valid(service_valid),
        .service_ready(service_ready), .service_kind(service_kind),
        .service_beat(service_beat), .output_valid(output_valid),
        .output_ready(output_ready), .output_tag(output_tag),
        .output_kind(output_kind), .output_width(output_width),
        .output_escape(output_escape), .output_values(output_values),
        .output_accept(output_accept), .protocol_error(protocol_error),
        .busy(busy), .parse_active(dut.parse_active_q),
        .parse_index(dut.parse_index_q),
        .transaction_active(dut.transaction_active_q),
        .expected_beat(dut.expected_beat_q),
        .request_fault(dut.request_fault_q),
        .m82_beat_accept(dut.m82_beat_accept),
        .m82_output_valid(dut.m82_output_valid),
        .output_negate(dut.output_negate_q),
        .request_last(dut.request_last),
        .request_semantically_valid(dut.request_semantically_valid),
        .request_violation(dut.request_violation),
        .accepted_grace_match(dut.accepted_grace_match)
    );

    always #1.5 clk_core = ~clk_core;
    always @(posedge clk_core) cycle_count <= cycle_count + 1;

    initial begin
        #200000;
        $fatal(1, "M102 combined watchdog timeout cycle=%0d", cycle_count);
    end

    function automatic integer words_for_code(input integer code);
        case (code)
            0: words_for_code = 24;
            1: words_for_code = 27;
            2: words_for_code = 30;
            3: words_for_code = 33;
            default: words_for_code = 0;
        endcase
    endfunction

    function automatic integer beats_for_width(input integer width);
        case (width)
            8: beats_for_width = 3;
            9, 10: beats_for_width = 4;
            11: beats_for_width = 5;
            default: beats_for_width = 0;
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
                $fatal(1, "M102 generated illegal terminal=%0d", cursor);
        end
    endtask

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            phase_load_valid = 1'b0;
            service_valid = 1'b0;
            output_ready = 1'b1;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
        end
    endtask

    task automatic load_phase;
        integer waited;
        begin
            do @(posedge clk_core); while (!phase_load_ready);
            @(negedge clk_core);
            phase_load_valid = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            phase_load_valid = 1'b0;
            waited = 0;
            while (!phase_loaded) begin
                @(posedge clk_core);
                #0.1;
                waited++;
                if (waited > 128)
                    $fatal(1, "M102 parser exceeded 128 cycles");
            end
            if (waited != 128 || metadata_error)
                $fatal(1, "M102 parser mismatch waited=%0d poison=%0d",
                       waited, metadata_error);
            parser_cycles += waited;
        end
    endtask

    task automatic build_payload(
        input integer width,
        input integer seed,
        output logic [1279:0] payload,
        output logic [1151:0] expected
    );
        integer signed value;
        logic [11:0] extended;
        begin
            payload = '0;
            expected = '0;
            for (int lane = 0; lane < 96; lane++) begin
                if (width == 8) begin
                    case (lane % 8)
                        0: value = -128;
                        1: value = 127;
                        default: value = ((lane*17 + seed*11) % 255) - 127;
                    endcase
                end else begin
                    value = ((lane*29 + seed*7) % ((1 << width)-1))
                          - ((1 << (width-1))-1);
                end
                case (width)
                    8: begin
                        payload[lane*8 +: 8] = value[7:0];
                        extended = {{4{value[7]}}, value[7:0]};
                    end
                    9: begin
                        payload[lane*9 +: 9] = value[8:0];
                        extended = {{3{value[8]}}, value[8:0]};
                    end
                    10: begin
                        payload[lane*10 +: 10] = value[9:0];
                        extended = {{2{value[9]}}, value[9:0]};
                    end
                    default: begin
                        payload[lane*11 +: 11] = value[10:0];
                        extended = {value[10], value[10:0]};
                    end
                endcase
                expected[lane*12 +: 12] = extended;
            end
        end
    endtask

    task automatic negate_expected(
        input logic [1151:0] original,
        output logic [1151:0] negated
    );
        begin
            negated = '0;
            for (int lane = 0; lane < 96; lane++)
                negated[lane*12 +: 12] =
                    (~original[lane*12 +: 12]) + 1'b1;
        end
    endtask

    task automatic drive_vector(
        input logic [1:0] kind,
        input integer pattern,
        input integer source,
        input integer block,
        input integer width,
        input logic negate,
        input logic [31:0] tag,
        input logic start_immediate,
        input logic inject_stall,
        input logic hold_last_request
    );
        logic [1279:0] payload;
        logic [1151:0] expected, transformed, held;
        logic accepted_now;
        integer beats, base_word, logical_word, base_bank, base_row;
        integer accepted_cycle;
        begin
            build_payload(width, tag[7:0], payload, expected);
            if (negate)
                negate_expected(expected, transformed);
            else
                transformed = expected;
            beats = beats_for_width(width);
            if (kind == 0) begin
                base_word = phase_metadata[384+pattern*13 +: 13];
                for (int prior = 0; prior < block; prior++)
                    base_word += words_for_code(
                        phase_metadata[(pattern*8+prior)*3 +: 3]);
            end else begin
                base_word = (source*8 + block)*24;
            end

            for (int beat = 0; beat < beats; beat++) begin
                if (!(beat == 0 && start_immediate))
                    @(negedge clk_core);
                service_valid = 1'b1;
                service_kind = kind;
                service_pattern = pattern;
                service_source = source;
                service_block = block;
                service_beat = beat;
                service_negate = negate;
                service_tag = tag;
                logical_word = base_word + beat*8;
                base_bank = logical_word & 7;
                base_row = logical_word >> 3;
                bank_words = '0;
                for (int word = 0; word < 8; word++)
                    bank_words[((base_bank+word)&7)*32 +: 32] =
                        payload[(beat*8+word)*32 +: 32];
                if (hold_last_request && beat == beats-1)
                    output_ready = 1'b0;
                #0.1;
                for (int bank = 0; bank < 8; bank++)
                    if (bank_row_addresses[bank*10 +: 10]
                            !== base_row + (bank < base_bank))
                        $fatal(1, "M102 bank row mismatch kind=%0d beat=%0d bank=%0d got=%0d expected=%0d semantic=%0d fault=%0d active=%0d code=%0d m82ready=%0d",
                               kind, beat, bank,
                               bank_row_addresses[bank*10 +: 10],
                               base_row + (bank < base_bank),
                               dut.request_semantically_valid,
                               dut.request_fault_q, dut.transaction_active_q,
                               dut.selected_code, dut.m82_beat_ready);
                if (bank_select_pwp !== (kind == 0))
                    $fatal(1, "M102 bank select mismatch kind=%0d", kind);
                do begin
                    @(posedge clk_core);
                    accepted_now = service_ready;
                    if (accepted_now) begin
                        #0.01;
                        if (hold_last_request && beat == beats-1) begin
                            if (!dut.accepted_grace_match
                                    || dut.request_violation
                                    || protocol_error || service_ready
                                    || !output_valid || !dut.m82_output_valid)
                                $fatal(1, "M102 candidate accepted request grace failed after transfer grace=%0d violation=%0d fault=%0d ready=%0d output=%0d m82=%0d",
                                       dut.accepted_grace_match,
                                       dut.request_violation,
                                       protocol_error, service_ready,
                                       output_valid, dut.m82_output_valid);
                            @(posedge clk_core); #0.01;
                            if (!dut.accepted_grace_match
                                    || dut.request_violation
                                    || protocol_error || service_ready
                                    || !output_valid)
                                $fatal(1, "M102 candidate accepted request grace was not stable");
                            accepted_grace_holds++;
                        end
                        service_valid = 1'b0;
                        #0.99;
                    end else begin
                        #1;
                    end
                    if (protocol_error)
                        $fatal(1, "M102 legal vector fault kind=%0d beat=%0d request_fault=%0d poison=%0d m82_error=%0d semantic=%0d active=%0d expected=%0d",
                               kind, beat, dut.request_fault_q,
                               dut.phase_poison_q, dut.m82_protocol_error,
                               dut.request_semantically_valid,
                               dut.transaction_active_q, dut.expected_beat_q);
                end while (!accepted_now);
                if (beat == beats-1 && dut.transaction_active_q)
                    $fatal(1, "M102 transaction did not retire kind=%0d beat=%0d descriptor_beats=%0d request_last=%0d expected=%0d",
                           kind, beat, dut.descriptor_beats, dut.request_last,
                           dut.expected_beat_q);
                if (beat == 0) begin
                    accepted_cycle = cycle_count;
                    if (start_immediate && previous_start_cycle >= 0) begin
                        if (accepted_cycle - previous_start_cycle
                                != previous_vector_beats)
                            $fatal(1, "M102 shared-slot II mismatch got=%0d expected=%0d kind=%0d tag=%h",
                                   accepted_cycle - previous_start_cycle,
                                   previous_vector_beats, kind, tag);
                        ii3_observations++;
                    end
                    previous_start_cycle = accepted_cycle;
                end
                accepted_beats++;
            end
            previous_vector_beats = beats;
            accepted_vectors++;
            case (kind)
                0: pwp_vectors++;
                1: correction_vectors++;
                2: fallback_vectors++;
            endcase
            if (inject_stall) begin
                @(negedge clk_core);
                service_valid = 1'b0;
                output_ready = 1'b0;
                held = output_values;
                repeat (3) begin
                    @(posedge clk_core);
                    #1;
                    if (!output_valid || output_values !== held)
                        $fatal(1, "M102 output changed during stall tag=%h", tag);
                    stall_cycles++;
                end
                @(negedge clk_core);
                output_ready = 1'b1;
            end else begin
                @(negedge clk_core);
                service_valid = 1'b0;
            end
            #1;
            if (!output_valid || output_tag != tag || output_kind != kind
                    || output_width != width || output_escape
                    || output_values !== transformed)
                $fatal(1, "M102 output mismatch kind=%0d tag=%h width=%0d",
                       kind, tag, width);
        end
    endtask

    task automatic attack_invalid_first(
        input logic [1:0] kind,
        input integer pattern,
        input integer source,
        input integer block,
        input integer beat,
        input logic negate
    );
        begin
            reset_dut();
            build_legal_metadata();
            load_phase();
            @(negedge clk_core);
            service_valid = 1'b1;
            service_kind = kind;
            service_pattern = pattern;
            service_source = source;
            service_block = block;
            service_beat = beat;
            service_negate = negate;
            service_tag = 32'hdead0000 | protocol_attacks;
            bank_words = '0;
            #0.1;
            if (service_ready)
                $fatal(1, "M102 invalid first request was ready kind=%0d pattern=%0d block=%0d beat=%0d negate=%0d",
                       kind, pattern, block, beat, negate);
            @(posedge clk_core); #1;
            if (!protocol_error || service_ready)
                $fatal(1, "M102 invalid first request was not fail-closed kind=%0d pattern=%0d block=%0d beat=%0d negate=%0d",
                       kind, pattern, block, beat, negate);
            protocol_attacks++;
            @(negedge clk_core);
            service_valid = 1'b0;
        end
    endtask

    task automatic attack_continuation_mutation(input integer mutation);
        logic accepted_first;
        begin
            reset_dut();
            build_legal_metadata();
            load_phase();
            @(negedge clk_core);
            service_valid = 1'b1;
            service_kind = 2'd1;
            service_pattern = 0;
            service_source = 3;
            service_block = 1;
            service_beat = 0;
            service_negate = 1'b0;
            service_tag = 32'hbad00000 | mutation;
            bank_words = '0;
            #0.1;
            accepted_first = service_ready;
            @(posedge clk_core);
            if (accepted_first) begin
                #0.01;
                service_valid = 1'b0;
                #0.99;
            end else begin
                #1;
            end
            if (!accepted_first || protocol_error)
                $fatal(1, "M102 mutation%0d first beat was not accepted",
                       mutation);
            @(negedge clk_core);
            service_valid = 1'b1;
            service_beat = 1;
            case (mutation)
                0: service_source = 4;
                1: service_pattern = 1;
                2: service_block = 2;
                3: service_kind = 2'd2;
                4: service_tag = service_tag + 1'b1;
                5: service_negate = 1'b1;
                default: $fatal(1, "M102 unknown continuation mutation");
            endcase
            #0.1;
            if (service_ready)
                $fatal(1, "M102 mutation%0d continuation was ready", mutation);
            @(posedge clk_core); #1;
            if (service_ready || !protocol_error || output_valid || output_accept)
                $fatal(1, "M102 mutation%0d was not fail-closed", mutation);
            protocol_attacks++;
            continuation_attacks++;
            @(negedge clk_core);
            service_valid = 1'b0;
            repeat (2) @(posedge clk_core);
            if (!protocol_error || service_ready || output_valid || output_accept)
                $fatal(1, "M102 mutation%0d fault was not sticky", mutation);
        end
    endtask

    task automatic attack_stalled_output_with_invalid_request;
        logic [1279:0] payload;
        logic [1151:0] unused_expected;
        integer base_word, logical_word, base_bank;
        begin
            reset_dut();
            build_legal_metadata();
            load_phase();
            build_payload(8, 32'h5a, payload, unused_expected);
            output_ready = 1'b0;
            base_word = (2*8 + 3)*24;
            for (int beat = 0; beat < 3; beat++) begin
                @(negedge clk_core);
                service_valid = 1'b1;
                service_kind = 2'd1;
                service_pattern = 0;
                service_source = 2;
                service_block = 3;
                service_beat = beat;
                service_negate = 1'b0;
                service_tag = 32'hf0170001;
                logical_word = base_word + beat*8;
                base_bank = logical_word & 7;
                bank_words = '0;
                for (int word = 0; word < 8; word++)
                    bank_words[((base_bank+word)&7)*32 +: 32] =
                        payload[(beat*8+word)*32 +: 32];
                do @(posedge clk_core); while (!service_ready);
                #0.01;
                service_valid = 1'b0;
            end
            @(negedge clk_core);
            service_valid = 1'b0;
            #0.1;
            if (!output_valid || !dut.m82_output_valid || protocol_error)
                $fatal(1, "M102 failed to create a clean stalled old output");

            // A reserved new service kind and ready release happen in the
            // same cycle.  The old M82 result must be quarantined before the
            // registering edge, then remain buffered under sticky fault.
            service_valid = 1'b1;
            service_kind = 2'd3;
            service_beat = 0;
            service_tag = 32'hf0170002;
            output_ready = 1'b1;
            #0.1;
            if (service_ready || !protocol_error || output_valid
                    || output_accept || !dut.m82_output_valid)
                $fatal(1, "M102 same-cycle release was not quarantined before edge ready=%0d fault=%0d valid=%0d accept=%0d m82=%0d",
                       service_ready, protocol_error, output_valid,
                       output_accept, dut.m82_output_valid);
            @(posedge clk_core); #1;
            if (!protocol_error || output_valid || output_accept
                    || !dut.m82_output_valid)
                $fatal(1, "M102 buffered output escaped top fault valid=%0d accept=%0d m82_valid=%0d fault=%0d",
                       output_valid, output_accept, dut.m82_output_valid,
                       protocol_error);
            @(negedge clk_core);
            service_valid = 1'b0;
            repeat (2) begin
                @(posedge clk_core); #1;
                if (!protocol_error || output_valid || output_accept
                        || !dut.m82_output_valid)
                    $fatal(1, "M102 fault quarantine was not sticky");
            end

            // A phase reload is not a recovery mechanism.  Only reset may
            // clear the request fault and release the quarantined output.
            @(negedge clk_core);
            phase_load_valid = 1'b1;
            #0.1;
            if (phase_load_ready)
                $fatal(1, "M102 sticky fault admitted a phase reload");
            @(posedge clk_core); #1;
            if (!protocol_error || phase_load_ready || output_valid
                    || output_accept || !dut.request_fault_q
                    || !dut.m82_output_valid)
                $fatal(1, "M102 phase reload cleared sticky fault");
            @(negedge clk_core);
            phase_load_valid = 1'b0;
            protocol_attacks++;
            fault_stall_attacks++;
        end
    endtask

    task automatic attack_poisoned_metadata;
        integer waited;
        begin
            reset_dut();
            build_legal_metadata();
            phase_metadata[0 +: 3] = 3'd5;
            do @(posedge clk_core); while (!phase_load_ready);
            @(negedge clk_core);
            phase_load_valid = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            phase_load_valid = 1'b0;
            waited = 0;
            while (!phase_loaded) begin
                @(posedge clk_core); #0.1;
                waited++;
                if (waited > 128)
                    $fatal(1, "M102 poisoned parser exceeded 128 cycles");
            end
            parser_cycles += waited;
            if (waited != 128 || !metadata_error || !protocol_error)
                $fatal(1, "M102 poisoned metadata admission mismatch waited=%0d metadata_error=%0d protocol_error=%0d",
                       waited, metadata_error, protocol_error);
            @(negedge clk_core);
            service_valid = 1'b1;
            service_kind = 2'd0;
            service_pattern = 0;
            service_source = 0;
            service_block = 0;
            service_beat = 0;
            service_negate = 1'b0;
            service_tag = 32'hbad50001;
            bank_words = '0;
            #0.1;
            if (service_ready || output_valid || output_accept)
                $fatal(1, "M102 poisoned metadata did not block service/output");
            @(posedge clk_core); #1;
            if (!protocol_error || output_valid || output_accept)
                $fatal(1, "M102 poisoned request did not enter fail-closed state");
            metadata_attacks++;
            @(negedge clk_core);
            service_valid = 1'b0;
        end
    endtask

    initial begin
        clk_core = 0;
        rst_core = 1;
        phase_load_valid = 0;
        phase_metadata = '0;
        service_valid = 0;
        service_kind = 0;
        service_pattern = 0;
        service_source = 0;
        service_block = 0;
        service_beat = 0;
        service_negate = 0;
        service_tag = 0;
        bank_words = 0;
        output_ready = 1;
        cycle_count = 0;
        parser_cycles = 0;
        accepted_beats = 0;
        accepted_vectors = 0;
        pwp_vectors = 0;
        correction_vectors = 0;
        fallback_vectors = 0;
        stall_cycles = 0;
        protocol_attacks = 0;
        continuation_attacks = 0;
        metadata_attacks = 0;
        fault_stall_attacks = 0;
        accepted_grace_holds = 0;
        ii3_observations = 0;
        previous_start_cycle = -1;
        previous_vector_beats = 0;

        reset_dut();
        build_legal_metadata();
        load_phase();

        // All service kinds and PWP widths share one slot without mode-switch
        // bubbles.  The final width11 result is held under backpressure.
        drive_vector(2'd0, 0, 0, 0, 8, 1'b0, 32'h0a00, 1'b0, 1'b0, 1'b0);
        drive_vector(2'd1, 0, 0, 0, 8, 1'b0, 32'hc000, 1'b1, 1'b0, 1'b0);
        drive_vector(2'd0, 0, 0, 1, 9, 1'b0, 32'h0a01, 1'b1, 1'b0, 1'b0);
        drive_vector(2'd1, 0, 15, 1, 8, 1'b1, 32'hc001, 1'b1, 1'b0, 1'b0);
        drive_vector(2'd2, 0, 0, 4, 8, 1'b0, 32'hfa10, 1'b1, 1'b0, 1'b0);
        drive_vector(2'd2, 1, 15, 1, 8, 1'b0, 32'hfa11, 1'b1, 1'b0, 1'b0);
        drive_vector(2'd0, 0, 0, 2, 10, 1'b0, 32'h0a02, 1'b1, 1'b0, 1'b0);
        drive_vector(2'd0, 0, 0, 3, 11, 1'b0, 32'h0a03, 1'b1, 1'b1, 1'b1);

        // Every continuation-owned identity field is independently attacked.
        for (int mutation = 0; mutation < 6; mutation++)
            attack_continuation_mutation(mutation);

        // Four distinct invalid first-request classes plus an illegal mode.
        attack_invalid_first(2'd0, 0, 0, 4, 0, 1'b0); // PWP on code4.
        attack_invalid_first(2'd2, 0, 0, 0, 0, 1'b0); // fallback on code0.
        attack_invalid_first(2'd0, 0, 0, 0, 0, 1'b1); // negate on PWP.
        attack_invalid_first(2'd1, 0, 0, 0, 1, 1'b0); // orphan beat1.
        attack_invalid_first(2'd3, 0, 0, 0, 0, 1'b0); // reserved kind.

        attack_stalled_output_with_invalid_request();
        attack_poisoned_metadata();

        if (parser_cycles != 1792 || accepted_vectors != 8
                || accepted_beats != 28 || pwp_vectors != 4
                || correction_vectors != 2 || fallback_vectors != 2
                || stall_cycles != 3 || protocol_attacks != 12
                || continuation_attacks != 6
                || metadata_attacks != 1 || fault_stall_attacks != 1
                || accepted_grace_holds != 1
                || ii3_observations != 7)
            $fatal(1, "M102 coverage mismatch parser=%0d vectors=%0d beats=%0d pwp=%0d correction=%0d fallback=%0d stalls=%0d attacks=%0d continuation=%0d metadata=%0d fault_stall=%0d grace=%0d ii3=%0d",
                   parser_cycles, accepted_vectors, accepted_beats, pwp_vectors,
                   correction_vectors, fallback_vectors, stall_cycles,
                   protocol_attacks, continuation_attacks, metadata_attacks,
                   fault_stall_attacks, accepted_grace_holds,
                   ii3_observations);
        $display("PASS M102 combined-r4 parser_cycles=1792 vectors=8 beats=28 pwp=4 correction=2 fallback=2 stalls=3 protocol_attacks=12 continuation_attacks=6 metadata_attacks=1 same_cycle_release_attacks=1 phase_reload_attacks=1 accepted_grace_holds=1 shared_slot_ii_checks=%0d",
                 ii3_observations);
        $finish;
    end
endmodule

`default_nettype wire
