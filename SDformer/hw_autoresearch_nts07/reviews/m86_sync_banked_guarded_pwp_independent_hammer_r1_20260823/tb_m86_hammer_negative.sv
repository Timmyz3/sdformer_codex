`timescale 1ns/1ps
`default_nettype none

// Independent directed flow-control and loader attack bench.  It does not
// import the production TB or any of its expected-value helpers.
module tb_m86_hammer_negative;
    localparam int ROWS = 460;
    localparam int QDEPTH = 32;

    logic clk_core, rst_core;
    logic payload_load_valid, payload_load_ready, payload_load_accept;
    logic [9:0] payload_load_row;
    logic [255:0] payload_load_words;
    logic phase_load_valid, phase_load_ready, phase_loaded, metadata_error;
    logic [591:0] phase_metadata;
    logic descriptor_valid, descriptor_ready, descriptor_accept;
    logic [3:0] descriptor_pattern;
    logic [2:0] descriptor_block;
    logic [31:0] descriptor_tag;
    logic output_valid, output_ready, output_escape, output_accept;
    logic [31:0] output_tag;
    logic [3:0] output_width;
    logic [1151:0] output_values;
    logic protocol_error, busy, bank_read_issue, bank_response_enqueue;
    logic [2:0] bank_read_beat, response_fifo_level;

    integer entry_code [0:127];
    integer entry_start [0:127];
    logic [31:0] expected_tag_q [0:QDEPTH-1];
    logic [3:0] expected_width_q [0:QDEPTH-1];
    logic expected_escape_q [0:QDEPTH-1];
    logic [1151:0] expected_values_q [0:QDEPTH-1];
    integer expected_rptr, expected_wptr, expected_count, output_count;
    integer cycle_count, read_issues, responses, prev_issue;
    integer simultaneous_push_pop, full_hold_cycles, collision_hold_cycles;
    integer missing_row_checks, oob_attacks, duplicate_attacks;

    sync_banked_guarded_pwp_frontend dut (.*);
    sync_banked_guarded_pwp_frontend_assertions sva (.*);

    always #1.5 clk_core = ~clk_core;
    initial begin
        #200000;
        $fatal(1, "M86 independent hammer watchdog");
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

    function automatic logic [31:0] payload_word(input integer word_index);
        payload_word = 32'h80ff_7f00
                     ^ (32'h0102_0408 * word_index);
    endfunction

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            rst_core = 1;
            payload_load_valid = 0;
            phase_load_valid = 0;
            descriptor_valid = 0;
            output_ready = 1;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 0;
            prev_issue = 0;
        end
    endtask

    task automatic build_metadata(input integer first_code);
        integer cursor;
        begin
            phase_metadata = '0;
            for (int entry = 0; entry < 128; entry++)
                entry_code[entry] = (entry == 0) ? first_code : 0;
            cursor = 0;
            for (int pattern = 0; pattern < 16; pattern++) begin
                phase_metadata[384 + pattern*13 +: 13] = cursor[12:0];
                for (int block = 0; block < 8; block++) begin
                    integer entry;
                    entry = pattern*8 + block;
                    entry_start[entry] = cursor;
                    phase_metadata[entry*3 +: 3] = entry_code[entry][2:0];
                    cursor = cursor + words_for_code(entry_code[entry]);
                end
            end
        end
    endtask

    task automatic drive_row(input integer row_value);
        begin
            @(negedge clk_core);
            payload_load_row = row_value[9:0];
            for (int bank = 0; bank < 8; bank++)
                payload_load_words[bank*32 +: 32] =
                    payload_word(row_value*8 + bank);
            payload_load_valid = 1;
            do @(posedge clk_core); while (!payload_load_ready);
            #0.1 payload_load_valid = 0;
        end
    endtask

    task automatic load_rows(input integer last_exclusive);
        begin
            for (int row = 0; row < last_exclusive; row++)
                drive_row(row);
        end
    endtask

    task automatic commit_phase;
        begin
            @(negedge clk_core);
            phase_load_valid = 1;
            do @(posedge clk_core); while (!phase_load_ready);
            #0.1 phase_load_valid = 0;
            #1;
            if (!phase_loaded || metadata_error || protocol_error)
                $fatal(1, "M86 hammer legal phase commit failed");
        end
    endtask

    task automatic expected_push_regular(
        input integer entry, input logic [31:0] tag
    );
        logic [1279:0] packed_bits;
        logic [1151:0] values;
        integer width;
        begin
            width = 8 + entry_code[entry];
            packed_bits = '0;
            values = '0;
            for (int word = 0; word < words_for_code(entry_code[entry]); word++)
                packed_bits[word*32 +: 32] =
                    payload_word(entry_start[entry] + word);
            for (int lane = 0; lane < 96; lane++) begin
                case (width)
                    8: values[lane*12 +: 12] = {
                        {4{packed_bits[lane*8+7]}},
                        packed_bits[lane*8 +: 8]};
                    9: values[lane*12 +: 12] = {
                        {3{packed_bits[lane*9+8]}},
                        packed_bits[lane*9 +: 9]};
                    default: $fatal(1, "M86 hammer unexpected width");
                endcase
            end
            expected_tag_q[expected_wptr] = tag;
            expected_width_q[expected_wptr] = width[3:0];
            expected_escape_q[expected_wptr] = 0;
            expected_values_q[expected_wptr] = values;
            expected_wptr = (expected_wptr + 1) % QDEPTH;
            expected_count++;
        end
    endtask

    task automatic expected_push_escape(input logic [31:0] tag);
        begin
            expected_tag_q[expected_wptr] = tag;
            expected_width_q[expected_wptr] = 12;
            expected_escape_q[expected_wptr] = 1;
            expected_values_q[expected_wptr] = '0;
            expected_wptr = (expected_wptr + 1) % QDEPTH;
            expected_count++;
        end
    endtask

    task automatic drive_descriptor(
        input integer entry, input logic [31:0] tag
    );
        begin
            @(negedge clk_core);
            descriptor_pattern = entry / 8;
            descriptor_block = entry % 8;
            descriptor_tag = tag;
            descriptor_valid = 1;
            do @(posedge clk_core); while (!descriptor_ready);
            #0.1 descriptor_valid = 0;
        end
    endtask

    always @(posedge clk_core) begin
        if (rst_core) begin
            prev_issue = 0;
        end else begin
            cycle_count++;
            if (bank_response_enqueue !== prev_issue[0])
                $fatal(1, "M86 hammer response latency mismatch cycle=%0d issue_prev=%0d response=%0d",
                       cycle_count, prev_issue, bank_response_enqueue);
            prev_issue = bank_read_issue;
            if (bank_read_issue) read_issues++;
            if (bank_response_enqueue) responses++;
            if (dut.fifo_push && dut.fifo_pop) simultaneous_push_pop++;
            if (response_fifo_level > 4)
                $fatal(1, "M86 hammer FIFO overflow level=%0d",
                       response_fifo_level);
            if (response_fifo_level == 4 && !dut.fifo_pop
                    && bank_read_issue)
                $fatal(1, "M86 hammer issued into non-popping full FIFO");
            if (output_accept) begin
                if (expected_count == 0)
                    $fatal(1, "M86 hammer unexpected output");
                if (output_tag != expected_tag_q[expected_rptr]
                        || output_width != expected_width_q[expected_rptr]
                        || output_escape != expected_escape_q[expected_rptr]
                        || output_values !== expected_values_q[expected_rptr])
                    $fatal(1, "M86 hammer output mismatch tag=%h expected=%h",
                           output_tag, expected_tag_q[expected_rptr]);
                expected_rptr = (expected_rptr + 1) % QDEPTH;
                expected_count--;
                output_count++;
            end
        end
    end

    initial begin
        logic [31:0] held_tag;
        logic [1151:0] held_values;
        clk_core = 0;
        rst_core = 1;
        payload_load_valid = 0;
        payload_load_row = 0;
        payload_load_words = 0;
        phase_load_valid = 0;
        phase_metadata = 0;
        descriptor_valid = 0;
        descriptor_pattern = 0;
        descriptor_block = 0;
        descriptor_tag = 0;
        output_ready = 1;
        expected_rptr = 0;
        expected_wptr = 0;
        expected_count = 0;
        output_count = 0;
        cycle_count = 0;
        read_issues = 0;
        responses = 0;
        prev_issue = 0;
        simultaneous_push_pop = 0;
        full_hold_cycles = 0;
        collision_hold_cycles = 0;
        missing_row_checks = 0;
        oob_attacks = 0;
        duplicate_attacks = 0;
        repeat (4) @(posedge clk_core);
        @(negedge clk_core); rst_core = 0;

        // The phase cannot commit with one of the 460 rows absent.
        build_metadata(1);
        load_rows(ROWS-1);
        @(negedge clk_core); phase_load_valid = 1;
        @(posedge clk_core); #1;
        if (phase_load_ready || phase_loaded || protocol_error)
            $fatal(1, "M86 hammer missing-row phase was not blocked");
        missing_row_checks++;
        @(negedge clk_core); phase_load_valid = 0;
        drive_row(ROWS-1);
        commit_phase();

        // Both independent ready channels suppress themselves if the payload
        // producer and descriptor producer assert valid together.  Record the
        // silent deadlock rather than blessing it as a protocol feature.
        expected_push_regular(1, 32'h8600_0001);
        @(negedge clk_core);
        descriptor_pattern = 0;
        descriptor_block = 1;
        descriptor_tag = 32'h8600_0001;
        descriptor_valid = 1;
        payload_load_row = 0;
        payload_load_words = '0;
        payload_load_valid = 1;
        repeat (4) begin
            @(posedge clk_core); #1;
            if (descriptor_ready || payload_load_ready
                    || descriptor_accept || payload_load_accept
                    || protocol_error || busy)
                $fatal(1, "M86 hammer simultaneous-valid collision behavior drift");
            collision_hold_cycles++;
        end
        @(negedge clk_core); payload_load_valid = 0;
        do @(posedge clk_core); while (!descriptor_ready);
        #0.1 descriptor_valid = 0;
        while (expected_count != 0) @(posedge clk_core);

        // Fill and hold the four-entry response FIFO behind a stalled output,
        // then release it and require ordered, bit-exact drain plus concurrent
        // push/pop activity.
        output_ready = 0;
        fork
            begin
                for (int entry = 2; entry < 12; entry++) begin
                    expected_push_regular(entry, 32'h8600_0100 + entry);
                    drive_descriptor(entry, 32'h8600_0100 + entry);
                end
            end
            begin
                wait (response_fifo_level == 4 && output_valid);
                held_tag = output_tag;
                held_values = output_values;
                repeat (6) begin
                    @(posedge clk_core); #1;
                    if (response_fifo_level != 4 || !output_valid
                            || output_tag != held_tag
                            || output_values != held_values)
                        $fatal(1, "M86 hammer full-FIFO hold instability");
                    full_hold_cycles++;
                end
                @(negedge clk_core); output_ready = 1;
            end
        join
        while (busy || expected_count != 0) @(posedge clk_core);
        if (protocol_error || simultaneous_push_pop == 0)
            $fatal(1, "M86 hammer legal FIFO recovery failed simultaneous=%0d",
                   simultaneous_push_pop);

        // A real escape control still traverses the registered read-response
        // stage and produces a zero width-12 control vector.
        reset_dut();
        build_metadata(4);
        load_rows(ROWS);
        commit_phase();
        expected_push_escape(32'h8600_00e5);
        drive_descriptor(0, 32'h8600_00e5);
        while (expected_count != 0 || busy) @(posedge clk_core);
        if (protocol_error)
            $fatal(1, "M86 hammer legal escape faulted");

        // Out-of-range and duplicate loader requests are independent attacks.
        reset_dut();
        @(negedge clk_core);
        payload_load_row = 10'd460;
        payload_load_words = '0;
        payload_load_valid = 1;
        @(posedge clk_core); #1;
        if (payload_load_ready || payload_load_accept || !protocol_error
                || descriptor_ready)
            $fatal(1, "M86 hammer OOB row did not fail closed");
        oob_attacks++;
        @(negedge clk_core); payload_load_valid = 0;

        reset_dut();
        drive_row(0);
        drive_row(0);
        #1;
        if (!protocol_error || descriptor_ready)
            $fatal(1, "M86 hammer duplicate row did not fail closed");
        duplicate_attacks++;

        if (missing_row_checks != 1 || collision_hold_cycles != 4
                || full_hold_cycles != 6 || simultaneous_push_pop == 0
                || oob_attacks != 1 || duplicate_attacks != 1
                || read_issues != responses || output_count != 12)
            $fatal(1, "M86 hammer coverage mismatch missing=%0d collision=%0d full=%0d pushpop=%0d oob=%0d dup=%0d issue=%0d response=%0d output=%0d",
                   missing_row_checks, collision_hold_cycles,
                   full_hold_cycles, simultaneous_push_pop, oob_attacks,
                   duplicate_attacks, read_issues, responses, output_count);
        $display("PASS M86 independent hammer one_cycle_response=1 missing_row=1 oob_row=1 duplicate_row=1 fifo_full_hold=6 simultaneous_push_pop=%0d bit_exact_outputs=12 simultaneous_valid_deadlock_cycles=4",
                 simultaneous_push_pop);
        $finish;
    end
endmodule

`default_nettype wire
