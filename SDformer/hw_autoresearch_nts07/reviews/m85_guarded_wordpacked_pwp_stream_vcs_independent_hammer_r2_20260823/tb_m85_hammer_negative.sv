`timescale 1ns/1ps
`default_nettype none

// Independent directed negative/backpressure checks.  This testbench does not
// import the production TB or its expected-value helpers.
module tb_m85_hammer_negative;
    logic clk_core, rst_core;
    logic phase_load_valid, phase_load_ready, phase_loaded, metadata_error;
    logic [591:0] phase_metadata;
    logic lookup_valid, lookup_ready;
    logic [3:0] lookup_pattern;
    logic [2:0] lookup_block, lookup_beat;
    logic [31:0] lookup_tag;
    logic [255:0] bank_words;
    logic [79:0] bank_row_addresses;
    logic output_valid, output_ready, output_escape, output_accept;
    logic [31:0] output_tag;
    logic [3:0] output_width;
    logic [1151:0] output_values;
    logic protocol_error, busy;
    integer poison_checks, invalid_lookup_checks, backpressure_checks;

    guarded_wordpacked_pwp_stream dut (.*);
    guarded_wordpacked_pwp_stream_assertions sva (.*);

    always #1.5 clk_core = ~clk_core;
    initial begin
        #100000;
        $fatal(1, "M85 independent negative watchdog");
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

    task automatic reset_dut;
        begin
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

    task automatic build_uniform_metadata(input integer code);
        integer cursor;
        begin
            phase_metadata = '0;
            cursor = 0;
            for (int pattern = 0; pattern < 16; pattern++) begin
                phase_metadata[384 + pattern*13 +: 13] = cursor[12:0];
                for (int block = 0; block < 8; block++) begin
                    phase_metadata[(pattern*8+block)*3 +: 3] = code[2:0];
                    cursor += words_for_code(code);
                end
            end
        end
    endtask

    task automatic load_and_check(input logic expect_error);
        begin
            if (!phase_load_ready)
                $fatal(1, "M85 hammer phase loader unexpectedly busy");
            @(negedge clk_core);
            phase_load_valid = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            phase_load_valid = 1'b0;
            #1;
            if (!phase_loaded || metadata_error !== expect_error
                    || protocol_error !== expect_error)
                $fatal(1, "M85 hammer load result mismatch expected_error=%0d loaded=%0d metadata_error=%0d protocol_error=%0d",
                       expect_error, phase_loaded, metadata_error, protocol_error);
            if (expect_error) poison_checks++;
        end
    endtask

    task automatic expect_invalid_lookup(
        input integer pattern, input integer block, input integer beat
    );
        begin
            @(negedge clk_core);
            lookup_valid = 1'b1;
            lookup_pattern = pattern[3:0];
            lookup_block = block[2:0];
            lookup_beat = beat[2:0];
            lookup_tag = 32'hbad00000 | invalid_lookup_checks;
            bank_words = '0;
            #1;
            if (lookup_ready)
                $fatal(1, "M85 hammer invalid lookup ready before edge");
            @(posedge clk_core);
            #1;
            if (!protocol_error || lookup_ready)
                $fatal(1, "M85 hammer invalid lookup did not fail closed");
            lookup_valid = 1'b0;
            invalid_lookup_checks++;
        end
    endtask

    task automatic drive_beat(
        input integer pattern, input integer block, input integer beat,
        input integer tag, input logic [255:0] data
    );
        begin
            @(negedge clk_core);
            lookup_valid = 1'b1;
            lookup_pattern = pattern[3:0];
            lookup_block = block[2:0];
            lookup_beat = beat[2:0];
            lookup_tag = beat == 0 ? tag : 0;
            bank_words = data;
            #1;
            if (!lookup_ready)
                $fatal(1, "M85 hammer legal beat unexpectedly stalled beat=%0d", beat);
            @(posedge clk_core);
            #1;
            if (protocol_error)
                $fatal(1, "M85 hammer legal beat faulted beat=%0d", beat);
        end
    endtask

    initial begin
        logic [31:0] held_tag;
        logic [3:0] held_width;
        logic [1151:0] held_values;
        clk_core = 0;
        rst_core = 1;
        phase_load_valid = 0;
        phase_metadata = '0;
        lookup_valid = 0;
        lookup_pattern = 0;
        lookup_block = 0;
        lookup_beat = 0;
        lookup_tag = 0;
        bank_words = '0;
        output_ready = 1;
        poison_checks = 0;
        invalid_lookup_checks = 0;
        backpressure_checks = 0;
        repeat (4) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 0;

        // Lookup before any metadata load must poison the interface.
        expect_invalid_lookup(0, 0, 0);

        // Reserved codes 5, 6, and 7, including a predecessor position.
        for (int reserved = 5; reserved <= 7; reserved++) begin
            reset_dut();
            build_uniform_metadata(0);
            phase_metadata[0 +: 3] = reserved[2:0];
            load_and_check(1'b1);
            if (lookup_ready) $fatal(1, "M85 reserved metadata enabled lookup");
        end

        // Inconsistent canonical base, capacity overflow, and zero terminal.
        reset_dut();
        build_uniform_metadata(0);
        phase_metadata[384 + 4*13 +: 13] ^= 13'd1;
        load_and_check(1'b1);

        reset_dut();
        build_uniform_metadata(3);
        load_and_check(1'b1);

        reset_dut();
        build_uniform_metadata(4);
        load_and_check(1'b1);

        // Legal metadata accepts and exposes exact cross-row bank addresses.
        reset_dut();
        build_uniform_metadata(1);
        load_and_check(1'b0);
        @(negedge clk_core);
        lookup_valid = 1'b1;
        lookup_pattern = 0;
        lookup_block = 1;
        lookup_beat = 0;
        lookup_tag = 32'h1234;
        bank_words = '0;
        #1;
        if (!lookup_ready)
            $fatal(1, "M85 cross-row probe not ready");
        for (int bank = 0; bank < 8; bank++) begin
            if (bank_row_addresses[bank*10 +: 10] != (bank < 3 ? 4 : 3))
                $fatal(1, "M85 bank row mismatch bank=%0d got=%0d", bank,
                       bank_row_addresses[bank*10 +: 10]);
        end
        @(posedge clk_core);

        // Regular beat beyond the descriptor and escape beat other than zero.
        reset_dut();
        build_uniform_metadata(0);
        load_and_check(1'b0);
        expect_invalid_lookup(0, 0, 3);

        reset_dut();
        build_uniform_metadata(0);
        phase_metadata[0 +: 3] = 3'd4;
        for (int pattern = 0, cursor = 0; pattern < 16; pattern++) begin
            phase_metadata[384 + pattern*13 +: 13] = cursor[12:0];
            for (int block = 0; block < 8; block++) begin
                integer selected;
                selected = pattern == 0 && block == 0 ? 4 : 0;
                cursor += words_for_code(selected);
            end
        end
        load_and_check(1'b0);
        expect_invalid_lookup(0, 0, 1);

        // One directed output stall: payload and tag must remain stable, and
        // the next legal start is blocked until the held output retires.
        reset_dut();
        build_uniform_metadata(0);
        load_and_check(1'b0);
        output_ready = 1'b1;
        drive_beat(0, 0, 0, 32'h85, 256'h0123);
        drive_beat(0, 0, 1, 0, 256'h4567);
        output_ready = 1'b0;
        drive_beat(0, 0, 2, 0, 256'h89ab);
        if (!output_valid || output_tag != 32'h85 || output_width != 8)
            $fatal(1, "M85 hammer held output missing");
        held_tag = output_tag;
        held_width = output_width;
        held_values = output_values;
        @(negedge clk_core);
        lookup_valid = 1'b1;
        lookup_pattern = 0;
        lookup_block = 1;
        lookup_beat = 0;
        lookup_tag = 32'h86;
        bank_words = 256'hcdef;
        repeat (4) begin
            @(posedge clk_core);
            #1;
            if (lookup_ready || !output_valid || output_tag != held_tag
                    || output_width != held_width || output_values != held_values)
                $fatal(1, "M85 hammer output stall instability");
            backpressure_checks++;
        end
        @(negedge clk_core);
        output_ready = 1'b1;
        #1;
        if (!lookup_ready)
            $fatal(1, "M85 hammer elastic retirement did not reopen input");
        @(posedge clk_core);
        #1;
        if (protocol_error)
            $fatal(1, "M85 hammer elastic retirement faulted");
        lookup_valid = 1'b0;

        if (poison_checks != 6 || invalid_lookup_checks != 3
                || backpressure_checks != 4)
            $fatal(1, "M85 hammer coverage mismatch poison=%0d invalid=%0d bp=%0d",
                   poison_checks, invalid_lookup_checks, backpressure_checks);
        $display("PASS M85 independent negative poison=6 invalid_lookup=3 cross_row_address=8 held_output_cycles=4 elastic_overlap=1");
        $finish;
    end
endmodule

`default_nettype wire
