`timescale 1ns/1ps
`default_nettype none

module tb_m139_epoch_safe_fallthrough_tagged_16bank_response_bridge;
    logic clk_core = 1'b0;
    logic rst_core = 1'b1;
    always #1.5 clk_core = ~clk_core;

    logic request_valid, request_ready, request_start, request_last;
    logic [11:0] logical_base_word;
    logic [3:0] request_width;
    logic [31:0] request_tag;
    logic request_accept;
    logic macro_flush_req, macro_flush_ack;
    logic macro_request_valid;
    logic [127:0] macro_bank_row_addresses;
    logic [15:0] macro_request_token;
    logic macro_response_valid;
    logic [15:0] macro_response_token;
    logic [511:0] macro_bank_words;
    logic response_valid, response_ready;
    logic [511:0] response_logical_words;
    logic response_start, response_last;
    logic [3:0] response_width;
    logic [31:0] response_tag;
    logic [15:0] response_token;
    logic response_accept, protocol_error, recovery_active;
    logic pending_response;
    logic [1:0] buffered_responses;
    logic busy;

    m139_epoch_safe_fallthrough_tagged_16bank_response_bridge dut (.*);

    typedef struct packed {
        logic [511:0] words;
        logic start_bit;
        logic last_bit;
        logic [3:0] width;
        logic [31:0] tag_bits;
        logic [15:0] token;
    } expected_t;
    expected_t expected_q[$];
    expected_t expected_item;

    localparam int MACRO_QUIET = 0;
    localparam int MACRO_AUTO = 1;
    localparam int MACRO_MANUAL = 2;
    localparam int MACRO_WRONG = 3;
    integer macro_mode;
    logic manual_response_valid;
    logic [15:0] manual_response_token;
    logic [511:0] manual_bank_words;

    logic scoreboard_enable, stall_enable, force_ready;
    integer wall_cycle, positive_requests, positive_outputs;
    integer logical_word_checks, ii1_checks, last_accept_cycle;
    integer flush_handshakes, reset_recoveries, initial_high_ack_tests;
    integer stale_drain_attacks, completion_collision_attacks;
    integer post_completion_attacks, wrong_token_attacks;
    integer reset_pending_attacks, reset_skid_attacks;
    integer stall_cycles, skid_cycles, wrap_requests, wrap_crossings;
    logic [15:0] expected_request_token;

    function automatic logic [31:0] word_value(input logic [12:0] index);
        word_value = 32'h39e1_0000 ^ {19'd0, index}
                   ^ ({19'd0, index} << 9);
    endfunction

    function automatic logic [511:0] logical_vector(input logic [11:0] base);
        logic [511:0] value;
        value = '0;
        for (int word = 0; word < 16; word++)
            value[word*32 +: 32] = word_value({1'b0, base} + word);
        return value;
    endfunction

    function automatic logic [511:0] physical_bank_vector(
        input logic [11:0] base
    );
        logic [511:0] value;
        int bank;
        value = '0;
        for (int word = 0; word < 16; word++) begin
            bank = (base[3:0] + word) & 15;
            value[bank*32 +: 32] = word_value({1'b0, base} + word);
        end
        return value;
    endfunction

    task automatic clear_request;
        request_valid = 1'b0;
        logical_base_word = '0;
        request_start = 1'b0;
        request_last = 1'b0;
        request_width = '0;
        request_tag = '0;
    endtask

    task automatic assert_reset(input int cycles);
        @(negedge clk_core);
        rst_core = 1'b1;
        clear_request();
        scoreboard_enable = 1'b0;
        force_ready = 1'b1;
        stall_enable = 1'b0;
        repeat (cycles) @(posedge clk_core);
        reset_recoveries++;
    endtask

    task automatic release_reset;
        @(negedge clk_core);
        rst_core = 1'b0;
    endtask

    task automatic complete_flush(input bit start_with_high_ack);
        begin
            macro_mode = MACRO_QUIET;
            manual_response_valid = 1'b0;
            if (start_with_high_ack) begin
                macro_flush_ack = 1'b1;
                repeat (2) begin
                    @(posedge clk_core);
                    if (!recovery_active || !macro_flush_req
                            || request_ready || macro_request_valid)
                        $fatal(1, "M139 stale high ack released recovery");
                end
                initial_high_ack_tests++;
            end
            @(negedge clk_core);
            macro_flush_ack = 1'b0;
            @(posedge clk_core);
            @(negedge clk_core);
            macro_flush_ack = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            macro_flush_ack = 1'b0;
            @(posedge clk_core);
            @(negedge clk_core);
            #1ps;
            if (recovery_active || macro_flush_req || protocol_error
                    || !request_ready)
                $fatal(1, "M139 four-phase flush did not enter run");
            flush_handshakes++;
        end
    endtask

    task automatic reset_and_flush(input bit initially_high);
        begin
            assert_reset(2);
            release_reset();
            complete_flush(initially_high);
        end
    endtask

    task automatic drive_stream(input int count, input int seed,
                                input bit enable_stalls,
                                input bit count_as_wrap);
        logic [11:0] base;
        begin
            macro_mode = MACRO_AUTO;
            scoreboard_enable = 1'b1;
            stall_enable = enable_stalls;
            force_ready = 1'b1;
            for (int item = 0; item < count; item++) begin
                base = (seed + item*29) % 3665;
                @(negedge clk_core);
                request_valid = 1'b1;
                logical_base_word = base;
                request_start = item[0];
                request_last = item[1];
                request_width = 4'd8 + (item % 4);
                request_tag = 32'h1390_0000 + seed + item;
                @(posedge clk_core);
                while (!request_accept)
                    @(posedge clk_core);
                if (count_as_wrap)
                    wrap_requests++;
            end
            @(negedge clk_core);
            clear_request();
            stall_enable = 1'b0;
            force_ready = 1'b1;
        end
    endtask

    task automatic accept_one(input logic [11:0] base,
                              input logic [31:0] tag);
        begin
            @(negedge clk_core);
            request_valid = 1'b1;
            logical_base_word = base;
            request_start = 1'b1;
            request_last = 1'b0;
            request_width = 4'd9;
            request_tag = tag;
            @(posedge clk_core);
            while (!request_accept)
                @(posedge clk_core);
            @(negedge clk_core);
            clear_request();
        end
    endtask

    always @(negedge clk_core) begin
        if (rst_core)
            response_ready = 1'b0;
        else if (stall_enable)
            response_ready = wall_cycle % 7 > 1;
        else
            response_ready = force_ready;
    end

    always @(posedge clk_core) begin
        wall_cycle++;
        if (rst_core) begin
            expected_q.delete();
            expected_request_token = '0;
            last_accept_cycle = -100;
        end else begin
            if (request_accept) begin
                if (macro_request_token !== expected_request_token)
                    $fatal(1, "M139 token mismatch expected=%0h got=%0h",
                           expected_request_token, macro_request_token);
                if (expected_request_token == 16'hffff)
                    wrap_crossings++;
                expected_request_token = expected_request_token + 1'b1;
                for (int bank = 0; bank < 16; bank++) begin
                    if (macro_bank_row_addresses[bank*8 +: 8]
                            !== logical_base_word[11:4]
                                + (bank < logical_base_word[3:0]))
                        $fatal(1, "M139 address mismatch bank=%0d", bank);
                end
                if (scoreboard_enable) begin
                    expected_item.words = logical_vector(logical_base_word);
                    expected_item.start_bit = request_start;
                    expected_item.last_bit = request_last;
                    expected_item.width = request_width;
                    expected_item.tag_bits = request_tag;
                    expected_item.token = macro_request_token;
                    expected_q.push_back(expected_item);
                    positive_requests++;
                    if (last_accept_cycle + 1 == wall_cycle)
                        ii1_checks++;
                    last_accept_cycle = wall_cycle;
                end
            end

            if (response_valid && !response_ready)
                stall_cycles++;
            if (buffered_responses == 1)
                skid_cycles++;
            if (response_accept && scoreboard_enable) begin
                if (expected_q.size() == 0)
                    $fatal(1, "M139 output without expected item");
                expected_item = expected_q.pop_front();
                if (response_logical_words !== expected_item.words
                        || response_start !== expected_item.start_bit
                        || response_last !== expected_item.last_bit
                        || response_width !== expected_item.width
                        || response_tag !== expected_item.tag_bits
                        || response_token !== expected_item.token)
                    $fatal(1, "M139 response mismatch token=%0h",
                           response_token);
                positive_outputs++;
                logical_word_checks += 16;
            end
        end

        case (macro_mode)
            MACRO_AUTO: begin
                macro_response_valid <= macro_request_valid;
                macro_response_token <= macro_request_token;
                macro_bank_words <= physical_bank_vector(logical_base_word);
            end
            MACRO_MANUAL: begin
                macro_response_valid <= manual_response_valid;
                macro_response_token <= manual_response_token;
                macro_bank_words <= manual_bank_words;
            end
            MACRO_WRONG: begin
                macro_response_valid <= macro_request_valid;
                macro_response_token <= macro_request_token ^ 16'h0001;
                macro_bank_words <= physical_bank_vector(logical_base_word);
            end
            default: begin
                macro_response_valid <= 1'b0;
                macro_response_token <= '0;
                macro_bank_words <= '0;
            end
        endcase
    end

    initial begin : test_sequence
        clear_request();
        macro_flush_ack = 1'b1;
        macro_response_valid = 1'b0;
        macro_response_token = '0;
        macro_bank_words = '0;
        manual_response_valid = 1'b0;
        manual_response_token = '0;
        manual_bank_words = '0;
        macro_mode = MACRO_QUIET;
        response_ready = 1'b0;
        scoreboard_enable = 1'b0;
        stall_enable = 1'b0;
        force_ready = 1'b1;
        wall_cycle = 0;
        positive_requests = 0;
        positive_outputs = 0;
        logical_word_checks = 0;
        ii1_checks = 0;
        last_accept_cycle = -100;
        flush_handshakes = 0;
        reset_recoveries = 0;
        initial_high_ack_tests = 0;
        stale_drain_attacks = 0;
        completion_collision_attacks = 0;
        post_completion_attacks = 0;
        wrong_token_attacks = 0;
        reset_pending_attacks = 0;
        reset_skid_attacks = 0;
        stall_cycles = 0;
        skid_cycles = 0;
        wrap_requests = 0;
        wrap_crossings = 0;
        expected_request_token = '0;

        repeat (3) @(posedge clk_core);
        release_reset();
        complete_flush(1'b1);

        // Replay the M137 positive shape, including a real skid stall.
        drive_stream(96, 3, 1'b0, 1'b0);
        drive_stream(32, 101, 1'b1, 1'b0);
        wait (expected_q.size() == 0 && !dut.bridge.busy);

        // Reset with an outstanding request; the stale token-0 response is
        // drained before the fresh low/high/low acknowledgement completes.
        reset_and_flush(1'b0);
        macro_mode = MACRO_QUIET;
        scoreboard_enable = 1'b0;
        accept_one(12'd16, 32'hdead_0000);
        reset_pending_attacks++;
        @(negedge clk_core);
        rst_core = 1'b1;
        macro_mode = MACRO_MANUAL;
        manual_response_valid = 1'b1;
        manual_response_token = 16'h0000;
        manual_bank_words = physical_bank_vector(12'd16);
        repeat (2) @(posedge clk_core);
        reset_recoveries++;
        release_reset();
        macro_flush_ack = 1'b0;
        repeat (2) @(posedge clk_core);
        if (response_valid || protocol_error || request_ready)
            $fatal(1, "M139 stale drain escaped WAIT_LOW/HIGH");
        stale_drain_attacks++;
        @(negedge clk_core);
        manual_response_valid = 1'b0;
        @(posedge clk_core);
        @(negedge clk_core);
        macro_flush_ack = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        macro_flush_ack = 1'b0;
        @(posedge clk_core);
        flush_handshakes++;
        macro_mode = MACRO_AUTO;
        scoreboard_enable = 1'b1;
        accept_one(12'd64, 32'hf2e5_0000);
        wait (expected_q.size() == 0 && !dut.bridge.busy);
        if (protocol_error)
            $fatal(1, "M139 fresh token-0 request failed after stale drain");

        // Completion edge collision must fault instead of releasing service.
        assert_reset(2);
        release_reset();
        macro_flush_ack = 1'b0;
        macro_mode = MACRO_MANUAL;
        manual_response_valid = 1'b1;
        manual_response_token = 16'h0033;
        manual_bank_words = '1;
        @(posedge clk_core);
        @(negedge clk_core);
        macro_flush_ack = 1'b1;
        @(posedge clk_core);
        #1ps;
        if (!protocol_error || request_ready || response_valid)
            $fatal(1, "M139 completion collision did not fail closed");
        completion_collision_attacks++;

        // A return in WAIT_DROP after a clean completion is also fatal.
        assert_reset(2);
        release_reset();
        macro_mode = MACRO_QUIET;
        macro_flush_ack = 1'b0;
        @(posedge clk_core);
        @(negedge clk_core);
        macro_flush_ack = 1'b1;
        @(posedge clk_core);
        macro_mode = MACRO_MANUAL;
        manual_response_valid = 1'b1;
        manual_response_token = 16'h0044;
        manual_bank_words = '1;
        @(posedge clk_core);
        @(negedge clk_core);
        #1ps;
        if (!protocol_error || request_ready || response_valid)
            $fatal(1, "M139 post-completion response did not fault");
        post_completion_attacks++;

        // Wrong token remains fail-closed in normal operation.
        reset_and_flush(1'b0);
        macro_mode = MACRO_WRONG;
        scoreboard_enable = 1'b0;
        accept_one(12'd31, 32'hbad0_0139);
        @(posedge clk_core);
        #1ps;
        if (!protocol_error || response_valid)
            $fatal(1, "M139 inherited wrong-token attack did not fault");
        wrong_token_attacks++;

        // Reset from a populated skid and require a fresh flush before reuse.
        reset_and_flush(1'b0);
        macro_mode = MACRO_AUTO;
        scoreboard_enable = 1'b0;
        force_ready = 1'b0;
        accept_one(12'd47, 32'h5a1d_0001);
        @(posedge clk_core);
        @(negedge clk_core);
        if (buffered_responses != 1)
            $fatal(1, "M139 skid setup failed");
        reset_skid_attacks++;
        assert_reset(2);
        release_reset();
        complete_flush(1'b0);
        force_ready = 1'b1;
        if (response_valid || buffered_responses != 0 || protocol_error)
            $fatal(1, "M139 reset-skid recovery leaked state");

        // Natural token wrap remains lossless and II1 after the flush repair.
        reset_and_flush(1'b0);
        drive_stream(65538, 211, 1'b0, 1'b1);
        wait (expected_q.size() == 0 && !dut.bridge.busy);
        repeat (2) @(posedge clk_core);

        if (positive_requests != 65667 || positive_outputs != 65667
                || logical_word_checks != 1050672
                || wrap_requests != 65538 || wrap_crossings != 1
                || ii1_checks < 65650 || flush_handshakes < 7
                || initial_high_ack_tests != 1 || stale_drain_attacks != 1
                || completion_collision_attacks != 1
                || post_completion_attacks != 1 || wrong_token_attacks != 1
                || reset_pending_attacks != 1 || reset_skid_attacks != 1
                || stall_cycles == 0 || skid_cycles == 0)
            $fatal(1, "M139 counter deficit req=%0d out=%0d words=%0d wrap_req=%0d wrap=%0d ii1=%0d flush=%0d stall=%0d skid=%0d",
                   positive_requests, positive_outputs, logical_word_checks,
                   wrap_requests, wrap_crossings, ii1_checks,
                   flush_handshakes, stall_cycles, skid_cycles);
        $display("PASS M139 epoch-safe fallthrough tagged 16-bank bridge VCS requests=%0d outputs=%0d words=%0d wrap_requests=%0d wrap_crossings=%0d ii1=%0d flushes=%0d initial_high_ack=%0d stale_drain=%0d completion_collision=%0d post_completion=%0d wrong_token=%0d reset_pending=%0d reset_skid=%0d stalls=%0d skid_cycles=%0d flush_fsm_bits=2 normal_ii=1 delivery_latency=1 macro=false physical_speedup=false system_speedup=false headline=false",
                 positive_requests, positive_outputs, logical_word_checks,
                 wrap_requests, wrap_crossings, ii1_checks,
                 flush_handshakes, initial_high_ack_tests,
                 stale_drain_attacks, completion_collision_attacks,
                 post_completion_attacks, wrong_token_attacks,
                 reset_pending_attacks, reset_skid_attacks,
                 stall_cycles, skid_cycles);
        $finish;
    end

    initial begin
        #3000000;
        $fatal(1, "M139 directed VCS timeout");
    end
endmodule

`default_nettype wire
