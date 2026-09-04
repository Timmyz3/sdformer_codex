`timescale 1ns/1ps
`default_nettype none

module tb_m136_latency_tagged_16bank_response_bridge;
    logic clk_core = 1'b0;
    logic rst_core = 1'b1;
    always #1.5 clk_core = ~clk_core;

    logic request_valid;
    logic request_ready;
    logic [11:0] logical_base_word;
    logic request_start;
    logic request_last;
    logic [3:0] request_width;
    logic [31:0] request_tag;
    logic request_accept;
    logic macro_request_valid;
    logic [127:0] macro_bank_row_addresses;
    logic [15:0] macro_request_token;
    logic macro_response_valid;
    logic [15:0] macro_response_token;
    logic [511:0] macro_bank_words;
    logic response_valid;
    logic response_ready;
    logic [511:0] response_logical_words;
    logic response_start;
    logic response_last;
    logic [3:0] response_width;
    logic [31:0] response_tag;
    logic [15:0] response_token;
    logic response_accept;
    logic protocol_error;
    logic pending_response;
    logic [1:0] buffered_responses;
    logic busy;

    m136_latency_tagged_16bank_response_bridge dut (.*);

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

    integer macro_mode;
    localparam int MACRO_AUTO = 0;
    localparam int MACRO_MISSING = 1;
    localparam int MACRO_WRONG_TOKEN = 2;
    localparam int MACRO_UNSOLICITED = 3;
    logic model_pending_valid;
    logic [11:0] model_pending_base;
    logic [15:0] model_pending_token;

    logic scoreboard_enable;
    logic stall_enable;
    logic force_ready;
    integer wall_cycle;
    integer positive_requests;
    integer positive_outputs;
    integer logical_word_checks;
    integer ii1_checks;
    integer stall_cycles;
    integer fifo_full_cycles;
    integer row_crossing_requests;
    integer wrong_token_attacks;
    integer missing_response_attacks;
    integer unsolicited_response_attacks;
    integer illegal_base_attacks;
    integer reset_recoveries;
    integer last_accept_cycle;
    logic [15:0] expected_request_token;

    function automatic logic [31:0] word_value(input logic [12:0] word_index);
        word_value = 32'h6d13_0000 ^ {19'h0, word_index}
                   ^ ({19'h0, word_index} << 7);
    endfunction

    function automatic logic [511:0] logical_vector(input logic [11:0] base);
        logic [511:0] value;
        value = '0;
        for (int word = 0; word < 16; word++)
            value[word*32 +: 32] = word_value({1'b0, base} + word);
        return value;
    endfunction

    function automatic logic [511:0] physical_bank_vector(input logic [11:0] base);
        logic [511:0] value;
        int bank;
        value = '0;
        for (int word = 0; word < 16; word++) begin
            bank = (base[3:0] + word) & 15;
            value[bank*32 +: 32] = word_value({1'b0, base} + word);
        end
        return value;
    endfunction

    task automatic apply_reset;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            request_valid = 1'b0;
            macro_mode = MACRO_MISSING;
            force_ready = 1'b1;
            stall_enable = 1'b0;
            repeat (2) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 1'b0;
            reset_recoveries = reset_recoveries + 1;
        end
    endtask

    task automatic drive_positive_requests(
        input int count,
        input int seed,
        input logic enable_stalls
    );
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
                request_tag = 32'h1360_0000 + seed + item;
                @(posedge clk_core);
                while (!request_accept)
                    @(posedge clk_core);
            end
            @(negedge clk_core);
            request_valid = 1'b0;
            stall_enable = 1'b0;
            force_ready = 1'b1;
        end
    endtask

    task automatic accept_one_attack_request(input logic [11:0] base);
        begin
            scoreboard_enable = 1'b0;
            @(negedge clk_core);
            request_valid = 1'b1;
            logical_base_word = base;
            request_start = 1'b1;
            request_last = 1'b0;
            request_width = 4'd9;
            request_tag = 32'hbad0_0136;
            @(posedge clk_core);
            while (!request_accept)
                @(posedge clk_core);
            @(negedge clk_core);
            request_valid = 1'b0;
        end
    endtask

    always @(posedge clk_core) begin
        wall_cycle = wall_cycle + 1;
        if (rst_core) begin
            model_pending_valid = 1'b0;
            model_pending_base = '0;
            model_pending_token = '0;
            macro_response_valid <= 1'b0;
            macro_response_token <= '0;
            macro_bank_words <= '0;
            expected_request_token = '0;
            expected_q.delete();
            last_accept_cycle = -100;
        end else begin
            model_pending_valid = request_accept;
            if (request_accept) begin
                model_pending_base = logical_base_word;
                model_pending_token = macro_request_token;
                if (macro_request_token !== expected_request_token)
                    $fatal(1, "M136 request token mismatch expected=%0d observed=%0d",
                           expected_request_token, macro_request_token);
                expected_request_token = expected_request_token + 1'b1;
                for (int bank = 0; bank < 16; bank++) begin
                    if (macro_bank_row_addresses[bank*8 +: 8]
                            !== logical_base_word[11:4]
                                + (bank < logical_base_word[3:0]))
                        $fatal(1, "M136 address mismatch base=%0d bank=%0d",
                               logical_base_word, bank);
                end
                if (scoreboard_enable) begin
                    expected_item.words = logical_vector(logical_base_word);
                    expected_item.start_bit = request_start;
                    expected_item.last_bit = request_last;
                    expected_item.width = request_width;
                    expected_item.tag_bits = request_tag;
                    expected_item.token = macro_request_token;
                    expected_q.push_back(expected_item);
                    positive_requests = positive_requests + 1;
                    if (last_accept_cycle + 1 == wall_cycle)
                        ii1_checks = ii1_checks + 1;
                    last_accept_cycle = wall_cycle;
                    if (logical_base_word[3:0] != 0)
                        row_crossing_requests = row_crossing_requests + 1;
                end
            end

            // A synchronous one-cycle macro wrapper launches its response
            // immediately after the request-accepting edge; the DUT samples it
            // at the following edge.  Updating here avoids half-cycle-only
            // valid pulses and exercises true edge-to-edge latency alignment.
            case (macro_mode)
                MACRO_AUTO: begin
                    macro_response_valid <= request_accept;
                    macro_response_token <= macro_request_token;
                    macro_bank_words <= physical_bank_vector(logical_base_word);
                end
                MACRO_WRONG_TOKEN: begin
                    macro_response_valid <= request_accept;
                    macro_response_token <= macro_request_token ^ 16'h0001;
                    macro_bank_words <= physical_bank_vector(logical_base_word);
                end
                MACRO_UNSOLICITED: begin
                    macro_response_valid <= 1'b1;
                    macro_response_token <= 16'hf136;
                    macro_bank_words <= 512'h1;
                end
                default: begin
                    macro_response_valid <= 1'b0;
                    macro_response_token <= '0;
                    macro_bank_words <= '0;
                end
            endcase

            if (response_valid && !response_ready)
                stall_cycles = stall_cycles + 1;
            if (buffered_responses == 2)
                fifo_full_cycles = fifo_full_cycles + 1;

            if (response_accept) begin
                if (expected_q.size() == 0)
                    $fatal(1, "M136 unexpected response token=%0d", response_token);
                expected_item = expected_q.pop_front();
                if (response_logical_words !== expected_item.words
                        || response_start !== expected_item.start_bit
                        || response_last !== expected_item.last_bit
                        || response_width !== expected_item.width
                        || response_tag !== expected_item.tag_bits
                        || response_token !== expected_item.token)
                    $fatal(1, "M136 response mismatch token=%0d", response_token);
                positive_outputs = positive_outputs + 1;
                logical_word_checks = logical_word_checks + 16;
            end
        end
    end

    always @(negedge clk_core) begin
        if (rst_core) begin
            response_ready = 1'b0;
        end else begin
            if (stall_enable)
                response_ready = wall_cycle % 7 != 0 && wall_cycle % 7 != 1;
            else
                response_ready = force_ready;
        end
    end

    initial begin
        request_valid = 1'b0;
        logical_base_word = '0;
        request_start = 1'b0;
        request_last = 1'b0;
        request_width = '0;
        request_tag = '0;
        macro_response_valid = 1'b0;
        macro_response_token = '0;
        macro_bank_words = '0;
        response_ready = 1'b0;
        macro_mode = MACRO_MISSING;
        scoreboard_enable = 1'b0;
        stall_enable = 1'b0;
        force_ready = 1'b1;
        wall_cycle = 0;
        positive_requests = 0;
        positive_outputs = 0;
        logical_word_checks = 0;
        ii1_checks = 0;
        stall_cycles = 0;
        fifo_full_cycles = 0;
        row_crossing_requests = 0;
        wrong_token_attacks = 0;
        missing_response_attacks = 0;
        unsolicited_response_attacks = 0;
        illegal_base_attacks = 0;
        reset_recoveries = 0;
        last_accept_cycle = -100;
        expected_request_token = '0;

        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        macro_mode = MACRO_AUTO;
        response_ready = 1'b1;

        drive_positive_requests(96, 3, 1'b0);
        drive_positive_requests(32, 101, 1'b1);
        wait (expected_q.size() == 0 && !busy);
        repeat (3) @(posedge clk_core);
        if (positive_requests != 128 || positive_outputs != 128
                || logical_word_checks != 2048 || ii1_checks < 95
                || row_crossing_requests < 100 || stall_cycles == 0
                || fifo_full_cycles == 0)
            $fatal(1, "M136 positive coverage deficit req=%0d out=%0d words=%0d ii1=%0d crossing=%0d stall=%0d full=%0d",
                   positive_requests, positive_outputs, logical_word_checks,
                   ii1_checks, row_crossing_requests, stall_cycles,
                   fifo_full_cycles);

        apply_reset();
        macro_mode = MACRO_WRONG_TOKEN;
        accept_one_attack_request(12'd31);
        @(posedge clk_core);
        if (!protocol_error || response_valid)
            $fatal(1, "M136 wrong-token attack did not quarantine");
        wrong_token_attacks = wrong_token_attacks + 1;
        @(posedge clk_core);

        apply_reset();
        macro_mode = MACRO_MISSING;
        accept_one_attack_request(12'd47);
        @(posedge clk_core);
        if (!protocol_error || response_valid)
            $fatal(1, "M136 missing-response attack did not quarantine");
        missing_response_attacks = missing_response_attacks + 1;
        @(posedge clk_core);

        apply_reset();
        macro_mode = MACRO_UNSOLICITED;
        @(posedge clk_core);
        @(negedge clk_core);
        if (!protocol_error || response_valid)
            $fatal(1, "M136 unsolicited-response attack did not quarantine");
        unsolicited_response_attacks = unsolicited_response_attacks + 1;
        macro_mode = MACRO_MISSING;
        @(posedge clk_core);

        apply_reset();
        macro_mode = MACRO_MISSING;
        scoreboard_enable = 1'b0;
        @(negedge clk_core);
        request_valid = 1'b1;
        logical_base_word = 12'd3666;
        request_start = 1'b1;
        request_last = 1'b0;
        request_width = 4'd8;
        request_tag = 32'hbad0_ffff;
        @(posedge clk_core);
        if (request_accept || !protocol_error)
            $fatal(1, "M136 illegal-base attack did not quarantine");
        illegal_base_attacks = illegal_base_attacks + 1;
        @(negedge clk_core);
        request_valid = 1'b0;
        @(posedge clk_core);

        $display("PASS M136 one-cycle tagged 16-bank response bridge VCS requests=%0d outputs=%0d words=%0d ii1=%0d stalls=%0d fifo_full=%0d row_crossings=%0d wrong_token=%0d missing=%0d unsolicited=%0d illegal_base=%0d reset_recoveries=%0d fifo_depth=2 latency=1 macro=false physical_speedup=false system_speedup=false headline=false",
                 positive_requests, positive_outputs, logical_word_checks,
                 ii1_checks, stall_cycles, fifo_full_cycles,
                 row_crossing_requests, wrong_token_attacks,
                 missing_response_attacks, unsolicited_response_attacks,
                 illegal_base_attacks, reset_recoveries);
        $finish;
    end

    initial begin
        #200000;
        $fatal(1, "M136 watchdog timeout");
    end
endmodule

`default_nettype wire
