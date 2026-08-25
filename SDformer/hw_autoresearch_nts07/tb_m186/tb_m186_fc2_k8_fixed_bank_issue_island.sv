`timescale 1ns/1ps
`default_nettype none

module tb_m186_fc2_k8_fixed_bank_issue_island;
    localparam int LANES = 96;
    localparam int MAX_RESULTS = 2000;

    logic clk_core, rst_core;
    logic header_valid, header_ready, header_accept;
    logic [23:0] header_tag;
    logic [3:0] header_output_blocks;
    logic [5:0] header_descriptor_count;
    logic descriptor_valid, descriptor_ready, descriptor_accept;
    logic [4:0] descriptor_beat_index;
    logic [95:0] descriptor_bitmap;
    logic weight_request_valid, weight_request_ready;
    logic [23:0] weight_request_tag;
    logic [2:0] weight_request_output_block;
    logic [3:0] weight_request_source_count;
    logic [7:0] weight_request_bank_valid;
    logic [11:0] weight_request_source_channel [0:7];
    logic weight_request_accept;
    logic weight_response_valid, weight_response_ready;
    logic signed [7:0] weight_response [0:7][0:LANES-1];
    logic signed [23:0] accumulator_context [0:LANES-1];
    logic weight_response_accept;
    logic result_valid, result_ready, result_accept;
    logic [23:0] result_token_tag;
    logic [2:0] result_output_block;
    logic [3:0] result_source_count;
    logic [7:0] result_bank_mask;
    logic signed [23:0] result_accumulator [0:LANES-1];
    logic token_done_valid, token_done_ready, token_done_accept;
    logic [23:0] token_done_tag;
    logic token_done_had_event;
    logic protocol_error, numeric_overflow, busy;

    logic random_request_stall_mode;
    logic random_result_stall_mode;
    logic random_done_stall_mode;
    logic manual_ready_mode;
    logic manual_response_mode;
    logic force_overflow_response;
    logic scoreboard_enabled;

    logic service_pending;
    logic [23:0] service_tag;
    logic [2:0] service_block;
    logic [7:0] service_mask;
    logic [11:0] service_channel [0:7];
    integer service_delay;
    logic response_accept_q;

    logic [23:0] expected_tag [0:MAX_RESULTS-1];
    logic [2:0] expected_block [0:MAX_RESULTS-1];
    logic [3:0] expected_count [0:MAX_RESULTS-1];
    logic [7:0] expected_mask [0:MAX_RESULTS-1];
    logic signed [23:0] expected_accumulator
        [0:MAX_RESULTS-1][0:LANES-1];
    logic [95:0] token_bitmap [0:31];

    integer expected_write, expected_read;
    integer accepted_headers, accepted_descriptors, accepted_tokens;
    integer accepted_requests, accepted_responses, accepted_results;
    integer input_events, expected_replayed_terms, observed_request_terms;
    integer request_stall_cycles, response_stall_cycles;
    integer result_stall_cycles, done_wait_cycles;
    integer same_cycle_response_request_replace;
    integer nonprefix_requests;
    integer final_headers, final_descriptors, final_tokens;
    integer final_requests, final_responses, final_results;
    integer final_input_events, final_expected_terms, final_observed_terms;
    integer final_request_stalls, final_response_stalls, final_result_stalls;
    integer final_done_wait, final_replace, final_nonprefix;

    m186_fc2_k8_fixed_bank_issue_island dut (.*);
    bind m186_fc2_k8_fixed_bank_issue_island
        m186_fc2_k8_fixed_bank_issue_island_assertions sva (.*);

    initial clk_core = 1'b0;
    always #1.5 clk_core = ~clk_core;

    function automatic logic [95:0] make_bitmap(
            input integer pattern, input integer descriptor);
        logic [95:0] value;
        integer forced_bit;
        begin
            value = '0;
            if (pattern == 1 && descriptor == 0) begin
                value[0] = 1'b1;
                value[7] = 1'b1;
            end else begin
                for (int bit_index = 0; bit_index < 96; bit_index++) begin
                    if (((bit_index + descriptor*5 + pattern*7)
                            % (15 + pattern)) == 0)
                        value[bit_index] = 1'b1;
                end
                forced_bit = (descriptor*13 + pattern*9) % 96;
                value[forced_bit] = 1'b1;
                if ((descriptor % 4) == 0)
                    value[(descriptor + pattern) % 8] = 1'b1;
            end
            return value;
        end
    endfunction

    function automatic integer popcount96(input logic [95:0] value);
        integer count;
        begin
            count = 0;
            for (int bit_index = 0; bit_index < 96; bit_index++)
                count += value[bit_index];
            return count;
        end
    endfunction

    function automatic integer popcount8(input logic [7:0] value);
        integer count;
        begin
            count = 0;
            for (int bit_index = 0; bit_index < 8; bit_index++)
                count += value[bit_index];
            return count;
        end
    endfunction

    task automatic send_header(
            input logic [23:0] tag_value,
            input integer output_blocks_value,
            input integer descriptor_count);
        begin
            @(negedge clk_core);
            header_tag = tag_value;
            header_output_blocks = output_blocks_value;
            header_descriptor_count = descriptor_count;
            header_valid = 1'b1;
            do @(posedge clk_core); while (!header_accept);
            @(negedge clk_core);
            header_valid = 1'b0;
        end
    endtask

    task automatic send_descriptors(input integer descriptor_count);
        integer descriptor;
        begin
            if (descriptor_count != 0) begin
                descriptor = 0;
                @(negedge clk_core);
                descriptor_beat_index = 0;
                descriptor_bitmap = token_bitmap[0];
                descriptor_valid = 1'b1;
                while (descriptor < descriptor_count) begin
                    @(posedge clk_core);
                    if (descriptor_accept) begin
                        descriptor++;
                        if (descriptor < descriptor_count) begin
                            @(negedge clk_core);
                            descriptor_beat_index = descriptor[4:0];
                            descriptor_bitmap = token_bitmap[descriptor];
                        end
                    end
                end
                @(negedge clk_core);
                descriptor_valid = 1'b0;
            end
        end
    endtask

    task automatic drive_token(
            input integer pattern,
            input logic [23:0] tag_value,
            input integer output_blocks_value,
            input integer descriptor_count);
        integer events;
        integer target_done;
        begin
            events = 0;
            for (int descriptor = 0;
                    descriptor < descriptor_count; descriptor++) begin
                token_bitmap[descriptor] = make_bitmap(pattern, descriptor);
                if (token_bitmap[descriptor] == 0)
                    $fatal(1, "M186 generated empty descriptor");
                events += popcount96(token_bitmap[descriptor]);
            end
            input_events += events;
            expected_replayed_terms += events * output_blocks_value;
            target_done = accepted_tokens + 1;
            send_header(tag_value, output_blocks_value, descriptor_count);
            send_descriptors(descriptor_count);
            wait (accepted_tokens == target_done);
            if (token_done_tag !== tag_value)
                $fatal(1, "M186 token completion tag drift");
            @(negedge clk_core);
        end
    endtask

    // Capture each accepted request into the one-outstanding response model;
    // create the exact numerical scoreboard at response acceptance.
    always @(posedge clk_core) begin
        integer total;
        if (rst_core) begin
            service_pending <= 1'b0;
            service_tag <= '0;
            service_block <= '0;
            service_mask <= '0;
            service_delay <= 0;
            response_accept_q <= 1'b0;
            accepted_headers <= 0;
            accepted_descriptors <= 0;
            accepted_tokens <= 0;
            accepted_requests <= 0;
            accepted_responses <= 0;
            accepted_results <= 0;
            observed_request_terms <= 0;
            request_stall_cycles <= 0;
            response_stall_cycles <= 0;
            result_stall_cycles <= 0;
            done_wait_cycles <= 0;
            same_cycle_response_request_replace <= 0;
            nonprefix_requests <= 0;
        end else begin
            response_accept_q <= weight_response_accept;
            if (header_accept)
                accepted_headers <= accepted_headers + 1;
            if (descriptor_accept)
                accepted_descriptors <= accepted_descriptors + 1;
            if (token_done_accept && scoreboard_enabled)
                accepted_tokens <= accepted_tokens + 1;
            if (weight_request_valid && !weight_request_ready)
                request_stall_cycles <= request_stall_cycles + 1;
            if (weight_response_valid && !weight_response_ready)
                response_stall_cycles <= response_stall_cycles + 1;
            if (result_valid && !result_ready)
                result_stall_cycles <= result_stall_cycles + 1;
            if (dut.m184_done_valid && !token_done_valid)
                done_wait_cycles <= done_wait_cycles + 1;
            if (weight_response_accept && weight_request_accept)
                same_cycle_response_request_replace
                    <= same_cycle_response_request_replace + 1;

            if (weight_response_accept && !weight_request_accept)
                service_pending <= 1'b0;
            if (weight_request_accept) begin
                if (service_pending && !weight_response_accept)
                    $fatal(1, "M186 request model overflow");
                service_pending <= 1'b1;
                service_tag <= weight_request_tag;
                service_block <= weight_request_output_block;
                service_mask <= weight_request_bank_valid;
                for (int bank = 0; bank < 8; bank++) begin
                    service_channel[bank]
                        <= weight_request_source_channel[bank];
                    if (weight_request_bank_valid[bank]
                            && weight_request_source_channel[bank][2:0]
                                != bank[2:0])
                        $fatal(1, "M186 request channel-bank mismatch");
                    if (!weight_request_bank_valid[bank]
                            && weight_request_source_channel[bank] != 0)
                        $fatal(1, "M186 invalid bank carried channel");
                end
                if (weight_request_source_count
                        != popcount8(weight_request_bank_valid))
                    $fatal(1, "M186 request popcount mismatch");
                accepted_requests <= accepted_requests + 1;
                observed_request_terms <= observed_request_terms
                    + weight_request_source_count;
                if (weight_request_bank_valid[7]
                        && !weight_request_bank_valid[6]
                        && weight_request_bank_valid != 8'hff)
                    nonprefix_requests <= nonprefix_requests + 1;
                service_delay <= $urandom_range(0, 3);
            end else if (service_pending && !weight_response_valid
                    && service_delay > 0) begin
                service_delay <= service_delay - 1;
            end

            if (weight_response_accept) begin
                accepted_responses <= accepted_responses + 1;
                if (scoreboard_enabled) begin
                    if (expected_write >= MAX_RESULTS)
                        $fatal(1, "M186 expected result overflow");
                    expected_tag[expected_write] = service_tag;
                    expected_block[expected_write] = service_block;
                    expected_count[expected_write]
                        = popcount8(service_mask);
                    expected_mask[expected_write] = service_mask;
                    for (int lane = 0; lane < LANES; lane++) begin
                        total = $signed(accumulator_context[lane]);
                        for (int bank = 0; bank < 8; bank++) begin
                            if (service_mask[bank])
                                total = total
                                    + $signed(weight_response[bank][lane]);
                        end
                        expected_accumulator[expected_write][lane] = total;
                    end
                    expected_write = expected_write + 1;
                end
            end

            if (result_accept && scoreboard_enabled) begin
                if (expected_read >= expected_write)
                    $fatal(1, "M186 unexpected arithmetic result");
                if (result_token_tag !== expected_tag[expected_read]
                        || result_output_block
                            !== expected_block[expected_read]
                        || result_source_count
                            !== expected_count[expected_read]
                        || result_bank_mask !== expected_mask[expected_read])
                    $fatal(1, "M186 result header mismatch index=%0d",
                        expected_read);
                for (int lane = 0; lane < LANES; lane++) begin
                    if (result_accumulator[lane]
                            !== expected_accumulator[expected_read][lane])
                        $fatal(1,
                            "M186 result mismatch index=%0d lane=%0d got=%0d expected=%0d",
                            expected_read, lane, result_accumulator[lane],
                            expected_accumulator[expected_read][lane]);
                end
                expected_read = expected_read + 1;
                accepted_results <= accepted_results + 1;
            end
        end
    end

    // Banked response and sink backpressure model.
    always @(negedge clk_core) begin
        if (rst_core) begin
            weight_request_ready = 1'b0;
            result_ready = 1'b0;
            token_done_ready = 1'b0;
            weight_response_valid = 1'b0;
        end else begin
            if (!manual_ready_mode) begin
                weight_request_ready = random_request_stall_mode
                    ? ($urandom_range(0, 4) != 0) : 1'b1;
                result_ready = random_result_stall_mode
                    ? ($urandom_range(0, 3) != 0) : 1'b1;
                token_done_ready = random_done_stall_mode
                    ? ($urandom_range(0, 4) != 0) : 1'b1;
            end
            if (!manual_response_mode) begin
                if (response_accept_q)
                    weight_response_valid = 1'b0;
                if (service_pending && !weight_response_valid
                        && service_delay == 0) begin
                    weight_response_valid = 1'b1;
                    for (int lane = 0; lane < LANES; lane++) begin
                        accumulator_context[lane]
                            = (($signed(service_tag[15:0])
                                + service_block*37 + lane*11) % 200001)
                                - 100000;
                        for (int bank = 0; bank < 8; bank++) begin
                            if (force_overflow_response)
                                weight_response[bank][lane] = 8'sd127;
                            else
                                weight_response[bank][lane]
                                    = ((service_channel[bank]
                                        + lane*3 + bank*17) % 255) - 127;
                        end
                        if (force_overflow_response)
                            accumulator_context[lane] = 24'sh7ffff0;
                    end
                end
            end
        end
    end

    initial begin
        rst_core = 1'b1;
        header_valid = 1'b0;
        header_tag = '0;
        header_output_blocks = 1;
        header_descriptor_count = 0;
        descriptor_valid = 1'b0;
        descriptor_beat_index = 0;
        descriptor_bitmap = 0;
        weight_request_ready = 1'b0;
        weight_response_valid = 1'b0;
        result_ready = 1'b0;
        token_done_ready = 1'b0;
        random_request_stall_mode = 1'b0;
        random_result_stall_mode = 1'b0;
        random_done_stall_mode = 1'b0;
        manual_ready_mode = 1'b0;
        manual_response_mode = 1'b0;
        force_overflow_response = 1'b0;
        scoreboard_enabled = 1'b1;
        expected_write = 0;
        expected_read = 0;
        input_events = 0;
        expected_replayed_terms = 0;
        repeat (5) @(posedge clk_core);
        @(negedge clk_core); rst_core = 1'b0;

        random_request_stall_mode = 1'b1;
        random_result_stall_mode = 1'b1;
        random_done_stall_mode = 1'b1;
        drive_token(1, 24'ha10000, 1, 0);
        drive_token(1, 24'ha20000, 1, 4);
        drive_token(2, 24'ha30000, 2, 8);
        drive_token(3, 24'ha40000, 4, 10);
        drive_token(4, 24'ha50000, 8, 12);
        wait (expected_read == expected_write);
        wait (!busy);
        random_request_stall_mode = 1'b0;
        random_result_stall_mode = 1'b0;
        random_done_stall_mode = 1'b0;
        repeat (2) @(posedge clk_core);

        if (accepted_headers != 5 || accepted_descriptors != 34
                || accepted_tokens != 5
                || accepted_requests == 0
                || accepted_requests != accepted_responses
                || accepted_requests != accepted_results
                || expected_write != expected_read
                || observed_request_terms != expected_replayed_terms)
            $fatal(1, "M186 conservation drift h=%0d d=%0d t=%0d req/rsp/res=%0d/%0d/%0d terms=%0d/%0d scoreboard=%0d/%0d",
                accepted_headers, accepted_descriptors, accepted_tokens,
                accepted_requests, accepted_responses, accepted_results,
                observed_request_terms, expected_replayed_terms,
                expected_write, expected_read);
        if (request_stall_cycles == 0 || response_stall_cycles == 0
                || result_stall_cycles == 0 || done_wait_cycles == 0
                || same_cycle_response_request_replace == 0
                || nonprefix_requests == 0)
            $fatal(1, "M186 coverage counters missing");

        final_headers = accepted_headers;
        final_descriptors = accepted_descriptors;
        final_tokens = accepted_tokens;
        final_requests = accepted_requests;
        final_responses = accepted_responses;
        final_results = accepted_results;
        final_input_events = input_events;
        final_expected_terms = expected_replayed_terms;
        final_observed_terms = observed_request_terms;
        final_request_stalls = request_stall_cycles;
        final_response_stalls = response_stall_cycles;
        final_result_stalls = result_stall_cycles;
        final_done_wait = done_wait_cycles;
        final_replace = same_cycle_response_request_replace;
        final_nonprefix = nonprefix_requests;
        scoreboard_enabled = 1'b0;

        // Numeric overflow: keep the accepted result drainable.
        @(negedge clk_core); rst_core = 1'b1;
        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        manual_response_mode = 1'b1;
        manual_ready_mode = 1'b1;
        force_overflow_response = 1'b1;
        weight_request_ready = 1'b1;
        result_ready = 1'b0;
        token_done_ready = 1'b1;
        token_bitmap[0] = 96'hff;
        send_header(24'hfff861, 1, 1);
        send_descriptors(1);
        wait (dut.pending_valid_q);
        @(negedge clk_core);
        for (int lane = 0; lane < LANES; lane++) begin
            accumulator_context[lane] = 24'sh7ffff0;
            for (int bank = 0; bank < 8; bank++)
                weight_response[bank][lane] = 8'sd127;
        end
        weight_response_valid = 1'b1;
        do @(posedge clk_core); while (!weight_response_accept);
        @(negedge clk_core); weight_response_valid = 1'b0;
        wait (numeric_overflow && result_valid);
        @(negedge clk_core); result_ready = 1'b1;
        wait (result_accept);
        repeat (2) @(posedge clk_core);
        if (!numeric_overflow || header_ready || descriptor_ready
                || weight_request_valid)
            $fatal(1, "M186 overflow fail-close missing");

        // Unsolicited response after reset must become a sticky protocol fault.
        @(negedge clk_core); rst_core = 1'b1;
        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        result_ready = 1'b1;
        weight_response_valid = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core); weight_response_valid = 1'b0;
        repeat (2) @(posedge clk_core);
        if (!protocol_error || header_ready || descriptor_ready
                || weight_request_valid || weight_response_ready)
            $fatal(1, "M186 unsolicited response fail-close missing");

        $display("PASS M186 FC2 K8 fixed-bank issue island VCS headers=%0d descriptors=%0d tokens=%0d requests=%0d responses=%0d results=%0d bitmap_events=%0d replayed_source_terms_expected=%0d replayed_source_terms_observed=%0d request_stall_cycles=%0d response_stall_cycles=%0d result_stall_cycles=%0d done_wait_cycles=%0d same_cycle_response_request_replace=%0d nonprefix_requests=%0d outstanding_slots=1 in_order_response=true direct_fixed_bank_mask=true bank_id_payload=false prefix_packing=false weight_response_payload_bits=6144 overflow_attacks=1 unsolicited_response_attacks=1 accumulator_context_external=true descriptor_producer=false weight_sram_macro=false bn2=false residual=false complete_fc2=false physical_speedup=false system_speedup=false headline=false",
            final_headers, final_descriptors, final_tokens,
            final_requests, final_responses, final_results,
            final_input_events, final_expected_terms, final_observed_terms,
            final_request_stalls, final_response_stalls,
            final_result_stalls, final_done_wait,
            final_replace, final_nonprefix);
        $finish;
    end

    initial begin
        #5000000;
        $fatal(1, "M186 watchdog timeout");
    end
endmodule

`default_nettype wire
