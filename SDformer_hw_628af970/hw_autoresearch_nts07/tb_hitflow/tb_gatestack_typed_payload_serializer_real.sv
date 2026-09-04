`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_typed_payload_serializer_real #(
    parameter int BITMAP_BYPASS_ENABLE = 0
);
    localparam logic [1:0] FORMAT_RAW = 2'd0;
    localparam logic [1:0] FORMAT_IPD32W = 2'd1;
    localparam logic [1:0] FORMAT_FADC24 = 2'd2;

    logic clk_core;
    logic rst_core;
    logic begin_valid;
    logic begin_ready;
    logic begin_context_id;
    logic [4:0] begin_head_id;
    logic [31:0] begin_tag;
    logic [1:0] begin_format;
    logic [15:0] begin_expected_payload_bits;
    logic [3:0] begin_active_classes;
    logic [7:0] begin_active_tokens;
    logic [7:0] begin_term_count;
    logic [12:0] begin_event_count;
    logic [7:0] begin_bitmap_term_count;
    logic [12:0] begin_fadc_destination_bytes;
    logic descriptor_valid;
    logic descriptor_ready;
    logic [8:0] descriptor_gate_code;
    logic [4:0] descriptor_lane_id;
    logic [7:0] descriptor_destination_count;
    logic descriptor_last;
    logic destination_valid;
    logic destination_ready;
    logic [7:0] destination_token_id;
    logic destination_last_for_term;
    logic destination_bitmap_valid;
    logic destination_bitmap_ready;
    logic [161:0] destination_bitmap;
    logic raw_token_valid;
    logic raw_token_ready;
    logic [7:0] raw_token_id;
    logic [8:0] raw_gate_code;
    logic [31:0] raw_k_bits;
    logic commit_begin_valid;
    logic commit_begin_ready;
    logic commit_context_id;
    logic [4:0] commit_head_id;
    logic [31:0] commit_tag;
    logic commit_mode_is_csr;
    logic [15:0] commit_payload_bits;
    logic commit_word_valid;
    logic commit_word_ready;
    logic [63:0] commit_word_data;
    logic commit_word_last;
    logic done_valid;
    logic done_ready;
    logic [31:0] done_tag;
    logic [1:0] done_format;
    logic done_error;
    logic [7:0] done_word_count;
    logic protocol_error;
    logic [31:0] count_heads;
    logic [31:0] count_committed_heads;
    logic [31:0] count_aborted_heads;
    logic [31:0] count_committed_words;
    logic [31:0] count_input_stall_cycles;
    logic [31:0] count_output_stall_cycles;

    logic [23:0] descriptor_mem [0:127];
    logic [7:0] destination_mem [0:8191];
    logic [40:0] raw_record_mem [0:161];
    logic [63:0] expected_word_mem [0:103];
    logic [31:0] prng_q;
    int observed_commits;

    gatestack_typed_payload_serializer #(
        .BITMAP_BYPASS_ENABLE(BITMAP_BYPASS_ENABLE)
    ) dut (.*);

    always #5 clk_core <= ~clk_core;

    task automatic advance_prng;
        begin
            prng_q = {prng_q[30:0],
                prng_q[31] ^ prng_q[21] ^ prng_q[1] ^ prng_q[0]};
        end
    endtask

    task automatic run_case(
        input logic [1:0] format_value,
        input logic [31:0] tag_value,
        input logic [4:0] head_value,
        input logic [3:0] active_classes,
        input logic [7:0] active_tokens,
        input int terms,
        input int events,
        input logic [7:0] bitmap_terms,
        input logic [12:0] fadc_destination_bytes,
        input int payload_bits,
        input int word_count
    );
        int destination_index;
        int count;
        int stalls;
        logic [161:0] bitmap_value;
        begin
            while (!begin_ready) @(posedge clk_core);
            @(negedge clk_core);
            begin_valid = 1'b1;
            begin_context_id = 1'b0;
            begin_head_id = head_value;
            begin_tag = tag_value;
            begin_format = format_value;
            begin_expected_payload_bits = 16'(payload_bits);
            begin_active_classes = active_classes;
            begin_active_tokens = active_tokens;
            begin_term_count = 8'(terms);
            begin_event_count = 13'(events);
            begin_bitmap_term_count = bitmap_terms;
            begin_fadc_destination_bytes = fadc_destination_bytes;
            @(posedge clk_core);
            @(negedge clk_core);
            begin_valid = 1'b0;

            if (format_value == FORMAT_RAW) begin
                for (int token = 0; token < 162; token = token + 1) begin
                    advance_prng();
                    repeat (32'(prng_q[1:0])) @(posedge clk_core);
                    @(negedge clk_core);
                    raw_token_valid = 1'b1;
                    raw_token_id = 8'(token);
                    raw_k_bits = raw_record_mem[token][31:0];
                    raw_gate_code = raw_record_mem[token][40:32];
                    do @(posedge clk_core); while (!raw_token_ready);
                    @(negedge clk_core);
                    raw_token_valid = 1'b0;
                end
            end else if (terms != 0) begin
                for (int term = 0; term < terms; term = term + 1) begin
                    advance_prng();
                    repeat (32'(prng_q[1:0])) @(posedge clk_core);
                    @(negedge clk_core);
                    descriptor_valid = 1'b1;
                    descriptor_gate_code = descriptor_mem[term][8:0];
                    descriptor_lane_id = descriptor_mem[term][13:9];
                    descriptor_destination_count = descriptor_mem[term][21:14];
                    descriptor_last = term == terms - 1;
                    do @(posedge clk_core); while (!descriptor_ready);
                    @(negedge clk_core);
                    descriptor_valid = 1'b0;
                end

                destination_index = 0;
                for (int term = 0; term < terms; term = term + 1) begin
                    count = 32'(descriptor_mem[term][21:14]);
                    if (BITMAP_BYPASS_ENABLE != 32'd0 &&
                        format_value == FORMAT_FADC24 && count > 21) begin
                        bitmap_value = '0;
                        for (int item = 0; item < count; item = item + 1)
                            bitmap_value[
                                destination_mem[destination_index + item]
                            ] = 1'b1;
                        advance_prng();
                        repeat (32'(prng_q[1:0])) @(posedge clk_core);
                        @(negedge clk_core);
                        destination_bitmap_valid = 1'b1;
                        destination_bitmap = bitmap_value;
                        do @(posedge clk_core);
                        while (!destination_bitmap_ready);
                        @(negedge clk_core);
                        destination_bitmap_valid = 1'b0;
                        destination_bitmap = '0;
                        destination_index = destination_index + count;
                    end else begin
                        for (int item = 0; item < count; item = item + 1) begin
                            advance_prng();
                            repeat (32'(prng_q[1:0])) @(posedge clk_core);
                            @(negedge clk_core);
                            destination_valid = 1'b1;
                            destination_token_id =
                                destination_mem[destination_index];
                            destination_last_for_term = item == count - 1;
                            do @(posedge clk_core); while (!destination_ready);
                            @(negedge clk_core);
                            destination_valid = 1'b0;
                            destination_index = destination_index + 1;
                        end
                    end
                end
                if (destination_index != events)
                    $fatal(1, "destination stream length mismatch");
            end

            while (!commit_begin_valid) @(posedge clk_core);
            advance_prng();
            stalls = 1 + 32'(prng_q[2:0]);
            repeat (stalls) @(posedge clk_core);
            if (commit_context_id != 0 || commit_head_id != head_value ||
                commit_tag != tag_value ||
                commit_mode_is_csr != (format_value != FORMAT_RAW) ||
                32'(commit_payload_bits) != payload_bits)
                $fatal(1, "commit begin metadata mismatch tag=%h", tag_value);
            @(negedge clk_core);
            commit_begin_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            commit_begin_ready = 1'b0;

            for (int word = 0; word < word_count; word = word + 1) begin
                advance_prng();
                stalls = 32'(prng_q[1:0]);
                repeat (stalls) @(posedge clk_core);
                if (!commit_word_valid ||
                    commit_word_data != expected_word_mem[word] ||
                    commit_word_last != (word == word_count - 1))
                    $fatal(1,
                        "commit word mismatch tag=%h word=%0d got=%h expected=%h last=%0d",
                        tag_value, word, commit_word_data,
                        expected_word_mem[word], commit_word_last);
                @(negedge clk_core);
                commit_word_ready = 1'b1;
                @(posedge clk_core);
                @(negedge clk_core);
                commit_word_ready = 1'b0;
            end

            while (!done_valid) @(posedge clk_core);
            if (done_tag != tag_value || done_format != format_value ||
                done_error || 32'(done_word_count) != word_count)
                $fatal(1, "serializer done mismatch tag=%h", tag_value);
            @(negedge clk_core);
            done_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            done_ready = 1'b0;
            observed_commits = observed_commits + 1;
        end
    endtask

    task automatic run_bad_begin;
        begin
            while (!begin_ready) @(posedge clk_core);
            @(negedge clk_core);
            begin_valid = 1'b1;
            begin_context_id = 1'b0;
            begin_head_id = 5'd7;
            begin_tag = 32'h69ff_0007;
            begin_format = FORMAT_IPD32W;
            begin_expected_payload_bits = 16'd129; // IPD must be byte aligned.
            begin_active_classes = 4'd0;
            begin_active_tokens = 8'd0;
            begin_term_count = 8'd0;
            begin_event_count = 13'd0;
            begin_bitmap_term_count = 8'd0;
            begin_fadc_destination_bytes = 13'd0;
            @(posedge clk_core);
            @(negedge clk_core);
            begin_valid = 1'b0;
            repeat (3) @(posedge clk_core);
            if (!done_valid || !done_error || commit_begin_valid)
                $fatal(1, "bad begin was not atomically aborted");
            @(negedge clk_core);
            done_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            done_ready = 1'b0;
        end
    endtask

    task automatic run_bad_bitmap_popcount;
        logic [161:0] bad_bitmap;
        int wait_cycles;
        begin
            while (!begin_ready) @(posedge clk_core);
            @(negedge clk_core);
            begin_valid = 1'b1;
            begin_context_id = 1'b0;
            begin_head_id = 5'd8;
            begin_tag = 32'h69ff_0008;
            begin_format = FORMAT_FADC24;
            begin_expected_payload_bits = 16'd320;
            begin_active_classes = 4'd1;
            begin_active_tokens = 8'd22;
            begin_term_count = 8'd1;
            begin_event_count = 13'd22;
            begin_bitmap_term_count = 8'd1;
            begin_fadc_destination_bytes = 13'd21;
            @(posedge clk_core);
            @(negedge clk_core);
            begin_valid = 1'b0;

            descriptor_valid = 1'b1;
            descriptor_gate_code = 9'd64;
            descriptor_lane_id = 5'd0;
            descriptor_destination_count = 8'd22;
            descriptor_last = 1'b1;
            do @(posedge clk_core); while (!descriptor_ready);
            @(negedge clk_core);
            descriptor_valid = 1'b0;

            bad_bitmap = '0;
            for (int token = 0; token < 21; token = token + 1)
                bad_bitmap[token] = 1'b1;
            destination_bitmap_valid = 1'b1;
            destination_bitmap = bad_bitmap;
            do @(posedge clk_core); while (!destination_bitmap_ready);
            @(negedge clk_core);
            destination_bitmap_valid = 1'b0;
            destination_bitmap = '0;

            wait_cycles = 0;
            while (!done_valid && !commit_begin_valid && wait_cycles < 30) begin
                @(posedge clk_core);
                wait_cycles = wait_cycles + 1;
            end
            if (!done_valid || !done_error || commit_begin_valid)
                $fatal(1, "bad bitmap popcount was not atomically aborted");
            @(negedge clk_core);
            done_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            done_ready = 1'b0;
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        begin_valid = 1'b0;
        begin_context_id = '0;
        begin_head_id = '0;
        begin_tag = '0;
        begin_format = '0;
        begin_expected_payload_bits = '0;
        begin_active_classes = '0;
        begin_active_tokens = '0;
        begin_term_count = '0;
        begin_event_count = '0;
        begin_bitmap_term_count = '0;
        begin_fadc_destination_bytes = '0;
        descriptor_valid = 1'b0;
        descriptor_gate_code = '0;
        descriptor_lane_id = '0;
        descriptor_destination_count = '0;
        descriptor_last = 1'b0;
        destination_valid = 1'b0;
        destination_token_id = '0;
        destination_last_for_term = 1'b0;
        destination_bitmap_valid = 1'b0;
        destination_bitmap = '0;
        raw_token_valid = 1'b0;
        raw_token_id = '0;
        raw_gate_code = '0;
        raw_k_bits = '0;
        commit_begin_ready = 1'b0;
        commit_word_ready = 1'b0;
        done_ready = 1'b0;
        prng_q = 32'h5e21_a119;
        observed_commits = 0;
        repeat (4) @(posedge clk_core);
        rst_core = 1'b0;

        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/ipd_s0_h0/descriptors.memh", descriptor_mem, 0, 31);
        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/ipd_s0_h0/destinations.memh", destination_mem, 0, 126);
        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/ipd_s0_h0/raw_records.memh", raw_record_mem);
        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/ipd_s0_h0/expected_words.memh", expected_word_mem, 0, 33);
        run_case(FORMAT_IPD32W, 32'h6900_0000, 0, 2, 56, 32, 127, 1, 124, 2168, 34);

        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/fadc_s3_h4/descriptors.memh", descriptor_mem, 0, 60);
        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/fadc_s3_h4/destinations.memh", destination_mem, 0, 813);
        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/fadc_s3_h4/raw_records.memh", raw_record_mem);
        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/fadc_s3_h4/expected_words.memh", expected_word_mem, 0, 101);
        run_case(FORMAT_FADC24, 32'h6903_0004, 4, 3, 153, 61, 814, 15, 616, 6520, 102);

        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/raw_s0_h0/raw_records.memh", raw_record_mem);
        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/raw_s0_h0/expected_words.memh", expected_word_mem, 0, 103);
        run_case(FORMAT_RAW, 32'h69f0_0000, 5, 2, 56, 32, 127, 1, 124, 6642, 104);

        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/ipd_empty_s1_h0/raw_records.memh", raw_record_mem);
        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/ipd_empty_s1_h0/expected_words.memh", expected_word_mem, 0, 1);
        run_case(FORMAT_IPD32W, 32'h6901_0000, 6, 0, 0, 0, 0, 0, 0, 128, 2);

        run_bad_begin();
        if (BITMAP_BYPASS_ENABLE != 32'd0)
            run_bad_bitmap_popcount();

        if (observed_commits != 4 ||
            count_heads != 32'(5 + BITMAP_BYPASS_ENABLE) ||
            count_committed_heads != 4 ||
            count_aborted_heads != 32'(1 + BITMAP_BYPASS_ENABLE) ||
            count_committed_words != 242 || !protocol_error ||
            destination_bitmap_ready)
            $fatal(1,
                "serializer counters mismatch commits=%0d heads=%0d committed=%0d aborted=%0d words=%0d error=%0d",
                observed_commits, count_heads, count_committed_heads,
                count_aborted_heads, count_committed_words, protocol_error);
        $display(
            "PASS: typed payload serializer real vectors commits=%0d words=%0d aborts=%0d input_stalls=%0d output_stalls=%0d",
            count_committed_heads, count_committed_words,
            count_aborted_heads, count_input_stall_cycles,
            count_output_stall_cycles);
        $finish;
    end

endmodule

`default_nettype wire
