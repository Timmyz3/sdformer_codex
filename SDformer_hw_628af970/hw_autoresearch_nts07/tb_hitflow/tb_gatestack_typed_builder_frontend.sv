`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_typed_builder_frontend;
    localparam logic [1:0] FORMAT_RAW = 2'd0;
    localparam logic [1:0] FORMAT_IPD32W = 2'd1;
    localparam logic [1:0] FORMAT_FADC24 = 2'd2;

    logic clk_core;
    logic rst_core;
    logic head_start_valid;
    logic head_start_ready;
    logic [31:0] head_start_tag;
    logic [3:0] head_start_active_classes;
    logic term_valid;
    logic term_ready;
    logic [7:0] term_destination_count;
    logic head_end_valid;
    logic head_end_ready;
    logic head_end_builder_error;
    logic decision_valid;
    logic decision_ready;
    logic [31:0] decision_tag;
    logic [1:0] decision_format;
    logic [2:0] decision_reason;
    logic [15:0] decision_payload_bits;
    logic [7:0] decision_word_count;
    logic [15:0] decision_ipd_payload_bytes;
    logic [15:0] decision_fadc_payload_bytes;
    logic [7:0] decision_term_count;
    logic [12:0] decision_event_count;
    logic [7:0] decision_bitmap_term_count;
    logic decision_metadata_overflow;
    logic [31:0] count_heads;
    logic [31:0] count_terms;
    logic [31:0] count_invalid_terms;
    logic [31:0] count_metadata_overflows;
    logic [31:0] prng_q;
    int expected_heads;
    int expected_terms;
    int expected_overflows;

    gatestack_typed_builder_frontend dut (.*);

    always #5 clk_core <= ~clk_core;

    task automatic advance_prng;
        begin
            prng_q = {prng_q[30:0],
                prng_q[31] ^ prng_q[21] ^ prng_q[1] ^ prng_q[0]};
        end
    endtask

    task automatic run_uniform_case(
        input int classes,
        input int terms,
        input int destinations,
        input logic builder_error,
        input logic [31:0] tag
    );
        int event_sum;
        int fadc_dest_sum;
        int term_sat;
        int event_sat;
        int bitmap_terms;
        int ipd_bytes;
        int fadc_bytes;
        int expected_bits;
        int expected_words;
        int stalls;
        logic overflow;
        logic [1:0] expected_format;
        logic [2:0] expected_reason;
        begin
            event_sum = terms * destinations;
            fadc_dest_sum = terms * ((destinations > 21) ? 21 : destinations);
            term_sat = (terms > 255) ? 255 : terms;
            event_sat = (event_sum > 8191) ? 8191 : event_sum;
            bitmap_terms = (destinations > 21) ? term_sat : 0;
            overflow = terms > 255 || event_sum > 8191 || builder_error;
            ipd_bytes = 16 + 8 * ((term_sat + 1) / 2) + event_sat;
            fadc_bytes = 16 + 3 * term_sat + fadc_dest_sum;

            if (overflow) begin
                expected_format = FORMAT_RAW;
                expected_reason = 3'd4;
                expected_bits = 6642;
                expected_overflows = expected_overflows + 1;
            end else if (classes <= 4 && ipd_bytes <= 832) begin
                expected_format = FORMAT_IPD32W;
                expected_reason = 3'd0;
                expected_bits = ipd_bytes * 8;
            end else if (fadc_bytes <= 832) begin
                expected_format = FORMAT_FADC24;
                expected_reason = (classes <= 4) ? 3'd2 : 3'd1;
                expected_bits = fadc_bytes * 8;
            end else begin
                expected_format = FORMAT_RAW;
                expected_reason = 3'd3;
                expected_bits = 6642;
            end
            expected_words = (expected_bits + 63) / 64;

            while (!head_start_ready) @(posedge clk_core);
            @(negedge clk_core);
            head_start_valid = 1'b1;
            head_start_tag = tag;
            head_start_active_classes = 4'(classes);
            @(posedge clk_core);
            @(negedge clk_core);
            head_start_valid = 1'b0;

            for (int index = 0; index < terms; index = index + 1) begin
                term_valid = 1'b1;
                term_destination_count = 8'(destinations);
                do @(posedge clk_core); while (!term_ready);
                @(negedge clk_core);
                term_valid = 1'b0;
                expected_terms = expected_terms + 1;
            end

            while (!head_end_ready) @(posedge clk_core);
            @(negedge clk_core);
            head_end_valid = 1'b1;
            head_end_builder_error = builder_error;
            @(posedge clk_core);
            @(negedge clk_core);
            head_end_valid = 1'b0;
            head_end_builder_error = 1'b0;

            advance_prng();
            stalls = 1 + 32'(prng_q[2:0]);
            decision_ready = 1'b0;
            repeat (stalls) @(posedge clk_core);
            if (!decision_valid || decision_tag != tag ||
                decision_format != expected_format ||
                decision_reason != expected_reason ||
                32'(decision_payload_bits) != expected_bits ||
                32'(decision_word_count) != expected_words ||
                32'(decision_ipd_payload_bytes) != ipd_bytes ||
                32'(decision_fadc_payload_bytes) != fadc_bytes ||
                32'(decision_term_count) != term_sat ||
                32'(decision_event_count) != event_sat ||
                32'(decision_bitmap_term_count) != bitmap_terms ||
                decision_metadata_overflow != overflow) begin
                $fatal(1,
                    "typed policy mismatch tag=%h fmt=%0d/%0d reason=%0d/%0d ipd=%0d/%0d fadc=%0d/%0d terms=%0d/%0d events=%0d/%0d overflow=%0d/%0d",
                    tag, decision_format, expected_format,
                    decision_reason, expected_reason,
                    decision_ipd_payload_bytes, ipd_bytes,
                    decision_fadc_payload_bytes, fadc_bytes,
                    decision_term_count, term_sat,
                    decision_event_count, event_sat,
                    decision_metadata_overflow, overflow);
            end
            @(negedge clk_core);
            decision_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            decision_ready = 1'b0;
            expected_heads = expected_heads + 1;
        end
    endtask

    initial begin
        int terms;
        int destinations;
        int classes;
        clk_core = 1'b0;
        rst_core = 1'b1;
        head_start_valid = 1'b0;
        head_start_tag = '0;
        head_start_active_classes = '0;
        term_valid = 1'b0;
        term_destination_count = '0;
        head_end_valid = 1'b0;
        head_end_builder_error = 1'b0;
        decision_ready = 1'b0;
        prng_q = 32'hb17d_2026;
        expected_heads = 0;
        expected_terms = 0;
        expected_overflows = 0;
        repeat (4) @(posedge clk_core);
        rst_core = 1'b0;

        run_uniform_case(0, 0, 1, 1'b0, 32'hb100_0000);
        run_uniform_case(4, 6, 132, 1'b0, 32'hb100_0001); // IPD exactly 832 B
        run_uniform_case(4, 6, 133, 1'b0, 32'hb100_0002); // IPD overflow, FADC fit
        run_uniform_case(5, 1, 1, 1'b0, 32'hb100_0003);   // complete class>4 stream
        run_uniform_case(4, 255, 1, 1'b0, 32'hb100_0004); // both compressed formats overflow
        run_uniform_case(4, 256, 1, 1'b0, 32'hb100_0005); // counter overflow fail-safe
        run_uniform_case(4, 1, 1, 1'b1, 32'hb100_0006);   // incomplete upstream stream

        for (int trial = 0; trial < 300; trial = trial + 1) begin
            advance_prng();
            terms = 32'(prng_q[7:0]) % 129;
            advance_prng();
            destinations = 1 + (32'(prng_q[7:0]) % 162);
            advance_prng();
            classes = 32'(prng_q[2:0]);
            run_uniform_case(classes, terms, destinations, 1'b0,
                             32'hb200_0000 + trial);
        end

        if (count_heads != expected_heads ||
            count_terms != expected_terms || count_invalid_terms != 0 ||
            count_metadata_overflows != expected_overflows) begin
            $fatal(1,
                "typed builder counters mismatch heads=%0d/%0d terms=%0d/%0d invalid=%0d overflow=%0d/%0d",
                count_heads, expected_heads, count_terms, expected_terms,
                count_invalid_terms, count_metadata_overflows,
                expected_overflows);
        end

        $display(
            "PASS: typed builder frontend heads=%0d terms=%0d metadata_overflows=%0d",
            count_heads, count_terms, count_metadata_overflows);
        $finish;
    end

endmodule

`default_nettype wire
