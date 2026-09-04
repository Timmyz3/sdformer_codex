`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_canonical_head_workspace_c0 #(
    parameter int DESTINATION_SCAN_MODE = 1,
    parameter int EXPLICIT_BITMAP_BANK_ENABLE = 0
);
    localparam logic [1:0] FORMAT_RAW = 2'd0;
    localparam logic [1:0] FORMAT_IPD32W = 2'd1;
    localparam logic [1:0] FORMAT_FADC24 = 2'd2;

    logic clk_core, rst_core;
    logic head_begin_valid, head_begin_ready, head_context_id;
    logic [4:0] head_id;
    logic [31:0] head_tag;
    logic token_valid, token_ready;
    logic [7:0] token_id;
    logic [8:0] token_gate_code;
    logic [31:0] token_k_bits;
    logic token_last;
    logic metadata_valid, metadata_ready, metadata_context_id;
    logic [4:0] metadata_head_id;
    logic [31:0] metadata_tag;
    logic [3:0] metadata_active_classes;
    logic [7:0] metadata_active_tokens, metadata_term_count;
    logic [12:0] metadata_event_count;
    logic [7:0] metadata_bitmap_term_count;
    logic [12:0] metadata_fadc_destination_bytes;
    logic metadata_overflow, raw_capture_error;
    logic emit_start_valid, emit_start_ready;
    logic [1:0] emit_start_format;
    logic descriptor_valid, descriptor_ready;
    logic [8:0] descriptor_gate_code;
    logic [4:0] descriptor_lane_id;
    logic [7:0] descriptor_destination_count;
    logic descriptor_last;
    logic destination_valid, destination_ready;
    logic [7:0] destination_token_id;
    logic destination_last_for_term;
    logic destination_bitmap_valid, destination_bitmap_ready;
    logic [161:0] destination_bitmap;
    logic raw_token_valid, raw_token_ready;
    logic [7:0] raw_token_id;
    logic [8:0] raw_gate_code;
    logic [31:0] raw_k_bits;
    logic emit_done_valid, emit_done_ready;
    logic [31:0] emit_done_tag;
    logic emit_done_error, protocol_error;
    logic [31:0] count_heads, count_raw_fallback_heads;
    logic [31:0] count_emitted_terms, count_emitted_destinations;
    logic [31:0] count_destination_scan_cycles;
    logic [31:0] count_output_stall_cycles;

    logic [40:0] raw_record_mem [0:161];
    logic [23:0] descriptor_mem [0:127];
    logic [7:0] destination_mem [0:8191];
    logic [31:0] prng_q;

    gatestack_canonical_head_workspace_c0 #(
        .DESTINATION_SCAN_MODE(DESTINATION_SCAN_MODE),
        .EXPLICIT_BITMAP_BANK_ENABLE(EXPLICIT_BITMAP_BANK_ENABLE)
    ) dut (.*);
    always #5 clk_core <= ~clk_core;

    task automatic advance_prng;
        begin
            prng_q = {prng_q[30:0],
                prng_q[31] ^ prng_q[21] ^ prng_q[1] ^ prng_q[0]};
        end
    endtask

    task automatic begin_head(
        input logic [31:0] tag_value,
        input logic [4:0] head_value
    );
        begin
            @(negedge clk_core);
            head_begin_valid = 1'b1;
            head_context_id = 1'b0;
            head_id = head_value;
            head_tag = tag_value;
            do @(posedge clk_core); while (!head_begin_ready);
            @(negedge clk_core);
            head_begin_valid = 1'b0;
        end
    endtask

    task automatic send_loaded_tokens;
        begin
            for (int token = 0; token < 162; token = token + 1) begin
                advance_prng();
                repeat (32'(prng_q[1:0])) @(posedge clk_core);
                @(negedge clk_core);
                token_valid = 1'b1;
                token_id = 8'(token);
                token_k_bits = raw_record_mem[token][31:0];
                token_gate_code = raw_record_mem[token][40:32];
                token_last = token == 161;
                do @(posedge clk_core); while (!token_ready);
                @(negedge clk_core);
                token_valid = 1'b0;
            end
        end
    endtask

    task automatic accept_and_check_metadata(
        input logic [31:0] tag_value,
        input logic [4:0] head_value,
        input logic [3:0] active_classes,
        input logic [7:0] active_tokens,
        input logic [7:0] terms,
        input logic [12:0] events,
        input logic [7:0] bitmap_terms,
        input logic [12:0] fadc_bytes,
        input logic expect_overflow
    );
        begin
            while (!metadata_valid) @(posedge clk_core);
            if (metadata_context_id != 0 || metadata_head_id != head_value ||
                metadata_tag != tag_value ||
                metadata_active_classes != active_classes ||
                metadata_active_tokens != active_tokens ||
                metadata_term_count != terms ||
                metadata_event_count != events ||
                metadata_bitmap_term_count != bitmap_terms ||
                metadata_fadc_destination_bytes != fadc_bytes ||
                metadata_overflow != expect_overflow || raw_capture_error)
                $fatal(1, "workspace metadata mismatch tag=%h", tag_value);
            @(negedge clk_core);
            metadata_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            metadata_ready = 1'b0;
        end
    endtask

    task automatic start_emit(input logic [1:0] format_value);
        begin
            @(negedge clk_core);
            emit_start_valid = 1'b1;
            emit_start_format = format_value;
            do @(posedge clk_core); while (!emit_start_ready);
            @(negedge clk_core);
            emit_start_valid = 1'b0;
        end
    endtask

    task automatic check_compressed_emit(
        input logic [31:0] tag_value,
        input int terms,
        input int events
    );
        int destination_index;
        int destination_term;
        int destination_seen;
        logic expected_destination_last;
        begin
            for (int term = 0; term < terms; term = term + 1) begin
                while (!descriptor_valid) @(posedge clk_core);
                advance_prng();
                repeat (32'(prng_q[2:1])) @(posedge clk_core);
                if (descriptor_gate_code != descriptor_mem[term][8:0] ||
                    descriptor_lane_id != descriptor_mem[term][13:9] ||
                    descriptor_destination_count != descriptor_mem[term][21:14] ||
                    descriptor_last != (term == terms - 1))
                    $fatal(1, "descriptor mismatch term=%0d", term);
                @(negedge clk_core);
                descriptor_ready = 1'b1;
                @(posedge clk_core);
                @(negedge clk_core);
                descriptor_ready = 1'b0;
            end

            destination_index = 0;
            destination_term = 0;
            destination_seen = 0;
            while (destination_index < events) begin
                while (!destination_valid) @(posedge clk_core);
                advance_prng();
                repeat (32'(prng_q[1:0])) @(posedge clk_core);
                expected_destination_last =
                    destination_seen + 1 ==
                    32'(descriptor_mem[destination_term][21:14]);
                if (destination_token_id != destination_mem[destination_index] ||
                    destination_last_for_term != expected_destination_last)
                    $fatal(1,
                        "destination mismatch index=%0d term=%0d seen=%0d actual_token=%0d expected_token=%0d actual_last=%0b expected_last=%0b",
                        destination_index, destination_term, destination_seen,
                        destination_token_id,
                        destination_mem[destination_index],
                        destination_last_for_term,
                        expected_destination_last);
                @(negedge clk_core);
                destination_ready = 1'b1;
                @(posedge clk_core);
                destination_index = destination_index + 1;
                if (expected_destination_last) begin
                    destination_term = destination_term + 1;
                    destination_seen = 0;
                end else begin
                    destination_seen = destination_seen + 1;
                end
                @(negedge clk_core);
                destination_ready = 1'b0;
            end
            while (!emit_done_valid) @(posedge clk_core);
            if (emit_done_tag != tag_value || emit_done_error)
                $fatal(1, "compressed emit done mismatch");
            @(negedge clk_core);
            emit_done_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            emit_done_ready = 1'b0;
        end
    endtask

    task automatic check_raw_emit(
        input logic [31:0] tag_value,
        input logic synthetic_overflow
    );
        logic [40:0] expected;
        begin
            for (int token = 0; token < 162; token = token + 1) begin
                while (!raw_token_valid) @(posedge clk_core);
                advance_prng();
                repeat (32'(prng_q[1:0])) @(posedge clk_core);
                expected = synthetic_overflow ?
                    {(token < 5) ? 9'(token + 1) : 9'd0,
                     (token < 5) ? 32'd1 : 32'd0} : raw_record_mem[token];
                if (raw_token_id != 8'(token) ||
                    {raw_gate_code, raw_k_bits} != expected)
                    $fatal(1, "RAW replay mismatch token=%0d", token);
                @(negedge clk_core);
                raw_token_ready = 1'b1;
                @(posedge clk_core);
                @(negedge clk_core);
                raw_token_ready = 1'b0;
            end
            while (!emit_done_valid) @(posedge clk_core);
            if (emit_done_tag != tag_value || emit_done_error)
                $fatal(1, "RAW emit done mismatch");
            @(negedge clk_core);
            emit_done_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            emit_done_ready = 1'b0;
        end
    endtask

    task automatic send_synthetic_overflow_tokens;
        begin
            for (int token = 0; token < 162; token = token + 1) begin
                @(negedge clk_core);
                token_valid = 1'b1;
                token_id = 8'(token);
                token_gate_code = (token < 5) ? 9'(token + 1) : 9'd0;
                token_k_bits = (token < 5) ? 32'd1 : 32'd0;
                token_last = token == 161;
                do @(posedge clk_core); while (!token_ready);
                @(negedge clk_core);
                token_valid = 1'b0;
            end
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        head_begin_valid = 1'b0;
        token_valid = 1'b0;
        metadata_ready = 1'b0;
        emit_start_valid = 1'b0;
        descriptor_ready = 1'b0;
        destination_ready = 1'b0;
        destination_bitmap_ready = 1'b0;
        raw_token_ready = 1'b0;
        emit_done_ready = 1'b0;
        head_context_id = '0;
        head_id = '0;
        head_tag = '0;
        token_id = '0;
        token_gate_code = '0;
        token_k_bits = '0;
        token_last = 1'b0;
        emit_start_format = '0;
        prng_q = 32'hc001_c0de;
        repeat (4) @(posedge clk_core);
        rst_core = 1'b0;

        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/ipd_s0_h0/raw_records.memh", raw_record_mem);
        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/ipd_s0_h0/descriptors.memh", descriptor_mem, 0, 31);
        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/ipd_s0_h0/destinations.memh", destination_mem, 0, 126);
        begin_head(32'h6900_0000, 5'd0);
        send_loaded_tokens();
        accept_and_check_metadata(32'h6900_0000, 5'd0, 4'd2, 8'd56,
            8'd32, 13'd127, 8'd1, 13'd124, 1'b0);
        start_emit(FORMAT_IPD32W);
        check_compressed_emit(32'h6900_0000, 32, 127);

        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/fadc_s3_h4/raw_records.memh", raw_record_mem);
        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/fadc_s3_h4/descriptors.memh", descriptor_mem, 0, 60);
        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/fadc_s3_h4/destinations.memh", destination_mem, 0, 813);
        begin_head(32'h6903_0004, 5'd4);
        send_loaded_tokens();
        accept_and_check_metadata(32'h6903_0004, 5'd4, 4'd3, 8'd153,
            8'd61, 13'd814, 8'd15, 13'd616, 1'b0);
        start_emit(FORMAT_FADC24);
        check_compressed_emit(32'h6903_0004, 61, 814);

        begin_head(32'h69ff_0001, 5'd7);
        send_synthetic_overflow_tokens();
        accept_and_check_metadata(32'h69ff_0001, 5'd7, 4'd4, 8'd5,
            8'd4, 13'd5, 8'd0, 13'd4, 1'b1);
        start_emit(FORMAT_RAW);
        check_raw_emit(32'h69ff_0001, 1'b1);

        repeat (3) @(posedge clk_core);
        if (destination_bitmap_valid && destination_bitmap == '0 ||
            protocol_error || count_heads != 3 ||
            count_raw_fallback_heads != 1 || count_emitted_terms != 93 ||
            count_emitted_destinations != 941 ||
            count_destination_scan_cycles !=
                ((DESTINATION_SCAN_MODE == 0) ? 32'd10832 : 32'd941) ||
            count_output_stall_cycles == 0)
            $fatal(1, "workspace final counters mismatch");
        $display(
            "PASS: canonical workspace heads=%0d terms=%0d destinations=%0d scan_cycles=%0d output_stalls=%0d raw_fallback=%0d",
            count_heads, count_emitted_terms, count_emitted_destinations,
            count_destination_scan_cycles, count_output_stall_cycles,
            count_raw_fallback_heads);
        $finish;
    end

endmodule

`default_nettype wire
