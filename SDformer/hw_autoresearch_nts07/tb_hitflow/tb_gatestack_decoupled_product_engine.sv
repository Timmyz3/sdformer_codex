`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_decoupled_product_engine;
    localparam int OUT_TILE = 3;
    localparam int PRODUCT_W = 17;
    logic clk_core;
    logic rst_core;
    logic clear_error;
    logic term_valid;
    logic term_ready;
    logic [15:0] term_tag;
    logic [8:0] term_gate_code;
    logic [5:0] term_input_channel;
    logic [3:0] term_output_tile;
    logic [12:0] term_issue_seq;
    logic weight_req_valid;
    logic weight_req_ready;
    logic [15:0] weight_req_tag;
    logic [5:0] weight_req_input_channel;
    logic [3:0] weight_req_output_tile;
    logic weight_rsp_valid;
    logic weight_rsp_ready;
    logic [15:0] weight_rsp_tag;
    logic [5:0] weight_rsp_input_channel;
    logic [3:0] weight_rsp_output_tile;
    logic [23:0] weight_rsp_weights;
    logic product_valid;
    logic product_ready;
    logic [15:0] product_tag;
    logic [5:0] product_input_channel;
    logic [3:0] product_output_tile;
    logic [12:0] product_issue_seq;
    logic [50:0] product_values;
    logic protocol_error;
    logic [31:0] count_terms;
    logic [31:0] count_weight_requests;
    logic [31:0] count_products;
    logic [31:0] count_weight_wait_cycles;
    logic [31:0] count_output_stall_cycles;
    logic signed [PRODUCT_W-1:0] observed_product;

    gatestack_decoupled_product_engine #(
        .OUT_TILE(OUT_TILE),
        .INPUT_CH_W(6),
        .OUTPUT_TILE_W(4),
        .TAG_W(16)
    ) dut (.*);
    always #5 clk_core <= ~clk_core;

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        clear_error = 1'b0;
        term_valid = 1'b0;
        term_tag = '0;
        term_gate_code = '0;
        term_input_channel = '0;
        term_output_tile = '0;
        term_issue_seq = '0;
        weight_req_ready = 1'b0;
        weight_rsp_valid = 1'b0;
        weight_rsp_tag = '0;
        weight_rsp_input_channel = '0;
        weight_rsp_output_tile = '0;
        weight_rsp_weights = '0;
        product_ready = 1'b0;
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;

        @(negedge clk_core);
        term_tag = 16'h1234;
        term_gate_code = 9'd256;
        term_input_channel = 6'd7;
        term_output_tile = 4'd2;
        term_issue_seq = 13'd19;
        term_valid = 1'b1;
        do @(posedge clk_core); while (!term_ready);
        @(negedge clk_core);
        term_valid = 1'b0;
        repeat (2) @(posedge clk_core);
        if (!weight_req_valid || weight_req_tag != 16'h1234 ||
            weight_req_input_channel != 7 || weight_req_output_tile != 2) begin
            $fatal(1, "weight request mismatch");
        end
        @(negedge clk_core);
        weight_req_ready = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        weight_req_ready = 1'b0;
        repeat (2) @(posedge clk_core);
        @(negedge clk_core);
        weight_rsp_tag = 16'h1234;
        weight_rsp_input_channel = 6'd7;
        weight_rsp_output_tile = 4'd2;
        weight_rsp_weights[7:0] = -8'sd128;
        weight_rsp_weights[15:8] = 8'sd127;
        weight_rsp_weights[23:16] = -8'sd1;
        weight_rsp_valid = 1'b1;
        do @(posedge clk_core); while (!weight_rsp_ready);
        @(negedge clk_core);
        weight_rsp_valid = 1'b0;
        repeat (2) @(posedge clk_core);
        if (!product_valid || product_tag != 16'h1234 ||
            product_input_channel != 7 || product_output_tile != 2 ||
            product_issue_seq != 13'd19) begin
            $fatal(1, "product metadata mismatch");
        end
        observed_product = $signed(product_values[0 +: PRODUCT_W]);
        if (observed_product != -17'sd32768) $fatal(1, "product0 mismatch");
        observed_product = $signed(product_values[PRODUCT_W +: PRODUCT_W]);
        if (observed_product != 17'sd32512) $fatal(1, "product1 mismatch");
        observed_product = $signed(product_values[2*PRODUCT_W +: PRODUCT_W]);
        if (observed_product != -17'sd256) $fatal(1, "product2 mismatch");
        @(negedge clk_core);
        product_ready = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        product_ready = 1'b0;

        // Keep a legal transaction in flight while auditing a bad response.
        @(negedge clk_core);
        term_tag = 16'h5678;
        term_gate_code = 9'd3;
        term_input_channel = 6'd11;
        term_output_tile = 4'd5;
        term_issue_seq = 13'd27;
        term_valid = 1'b1;
        do @(posedge clk_core); while (!term_ready);
        @(negedge clk_core);
        term_valid = 1'b0;
        weight_req_ready = 1'b1;
        do @(posedge clk_core); while (!weight_req_valid);
        @(negedge clk_core);
        weight_req_ready = 1'b0;

        weight_rsp_tag = 16'h5679;
        weight_rsp_input_channel = 6'd11;
        weight_rsp_output_tile = 4'd5;
        weight_rsp_valid = 1'b1;
        @(posedge clk_core);
        if (weight_rsp_ready)
            $fatal(1, "wrong identity response was accepted");
        @(negedge clk_core);
        weight_rsp_valid = 1'b0;
        repeat (2) @(posedge clk_core);
        if (!protocol_error)
            $fatal(1, "first identity error did not set protocol_error");

        @(negedge clk_core);
        weight_rsp_tag = 16'h5678;
        weight_rsp_input_channel = 6'd11;
        weight_rsp_output_tile = 4'd5;
        clear_error = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        clear_error = 1'b0;
        if (protocol_error || !weight_rsp_ready || count_terms != 2 ||
            count_weight_requests != 2 || count_products != 1)
            $fatal(1, "clear_error changed in-flight state or counters");

        weight_rsp_tag = 16'h5678;
        weight_rsp_input_channel = 6'd11;
        weight_rsp_output_tile = 4'd5;
        weight_rsp_weights = {8'sd4, -8'sd3, 8'sd2};
        weight_rsp_valid = 1'b1;
        @(posedge clk_core);
        if (!weight_rsp_ready)
            $fatal(1, "legal response lost after clear_error");
        @(negedge clk_core);
        weight_rsp_valid = 1'b0;
        product_ready = 1'b1;
        do @(posedge clk_core); while (!product_valid);
        @(negedge clk_core);
        product_ready = 1'b0;

        // A different error after clear_error must be observable again.
        term_gate_code = 0;
        term_valid = 1'b1;
        @(posedge clk_core);
        if (term_ready) $fatal(1, "zero gate was accepted");
        @(negedge clk_core);
        term_valid = 1'b0;
        repeat (2) @(posedge clk_core);
        if (!protocol_error || count_terms != 2 ||
            count_weight_requests != 2 || count_products != 2 ||
            count_weight_wait_cycles == 0 || count_output_stall_cycles == 0) begin
            $fatal(1, "product engine counters/error mismatch");
        end
        @(negedge clk_core);
        clear_error = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        clear_error = 1'b0;
        if (protocol_error)
            $fatal(1, "second clear_error did not clear protocol_error");
        $display("PASS: decoupled product engine terms=%0d requests=%0d products=%0d weight_wait=%0d output_stall=%0d",
                 count_terms, count_weight_requests, count_products,
                 count_weight_wait_cycles, count_output_stall_cycles);
        $finish;
    end

    initial begin
        repeat (2000) @(posedge clk_core);
        $fatal(1, "decoupled product TB timeout");
    end
endmodule

`default_nettype wire
