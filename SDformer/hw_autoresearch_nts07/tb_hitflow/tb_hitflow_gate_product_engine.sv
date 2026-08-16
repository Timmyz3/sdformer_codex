`timescale 1ns/1ps
`default_nettype none

module tb_hitflow_gate_product_engine;
    localparam int TOKENS = 4;
    localparam int GATE_W = 9;
    localparam int WEIGHT_W = 8;
    localparam int PRODUCT_W = 17;
    localparam int OUT_TILE = 3;
    localparam int INPUT_CH_W = 6;
    localparam int OUTPUT_TILE_W = 4;
    localparam int TAG_W = 16;

    logic clk_core = 1'b0;
    logic rst_core;
    logic term_valid;
    logic term_ready;
    logic [TAG_W-1:0] term_tag;
    logic [GATE_W-1:0] term_gate_code;
    logic [INPUT_CH_W-1:0] term_input_channel;
    logic [OUTPUT_TILE_W-1:0] term_output_tile;
    logic [TOKENS-1:0] term_destination_bitmap;
    logic weight_req_valid;
    logic weight_req_ready;
    logic [TAG_W-1:0] weight_req_tag;
    logic [INPUT_CH_W-1:0] weight_req_input_channel;
    logic [OUTPUT_TILE_W-1:0] weight_req_output_tile;
    logic weight_rsp_valid;
    logic weight_rsp_ready;
    logic [TAG_W-1:0] weight_rsp_tag;
    logic [INPUT_CH_W-1:0] weight_rsp_input_channel;
    logic [OUTPUT_TILE_W-1:0] weight_rsp_output_tile;
    logic [(OUT_TILE*WEIGHT_W)-1:0] weight_rsp_weights;
    logic product_valid;
    logic product_ready;
    logic [TAG_W-1:0] product_tag;
    logic [INPUT_CH_W-1:0] product_input_channel;
    logic [OUTPUT_TILE_W-1:0] product_output_tile;
    logic [TOKENS-1:0] product_destination_bitmap;
    logic [(OUT_TILE*PRODUCT_W)-1:0] product_values;
    logic protocol_error;
    logic [31:0] count_terms;
    logic [31:0] count_weight_requests;
    logic [31:0] count_products;
    logic [31:0] count_weight_wait_cycles;
    logic [31:0] count_output_stall_cycles;
    logic signed [PRODUCT_W-1:0] observed_product;

    initial begin
        forever #1 clk_core = ~clk_core;
    end

    hitflow_gate_product_engine #(
        .TOKENS(TOKENS), .GATE_W(GATE_W), .WEIGHT_W(WEIGHT_W),
        .PRODUCT_W(PRODUCT_W), .OUT_TILE(OUT_TILE),
        .INPUT_CH_W(INPUT_CH_W), .OUTPUT_TILE_W(OUTPUT_TILE_W), .TAG_W(TAG_W)
    ) dut (.*);

    task automatic check(input logic condition, input string message);
        if (!condition) $fatal(1, "%s", message);
    endtask

    task automatic send_term(
        input logic [15:0] tag,
        input logic [8:0] gate,
        input logic [5:0] channel,
        input logic [3:0] tile,
        input logic [3:0] bitmap
    );
        begin
            term_tag = tag;
            term_gate_code = gate;
            term_input_channel = channel;
            term_output_tile = tile;
            term_destination_bitmap = bitmap;
            term_valid = 1'b1;
            do @(posedge clk_core); while (!term_ready);
            #0.1 term_valid = 1'b0;
        end
    endtask

    initial begin
        rst_core = 1'b1;
        term_valid = 1'b0;
        term_tag = '0;
        term_gate_code = '0;
        term_input_channel = '0;
        term_output_tile = '0;
        term_destination_bitmap = '0;
        weight_req_ready = 1'b0;
        weight_rsp_valid = 1'b0;
        weight_rsp_tag = '0;
        weight_rsp_input_channel = '0;
        weight_rsp_output_tile = '0;
        weight_rsp_weights = '0;
        product_ready = 1'b0;
        repeat (3) @(posedge clk_core);
        #0.1 rst_core = 1'b0;

        $display("阶段1：权重请求反压、边界乘积和产品输出稳定");
        send_term(16'h1111, 9'd256, 6'd7, 4'd2, 4'b1011);
        check(weight_req_valid, "term后必须发权重请求");
        check(weight_req_tag == 16'h1111 && weight_req_input_channel == 7 &&
              weight_req_output_tile == 2, "权重请求tag错误");
        repeat (2) begin
            @(posedge clk_core);
            #0.1;
            check(weight_req_valid && weight_req_tag == 16'h1111,
                  "权重请求反压期间不稳定");
        end
        weight_req_ready = 1'b1;
        @(posedge clk_core);
        #0.1 weight_req_ready = 1'b0;
        repeat (2) begin
            @(posedge clk_core);
            #0.1;
            check(!weight_rsp_ready, "无权重响应时不得误握手");
        end

        weight_rsp_valid = 1'b1;
        weight_rsp_tag = 16'h1112;
        weight_rsp_input_channel = 7;
        weight_rsp_output_tile = 2;
        weight_rsp_weights[0 +: 8] = -128;
        weight_rsp_weights[8 +: 8] = 127;
        weight_rsp_weights[16 +: 8] = -1;
        #0.1;
        check(protocol_error && !weight_rsp_ready, "错tag权重响应必须拒绝");
        weight_rsp_tag = 16'h1111;
        do @(posedge clk_core); while (!weight_rsp_ready);
        #0.1 weight_rsp_valid = 1'b0;

        check(product_valid, "权重响应后必须产生product");
        check(product_tag == 16'h1111 && product_destination_bitmap == 4'b1011,
              "product元数据错误");
        check(product_input_channel == 7 && product_output_tile == 2,
              "product通道或输出分块错误");
        observed_product = $signed(product_values[0 +: PRODUCT_W]);
        check(observed_product == -17'sd32768, "256乘-128边界错误");
        observed_product = $signed(product_values[PRODUCT_W +: PRODUCT_W]);
        check(observed_product == 17'sd32512, "256乘127边界错误");
        observed_product = $signed(product_values[2*PRODUCT_W +: PRODUCT_W]);
        check(observed_product == -17'sd256, "256乘-1错误");
        repeat (2) begin
            @(posedge clk_core);
            #0.1;
            check(product_valid && product_destination_bitmap == 4'b1011,
                  "product反压期间不稳定");
        end
        product_ready = 1'b1;
        @(posedge clk_core);
        #0.1 product_ready = 1'b0;

        $display("阶段2：普通正负乘积和计数器");
        send_term(16'h2222, 9'd64, 6'd9, 4'd3, 4'b0100);
        weight_req_ready = 1'b1;
        @(posedge clk_core);
        #0.1 weight_req_ready = 1'b0;
        weight_rsp_valid = 1'b1;
        weight_rsp_tag = 16'h2222;
        weight_rsp_input_channel = 9;
        weight_rsp_output_tile = 3;
        weight_rsp_weights[0 +: 8] = 2;
        weight_rsp_weights[8 +: 8] = -3;
        weight_rsp_weights[16 +: 8] = 0;
        do @(posedge clk_core); while (!weight_rsp_ready);
        #0.1 weight_rsp_valid = 1'b0;
        check(product_input_channel == 9 && product_output_tile == 3,
              "第二组product通道或输出分块错误");
        observed_product = $signed(product_values[0 +: PRODUCT_W]);
        check(observed_product == 17'sd128, "64乘2错误");
        observed_product = $signed(product_values[PRODUCT_W +: PRODUCT_W]);
        check(observed_product == -17'sd192, "64乘-3错误");
        observed_product = $signed(product_values[2*PRODUCT_W +: PRODUCT_W]);
        check(observed_product == 17'sd0, "64乘0错误");
        product_ready = 1'b1;
        @(posedge clk_core);
        #0.1 product_ready = 1'b0;
        check(count_terms == 2 && count_weight_requests == 2 && count_products == 2,
              "product engine计数器错误");
        check(count_weight_wait_cycles >= 2 && count_output_stall_cycles >= 2,
              "反压周期计数器未覆盖预期等待");

        $display("阶段3：零gate和空bitmap必须拒绝");
        term_valid = 1'b1;
        term_gate_code = 0;
        term_destination_bitmap = 4'b0001;
        #0.1;
        check(protocol_error && !term_ready, "零gate term必须拒绝");
        term_gate_code = 1;
        term_destination_bitmap = 0;
        #0.1;
        check(protocol_error && !term_ready, "空bitmap term必须拒绝");
        term_valid = 1'b0;

        $display("PASS: HIT-Flow gate product engine");
        $finish;
    end

endmodule

`default_nettype wire
