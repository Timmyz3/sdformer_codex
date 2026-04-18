module tb_top;
    localparam int LANES = 8;
    localparam int DATA_W = 8;

    logic clk;
    logic rst_n;
    logic in_valid;
    logic in_last;
    logic in_ready;
    logic [LANES*DATA_W-1:0] in_data;
    logic out_valid;
    logic out_last;
    logic out_ready;
    logic [LANES-1:0] out_data;

    top #(
        .LANES(LANES),
        .DATA_W(DATA_W),
        .THRESHOLD(4)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .in_last(in_last),
        .in_ready(in_ready),
        .in_data(in_data),
        .out_valid(out_valid),
        .out_last(out_last),
        .out_ready(out_ready),
        .out_data(out_data)
    );

    always #5 clk = ~clk;

    task automatic drive(input logic [LANES*DATA_W-1:0] sample, input logic [LANES-1:0] expected);
        begin
            @(negedge clk);
            in_valid <= 1'b1;
            in_data <= sample;
            in_last <= 1'b0;
            @(posedge clk);
            #1;
            if (out_valid !== 1'b1) begin
                $display("FAIL: out_valid not asserted");
                $finish;
            end
            if (out_data !== expected) begin
                $display("FAIL: expected %b got %b", expected, out_data);
                $finish;
            end
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        in_valid = 1'b0;
        in_last = 1'b0;
        in_data = '0;
        out_ready = 1'b1;

        repeat (2) @(posedge clk);
        rst_n = 1'b1;

        drive(64'h0807060504030201, 8'b11111000);
        drive(64'h0101010101010101, 8'b00000100);
        drive(64'h0000000004040404, 8'b00001111);

        $display("PASS");
        $finish;
    end
endmodule
