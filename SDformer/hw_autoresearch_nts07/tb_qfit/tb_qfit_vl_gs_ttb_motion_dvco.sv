`timescale 1ns/1ps
`default_nettype none

module tb_qfit_vl_gs_ttb_motion_dvco;
    localparam int GATE_W = 9;
    localparam int PAYLOAD_W = 16;

    logic clk_core, rst_core;
    logic build_start_valid, build_start_ready, build_raw_mode;
    logic build_update_valid, build_update_ready;
    logic [1:0] build_update_slot;
    logic [GATE_W-1:0] build_update_gate;
    logic build_commit_valid, build_commit_ready;
    logic body_context_valid, body_context_ready, body_context_raw_mode;
    logic body_valid, body_ready;
    logic [1:0] body_slot;
    logic [GATE_W-1:0] body_raw_gate;
    logic [PAYLOAD_W-1:0] body_payload;
    logic body_last;
    logic out_valid, out_ready, out_last;
    logic [GATE_W-1:0] out_gate;
    logic [PAYLOAD_W-1:0] out_payload;
    logic protocol_error;
    logic [31:0] overlap_cycles, build_wait, body_wait;
    integer expected_gate [0:15];
    integer expected_payload [0:15];
    integer expected_count, retired;
    logic [15:0] lfsr_q;

    always #5 clk_core = ~clk_core;
    assign out_ready = lfsr_q[0] || lfsr_q[3];

    qfit_vl_gs_ttb_motion_dvco #(
        .SLOTS(4), .GATE_W(GATE_W), .PAYLOAD_W(PAYLOAD_W)
    ) dut (
        .clk_core(clk_core), .rst_core(rst_core),
        .build_start_valid(build_start_valid),
        .build_start_ready(build_start_ready),
        .build_raw_mode(build_raw_mode),
        .build_update_valid(build_update_valid),
        .build_update_ready(build_update_ready),
        .build_update_slot(build_update_slot),
        .build_update_gate(build_update_gate),
        .build_commit_valid(build_commit_valid),
        .build_commit_ready(build_commit_ready),
        .body_context_valid(body_context_valid),
        .body_context_ready(body_context_ready),
        .body_context_raw_mode(body_context_raw_mode),
        .body_valid(body_valid), .body_ready(body_ready),
        .body_slot(body_slot), .body_raw_gate(body_raw_gate),
        .body_payload(body_payload), .body_last(body_last),
        .out_valid(out_valid), .out_ready(out_ready), .out_gate(out_gate),
        .out_payload(out_payload), .out_last(out_last),
        .protocol_error(protocol_error),
        .perf_overlap_cycles(overlap_cycles),
        .perf_build_wait_bank(build_wait),
        .perf_body_wait_context(body_wait)
    );

    task automatic build_start(input integer raw_mode);
        @(negedge clk_core);
        build_raw_mode = raw_mode != 0;
        build_start_valid = 1'b1;
        do @(posedge clk_core); while (!build_start_ready);
        @(negedge clk_core);
        build_start_valid = 1'b0;
    endtask

    task automatic build_update(input integer slot, input integer gate);
        @(negedge clk_core);
        build_update_slot = 2'(slot);
        build_update_gate = GATE_W'(gate);
        build_update_valid = 1'b1;
        do @(posedge clk_core); while (!build_update_ready);
        @(negedge clk_core);
        build_update_valid = 1'b0;
    endtask

    task automatic build_commit;
        @(negedge clk_core);
        build_commit_valid = 1'b1;
        do @(posedge clk_core); while (!build_commit_ready);
        @(negedge clk_core);
        build_commit_valid = 1'b0;
    endtask

    task automatic body_start(input integer expected_raw);
        @(negedge clk_core);
        body_context_ready = 1'b1;
        do @(posedge clk_core); while (!body_context_valid);
        if (body_context_raw_mode != (expected_raw != 0))
            $fatal(1, "DVCO context模式失配");
        @(negedge clk_core);
        body_context_ready = 1'b0;
    endtask

    task automatic body_term(
        input integer slot,
        input integer raw_gate,
        input integer gate,
        input integer payload,
        input integer last
    );
        expected_gate[expected_count] = gate;
        expected_payload[expected_count] = payload;
        expected_count = expected_count + 1;
        @(negedge clk_core);
        body_slot = 2'(slot);
        body_raw_gate = GATE_W'(raw_gate);
        body_payload = PAYLOAD_W'(payload);
        body_last = last != 0;
        body_valid = 1'b1;
        do @(posedge clk_core); while (!body_ready);
        @(negedge clk_core);
        body_valid = 1'b0;
    endtask

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            lfsr_q <= 16'h91a7;
            retired <= 0;
        end else begin
            lfsr_q <= {
                lfsr_q[14:0],
                lfsr_q[15] ^ lfsr_q[13] ^ lfsr_q[12] ^ lfsr_q[10]
            };
`ifndef __ICARUS__
            if ($past(!rst_core && out_valid && !out_ready))
                assert(out_valid && $stable({out_gate, out_payload, out_last}));
`endif
            if (out_valid && out_ready) begin
                if (integer'(out_gate) != expected_gate[retired]
                    || integer'(out_payload) != expected_payload[retired])
                    $fatal(
                        1,
                        "DVCO输出失配 index=%0d gate=%0d/%0d payload=%0d/%0d",
                        retired, integer'(out_gate), expected_gate[retired],
                        integer'(out_payload), expected_payload[retired]
                    );
                retired <= retired + 1;
            end
        end
    end

    initial begin
        integer timeout;
        clk_core = 1'b0;
        rst_core = 1'b1;
        build_start_valid = 1'b0;
        build_raw_mode = 1'b0;
        build_update_valid = 1'b0;
        build_update_slot = '0;
        build_update_gate = '0;
        build_commit_valid = 1'b0;
        body_context_ready = 1'b0;
        body_valid = 1'b0;
        body_slot = '0;
        body_raw_gate = '0;
        body_payload = '0;
        body_last = 1'b0;
        expected_count = 0;
        repeat (5) @(negedge clk_core);
        rst_core = 1'b0;

        build_start(0);
        build_update(0, 5);
        build_update(1, 9);
        build_commit();
        body_start(0);

        // body消费bank0期间，在bank1构建raw context。
        build_start(1);
        build_commit();
        @(negedge clk_core);
        build_start_valid = 1'b1;
        repeat (3) begin
            @(negedge clk_core);
            if (build_start_ready)
                $fatal(1, "两个bank占满时错误允许覆盖");
        end
        build_start_valid = 1'b0;
        body_term(0, 0, 5, 101, 0);
        body_term(1, 0, 9, 102, 1);

        body_start(1);
        fork
            begin
                build_start(0);
                build_update(0, 31);
                build_commit();
            end
            begin
                body_term(0, 21, 21, 103, 0);
                body_term(0, 22, 22, 104, 1);
            end
        join
        body_start(0);
        body_term(0, 0, 31, 105, 1);

        timeout = 0;
        while (retired != expected_count && timeout < 1000) begin
            @(negedge clk_core);
            timeout = timeout + 1;
        end
        if (timeout >= 1000) $fatal(1, "DVCO退休超时");
        if (protocol_error) $fatal(1, "DVCO合法流触发协议错误");
        if (overlap_cycles == 0 || build_wait == 0)
            $fatal(1, "DVCO未覆盖重叠或bank占满");
        $display(
            "PASS Motion DVCO terms=%0d overlap=%0d bank_wait=%0d",
            retired, overlap_cycles, build_wait
        );
        $finish;
    end
endmodule

`default_nettype wire
