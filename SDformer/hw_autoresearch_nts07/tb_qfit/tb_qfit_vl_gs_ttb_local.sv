`timescale 1ns/1ps
`default_nettype none

module tb_qfit_vl_gs_ttb_local;
    localparam int GATE_W = 9;
    localparam int PAYLOAD_W = 16;

    logic clk_core, rst_core;
    logic start, finish, active, done;
    logic allocator_active, allocator_done;
    logic in_valid, in_ready, in_set, in_last;
    logic [GATE_W-1:0] in_gate;
    logic [PAYLOAD_W-1:0] in_payload;
    logic alloc_update_valid, alloc_update_ready, alloc_update_set;
    logic alloc_update_slot;
    logic [GATE_W-1:0] alloc_update_gate;
    logic alloc_primary_valid, alloc_primary_ready, alloc_primary_set;
    logic alloc_primary_slot, alloc_primary_exception, alloc_primary_last;
    logic [PAYLOAD_W-1:0] alloc_primary_payload;
    logic alloc_exception_valid, alloc_exception_ready;
    logic [GATE_W-1:0] alloc_exception_gate;
    logic dec_update_valid, dec_update_ready;
    logic dec_primary_valid, dec_primary_ready;
    logic dec_exception_valid, dec_exception_ready;
    logic out_valid, out_ready, out_last;
    logic [GATE_W-1:0] out_gate;
    logic [PAYLOAD_W-1:0] out_payload;
    logic alloc_error, dec_error;
    logic [31:0] fills, hits, bypasses;
    logic [31:0] perf_updates, perf_slots, perf_exceptions, perf_stalls;
    logic [15:0] lfsr_q;
    logic force_pattern;
    logic allow_update, allow_primary, allow_exception, allow_out;
    integer expected_gate [0:15];
    integer expected_payload [0:15];
    integer expected_last [0:15];
    integer expected_count, retired;

    always #5 clk_core = ~clk_core;

    assign dec_update_valid = alloc_update_valid
        && (force_pattern ? allow_update : lfsr_q[1]);
    assign alloc_update_ready = dec_update_ready
        && (force_pattern ? allow_update : lfsr_q[1]);
    assign dec_primary_valid = alloc_primary_valid
        && (force_pattern ? allow_primary : lfsr_q[2]);
    assign alloc_primary_ready = dec_primary_ready
        && (force_pattern ? allow_primary : lfsr_q[2]);
    assign dec_exception_valid = alloc_exception_valid
        && (force_pattern ? allow_exception : lfsr_q[3]);
    assign alloc_exception_ready = dec_exception_ready
        && (force_pattern ? allow_exception : lfsr_q[3]);
    assign out_ready = force_pattern
        ? allow_out : (lfsr_q[4] || lfsr_q[7]);

    qfit_vl_gs_ttb_local_allocator #(
        .SETS(2), .SLOTS(2), .GATE_W(GATE_W), .PAYLOAD_W(PAYLOAD_W)
    ) u_allocator (
        .clk_core(clk_core), .rst_core(rst_core),
        .lifecycle_start(start), .lifecycle_end(finish),
        .lifecycle_active(allocator_active),
        .lifecycle_done(allocator_done),
        .in_valid(in_valid), .in_ready(in_ready), .in_set(in_set),
        .in_gate(in_gate), .in_payload(in_payload), .in_last(in_last),
        .update_valid(alloc_update_valid),
        .update_ready(alloc_update_ready),
        .update_set(alloc_update_set), .update_slot(alloc_update_slot),
        .update_gate(alloc_update_gate),
        .primary_valid(alloc_primary_valid),
        .primary_ready(alloc_primary_ready),
        .primary_set(alloc_primary_set), .primary_slot(alloc_primary_slot),
        .primary_use_exception(alloc_primary_exception),
        .primary_payload(alloc_primary_payload),
        .primary_last(alloc_primary_last),
        .exception_valid(alloc_exception_valid),
        .exception_ready(alloc_exception_ready),
        .exception_gate(alloc_exception_gate),
        .protocol_error(alloc_error), .perf_fills(fills),
        .perf_hits(hits), .perf_bypasses(bypasses)
    );

    qfit_vl_gs_ttb_slot_decoder #(
        .SETS(2), .SLOTS(2), .GATE_W(GATE_W), .PAYLOAD_W(PAYLOAD_W)
    ) u_decoder (
        .clk_core(clk_core), .rst_core(rst_core),
        .lifecycle_start(start), .lifecycle_end(finish),
        .lifecycle_active(active), .lifecycle_done(done),
        .update_valid(dec_update_valid), .update_ready(dec_update_ready),
        .update_set(alloc_update_set), .update_slot(alloc_update_slot),
        .update_gate(alloc_update_gate),
        .primary_valid(dec_primary_valid), .primary_ready(dec_primary_ready),
        .primary_set(alloc_primary_set), .primary_slot(alloc_primary_slot),
        .primary_use_exception(alloc_primary_exception),
        .primary_payload(alloc_primary_payload),
        .primary_last(alloc_primary_last),
        .exception_valid(dec_exception_valid),
        .exception_ready(dec_exception_ready),
        .exception_gate(alloc_exception_gate),
        .out_valid(out_valid), .out_ready(out_ready), .out_gate(out_gate),
        .out_payload(out_payload), .out_last(out_last),
        .protocol_error(dec_error), .perf_updates(perf_updates),
        .perf_slot_terms(perf_slots),
        .perf_exception_terms(perf_exceptions),
        .perf_output_stalls(perf_stalls)
    );

    task automatic send_term(
        input integer set_id,
        input integer gate,
        input integer payload,
        input integer last
    );
        expected_gate[expected_count] = gate;
        expected_payload[expected_count] = payload;
        expected_last[expected_count] = last;
        expected_count = expected_count + 1;
        @(negedge clk_core);
        in_set = set_id != 0;
        in_gate = GATE_W'(gate);
        in_payload = PAYLOAD_W'(payload);
        in_last = last != 0;
        in_valid = 1'b1;
        while (!in_ready) @(negedge clk_core);
        @(negedge clk_core);
        in_valid = 1'b0;
    endtask

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            lfsr_q <= 16'h8e17;
            retired <= 0;
        end else begin
            lfsr_q <= {
                lfsr_q[14:0],
                lfsr_q[15] ^ lfsr_q[13] ^ lfsr_q[12] ^ lfsr_q[10]
            };
`ifndef __ICARUS__
            if ($past(!rst_core && out_valid && !out_ready))
                assert (
                    out_valid
                    && $stable({out_gate, out_payload, out_last})
                );
`endif
            if (out_valid && out_ready) begin
                if (
                    integer'(out_gate) != expected_gate[retired]
                    || integer'(out_payload) != expected_payload[retired]
                    || integer'(out_last) != expected_last[retired]
                ) $fatal(1, "Local5 gate重构失配 index=%0d", retired);
                retired <= retired + 1;
            end
        end
    end

    initial begin
        integer timeout;
        clk_core = 1'b0;
        rst_core = 1'b1;
        start = 1'b0;
        finish = 1'b0;
        in_valid = 1'b0;
        in_set = 1'b0;
        in_gate = '0;
        in_payload = '0;
        in_last = 1'b0;
        expected_count = 0;
        force_pattern = 1'b0;
        allow_update = 1'b1;
        allow_primary = 1'b1;
        allow_exception = 1'b1;
        allow_out = 1'b1;
        repeat (5) @(negedge clk_core);
        rst_core = 1'b0;
        @(negedge clk_core);
        start = 1'b1;
        in_valid = 1'b1;
        in_gate = GATE_W'(5);
        #1;
        if (in_ready)
            $fatal(1, "Local5 start周期错误接受term");
        @(negedge clk_core);
        start = 1'b0;
        in_valid = 1'b0;

        send_term(0, 5, 201, 0);
        send_term(0, 9, 202, 0);
        send_term(0, 5, 203, 0);
        force_pattern = 1'b1;
        allow_primary = 1'b1;
        allow_exception = 1'b0;
        send_term(0, 11, 204, 0);
        repeat (3) @(negedge clk_core);
        allow_primary = 1'b0;
        allow_exception = 1'b1;
        repeat (3) @(negedge clk_core);
        force_pattern = 1'b0;
        send_term(1, 21, 205, 1);

        timeout = 0;
        while (retired != expected_count && timeout < 1000) begin
            @(negedge clk_core);
            timeout = timeout + 1;
        end
        if (timeout >= 1000) $fatal(1, "Local5退休超时");
        @(negedge clk_core);
        finish = 1'b1;
        in_valid = 1'b1;
        #1;
        if (in_ready)
            $fatal(1, "Local5 end周期错误接受term");
        @(negedge clk_core);
        if (!done || !allocator_done)
            $fatal(1, "Local5 lifecycle未结束");
        finish = 1'b0;
        in_valid = 1'b0;
        if (alloc_error || dec_error) $fatal(1, "Local5合法流协议错误");
        if (
            fills != 3 || hits != 1 || bypasses != 1
            || perf_updates != 3 || perf_slots != 4
            || perf_exceptions != 1 || perf_stalls == 0
        ) $fatal(1, "Local5性能计数失配");
        $display(
            "PASS VL-GS-TTB Local5 fills=%0d hits=%0d bypass=%0d stalls=%0d",
            fills, hits, bypasses, perf_stalls
        );
        $finish;
    end
endmodule

`default_nettype wire
