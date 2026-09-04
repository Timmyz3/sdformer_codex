`timescale 1ns/1ps
`default_nettype none

module tb_qfit_vl_gs_ttb_motion;
    localparam int GATE_W = 9;
    localparam int PAYLOAD_W = 16;

    logic clk_core, rst_core;
    logic start, raw_mode, finish, active, done;
    logic encoder_active, encoder_done;
    logic header_valid, header_ready;
    logic [1:0] header_slot;
    logic [GATE_W-1:0] header_gate;
    logic in_valid, in_ready, in_last;
    logic [GATE_W-1:0] in_gate;
    logic [PAYLOAD_W-1:0] in_payload;
    logic enc_update_valid, enc_update_ready;
    logic [1:0] enc_update_slot;
    logic [GATE_W-1:0] enc_update_gate;
    logic enc_primary_valid, enc_primary_ready;
    logic [1:0] enc_primary_slot;
    logic enc_primary_exception, enc_primary_last;
    logic [PAYLOAD_W-1:0] enc_primary_payload;
    logic enc_exception_valid, enc_exception_ready;
    logic [GATE_W-1:0] enc_exception_gate;
    logic dec_update_valid, dec_update_ready;
    logic dec_primary_valid, dec_primary_ready;
    logic dec_exception_valid, dec_exception_ready;
    logic out_valid, out_ready, out_last;
    logic [GATE_W-1:0] out_gate;
    logic [PAYLOAD_W-1:0] out_payload;
    logic enc_error, dec_error;
    logic [31:0] perf_updates, perf_slots, perf_exceptions, perf_stalls;
    logic [15:0] lfsr_q;
    logic force_pattern;
    logic allow_update, allow_primary, allow_exception, allow_out;
    integer expected_gate [0:15];
    integer expected_payload [0:15];
    integer expected_last [0:15];
    integer expected_count, retired;

    always #5 clk_core = ~clk_core;

    assign dec_update_valid = enc_update_valid
        && (force_pattern ? allow_update : lfsr_q[1]);
    assign enc_update_ready = dec_update_ready
        && (force_pattern ? allow_update : lfsr_q[1]);
    assign dec_primary_valid = enc_primary_valid
        && (force_pattern ? allow_primary : lfsr_q[2]);
    assign enc_primary_ready = dec_primary_ready
        && (force_pattern ? allow_primary : lfsr_q[2]);
    assign dec_exception_valid = enc_exception_valid
        && (force_pattern ? allow_exception : lfsr_q[3]);
    assign enc_exception_ready = dec_exception_ready
        && (force_pattern ? allow_exception : lfsr_q[3]);
    assign out_ready = force_pattern
        ? allow_out : (lfsr_q[4] || lfsr_q[7]);

    qfit_vl_gs_ttb_motion_encoder #(
        .SLOTS(4), .GATE_W(GATE_W), .PAYLOAD_W(PAYLOAD_W)
    ) u_encoder (
        .clk_core(clk_core), .rst_core(rst_core),
        .lifecycle_start(start), .lifecycle_end(finish),
        .lifecycle_raw_mode(raw_mode),
        .lifecycle_active(encoder_active), .lifecycle_done(encoder_done),
        .header_valid(header_valid), .header_ready(header_ready),
        .header_slot(header_slot), .header_gate(header_gate),
        .update_valid(enc_update_valid), .update_ready(enc_update_ready),
        .update_slot(enc_update_slot), .update_gate(enc_update_gate),
        .in_valid(in_valid), .in_ready(in_ready), .in_gate(in_gate),
        .in_payload(in_payload), .in_last(in_last),
        .primary_valid(enc_primary_valid),
        .primary_ready(enc_primary_ready),
        .primary_slot(enc_primary_slot),
        .primary_use_exception(enc_primary_exception),
        .primary_payload(enc_primary_payload),
        .primary_last(enc_primary_last),
        .exception_valid(enc_exception_valid),
        .exception_ready(enc_exception_ready),
        .exception_gate(enc_exception_gate),
        .protocol_error(enc_error)
    );

    qfit_vl_gs_ttb_slot_decoder #(
        .SETS(1), .SLOTS(4), .GATE_W(GATE_W), .PAYLOAD_W(PAYLOAD_W)
    ) u_decoder (
        .clk_core(clk_core), .rst_core(rst_core),
        .lifecycle_start(start), .lifecycle_end(finish),
        .lifecycle_active(active), .lifecycle_done(done),
        .update_valid(dec_update_valid), .update_ready(dec_update_ready),
        .update_set('0), .update_slot(enc_update_slot),
        .update_gate(enc_update_gate),
        .primary_valid(dec_primary_valid), .primary_ready(dec_primary_ready),
        .primary_set('0), .primary_slot(enc_primary_slot),
        .primary_use_exception(enc_primary_exception),
        .primary_payload(enc_primary_payload),
        .primary_last(enc_primary_last),
        .exception_valid(dec_exception_valid),
        .exception_ready(dec_exception_ready),
        .exception_gate(enc_exception_gate),
        .out_valid(out_valid), .out_ready(out_ready), .out_gate(out_gate),
        .out_payload(out_payload), .out_last(out_last),
        .protocol_error(dec_error), .perf_updates(perf_updates),
        .perf_slot_terms(perf_slots),
        .perf_exception_terms(perf_exceptions),
        .perf_output_stalls(perf_stalls)
    );

    task automatic pulse_start(input logic use_raw);
        @(negedge clk_core);
        raw_mode = use_raw;
        start = 1'b1;
        @(negedge clk_core);
        start = 1'b0;
    endtask

    task automatic send_header(input integer slot, input integer gate);
        @(negedge clk_core);
        header_slot = 2'(slot);
        header_gate = GATE_W'(gate);
        header_valid = 1'b1;
        while (!header_ready) @(negedge clk_core);
        @(negedge clk_core);
        header_valid = 1'b0;
    endtask

    task automatic send_term(
        input integer gate, input integer payload, input integer last
    );
        expected_gate[expected_count] = gate;
        expected_payload[expected_count] = payload;
        expected_last[expected_count] = last;
        expected_count = expected_count + 1;
        @(negedge clk_core);
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
            lfsr_q <= 16'hd431;
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
                ) $fatal(1, "Motion gate重构失配 index=%0d", retired);
                retired <= retired + 1;
            end
        end
    end

    initial begin
        integer timeout;
        clk_core = 1'b0;
        rst_core = 1'b1;
        start = 1'b0;
        raw_mode = 1'b0;
        finish = 1'b0;
        header_valid = 1'b0;
        header_slot = '0;
        header_gate = '0;
        in_valid = 1'b0;
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
        raw_mode = 1'b0;
        start = 1'b1;
        in_valid = 1'b1;
        in_gate = GATE_W'(7);
        #1;
        if (in_ready)
            $fatal(1, "Motion start周期错误接受term");
        @(negedge clk_core);
        start = 1'b0;
        in_valid = 1'b0;
        send_header(0, 7);
        send_header(1, 13);
        send_term(7, 101, 0);
        send_term(13, 102, 1);
        while (retired != 2)
            @(negedge clk_core);
        @(negedge clk_core);
        finish = 1'b1;
        in_valid = 1'b1;
        #1;
        if (in_ready)
            $fatal(1, "Motion end周期错误接受term");
        @(negedge clk_core);
        if (!done || !encoder_done)
            $fatal(1, "Motion fast lifecycle未结束");
        finish = 1'b0;
        in_valid = 1'b0;
        if (
            perf_updates != 2 || perf_slots != 2 || perf_exceptions != 0
        ) $fatal(1, "Motion fast计数失配");

        pulse_start(1'b1);
        force_pattern = 1'b1;
        allow_primary = 1'b0;
        allow_exception = 1'b1;
        send_term(99, 103, 0);
        repeat (3) @(negedge clk_core);
        allow_primary = 1'b1;
        allow_exception = 1'b0;
        allow_out = 1'b0;
        repeat (3) @(negedge clk_core);
        allow_out = 1'b1;
        repeat (2) @(negedge clk_core);
        force_pattern = 1'b0;
        send_term(101, 104, 1);

        timeout = 0;
        while (retired != expected_count && timeout < 1000) begin
            @(negedge clk_core);
            timeout = timeout + 1;
        end
        if (timeout >= 1000) $fatal(1, "Motion退休超时");
        @(negedge clk_core);
        finish = 1'b1;
        in_valid = 1'b1;
        #1;
        if (in_ready)
            $fatal(1, "Motion raw end周期错误接受term");
        @(negedge clk_core);
        if (!done || !encoder_done)
            $fatal(1, "Motion raw lifecycle未结束");
        finish = 1'b0;
        in_valid = 1'b0;
        if (enc_error || dec_error) $fatal(1, "Motion合法流协议错误");
        if (
            perf_updates != 0 || perf_slots != 0 || perf_exceptions != 2
            || perf_stalls == 0
        ) $fatal(1, "Motion raw性能计数失配");
        $display(
            "PASS VL-GS-TTB Motion updates=%0d slots=%0d raw=%0d stalls=%0d",
            perf_updates, perf_slots, perf_exceptions, perf_stalls
        );
        $finish;
    end
endmodule

`default_nettype wire
