`timescale 1ns/1ps
`default_nettype none

module tb_qfit_vl_gs_ttb_abic_trace;
    localparam int SETS = 32;
    localparam int SLOTS = 4;
    localparam int GATE_W = 9;
    localparam int PAYLOAD_W = 16;
    localparam int EXPECTED_TERMS = 1498;

    logic clk_core, rst_core, start, finish, active, done;
    logic update_valid, update_ready;
    logic [4:0] update_set;
    logic [1:0] update_slot;
    logic [GATE_W-1:0] update_gate;
    logic primary_valid, primary_ready;
    logic [4:0] primary_set;
    logic [1:0] primary_slot;
    logic primary_use_exception;
    logic [PAYLOAD_W-1:0] primary_payload;
    logic primary_last;
    logic exception_valid, exception_ready;
    logic [GATE_W-1:0] exception_gate;
    logic out_valid, out_ready, out_last, protocol_error;
    logic [GATE_W-1:0] out_gate;
    logic [PAYLOAD_W-1:0] out_payload;
    logic [31:0] commit_forwards, output_stalls;
    logic tb_valid [0:SETS-1][0:SLOTS-1];
    integer tb_gate [0:SETS-1][0:SLOTS-1];
    integer expected_gate [0:EXPECTED_TERMS-1];
    integer retired;
    integer cycle_count;

    always #5 clk_core = ~clk_core;
    assign out_ready = 1'b1;

    qfit_vl_gs_ttb_abic_decoder #(
        .SETS(SETS), .SLOTS(SLOTS), .GATE_W(GATE_W),
        .PAYLOAD_W(PAYLOAD_W)
    ) dut (
        .clk_core(clk_core), .rst_core(rst_core),
        .lifecycle_start(start), .lifecycle_end(finish),
        .lifecycle_active(active), .lifecycle_done(done),
        .update_valid(update_valid), .update_ready(update_ready),
        .update_set(update_set), .update_slot(update_slot),
        .update_gate(update_gate),
        .primary_valid(primary_valid), .primary_ready(primary_ready),
        .primary_set(primary_set), .primary_slot(primary_slot),
        .primary_use_exception(primary_use_exception),
        .primary_payload(primary_payload), .primary_last(primary_last),
        .exception_valid(exception_valid),
        .exception_ready(exception_ready),
        .exception_gate(exception_gate),
        .out_valid(out_valid), .out_ready(out_ready), .out_gate(out_gate),
        .out_payload(out_payload), .out_last(out_last),
        .protocol_error(protocol_error),
        .perf_commit_forwards(commit_forwards),
        .perf_output_stalls(output_stalls)
    );

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            retired <= 0;
            cycle_count <= 0;
        end else begin
            if (active)
                cycle_count <= cycle_count + 1;
`ifndef __ICARUS__
            if ($past(!rst_core && out_valid && !out_ready))
                assert(out_valid && $stable({out_gate, out_payload, out_last}));
`endif
            if (out_valid && out_ready) begin
                if (integer'(out_payload) != retired
                    || integer'(out_gate) != expected_gate[retired])
                    $fatal(
                        1,
                        "ABIC trace失配 index=%0d gate=%0d/%0d payload=%0d",
                        retired, integer'(out_gate), expected_gate[retired],
                        integer'(out_payload)
                    );
                retired <= retired + 1;
            end
        end
    end

    initial begin
        integer fd, scan, seq, plane, y, x, lane, gate, mask, window_last;
        integer term_count, slot, hit_slot, free_slot;
        reg [8*256-1:0] header;
        clk_core = 1'b0;
        rst_core = 1'b1;
        start = 1'b0;
        finish = 1'b0;
        update_valid = 1'b0;
        update_set = '0;
        update_slot = '0;
        update_gate = '0;
        primary_valid = 1'b0;
        primary_set = '0;
        primary_slot = '0;
        primary_use_exception = 1'b0;
        primary_payload = '0;
        primary_last = 1'b0;
        exception_valid = 1'b0;
        exception_gate = '0;
        for (integer set_id = 0; set_id < SETS; set_id++)
            for (integer way = 0; way < SLOTS; way++) begin
                tb_valid[set_id][way] = 1'b0;
                tb_gate[set_id][way] = 0;
            end
        repeat (5) @(negedge clk_core);
        rst_core = 1'b0;
        @(negedge clk_core);
        start = 1'b1;
        @(negedge clk_core);
        start = 1'b0;

        fd = $fopen(
            "results/qfit_local5_projection_tile_yosys_20260731/ordered_term_trace.csv",
            "r"
        );
        if (fd == 0) $fatal(1, "无法打开Local5 trace");
        scan = $fgets(header, fd);
        term_count = 0;
        while (!$feof(fd)) begin
            scan = $fscanf(
                fd, "%d,%d,%d,%d,%d,%d,%d,%d\n",
                seq, plane, y, x, lane, gate, mask, window_last
            );
            if (scan == 8) begin
                hit_slot = -1;
                free_slot = -1;
                for (slot = 0; slot < SLOTS; slot = slot + 1) begin
                    if (tb_valid[lane][slot] && tb_gate[lane][slot] == gate)
                        hit_slot = slot;
                    if (!tb_valid[lane][slot] && free_slot < 0)
                        free_slot = slot;
                end
                @(negedge clk_core);
                primary_set = 5'(lane);
                primary_payload = PAYLOAD_W'(term_count);
                primary_last = term_count == EXPECTED_TERMS - 1;
                expected_gate[term_count] = gate;
                if (hit_slot >= 0) begin
                    primary_slot = 2'(hit_slot);
                    primary_use_exception = 1'b0;
                    update_valid = 1'b0;
                    exception_valid = 1'b0;
                end else if (free_slot >= 0) begin
                    primary_slot = 2'(free_slot);
                    primary_use_exception = 1'b0;
                    update_valid = 1'b1;
                    update_set = 5'(lane);
                    update_slot = 2'(free_slot);
                    update_gate = GATE_W'(gate);
                    exception_valid = 1'b0;
                    tb_valid[lane][free_slot] = 1'b1;
                    tb_gate[lane][free_slot] = gate;
                end else begin
                    primary_slot = '0;
                    primary_use_exception = 1'b1;
                    update_valid = 1'b0;
                    exception_valid = 1'b1;
                    exception_gate = GATE_W'(gate);
                end
                primary_valid = 1'b1;
                do @(posedge clk_core); while (
                    !primary_ready
                    || (update_valid && !update_ready)
                    || (exception_valid && !exception_ready)
                );
                term_count = term_count + 1;
            end
        end
        $fclose(fd);
        @(negedge clk_core);
        primary_valid = 1'b0;
        update_valid = 1'b0;
        exception_valid = 1'b0;
        while (retired != term_count) @(negedge clk_core);
        @(negedge clk_core);
        finish = 1'b1;
        @(negedge clk_core);
        finish = 1'b0;
        if (!done || protocol_error) $fatal(1, "ABIC生命周期或协议错误");
        if (term_count != EXPECTED_TERMS || commit_forwards != 96)
            $fatal(1, "ABIC计数错误 terms=%0d forwards=%0d", term_count,
                commit_forwards);
        // 1498个term外仅含start、首装载、末退休和end四拍生命周期开销。
        if (cycle_count != EXPECTED_TERMS + 4)
            $fatal(1, "ABIC未达到一term/拍 cycles=%0d", cycle_count);
        $display(
            "PASS Local5 ABIC terms=%0d cycles=%0d forwards=%0d",
            term_count, cycle_count, commit_forwards
        );
        $finish;
    end
endmodule

`default_nettype wire
