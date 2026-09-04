`timescale 1ns/1ps
`default_nettype none

module tb_qfit_exposure_relation_vault;
    localparam int TOTAL = 450;
    localparam int MAX_HEADS = 24;
    localparam int HEAD_W = $clog2(MAX_HEADS);
    localparam int PTR_W = $clog2(513);

    logic clk_core;
    logic rst_core;
    logic window_start;
    logic head_start;
    logic head_ready;
    logic [HEAD_W-1:0] head_index;
    logic in_valid;
    logic in_ready;
    logic [8:0] in_source_id;
    logic [3:0] in_y;
    logic [3:0] in_x;
    logic [31:0] in_k;
    logic [44:0] in_gates;
    logic [4:0] in_valid_mask;
    logic in_last;
    logic live_valid;
    logic live_ready;
    logic [8:0] live_source_id;
    logic [3:0] live_y;
    logic [3:0] live_x;
    logic [31:0] live_k;
    logic [44:0] live_gates;
    logic [4:0] live_valid_mask;
    logic live_last;
    logic head_done;
    logic head_resident;
    logic head_critical;
    logic head_overflow;
    logic [31:0] head_service_cycles;
    logic [PTR_W-1:0] head_record_count;
    logic replay_start;
    logic replay_cmd_ready;
    logic [HEAD_W-1:0] replay_head_index;
    logic replay_valid;
    logic replay_ready;
    logic [8:0] replay_source_id;
    logic [3:0] replay_y;
    logic [3:0] replay_x;
    logic [31:0] replay_k;
    logic [44:0] replay_gates;
    logic [4:0] replay_valid_mask;
    logic replay_last;
    logic replay_done;
    logic replay_miss;
    logic protocol_error;
    logic [31:0] perf_speculative_writes;
    logic [31:0] perf_discarded_writes;
    logic [31:0] perf_committed_records;
    logic [31:0] perf_replay_reads;
    logic [31:0] perf_capacity_misses;
    int live_count;

    qfit_exposure_relation_vault dut (.*);

    always #5 clk_core = ~clk_core;

    task automatic run_head(
        input int head,
        input int active_sources,
        input bit expected_resident,
        input bit expected_critical,
        input bit expected_overflow
    );
        int sent;
        bit fire;
        while (!head_ready)
            @(negedge clk_core);
        head_index = HEAD_W'(head);
        head_start = 1'b1;
        @(negedge clk_core);
        head_start = 1'b0;
        sent = 0;
        in_valid = 1'b1;
        while (sent < TOTAL) begin
            in_source_id = 9'(sent);
            in_y = 4'((sent % 225) / 15);
            in_x = 4'((sent % 225) % 15);
            in_k = sent < active_sources ? 32'h0000_0001 : '0;
            in_gates = '0;
            in_gates[8:0] = 9'd1;
            in_valid_mask = 5'b00001;
            in_last = sent == TOTAL - 1;
            @(posedge clk_core);
            fire = in_ready;
            @(negedge clk_core);
            if (fire)
                sent = sent + 1;
        end
        in_valid = 1'b0;
        in_last = 1'b0;
        while (!head_done)
            @(negedge clk_core);
        if (head_resident != expected_resident)
            $fatal(1, "head %0d resident mismatch", head);
        if (head_critical != expected_critical)
            $fatal(1, "head %0d critical mismatch", head);
        if (head_overflow != expected_overflow)
            $fatal(1, "head %0d overflow mismatch", head);
        if (head_service_cycles != 15 + active_sources)
            $fatal(1, "head %0d service mismatch", head);
        if (
            head_record_count
            != (expected_resident ? PTR_W'(active_sources) : '0)
        )
            $fatal(1, "head %0d record count mismatch", head);
    endtask

    task automatic replay_head(
        input int head,
        input int active_sources,
        input bit expected_miss
    );
        int received;
        bit saw_miss;
        while (!replay_cmd_ready)
            @(negedge clk_core);
        replay_head_index = HEAD_W'(head);
        replay_start = 1'b1;
        @(negedge clk_core);
        replay_start = 1'b0;
        received = 0;
        saw_miss = replay_miss;
        while (!replay_done) begin
            replay_ready = ($urandom_range(0, 4) != 0);
            @(posedge clk_core);
            if (replay_miss)
                saw_miss = 1'b1;
            if (replay_valid && replay_ready) begin
                if (replay_source_id != 9'(received))
                    $fatal(
                        1,
                        "head %0d replay source mismatch got=%0d exp=%0d",
                        head,
                        replay_source_id,
                        received
                    );
                if (replay_k != 32'h0000_0001)
                    $fatal(1, "head %0d replay K mismatch", head);
                if (replay_gates[8:0] != 9'd1)
                    $fatal(1, "head %0d replay gate mismatch", head);
                if (replay_valid_mask != 5'b00001)
                    $fatal(1, "head %0d replay mask mismatch", head);
                if (replay_y != 4'((received % 225) / 15))
                    $fatal(1, "head %0d replay y mismatch", head);
                if (replay_x != 4'((received % 225) % 15))
                    $fatal(1, "head %0d replay x mismatch", head);
                if (replay_last != (received == active_sources - 1))
                    $fatal(1, "head %0d replay last mismatch", head);
                received = received + 1;
            end
            @(negedge clk_core);
        end
        replay_ready = 1'b0;
        if (saw_miss != expected_miss)
            $fatal(1, "head %0d replay miss mismatch", head);
        if (received != (expected_miss ? 0 : active_sources))
            $fatal(1, "head %0d replay count mismatch", head);
    endtask

    always_ff @(posedge clk_core) begin
        if (rst_core)
            live_count <= 0;
        else if (live_valid && live_ready) begin
            if (live_source_id != 9'(live_count % TOTAL))
                $fatal(1, "live source mismatch");
            live_count <= live_count + 1;
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        window_start = 1'b0;
        head_start = 1'b0;
        head_index = '0;
        in_valid = 1'b0;
        in_source_id = '0;
        in_y = '0;
        in_x = '0;
        in_k = '0;
        in_gates = '0;
        in_valid_mask = '0;
        in_last = 1'b0;
        live_ready = 1'b0;
        replay_start = 1'b0;
        replay_head_index = '0;
        replay_ready = 1'b0;
        live_count = 0;
        repeat (5) @(negedge clk_core);
        rst_core = 1'b0;
        window_start = 1'b1;
        @(negedge clk_core);
        window_start = 1'b0;
        live_ready = 1'b1;

        run_head(0, 20, 1'b1, 1'b1, 1'b0);
        run_head(1, 450, 1'b0, 1'b0, 1'b0);
        run_head(2, 20, 1'b1, 1'b1, 1'b0);
        run_head(3, 434, 1'b1, 1'b1, 1'b0);
        run_head(4, 50, 1'b0, 1'b1, 1'b1);

        replay_head(0, 20, 1'b0);
        replay_head(1, 0, 1'b1);
        replay_head(2, 20, 1'b0);
        replay_head(3, 434, 1'b0);
        replay_head(4, 0, 1'b1);

        if (protocol_error)
            $fatal(1, "unexpected protocol error");
        if (perf_committed_records != 474)
            $fatal(1, "committed record mismatch");
        if (perf_capacity_misses != 1)
            $fatal(1, "capacity miss mismatch");
        if (perf_replay_reads != 474)
            $fatal(1, "replay read mismatch");
        $display(
            "PASS exposure relation vault live=%0d spec=%0d discard=%0d committed=%0d replay=%0d miss=%0d",
            live_count,
            perf_speculative_writes,
            perf_discarded_writes,
            perf_committed_records,
            perf_replay_reads,
            perf_capacity_misses
        );
        $finish;
    end
endmodule

`default_nettype wire
