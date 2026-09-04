`timescale 1ns/1ps
`default_nettype none
module tb_m474_fused_parent_dual_update_pipeline;
    localparam int LANES = 96;
    localparam int ROW_BITS = 6;
    localparam int ROWS = 1 << ROW_BITS;

    logic clk_core, reset_n;
    logic prefetch_valid, prefetch_ready;
    logic [ROW_BITS-1:0] prefetch_parent_id;
    logic scratch_read_enable;
    logic [ROW_BITS-1:0] scratch_read_address;
    logic [LANES*12-1:0] scratch_read_data;
    logic issue_valid, issue_ready, issue_first, issue_last;
    logic [ROW_BITS-1:0] issue_row_id;
    logic issue_parent_valid;
    logic [ROW_BITS-1:0] issue_parent_id;
    logic [LANES*12-1:0] issue_residual_data;
    logic [LANES*19-1:0] issue_psum_prior;
    logic scratch_write_enable;
    logic [ROW_BITS-1:0] scratch_write_address;
    logic [LANES*12-1:0] scratch_write_data;
    logic psum_write_valid, psum_write_ready;
    logic [ROW_BITS-1:0] psum_write_address;
    logic [LANES*19-1:0] psum_write_data;
    logic row_complete, protocol_error, row_active, parent_buffer_valid;
    logic debug_forward_event, debug_scratch_read_event;
    logic debug_overflow_block_event;
    logic [31:0] debug_issue_accepts, debug_row_completions;
    logic [31:0] debug_forward_hits, debug_scratch_reads;
    logic [31:0] debug_stall_cycles;

    logic [LANES*12-1:0] scratch_mem [0:ROWS-1];
    logic [ROW_BITS-1:0] scratch_read_address_q;
    integer expected_row [0:4][0:LANES-1];
    integer expected_psum [0:4][0:LANES-1];
    integer cycles, writes_seen, psum_seen, exact_seen, partial_seen;
    integer back_to_back_completions, previous_completion_cycle;
    integer previous_read_cycle, one_ahead_read_issues;
    bit expect_protocol_error;

    m474_fused_parent_dual_update_pipeline #(
        .LANES(LANES), .ROW_BITS(ROW_BITS)
    ) dut (.*);
    m474_fused_parent_dual_update_assertions #(
        .LANES(LANES), .ROW_BITS(ROW_BITS)
    ) sva (.*);

    always #1.5 clk_core = ~clk_core;
    assign scratch_read_data = scratch_mem[scratch_read_address_q];

    always @(posedge clk_core) begin
        if (scratch_read_enable)
            scratch_read_address_q <= scratch_read_address;
        if (scratch_write_enable)
            scratch_mem[scratch_write_address] <= scratch_write_data;
    end

    function automatic integer residual_value(
        input integer kind, input integer lane
    );
        case (kind)
            0: residual_value = (lane % 5) - 2;
            1: residual_value = 0;
            2: residual_value = -3 + (lane % 2);
            3: residual_value = 5 - (lane % 3);
            4: residual_value = 10 - (lane % 4);
            5: residual_value = 2047;
            default: residual_value = 0;
        endcase
    endfunction

    task automatic clear_drives;
        begin
            prefetch_valid = 0;
            prefetch_parent_id = 0;
            issue_valid = 0;
            issue_row_id = 0;
            issue_first = 0;
            issue_last = 0;
            issue_parent_valid = 0;
            issue_parent_id = 0;
            issue_residual_data = 0;
            issue_psum_prior = 0;
        end
    endtask

    task automatic apply_reset;
        begin
            reset_n = 0;
            repeat (4) @(posedge clk_core);
            @(negedge clk_core);
            reset_n = 1;
        end
    endtask

    task automatic prefetch_parent(input integer parent_id);
        begin
            @(negedge clk_core);
            prefetch_valid = 1;
            prefetch_parent_id = parent_id[ROW_BITS-1:0];
            do @(posedge clk_core); while (!prefetch_ready);
            @(negedge clk_core);
            prefetch_valid = 0;
            // The dependent issue is driven in this low phase and must accept
            // at the immediately following edge from the macro's registered Q.
        end
    endtask

    task automatic drive_issue(
        input integer row_id,
        input bit first,
        input bit last,
        input bit parent_valid,
        input integer parent_id,
        input integer residual_kind,
        input integer psum_base,
        input bit also_prefetch,
        input integer next_parent_id
    );
        integer l, value;
        begin
            // Reuse the current low phase when a prior issue task has just
            // retired. This intentionally generates adjacent-cycle issues.
            if (clk_core !== 1'b0)
                @(negedge clk_core);
            issue_valid = 1;
            issue_row_id = row_id[ROW_BITS-1:0];
            issue_first = first;
            issue_last = last;
            issue_parent_valid = parent_valid;
            issue_parent_id = parent_id[ROW_BITS-1:0];
            issue_residual_data = '0;
            issue_psum_prior = '0;
            for (l = 0; l < LANES; l = l + 1) begin
                value = residual_value(residual_kind, l);
                issue_residual_data[l*12 +: 12] = value[11:0];
                value = psum_base + l;
                issue_psum_prior[l*19 +: 19] = value[18:0];
            end
            prefetch_valid = also_prefetch;
            prefetch_parent_id = next_parent_id[ROW_BITS-1:0];
            do @(posedge clk_core); while (!issue_ready
                || (also_prefetch && !prefetch_ready));
            @(negedge clk_core);
            issue_valid = 0;
            prefetch_valid = 0;
        end
    endtask

    always @(posedge clk_core) begin : monitor
        integer l, got;
        if (reset_n) begin
            cycles = cycles + 1;
            if (protocol_error && !expect_protocol_error)
                $fatal(1, "unexpected protocol error cycle=%0d", cycles);
            if (issue_valid && issue_ready) begin
                if (issue_parent_valid && issue_residual_data == '0)
                    exact_seen = exact_seen + 1;
                if (issue_parent_valid && issue_residual_data != '0)
                    partial_seen = partial_seen + 1;
                if (issue_parent_valid && previous_read_cycle >= 0
                        && cycles == previous_read_cycle + 1)
                    one_ahead_read_issues = one_ahead_read_issues + 1;
            end
            if (scratch_read_enable)
                previous_read_cycle = cycles;
            if (scratch_write_enable) begin
                for (l = 0; l < LANES; l = l + 1) begin
                    got = $signed(scratch_write_data[l*12 +: 12]);
                    if (got !== expected_row[scratch_write_address][l])
                        $fatal(1, "row mismatch row=%0d lane=%0d got=%0d exp=%0d",
                            scratch_write_address, l, got,
                            expected_row[scratch_write_address][l]);
                end
                writes_seen = writes_seen + 1;
                if (previous_completion_cycle >= 0
                        && cycles == previous_completion_cycle + 1)
                    back_to_back_completions = back_to_back_completions + 1;
                previous_completion_cycle = cycles;
            end
            if (psum_write_valid && psum_write_ready) begin
                for (l = 0; l < LANES; l = l + 1) begin
                    got = $signed(psum_write_data[l*19 +: 19]);
                    if (got !== expected_psum[psum_write_address][l])
                        $fatal(1, "psum mismatch row=%0d lane=%0d got=%0d exp=%0d",
                            psum_write_address, l, got,
                            expected_psum[psum_write_address][l]);
                end
                psum_seen = psum_seen + 1;
            end
        end
    end

    initial begin : test
        integer l, r0, a, b, r2, r4;
        clk_core = 0;
        reset_n = 0;
        psum_write_ready = 1;
        expect_protocol_error = 0;
        cycles = 0;
        writes_seen = 0;
        psum_seen = 0;
        exact_seen = 0;
        partial_seen = 0;
        back_to_back_completions = 0;
        previous_completion_cycle = -1;
        previous_read_cycle = -1;
        one_ahead_read_issues = 0;
        scratch_read_address_q = 0;
        clear_drives();
        for (integer r = 0; r < ROWS; r = r + 1)
            scratch_mem[r] = '0;
        for (l = 0; l < LANES; l = l + 1) begin
            r0 = residual_value(0, l);
            a = residual_value(2, l);
            b = residual_value(3, l);
            r2 = r0 + a + b;
            r4 = residual_value(4, l);
            expected_row[0][l] = r0;
            expected_psum[0][l] = 100 + l + r0;
            expected_row[1][l] = r0;
            expected_psum[1][l] = 200 + l + r0;
            expected_row[2][l] = r2;
            expected_psum[2][l] = -100 + l + r2;
            expected_row[3][l] = r2;
            expected_psum[3][l] = 50 + l + r2;
            expected_row[4][l] = r4;
            expected_psum[4][l] = l + r4;
        end

        apply_reset();

        // Direct row0 completes while row0 is simultaneously prefetched for
        // row1. The read is suppressed and the write is forwarded.
        drive_issue(0, 1, 1, 0, 0, 0, 100, 1, 0);
        drive_issue(1, 1, 1, 1, 0, 1, 200, 1, 0);

        // Row1 consumes its forwarded parent while issuing a nonmatching
        // prefetch for row0. The normal macro read supplies the immediately
        // following two-beat partial-parent row2 with no capture bubble.
        drive_issue(2, 1, 0, 1, 0, 2, -100, 0, 0);
        drive_issue(2, 0, 1, 1, 0, 3, 0, 1, 2);
        drive_issue(3, 1, 1, 1, 2, 1, 50, 0, 0);

        // Final-output backpressure must hold the issue and psum payload while
        // producing no scratch write. Release after two full stall cycles.
        @(negedge clk_core);
        psum_write_ready = 0;
        fork
            begin
                repeat (3) @(posedge clk_core);
                @(negedge clk_core);
                psum_write_ready = 1;
            end
            drive_issue(4, 1, 1, 0, 0, 4, 0, 0, 0);
        join

        repeat (3) @(posedge clk_core);
        if (writes_seen != 5 || psum_seen != 5)
            $fatal(1, "completion count mismatch scratch=%0d psum=%0d",
                writes_seen, psum_seen);
        if (debug_issue_accepts != 6 || debug_row_completions != 5)
            $fatal(1, "debug issue/completion mismatch %0d/%0d",
                debug_issue_accepts, debug_row_completions);
        if (debug_forward_hits != 2 || debug_scratch_reads != 1)
            $fatal(1, "forward/read mismatch %0d/%0d",
                debug_forward_hits, debug_scratch_reads);
        if (exact_seen != 2 || partial_seen != 2)
            $fatal(1, "parent coverage mismatch exact=%0d partialbeats=%0d",
                exact_seen, partial_seen);
        if (back_to_back_completions < 2)
            $fatal(1, "back-to-back completion coverage=%0d",
                back_to_back_completions);
        if (one_ahead_read_issues != 1)
            $fatal(1, "one-ahead synchronous read coverage=%0d",
                one_ahead_read_issues);
        if (debug_stall_cycles < 2)
            $fatal(1, "stall coverage=%0d", debug_stall_cycles);

        // Fail-closed atomicity attack: parent row0 + maximum signed12
        // residual overflows row scratch. Neither scratch nor psum may write.
        prefetch_parent(0);
        if (clk_core !== 1'b0)
            @(negedge clk_core);
        expect_protocol_error = 1;
        issue_valid = 1;
        issue_row_id = 5;
        issue_first = 1;
        issue_last = 1;
        issue_parent_valid = 1;
        issue_parent_id = 0;
        for (l = 0; l < LANES; l = l + 1) begin
            r0 = residual_value(5, l);
            issue_residual_data[l*12 +: 12] = r0[11:0];
            issue_psum_prior[l*19 +: 19] = l;
        end
        repeat (3) @(posedge clk_core);
        if (!protocol_error)
            $fatal(1, "overflow attack did not latch fault");
        if (writes_seen != 5 || psum_seen != 5)
            $fatal(1, "overflow attack leaked write scratch=%0d psum=%0d",
                writes_seen, psum_seen);

        $display("PASS M474 directed issues=%0d rows=%0d forward=%0d reads=%0d stalls=%0d b2b=%0d oneahead=%0d exact=%0d partialbeats=%0d overflow_attacks=1",
            debug_issue_accepts, debug_row_completions,
            debug_forward_hits, debug_scratch_reads, debug_stall_cycles,
            back_to_back_completions, one_ahead_read_issues,
            exact_seen, partial_seen);
        $finish;
    end

    initial begin
        #20000;
        $fatal(1, "M474 timeout");
    end
endmodule
`default_nettype wire
