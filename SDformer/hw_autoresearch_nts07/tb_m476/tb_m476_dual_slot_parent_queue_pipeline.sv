`timescale 1ns/1ps
`default_nettype none
module tb_m476_dual_slot_parent_queue_pipeline;
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
    logic row_complete, protocol_error, row_active;
    logic [1:0] parent_queue_occupancy;
    logic parent_queue_full;
    logic debug_forward_event, debug_scratch_read_event;
    logic debug_read_response_event, debug_dual_enqueue_event;
    logic debug_overflow_block_event;

    logic [LANES*12-1:0] scratch_mem [0:ROWS-1];
    logic [ROW_BITS-1:0] scratch_read_address_q;
    integer expected_row [0:ROWS-1][0:LANES-1];
    integer expected_psum [0:ROWS-1][0:LANES-1];
    bit expected_valid [0:ROWS-1];

    integer cycles, issues_seen, writes_seen, psum_seen;
    integer forward_seen, reads_seen, responses_seen, dual_enqueue_seen;
    integer full_seen, full_consume_seen, stalls_seen;
    integer exact_seen, partial_seen, previous_completion_cycle, b2b_seen;
    integer id_attacks, overflow_attacks, overflow_block_seen;
    bit expect_protocol_error;

    m476_dual_slot_parent_queue_pipeline #(
        .LANES(LANES), .ROW_BITS(ROW_BITS)
    ) dut (.*);
    m476_dual_slot_parent_queue_assertions #(
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
            6: residual_value = (lane % 7) - 3;
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
            clear_drives();
            reset_n = 0;
            repeat (4) @(posedge clk_core);
            @(negedge clk_core);
            reset_n = 1;
            expect_protocol_error = 0;
        end
    endtask

    task automatic set_issue_payload(
        input integer row_id,
        input bit first,
        input bit last,
        input bit parent_valid,
        input integer parent_id,
        input integer residual_kind,
        input integer psum_base
    );
        integer l, value;
        begin
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
        end
    endtask

    task automatic issue_once(
        input integer row_id,
        input bit first,
        input bit last,
        input bit parent_valid,
        input integer parent_id,
        input integer residual_kind,
        input integer psum_base
    );
        begin
            if (clk_core !== 1'b0)
                @(negedge clk_core);
            set_issue_payload(row_id, first, last, parent_valid, parent_id,
                residual_kind, psum_base);
            issue_valid = 1;
            do @(posedge clk_core); while (!issue_ready);
            @(negedge clk_core);
            issue_valid = 0;
        end
    endtask

    task automatic prefetch_once(input integer parent_id);
        begin
            if (clk_core !== 1'b0)
                @(negedge clk_core);
            prefetch_parent_id = parent_id[ROW_BITS-1:0];
            prefetch_valid = 1;
            do @(posedge clk_core); while (!prefetch_ready);
            @(negedge clk_core);
            prefetch_valid = 0;
        end
    endtask

    task automatic issue_with_forward(
        input integer row_id,
        input bit first,
        input bit last,
        input bit parent_valid,
        input integer parent_id,
        input integer residual_kind,
        input integer psum_base
    );
        begin
            if (clk_core !== 1'b0)
                @(negedge clk_core);
            set_issue_payload(row_id, first, last, parent_valid, parent_id,
                residual_kind, psum_base);
            issue_valid = 1;
            prefetch_valid = 1;
            prefetch_parent_id = row_id[ROW_BITS-1:0];
            #0;
            if (!issue_ready || !prefetch_ready)
                $fatal(1, "joint issue/forward unexpectedly not ready");
            @(posedge clk_core);
            @(negedge clk_core);
            issue_valid = 0;
            prefetch_valid = 0;
        end
    endtask

    task automatic full_consume_with_held_prefetch(
        input integer row_id,
        input integer parent_id,
        input integer psum_base
    );
        begin
            if (clk_core !== 1'b0)
                @(negedge clk_core);
            if (!parent_queue_full)
                $fatal(1, "full-queue test entered without two entries");
            set_issue_payload(row_id, 1, 1, 1, parent_id, 1, psum_base);
            issue_valid = 1;
            prefetch_valid = 1;
            prefetch_parent_id = row_id[ROW_BITS-1:0];
            #0;
            if (!issue_ready || prefetch_ready)
                $fatal(1, "full queue did not decouple issue/prefetch");
            @(posedge clk_core);
            @(negedge clk_core);
            issue_valid = 0;
            // The stalled prefetch ID remains stable and must be accepted now
            // that the prior issue has consumed slot0.
            #0;
            if (!prefetch_ready)
                $fatal(1, "held prefetch did not become ready after consume");
            @(posedge clk_core);
            @(negedge clk_core);
            prefetch_valid = 0;
        end
    endtask

    always @(posedge clk_core) begin : monitor
        integer l, got;
        if (reset_n) begin
            cycles = cycles + 1;
            if (protocol_error && !expect_protocol_error)
                $fatal(1, "unexpected protocol error cycle=%0d", cycles);
            if (issue_valid && !issue_ready)
                stalls_seen = stalls_seen + 1;
            if (issue_valid && issue_ready) begin
                issues_seen = issues_seen + 1;
                if (issue_parent_valid && issue_residual_data == '0)
                    exact_seen = exact_seen + 1;
                if (issue_parent_valid && issue_residual_data != '0)
                    partial_seen = partial_seen + 1;
            end
            if (debug_forward_event)
                forward_seen = forward_seen + 1;
            if (debug_scratch_read_event)
                reads_seen = reads_seen + 1;
            if (debug_read_response_event)
                responses_seen = responses_seen + 1;
            if (debug_dual_enqueue_event)
                dual_enqueue_seen = dual_enqueue_seen + 1;
            if (debug_overflow_block_event)
                overflow_block_seen = overflow_block_seen + 1;
            if (parent_queue_full)
                full_seen = full_seen + 1;
            if (parent_queue_full && issue_valid && issue_ready && issue_last
                    && issue_parent_valid && !prefetch_ready)
                full_consume_seen = full_consume_seen + 1;

            if (scratch_write_enable) begin
                if (!expected_valid[scratch_write_address])
                    $fatal(1, "unexpected row write address=%0d",
                        scratch_write_address);
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
                    b2b_seen = b2b_seen + 1;
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
        integer l, p10, r0, a, b, r22, r4;
        clk_core = 0;
        reset_n = 0;
        psum_write_ready = 1;
        expect_protocol_error = 0;
        cycles = 0;
        issues_seen = 0;
        writes_seen = 0;
        psum_seen = 0;
        forward_seen = 0;
        reads_seen = 0;
        responses_seen = 0;
        dual_enqueue_seen = 0;
        full_seen = 0;
        full_consume_seen = 0;
        stalls_seen = 0;
        exact_seen = 0;
        partial_seen = 0;
        previous_completion_cycle = -1;
        b2b_seen = 0;
        id_attacks = 0;
        overflow_attacks = 0;
        overflow_block_seen = 0;
        scratch_read_address_q = 0;
        clear_drives();
        for (integer r = 0; r < ROWS; r = r + 1) begin
            scratch_mem[r] = '0;
            expected_valid[r] = 0;
            for (l = 0; l < LANES; l = l + 1) begin
                expected_row[r][l] = 0;
                expected_psum[r][l] = 0;
            end
        end

        // Row10 is a pre-existing parent in the external 64x1152 scratch.
        for (l = 0; l < LANES; l = l + 1) begin
            p10 = residual_value(6, l);
            scratch_mem[10][l*12 +: 12] = p10[11:0];
            r0 = residual_value(0, l);
            a = residual_value(2, l);
            b = residual_value(3, l);
            r22 = r0 + a + b;
            r4 = residual_value(4, l);

            expected_row[20][l] = r0;
            expected_psum[20][l] = 100 + l + r0;
            expected_row[21][l] = p10;
            expected_psum[21][l] = 200 + l + p10;
            expected_row[22][l] = r22;
            expected_psum[22][l] = -100 + l + r22;
            expected_row[23][l] = p10;
            expected_psum[23][l] = 50 + l + p10;
            expected_row[24][l] = r4;
            expected_psum[24][l] = l + r4;
        end
        expected_valid[20] = 1;
        expected_valid[21] = 1;
        expected_valid[22] = 1;
        expected_valid[23] = 1;
        expected_valid[24] = 1;

        apply_reset();

        // Issue a real macro read for row10.  On its response cycle, direct
        // row20 completes and is forwarded into the second queue slot.  This
        // explicitly covers two enqueues in one edge without pop credit.
        prefetch_once(10);
        issue_with_forward(20, 1, 1, 0, 0, 0, 100);
        if (!parent_queue_full || parent_queue_occupancy != 2)
            $fatal(1, "dual enqueue did not fill queue occupancy=%0d",
                parent_queue_occupancy);

        // Full queue must not use same-cycle consume as prefetch capacity.
        // Row21 consumes parent10; its held prefetch is accepted one cycle
        // later as a real scratch read after row21 has committed.
        full_consume_with_held_prefetch(21, 10, 200);

        // Queue head is row20 while row21's read is pending.  The first beat
        // of row22 overlaps the row21 response; the final beat consumes row20.
        issue_once(22, 1, 0, 1, 20, 2, -100);
        issue_once(22, 0, 1, 1, 20, 3, 0);
        // The compacted second slot becomes head, enabling an immediately
        // adjacent exact-parent completion.
        issue_once(23, 1, 1, 1, 21, 1, 50);

        // Final-output backpressure must hold payload and produce no write.
        @(negedge clk_core);
        psum_write_ready = 0;
        fork
            begin
                repeat (3) @(posedge clk_core);
                @(negedge clk_core);
                psum_write_ready = 1;
            end
            issue_once(24, 1, 1, 0, 0, 4, 0);
        join

        repeat (3) @(posedge clk_core);
        if (writes_seen != 5 || psum_seen != 5 || issues_seen != 6)
            $fatal(1, "completion mismatch issues=%0d scratch=%0d psum=%0d",
                issues_seen, writes_seen, psum_seen);
        if (dual_enqueue_seen != 1 || full_consume_seen < 1 || full_seen < 1)
            $fatal(1, "queue coverage dual=%0d fullconsume=%0d full=%0d",
                dual_enqueue_seen, full_consume_seen, full_seen);
        if (forward_seen != 1 || reads_seen != 2 || responses_seen != 2)
            $fatal(1, "transport coverage fwd=%0d read=%0d response=%0d",
                forward_seen, reads_seen, responses_seen);
        if (exact_seen != 2 || partial_seen != 2 || b2b_seen < 2)
            $fatal(1, "work coverage exact=%0d partial=%0d b2b=%0d",
                exact_seen, partial_seen, b2b_seen);
        if (stalls_seen < 2)
            $fatal(1, "output stall coverage=%0d", stalls_seen);

        // ID mismatch attack: row20 is present but issue claims row19.
        prefetch_once(20);
        @(posedge clk_core);
        @(negedge clk_core);
        expect_protocol_error = 1;
        set_issue_payload(25, 1, 1, 1, 19, 1, 0);
        issue_valid = 1;
        repeat (3) @(posedge clk_core);
        if (!protocol_error)
            $fatal(1, "ID mismatch did not latch fault");
        if (writes_seen != 5 || psum_seen != 5)
            $fatal(1, "ID mismatch leaked a write");
        id_attacks = id_attacks + 1;

        // Reset only the DUT protocol state, retain external memories/counters,
        // then attack signed12 row overflow with a valid parent.
        apply_reset();
        prefetch_once(20);
        @(posedge clk_core);
        @(negedge clk_core);
        expect_protocol_error = 1;
        set_issue_payload(25, 1, 1, 1, 20, 5, 0);
        issue_valid = 1;
        repeat (3) @(posedge clk_core);
        if (!protocol_error || overflow_block_seen < 1)
            $fatal(1, "overflow attack did not fail closed");
        if (writes_seen != 5 || psum_seen != 5)
            $fatal(1, "overflow attack leaked a write");
        overflow_attacks = overflow_attacks + 1;

        $display("PASS M476 directed issues=%0d rows=%0d forward=%0d reads=%0d responses=%0d dual_enqueue=%0d full=%0d fullconsume=%0d stalls=%0d b2b=%0d exact=%0d partialbeats=%0d id_attacks=%0d overflow_attacks=%0d",
            issues_seen, writes_seen, forward_seen, reads_seen, responses_seen,
            dual_enqueue_seen, full_seen, full_consume_seen, stalls_seen,
            b2b_seen, exact_seen, partial_seen, id_attacks, overflow_attacks);
        $finish;
    end

    initial begin
        #30000;
        $fatal(1, "M476 timeout");
    end
endmodule
`default_nettype wire
