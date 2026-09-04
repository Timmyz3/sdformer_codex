`timescale 1ns/1ps
`default_nettype none

module tb_h87_t5_open_run_file;
    localparam int POSITIONS = 4;
    localparam int POSITION_W = 2;
    localparam int MAX_EXPECTED = 64;

    logic clk_core;
    logic rst_core;
    logic row_start;
    logic row_start_ready;
    logic in_valid;
    logic in_ready;
    logic [2:0] in_window_index;
    logic [POSITION_W-1:0] in_position_id;
    logic [7:0] in_score0_q7;
    logic [7:0] in_score1_q7;
    logic [1:0] in_active_mask;
    logic packet_valid;
    logic packet_ready;
    logic [1:0] packet_desc_count;
    logic [POSITION_W-1:0] packet_desc0_position;
    logic packet_desc0_group;
    logic [7:0] packet_desc0_score_q7;
    logic [4:0] packet_desc0_temporal_mask;
    logic [4:0] packet_desc0_active_mask;
    logic packet_desc0_group_last;
    logic packet_desc0_row_last;
    logic [POSITION_W-1:0] packet_desc1_position;
    logic packet_desc1_group;
    logic [7:0] packet_desc1_score_q7;
    logic [4:0] packet_desc1_temporal_mask;
    logic [4:0] packet_desc1_active_mask;
    logic packet_desc1_group_last;
    logic packet_desc1_row_last;
    logic row_done;
    logic protocol_error;
    logic [31:0] perf_input_pairs;
    logic [31:0] perf_original_slots;
    logic [31:0] perf_run_descriptors;
    logic [31:0] perf_equal_edges;
    logic [31:0] perf_spill_packets;
    logic [31:0] perf_input_stall_cycles;

    logic [7:0] scores [0:POSITIONS-1][0:9];
    logic active [0:POSITIONS-1][0:9];
    logic [7:0] ref_score [0:POSITIONS-1];
    logic [4:0] ref_temporal [0:POSITIONS-1];
    logic [4:0] ref_active [0:POSITIONS-1];

    logic [POSITION_W-1:0] exp_position [0:MAX_EXPECTED-1];
    logic exp_group [0:MAX_EXPECTED-1];
    logic [7:0] exp_score [0:MAX_EXPECTED-1];
    logic [4:0] exp_temporal [0:MAX_EXPECTED-1];
    logic [4:0] exp_active [0:MAX_EXPECTED-1];
    logic exp_group_last [0:MAX_EXPECTED-1];
    logic exp_row_last [0:MAX_EXPECTED-1];

    integer exp_count;
    integer exp_index;
    integer errors;
    integer row_done_count;
    integer p;
    integer w;
    integer t;
    integer timeout;
    logic [15:0] lfsr_q;

    h87_t5_open_run_file #(
        .POSITIONS(POSITIONS),
        .POSITION_W(POSITION_W)
    ) dut (.*);

    always #1 clk_core = ~clk_core;

    always_ff @(posedge clk_core) begin
        if (rst_core)
            lfsr_q <= 16'h1d3f;
        else
            lfsr_q <= {lfsr_q[14:0],
                       lfsr_q[15] ^ lfsr_q[13] ^ lfsr_q[12] ^ lfsr_q[10]};
    end

    assign packet_ready = lfsr_q[0] | lfsr_q[5];

    task automatic append_expected(
        input integer pos,
        input logic group_id,
        input logic [7:0] score,
        input logic [4:0] temporal_mask,
        input logic [4:0] active_mask,
        input logic group_last,
        input logic row_last
    );
        begin
            exp_position[exp_count] = POSITION_W'(pos);
            exp_group[exp_count] = group_id;
            exp_score[exp_count] = score;
            exp_temporal[exp_count] = temporal_mask;
            exp_active[exp_count] = active_mask;
            exp_group_last[exp_count] = group_last;
            exp_row_last[exp_count] = row_last;
            exp_count = exp_count + 1;
        end
    endtask

    task automatic ref_consume(
        input integer pos,
        input logic group_id,
        input integer slot,
        input logic [7:0] score,
        input logic active_bit
    );
        begin
            if (score == ref_score[pos]) begin
                ref_temporal[pos][slot] = 1'b1;
                ref_active[pos][slot] = active_bit;
            end else begin
                append_expected(pos, group_id, ref_score[pos],
                                ref_temporal[pos], ref_active[pos], 1'b0, 1'b0);
                ref_score[pos] = score;
                ref_temporal[pos] = 5'b00001 << slot;
                ref_active[pos] = ({4'b0000, active_bit}) << slot;
            end
        end
    endtask

    task automatic ref_finalize(
        input integer pos,
        input logic group_id,
        input logic row_last
    );
        begin
            append_expected(pos, group_id, ref_score[pos],
                            ref_temporal[pos], ref_active[pos], 1'b1, row_last);
        end
    endtask

    task automatic drive_pair(input integer window_index, input integer pos);
        integer slot0;
        begin
            slot0 = 2 * window_index;
            @(negedge clk_core);
            in_valid = 1'b1;
            in_window_index = 3'(window_index);
            in_position_id = POSITION_W'(pos);
            in_score0_q7 = scores[pos][slot0];
            in_score1_q7 = scores[pos][slot0 + 1];
            in_active_mask = {active[pos][slot0 + 1], active[pos][slot0]};
            do @(posedge clk_core); while (!in_ready);
            @(negedge clk_core);
            in_valid = 1'b0;
        end
    endtask

    task automatic compare_descriptor0;
        begin
            if (exp_index >= exp_count) begin
                $display("ERROR unexpected desc0");
                errors = errors + 1;
            end else begin
                if (packet_desc0_position !== exp_position[exp_index]
                    || packet_desc0_group !== exp_group[exp_index]
                    || packet_desc0_score_q7 !== exp_score[exp_index]
                    || packet_desc0_temporal_mask !== exp_temporal[exp_index]
                    || packet_desc0_active_mask !== exp_active[exp_index]
                    || packet_desc0_group_last !== exp_group_last[exp_index]
                    || packet_desc0_row_last !== exp_row_last[exp_index]) begin
                    $display("ERROR desc0 idx=%0d got p/g/s/t/a/gl/rl=%0d/%0d/%0d/%b/%b/%0d/%0d expected=%0d/%0d/%0d/%b/%b/%0d/%0d",
                             exp_index,
                             packet_desc0_position, packet_desc0_group,
                             packet_desc0_score_q7, packet_desc0_temporal_mask,
                             packet_desc0_active_mask, packet_desc0_group_last,
                             packet_desc0_row_last,
                             exp_position[exp_index], exp_group[exp_index],
                             exp_score[exp_index], exp_temporal[exp_index],
                             exp_active[exp_index], exp_group_last[exp_index],
                             exp_row_last[exp_index]);
                    errors = errors + 1;
                end
                exp_index = exp_index + 1;
            end
        end
    endtask

    task automatic compare_descriptor1;
        begin
            if (exp_index >= exp_count) begin
                $display("ERROR unexpected desc1");
                errors = errors + 1;
            end else begin
                if (packet_desc1_position !== exp_position[exp_index]
                    || packet_desc1_group !== exp_group[exp_index]
                    || packet_desc1_score_q7 !== exp_score[exp_index]
                    || packet_desc1_temporal_mask !== exp_temporal[exp_index]
                    || packet_desc1_active_mask !== exp_active[exp_index]
                    || packet_desc1_group_last !== exp_group_last[exp_index]
                    || packet_desc1_row_last !== exp_row_last[exp_index]) begin
                    $display("ERROR desc1 idx=%0d", exp_index);
                    errors = errors + 1;
                end
                exp_index = exp_index + 1;
            end
        end
    endtask

    always @(posedge clk_core) begin
        if (!rst_core && packet_valid && packet_ready) begin
            compare_descriptor0();
            if (packet_desc_count == 2'd2)
                compare_descriptor1();
        end
        if (!rst_core && row_done)
            row_done_count = row_done_count + 1;
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        row_start = 1'b0;
        in_valid = 1'b0;
        in_window_index = '0;
        in_position_id = '0;
        in_score0_q7 = '0;
        in_score1_q7 = '0;
        in_active_mask = '0;
        exp_count = 0;
        exp_index = 0;
        errors = 0;
        row_done_count = 0;

        for (p = 0; p < POSITIONS; p = p + 1)
            for (t = 0; t < 10; t = t + 1)
                active[p][t] = ((p + t) % 3) != 0;

        for (t = 0; t < 10; t = t + 1) begin
            scores[0][t] = 8'd11;
            scores[1][t] = 8'(t);
        end
        scores[2][0] = 8'd1; scores[2][1] = 8'd1;
        scores[2][2] = 8'd2; scores[2][3] = 8'd2; scores[2][4] = 8'd2;
        scores[2][5] = 8'd3; scores[2][6] = 8'd4; scores[2][7] = 8'd4;
        scores[2][8] = 8'd5; scores[2][9] = 8'd5;
        scores[3][0] = 8'd7; scores[3][1] = 8'd8; scores[3][2] = 8'd7;
        scores[3][3] = 8'd8; scores[3][4] = 8'd7;
        scores[3][5] = 8'd9; scores[3][6] = 8'd9; scores[3][7] = 8'd9;
        scores[3][8] = 8'd9; scores[3][9] = 8'd10;

        // Independent reference follows window-major arrival and closes a run
        // only on score change or a T=5 group boundary.
        for (w = 0; w < 5; w = w + 1) begin
            for (p = 0; p < POSITIONS; p = p + 1) begin
                unique case (w)
                    0: begin
                        ref_score[p] = scores[p][0];
                        ref_temporal[p] = 5'b00001;
                        ref_active[p] = {4'b0000, active[p][0]};
                        ref_consume(p, 1'b0, 1, scores[p][1], active[p][1]);
                    end
                    1: begin
                        ref_consume(p, 1'b0, 2, scores[p][2], active[p][2]);
                        ref_consume(p, 1'b0, 3, scores[p][3], active[p][3]);
                    end
                    2: begin
                        ref_consume(p, 1'b0, 4, scores[p][4], active[p][4]);
                        ref_finalize(p, 1'b0, 1'b0);
                        ref_score[p] = scores[p][5];
                        ref_temporal[p] = 5'b00001;
                        ref_active[p] = {4'b0000, active[p][5]};
                    end
                    3: begin
                        ref_consume(p, 1'b1, 1, scores[p][6], active[p][6]);
                        ref_consume(p, 1'b1, 2, scores[p][7], active[p][7]);
                    end
                    4: begin
                        ref_consume(p, 1'b1, 3, scores[p][8], active[p][8]);
                        ref_consume(p, 1'b1, 4, scores[p][9], active[p][9]);
                        ref_finalize(p, 1'b1, p == POSITIONS - 1);
                    end
                endcase
            end
        end

        if (exp_count != 24) begin
            $display("ERROR reference descriptor count=%0d expected=24", exp_count);
            errors = errors + 1;
        end

        repeat (4) @(posedge clk_core);
        rst_core = 1'b0;
        @(negedge clk_core);
        if (!row_start_ready) begin
            $display("ERROR row_start_ready low after reset");
            errors = errors + 1;
        end
        row_start = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        row_start = 1'b0;

        for (w = 0; w < 5; w = w + 1)
            for (p = 0; p < POSITIONS; p = p + 1)
                drive_pair(w, p);

        timeout = 0;
        while (row_done_count == 0 && timeout < 2000) begin
            @(posedge clk_core);
            timeout = timeout + 1;
        end
        repeat (4) @(posedge clk_core);

        if (timeout >= 2000) begin
            $display("ERROR timeout waiting for row_done");
            errors = errors + 1;
        end
        if (exp_index != exp_count) begin
            $display("ERROR descriptors retired=%0d expected=%0d", exp_index, exp_count);
            errors = errors + 1;
        end
        if (row_done_count != 1) begin
            $display("ERROR row_done_count=%0d", row_done_count);
            errors = errors + 1;
        end
        if (protocol_error) begin
            $display("ERROR protocol_error set");
            errors = errors + 1;
        end
        if (perf_input_pairs != 20 || perf_original_slots != 40
            || perf_run_descriptors != 24 || perf_equal_edges != 16
            || perf_spill_packets != 1) begin
            $display("ERROR perf pairs/slots/desc/equal/spill=%0d/%0d/%0d/%0d/%0d",
                     perf_input_pairs, perf_original_slots,
                     perf_run_descriptors, perf_equal_edges, perf_spill_packets);
            errors = errors + 1;
        end

        if (errors == 0)
            $display("PASS h87_t5_open_run_file desc=%0d equal=%0d spill=%0d stalls=%0d",
                     perf_run_descriptors, perf_equal_edges,
                     perf_spill_packets, perf_input_stall_cycles);
        else
            $display("FAIL h87_t5_open_run_file errors=%0d", errors);
        $finish;
    end
endmodule

`default_nettype wire
