`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off BLKSEQ */
/* verilator lint_off UNUSEDSIGNAL */

module tb_gatestack_ppdi_executor_acc_flush;
    localparam int TOKENS = 4;
    localparam int OUT_TILE = 4;
    localparam int TOKEN_W = 2;
    localparam int TAG_W = 12;
    localparam int PRODUCT_W = 17;
    localparam int ACC_W = 32;

    logic clk_core, rst_core, flush, clear_error;
    logic cmd_valid, cmd_ready, cmd_term_first, cmd_term_last, cmd_head_last;
    logic [TAG_W-1:0] cmd_group_tag;
    logic [7:0] cmd_sequence;
    logic [5:0] cmd_term_issue_seq;
    logic [6:0] cmd_input_channel;
    logic [8:0] cmd_gate_code;
    logic [4:0] cmd_lane_id;
    logic [1:0] cmd_destination_valid;
    logic [(2*TOKEN_W)-1:0] cmd_destination_tokens;
    logic [3:0] logical_supertile;
    logic weight_req_valid, weight_req_ready;
    logic [TAG_W-1:0] weight_req_tag;
    logic [6:0] weight_req_input_channel;
    logic [5:0] weight_req_output_tile;
    logic [2:0] weight_req_epoch;
    logic weight_rsp_valid, weight_rsp_ready;
    logic [TAG_W-1:0] weight_rsp_tag;
    logic [6:0] weight_rsp_input_channel;
    logic [5:0] weight_rsp_output_tile;
    logic [2:0] weight_rsp_epoch;
    logic [(OUT_TILE*8)-1:0] weight_rsp_weights;
    logic [1:0] exec_update_valid, exec_update_ready;
    logic [(2*TOKEN_W)-1:0] exec_update_tokens;
    logic [TAG_W-1:0] exec_update_tag;
    logic [(OUT_TILE*PRODUCT_W)-1:0] exec_update_values;
    logic term_done, term_done_head_last, exec_protocol_error;
    logic [TAG_W-1:0] term_done_group_tag;
    logic [5:0] term_done_issue_seq;
    logic [31:0] stale_count;

    logic group_start_valid, group_start_ready;
    logic [TAG_W-1:0] group_start_tag;
    logic [1:0] acc_update_valid, acc_update_ready;
    logic [(2*TOKEN_W)-1:0] acc_update_tokens;
    logic [TAG_W-1:0] acc_update_tag;
    logic acc_update_is_bias;
    logic [(OUT_TILE*PRODUCT_W)-1:0] acc_update_values;
    logic [(OUT_TILE*ACC_W)-1:0] acc_bias_values;
    logic [1:0] final_valid, final_ready;
    logic [(2*TOKEN_W)-1:0] final_token_ids;
    logic [TAG_W-1:0] final_tag;
    logic [(2*OUT_TILE*ACC_W)-1:0] final_values;
    logic group_finish_valid, group_finish_ready;
    logic [TAG_W-1:0] group_finish_tag;
    logic acc_protocol_error, accumulator_overflow;
    logic [31:0] count_updates, count_writes, count_bias_commits;
    logic [31:0] count_bank_stall_cycles, count_final_stall_cycles;

    logic [1:0] exec_port_allow;
    logic bias_mode;
    logic [1:0] bias_valid;
    logic [(2*TOKEN_W)-1:0] bias_tokens;
    logic [TAG_W-1:0] bias_tag;
    integer old_even_commits;
    integer old_odd_commits;
    integer final_token2_seen;

    assign exec_update_ready = bias_mode ? 2'b00 :
                               (acc_update_ready & exec_port_allow);
    assign acc_update_valid = bias_mode ? bias_valid :
                              (exec_update_valid & exec_port_allow);
    assign acc_update_tokens = bias_mode ? bias_tokens : exec_update_tokens;
    assign acc_update_tag = bias_mode ? bias_tag : exec_update_tag;
    assign acc_update_is_bias = bias_mode;
    assign acc_update_values = exec_update_values;

    gatestack_ppdi_dctf32_bank_executor #(
        .BANK_ID(0), .BANK_COUNT(3), .TOKENS(TOKENS),
        .OUT_TILE(OUT_TILE), .GATE_W(9), .WEIGHT_W(8),
        .PRODUCT_W(PRODUCT_W), .GROUP_TAG_W(TAG_W),
        .CMD_SEQUENCE_W(8), .ISSUE_SEQ_W(6), .INPUT_CH_W(7),
        .LANE_ID_W(5), .TOKEN_ID_W(TOKEN_W), .OUTPUT_TILE_W(6),
        .LOGICAL_SUPERTILE_W(4), .EPOCH_W(3)
    ) u_executor (
        .clk_core(clk_core), .rst_core(rst_core), .flush(flush),
        .clear_error(clear_error), .cmd_valid(cmd_valid),
        .cmd_ready(cmd_ready), .cmd_group_tag(cmd_group_tag),
        .cmd_sequence(cmd_sequence), .cmd_term_issue_seq(cmd_term_issue_seq),
        .cmd_term_first(cmd_term_first), .cmd_term_last(cmd_term_last),
        .cmd_head_last(cmd_head_last),
        .cmd_input_channel(cmd_input_channel),
        .cmd_gate_code(cmd_gate_code), .cmd_lane_id(cmd_lane_id),
        .cmd_destination_valid(cmd_destination_valid),
        .cmd_destination_tokens(cmd_destination_tokens),
        .logical_supertile(logical_supertile),
        .weight_req_valid(weight_req_valid),
        .weight_req_ready(weight_req_ready), .weight_req_tag(weight_req_tag),
        .weight_req_input_channel(weight_req_input_channel),
        .weight_req_output_tile(weight_req_output_tile),
        .weight_req_epoch(weight_req_epoch),
        .weight_rsp_valid(weight_rsp_valid),
        .weight_rsp_ready(weight_rsp_ready), .weight_rsp_tag(weight_rsp_tag),
        .weight_rsp_input_channel(weight_rsp_input_channel),
        .weight_rsp_output_tile(weight_rsp_output_tile),
        .weight_rsp_epoch(weight_rsp_epoch),
        .weight_rsp_weights(weight_rsp_weights),
        .acc_update_valid(exec_update_valid),
        .acc_update_ready(exec_update_ready),
        .acc_update_token_ids(exec_update_tokens),
        .acc_update_tag(exec_update_tag),
        .acc_update_values(exec_update_values), .term_done(term_done),
        .term_done_group_tag(term_done_group_tag),
        .term_done_issue_seq(term_done_issue_seq),
        .term_done_head_last(term_done_head_last),
        .protocol_error(exec_protocol_error),
        .count_stale_weight_responses(stale_count)
    );

    hitflow_banked_accumulator #(
        .TOKENS(TOKENS), .BANKS(2), .PRODUCT_W(PRODUCT_W),
        .ACC_W(ACC_W), .OUT_TILE(OUT_TILE), .TAG_W(TAG_W),
        .TOKEN_ID_W(TOKEN_W)
    ) u_accumulator (
        .clk_core(clk_core), .rst_core(rst_core), .flush(flush),
        .group_start_valid(group_start_valid),
        .group_start_ready(group_start_ready),
        .group_start_tag(group_start_tag), .update_valid(acc_update_valid),
        .update_ready(acc_update_ready), .update_token_ids(acc_update_tokens),
        .update_tag(acc_update_tag), .update_is_bias(acc_update_is_bias),
        .update_values(acc_update_values),
        .update_bias_values(acc_bias_values), .final_valid(final_valid),
        .final_ready(final_ready), .final_token_ids(final_token_ids),
        .final_tag(final_tag), .final_values(final_values),
        .group_finish_valid(group_finish_valid),
        .group_finish_ready(group_finish_ready),
        .group_finish_tag(group_finish_tag),
        .protocol_error(acc_protocol_error),
        .accumulator_overflow(accumulator_overflow),
        .count_updates(count_updates), .count_writes(count_writes),
        .count_bias_commits(count_bias_commits),
        .count_bank_stall_cycles(count_bank_stall_cycles),
        .count_final_stall_cycles(count_final_stall_cycles)
    );

    always #5 clk_core = ~clk_core;

    task automatic start_group(input logic [TAG_W-1:0] tag_value);
        begin
            @(negedge clk_core);
            group_start_tag = tag_value;
            group_start_valid = 1'b1;
            do @(posedge clk_core); while (!group_start_ready);
            @(negedge clk_core);
            group_start_valid = 1'b0;
        end
    endtask

    task automatic set_command(
        input logic [8:0] gate_value,
        input logic [1:0] valid_value
    );
        begin
            @(negedge clk_core);
            cmd_group_tag = 12'ha55;
            cmd_sequence = 8'd0;
            cmd_term_issue_seq = 6'd3;
            cmd_term_first = 1'b1;
            cmd_term_last = 1'b1;
            cmd_head_last = 1'b0;
            cmd_input_channel = 7'd9;
            cmd_gate_code = gate_value;
            cmd_lane_id = 5'd1;
            cmd_destination_valid = valid_value;
            cmd_destination_tokens = {2'd3, 2'd2};
            logical_supertile = 4'd0;
            cmd_valid = 1'b1;
        end
    endtask

    task automatic serve_weight(input integer base_value);
        integer lane;
        logic [2:0] epoch_value;
        begin
            @(negedge clk_core);
            weight_req_ready = 1'b1;
            do @(posedge clk_core); while (!weight_req_valid);
            epoch_value = weight_req_epoch;
            @(negedge clk_core);
            weight_req_ready = 1'b0;
            weight_rsp_tag = 12'ha55;
            weight_rsp_input_channel = 7'd9;
            weight_rsp_output_tile = 6'd0;
            weight_rsp_epoch = epoch_value;
            for (lane = 0; lane < OUT_TILE; lane = lane + 1)
                weight_rsp_weights[(lane*8) +: 8] = 8'(base_value + lane);
            weight_rsp_valid = 1'b1;
            do @(posedge clk_core); while (!weight_rsp_ready);
            @(negedge clk_core);
            weight_rsp_valid = 1'b0;
        end
    endtask

    task automatic commit_bias(input integer token_value);
        begin
            @(negedge clk_core);
            bias_tokens = '0;
            bias_tokens[((token_value & 1)*TOKEN_W) +: TOKEN_W] =
                TOKEN_W'(token_value);
            bias_valid = token_value[0] ? 2'b10 : 2'b01;
            do @(posedge clk_core);
            while ((bias_valid & acc_update_ready) == 2'b00);
            @(negedge clk_core);
            bias_valid = '0;
        end
    endtask

    always @(posedge clk_core) begin : p_final_scoreboard
        integer bank;
        integer token;
        integer lane;
        integer signed observed;
        integer expected;
        if (!rst_core && !flush) begin
            if (exec_update_valid[0] && exec_update_ready[0] &&
                cmd_gate_code == 9'd7)
                old_even_commits = old_even_commits + 1;
            if (exec_update_valid[1] && exec_update_ready[1] &&
                cmd_gate_code == 9'd7)
                old_odd_commits = old_odd_commits + 1;
            for (bank = 0; bank < 2; bank = bank + 1) begin
                if (final_valid[bank] && final_ready[bank]) begin
                    token = 32'(final_token_ids[(bank*TOKEN_W) +: TOKEN_W]);
                    if (final_tag != 12'ha55)
                        $fatal(1, "PPDI/Acc final tag mismatch");
                    if (token == 2) begin
                        for (lane = 0; lane < OUT_TILE; lane = lane + 1) begin
                            observed = 32'($signed(final_values[
                                (bank*OUT_TILE*ACC_W) + (lane*ACC_W) +:
                                ACC_W]));
                            expected = 2 * (3 + lane);
                            if (observed != expected)
                                $fatal(1,
                                    "old partial write leaked lane=%0d got=%0d exp=%0d",
                                    lane, observed, expected);
                        end
                        final_token2_seen = final_token2_seen + 1;
                    end
                end
            end
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        flush = 1'b0;
        clear_error = 1'b0;
        cmd_valid = 1'b0;
        weight_req_ready = 1'b0;
        weight_rsp_valid = 1'b0;
        group_start_valid = 1'b0;
        group_finish_valid = 1'b0;
        final_ready = 2'b11;
        exec_port_allow = 2'b01;
        bias_mode = 1'b0;
        bias_valid = '0;
        bias_tokens = '0;
        bias_tag = 12'ha55;
        acc_bias_values = '0;
        old_even_commits = 0;
        old_odd_commits = 0;
        final_token2_seen = 0;
        repeat (4) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        start_group(12'ha55);
        fork
            set_command(9'd7, 2'b11);
            serve_weight(10);
        join
        while (old_even_commits == 0) @(posedge clk_core);
        repeat (3) @(posedge clk_core);
        if (old_even_commits != 1 || old_odd_commits != 0 || term_done ||
            exec_update_valid != 2'b10)
            $fatal(1, "PPDI partial pre-flush setup mismatch");

        @(negedge clk_core);
        flush = 1'b1;
        #1;
        if (exec_update_valid != 2'b00 || acc_update_ready != 2'b00 ||
            final_valid != 2'b00 || term_done)
            $fatal(1, "PPDI/Acc common flush did not mask interfaces");
        @(posedge clk_core);
        @(negedge clk_core);
        flush = 1'b0;
        cmd_valid = 1'b0;

        start_group(12'ha55);
        fork
            begin
                set_command(9'd2, 2'b01);
                do @(posedge clk_core); while (!cmd_ready);
                @(negedge clk_core);
                cmd_valid = 1'b0;
            end
            serve_weight(3);
        join
        if (!term_done && exec_protocol_error)
            $fatal(1, "PPDI replacement execution error");
        repeat (3) @(posedge clk_core);

        bias_mode = 1'b1;
        for (integer token = 0; token < TOKENS; token = token + 1)
            commit_bias(token);
        while (count_bias_commits != 4) @(posedge clk_core);
        repeat (3) @(posedge clk_core);
        group_finish_valid = 1'b1;
        while (!group_finish_ready) @(negedge clk_core);
        @(posedge clk_core);
        @(negedge clk_core);
        group_finish_valid = 1'b0;

        if (final_token2_seen != 1 || old_even_commits != 1 ||
            old_odd_commits != 0 || exec_protocol_error ||
            acc_protocol_error || accumulator_overflow)
            $fatal(1,
                "PPDI/Acc partial flush recovery mismatch final2=%0d old=%0d/%0d",
                final_token2_seen, old_even_commits, old_odd_commits);

        $display("PASS PPDI EXECUTOR ACC FLUSH old_partial=1/0 replacement_final_token2=%0d bias=%0d updates=%0d writes=%0d",
                 final_token2_seen, count_bias_commits, count_updates,
                 count_writes);
        $finish;
    end

    initial begin
        repeat (3000) @(posedge clk_core);
        $display("TIMEOUT flush=%b cmd=%b/%b term_done=%b weight=%b/%b rsp=%b/%b group_start=%b/%b update=%b/%b bias=%b final=%b/%b finish=%b/%b exec_state=%0d acc_active=%b writes=%0d bias_count=%0d final2=%0d",
                 flush, cmd_valid, cmd_ready, term_done,
                 weight_req_valid, weight_req_ready, weight_rsp_valid,
                 weight_rsp_ready, group_start_valid, group_start_ready,
                 acc_update_valid, acc_update_ready, bias_mode,
                 final_valid, final_ready, group_finish_valid,
                 group_finish_ready, u_executor.u_product_engine.state_q,
                 u_accumulator.group_active_q, count_writes,
                 count_bias_commits, final_token2_seen);
        $fatal(1, "PPDI executor/Acc flush TB timeout");
    end
endmodule

/* verilator lint_on UNUSEDSIGNAL */
/* verilator lint_on BLKSEQ */

`default_nettype wire
