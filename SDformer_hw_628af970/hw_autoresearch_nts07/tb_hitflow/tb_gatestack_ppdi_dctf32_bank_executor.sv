`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off BLKSEQ */
/* verilator lint_off UNUSEDSIGNAL */

module tb_gatestack_ppdi_dctf32_bank_executor;
    localparam int BANK_ID = 1;
    localparam int TOKENS = 18;
    localparam int OUT_TILE = 32;
    localparam int GATE_W = 9;
    localparam int WEIGHT_W = 8;
    localparam int PRODUCT_W = 17;
    localparam int GROUP_TAG_W = 12;
    localparam int CMD_SEQUENCE_W = 8;
    localparam int ISSUE_SEQ_W = 6;
    localparam int INPUT_CH_W = 7;
    localparam int LANE_ID_W = 5;
    localparam int TOKEN_ID_W = 5;
    localparam int OUTPUT_TILE_W = 6;
    localparam int LOGICAL_SUPERTILE_W = 4;
    localparam int EPOCH_W = 3;
    localparam int EPOCH_COUNT = 1 << EPOCH_W;

    logic clk_core;
    logic rst_core;
    logic flush;
    logic clear_error;
    logic cmd_valid;
    logic cmd_ready;
    logic [GROUP_TAG_W-1:0] cmd_group_tag;
    logic [CMD_SEQUENCE_W-1:0] cmd_sequence;
    logic [ISSUE_SEQ_W-1:0] cmd_term_issue_seq;
    logic cmd_term_first;
    logic cmd_term_last;
    logic cmd_head_last;
    logic [INPUT_CH_W-1:0] cmd_input_channel;
    logic [GATE_W-1:0] cmd_gate_code;
    logic [LANE_ID_W-1:0] cmd_lane_id;
    logic [1:0] cmd_destination_valid;
    logic [(2*TOKEN_ID_W)-1:0] cmd_destination_tokens;
    logic [LOGICAL_SUPERTILE_W-1:0] logical_supertile;
    logic weight_req_valid;
    logic weight_req_ready;
    logic [GROUP_TAG_W-1:0] weight_req_tag;
    logic [INPUT_CH_W-1:0] weight_req_input_channel;
    logic [OUTPUT_TILE_W-1:0] weight_req_output_tile;
    logic [EPOCH_W-1:0] weight_req_epoch;
    logic weight_rsp_valid;
    logic weight_rsp_ready;
    logic [GROUP_TAG_W-1:0] weight_rsp_tag;
    logic [INPUT_CH_W-1:0] weight_rsp_input_channel;
    logic [OUTPUT_TILE_W-1:0] weight_rsp_output_tile;
    logic [EPOCH_W-1:0] weight_rsp_epoch;
    logic [(OUT_TILE*WEIGHT_W)-1:0] weight_rsp_weights;
    logic [1:0] acc_update_valid;
    logic [1:0] acc_update_ready;
    logic [(2*TOKEN_ID_W)-1:0] acc_update_token_ids;
    logic [GROUP_TAG_W-1:0] acc_update_tag;
    logic [(OUT_TILE*PRODUCT_W)-1:0] acc_update_values;
    logic term_done;
    logic [GROUP_TAG_W-1:0] term_done_group_tag;
    logic [ISSUE_SEQ_W-1:0] term_done_issue_seq;
    logic term_done_head_last;
    logic protocol_error;
    logic [31:0] count_stale_weight_responses;

    integer cycle_count;
    integer weight_request_count;
    integer acc_fire_count [0:1];
    integer command_count;
    integer term_done_count;
    integer token_fire_count [0:TOKENS-1];
    integer expected_weight_base;
    integer total_cycles;
    integer total_weight_requests;
    integer total_acc_fires [0:1];
    integer total_commands;
    integer total_term_done;
    integer total_stale_drops;
    integer total_zero_gate_commands;
    integer total_zero_gate_term_done;
    logic split_pair_seen;
    logic same_cycle_pair_seen;
    logic only_even_seen;
    logic only_odd_seen;
    logic partial_flush_seen;
    logic malformed_seen;
    logic stale_seen;
    logic stale_during_acc_stall_seen;
    logic child_clear_seen;
    logic paired_continuation_seen;
    logic odd_first_seen;
    logic epoch_wrap_guard_seen;
    logic zero_single_pair_seen;
    logic zero_multi_parity_seen;
    logic zero_head_last_seen;
    logic zero_first_stable_seen;
    logic zero_flush_seen;
    logic [EPOCH_W-1:0] wrap_epochs [0:EPOCH_COUNT-1];

    gatestack_ppdi_dctf32_bank_executor #(
        .BANK_ID(BANK_ID), .TOKENS(TOKENS), .OUT_TILE(OUT_TILE),
        .GATE_W(GATE_W), .WEIGHT_W(WEIGHT_W), .PRODUCT_W(PRODUCT_W),
        .GROUP_TAG_W(GROUP_TAG_W), .CMD_SEQUENCE_W(CMD_SEQUENCE_W),
        .ISSUE_SEQ_W(ISSUE_SEQ_W), .INPUT_CH_W(INPUT_CH_W),
        .LANE_ID_W(LANE_ID_W), .TOKEN_ID_W(TOKEN_ID_W),
        .OUTPUT_TILE_W(OUTPUT_TILE_W),
        .LOGICAL_SUPERTILE_W(LOGICAL_SUPERTILE_W), .EPOCH_W(EPOCH_W)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    task automatic apply_reset;
        integer token;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            flush = 1'b0;
            clear_error = 1'b0;
            cmd_valid = 1'b0;
            weight_req_ready = 1'b0;
            weight_rsp_valid = 1'b0;
            acc_update_ready = '0;
            repeat (3) @(posedge clk_core);
            for (token = 0; token < TOKENS; token = token + 1)
                token_fire_count[token] = 0;
            @(negedge clk_core);
            rst_core = 1'b0;
        end
    endtask

    task automatic set_command(
        input logic [GROUP_TAG_W-1:0] tag_value,
        input logic [CMD_SEQUENCE_W-1:0] sequence_value,
        input logic [ISSUE_SEQ_W-1:0] issue_value,
        input logic first_value,
        input logic last_value,
        input logic head_last_value,
        input logic [INPUT_CH_W-1:0] channel_value,
        input logic [GATE_W-1:0] gate_value,
        input logic [LANE_ID_W-1:0] lane_value,
        input logic [1:0] destination_valid_value,
        input logic [TOKEN_ID_W-1:0] even_token_value,
        input logic [TOKEN_ID_W-1:0] odd_token_value,
        input logic [LOGICAL_SUPERTILE_W-1:0] supertile_value
    );
        begin
            @(negedge clk_core);
            cmd_group_tag = tag_value;
            cmd_sequence = sequence_value;
            cmd_term_issue_seq = issue_value;
            cmd_term_first = first_value;
            cmd_term_last = last_value;
            cmd_head_last = head_last_value;
            cmd_input_channel = channel_value;
            cmd_gate_code = gate_value;
            cmd_lane_id = lane_value;
            cmd_destination_valid = destination_valid_value;
            cmd_destination_tokens = {odd_token_value, even_token_value};
            logical_supertile = supertile_value;
            cmd_valid = 1'b1;
        end
    endtask

    task automatic wait_command_accept;
        begin
            do @(posedge clk_core); while (!cmd_ready);
            @(negedge clk_core);
            cmd_valid = 1'b0;
        end
    endtask

    task automatic capture_weight_request(
        input logic [GROUP_TAG_W-1:0] expected_tag,
        input logic [INPUT_CH_W-1:0] expected_channel,
        input logic [OUTPUT_TILE_W-1:0] expected_tile,
        output logic [EPOCH_W-1:0] captured_epoch
    );
        begin
            while (!weight_req_valid) @(posedge clk_core);
            repeat (2) @(posedge clk_core);
            @(negedge clk_core);
            weight_req_ready = 1'b1;
            do @(posedge clk_core); while (!weight_req_valid);
            if (weight_req_tag != expected_tag ||
                weight_req_input_channel != expected_channel ||
                weight_req_output_tile != expected_tile)
                $fatal(1, "PPDI weight request identity mismatch");
            captured_epoch = weight_req_epoch;
            @(negedge clk_core);
            weight_req_ready = 1'b0;
        end
    endtask

    task automatic send_weight_response(
        input logic [GROUP_TAG_W-1:0] response_tag,
        input logic [INPUT_CH_W-1:0] response_channel,
        input logic [OUTPUT_TILE_W-1:0] response_tile,
        input logic [EPOCH_W-1:0] response_epoch,
        input integer weight_base
    );
        integer lane;
        begin
            expected_weight_base = weight_base;
            @(negedge clk_core);
            weight_rsp_tag = response_tag;
            weight_rsp_input_channel = response_channel;
            weight_rsp_output_tile = response_tile;
            weight_rsp_epoch = response_epoch;
            for (lane = 0; lane < OUT_TILE; lane = lane + 1)
                weight_rsp_weights[(lane*WEIGHT_W) +: WEIGHT_W] =
                    WEIGHT_W'(weight_base + lane);
            weight_rsp_valid = 1'b1;
            do @(posedge clk_core); while (!weight_rsp_ready);
            @(negedge clk_core);
            weight_rsp_valid = 1'b0;
        end
    endtask

    task automatic serve_weight(
        input logic [GROUP_TAG_W-1:0] expected_tag,
        input logic [INPUT_CH_W-1:0] expected_channel,
        input logic [OUTPUT_TILE_W-1:0] expected_tile,
        input integer weight_base
    );
        logic [EPOCH_W-1:0] captured_epoch;
        begin
            capture_weight_request(expected_tag, expected_channel,
                                   expected_tile, captured_epoch);
            send_weight_response(expected_tag, expected_channel,
                                 expected_tile, captured_epoch, weight_base);
        end
    endtask

    task automatic accept_mask(input logic [1:0] ready_mask);
        begin
            while (acc_update_valid == 2'b00) @(posedge clk_core);
            @(negedge clk_core);
            acc_update_ready = ready_mask;
            do @(posedge clk_core);
            while ((acc_update_valid & ready_mask) == 2'b00);
            @(negedge clk_core);
            acc_update_ready = '0;
        end
    endtask

    always @(posedge clk_core) begin : p_scoreboard
        integer port;
        integer token;
        integer lane;
        integer signed observed_product;
        integer signed expected_product;
        if (rst_core) begin
            cycle_count = 0;
            weight_request_count = 0;
            acc_fire_count[0] = 0;
            acc_fire_count[1] = 0;
            command_count = 0;
            term_done_count = 0;
        end else begin
            cycle_count = cycle_count + 1;
            total_cycles = total_cycles + 1;
            if (weight_req_valid && weight_req_ready) begin
                weight_request_count = weight_request_count + 1;
                total_weight_requests = total_weight_requests + 1;
            end
            if (cmd_valid && cmd_ready) begin
                command_count = command_count + 1;
                total_commands = total_commands + 1;
                if (cmd_gate_code == '0)
                    total_zero_gate_commands = total_zero_gate_commands + 1;
            end
            if (dut.stale_weight_response_fire)
                total_stale_drops = total_stale_drops + 1;
            for (port = 0; port < 2; port = port + 1) begin
                if (acc_update_valid[port] && acc_update_ready[port]) begin
                    token = 32'(acc_update_token_ids[
                        (port*TOKEN_ID_W) +: TOKEN_ID_W]);
                    if ((token & 1) != port ||
                        token != 32'(cmd_destination_tokens[
                            (port*TOKEN_ID_W) +: TOKEN_ID_W]) ||
                        !cmd_destination_valid[port] ||
                        acc_update_tag != cmd_group_tag)
                        $fatal(1, "PPDI Acc identity mismatch port=%0d token=%0d",
                               port, token);
                    if (token_fire_count[token] != 0)
                        $fatal(1, "PPDI duplicate token update token=%0d", token);
                    token_fire_count[token] = 1;
                    acc_fire_count[port] = acc_fire_count[port] + 1;
                    total_acc_fires[port] = total_acc_fires[port] + 1;
                    for (lane = 0; lane < OUT_TILE; lane = lane + 1) begin
                        observed_product = 32'($signed(acc_update_values[
                            (lane*PRODUCT_W) +: PRODUCT_W]));
                        expected_product = 32'(cmd_gate_code) *
                                           (expected_weight_base + lane);
                        if (observed_product != expected_product)
                            $fatal(1,
                                "PPDI product mismatch port=%0d lane=%0d got=%0d exp=%0d",
                                port, lane, observed_product,
                                expected_product);
                    end
                end
            end
            if (term_done) begin
                if (!cmd_valid || !cmd_ready || !cmd_term_last ||
                    term_done_group_tag != cmd_group_tag ||
                    term_done_issue_seq != cmd_term_issue_seq ||
                    term_done_head_last != cmd_head_last)
                    $fatal(1, "PPDI term completion mismatch");
                term_done_count = term_done_count + 1;
                total_term_done = total_term_done + 1;
                if (cmd_gate_code == '0)
                    total_zero_gate_term_done =
                        total_zero_gate_term_done + 1;
            end
            if (flush && (cmd_ready || weight_req_valid || weight_rsp_ready ||
                          acc_update_valid != 2'b00 || term_done))
                $fatal(1, "PPDI flush output masking failed");
        end
    end

    initial begin : p_test
        logic [EPOCH_W-1:0] canceled_epoch;
        logic [EPOCH_W-1:0] replacement_epoch;
        integer token;
        clk_core = 1'b0;
        rst_core = 1'b1;
        flush = 1'b0;
        clear_error = 1'b0;
        cmd_valid = 1'b0;
        cmd_group_tag = '0;
        cmd_sequence = '0;
        cmd_term_issue_seq = '0;
        cmd_term_first = 1'b0;
        cmd_term_last = 1'b0;
        cmd_head_last = 1'b0;
        cmd_input_channel = '0;
        cmd_gate_code = '0;
        cmd_lane_id = '0;
        cmd_destination_valid = '0;
        cmd_destination_tokens = '0;
        logical_supertile = '0;
        weight_req_ready = 1'b0;
        weight_rsp_valid = 1'b0;
        weight_rsp_tag = '0;
        weight_rsp_input_channel = '0;
        weight_rsp_output_tile = '0;
        weight_rsp_epoch = '0;
        weight_rsp_weights = '0;
        acc_update_ready = '0;
        expected_weight_base = 0;
        total_cycles = 0;
        total_weight_requests = 0;
        total_acc_fires[0] = 0;
        total_acc_fires[1] = 0;
        total_commands = 0;
        total_term_done = 0;
        total_stale_drops = 0;
        total_zero_gate_commands = 0;
        total_zero_gate_term_done = 0;
        split_pair_seen = 1'b0;
        same_cycle_pair_seen = 1'b0;
        only_even_seen = 1'b0;
        only_odd_seen = 1'b0;
        partial_flush_seen = 1'b0;
        malformed_seen = 1'b0;
        stale_seen = 1'b0;
        stale_during_acc_stall_seen = 1'b0;
        child_clear_seen = 1'b0;
        paired_continuation_seen = 1'b0;
        odd_first_seen = 1'b0;
        epoch_wrap_guard_seen = 1'b0;
        zero_single_pair_seen = 1'b0;
        zero_multi_parity_seen = 1'b0;
        zero_head_last_seen = 1'b0;
        zero_first_stable_seen = 1'b0;
        zero_flush_seen = 1'b0;
        for (token = 0; token < TOKENS; token = token + 1)
            token_fire_count[token] = 0;

        repeat (4) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        // A paired zero-gate term is latched without touching the product
        // engine. The source holds the first command until next-cycle retire.
        set_command(12'h101, 8'd0, 6'd1, 1'b1, 1'b1, 1'b1,
                    7'd3, 9'd0, 5'd1, 2'b11, 5'd0, 5'd1, 4'd1);
        if (cmd_ready || weight_req_valid || acc_update_valid != 2'b00)
            $fatal(1, "PPDI zero-gate first beat was not latch-only");
        @(posedge clk_core);
        #1;
        if (!cmd_valid || !cmd_ready || weight_req_valid ||
            acc_update_valid != 2'b00 || !term_done ||
            !term_done_head_last)
            $fatal(1, "PPDI zero-gate paired retire mismatch");
        zero_first_stable_seen = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        cmd_valid = 1'b0;
        if (weight_request_count != 0 || acc_fire_count[0] != 0 ||
            acc_fire_count[1] != 0 || command_count != 1 ||
            term_done_count != 1)
            $fatal(1, "PPDI zero-gate paired accounting mismatch");
        zero_single_pair_seen = 1'b1;
        zero_head_last_seen = 1'b1;

        // A multi-command zero-gate term preserves sequence and parity masks;
        // each whole command retires while no add-zero update is emitted.
        apply_reset();
        set_command(12'h202, 8'd0, 6'd2, 1'b1, 1'b0, 1'b0,
                    7'd4, 9'd0, 5'd2, 2'b01, 5'd2, 5'd0, 4'd2);
        wait_command_accept();
        set_command(12'h202, 8'd1, 6'd2, 1'b0, 1'b1, 1'b0,
                    7'd4, 9'd0, 5'd2, 2'b10, 5'd0, 5'd3, 4'd2);
        wait_command_accept();
        if (weight_request_count != 0 || acc_fire_count[0] != 0 ||
            acc_fire_count[1] != 0 || command_count != 2 ||
            term_done_count != 1)
            $fatal(1, "PPDI zero-gate multi-command accounting mismatch");
        zero_multi_parity_seen = 1'b1;

        // Flush after a non-last zero command. The offered continuation is
        // canceled with no completion, weight access, or accumulator update.
        apply_reset();
        set_command(12'h303, 8'd0, 6'd3, 1'b1, 1'b0, 1'b0,
                    7'd5, 9'd0, 5'd3, 2'b01, 5'd4, 5'd0, 4'd1);
        wait_command_accept();
        set_command(12'h303, 8'd1, 6'd3, 1'b0, 1'b1, 1'b1,
                    7'd5, 9'd0, 5'd3, 2'b10, 5'd0, 5'd5, 4'd1);
        flush = 1'b1;
        #1;
        if (cmd_ready || weight_req_valid || weight_rsp_ready ||
            acc_update_valid != 2'b00 || term_done || term_done_head_last)
            $fatal(1, "PPDI zero-gate flush was not combinational");
        @(posedge clk_core);
        @(negedge clk_core);
        flush = 1'b0;
        cmd_valid = 1'b0;
        if (weight_request_count != 0 || acc_fire_count[0] != 0 ||
            acc_fire_count[1] != 0 || command_count != 1 ||
            term_done_count != 0 || dut.term_active_q)
            $fatal(1, "PPDI zero-gate flush lifecycle mismatch");
        zero_flush_seen = 1'b1;

        // Seed one canceled request so a genuine pending stale response can
        // arrive while the following pair is stalled at its odd Acc port.
        apply_reset();
        set_command(12'h440, 8'd0, 6'd10, 1'b1, 1'b1, 1'b0,
                    7'd22, 9'd2, 5'd1, 2'b01, 5'd0, 5'd0, 4'd3);
        capture_weight_request(12'h440, 7'd22, 6'd10, canceled_epoch);
        @(negedge clk_core);
        flush = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        flush = 1'b0;
        cmd_valid = 1'b0;

        // Split a paired last command: even commits first, odd remains valid.
        fork
            begin
                set_command(12'h451, 8'd0, 6'd11, 1'b1, 1'b1, 1'b1,
                            7'd23, 9'd3, 5'd7, 2'b11, 5'd2, 5'd3, 4'd3);
                wait_command_accept();
            end
            serve_weight(12'h451, 7'd23, 6'd10, -16);
            begin
                while (acc_update_valid != 2'b11) @(posedge clk_core);
                accept_mask(2'b01);
                #1;
                if (acc_update_valid != 2'b10 || cmd_ready || term_done)
                    $fatal(1, "PPDI split pair did not retain odd destination");
                @(negedge clk_core);
                weight_rsp_epoch = canceled_epoch;
                weight_rsp_tag = 12'h441;
                weight_rsp_input_channel = 7'd22;
                weight_rsp_output_tile = 6'd10;
                weight_rsp_valid = 1'b1;
                @(posedge clk_core);
                if (!weight_rsp_ready || acc_update_valid != 2'b10 ||
                    cmd_ready || term_done)
                    $fatal(1,
                        "PPDI stale drain disturbed stalled odd destination");
                @(negedge clk_core);
                weight_rsp_valid = 1'b0;
                #1;
                if (acc_update_valid != 2'b10 ||
                    count_stale_weight_responses != 1 ||
                    !dut.stale_epoch_pending_q[canceled_epoch] ||
                    !protocol_error)
                    $fatal(1,
                        "PPDI wrong stale identity cleared pending generation");
                clear_error = 1'b1;
                @(posedge clk_core);
                @(negedge clk_core);
                clear_error = 1'b0;
                weight_rsp_epoch = canceled_epoch;
                weight_rsp_tag = 12'h440;
                weight_rsp_input_channel = 7'd22;
                weight_rsp_output_tile = 6'd10;
                weight_rsp_valid = 1'b1;
                @(posedge clk_core);
                if (!weight_rsp_ready || acc_update_valid != 2'b10)
                    $fatal(1, "PPDI correct pending stale was not drained");
                @(negedge clk_core);
                weight_rsp_valid = 1'b0;
                #1;
                if (acc_update_valid != 2'b10 ||
                    count_stale_weight_responses != 2 ||
                    dut.stale_epoch_pending_q[canceled_epoch])
                    $fatal(1, "PPDI stalled valid did not survive stale drain");
                stale_during_acc_stall_seen = 1'b1;
                stale_seen = 1'b1;
                repeat (2) @(posedge clk_core);
                accept_mask(2'b10);
            end
        join
        if (weight_request_count != 2 || acc_fire_count[0] != 1 ||
            acc_fire_count[1] != 1 || command_count != 1 ||
            term_done_count != 1)
            $fatal(1, "PPDI split-pair accounting mismatch");
        split_pair_seen = 1'b1;

        // One product is reused by an even-only then odd-only command.
        apply_reset();
        fork
            begin
                set_command(12'h522, 8'd0, 6'd12, 1'b1, 1'b0, 1'b0,
                            7'd19, 9'd5, 5'd2, 2'b01, 5'd4, 5'd0, 4'd2);
                wait_command_accept();
            end
            serve_weight(12'h522, 7'd19, 6'd7, -8);
            accept_mask(2'b01);
        join
        only_even_seen = 1'b1;
        fork
            begin
                set_command(12'h522, 8'd1, 6'd12, 1'b0, 1'b1, 1'b0,
                            7'd19, 9'd5, 5'd2, 2'b10, 5'd0, 5'd5, 4'd2);
                wait_command_accept();
            end
            accept_mask(2'b10);
        join
        only_odd_seen = 1'b1;
        if (weight_request_count != 1 || command_count != 2 ||
            acc_fire_count[0] != 1 || acc_fire_count[1] != 1 ||
            term_done_count != 1)
            $fatal(1, "PPDI single-parity product reuse mismatch");

        // Both destinations commit in one cycle.
        apply_reset();
        fork
            begin
                set_command(12'h533, 8'd0, 6'd13, 1'b1, 1'b1, 1'b0,
                            7'd17, 9'd2, 5'd1, 2'b11, 5'd6, 5'd7, 4'd1);
                wait_command_accept();
            end
            serve_weight(12'h533, 7'd17, 6'd4, -4);
            accept_mask(2'b11);
        join
        if (acc_fire_count[0] != 1 || acc_fire_count[1] != 1 ||
            command_count != 1 || term_done_count != 1)
            $fatal(1, "PPDI same-cycle pair mismatch");
        same_cycle_pair_seen = 1'b1;

        // A paired non-last command keeps the product resident; the paired
        // last continuation commits odd first and releases the product once.
        apply_reset();
        fork
            begin
                set_command(12'h544, 8'd0, 6'd14, 1'b1, 1'b0, 1'b0,
                            7'd21, 9'd4, 5'd3, 2'b11, 5'd8, 5'd9, 4'd2);
                wait_command_accept();
            end
            serve_weight(12'h544, 7'd21, 6'd7, -10);
            accept_mask(2'b11);
        join
        if (term_done_count != 0 || weight_request_count != 1)
            $fatal(1, "PPDI paired non-last released product early");
        fork
            begin
                set_command(12'h544, 8'd1, 6'd14, 1'b0, 1'b1, 1'b0,
                            7'd21, 9'd4, 5'd3, 2'b11, 5'd10, 5'd11, 4'd2);
                wait_command_accept();
            end
            begin
                while (acc_update_valid != 2'b11) @(posedge clk_core);
                accept_mask(2'b10);
                #1;
                if (acc_update_valid != 2'b01 || cmd_ready || term_done)
                    $fatal(1, "PPDI odd-first split lost even destination");
                odd_first_seen = 1'b1;
                accept_mask(2'b01);
            end
        join
        if (weight_request_count != 1 || command_count != 2 ||
            acc_fire_count[0] != 2 || acc_fire_count[1] != 2 ||
            term_done_count != 1)
            $fatal(1, "PPDI paired continuation accounting mismatch");
        paired_continuation_seen = 1'b1;

        // Empty mask and parity violations are rejected before weight access.
        apply_reset();
        set_command(12'h601, 8'd0, 6'd1, 1'b1, 1'b1, 1'b0,
                    7'd4, 9'd2, 5'd1, 2'b00, 5'd0, 5'd0, 4'd1);
        repeat (2) @(posedge clk_core);
        if (!protocol_error || cmd_ready || weight_req_valid)
            $fatal(1, "PPDI empty mask escaped rejection");
        @(negedge clk_core);
        cmd_valid = 1'b0;
        clear_error = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        clear_error = 1'b0;
        set_command(12'h602, 8'd0, 6'd2, 1'b1, 1'b1, 1'b0,
                    7'd5, 9'd2, 5'd1, 2'b11, 5'd3, 5'd4, 4'd1);
        repeat (2) @(posedge clk_core);
        if (!protocol_error || cmd_ready || weight_req_valid)
            $fatal(1, "PPDI parity violation escaped rejection");
        @(negedge clk_core);
        cmd_valid = 1'b0;
        malformed_seen = 1'b1;

        // A previously latched child error is cleared by one clear pulse; a
        // same-epoch wrong-identity response itself requires flush/abort.
        apply_reset();
        set_command(12'h688, 8'd0, 6'd6, 1'b1, 1'b1, 1'b0,
                    7'd12, 9'd3, 5'd2, 2'b01, 5'd2, 5'd0, 4'd1);
        capture_weight_request(12'h688, 7'd12, 6'd4, replacement_epoch);
        @(negedge clk_core);
        weight_rsp_tag = 12'h689;
        weight_rsp_input_channel = 7'd12;
        weight_rsp_output_tile = 6'd4;
        weight_rsp_epoch = replacement_epoch;
        weight_rsp_valid = 1'b1;
        repeat (2) @(posedge clk_core);
        #1;
        if (weight_rsp_ready || !dut.engine_protocol_error || !protocol_error)
            $fatal(1, "PPDI child identity error was not isolated");
        @(negedge clk_core);
        weight_rsp_valid = 1'b0;
        clear_error = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        clear_error = 1'b0;
        #1;
        if (dut.engine_protocol_error || protocol_error)
            $fatal(1, "PPDI single-cycle clear did not clear child/parent");
        child_clear_seen = 1'b1;
        flush = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        flush = 1'b0;
        cmd_valid = 1'b0;

        // Fill every epoch with one canceled outstanding request. Hardware
        // must block instead of reusing an epoch that can still return.
        apply_reset();
        for (integer generation = 0; generation < EPOCH_COUNT;
             generation = generation + 1) begin
            set_command(GROUP_TAG_W'(32'd1824 + generation), 8'd0,
                        ISSUE_SEQ_W'(generation), 1'b1, 1'b1, 1'b0,
                        7'd33, 9'd2, 5'd1, 2'b01, 5'd2, 5'd0, 4'd1);
            capture_weight_request(GROUP_TAG_W'(32'd1824 + generation),
                                   7'd33, 6'd4,
                                   wrap_epochs[generation]);
            for (integer prior = 0; prior < generation;
                 prior = prior + 1) begin
                if (wrap_epochs[generation] == wrap_epochs[prior])
                    $fatal(1, "PPDI reused pending epoch generation=%0d",
                           generation);
            end
            @(negedge clk_core);
            flush = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            flush = 1'b0;
            cmd_valid = 1'b0;
            if ((generation < EPOCH_COUNT-1) &&
                dut.epoch_space_exhausted_q)
                $fatal(1, "PPDI epoch guard blocked before table full");
        end
        #1;
        if (!dut.epoch_space_exhausted_q || !protocol_error ||
            dut.stale_epoch_pending_q != {EPOCH_COUNT{1'b1}})
            $fatal(1, "PPDI epoch exhaustion did not fail closed");
        @(negedge clk_core);
        weight_rsp_epoch = wrap_epochs[0];
        weight_rsp_tag = 12'h720;
        weight_rsp_input_channel = 7'd33;
        weight_rsp_output_tile = 6'd4;
        weight_rsp_valid = 1'b1;
        @(posedge clk_core);
        if (!weight_rsp_ready)
            $fatal(1, "PPDI blocked epoch did not drain pending response");
        @(negedge clk_core);
        weight_rsp_valid = 1'b0;
        #1;
        if (dut.epoch_space_exhausted_q ||
            dut.stale_epoch_pending_q[wrap_epochs[0]] ||
            dut.epoch_q != wrap_epochs[0])
            $fatal(1, "PPDI epoch guard did not recover freed generation");
        clear_error = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        clear_error = 1'b0;
        if (protocol_error)
            $fatal(1, "PPDI epoch guard error did not clear after recovery");
        epoch_wrap_guard_seen = 1'b1;

        // Flush after only the even destination committed, then prove that an
        // old response is dropped by epoch before a clean replacement term.
        apply_reset();
        set_command(12'h711, 8'd0, 6'd15, 1'b1, 1'b1, 1'b0,
                    7'd42, 9'd7, 5'd4, 2'b11, 5'd8, 5'd9, 4'd3);
        capture_weight_request(12'h711, 7'd42, 6'd10, canceled_epoch);
        send_weight_response(12'h711, 7'd42, 6'd10, canceled_epoch, -6);
        accept_mask(2'b01);
        if (cmd_ready || term_done || acc_update_valid != 2'b10)
            $fatal(1, "PPDI partial command completed before flush");
        @(negedge clk_core);
        flush = 1'b1;
        #1;
        if (cmd_ready || weight_req_valid || weight_rsp_ready ||
            acc_update_valid != 2'b00 || term_done)
            $fatal(1, "PPDI partial flush was not combinational");
        @(posedge clk_core);
        @(negedge clk_core);
        flush = 1'b0;
        cmd_valid = 1'b0;
        partial_flush_seen = 1'b1;

        set_command(12'h711, 8'd9, 6'd16, 1'b1, 1'b1, 1'b1,
                    7'd42, 9'd2, 5'd5, 2'b10, 5'd0, 5'd11, 4'd3);
        capture_weight_request(12'h711, 7'd42, 6'd10, replacement_epoch);
        if (replacement_epoch == canceled_epoch)
            $fatal(1, "PPDI epoch did not advance across flush");
        fork
            begin
                send_weight_response(12'h711, 7'd42, 6'd10,
                                     replacement_epoch, -2);
            end
            begin
                accept_mask(2'b10);
            end
            begin
                wait_command_accept();
            end
        join
        if (command_count != 1 || acc_fire_count[0] != 1 ||
            acc_fire_count[1] != 1 || term_done_count != 1 ||
            count_stale_weight_responses != 0 || protocol_error)
            $fatal(1, "PPDI post-flush recovery mismatch");

        if (!split_pair_seen || !same_cycle_pair_seen || !only_even_seen ||
            !only_odd_seen || !partial_flush_seen || !malformed_seen ||
            !stale_seen || !stale_during_acc_stall_seen || !child_clear_seen ||
            !paired_continuation_seen || !odd_first_seen ||
            !epoch_wrap_guard_seen || !zero_single_pair_seen ||
            !zero_multi_parity_seen || !zero_head_last_seen ||
            !zero_first_stable_seen || !zero_flush_seen)
            $fatal(1, "PPDI required coverage marker missing");

        if (total_commands != 11 || total_weight_requests != 16 ||
            total_acc_fires[0] != 6 || total_acc_fires[1] != 6 ||
            total_term_done != 7 || total_stale_drops != 3 ||
            total_zero_gate_commands != 4 ||
            total_zero_gate_term_done != 2)
            $fatal(1,
                "PPDI lifetime accounting mismatch cmd=%0d weight=%0d acc=%0d/%0d done=%0d stale=%0d zero=%0d/%0d",
                total_commands, total_weight_requests, total_acc_fires[0],
                total_acc_fires[1], total_term_done, total_stale_drops,
                total_zero_gate_commands, total_zero_gate_term_done);

        $display("PASS PPDI DCTF32 BANK EXECUTOR cycles=%0d commands=%0d weight_req=%0d acc={%0d,%0d} done=%0d stale=%0d zero={%0d,%0d} coverage=split_pair,same_cycle,only_even,only_odd,partial_flush,malformed,stale_epoch,stale_acc_stall,child_clear,paired_continuation,odd_first,epoch_wrap_guard,zero_single_pair,zero_multi_parity,zero_head_last,zero_first_stable,zero_flush",
                 total_cycles, total_commands, total_weight_requests,
                 total_acc_fires[0], total_acc_fires[1], total_term_done,
                 total_stale_drops, total_zero_gate_commands,
                 total_zero_gate_term_done);
        $finish;
    end

    initial begin
        repeat (5000) @(posedge clk_core);
        $fatal(1, "PPDI DCTF32 executor TB timeout");
    end
endmodule

/* verilator lint_on UNUSEDSIGNAL */
/* verilator lint_on BLKSEQ */

`default_nettype wire
