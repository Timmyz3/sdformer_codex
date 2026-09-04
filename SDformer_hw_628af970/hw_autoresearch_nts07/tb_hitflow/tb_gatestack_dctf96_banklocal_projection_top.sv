`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off BLKSEQ */
/* verilator lint_off UNUSEDSIGNAL */

module tb_gatestack_dctf96_banklocal_projection_top #(
    parameter integer ADAPTER_CONTEXTS = 1,
    parameter bit PPDI_ENABLE = 1'b0,
    parameter bit IMPLICIT_BIAS_FINALIZE_ENABLE = 1'b0,
    parameter bit STATIONARY_BIAS_TEST =
        IMPLICIT_BIAS_FINALIZE_ENABLE
);
    localparam int Q = 2;
    localparam int TOKENS = 6;
    localparam int EVENT_WAYS = 4;
    localparam int OUT_TILE = 4;
    localparam int GATE_W = 9;
    localparam int WEIGHT_W = 8;
    localparam int PRODUCT_W = GATE_W + WEIGHT_W;
    localparam int ACC_W = 32;
    localparam int TAG_W = 12;
    localparam int CMD_SEQUENCE_W = 8;
    localparam int ISSUE_SEQ_W = 6;
    localparam int INPUT_CH_W = 6;
    localparam int LANE_ID_W = 3;
    localparam int TOKEN_ID_W = 3;
    localparam int OUTPUT_TILE_W = 5;
    localparam int LOGICAL_SUPERTILE_W = 4;
    localparam int HEAD_COUNT_W = 2;
    localparam int EPOCH_W = 3;
    localparam int COUNTER_W = 32;
    localparam int WAY_COUNT_W = 3;
    localparam logic [TAG_W-1:0] TEST_TAG = 12'h5a7;
    localparam logic [LOGICAL_SUPERTILE_W-1:0] TEST_SUPERTILE = 4'd2;

    logic clk_core;
    logic rst_core;
    logic flush;
    logic tile_start_valid;
    logic tile_start_ready;
    logic [TAG_W-1:0] tile_start_tag;
    logic [LOGICAL_SUPERTILE_W-1:0] tile_start_logical_supertile;
    logic [HEAD_COUNT_W-1:0] tile_start_head_count;
    logic head_start_valid;
    logic head_start_ready;
    logic [TAG_W-1:0] head_start_tag;
    logic [HEAD_COUNT_W-1:0] head_start_index;
    logic [INPUT_CH_W-1:0] head_start_input_channel_base;
    logic head_start_last;
    logic term_valid;
    logic term_ready;
    logic [GATE_W-1:0] term_gate_code;
    logic [LANE_ID_W-1:0] term_lane_id;
    logic [7:0] term_destination_count;
    logic [ISSUE_SEQ_W-1:0] term_issue_seq;
    logic term_head_last;
    logic event_valid;
    logic event_ready;
    logic [GATE_W-1:0] event_gate_code;
    logic [LANE_ID_W-1:0] event_lane_id;
    logic [EVENT_WAYS-1:0] event_token_valid;
    logic [(EVENT_WAYS*TOKEN_ID_W)-1:0] event_token_ids;
    logic [WAY_COUNT_W-1:0] event_count;
    logic [ISSUE_SEQ_W-1:0] event_issue_seq;
    logic event_term_first;
    logic event_term_last;
    logic event_head_last;
    logic source_done_valid;
    logic source_done_ready;
    logic [TAG_W-1:0] source_done_tag;
    logic source_done_error;
    logic head_done_valid;
    logic head_done_ready;
    logic [TAG_W-1:0] head_done_tag;
    logic [HEAD_COUNT_W-1:0] head_done_index;
    logic head_done_last;
    logic head_done_error;
    logic [2:0] weight_req_valid;
    logic [2:0] weight_req_ready;
    logic [(3*TAG_W)-1:0] weight_req_tags;
    logic [(3*INPUT_CH_W)-1:0] weight_req_input_channels;
    logic [(3*OUTPUT_TILE_W)-1:0] weight_req_output_tiles;
    logic [(3*EPOCH_W)-1:0] weight_req_epochs;
    logic [2:0] weight_rsp_valid;
    logic [2:0] weight_rsp_ready;
    logic [(3*TAG_W)-1:0] weight_rsp_tags;
    logic [(3*INPUT_CH_W)-1:0] weight_rsp_input_channels;
    logic [(3*OUTPUT_TILE_W)-1:0] weight_rsp_output_tiles;
    logic [(3*EPOCH_W)-1:0] weight_rsp_epochs;
    logic [(3*OUT_TILE*WEIGHT_W)-1:0] weight_rsp_weights;
    logic [2:0] bias_req_valid;
    logic [2:0] bias_req_ready;
    logic [(3*TAG_W)-1:0] bias_req_tags;
    logic [(3*OUTPUT_TILE_W)-1:0] bias_req_output_tiles;
    logic [(3*TOKEN_ID_W)-1:0] bias_req_token_ids;
    logic [(3*EPOCH_W)-1:0] bias_req_epochs;
    logic [2:0] bias_rsp_valid;
    logic [2:0] bias_rsp_ready;
    logic [(3*TAG_W)-1:0] bias_rsp_tags;
    logic [(3*OUTPUT_TILE_W)-1:0] bias_rsp_output_tiles;
    logic [(3*TOKEN_ID_W)-1:0] bias_rsp_token_ids;
    logic [(3*EPOCH_W)-1:0] bias_rsp_epochs;
    logic [(3*OUT_TILE*ACC_W)-1:0] bias_rsp_values;
    logic [5:0] final_valid;
    logic [5:0] final_ready;
    logic [(6*TOKEN_ID_W)-1:0] final_token_ids;
    logic [(3*TAG_W)-1:0] final_tags;
    logic [(6*OUT_TILE*ACC_W)-1:0] final_values;
    logic tile_done_valid;
    logic tile_done_ready;
    logic [TAG_W-1:0] tile_done_tag;
    logic tile_done_error;
    logic protocol_error;
    logic accumulator_overflow;
    logic [COUNTER_W-1:0] count_heads;
    logic [COUNTER_W-1:0] count_issued_terms;
    logic [(3*COUNTER_W)-1:0] count_completed_terms;
    logic [(3*COUNTER_W)-1:0] count_bias_commits;
    logic [(3*COUNTER_W)-1:0] count_stale_weight_responses;
    logic [(3*COUNTER_W)-1:0] count_stale_bias_responses;

    integer cycle_count;
    integer weight_request_count [0:2];
    integer weight_response_order [0:2];
    integer weight_response_sequence;
    integer bias_request_count [0:2];
    integer bias_response_order [0:2];
    integer bias_response_sequence;
    integer final_fire_count [0:5];
    integer final_stall_seen [0:5];
    integer observed [0:2][0:TOKENS-1][0:OUT_TILE-1];
    logic [TOKENS-1:0] observed_valid [0:2];
    logic final_sink_enable;
    logic wrong_current_seen;
    logic old_epoch_drop_seen;
    logic same_tag_recovery_seen;
    logic zero_term_head_seen;
    logic multi_destination_seen;
    logic [EPOCH_W-1:0] old_bias_epoch [0:2];
    logic [TOKEN_ID_W-1:0] old_bias_token [0:2];

    gatestack_dctf96_banklocal_projection_top #(
        .Q(Q),
        .TOKENS(TOKENS),
        .EVENT_WAYS(EVENT_WAYS),
        .OUT_TILE(OUT_TILE),
        .GATE_W(GATE_W),
        .WEIGHT_W(WEIGHT_W),
        .PRODUCT_W(PRODUCT_W),
        .ACC_W(ACC_W),
        .TAG_W(TAG_W),
        .CMD_SEQUENCE_W(CMD_SEQUENCE_W),
        .ISSUE_SEQ_W(ISSUE_SEQ_W),
        .INPUT_CH_W(INPUT_CH_W),
        .INPUT_CHANNELS(48),
        .LANE_ID_W(LANE_ID_W),
        .TOKEN_ID_W(TOKEN_ID_W),
        .OUTPUT_TILE_W(OUTPUT_TILE_W),
        .LOGICAL_SUPERTILE_W(LOGICAL_SUPERTILE_W),
        .HEAD_COUNT_W(HEAD_COUNT_W),
        .EPOCH_W(EPOCH_W),
        .COUNTER_W(COUNTER_W),
        .ADAPTER_CONTEXTS(ADAPTER_CONTEXTS),
        .PPDI_ENABLE(PPDI_ENABLE),
        .IMPLICIT_BIAS_FINALIZE_ENABLE(IMPLICIT_BIAS_FINALIZE_ENABLE),
        .WAY_COUNT_W(WAY_COUNT_W)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    function automatic integer weight_value(
        input integer bank,
        input integer lane
    );
        begin
            case (bank)
                0: weight_value = 1 + lane;
                1: weight_value = -2 + lane;
                default: weight_value = 3 + lane;
            endcase
        end
    endfunction

    function automatic integer bias_value(
        input integer bank,
        input integer token,
        input integer lane
    );
        begin
            bias_value = ((bank + 1) * 1000) + (token * 100) +
                         (lane * 10) + bank;
            if (STATIONARY_BIAS_TEST)
                bias_value = ((bank + 1) * 1000) + (lane * 10) + bank;
        end
    endfunction

    task automatic start_tile;
        begin
            @(negedge clk_core);
            tile_start_tag = TEST_TAG;
            tile_start_logical_supertile = TEST_SUPERTILE;
            tile_start_head_count = HEAD_COUNT_W'(2);
            tile_start_valid = 1'b1;
            do @(posedge clk_core); while (!tile_start_ready);
            @(negedge clk_core);
            tile_start_valid = 1'b0;
        end
    endtask

    task automatic start_head(
        input integer index_value,
        input integer base_value,
        input logic last_value
    );
        begin
            @(negedge clk_core);
            head_start_tag = TEST_TAG;
            head_start_index = HEAD_COUNT_W'(index_value);
            head_start_input_channel_base = INPUT_CH_W'(base_value);
            head_start_last = last_value;
            head_start_valid = 1'b1;
            do @(posedge clk_core); while (!head_start_ready);
            @(negedge clk_core);
            head_start_valid = 1'b0;
        end
    endtask

    task automatic drive_multidestination_term(
        input integer gate_value,
        input integer lane_value
    );
        begin
            @(negedge clk_core);
            term_gate_code = GATE_W'(gate_value);
            term_lane_id = LANE_ID_W'(lane_value);
            term_destination_count = 8'd3;
            term_issue_seq = ISSUE_SEQ_W'(0);
            term_head_last = 1'b1;
            term_valid = 1'b1;
            do @(posedge clk_core); while (!term_ready);
            multi_destination_seen = 1'b1;
            @(negedge clk_core);
            term_valid = 1'b0;
            event_gate_code = GATE_W'(gate_value);
            event_lane_id = LANE_ID_W'(lane_value);
            event_token_valid = 4'b0111;
            event_token_ids = '0;
            event_token_ids[0 +: TOKEN_ID_W] = TOKEN_ID_W'(0);
            event_token_ids[TOKEN_ID_W +: TOKEN_ID_W] = TOKEN_ID_W'(3);
            event_token_ids[(2*TOKEN_ID_W) +: TOKEN_ID_W] = TOKEN_ID_W'(5);
            event_count = WAY_COUNT_W'(3);
            event_issue_seq = ISSUE_SEQ_W'(0);
            event_term_first = 1'b1;
            event_term_last = 1'b1;
            event_head_last = 1'b1;
            while (!event_ready) @(negedge clk_core);
            source_done_tag = TEST_TAG;
            source_done_error = 1'b0;
            source_done_valid = 1'b1;
            event_valid = 1'b1;
            do @(posedge clk_core); while (!(event_ready &&
                                             source_done_ready));
            @(negedge clk_core);
            event_valid = 1'b0;
            source_done_valid = 1'b0;
        end
    endtask

    task automatic finish_source;
        begin
            @(negedge clk_core);
            source_done_tag = TEST_TAG;
            source_done_error = 1'b0;
            source_done_valid = 1'b1;
            do @(posedge clk_core); while (!source_done_ready);
            @(negedge clk_core);
            source_done_valid = 1'b0;
        end
    endtask

    task automatic wait_head_completion(
        input integer expected_index,
        input logic expected_last
    );
        begin
            while (!head_done_valid) @(posedge clk_core);
            if (head_done_tag !== TEST_TAG ||
                head_done_index !== HEAD_COUNT_W'(expected_index) ||
                head_done_last !== expected_last || head_done_error)
                $fatal(1, "head completion mismatch index=%0d", expected_index);
            do @(posedge clk_core); while (!head_done_ready);
            @(negedge clk_core);
        end
    endtask

    task automatic serve_weight_request(
        input integer bank,
        input integer request_stall,
        input integer response_delay,
        output logic [EPOCH_W-1:0] captured_epoch
    );
        integer lane;
        logic [INPUT_CH_W-1:0] captured_channel;
        logic [OUTPUT_TILE_W-1:0] captured_tile;
        begin
            weight_req_ready[bank] = 1'b0;
            while (!weight_req_valid[bank]) @(posedge clk_core);
            repeat (request_stall) @(posedge clk_core);
            @(negedge clk_core);
            weight_req_ready[bank] = 1'b1;
            do @(posedge clk_core); while (!weight_req_valid[bank]);
            if (weight_req_tags[(bank*TAG_W) +: TAG_W] !== TEST_TAG ||
                weight_req_input_channels[(bank*INPUT_CH_W) +: INPUT_CH_W] !==
                    INPUT_CH_W'(5) ||
                weight_req_output_tiles[(bank*OUTPUT_TILE_W) +:
                                        OUTPUT_TILE_W] !==
                    OUTPUT_TILE_W'((3 * TEST_SUPERTILE) + bank))
                $fatal(1, "bank%0d weight request identity mismatch", bank);
            captured_channel = weight_req_input_channels[
                (bank*INPUT_CH_W) +: INPUT_CH_W];
            captured_tile = weight_req_output_tiles[
                (bank*OUTPUT_TILE_W) +: OUTPUT_TILE_W];
            captured_epoch = weight_req_epochs[(bank*EPOCH_W) +: EPOCH_W];
            @(negedge clk_core);
            weight_req_ready[bank] = 1'b0;
            repeat (response_delay) @(posedge clk_core);
            @(negedge clk_core);
            weight_rsp_tags[(bank*TAG_W) +: TAG_W] = TEST_TAG;
            weight_rsp_input_channels[(bank*INPUT_CH_W) +: INPUT_CH_W] =
                captured_channel;
            weight_rsp_output_tiles[(bank*OUTPUT_TILE_W) +: OUTPUT_TILE_W] =
                captured_tile;
            weight_rsp_epochs[(bank*EPOCH_W) +: EPOCH_W] = captured_epoch;
            for (lane = 0; lane < OUT_TILE; lane = lane + 1)
                weight_rsp_weights[(bank*OUT_TILE*WEIGHT_W) +
                                   (lane*WEIGHT_W) +: WEIGHT_W] =
                    WEIGHT_W'(weight_value(bank, lane));
            weight_rsp_valid[bank] = 1'b1;
            do @(posedge clk_core); while (!weight_rsp_ready[bank]);
            @(negedge clk_core);
            weight_rsp_valid[bank] = 1'b0;
        end
    endtask

    task automatic run_nonzero_head(
        output logic [EPOCH_W-1:0] epoch0,
        output logic [EPOCH_W-1:0] epoch1,
        output logic [EPOCH_W-1:0] epoch2
    );
        begin
            start_head(0, 4, 1'b0);
            fork
                serve_weight_request(0, 1, 4, epoch0);
                serve_weight_request(1, 3, 1, epoch1);
                serve_weight_request(2, 5, 2, epoch2);
                drive_multidestination_term(3, 1);
            join
            wait_head_completion(0, 1'b0);
        end
    endtask

    task automatic run_zero_term_head;
        begin
            start_head(1, 12, 1'b1);
            finish_source();
            zero_term_head_seen = 1'b1;
            wait_head_completion(1, 1'b1);
        end
    endtask

    task automatic capture_bias_request(
        input integer bank,
        input integer expected_token,
        input integer request_stall,
        output logic [EPOCH_W-1:0] captured_epoch
    );
        begin
            bias_req_ready[bank] = 1'b0;
            while (!bias_req_valid[bank]) @(posedge clk_core);
            repeat (request_stall) @(posedge clk_core);
            @(negedge clk_core);
            bias_req_ready[bank] = 1'b1;
            do @(posedge clk_core); while (!bias_req_valid[bank]);
            if (bias_req_tags[(bank*TAG_W) +: TAG_W] !== TEST_TAG ||
                bias_req_output_tiles[(bank*OUTPUT_TILE_W) +:
                                      OUTPUT_TILE_W] !==
                    OUTPUT_TILE_W'((3 * TEST_SUPERTILE) + bank) ||
                bias_req_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W] !==
                    TOKEN_ID_W'(expected_token))
                $fatal(1, "bank%0d bias request mismatch token=%0d",
                       bank, expected_token);
            captured_epoch = bias_req_epochs[(bank*EPOCH_W) +: EPOCH_W];
            @(negedge clk_core);
            bias_req_ready[bank] = 1'b0;
        end
    endtask

    task automatic send_bias_response(
        input integer bank,
        input integer token,
        input logic [EPOCH_W-1:0] response_epoch,
        input integer response_delay
    );
        integer lane;
        begin
            repeat (response_delay) @(posedge clk_core);
            @(negedge clk_core);
            bias_rsp_tags[(bank*TAG_W) +: TAG_W] = TEST_TAG;
            bias_rsp_output_tiles[(bank*OUTPUT_TILE_W) +: OUTPUT_TILE_W] =
                OUTPUT_TILE_W'((3 * TEST_SUPERTILE) + bank);
            bias_rsp_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W] =
                TOKEN_ID_W'(token);
            bias_rsp_epochs[(bank*EPOCH_W) +: EPOCH_W] = response_epoch;
            for (lane = 0; lane < OUT_TILE; lane = lane + 1)
                bias_rsp_values[(bank*OUT_TILE*ACC_W) + (lane*ACC_W) +:
                                ACC_W] = ACC_W'(bias_value(bank, token, lane));
            bias_rsp_valid[bank] = 1'b1;
            do @(posedge clk_core); while (!bias_rsp_ready[bank]);
            @(negedge clk_core);
            bias_rsp_valid[bank] = 1'b0;
        end
    endtask

    task automatic inject_wrong_current_bias(
        input integer bank,
        input logic [EPOCH_W-1:0] response_epoch
    );
        integer commits_before;
        begin
            commits_before = count_bias_commits[(bank*COUNTER_W) +:
                                                COUNTER_W];
            @(negedge clk_core);
            bias_rsp_tags[(bank*TAG_W) +: TAG_W] = TEST_TAG;
            bias_rsp_output_tiles[(bank*OUTPUT_TILE_W) +: OUTPUT_TILE_W] =
                OUTPUT_TILE_W'((3 * TEST_SUPERTILE) + bank);
            bias_rsp_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W] =
                TOKEN_ID_W'(1);
            bias_rsp_epochs[(bank*EPOCH_W) +: EPOCH_W] = response_epoch;
            bias_rsp_values[(bank*OUT_TILE*ACC_W) +: (OUT_TILE*ACC_W)] = '1;
            bias_rsp_valid[bank] = 1'b1;
            @(posedge clk_core);
            #1;
            if (!bias_rsp_ready[bank] || !protocol_error ||
                dut.bias_commit_fire[bank] ||
                !dut.bias_outstanding_q[bank])
                $fatal(1,
                       "wrong-current drop bank%0d ready=%b protocol=%b commit=%b outstanding=%b",
                       bank, bias_rsp_ready[bank], protocol_error,
                       dut.bias_commit_fire[bank],
                       dut.bias_outstanding_q[bank]);
            @(negedge clk_core);
            bias_rsp_valid[bank] = 1'b0;
            if (count_bias_commits[(bank*COUNTER_W) +: COUNTER_W] !=
                COUNTER_W'(commits_before))
                $fatal(1, "wrong-current bias committed on bank%0d", bank);
            wrong_current_seen = 1'b1;
        end
    endtask

    task automatic serve_recovery_bias_bank(
        input integer bank
    );
        integer token;
        logic [EPOCH_W-1:0] epoch_value;
        begin
            for (token = 0;
                 token < (IMPLICIT_BIAS_FINALIZE_ENABLE ? 1 : TOKENS);
                 token = token + 1) begin
                capture_bias_request(bank, token, (bank + token) % 3,
                                     epoch_value);
                send_bias_response(bank, token, epoch_value,
                                   1 + ((bank * 2 + token) % 4));
            end
        end
    endtask

    always_comb begin
        head_done_ready = 1'b1;
        tile_done_ready = 1'b1;
        final_ready = '0;
        for (int port = 0; port < 6; port = port + 1) begin
            if (final_sink_enable && (final_stall_seen[port] != 0)) begin
                case (port)
                    0: final_ready[port] = (cycle_count % 3) != 0;
                    1: final_ready[port] = (cycle_count % 4) != 1;
                    2: final_ready[port] = (cycle_count % 5) != 2;
                    3: final_ready[port] = (cycle_count % 6) != 3;
                    4: final_ready[port] = (cycle_count % 7) != 4;
                    default: final_ready[port] = (cycle_count % 8) != 5;
                endcase
            end
        end
    end

    always @(posedge clk_core) begin : p_scoreboard
        integer bank;
        integer port;
        integer token;
        integer lane;
        integer signed actual_value;
        integer signed expected_value;
        if (rst_core) begin
            cycle_count = 0;
        end else begin
            cycle_count = cycle_count + 1;
            for (bank = 0; bank < 3; bank = bank + 1) begin
                if (weight_req_valid[bank] && weight_req_ready[bank])
                    weight_request_count[bank] =
                        weight_request_count[bank] + 1;
                if (weight_rsp_valid[bank] && weight_rsp_ready[bank]) begin
                    weight_response_sequence = weight_response_sequence + 1;
                    weight_response_order[bank] = weight_response_sequence;
                end
                if (bias_req_valid[bank] && bias_req_ready[bank])
                    bias_request_count[bank] = bias_request_count[bank] + 1;
                if (bias_rsp_valid[bank] && bias_rsp_ready[bank] &&
                    (bias_rsp_epochs[(bank*EPOCH_W) +: EPOCH_W] ==
                     dut.bias_epoch_q)) begin
                    bias_response_sequence = bias_response_sequence + 1;
                    bias_response_order[bank] = bias_response_sequence;
                end
            end
            for (port = 0; port < 6; port = port + 1) begin
                if (final_valid[port] && !final_ready[port])
                    final_stall_seen[port] = 1;
                if (final_valid[port] && final_ready[port]) begin
                    bank = port / 2;
                    token = 32'(final_token_ids[
                        (port*TOKEN_ID_W) +: TOKEN_ID_W]);
                    if (token >= TOKENS || ((token & 1) != (port & 1)))
                        $fatal(1, "final port%0d token mismatch %0d", port,
                               token);
                    if (final_tags[(bank*TAG_W) +: TAG_W] !== TEST_TAG)
                        $fatal(1, "final bank%0d tag mismatch", bank);
                    if (observed_valid[bank][token])
                        $fatal(1, "duplicate final bank%0d token%0d", bank,
                               token);
                    observed_valid[bank][token] = 1'b1;
                    final_fire_count[port] = final_fire_count[port] + 1;
                    for (lane = 0; lane < OUT_TILE; lane = lane + 1) begin
                        actual_value = $signed(final_values[
                            (port*OUT_TILE*ACC_W) + (lane*ACC_W) +: ACC_W]);
                        expected_value = bias_value(bank, token, lane);
                        if ((token == 0) || (token == 3) || (token == 5))
                            expected_value = expected_value +
                                             (3 * weight_value(bank, lane));
                        observed[bank][token][lane] = actual_value;
                        if (actual_value != expected_value)
                            $fatal(1,
                                "final mismatch bank=%0d token=%0d lane=%0d got=%0d expected=%0d",
                                bank, token, lane, actual_value,
                                expected_value);
                    end
                end
            end
        end
    end

    initial begin : p_test
        integer bank;
        integer port;
        logic [EPOCH_W-1:0] aborted_weight_epoch [0:2];
        logic [EPOCH_W-1:0] recovery_weight_epoch [0:2];
        logic [EPOCH_W-1:0] first_bias_epoch [0:2];
        clk_core = 1'b0;
        rst_core = 1'b1;
        flush = 1'b0;
        tile_start_valid = 1'b0;
        tile_start_tag = TEST_TAG;
        tile_start_logical_supertile = TEST_SUPERTILE;
        tile_start_head_count = HEAD_COUNT_W'(2);
        head_start_valid = 1'b0;
        head_start_tag = TEST_TAG;
        head_start_index = '0;
        head_start_input_channel_base = '0;
        head_start_last = 1'b0;
        term_valid = 1'b0;
        term_gate_code = '0;
        term_lane_id = '0;
        term_destination_count = '0;
        term_issue_seq = '0;
        term_head_last = 1'b0;
        event_valid = 1'b0;
        event_gate_code = '0;
        event_lane_id = '0;
        event_token_valid = '0;
        event_token_ids = '0;
        event_count = '0;
        event_issue_seq = '0;
        event_term_first = 1'b0;
        event_term_last = 1'b0;
        event_head_last = 1'b0;
        source_done_valid = 1'b0;
        source_done_tag = TEST_TAG;
        source_done_error = 1'b0;
        weight_req_ready = '0;
        weight_rsp_valid = '0;
        weight_rsp_tags = '0;
        weight_rsp_input_channels = '0;
        weight_rsp_output_tiles = '0;
        weight_rsp_epochs = '0;
        weight_rsp_weights = '0;
        bias_req_ready = '0;
        bias_rsp_valid = '0;
        bias_rsp_tags = '0;
        bias_rsp_output_tiles = '0;
        bias_rsp_token_ids = '0;
        bias_rsp_epochs = '0;
        bias_rsp_values = '0;
        final_sink_enable = 1'b0;
        wrong_current_seen = 1'b0;
        old_epoch_drop_seen = 1'b0;
        same_tag_recovery_seen = 1'b0;
        zero_term_head_seen = 1'b0;
        multi_destination_seen = 1'b0;
        weight_response_sequence = 0;
        bias_response_sequence = 0;
        for (bank = 0; bank < 3; bank = bank + 1) begin
            weight_request_count[bank] = 0;
            weight_response_order[bank] = 0;
            bias_request_count[bank] = 0;
            bias_response_order[bank] = 0;
            observed_valid[bank] = '0;
            old_bias_epoch[bank] = '0;
            old_bias_token[bank] = '0;
            for (integer token = 0; token < TOKENS; token = token + 1)
                for (integer lane = 0; lane < OUT_TILE; lane = lane + 1)
                    observed[bank][token][lane] = 0;
        end
        for (port = 0; port < 6; port = port + 1) begin
            final_fire_count[port] = 0;
            final_stall_seen[port] = 0;
        end

        repeat (6) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        start_tile();
        run_nonzero_head(aborted_weight_epoch[0], aborted_weight_epoch[1],
                         aborted_weight_epoch[2]);
        run_zero_term_head();

        fork
            capture_bias_request(0, 0, 1, first_bias_epoch[0]);
            capture_bias_request(1, 0, 3, first_bias_epoch[1]);
            capture_bias_request(2, 0, 5, first_bias_epoch[2]);
        join
        if (IMPLICIT_BIAS_FINALIZE_ENABLE) begin
            fork
                send_bias_response(0, 0, first_bias_epoch[0], 1);
                begin
                    inject_wrong_current_bias(1, first_bias_epoch[1]);
                    send_bias_response(1, 0, first_bias_epoch[1], 1);
                end
                send_bias_response(2, 0, first_bias_epoch[2], 4);
            join
            for (bank = 0; bank < 3; bank = bank + 1) begin
                old_bias_epoch[bank] = first_bias_epoch[bank];
                old_bias_token[bank] = '0;
            end
        end else begin
            fork
                begin
                    send_bias_response(0, 0, first_bias_epoch[0], 1);
                    capture_bias_request(0, 1, 0, old_bias_epoch[0]);
                    old_bias_token[0] = TOKEN_ID_W'(1);
                end
                begin
                    inject_wrong_current_bias(1, first_bias_epoch[1]);
                    send_bias_response(1, 0, first_bias_epoch[1], 1);
                    capture_bias_request(1, 1, 0, old_bias_epoch[1]);
                    old_bias_token[1] = TOKEN_ID_W'(1);
                end
                begin
                    send_bias_response(2, 0, first_bias_epoch[2], 4);
                    capture_bias_request(2, 1, 0, old_bias_epoch[2]);
                    old_bias_token[2] = TOKEN_ID_W'(1);
                end
            join
        end

        @(negedge clk_core);
        flush = 1'b1;
        weight_rsp_valid = 3'b111;
        for (bank = 0; bank < 3; bank = bank + 1)
            weight_rsp_epochs[(bank*EPOCH_W) +: EPOCH_W] =
                aborted_weight_epoch[bank];
        repeat (3) begin
            @(posedge clk_core);
            if (weight_rsp_ready != '0)
                $fatal(1, "long flush exposed weight response ready");
        end
        @(negedge clk_core);
        weight_rsp_valid = '0;
        flush = 1'b0;
        if (protocol_error)
            $fatal(1, "flush did not clear protocol error");
        if (dut.bias_epoch_q != (first_bias_epoch[0] + 1'b1))
            $fatal(1, "long flush advanced bias epoch more than once");

        fork
            send_bias_response(0, 32'(old_bias_token[0]),
                               old_bias_epoch[0], 3);
            send_bias_response(1, 32'(old_bias_token[1]),
                               old_bias_epoch[1], 1);
            send_bias_response(2, 32'(old_bias_token[2]),
                               old_bias_epoch[2], 5);
        join
        repeat (2) @(posedge clk_core);
        if (count_stale_bias_responses[0 +: COUNTER_W] != 1 ||
            count_stale_bias_responses[COUNTER_W +: COUNTER_W] != 1 ||
            count_stale_bias_responses[(2*COUNTER_W) +: COUNTER_W] != 1)
            $fatal(1, "old bias epoch responses were not dropped per bank");
        old_epoch_drop_seen = 1'b1;

        start_tile();
        same_tag_recovery_seen = 1'b1;
        run_nonzero_head(recovery_weight_epoch[0], recovery_weight_epoch[1],
                         recovery_weight_epoch[2]);
        for (bank = 0; bank < 3; bank = bank + 1)
            if (recovery_weight_epoch[bank] !=
                (aborted_weight_epoch[bank] + 1'b1))
                $fatal(1, "bank%0d long-flush weight epoch mismatch", bank);
        run_zero_term_head();
        final_sink_enable = 1'b1;
        fork
            serve_recovery_bias_bank(0);
            serve_recovery_bias_bank(1);
            serve_recovery_bias_bank(2);
        join

        while (!tile_done_valid) @(posedge clk_core);
        if (tile_done_tag !== TEST_TAG || tile_done_error || protocol_error ||
            accumulator_overflow)
            $fatal(1, "recovery tile completion status mismatch");
        do @(posedge clk_core); while (!tile_done_ready);
        @(negedge clk_core);

        for (bank = 0; bank < 3; bank = bank + 1) begin
            if (observed_valid[bank] != {TOKENS{1'b1}})
                $fatal(1, "bank%0d missing final tokens mask=%b", bank,
                       observed_valid[bank]);
            if (weight_request_count[bank] != 2)
                $fatal(1, "bank%0d weight request count=%0d", bank,
                       weight_request_count[bank]);
            if (bias_request_count[bank] <
                (IMPLICIT_BIAS_FINALIZE_ENABLE ? 2 : (TOKENS + 1)))
                $fatal(1, "bank%0d bias request coverage count=%0d", bank,
                       bias_request_count[bank]);
            if (count_completed_terms[(bank*COUNTER_W) +: COUNTER_W] != 2)
                $fatal(1, "bank%0d completed term count mismatch", bank);
        end
        for (port = 0; port < 6; port = port + 1) begin
            if ((final_stall_seen[port] == 0) ||
                (final_fire_count[port] != 3))
                $fatal(1, "final port%0d backpressure/fire mismatch stall=%0d fire=%0d",
                       port, final_stall_seen[port], final_fire_count[port]);
        end
        if (!wrong_current_seen || !old_epoch_drop_seen ||
            !same_tag_recovery_seen || !zero_term_head_seen ||
            !multi_destination_seen || count_heads != 4 ||
            count_issued_terms != 2)
            $fatal(1, "required coverage/status mismatch heads=%0d terms=%0d",
                   count_heads, count_issued_terms);
        if ((weight_response_order[0] == weight_response_order[1]) ||
            (weight_response_order[1] == weight_response_order[2]) ||
            (weight_response_order[0] == weight_response_order[2]) ||
            (bias_response_order[0] == bias_response_order[1]) ||
            (bias_response_order[1] == bias_response_order[2]) ||
            (bias_response_order[0] == bias_response_order[2]))
            $fatal(1, "bank response staggering was not observed");

        $display("PASS DCTF96 BANKLOCAL PROJECTION cycles=%0d heads=%0d terms=%0d finals=18 coverage=two_head,multidestination,zero_head,three_bank_skew,six_final_backpressure,wrong_current,mid_bias_flush,old_epoch_drop,same_tag_recovery",
                 cycle_count, count_heads, count_issued_terms);
        $finish;
    end

    initial begin
        repeat (20000) @(posedge clk_core);
        $fatal(1, "DCTF96 bank-local projection TB timeout");
    end
endmodule

`default_nettype wire
