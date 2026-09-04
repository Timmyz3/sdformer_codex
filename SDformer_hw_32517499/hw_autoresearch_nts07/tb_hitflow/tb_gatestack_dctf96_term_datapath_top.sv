`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off BLKSEQ */
/* verilator lint_off UNUSEDSIGNAL */

module tb_gatestack_dctf96_term_datapath_top #(
    parameter integer ADAPTER_CONTEXTS = 1,
    parameter bit PPDI_ENABLE = 1'b0,
    parameter bit EXTENDED_2C_AUDIT = (ADAPTER_CONTEXTS == 2)
);
    localparam int Q = 2;
    localparam int TOKENS = 32;
    localparam int EVENT_WAYS = 4;
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
    localparam int OUTPUT_TILE_W = 5;
    localparam int LOGICAL_SUPERTILE_W = 4;
    localparam int EPOCH_W = 3;
    localparam int COUNTER_W = 32;
    localparam int WAY_COUNT_W = 3;
    localparam int OCC_W = $clog2(Q + 1);

    logic clk_core;
    logic rst_core;
    logic flush;
    logic clear_error;
    logic term_valid;
    logic term_ready;
    logic [GROUP_TAG_W-1:0] term_tag;
    logic [GATE_W-1:0] term_gate_code;
    logic [LANE_ID_W-1:0] term_lane_id;
    logic [7:0] term_destination_count;
    logic [ISSUE_SEQ_W-1:0] term_issue_seq;
    logic term_head_last;
    logic [LOGICAL_SUPERTILE_W-1:0] logical_supertile;
    logic [INPUT_CH_W-1:0] head_input_channel_base;
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
    logic [2:0] weight_req_valid;
    logic [2:0] weight_req_ready;
    logic [(3*GROUP_TAG_W)-1:0] weight_req_tags;
    logic [(3*INPUT_CH_W)-1:0] weight_req_input_channels;
    logic [(3*OUTPUT_TILE_W)-1:0] weight_req_output_tiles;
    logic [(3*EPOCH_W)-1:0] weight_req_epochs;
    logic [2:0] weight_rsp_valid;
    logic [2:0] weight_rsp_ready;
    logic [(3*GROUP_TAG_W)-1:0] weight_rsp_tags;
    logic [(3*INPUT_CH_W)-1:0] weight_rsp_input_channels;
    logic [(3*OUTPUT_TILE_W)-1:0] weight_rsp_output_tiles;
    logic [(3*EPOCH_W)-1:0] weight_rsp_epochs;
    logic [(3*OUT_TILE*WEIGHT_W)-1:0] weight_rsp_weights;
    logic [5:0] acc_update_valid;
    logic [5:0] acc_update_ready;
    logic [(6*TOKEN_ID_W)-1:0] acc_update_token_ids;
    logic [(3*GROUP_TAG_W)-1:0] acc_update_tags;
    logic [(3*OUT_TILE*PRODUCT_W)-1:0] acc_update_values;
    logic [2:0] bank_term_done;
    logic [(3*GROUP_TAG_W)-1:0] bank_term_done_group_tags;
    logic [(3*ISSUE_SEQ_W)-1:0] bank_term_done_issue_seqs;
    logic [2:0] bank_term_done_head_last;
    logic head_compute_done;
    logic [GROUP_TAG_W-1:0] head_compute_done_group_tag;
    logic [ISSUE_SEQ_W-1:0] head_compute_done_issue_seq;
    logic dispatch_retire_valid;
    logic [GROUP_TAG_W-1:0] dispatch_retire_group_tag;
    logic [CMD_SEQUENCE_W-1:0] dispatch_retire_sequence;
    logic [ISSUE_SEQ_W-1:0] dispatch_retire_issue_seq;
    logic dispatch_retire_term_first;
    logic dispatch_retire_term_last;
    logic dispatch_retire_head_last;
    logic [OCC_W-1:0] fabric_occupancy;
    logic [COUNTER_W-1:0] fabric_count_accepted;
    logic [(3*COUNTER_W)-1:0] fabric_count_bank_consumed;
    logic [COUNTER_W-1:0] fabric_count_retired;
    logic [COUNTER_W-1:0] fabric_count_input_stall;
    logic [(3*COUNTER_W)-1:0] fabric_count_bank_stall;
    logic [COUNTER_W-1:0] fabric_max_occupancy;
    logic [COUNTER_W-1:0] fabric_count_skew_cycles;
    logic [COUNTER_W-1:0] issued_terms;
    logic [(3*COUNTER_W)-1:0] completed_terms;
    logic [(3*COUNTER_W)-1:0] count_stale_weight_responses;
    logic datapath_idle;
    logic protocol_error;

    integer cycle_count;
    integer weight_request_count [0:2];
    integer update_count [0:2];
    integer parity_update_count [0:5];
    integer token_update_count [0:2][0:TOKENS-1];
    integer term_done_count [0:2];
    integer acc_stall_seen [0:5];
    integer weight_stall_seen [0:2];
    integer head_done_count;
    integer head_done_cycle;
    integer head_retire_cycle;
    integer first_nonlast_retire_cycle;
    integer first_term_done_cycle;
    logic [2:0] head_bank_done_seen;
    logic [2:0] head_done_this_cycle;
    logic fabric_full_seen;
    logic flush_mask_seen;
    logic aba_drop_seen;
    logic cross_term_overlap_seen;
    logic idle_collect_seen;
    logic idle_emit_seen;
    logic idle_fabric_seen;
    logic idle_executor_seen;
    logic [EPOCH_W-1:0] canceled_epoch [0:2];
    logic [EPOCH_W-1:0] replacement_epoch [0:2];
    logic adapter_collect_active;
    logic adapter_emit_active;
    logic [1:0] adapter_context_valid;
    logic [2:0] executor_active_debug;
    integer mixed_fabric_accepted_before;
    integer mixed_issued_before;
    integer mixed_weight_before [0:2];
    integer mixed_update_before [0:2];
    integer mixed_done_before [0:2];

    gatestack_dctf96_term_datapath_top #(
        .Q(Q),
        .TOKENS(TOKENS),
        .EVENT_WAYS(EVENT_WAYS),
        .OUT_TILE(OUT_TILE),
        .GATE_W(GATE_W),
        .WEIGHT_W(WEIGHT_W),
        .PRODUCT_W(PRODUCT_W),
        .GROUP_TAG_W(GROUP_TAG_W),
        .CMD_SEQUENCE_W(CMD_SEQUENCE_W),
        .ISSUE_SEQ_W(ISSUE_SEQ_W),
        .INPUT_CH_W(INPUT_CH_W),
        .INPUT_CHANNELS(96),
        .LANE_ID_W(LANE_ID_W),
        .TOKEN_ID_W(TOKEN_ID_W),
        .OUTPUT_TILE_W(OUTPUT_TILE_W),
        .LOGICAL_SUPERTILE_W(LOGICAL_SUPERTILE_W),
        .EPOCH_W(EPOCH_W),
        .COUNTER_W(COUNTER_W),
        .ADAPTER_CONTEXTS(ADAPTER_CONTEXTS),
        .PPDI_ENABLE(PPDI_ENABLE),
        .WAY_COUNT_W(WAY_COUNT_W)
    ) dut (.*);

    generate
        if (PPDI_ENABLE) begin : g_tb_ppdi_adapter_2c
            assign adapter_collect_active =
                dut.g_ppdi_adapter_2c.u_ppdi_term_event_adapter_2c.
                    fill_active_q;
            assign adapter_emit_active = dut.adapter_cmd_valid;
            assign adapter_context_valid =
                dut.g_ppdi_adapter_2c.u_ppdi_term_event_adapter_2c.
                    context_valid_q;
        end else if (ADAPTER_CONTEXTS == 1) begin : g_tb_adapter_1c
            assign adapter_collect_active =
                dut.g_adapter_1c.u_term_event_adapter.state_q == 2'd1;
            assign adapter_emit_active =
                dut.g_adapter_1c.u_term_event_adapter.state_q == 2'd3;
            assign adapter_context_valid = 2'b00;
        end else begin : g_tb_adapter_2c
            assign adapter_collect_active =
                dut.g_adapter_2c.u_term_event_adapter_2c.fill_active_q;
            assign adapter_emit_active = dut.adapter_cmd_valid;
            assign adapter_context_valid =
                dut.g_adapter_2c.u_term_event_adapter_2c.context_valid_q;
        end
    endgenerate

    generate
        if (PPDI_ENABLE) begin : g_tb_ppdi_executor_debug
            for (genvar bank = 0; bank < 3; bank = bank + 1) begin : g_bank
                assign executor_active_debug[bank] =
                    dut.g_executor[bank].g_ppdi_executor.u_ppdi_executor.
                        term_active_q;
            end
        end else begin : g_tb_scalar_executor_debug
            for (genvar bank = 0; bank < 3; bank = bank + 1) begin : g_bank
                assign executor_active_debug[bank] =
                    dut.g_executor[bank].g_scalar_executor.u_executor.
                        term_active_q;
            end
        end
    endgenerate

    always #5 clk_core = ~clk_core;

    task automatic drive_term(
        input logic [GROUP_TAG_W-1:0] tag_value,
        input logic [GATE_W-1:0] gate_value,
        input logic [LANE_ID_W-1:0] lane_value,
        input logic [ISSUE_SEQ_W-1:0] issue_value,
        input logic head_last_value,
        input logic [LOGICAL_SUPERTILE_W-1:0] supertile_value,
        input logic [INPUT_CH_W-1:0] base_value,
        input integer destination_count_value,
        input logic [TOKEN_ID_W-1:0] token0,
        input logic [TOKEN_ID_W-1:0] token1,
        input logic [TOKEN_ID_W-1:0] token2
    );
        begin
            @(negedge clk_core);
            term_tag = tag_value;
            term_gate_code = gate_value;
            term_lane_id = lane_value;
            term_destination_count = 8'(destination_count_value);
            term_issue_seq = issue_value;
            term_head_last = head_last_value;
            logical_supertile = supertile_value;
            head_input_channel_base = base_value;
            term_valid = 1'b1;
            do @(posedge clk_core); while (!term_ready);
            @(negedge clk_core);
            term_valid = 1'b0;

            event_gate_code = gate_value;
            event_lane_id = lane_value;
            event_token_valid = '0;
            event_token_ids = '0;
            event_token_valid[0] = 1'b1;
            event_token_ids[0 +: TOKEN_ID_W] = token0;
            if (destination_count_value > 1) begin
                event_token_valid[1] = 1'b1;
                event_token_ids[TOKEN_ID_W +: TOKEN_ID_W] = token1;
            end
            if (destination_count_value > 2) begin
                event_token_valid[2] = 1'b1;
                event_token_ids[(2*TOKEN_ID_W) +: TOKEN_ID_W] = token2;
            end
            event_count = WAY_COUNT_W'(destination_count_value);
            event_issue_seq = issue_value;
            event_term_first = 1'b1;
            event_term_last = 1'b1;
            event_head_last = head_last_value;
            event_valid = 1'b1;
            do @(posedge clk_core); while (!event_ready);
            @(negedge clk_core);
            event_valid = 1'b0;
        end
    endtask

    task automatic capture_weight_request(
        input integer bank,
        input integer stall_cycles,
        input logic [GROUP_TAG_W-1:0] expected_tag,
        input logic [INPUT_CH_W-1:0] expected_channel,
        input logic [OUTPUT_TILE_W-1:0] expected_tile,
        output logic [EPOCH_W-1:0] captured_epoch
    );
        begin
            weight_req_ready[bank] = 1'b0;
            while (!weight_req_valid[bank]) @(posedge clk_core);
            repeat (stall_cycles) @(posedge clk_core);
            @(negedge clk_core);
            weight_req_ready[bank] = 1'b1;
            do @(posedge clk_core); while (!weight_req_valid[bank]);
            if (weight_req_tags[(bank*GROUP_TAG_W) +: GROUP_TAG_W] !==
                    expected_tag ||
                weight_req_input_channels[(bank*INPUT_CH_W) +: INPUT_CH_W] !==
                    expected_channel ||
                weight_req_output_tiles[(bank*OUTPUT_TILE_W) +:
                                        OUTPUT_TILE_W] !== expected_tile)
                $fatal(1, "bank%0d weight request mismatch", bank);
            captured_epoch = weight_req_epochs[(bank*EPOCH_W) +: EPOCH_W];
            @(negedge clk_core);
            weight_req_ready[bank] = 1'b0;
        end
    endtask

    task automatic send_weight_response(
        input integer bank,
        input integer delay_cycles,
        input logic [GROUP_TAG_W-1:0] response_tag,
        input logic [INPUT_CH_W-1:0] response_channel,
        input logic [OUTPUT_TILE_W-1:0] response_tile,
        input logic [EPOCH_W-1:0] response_epoch,
        input integer weight_base
    );
        integer lane;
        begin
            repeat (delay_cycles) @(posedge clk_core);
            @(negedge clk_core);
            weight_rsp_tags[(bank*GROUP_TAG_W) +: GROUP_TAG_W] = response_tag;
            weight_rsp_input_channels[(bank*INPUT_CH_W) +: INPUT_CH_W] =
                response_channel;
            weight_rsp_output_tiles[(bank*OUTPUT_TILE_W) +: OUTPUT_TILE_W] =
                response_tile;
            weight_rsp_epochs[(bank*EPOCH_W) +: EPOCH_W] = response_epoch;
            for (lane = 0; lane < OUT_TILE; lane = lane + 1)
                weight_rsp_weights[
                    (bank*OUT_TILE*WEIGHT_W) + (lane*WEIGHT_W) +:
                    WEIGHT_W] = WEIGHT_W'(weight_base + lane);
            weight_rsp_valid[bank] = 1'b1;
            do @(posedge clk_core); while (!weight_rsp_ready[bank]);
            @(negedge clk_core);
            weight_rsp_valid[bank] = 1'b0;
        end
    endtask

    task automatic serve_weight_request(
        input integer bank,
        input integer request_stall_cycles,
        input integer response_delay_cycles,
        input logic [GROUP_TAG_W-1:0] expected_tag,
        input logic [INPUT_CH_W-1:0] expected_channel,
        input logic [OUTPUT_TILE_W-1:0] expected_tile,
        input integer weight_base
    );
        logic [EPOCH_W-1:0] epoch_value;
        begin
            capture_weight_request(bank, request_stall_cycles, expected_tag,
                                   expected_channel, expected_tile,
                                   epoch_value);
            send_weight_response(bank, response_delay_cycles, expected_tag,
                                 expected_channel, expected_tile, epoch_value,
                                 weight_base);
        end
    endtask

    task automatic sink_acc_channel(
        input integer channel,
        input integer expected_updates
    );
        integer update;
        integer stall_cycles;
        begin
            acc_update_ready[channel] = 1'b0;
            for (update = 0; update < expected_updates;
                 update = update + 1) begin
                while (!acc_update_valid[channel]) @(posedge clk_core);
                stall_cycles = 1 + ((channel + update) % 3);
                repeat (stall_cycles) @(posedge clk_core);
                @(negedge clk_core);
                acc_update_ready[channel] = 1'b1;
                do @(posedge clk_core); while (!acc_update_valid[channel]);
                @(negedge clk_core);
                acc_update_ready[channel] = 1'b0;
            end
        end
    endtask

    always @(posedge clk_core) begin : p_scoreboard
        integer bank;
        integer channel;
        integer token;
        integer lane;
        integer expected_gate;
        integer expected_weight_base;
        integer signed observed_product;
        integer signed expected_product;
        logic [2:0] current_head_done;
        if (rst_core) begin
            cycle_count = 0;
        end else begin
            if (datapath_idle !== (dut.adapter_idle &&
                                  !dut.illegal_drop_active_q &&
                                  (fabric_occupancy == '0) &&
                                  (dut.track_count_q == '0)))
                $fatal(1, "datapath_idle equation mismatch");
            if (adapter_collect_active) begin
                idle_collect_seen = 1'b1;
                if (datapath_idle)
                    $fatal(1, "datapath_idle asserted during collection");
            end
            if (adapter_emit_active) begin
                idle_emit_seen = 1'b1;
                if (datapath_idle)
                    $fatal(1, "datapath_idle asserted during emit");
            end
            if (fabric_occupancy != '0) begin
                idle_fabric_seen = 1'b1;
                if (datapath_idle)
                    $fatal(1, "datapath_idle asserted with fabric occupancy");
            end
            if (|executor_active_debug) begin
                idle_executor_seen = 1'b1;
                if (datapath_idle)
                    $fatal(1, "datapath_idle asserted with executor activity");
            end
            cycle_count = cycle_count + 1;
            current_head_done = bank_term_done & bank_term_done_head_last;
            if (fabric_occupancy == OCC_W'(Q))
                fabric_full_seen = 1'b1;
            if (term_valid && term_ready && (term_issue_seq == 6'd11) &&
                ((fabric_occupancy != '0) ||
                 (term_done_count[0] == 0) ||
                 (term_done_count[1] == 0) ||
                 (term_done_count[2] == 0)))
                cross_term_overlap_seen = 1'b1;
            for (bank = 0; bank < 3; bank = bank + 1) begin
                if (weight_req_valid[bank] && weight_req_ready[bank])
                    weight_request_count[bank] =
                        weight_request_count[bank] + 1;
                if (weight_req_valid[bank] && !weight_req_ready[bank])
                    weight_stall_seen[bank] = 1;
                if (bank_term_done[bank]) begin
                    if (bank_term_done_group_tags[
                            (bank*GROUP_TAG_W) +: GROUP_TAG_W] == 12'h155) begin
                        if (bank_term_done_issue_seqs[
                                (bank*ISSUE_SEQ_W) +: ISSUE_SEQ_W] != 6'd7 ||
                            bank_term_done_head_last[bank])
                            $fatal(1,
                                "mixed legal term completion metadata mismatch bank=%0d",
                                bank);
                    end
                    term_done_count[bank] = term_done_count[bank] + 1;
                    if (first_term_done_cycle < 0)
                        first_term_done_cycle = cycle_count;
                end
            end
            head_done_this_cycle = current_head_done;
            if (current_head_done != '0)
                head_bank_done_seen = head_bank_done_seen | current_head_done;

            for (channel = 0; channel < 6; channel = channel + 1) begin
                if (acc_update_valid[channel] && !acc_update_ready[channel])
                    acc_stall_seen[channel] = 1;
                if (acc_update_valid[channel] && acc_update_ready[channel]) begin
                    bank = channel / 2;
                    token = 32'(acc_update_token_ids[
                        (channel*TOKEN_ID_W) +: TOKEN_ID_W]);
                    if ((token & 1) != (channel & 1))
                        $fatal(1, "channel%0d parity mismatch token=%0d",
                               channel, token);
                    if (token_update_count[bank][token] != 0)
                        $fatal(1, "bank%0d token%0d duplicate Acc update",
                               bank, token);
                    token_update_count[bank][token] = 1;
                    update_count[bank] = update_count[bank] + 1;
                    parity_update_count[channel] =
                        parity_update_count[channel] + 1;
                    case (token)
                        2, 3, 4: begin
                            expected_gate = 2;
                            expected_weight_base = 2 + (bank * 9);
                            if (acc_update_tags[
                                (bank*GROUP_TAG_W) +: GROUP_TAG_W] != 12'h120)
                                $fatal(1, "term0 Acc tag mismatch");
                        end
                        5, 6: begin
                            expected_gate = 3;
                            expected_weight_base = 5 + (bank * 9);
                            if (acc_update_tags[
                                (bank*GROUP_TAG_W) +: GROUP_TAG_W] != 12'h120)
                                $fatal(1, "term1 Acc tag mismatch");
                        end
                        7: begin
                            expected_gate = 4;
                            expected_weight_base = 8 + (bank * 9);
                            if (acc_update_tags[
                                (bank*GROUP_TAG_W) +: GROUP_TAG_W] != 12'h2aa)
                                $fatal(1, "ABA Acc tag mismatch");
                        end
                        12, 13, 14: begin
                            expected_gate = 5;
                            expected_weight_base = 32 + (bank * 9);
                            if (acc_update_tags[
                                (bank*GROUP_TAG_W) +: GROUP_TAG_W] != 12'h155)
                                $fatal(1, "mixed legal Acc tag mismatch");
                        end
                        default: $fatal(1, "unexpected Acc token %0d", token);
                    endcase
                    for (lane = 0; lane < OUT_TILE; lane = lane + 1) begin
                        observed_product = 32'($signed(acc_update_values[
                            (bank*OUT_TILE*PRODUCT_W) +
                            (lane*PRODUCT_W) +: PRODUCT_W]));
                        expected_product = expected_gate *
                                           (expected_weight_base + lane);
                        if (observed_product != expected_product)
                            $fatal(1,
                                "bank%0d token%0d lane%0d product got=%0d exp=%0d",
                                bank, token, lane, observed_product,
                                expected_product);
                    end
                end
            end

            if (dispatch_retire_valid && !dispatch_retire_term_last &&
                first_nonlast_retire_cycle < 0)
                first_nonlast_retire_cycle = cycle_count;
            if (dispatch_retire_valid && dispatch_retire_head_last)
                head_retire_cycle = cycle_count;
            if (head_compute_done) begin
                if ((head_bank_done_seen | current_head_done) != 3'b111 ||
                    current_head_done == '0)
                    $fatal(1, "head done did not coincide with third bank completion");
                if (head_compute_done_group_tag != 12'h120 ||
                    head_compute_done_issue_seq != 6'd11)
                    $fatal(1, "head done metadata mismatch");
                head_done_count = head_done_count + 1;
                head_done_cycle = cycle_count;
            end
            if (flush) begin
                if (term_ready || event_ready || weight_req_valid != '0 ||
                    weight_rsp_ready != '0 || acc_update_valid != '0 ||
                    bank_term_done != '0 || head_compute_done ||
                    dispatch_retire_valid)
                    $fatal(1, "flush did not mask top interfaces");
                flush_mask_seen = 1'b1;
                head_bank_done_seen = '0;
            end
        end
    end

    initial begin : p_test
        integer bank;
        integer channel;
        integer token;
        clk_core = 1'b0;
        rst_core = 1'b1;
        flush = 1'b0;
        clear_error = 1'b0;
        term_valid = 1'b0;
        term_tag = '0;
        term_gate_code = '0;
        term_lane_id = '0;
        term_destination_count = '0;
        term_issue_seq = '0;
        term_head_last = 1'b0;
        logical_supertile = '0;
        head_input_channel_base = '0;
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
        weight_req_ready = '0;
        weight_rsp_valid = '0;
        weight_rsp_tags = '0;
        weight_rsp_input_channels = '0;
        weight_rsp_output_tiles = '0;
        weight_rsp_epochs = '0;
        weight_rsp_weights = '0;
        acc_update_ready = '0;
        cycle_count = 0;
        head_done_count = 0;
        head_done_cycle = -1;
        head_retire_cycle = -1;
        first_nonlast_retire_cycle = -1;
        first_term_done_cycle = -1;
        head_bank_done_seen = '0;
        head_done_this_cycle = '0;
        fabric_full_seen = 1'b0;
        flush_mask_seen = 1'b0;
        aba_drop_seen = 1'b0;
        cross_term_overlap_seen = 1'b0;
        idle_collect_seen = 1'b0;
        idle_emit_seen = 1'b0;
        idle_fabric_seen = 1'b0;
        idle_executor_seen = 1'b0;
        for (bank = 0; bank < 3; bank = bank + 1) begin
            weight_request_count[bank] = 0;
            update_count[bank] = 0;
            term_done_count[bank] = 0;
            weight_stall_seen[bank] = 0;
            canceled_epoch[bank] = '0;
            replacement_epoch[bank] = '0;
            for (token = 0; token < TOKENS; token = token + 1)
                token_update_count[bank][token] = 0;
        end
        for (channel = 0; channel < 6; channel = channel + 1) begin
            parity_update_count[channel] = 0;
            acc_stall_seen[channel] = 0;
        end

        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        // Illegal channel metadata is consumed and its payload is drained.
        @(negedge clk_core);
        term_tag = 12'hbad;
        term_gate_code = 9'd2;
        term_lane_id = 5'd2;
        term_destination_count = 8'd1;
        term_issue_seq = 6'd1;
        term_head_last = 1'b0;
        logical_supertile = 4'd0;
        head_input_channel_base = 7'd95;
        term_valid = 1'b1;
        do @(posedge clk_core); while (!term_ready);
        @(negedge clk_core);
        term_valid = 1'b0;
        event_gate_code = 9'd2;
        event_lane_id = 5'd2;
        event_token_valid = 4'b0001;
        event_token_ids = {5'd0, 5'd0, 5'd0, 5'd3};
        event_count = 3'd1;
        event_issue_seq = 6'd1;
        event_term_first = 1'b1;
        event_term_last = 1'b1;
        event_head_last = 1'b0;
        event_valid = 1'b1;
        do @(posedge clk_core); while (!event_ready);
        @(negedge clk_core);
        event_valid = 1'b0;
        if (weight_req_valid != '0 || acc_update_valid != '0 ||
            issued_terms != 0 || !dut.adapter_idle ||
            dut.illegal_drop_active_q)
            $fatal(1, "illegal input channel drain had side effects");

        // 3*15+2 exceeds a five-bit physical output tile and is also drained.
        @(negedge clk_core);
        term_tag = 12'hbae;
        term_gate_code = 9'd2;
        term_lane_id = 5'd1;
        term_destination_count = 8'd1;
        term_issue_seq = 6'd2;
        logical_supertile = 4'd15;
        head_input_channel_base = 7'd4;
        term_valid = 1'b1;
        do @(posedge clk_core); while (!term_ready);
        @(negedge clk_core);
        term_valid = 1'b0;
        event_gate_code = 9'd2;
        event_lane_id = 5'd1;
        event_token_valid = 4'b0001;
        event_token_ids = {5'd0, 5'd0, 5'd0, 5'd4};
        event_count = 3'd1;
        event_issue_seq = 6'd2;
        event_term_first = 1'b1;
        event_term_last = 1'b1;
        event_head_last = 1'b0;
        event_valid = 1'b1;
        do @(posedge clk_core); while (!event_ready);
        @(negedge clk_core);
        event_valid = 1'b0;
        if (!protocol_error || issued_terms != 0 ||
            weight_req_valid != '0 || acc_update_valid != '0 ||
            !dut.adapter_idle || dut.illegal_drop_active_q)
            $fatal(1, "illegal metadata audit mismatch");

        // A zero-destination illegal term is consumed without entering drain.
        @(negedge clk_core);
        term_destination_count = 8'd0;
        logical_supertile = 4'd0;
        head_input_channel_base = 7'd95;
        term_valid = 1'b1;
        do @(posedge clk_core); while (!term_ready);
        @(negedge clk_core);
        term_valid = 1'b0;
        if (dut.illegal_drop_active_q || !term_ready || issued_terms != 0)
            $fatal(1, "zero-destination illegal term did not recover");

        // Flush aborts an illegal drain without waiting for the old last event.
        term_destination_count = 8'd2;
        term_valid = 1'b1;
        do @(posedge clk_core); while (!term_ready);
        @(negedge clk_core);
        term_valid = 1'b0;
        if (!dut.illegal_drop_active_q)
            $fatal(1, "nonempty illegal term did not enter drain");
        flush = 1'b1;
        #1;
        if (term_ready || event_ready)
            $fatal(1, "flush did not mask illegal drain interfaces");
        @(posedge clk_core);
        @(negedge clk_core);
        flush = 1'b0;
        #1;
        if (dut.illegal_drop_active_q || !term_ready)
            $fatal(1, "flush did not release illegal drain");

        // A newly accepted illegal term wins over a concurrent clear_error.
        term_destination_count = 8'd0;
        clear_error = 1'b1;
        term_valid = 1'b1;
        do @(posedge clk_core); while (!term_ready);
        @(negedge clk_core);
        clear_error = 1'b0;
        term_valid = 1'b0;
        if (!protocol_error)
            $fatal(1, "concurrent clear swallowed new illegal metadata error");

        if (EXTENDED_2C_AUDIT) begin
            // Fill a legal context until its commands occupy the fabric.
            drive_term(12'h155, 9'd5, 5'd5, 6'd7, 1'b0,
                       4'd0, 7'd8, 3, 5'd12, 5'd13, 5'd14);
            while (fabric_occupancy != OCC_W'(Q)) @(posedge clk_core);
            mixed_fabric_accepted_before = fabric_count_accepted;
            mixed_issued_before = issued_terms;
            for (bank = 0; bank < 3; bank = bank + 1) begin
                mixed_weight_before[bank] = weight_request_count[bank];
                mixed_update_before[bank] = update_count[bank];
                mixed_done_before[bank] = term_done_count[bank];
            end

            // The free context accepts an illegal term and drains its payload
            // while the previously committed legal context remains in flight.
            @(negedge clk_core);
            term_tag = 12'hbad;
            term_gate_code = 9'd6;
            term_lane_id = 5'd2;
            term_destination_count = 8'd1;
            term_issue_seq = 6'd8;
            term_head_last = 1'b0;
            logical_supertile = 4'd0;
            head_input_channel_base = 7'd95;
            term_valid = 1'b1;
            do @(posedge clk_core); while (!term_ready);
            @(negedge clk_core);
            term_valid = 1'b0;
            event_gate_code = 9'd6;
            event_lane_id = 5'd2;
            event_token_valid = 4'b0001;
            event_token_ids = {5'd0, 5'd0, 5'd0, 5'd9};
            event_count = 3'd1;
            event_issue_seq = 6'd8;
            event_term_first = 1'b1;
            event_term_last = 1'b1;
            event_head_last = 1'b0;
            event_valid = 1'b1;
            do @(posedge clk_core); while (!event_ready);
            @(negedge clk_core);
            event_valid = 1'b0;
            if (dut.illegal_drop_active_q ||
                fabric_count_accepted != mixed_fabric_accepted_before ||
                issued_terms != mixed_issued_before ||
                adapter_context_valid != 2'b01)
                $fatal(1, "2C legal/illegal overlap isolation failed");

            // Release the previously blocked legal context and prove that its
            // payload and sidebands survived the overlapping illegal drain.
            fork
                sink_acc_channel(0, 2);
                sink_acc_channel(1, 1);
                sink_acc_channel(2, 2);
                sink_acc_channel(3, 1);
                sink_acc_channel(4, 2);
                sink_acc_channel(5, 1);
                serve_weight_request(0, 1, 2, 12'h155, 7'd13, 5'd0, 32);
                serve_weight_request(1, 2, 1, 12'h155, 7'd13, 5'd1, 41);
                serve_weight_request(2, 1, 3, 12'h155, 7'd13, 5'd2, 50);
            join
            while ((term_done_count[0] == mixed_done_before[0]) ||
                   (term_done_count[1] == mixed_done_before[1]) ||
                   (term_done_count[2] == mixed_done_before[2]))
                @(posedge clk_core);
            repeat (2) @(posedge clk_core);
            for (bank = 0; bank < 3; bank = bank + 1) begin
                if (weight_request_count[bank] !=
                        mixed_weight_before[bank] + 1 ||
                    update_count[bank] != mixed_update_before[bank] + 3 ||
                    term_done_count[bank] != mixed_done_before[bank] + 1 ||
                    token_update_count[bank][12] != 1 ||
                    token_update_count[bank][13] != 1 ||
                    token_update_count[bank][14] != 1)
                    $fatal(1,
                        "2C mixed legal recovery mismatch bank=%0d req=%0d update=%0d done=%0d tokens=%0d/%0d/%0d",
                        bank, weight_request_count[bank], update_count[bank],
                        term_done_count[bank], token_update_count[bank][12],
                        token_update_count[bank][13],
                        token_update_count[bank][14]);
            end
            if (issued_terms != 1 || fabric_count_accepted != 3 ||
                adapter_context_valid != 2'b00 || !datapath_idle)
                $fatal(1,
                    "2C mixed legal context did not retire cleanly issued=%0d accepted=%0d context=%b idle=%0b",
                    issued_terms, fabric_count_accepted,
                    adapter_context_valid, datapath_idle);

            @(negedge clk_core);
            flush = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            flush = 1'b0;
            for (bank = 0; bank < 3; bank = bank + 1) begin
                weight_request_count[bank] = 0;
                update_count[bank] = 0;
                term_done_count[bank] = 0;
                weight_stall_seen[bank] = 0;
            end
            for (channel = 0; channel < 6; channel = channel + 1) begin
                parity_update_count[channel] = 0;
                acc_stall_seen[channel] = 0;
            end
            first_nonlast_retire_cycle = -1;
            first_term_done_cycle = -1;
            head_retire_cycle = -1;
            head_done_cycle = -1;
        end

        fork
            sink_acc_channel(0, 3);
            sink_acc_channel(1, 3);
            sink_acc_channel(2, 3);
            sink_acc_channel(3, 3);
            sink_acc_channel(4, 3);
            sink_acc_channel(5, 3);
        join_none

        fork
            begin
                drive_term(12'h120, 9'd2, 5'd2, 6'd10, 1'b0,
                           4'd0, 7'd10, 3, 5'd2, 5'd3, 5'd4);
                drive_term(12'h120, 9'd3, 5'd3, 6'd11, 1'b1,
                           4'd1, 7'd20, 2, 5'd5, 5'd6, 5'd0);
            end
            begin
                serve_weight_request(0, 2, 5, 12'h120, 7'd12, 5'd0, 2);
                serve_weight_request(0, 1, 1, 12'h120, 7'd23, 5'd3, 5);
            end
            begin
                serve_weight_request(1, 3, 1, 12'h120, 7'd12, 5'd1, 11);
                serve_weight_request(1, 2, 6, 12'h120, 7'd23, 5'd4, 14);
            end
            begin
                serve_weight_request(2, 1, 3, 12'h120, 7'd12, 5'd2, 20);
                serve_weight_request(2, 3, 3, 12'h120, 7'd23, 5'd5, 23);
            end
        join

        while (head_done_count == 0) @(posedge clk_core);
        repeat (3) @(posedge clk_core);
        if (head_done_count != 1 || head_retire_cycle < 0 ||
            head_retire_cycle > head_done_cycle ||
            first_nonlast_retire_cycle < 0 || first_term_done_cycle < 0 ||
            first_nonlast_retire_cycle >= head_done_cycle)
            $fatal(1, "dispatch/compute ordering coverage failed nonlast_retire=%0d first_done=%0d head_retire=%0d head_done=%0d",
                   first_nonlast_retire_cycle, first_term_done_cycle,
                   head_retire_cycle, head_done_cycle);

        // Cancel an in-flight same-identity term after all bank requests.
        fork
            drive_term(12'h2aa, 9'd4, 5'd4, 6'd20, 1'b0,
                       4'd2, 7'd30, 1, 5'd8, 5'd0, 5'd0);
            capture_weight_request(0, 1, 12'h2aa, 7'd34, 5'd6,
                                   canceled_epoch[0]);
            capture_weight_request(1, 2, 12'h2aa, 7'd34, 5'd7,
                                   canceled_epoch[1]);
            capture_weight_request(2, 3, 12'h2aa, 7'd34, 5'd8,
                                   canceled_epoch[2]);
        join
        @(negedge clk_core);
        flush = 1'b1;
        #1;
        if (term_ready || event_ready || weight_req_valid != '0 ||
            weight_rsp_ready != '0 || acc_update_valid != '0 ||
            bank_term_done != '0 || head_compute_done ||
            dispatch_retire_valid)
            $fatal(1, "combinational flush masking failed");
        @(posedge clk_core);
        @(negedge clk_core);
        flush = 1'b0;

        fork
            drive_term(12'h2aa, 9'd4, 5'd4, 6'd21, 1'b0,
                       4'd2, 7'd30, 1, 5'd7, 5'd0, 5'd0);
            capture_weight_request(0, 1, 12'h2aa, 7'd34, 5'd6,
                                   replacement_epoch[0]);
            capture_weight_request(1, 2, 12'h2aa, 7'd34, 5'd7,
                                   replacement_epoch[1]);
            capture_weight_request(2, 1, 12'h2aa, 7'd34, 5'd8,
                                   replacement_epoch[2]);
        join
        for (bank = 0; bank < 3; bank = bank + 1) begin
            if (replacement_epoch[bank] == canceled_epoch[bank])
                $fatal(1, "bank%0d epoch did not advance across flush", bank);
        end

        fork
            send_weight_response(0, 0, 12'h2aa, 7'd34, 5'd6,
                                 canceled_epoch[0], 61);
            send_weight_response(1, 2, 12'h2aa, 7'd34, 5'd7,
                                 canceled_epoch[1], 62);
            send_weight_response(2, 1, 12'h2aa, 7'd34, 5'd8,
                                 canceled_epoch[2], 63);
        join
        aba_drop_seen = 1'b1;
        if (acc_update_valid != '0 || bank_term_done != '0)
            $fatal(1, "stale ABA response polluted compute outputs");

        fork
            send_weight_response(0, 2, 12'h2aa, 7'd34, 5'd6,
                                 replacement_epoch[0], 8);
            send_weight_response(1, 0, 12'h2aa, 7'd34, 5'd7,
                                 replacement_epoch[1], 17);
            send_weight_response(2, 4, 12'h2aa, 7'd34, 5'd8,
                                 replacement_epoch[2], 26);
        join

        while ((term_done_count[0] < 3) || (term_done_count[1] < 3) ||
               (term_done_count[2] < 3)) @(posedge clk_core);
        repeat (3) @(posedge clk_core);

        for (bank = 0; bank < 3; bank = bank + 1) begin
            if (weight_request_count[bank] != 4 || update_count[bank] != 6 ||
                term_done_count[bank] != 3 ||
                count_stale_weight_responses[
                    (bank*COUNTER_W) +: COUNTER_W] != 1 ||
                completed_terms[(bank*COUNTER_W) +: COUNTER_W] !=
                    COUNTER_W'(3 + (EXTENDED_2C_AUDIT ? 1 : 0)) ||
                (weight_stall_seen[bank] == 0))
                $fatal(1, "bank%0d accounting mismatch req=%0d update=%0d done=%0d completed=%0d stale=%0d stall=%0d",
                       bank, weight_request_count[bank], update_count[bank],
                       term_done_count[bank], completed_terms[
                           (bank*COUNTER_W) +: COUNTER_W],
                       count_stale_weight_responses[
                           (bank*COUNTER_W) +: COUNTER_W],
                       weight_stall_seen[bank]);
            for (token = 2; token <= 7; token = token + 1) begin
                if (token_update_count[bank][token] != 1)
                    $fatal(1, "bank%0d token%0d update count mismatch",
                           bank, token);
            end
            if (token_update_count[bank][8] != 0)
                $fatal(1, "canceled token polluted bank%0d", bank);
        end
        for (channel = 0; channel < 6; channel = channel + 1) begin
            if (parity_update_count[channel] != 3 ||
                (acc_stall_seen[channel] == 0))
                $fatal(1, "Acc channel%0d coverage mismatch updates=%0d stall=%0d",
                       channel, parity_update_count[channel],
                       acc_stall_seen[channel]);
        end
        if (issued_terms != COUNTER_W'(
                4 + (EXTENDED_2C_AUDIT ? 1 : 0)) ||
            !protocol_error ||
            (dut.executor_protocol_error != '0) ||
            dut.adapter_protocol_error || dut.tracking_protocol_error ||
            !fabric_full_seen ||
            !flush_mask_seen || !aba_drop_seen || head_done_count != 1 ||
            !cross_term_overlap_seen || fabric_count_skew_cycles == 0 ||
            !idle_collect_seen || !idle_emit_seen || !idle_fabric_seen ||
            !idle_executor_seen || !datapath_idle)
            $fatal(1, "top coverage/accounting failed issued=%0d error=%0b child_error=%0b/%0b/%0b full=%0b flush=%0b aba=%0b overlap=%0b head=%0d skew=%0d",
                   issued_terms, protocol_error, dut.executor_protocol_error,
                   dut.adapter_protocol_error, dut.tracking_protocol_error,
                   fabric_full_seen, flush_mask_seen,
                   aba_drop_seen, cross_term_overlap_seen, head_done_count,
                   fabric_count_skew_cycles);

        @(negedge clk_core);
        clear_error = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        clear_error = 1'b0;
        if (protocol_error)
            $fatal(1, "clear_error did not clear DCTF96 sticky error");

        $display("PASS DCTF96 TERM DATAPATH cycles=%0d issued=%0d completed={%0d,%0d,%0d} weight_req={%0d,%0d,%0d} acc_updates={%0d,%0d,%0d} parity={%0d,%0d,%0d,%0d,%0d,%0d} stale={%0d,%0d,%0d} head_done_cycle=%0d head_retire_cycle=%0d max_occ=%0d skew=%0d",
                 cycle_count, issued_terms,
                 completed_terms[0 +: COUNTER_W],
                 completed_terms[COUNTER_W +: COUNTER_W],
                 completed_terms[(2*COUNTER_W) +: COUNTER_W],
                 weight_request_count[0], weight_request_count[1],
                 weight_request_count[2], update_count[0], update_count[1],
                 update_count[2], parity_update_count[0],
                 parity_update_count[1], parity_update_count[2],
                 parity_update_count[3], parity_update_count[4],
                 parity_update_count[5],
                 count_stale_weight_responses[0 +: COUNTER_W],
                 count_stale_weight_responses[COUNTER_W +: COUNTER_W],
                 count_stale_weight_responses[(2*COUNTER_W) +: COUNTER_W],
                 head_done_cycle, head_retire_cycle, fabric_max_occupancy,
                 fabric_count_skew_cycles);
        $finish;
    end

    initial begin
        repeat (10000) @(posedge clk_core);
        $display("TIMEOUT state cycle=%0d term=%b/%b event=%b/%b wreq=%b/%b wrsp=%b/%b acc=%b/%b done=%b issued=%0d completed=%h occ=%0d error=%b",
                 cycle_count, term_valid, term_ready, event_valid, event_ready,
                 weight_req_valid, weight_req_ready, weight_rsp_valid,
                 weight_rsp_ready, acc_update_valid, acc_update_ready,
                 bank_term_done, issued_terms, completed_terms,
                 fabric_occupancy, protocol_error);
        $fatal(1, "DCTF96 term datapath TB timeout");
    end
endmodule

/* verilator lint_on UNUSEDSIGNAL */
/* verilator lint_on BLKSEQ */

`default_nettype wire
