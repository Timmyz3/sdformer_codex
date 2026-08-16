`timescale 1ns/1ps
`default_nettype none

// PPDI bank executor. One even and one odd destination may share the product
// of a term command. Each accumulator port can handshake independently, while
// the command retires only after every valid destination has committed once.
module gatestack_ppdi_dctf32_bank_executor #(
    parameter int BANK_ID = 0,
    parameter int BANK_COUNT = 3,
    parameter int TOKENS = 162,
    parameter int OUT_TILE = 32,
    parameter int GATE_W = 9,
    parameter int WEIGHT_W = 8,
    parameter int PRODUCT_W = GATE_W + WEIGHT_W,
    parameter int GROUP_TAG_W = 32,
    parameter int CMD_SEQUENCE_W = 16,
    parameter int ISSUE_SEQ_W = 13,
    parameter int INPUT_CH_W = 10,
    parameter int LANE_ID_W = 5,
    parameter int TOKEN_ID_W = 8,
    parameter int OUTPUT_TILE_W = 8,
    parameter int LOGICAL_SUPERTILE_W = OUTPUT_TILE_W,
    parameter int EPOCH_W = 4,
    parameter int COUNTER_W = 32
) (
    input  logic                                      clk_core,
    input  logic                                      rst_core,
    input  logic                                      flush,
    input  logic                                      clear_error,

    input  logic                                      cmd_valid,
    output logic                                      cmd_ready,
    input  logic [GROUP_TAG_W-1:0]                    cmd_group_tag,
    input  logic [CMD_SEQUENCE_W-1:0]                 cmd_sequence,
    input  logic [ISSUE_SEQ_W-1:0]                    cmd_term_issue_seq,
    input  logic                                      cmd_term_first,
    input  logic                                      cmd_term_last,
    input  logic                                      cmd_head_last,
    input  logic [INPUT_CH_W-1:0]                     cmd_input_channel,
    input  logic [GATE_W-1:0]                         cmd_gate_code,
    input  logic [LANE_ID_W-1:0]                      cmd_lane_id,
    input  logic [1:0]                                cmd_destination_valid,
    input  logic [(2*TOKEN_ID_W)-1:0]                 cmd_destination_tokens,
    input  logic [LOGICAL_SUPERTILE_W-1:0]            logical_supertile,

    output logic                                      weight_req_valid,
    input  logic                                      weight_req_ready,
    output logic [GROUP_TAG_W-1:0]                    weight_req_tag,
    output logic [INPUT_CH_W-1:0]                     weight_req_input_channel,
    output logic [OUTPUT_TILE_W-1:0]                  weight_req_output_tile,
    output logic [EPOCH_W-1:0]                        weight_req_epoch,
    input  logic                                      weight_rsp_valid,
    output logic                                      weight_rsp_ready,
    input  logic [GROUP_TAG_W-1:0]                    weight_rsp_tag,
    input  logic [INPUT_CH_W-1:0]                     weight_rsp_input_channel,
    input  logic [OUTPUT_TILE_W-1:0]                  weight_rsp_output_tile,
    input  logic [EPOCH_W-1:0]                        weight_rsp_epoch,
    input  logic [(OUT_TILE*WEIGHT_W)-1:0]            weight_rsp_weights,

    output logic [1:0]                                acc_update_valid,
    input  logic [1:0]                                acc_update_ready,
    output logic [(2*TOKEN_ID_W)-1:0]                 acc_update_token_ids,
    output logic [GROUP_TAG_W-1:0]                    acc_update_tag,
    output logic [(OUT_TILE*PRODUCT_W)-1:0]           acc_update_values,

    output logic                                      term_done,
    output logic [GROUP_TAG_W-1:0]                    term_done_group_tag,
    output logic [ISSUE_SEQ_W-1:0]                    term_done_issue_seq,
    output logic                                      term_done_head_last,
    output logic                                      protocol_error,
    output logic [COUNTER_W-1:0]                      count_stale_weight_responses
);
    localparam int ENGINE_TAG_W = EPOCH_W + GROUP_TAG_W;
    localparam int EPOCH_COUNT = 32'd1 << EPOCH_W;

    logic term_active_q;
    logic first_pending_q;
    logic sequence_valid_q;
    logic [GROUP_TAG_W-1:0] term_group_tag_q;
    logic [CMD_SEQUENCE_W-1:0] first_sequence_q;
    logic [CMD_SEQUENCE_W-1:0] next_sequence_q;
    logic [ISSUE_SEQ_W-1:0] term_issue_seq_q;
    logic [INPUT_CH_W-1:0] term_input_channel_q;
    logic [GATE_W-1:0] term_gate_code_q;
    logic [LANE_ID_W-1:0] term_lane_id_q;
    logic [1:0] first_destination_valid_q;
    logic [(2*TOKEN_ID_W)-1:0] first_destination_tokens_q;
    logic [1:0] destination_done_q;
    logic [LOGICAL_SUPERTILE_W-1:0] logical_supertile_q;
    logic [OUTPUT_TILE_W-1:0] physical_tile_q;
    logic [EPOCH_W-1:0] epoch_q;
    logic [EPOCH_W-1:0] term_epoch_q;
    logic [EPOCH_COUNT-1:0] stale_epoch_pending_q;
    logic [GROUP_TAG_W-1:0] stale_epoch_tag_q [0:EPOCH_COUNT-1];
    logic [INPUT_CH_W-1:0] stale_epoch_channel_q [0:EPOCH_COUNT-1];
    logic [OUTPUT_TILE_W-1:0] stale_epoch_tile_q [0:EPOCH_COUNT-1];
    logic current_weight_outstanding_q;
    logic [GROUP_TAG_W-1:0] current_weight_tag_q;
    logic [INPUT_CH_W-1:0] current_weight_channel_q;
    logic [OUTPUT_TILE_W-1:0] current_weight_tile_q;
    logic epoch_space_exhausted_q;
    logic first_term_last_q;
    logic first_head_last_q;

    logic [OUTPUT_TILE_W-1:0] physical_tile_comb;
    logic destination_contract_ok;
    logic start_sequence_ok;
    logic start_contract_ok;
    logic identity_matches;
    logic first_command_matches;
    logic continuation_matches;
    logic active_command_ok;
    logic command_protocol_bad;
    logic acc_update_enable;
    logic all_destinations_complete;
    logic [1:0] acc_port_fire;
    logic update_fire;
    logic start_fire;
    logic zero_gate_start_fire;
    logic zero_gate_active;
    logic zero_gate_retire_enable;

    logic engine_term_valid;
    logic engine_term_ready;
    logic engine_weight_req_valid;
    logic engine_weight_rsp_valid;
    logic engine_weight_rsp_ready;
    logic [ENGINE_TAG_W-1:0] engine_weight_req_tag;
    logic engine_product_valid;
    logic engine_product_ready;
    logic [ENGINE_TAG_W-1:0] engine_product_tag;
    logic [INPUT_CH_W-1:0] engine_product_input_channel;
    logic [OUTPUT_TILE_W-1:0] engine_product_output_tile;
    logic [ISSUE_SEQ_W-1:0] engine_product_issue_seq;
    logic [(OUT_TILE*PRODUCT_W)-1:0] engine_product_values;
    logic engine_product_identity_ok;
    logic engine_protocol_error;
    logic weight_response_is_stale;
    logic stale_weight_response_fire;
    logic stale_response_was_pending;
    logic stale_response_identity_matches;
    logic unknown_stale_response_fire;
    logic weight_request_fire;
    logic current_weight_response_fire;
    logic [EPOCH_COUNT-1:0] pending_after_flush_comb;
    logic [EPOCH_W-1:0] next_epoch_comb;
    logic next_epoch_found_comb;

    /* verilator lint_off UNUSEDSIGNAL */
    logic [COUNTER_W-1:0] engine_count_terms;
    logic [COUNTER_W-1:0] engine_count_weight_requests;
    logic [COUNTER_W-1:0] engine_count_products;
    logic [COUNTER_W-1:0] engine_count_weight_wait_cycles;
    logic [COUNTER_W-1:0] engine_count_output_stall_cycles;
    /* verilator lint_on UNUSEDSIGNAL */

    assign physical_tile_comb = OUTPUT_TILE_W'(
        32'(logical_supertile) * 32'(BANK_COUNT) + 32'(BANK_ID));
    assign destination_contract_ok =
        (cmd_destination_valid != 2'b00) &&
        (!cmd_destination_valid[0] ||
            ((32'(cmd_destination_tokens[0 +: TOKEN_ID_W]) < TOKENS) &&
             !cmd_destination_tokens[0])) &&
        (!cmd_destination_valid[1] ||
            ((32'(cmd_destination_tokens[TOKEN_ID_W +: TOKEN_ID_W]) < TOKENS) &&
             cmd_destination_tokens[TOKEN_ID_W]));
    assign start_sequence_ok = !sequence_valid_q ||
                               (cmd_sequence == next_sequence_q);
    assign start_contract_ok = cmd_term_first &&
                               (!cmd_head_last || cmd_term_last) &&
                               destination_contract_ok &&
                               start_sequence_ok;

    assign identity_matches =
        (cmd_group_tag == term_group_tag_q) &&
        (cmd_term_issue_seq == term_issue_seq_q) &&
        (cmd_input_channel == term_input_channel_q) &&
        (cmd_gate_code == term_gate_code_q) &&
        (cmd_lane_id == term_lane_id_q) &&
        (logical_supertile == logical_supertile_q);
    assign first_command_matches = identity_matches && cmd_term_first &&
        (cmd_sequence == first_sequence_q) &&
        (cmd_destination_valid == first_destination_valid_q) &&
        (cmd_destination_tokens == first_destination_tokens_q) &&
        (cmd_term_last == first_term_last_q) &&
        (cmd_head_last == first_head_last_q) && destination_contract_ok;
    assign continuation_matches = identity_matches && !cmd_term_first &&
        (cmd_sequence == next_sequence_q) &&
        (!cmd_head_last || cmd_term_last) && destination_contract_ok;
    assign active_command_ok = first_pending_q ? first_command_matches :
                                                continuation_matches;
    assign command_protocol_bad = cmd_valid && !flush &&
        ((!term_active_q && !start_contract_ok) ||
         (term_active_q && !active_command_ok));

    assign engine_term_valid = !flush && !epoch_space_exhausted_q &&
                               cmd_valid && !term_active_q &&
                               start_contract_ok && (cmd_gate_code != '0);
    assign zero_gate_start_fire = !flush && cmd_valid && !term_active_q &&
                                  start_contract_ok && (cmd_gate_code == '0);
    assign start_fire = zero_gate_start_fire ||
                        (engine_term_valid && engine_term_ready);
    assign zero_gate_active = term_active_q && (term_gate_code_q == '0);
    assign engine_product_identity_ok =
        (engine_product_tag == {term_epoch_q, term_group_tag_q}) &&
        (engine_product_input_channel == term_input_channel_q) &&
        (engine_product_output_tile == physical_tile_q) &&
        (engine_product_issue_seq == term_issue_seq_q);

    assign acc_update_enable = !flush && term_active_q && cmd_valid &&
        !zero_gate_active && active_command_ok &&
        engine_product_valid && engine_product_identity_ok;
    assign acc_update_valid = acc_update_enable ?
                              (cmd_destination_valid & ~destination_done_q) :
                              2'b00;
    assign acc_update_token_ids = cmd_destination_tokens;
    assign acc_update_tag = term_group_tag_q;
    assign acc_update_values = engine_product_values;
    assign acc_port_fire = acc_update_valid & acc_update_ready;
    assign all_destinations_complete =
        &((~cmd_destination_valid) | destination_done_q | acc_update_ready);

    assign zero_gate_retire_enable = !flush && zero_gate_active &&
                                     cmd_valid && active_command_ok;
    assign cmd_ready = zero_gate_retire_enable ||
                       (acc_update_enable && all_destinations_complete);
    assign update_fire = cmd_valid && cmd_ready;
    assign engine_product_ready = !flush && update_fire && cmd_term_last;
    assign term_done = !flush && update_fire && cmd_term_last;
    assign term_done_group_tag = term_group_tag_q;
    assign term_done_issue_seq = term_issue_seq_q;
    assign term_done_head_last = term_done && cmd_head_last;

    assign weight_req_valid = !flush && engine_weight_req_valid;
    assign weight_req_tag = engine_weight_req_tag[GROUP_TAG_W-1:0];
    assign weight_req_epoch = engine_weight_req_tag[ENGINE_TAG_W-1 -: EPOCH_W];
    assign weight_request_fire = weight_req_valid && weight_req_ready;
    assign stale_response_was_pending =
        stale_epoch_pending_q[weight_rsp_epoch];
    assign stale_response_identity_matches = stale_response_was_pending &&
        (weight_rsp_tag == stale_epoch_tag_q[weight_rsp_epoch]) &&
        (weight_rsp_input_channel ==
            stale_epoch_channel_q[weight_rsp_epoch]) &&
        (weight_rsp_output_tile == stale_epoch_tile_q[weight_rsp_epoch]);
    assign weight_response_is_stale = stale_response_was_pending ||
                                      (weight_rsp_epoch != epoch_q);
    assign weight_rsp_ready = !flush &&
        ((weight_rsp_valid && weight_response_is_stale) ||
         engine_weight_rsp_ready);
    assign stale_weight_response_fire = weight_rsp_valid &&
                                        weight_rsp_ready &&
                                        weight_response_is_stale;
    assign unknown_stale_response_fire = stale_weight_response_fire &&
                                         !stale_response_identity_matches;
    assign engine_weight_rsp_valid = weight_rsp_valid && !flush &&
                                     !weight_response_is_stale;
    assign current_weight_response_fire = engine_weight_rsp_valid &&
                                          engine_weight_rsp_ready;

    always_comb begin
        pending_after_flush_comb = stale_epoch_pending_q;
        if (current_weight_outstanding_q)
            pending_after_flush_comb[epoch_q] = 1'b1;
        next_epoch_comb = epoch_q;
        next_epoch_found_comb = 1'b0;
        for (int offset = 32'd1; offset <= EPOCH_COUNT;
             offset = offset + 32'd1) begin
            if (!next_epoch_found_comb &&
                !pending_after_flush_comb[EPOCH_W'(
                    32'(epoch_q) + offset)]) begin
                next_epoch_comb = EPOCH_W'(32'(epoch_q) + offset);
                next_epoch_found_comb = 1'b1;
            end
        end
    end

    gatestack_decoupled_product_engine #(
        .GATE_W(GATE_W), .WEIGHT_W(WEIGHT_W), .PRODUCT_W(PRODUCT_W),
        .OUT_TILE(OUT_TILE), .INPUT_CH_W(INPUT_CH_W),
        .OUTPUT_TILE_W(OUTPUT_TILE_W), .ISSUE_SEQ_W(ISSUE_SEQ_W),
        .TAG_W(ENGINE_TAG_W), .COUNTER_W(COUNTER_W)
    ) u_product_engine (
        .clk_core(clk_core), .rst_core(rst_core || flush),
        .clear_error(clear_error), .term_valid(engine_term_valid),
        .term_ready(engine_term_ready),
        .term_tag({epoch_q, cmd_group_tag}),
        .term_gate_code(cmd_gate_code),
        .term_input_channel(cmd_input_channel),
        .term_output_tile(physical_tile_comb),
        .term_issue_seq(cmd_term_issue_seq),
        .weight_req_valid(engine_weight_req_valid),
        .weight_req_ready(weight_req_ready),
        .weight_req_tag(engine_weight_req_tag),
        .weight_req_input_channel(weight_req_input_channel),
        .weight_req_output_tile(weight_req_output_tile),
        .weight_rsp_valid(engine_weight_rsp_valid),
        .weight_rsp_ready(engine_weight_rsp_ready),
        .weight_rsp_tag({weight_rsp_epoch, weight_rsp_tag}),
        .weight_rsp_input_channel(weight_rsp_input_channel),
        .weight_rsp_output_tile(weight_rsp_output_tile),
        .weight_rsp_weights(weight_rsp_weights),
        .product_valid(engine_product_valid),
        .product_ready(engine_product_ready),
        .product_tag(engine_product_tag),
        .product_input_channel(engine_product_input_channel),
        .product_output_tile(engine_product_output_tile),
        .product_issue_seq(engine_product_issue_seq),
        .product_values(engine_product_values),
        .protocol_error(engine_protocol_error),
        .count_terms(engine_count_terms),
        .count_weight_requests(engine_count_weight_requests),
        .count_products(engine_count_products),
        .count_weight_wait_cycles(engine_count_weight_wait_cycles),
        .count_output_stall_cycles(engine_count_output_stall_cycles)
    );

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            term_active_q <= 1'b0;
            first_pending_q <= 1'b0;
            sequence_valid_q <= 1'b0;
            term_group_tag_q <= '0;
            first_sequence_q <= '0;
            next_sequence_q <= '0;
            term_issue_seq_q <= '0;
            term_input_channel_q <= '0;
            term_gate_code_q <= '0;
            term_lane_id_q <= '0;
            first_destination_valid_q <= '0;
            first_destination_tokens_q <= '0;
            destination_done_q <= '0;
            logical_supertile_q <= '0;
            physical_tile_q <= '0;
            epoch_q <= '0;
            term_epoch_q <= '0;
            stale_epoch_pending_q <= '0;
            current_weight_outstanding_q <= 1'b0;
            current_weight_tag_q <= '0;
            current_weight_channel_q <= '0;
            current_weight_tile_q <= '0;
            epoch_space_exhausted_q <= 1'b0;
            first_term_last_q <= 1'b0;
            first_head_last_q <= 1'b0;
            protocol_error <= 1'b0;
            count_stale_weight_responses <= '0;
        end else if (flush) begin
            term_active_q <= 1'b0;
            first_pending_q <= 1'b0;
            sequence_valid_q <= 1'b0;
            destination_done_q <= '0;
            stale_epoch_pending_q <= pending_after_flush_comb;
            current_weight_outstanding_q <= 1'b0;
            if (current_weight_outstanding_q) begin
                stale_epoch_tag_q[epoch_q] <= current_weight_tag_q;
                stale_epoch_channel_q[epoch_q] <= current_weight_channel_q;
                stale_epoch_tile_q[epoch_q] <= current_weight_tile_q;
            end
            if (clear_error)
                protocol_error <= 1'b0;
            if (next_epoch_found_comb) begin
                epoch_q <= next_epoch_comb;
                epoch_space_exhausted_q <= 1'b0;
            end else begin
                epoch_space_exhausted_q <= 1'b1;
                protocol_error <= 1'b1;
            end
        end else begin
            if (clear_error)
                protocol_error <= 1'b0;
            if (start_fire) begin
                term_active_q <= 1'b1;
                first_pending_q <= 1'b1;
                term_group_tag_q <= cmd_group_tag;
                first_sequence_q <= cmd_sequence;
                term_issue_seq_q <= cmd_term_issue_seq;
                term_input_channel_q <= cmd_input_channel;
                term_gate_code_q <= cmd_gate_code;
                term_lane_id_q <= cmd_lane_id;
                first_destination_valid_q <= cmd_destination_valid;
                first_destination_tokens_q <= cmd_destination_tokens;
                destination_done_q <= '0;
                logical_supertile_q <= logical_supertile;
                physical_tile_q <= physical_tile_comb;
                term_epoch_q <= epoch_q;
                first_term_last_q <= cmd_term_last;
                first_head_last_q <= cmd_head_last;
            end
            if (|acc_port_fire)
                destination_done_q <= destination_done_q | acc_port_fire;
            if (update_fire) begin
                sequence_valid_q <= 1'b1;
                next_sequence_q <= cmd_sequence + 1'b1;
                first_pending_q <= 1'b0;
                destination_done_q <= '0;
                if (cmd_term_last)
                    term_active_q <= 1'b0;
            end
            if (weight_request_fire) begin
                current_weight_outstanding_q <= 1'b1;
                current_weight_tag_q <= weight_req_tag;
                current_weight_channel_q <= weight_req_input_channel;
                current_weight_tile_q <= weight_req_output_tile;
            end
            if (current_weight_response_fire)
                current_weight_outstanding_q <= 1'b0;
            if (command_protocol_bad ||
                (engine_product_valid && !engine_product_identity_ok) ||
                (!clear_error && engine_protocol_error) ||
                unknown_stale_response_fire)
                protocol_error <= 1'b1;
            if (stale_weight_response_fire) begin
                count_stale_weight_responses <=
                    count_stale_weight_responses + 1'b1;
                if (stale_response_identity_matches) begin
                    stale_epoch_pending_q[weight_rsp_epoch] <= 1'b0;
                    if (epoch_space_exhausted_q) begin
                        epoch_q <= weight_rsp_epoch;
                        epoch_space_exhausted_q <= 1'b0;
                    end
                end
            end
        end
    end
endmodule

`default_nettype wire
