`timescale 1ns/1ps
`default_nettype none

// Passive simulation-only telemetry. This module has no outputs and cannot
// affect ready/valid, state, memory, or result signals in the observed DUT.
module local5_phase_semantic_monitor_v1 #(
    parameter int HEAD_W = 5,
    parameter int OUTPUT_TILE_W = 5,
    parameter int TOKEN_ID_W = 9,
    parameter int RESULT_ADDR_W = 14,
    parameter int OUT_W = 5,
    parameter int LANE_W = 5
) (
    input logic clk_core,
    input logic rst_core,

    input logic group_valid,
    input logic group_ready,
    input logic group_done_valid,
    input logic group_done_ready,
    input logic tile_start_valid,
    input logic tile_start_ready,
    input logic [1:0] tile_start_stage,
    input logic [2:0] tile_start_block,
    input logic [8:0] tile_start_window,
    input logic [OUTPUT_TILE_W-1:0] tile_start_output_tile,
    input logic tile_done_valid,
    input logic tile_done_ready,
    input logic head_job_valid,
    input logic head_job_ready,
    input logic [HEAD_W-1:0] head_job_input_head,
    input logic [OUTPUT_TILE_W-1:0] head_job_output_tile,
    input logic head_done_valid,
    input logic head_done_ready,
    input logic [HEAD_W-1:0] head_done_input_head,

    input logic token_req_valid,
    input logic token_req_ready,
    input logic [HEAD_W-1:0] token_req_input_head,
    input logic [TOKEN_ID_W-1:0] token_req_token_id,
    input logic token_rsp_valid,
    input logic token_rsp_ready,
    input logic [HEAD_W-1:0] token_rsp_input_head,
    input logic [TOKEN_ID_W-1:0] token_rsp_token_id,

    input logic weight_req_valid,
    input logic weight_req_ready,
    input logic [HEAD_W-1:0] weight_req_input_head,
    input logic [OUTPUT_TILE_W-1:0] weight_req_output_tile,
    input logic [LANE_W-1:0] weight_req_lane,
    input logic [OUT_W-1:0] weight_req_out,
    input logic weight_rsp_valid,
    input logic weight_rsp_ready,
    input logic [HEAD_W-1:0] weight_rsp_input_head,
    input logic [OUTPUT_TILE_W-1:0] weight_rsp_output_tile,
    input logic [LANE_W-1:0] weight_rsp_lane,
    input logic [OUT_W-1:0] weight_rsp_out,

    input logic tile_result_valid,
    input logic tile_result_ready,
    input logic [OUTPUT_TILE_W-1:0] tile_result_output_tile,
    input logic tile_result_plane,
    input logic [3:0] tile_result_y,
    input logic [3:0] tile_result_x,
    input logic [OUT_W-1:0] tile_result_out,

    input logic [3:0] tx_state,
    input logic acc_state,
    input logic [4:0] head_state,
    input logic [HEAD_W-1:0] active_head,
    input logic memory_command_valid,
    input logic memory_command_write,
    input logic [RESULT_ADDR_W-1:0] memory_command_addr,
    input logic tcfm_term_commit,
    input logic [4:0] tcfm_bank_update_mask,
    input logic tcfm_term_source_plane,
    input logic [3:0] tcfm_term_source_y,
    input logic [3:0] tcfm_term_source_x,
    input logic [LANE_W-1:0] tcfm_term_lane,
    input logic protocol_error,
    input logic scheduler_error
);
    integer telemetry_fd;
    integer cycle_q;
    integer phase_sequence_q;
    integer resource_sequence_q;
    integer active_head_phase_q;
    integer active_head_phase_start_q;
    integer active_head_phase_tile_q;
    integer active_head_phase_head_q;
    integer active_head_transaction_start_q;
    integer active_head_transaction_tile_q;
    integer active_head_transaction_head_q;
    integer active_tile_phase_start_q;
    integer active_tile_q;
    integer active_group_phase_start_q;
    integer active_drain_phase_start_q;
    integer active_drain_tile_q;
    integer expected_stage_q;
    integer expected_block_q;
    integer expected_window_q;
    integer observed_stage_q;
    integer observed_block_q;
    integer observed_window_q;
    integer source_id;
    string telemetry_path;
    bit telemetry_closed_q;

    function automatic integer classify_head_phase(input logic [4:0] state);
        begin
            case (state)
                5'd1, 5'd2: classify_head_phase = 1;
                5'd3, 5'd4, 5'd5, 5'd6, 5'd7, 5'd8, 5'd9:
                    classify_head_phase = 2;
                5'd10, 5'd11, 5'd12: classify_head_phase = 3;
                5'd13, 5'd14: classify_head_phase = 4;
                5'd15: classify_head_phase = 5;
                default: classify_head_phase = 0;
            endcase
        end
    endfunction

    function automatic string head_phase_name(input integer phase);
        begin
            case (phase)
                1: head_phase_name = "HEAD_WEIGHT";
                2: head_phase_name = "HEAD_FRONTEND";
                3: head_phase_name = "HEAD_READOUT";
                4: head_phase_name = "HEAD_RELEASE";
                5: head_phase_name = "HEAD_ERROR";
                default: head_phase_name = "INVALID";
            endcase
        end
    endfunction

    task automatic write_phase(
        input integer tile,
        input integer head,
        input string role,
        input integer start_cycle,
        input integer end_cycle
    );
        begin
            if (end_cycle < start_cycle)
                $fatal(1, "telemetry phase has negative duration");
            $fwrite(telemetry_fd,
                "P,%0d,%0d,%0d,%0d,%0d,%0d,%s,%0d,%0d,%0d,RTL_DIRECT\n",
                phase_sequence_q, observed_stage_q, observed_block_q,
                observed_window_q, tile, head, role, start_cycle, end_cycle,
                end_cycle - start_cycle + 1);
            phase_sequence_q = phase_sequence_q + 1;
        end
    endtask

    task automatic write_resource(
        input integer tile,
        input integer head,
        input string resource,
        input integer identity0,
        input integer identity1,
        input integer identity2
    );
        begin
            $fwrite(telemetry_fd,
                "R,%0d,%0d,%0d,%0d,%0d,%0d,%s,%0d,%0d,%0d,%0d,RTL_DIRECT\n",
                resource_sequence_q, observed_stage_q, observed_block_q,
                observed_window_q, tile, head, resource, cycle_q,
                identity0, identity1, identity2);
            resource_sequence_q = resource_sequence_q + 1;
        end
    endtask

    initial begin
        telemetry_fd = 0;
        telemetry_closed_q = 1'b0;
        if (!$value$plusargs("PHASE_TELEMETRY=%s", telemetry_path)
            || !$value$plusargs("TELEMETRY_STAGE=%d", expected_stage_q)
            || !$value$plusargs("TELEMETRY_BLOCK=%d", expected_block_q)
            || !$value$plusargs("TELEMETRY_WINDOW=%d", expected_window_q))
            $fatal(1, "phase telemetry plusargs are mandatory");
        telemetry_fd = $fopen(telemetry_path, "w");
        if (telemetry_fd == 0)
            $fatal(1, "cannot open phase telemetry output");
        $fwrite(telemetry_fd, "SCHEMA,local5_phase_semantic_telemetry_v1\n");
        $fwrite(telemetry_fd, "ORIGIN,RTL_DIRECT\n");
        $fwrite(telemetry_fd,
            "COLUMNS_P,record,sequence,stage,block,window,tile,head,role,start_cycle,end_cycle,duration,origin\n");
        $fwrite(telemetry_fd,
            "COLUMNS_R,record,sequence,stage,block,window,tile,head,resource,cycle,identity0,identity1,identity2,origin\n");
    end

    always @(posedge clk_core) begin : p_passive_telemetry
        integer observed_head_phase;
        if (rst_core) begin
            cycle_q = 0;
            phase_sequence_q = 0;
            resource_sequence_q = 0;
            active_head_phase_q = 0;
            active_head_phase_start_q = -1;
            active_head_phase_tile_q = -1;
            active_head_phase_head_q = -1;
            active_head_transaction_start_q = -1;
            active_head_transaction_tile_q = -1;
            active_head_transaction_head_q = -1;
            active_tile_phase_start_q = -1;
            active_tile_q = -1;
            active_group_phase_start_q = -1;
            active_drain_phase_start_q = -1;
            active_drain_tile_q = -1;
            observed_stage_q = -1;
            observed_block_q = -1;
            observed_window_q = -1;
        end else if (!telemetry_closed_q) begin
            if (protocol_error || scheduler_error)
                $fatal(1, "observed protocol error during telemetry pilot");

            if (group_valid && group_ready)
                active_group_phase_start_q = cycle_q;

            if (tile_start_valid && tile_start_ready) begin
                if (32'(tile_start_stage) != expected_stage_q
                    || 32'(tile_start_block) != expected_block_q
                    || 32'(tile_start_window) != expected_window_q)
                    $fatal(1, "telemetry identity does not match observed tile start");
                observed_stage_q = 32'(tile_start_stage);
                observed_block_q = 32'(tile_start_block);
                observed_window_q = 32'(tile_start_window);
                active_tile_phase_start_q = cycle_q;
                active_tile_q = 32'(tile_start_output_tile);
            end

            observed_head_phase = classify_head_phase(head_state);

            if (head_job_valid && head_job_ready) begin
                if (active_head_transaction_start_q >= 0)
                    $fatal(1, "overlapping head transaction in telemetry");
                active_head_transaction_start_q = cycle_q;
                active_head_transaction_tile_q = 32'(head_job_output_tile);
                active_head_transaction_head_q = 32'(head_job_input_head);
            end

            if (observed_head_phase != active_head_phase_q) begin
                if (active_head_phase_q != 0)
                    write_phase(
                        active_head_phase_tile_q, active_head_phase_head_q,
                        head_phase_name(active_head_phase_q),
                        active_head_phase_start_q, cycle_q - 1
                    );
                active_head_phase_q = observed_head_phase;
                if (observed_head_phase != 0) begin
                    active_head_phase_start_q = cycle_q;
                    active_head_phase_tile_q = active_tile_q;
                    active_head_phase_head_q = 32'(active_head);
                end
            end

            if (head_done_valid && head_done_ready) begin
                if (active_head_transaction_start_q < 0
                    || active_head_transaction_head_q
                       != 32'(head_done_input_head))
                    $fatal(1, "head done does not match telemetry head start");
                write_phase(active_head_transaction_tile_q,
                    active_head_transaction_head_q, "HEAD_TRANSACTION",
                    active_head_transaction_start_q, cycle_q);
                active_head_transaction_start_q = -1;
            end

            if (tx_state >= 4 && tx_state <= 6
                && active_drain_phase_start_q < 0) begin
                active_drain_phase_start_q = cycle_q;
                active_drain_tile_q = active_tile_q;
            end else if (!(tx_state >= 4 && tx_state <= 6)
                         && active_drain_phase_start_q >= 0) begin
                write_phase(active_drain_tile_q, -1, "TILE_DRAIN",
                    active_drain_phase_start_q, cycle_q - 1);
                active_drain_phase_start_q = -1;
            end

            if (token_req_valid && token_req_ready)
                write_resource(active_tile_q, 32'(token_req_input_head),
                    "RELATION_REQ_ACCEPT", 32'(token_req_token_id), 0, 0);
            if (token_rsp_valid && token_rsp_ready)
                write_resource(active_tile_q, 32'(token_rsp_input_head),
                    "RELATION_RSP_ACCEPT", 32'(token_rsp_token_id), 0, 0);
            if (weight_req_valid && weight_req_ready)
                write_resource(32'(weight_req_output_tile),
                    32'(weight_req_input_head), "WEIGHT_REQ_ACCEPT",
                    32'(weight_req_lane), 32'(weight_req_out), 0);
            if (weight_rsp_valid && weight_rsp_ready)
                write_resource(32'(weight_rsp_output_tile),
                    32'(weight_rsp_input_head), "WEIGHT_RSP_ACCEPT",
                    32'(weight_rsp_lane), 32'(weight_rsp_out), 0);
            if (tile_result_valid && tile_result_ready) begin
                source_id = (32'(tile_result_plane) * 225)
                          + (32'(tile_result_y) * 15) + 32'(tile_result_x);
                write_resource(32'(tile_result_output_tile), -1,
                    "FINAL_ACCEPT", source_id, 32'(tile_result_out), 0);
            end
            if (memory_command_valid)
                write_resource(active_tile_q, -1, "CROSS_ACC_CMD",
                    32'(memory_command_addr), 32'(memory_command_write),
                    32'(acc_state));
            if (tcfm_term_commit) begin
                source_id = (32'(tcfm_term_source_plane) * 225)
                          + (32'(tcfm_term_source_y) * 15)
                          + 32'(tcfm_term_source_x);
                write_resource(active_tile_q, 32'(active_head),
                    "TCFM5_BANK_UPDATE_MASK", source_id,
                    32'(tcfm_term_lane), 32'(tcfm_bank_update_mask));
            end

            if (tile_done_valid && tile_done_ready) begin
                if (active_drain_phase_start_q >= 0) begin
                    write_phase(active_drain_tile_q, -1, "TILE_DRAIN",
                        active_drain_phase_start_q, cycle_q);
                    active_drain_phase_start_q = -1;
                end
                if (active_tile_phase_start_q < 0)
                    $fatal(1, "tile done without telemetry tile start");
                write_phase(active_tile_q, -1, "TILE_TRANSACTION",
                    active_tile_phase_start_q, cycle_q);
                active_tile_phase_start_q = -1;
            end

            if (group_done_valid && group_done_ready) begin
                if (active_head_phase_q != 0)
                    $fatal(1, "group done with an open head phase");
                if (active_group_phase_start_q < 0)
                    $fatal(1, "group done without telemetry group start");
                write_phase(-1, -1, "GROUP_TRANSACTION",
                    active_group_phase_start_q, cycle_q);
                $fwrite(telemetry_fd,
                    "END,%0d,%0d,%0d,RTL_DIRECT\n",
                    cycle_q, phase_sequence_q, resource_sequence_q);
                $fclose(telemetry_fd);
                telemetry_fd = 0;
                telemetry_closed_q = 1'b1;
            end
            cycle_q = cycle_q + 1;
        end
    end
endmodule

`default_nettype wire
