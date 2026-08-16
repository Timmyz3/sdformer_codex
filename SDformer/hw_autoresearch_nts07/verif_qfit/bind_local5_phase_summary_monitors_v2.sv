`timescale 1ns/1ps
`default_nettype none

bind tb_qfit_local5_memo_multitile_cross_head
    local5_phase_summary_monitor_v2 #(
        .H(HEADS)
    ) u_local5_phase_summary_monitor_v2 (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .group_valid(group_valid),
        .group_ready(group_ready),
        .group_done_valid(group_done_valid),
        .group_done_ready(group_done_ready),
        .tile_start_valid(tile_start_valid),
        .tile_start_ready(tile_start_ready),
        .tile_start_stage(u_executor.tile_start_stage),
        .tile_start_block(u_executor.tile_start_block),
        .tile_start_window(u_executor.tile_start_window),
        .tile_start_output_tile(tile_start_output_tile),
        .tile_done_valid(tile_done_valid),
        .tile_done_ready(tile_done_ready),
        .head_job_valid(head_job_valid),
        .head_job_ready(head_job_ready),
        .head_job_input_head(head_job_input_head),
        .head_job_output_tile(head_job_output_tile),
        .head_done_valid(head_done_valid),
        .head_done_ready(head_done_ready),
        .head_done_input_head(head_done_input_head),
        .token_req_valid(token_req_valid),
        .token_req_ready(token_req_ready),
        .token_req_input_head(token_req_input_head),
        .token_req_token_id(token_req_token_id),
        .token_rsp_valid(token_rsp_valid),
        .token_rsp_ready(token_rsp_ready),
        .token_rsp_input_head(token_rsp_input_head),
        .token_rsp_token_id(token_rsp_token_id),
        .weight_req_valid(weight_req_valid),
        .weight_req_ready(weight_req_ready),
        .weight_req_input_head(weight_req_input_head),
        .weight_req_output_tile(weight_req_output_tile),
        .weight_req_lane(weight_req_lane),
        .weight_req_out(weight_req_out),
        .weight_rsp_valid(dut_weight_rsp_valid),
        .weight_rsp_ready(dut_weight_rsp_ready),
        .weight_rsp_input_head(weight_rsp_input_head),
        .weight_rsp_output_tile(weight_rsp_output_tile),
        .weight_rsp_lane(weight_rsp_lane),
        .weight_rsp_out(weight_rsp_out),
        .tile_result_valid(tile_result_valid),
        .tile_result_ready(tile_result_ready),
        .tile_result_output_tile(tile_result_output_tile),
        .tile_result_plane(tile_result_plane),
        .tile_result_y(tile_result_y),
        .tile_result_x(tile_result_x),
        .tile_result_out(tile_result_out),
        .tx_state(u_executor.tx_state_q),
        .head_state(
            u_executor.g_baseline_head_engine.u_head_engine.state_q
        ),
        .active_head(u_executor.active_head_q),
        .memory_command_valid(u_executor.memory_command_valid),
        .memory_command_write(u_executor.memory_command_write),
        .memory_command_addr(u_executor.memory_command_addr),
        .memory_command_write_data(u_executor.memory_command_write_data),
        .tcfm_term_commit(
            u_executor.g_baseline_head_engine.u_head_engine.u_tile.term_valid
            && u_executor.g_baseline_head_engine.u_head_engine.u_tile.term_ready
        ),
        .tcfm_term_source_plane(
            u_executor.g_baseline_head_engine.u_head_engine.u_tile
                .term_source_plane
        ),
        .tcfm_term_source_y(
            u_executor.g_baseline_head_engine.u_head_engine.u_tile.term_source_y
        ),
        .tcfm_term_source_x(
            u_executor.g_baseline_head_engine.u_head_engine.u_tile.term_source_x
        ),
        .tcfm_term_lane(
            u_executor.g_baseline_head_engine.u_head_engine.u_tile.term_lane
        ),
        .tcfm_term_destination_mask(
            u_executor.g_baseline_head_engine.u_head_engine.u_tile
                .term_destination_mask
        ),
        .protocol_error(protocol_error),
        .scheduler_error(scheduler_error)
    );

bind qfit_single_port_acc_memory
    local5_cross_acc_summary_monitor_v2 #(
        .DEPTH(DEPTH),
        .VEC_W(VEC_W),
        .ADDR_W(ADDR_W)
    ) u_local5_cross_acc_summary_monitor_v2 (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .command_valid(command_valid),
        .command_write(command_write),
        .command_addr(command_addr),
        .command_write_data(command_write_data)
    );

bind qfit_tcfm5_projection_top
    local5_tcfm5_summary_monitor_v2 #(
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .HEAD_DIM(HEAD_DIM),
        .LANE_W(LANE_W),
        .Y_W(Y_W),
        .X_W(X_W),
        .PLANE_W(PLANE_W),
        .BANK_ADDR_W(BANK_ADDR_W)
    ) u_local5_tcfm5_summary_monitor_v2 (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .term_commit(term_commit),
        .term_source_plane(term_source_plane),
        .term_source_y(term_source_y),
        .term_source_x(term_source_x),
        .term_lane(term_lane),
        .term_destination_mask(term_destination_mask),
        .actual_bank_mask({
            bank_write_enable[4], bank_write_enable[3],
            bank_write_enable[2], bank_write_enable[1],
            bank_write_enable[0]
        }),
        .actual_bank_addr_flat({
            bank_write_addr[4], bank_write_addr[3], bank_write_addr[2],
            bank_write_addr[1], bank_write_addr[0]
        })
    );

`default_nettype wire
