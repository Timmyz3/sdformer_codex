`timescale 1ns/1ps
`default_nettype none

module m241r2_elastic_tagged_checkpoint_no_forward_accumulator_assertions #(
    parameter int LANES = 8,
    parameter int ROWS = 384,
    parameter int ACC_BITS = 19,
    parameter int SEQUENCE_BITS = 32,
    parameter int PARTITION_BITS = 9,
    parameter int WINDOW_BITS = 16,
    parameter int EPOCH_BITS = 16,
    parameter int PAYLOAD_BITS = 32,
    parameter int ORDER_BITS = 16,
    parameter int ROW_BITS = $clog2(ROWS),
    parameter int ACC_ADDR_BITS = $clog2(2 * ROWS)
) (
    input logic                         clk_core,
    input logic                         rst_core,
    input logic                         loader_binding_valid,
    input logic                         context_open_valid,
    input logic                         context_open_ready,
    input logic                         context_open_accept,
    input logic                         descriptor_valid,
    input logic                         descriptor_ready,
    input logic [WINDOW_BITS-1:0]       descriptor_window,
    input logic [ORDER_BITS-1:0]        descriptor_order,
    input logic [3:0]                   descriptor_destination_valid,
    input logic [3:0]                   descriptor_negate,
    input logic                         descriptor_accept,
    input logic                         context_close_valid,
    input logic                         context_close_ready,
    input logic                         context_close_accept,
    input logic                         window_done,
    input logic                         weight_req_valid,
    input logic                         weight_req_ready,
    input logic                         weight_req_accept,
    input logic [3:0]                   weight_req_bank_valid,
    input logic [4:0]                   weight_req_addr [0:3],
    input logic [SEQUENCE_BITS-1:0]     weight_req_sequence,
    input logic [1:0]                   weight_req_operator,
    input logic [PARTITION_BITS-1:0]    weight_req_partition,
    input logic [WINDOW_BITS-1:0]       weight_req_window,
    input logic [EPOCH_BITS-1:0]        weight_req_weight_epoch,
    input logic [PAYLOAD_BITS-1:0]      weight_req_payload_id,
    input logic [ORDER_BITS-1:0]        weight_req_order,
    input logic [3:0]                   weight_req_source,
    input logic                         weight_req_half,
    input logic                         weight_rsp_valid,
    input logic                         weight_rsp_ready,
    input logic                         weight_rsp_accept,
    input logic [3:0]                   weight_rsp_bank_valid,
    input logic signed [7:0]            weight_rsp_data
                                            [0:3][0:LANES-1],
    input logic [3:0]                   weight_cache_hit,
    input logic [3:0]                   weight_cache_miss,
    input logic                         acc_req_valid,
    input logic                         acc_req_ready,
    input logic                         acc_req_accept,
    input logic [3:0]                   acc_req_bank_valid,
    input logic [ACC_ADDR_BITS-1:0]     acc_req_addr [0:3],
    input logic [SEQUENCE_BITS-1:0]     acc_req_sequence,
    input logic [WINDOW_BITS-1:0]       acc_req_window,
    input logic [EPOCH_BITS-1:0]        acc_req_weight_epoch,
    input logic [PAYLOAD_BITS-1:0]      acc_req_payload_id,
    input logic [ORDER_BITS-1:0]        acc_req_order,
    input logic                         acc_rsp_valid,
    input logic                         acc_rsp_ready,
    input logic                         acc_rsp_accept,
    input logic [3:0]                   acc_rsp_bank_valid,
    input logic signed [ACC_BITS-1:0]   acc_rsp_data
                                            [0:3][0:LANES-1],
    input logic [3:0]                   acc_wr_en,
    input logic [ACC_ADDR_BITS-1:0]     acc_wr_addr [0:3],
    input logic signed [ACC_BITS-1:0]   acc_wr_data [0:3][0:LANES-1],
    input logic                         commit_valid,
    input logic                         commit_ready,
    input logic                         commit_accept,
    input logic [ORDER_BITS-1:0]        commit_order,
    input logic [WINDOW_BITS-1:0]       commit_window,
    input logic [ROW_BITS-1:0]          commit_row,
    input logic [3:0]                   commit_bank_valid,
    input logic [2:0]                   commit_destination [0:3],
    input logic                         commit_last,
    input logic                         abort_valid,
    input logic                         abort_ready,
    input logic                         abort_accept,
    input logic [ORDER_BITS-1:0]        abort_order,
    input logic [WINDOW_BITS-1:0]       abort_window,
    input logic [1:0]                   abort_discarded_tokens,
    input logic                         context_abort,
    input logic                         rmw_alias_stall,
    input logic [ORDER_BITS-1:0]        next_descriptor_order,
    input logic                         context_active,
    input logic                         protocol_error,
    input logic                         overflow_error,
    input logic                         busy
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_open_accept_definition:
        assert property (context_open_accept
                         == (context_open_valid && context_open_ready));
    ap_descriptor_accept_definition:
        assert property (descriptor_accept
                         == (descriptor_valid && descriptor_ready));
    ap_close_accept_definition:
        assert property (context_close_accept
                         == (context_close_valid && context_close_ready));
    ap_weight_req_accept_definition:
        assert property (weight_req_accept
                         == (weight_req_valid && weight_req_ready));
    ap_weight_rsp_accept_definition:
        assert property (weight_rsp_accept
                         == (weight_rsp_valid && weight_rsp_ready));
    ap_acc_req_accept_definition:
        assert property (acc_req_accept
                         == (acc_req_valid && acc_req_ready));
    ap_acc_rsp_accept_definition:
        assert property (acc_rsp_accept
                         == (acc_rsp_valid && acc_rsp_ready));
    ap_commit_accept_definition:
        assert property (commit_accept
                         == (commit_valid && commit_ready));
    ap_abort_accept_definition:
        assert property (abort_accept == (abort_valid && abort_ready));
    ap_exact_descriptor_order:
        assert property (descriptor_accept
                         |-> descriptor_order == next_descriptor_order);
    ap_context_requires_loader_binding:
        assert property (context_open_accept |-> loader_binding_valid);
    ap_cache_partition:
        assert property (descriptor_accept
            |-> $countones(weight_cache_hit) + $countones(weight_cache_miss)
                == $countones(descriptor_destination_valid));

    ap_weight_request_stable_under_stall:
        assert property (weight_req_valid && !weight_req_ready
            |=> weight_req_valid
                && $stable({weight_req_bank_valid, weight_req_sequence,
                            weight_req_operator, weight_req_partition,
                            weight_req_window, weight_req_weight_epoch,
                            weight_req_payload_id, weight_req_order,
                            weight_req_source, weight_req_half,
                            weight_req_addr[0], weight_req_addr[1],
                            weight_req_addr[2], weight_req_addr[3]}));
    ap_acc_request_stable_under_stall:
        assert property (acc_req_valid && !acc_req_ready
            |=> acc_req_valid
                && $stable({acc_req_bank_valid, acc_req_sequence,
                            acc_req_window, acc_req_weight_epoch,
                            acc_req_payload_id, acc_req_order,
                            acc_req_addr[0], acc_req_addr[1],
                            acc_req_addr[2], acc_req_addr[3]}));
    ap_weight_response_stable_under_stall:
        assert property (weight_rsp_valid && !weight_rsp_ready
            |=> weight_rsp_valid && $stable(weight_rsp_bank_valid));
    ap_acc_response_stable_under_stall:
        assert property (acc_rsp_valid && !acc_rsp_ready
            |=> acc_rsp_valid && $stable(acc_rsp_bank_valid));
    ap_stale_weight_response_not_consumed:
        assert property (weight_rsp_valid && protocol_error
                         |-> !weight_rsp_accept);
    ap_stale_acc_response_not_consumed:
        assert property (acc_rsp_valid && protocol_error
                         |-> !acc_rsp_accept);
    ap_weight_response_accept_has_request:
        assert property (weight_rsp_accept |-> busy && context_active);
    ap_acc_response_accept_has_request:
        assert property (acc_rsp_accept |-> busy && context_active);

    ap_commit_abort_mutually_exclusive:
        assert property (!(commit_valid && abort_valid));
    ap_overflow_never_successfully_commits:
        assert property (overflow_error
                         |-> !commit_valid && !commit_accept);
    ap_overflow_suppresses_all_writes:
        assert property (overflow_error |-> acc_wr_en == 4'b0000);
    ap_abort_has_no_writes:
        assert property (abort_valid |-> acc_wr_en == 4'b0000);
    ap_clean_commit_writes_valid_banks:
        assert property (commit_accept |-> acc_wr_en == commit_bank_valid);
    ap_no_write_without_successful_commit:
        assert property ((|acc_wr_en) |-> commit_accept);
    ap_context_abort_follows_abort_accept:
        assert property (abort_accept |=> context_abort);
    ap_protocol_fault_sticky:
        assert property (protocol_error |=> protocol_error);
    ap_overflow_fault_sticky:
        assert property (overflow_error |=> overflow_error);
    ap_done_follows_close:
        assert property (context_close_accept |=> window_done);
    ap_context_active_is_busy:
        assert property (context_active |-> busy);
    ap_reset_flushes_external_success:
        assert property (rst_core |=> !commit_valid && !abort_valid
                         && acc_wr_en == 4'b0000);

    ap_commit_stable_under_stall:
        assert property (commit_valid && !commit_ready
            |=> commit_valid
                && $stable({commit_order, commit_window, commit_row,
                            commit_bank_valid, commit_destination[0],
                            commit_destination[1], commit_destination[2],
                            commit_destination[3], commit_last}));
    ap_abort_stable_under_stall:
        assert property (abort_valid && !abort_ready
            |=> abort_valid
                && $stable({abort_order, abort_window,
                            abort_discarded_tokens}));

    generate
        for (genvar bank = 0; bank < 4; bank++) begin : g_bank
            ap_dense_write_address:
                assert property (acc_wr_en[bank]
                    |-> acc_wr_addr[bank]
                        == ((commit_destination[bank][2] ? ROWS : 0)
                            + commit_row));
            for (genvar lane = 0; lane < LANES; lane++) begin : g_lane
                ap_weight_response_data_stable_under_stall:
                    assert property (weight_rsp_valid && !weight_rsp_ready
                                     |=> $stable(
                                         weight_rsp_data[bank][lane]));
                ap_acc_response_data_stable_under_stall:
                    assert property (acc_rsp_valid && !acc_rsp_ready
                                     |=> $stable(
                                         acc_rsp_data[bank][lane]));
                ap_write_data_stable_under_commit_stall:
                    assert property (commit_valid && !commit_ready
                                     |=> $stable(acc_wr_data[bank][lane]));
            end
        end
    endgenerate

    cp_weight_request_stall:
        cover property (weight_req_valid && !weight_req_ready
                        ##1 weight_req_accept);
    cp_weight_response_stall:
        cover property (weight_rsp_valid && !weight_rsp_ready
                        ##1 weight_rsp_accept);
    cp_acc_request_stall:
        cover property (acc_req_valid && !acc_req_ready ##1 acc_req_accept);
    cp_acc_response_stall:
        cover property (acc_rsp_valid && !acc_rsp_ready ##1 acc_rsp_accept);
    cp_cache_reuse:
        cover property (descriptor_accept && |weight_cache_hit);
    cp_full4:
        cover property (descriptor_accept
                        && descriptor_destination_valid == 4'b1111);
    cp_negate:
        cover property (descriptor_accept && |descriptor_negate);
    cp_commit_stall:
        cover property (commit_valid && !commit_ready
                        ##1 commit_accept);
    cp_alias_interlock:
        cover property (rmw_alias_stall);
    cp_stale_weight_response:
        cover property (weight_rsp_valid && protocol_error
                        && !weight_rsp_accept);
    cp_stale_acc_response:
        cover property (acc_rsp_valid && protocol_error
                        && !acc_rsp_accept);
    cp_overflow_abort_with_two_younger:
        cover property (abort_valid && abort_discarded_tokens == 2
                        && !commit_accept && acc_wr_en == 4'b0000);
    cp_abort_stall:
        cover property (abort_valid && !abort_ready ##1 abort_accept);
    cp_context_abort:
        cover property (context_abort);
    cp_window_done:
        cover property (window_done);
endmodule

`default_nettype wire
