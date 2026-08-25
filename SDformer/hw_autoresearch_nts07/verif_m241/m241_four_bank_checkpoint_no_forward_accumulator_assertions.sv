`timescale 1ns/1ps
`default_nettype none

module m241_four_bank_checkpoint_no_forward_accumulator_assertions #(
    parameter int LANES = 8,
    parameter int ROWS = 384,
    parameter int ACC_BITS = 19,
    parameter int ORDER_BITS = 16,
    parameter int ROW_BITS = $clog2(ROWS),
    parameter int ACC_ADDR_BITS = $clog2(2 * ROWS)
) (
    input logic                         clk_core,
    input logic                         rst_core,
    input logic                         context_open_valid,
    input logic                         context_open_ready,
    input logic                         context_open_accept,
    input logic                         descriptor_valid,
    input logic                         descriptor_ready,
    input logic [ORDER_BITS-1:0]        descriptor_order,
    input logic [ROW_BITS-1:0]          descriptor_row,
    input logic [3:0]                   descriptor_destination_valid,
    input logic [3:0]                   descriptor_negate,
    input logic                         descriptor_last,
    input logic                         descriptor_accept,
    input logic                         context_close_valid,
    input logic                         context_close_ready,
    input logic                         context_close_accept,
    input logic                         window_done,
    input logic [3:0]                   weight_rd_en,
    input logic [3:0]                   weight_cache_hit,
    input logic [3:0]                   weight_cache_miss,
    input logic [3:0]                   acc_rd_en,
    input logic [ACC_ADDR_BITS-1:0]     acc_rd_addr [0:3],
    input logic [3:0]                   acc_wr_en,
    input logic [ACC_ADDR_BITS-1:0]     acc_wr_addr [0:3],
    input logic signed [ACC_BITS-1:0]   acc_wr_data [0:3][0:LANES-1],
    input logic                         commit_valid,
    input logic                         commit_ready,
    input logic                         commit_accept,
    input logic [ORDER_BITS-1:0]        commit_order,
    input logic [ROW_BITS-1:0]          commit_row,
    input logic [3:0]                   commit_bank_valid,
    input logic [2:0]                   commit_destination [0:3],
    input logic                         commit_last,
    input logic                         rmw_alias_stall,
    input logic [ORDER_BITS-1:0]        next_descriptor_order,
    input logic                         context_active,
    input logic                         protocol_error,
    input logic                         overflow_error,
    input logic                         busy
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_context_open_accept_definition:
        assert property (context_open_accept
                         == (context_open_valid && context_open_ready));
    ap_descriptor_accept_definition:
        assert property (descriptor_accept
                         == (descriptor_valid && descriptor_ready));
    ap_context_close_accept_definition:
        assert property (context_close_accept
                         == (context_close_valid && context_close_ready));
    ap_commit_accept_definition:
        assert property (commit_accept == (commit_valid && commit_ready));
    ap_accepted_order_is_exact:
        assert property (descriptor_accept
                         |-> descriptor_order == next_descriptor_order);
    ap_weight_accounting:
        assert property (descriptor_accept
            |-> $countones(weight_rd_en) + $countones(weight_cache_hit)
                == $countones(descriptor_destination_valid));
    ap_weight_hit_miss_partition:
        assert property (descriptor_accept
            |-> (weight_cache_hit ^ weight_cache_miss)
                == (weight_cache_hit | weight_cache_miss));
    ap_weight_read_is_miss:
        assert property (weight_rd_en == weight_cache_miss);
    ap_no_weight_activity_without_accept:
        assert property ((|weight_rd_en) || (|weight_cache_hit)
                         || (|weight_cache_miss) |-> descriptor_accept);
    ap_alias_stall_suppresses_acc_read:
        assert property (rmw_alias_stall |-> acc_rd_en == 4'b0000);
    ap_no_acc_read_without_busy:
        assert property ((|acc_rd_en) |-> busy && context_active);
    ap_no_write_without_commit:
        assert property ((|acc_wr_en) |-> commit_accept);
    ap_overflow_suppresses_all_writes:
        assert property (overflow_error |-> acc_wr_en == 4'b0000);
    ap_clean_commit_writes_all_valid_banks:
        assert property (commit_accept && !overflow_error
                         |-> acc_wr_en == commit_bank_valid);
    ap_protocol_fault_is_sticky:
        assert property (protocol_error |=> protocol_error);
    ap_overflow_fault_is_sticky:
        assert property (overflow_error |=> overflow_error);
    ap_done_follows_close:
        assert property (context_close_accept |=> window_done);
    ap_busy_when_context_active:
        assert property (context_active |-> busy);
    ap_commit_metadata_stable_under_stall:
        assert property (commit_valid && !commit_ready
            |=> commit_valid
                && $stable({commit_order, commit_row, commit_bank_valid,
                            commit_destination[0], commit_destination[1],
                            commit_destination[2], commit_destination[3],
                            commit_last}));
    ap_reset_flushes_commit:
        assert property (rst_core |=> !commit_valid && acc_wr_en == 4'b0000);

    generate
        for (genvar bank = 0; bank < 4; bank++) begin : g_bank
            ap_dense_write_address_matches_metadata:
                assert property (acc_wr_en[bank]
                    |-> acc_wr_addr[bank]
                        == ((commit_destination[bank][2] ? ROWS : 0)
                            + commit_row));
            for (genvar lane = 0; lane < LANES; lane++) begin : g_lane
                ap_write_data_stable_under_commit_stall:
                    assert property (commit_valid && !commit_ready
                                     |=> $stable(acc_wr_data[bank][lane]));
            end
        end
    endgenerate

    cp_all_four_weight_banks:
        cover property (descriptor_accept && weight_rd_en == 4'b1111);
    cp_weight_cache_reuse:
        cover property (descriptor_accept && |weight_cache_hit);
    cp_full4_descriptor:
        cover property (descriptor_accept
                        && descriptor_destination_valid == 4'b1111);
    cp_tail_descriptor:
        cover property (descriptor_accept
                        && descriptor_destination_valid != 4'b1111);
    cp_negated_descriptor:
        cover property (descriptor_accept && |descriptor_negate);
    cp_all_four_accumulator_writes:
        cover property (acc_wr_en == 4'b1111);
    cp_commit_stall:
        cover property (commit_valid && !commit_ready
                        ##1 commit_valid && commit_ready);
    cp_rmw_alias_interlock:
        cover property (rmw_alias_stall);
    cp_protocol_fault_with_older_commit:
        cover property (protocol_error && commit_valid);
    cp_overflow_fault:
        cover property (overflow_error);
    cp_window_done:
        cover property (window_done);
endmodule

`default_nettype wire
