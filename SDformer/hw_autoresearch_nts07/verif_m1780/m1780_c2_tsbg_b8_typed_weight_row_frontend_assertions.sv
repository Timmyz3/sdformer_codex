`timescale 1ns/1ps
`default_nettype none

module m1780_c2_tsbg_b8_typed_weight_row_frontend_assertions #(
    parameter int BANKS = 8,
    parameter int LANES = 16,
    parameter int TAG_BITS = 24,
    parameter int GROUP_BITS = 6
) (
    input logic clk_core,
    input logic rst_core,
    input logic load_valid,
    input logic load_ready,
    input logic [2:0] load_context,
    input logic [GROUP_BITS-1:0] load_group,
    input logic signed [7:0] load_source_value [0:15],
    input logic load_last,
    input logic load_accept,
    input logic mem_req_valid,
    input logic mem_req_ready,
    input logic [GROUP_BITS-1:0] mem_req_group,
    input logic mem_req_half,
    input logic [2:0] mem_req_slice,
    input logic mem_rsp_valid,
    input logic mem_rsp_ready,
    input logic issue_valid,
    input logic issue_ready,
    input logic [2:0] issue_context,
    input logic [GROUP_BITS-1:0] issue_group,
    input logic issue_half,
    input logic [2:0] issue_slice,
    input logic [BANKS-1:0] issue_bank_valid,
    input logic signed [7:0] issue_source_value [0:BANKS-1],
    input logic signed [7:0] issue_weight [0:BANKS-1][0:LANES-1],
    input logic commit_valid,
    input logic commit_ready,
    input logic [2:0] commit_context,
    input logic [TAG_BITS-1:0] commit_tag,
    input logic [2:0] commit_slice,
    input logic signed [23:0] commit_accumulator [0:LANES-1],
    input logic commit_terminal,
    input logic protocol_error
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_load_accept_definition: assert property (
        load_accept |-> load_valid && load_ready);
    ap_mem_req_stable: assert property (
        mem_req_valid && !mem_req_ready |=> mem_req_valid
            && $stable({mem_req_group, mem_req_half, mem_req_slice}));
    ap_issue_header_stable: assert property (
        issue_valid && !issue_ready |=> issue_valid
            && $stable({issue_context, issue_group, issue_half,
                        issue_slice, issue_bank_valid}));
    ap_issue_payload_stable: assert property (
        issue_valid && !issue_ready |=>
            $stable(issue_source_value) && $stable(issue_weight));
    ap_commit_header_stable: assert property (
        commit_valid && !commit_ready |=> commit_valid
            && $stable({commit_context, commit_tag, commit_slice,
                        commit_terminal}));
    ap_commit_payload_stable: assert property (
        commit_valid && !commit_ready |=> $stable(commit_accumulator));
    ap_fault_closes_load: assert property (protocol_error |-> !load_ready);
    ap_issue_nonempty: assert property (issue_valid |-> issue_bank_valid != 0);
    ap_terminal_only_last_slice: assert property (
        commit_valid && commit_terminal |-> commit_slice == 3'd5);

    generate
        for (genvar source = 0; source < 16; source++) begin : g_load_value
            ap_load_typed_value: assert property (load_accept |->
                load_source_value[source] == -8'sd1
                || load_source_value[source] == 8'sd0
                || load_source_value[source] == 8'sd1);
        end
        for (genvar bank = 0; bank < BANKS; bank++) begin : g_issue_value
            ap_issue_typed_value: assert property (issue_valid |->
                issue_source_value[bank] == -8'sd1
                || issue_source_value[bank] == 8'sd0
                || issue_source_value[bank] == 8'sd1);
            ap_mask_matches_value: assert property (issue_valid |->
                issue_bank_valid[bank] == (issue_source_value[bank] != 0));
        end
    endgenerate

    cp_negative_source: cover property (
        issue_valid && issue_ready && issue_source_value[0] == -8'sd1);
    cp_positive_source: cover property (
        issue_valid && issue_ready && issue_source_value[0] == 8'sd1);
    cp_issue_stall: cover property (issue_valid && !issue_ready);
    cp_memory_stall: cover property (mem_req_valid && !mem_req_ready);
    cp_commit_stall: cover property (commit_valid && !commit_ready);
    cp_terminal: cover property (commit_valid && commit_ready && commit_terminal);
    cp_protocol_attack: cover property (protocol_error);
endmodule

`default_nettype wire
