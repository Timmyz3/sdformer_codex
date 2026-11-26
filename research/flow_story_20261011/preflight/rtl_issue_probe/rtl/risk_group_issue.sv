// Independent research prototype. No PE, tensor data, PSN, or memory datapath.
// A group has 1 or 4 members: window_id = base + member_index * stride.
// Input contract: ordered risks, and every generated address fits ADDR_W bits.
// Synchronous reset cancels remaining work; accepted descriptors cannot be undone.
module risk_group_issue #(
    parameter integer ADDR_W = 16,
    parameter integer GROUP_W = 16,
    parameter integer FORCE_FULL = 0
) (
    input  logic clk,
    input  logic rst,
    input  logic group_valid,
    output logic group_ready,
    input  logic [GROUP_W-1:0] group_id,
    input  logic [ADDR_W-1:0] group_base,
    input  logic [ADDR_W-1:0] group_stride,
    input  logic group_four,
    input  logic [7:0] group_threshold,
    input  logic risk_valid,
    output logic risk_ready,
    input  logic [7:0] risk_value,
    output logic compute_issue_valid,
    input  logic compute_issue_ready,
    output logic bypass_issue_valid,
    input  logic bypass_issue_ready,
    output logic [GROUP_W-1:0] issue_group_id,
    output logic [ADDR_W-1:0] issue_window_id,
    output logic [1:0] issue_member,
    output logic issue_last
);
    typedef enum logic [1:0] {IDLE=2'd0, COLLECT=2'd1, ISSUE=2'd2} state_t;
    state_t state;
    logic [GROUP_W-1:0] id_q;
    logic [ADDR_W-1:0] address_q, stride_q;
    logic [7:0] threshold_q, maximum_q;
    logic [1:0] last_member_q, risk_count_q, member_q;
    logic keep_q;
    logic [7:0] merged_risk;
    logic issue_fire;

    always_comb begin
        group_ready = (state == IDLE) && !rst;
        risk_ready = (state == COLLECT) && !rst;
        compute_issue_valid = (state == ISSUE) && keep_q && !rst;
        bypass_issue_valid = (state == ISSUE) && !keep_q && !rst;
        issue_group_id = id_q;
        issue_window_id = address_q;
        issue_member = member_q;
        issue_last = (member_q == last_member_q);
        merged_risk = (risk_value > maximum_q) ? risk_value : maximum_q;
        issue_fire = (compute_issue_valid && compute_issue_ready) ||
                     (bypass_issue_valid && bypass_issue_ready);
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            id_q <= '0;
            address_q <= '0;
            stride_q <= '0;
            threshold_q <= '0;
            maximum_q <= '0;
            last_member_q <= '0;
            risk_count_q <= '0;
            member_q <= '0;
            keep_q <= 1'b0;
        end else begin
            case (state)
                IDLE: if (group_valid && group_ready) begin
                    id_q <= group_id;
                    address_q <= group_base;
                    stride_q <= group_stride;
                    last_member_q <= group_four ? 2'd3 : 2'd0;
                    threshold_q <= group_threshold;
                    maximum_q <= '0;
                    risk_count_q <= '0;
                    member_q <= '0;
                    keep_q <= (FORCE_FULL != 0);
                    if (FORCE_FULL != 0) state <= ISSUE;
                    else state <= COLLECT;
                end
                COLLECT: if (risk_valid && risk_ready) begin
                    maximum_q <= merged_risk;
                    if (risk_count_q == last_member_q) begin
                        // Equality is conservative: retain the group.
                        keep_q <= (merged_risk >= threshold_q);
                        state <= ISSUE;
                    end else risk_count_q <= risk_count_q + 2'd1;
                end
                ISSUE: if (issue_fire) begin
                    if (member_q == last_member_q) state <= IDLE;
                    else begin
                        member_q <= member_q + 2'd1;
                        address_q <= address_q + stride_q;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
