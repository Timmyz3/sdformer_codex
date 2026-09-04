`timescale 1ns/1ps
`default_nettype none

// M194: elastic bank-head selector for token-owned adjacent-window fusion.
//
// Two resident FC2 windows may share one Acc24 only when their token tags are
// equal.  For every physical bank this module selects the earlier window's
// head, falling through to the later window when the earlier queue is empty.
// Thus at most one weight read is requested from each bank per cycle and the
// eight returned weights still feed one ordinary fixed-bank accumulator.
// Cross-token pairs, malformed counts and channel/bank mismatches fail closed.
// Queue storage, head advancement, SRAM response and complete-FC2 control are
// deliberately outside this steering screen.
module m194_fc2_same_token_pair_bank_head_selector #(
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int COUNT_BITS = 8
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         pair_valid,
    output logic                         pair_ready,
    input  logic [1:0]                   window_valid,
    input  logic [TAG_BITS-1:0]          window_token_tag [0:1],
    input  logic [COUNT_BITS-1:0]        window_bank_count [0:1][0:7],
    input  logic [CHANNEL_BITS-1:0]      window_head_channel [0:1][0:7],
    output logic                         pair_accept,

    output logic                         issue_valid,
    input  logic                         issue_ready,
    output logic [TAG_BITS-1:0]          issue_token_tag,
    output logic [3:0]                   issue_source_count,
    output logic [7:0]                   issue_bank_valid,
    output logic [7:0]                   issue_selected_window,
    output logic [CHANNEL_BITS-1:0]      issue_source_channel [0:7],
    output logic                         issue_pair_last,
    output logic                         issue_accept,

    output logic                         protocol_error,
    output logic                         busy
);
    logic fault_q;
    logic issue_valid_q;
    logic [TAG_BITS-1:0] issue_token_tag_q;
    logic [3:0] issue_source_count_q;
    logic [7:0] issue_bank_valid_q;
    logic [7:0] issue_selected_window_q;
    logic [CHANNEL_BITS-1:0] issue_source_channel_q [0:7];
    logic issue_pair_last_q;

    logic shape_legal;
    logic token_legal;
    logic count_channel_legal;
    logic nonempty;
    logic [3:0] candidate_source_count;
    logic [7:0] candidate_bank_valid;
    logic [7:0] candidate_selected_window;
    logic [CHANNEL_BITS-1:0] candidate_source_channel [0:7];
    logic candidate_pair_last;
    logic illegal_request;
    logic slot_open;

    always_comb begin : candidate_selection
        logic [COUNT_BITS:0] merged_count;
        shape_legal = window_valid != 2'b00;
        token_legal = !(window_valid == 2'b11)
            || window_token_tag[0] == window_token_tag[1];
        count_channel_legal = 1'b1;
        nonempty = 1'b0;
        candidate_source_count = '0;
        candidate_bank_valid = '0;
        candidate_selected_window = '0;
        candidate_pair_last = 1'b1;
        for (int bank = 0; bank < 8; bank++) begin
            for (int window = 0; window < 2; window++) begin
                if (!window_valid[window]
                        && window_bank_count[window][bank] != 0)
                    count_channel_legal = 1'b0;
                if (window_valid[window]
                        && window_bank_count[window][bank] != 0
                        && window_head_channel[window][bank][2:0]
                            != bank[2:0])
                    count_channel_legal = 1'b0;
            end
            merged_count = (window_valid[0]
                    ? {1'b0, window_bank_count[0][bank]} : '0)
                + (window_valid[1]
                    ? {1'b0, window_bank_count[1][bank]} : '0);
            candidate_bank_valid[bank] = merged_count != 0;
            candidate_source_count = candidate_source_count
                + candidate_bank_valid[bank];
            nonempty = nonempty || candidate_bank_valid[bank];
            if (window_valid[0] && window_bank_count[0][bank] != 0) begin
                candidate_selected_window[bank] = 1'b0;
                candidate_source_channel[bank]
                    = window_head_channel[0][bank];
            end else if (window_valid[1]
                    && window_bank_count[1][bank] != 0) begin
                candidate_selected_window[bank] = 1'b1;
                candidate_source_channel[bank]
                    = window_head_channel[1][bank];
            end else begin
                candidate_selected_window[bank] = 1'b0;
                candidate_source_channel[bank] = '0;
            end
            if (merged_count > 1)
                candidate_pair_last = 1'b0;
        end
    end

    assign illegal_request = pair_valid
        && !(shape_legal && token_legal && count_channel_legal && nonempty);
    assign slot_open = !issue_valid_q || issue_accept;
    assign pair_ready = !fault_q && slot_open
        && shape_legal && token_legal && count_channel_legal && nonempty;
    assign pair_accept = pair_valid && pair_ready;
    assign issue_valid = issue_valid_q;
    assign issue_accept = issue_valid_q && issue_ready;
    assign issue_token_tag = issue_token_tag_q;
    assign issue_source_count = issue_source_count_q;
    assign issue_bank_valid = issue_bank_valid_q;
    assign issue_selected_window = issue_selected_window_q;
    assign issue_pair_last = issue_pair_last_q;
    assign protocol_error = fault_q || illegal_request;
    assign busy = issue_valid_q;

    generate
        for (genvar bank = 0; bank < 8; bank++) begin : g_channel
            assign issue_source_channel[bank]
                = issue_source_channel_q[bank];
        end
    endgenerate

    always_ff @(posedge clk_core) begin : elastic_state
        if (rst_core) begin
            fault_q <= 1'b0;
            issue_valid_q <= 1'b0;
            issue_token_tag_q <= '0;
            issue_source_count_q <= '0;
            issue_bank_valid_q <= '0;
            issue_selected_window_q <= '0;
            issue_pair_last_q <= 1'b0;
            for (int bank = 0; bank < 8; bank++)
                issue_source_channel_q[bank] <= '0;
        end else begin
            if (illegal_request)
                fault_q <= 1'b1;
            if (issue_accept && !pair_accept)
                issue_valid_q <= 1'b0;
            if (pair_accept) begin
                issue_valid_q <= 1'b1;
                issue_token_tag_q <= window_valid[0]
                    ? window_token_tag[0] : window_token_tag[1];
                issue_source_count_q <= candidate_source_count;
                issue_bank_valid_q <= candidate_bank_valid;
                issue_selected_window_q <= candidate_selected_window;
                issue_pair_last_q <= candidate_pair_last;
                for (int bank = 0; bank < 8; bank++)
                    issue_source_channel_q[bank]
                        <= candidate_source_channel[bank];
            end
        end
    end
endmodule

`default_nettype wire
