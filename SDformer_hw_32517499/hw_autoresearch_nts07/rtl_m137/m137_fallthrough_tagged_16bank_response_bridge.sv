`timescale 1ns/1ps
`default_nettype none

// Storage-reduced successor to M136.  A correct one-cycle SRAM response falls
// through directly to the consumer; only a stalled response is captured in a
// single 512-bit skid register.  Capacity is reserved before accepting the next
// request, so a future macro response can never overwrite stalled data.
module m137_fallthrough_tagged_16bank_response_bridge (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         request_valid,
    output logic                         request_ready,
    input  logic [11:0]                  logical_base_word,
    input  logic                         request_start,
    input  logic                         request_last,
    input  logic [3:0]                   request_width,
    input  logic [31:0]                  request_tag,
    output logic                         request_accept,

    output logic                         macro_request_valid,
    output logic [127:0]                 macro_bank_row_addresses,
    output logic [15:0]                  macro_request_token,
    input  logic                         macro_response_valid,
    input  logic [15:0]                  macro_response_token,
    input  logic [511:0]                 macro_bank_words,

    output logic                         response_valid,
    input  logic                         response_ready,
    output logic [511:0]                 response_logical_words,
    output logic                         response_start,
    output logic                         response_last,
    output logic [3:0]                   response_width,
    output logic [31:0]                  response_tag,
    output logic [15:0]                  response_token,
    output logic                         response_accept,

    output logic                         protocol_error,
    output logic                         pending_response,
    output logic [1:0]                   buffered_responses,
    output logic                         busy
);
    localparam int WORDS = 3680;

    logic fault_q;
    logic [15:0] next_token_q;

    logic pending_q;
    logic [3:0] pending_base_bank_q;
    logic pending_start_q;
    logic pending_last_q;
    logic [3:0] pending_width_q;
    logic [31:0] pending_tag_q;
    logic [15:0] pending_token_q;

    logic skid_valid_q;
    logic [511:0] skid_words_q;
    logic skid_start_q;
    logic skid_last_q;
    logic [3:0] skid_width_q;
    logic [31:0] skid_tag_q;
    logic [15:0] skid_token_q;

    logic [12:0] request_end_word;
    logic request_legal;
    logic mapper_violation;
    logic response_identity_good;
    logic response_violation;
    logic quarantine;
    logic [511:0] rotated_response_words;
    logic direct_response;
    logic skid_pop;
    logic skid_store;
    logic skid_full_after_edge;

    always_comb begin : form_conflict_free_bank_request
        request_end_word = {1'b0, logical_base_word} + 13'd15;
        request_legal = request_end_word < WORDS;
        macro_bank_row_addresses = '0;
        for (int bank = 0; bank < 16; bank++) begin
            macro_bank_row_addresses[bank*8 +: 8] =
                logical_base_word[11:4] + (bank < logical_base_word[3:0]);
        end
    end

    always_comb begin : rotate_returned_physical_banks
        rotated_response_words = '0;
        for (int word = 0; word < 16; word++) begin
            rotated_response_words[word*32 +: 32] = macro_bank_words[
                (((pending_base_bank_q + word) & 4'hf)*32) +: 32];
        end
    end

    assign mapper_violation = request_valid && !request_legal;
    assign response_identity_good = pending_q && macro_response_valid
                                  && macro_response_token == pending_token_q;
    assign response_violation = (macro_response_valid != pending_q)
                              || (macro_response_valid && pending_q
                                  && macro_response_token != pending_token_q)
                              || (skid_valid_q && response_identity_good);
    assign quarantine = fault_q || mapper_violation || response_violation;

    assign direct_response = !skid_valid_q && response_identity_good;
    assign response_valid = !rst_core && !quarantine
                          && (skid_valid_q || direct_response);
    assign response_logical_words = skid_valid_q ? skid_words_q
                                  : direct_response ? rotated_response_words : '0;
    assign response_start = skid_valid_q ? skid_start_q
                          : direct_response ? pending_start_q : 1'b0;
    assign response_last = skid_valid_q ? skid_last_q
                         : direct_response ? pending_last_q : 1'b0;
    assign response_width = skid_valid_q ? skid_width_q
                          : direct_response ? pending_width_q : '0;
    assign response_tag = skid_valid_q ? skid_tag_q
                        : direct_response ? pending_tag_q : '0;
    assign response_token = skid_valid_q ? skid_token_q
                          : direct_response ? pending_token_q : '0;
    assign response_accept = response_valid && response_ready;

    assign skid_pop = skid_valid_q && response_accept;
    assign skid_store = direct_response && !response_ready && !quarantine;
    assign skid_full_after_edge = (skid_valid_q && !skid_pop) || skid_store;

    // A request is accepted only if the post-edge skid state is empty.  Its
    // one-cycle-later response can then either fall through or occupy the skid.
    assign request_ready = !rst_core && !quarantine
                         && (!request_valid || request_legal)
                         && !skid_full_after_edge;
    assign request_accept = request_valid && request_ready;
    assign macro_request_valid = request_accept;
    assign macro_request_token = next_token_q;

    assign protocol_error = !rst_core && quarantine;
    assign pending_response = pending_q;
    assign buffered_responses = {1'b0, skid_valid_q};
    assign busy = pending_q || skid_valid_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            fault_q <= 1'b0;
            next_token_q <= '0;
            pending_q <= 1'b0;
            pending_base_bank_q <= '0;
            pending_start_q <= 1'b0;
            pending_last_q <= 1'b0;
            pending_width_q <= '0;
            pending_tag_q <= '0;
            pending_token_q <= '0;
            skid_valid_q <= 1'b0;
            skid_words_q <= '0;
            skid_start_q <= 1'b0;
            skid_last_q <= 1'b0;
            skid_width_q <= '0;
            skid_tag_q <= '0;
            skid_token_q <= '0;
        end else if (mapper_violation || response_violation) begin
            fault_q <= 1'b1;
            pending_q <= 1'b0;
            skid_valid_q <= 1'b0;
        end else if (!fault_q) begin
            pending_q <= request_accept;
            if (request_accept) begin
                pending_base_bank_q <= logical_base_word[3:0];
                pending_start_q <= request_start;
                pending_last_q <= request_last;
                pending_width_q <= request_width;
                pending_tag_q <= request_tag;
                pending_token_q <= next_token_q;
                next_token_q <= next_token_q + 1'b1;
            end

            if (skid_pop)
                skid_valid_q <= 1'b0;
            if (skid_store) begin
                skid_valid_q <= 1'b1;
                skid_words_q <= rotated_response_words;
                skid_start_q <= pending_start_q;
                skid_last_q <= pending_last_q;
                skid_width_q <= pending_width_q;
                skid_tag_q <= pending_tag_q;
                skid_token_q <= pending_token_q;
            end
        end
    end
endmodule

`default_nettype wire
