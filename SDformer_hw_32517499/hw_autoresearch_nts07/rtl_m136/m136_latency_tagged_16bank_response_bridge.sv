`timescale 1ns/1ps
`default_nettype none

// Fixed-one-cycle, latency-tagged bridge between a 16-bank SRAM request and
// the 512-bit PWP beat consumer.  M134 proved the address permutation; M136
// stores the accepted base-bank and metadata until the macro response arrives,
// checks an echoed transaction token, rotates physical-bank order back to
// logical-word order, and absorbs up to two returned beats under backpressure.
// The SRAM macros themselves remain outside this module.
module m136_latency_tagged_16bank_response_bridge (
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
    localparam int FIFO_DEPTH = 2;

    logic fault_q;
    logic [15:0] next_token_q;

    logic pending_q;
    logic [3:0] pending_base_bank_q;
    logic pending_start_q;
    logic pending_last_q;
    logic [3:0] pending_width_q;
    logic [31:0] pending_tag_q;
    logic [15:0] pending_token_q;

    logic [511:0] fifo_words_q [0:FIFO_DEPTH-1];
    logic fifo_start_q [0:FIFO_DEPTH-1];
    logic fifo_last_q [0:FIFO_DEPTH-1];
    logic [3:0] fifo_width_q [0:FIFO_DEPTH-1];
    logic [31:0] fifo_tag_q [0:FIFO_DEPTH-1];
    logic [15:0] fifo_token_q [0:FIFO_DEPTH-1];
    logic read_pointer_q;
    logic write_pointer_q;
    logic [1:0] fifo_count_q;

    logic [12:0] request_end_word;
    logic request_legal;
    logic mapper_violation;
    logic response_violation;
    logic quarantine;
    logic pop_now;
    logic enqueue_now;
    logic [2:0] projected_count;
    logic [511:0] rotated_response_words;

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
    assign response_violation = (macro_response_valid != pending_q)
                              || (macro_response_valid && pending_q
                                  && macro_response_token != pending_token_q);
    assign quarantine = fault_q || mapper_violation || response_violation;

    assign pop_now = fifo_count_q != 0 && response_ready && !quarantine;
    assign enqueue_now = pending_q && macro_response_valid
                       && macro_response_token == pending_token_q
                       && !quarantine;
    assign projected_count = {1'b0, fifo_count_q}
                           + enqueue_now - pop_now;

    // Reserve space for the response that a newly accepted request must return
    // on the next cycle, even if the consumer stalls in that future cycle.
    assign request_ready = !rst_core && !quarantine
                         && (!request_valid || request_legal)
                         && projected_count < FIFO_DEPTH;
    assign request_accept = request_valid && request_ready;
    assign macro_request_valid = request_accept;
    assign macro_request_token = next_token_q;

    assign response_valid = !rst_core && !quarantine && fifo_count_q != 0;
    assign response_logical_words = response_valid
                                  ? fifo_words_q[read_pointer_q] : '0;
    assign response_start = response_valid ? fifo_start_q[read_pointer_q] : 1'b0;
    assign response_last = response_valid ? fifo_last_q[read_pointer_q] : 1'b0;
    assign response_width = response_valid ? fifo_width_q[read_pointer_q] : '0;
    assign response_tag = response_valid ? fifo_tag_q[read_pointer_q] : '0;
    assign response_token = response_valid ? fifo_token_q[read_pointer_q] : '0;
    assign response_accept = response_valid && response_ready;

    assign protocol_error = !rst_core && quarantine;
    assign pending_response = pending_q;
    assign buffered_responses = fifo_count_q;
    assign busy = pending_q || fifo_count_q != 0;

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
            read_pointer_q <= 1'b0;
            write_pointer_q <= 1'b0;
            fifo_count_q <= '0;
            for (int slot = 0; slot < FIFO_DEPTH; slot++) begin
                fifo_words_q[slot] <= '0;
                fifo_start_q[slot] <= 1'b0;
                fifo_last_q[slot] <= 1'b0;
                fifo_width_q[slot] <= '0;
                fifo_tag_q[slot] <= '0;
                fifo_token_q[slot] <= '0;
            end
        end else if (mapper_violation || response_violation) begin
            fault_q <= 1'b1;
            pending_q <= 1'b0;
            read_pointer_q <= 1'b0;
            write_pointer_q <= 1'b0;
            fifo_count_q <= '0;
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

            if (enqueue_now) begin
                fifo_words_q[write_pointer_q] <= rotated_response_words;
                fifo_start_q[write_pointer_q] <= pending_start_q;
                fifo_last_q[write_pointer_q] <= pending_last_q;
                fifo_width_q[write_pointer_q] <= pending_width_q;
                fifo_tag_q[write_pointer_q] <= pending_tag_q;
                fifo_token_q[write_pointer_q] <= pending_token_q;
                write_pointer_q <= write_pointer_q + 1'b1;
            end
            if (pop_now)
                read_pointer_q <= read_pointer_q + 1'b1;

            case ({enqueue_now, pop_now})
                2'b10: fifo_count_q <= fifo_count_q + 1'b1;
                2'b01: fifo_count_q <= fifo_count_q - 1'b1;
                default: fifo_count_q <= fifo_count_q;
            endcase
        end
    end
endmodule

`default_nettype wire
