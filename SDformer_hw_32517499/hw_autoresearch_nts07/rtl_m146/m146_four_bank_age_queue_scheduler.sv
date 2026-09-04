`timescale 1ns/1ps
`default_nettype none

// Comparator-free four-bank service scheduler.
//
// M142 finds the oldest filled/waiting bank with parallel 32-bit sequence
// comparisons.  M146 carries the same 32-bit identity only as payload and
// preserves age with two four-entry, two-bit bank-index FIFOs.  The expensive
// sequence age comparator is therefore absent from both engine issue paths;
// full-width equality checks remain on completion identity for fail-closed
// stale-response rejection.
// Engine arithmetic and SRAMs are deliberately outside this standalone block.
module m146_four_bank_age_queue_scheduler #(
    parameter int TAG_BITS = 16,
    parameter int SEQUENCE_BITS = 32,
    parameter int BANKS = 4
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         fill_valid,
    output logic                         fill_ready,
    input  logic [1:0]                   fill_bank,
    input  logic [TAG_BITS-1:0]          fill_window_tag,
    input  logic [SEQUENCE_BITS-1:0]     fill_sequence,
    output logic                         fill_accept,

    output logic                         pwp_valid,
    input  logic                         pwp_ready,
    output logic [1:0]                   pwp_bank,
    output logic [TAG_BITS-1:0]          pwp_window_tag,
    output logic [SEQUENCE_BITS-1:0]     pwp_sequence,
    output logic                         pwp_accept,
    input  logic                         pwp_done_valid,
    input  logic [1:0]                   pwp_done_bank,
    input  logic [TAG_BITS-1:0]          pwp_done_window_tag,
    input  logic [SEQUENCE_BITS-1:0]     pwp_done_sequence,

    output logic                         correction_valid,
    input  logic                         correction_ready,
    output logic [1:0]                   correction_bank,
    output logic [TAG_BITS-1:0]          correction_window_tag,
    output logic [SEQUENCE_BITS-1:0]     correction_sequence,
    output logic                         correction_accept,
    input  logic                         correction_done_valid,
    input  logic [1:0]                   correction_done_bank,
    input  logic [TAG_BITS-1:0]          correction_done_window_tag,
    input  logic [SEQUENCE_BITS-1:0]     correction_done_sequence,

    output logic                         release_valid,
    output logic [1:0]                   release_bank,
    output logic [TAG_BITS-1:0]          release_window_tag,
    output logic [SEQUENCE_BITS-1:0]     release_sequence,
    output logic [BANKS-1:0]             observed_bank_free,
    output logic [2:0]                   observed_pwp_queue_count,
    output logic [2:0]                   observed_correction_queue_count,
    output logic                         observed_pwp_busy,
    output logic                         observed_correction_busy,
    output logic [SEQUENCE_BITS-1:0]     observed_next_fill_sequence,
    output logic                         protocol_error,
    output logic                         busy
);
    typedef enum logic [2:0] {
        BANK_FREE = 3'd0,
        BANK_FILLED = 3'd1,
        BANK_PWP = 3'd2,
        BANK_WAIT_CORRECTION = 3'd3,
        BANK_CORRECTION = 3'd4
    } bank_state_t;

    bank_state_t bank_state_q [0:BANKS-1];
    logic [TAG_BITS-1:0] bank_tag_q [0:BANKS-1];
    logic [SEQUENCE_BITS-1:0] bank_sequence_q [0:BANKS-1];
    logic [BANKS-1:0] bank_live_q;
    logic [SEQUENCE_BITS-1:0] next_fill_sequence_q;

    logic [1:0] pwp_fifo_q [0:BANKS-1];
    logic [1:0] correction_fifo_q [0:BANKS-1];
    logic [1:0] pwp_head_q, pwp_tail_q;
    logic [1:0] correction_head_q, correction_tail_q;
    logic [2:0] pwp_count_q, correction_count_q;

    logic pwp_busy_q, correction_busy_q;
    logic [1:0] pwp_active_bank_q, correction_active_bank_q;
    logic [TAG_BITS-1:0] pwp_active_tag_q, correction_active_tag_q;
    logic [SEQUENCE_BITS-1:0] pwp_active_sequence_q;
    logic [SEQUENCE_BITS-1:0] correction_active_sequence_q;
    logic fault_q;
    logic illegal_fill, illegal_pwp_done, illegal_correction_done;
    logic illegal_internal_state;
    logic illegal_event, quarantine;

`ifndef SYNTHESIS
    initial begin
        if (BANKS != 4 || TAG_BITS < 1 || SEQUENCE_BITS != 32)
            $fatal(1, "M146 production geometry drift");
    end
`endif

    always_comb begin
        observed_bank_free = ~bank_live_q;
        observed_pwp_queue_count = pwp_count_q;
        observed_correction_queue_count = correction_count_q;
        observed_pwp_busy = pwp_busy_q;
        observed_correction_busy = correction_busy_q;
        observed_next_fill_sequence = next_fill_sequence_q;
    end

    assign illegal_fill = fill_valid
        && (bank_live_q[fill_bank]
            || fill_sequence != next_fill_sequence_q);
    assign illegal_pwp_done = pwp_done_valid
        && (!pwp_busy_q
            || pwp_done_bank != pwp_active_bank_q
            || pwp_done_window_tag != pwp_active_tag_q
            || pwp_done_sequence != pwp_active_sequence_q);
    assign illegal_correction_done = correction_done_valid
        && (!correction_busy_q
            || correction_done_bank != correction_active_bank_q
            || correction_done_window_tag != correction_active_tag_q
            || correction_done_sequence != correction_active_sequence_q);
    assign illegal_internal_state =
           (pwp_count_q != 0
            && (!bank_live_q[pwp_fifo_q[pwp_head_q]]
                || bank_state_q[pwp_fifo_q[pwp_head_q]] != BANK_FILLED))
        || (correction_count_q != 0
            && (!bank_live_q[correction_fifo_q[correction_head_q]]
                || bank_state_q[correction_fifo_q[correction_head_q]]
                   != BANK_WAIT_CORRECTION))
        || (pwp_busy_q
            && (!bank_live_q[pwp_active_bank_q]
                || bank_state_q[pwp_active_bank_q] != BANK_PWP))
        || (correction_busy_q
            && (!bank_live_q[correction_active_bank_q]
                || bank_state_q[correction_active_bank_q]
                   != BANK_CORRECTION));
    assign illegal_event = illegal_fill || illegal_pwp_done
                         || illegal_correction_done
                         || illegal_internal_state;
    assign quarantine = fault_q || illegal_event;
    assign protocol_error = !rst_core && quarantine;

    assign fill_ready = !rst_core && !quarantine
        && !bank_live_q[fill_bank] && pwp_count_q < BANKS
        && (!fill_valid || fill_sequence == next_fill_sequence_q);
    assign fill_accept = fill_valid && fill_ready;

    assign pwp_bank = pwp_fifo_q[pwp_head_q];
    assign pwp_window_tag = bank_tag_q[pwp_bank];
    assign pwp_sequence = bank_sequence_q[pwp_bank];
    assign pwp_valid = !rst_core && !quarantine && !pwp_busy_q
                     && pwp_count_q != 0;
    assign pwp_accept = pwp_valid && pwp_ready;

    assign correction_bank = correction_fifo_q[correction_head_q];
    assign correction_window_tag = bank_tag_q[correction_bank];
    assign correction_sequence = bank_sequence_q[correction_bank];
    assign correction_valid = !rst_core && !quarantine
                            && !correction_busy_q
                            && correction_count_q != 0;
    assign correction_accept = correction_valid && correction_ready;

    assign release_valid = !rst_core && correction_done_valid
                         && !illegal_correction_done && !quarantine;
    assign release_bank = correction_done_bank;
    assign release_window_tag = correction_done_window_tag;
    assign release_sequence = correction_done_sequence;
    assign busy = pwp_busy_q || correction_busy_q
                || pwp_count_q != 0 || correction_count_q != 0
                || |bank_live_q;

    always_ff @(posedge clk_core) begin : scheduler_state
        if (rst_core) begin
            bank_live_q <= '0;
            next_fill_sequence_q <= '0;
            pwp_head_q <= '0;
            pwp_tail_q <= '0;
            correction_head_q <= '0;
            correction_tail_q <= '0;
            pwp_count_q <= '0;
            correction_count_q <= '0;
            pwp_busy_q <= 1'b0;
            correction_busy_q <= 1'b0;
            pwp_active_bank_q <= '0;
            correction_active_bank_q <= '0;
            pwp_active_tag_q <= '0;
            correction_active_tag_q <= '0;
            pwp_active_sequence_q <= '0;
            correction_active_sequence_q <= '0;
            fault_q <= 1'b0;
            for (int bank = 0; bank < BANKS; bank++) begin
                bank_state_q[bank] <= BANK_FREE;
                bank_tag_q[bank] <= '0;
                bank_sequence_q[bank] <= '0;
                pwp_fifo_q[bank] <= '0;
                correction_fifo_q[bank] <= '0;
            end
        end else begin
            if (illegal_event)
                fault_q <= 1'b1;

            if (!quarantine) begin
                case ({fill_accept, pwp_accept})
                    2'b10: pwp_count_q <= pwp_count_q + 1'b1;
                    2'b01: pwp_count_q <= pwp_count_q - 1'b1;
                    default: pwp_count_q <= pwp_count_q;
                endcase
                if (fill_accept) begin
                    bank_live_q[fill_bank] <= 1'b1;
                    bank_state_q[fill_bank] <= BANK_FILLED;
                    bank_tag_q[fill_bank] <= fill_window_tag;
                    bank_sequence_q[fill_bank] <= fill_sequence;
                    pwp_fifo_q[pwp_tail_q] <= fill_bank;
                    pwp_tail_q <= pwp_tail_q + 1'b1;
                    next_fill_sequence_q <= next_fill_sequence_q + 1'b1;
                end
                if (pwp_accept) begin
                    pwp_head_q <= pwp_head_q + 1'b1;
                    pwp_busy_q <= 1'b1;
                    pwp_active_bank_q <= pwp_bank;
                    pwp_active_tag_q <= pwp_window_tag;
                    pwp_active_sequence_q <= pwp_sequence;
                    bank_state_q[pwp_bank] <= BANK_PWP;
                end

                case ({pwp_done_valid, correction_accept})
                    2'b10:
                        correction_count_q
                            <= correction_count_q + 1'b1;
                    2'b01:
                        correction_count_q
                            <= correction_count_q - 1'b1;
                    default:
                        correction_count_q <= correction_count_q;
                endcase
                if (pwp_done_valid) begin
                    pwp_busy_q <= 1'b0;
                    bank_state_q[pwp_active_bank_q]
                        <= BANK_WAIT_CORRECTION;
                    correction_fifo_q[correction_tail_q]
                        <= pwp_active_bank_q;
                    correction_tail_q <= correction_tail_q + 1'b1;
                end
                if (correction_accept) begin
                    correction_head_q <= correction_head_q + 1'b1;
                    correction_busy_q <= 1'b1;
                    correction_active_bank_q <= correction_bank;
                    correction_active_tag_q <= correction_window_tag;
                    correction_active_sequence_q <= correction_sequence;
                    bank_state_q[correction_bank] <= BANK_CORRECTION;
                end
                if (correction_done_valid) begin
                    correction_busy_q <= 1'b0;
                    bank_live_q[correction_active_bank_q] <= 1'b0;
                    bank_state_q[correction_active_bank_q] <= BANK_FREE;
                end
            end
        end
    end
endmodule

`default_nettype wire
