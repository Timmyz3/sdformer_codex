`timescale 1ns/1ps
`default_nettype none

// Sequence/barrier closure around the sealed M142 bounded-overlap controller.
//
// M144 leaves descriptorization and bank ownership in M142.  It adds the
// minimum missing contracts exposed by the M143r2 independent review:
//   * exact relative row count/order for each unit;
//   * a 32-bit sequence echoed through both engine completion paths; and
//   * one bounded outer fence.  Post-fence PWP/correction launches are held
//     until the external commit for that fence completes, while the producer
//     may use the remaining finite banks as lookahead.
module m144_sequence_fenced_raw128_overlap_wrapper #(
    parameter int TAG_BITS = 16,
    parameter int ROW_BITS = 9,
    parameter int BANKS = 4,
    parameter int SEQUENCE_BITS = 32
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         row_valid,
    output logic                         row_ready,
    input  logic                         row_window_start,
    input  logic                         row_window_end,
    input  logic [TAG_BITS-1:0]          row_window_tag,
    input  logic [ROW_BITS-1:0]          row_id,
    input  logic [ROW_BITS-1:0]          row_window_rows,
    input  logic [15:0]                  row_source_mask [0:7],
    input  logic [15:0]                  row_negate_mask [0:7],
    output logic                         row_accept,

    output logic                         descriptor_valid,
    input  logic                         descriptor_ready,
    output logic [1:0]                   descriptor_bank,
    output logic [TAG_BITS-1:0]          descriptor_window_tag,
    output logic [ROW_BITS-1:0]          descriptor_row,
    output logic [2:0]                   descriptor_block,
    output logic [1:0]                   descriptor_source_count_m1,
    output logic [3:0]                   descriptor_source [0:3],
    output logic [3:0]                   descriptor_negate,
    output logic                         descriptor_row_last,
    output logic                         descriptor_window_last,
    output logic                         descriptor_accept,

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

    input  logic                         outer_barrier_valid,
    output logic                         outer_barrier_ready,
    input  logic [TAG_BITS-1:0]          outer_barrier_tag,
    output logic                         outer_barrier_accept,
    output logic                         outer_commit_valid,
    input  logic                         outer_commit_done_valid,
    input  logic [TAG_BITS-1:0]          outer_commit_done_tag,
    output logic                         outer_commit_done_accept,
    output logic [TAG_BITS-1:0]          outer_commit_tag,
    output logic [SEQUENCE_BITS-1:0]     outer_commit_fence_sequence,

    output logic [BANKS-1:0]             observed_bank_free,
    output logic [BANKS-1:0]             observed_bank_fill,
    output logic [BANKS-1:0]             observed_bank_filled,
    output logic [BANKS-1:0]             observed_bank_pwp,
    output logic [BANKS-1:0]             observed_bank_wait_correction,
    output logic [BANKS-1:0]             observed_bank_correction,
    output logic                         observed_window_open,
    output logic                         observed_pwp_busy,
    output logic                         observed_correction_busy,
    output logic                         observed_barrier_active,
    output logic [SEQUENCE_BITS-1:0]     observed_next_sequence,
    output logic [SEQUENCE_BITS-1:0]     observed_next_completion_sequence,
    output logic                         protocol_error,
    output logic                         busy
);
    logic lower_row_valid, lower_row_ready, lower_row_accept;
    logic lower_pwp_valid, lower_pwp_ready, lower_pwp_accept;
    logic lower_correction_valid, lower_correction_ready;
    logic lower_correction_accept;
    logic lower_pwp_done_valid, lower_correction_done_valid;
    logic lower_protocol_error, lower_busy;

    logic wrapper_fault_q, lower_fault_q;
    logic unit_open_q;
    logic [TAG_BITS-1:0] unit_tag_q;
    logic [ROW_BITS-1:0] unit_rows_q;
    logic [ROW_BITS-1:0] expected_row_q;
    logic row_semantically_valid;
    logic illegal_row;

    logic [SEQUENCE_BITS-1:0] bank_sequence_q [0:BANKS-1];
    logic [BANKS-1:0] bank_sequence_valid_q;
    logic [BANKS-1:0] bank_post_fence_q;
    logic [SEQUENCE_BITS-1:0] next_sequence_q;
    logic [SEQUENCE_BITS-1:0] next_completion_sequence_q;
    logic allocation_bank_valid;
    logic [1:0] allocation_bank;

    logic barrier_active_q;
    logic [TAG_BITS-1:0] barrier_tag_q;
    logic [SEQUENCE_BITS-1:0] barrier_fence_q;
    logic barrier_service_block;
    logic barrier_drained;
    logic illegal_pwp_sequence;
    logic illegal_correction_sequence;
    logic illegal_completion_order;
    logic illegal_commit_done;
    logic illegal_internal_launch;
    logic persistent_quarantine;
    logic external_illegal_event;
    logic wrapper_illegal_event;
    logic wrapper_quarantine;

`ifndef SYNTHESIS
    initial begin
        if (BANKS != 4 || ROW_BITS != 9 || SEQUENCE_BITS < 18)
            $fatal(1, "M144 production geometry drift");
    end
`endif

    always_comb begin : choose_allocation_bank
        allocation_bank_valid = 1'b0;
        allocation_bank = '0;
        for (int bank = 0; bank < BANKS; bank++) begin
            if (!allocation_bank_valid && observed_bank_free[bank]) begin
                allocation_bank_valid = 1'b1;
                allocation_bank = bank[1:0];
            end
        end
    end

    always_comb begin : audit_row_order
        if (!unit_open_q) begin
            row_semantically_valid = row_window_start
                && row_window_rows != 0
                && row_window_rows <= 9'd384
                && row_id == 0
                && row_window_end == (row_window_rows == 1);
        end else begin
            row_semantically_valid = !row_window_start
                && row_window_tag == unit_tag_q
                && row_window_rows == unit_rows_q
                && row_id == expected_row_q
                && row_window_end
                   == (expected_row_q + 1'b1 == unit_rows_q);
        end
    end

    assign illegal_row = row_valid && !row_semantically_valid;
    assign persistent_quarantine = wrapper_fault_q || lower_fault_q;
    // Internal launch consistency is checked and latched, but is kept out of
    // the same-cycle quarantine cone.  The launch handshakes are explicitly
    // gated by bank_sequence_valid_q below.  This preserves fail-closed
    // behavior without feeding lower.valid -> wrapper.quarantine ->
    // lower.row_valid -> lower.quarantine -> lower.valid.
    assign wrapper_quarantine = persistent_quarantine
                              || external_illegal_event;
    assign lower_row_valid = row_valid && row_semantically_valid
                           && !wrapper_quarantine;
    assign row_ready = lower_row_ready && row_semantically_valid
                     && !wrapper_quarantine;
    assign row_accept = row_valid && row_ready;

    assign pwp_sequence = bank_sequence_q[pwp_bank];
    assign correction_sequence = bank_sequence_q[correction_bank];
    // Classification is captured when a bank is allocated.  The PWP launch
    // critical path therefore uses a one-bit bank flag instead of another
    // 32-bit sequence comparator after M142's oldest-bank selector.
    assign barrier_service_block = barrier_active_q
        && lower_pwp_valid && bank_post_fence_q[pwp_bank];
    assign pwp_valid = lower_pwp_valid
                     && bank_sequence_valid_q[pwp_bank]
                     && !barrier_service_block
                     && !wrapper_quarantine;
    assign lower_pwp_ready = pwp_ready
                           && bank_sequence_valid_q[pwp_bank]
                           && !barrier_service_block
                           && !wrapper_quarantine;
    assign pwp_accept = pwp_valid && pwp_ready;

    // Correction is ordered behind PWP.  Gate it independently as a defensive
    // contract in case a future lower controller exposes a later wait bank.
    assign correction_valid = lower_correction_valid
        && bank_sequence_valid_q[correction_bank]
        && !(barrier_active_q
             && bank_post_fence_q[correction_bank])
        && !wrapper_quarantine;
    assign lower_correction_ready = correction_ready
        && bank_sequence_valid_q[correction_bank]
        && !(barrier_active_q
             && bank_post_fence_q[correction_bank])
        && !wrapper_quarantine;
    assign correction_accept = correction_valid && correction_ready;

    assign illegal_internal_launch
        = (lower_pwp_valid && !bank_sequence_valid_q[pwp_bank])
          || (lower_correction_valid
              && !bank_sequence_valid_q[correction_bank]);
    assign illegal_pwp_sequence = pwp_done_valid
        && (!bank_sequence_valid_q[pwp_done_bank]
            || pwp_done_sequence != bank_sequence_q[pwp_done_bank]);
    assign illegal_correction_sequence = correction_done_valid
        && (!bank_sequence_valid_q[correction_done_bank]
            || correction_done_sequence
               != bank_sequence_q[correction_done_bank]);
    assign illegal_completion_order = correction_done_valid
        && correction_done_sequence != next_completion_sequence_q;
    assign lower_pwp_done_valid = pwp_done_valid
        && !illegal_pwp_sequence && !wrapper_quarantine;
    assign lower_correction_done_valid = correction_done_valid
        && !illegal_correction_sequence && !illegal_completion_order
        && !wrapper_quarantine;

    assign outer_barrier_ready = !wrapper_quarantine && !barrier_active_q
                               && !unit_open_q && next_sequence_q != 0;
    assign outer_barrier_accept = outer_barrier_valid
                                && outer_barrier_ready;
    assign barrier_drained = barrier_active_q
        && next_completion_sequence_q > barrier_fence_q;
    // Keep the offer independent of outer_commit_done_valid.  A malformed
    // acknowledgement is rejected and latched below; feeding it back into the
    // offer would create a combinational valid/error loop.
    assign outer_commit_valid = barrier_drained && !persistent_quarantine;
    assign outer_commit_tag = barrier_tag_q;
    assign outer_commit_fence_sequence = barrier_fence_q;
    assign illegal_commit_done = outer_commit_done_valid
        && (!outer_commit_valid
            || outer_commit_done_tag != barrier_tag_q);
    assign outer_commit_done_accept = outer_commit_done_valid
        && outer_commit_valid && !wrapper_illegal_event;

    assign external_illegal_event = illegal_row || illegal_pwp_sequence
        || illegal_correction_sequence || illegal_completion_order
        || illegal_commit_done;
    assign wrapper_illegal_event = external_illegal_event
        || illegal_internal_launch;
    assign protocol_error = !rst_core
        && (wrapper_fault_q || lower_fault_q || wrapper_illegal_event
            || lower_protocol_error);
    assign observed_barrier_active = barrier_active_q;
    assign observed_next_sequence = next_sequence_q;
    assign observed_next_completion_sequence
        = next_completion_sequence_q;
    assign busy = lower_busy || unit_open_q || barrier_active_q;

    m142_sparse_mask_k4_bounded_overlap_controller #(
        .TAG_BITS(TAG_BITS), .ROW_BITS(ROW_BITS),
        .SEQUENCE_BITS(SEQUENCE_BITS), .BANKS(BANKS)
    ) lower (
        .clk_core(clk_core), .rst_core(rst_core),
        .row_valid(lower_row_valid), .row_ready(lower_row_ready),
        .row_window_start(row_window_start),
        .row_window_end(row_window_end),
        .row_window_tag(row_window_tag), .row_id(row_id),
        .row_source_mask(row_source_mask),
        .row_negate_mask(row_negate_mask),
        .row_accept(lower_row_accept),
        .descriptor_valid(descriptor_valid),
        .descriptor_ready(descriptor_ready),
        .descriptor_bank(descriptor_bank),
        .descriptor_window_tag(descriptor_window_tag),
        .descriptor_row(descriptor_row),
        .descriptor_block(descriptor_block),
        .descriptor_source_count_m1(descriptor_source_count_m1),
        .descriptor_source(descriptor_source),
        .descriptor_negate(descriptor_negate),
        .descriptor_row_last(descriptor_row_last),
        .descriptor_window_last(descriptor_window_last),
        .descriptor_accept(descriptor_accept),
        .pwp_valid(lower_pwp_valid), .pwp_ready(lower_pwp_ready),
        .pwp_bank(pwp_bank), .pwp_window_tag(pwp_window_tag),
        .pwp_accept(lower_pwp_accept),
        .pwp_done_valid(lower_pwp_done_valid),
        .pwp_done_bank(pwp_done_bank),
        .pwp_done_window_tag(pwp_done_window_tag),
        .correction_valid(lower_correction_valid),
        .correction_ready(lower_correction_ready),
        .correction_bank(correction_bank),
        .correction_window_tag(correction_window_tag),
        .correction_accept(lower_correction_accept),
        .correction_done_valid(lower_correction_done_valid),
        .correction_done_bank(correction_done_bank),
        .correction_done_window_tag(correction_done_window_tag),
        .observed_bank_free(observed_bank_free),
        .observed_bank_fill(observed_bank_fill),
        .observed_bank_filled(observed_bank_filled),
        .observed_bank_pwp(observed_bank_pwp),
        .observed_bank_wait_correction(
            observed_bank_wait_correction),
        .observed_bank_correction(observed_bank_correction),
        .observed_window_open(observed_window_open),
        .observed_pwp_busy(observed_pwp_busy),
        .observed_correction_busy(observed_correction_busy),
        .protocol_error(lower_protocol_error), .busy(lower_busy)
    );

    always_ff @(posedge clk_core) begin : wrapper_state
        if (rst_core) begin
            wrapper_fault_q <= 1'b0;
            lower_fault_q <= 1'b0;
            unit_open_q <= 1'b0;
            unit_tag_q <= '0;
            unit_rows_q <= '0;
            expected_row_q <= '0;
            bank_sequence_valid_q <= '0;
            bank_post_fence_q <= '0;
            next_sequence_q <= '0;
            next_completion_sequence_q <= '0;
            barrier_active_q <= 1'b0;
            barrier_tag_q <= '0;
            barrier_fence_q <= '0;
            for (int bank = 0; bank < BANKS; bank++)
                bank_sequence_q[bank] <= '0;
        end else begin
            if (wrapper_illegal_event)
                wrapper_fault_q <= 1'b1;
            if (lower_protocol_error)
                lower_fault_q <= 1'b1;

            if (!wrapper_quarantine && lower_row_accept) begin
                if (row_window_start) begin
                    if (!allocation_bank_valid)
                        wrapper_fault_q <= 1'b1;
                    else begin
                        bank_sequence_q[allocation_bank]
                            <= next_sequence_q;
                        bank_sequence_valid_q[allocation_bank] <= 1'b1;
                        bank_post_fence_q[allocation_bank]
                            <= barrier_active_q
                               || outer_barrier_accept;
                    end
                    next_sequence_q <= next_sequence_q + 1'b1;
                    unit_tag_q <= row_window_tag;
                    unit_rows_q <= row_window_rows;
                    expected_row_q <= 1;
                    unit_open_q <= !row_window_end;
                end else if (row_window_end) begin
                    unit_open_q <= 1'b0;
                end else begin
                    expected_row_q <= expected_row_q + 1'b1;
                end
            end

            if (!wrapper_quarantine && lower_correction_done_valid) begin
                bank_sequence_valid_q[correction_done_bank] <= 1'b0;
                bank_post_fence_q[correction_done_bank] <= 1'b0;
                next_completion_sequence_q
                    <= next_completion_sequence_q + 1'b1;
            end

            if (!wrapper_quarantine && outer_barrier_accept) begin
                barrier_active_q <= 1'b1;
                barrier_tag_q <= outer_barrier_tag;
                barrier_fence_q <= next_sequence_q - 1'b1;
            end
            if (!wrapper_quarantine && outer_commit_done_accept) begin
                barrier_active_q <= 1'b0;
                bank_post_fence_q <= '0;
            end
        end
    end
endmodule

`default_nettype wire
