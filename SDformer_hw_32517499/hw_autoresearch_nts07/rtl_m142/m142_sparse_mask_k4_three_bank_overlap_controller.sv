`timescale 1ns/1ps
`default_nettype none

// M142 is the bounded control/data boundary selected by the M141r3 cycle DSE.
// It deliberately does not implement the PWP or correction arithmetic, nor
// descriptor/result SRAM macros.  It provides:
//   * one ordered 128-bit row (eight 16-bit signed source masks) to a
//     canonical block-major K1..K4 descriptor lane;
//   * a compile-time bounded three- or four-bank ownership ring;
//   * fill -> PWP -> correction ownership transfer; and
//   * release only after the matching correction completion.
// PWP and correction are independent ready/valid endpoints and may own two
// different banks concurrently.  No unbounded FIFO or implicit bank exists.
module m142_sparse_mask_k4_bounded_overlap_controller #(
    parameter int TAG_BITS = 16,
    parameter int ROW_BITS = 9,
    parameter int SEQUENCE_BITS = 32,
    parameter int BANKS = 4
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         row_valid,
    output logic                         row_ready,
    input  logic                         row_window_start,
    input  logic                         row_window_end,
    input  logic [TAG_BITS-1:0]          row_window_tag,
    input  logic [ROW_BITS-1:0]          row_id,
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
    output logic                         pwp_accept,
    input  logic                         pwp_done_valid,
    input  logic [1:0]                   pwp_done_bank,
    input  logic [TAG_BITS-1:0]          pwp_done_window_tag,

    output logic                         correction_valid,
    input  logic                         correction_ready,
    output logic [1:0]                   correction_bank,
    output logic [TAG_BITS-1:0]          correction_window_tag,
    output logic                         correction_accept,
    input  logic                         correction_done_valid,
    input  logic [1:0]                   correction_done_bank,
    input  logic [TAG_BITS-1:0]          correction_done_window_tag,

    output logic [BANKS-1:0]             observed_bank_free,
    output logic [BANKS-1:0]             observed_bank_fill,
    output logic [BANKS-1:0]             observed_bank_filled,
    output logic [BANKS-1:0]             observed_bank_pwp,
    output logic [BANKS-1:0]             observed_bank_wait_correction,
    output logic [BANKS-1:0]             observed_bank_correction,
    output logic                         observed_window_open,
    output logic                         observed_pwp_busy,
    output logic                         observed_correction_busy,
    output logic                         protocol_error,
    output logic                         busy
);
    typedef enum logic [2:0] {
        BANK_FREE = 3'd0,
        BANK_FILL = 3'd1,
        BANK_FILLED = 3'd2,
        BANK_PWP = 3'd3,
        BANK_WAIT_CORRECTION = 3'd4,
        BANK_CORRECTION = 3'd5
    } bank_state_t;

    bank_state_t bank_state_q [0:BANKS-1];
    logic [TAG_BITS-1:0] bank_tag_q [0:BANKS-1];
    logic [SEQUENCE_BITS-1:0] bank_sequence_q [0:BANKS-1];
    logic [SEQUENCE_BITS-1:0] next_sequence_q;

    logic request_fault_q;
    logic window_open_q;
    logic [1:0] fill_bank_q;
    logic [TAG_BITS-1:0] fill_tag_q;

    logic mask_valid_q;
    logic [15:0] mask_q [0:7];
    logic [15:0] negate_mask_q [0:7];
    logic [ROW_BITS-1:0] mask_row_q;
    logic mask_window_end_q;

    logic pwp_busy_q;
    logic [1:0] pwp_active_bank_q;
    logic [TAG_BITS-1:0] pwp_active_tag_q;
    logic correction_busy_q;
    logic [1:0] correction_active_bank_q;
    logic [TAG_BITS-1:0] correction_active_tag_q;

    logic free_bank_valid;
    logic [1:0] free_bank;
    logic pwp_candidate_valid;
    logic [1:0] pwp_candidate_bank;
    logic [SEQUENCE_BITS-1:0] pwp_candidate_sequence;
    logic correction_candidate_valid;
    logic [1:0] correction_candidate_bank;
    logic [SEQUENCE_BITS-1:0] correction_candidate_sequence;

    logic [15:0] descriptor_selected_mask;
    logic [15:0] descriptor_remaining_mask;
    logic descriptor_other_blocks_nonzero;
    logic [2:0] descriptor_source_count;
    logic [2:0] descriptor_selected_block;
    logic row_any_source;
    logic row_dirty_negate;
    logic descriptor_fire;
    logic descriptor_slot_available;
    logic row_audit_enable;
    logic row_protocol_valid;
    logic row_has_bank_capacity;
    logic illegal_row;
    logic illegal_pwp_done;
    logic illegal_correction_done;
    logic illegal_event;
    logic quarantine;

`ifndef SYNTHESIS
    initial begin
        if (TAG_BITS < 1 || ROW_BITS < 1 || SEQUENCE_BITS < 18
                || (BANKS != 3 && BANKS != 4))
            $fatal(1, "M142 parameter contract drift");
    end
`endif

    always_comb begin : observe_banks_and_select_free
        observed_bank_free = '0;
        observed_bank_fill = '0;
        observed_bank_filled = '0;
        observed_bank_pwp = '0;
        observed_bank_wait_correction = '0;
        observed_bank_correction = '0;
        free_bank_valid = 1'b0;
        free_bank = '0;
        for (int bank = 0; bank < BANKS; bank++) begin
            case (bank_state_q[bank])
                BANK_FREE: begin
                    observed_bank_free[bank] = 1'b1;
                    if (!free_bank_valid) begin
                        free_bank_valid = 1'b1;
                        free_bank = bank[1:0];
                    end
                end
                BANK_FILL: observed_bank_fill[bank] = 1'b1;
                BANK_FILLED: observed_bank_filled[bank] = 1'b1;
                BANK_PWP: observed_bank_pwp[bank] = 1'b1;
                BANK_WAIT_CORRECTION:
                    observed_bank_wait_correction[bank] = 1'b1;
                BANK_CORRECTION:
                    observed_bank_correction[bank] = 1'b1;
                default: begin end
            endcase
        end
    end

    always_comb begin : canonical_k4_extract
        logic [15:0] work_mask;
        logic found;
        logic block_found;
        logic [3:0] slot_valid;
        descriptor_selected_mask = '0;
        descriptor_source_count = '0;
        descriptor_selected_block = '0;
        descriptor_other_blocks_nonzero = 1'b0;
        descriptor_negate = '0;
        slot_valid = '0;
        for (int slot = 0; slot < 4; slot++)
            descriptor_source[slot] = '0;
        block_found = 1'b0;
        for (int block = 0; block < 8; block++) begin
            if (!block_found && mask_q[block] != 0) begin
                descriptor_selected_block = block[2:0];
                block_found = 1'b1;
            end
        end
        work_mask = mask_q[descriptor_selected_block];
        for (int slot = 0; slot < 4; slot++) begin
            found = 1'b0;
            for (int source = 0; source < 16; source++) begin
                if (!found && work_mask[source]) begin
                    descriptor_source[slot] = source[3:0];
                    descriptor_negate[slot]
                        = negate_mask_q[descriptor_selected_block][source];
                    descriptor_selected_mask[source] = 1'b1;
                    work_mask[source] = 1'b0;
                    slot_valid[slot] = 1'b1;
                    found = 1'b1;
                end
            end
        end
        descriptor_source_count
            = {2'b0, slot_valid[0]} + {2'b0, slot_valid[1]}
              + {2'b0, slot_valid[2]} + {2'b0, slot_valid[3]};
        descriptor_remaining_mask
            = mask_q[descriptor_selected_block]
              & ~descriptor_selected_mask;
        for (int block = 0; block < 8; block++) begin
            if (block != descriptor_selected_block
                    && mask_q[block] != 0)
                descriptor_other_blocks_nonzero = 1'b1;
        end
    end

    always_comb begin : audit_raw_row
        row_any_source = 1'b0;
        row_dirty_negate = 1'b0;
        for (int block = 0; block < 8; block++) begin
            if (row_source_mask[block] != 0)
                row_any_source = 1'b1;
            if ((row_negate_mask[block] & ~row_source_mask[block]) != 0)
                row_dirty_negate = 1'b1;
        end
    end

    assign descriptor_valid = !rst_core && mask_valid_q && !quarantine;
    assign descriptor_bank = fill_bank_q;
    assign descriptor_window_tag = fill_tag_q;
    assign descriptor_row = mask_row_q;
    assign descriptor_block = descriptor_selected_block;
    assign descriptor_source_count_m1 = descriptor_source_count[1:0] - 1'b1;
    assign descriptor_row_last = descriptor_remaining_mask == 0
                               && !descriptor_other_blocks_nonzero;
    assign descriptor_window_last = mask_window_end_q && descriptor_row_last;
    assign descriptor_accept = descriptor_valid && descriptor_ready;
    assign descriptor_fire = descriptor_accept;

    // A new row may replace a row whose final descriptor is accepted on this
    // edge.  A closing row cannot simultaneously open a new window; the bank
    // ownership transition remains explicit for one cycle.
    assign descriptor_slot_available = !mask_valid_q
        || (descriptor_fire && descriptor_row_last && !mask_window_end_q);
    assign row_audit_enable = !mask_valid_q
        || (descriptor_ready && descriptor_row_last && !mask_window_end_q);
    assign row_protocol_valid = window_open_q
        ? (!row_window_start && row_window_tag == fill_tag_q)
        : row_window_start;
    assign row_has_bank_capacity = window_open_q || free_bank_valid;
    assign row_ready = !rst_core && !quarantine && descriptor_slot_available
        && row_has_bank_capacity
        && (!row_valid || (row_protocol_valid
                           && !row_dirty_negate));
    assign row_accept = row_valid && row_ready;
    // Format/protocol faults are independent of ready/capacity.  Keeping this
    // cone independent of descriptor_fire prevents quarantine from feeding
    // back through valid/ready during synthesis.
    assign illegal_row = row_valid && row_audit_enable
        && row_has_bank_capacity
        && (!row_protocol_valid
            || row_dirty_negate);

    always_comb begin : select_oldest_pwp_bank
        pwp_candidate_valid = 1'b0;
        pwp_candidate_bank = '0;
        pwp_candidate_sequence = '1;
        for (int bank = 0; bank < BANKS; bank++) begin
            if (bank_state_q[bank] == BANK_FILLED
                    && (!pwp_candidate_valid
                        || bank_sequence_q[bank]
                           < pwp_candidate_sequence)) begin
                pwp_candidate_valid = 1'b1;
                pwp_candidate_bank = bank[1:0];
                pwp_candidate_sequence = bank_sequence_q[bank];
            end
        end
    end

    always_comb begin : select_oldest_correction_bank
        correction_candidate_valid = 1'b0;
        correction_candidate_bank = '0;
        correction_candidate_sequence = '1;
        for (int bank = 0; bank < BANKS; bank++) begin
            if (bank_state_q[bank] == BANK_WAIT_CORRECTION
                    && (!correction_candidate_valid
                        || bank_sequence_q[bank]
                           < correction_candidate_sequence)) begin
                correction_candidate_valid = 1'b1;
                correction_candidate_bank = bank[1:0];
                correction_candidate_sequence = bank_sequence_q[bank];
            end
        end
    end

    assign pwp_valid = !rst_core && !quarantine && !pwp_busy_q
                     && pwp_candidate_valid;
    assign pwp_bank = pwp_candidate_bank;
    assign pwp_window_tag = bank_tag_q[pwp_candidate_bank];
    assign pwp_accept = pwp_valid && pwp_ready;

    assign correction_valid = !rst_core && !quarantine
                            && !correction_busy_q
                            && correction_candidate_valid;
    assign correction_bank = correction_candidate_bank;
    assign correction_window_tag = bank_tag_q[correction_candidate_bank];
    assign correction_accept = correction_valid && correction_ready;

    assign illegal_pwp_done = pwp_done_valid
        && (!pwp_busy_q
            || pwp_done_bank != pwp_active_bank_q
            || pwp_done_window_tag != pwp_active_tag_q);
    assign illegal_correction_done = correction_done_valid
        && (!correction_busy_q
            || correction_done_bank != correction_active_bank_q
            || correction_done_window_tag != correction_active_tag_q);
    assign illegal_event = illegal_row || illegal_pwp_done
                         || illegal_correction_done;
    assign quarantine = request_fault_q || illegal_event;
    assign protocol_error = !rst_core && quarantine;

    assign observed_window_open = window_open_q;
    assign observed_pwp_busy = pwp_busy_q;
    assign observed_correction_busy = correction_busy_q;
    assign busy = window_open_q || mask_valid_q || pwp_busy_q
                || correction_busy_q || !(&observed_bank_free);

    always_ff @(posedge clk_core) begin : state_update
        if (rst_core) begin
            request_fault_q <= 1'b0;
            window_open_q <= 1'b0;
            fill_bank_q <= '0;
            fill_tag_q <= '0;
            mask_valid_q <= 1'b0;
            mask_row_q <= '0;
            mask_window_end_q <= 1'b0;
            next_sequence_q <= '0;
            pwp_busy_q <= 1'b0;
            pwp_active_bank_q <= '0;
            pwp_active_tag_q <= '0;
            correction_busy_q <= 1'b0;
            correction_active_bank_q <= '0;
            correction_active_tag_q <= '0;
            for (int bank = 0; bank < BANKS; bank++) begin
                bank_state_q[bank] <= BANK_FREE;
                bank_tag_q[bank] <= '0;
                bank_sequence_q[bank] <= '0;
            end
            for (int block = 0; block < 8; block++) begin
                mask_q[block] <= '0;
                negate_mask_q[block] <= '0;
            end
        end else begin
            if (illegal_event)
                request_fault_q <= 1'b1;

            if (!quarantine) begin
                if (descriptor_fire) begin
                    if (descriptor_row_last) begin
                        mask_valid_q <= 1'b0;
                        for (int block = 0; block < 8; block++) begin
                            mask_q[block] <= '0;
                            negate_mask_q[block] <= '0;
                        end
                        if (mask_window_end_q) begin
                            window_open_q <= 1'b0;
                            bank_state_q[fill_bank_q] <= BANK_FILLED;
                        end
                    end else begin
                        mask_q[descriptor_selected_block]
                            <= descriptor_remaining_mask;
                        negate_mask_q[descriptor_selected_block]
                            <= negate_mask_q[descriptor_selected_block]
                               & descriptor_remaining_mask;
                    end
                end

                if (row_accept) begin
                    logic [1:0] accepted_bank;
                    accepted_bank = window_open_q ? fill_bank_q : free_bank;
                    if (!window_open_q) begin
                        fill_bank_q <= free_bank;
                        fill_tag_q <= row_window_tag;
                        bank_tag_q[free_bank] <= row_window_tag;
                        bank_sequence_q[free_bank] <= next_sequence_q;
                        next_sequence_q <= next_sequence_q + 1'b1;
                        bank_state_q[free_bank] <= BANK_FILL;
                        window_open_q <= 1'b1;
                    end

                    if (!row_any_source) begin
                        mask_valid_q <= 1'b0;
                        if (row_window_end) begin
                            window_open_q <= 1'b0;
                            bank_state_q[accepted_bank] <= BANK_FILLED;
                        end
                    end else begin
                        mask_valid_q <= 1'b1;
                        for (int block = 0; block < 8; block++) begin
                            mask_q[block] <= row_source_mask[block];
                            negate_mask_q[block] <= row_negate_mask[block];
                        end
                        mask_row_q <= row_id;
                        mask_window_end_q <= row_window_end;
                    end
                end

                if (pwp_accept) begin
                    pwp_busy_q <= 1'b1;
                    pwp_active_bank_q <= pwp_bank;
                    pwp_active_tag_q <= pwp_window_tag;
                    bank_state_q[pwp_bank] <= BANK_PWP;
                end
                if (pwp_done_valid) begin
                    pwp_busy_q <= 1'b0;
                    bank_state_q[pwp_active_bank_q]
                        <= BANK_WAIT_CORRECTION;
                end

                if (correction_accept) begin
                    correction_busy_q <= 1'b1;
                    correction_active_bank_q <= correction_bank;
                    correction_active_tag_q <= correction_window_tag;
                    bank_state_q[correction_bank] <= BANK_CORRECTION;
                end
                if (correction_done_valid) begin
                    correction_busy_q <= 1'b0;
                    bank_state_q[correction_active_bank_q] <= BANK_FREE;
                end
            end
        end
    end
endmodule

`default_nettype wire
