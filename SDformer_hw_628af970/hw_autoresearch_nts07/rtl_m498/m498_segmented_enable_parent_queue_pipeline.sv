`timescale 1ns/1ps
`default_nettype none

// A synthesis-only explicit buffer leaf is used because M479 showed that
// replicated wires and keep attributes are legally collapsed back into a
// single high-fanout clock-enable net by Design Compiler.  VCS sees the exact
// zero-latency Boolean identity; the TSMC28 DC point sees a preserved BUFFD1.
module m498_physical_enable_buffer (
    input  logic enable_in,
    output logic enable_out
);
`ifdef SYNTHESIS
    (* dont_touch = "true" *) BUFFD1BWP35P140 u_physical_buffer (
        .I(enable_in),
        .Z(enable_out)
    );
`else
    assign enable_out = enable_in;
`endif
endmodule

// M476 removes M474's same-cycle consume->prefetch capacity dependency.
// Two compacted parent-response slots reserve space for one synchronous read
// in flight, so a current issue ID can never sit on the next-prefetch ready
// path.  A same-address final-write/prefetch is still forwarded exactly.
// The external parent scratch is 64 words x 1152 bits (9 KiB at LANES=96),
// not a 144-byte total memory.  This module contains only the two 144-byte
// response slots; the 64-word 1R1W scratch and resident psum store are cuts.
module m498_segmented_enable_parent_queue_pipeline #(
    parameter int LANES = 96,
    parameter int ROW_BITS = 6
) (
    input  logic clk_core,
    input  logic reset_n,

    input  logic prefetch_valid,
    output logic prefetch_ready,
    input  logic [ROW_BITS-1:0] prefetch_parent_id,
    output logic scratch_read_enable,
    output logic [ROW_BITS-1:0] scratch_read_address,
    input  logic [LANES*12-1:0] scratch_read_data,

    input  logic issue_valid,
    output logic issue_ready,
    input  logic [ROW_BITS-1:0] issue_row_id,
    input  logic issue_first,
    input  logic issue_last,
    input  logic issue_parent_valid,
    input  logic [ROW_BITS-1:0] issue_parent_id,
    input  logic [LANES*12-1:0] issue_residual_data,
    input  logic [LANES*19-1:0] issue_psum_prior,

    output logic scratch_write_enable,
    output logic [ROW_BITS-1:0] scratch_write_address,
    output logic [LANES*12-1:0] scratch_write_data,
    output logic psum_write_valid,
    input  logic psum_write_ready,
    output logic [ROW_BITS-1:0] psum_write_address,
    output logic [LANES*19-1:0] psum_write_data,
    output logic row_complete,

    output logic protocol_error,
    output logic row_active,
    output logic [1:0] parent_queue_occupancy,
    output logic parent_queue_full,
    output logic debug_forward_event,
    output logic debug_scratch_read_event,
    output logic debug_read_response_event,
    output logic debug_dual_enqueue_event,
    output logic debug_overflow_block_event
);
    logic fault_q;
    logic row_active_q;
    logic [ROW_BITS-1:0] row_id_q, row_parent_id_q;
    logic row_parent_valid_q;
    logic signed [12:0] row_acc_q [0:LANES-1];
    logic signed [19:0] psum_acc_q [0:LANES-1];

    logic slot0_valid_q, slot1_valid_q;
    logic [ROW_BITS-1:0] slot0_id_q, slot1_id_q;
    logic [LANES*12-1:0] slot0_data_q, slot1_data_q;
    logic slot0_valid_n, slot1_valid_n;
    logic [ROW_BITS-1:0] slot0_id_n, slot1_id_n;
    logic [LANES*12-1:0] slot0_data_n, slot1_data_n;

    logic read_pending_q;
    logic [ROW_BITS-1:0] read_pending_id_q;

    logic signed [13:0] row_partial_w [0:LANES-1];
    logic signed [13:0] row_final_w [0:LANES-1];
    logic signed [20:0] psum_final_w [0:LANES-1];
    logic row_overflow_w, psum_overflow_w;
    logic metadata_ok_w, parent_ready_w, final_outputs_ready_w;
    logic issue_accept_w, consume_parent_w;
    logic lane_commit_root_w;
    localparam int ENABLE_GROUP_LANES = 8;
    localparam int ENABLE_GROUPS =
        (LANES + ENABLE_GROUP_LANES - 1) / ENABLE_GROUP_LANES;
    logic [ENABLE_GROUPS-1:0] group_enable_w;
    logic [LANES-1:0] row_lane_enable_w;
    logic [LANES-1:0] psum_lane_enable_w;
    logic prefetch_accept_w, forward_match_w;
    logic [1:0] queue_count_w, reserved_count_w;
    logic [LANES*12-1:0] row_final_packed_w;

    integer lane_comb;
    integer lane_ff;

    // The root drives only one branch buffer per eight lanes.  Each branch
    // drives sixteen leaf buffers (row + psum for eight lanes), and each leaf
    // drives at most 13 or 20 state bits.  Thus every deliberate enable-tree
    // electrical fanout is <=20, below the 32-load contract that M479 missed.
    // The !issue_last term also removes the redundant outer issue_accept guard
    // that allowed M479's nominal lane enables to be optimized away.
    assign lane_commit_root_w = issue_accept_w && !issue_last;
    generate
        for (genvar enable_group = 0;
                enable_group < ENABLE_GROUPS;
                enable_group = enable_group + 1) begin : g_enable_branch
            (* dont_touch = "true" *)
            m498_physical_enable_buffer branch_buffer (
                .enable_in(lane_commit_root_w),
                .enable_out(group_enable_w[enable_group])
            );
            for (genvar enable_lane = 0;
                    enable_lane < ENABLE_GROUP_LANES;
                    enable_lane = enable_lane + 1) begin : g_enable_leaf
                localparam int ABSOLUTE_LANE =
                    enable_group * ENABLE_GROUP_LANES + enable_lane;
                if (ABSOLUTE_LANE < LANES) begin : g_live_lane
                    (* dont_touch = "true" *)
                    m498_physical_enable_buffer row_leaf_buffer (
                        .enable_in(group_enable_w[enable_group]),
                        .enable_out(row_lane_enable_w[ABSOLUTE_LANE])
                    );
                    (* dont_touch = "true" *)
                    m498_physical_enable_buffer psum_leaf_buffer (
                        .enable_in(group_enable_w[enable_group]),
                        .enable_out(psum_lane_enable_w[ABSOLUTE_LANE])
                    );
                end
            end
        end
    endgenerate

    always_comb begin
        row_overflow_w = 1'b0;
        psum_overflow_w = 1'b0;
        row_final_packed_w = '0;
        psum_write_data = '0;
        for (lane_comb = 0; lane_comb < LANES; lane_comb = lane_comb + 1) begin
            row_partial_w[lane_comb] =
                (issue_first ? 14'sd0 : $signed(row_acc_q[lane_comb]))
                + $signed(issue_residual_data[lane_comb*12 +: 12]);
            row_final_w[lane_comb] = row_partial_w[lane_comb]
                + (issue_parent_valid
                    ? $signed(slot0_data_q[lane_comb*12 +: 12])
                    : 14'sd0);
            psum_final_w[lane_comb] =
                (issue_first
                    ? $signed(issue_psum_prior[lane_comb*19 +: 19])
                    : $signed(psum_acc_q[lane_comb]))
                + $signed(issue_residual_data[lane_comb*12 +: 12])
                + ((issue_last && issue_parent_valid)
                    ? $signed(slot0_data_q[lane_comb*12 +: 12])
                    : 21'sd0);
            row_final_packed_w[lane_comb*12 +: 12] =
                row_final_w[lane_comb][11:0];
            psum_write_data[lane_comb*19 +: 19] =
                psum_final_w[lane_comb][18:0];
            if (row_final_w[lane_comb] < -14'sd2048
                    || row_final_w[lane_comb] > 14'sd2047)
                row_overflow_w = 1'b1;
            if (psum_final_w[lane_comb] < -21'sd262144
                    || psum_final_w[lane_comb] > 21'sd262143)
                psum_overflow_w = 1'b1;
        end
    end

    always_comb begin
        queue_count_w = {1'b0, slot0_valid_q} + {1'b0, slot1_valid_q};
        reserved_count_w = queue_count_w + {1'b0, read_pending_q};
        metadata_ok_w = issue_first
            ? !row_active_q
            : row_active_q && issue_row_id == row_id_q
                && issue_parent_valid == row_parent_valid_q
                && (!issue_parent_valid || issue_parent_id == row_parent_id_q);
        parent_ready_w = !issue_parent_valid
            || (slot0_valid_q && slot0_id_q == issue_parent_id);
        final_outputs_ready_w = !issue_last
            || (psum_write_ready && !row_overflow_w && !psum_overflow_w);
        issue_ready = !fault_q && metadata_ok_w && parent_ready_w
            && final_outputs_ready_w;
        issue_accept_w = issue_valid && issue_ready;
        consume_parent_w = issue_accept_w && issue_last && issue_parent_valid;

        scratch_write_enable = issue_accept_w && issue_last;
        scratch_write_address = issue_row_id;
        scratch_write_data = row_final_packed_w;
        psum_write_valid = !fault_q && issue_valid && metadata_ok_w
            && parent_ready_w && issue_last
            && !row_overflow_w && !psum_overflow_w;
        psum_write_address = issue_row_id;
        row_complete = scratch_write_enable;

        // Deliberately do not credit a same-cycle consume.  At occupancy two,
        // prefetch stalls for one cycle while issue can still consume slot0.
        // This severs M474's issue_parent_id -> consume -> prefetch path.
        prefetch_ready = !fault_q && reserved_count_w < 2;
        prefetch_accept_w = prefetch_valid && prefetch_ready;
        forward_match_w = prefetch_accept_w && scratch_write_enable
            && prefetch_parent_id == scratch_write_address;
        scratch_read_enable = prefetch_accept_w && !forward_match_w;
        scratch_read_address = prefetch_parent_id;
    end

    // Compact queue next-state: optional head pop, then the returning macro
    // response, then an optional same-cycle RAW-forwarded word.  The reserved
    // capacity rule guarantees at most two entries without using pop credit.
    always_comb begin
        slot0_valid_n = slot0_valid_q;
        slot0_id_n = slot0_id_q;
        slot0_data_n = slot0_data_q;
        slot1_valid_n = slot1_valid_q;
        slot1_id_n = slot1_id_q;
        slot1_data_n = slot1_data_q;

        if (consume_parent_w) begin
            slot0_valid_n = slot1_valid_q;
            slot0_id_n = slot1_id_q;
            slot0_data_n = slot1_data_q;
            slot1_valid_n = 1'b0;
        end

        if (read_pending_q) begin
            if (!slot0_valid_n) begin
                slot0_valid_n = 1'b1;
                slot0_id_n = read_pending_id_q;
                slot0_data_n = scratch_read_data;
            end else begin
                slot1_valid_n = 1'b1;
                slot1_id_n = read_pending_id_q;
                slot1_data_n = scratch_read_data;
            end
        end

        if (forward_match_w) begin
            if (!slot0_valid_n) begin
                slot0_valid_n = 1'b1;
                slot0_id_n = prefetch_parent_id;
                slot0_data_n = scratch_write_data;
            end else begin
                slot1_valid_n = 1'b1;
                slot1_id_n = prefetch_parent_id;
                slot1_data_n = scratch_write_data;
            end
        end
    end

    assign protocol_error = fault_q;
    assign row_active = row_active_q;
    assign parent_queue_occupancy = queue_count_w;
    assign parent_queue_full = slot0_valid_q && slot1_valid_q;
    assign debug_forward_event = forward_match_w;
    assign debug_scratch_read_event = scratch_read_enable;
    assign debug_read_response_event = read_pending_q;
    assign debug_dual_enqueue_event = read_pending_q && forward_match_w;
    assign debug_overflow_block_event = !fault_q && issue_valid
        && metadata_ok_w && parent_ready_w && issue_last
        && (row_overflow_w || psum_overflow_w);

    always_ff @(posedge clk_core or negedge reset_n) begin
        if (!reset_n) begin
            fault_q <= 1'b0;
            row_active_q <= 1'b0;
            row_id_q <= '0;
            row_parent_valid_q <= 1'b0;
            row_parent_id_q <= '0;
            slot0_valid_q <= 1'b0;
            slot0_id_q <= '0;
            slot0_data_q <= '0;
            slot1_valid_q <= 1'b0;
            slot1_id_q <= '0;
            slot1_data_q <= '0;
            read_pending_q <= 1'b0;
            read_pending_id_q <= '0;
            for (lane_ff = 0; lane_ff < LANES; lane_ff = lane_ff + 1) begin
                row_acc_q[lane_ff] <= '0;
                psum_acc_q[lane_ff] <= '0;
            end
        end else begin
            if (issue_valid && !fault_q && !metadata_ok_w)
                fault_q <= 1'b1;
            if (issue_valid && !fault_q && metadata_ok_w
                    && issue_parent_valid && !parent_ready_w)
                fault_q <= 1'b1;
            if (issue_valid && !fault_q && metadata_ok_w && parent_ready_w
                    && issue_last && (row_overflow_w || psum_overflow_w))
                fault_q <= 1'b1;
            if (slot1_valid_q && !slot0_valid_q)
                fault_q <= 1'b1;

            slot0_valid_q <= slot0_valid_n;
            slot0_id_q <= slot0_id_n;
            slot0_data_q <= slot0_data_n;
            slot1_valid_q <= slot1_valid_n;
            slot1_id_q <= slot1_id_n;
            slot1_data_q <= slot1_data_n;

            read_pending_q <= scratch_read_enable;
            if (scratch_read_enable)
                read_pending_id_q <= scratch_read_address;

            if (issue_accept_w) begin
                if (issue_first) begin
                    row_id_q <= issue_row_id;
                    row_parent_valid_q <= issue_parent_valid;
                    row_parent_id_q <= issue_parent_id;
                end
                if (issue_last) begin
                    row_active_q <= 1'b0;
                end else begin
                    row_active_q <= 1'b1;
                end
            end
            for (lane_ff = 0; lane_ff < LANES; lane_ff = lane_ff + 1) begin
                if (row_lane_enable_w[lane_ff])
                    row_acc_q[lane_ff] <= row_partial_w[lane_ff][12:0];
                if (psum_lane_enable_w[lane_ff])
                    psum_acc_q[lane_ff] <= psum_final_w[lane_ff][19:0];
            end
        end
    end
endmodule

`default_nettype wire
