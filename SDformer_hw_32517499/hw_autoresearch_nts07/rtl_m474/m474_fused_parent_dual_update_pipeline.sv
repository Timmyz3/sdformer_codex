`timescale 1ns/1ps
`default_nettype none

// M474 isolates the only physical assumption separating M473's fused and
// unfused cycle points.  A scheduler prefetches the next row's parent through
// a synchronous 144-byte 1R1W scratch cut.  A same-address final-write/read
// hazard is forwarded, so every accepted residual beat remains one work cycle;
// the final beat also writes the signed12 row result and signed19 psum result.
module m474_fused_parent_dual_update_pipeline #(
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
    output logic parent_buffer_valid,
    output logic debug_forward_event,
    output logic debug_scratch_read_event,
    output logic debug_overflow_block_event,
    output logic [31:0] debug_issue_accepts,
    output logic [31:0] debug_row_completions,
    output logic [31:0] debug_forward_hits,
    output logic [31:0] debug_scratch_reads,
    output logic [31:0] debug_stall_cycles
);
    logic fault_q;
    logic row_active_q;
    logic [ROW_BITS-1:0] row_id_q, row_parent_id_q;
    logic row_parent_valid_q;
    logic signed [12:0] row_acc_q [0:LANES-1];
    logic signed [19:0] psum_acc_q [0:LANES-1];

    logic parent_valid_q, read_pending_q;
    logic [ROW_BITS-1:0] parent_id_q, read_pending_id_q;
    logic [LANES*12-1:0] parent_data_q;

    logic signed [13:0] row_partial_w [0:LANES-1];
    logic signed [13:0] row_final_w [0:LANES-1];
    logic signed [20:0] psum_final_w [0:LANES-1];
    logic row_overflow_w, psum_overflow_w;
    logic metadata_ok_w, parent_ready_w, final_outputs_ready_w;
    logic pending_parent_match_w;
    logic issue_accept_w, consume_parent_w, prefetch_space_w;
    logic prefetch_accept_w, forward_match_w;
    logic [LANES*12-1:0] row_final_packed_w;
    logic [LANES*12-1:0] parent_source_data_w;

    logic [31:0] issue_accepts_q, row_completions_q, forward_hits_q;
    logic [31:0] scratch_reads_q, stall_cycles_q;

    integer lane_comb;
    integer lane_ff;
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
                + ((issue_parent_valid)
                    ? $signed(parent_source_data_w[lane_comb*12 +: 12])
                    : 14'sd0);
            psum_final_w[lane_comb] =
                (issue_first
                    ? $signed(issue_psum_prior[lane_comb*19 +: 19])
                    : $signed(psum_acc_q[lane_comb]))
                + $signed(issue_residual_data[lane_comb*12 +: 12])
                + ((issue_last && issue_parent_valid)
                    ? $signed(parent_source_data_w[lane_comb*12 +: 12])
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
        metadata_ok_w = issue_first
            ? !row_active_q
            : row_active_q && issue_row_id == row_id_q
                && issue_parent_valid == row_parent_valid_q
                && (!issue_parent_valid || issue_parent_id == row_parent_id_q);
        pending_parent_match_w = read_pending_q
            && read_pending_id_q == issue_parent_id;
        parent_ready_w = !issue_parent_valid
            || (parent_valid_q && parent_id_q == issue_parent_id)
            || pending_parent_match_w;
        // A synchronous macro's registered Q is available throughout the
        // cycle after scratch_read_enable. Consume that Q directly instead of
        // adding a second capture/issue bubble.
        parent_source_data_w = (parent_valid_q
            && parent_id_q == issue_parent_id)
            ? parent_data_q : scratch_read_data;
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

        prefetch_space_w = (!parent_valid_q && !read_pending_q)
            || consume_parent_w;
        forward_match_w = prefetch_valid && prefetch_space_w
            && scratch_write_enable
            && prefetch_parent_id == scratch_write_address;
        prefetch_ready = !fault_q && prefetch_space_w;
        prefetch_accept_w = prefetch_valid && prefetch_ready;
        scratch_read_enable = prefetch_accept_w && !forward_match_w;
        scratch_read_address = prefetch_parent_id;
    end

    assign protocol_error = fault_q;
    assign row_active = row_active_q;
    assign parent_buffer_valid = parent_valid_q;
    assign debug_forward_event = prefetch_accept_w && forward_match_w;
    assign debug_scratch_read_event = scratch_read_enable;
    assign debug_overflow_block_event = !fault_q && issue_valid
        && metadata_ok_w && parent_ready_w && issue_last
        && (row_overflow_w || psum_overflow_w);
    assign debug_issue_accepts = issue_accepts_q;
    assign debug_row_completions = row_completions_q;
    assign debug_forward_hits = forward_hits_q;
    assign debug_scratch_reads = scratch_reads_q;
    assign debug_stall_cycles = stall_cycles_q;

    always_ff @(posedge clk_core or negedge reset_n) begin
        if (!reset_n) begin
            fault_q <= 1'b0;
            row_active_q <= 1'b0;
            row_id_q <= '0;
            row_parent_valid_q <= 1'b0;
            row_parent_id_q <= '0;
            parent_valid_q <= 1'b0;
            parent_id_q <= '0;
            parent_data_q <= '0;
            read_pending_q <= 1'b0;
            read_pending_id_q <= '0;
            issue_accepts_q <= '0;
            row_completions_q <= '0;
            forward_hits_q <= '0;
            scratch_reads_q <= '0;
            stall_cycles_q <= '0;
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
                    && issue_last
                    && (row_overflow_w || psum_overflow_w))
                fault_q <= 1'b1;

            if (issue_valid && !issue_ready)
                stall_cycles_q <= stall_cycles_q + 1'b1;
            if (issue_accept_w) begin
                issue_accepts_q <= issue_accepts_q + 1'b1;
                if (issue_first) begin
                    row_id_q <= issue_row_id;
                    row_parent_valid_q <= issue_parent_valid;
                    row_parent_id_q <= issue_parent_id;
                end
                if (issue_last) begin
                    row_active_q <= 1'b0;
                    row_completions_q <= row_completions_q + 1'b1;
                end else begin
                    row_active_q <= 1'b1;
                    for (lane_ff = 0; lane_ff < LANES; lane_ff = lane_ff + 1) begin
                        row_acc_q[lane_ff] <= row_partial_w[lane_ff][12:0];
                        psum_acc_q[lane_ff] <= psum_final_w[lane_ff][19:0];
                    end
                end
            end

            // A real synchronous read response arrives exactly one cycle
            // after scratch_read_enable.  A same-cycle write/read hazard is
            // suppressed and captured through forwarding instead.
            read_pending_q <= scratch_read_enable;
            if (scratch_read_enable)
                read_pending_id_q <= scratch_read_address;

            if (consume_parent_w)
                parent_valid_q <= 1'b0;
            if (read_pending_q && !consume_parent_w) begin
                parent_valid_q <= 1'b1;
                parent_id_q <= read_pending_id_q;
                parent_data_q <= scratch_read_data;
            end
            if (prefetch_accept_w && forward_match_w) begin
                parent_valid_q <= 1'b1;
                parent_id_q <= prefetch_parent_id;
                parent_data_q <= scratch_write_data;
                forward_hits_q <= forward_hits_q + 1'b1;
            end
            if (scratch_read_enable)
                scratch_reads_q <= scratch_reads_q + 1'b1;
        end
    end
endmodule

`default_nettype wire
