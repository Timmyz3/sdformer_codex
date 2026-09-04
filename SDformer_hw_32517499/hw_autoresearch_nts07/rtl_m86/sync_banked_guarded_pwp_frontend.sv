`timescale 1ns/1ps
`default_nettype none

// M86 replaces M85's combinational bank-data assumption with an explicit
// one-cycle, eight-bank synchronous read path.  One aggregate DMA row writes
// one word into each bank; descriptors then autonomously issue the 3/4/4/5
// reads required by signed8/9/10/11 PWP records.  A four-entry response FIFO
// decouples the SRAM response from M85/M82 output backpressure.
module sync_banked_guarded_pwp_frontend #(
    parameter int ROWS = 460,
    parameter int ROW_W = 10,
    parameter int TAG_W = 32,
    parameter int FIFO_DEPTH = 4
) (
    input  logic                     clk_core,
    input  logic                     rst_core,

    input  logic                     payload_load_valid,
    output logic                     payload_load_ready,
    input  logic [ROW_W-1:0]         payload_load_row,
    input  logic [255:0]             payload_load_words,
    output logic                     payload_load_accept,

    input  logic                     phase_load_valid,
    output logic                     phase_load_ready,
    input  logic [591:0]             phase_metadata,
    output logic                     phase_loaded,
    output logic                     metadata_error,

    input  logic                     descriptor_valid,
    output logic                     descriptor_ready,
    input  logic [3:0]               descriptor_pattern,
    input  logic [2:0]               descriptor_block,
    input  logic [TAG_W-1:0]         descriptor_tag,
    output logic                     descriptor_accept,

    output logic                     output_valid,
    input  logic                     output_ready,
    output logic [TAG_W-1:0]         output_tag,
    output logic [3:0]               output_width,
    output logic                     output_escape,
    output logic [96*12-1:0]         output_values,
    output logic                     output_accept,

    output logic                     protocol_error,
    output logic                     busy,
    output logic                     bank_read_issue,
    output logic [2:0]               bank_read_beat,
    output logic                     bank_response_enqueue,
    output logic [2:0]               response_fifo_level
);
    logic [31:0] bank_mem [0:7][0:ROWS-1];
    logic [ROWS-1:0] row_written_q;
    logic [591:0] metadata_q;
    logic phase_committed_q, frontend_fault_q;

    logic active_q;
    logic [3:0] active_pattern_q;
    logic [2:0] active_block_q, active_beat_q;
    logic [TAG_W-1:0] active_tag_q;

    logic [2:0] issue_code, issue_beats;
    logic issue_escape, issue_last, issue_valid, issue_capacity;
    logic [13:0] issue_prefix_words, issue_logical_word;
    logic [2:0] issue_base_bank;
    logic [ROW_W-1:0] issue_base_row;
    logic [8*ROW_W-1:0] issue_rows;
    integer issue_prior_code;

    logic rd_pending_q;
    logic rd_escape_q;
    logic [3:0] rd_pattern_q;
    logic [2:0] rd_block_q, rd_beat_q;
    logic [TAG_W-1:0] rd_tag_q;
    logic [8*ROW_W-1:0] rd_rows_q;

    localparam int FIFO_PTR_W = $clog2(FIFO_DEPTH);
    logic [255:0] fifo_bank_words_q [0:FIFO_DEPTH-1];
    logic [3:0] fifo_pattern_q [0:FIFO_DEPTH-1];
    logic [2:0] fifo_block_q [0:FIFO_DEPTH-1];
    logic [2:0] fifo_beat_q [0:FIFO_DEPTH-1];
    logic [TAG_W-1:0] fifo_tag_q [0:FIFO_DEPTH-1];
    logic [FIFO_PTR_W-1:0] fifo_read_ptr_q, fifo_write_ptr_q;
    logic [FIFO_PTR_W:0] fifo_count_q;
    logic fifo_push, fifo_pop;

    logic m85_phase_load_ready, m85_phase_loaded, m85_metadata_error;
    logic m85_lookup_ready, m85_protocol_error, m85_busy;
    logic [8*ROW_W-1:0] unused_bank_rows;

`ifndef SYNTHESIS
    initial begin
        if (ROWS != 460 || ROW_W != 10 || TAG_W != 32 || FIFO_DEPTH != 4)
            $fatal(1, "M86 frozen synchronous bank geometry drift");
    end
`endif

    function automatic integer words_for_code(input integer code);
        case (code)
            0: words_for_code = 24;
            1: words_for_code = 27;
            2: words_for_code = 30;
            3: words_for_code = 33;
            4: words_for_code = 0;
            default: words_for_code = 0;
        endcase
    endfunction

    assign fifo_push = rd_pending_q;
    assign fifo_pop = (fifo_count_q != 0) && m85_lookup_ready;
    // A response already in flight consumes capacity.  A same-edge pop may
    // be credited because the FIFO read and write ports are independent.
    always_comb begin
        integer occupied_after_pop;
        occupied_after_pop = fifo_count_q + (rd_pending_q ? 1 : 0)
                           - (fifo_pop ? 1 : 0);
        issue_capacity = occupied_after_pop < FIFO_DEPTH;
    end

    always_comb begin : descriptor_address_generation
        issue_code = metadata_q[
            (active_pattern_q*8 + active_block_q)*3 +: 3];
        issue_prefix_words = {1'b0, metadata_q[
            384 + active_pattern_q*13 +: 13]};
        for (int prior = 0; prior < 8; prior++) begin
            if (prior < active_block_q) begin
                issue_prior_code = metadata_q[
                    (active_pattern_q*8 + prior)*3 +: 3];
                issue_prefix_words = issue_prefix_words
                                   + words_for_code(issue_prior_code);
            end
        end
        case (issue_code)
            3'd0: issue_beats = 3;
            3'd1, 3'd2: issue_beats = 4;
            3'd3: issue_beats = 5;
            default: issue_beats = 0;
        endcase
        issue_escape = issue_code == 3'd4;
        issue_last = issue_escape || active_beat_q + 1'b1 == issue_beats;
        issue_logical_word = issue_prefix_words + active_beat_q*8;
        issue_base_bank = issue_logical_word[2:0];
        issue_base_row = issue_logical_word[12:3];
        issue_rows = '0;
        for (int bank = 0; bank < 8; bank++)
            issue_rows[bank*ROW_W +: ROW_W] =
                issue_base_row + (bank < issue_base_bank);
        issue_valid = active_q && issue_capacity && !frontend_fault_q
                    && !m85_metadata_error && issue_code <= 4
                    && (issue_escape || issue_base_row < ROWS);
    end

    assign bank_read_issue = issue_valid;
    assign bank_read_beat = active_beat_q;
    assign bank_response_enqueue = fifo_push;
    assign response_fifo_level = fifo_count_q;
    assign descriptor_ready = phase_committed_q && m85_phase_loaded
                            && !m85_metadata_error && !frontend_fault_q
                            && !payload_load_valid
                            && (!active_q || (issue_valid && issue_last));
    assign descriptor_accept = descriptor_valid && descriptor_ready;

    assign payload_load_ready = !active_q && !rd_pending_q
                              && fifo_count_q == 0 && !m85_busy
                              && !descriptor_valid && !phase_load_valid
                              && payload_load_row < ROWS;
    assign payload_load_accept = payload_load_valid && payload_load_ready;
    assign phase_load_ready = !active_q && !rd_pending_q
                            && fifo_count_q == 0 && !m85_busy
                            && (&row_written_q) && m85_phase_load_ready;
    assign phase_loaded = phase_committed_q && m85_phase_loaded;
    assign metadata_error = m85_metadata_error;
    assign protocol_error = frontend_fault_q || m85_protocol_error;
    assign busy = active_q || rd_pending_q || fifo_count_q != 0 || m85_busy;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            row_written_q <= '0;
            metadata_q <= '0;
            phase_committed_q <= 1'b0;
            frontend_fault_q <= 1'b0;
            active_q <= 1'b0;
            active_pattern_q <= '0;
            active_block_q <= '0;
            active_beat_q <= '0;
            active_tag_q <= '0;
            rd_pending_q <= 1'b0;
            rd_escape_q <= 1'b0;
            rd_pattern_q <= '0;
            rd_block_q <= '0;
            rd_beat_q <= '0;
            rd_tag_q <= '0;
            rd_rows_q <= '0;
            fifo_read_ptr_q <= '0;
            fifo_write_ptr_q <= '0;
            fifo_count_q <= '0;
        end else begin
            if (payload_load_valid && payload_load_row >= ROWS)
                frontend_fault_q <= 1'b1;
            if (payload_load_accept) begin
                if (row_written_q[payload_load_row])
                    frontend_fault_q <= 1'b1;
                for (int bank = 0; bank < 8; bank++)
                    bank_mem[bank][payload_load_row] <=
                        payload_load_words[bank*32 +: 32];
                row_written_q[payload_load_row] <= 1'b1;
                phase_committed_q <= 1'b0;
            end
            if (phase_load_valid && phase_load_ready) begin
                metadata_q <= phase_metadata;
                phase_committed_q <= 1'b1;
                row_written_q <= '0;
            end

            if (descriptor_accept && (!active_q
                    || (issue_valid && issue_last))) begin
                active_q <= 1'b1;
                active_pattern_q <= descriptor_pattern;
                active_block_q <= descriptor_block;
                active_beat_q <= '0;
                active_tag_q <= descriptor_tag;
            end else if (issue_valid) begin
                if (issue_last) begin
                    active_q <= 1'b0;
                    active_beat_q <= '0;
                end else begin
                    active_beat_q <= active_beat_q + 1'b1;
                end
            end

            rd_pending_q <= issue_valid;
            if (issue_valid) begin
                rd_escape_q <= issue_escape;
                rd_pattern_q <= active_pattern_q;
                rd_block_q <= active_block_q;
                rd_beat_q <= active_beat_q;
                rd_tag_q <= active_beat_q == 0 ? active_tag_q : '0;
                rd_rows_q <= issue_rows;
            end

            if (fifo_push) begin
                fifo_pattern_q[fifo_write_ptr_q] <= rd_pattern_q;
                fifo_block_q[fifo_write_ptr_q] <= rd_block_q;
                fifo_beat_q[fifo_write_ptr_q] <= rd_beat_q;
                fifo_tag_q[fifo_write_ptr_q] <= rd_tag_q;
                for (int bank = 0; bank < 8; bank++) begin
                    if (rd_escape_q)
                        fifo_bank_words_q[fifo_write_ptr_q][bank*32 +: 32]
                            <= '0;
                    else
                        fifo_bank_words_q[fifo_write_ptr_q][bank*32 +: 32]
                            <= bank_mem[bank][rd_rows_q[
                                bank*ROW_W +: ROW_W]];
                end
                fifo_write_ptr_q <= fifo_write_ptr_q + 1'b1;
            end
            if (fifo_pop)
                fifo_read_ptr_q <= fifo_read_ptr_q + 1'b1;
            case ({fifo_push, fifo_pop})
                2'b10: fifo_count_q <= fifo_count_q + 1'b1;
                2'b01: fifo_count_q <= fifo_count_q - 1'b1;
                default: fifo_count_q <= fifo_count_q;
            endcase
        end
    end

    guarded_wordpacked_pwp_stream m85_stream (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .phase_load_valid(phase_load_valid && phase_load_ready),
        .phase_load_ready(m85_phase_load_ready),
        .phase_metadata(phase_metadata),
        .phase_loaded(m85_phase_loaded),
        .metadata_error(m85_metadata_error),
        .lookup_valid(fifo_count_q != 0),
        .lookup_ready(m85_lookup_ready),
        .lookup_pattern(fifo_pattern_q[fifo_read_ptr_q]),
        .lookup_block(fifo_block_q[fifo_read_ptr_q]),
        .lookup_beat(fifo_beat_q[fifo_read_ptr_q]),
        .lookup_tag(fifo_tag_q[fifo_read_ptr_q]),
        .bank_words(fifo_bank_words_q[fifo_read_ptr_q]),
        .bank_row_addresses(unused_bank_rows),
        .output_valid(output_valid),
        .output_ready(output_ready),
        .output_tag(output_tag),
        .output_width(output_width),
        .output_escape(output_escape),
        .output_values(output_values),
        .output_accept(output_accept),
        .protocol_error(m85_protocol_error),
        .busy(m85_busy)
    );
endmodule

`default_nettype wire
