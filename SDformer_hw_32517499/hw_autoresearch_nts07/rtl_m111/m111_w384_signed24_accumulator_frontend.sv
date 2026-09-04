`timescale 1ns/1ps
`default_nettype none

// W384 x eight-block x 96-lane signed24 accumulator macro frontend.
//
// One global vector update can be accepted per cycle.  Each of the eight
// external banks provides one synchronous read plus one write port.  A small
// valid bitmap clears a window without sweeping the 884736-byte data array.
// Consecutive updates to the same (block,row) are fail-closed because their
// read and the prior write would share an undefined macro RDW edge.  The M108
// ordering contract spaces such updates; other addresses retain II=1.
module m111_w384_signed24_accumulator_frontend #(
    parameter int WIN_ROWS = 384,
    parameter int ROW_W = 9,
    parameter int BANKS = 8,
    parameter int BANK_W = 3,
    parameter int LANES = 96,
    parameter int ACC_BITS = 24,
    parameter int VECTOR_BITS = LANES * ACC_BITS
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         window_start_valid,
    output logic                         window_start_ready,
    output logic                         window_start_accept,

    input  logic                         update_valid,
    output logic                         update_ready,
    input  logic [BANK_W-1:0]            update_block,
    input  logic [ROW_W-1:0]             update_row,
    input  logic [VECTOR_BITS-1:0]       update_delta,
    output logic                         update_accept,

    input  logic                         window_end_valid,
    output logic                         window_end_ready,
    output logic                         window_end_accept,

    output logic                         commit_valid,
    input  logic                         commit_ready,
    output logic [BANK_W-1:0]            commit_block,
    output logic [ROW_W-1:0]             commit_row,
    output logic [VECTOR_BITS-1:0]       commit_data,
    output logic                         commit_last,
    output logic                         window_done,

    output logic [BANKS-1:0]             mem_rd_en,
    output logic [ROW_W-1:0]             mem_rd_addr [0:BANKS-1],
    input  logic [VECTOR_BITS-1:0]       mem_rd_data [0:BANKS-1],
    output logic [BANKS-1:0]             mem_wr_en,
    output logic [ROW_W-1:0]             mem_wr_addr [0:BANKS-1],
    output logic [VECTOR_BITS-1:0]       mem_wr_data [0:BANKS-1],

    output logic                         protocol_error,
    output logic                         window_active,
    output logic                         busy
);
    logic request_fault_q;
    logic window_active_q;
    logic [WIN_ROWS-1:0] row_valid_q [0:BANKS-1];

    logic update_pipe_valid_q;
    logic [BANK_W-1:0] update_pipe_block_q;
    logic [ROW_W-1:0] update_pipe_row_q;
    logic [VECTOR_BITS-1:0] update_pipe_delta_q;
    logic update_pipe_base_valid_q;

    logic commit_active_q;
    logic [BANK_W-1:0] commit_issue_block_q;
    logic [ROW_W-1:0] commit_issue_row_q;
    logic commit_pipe_valid_q;
    logic [BANK_W-1:0] commit_pipe_block_q;
    logic [ROW_W-1:0] commit_pipe_row_q;
    logic commit_pipe_row_valid_q;
    logic commit_pipe_last_q;

    logic update_row_in_range;
    logic same_address_rdw_hazard;
    logic start_semantically_valid;
    logic update_semantically_valid;
    logic end_semantically_valid;
    logic request_collision;
    logic illegal_request;
    logic issue_commit;
    logic update_pipe_overflow;
    logic [VECTOR_BITS-1:0] update_write_vector;
    logic signed [ACC_BITS:0] lane_base_ext [0:LANES-1];
    logic signed [ACC_BITS:0] lane_delta_ext [0:LANES-1];
    logic signed [ACC_BITS:0] lane_sum_ext [0:LANES-1];

`ifndef SYNTHESIS
    initial begin
        if (WIN_ROWS != 384 || ROW_W != 9 || BANKS != 8 || BANK_W != 3
                || LANES != 96 || ACC_BITS != 24 || VECTOR_BITS != 2304)
            $fatal(1, "M111 production geometry drift");
    end
`endif

    always_comb begin : request_audit
        update_row_in_range = update_row < WIN_ROWS;
        same_address_rdw_hazard = update_pipe_valid_q && update_valid
                                && update_block == update_pipe_block_q
                                && update_row == update_pipe_row_q;
        start_semantically_valid = !window_active_q && !commit_active_q
                                 && !commit_pipe_valid_q
                                 && !update_pipe_valid_q;
        update_semantically_valid = window_active_q && !commit_active_q
                                  && update_row_in_range
                                  && !same_address_rdw_hazard;
        end_semantically_valid = window_active_q && !commit_active_q
                               && !update_pipe_valid_q && !update_valid;
        request_collision = (window_start_valid && update_valid)
                          || (window_start_valid && window_end_valid)
                          || (update_valid && window_end_valid);
        illegal_request = request_collision
                        || (window_start_valid && !start_semantically_valid)
                        || (update_valid && !update_semantically_valid)
                        || (window_end_valid && !end_semantically_valid);
    end

    assign protocol_error = request_fault_q || illegal_request
                          || update_pipe_overflow;
    assign window_start_ready = !protocol_error && start_semantically_valid
                              && !update_valid && !window_end_valid;
    assign update_ready = !protocol_error && update_semantically_valid
                        && !window_start_valid && !window_end_valid;
    assign window_end_ready = !protocol_error && end_semantically_valid
                            && !window_start_valid && !update_valid;
    assign window_start_accept = window_start_valid && window_start_ready;
    assign update_accept = update_valid && update_ready;
    assign window_end_accept = window_end_valid && window_end_ready;

    assign issue_commit = commit_active_q
                        && (!commit_pipe_valid_q || commit_ready)
                        && !protocol_error;
    assign commit_valid = commit_pipe_valid_q && !protocol_error;
    assign commit_block = commit_pipe_block_q;
    assign commit_row = commit_pipe_row_q;
    assign commit_data = commit_pipe_row_valid_q
                       ? mem_rd_data[commit_pipe_block_q] : '0;
    assign commit_last = commit_pipe_last_q;
    assign window_active = window_active_q;
    assign busy = window_active_q || update_pipe_valid_q || commit_active_q
                || commit_pipe_valid_q;

    always_comb begin : vector_add_and_macro_ports
        update_pipe_overflow = 1'b0;
        update_write_vector = '0;
        for (int lane = 0; lane < LANES; lane++) begin
            lane_base_ext[lane] = update_pipe_base_valid_q
                ? {mem_rd_data[update_pipe_block_q]
                    [lane * ACC_BITS + ACC_BITS - 1],
                   mem_rd_data[update_pipe_block_q]
                    [lane * ACC_BITS +: ACC_BITS]}
                : '0;
            lane_delta_ext[lane] = {
                update_pipe_delta_q[lane * ACC_BITS + ACC_BITS - 1],
                update_pipe_delta_q[lane * ACC_BITS +: ACC_BITS]};
            lane_sum_ext[lane] = lane_base_ext[lane] + lane_delta_ext[lane];
            update_write_vector[lane * ACC_BITS +: ACC_BITS]
                = lane_sum_ext[lane][ACC_BITS-1:0];
            if (update_pipe_valid_q
                    && lane_sum_ext[lane][ACC_BITS]
                       != lane_sum_ext[lane][ACC_BITS-1])
                update_pipe_overflow = 1'b1;
        end

        mem_rd_en = '0;
        mem_wr_en = '0;
        for (int bank = 0; bank < BANKS; bank++) begin
            mem_rd_addr[bank] = '0;
            mem_wr_addr[bank] = '0;
            mem_wr_data[bank] = '0;
        end
        if (update_accept) begin
            mem_rd_en[update_block] = 1'b1;
            mem_rd_addr[update_block] = update_row;
        end else if (issue_commit && row_valid_q[commit_issue_block_q]
                                      [commit_issue_row_q]) begin
            mem_rd_en[commit_issue_block_q] = 1'b1;
            mem_rd_addr[commit_issue_block_q] = commit_issue_row_q;
        end
        // A same-cycle malformed new request cannot erase an older accepted
        // update.  Only a sticky prior fault or arithmetic overflow suppresses
        // the buffered write.
        if (update_pipe_valid_q && !request_fault_q
                && !update_pipe_overflow) begin
            mem_wr_en[update_pipe_block_q] = 1'b1;
            mem_wr_addr[update_pipe_block_q] = update_pipe_row_q;
            mem_wr_data[update_pipe_block_q] = update_write_vector;
        end
    end

    always_ff @(posedge clk_core) begin : state_update
        if (rst_core) begin
            request_fault_q <= 1'b0;
            window_active_q <= 1'b0;
            update_pipe_valid_q <= 1'b0;
            update_pipe_block_q <= '0;
            update_pipe_row_q <= '0;
            update_pipe_delta_q <= '0;
            update_pipe_base_valid_q <= 1'b0;
            commit_active_q <= 1'b0;
            commit_issue_block_q <= '0;
            commit_issue_row_q <= '0;
            commit_pipe_valid_q <= 1'b0;
            commit_pipe_block_q <= '0;
            commit_pipe_row_q <= '0;
            commit_pipe_row_valid_q <= 1'b0;
            commit_pipe_last_q <= 1'b0;
            window_done <= 1'b0;
            for (int bank = 0; bank < BANKS; bank++)
                row_valid_q[bank] <= '0;
        end else begin
            window_done <= 1'b0;
            if (illegal_request || update_pipe_overflow)
                request_fault_q <= 1'b1;

            if (update_pipe_valid_q && !request_fault_q
                    && !update_pipe_overflow)
                row_valid_q[update_pipe_block_q][update_pipe_row_q] <= 1'b1;

            if (!request_fault_q && !update_pipe_overflow) begin
                update_pipe_valid_q <= update_accept;
                if (update_accept) begin
                    update_pipe_block_q <= update_block;
                    update_pipe_row_q <= update_row;
                    update_pipe_delta_q <= update_delta;
                    update_pipe_base_valid_q
                        <= row_valid_q[update_block][update_row];
                end

                if (window_start_accept) begin
                    window_active_q <= 1'b1;
                    for (int bank = 0; bank < BANKS; bank++)
                        row_valid_q[bank] <= '0;
                end

                if (window_end_accept) begin
                    window_active_q <= 1'b0;
                    commit_active_q <= 1'b1;
                    commit_issue_block_q <= '0;
                    commit_issue_row_q <= '0;
                end

                if (commit_pipe_valid_q && commit_ready) begin
                    commit_pipe_valid_q <= 1'b0;
                    if (commit_pipe_last_q)
                        window_done <= 1'b1;
                end
                if (issue_commit) begin
                    commit_pipe_valid_q <= 1'b1;
                    commit_pipe_block_q <= commit_issue_block_q;
                    commit_pipe_row_q <= commit_issue_row_q;
                    commit_pipe_row_valid_q
                        <= row_valid_q[commit_issue_block_q]
                                      [commit_issue_row_q];
                    commit_pipe_last_q
                        <= commit_issue_block_q == BANKS-1
                        && commit_issue_row_q == WIN_ROWS-1;
                    if (commit_issue_row_q == WIN_ROWS-1) begin
                        commit_issue_row_q <= '0;
                        if (commit_issue_block_q == BANKS-1) begin
                            commit_issue_block_q <= '0;
                            commit_active_q <= 1'b0;
                        end else begin
                            commit_issue_block_q <= commit_issue_block_q + 1'b1;
                        end
                    end else begin
                        commit_issue_row_q <= commit_issue_row_q + 1'b1;
                    end
                end
            end
        end
    end
endmodule

`default_nettype wire
