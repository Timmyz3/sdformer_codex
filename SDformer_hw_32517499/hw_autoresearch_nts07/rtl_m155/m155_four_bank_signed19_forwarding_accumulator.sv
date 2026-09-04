`timescale 1ns/1ps
`default_nettype none

// Four conflict-free destination-bank signed19 accumulator.
//
// One accepted descriptor may update one destination in each modulo-four bank.
// Destinations d and d+4 share a physical bank and are rejected in the same
// descriptor.  Each bank is a 768x(96x19) external 1R1W SRAM: address is
// {destination[2], row[8:0]}.  A bank-local one-entry RMW pipeline sustains one
// vector update per cycle.  Consecutive updates to the same address suppress
// the undefined macro read and forward the older computed write vector.
//
// Input contributions are signed11 (PWP or folded correction range).  Negation
// first widens to signed12, preserving -(-1024)=+1024, before exact signed19
// accumulation.  Any lane overflow rejects the entire four-bank write atomically
// and raises a sticky fault.  SRAM macros and commit/drain scheduling are cuts.
module m155_four_bank_signed19_forwarding_accumulator #(
    parameter int LANES = 96,
    parameter int ROWS = 384,
    parameter int ACC_BITS = 19
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         window_start_valid,
    output logic                         window_start_ready,
    output logic                         window_start_accept,

    input  logic                         update_valid,
    output logic                         update_ready,
    input  logic [8:0]                   update_row,
    input  logic [3:0]                   update_group_valid,
    input  logic [2:0]                   update_destination [0:3],
    input  logic [3:0]                   update_negate,
    input  logic signed [10:0]           update_vector [0:3][0:LANES-1],
    output logic                         update_accept,

    input  logic                         window_end_valid,
    output logic                         window_end_ready,
    output logic                         window_end_accept,
    output logic                         window_done,

    output logic [3:0]                   acc_rd_en,
    output logic [9:0]                   acc_rd_addr [0:3],
    input  logic signed [ACC_BITS-1:0]   acc_rd_data [0:3][0:LANES-1],
    output logic [3:0]                   acc_wr_en,
    output logic [9:0]                   acc_wr_addr [0:3],
    output logic signed [ACC_BITS-1:0]   acc_wr_data [0:3][0:LANES-1],

    output logic [3:0]                   same_address_forward,
    output logic                         window_active,
    output logic                         protocol_error,
    output logic                         overflow_error,
    output logic                         busy
);
    logic fault_q;
    logic window_active_q;
    logic [767:0] address_valid_q [0:3];

    logic [3:0] pipe_valid_q;
    logic [9:0] pipe_addr_q [0:3];
    logic signed [11:0] pipe_delta_q [0:3][0:LANES-1];
    logic [3:0] pipe_base_valid_q;
    logic [3:0] pipe_base_forward_q;
    logic signed [ACC_BITS-1:0] pipe_base_forward_data_q
        [0:3][0:LANES-1];

    logic [3:0] incoming_bank_valid;
    logic [9:0] incoming_addr [0:3];
    logic signed [11:0] incoming_delta [0:3][0:LANES-1];
    logic illegal_shape;
    logic bank_conflict;
    logic row_out_of_range;
    logic request_collision;
    logic illegal_request;
    logic pipe_overflow_any;
    logic pipe_overflow [0:3][0:LANES-1];
    logic signed [ACC_BITS:0] write_sum [0:3][0:LANES-1];

`ifndef SYNTHESIS
    initial begin
        if (LANES != 96 || ROWS != 384 || ACC_BITS != 19)
            $fatal(1, "M155 production geometry drift");
    end
`endif

    always_comb begin : route_and_protocol
        case (update_group_valid)
            4'b0001, 4'b0011, 4'b0111, 4'b1111:
                illegal_shape = 1'b0;
            default:
                illegal_shape = 1'b1;
        endcase
        bank_conflict = 1'b0;
        for (int later = 1; later < 4; later++) begin
            for (int earlier = 0; earlier < later; earlier++) begin
                if (update_group_valid[later]
                        && update_group_valid[earlier]
                        && update_destination[later][1:0]
                           == update_destination[earlier][1:0])
                    bank_conflict = 1'b1;
            end
        end
        row_out_of_range = update_row >= ROWS;
        request_collision = (window_start_valid && update_valid)
                          || (window_start_valid && window_end_valid)
                          || (update_valid && window_end_valid);
        illegal_request = request_collision
            || (window_start_valid && (window_active_q || |pipe_valid_q))
            || (update_valid
                && (!window_active_q || illegal_shape || bank_conflict
                    || row_out_of_range))
            || (window_end_valid
                && (!window_active_q || |pipe_valid_q || update_valid));

        incoming_bank_valid = '0;
        for (int bank = 0; bank < 4; bank++) begin
            incoming_addr[bank] = '0;
            for (int lane = 0; lane < LANES; lane++)
                incoming_delta[bank][lane] = '0;
        end
        for (int tuple = 0; tuple < 4; tuple++) begin
            if (update_group_valid[tuple]) begin
                incoming_bank_valid[update_destination[tuple][1:0]] = 1'b1;
                incoming_addr[update_destination[tuple][1:0]] = {
                    update_destination[tuple][2], update_row};
                for (int lane = 0; lane < LANES; lane++) begin
                    logic signed [11:0] widened;
                    widened = {{1{update_vector[tuple][lane][10]}},
                               update_vector[tuple][lane]};
                    incoming_delta[update_destination[tuple][1:0]][lane]
                        = update_negate[tuple] ? -widened : widened;
                end
            end
        end
    end

    always_comb begin : add_and_macro_ports
        pipe_overflow_any = 1'b0;
        acc_rd_en = '0;
        acc_wr_en = '0;
        same_address_forward = '0;
        for (int bank = 0; bank < 4; bank++) begin
            acc_rd_addr[bank] = incoming_addr[bank];
            acc_wr_addr[bank] = pipe_addr_q[bank];
            same_address_forward[bank] = pipe_valid_q[bank]
                && incoming_bank_valid[bank]
                && pipe_addr_q[bank] == incoming_addr[bank];
            for (int lane = 0; lane < LANES; lane++) begin
                logic signed [ACC_BITS:0] base_ext;
                logic signed [ACC_BITS:0] delta_ext;
                if (pipe_base_forward_q[bank])
                    base_ext = {pipe_base_forward_data_q[bank][lane]
                                [ACC_BITS-1],
                                pipe_base_forward_data_q[bank][lane]};
                else if (pipe_base_valid_q[bank])
                    base_ext = {acc_rd_data[bank][lane][ACC_BITS-1],
                                acc_rd_data[bank][lane]};
                else
                    base_ext = '0;
                delta_ext = {{(ACC_BITS + 1 - 12)
                              {pipe_delta_q[bank][lane][11]}},
                             pipe_delta_q[bank][lane]};
                write_sum[bank][lane] = base_ext + delta_ext;
                pipe_overflow[bank][lane] = pipe_valid_q[bank]
                    && write_sum[bank][lane][ACC_BITS]
                       != write_sum[bank][lane][ACC_BITS-1];
                if (pipe_overflow[bank][lane])
                    pipe_overflow_any = 1'b1;
                acc_wr_data[bank][lane]
                    = write_sum[bank][lane][ACC_BITS-1:0];
            end
            if (update_accept && incoming_bank_valid[bank]
                    && !same_address_forward[bank]
                    && address_valid_q[bank][incoming_addr[bank]])
                acc_rd_en[bank] = 1'b1;
        end
        for (int bank = 0; bank < 4; bank++) begin
            if (pipe_valid_q[bank] && !fault_q && !pipe_overflow_any)
                acc_wr_en[bank] = 1'b1;
        end
    end

    assign window_start_ready = !rst_core && !fault_q
                              && !window_active_q && !(|pipe_valid_q)
                              && !update_valid && !window_end_valid;
    assign update_ready = !rst_core && !fault_q && !pipe_overflow_any
                        && window_active_q && !illegal_shape
                        && !bank_conflict && !row_out_of_range
                        && !window_start_valid && !window_end_valid;
    assign window_end_ready = !rst_core && !fault_q && window_active_q
                            && !(|pipe_valid_q) && !window_start_valid
                            && !update_valid;
    assign window_start_accept = window_start_valid && window_start_ready;
    assign update_accept = update_valid && update_ready;
    assign window_end_accept = window_end_valid && window_end_ready;
    assign window_active = window_active_q;
    assign protocol_error = !rst_core && (fault_q || illegal_request);
    assign overflow_error = !rst_core && pipe_overflow_any;
    assign busy = window_active_q || |pipe_valid_q;

    always_ff @(posedge clk_core) begin : state_update
        if (rst_core) begin
            fault_q <= 1'b0;
            window_active_q <= 1'b0;
            pipe_valid_q <= '0;
            pipe_base_valid_q <= '0;
            pipe_base_forward_q <= '0;
            window_done <= 1'b0;
            for (int bank = 0; bank < 4; bank++) begin
                address_valid_q[bank] <= '0;
                pipe_addr_q[bank] <= '0;
                for (int lane = 0; lane < LANES; lane++) begin
                    pipe_delta_q[bank][lane] <= '0;
                    pipe_base_forward_data_q[bank][lane] <= '0;
                end
            end
        end else begin
            window_done <= 1'b0;
            if (illegal_request || pipe_overflow_any)
                fault_q <= 1'b1;

            // An illegal younger request cannot suppress older accepted writes.
            if (!fault_q && !pipe_overflow_any) begin
                for (int bank = 0; bank < 4; bank++) begin
                    if (pipe_valid_q[bank])
                        address_valid_q[bank][pipe_addr_q[bank]] <= 1'b1;
                    pipe_valid_q[bank]
                        <= update_accept && incoming_bank_valid[bank];
                    if (update_accept && incoming_bank_valid[bank]) begin
                        pipe_addr_q[bank] <= incoming_addr[bank];
                        pipe_base_valid_q[bank]
                            <= same_address_forward[bank]
                               || address_valid_q[bank][incoming_addr[bank]];
                        pipe_base_forward_q[bank]
                            <= same_address_forward[bank];
                        for (int lane = 0; lane < LANES; lane++) begin
                            pipe_delta_q[bank][lane]
                                <= incoming_delta[bank][lane];
                            pipe_base_forward_data_q[bank][lane]
                                <= acc_wr_data[bank][lane];
                        end
                    end
                end

                if (window_start_accept) begin
                    window_active_q <= 1'b1;
                    for (int bank = 0; bank < 4; bank++)
                        address_valid_q[bank] <= '0;
                end
                if (window_end_accept) begin
                    window_active_q <= 1'b0;
                    window_done <= 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
