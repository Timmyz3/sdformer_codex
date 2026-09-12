// Isolated metadata arithmetic, not a source/coalescer/NRV accelerator.
// The caller owns RF94, geometry RF, SRAM, issue and RF writeback arbitration.
module metadata_primitive (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         in_valid,
    output logic         in_ready,
    input  logic [1:0]   operation_i,       // 0: IMETA_GATE, 1: IMETA_WORD, 2: permission
    // The upper six bits of each lane are intentionally ignored by contract.
    /* verilator lint_off UNUSED */
    input  logic [127:0] gate_words_i,      // uint16 lane 0 occupies bits 15:0
    /* verilator lint_on UNUSED */
    input  logic [23:0]  summary_i,         // existing RF94, already cleared per pixel
    input  logic [4:0]   group_i,           // GATE: even 0..22; WORD/permission: 0..23
    input  logic [23:0]  geometry0_i,       // existing geometry lane 4
    input  logic [23:0]  geometry1_i,       // existing geometry lane 5
    input  logic [1:0]   position_valid_i,  // existing geometry lane 6, low two bits
    output logic         out_valid,
    input  logic         out_ready,
    output logic [1:0]   operation_o,
    output logic [23:0]  summary_o,
    output logic [1:0]   occupied_o,
    output logic [1:0]   permission_o
);
    logic [1:0] occupied;
    logic [23:0] next_summary;
    logic [1:0] next_occupied;
    logic [1:0] next_permission;

    // Upper six bits of every uint16 lane must not affect occupancy.
    always_comb begin
        occupied[0] = (|gate_words_i[9:0])   | (|gate_words_i[25:16]) |
                      (|gate_words_i[41:32]) | (|gate_words_i[57:48]);
        occupied[1] = (|gate_words_i[73:64]) | (|gate_words_i[89:80]) |
                      (|gate_words_i[105:96]) | (|gate_words_i[121:112]);
        next_summary = '0;
        next_occupied = '0;
        next_permission = '0;
        case (operation_i)
            2'd0: begin
                next_summary = summary_i;
                next_occupied = occupied;
                // Match CPU IMETA_GATE's OR accumulation exactly. Inputs outside
                // the declared even-index contract leave the summary unchanged.
                if (group_i < 5'd23 && !group_i[0])
                    next_summary[group_i +: 2] = summary_i[group_i +: 2] | occupied;
            end
            2'd1: begin
                next_summary = summary_i;
                next_occupied = {1'b0, occupied[0]};
                if (group_i < 5'd24)
                    next_summary[group_i] = summary_i[group_i] | occupied[0];
            end
            2'd2: begin
                if (group_i < 5'd24) begin
                    next_permission[0] = position_valid_i[0] & geometry0_i[group_i];
                    next_permission[1] = position_valid_i[1] & geometry1_i[group_i];
                end
            end
            default: begin end
        endcase
    end

    // One elastic result register. A stalled result and all its fields hold.
    // No per-pixel summary is stored here: summary_i comes from the existing RF.
    assign in_ready = !out_valid || out_ready;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            operation_o <= '0;
            summary_o <= '0;
            occupied_o <= '0;
            permission_o <= '0;
        end else if (in_ready) begin
            out_valid <= in_valid;
            if (in_valid) begin
                operation_o <= operation_i;
                summary_o <= next_summary;
                occupied_o <= next_occupied;
                permission_o <= next_permission;
            end
        end
    end
endmodule
