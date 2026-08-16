`timescale 1ns/1ps
`default_nettype none

// Occupied-Bit Iterator: hierarchically selects one active {slot,lane} entry.
module gatestack_obi_iterator #(
    parameter int SLOTS      = 4,
    parameter int LANES      = 32,
    parameter int TAG_W      = 32,
    parameter int COUNTER_W  = 32,
    parameter int SLOT_ID_W  = (SLOTS <= 1) ? 1 : $clog2(SLOTS),
    parameter int LANE_ID_W  = (LANES <= 1) ? 1 : $clog2(LANES)
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         load_valid,
    output logic                         load_ready,
    input  logic [TAG_W-1:0]             load_tag,
    input  logic [(SLOTS*LANES)-1:0]     load_occupied_mask,

    output logic                         entry_valid,
    input  logic                         entry_ready,
    output logic [TAG_W-1:0]             entry_tag,
    output logic [SLOT_ID_W-1:0]         entry_slot_id,
    output logic [LANE_ID_W-1:0]         entry_lane_id,
    output logic                         entry_last,

    output logic                         done_valid,
    input  logic                         done_ready,
    output logic [TAG_W-1:0]             done_tag,

    output logic [COUNTER_W-1:0]         count_loads,
    output logic [COUNTER_W-1:0]         count_entries,
    output logic [COUNTER_W-1:0]         count_entry_stall_cycles
);

    localparam int MASK_W = SLOTS * LANES;

    logic active_q;
    logic done_q;
    logic [TAG_W-1:0] tag_q;
    logic [MASK_W-1:0] pending_q;

    logic selected_valid;
    logic [SLOT_ID_W-1:0] selected_slot;
    logic [LANE_ID_W-1:0] selected_lane;
    logic [MASK_W-1:0] selected_onehot;
    logic slot_found;
    logic lane_found;
    logic entry_fire;
    logic load_fire;
    logic done_fire;

    assign load_ready = !active_q && !done_q;
    assign load_fire = load_valid && load_ready;
    assign entry_valid = active_q && selected_valid;
    assign entry_fire = entry_valid && entry_ready;
    assign entry_tag = tag_q;
    assign entry_slot_id = selected_slot;
    assign entry_lane_id = selected_lane;
    assign entry_last = entry_valid && (pending_q == selected_onehot);
    assign done_valid = done_q;
    assign done_fire = done_valid && done_ready;
    assign done_tag = tag_q;

    // Two-level priority selection limits the longest chain to SLOTS + LANES.
    always_comb begin
        selected_valid = 1'b0;
        selected_slot = '0;
        selected_lane = '0;
        selected_onehot = '0;
        slot_found = 1'b0;
        lane_found = 1'b0;

        for (int slot = 0; slot < SLOTS; slot = slot + 1) begin
            if (!slot_found &&
                (pending_q[(slot*LANES) +: LANES] != '0)) begin
                selected_slot = SLOT_ID_W'(slot);
                slot_found = 1'b1;
            end
        end

        for (int lane = 0; lane < LANES; lane = lane + 1) begin
            if (slot_found && !lane_found &&
                pending_q[(selected_slot*LANES) + lane]) begin
                selected_lane = LANE_ID_W'(lane);
                lane_found = 1'b1;
            end
        end

        selected_valid = slot_found && lane_found;
        if (selected_valid) begin
            selected_onehot[(selected_slot*LANES) + 32'(selected_lane)] = 1'b1;
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            active_q <= 1'b0;
            done_q <= 1'b0;
            tag_q <= '0;
            pending_q <= '0;
            count_loads <= '0;
            count_entries <= '0;
            count_entry_stall_cycles <= '0;
        end else begin
            if (load_fire) begin
                active_q <= (load_occupied_mask != '0);
                done_q <= (load_occupied_mask == '0);
                tag_q <= load_tag;
                pending_q <= load_occupied_mask;
                count_loads <= count_loads + 1'b1;
            end else if (entry_fire) begin
                pending_q <= pending_q & ~selected_onehot;
                count_entries <= count_entries + 1'b1;
                if (entry_last) begin
                    active_q <= 1'b0;
                    done_q <= 1'b1;
                end
            end else if (done_fire) begin
                done_q <= 1'b0;
            end

            if (entry_valid && !entry_ready) begin
                count_entry_stall_cycles <= count_entry_stall_cycles + 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
