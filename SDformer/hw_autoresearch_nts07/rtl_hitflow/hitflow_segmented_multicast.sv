`timescale 1ns/1ps
`default_nettype none

module hitflow_segmented_multicast #(
    parameter int TOKENS         = 162,
    parameter int SEGMENT_TOKENS = 18,
    parameter int BANKS          = 2,
    parameter int PRODUCT_W      = 17,
    parameter int OUT_TILE       = 8,
    parameter int TAG_W          = 32,
    parameter int COUNTER_W      = 32,
    parameter int TOKEN_ID_W     = (TOKENS <= 1) ? 1 : $clog2(TOKENS)
) (
    input  logic                              clk_core,
    input  logic                              rst_core,

    input  logic                              product_valid,
    output logic                              product_ready,
    input  logic [TAG_W-1:0]                  product_tag,
    input  logic [TOKENS-1:0]                 product_destination_bitmap,
    input  logic [(OUT_TILE*PRODUCT_W)-1:0]   product_values,

    output logic [BANKS-1:0]                  update_valid,
    input  logic [BANKS-1:0]                  update_ready,
    output logic [(BANKS*TOKEN_ID_W)-1:0]     update_token_ids,
    output logic [TAG_W-1:0]                  update_tag,
    output logic [(OUT_TILE*PRODUCT_W)-1:0]   update_values,

    output logic                              product_done_valid,
    input  logic                              product_done_ready,
    output logic [TAG_W-1:0]                  product_done_tag,
    output logic                              protocol_error,

    output logic [COUNTER_W-1:0]              count_products,
    output logic [COUNTER_W-1:0]              count_destinations,
    output logic [COUNTER_W-1:0]              count_issue_cycles,
    output logic [COUNTER_W-1:0]              count_segment_advances,
    output logic [COUNTER_W-1:0]              count_bank_stall_cycles
);

    localparam bit BANK_ALIGNED_SEGMENTS =
        (SEGMENT_TOKENS <= TOKENS) && ((SEGMENT_TOKENS % BANKS) == 0);

    logic active_q;
    logic done_q;
    logic [TAG_W-1:0] tag_q;
    logic [SEGMENT_TOKENS-1:0] segment_pending_q;
    logic [TOKENS-1:0] remaining_q;
    logic [(OUT_TILE*PRODUCT_W)-1:0] values_q;
    logic [TOKEN_ID_W:0] segment_base_q;

    logic [BANKS-1:0] bank_found;
    logic [SEGMENT_TOKENS-1:0] clear_mask;
    logic [SEGMENT_TOKENS-1:0] segment_after;
    logic [COUNTER_W-1:0] destination_fire_count;
    logic product_fire;
    logic done_fire;

    // Ready when idle. Empty-bitmap rejection is handled by protocol_error and
    // by not asserting ready for an invalid presented product.
    assign product_ready = !active_q && !done_q && BANK_ALIGNED_SEGMENTS &&
                           (!product_valid || (product_destination_bitmap != '0));
    assign product_fire = product_valid && product_ready;
    assign update_tag = tag_q;
    assign update_values = values_q;
    assign product_done_valid = done_q;
    assign product_done_tag = tag_q;
    assign done_fire = product_done_valid && product_done_ready;
    assign protocol_error = !BANK_ALIGNED_SEGMENTS ||
                            (product_valid && !active_q && !done_q &&
                             (product_destination_bitmap == '0));

    // Split request presentation from ready/clear so token_ids do not form a
    // combinational loop through accumulator ready.
    always_comb begin
        update_valid = '0;
        update_token_ids = '0;
        bank_found = '0;

        for (int offset = 0; offset < SEGMENT_TOKENS;
             offset = offset + 1) begin
            if (active_q && segment_pending_q[offset] &&
                !bank_found[offset % BANKS]) begin
                update_valid[offset % BANKS] = 1'b1;
                update_token_ids[((offset % BANKS)*TOKEN_ID_W) +: TOKEN_ID_W] =
                    TOKEN_ID_W'(segment_base_q) + TOKEN_ID_W'(offset);
                bank_found[offset % BANKS] = 1'b1;
            end
        end
    end

    always_comb begin
        clear_mask = '0;
        destination_fire_count = '0;
        for (int offset = 0; offset < SEGMENT_TOKENS;
             offset = offset + 1) begin
            if (segment_pending_q[offset] &&
                update_valid[offset % BANKS] && update_ready[offset % BANKS] &&
                (update_token_ids[((offset % BANKS)*TOKEN_ID_W) +: TOKEN_ID_W] ==
                 (TOKEN_ID_W'(segment_base_q) + TOKEN_ID_W'(offset)))) begin
                clear_mask[offset] = 1'b1;
                destination_fire_count = destination_fire_count + 1'b1;
            end
        end

        segment_after = segment_pending_q & ~clear_mask;
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            active_q                   <= 1'b0;
            done_q                     <= 1'b0;
            tag_q                      <= '0;
            segment_pending_q          <= '0;
            remaining_q                <= '0;
            values_q                   <= '0;
            segment_base_q             <= '0;
            count_products             <= '0;
            count_destinations         <= '0;
            count_issue_cycles         <= '0;
            count_segment_advances     <= '0;
            count_bank_stall_cycles    <= '0;
        end else begin
            if (done_fire) begin
                done_q <= 1'b0;
            end

            if (product_fire) begin
                active_q       <= 1'b1;
                tag_q          <= product_tag;
                segment_pending_q <=
                    product_destination_bitmap[SEGMENT_TOKENS-1:0];
                remaining_q <= product_destination_bitmap >> SEGMENT_TOKENS;
                values_q       <= product_values;
                segment_base_q <= '0;
                count_products <= count_products + 1'b1;
            end else if (active_q) begin
                segment_pending_q <= segment_after;
                if (update_valid != '0) begin
                    count_issue_cycles <= count_issue_cycles + 1'b1;
                end
                if ((update_valid & ~update_ready) != '0) begin
                    count_bank_stall_cycles <= count_bank_stall_cycles + 1'b1;
                end
                count_destinations <= count_destinations + destination_fire_count;

                if (segment_after == '0) begin
                    if (remaining_q == '0) begin
                        active_q <= 1'b0;
                        done_q   <= 1'b1;
                    end else begin
                        segment_pending_q <= remaining_q[SEGMENT_TOKENS-1:0];
                        remaining_q <= remaining_q >> SEGMENT_TOKENS;
                        segment_base_q <= segment_base_q +
                                          (TOKEN_ID_W+1)'(SEGMENT_TOKENS);
                        count_segment_advances <= count_segment_advances + 1'b1;
                    end
                end
            end
        end
    end

endmodule

`default_nettype wire
