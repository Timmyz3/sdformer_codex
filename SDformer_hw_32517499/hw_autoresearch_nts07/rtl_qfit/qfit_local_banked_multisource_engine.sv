`timescale 1ns/1ps
`default_nettype none

// Exact Local tile engine with one conflict-free source per weight bank/cycle.
//
// The activation frontier is split by source_index % ISSUE_WIDTH.  A small
// WORD_BITS window is selected first, then each bank performs only a
// WORD_BITS/ISSUE_WIDTH priority search.  Weight storage is intentionally an
// external one-read-per-bank synchronous interface: synthesis must not turn a
// 256 x OUT_LANES table into flops or a 256:1 combinational mux again.
module qfit_local_banked_multisource_engine #(
    parameter int TILE_BITS = 256,
    parameter int WORD_BITS = 32,
    parameter int ISSUE_WIDTH = 4,
    parameter int OUT_LANES = 16,
    parameter int TAG_W = 32,
    parameter int W_W = 8,
    parameter int ACC_W = 32,
    parameter int INDEX_W = (TILE_BITS <= 1) ? 1 : $clog2(TILE_BITS),
    parameter int COUNT_W = $clog2(TILE_BITS + 1),
    parameter int BANK_BITS = (ISSUE_WIDTH <= 1) ? 0 : $clog2(ISSUE_WIDTH),
    parameter int BANK_ADDR_W = INDEX_W - BANK_BITS,
    parameter int WORD_COUNT = TILE_BITS / WORD_BITS,
    parameter int WORD_INDEX_W = (WORD_COUNT <= 1) ? 1 : $clog2(WORD_COUNT)
) (
    input  logic                              clk_core,
    input  logic                              rst_core,

    input  logic                              command_valid,
    output logic                              command_ready,
    input  logic [TAG_W-1:0]                  command_tag,
    input  logic [TILE_BITS-1:0]              command_current_bits,
    input  logic [OUT_LANES*ACC_W-1:0]        command_seed_acc,

    output logic                              weight_request_valid,
    input  logic                              weight_request_ready,
    output logic [ISSUE_WIDTH-1:0]            weight_request_bank_valid,
    output logic [ISSUE_WIDTH*BANK_ADDR_W-1:0] weight_request_bank_addr,
    output logic                              weight_request_last,

    input  logic                              weight_response_valid,
    output logic                              weight_response_ready,
    input  logic [ISSUE_WIDTH-1:0]            weight_response_bank_valid,
    input  logic [ISSUE_WIDTH*OUT_LANES*W_W-1:0] weight_response_data,

    output logic                              output_valid,
    input  logic                              output_ready,
    output logic [TAG_W-1:0]                  output_tag,
    output logic [COUNT_W-1:0]                output_source_count,
    output logic [OUT_LANES*ACC_W-1:0]        output_acc,

    output logic                              protocol_error
);
    logic active_q;
    logic issued_all_q;
    logic pending_q;
    logic pending_last_q;
    logic faulted_q;
    logic output_valid_q;
    logic [TAG_W-1:0] tag_q;
    logic [TILE_BITS-1:0] remaining_q;
    logic [WORD_COUNT-1:0] bank_word_nonempty_q [0:ISSUE_WIDTH-1];
    logic [ISSUE_WIDTH-1:0] pending_bank_valid_q;
    logic [COUNT_W-1:0] source_count_q;
    logic signed [ACC_W-1:0] acc_q [0:OUT_LANES-1];

    logic [ISSUE_WIDTH-1:0] selected_word_found;
    logic [WORD_INDEX_W-1:0] selected_word [0:ISSUE_WIDTH-1];
    logic [TILE_BITS-1:0] selection_mask;
    logic selected_valid;
    logic [TILE_BITS-1:0] remaining_after_request;
    logic [COUNT_W-1:0] request_source_count;
    logic response_contract_valid;
    logic can_issue_request;
    logic command_fire;
    logic request_fire;
    logic response_fire;
    logic output_fire;
    logic signed [ACC_W-1:0] response_sum [0:OUT_LANES-1];

    function automatic logic signed [ACC_W-1:0] extend_weight(
        input logic [W_W-1:0] value
    );
        extend_weight = {{(ACC_W-W_W){value[W_W-1]}}, value};
    endfunction

    function automatic logic [COUNT_W-1:0] popcount_banks(
        input logic [ISSUE_WIDTH-1:0] value
    );
        logic [COUNT_W-1:0] count;
        begin
            count = '0;
            for (int bank = 0; bank < ISSUE_WIDTH; bank = bank + 1)
                count = count + COUNT_W'(value[bank]);
            popcount_banks = count;
        end
    endfunction

    function automatic logic word_has_bank_sources(
        input logic [TILE_BITS-1:0] value,
        input integer word_index,
        input integer bank_index
    );
        logic found;
        begin
            found = 1'b0;
            for (int position = bank_index; position < WORD_BITS; position = position + ISSUE_WIDTH)
                found = found | value[word_index*WORD_BITS + position];
            word_has_bank_sources = found;
        end
    endfunction

    initial begin
        if (ACC_W < W_W)
            $error("ACC_W must be at least W_W");
        if (TILE_BITS % WORD_BITS != 0)
            $error("TILE_BITS must be divisible by WORD_BITS");
        if (WORD_BITS % ISSUE_WIDTH != 0)
            $error("WORD_BITS must be divisible by ISSUE_WIDTH");
        if (ISSUE_WIDTH < 1)
            $error("ISSUE_WIDTH must be positive");
        if ((ISSUE_WIDTH & (ISSUE_WIDTH - 1)) != 0)
            $error("ISSUE_WIDTH must be a power of two");
    end

    always_comb begin
        selected_word_found = '0;
        for (int bank = 0; bank < ISSUE_WIDTH; bank = bank + 1) begin
            selected_word[bank] = '0;
            for (int word = 0; word < WORD_COUNT; word = word + 1) begin
                if (!selected_word_found[bank] && bank_word_nonempty_q[bank][word]) begin
                    selected_word_found[bank] = 1'b1;
                    selected_word[bank] = WORD_INDEX_W'(word);
                end
            end
        end

        weight_request_bank_valid = '0;
        weight_request_bank_addr = '0;
        selection_mask = '0;
        for (int word = 0; word < WORD_COUNT; word = word + 1) begin
            for (int bank = 0; bank < ISSUE_WIDTH; bank = bank + 1) begin
                logic bank_found;
                bank_found = 1'b0;
                for (int position = bank; position < WORD_BITS; position = position + ISSUE_WIDTH) begin
                    if (selected_word_found[bank] && selected_word[bank] == WORD_INDEX_W'(word)
                            && !bank_found && remaining_q[word*WORD_BITS + position]) begin
                        weight_request_bank_valid[bank] = 1'b1;
                        weight_request_bank_addr[bank*BANK_ADDR_W +: BANK_ADDR_W]
                            = BANK_ADDR_W'((word*WORD_BITS + position) >> BANK_BITS);
                        selection_mask[word*WORD_BITS + position] = 1'b1;
                        bank_found = 1'b1;
                    end
                end
            end
        end
        selected_valid = |weight_request_bank_valid;
        remaining_after_request = remaining_q & ~selection_mask;
        request_source_count = popcount_banks(weight_request_bank_valid);
    end

    always_comb begin
        for (int lane = 0; lane < OUT_LANES; lane = lane + 1) begin
            response_sum[lane] = '0;
            for (int bank = 0; bank < ISSUE_WIDTH; bank = bank + 1) begin
                if (weight_response_bank_valid[bank])
                    response_sum[lane] = response_sum[lane] + extend_weight(
                        weight_response_data[(bank*OUT_LANES + lane)*W_W +: W_W]
                    );
            end
        end
    end

    assign response_contract_valid = weight_response_bank_valid == pending_bank_valid_q;
    assign weight_response_ready = pending_q;
    assign response_fire = weight_response_valid && weight_response_ready;
    assign can_issue_request = !pending_q
                            || (weight_response_valid && response_contract_valid);
    assign weight_request_valid = active_q && !issued_all_q
                               && selected_valid && can_issue_request && !faulted_q;
    assign weight_request_last = weight_request_valid && remaining_after_request == '0;
    assign request_fire = weight_request_valid && weight_request_ready;
    assign command_ready = !active_q && !pending_q && !output_valid_q && !faulted_q;
    assign command_fire = command_valid && command_ready;
    assign output_valid = output_valid_q;
    assign output_fire = output_valid && output_ready;
    assign output_tag = tag_q;
    assign output_source_count = source_count_q;
    assign protocol_error = faulted_q;

    generate
        for (genvar lane = 0; lane < OUT_LANES; lane = lane + 1) begin : g_output
            assign output_acc[lane*ACC_W +: ACC_W] = acc_q[lane];
        end
    endgenerate

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            active_q <= 1'b0;
            issued_all_q <= 1'b0;
            pending_q <= 1'b0;
            pending_last_q <= 1'b0;
            pending_bank_valid_q <= '0;
            faulted_q <= 1'b0;
            output_valid_q <= 1'b0;
            tag_q <= '0;
            remaining_q <= '0;
            for (int bank = 0; bank < ISSUE_WIDTH; bank = bank + 1)
                bank_word_nonempty_q[bank] <= '0;
            source_count_q <= '0;
            for (int lane = 0; lane < OUT_LANES; lane = lane + 1)
                acc_q[lane] <= '0;
        end else begin
            if (command_fire) begin
                active_q <= command_current_bits != '0;
                issued_all_q <= command_current_bits == '0;
                tag_q <= command_tag;
                remaining_q <= command_current_bits;
                source_count_q <= '0;
                output_valid_q <= command_current_bits == '0;
                for (int bank = 0; bank < ISSUE_WIDTH; bank = bank + 1) begin
                    for (int word = 0; word < WORD_COUNT; word = word + 1)
                        bank_word_nonempty_q[bank][word]
                            <= word_has_bank_sources(command_current_bits, word, bank);
                end
                for (int lane = 0; lane < OUT_LANES; lane = lane + 1)
                    acc_q[lane] <= command_seed_acc[lane*ACC_W +: ACC_W];
            end

            if (request_fire) begin
                remaining_q <= remaining_after_request;
                source_count_q <= source_count_q + request_source_count;
                for (int bank = 0; bank < ISSUE_WIDTH; bank = bank + 1) begin
                    if (weight_request_bank_valid[bank])
                        bank_word_nonempty_q[bank][selected_word[bank]]
                            <= word_has_bank_sources(
                                remaining_after_request, selected_word[bank], bank
                            );
                end
                issued_all_q <= remaining_after_request == '0;
                pending_q <= 1'b1;
                pending_last_q <= remaining_after_request == '0;
                pending_bank_valid_q <= weight_request_bank_valid;
            end

            if (response_fire) begin
                if (!response_contract_valid) begin
                    faulted_q <= 1'b1;
                    active_q <= 1'b0;
                    issued_all_q <= 1'b0;
                    pending_q <= 1'b0;
                    output_valid_q <= 1'b0;
                end else begin
                    for (int lane = 0; lane < OUT_LANES; lane = lane + 1)
                        acc_q[lane] <= acc_q[lane] + response_sum[lane];
                    if (pending_last_q) begin
                        active_q <= 1'b0;
                        output_valid_q <= 1'b1;
                    end
                    if (!request_fire) begin
                        pending_q <= 1'b0;
                        pending_last_q <= 1'b0;
                        pending_bank_valid_q <= '0;
                    end
                end
            end

            if (weight_response_valid && !pending_q) begin
                faulted_q <= 1'b1;
                active_q <= 1'b0;
                issued_all_q <= 1'b0;
                output_valid_q <= 1'b0;
            end

            if (output_fire) begin
                output_valid_q <= 1'b0;
                issued_all_q <= 1'b0;
            end
        end
    end
endmodule

`default_nettype wire
