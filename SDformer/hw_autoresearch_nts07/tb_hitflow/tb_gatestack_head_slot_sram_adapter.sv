`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_head_slot_sram_adapter;
    localparam int CONTEXTS = 2;
    localparam int HEADS = 3;
    localparam int HEAD_BITS = 6642;
    localparam int SLOT_CAPACITY_BITS = 6656;
    localparam int WORD_W = 64;
    logic clk_core;
    logic rst_core;
    logic commit_begin_valid;
    logic commit_begin_ready;
    logic commit_context_id;
    logic [1:0] commit_head_id;
    logic [31:0] commit_tag;
    logic commit_mode_is_csr;
    logic [15:0] commit_payload_bits;
    logic commit_word_valid;
    logic commit_word_ready;
    logic [63:0] commit_word_data;
    logic commit_word_last;
    logic inspect_valid;
    logic inspect_ready;
    logic inspect_context_id;
    logic [1:0] inspect_head_id;
    logic inspect_meta_valid;
    logic inspect_meta_ready;
    logic inspect_exists;
    logic [31:0] inspect_tag;
    logic inspect_mode_is_csr;
    logic [1:0] inspect_format;
    logic [15:0] inspect_payload_bits;
    logic [15:0] inspect_word_count;
    logic replay_begin_valid;
    logic replay_begin_ready;
    logic replay_context_id;
    logic [1:0] replay_head_id;
    logic [6:0] replay_start_word;
    logic replay_word_valid;
    logic replay_word_ready;
    logic [63:0] replay_word_data;
    logic [6:0] replay_word_index;
    logic replay_word_last;
    logic [31:0] replay_tag;
    logic replay_mode_is_csr;
    logic [1:0] replay_format;
    logic [15:0] replay_payload_bits;
    logic release_valid;
    logic release_ready;
    logic release_context_id;
    logic [1:0] release_head_id;
    logic commit_session_active;
    logic replay_session_active;
    logic [5:0] slot_valid_flat;
    logic protocol_error;
    logic [31:0] count_commit_heads;
    logic [31:0] count_replay_heads;
    logic [31:0] count_release_heads;
    logic [31:0] count_invalid_headers;
    logic [31:0] count_commit_stall_cycles;
    logic [31:0] count_replay_stall_cycles;
    int overlap_cycles;

    gatestack_head_slot_sram_adapter #(
        .CONTEXTS(CONTEXTS),
        .HEADS(HEADS),
        .HEAD_BITS(HEAD_BITS),
        .SLOT_CAPACITY_BITS(SLOT_CAPACITY_BITS),
        .WORD_W(WORD_W)
    ) dut (.*);

    always #5 clk_core <= ~clk_core;

    function automatic logic [63:0] word_pattern(
        input logic [31:0] seed,
        input int index
    );
        word_pattern = {seed ^ 32'(index), seed + 32'(index * 17)};
    endfunction

    function automatic logic [63:0] payload_word(
        input logic is_csr,
        input logic [31:0] tag,
        input logic [31:0] seed,
        input int index
    );
        if (is_csr && index == 0)
            payload_word = {tag, 11'd0, 1'b1, 4'd1, 16'h4753};
        else
            payload_word = word_pattern(seed, index);
    endfunction

    task automatic begin_commit(
        input logic context_id,
        input logic [1:0] head_id,
        input logic [31:0] tag,
        input logic is_csr,
        input logic [15:0] payload_bits
    );
        begin
            @(negedge clk_core);
            commit_context_id = 1'(context_id);
            commit_head_id = 2'(head_id);
            commit_tag = tag;
            commit_mode_is_csr = is_csr;
            commit_payload_bits = 16'(payload_bits);
            commit_begin_valid = 1'b1;
            do @(posedge clk_core); while (!commit_begin_ready);
            @(negedge clk_core);
            commit_begin_valid = 1'b0;
        end
    endtask

    task automatic commit_slot(
        input logic context_id,
        input logic [1:0] head_id,
        input logic [31:0] tag,
        input logic is_csr,
        input logic [15:0] payload_bits,
        input logic [31:0] seed
    );
        int words;
        begin
            words = (32'(payload_bits) + WORD_W - 1) / WORD_W;
            begin_commit(context_id, head_id, tag, is_csr, payload_bits);
            for (int index = 0; index < words; index = index + 1) begin
                if ((index % 5) == 2) @(posedge clk_core);
                @(negedge clk_core);
                commit_word_valid = 1'b1;
                commit_word_data = payload_word(is_csr, tag, seed, index);
                commit_word_last = (index == words - 1);
                do @(posedge clk_core); while (!commit_word_ready);
                @(negedge clk_core);
                commit_word_valid = 1'b0;
                commit_word_last = 1'b0;
            end
        end
    endtask

    task automatic replay_slot(
        input logic context_id,
        input logic [1:0] head_id,
        input logic [31:0] expected_tag,
        input logic expected_is_csr,
        input logic [15:0] expected_payload_bits,
        input logic [31:0] seed,
        input int start_word
    );
        int words;
        int received;
        int replay_cycles;
        begin
            words = (32'(expected_payload_bits) + WORD_W - 1) / WORD_W;
            received = 0;
            replay_cycles = 0;
            @(negedge clk_core);
            replay_context_id = 1'(context_id);
            replay_head_id = 2'(head_id);
            replay_start_word = 7'(start_word);
            replay_begin_valid = 1'b1;
            do @(posedge clk_core); while (!replay_begin_ready);
            @(negedge clk_core);
            replay_begin_valid = 1'b0;
            while (received < words - start_word) begin
                @(negedge clk_core);
                replay_word_ready = ((replay_cycles % 3) != 1);
                @(posedge clk_core);
                if (replay_word_valid && replay_word_ready) begin
                    if (replay_word_index != 7'(received) ||
                        replay_word_data != payload_word(expected_is_csr,
                            expected_tag, seed, received + start_word) ||
                        replay_word_last !=
                            (received == words - start_word - 1) ||
                        replay_tag != expected_tag ||
                        replay_mode_is_csr != expected_is_csr ||
                        replay_format != (expected_is_csr ? 2'd1 : 2'd0) ||
                        replay_payload_bits != 16'(expected_payload_bits)) begin
                        $fatal(1, "replay mismatch head=%0d word=%0d", head_id,
                               received);
                    end
                    received = received + 1;
                end
                replay_cycles = replay_cycles + 1;
            end
            @(negedge clk_core);
            replay_word_ready = 1'b0;
        end
    endtask

    task automatic inspect_slot(
        input logic context_id,
        input logic [1:0] head_id,
        input logic expected_exists,
        input logic [31:0] expected_tag,
        input logic expected_is_csr,
        input logic [15:0] expected_payload_bits
    );
        logic [15:0] expected_words;
        begin
            expected_words = 16'((32'(expected_payload_bits) + WORD_W - 1) /
                                 WORD_W);
            @(negedge clk_core);
            inspect_context_id = context_id;
            inspect_head_id = head_id;
            inspect_valid = 1'b1;
            do @(posedge clk_core); while (!inspect_ready);
            @(negedge clk_core);
            inspect_valid = 1'b0;
            repeat (2) @(posedge clk_core);
            if (!inspect_meta_valid || inspect_exists != expected_exists ||
                (expected_exists &&
                 (inspect_tag != expected_tag ||
                  inspect_mode_is_csr != expected_is_csr ||
                  inspect_format != (expected_is_csr ? 2'd1 : 2'd0) ||
                  inspect_payload_bits != expected_payload_bits ||
                  inspect_word_count != 16'(expected_words)))) begin
                $fatal(1, "inspect mismatch head=%0d", head_id);
            end
            @(negedge clk_core);
            inspect_meta_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            inspect_meta_ready = 1'b0;
        end
    endtask

    task automatic release_slot(
        input logic context_id,
        input logic [1:0] head_id
    );
        begin
            @(negedge clk_core);
            release_context_id = 1'(context_id);
            release_head_id = 2'(head_id);
            release_valid = 1'b1;
            do @(posedge clk_core); while (!release_ready);
            @(negedge clk_core);
            release_valid = 1'b0;
        end
    endtask

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            overlap_cycles <= 0;
        end else if (commit_session_active && replay_session_active) begin
            overlap_cycles <= overlap_cycles + 1;
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        commit_begin_valid = 1'b0;
        commit_context_id = '0;
        commit_head_id = '0;
        commit_tag = '0;
        commit_mode_is_csr = 1'b0;
        commit_payload_bits = '0;
        commit_word_valid = 1'b0;
        commit_word_data = '0;
        commit_word_last = 1'b0;
        inspect_valid = 1'b0;
        inspect_context_id = '0;
        inspect_head_id = '0;
        inspect_meta_ready = 1'b0;
        replay_begin_valid = 1'b0;
        replay_context_id = '0;
        replay_head_id = '0;
        replay_start_word = '0;
        replay_word_ready = 1'b0;
        release_valid = 1'b0;
        release_context_id = '0;
        release_head_id = '0;
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;

        // A long CSR slot creates enough replay time to overlap a full RAW commit.
        commit_slot(0, 0, 32'hc500_0000, 1'b1, 4096, 32'h1357_0000);
        if (!slot_valid_flat[0] || protocol_error) $fatal(1, "CSR commit failed");
        inspect_slot(0, 0, 1'b1, 32'hc500_0000, 1'b1, 4096);

        fork
            replay_slot(0, 0, 32'hc500_0000, 1'b1, 4096,
                        32'h1357_0000, 0);
            commit_slot(0, 1, 32'hc500_0001, 1'b0, 16'(HEAD_BITS),
                        32'h2468_0000);
        join
        if (!slot_valid_flat[1] || overlap_cycles == 0 || protocol_error)
            $fatal(1, "1W1R overlap failed");

        release_slot(0, 0);
        inspect_slot(0, 0, 1'b0, 32'd0, 1'b0, 16'd0);
        replay_slot(0, 1, 32'hc500_0001, 1'b0, 16'(HEAD_BITS),
                    32'h2468_0000, 0);
        release_slot(0, 1);

        // Compressed formats may use all 104 physical words; RAW legality is
        // checked separately by the replay planner against HEAD_BITS=6642.
        commit_slot(0, 0, 32'hc500_0010, 1'b1,
                    16'(SLOT_CAPACITY_BITS), 32'h369c_0000);
        replay_slot(0, 0, 32'hc500_0010, 1'b1,
                    16'(SLOT_CAPACITY_BITS), 32'h369c_0000, 0);
        release_slot(0, 0);

        // Unknown CSR magic drains the transaction but never publishes a slot.
        begin_commit(0, 0, 32'hc500_00ff, 1'b1, 130);
        for (int index = 0; index < 3; index = index + 1) begin
            @(negedge clk_core);
            commit_word_valid = 1'b1;
            commit_word_data = index == 0 ?
                64'hc500_00ff_0001_dead : word_pattern(32'hbad0_0000, index);
            commit_word_last = index == 2;
            do @(posedge clk_core); while (!commit_word_ready);
            @(negedge clk_core);
            commit_word_valid = 1'b0;
            commit_word_last = 1'b0;
        end
        inspect_slot(0, 0, 1'b0, 32'd0, 1'b0, 16'd0);
        if (slot_valid_flat[0] || count_invalid_headers != 1)
            $fatal(1, "invalid CSR header became visible");

        // A short tail payload checks ceil(bits/64), metadata and context mapping.
        commit_slot(1, 2, 32'hc501_0002, 1'b1, 130, 32'h55aa_0000);
        replay_slot(1, 2, 32'hc501_0002, 1'b1, 130,
                    32'h55aa_0000, 1);

        // start_word equal to slot_words is outside the replay subrange.
        @(negedge clk_core);
        replay_context_id = 1'b1;
        replay_head_id = 2'd2;
        replay_start_word = 7'd3;
        replay_begin_valid = 1'b1;
        @(posedge clk_core);
        if (replay_begin_ready) $fatal(1, "invalid start_word was accepted");
        @(negedge clk_core);
        replay_begin_valid = 1'b0;
        repeat (2) @(posedge clk_core);
        if (!protocol_error) $fatal(1, "invalid start_word was not flagged");
        release_slot(1, 2);
        if (slot_valid_flat != '0) $fatal(1, "release did not clear slots");

        // Premature last aborts the transaction and cannot expose a valid slot.
        begin_commit(1, 1, 32'hdead_0001, 1'b1, 130);
        @(negedge clk_core);
        commit_word_valid = 1'b1;
        commit_word_data = 64'hdead_beef_0000_0000;
        commit_word_last = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        commit_word_valid = 1'b0;
        commit_word_last = 1'b0;
        repeat (2) @(posedge clk_core);
        if (!protocol_error || slot_valid_flat[4])
            $fatal(1, "bad commit was not rejected");

        if (count_commit_heads != 4 || count_replay_heads != 4 ||
            count_release_heads != 4)
            $fatal(1, "transaction counters mismatch");
        $display("PASS: GateStack head-slot adapter commits=%0d replays=%0d releases=%0d overlap=%0d commit_stall=%0d replay_stall=%0d",
                 count_commit_heads, count_replay_heads, count_release_heads,
                 overlap_cycles, count_commit_stall_cycles,
                 count_replay_stall_cycles);
        $finish;
    end

    initial begin
        repeat (20000) @(posedge clk_core);
        $fatal(1, "head-slot TB timeout");
    end

endmodule

`default_nettype wire
