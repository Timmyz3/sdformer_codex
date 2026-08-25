`timescale 1ns/1ps
`default_nettype none

module tb_m142_sparse_mask_k4_bounded_overlap_controller;
    localparam int TAG_BITS = 16;
    localparam int ROW_BITS = 9;
    localparam int BANKS = 4;
    localparam int WINDOWS = 32;

    logic clk_core = 0;
    logic rst_core;
    always #1.5 clk_core = ~clk_core;

    logic row_valid, row_ready, row_window_start, row_window_end;
    logic [TAG_BITS-1:0] row_window_tag;
    logic [ROW_BITS-1:0] row_id;
    logic [15:0] row_source_mask [0:7];
    logic [15:0] row_negate_mask [0:7];
    logic row_accept;
    logic descriptor_valid, descriptor_ready;
    logic [1:0] descriptor_bank;
    logic [TAG_BITS-1:0] descriptor_window_tag;
    logic [ROW_BITS-1:0] descriptor_row;
    logic [2:0] descriptor_block;
    logic [1:0] descriptor_source_count_m1;
    logic [3:0] descriptor_source [0:3];
    logic [3:0] descriptor_negate;
    logic descriptor_row_last, descriptor_window_last, descriptor_accept;
    logic pwp_valid, pwp_ready, pwp_accept;
    logic [1:0] pwp_bank;
    logic [TAG_BITS-1:0] pwp_window_tag;
    logic pwp_done_valid;
    logic [1:0] pwp_done_bank;
    logic [TAG_BITS-1:0] pwp_done_window_tag;
    logic correction_valid, correction_ready, correction_accept;
    logic [1:0] correction_bank;
    logic [TAG_BITS-1:0] correction_window_tag;
    logic correction_done_valid;
    logic [1:0] correction_done_bank;
    logic [TAG_BITS-1:0] correction_done_window_tag;
    logic [BANKS-1:0] observed_bank_free, observed_bank_fill;
    logic [BANKS-1:0] observed_bank_filled, observed_bank_pwp;
    logic [BANKS-1:0] observed_bank_wait_correction;
    logic [BANKS-1:0] observed_bank_correction;
    logic observed_window_open, observed_pwp_busy;
    logic observed_correction_busy, protocol_error, busy;

    logic attack_mux;
    logic attack_pwp_done_valid, attack_correction_done_valid;
    logic [1:0] attack_done_bank;
    logic [TAG_BITS-1:0] attack_done_tag;
    logic model_pwp_done_valid, model_correction_done_valid;
    logic [1:0] model_pwp_done_bank, model_correction_done_bank;
    logic [TAG_BITS-1:0] model_pwp_done_tag, model_correction_done_tag;

    typedef struct packed {
        logic [TAG_BITS-1:0] tag;
        logic [ROW_BITS-1:0] row;
        logic [2:0] block_id;
        logic [1:0] count_m1;
        logic [15:0] sources;
        logic [3:0] negate;
        logic row_last;
        logic window_last;
    } expected_descriptor_t;
    expected_descriptor_t expected_descriptors[$];
    logic [TAG_BITS-1:0] expected_pwp_tags[$];
    logic [TAG_BITS-1:0] expected_correction_tags[$];

    int unsigned cycle_count;
    int unsigned rows_accepted, descriptors_accepted, sources_checked;
    int unsigned pwp_accepted, correction_accepted, corrections_completed;
    int unsigned descriptor_stall_cycles, pwp_stall_cycles;
    int unsigned correction_stall_cycles, descriptor_ii1;
    int unsigned zero_rows, bank_reuses;
    int signed last_descriptor_cycle;
    bit saw_engine_overlap, saw_four_owned, saw_k4, saw_bank_reuse;
    bit tag_bank_valid [0:255];
    logic [1:0] tag_bank [0:255];
    bit bank_seen [0:BANKS-1];

    assign pwp_done_valid = attack_mux
        ? attack_pwp_done_valid : model_pwp_done_valid;
    assign pwp_done_bank = attack_mux
        ? attack_done_bank : model_pwp_done_bank;
    assign pwp_done_window_tag = attack_mux
        ? attack_done_tag : model_pwp_done_tag;
    assign correction_done_valid = attack_mux
        ? attack_correction_done_valid : model_correction_done_valid;
    assign correction_done_bank = attack_mux
        ? attack_done_bank : model_correction_done_bank;
    assign correction_done_window_tag = attack_mux
        ? attack_done_tag : model_correction_done_tag;

    m142_sparse_mask_k4_bounded_overlap_controller #(
        .TAG_BITS(TAG_BITS), .ROW_BITS(ROW_BITS), .BANKS(BANKS)
    ) dut (.*);

    m142_sparse_mask_k4_bounded_overlap_controller_assertions #(
        .TAG_BITS(TAG_BITS), .ROW_BITS(ROW_BITS), .BANKS(BANKS)
    ) sva (.*);

    task automatic enqueue_expected_descriptors(
        input logic [TAG_BITS-1:0] tag,
        input logic [ROW_BITS-1:0] row,
        input logic [127:0] masks,
        input logic [127:0] negate_masks,
        input logic window_last
    );
        logic [15:0] remaining;
        logic found;
        logic future_nonzero;
        expected_descriptor_t item;
        int count;
        for (int block = 0; block < 8; block++) begin
            remaining = masks[block * 16 +: 16];
            while (remaining != 0) begin
                item = '0;
                item.tag = tag;
                item.row = row;
                item.block_id = block[2:0];
                count = 0;
                for (int slot = 0; slot < 4; slot++) begin
                    found = 1'b0;
                    for (int source = 0; source < 16; source++) begin
                        if (!found && remaining[source]) begin
                            item.sources[slot * 4 +: 4] = source[3:0];
                            item.negate[slot]
                                = negate_masks[block * 16 + source];
                            remaining[source] = 1'b0;
                            count++;
                            found = 1'b1;
                        end
                    end
                end
                item.count_m1 = count - 1;
                future_nonzero = 1'b0;
                for (int future = block + 1; future < 8; future++) begin
                    if (masks[future * 16 +: 16] != 0)
                        future_nonzero = 1'b1;
                end
                item.row_last = remaining == 0 && !future_nonzero;
                item.window_last = window_last && item.row_last;
                expected_descriptors.push_back(item);
            end
        end
    endtask

    task automatic drive_row(
        input logic start_window,
        input logic end_window,
        input logic [TAG_BITS-1:0] tag,
        input logic [ROW_BITS-1:0] row,
        input logic [127:0] masks,
        input logic [127:0] negate_masks
    );
        if (start_window)
            expected_pwp_tags.push_back(tag);
        if (masks == 0)
            zero_rows++;
        enqueue_expected_descriptors(tag, row, masks, negate_masks,
                                     end_window);
        @(negedge clk_core);
        row_valid = 1'b1;
        row_window_start = start_window;
        row_window_end = end_window;
        row_window_tag = tag;
        row_id = row;
        for (int block = 0; block < 8; block++) begin
            row_source_mask[block] = masks[block * 16 +: 16];
            row_negate_mask[block]
                = negate_masks[block * 16 +: 16];
        end
        while (!row_ready)
            @(negedge clk_core);
        @(posedge clk_core);
        if (!row_accept)
            $fatal(1, "M142 expected row acceptance tag=%0d row=%0d",
                   tag, row);
        @(negedge clk_core);
        row_valid = 1'b0;
        row_window_start = 1'b0;
        row_window_end = 1'b0;
        for (int block = 0; block < 8; block++) begin
            row_source_mask[block] = '0;
            row_negate_mask[block] = '0;
        end
    endtask

    task automatic apply_reset;
        @(negedge clk_core);
        rst_core = 1'b1;
        row_valid = 1'b0;
        attack_pwp_done_valid = 1'b0;
        attack_correction_done_valid = 1'b0;
        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        repeat (2) @(posedge clk_core);
    endtask

    always_comb begin
        descriptor_ready = (cycle_count % 11) != 4;
        pwp_ready = (cycle_count % 7) != 2;
        correction_ready = (cycle_count % 9) != 3;
    end

    always @(posedge clk_core) begin : clock_counter
        if (rst_core)
            cycle_count <= 0;
        else
            cycle_count <= cycle_count + 1;
    end

    always @(posedge clk_core) begin : pwp_engine_model
        static logic engine_busy;
        static int unsigned countdown;
        static logic [1:0] held_bank;
        static logic [TAG_BITS-1:0] held_tag;
        if (rst_core || attack_mux) begin
            engine_busy <= 1'b0;
            countdown <= 0;
            held_bank <= '0;
            held_tag <= '0;
            model_pwp_done_valid <= 1'b0;
            model_pwp_done_bank <= '0;
            model_pwp_done_tag <= '0;
        end else begin
            model_pwp_done_valid <= 1'b0;
            if (pwp_accept) begin
                if (engine_busy)
                    $fatal(1, "M142 PWP engine double launch");
                engine_busy <= 1'b1;
                countdown <= 80 + pwp_window_tag[2:0];
                held_bank <= pwp_bank;
                held_tag <= pwp_window_tag;
            end else if (engine_busy && countdown != 0) begin
                countdown <= countdown - 1;
            end else if (engine_busy) begin
                model_pwp_done_valid <= 1'b1;
                model_pwp_done_bank <= held_bank;
                model_pwp_done_tag <= held_tag;
                engine_busy <= 1'b0;
            end
        end
    end

    always @(posedge clk_core) begin : correction_engine_model
        static logic engine_busy;
        static int unsigned countdown;
        static logic [1:0] held_bank;
        static logic [TAG_BITS-1:0] held_tag;
        if (rst_core || attack_mux) begin
            engine_busy <= 1'b0;
            countdown <= 0;
            held_bank <= '0;
            held_tag <= '0;
            model_correction_done_valid <= 1'b0;
            model_correction_done_bank <= '0;
            model_correction_done_tag <= '0;
        end else begin
            model_correction_done_valid <= 1'b0;
            if (correction_accept) begin
                if (engine_busy)
                    $fatal(1, "M142 correction engine double launch");
                engine_busy <= 1'b1;
                countdown <= 160 + correction_window_tag[3:0];
                held_bank <= correction_bank;
                held_tag <= correction_window_tag;
            end else if (engine_busy && countdown != 0) begin
                countdown <= countdown - 1;
            end else if (engine_busy) begin
                model_correction_done_valid <= 1'b1;
                model_correction_done_bank <= held_bank;
                model_correction_done_tag <= held_tag;
                engine_busy <= 1'b0;
            end
        end
    end

    always @(posedge clk_core) begin : scoreboard
        expected_descriptor_t expected;
        logic [15:0] actual_sources;
        if (!rst_core) begin
            if (row_accept)
                rows_accepted <= rows_accepted + 1;
            if (descriptor_valid && !descriptor_ready)
                descriptor_stall_cycles <= descriptor_stall_cycles + 1;
            if (pwp_valid && !pwp_ready)
                pwp_stall_cycles <= pwp_stall_cycles + 1;
            if (correction_valid && !correction_ready)
                correction_stall_cycles <= correction_stall_cycles + 1;
            if (observed_pwp_busy && observed_correction_busy)
                saw_engine_overlap <= 1'b1;
            if (observed_bank_free == 0)
                saw_four_owned <= 1'b1;

            if (descriptor_accept) begin
                if (expected_descriptors.size() == 0)
                    $fatal(1, "M142 unexpected descriptor");
                expected = expected_descriptors.pop_front();
                actual_sources = {descriptor_source[3],
                                  descriptor_source[2],
                                  descriptor_source[1],
                                  descriptor_source[0]};
                if (descriptor_window_tag !== expected.tag
                        || descriptor_row !== expected.row
                        || descriptor_block !== expected.block_id
                        || descriptor_source_count_m1
                           !== expected.count_m1
                        || actual_sources !== expected.sources
                        || descriptor_negate !== expected.negate
                        || descriptor_row_last !== expected.row_last
                        || descriptor_window_last
                           !== expected.window_last)
                    $fatal(1, "M142 descriptor mismatch tag=%0d row=%0d",
                           descriptor_window_tag, descriptor_row);
                if (descriptor_source_count_m1 == 3)
                    saw_k4 <= 1'b1;
                descriptors_accepted <= descriptors_accepted + 1;
                sources_checked <= sources_checked
                    + descriptor_source_count_m1 + 1;
                if (last_descriptor_cycle + 1 == cycle_count)
                    descriptor_ii1 <= descriptor_ii1 + 1;
                last_descriptor_cycle <= cycle_count;
                if (descriptor_window_tag < 256) begin
                    if (!tag_bank_valid[descriptor_window_tag]) begin
                        tag_bank_valid[descriptor_window_tag] <= 1'b1;
                        tag_bank[descriptor_window_tag] <= descriptor_bank;
                    end else if (tag_bank[descriptor_window_tag]
                                 != descriptor_bank) begin
                        $fatal(1, "M142 tag changed banks during fill");
                    end
                end
            end

            if (pwp_accept) begin
                if (expected_pwp_tags.size() == 0
                        || expected_pwp_tags.pop_front()
                           !== pwp_window_tag)
                    $fatal(1, "M142 PWP order mismatch tag=%0d",
                           pwp_window_tag);
                expected_correction_tags.push_back(pwp_window_tag);
                pwp_accepted <= pwp_accepted + 1;
                if (pwp_window_tag < 256) begin
                    if (tag_bank_valid[pwp_window_tag]
                            && tag_bank[pwp_window_tag] != pwp_bank)
                        $fatal(1, "M142 PWP bank identity mismatch");
                    tag_bank_valid[pwp_window_tag] <= 1'b1;
                    tag_bank[pwp_window_tag] <= pwp_bank;
                end
                if (bank_seen[pwp_bank]) begin
                    bank_reuses <= bank_reuses + 1;
                    saw_bank_reuse <= 1'b1;
                end
                bank_seen[pwp_bank] <= 1'b1;
            end

            if (correction_accept) begin
                if (expected_correction_tags.size() == 0
                        || expected_correction_tags.pop_front()
                           !== correction_window_tag)
                    $fatal(1, "M142 correction order mismatch tag=%0d",
                           correction_window_tag);
                if (correction_window_tag < 256
                        && tag_bank[correction_window_tag]
                           != correction_bank)
                    $fatal(1, "M142 correction bank identity mismatch");
                correction_accepted <= correction_accepted + 1;
            end
            if (correction_done_valid && !attack_mux)
                corrections_completed <= corrections_completed + 1;
        end
    end

    initial begin : stimulus
        int watchdog;
        rst_core = 1'b1;
        row_valid = 1'b0;
        row_window_start = 1'b0;
        row_window_end = 1'b0;
        row_window_tag = '0;
        row_id = '0;
        for (int block = 0; block < 8; block++) begin
            row_source_mask[block] = '0;
            row_negate_mask[block] = '0;
        end
        attack_mux = 1'b0;
        attack_pwp_done_valid = 1'b0;
        attack_correction_done_valid = 1'b0;
        attack_done_bank = '0;
        attack_done_tag = '0;
        cycle_count = 0;
        rows_accepted = 0;
        descriptors_accepted = 0;
        sources_checked = 0;
        pwp_accepted = 0;
        correction_accepted = 0;
        corrections_completed = 0;
        descriptor_stall_cycles = 0;
        pwp_stall_cycles = 0;
        correction_stall_cycles = 0;
        descriptor_ii1 = 0;
        zero_rows = 0;
        bank_reuses = 0;
        last_descriptor_cycle = -100;
        saw_engine_overlap = 1'b0;
        saw_four_owned = 1'b0;
        saw_k4 = 1'b0;
        saw_bank_reuse = 1'b0;
        for (int tag = 0; tag < 256; tag++) begin
            tag_bank_valid[tag] = 1'b0;
            tag_bank[tag] = '0;
        end
        for (int bank = 0; bank < BANKS; bank++)
            bank_seen[bank] = 1'b0;

        apply_reset();
        for (int window = 0; window < WINDOWS; window++) begin
            drive_row(1'b1, 1'b0, window + 16,
                      0, 128'h0, 128'h0);
            drive_row(1'b0, 1'b0, window + 16,
                      1,
                      {80'h0, 16'h0011, 16'h000f, 16'h0f3d},
                      {80'h0, 16'h0010, 16'h0005, 16'h0121});
            drive_row(1'b0, 1'b1, window + 16,
                      2, {8{16'hffff}}, {8{16'ha5a5}});
        end

        watchdog = 0;
        while (corrections_completed != WINDOWS && watchdog < 5000) begin
            @(posedge clk_core);
            watchdog++;
        end
        if (watchdog == 5000)
            $fatal(1, "M142 normal traffic failed to drain");
        repeat (4) @(posedge clk_core);
        if (expected_descriptors.size() != 0
                || expected_pwp_tags.size() != 0
                || expected_correction_tags.size() != 0
                || busy || protocol_error)
            $fatal(1, "M142 normal scoreboard did not close");
        if (rows_accepted != WINDOWS * 3
                || descriptors_accepted != WINDOWS * 37
                || sources_checked != WINDOWS * 143
                || pwp_accepted != WINDOWS
                || correction_accepted != WINDOWS
                || corrections_completed != WINDOWS
                || !saw_engine_overlap || !saw_four_owned || !saw_k4
                || !saw_bank_reuse || descriptor_ii1 == 0)
            $fatal(1, "M142 normal coverage contract incomplete");

        // Unexpected PWP completion must fail closed.
        attack_mux = 1'b1;
        apply_reset();
        @(negedge clk_core);
        attack_done_bank = 2'd0;
        attack_done_tag = 16'hdead;
        attack_pwp_done_valid = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        attack_pwp_done_valid = 1'b0;
        if (!protocol_error)
            $fatal(1, "M142 unexpected PWP completion not quarantined");

        // Unexpected correction completion must fail closed.
        apply_reset();
        @(negedge clk_core);
        attack_done_bank = 2'd1;
        attack_done_tag = 16'hbeef;
        attack_correction_done_valid = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        attack_correction_done_valid = 1'b0;
        if (!protocol_error)
            $fatal(1, "M142 unexpected correction completion not quarantined");

        // A sign outside the raw source mask is malformed, not backpressure.
        apply_reset();
        @(negedge clk_core);
        row_valid = 1'b1;
        row_window_start = 1'b1;
        row_window_end = 1'b1;
        row_window_tag = 16'h1234;
        row_id = 9'd9;
        row_source_mask[0] = 16'h0001;
        row_negate_mask[0] = 16'h0002;
        @(posedge clk_core);
        @(negedge clk_core);
        row_valid = 1'b0;
        if (!protocol_error || row_accept)
            $fatal(1, "M142 dirty sign padding not quarantined");

        $display("PASS M142 bounded overlap VCS banks=4 windows=%0d rows=%0d zero_rows=%0d descriptors=%0d sources=%0d pwp=%0d correction=%0d completed=%0d descriptor_ii1=%0d descriptor_stalls=%0d pwp_stalls=%0d correction_stalls=%0d bank_reuses=%0d early_pwp=0 raw_zero_rows_accepted=true protocol_attacks=3 pwp_correction_overlap=1 all_banks_owned=1 engine_arithmetic=false sram_macro=false physical_speedup=false system_speedup=false headline=false",
                 WINDOWS, rows_accepted, zero_rows, descriptors_accepted,
                 sources_checked, pwp_accepted, correction_accepted,
                 corrections_completed, descriptor_ii1,
                 descriptor_stall_cycles, pwp_stall_cycles,
                 correction_stall_cycles, bank_reuses);
        $finish;
    end

    initial begin : watchdog
        #1000000;
        $fatal(1, "M142 watchdog timeout");
    end
endmodule

`default_nettype wire
