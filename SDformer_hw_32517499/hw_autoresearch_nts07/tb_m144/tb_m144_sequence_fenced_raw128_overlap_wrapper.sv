`timescale 1ns/1ps
`default_nettype none

module tb_m144_sequence_fenced_raw128_overlap_wrapper;
    localparam int TAG_BITS = 16;
    localparam int ROW_BITS = 9;
    localparam int BANKS = 4;
    localparam int SEQUENCE_BITS = 32;
    localparam int NORMAL_JOBS = 5;

    logic clk_core = 0;
    logic rst_core;
    always #1.5 clk_core = ~clk_core;

    logic row_valid, row_ready, row_window_start, row_window_end;
    logic [TAG_BITS-1:0] row_window_tag;
    logic [ROW_BITS-1:0] row_id, row_window_rows;
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
    logic [SEQUENCE_BITS-1:0] pwp_sequence;
    logic pwp_done_valid;
    logic [1:0] pwp_done_bank;
    logic [TAG_BITS-1:0] pwp_done_window_tag;
    logic [SEQUENCE_BITS-1:0] pwp_done_sequence;
    logic correction_valid, correction_ready, correction_accept;
    logic [1:0] correction_bank;
    logic [TAG_BITS-1:0] correction_window_tag;
    logic [SEQUENCE_BITS-1:0] correction_sequence;
    logic correction_done_valid;
    logic [1:0] correction_done_bank;
    logic [TAG_BITS-1:0] correction_done_window_tag;
    logic [SEQUENCE_BITS-1:0] correction_done_sequence;
    logic outer_barrier_valid, outer_barrier_ready;
    logic [TAG_BITS-1:0] outer_barrier_tag;
    logic outer_barrier_accept, outer_commit_valid;
    logic outer_commit_done_valid, outer_commit_done_accept;
    logic [TAG_BITS-1:0] outer_commit_done_tag, outer_commit_tag;
    logic [SEQUENCE_BITS-1:0] outer_commit_fence_sequence;
    logic [BANKS-1:0] observed_bank_free, observed_bank_fill;
    logic [BANKS-1:0] observed_bank_filled, observed_bank_pwp;
    logic [BANKS-1:0] observed_bank_wait_correction;
    logic [BANKS-1:0] observed_bank_correction;
    logic observed_window_open, observed_pwp_busy;
    logic observed_correction_busy, observed_barrier_active;
    logic [SEQUENCE_BITS-1:0] observed_next_sequence;
    logic [SEQUENCE_BITS-1:0] observed_next_completion_sequence;
    logic protocol_error, busy;

    logic engine_enable, attack_mode;
    int unsigned cycle_count, rows_accepted, descriptors_accepted;
    int unsigned pwp_accepted, correction_accepted;
    int unsigned corrections_completed, barrier_accepts, commit_accepts;
    int unsigned protocol_attacks, next_expected_pwp;
    int unsigned next_expected_correction;
    bit saw_four_owned, saw_post_fence_lookahead;
    bit saw_barrier_block, saw_commit_stall, saw_min_endpoint;

    m144_sequence_fenced_raw128_overlap_wrapper #(
        .TAG_BITS(TAG_BITS), .ROW_BITS(ROW_BITS), .BANKS(BANKS),
        .SEQUENCE_BITS(SEQUENCE_BITS)
    ) dut (.*);

    m144_sequence_fenced_raw128_overlap_wrapper_assertions #(
        .TAG_BITS(TAG_BITS), .SEQUENCE_BITS(SEQUENCE_BITS),
        .BANKS(BANKS)
    ) sva (.*);

    task automatic clear_request_inputs;
        row_valid = 1'b0;
        row_window_start = 1'b0;
        row_window_end = 1'b0;
        row_window_tag = '0;
        row_id = '0;
        row_window_rows = '0;
        for (int block = 0; block < 8; block++) begin
            row_source_mask[block] = '0;
            row_negate_mask[block] = '0;
        end
        outer_barrier_valid = 1'b0;
        outer_barrier_tag = '0;
        outer_commit_done_valid = 1'b0;
        outer_commit_done_tag = '0;
    endtask

    task automatic apply_reset;
        @(negedge clk_core);
        rst_core = 1'b1;
        clear_request_inputs();
        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        repeat (2) @(posedge clk_core);
    endtask

    task automatic drive_one_row_job(
        input logic [TAG_BITS-1:0] tag,
        input logic [127:0] masks,
        input logic [127:0] negate_masks
    );
        @(negedge clk_core);
        row_valid = 1'b1;
        row_window_start = 1'b1;
        row_window_end = 1'b1;
        row_window_tag = tag;
        row_id = 0;
        row_window_rows = 1;
        for (int block = 0; block < 8; block++) begin
            row_source_mask[block] = masks[block * 16 +: 16];
            row_negate_mask[block]
                = negate_masks[block * 16 +: 16];
        end
        #0.1;
        while (!row_ready)
            @(negedge clk_core);
        @(posedge clk_core);
        if (!row_accept)
            $fatal(1, "M144 expected row acceptance tag=%0d", tag);
        #0.1;
        row_valid = 1'b0;
        row_window_start = 1'b0;
        row_window_end = 1'b0;
        for (int block = 0; block < 8; block++) begin
            row_source_mask[block] = '0;
            row_negate_mask[block] = '0;
        end
    endtask

    task automatic send_barrier(input logic [TAG_BITS-1:0] tag);
        @(negedge clk_core);
        outer_barrier_valid = 1'b1;
        outer_barrier_tag = tag;
        #0.1;
        while (!outer_barrier_ready)
            @(negedge clk_core);
        @(posedge clk_core);
        if (!outer_barrier_accept)
            $fatal(1, "M144 expected barrier acceptance");
        #0.1;
        outer_barrier_valid = 1'b0;
    endtask

    always_comb begin
        descriptor_ready = (cycle_count % 7) != 3;
        pwp_ready = engine_enable && (cycle_count % 5) != 1;
        correction_ready = engine_enable && (cycle_count % 6) != 2;
    end

    always @(posedge clk_core) begin
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
        static logic [SEQUENCE_BITS-1:0] held_sequence;
        static int unsigned launch_cycle;
        if (rst_core || attack_mode) begin
            engine_busy <= 1'b0;
            countdown <= 0;
            pwp_done_valid <= 1'b0;
            pwp_done_bank <= '0;
            pwp_done_window_tag <= '0;
            pwp_done_sequence <= '0;
        end else begin
            pwp_done_valid <= 1'b0;
            if (pwp_accept) begin
                if (engine_busy)
                    $fatal(1, "M144 PWP double launch");
                engine_busy <= 1'b1;
                countdown <= 2 + pwp_sequence[0];
                held_bank <= pwp_bank;
                held_tag <= pwp_window_tag;
                held_sequence <= pwp_sequence;
                launch_cycle <= cycle_count;
            end else if (engine_busy && countdown != 0) begin
                countdown <= countdown - 1;
            end else if (engine_busy) begin
                if (cycle_count <= launch_cycle)
                    $fatal(1, "M144 zero-cycle PWP endpoint");
                pwp_done_valid <= 1'b1;
                pwp_done_bank <= held_bank;
                pwp_done_window_tag <= held_tag;
                pwp_done_sequence <= held_sequence;
                engine_busy <= 1'b0;
                saw_min_endpoint <= 1'b1;
            end
        end
    end

    always @(posedge clk_core) begin : correction_engine_model
        static logic engine_busy;
        static int unsigned countdown;
        static logic [1:0] held_bank;
        static logic [TAG_BITS-1:0] held_tag;
        static logic [SEQUENCE_BITS-1:0] held_sequence;
        if (rst_core || attack_mode) begin
            engine_busy <= 1'b0;
            countdown <= 0;
            correction_done_valid <= 1'b0;
            correction_done_bank <= '0;
            correction_done_window_tag <= '0;
            correction_done_sequence <= '0;
        end else begin
            correction_done_valid <= 1'b0;
            if (correction_accept) begin
                if (engine_busy)
                    $fatal(1, "M144 correction double launch");
                engine_busy <= 1'b1;
                countdown <= 3 + correction_sequence[0];
                held_bank <= correction_bank;
                held_tag <= correction_window_tag;
                held_sequence <= correction_sequence;
            end else if (engine_busy && countdown != 0) begin
                countdown <= countdown - 1;
            end else if (engine_busy) begin
                correction_done_valid <= 1'b1;
                correction_done_bank <= held_bank;
                correction_done_window_tag <= held_tag;
                correction_done_sequence <= held_sequence;
                engine_busy <= 1'b0;
            end
        end
    end

    always @(posedge clk_core) begin : normal_scoreboard
        if (!rst_core && !attack_mode) begin
            if (row_accept)
                rows_accepted <= rows_accepted + 1;
            if (descriptor_accept)
                descriptors_accepted <= descriptors_accepted + 1;
            if (outer_barrier_accept)
                barrier_accepts <= barrier_accepts + 1;
            if (outer_commit_done_accept)
                commit_accepts <= commit_accepts + 1;
            if (observed_bank_free == 0)
                saw_four_owned <= 1'b1;
            if (observed_barrier_active && row_accept)
                saw_post_fence_lookahead <= 1'b1;
            if (observed_barrier_active && dut.lower_pwp_valid
                    && dut.pwp_sequence > outer_commit_fence_sequence
                    && !pwp_valid)
                saw_barrier_block <= 1'b1;
            if (outer_commit_valid && !outer_commit_done_valid)
                saw_commit_stall <= 1'b1;

            if (pwp_accept) begin
                if (pwp_sequence !== next_expected_pwp
                        || pwp_window_tag !== 16'd100
                                             + next_expected_pwp)
                    $fatal(1, "M144 PWP sequence/tag mismatch");
                next_expected_pwp <= next_expected_pwp + 1;
                pwp_accepted <= pwp_accepted + 1;
            end
            if (correction_accept) begin
                if (correction_sequence !== next_expected_correction
                        || correction_window_tag !== 16'd100
                                                + next_expected_correction)
                    $fatal(1, "M144 correction sequence/tag mismatch");
                next_expected_correction <= next_expected_correction + 1;
                correction_accepted <= correction_accepted + 1;
            end
            if (correction_done_valid)
                corrections_completed <= corrections_completed + 1;
        end
    end

    initial begin : stimulus
        int watchdog;
        logic [1:0] held_attack_bank;
        logic [TAG_BITS-1:0] held_attack_tag;
        logic [SEQUENCE_BITS-1:0] held_attack_sequence;

        rst_core = 1'b1;
        engine_enable = 1'b0;
        attack_mode = 1'b0;
        clear_request_inputs();
        cycle_count = 0;
        rows_accepted = 0;
        descriptors_accepted = 0;
        pwp_accepted = 0;
        correction_accepted = 0;
        corrections_completed = 0;
        barrier_accepts = 0;
        commit_accepts = 0;
        protocol_attacks = 0;
        next_expected_pwp = 0;
        next_expected_correction = 0;
        saw_four_owned = 1'b0;
        saw_post_fence_lookahead = 1'b0;
        saw_barrier_block = 1'b0;
        saw_commit_stall = 1'b0;
        saw_min_endpoint = 1'b0;

        apply_reset();
        drive_one_row_job(16'd100, 128'h0, 128'h0);
        drive_one_row_job(16'd101,
                          {96'h0, 16'h000f, 16'h00f3},
                          {96'h0, 16'h0005, 16'h0051});
        send_barrier(16'hb144);
        if (outer_commit_fence_sequence !== 1)
            $fatal(1, "M144 barrier fence=%0d next=%0d rows=%0d did not close at sequence 1",
                   outer_commit_fence_sequence, observed_next_sequence,
                   rows_accepted);
        drive_one_row_job(16'd102, 128'h0, 128'h0);
        drive_one_row_job(16'd103,
                          {112'h0, 16'hffff},
                          {112'h0, 16'ha5a5});
        engine_enable = 1'b1;

        watchdog = 0;
        while (!outer_commit_valid && watchdog < 500) begin
            @(posedge clk_core);
            watchdog++;
        end
        if (watchdog == 500)
            $fatal(1, "M144 fence failed to drain");
        if (pwp_accepted != 2 || correction_accepted != 2
                || corrections_completed != 2)
            $fatal(1, "M144 crossed barrier before commit");

        repeat (3) @(posedge clk_core);
        drive_one_row_job(16'd104,
                          {64'h0, 16'h0001, 48'h0},
                          {64'h0, 16'h0001, 48'h0});
        repeat (2) @(posedge clk_core);
        @(negedge clk_core);
        outer_commit_done_valid = 1'b1;
        outer_commit_done_tag = 16'hb144;
        @(posedge clk_core);
        if (!outer_commit_done_accept)
            $fatal(1, "M144 correct commit acknowledgement rejected");
        @(negedge clk_core);
        outer_commit_done_valid = 1'b0;

        watchdog = 0;
        while (corrections_completed != NORMAL_JOBS
                && watchdog < 1000) begin
            @(posedge clk_core);
            watchdog++;
        end
        if (watchdog == 1000)
            $fatal(1, "M144 normal traffic failed to drain");
        repeat (5) @(posedge clk_core);
        if (busy || protocol_error || rows_accepted != NORMAL_JOBS
                || pwp_accepted != NORMAL_JOBS
                || correction_accepted != NORMAL_JOBS
                || corrections_completed != NORMAL_JOBS
                || barrier_accepts != 1 || commit_accepts != 1
                || !saw_four_owned || !saw_post_fence_lookahead
                || !saw_barrier_block || !saw_commit_stall
                || !saw_min_endpoint)
            $fatal(1, "M144 normal contract incomplete");

        // Malformed relative row identity is a protocol error, not pressure.
        attack_mode = 1'b1;
        engine_enable = 1'b0;
        apply_reset();
        @(negedge clk_core);
        row_valid = 1'b1;
        row_window_start = 1'b1;
        row_window_end = 1'b0;
        row_window_tag = 16'hdead;
        row_window_rows = 2;
        row_id = 1;
        @(posedge clk_core);
        @(negedge clk_core);
        row_valid = 1'b0;
        if (!protocol_error || row_accept)
            $fatal(1, "M144 malformed first row not quarantined");
        protocol_attacks++;

        // A completion with the wrong 32-bit identity must not reach M142.
        apply_reset();
        engine_enable = 1'b1;
        drive_one_row_job(16'h1111, 128'h0, 128'h0);
        watchdog = 0;
        while (!pwp_accept && watchdog < 100) begin
            @(posedge clk_core);
            watchdog++;
        end
        if (watchdog == 100)
            $fatal(1, "M144 attack PWP did not launch");
        held_attack_bank = pwp_bank;
        held_attack_tag = pwp_window_tag;
        held_attack_sequence = pwp_sequence;
        @(negedge clk_core);
        pwp_done_valid = 1'b1;
        pwp_done_bank = held_attack_bank;
        pwp_done_window_tag = held_attack_tag;
        pwp_done_sequence = held_attack_sequence + 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        pwp_done_valid = 1'b0;
        if (!protocol_error)
            $fatal(1, "M144 wrong PWP sequence not quarantined");
        protocol_attacks++;

        // An acknowledgement without an offered commit fails closed.
        apply_reset();
        @(negedge clk_core);
        outer_commit_done_valid = 1'b1;
        outer_commit_done_tag = 16'hbadd;
        @(posedge clk_core);
        @(negedge clk_core);
        outer_commit_done_valid = 1'b0;
        if (!protocol_error || outer_commit_done_accept)
            $fatal(1, "M144 unsolicited commit done not quarantined");
        protocol_attacks++;

        $display("PASS M144 sequence-fenced wrapper VCS banks=4 jobs=%0d rows=%0d descriptors=%0d pwp=%0d correction=%0d completed=%0d barriers=%0d commits=%0d protocol_attacks=%0d sequence_bits=32 exact_relative_rows=true post_fence_lookahead=true zero_work_endpoint_floor=true engine_arithmetic=false sram_macro=false physical_speedup=false system_speedup=false headline=false",
                 NORMAL_JOBS, rows_accepted, descriptors_accepted,
                 pwp_accepted, correction_accepted,
                 corrections_completed, barrier_accepts,
                 commit_accepts, protocol_attacks);
        $finish;
    end

    initial begin : watchdog
        #100000;
        $fatal(1, "M144 watchdog timeout");
    end
endmodule

`default_nettype wire
