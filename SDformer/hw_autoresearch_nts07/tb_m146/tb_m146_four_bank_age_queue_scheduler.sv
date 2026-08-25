`timescale 1ns/1ps
`default_nettype none

module tb_m146_four_bank_age_queue_scheduler;
    localparam int TAG_BITS = 16;
    localparam int SEQUENCE_BITS = 32;
    localparam int BANKS = 4;
    localparam int JOBS = 40;

    logic clk_core = 0;
    logic rst_core;
    always #1.5 clk_core = ~clk_core;

    logic fill_valid, fill_ready, fill_accept;
    logic [1:0] fill_bank;
    logic [TAG_BITS-1:0] fill_window_tag;
    logic [SEQUENCE_BITS-1:0] fill_sequence;
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
    logic release_valid;
    logic [1:0] release_bank;
    logic [TAG_BITS-1:0] release_window_tag;
    logic [SEQUENCE_BITS-1:0] release_sequence;
    logic [BANKS-1:0] observed_bank_free;
    logic [2:0] observed_pwp_queue_count;
    logic [2:0] observed_correction_queue_count;
    logic observed_pwp_busy, observed_correction_busy;
    logic [SEQUENCE_BITS-1:0] observed_next_fill_sequence;
    logic protocol_error, busy;

    logic engine_enable, attack_mode;
    logic model_pwp_done_valid, model_correction_done_valid;
    logic [1:0] model_pwp_done_bank, model_correction_done_bank;
    logic [TAG_BITS-1:0] model_pwp_done_tag;
    logic [TAG_BITS-1:0] model_correction_done_tag;
    logic [SEQUENCE_BITS-1:0] model_pwp_done_sequence;
    logic [SEQUENCE_BITS-1:0] model_correction_done_sequence;
    logic attack_pwp_done_valid, attack_correction_done_valid;
    logic [1:0] attack_done_bank;
    logic [TAG_BITS-1:0] attack_done_tag;
    logic [SEQUENCE_BITS-1:0] attack_done_sequence;

    int unsigned cycle_count, fills, pwp_launches, correction_launches;
    int unsigned releases, pwp_stalls, correction_stalls;
    int unsigned bank_reuses, protocol_attacks;
    int unsigned expected_pwp, expected_correction, expected_release;
    bit bank_seen [0:BANKS-1];
    bit saw_all_live, saw_overlap, saw_pwp_full;

    assign pwp_done_valid = attack_mode
        ? attack_pwp_done_valid : model_pwp_done_valid;
    assign pwp_done_bank = attack_mode
        ? attack_done_bank : model_pwp_done_bank;
    assign pwp_done_window_tag = attack_mode
        ? attack_done_tag : model_pwp_done_tag;
    assign pwp_done_sequence = attack_mode
        ? attack_done_sequence : model_pwp_done_sequence;
    assign correction_done_valid = attack_mode
        ? attack_correction_done_valid : model_correction_done_valid;
    assign correction_done_bank = attack_mode
        ? attack_done_bank : model_correction_done_bank;
    assign correction_done_window_tag = attack_mode
        ? attack_done_tag : model_correction_done_tag;
    assign correction_done_sequence = attack_mode
        ? attack_done_sequence : model_correction_done_sequence;

    m146_four_bank_age_queue_scheduler dut (.*);
    m146_four_bank_age_queue_scheduler_assertions sva (.*);

    task automatic apply_reset;
        @(negedge clk_core);
        rst_core = 1'b1;
        fill_valid = 1'b0;
        attack_pwp_done_valid = 1'b0;
        attack_correction_done_valid = 1'b0;
        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        repeat (2) @(posedge clk_core);
    endtask

    task automatic drive_fill(input int unsigned sequence_id);
        logic [1:0] selected;
        bit found;
        @(negedge clk_core);
        found = 1'b0;
        selected = '0;
        while (!found) begin
            for (int bank = 0; bank < BANKS; bank++) begin
                if (!found && observed_bank_free[bank]) begin
                    found = 1'b1;
                    selected = bank[1:0];
                end
            end
            if (!found)
                @(negedge clk_core);
        end
        fill_valid = 1'b1;
        fill_bank = selected;
        fill_window_tag = 16'd1000 + sequence_id;
        fill_sequence = sequence_id;
        #0.1;
        while (!fill_ready)
            @(negedge clk_core);
        @(posedge clk_core);
        if (!fill_accept)
            $fatal(1, "M146 expected fill acceptance sequence=%0d",
                   sequence_id);
        #0.1;
        fill_valid = 1'b0;
    endtask

    always_comb begin
        pwp_ready = engine_enable && (cycle_count % 5) != 1;
        correction_ready = engine_enable && (cycle_count % 7) != 2;
    end

    always @(posedge clk_core) begin
        if (rst_core)
            cycle_count <= 0;
        else
            cycle_count <= cycle_count + 1;
    end

    always @(posedge clk_core) begin : pwp_engine
        static logic active;
        static int unsigned countdown;
        static logic [1:0] held_bank;
        static logic [TAG_BITS-1:0] held_tag;
        static logic [SEQUENCE_BITS-1:0] held_sequence;
        if (rst_core || attack_mode) begin
            active <= 1'b0;
            countdown <= 0;
            model_pwp_done_valid <= 1'b0;
            model_pwp_done_bank <= '0;
            model_pwp_done_tag <= '0;
            model_pwp_done_sequence <= '0;
        end else begin
            model_pwp_done_valid <= 1'b0;
            if (pwp_accept) begin
                if (active)
                    $fatal(1, "M146 PWP double launch");
                active <= 1'b1;
                countdown <= 2 + pwp_sequence[1:0];
                held_bank <= pwp_bank;
                held_tag <= pwp_window_tag;
                held_sequence <= pwp_sequence;
            end else if (active && countdown != 0) begin
                countdown <= countdown - 1;
            end else if (active) begin
                model_pwp_done_valid <= 1'b1;
                model_pwp_done_bank <= held_bank;
                model_pwp_done_tag <= held_tag;
                model_pwp_done_sequence <= held_sequence;
                active <= 1'b0;
            end
        end
    end

    always @(posedge clk_core) begin : correction_engine
        static logic active;
        static int unsigned countdown;
        static logic [1:0] held_bank;
        static logic [TAG_BITS-1:0] held_tag;
        static logic [SEQUENCE_BITS-1:0] held_sequence;
        if (rst_core || attack_mode) begin
            active <= 1'b0;
            countdown <= 0;
            model_correction_done_valid <= 1'b0;
            model_correction_done_bank <= '0;
            model_correction_done_tag <= '0;
            model_correction_done_sequence <= '0;
        end else begin
            model_correction_done_valid <= 1'b0;
            if (correction_accept) begin
                if (active)
                    $fatal(1, "M146 correction double launch");
                active <= 1'b1;
                countdown <= 5 + correction_sequence[1:0];
                held_bank <= correction_bank;
                held_tag <= correction_window_tag;
                held_sequence <= correction_sequence;
            end else if (active && countdown != 0) begin
                countdown <= countdown - 1;
            end else if (active) begin
                model_correction_done_valid <= 1'b1;
                model_correction_done_bank <= held_bank;
                model_correction_done_tag <= held_tag;
                model_correction_done_sequence <= held_sequence;
                active <= 1'b0;
            end
        end
    end

    always @(posedge clk_core) begin : scoreboard
        if (!rst_core && !attack_mode) begin
            if (fill_accept) begin
                if (fill_sequence !== fills)
                    $fatal(1, "M146 fill sequence mismatch");
                if (bank_seen[fill_bank])
                    bank_reuses <= bank_reuses + 1;
                bank_seen[fill_bank] <= 1'b1;
                fills <= fills + 1;
            end
            if (pwp_valid && !pwp_ready)
                pwp_stalls <= pwp_stalls + 1;
            if (correction_valid && !correction_ready)
                correction_stalls <= correction_stalls + 1;
            if (pwp_accept) begin
                if (pwp_sequence !== expected_pwp
                        || pwp_window_tag !== 16'd1000 + expected_pwp)
                    $fatal(1, "M146 PWP FIFO order mismatch");
                expected_pwp <= expected_pwp + 1;
                pwp_launches <= pwp_launches + 1;
            end
            if (correction_accept) begin
                if (correction_sequence !== expected_correction
                        || correction_window_tag
                           !== 16'd1000 + expected_correction)
                    $fatal(1, "M146 correction FIFO order mismatch");
                expected_correction <= expected_correction + 1;
                correction_launches <= correction_launches + 1;
            end
            if (release_valid) begin
                if (release_sequence !== expected_release
                        || release_window_tag
                           !== 16'd1000 + expected_release)
                    $fatal(1, "M146 release order mismatch");
                expected_release <= expected_release + 1;
                releases <= releases + 1;
            end
            if (observed_bank_free == 0)
                saw_all_live <= 1'b1;
            if (observed_pwp_busy && observed_correction_busy)
                saw_overlap <= 1'b1;
            if (observed_pwp_queue_count == BANKS)
                saw_pwp_full <= 1'b1;
        end
    end

    initial begin : stimulus
        int watchdog;
        logic [1:0] held_bank;
        logic [TAG_BITS-1:0] held_tag;
        logic [SEQUENCE_BITS-1:0] held_sequence;

        rst_core = 1'b1;
        fill_valid = 1'b0;
        fill_bank = '0;
        fill_window_tag = '0;
        fill_sequence = '0;
        engine_enable = 1'b0;
        attack_mode = 1'b0;
        attack_pwp_done_valid = 1'b0;
        attack_correction_done_valid = 1'b0;
        attack_done_bank = '0;
        attack_done_tag = '0;
        attack_done_sequence = '0;
        cycle_count = 0;
        fills = 0;
        pwp_launches = 0;
        correction_launches = 0;
        releases = 0;
        pwp_stalls = 0;
        correction_stalls = 0;
        bank_reuses = 0;
        protocol_attacks = 0;
        expected_pwp = 0;
        expected_correction = 0;
        expected_release = 0;
        saw_all_live = 1'b0;
        saw_overlap = 1'b0;
        saw_pwp_full = 1'b0;
        for (int bank = 0; bank < BANKS; bank++)
            bank_seen[bank] = 1'b0;

        apply_reset();
        for (int sequence_id = 0; sequence_id < 4; sequence_id++)
            drive_fill(sequence_id);
        engine_enable = 1'b1;
        for (int sequence_id = 4; sequence_id < JOBS; sequence_id++)
            drive_fill(sequence_id);
        watchdog = 0;
        while (releases != JOBS && watchdog < 3000) begin
            @(posedge clk_core);
            watchdog++;
        end
        if (watchdog == 3000)
            $fatal(1, "M146 normal traffic failed to drain");
        repeat (5) @(posedge clk_core);
        if (busy || protocol_error || fills != JOBS
                || pwp_launches != JOBS
                || correction_launches != JOBS || releases != JOBS
                || bank_reuses < JOBS - BANKS || !saw_all_live
                || !saw_overlap || !saw_pwp_full)
            $fatal(1, "M146 normal contract incomplete");

        attack_mode = 1'b1;
        engine_enable = 1'b1;
        apply_reset();
        drive_fill(0);
        watchdog = 0;
        while (!pwp_accept && watchdog < 100) begin
            @(posedge clk_core);
            watchdog++;
        end
        if (watchdog == 100)
            $fatal(1, "M146 reset-release setup PWP did not launch");
        held_bank = pwp_bank;
        held_tag = pwp_window_tag;
        held_sequence = pwp_sequence;
        @(negedge clk_core);
        attack_done_bank = held_bank;
        attack_done_tag = held_tag;
        attack_done_sequence = held_sequence;
        attack_pwp_done_valid = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        attack_pwp_done_valid = 1'b0;
        watchdog = 0;
        while (!correction_accept && watchdog < 100) begin
            @(posedge clk_core);
            watchdog++;
        end
        if (watchdog == 100)
            $fatal(1, "M146 reset-release setup correction did not launch");
        held_bank = correction_bank;
        held_tag = correction_window_tag;
        held_sequence = correction_sequence;
        @(negedge clk_core);
        rst_core = 1'b1;
        attack_done_bank = held_bank;
        attack_done_tag = held_tag;
        attack_done_sequence = held_sequence;
        attack_correction_done_valid = 1'b1;
        #0.1;
        if (release_valid)
            $fatal(1, "M146 release leaked during reset assertion");
        @(posedge clk_core);
        #0.1;
        if (release_valid)
            $fatal(1, "M146 release leaked on reset edge");
        @(negedge clk_core);
        attack_correction_done_valid = 1'b0;
        rst_core = 1'b0;
        repeat (2) @(posedge clk_core);
        protocol_attacks++;

        engine_enable = 1'b0;
        apply_reset();
        @(negedge clk_core);
        fill_valid = 1'b1;
        fill_bank = 0;
        fill_window_tag = 16'h1111;
        fill_sequence = 7;
        @(posedge clk_core);
        @(negedge clk_core);
        fill_valid = 1'b0;
        if (!protocol_error || fill_accept)
            $fatal(1, "M146 wrong fill sequence not quarantined");
        protocol_attacks++;

        apply_reset();
        drive_fill(0);
        @(negedge clk_core);
        fill_valid = 1'b1;
        fill_bank = 0;
        fill_window_tag = 16'h2222;
        fill_sequence = 1;
        @(posedge clk_core);
        @(negedge clk_core);
        fill_valid = 1'b0;
        if (!protocol_error || fill_accept)
            $fatal(1, "M146 live-bank refill not quarantined");
        protocol_attacks++;

        apply_reset();
        engine_enable = 1'b1;
        drive_fill(0);
        watchdog = 0;
        while (!pwp_accept && watchdog < 100) begin
            @(posedge clk_core);
            watchdog++;
        end
        if (watchdog == 100)
            $fatal(1, "M146 attack PWP did not launch");
        held_bank = pwp_bank;
        held_tag = pwp_window_tag;
        held_sequence = pwp_sequence;
        @(negedge clk_core);
        attack_done_bank = held_bank;
        attack_done_tag = held_tag;
        attack_done_sequence = held_sequence + 1'b1;
        attack_pwp_done_valid = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        attack_pwp_done_valid = 1'b0;
        if (!protocol_error)
            $fatal(1, "M146 wrong completion sequence not quarantined");
        protocol_attacks++;

        $display("PASS M146r2 age-queue scheduler VCS banks=4 jobs=%0d fills=%0d pwp=%0d correction=%0d releases=%0d pwp_stalls=%0d correction_stalls=%0d bank_reuses=%0d protocol_attacks=%0d reset_release_guard=true sequence_age_comparators=0 completion_identity_equality=true sequence_bits=32 pwp_correction_overlap=1 engine_arithmetic=false sram_macro=false physical_speedup=false system_speedup=false headline=false",
                 JOBS, fills, pwp_launches, correction_launches,
                 releases, pwp_stalls, correction_stalls,
                 bank_reuses, protocol_attacks);
        $finish;
    end

    initial begin : watchdog
        #100000;
        $fatal(1, "M146 watchdog timeout");
    end
endmodule

`default_nettype wire
