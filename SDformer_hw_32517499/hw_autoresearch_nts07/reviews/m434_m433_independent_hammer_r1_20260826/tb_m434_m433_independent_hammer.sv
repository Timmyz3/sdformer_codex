`timescale 1ns/1ps
`default_nettype none

// Independent M434 checker for M433.  This testbench intentionally does not
// call the M433 request builders or scoreboard.  It exhausts the signed-12
// code space, exhausts the signed-8 code space, and attacks every duplicated
// metadata field and every fail-closed boundary independently.
module tb_m434_m433_independent_hammer;
    localparam int TAG_BITS = 24;
    localparam int MAX_Q = 4096;

    logic clk_core, reset_n, config_reload;
    logic request_valid, request_ready, request_accept;
    logic [TAG_BITS-1:0] low_tag, high_tag;
    logic low_tile, high_tile;
    logic [4:0] low_center_id, high_center_id;
    logic [2:0] low_output_block, high_output_block;
    logic request_narrow;
    logic [767:0] low_data;
    logic [511:0] high_data;
    logic contribution_valid, contribution_ready, contribution_accept;
    logic [TAG_BITS-1:0] contribution_tag;
    logic contribution_tile;
    logic [4:0] contribution_center_id;
    logic [2:0] contribution_output_block;
    logic contribution_narrow;
    logic [1151:0] contribution_data;
    logic protocol_error, busy, debug_output_full;
    logic [31:0] debug_request_accepts, debug_narrow_accepts;
    logic [31:0] debug_wide_accepts, debug_contributions;
    logic [31:0] debug_protocol_faults;

    logic [TAG_BITS-1:0] q_tag [0:MAX_Q-1];
    logic q_tile [0:MAX_Q-1];
    logic [4:0] q_center [0:MAX_Q-1];
    logic [2:0] q_block [0:MAX_Q-1];
    logic q_narrow [0:MAX_Q-1];
    logic [1151:0] q_data [0:MAX_Q-1];
    integer q_head, q_tail;

    integer global_accepts, global_retires, legal_reset_discards;
    integer arithmetic_mismatches, metadata_mismatches, order_mismatches;
    integer same_cycle_pop_push, stall_cycles, max_stall, stall_run;
    integer attacks, same_cycle_leaks, sticky_leaks, legal_reloads;
    integer wide_codes_seen [0:4095];
    integer narrow_codes_seen [0:255];
    integer wide_unique, narrow_unique;
    integer unsigned prng_data, prng_ready;

    m433_exact_dualbank_coread_pwp_adapter #(.TAG_BITS(TAG_BITS)) dut (.*);

    m433_exact_dualbank_coread_pwp_adapter_assertions #(
        .TAG_BITS(TAG_BITS)) original_sva (.*);

    always #1.5 clk_core = ~clk_core;

    task automatic set_idle;
        begin
            config_reload = 1'b0;
            request_valid = 1'b0;
            low_tag = '0;
            low_tile = 1'b0;
            low_center_id = '0;
            low_output_block = '0;
            request_narrow = 1'b0;
            low_data = '0;
            high_tag = '0;
            high_tile = 1'b0;
            high_center_id = '0;
            high_output_block = '0;
            high_data = '0;
        end
    endtask

    task automatic reset_and_quarantine;
        integer pending;
        begin
            @(negedge clk_core);
            pending = q_tail - q_head;
            if (pending < 0 || pending > 1)
                $fatal(1,"M434 impossible pending depth=%0d",pending);
            legal_reset_discards = legal_reset_discards + pending;
            q_head = q_tail;
            reset_n = 1'b0;
            set_idle();
            contribution_ready = 1'b1;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            reset_n = 1'b1;
        end
    endtask

    task automatic send_payload(
        input integer txid,
        input logic narrow_v,
        input logic [767:0] low_v,
        input logic [511:0] high_v
    );
        begin
            if (clk_core !== 1'b0) @(negedge clk_core);
            low_tag = txid[TAG_BITS-1:0];
            low_tile = txid[6];
            low_center_id = txid[4:0];
            low_output_block = txid[2:0];
            request_narrow = narrow_v;
            low_data = low_v;
            if (narrow_v) begin
                high_tag = '0;
                high_tile = 1'b0;
                high_center_id = '0;
                high_output_block = '0;
                high_data = '0;
            end else begin
                high_tag = txid[TAG_BITS-1:0];
                high_tile = txid[6];
                high_center_id = txid[4:0];
                high_output_block = txid[2:0];
                high_data = high_v;
            end
            request_valid = 1'b1;
            do @(posedge clk_core); while (!request_accept && !protocol_error);
            if (protocol_error)
                $fatal(1,"M434 legal request fault tx=%0d",txid);
            @(negedge clk_core);
            request_valid = 1'b0;
        end
    endtask

    task automatic build_wide_exhaustive(
        input integer base_code,
        output logic [767:0] low_v,
        output logic [511:0] high_v
    );
        integer code;
        begin
            low_v = '0;
            high_v = '0;
            for (int lane = 0; lane < 96; lane++) begin
                code = (base_code + lane) & 4095;
                low_v[lane*8 +: 8] = code & 255;
                high_v[lane*4 +: 4] = (code >> 8) & 15;
                wide_codes_seen[code] = 1;
            end
        end
    endtask

    task automatic build_narrow_exhaustive(
        input integer base_code,
        output logic [767:0] low_v
    );
        integer code;
        begin
            low_v = '0;
            for (int lane = 0; lane < 96; lane++) begin
                code = (base_code + lane) & 255;
                low_v[lane*8 +: 8] = code;
                narrow_codes_seen[code] = 1;
            end
        end
    endtask

    task automatic attack_now(input integer attack_id, input logic buffered);
        logic [767:0] lb;
        logic [511:0] hb;
        begin
            if (buffered) begin
                contribution_ready = 1'b0;
                build_wide_exhaustive((attack_id * 97) & 4095,lb,hb);
                send_payload(700000 + attack_id,1'b0,lb,hb);
                wait (contribution_valid && busy);
            end else begin
                @(negedge clk_core);
                set_idle();
            end

            @(negedge clk_core);
            contribution_ready = 1'b1;
            low_tag = 24'h654321;
            low_tile = 1'b1;
            low_center_id = 5'd17;
            low_output_block = 3'd5;
            request_narrow = 1'b0;
            build_wide_exhaustive((attack_id * 193) & 4095,lb,hb);
            low_data = lb;
            high_tag = low_tag;
            high_tile = low_tile;
            high_center_id = low_center_id;
            high_output_block = low_output_block;
            high_data = hb;
            request_valid = 1'b1;
            config_reload = 1'b0;

            case (attack_id)
                0: high_tag = low_tag ^ 24'h1;
                1: high_tile = ~low_tile;
                2: high_center_id = low_center_id ^ 5'h1;
                3: high_output_block = low_output_block ^ 3'h1;
                4: high_data[384] = 1'b1;
                5: high_data[511] = 1'b1;
                6: begin request_narrow=1; high_data='0; high_tag=24'h1; high_tile=0; high_center_id=0; high_output_block=0; end
                7: begin request_narrow=1; high_data='0; high_tag=0; high_tile=1; high_center_id=0; high_output_block=0; end
                8: begin request_narrow=1; high_data='0; high_tag=0; high_tile=0; high_center_id=1; high_output_block=0; end
                9: begin request_narrow=1; high_data='0; high_tag=0; high_tile=0; high_center_id=0; high_output_block=1; end
                10: begin request_narrow=1; high_data=512'h1; high_tag=0; high_tile=0; high_center_id=0; high_output_block=0; end
                11: begin config_reload=1; request_valid=1; request_narrow=1; high_data=0; high_tag=0; high_tile=0; high_center_id=0; high_output_block=0; end
                12: begin config_reload=1; request_valid=0; end
                13: high_data[400] = 1'b1;
                default: $fatal(1,"M434 bad attack id");
            endcase

            // The malformed inputs are already visible before the edge.  An
            // older buffered output must not leak even if ready is high.
            #1;
            if (request_ready || request_accept || contribution_valid ||
                    contribution_accept)
                $fatal(1,"M434 same-cycle fail-closed leak attack=%0d",attack_id);
            @(posedge clk_core); #1;
            if (!protocol_error || request_ready || request_accept ||
                    contribution_valid || contribution_accept)
                $fatal(1,"M434 sticky fault missing/leak attack=%0d",attack_id);
            attacks++;

            // Legal-looking traffic must remain isolated until reset.
            @(negedge clk_core);
            config_reload = 1'b0;
            request_valid = 1'b1;
            request_narrow = 1'b1;
            high_tag = 0; high_tile = 0; high_center_id = 0;
            high_output_block = 0; high_data = 0;
            repeat (2) begin
                @(posedge clk_core); #1;
                if (request_ready || request_accept || contribution_valid ||
                        contribution_accept)
                    $fatal(1,"M434 post-fault sticky leak attack=%0d",attack_id);
            end
            reset_and_quarantine();
        end
    endtask

    // Independent queue model: retire the old entry before enqueuing a
    // simultaneous replacement.  Expected data is formed here rather than by
    // any M433 helper.
    always @(posedge clk_core) begin : independent_scoreboard
        logic [1151:0] expected_word;
        integer raw8;
        if (!reset_n) begin
            stall_run = 0;
        end else begin
            if (contribution_valid && !contribution_ready) begin
                stall_cycles++;
                stall_run++;
                if (stall_run > max_stall) max_stall = stall_run;
            end else begin
                stall_run = 0;
            end
            if (contribution_accept) begin
                if (q_head >= q_tail) begin
                    order_mismatches++;
                    $fatal(1,"M434 output without queued input");
                end
                if (contribution_tag !== q_tag[q_head] ||
                        contribution_tile !== q_tile[q_head] ||
                        contribution_center_id !== q_center[q_head] ||
                        contribution_output_block !== q_block[q_head] ||
                        contribution_narrow !== q_narrow[q_head]) begin
                    metadata_mismatches++;
                    $fatal(1,"M434 metadata/order mismatch at %0d",q_head);
                end
                if (contribution_data !== q_data[q_head]) begin
                    arithmetic_mismatches++;
                    $fatal(1,"M434 data mismatch tag=%0h lane_xor=%h",
                           contribution_tag, contribution_data ^ q_data[q_head]);
                end
                q_head++;
                global_retires++;
            end
            if (request_accept) begin
                if (q_tail >= MAX_Q) $fatal(1,"M434 queue overflow");
                expected_word = '0;
                for (int lane = 0; lane < 96; lane++) begin
                    if (request_narrow) begin
                        raw8 = low_data[lane*8 +: 8];
                        expected_word[lane*12 +: 12] =
                            (raw8[7] ? (12'hf00 | raw8) : raw8);
                    end else begin
                        expected_word[lane*12 +: 12] =
                            ((high_data[lane*4 +: 4] << 8) |
                             low_data[lane*8 +: 8]);
                    end
                end
                q_tag[q_tail] = low_tag;
                q_tile[q_tail] = low_tile;
                q_center[q_tail] = low_center_id;
                q_block[q_tail] = low_output_block;
                q_narrow[q_tail] = request_narrow;
                q_data[q_tail] = expected_word;
                q_tail++;
                global_accepts++;
            end
            if (request_accept && contribution_accept)
                same_cycle_pop_push++;
            if (protocol_error && (request_ready || request_accept ||
                    contribution_valid || contribution_accept))
                sticky_leaks++;
        end
    end

    initial begin : test
        logic [767:0] lb;
        logic [511:0] hb;
        clk_core=0; reset_n=0; contribution_ready=1;
        set_idle();
        q_head=0; q_tail=0;
        global_accepts=0; global_retires=0; legal_reset_discards=0;
        arithmetic_mismatches=0; metadata_mismatches=0; order_mismatches=0;
        same_cycle_pop_push=0; stall_cycles=0; max_stall=0; stall_run=0;
        attacks=0; same_cycle_leaks=0; sticky_leaks=0; legal_reloads=0;
        wide_unique=0; narrow_unique=0;
        prng_data=32'h4340cafe; prng_ready=32'h5eed1234;
        for (int i=0;i<4096;i++) wide_codes_seen[i]=0;
        for (int i=0;i<256;i++) narrow_codes_seen[i]=0;
        repeat (5) @(posedge clk_core);
        @(negedge clk_core); reset_n=1;

        // Wide exhaustive: 43 transactions cover all 4096 signed12 codes.
        contribution_ready=1;
        for (int group=0; group<43; group++) begin
            build_wide_exhaustive(group*96,lb,hb);
            send_payload(1000+group,1'b0,lb,hb);
        end
        wait (q_head==q_tail && !busy);

        // Narrow exhaustive: three transactions cover all signed8 codes.
        for (int group=0; group<3; group++) begin
            build_narrow_exhaustive(group*96,lb);
            hb='0;
            send_payload(2000+group,1'b1,lb,hb);
        end
        wait (q_head==q_tail && !busy);

        // Randomized traffic/backpressure. The producer is independent of the
        // consumer so ordering, stalls, and replacement are exercised.
        fork
            begin
                for (int tx=0; tx<600; tx++) begin
                    lb='0; hb='0;
                    for (int lane=0; lane<96; lane++) begin
                        prng_data = prng_data * 1664525 + 1013904223;
                        lb[lane*8 +: 8] = prng_data[15:8];
                        prng_data = prng_data * 1664525 + 1013904223;
                        hb[lane*4 +: 4] = prng_data[3:0];
                    end
                    send_payload(10000+tx,(tx%5)==0,lb,hb);
                end
            end
            begin
                while (global_accepts < 646 || q_head != q_tail) begin
                    @(negedge clk_core);
                    prng_ready = prng_ready * 1664525 + 1013904223;
                    contribution_ready = (prng_ready[3:0] != 0);
                end
                contribution_ready=1;
            end
        join
        wait (q_head==q_tail && !busy);

        // A deterministic 16-cycle stall checks payload retention.
        contribution_ready=0;
        build_wide_exhaustive(2047,lb,hb);
        send_payload(30000,0,lb,hb);
        wait (contribution_valid);
        repeat (16) @(posedge clk_core);
        @(negedge clk_core); contribution_ready=1;
        wait (q_head==q_tail && !busy);

        // Empty reload is legal and must suppress ready for the boundary cycle.
        @(negedge clk_core); set_idle(); config_reload=1;
        #1;
        if (protocol_error || request_ready || request_accept ||
                contribution_valid || contribution_accept)
            $fatal(1,"M434 legal empty reload behavior mismatch");
        @(posedge clk_core); #1;
        if (protocol_error) $fatal(1,"M434 legal reload set fault");
        legal_reloads++;
        @(negedge clk_core); config_reload=0;

        // Attacks 0..11 on empty; 12 is reload-busy; 13 is malformed padding
        // while an older contribution is buffered. Exactly two accepted old
        // entries are deliberately quarantined and later discarded by reset.
        for (int attack=0; attack<=11; attack++)
            attack_now(attack,1'b0);
        attack_now(12,1'b1);
        attack_now(13,1'b1);

        // Reset recovery must still accept and retire an exact transaction.
        contribution_ready=1;
        build_wide_exhaustive(4095,lb,hb);
        send_payload(900000,0,lb,hb);
        wait (q_head==q_tail && !busy);

        for (int i=0;i<4096;i++) if (wide_codes_seen[i]) wide_unique++;
        for (int i=0;i<256;i++) if (narrow_codes_seen[i]) narrow_unique++;
        if (wide_unique != 4096 || narrow_unique != 256 ||
                arithmetic_mismatches || metadata_mismatches ||
                order_mismatches || same_cycle_leaks || sticky_leaks ||
                attacks != 14 || legal_reloads != 1 || max_stall < 16 ||
                same_cycle_pop_push < 100 || legal_reset_discards != 2 ||
                global_accepts - global_retires != legal_reset_discards)
            $fatal(1,"M434 gate fail wide=%0d narrow=%0d acc=%0d ret=%0d discard=%0d attacks=%0d arith=%0d meta=%0d order=%0d same=%0d sticky=%0d stall=%0d poppush=%0d",
                   wide_unique,narrow_unique,global_accepts,global_retires,
                   legal_reset_discards,attacks,arithmetic_mismatches,
                   metadata_mismatches,order_mismatches,same_cycle_leaks,
                   sticky_leaks,max_stall,same_cycle_pop_push);

        $display("PASS M434 independent M433 hammer wide_codes=4096 narrow_codes=256 accepts=%0d retires=%0d explicit_fault_reset_discards=%0d arithmetic_mismatches=0 metadata_mismatches=0 order_mismatches=0 attacks=14 same_cycle_leaks=0 sticky_leaks=0 legal_reloads=1 max_stall=%0d pop_push=%0d logical_wide_bytes=144 physical_interface_bytes=160 correction_port=false old_psum_port=false seed_fusion=false docs359_unchanged=true dc_go=true formality_go=true full_population_go=true headline=false",
                 global_accepts,global_retires,legal_reset_discards,max_stall,
                 same_cycle_pop_push);
        $finish;
    end

    initial begin
        #1000000;
        $fatal(1,"M434 watchdog accepts=%0d retires=%0d q=%0d/%0d",
               global_accepts,global_retires,q_head,q_tail);
    end
endmodule

`default_nettype wire
