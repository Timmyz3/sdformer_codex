`timescale 1ns/1ps
`default_nettype none

module tb_m452_independent_m451_exhaustive;
    localparam int TAG_BITS = 24;
    localparam int MAX_EXPECTED = 26000;
    localparam int WIDE_PAIRS = 4096 * 256;
    localparam int NARROW_PAIRS = 256 * 256;

    logic clk_core, reset_n, config_reload;
    logic request_valid, request_ready, request_accept;
    logic [TAG_BITS-1:0] low_tag, high_tag, correction_tag;
    logic low_tile, high_tile, correction_tile;
    logic [4:0] low_center_id, high_center_id;
    logic [2:0] low_output_block, high_output_block;
    logic [2:0] correction_output_block;
    logic request_narrow, request_fuse_correction, correction_subtract;
    logic [767:0] low_data, correction_data;
    logic [511:0] high_data;
    logic contribution_valid, contribution_ready, contribution_accept;
    logic [TAG_BITS-1:0] contribution_tag;
    logic contribution_tile;
    logic [4:0] contribution_center_id;
    logic [2:0] contribution_output_block;
    logic contribution_narrow, contribution_fused;
    logic [1247:0] contribution_data;
    logic protocol_error, busy;
    logic [31:0] debug_request_accepts, debug_plain_accepts;
    logic [31:0] debug_fused_accepts, debug_contributions;
    logic [31:0] debug_protocol_faults;

    logic [TAG_BITS-1:0] expected_tag [0:MAX_EXPECTED-1];
    logic expected_tile [0:MAX_EXPECTED-1];
    logic [4:0] expected_center [0:MAX_EXPECTED-1];
    logic [2:0] expected_block [0:MAX_EXPECTED-1];
    logic expected_narrow [0:MAX_EXPECTED-1];
    logic expected_fused [0:MAX_EXPECTED-1];
    logic [1247:0] expected_data [0:MAX_EXPECTED-1];
    integer expected_head, expected_tail;
    integer legal_accepts, retired, arithmetic_mismatches;
    integer metadata_mismatches, unknown_outputs, failclosed_leaks;
    integer protocol_attacks, legal_reloads, max_stall, stall_run;
    integer pop_push, ii1_pairs;
    integer wide_pairs_checked, narrow_pairs_checked, plain_lanes_checked;
    logic previous_accept;

    m451_exact_k1_fused_pwp_correction_adapter #(
        .TAG_BITS(TAG_BITS)) dut (.*);
    m452_independent_m451_assertions #(
        .TAG_BITS(TAG_BITS)) m452_sva (.*);

    always #1.5 clk_core = ~clk_core;

    function automatic integer signed sx8(input integer raw);
        sx8 = (raw & 8'h80) ? (raw & 8'hff) - 256 : raw & 8'hff;
    endfunction

    function automatic integer signed sx12(input integer raw);
        sx12 = (raw & 12'h800) ? (raw & 12'hfff) - 4096 : raw & 12'hfff;
    endfunction

    task automatic clear_inputs;
        begin
            config_reload = 0;
            request_valid = 0;
            low_tag = 0; low_tile = 0; low_center_id = 0;
            low_output_block = 0; request_narrow = 0; low_data = 0;
            high_tag = 0; high_tile = 0; high_center_id = 0;
            high_output_block = 0; high_data = 0;
            request_fuse_correction = 0; correction_subtract = 0;
            correction_tag = 0; correction_tile = 0;
            correction_output_block = 0; correction_data = 0;
        end
    endtask

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            reset_n = 0;
            clear_inputs();
            contribution_ready = 1;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            reset_n = 1;
            expected_head = expected_tail;
        end
    endtask

    task automatic enqueue(
        input integer txid,
        input logic narrow_value,
        input logic fused_value,
        input logic [1247:0] result_bits
    );
        begin
            if (expected_tail >= MAX_EXPECTED)
                $fatal(1,"M452 scoreboard capacity exceeded");
            expected_tag[expected_tail] = txid[TAG_BITS-1:0];
            expected_tile[expected_tail] = txid[3];
            expected_center[expected_tail] = txid[4:0];
            expected_block[expected_tail] = txid[2:0];
            expected_narrow[expected_tail] = narrow_value;
            expected_fused[expected_tail] = fused_value;
            expected_data[expected_tail] = result_bits;
            expected_tail++;
        end
    endtask

    task automatic drive_request(
        input integer txid,
        input logic narrow_value,
        input logic fused_value,
        input logic subtract_value,
        input integer base_pair,
        input integer pair_extent
    );
        integer pair_index, pwp_raw, weight_raw;
        integer signed pwp, weight, result;
        integer raw12, raw13;
        logic [1247:0] result_bits;
        begin
            low_data = '0;
            high_data = '0;
            correction_data = '0;
            result_bits = '0;
            for (int lane = 0; lane < 96; lane++) begin
                pair_index = base_pair + lane;
                if (pair_index >= pair_extent)
                    pair_index = pair_extent - 1;
                if (narrow_value) begin
                    pwp_raw = (pair_index >> 8) & 8'hff;
                    pwp = sx8(pwp_raw);
                end else begin
                    pwp_raw = (pair_index >> 8) & 12'hfff;
                    pwp = sx12(pwp_raw);
                end
                weight_raw = pair_index & 8'hff;
                weight = sx8(weight_raw);
                result = fused_value ?
                    (subtract_value ? pwp - weight : pwp + weight) : pwp;
                raw12 = pwp & 12'hfff;
                raw13 = result & 13'h1fff;
                low_data[lane*8 +: 8] = raw12[7:0];
                if (!narrow_value)
                    high_data[lane*4 +: 4] = raw12[11:8];
                if (fused_value)
                    correction_data[lane*8 +: 8] = weight_raw[7:0];
                result_bits[lane*13 +: 13] = raw13[12:0];
            end

            low_tag = txid[TAG_BITS-1:0];
            low_tile = txid[3];
            low_center_id = txid[4:0];
            low_output_block = txid[2:0];
            request_narrow = narrow_value;
            if (narrow_value) begin
                high_tag = '0; high_tile = 0; high_center_id = '0;
                high_output_block = '0; high_data = '0;
            end else begin
                high_tag = txid[TAG_BITS-1:0];
                high_tile = txid[3];
                high_center_id = txid[4:0];
                high_output_block = txid[2:0];
            end
            request_fuse_correction = fused_value;
            correction_subtract = fused_value && subtract_value;
            if (fused_value) begin
                correction_tag = txid[TAG_BITS-1:0];
                correction_tile = txid[3];
                correction_output_block = txid[2:0];
            end else begin
                correction_tag = '0; correction_tile = 0;
                correction_output_block = '0; correction_data = '0;
            end
            enqueue(txid,narrow_value,fused_value,result_bits);
            request_valid = 1;
            do @(posedge clk_core); while (!request_accept);
            @(negedge clk_core);
            request_valid = 0;
        end
    endtask

    task automatic expect_fault;
        begin
            @(posedge clk_core); #1;
            if (!protocol_error || request_ready || request_accept ||
                    contribution_valid || contribution_accept)
                $fatal(1,"M452 fail-closed attack leaked");
            repeat (3) begin
                @(posedge clk_core); #1;
                if (!protocol_error || request_ready || contribution_valid)
                    $fatal(1,"M452 sticky fault/quiescence failure");
            end
            protocol_attacks++;
            reset_dut();
        end
    endtask

    always @(posedge clk_core) begin
        if (!reset_n) begin
            previous_accept <= 0;
            stall_run = 0;
        end else begin
            if (request_accept) begin
                legal_accepts++;
                if (previous_accept) ii1_pairs++;
            end
            previous_accept <= request_accept;
            if (request_accept && contribution_accept) pop_push++;
            if (contribution_valid && !contribution_ready) begin
                stall_run++;
                if (stall_run > max_stall) max_stall = stall_run;
            end else stall_run = 0;
        end
        if (protocol_error &&
                (request_ready || request_accept || contribution_valid ||
                 contribution_accept))
            failclosed_leaks++;
        if (contribution_accept) begin
            if (expected_head >= expected_tail)
                $fatal(1,"M452 unexpected contribution");
            if ($isunknown({contribution_tag,contribution_tile,
                            contribution_center_id,
                            contribution_output_block,
                            contribution_narrow,contribution_fused,
                            contribution_data}))
                unknown_outputs++;
            if (contribution_tag !== expected_tag[expected_head] ||
                    contribution_tile !== expected_tile[expected_head] ||
                    contribution_center_id !== expected_center[expected_head] ||
                    contribution_output_block !== expected_block[expected_head] ||
                    contribution_narrow !== expected_narrow[expected_head] ||
                    contribution_fused !== expected_fused[expected_head])
                metadata_mismatches++;
            if (contribution_data !== expected_data[expected_head])
                arithmetic_mismatches++;
            if (metadata_mismatches || arithmetic_mismatches)
                $fatal(1,"M452 scoreboard mismatch index=%0d",expected_head);
            expected_head++;
            retired++;
        end
    end

    initial begin
        integer txid;
        clk_core = 0;
        reset_n = 0;
        contribution_ready = 1;
        clear_inputs();
        expected_head = 0; expected_tail = 0;
        legal_accepts = 0; retired = 0;
        arithmetic_mismatches = 0; metadata_mismatches = 0;
        unknown_outputs = 0; failclosed_leaks = 0;
        protocol_attacks = 0; legal_reloads = 0;
        max_stall = 0; stall_run = 0; pop_push = 0; ii1_pairs = 0;
        wide_pairs_checked = 0; narrow_pairs_checked = 0;
        plain_lanes_checked = 0; previous_accept = 0;
        repeat (4) @(posedge clk_core);
        @(negedge clk_core); reset_n = 1;

        txid = 1;
        for (int subtract = 0; subtract <= 1; subtract++) begin
            for (int base = 0; base < WIDE_PAIRS; base += 96) begin
                drive_request(txid,0,1,subtract,base,WIDE_PAIRS);
                wide_pairs_checked +=
                    ((WIDE_PAIRS-base) >= 96) ? 96 : (WIDE_PAIRS-base);
                txid++;
            end
        end
        for (int subtract = 0; subtract <= 1; subtract++) begin
            for (int base = 0; base < NARROW_PAIRS; base += 96) begin
                drive_request(txid,1,1,subtract,base,NARROW_PAIRS);
                narrow_pairs_checked +=
                    ((NARROW_PAIRS-base) >= 96) ? 96 : (NARROW_PAIRS-base);
                txid++;
            end
        end
        for (int base = 0; base < 4096; base += 96) begin
            drive_request(txid,0,0,0,base << 8,WIDE_PAIRS);
            plain_lanes_checked += 96;
            txid++;
        end
        for (int base = 0; base < 256; base += 96) begin
            drive_request(txid,1,0,0,base << 8,NARROW_PAIRS);
            plain_lanes_checked += 96;
            txid++;
        end
        wait (expected_head == expected_tail && !busy);

        // Twelve-cycle stall and eventual exact retirement.
        contribution_ready = 0;
        drive_request(txid++,0,1,1,0,WIDE_PAIRS);
        repeat (13) @(posedge clk_core);
        @(negedge clk_core); contribution_ready = 1;
        wait (expected_head == expected_tail && !busy);

        // Legal empty reload.
        @(negedge clk_core); config_reload = 1;
        @(posedge clk_core); #1;
        if (protocol_error || request_ready || contribution_valid)
            $fatal(1,"M452 legal empty reload failed");
        legal_reloads++;
        @(negedge clk_core); config_reload = 0;

        // Attack 1: wide tag mismatch.
        @(negedge clk_core); clear_inputs();
        low_tag=1; high_tag=2; request_valid=1;
        expect_fault();
        // Attack 2: wide tile mismatch.
        @(negedge clk_core); clear_inputs();
        low_tag=3; high_tag=3; low_tile=0; high_tile=1; request_valid=1;
        expect_fault();
        // Attack 3: wide center mismatch.
        @(negedge clk_core); clear_inputs();
        low_tag=4; high_tag=4; low_center_id=1; high_center_id=2;
        request_valid=1;
        expect_fault();
        // Attack 4: wide output block mismatch.
        @(negedge clk_core); clear_inputs();
        low_tag=5; high_tag=5; low_output_block=1; high_output_block=2;
        request_valid=1;
        expect_fault();
        // Attack 5: wide physical padding nonzero.
        @(negedge clk_core); clear_inputs();
        low_tag=6; high_tag=6; high_data[511]=1; request_valid=1;
        expect_fault();
        // Attack 6: narrow nonzero high-side metadata/data.
        @(negedge clk_core); clear_inputs();
        request_narrow=1; high_tag=1; high_data=1; request_valid=1;
        expect_fault();
        // Attack 7: fused correction tag mismatch.
        @(negedge clk_core); clear_inputs();
        request_narrow=1; low_tag=7; request_fuse_correction=1;
        correction_tag=8; request_valid=1;
        expect_fault();
        // Attack 8: fused correction tile mismatch.
        @(negedge clk_core); clear_inputs();
        request_narrow=1; low_tag=9; low_tile=1;
        request_fuse_correction=1; correction_tag=9;
        correction_tile=0; request_valid=1;
        expect_fault();
        // Attack 9: fused correction block mismatch.
        @(negedge clk_core); clear_inputs();
        request_narrow=1; low_tag=10; low_output_block=2;
        request_fuse_correction=1; correction_tag=10;
        correction_output_block=3; request_valid=1;
        expect_fault();
        // Attack 10: plain request drives subtract control.
        @(negedge clk_core); clear_inputs();
        request_narrow=1; correction_subtract=1; request_valid=1;
        expect_fault();
        // Attack 11: plain request drives correction data.
        @(negedge clk_core); clear_inputs();
        request_narrow=1; correction_data=1; request_valid=1;
        expect_fault();
        // Attack 12: reload concurrent with a request on an empty adapter.
        @(negedge clk_core); clear_inputs();
        request_narrow=1; request_valid=1; config_reload=1;
        expect_fault();
        // Attack 13: reload while an accepted contribution is buffered.
        contribution_ready=0;
        drive_request(txid++,1,1,0,0,NARROW_PAIRS);
        @(negedge clk_core); config_reload=1;
        expect_fault();

        if (arithmetic_mismatches || metadata_mismatches || unknown_outputs ||
                failclosed_leaks || protocol_attacks != 13 ||
                legal_reloads != 1 || max_stall < 12 || pop_push < 1000 ||
                ii1_pairs < 1000 || wide_pairs_checked != 2*WIDE_PAIRS ||
                narrow_pairs_checked != 2*NARROW_PAIRS)
            $fatal(1,"M452 final independent gate failure");

        $display("PASS M452 independent M451 exhaustive legal_accepts=%0d retired=%0d wide_signed_pairs=%0d narrow_signed_pairs=%0d plain_lanes=%0d arithmetic_mismatches=%0d metadata_mismatches=%0d unknown_outputs=%0d protocol_attacks=%0d failclosed_leaks=%0d max_stall=%0d pop_push=%0d ii1_pairs=%0d legal_reloads=%0d quarantined=1 signed13_min=-2176 signed13_max=2175 old_psum_external=true memories_absent=true cycles=false system=false power=false",
                 legal_accepts,retired,wide_pairs_checked,
                 narrow_pairs_checked,plain_lanes_checked,
                 arithmetic_mismatches,metadata_mismatches,unknown_outputs,
                 protocol_attacks,failclosed_leaks,max_stall,pop_push,
                 ii1_pairs,legal_reloads);
        $finish;
    end
endmodule

`default_nettype wire
