`timescale 1ns/1ps
`default_nettype none

module tb_m170_fc2_k1_single_bank_accumulator;
    localparam int LANES = 96;
    localparam int MAX_RESULTS = 400;

    logic clk_core;
    logic rst_core;
    logic issue_valid;
    logic issue_ready;
    logic [23:0] issue_tag;
    logic issue_last;
    logic issue_slot_valid;
    logic [2:0] issue_bank_id;
    logic signed [7:0] issue_weight [0:LANES-1];
    logic signed [23:0] issue_accumulator [0:LANES-1];
    logic issue_accept;
    logic result_valid;
    logic result_ready;
    logic [23:0] result_tag;
    logic result_last;
    logic [2:0] result_source_count;
    logic [7:0] result_bank_mask;
    logic signed [23:0] result_accumulator [0:LANES-1];
    logic result_accept;
    logic [LANES-1:0] accepted_weight_active_mask;
    logic protocol_error;
    logic numeric_overflow;
    logic busy;

    logic random_stall_mode;
    logic force_stall_mode;
    logic throughput_mode;
    integer expected_write;
    integer expected_read;
    integer issue_count;
    integer result_count;
    integer accepted_weight_terms;
    integer output_stall_cycles;
    integer same_cycle_replace_count;
    integer consecutive_issue_hits;
    integer previous_issue_cycle;
    integer cycle_count;

    logic [23:0] expected_tag [0:MAX_RESULTS-1];
    logic expected_last [0:MAX_RESULTS-1];
    logic [7:0] expected_bank_mask [0:MAX_RESULTS-1];
    logic signed [23:0] expected_accumulator
        [0:MAX_RESULTS-1][0:LANES-1];

    m170_fc2_k1_single_bank_accumulator dut (.*);

    bind m170_fc2_k1_single_bank_accumulator
        m170_fc2_k1_single_bank_accumulator_assertions sva (.*);

    initial clk_core = 1'b0;
    always #1.5 clk_core = ~clk_core;

    task automatic load_legal_issue(input integer tag_value);
        integer slot_index;
        integer total;
        begin
            slot_index = expected_write;
            if (slot_index >= MAX_RESULTS)
                $fatal(1, "M170 expected scoreboard overflow");
            issue_tag = tag_value;
            issue_last = ((tag_value % 7) == 0);
            issue_slot_valid = 1'b1;
            issue_bank_id = tag_value % 8;
            expected_tag[slot_index] = issue_tag;
            expected_last[slot_index] = issue_last;
            expected_bank_mask[slot_index] = (8'b1 << issue_bank_id);
            for (int lane = 0; lane < LANES; lane++) begin
                issue_weight[lane] = (($urandom % 129) - 64);
                issue_accumulator[lane]
                    = (($urandom % 100001) - 50000);
                total = $signed(issue_accumulator[lane])
                    + $signed(issue_weight[lane]);
                expected_accumulator[slot_index][lane] = total;
            end
            expected_write = expected_write + 1;
        end
    endtask

    task automatic load_overflow_issue;
        integer slot_index;
        integer total;
        begin
            slot_index = expected_write;
            issue_tag = 24'hfff001;
            issue_last = 1'b1;
            issue_slot_valid = 1'b1;
            issue_bank_id = 3'd0;
            expected_tag[slot_index] = issue_tag;
            expected_last[slot_index] = issue_last;
            expected_bank_mask[slot_index] = 8'h01;
            for (int lane = 0; lane < LANES; lane++) begin
                issue_weight[lane] = 8'sd127;
                issue_accumulator[lane] = 24'sh7ffff0;
                total = $signed(issue_accumulator[lane]) + 127;
                expected_accumulator[slot_index][lane] = total;
            end
            expected_write = expected_write + 1;
        end
    endtask

    task automatic drive_issues(input integer count, input integer tag_base);
        integer sent;
        begin
            sent = 0;
            @(negedge clk_core);
            load_legal_issue(tag_base + sent);
            issue_valid = 1'b1;
            while (sent < count) begin
                @(posedge clk_core);
                if (issue_accept) begin
                    sent = sent + 1;
                    if (sent < count) begin
                        @(negedge clk_core);
                        load_legal_issue(tag_base + sent);
                    end else begin
                        @(negedge clk_core);
                        issue_valid = 1'b0;
                    end
                end
            end
        end
    endtask

    always @(negedge clk_core) begin
        if (rst_core || force_stall_mode)
            result_ready <= 1'b0;
        else if (random_stall_mode)
            result_ready <= ($urandom_range(0, 3) != 0);
        else
            result_ready <= 1'b1;
    end

    always @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
            issue_count <= 0;
            result_count <= 0;
            accepted_weight_terms <= 0;
            output_stall_cycles <= 0;
            same_cycle_replace_count <= 0;
            consecutive_issue_hits <= 0;
            previous_issue_cycle <= -1;
        end else begin
            cycle_count <= cycle_count + 1;
            if (issue_accept) begin
                if (accepted_weight_active_mask !== {LANES{1'b1}})
                    $fatal(1, "M170 accepted activity mask mismatch");
                issue_count <= issue_count + 1;
                accepted_weight_terms <= accepted_weight_terms + 1;
                if (throughput_mode) begin
                    if (previous_issue_cycle >= 0) begin
                        if (cycle_count - previous_issue_cycle != 1)
                            $fatal(1, "M170 issue II drift=%0d",
                                cycle_count - previous_issue_cycle);
                        consecutive_issue_hits <= consecutive_issue_hits + 1;
                    end
                    previous_issue_cycle <= cycle_count;
                end
            end
            if (result_valid && !result_ready)
                output_stall_cycles <= output_stall_cycles + 1;
            if (result_accept && issue_accept)
                same_cycle_replace_count <= same_cycle_replace_count + 1;
            if (result_accept) begin
                if (expected_read >= expected_write)
                    $fatal(1, "M170 unexpected result");
                if (result_tag !== expected_tag[expected_read]
                        || result_last !== expected_last[expected_read]
                        || result_source_count !== 3'd1
                        || result_bank_mask
                            !== expected_bank_mask[expected_read])
                    $fatal(1, "M170 header mismatch index=%0d",
                        expected_read);
                for (int lane = 0; lane < LANES; lane++) begin
                    if (result_accumulator[lane]
                            !== expected_accumulator[expected_read][lane])
                        $fatal(1, "M170 accumulator mismatch index=%0d lane=%0d got=%0d expected=%0d",
                            expected_read, lane, result_accumulator[lane],
                            expected_accumulator[expected_read][lane]);
                end
                expected_read <= expected_read + 1;
                result_count <= result_count + 1;
            end
        end
    end

    initial begin
        rst_core = 1'b1;
        issue_valid = 1'b0;
        issue_tag = '0;
        issue_last = 1'b0;
        issue_slot_valid = 1'b0;
        result_ready = 1'b0;
        random_stall_mode = 1'b0;
        force_stall_mode = 1'b0;
        throughput_mode = 1'b0;
        expected_write = 0;
        expected_read = 0;
        issue_count = 0;
        result_count = 0;
        accepted_weight_terms = 0;
        output_stall_cycles = 0;
        same_cycle_replace_count = 0;
        consecutive_issue_hits = 0;
        previous_issue_cycle = -1;
        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        throughput_mode = 1'b1;
        drive_issues(90, 24'h210000);
        wait (expected_read == expected_write);
        @(negedge clk_core);
        throughput_mode = 1'b0;
        previous_issue_cycle = -1;

        random_stall_mode = 1'b1;
        drive_issues(270, 24'h220000);
        wait (expected_read == expected_write);
        wait (!busy);
        @(negedge clk_core);
        random_stall_mode = 1'b0;

        // Accepted overflow remains drainable; an empty younger issue is
        // rejected in the following cycle and makes protocol_error sticky.
        force_stall_mode = 1'b1;
        @(negedge clk_core);
        load_overflow_issue();
        issue_valid = 1'b1;
        do @(posedge clk_core); while (!issue_accept);
        @(negedge clk_core);
        issue_tag = 24'hfff002;
        issue_slot_valid = 1'b0;
        @(posedge clk_core);
        @(negedge clk_core);
        issue_valid = 1'b0;
        force_stall_mode = 1'b0;
        wait (expected_read == expected_write);
        repeat (2) @(posedge clk_core);

        if (!numeric_overflow || !protocol_error || issue_ready)
            $fatal(1, "M170 numeric/protocol fail-close missing");
        if (issue_count != 361 || result_count != 361
                || accepted_weight_terms != 361)
            $fatal(1, "M170 population drift issues=%0d results=%0d terms=%0d",
                issue_count, result_count, accepted_weight_terms);
        if (consecutive_issue_hits < 89 || output_stall_cycles == 0
                || same_cycle_replace_count == 0)
            $fatal(1, "M170 recurrence/cover counters missing");

        $display("PASS M170 FC2 K1 single-bank accumulator VCS issues=361 results=361 accepted_weight_terms=361 output_lanes=96 accumulator_bits=24 weight_bits=8 weight_banks=8 max_sources_per_issue=1 consecutive_issue_ii1_hits=%0d same_cycle_result_replace=%0d output_stall_cycles=%0d overflow_attacks=1 empty_issue_attacks=1 multipliers=0 weight_payload_bits_per_full_issue=768 matched_m169_interface_state=true full_fc2=false bn2=false residual=false physical_speedup=false system_speedup=false headline=false",
            consecutive_issue_hits, same_cycle_replace_count,
            output_stall_cycles);
        $finish;
    end

    initial begin
        #200000;
        $fatal(1, "M170 watchdog timeout");
    end
endmodule

`default_nettype wire
