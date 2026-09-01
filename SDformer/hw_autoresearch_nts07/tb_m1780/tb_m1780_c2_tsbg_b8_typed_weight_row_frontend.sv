`timescale 1ns/1ps
`default_nettype none

module m1780_directed_8bank_row_memory #(
    parameter int GROUP_BITS = 6,
    parameter int BANKS = 8,
    parameter int LANES = 16
) (
    input  logic clk_core,
    input  logic rst_core,
    input  logic req_valid,
    output logic req_ready,
    input  logic [GROUP_BITS-1:0] req_group,
    input  logic req_half,
    input  logic [2:0] req_slice,
    input  logic req_accept,
    output logic rsp_valid,
    input  logic rsp_ready,
    output logic [GROUP_BITS-1:0] rsp_group,
    output logic rsp_half,
    output logic [2:0] rsp_slice,
    output logic signed [7:0] rsp_weight [0:BANKS-1][0:LANES-1],
    input  logic rsp_accept,
    output logic [31:0] request_count,
    output logic [31:0] response_count,
    output logic [31:0] request_stall_count
);
    logic pending_q;
    logic [GROUP_BITS-1:0] group_q;
    logic half_q;
    logic [2:0] slice_q;
    logic [31:0] cycle_q;
    integer raw_weight;

    assign req_ready = !pending_q && (cycle_q % 5 != 1);
    assign rsp_valid = pending_q;
    assign rsp_group = group_q;
    assign rsp_half = half_q;
    assign rsp_slice = slice_q;
    always_comb begin
        for (int bank = 0; bank < BANKS; bank++) begin
            for (int lane = 0; lane < LANES; lane++) begin
                raw_weight = (int'(group_q) * 17 + int'(half_q) * 11
                    + int'(slice_q) * 7 + bank * 5 + lane * 3) % 63 - 31;
                rsp_weight[bank][lane] = raw_weight;
            end
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            pending_q <= 0;
            group_q <= 0;
            half_q <= 0;
            slice_q <= 0;
            cycle_q <= 0;
            request_count <= 0;
            response_count <= 0;
            request_stall_count <= 0;
        end else begin
            cycle_q <= cycle_q + 1'b1;
            if (req_valid && !req_ready)
                request_stall_count <= request_stall_count + 1'b1;
            if (req_accept) begin
                if (pending_q) $fatal(1, "M1780 memory overwrote pending response");
                pending_q <= 1;
                group_q <= req_group;
                half_q <= req_half;
                slice_q <= req_slice;
                request_count <= request_count + 1'b1;
            end
            if (rsp_accept) begin
                pending_q <= 0;
                response_count <= response_count + 1'b1;
            end
        end
    end
endmodule

module tb_m1780_c2_tsbg_b8_typed_weight_row_frontend;
    localparam int BUNDLE = 8;
    localparam int GROUPS = 12;
    localparam int SLICES = 6;
    localparam int BANKS = 8;
    localparam int LANES = 16;
    localparam int GROUP_BITS = 6;
    localparam int EXPECTED_ROW_ACCESSES = BUNDLE * GROUPS;
    localparam int EXPECTED_ISSUES = BUNDLE * GROUPS * 2 * SLICES;
    localparam int EXPECTED_PRODUCTS = BUNDLE * GROUPS * 2 * SLICES * LANES;
    localparam int EXPECTED_COMMITS = BUNDLE * SLICES;

    logic clk_core, rst_core;
    logic load_valid;
    logic [2:0] load_context;
    logic [23:0] load_tag;
    logic [GROUP_BITS-1:0] load_group;
    logic signed [7:0] load_source_value [0:15];
    logic load_last;
    logic load_ready_base, load_ready_tsbg;
    logic load_accept_base, load_accept_tsbg;

`define DECLARE_SIDE(name) \
    logic mem_req_valid_``name, mem_req_ready_``name; \
    logic [GROUP_BITS-1:0] mem_req_group_``name; \
    logic mem_req_half_``name; \
    logic [2:0] mem_req_slice_``name; \
    logic mem_req_accept_``name; \
    logic mem_rsp_valid_``name, mem_rsp_ready_``name; \
    logic [GROUP_BITS-1:0] mem_rsp_group_``name; \
    logic mem_rsp_half_``name; \
    logic [2:0] mem_rsp_slice_``name; \
    logic signed [7:0] mem_rsp_weight_``name [0:BANKS-1][0:LANES-1]; \
    logic mem_rsp_accept_``name; \
    logic issue_valid_``name, issue_ready_``name; \
    logic [2:0] issue_context_``name; \
    logic [GROUP_BITS-1:0] issue_group_``name; \
    logic issue_half_``name; \
    logic [2:0] issue_slice_``name; \
    logic [BANKS-1:0] issue_bank_valid_``name; \
    logic [GROUP_BITS+3:0] issue_source_channel_``name [0:BANKS-1]; \
    logic signed [7:0] issue_source_value_``name [0:BANKS-1]; \
    logic signed [7:0] issue_weight_``name [0:BANKS-1][0:LANES-1]; \
    logic issue_accept_``name; \
    logic commit_valid_``name, commit_ready_``name; \
    logic [2:0] commit_context_``name; \
    logic [23:0] commit_tag_``name; \
    logic [2:0] commit_slice_``name; \
    logic signed [23:0] commit_accumulator_``name [0:LANES-1]; \
    logic commit_terminal_``name, commit_accept_``name; \
    logic bundle_done_valid_``name, bundle_done_ready_``name; \
    logic protocol_error_``name, numeric_overflow_``name, busy_``name; \
    logic [31:0] cycle_count_``name, row_access_count_``name; \
    logic [31:0] cache_hit_count_``name, cache_miss_count_``name; \
    logic [31:0] weight_beat_count_``name, issue_count_``name; \
    logic [31:0] product_count_``name, commit_count_``name; \
    logic [31:0] memory_request_count_``name, memory_response_count_``name; \
    logic [31:0] memory_stall_count_``name

    `DECLARE_SIDE(base);
    `DECLARE_SIDE(tsbg);
`undef DECLARE_SIDE

    logic signed [23:0] expected [0:BUNDLE-1][0:SLICES-1][0:LANES-1];
    logic signed [23:0] observed_base [0:BUNDLE-1][0:SLICES-1][0:LANES-1];
    logic signed [23:0] observed_tsbg [0:BUNDLE-1][0:SLICES-1][0:LANES-1];
    logic observed_base_valid [0:BUNDLE-1][0:SLICES-1];
    logic observed_tsbg_valid [0:BUNDLE-1][0:SLICES-1];
    integer tb_cycle;
    integer start_cycle, base_done_cycle, tsbg_done_cycle;
    integer terminal_base, terminal_tsbg;
    integer issue_stall_base, issue_stall_tsbg;
    integer commit_stall_base, commit_stall_tsbg;
    integer raw_weight;

    always #1.5 clk_core = ~clk_core;
    always_ff @(posedge clk_core) begin
        if (rst_core) tb_cycle <= 0;
        else tb_cycle <= tb_cycle + 1;
    end

`define CONNECT_DUT(name, mode) \
    m1780_c2_tsbg_b8_typed_weight_row_frontend #( \
        .SCHEDULE_MODE(mode), .SOURCE_GROUPS(GROUPS), \
        .OUTPUT_SLICES(SLICES)) dut_``name ( \
        .clk_core(clk_core), .rst_core(rst_core), \
        .load_valid(load_valid), .load_ready(load_ready_``name), \
        .load_context(load_context), .load_tag(load_tag), \
        .load_group(load_group), .load_source_value(load_source_value), \
        .load_last(load_last), .load_accept(load_accept_``name), \
        .mem_req_valid(mem_req_valid_``name), .mem_req_ready(mem_req_ready_``name), \
        .mem_req_group(mem_req_group_``name), .mem_req_half(mem_req_half_``name), \
        .mem_req_slice(mem_req_slice_``name), .mem_req_accept(mem_req_accept_``name), \
        .mem_rsp_valid(mem_rsp_valid_``name), .mem_rsp_ready(mem_rsp_ready_``name), \
        .mem_rsp_group(mem_rsp_group_``name), .mem_rsp_half(mem_rsp_half_``name), \
        .mem_rsp_slice(mem_rsp_slice_``name), .mem_rsp_weight(mem_rsp_weight_``name), \
        .mem_rsp_accept(mem_rsp_accept_``name), \
        .issue_valid(issue_valid_``name), .issue_ready(issue_ready_``name), \
        .issue_context(issue_context_``name), .issue_group(issue_group_``name), \
        .issue_half(issue_half_``name), .issue_slice(issue_slice_``name), \
        .issue_bank_valid(issue_bank_valid_``name), \
        .issue_source_channel(issue_source_channel_``name), \
        .issue_source_value(issue_source_value_``name), \
        .issue_weight(issue_weight_``name), .issue_accept(issue_accept_``name), \
        .commit_valid(commit_valid_``name), .commit_ready(commit_ready_``name), \
        .commit_context(commit_context_``name), .commit_tag(commit_tag_``name), \
        .commit_slice(commit_slice_``name), \
        .commit_accumulator(commit_accumulator_``name), \
        .commit_terminal(commit_terminal_``name), \
        .commit_accept(commit_accept_``name), \
        .bundle_done_valid(bundle_done_valid_``name), \
        .bundle_done_ready(bundle_done_ready_``name), \
        .protocol_error(protocol_error_``name), \
        .numeric_overflow(numeric_overflow_``name), .busy(busy_``name), \
        .debug_cycle_count(cycle_count_``name), \
        .debug_row_access_count(row_access_count_``name), \
        .debug_cache_hit_count(cache_hit_count_``name), \
        .debug_cache_miss_count(cache_miss_count_``name), \
        .debug_weight_beat_count(weight_beat_count_``name), \
        .debug_issue_count(issue_count_``name), \
        .debug_signed_product_count(product_count_``name), \
        .debug_commit_count(commit_count_``name)); \
    m1780_directed_8bank_row_memory memory_``name ( \
        .clk_core(clk_core), .rst_core(rst_core), \
        .req_valid(mem_req_valid_``name), .req_ready(mem_req_ready_``name), \
        .req_group(mem_req_group_``name), .req_half(mem_req_half_``name), \
        .req_slice(mem_req_slice_``name), .req_accept(mem_req_accept_``name), \
        .rsp_valid(mem_rsp_valid_``name), .rsp_ready(mem_rsp_ready_``name), \
        .rsp_group(mem_rsp_group_``name), .rsp_half(mem_rsp_half_``name), \
        .rsp_slice(mem_rsp_slice_``name), .rsp_weight(mem_rsp_weight_``name), \
        .rsp_accept(mem_rsp_accept_``name), \
        .request_count(memory_request_count_``name), \
        .response_count(memory_response_count_``name), \
        .request_stall_count(memory_stall_count_``name)); \
    m1780_c2_tsbg_b8_typed_weight_row_frontend_assertions sva_``name ( \
        .clk_core(clk_core), .rst_core(rst_core), \
        .load_valid(load_valid), .load_ready(load_ready_``name), \
        .load_context(load_context), .load_group(load_group), \
        .load_source_value(load_source_value), .load_last(load_last), \
        .load_accept(load_accept_``name), \
        .mem_req_valid(mem_req_valid_``name), .mem_req_ready(mem_req_ready_``name), \
        .mem_req_group(mem_req_group_``name), .mem_req_half(mem_req_half_``name), \
        .mem_req_slice(mem_req_slice_``name), \
        .mem_rsp_valid(mem_rsp_valid_``name), .mem_rsp_ready(mem_rsp_ready_``name), \
        .issue_valid(issue_valid_``name), .issue_ready(issue_ready_``name), \
        .issue_context(issue_context_``name), .issue_group(issue_group_``name), \
        .issue_half(issue_half_``name), .issue_slice(issue_slice_``name), \
        .issue_bank_valid(issue_bank_valid_``name), \
        .issue_source_value(issue_source_value_``name), \
        .issue_weight(issue_weight_``name), \
        .commit_valid(commit_valid_``name), .commit_ready(commit_ready_``name), \
        .commit_context(commit_context_``name), .commit_tag(commit_tag_``name), \
        .commit_slice(commit_slice_``name), \
        .commit_accumulator(commit_accumulator_``name), \
        .commit_terminal(commit_terminal_``name), \
        .protocol_error(protocol_error_``name));

    `CONNECT_DUT(base, 0);
    `CONNECT_DUT(tsbg, 1);
`undef CONNECT_DUT

    assign issue_ready_base = (tb_cycle % 7 != 2);
    assign issue_ready_tsbg = (tb_cycle % 7 != 2);
    assign commit_ready_base = (tb_cycle % 5 != 3);
    assign commit_ready_tsbg = (tb_cycle % 5 != 3);

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            base_done_cycle <= -1;
            tsbg_done_cycle <= -1;
            terminal_base <= 0;
            terminal_tsbg <= 0;
            issue_stall_base <= 0;
            issue_stall_tsbg <= 0;
            commit_stall_base <= 0;
            commit_stall_tsbg <= 0;
            for (int context = 0; context < BUNDLE; context++) begin
                for (int slice = 0; slice < SLICES; slice++) begin
                    observed_base_valid[context][slice] <= 0;
                    observed_tsbg_valid[context][slice] <= 0;
                    for (int lane = 0; lane < LANES; lane++) begin
                        observed_base[context][slice][lane] <= 0;
                        observed_tsbg[context][slice][lane] <= 0;
                    end
                end
            end
        end else begin
            if (issue_valid_base && !issue_ready_base)
                issue_stall_base <= issue_stall_base + 1;
            if (issue_valid_tsbg && !issue_ready_tsbg)
                issue_stall_tsbg <= issue_stall_tsbg + 1;
            if (commit_valid_base && !commit_ready_base)
                commit_stall_base <= commit_stall_base + 1;
            if (commit_valid_tsbg && !commit_ready_tsbg)
                commit_stall_tsbg <= commit_stall_tsbg + 1;
            if (commit_accept_base) begin
                observed_base_valid[commit_context_base][commit_slice_base] <= 1;
                if (commit_terminal_base) terminal_base <= terminal_base + 1;
                for (int lane = 0; lane < LANES; lane++)
                    observed_base[commit_context_base][commit_slice_base][lane]
                        <= commit_accumulator_base[lane];
            end
            if (commit_accept_tsbg) begin
                observed_tsbg_valid[commit_context_tsbg][commit_slice_tsbg] <= 1;
                if (commit_terminal_tsbg) terminal_tsbg <= terminal_tsbg + 1;
                for (int lane = 0; lane < LANES; lane++)
                    observed_tsbg[commit_context_tsbg][commit_slice_tsbg][lane]
                        <= commit_accumulator_tsbg[lane];
            end
            if (bundle_done_valid_base && base_done_cycle < 0)
                base_done_cycle <= tb_cycle;
            if (bundle_done_valid_tsbg && tsbg_done_cycle < 0)
                tsbg_done_cycle <= tb_cycle;
        end
    end

    task automatic send_descriptor(
        input int context, input int group, input logic last);
        int source0, source1;
        int value0, value1;
        begin
            source0 = (context + group) % 8;
            source1 = 8 + ((context * 3 + group) % 8);
            value0 = ((context + group) % 2 == 0) ? 1 : -1;
            value1 = -value0;
            @(negedge clk_core);
            load_valid = 1;
            load_context = context;
            load_tag = 24'h510000 + context;
            load_group = group;
            load_last = last;
            for (int source = 0; source < 16; source++)
                load_source_value[source] = 0;
            load_source_value[source0] = value0;
            load_source_value[source1] = value1;
            do @(posedge clk_core); while (!(load_accept_base && load_accept_tsbg));
            for (int slice = 0; slice < SLICES; slice++) begin
                for (int lane = 0; lane < LANES; lane++) begin
                    raw_weight = (group * 17 + 0 * 11 + slice * 7
                        + source0 * 5 + lane * 3) % 63 - 31;
                    expected[context][slice][lane] =
                        expected[context][slice][lane] + value0 * raw_weight;
                    raw_weight = (group * 17 + 1 * 11 + slice * 7
                        + (source1 - 8) * 5 + lane * 3) % 63 - 31;
                    expected[context][slice][lane] =
                        expected[context][slice][lane] + value1 * raw_weight;
                end
            end
            @(negedge clk_core);
            load_valid = 0;
            load_last = 0;
        end
    endtask

    initial begin : directed_test
        real speedup;
        clk_core = 0;
        rst_core = 1;
        load_valid = 0;
        load_context = 0;
        load_tag = 0;
        load_group = 0;
        load_last = 0;
        bundle_done_ready_base = 0;
        bundle_done_ready_tsbg = 0;
        for (int source = 0; source < 16; source++) load_source_value[source] = 0;
        for (int context = 0; context < BUNDLE; context++)
            for (int slice = 0; slice < SLICES; slice++)
                for (int lane = 0; lane < LANES; lane++)
                    expected[context][slice][lane] = 0;
        repeat (5) @(posedge clk_core);
        @(negedge clk_core); rst_core = 0;

        for (int context = 0; context < BUNDLE; context++)
            for (int group = 0; group < GROUPS; group++)
                send_descriptor(context, group, group == GROUPS - 1);
        start_cycle = tb_cycle;

        wait (bundle_done_valid_base && bundle_done_valid_tsbg);
        @(negedge clk_core);
        if (protocol_error_base || protocol_error_tsbg
                || numeric_overflow_base || numeric_overflow_tsbg)
            $fatal(1, "M1780 unexpected protocol/numeric fault");
        if (row_access_count_base != EXPECTED_ROW_ACCESSES
                || row_access_count_tsbg != EXPECTED_ROW_ACCESSES)
            $fatal(1, "M1780 row-work conservation failed");
        if (issue_count_base != EXPECTED_ISSUES
                || issue_count_tsbg != EXPECTED_ISSUES
                || product_count_base != EXPECTED_PRODUCTS
                || product_count_tsbg != EXPECTED_PRODUCTS
                || commit_count_base != EXPECTED_COMMITS
                || commit_count_tsbg != EXPECTED_COMMITS)
            $fatal(1, "M1780 compute/commit conservation failed");
        if (cache_miss_count_base != 96 || cache_miss_count_tsbg != 12
                || cache_hit_count_base != 0 || cache_hit_count_tsbg != 84)
            $fatal(1, "M1780 exact ordinary-LRU8 miss/hit ledger failed");
        if (weight_beat_count_base != 1152 || weight_beat_count_tsbg != 144
                || memory_request_count_base != weight_beat_count_base
                || memory_request_count_tsbg != weight_beat_count_tsbg
                || memory_response_count_base != weight_beat_count_base
                || memory_response_count_tsbg != weight_beat_count_tsbg)
            $fatal(1, "M1780 eight-bank weight-beat conservation failed");
        if (terminal_base != BUNDLE || terminal_tsbg != BUNDLE
                || issue_stall_base == 0 || issue_stall_tsbg == 0
                || commit_stall_base == 0 || commit_stall_tsbg == 0
                || memory_stall_count_base == 0 || memory_stall_count_tsbg == 0)
            $fatal(1, "M1780 directed coverage incomplete");
        for (int context = 0; context < BUNDLE; context++) begin
            for (int slice = 0; slice < SLICES; slice++) begin
                if (!observed_base_valid[context][slice]
                        || !observed_tsbg_valid[context][slice])
                    $fatal(1, "M1780 missing commit");
                for (int lane = 0; lane < LANES; lane++) begin
                    if (observed_base[context][slice][lane]
                            !== expected[context][slice][lane]
                            || observed_tsbg[context][slice][lane]
                            !== expected[context][slice][lane])
                        $fatal(1, "M1780 signed Acc24 mismatch");
                end
            end
        end
        speedup = real'(base_done_cycle - start_cycle)
            / real'(tsbg_done_cycle - start_cycle);
        if (speedup < 1.15)
            $fatal(1, "M1780 directed same-resource local gate below 1.15x");

        bundle_done_ready_base = 1;
        bundle_done_ready_tsbg = 1;
        @(posedge clk_core);
        @(negedge clk_core);
        bundle_done_ready_base = 0;
        bundle_done_ready_tsbg = 0;

        // One explicit malformed typed value must fail closed in both modes.
        load_valid = 1;
        load_context = 0;
        load_tag = 24'hbad001;
        load_group = 0;
        load_last = 1;
        for (int source = 0; source < 16; source++) load_source_value[source] = 0;
        load_source_value[0] = 8'sd2;
        @(posedge clk_core);
        @(negedge clk_core); load_valid = 0;
        @(posedge clk_core);
        if (!protocol_error_base || !protocol_error_tsbg
                || load_ready_base || load_ready_tsbg)
            $fatal(1, "M1780 malformed-value attack did not fail closed");

        $display("COVERAGE_M1780_TSBG_B8 mixed_signed=1 terminal=%0d "
                 "issue_stall_base=%0d issue_stall_tsbg=%0d "
                 "commit_stall_base=%0d commit_stall_tsbg=%0d "
                 "memory_stall_base=%0d memory_stall_tsbg=%0d attack=1",
                 terminal_tsbg, issue_stall_base, issue_stall_tsbg,
                 commit_stall_base, commit_stall_tsbg,
                 memory_stall_count_base, memory_stall_count_tsbg);
        $display("M1780_LEDGER baseline_cycles=%0d candidate_cycles=%0d "
                 "directed_speedup=%0.6f baseline_misses=%0d "
                 "candidate_misses=%0d baseline_weight_beats=%0d "
                 "candidate_weight_beats=%0d issues=%0d products=%0d commits=%0d",
                 base_done_cycle - start_cycle, tsbg_done_cycle - start_cycle,
                 speedup, cache_miss_count_base, cache_miss_count_tsbg,
                 weight_beat_count_base, weight_beat_count_tsbg,
                 issue_count_tsbg, product_count_tsbg, commit_count_tsbg);
        $display("PASS_M1780_C2_TSBG_B8_TYPED_WEIGHT_ROW_FRONTEND_DIRECTED");
        $finish;
    end

    initial begin
        #2000000;
        $fatal(1, "M1780 timeout");
    end
endmodule

`default_nettype wire
