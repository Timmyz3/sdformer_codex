`timescale 1ns/1ps
`default_nettype none

// M2217 production power harness.  The compile command must define exactly
// one M2217_SCHEDULE_MODE (0 ordinary or 1 TSBG) and elaborate only this top.
// One M2018 DUT is instantiated; the directed SRAM model is outside the UCLI
// SAIF scope.  Low/median/high workload identity is selected by plusarg from
// the pre-power M2217 selection manifest, never by a measured power result.
`ifndef M2217_SCHEDULE_MODE
`define M2217_SCHEDULE_MODE -1
`endif

module tb_m2217_m2018_tsbg_matched_native_saif_power;
    localparam int BUNDLE=4, GROUPS=48, SLICES=6, LANES=16;
    localparam int SCHEDULE_MODE=`M2217_SCHEDULE_MODE;
    localparam int PRELOAD_CYCLES=383;
    localparam int INTERNAL_ELEMENTS=228;
    localparam realtime CLOCK_PERIOD_NS=3.0;
    localparam string FIXTURE = "/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.memh";

    logic clk_core=0, rst_core, load_valid;
    logic [2:0] load_context;
    logic [23:0] load_tag;
    logic [5:0] load_group;
    logic [15:0] load_source_active, load_source_sign;
    logic load_last;
    integer tb_cycle, done_cycle, execute_start_cycle;
    integer terminal_count, reorder_count, last_response_bank;
    integer independent_stall_count;
    integer workload_slot, sample_id, layer_id, token_start, source_groups;
    integer expected_rows, expected_issues, expected_products, expected_commits;
    integer expected_misses, expected_hits, expected_evictions;
    integer expected_bundles, expected_reads, expected_cycles;
    string stratum, sequence_name, target_name, token_role;
    logic [31:0] fixture_word [0:368639];
    integer signed expected [0:BUNDLE-1][0:SLICES-1][0:LANES-1];
    logic observed [0:BUNDLE-1][0:SLICES-1];
    logic measurement_window_active=0;
    realtime measurement_begin_time;
    m2160_ordinary_side_if axis();

    always #1.5 clk_core = ~clk_core;
    initial begin repeat (100000) @(posedge clk_core);
        $fatal(1, "M2217 watchdog expired"); end
    always @(posedge clk_core)
        if (rst_core) tb_cycle <= 0; else tb_cycle <= tb_cycle + 1;

    m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend #(
        .SCHEDULE_MODE(SCHEDULE_MODE), .SOURCE_GROUPS(GROUPS)
    ) dut_axis (
        .clk_core(clk_core), .rst_core(rst_core),
        .load_valid(load_valid), .load_ready(axis.load_ready),
        .load_context(load_context), .load_tag(load_tag),
        .load_group(load_group), .load_source_active(load_source_active),
        .load_source_sign(load_source_sign), .load_last(load_last),
        .load_accept(axis.load_accept),
        .mem_req_valid(axis.mem_req_valid), .mem_req_ready(axis.mem_req_ready),
        .mem_req_epoch(axis.mem_req_epoch), .mem_req_slot(axis.mem_req_slot),
        .mem_req_generation(axis.mem_req_generation),
        .mem_req_tag(axis.mem_req_tag),
        .mem_req_output_block(axis.mem_req_output_block),
        .mem_req_slice(axis.mem_req_slice),
        .mem_req_source_channel(axis.mem_req_source_channel),
        .mem_req_accept(axis.mem_req_accept),
        .mem_rsp_valid(axis.mem_rsp_valid), .mem_rsp_ready(axis.mem_rsp_ready),
        .mem_rsp_epoch(axis.mem_rsp_epoch), .mem_rsp_slot(axis.mem_rsp_slot),
        .mem_rsp_generation(axis.mem_rsp_generation),
        .mem_rsp_tag(axis.mem_rsp_tag), .mem_rsp_weight(axis.mem_rsp_weight),
        .mem_rsp_accept(axis.mem_rsp_accept),
        .bridge_valid(axis.bridge_valid), .bridge_ready(axis.bridge_ready),
        .bridge_context(axis.bridge_context), .bridge_group(axis.bridge_group),
        .bridge_half(axis.bridge_half), .bridge_slice(axis.bridge_slice),
        .bridge_bank_valid(axis.bridge_bank_valid),
        .bridge_source_channel(axis.bridge_source_channel),
        .bridge_source_value(axis.bridge_source_value),
        .bridge_effective_weight(axis.bridge_effective_weight),
        .bridge_accept(axis.bridge_accept),
        .commit_valid(axis.commit_valid), .commit_ready(axis.commit_ready),
        .commit_context(axis.commit_context), .commit_tag(axis.commit_tag),
        .commit_slice(axis.commit_slice),
        .commit_accumulator(axis.commit_accumulator),
        .commit_terminal(axis.commit_terminal),
        .commit_accept(axis.commit_accept),
        .bundle_done_valid(axis.bundle_done_valid),
        .bundle_done_ready(axis.bundle_done_ready),
        .protocol_error(axis.protocol_error),
        .stale_response_seen(axis.stale_response_seen),
        .numeric_overflow(axis.numeric_overflow), .busy(axis.busy),
        .debug_cycle_count(axis.cycle_count),
        .debug_row_access_count(axis.row_access_count),
        .debug_cache_hit_count(axis.cache_hit_count),
        .debug_cache_miss_count(axis.cache_miss_count),
        .debug_cache_eviction_count(axis.cache_eviction_count),
        .debug_weight_bundle_beat_count(axis.weight_bundle_beat_count),
        .debug_scalar_bank_request_count(axis.scalar_bank_request_count),
        .debug_scalar_bank_response_count(axis.scalar_bank_response_count),
        .debug_issue_count(axis.issue_count),
        .debug_signed_product_count(axis.product_count),
        .debug_commit_count(axis.commit_count)
    );

    m1880_c2_tsbg_b4_real_channel_signed_frontend_assertions #(
        .SOURCE_GROUPS(GROUPS)
    ) sva_axis (
        .clk_core(clk_core), .rst_core(rst_core),
        .load_valid(load_valid), .load_ready(axis.load_ready),
        .load_accept(axis.load_accept), .load_context(load_context),
        .mem_req_valid(axis.mem_req_valid), .mem_req_ready(axis.mem_req_ready),
        .mem_req_epoch(axis.mem_req_epoch), .mem_req_slot(axis.mem_req_slot),
        .mem_req_generation(axis.mem_req_generation),
        .mem_req_tag(axis.mem_req_tag),
        .mem_req_output_block(axis.mem_req_output_block),
        .mem_req_slice(axis.mem_req_slice),
        .mem_req_source_channel(axis.mem_req_source_channel),
        .mem_req_accept(axis.mem_req_accept),
        .mem_rsp_valid(axis.mem_rsp_valid), .mem_rsp_ready(axis.mem_rsp_ready),
        .mem_rsp_epoch(axis.mem_rsp_epoch), .mem_rsp_slot(axis.mem_rsp_slot),
        .mem_rsp_generation(axis.mem_rsp_generation),
        .mem_rsp_tag(axis.mem_rsp_tag), .mem_rsp_weight(axis.mem_rsp_weight),
        .mem_rsp_accept(axis.mem_rsp_accept),
        .bridge_valid(axis.bridge_valid), .bridge_ready(axis.bridge_ready),
        .bridge_context(axis.bridge_context), .bridge_group(axis.bridge_group),
        .bridge_half(axis.bridge_half), .bridge_slice(axis.bridge_slice),
        .bridge_bank_valid(axis.bridge_bank_valid),
        .bridge_source_channel(axis.bridge_source_channel),
        .bridge_source_value(axis.bridge_source_value),
        .bridge_effective_weight(axis.bridge_effective_weight),
        .bridge_accept(axis.bridge_accept),
        .commit_valid(axis.commit_valid), .commit_ready(axis.commit_ready),
        .commit_context(axis.commit_context), .commit_tag(axis.commit_tag),
        .commit_slice(axis.commit_slice),
        .commit_accumulator(axis.commit_accumulator),
        .commit_terminal(axis.commit_terminal),
        .commit_accept(axis.commit_accept),
        .protocol_error(axis.protocol_error),
        .stale_response_seen(axis.stale_response_seen),
        .numeric_overflow(axis.numeric_overflow),
        .debug_cache_eviction_count(axis.cache_eviction_count),
        .debug_weight_bundle_beat_count(axis.weight_bundle_beat_count)
    );

    for (genvar bank=0; bank<8; bank++) begin : g_memory
        m2160_directed_scalar_bank_memory #(.BANK_ID(bank)) memory (
            .clk_core(clk_core), .rst_core(rst_core),
            .req_valid(axis.mem_req_valid[bank]),
            .req_ready(axis.mem_req_ready[bank]),
            .req_epoch(axis.mem_req_epoch[bank]),
            .req_slot(axis.mem_req_slot[bank]),
            .req_generation(axis.mem_req_generation[bank]),
            .req_tag(axis.mem_req_tag[bank]),
            .req_output_block(axis.mem_req_output_block[bank]),
            .req_slice(axis.mem_req_slice[bank]),
            .req_source_channel(axis.mem_req_source_channel[bank]),
            .req_accept(axis.mem_req_accept[bank]),
            .rsp_valid(axis.mem_rsp_valid[bank]),
            .rsp_ready(axis.mem_rsp_ready[bank]),
            .rsp_epoch(axis.mem_rsp_epoch[bank]),
            .rsp_slot(axis.mem_rsp_slot[bank]),
            .rsp_generation(axis.mem_rsp_generation[bank]),
            .rsp_tag(axis.mem_rsp_tag[bank]),
            .rsp_weight(axis.mem_rsp_weight[bank]),
            .rsp_accept(axis.mem_rsp_accept[bank]),
            .request_count(axis.memory_request_count[bank]),
            .response_count(axis.memory_response_count[bank]),
            .request_stall_count(axis.memory_stall_count[bank])
        );
    end

    function automatic integer directed_weight(
        input integer group_index, half_index, output_slice, bank, lane);
        integer value;
        begin
            value=(group_index*17+half_index*11+output_slice*7+bank*5+lane*3)%255-127;
            if (group_index==0 && half_index==0 && output_slice==0
                    && bank==0 && lane==0) value=-128;
            return value;
        end
    endfunction

    task automatic choose_window;
        begin
            if (!$value$plusargs("M2217_STRATUM=%s", stratum))
                $fatal(1, "M2217 missing stratum plusarg");
            if (stratum=="low") begin
                workload_slot=1606; sample_id=33; layer_id=15;
                sequence_name="zurich_city_12_a"; target_name="FC2";
                token_role="middle"; token_start=24000; source_groups=48;
                expected_rows=35; expected_issues=240; expected_products=4992;
                expected_commits=24;
                if (SCHEDULE_MODE==0) begin expected_misses=35; expected_hits=0;
                    expected_evictions=31; expected_bundles=420;
                    expected_reads=3360; expected_cycles=4717; end
                else begin expected_misses=35; expected_hits=0;
                    expected_evictions=31; expected_bundles=420;
                    expected_reads=3360; expected_cycles=4717; end
            end else if (stratum=="median") begin
                workload_slot=526; sample_id=10; layer_id=30;
                sequence_name="interlaken_01_a"; target_name="FC1";
                token_role="middle"; token_start=1500; source_groups=48;
                expected_rows=50; expected_issues=348; expected_products=6144;
                expected_commits=24;
                if (SCHEDULE_MODE==0) begin expected_misses=50; expected_hits=0;
                    expected_evictions=46; expected_bundles=600;
                    expected_reads=4800; expected_cycles=6733; end
                else begin expected_misses=32; expected_hits=18;
                    expected_evictions=28; expected_bundles=384;
                    expected_reads=3072; expected_cycles=4465; end
            end else if (stratum=="high") begin
                workload_slot=1071; sample_id=22; layer_id=13;
                sequence_name="thun_01_b"; target_name="FC2";
                token_role="first"; token_start=0; source_groups=48;
                expected_rows=163; expected_issues=1482; expected_products=39072;
                expected_commits=24;
                if (SCHEDULE_MODE==0) begin expected_misses=163; expected_hits=0;
                    expected_evictions=159; expected_bundles=1956;
                    expected_reads=15648; expected_cycles=22294; end
                else begin expected_misses=46; expected_hits=117;
                    expected_evictions=42; expected_bundles=552;
                    expected_reads=4416; expected_cycles=7554; end
            end else $fatal(1, "M2217 unknown stratum");
        end
    endtask

    task automatic prepare_descriptor(input integer ctx, group_index);
        integer source, value, fixture_index;
        begin
            fixture_index=workload_slot*BUNDLE*GROUPS+ctx*GROUPS+group_index;
            load_source_active=fixture_word[fixture_index][15:0];
            load_source_sign=fixture_word[fixture_index][31:16]&load_source_active;
            for (int os=0; os<SLICES; os++) for (int lane=0; lane<LANES; lane++)
                for (source=0; source<16; source++)
                    if (load_source_active[source]) begin
                        value=load_source_sign[source] ? -1 : 1;
                        expected[ctx][os][lane]+=value*directed_weight(
                            group_index,source/8,os,source%8,lane);
                    end
        end
    endtask

    task automatic send_descriptor;
        integer waits; logic accepted;
        begin
            accepted=0; @(negedge clk_core); load_valid=1;
            for (waits=0; waits<10000 && !accepted; waits++) begin
                @(posedge clk_core); if (axis.load_accept) begin
                    accepted=1; load_valid<=0; end
            end
            if (!accepted) $fatal(1,"M2217 load timeout");
            @(negedge clk_core); load_valid=0;
        end
    endtask

    task automatic load_window;
        begin
            for (int ctx=0;ctx<BUNDLE;ctx++) for (int group_index=0;
                    group_index<GROUPS;group_index++) begin
                prepare_descriptor(ctx,group_index); load_context=ctx;
                load_tag=24'h340000+ctx; load_group=group_index;
                load_last=(group_index==GROUPS-1); send_descriptor();
            end
        end
    endtask

    task automatic check_known;
        begin
            if ($isunknown({clk_core,rst_core,load_valid,load_context,load_tag,
                    load_group,load_source_active,load_source_sign,load_last,
                    axis.load_ready,axis.load_accept,axis.mem_req_valid,
                    axis.mem_req_ready,axis.mem_req_accept,axis.mem_rsp_valid,
                    axis.mem_rsp_ready,axis.mem_rsp_accept,axis.bridge_valid,
                    axis.bridge_ready,axis.bridge_accept,axis.commit_valid,
                    axis.commit_ready,axis.commit_accept,axis.bundle_done_valid,
                    axis.bundle_done_ready,axis.protocol_error,
                    axis.stale_response_seen,axis.numeric_overflow,axis.busy}))
                $fatal(1,"M2217 public X/Z");
            if (axis.protocol_error||axis.stale_response_seen||axis.numeric_overflow)
                $fatal(1,"M2217 DUT fault");
        end
    endtask

    task automatic check_completion;
        integer measured_cycles, req_sum, rsp_sum;
        begin
            measured_cycles=done_cycle-execute_start_cycle;
            req_sum=0; rsp_sum=0;
            for (int bank=0;bank<8;bank++) begin
                req_sum+=axis.memory_request_count[bank];
                rsp_sum+=axis.memory_response_count[bank];
            end
            if (measured_cycles!=expected_cycles
                    || axis.row_access_count!=expected_rows
                    || axis.issue_count!=expected_issues
                    || axis.product_count!=expected_products
                    || axis.commit_count!=expected_commits
                    || axis.cache_miss_count!=expected_misses
                    || axis.cache_hit_count!=expected_hits
                    || axis.cache_eviction_count!=expected_evictions
                    || axis.weight_bundle_beat_count!=expected_bundles
                    || axis.scalar_bank_request_count!=expected_reads
                    || axis.scalar_bank_response_count!=expected_reads
                    || req_sum!=expected_reads || rsp_sum!=expected_reads
                    || terminal_count!=4 || reorder_count==0
                    || independent_stall_count==0)
                $fatal(1,"M2217 completion ledger drift");
            for (int ctx=0;ctx<BUNDLE;ctx++) for (int os=0;os<SLICES;os++)
                if (!observed[ctx][os]) $fatal(1,"M2217 missing commit");
        end
    endtask

    always_comb begin
        axis.bridge_ready=(tb_cycle%11!=3);
        axis.commit_ready=(tb_cycle%13!=5);
        axis.bundle_done_ready=1;
    end
    always @(posedge clk_core) if (!rst_core) begin
        for (int bank=0;bank<8;bank++) begin
            if (axis.mem_rsp_accept[bank]) begin
                if (last_response_bank>=0 && bank<last_response_bank)
                    reorder_count<=reorder_count+1;
                last_response_bank<=bank;
            end
            if (axis.mem_req_valid[bank]&&!axis.mem_req_ready[bank])
                independent_stall_count<=independent_stall_count+1;
        end
        if (axis.commit_accept) begin
            if (observed[axis.commit_context][axis.commit_slice])
                $fatal(1,"M2217 duplicate commit");
            observed[axis.commit_context][axis.commit_slice]<=1;
            for (int lane=0;lane<LANES;lane++)
                if (axis.commit_accumulator[lane]!==
                        expected[axis.commit_context][axis.commit_slice][lane])
                    $fatal(1,"M2217 arithmetic mismatch");
            if (axis.commit_terminal) terminal_count<=terminal_count+1;
        end
        if (axis.bundle_done_valid && done_cycle<0) done_cycle<=tb_cycle;
    end
    always @(negedge clk_core) if (measurement_window_active) begin
        #0.01; check_known();
    end

    initial begin : single_axis_campaign
        if (SCHEDULE_MODE!=0 && SCHEDULE_MODE!=1)
            $fatal(1,"M2217 schedule mode compile definition missing/illegal");
        choose_window(); tb_cycle=0; done_cycle=-1; execute_start_cycle=-1;
        terminal_count=0; reorder_count=0; last_response_bank=-1;
        independent_stall_count=0; rst_core=1; load_valid=0;
        load_context=0; load_tag=0; load_group=0; load_source_active=0;
        load_source_sign=0; load_last=0;
        for (int ctx=0;ctx<BUNDLE;ctx++) for (int os=0;os<SLICES;os++) begin
            observed[ctx][os]=0;
            for (int lane=0;lane<LANES;lane++) expected[ctx][os][lane]=0;
        end
        $readmemh(FIXTURE,fixture_word);
        repeat (5) @(posedge clk_core); rst_core=0; load_window();
        execute_start_cycle=tb_cycle;
        if (execute_start_cycle!=PRELOAD_CYCLES)
            $fatal(1,"M2217 preload denominator drift");
        @(negedge clk_core); #0.01; check_known();
        measurement_begin_time=$realtime; measurement_window_active=1;
        $display("M2217_WINDOW_BEGIN axis_mode=%0d stratum=%s slot=%0d sample=%0d sequence=%s layer=%0d target=%s token_role=%s token_start=%0d source_groups=%0d preload_cycles=383 time_ns=%0.2f",
            SCHEDULE_MODE,stratum,workload_slot,sample_id,sequence_name,
            layer_id,target_name,token_role,token_start,source_groups,
            measurement_begin_time);
        $stop;
        wait(done_cycle>=0); @(negedge clk_core); #0.01;
        check_completion(); check_known(); measurement_window_active=0;
        if (($realtime-measurement_begin_time)!=expected_cycles*CLOCK_PERIOD_NS)
            $fatal(1,"M2217 physical duration drift");
        $display("M2217_WINDOW_END axis_mode=%0d stratum=%s cycles=%0d rows=%0d issues=%0d products=%0d commits=%0d bundles=%0d accepted_bank_requests=%0d duration_ns=%0.2f record_conservation=1 tx_required_zero=1",
            SCHEDULE_MODE,stratum,expected_cycles,expected_rows,
            expected_issues,expected_products,expected_commits,
            expected_bundles,expected_reads,$realtime-measurement_begin_time);
        $display("PASS_M2217_SINGLE_DUT_NATIVE_SAIF axis_mode=%0d stratum=%s arithmetic=1 ledger=1 frontends=1 second_axis=0 paper_result=0",
            SCHEDULE_MODE,stratum);
        $stop;
    end
endmodule

`default_nettype wire
