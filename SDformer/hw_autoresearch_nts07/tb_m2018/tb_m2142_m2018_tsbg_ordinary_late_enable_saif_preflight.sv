`timescale 1ns/1ps
`default_nettype none

// M2142 is an additive, ordinary-only native-SAIF acquisition preflight.
// It implements the only safe successor allowed by the independent M2140
// M2139 failure hammer: the UCLI monitor is enabled before time advances,
// observes reset plus the frozen 383-cycle preload, and is reset at this
// module's first stop before the exact 20,292-cycle execution window.
//
// The first-boundary census below is deliberately observational.  It neither
// forces nor deposits state, does not filter SAIF records, and changes no RTL.
// Every element from all five M2140 TX-fingerprint families is required known
// before the first $stop lets UCLI issue power -reset.
module tb_m2142_m2018_tsbg_ordinary_late_enable_saif_preflight;
    localparam int FROZEN_WORKLOAD_SLOT = 42;
    localparam int FROZEN_PRELOAD_CYCLES = 383;
    localparam int FROZEN_ROWS = 149;
    localparam int FROZEN_ISSUES = 1278;
    localparam int FROZEN_PRODUCTS = 29472;
    localparam int FROZEN_COMMITS = 24;
    localparam int FROZEN_BUNDLES = 1788;
    localparam int FROZEN_SCALAR_READS = 14304;
    localparam int FROZEN_CYCLES = 20292;
    localparam int FROZEN_INTERNAL_ELEMENTS = 228;
    localparam realtime FROZEN_CLOCK_PERIOD_NS = 3.0;

    logic measurement_window_active = 1'b0;
    realtime measurement_begin_time;
    tb_m2051_ep34_tsbg_full40_cycle core();

    task automatic check_axis_selection;
        begin
            if (!$test$plusargs("M2142_AXIS_ORDINARY"))
                $fatal(1, "M2142 requires ordinary-axis plusarg");
            if ($test$plusargs("M2125_AXIS_ORDINARY")
                    || $test$plusargs("M2125_AXIS_TSBG")
                    || $test$plusargs("M2142_AXIS_TSBG"))
                $fatal(1, "M2142 rejects predecessor/TSBG axis plusargs");
        end
    endtask

    task automatic check_workload_identity;
        begin
            if (core.workload_slot != FROZEN_WORKLOAD_SLOT
                    || core.sample_id != 0 || core.layer_id != 28
                    || core.is_fc2 != 0 || core.token_start != 0
                    || core.real_source_groups != 48
                    || core.expected_rows != FROZEN_ROWS
                    || core.expected_issues != FROZEN_ISSUES
                    || core.expected_products != FROZEN_PRODUCTS
                    || core.expected_base_misses != 149
                    || core.expected_base_hits != 0
                    || core.expected_base_evictions != 145
                    || core.expected_base_bundles != FROZEN_BUNDLES)
                $fatal(1, "M2142 frozen workload identity drift");
            if (core.full_execute_start_cycle != FROZEN_PRELOAD_CYCLES)
                $fatal(1, "M2142 preload denominator drift");
        end
    endtask

    task automatic census_internal_knownness;
        integer row_live_known, row_live_one;
        integer cache_valid_known, cache_valid_one;
        integer slot_valid_known, slot_valid_one;
        integer bridge_overflow_known, bridge_overflow_one;
        integer rsp_shape_legal_known, rsp_shape_legal_one;
        integer total_known;
        begin
            row_live_known = 0;
            row_live_one = 0;
            cache_valid_known = 0;
            cache_valid_one = 0;
            slot_valid_known = 0;
            slot_valid_one = 0;
            bridge_overflow_known = 0;
            bridge_overflow_one = 0;
            rsp_shape_legal_known = 0;
            rsp_shape_legal_one = 0;

            for (int ctx = 0; ctx < 4; ctx++) begin
                for (int group = 0; group < 48; group++) begin
                    if (!$isunknown(core.dut_base.implementation
                                             .row_live_q[ctx][group])) begin
                        row_live_known++;
                        row_live_one += core.dut_base.implementation
                                            .row_live_q[ctx][group];
                    end
                end
            end
            for (int entry = 0; entry < 4; entry++) begin
                if (!$isunknown(core.dut_base.implementation
                                         .cache_valid_q[entry])) begin
                    cache_valid_known++;
                    cache_valid_one += core.dut_base.implementation
                                            .cache_valid_q[entry];
                end
            end
            for (int slot = 0; slot < 8; slot++) begin
                if (!$isunknown(core.dut_base.implementation.adapter
                                         .slot_valid_q[slot])) begin
                    slot_valid_known++;
                    slot_valid_one += core.dut_base.implementation.adapter
                                           .slot_valid_q[slot];
                end
            end
            for (int lane = 0; lane < 16; lane++) begin
                if (!$isunknown(core.dut_base.implementation
                                         .bridge_overflow[lane])) begin
                    bridge_overflow_known++;
                    bridge_overflow_one += core.dut_base.implementation
                                                 .bridge_overflow[lane];
                end
            end
            for (int bank = 0; bank < 8; bank++) begin
                if (!$isunknown(core.dut_base.implementation.adapter
                                         .rsp_shape_legal[bank])) begin
                    rsp_shape_legal_known++;
                    rsp_shape_legal_one += core.dut_base.implementation.adapter
                                               .rsp_shape_legal[bank];
                end
            end
            total_known = row_live_known + cache_valid_known
                + slot_valid_known + bridge_overflow_known
                + rsp_shape_legal_known;
            $display("M2142_INTERNAL_KNOWNNESS_CENSUS phase=pre_power_reset row_live=%0d/192 row_live_one=%0d cache_valid=%0d/4 cache_valid_one=%0d slot_valid=%0d/8 slot_valid_one=%0d bridge_overflow=%0d/16 bridge_overflow_one=%0d rsp_shape_legal=%0d/8 rsp_shape_legal_one=%0d total=%0d/228 observe_only=1 force=0 deposit=0 mask=0 rtl_edit=0",
                row_live_known, row_live_one,
                cache_valid_known, cache_valid_one,
                slot_valid_known, slot_valid_one,
                bridge_overflow_known, bridge_overflow_one,
                rsp_shape_legal_known, rsp_shape_legal_one, total_known);
            if (row_live_known != 192 || cache_valid_known != 4
                    || slot_valid_known != 8
                    || bridge_overflow_known != 16
                    || rsp_shape_legal_known != 8
                    || total_known != FROZEN_INTERNAL_ELEMENTS)
                $fatal(1, "M2142 first-boundary internal knownness census failed");
        end
    endtask

    task automatic check_public_known;
        begin
            if ($isunknown({core.clk_core, core.rst_core,
                    core.load_context, core.load_tag, core.load_group,
                    core.load_source_active, core.load_source_sign,
                    core.load_last, core.load_valid_base,
                    core.base.load_ready, core.base.load_accept,
                    core.base.mem_req_valid, core.base.mem_req_ready,
                    core.base.mem_req_accept, core.base.mem_rsp_valid,
                    core.base.mem_rsp_ready, core.base.mem_rsp_accept,
                    core.base.bridge_valid, core.base.bridge_ready,
                    core.base.bridge_accept, core.base.commit_valid,
                    core.base.commit_ready, core.base.commit_accept,
                    core.base.bundle_done_valid,
                    core.base.bundle_done_ready, core.base.protocol_error,
                    core.base.stale_response_seen,
                    core.base.numeric_overflow, core.base.busy,
                    core.base.cycle_count, core.base.row_access_count,
                    core.base.cache_hit_count, core.base.cache_miss_count,
                    core.base.cache_eviction_count,
                    core.base.weight_bundle_beat_count,
                    core.base.scalar_bank_request_count,
                    core.base.scalar_bank_response_count,
                    core.base.issue_count, core.base.product_count,
                    core.base.commit_count}))
                $fatal(1, "M2142 ordinary public X/Z");
            if (core.base.protocol_error || core.base.stale_response_seen
                    || core.base.numeric_overflow)
                $fatal(1, "M2142 ordinary fault in measured window");
            for (int bank = 0; bank < 8; bank++) begin
                if (core.base.mem_req_valid[bank]
                        && $isunknown({core.base.mem_req_epoch[bank],
                            core.base.mem_req_slot[bank],
                            core.base.mem_req_generation[bank],
                            core.base.mem_req_tag[bank],
                            core.base.mem_req_output_block[bank],
                            core.base.mem_req_slice[bank],
                            core.base.mem_req_source_channel[bank]}))
                    $fatal(1, "M2142 ordinary request payload X/Z");
                if (core.base.mem_rsp_valid[bank]
                        && $isunknown({core.base.mem_rsp_epoch[bank],
                            core.base.mem_rsp_slot[bank],
                            core.base.mem_rsp_generation[bank],
                            core.base.mem_rsp_tag[bank]}))
                    $fatal(1, "M2142 ordinary response payload X/Z");
                if (core.base.bridge_valid
                        && core.base.bridge_bank_valid[bank]
                        && $isunknown({
                            core.base.bridge_source_channel[bank],
                            core.base.bridge_source_value[bank]}))
                    $fatal(1, "M2142 ordinary bridge payload X/Z");
                for (int lane = 0; lane < 16; lane++) begin
                    if (core.base.mem_rsp_valid[bank]
                            && $isunknown(core.base.mem_rsp_weight[bank][lane]))
                        $fatal(1, "M2142 ordinary response weight X/Z");
                    if (core.base.bridge_valid
                            && core.base.bridge_bank_valid[bank]
                            && $isunknown(core.base
                                      .bridge_effective_weight[bank][lane]))
                        $fatal(1, "M2142 ordinary effective weight X/Z");
                end
            end
            if (core.base.bridge_valid
                    && $isunknown({core.base.bridge_bank_valid,
                        core.base.bridge_context, core.base.bridge_group,
                        core.base.bridge_half, core.base.bridge_slice}))
                $fatal(1, "M2142 ordinary bridge header X/Z");
            if (core.base.commit_valid
                    && $isunknown({core.base.commit_context,
                        core.base.commit_tag, core.base.commit_slice,
                        core.base.commit_terminal}))
                $fatal(1, "M2142 ordinary commit payload X/Z");
            for (int lane = 0; lane < 16; lane++) begin
                if (core.base.commit_valid
                        && $isunknown(core.base.commit_accumulator[lane]))
                    $fatal(1, "M2142 ordinary commit accumulator X/Z");
            end
        end
    endtask

    task automatic check_completion;
        integer measured_cycles;
        begin
            measured_cycles = core.base_done_cycle
                - core.full_execute_start_cycle;
            if (measured_cycles != FROZEN_CYCLES
                    || core.base.row_access_count != FROZEN_ROWS
                    || core.base.issue_count != FROZEN_ISSUES
                    || core.base.product_count != FROZEN_PRODUCTS
                    || core.base.commit_count != FROZEN_COMMITS
                    || core.base.cache_miss_count != 149
                    || core.base.cache_hit_count != 0
                    || core.base.cache_eviction_count != 145
                    || core.base.weight_bundle_beat_count != FROZEN_BUNDLES
                    || core.base.scalar_bank_request_count
                        != FROZEN_SCALAR_READS
                    || core.base.scalar_bank_response_count
                        != FROZEN_SCALAR_READS)
                $fatal(1, "M2142 ordinary completion ledger drift");
        end
    endtask

    always @(negedge core.clk_core) begin : settled_window_monitor
        if (measurement_window_active) begin
            #0.01;
            check_public_known();
        end
    end

    initial begin : m2142_late_enable_causal_preflight
        check_axis_selection();
        wait (core.full_execute_start_cycle >= 0);
        @(negedge core.clk_core);
        #0.01;
        // This must be the first first-boundary diagnostic action.
        census_internal_knownness();
        check_workload_identity();
        check_public_known();
        measurement_begin_time = $realtime;
        measurement_window_active = 1'b1;
        $display("M2142_RTL_SAIF_WINDOW_BEGIN sampling=settled_negedge global_slot=42 sample=0 layer=28 is_fc2=0 token_start=0 source_groups=48 preload_cycles=383 time_ns=%0.2f next_ucli_action=power_reset",
            measurement_begin_time);
        $stop;
        wait (core.base_done_cycle >= 0);
        @(negedge core.clk_core);
        #0.01;
        check_completion();
        check_public_known();
        measurement_window_active = 1'b0;
        if (($realtime - measurement_begin_time)
                != FROZEN_CYCLES * FROZEN_CLOCK_PERIOD_NS)
            $fatal(1, "M2142 ordinary physical window duration drift");
        $display("M2142_RTL_SAIF_WINDOW_END axis=ordinary_lru4 sampling=settled_negedge measurement_cycles=20292 scalar_weight_reads=14304 duration_ns=%0.2f",
            $realtime - measurement_begin_time);
        $display("PASS_M2142_ORDINARY_LATE_ENABLE_SAIF_PREFLIGHT ledger_exact=1 internal_census_exact=1 enable_before_reset_preload=1 power_reset_at_first_stop=1 initreg_diagnostic_only=1 paper_citable=0");
        $stop;
    end
endmodule

`default_nettype wire
