`timescale 1ns/1ps
`default_nettype none

// M2105 source-only RTL activity wrapper.  It deliberately reuses the sealed
// M2051 fixed ep34 workload and instantiates the same M2018 implementation on
// both axes.  SCHEDULE_MODE=0/1 is the only DUT parameter difference.  UCLI
// records only core.dut_{base,tsbg}.implementation, never the testbench or
// directed SRAM model.  This source is not an EDA result; M2106 independent
// source admission is required before the single authorized campaign.
module tb_m2105_m2018_tsbg_rtl_saifmap_power;
    localparam int FROZEN_WORKLOAD_SLOT = 42;
    localparam int FROZEN_PRELOAD_CYCLES = 383;
    localparam int FROZEN_ROWS = 149;
    localparam int FROZEN_ISSUES = 1278;
    localparam int FROZEN_PRODUCTS = 29472;
    localparam int FROZEN_COMMITS = 24;
    localparam int FROZEN_BASE_BUNDLES = 1788;
    localparam int FROZEN_TSBG_BUNDLES = 576;
    localparam int FROZEN_BASE_SCALAR = 14304;
    localparam int FROZEN_TSBG_SCALAR = 4608;
    localparam int FROZEN_BASE_CYCLES = 20292;
    localparam int FROZEN_TSBG_CYCLES = 7569;

    bit axis_ordinary;
    bit axis_tsbg;
    logic measurement_window_active = 1'b0;
    tb_m2051_ep34_tsbg_full40_cycle core();

    task automatic check_axis_selection;
        begin
            axis_ordinary = $test$plusargs("M2105_AXIS_ORDINARY");
            axis_tsbg = $test$plusargs("M2105_AXIS_TSBG");
            if (axis_ordinary == axis_tsbg)
                $fatal(1, "M2105 requires exactly one axis plusarg");
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
                    || core.expected_tsbg_misses != 48
                    || core.expected_tsbg_hits != 101
                    || core.expected_tsbg_evictions != 44
                    || core.expected_base_bundles != FROZEN_BASE_BUNDLES
                    || core.expected_tsbg_bundles != FROZEN_TSBG_BUNDLES)
                $fatal(1, "M2105 fixed workload identity drift");
            if (core.full_execute_start_cycle != FROZEN_PRELOAD_CYCLES)
                $fatal(1, "M2105 preload denominator drift");
        end
    endtask

    task automatic check_selected_known;
        begin
            if ($isunknown({core.clk_core, core.rst_core,
                    core.load_context, core.load_tag, core.load_group,
                    core.load_source_active, core.load_source_sign,
                    core.load_last}))
                $fatal(1, "M2105 common stimulus X/Z in power window");
            if (axis_ordinary) begin
                if ($isunknown({core.load_valid_base, core.base.load_ready,
                        core.base.load_accept, core.base.mem_req_valid,
                        core.base.mem_req_ready, core.base.mem_req_accept,
                        core.base.mem_rsp_valid, core.base.mem_rsp_ready,
                        core.base.mem_rsp_accept, core.base.bridge_valid,
                        core.base.bridge_ready, core.base.bridge_accept,
                        core.base.commit_valid, core.base.commit_ready,
                        core.base.commit_accept, core.base.bundle_done_valid,
                        core.base.bundle_done_ready, core.base.protocol_error,
                        core.base.numeric_overflow, core.base.busy,
                        core.base.cycle_count, core.base.row_access_count,
                        core.base.cache_hit_count, core.base.cache_miss_count,
                        core.base.cache_eviction_count,
                        core.base.weight_bundle_beat_count,
                        core.base.scalar_bank_request_count,
                        core.base.scalar_bank_response_count,
                        core.base.issue_count, core.base.product_count,
                        core.base.commit_count}))
                    $fatal(1, "M2105 ordinary DUT public X/Z");
                if (core.base.protocol_error || core.base.numeric_overflow)
                    $fatal(1, "M2105 ordinary DUT fault in window");
            end else begin
                if ($isunknown({core.load_valid_tsbg, core.tsbg.load_ready,
                        core.tsbg.load_accept, core.tsbg.mem_req_valid,
                        core.tsbg.mem_req_ready, core.tsbg.mem_req_accept,
                        core.tsbg.mem_rsp_valid, core.tsbg.mem_rsp_ready,
                        core.tsbg.mem_rsp_accept, core.tsbg.bridge_valid,
                        core.tsbg.bridge_ready, core.tsbg.bridge_accept,
                        core.tsbg.commit_valid, core.tsbg.commit_ready,
                        core.tsbg.commit_accept, core.tsbg.bundle_done_valid,
                        core.tsbg.bundle_done_ready, core.tsbg.protocol_error,
                        core.tsbg.numeric_overflow, core.tsbg.busy,
                        core.tsbg.cycle_count, core.tsbg.row_access_count,
                        core.tsbg.cache_hit_count, core.tsbg.cache_miss_count,
                        core.tsbg.cache_eviction_count,
                        core.tsbg.weight_bundle_beat_count,
                        core.tsbg.scalar_bank_request_count,
                        core.tsbg.scalar_bank_response_count,
                        core.tsbg.issue_count, core.tsbg.product_count,
                        core.tsbg.commit_count}))
                    $fatal(1, "M2105 TSBG DUT public X/Z");
                if (core.tsbg.protocol_error || core.tsbg.numeric_overflow)
                    $fatal(1, "M2105 TSBG DUT fault in window");
            end
        end
    endtask

    task automatic check_completion;
        integer measured_cycles;
        begin
            if (axis_ordinary) begin
                measured_cycles = core.base_done_cycle
                    - core.full_execute_start_cycle;
                if (measured_cycles != FROZEN_BASE_CYCLES
                        || core.base.row_access_count != FROZEN_ROWS
                        || core.base.issue_count != FROZEN_ISSUES
                        || core.base.product_count != FROZEN_PRODUCTS
                        || core.base.commit_count != FROZEN_COMMITS
                        || core.base.cache_miss_count != 149
                        || core.base.cache_hit_count != 0
                        || core.base.cache_eviction_count != 145
                        || core.base.weight_bundle_beat_count
                            != FROZEN_BASE_BUNDLES
                        || core.base.scalar_bank_request_count
                            != FROZEN_BASE_SCALAR
                        || core.base.scalar_bank_response_count
                            != FROZEN_BASE_SCALAR)
                    $fatal(1, "M2105 ordinary completion ledger drift");
            end else begin
                measured_cycles = core.tsbg_done_cycle
                    - core.full_execute_start_cycle;
                if (measured_cycles != FROZEN_TSBG_CYCLES
                        || core.tsbg.row_access_count != FROZEN_ROWS
                        || core.tsbg.issue_count != FROZEN_ISSUES
                        || core.tsbg.product_count != FROZEN_PRODUCTS
                        || core.tsbg.commit_count != FROZEN_COMMITS
                        || core.tsbg.cache_miss_count != 48
                        || core.tsbg.cache_hit_count != 101
                        || core.tsbg.cache_eviction_count != 44
                        || core.tsbg.weight_bundle_beat_count
                            != FROZEN_TSBG_BUNDLES
                        || core.tsbg.scalar_bank_request_count
                            != FROZEN_TSBG_SCALAR
                        || core.tsbg.scalar_bank_response_count
                            != FROZEN_TSBG_SCALAR)
                    $fatal(1, "M2105 TSBG completion ledger drift");
            end
        end
    endtask

    always @(posedge core.clk_core) begin
        if (measurement_window_active) begin
            #0.01;
            check_selected_known();
        end
    end

    initial begin : m2105_fixed_window
        check_axis_selection();
        wait (core.full_execute_start_cycle >= 0);
        #0.01;
        check_workload_identity();
        check_selected_known();
        measurement_window_active = 1'b1;
        $display("M2105_RTL_SAIF_WINDOW_BEGIN global_slot=42 sample=0 layer=28 is_fc2=0 token_start=0 source_groups=48 preload_cycles=383");
        $stop;
        if (axis_ordinary)
            wait (core.base_done_cycle >= 0);
        else
            wait (core.tsbg_done_cycle >= 0);
        #0.01;
        check_completion();
        check_selected_known();
        measurement_window_active = 1'b0;
        if (axis_ordinary)
            $display("M2105_RTL_SAIF_WINDOW_END axis=ordinary_lru4 measurement_cycles=20292 scalar_weight_reads=14304");
        else
            $display("M2105_RTL_SAIF_WINDOW_END axis=tsbg_b4 measurement_cycles=7569 scalar_weight_reads=4608");
        $stop;
    end
endmodule

`default_nettype wire
