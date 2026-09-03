`timescale 1ns/1ps
`default_nettype none

// M2056 is a two-stop mapped-energy wrapper around the frozen M2051
// scoreboard.  It locks global workload slot 42, which is M2047 semantic
// anchor slot 0: sample 0 / layer 28 / FC1 / token start 0 / G48.  The first
// stop is after the exact 383-cycle descriptor preload and before useful
// execution.  The second stop is at the selected mapped axis completion.  A
// third UCLI run then lets M2051 execute its attacks, recovery, and final PASS.
module tb_m2056_m2018_tsbg_matched_mapped_energy;
    localparam int FROZEN_WORKLOAD_SLOT = 42;
    localparam int FROZEN_PRELOAD_CYCLES = 383;
    localparam int FROZEN_ROWS = 149;
    localparam int FROZEN_ISSUES = 1278;
    localparam int FROZEN_PRODUCTS = 29472;
    localparam int FROZEN_BASE_BUNDLES = 1788;
    localparam int FROZEN_TSBG_BUNDLES = 576;
    localparam int FROZEN_BASE_SCALAR = 14304;
    localparam int FROZEN_TSBG_SCALAR = 4608;
    localparam int FROZEN_BASE_CYCLES = 20292;
    localparam int FROZEN_TSBG_CYCLES = 7569;

    logic measurement_window_active = 1'b0;
    tb_m2051_ep34_tsbg_full40_cycle core();

    task automatic check_frozen_workload_identity;
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
                $fatal(1, "M2056 frozen global-slot42 workload identity drift");
            if (core.full_execute_start_cycle != FROZEN_PRELOAD_CYCLES)
                $fatal(1, "M2056 descriptor preload is not exactly 383 cycles");
        end
    endtask

    task automatic check_selected_public_known;
        begin
`ifdef M2056_AXIS_ORDINARY
            if ($isunknown({core.clk_core, core.rst_core,
                    core.load_valid_base, core.load_context, core.load_tag,
                    core.load_group, core.load_source_active,
                    core.load_source_sign, core.load_last,
                    core.base.load_ready, core.base.load_accept}))
                $fatal(1, "M2056 ordinary mapped load/reset X/Z");
            if ($isunknown({
                    core.base.mem_req_valid, core.base.mem_req_ready,
                    core.base.mem_req_accept, core.base.mem_rsp_valid,
                    core.base.mem_rsp_ready, core.base.mem_rsp_accept}))
                $fatal(1, "M2056 ordinary mapped memory handshake X/Z");
            if ($isunknown({
                    core.base.bridge_valid, core.base.bridge_ready,
                    core.base.bridge_context, core.base.bridge_group,
                    core.base.bridge_half, core.base.bridge_slice,
                    core.base.bridge_bank_valid, core.base.bridge_accept,
                    core.base.commit_valid, core.base.commit_ready,
                    core.base.commit_context, core.base.commit_tag,
                    core.base.commit_slice, core.base.commit_terminal,
                    core.base.commit_accept, core.base.bundle_done_valid,
                    core.base.bundle_done_ready, core.base.protocol_error,
                    core.base.stale_response_seen, core.base.numeric_overflow,
                    core.base.busy}))
                $fatal(1, "M2056 ordinary mapped bridge/commit/control X/Z");
            if ($isunknown({core.base.cycle_count,
                    core.base.row_access_count, core.base.cache_hit_count,
                    core.base.cache_miss_count, core.base.cache_eviction_count,
                    core.base.weight_bundle_beat_count,
                    core.base.scalar_bank_request_count,
                    core.base.scalar_bank_response_count,
                    core.base.issue_count, core.base.product_count,
                    core.base.commit_count}))
                $fatal(1, "M2056 ordinary mapped counter X/Z");
            for (int bank = 0; bank < 8; bank++) begin
                if ($isunknown({core.base.mem_req_epoch[bank],
                        core.base.mem_req_slot[bank],
                        core.base.mem_req_generation[bank],
                        core.base.mem_req_tag[bank],
                        core.base.mem_req_output_block[bank],
                        core.base.mem_req_slice[bank],
                        core.base.mem_req_source_channel[bank],
                        core.base.mem_rsp_epoch[bank],
                        core.base.mem_rsp_slot[bank],
                        core.base.mem_rsp_generation[bank],
                        core.base.mem_rsp_tag[bank],
                        core.base.bridge_source_channel[bank],
                        core.base.bridge_source_value[bank]}))
                    $fatal(1, "M2056 ordinary mapped bank metadata X/Z bank=%0d", bank);
                for (int lane = 0; lane < 16; lane++) begin
                    if ($isunknown({core.base.mem_rsp_weight[bank][lane],
                            core.base.bridge_effective_weight[bank][lane]}))
                        $fatal(1, "M2056 ordinary mapped payload X/Z bank=%0d lane=%0d",
                            bank, lane);
                end
            end
            for (int lane = 0; lane < 16; lane++) begin
                if ($isunknown(core.base.commit_accumulator[lane]))
                    $fatal(1, "M2056 ordinary mapped accumulator X/Z lane=%0d", lane);
            end
            if (core.base.protocol_error || core.base.stale_response_seen
                    || core.base.numeric_overflow)
                $fatal(1, "M2056 ordinary mapped fault in clean SAIF window");
`elsif M2056_AXIS_TSBG
            if ($isunknown({core.clk_core, core.rst_core,
                    core.load_valid_tsbg, core.load_context, core.load_tag,
                    core.load_group, core.load_source_active,
                    core.load_source_sign, core.load_last,
                    core.tsbg.load_ready, core.tsbg.load_accept}))
                $fatal(1, "M2056 TSBG mapped load/reset X/Z");
            if ($isunknown({
                    core.tsbg.mem_req_valid, core.tsbg.mem_req_ready,
                    core.tsbg.mem_req_accept, core.tsbg.mem_rsp_valid,
                    core.tsbg.mem_rsp_ready, core.tsbg.mem_rsp_accept}))
                $fatal(1, "M2056 TSBG mapped memory handshake X/Z");
            if ($isunknown({
                    core.tsbg.bridge_valid, core.tsbg.bridge_ready,
                    core.tsbg.bridge_context, core.tsbg.bridge_group,
                    core.tsbg.bridge_half, core.tsbg.bridge_slice,
                    core.tsbg.bridge_bank_valid, core.tsbg.bridge_accept,
                    core.tsbg.commit_valid, core.tsbg.commit_ready,
                    core.tsbg.commit_context, core.tsbg.commit_tag,
                    core.tsbg.commit_slice, core.tsbg.commit_terminal,
                    core.tsbg.commit_accept, core.tsbg.bundle_done_valid,
                    core.tsbg.bundle_done_ready, core.tsbg.protocol_error,
                    core.tsbg.stale_response_seen, core.tsbg.numeric_overflow,
                    core.tsbg.busy}))
                $fatal(1, "M2056 TSBG mapped bridge/commit/control X/Z");
            if ($isunknown({core.tsbg.cycle_count,
                    core.tsbg.row_access_count, core.tsbg.cache_hit_count,
                    core.tsbg.cache_miss_count, core.tsbg.cache_eviction_count,
                    core.tsbg.weight_bundle_beat_count,
                    core.tsbg.scalar_bank_request_count,
                    core.tsbg.scalar_bank_response_count,
                    core.tsbg.issue_count, core.tsbg.product_count,
                    core.tsbg.commit_count}))
                $fatal(1, "M2056 TSBG mapped counter X/Z");
            for (int bank = 0; bank < 8; bank++) begin
                if ($isunknown({core.tsbg.mem_req_epoch[bank],
                        core.tsbg.mem_req_slot[bank],
                        core.tsbg.mem_req_generation[bank],
                        core.tsbg.mem_req_tag[bank],
                        core.tsbg.mem_req_output_block[bank],
                        core.tsbg.mem_req_slice[bank],
                        core.tsbg.mem_req_source_channel[bank],
                        core.tsbg.mem_rsp_epoch[bank],
                        core.tsbg.mem_rsp_slot[bank],
                        core.tsbg.mem_rsp_generation[bank],
                        core.tsbg.mem_rsp_tag[bank],
                        core.tsbg.bridge_source_channel[bank],
                        core.tsbg.bridge_source_value[bank]}))
                    $fatal(1, "M2056 TSBG mapped bank metadata X/Z bank=%0d", bank);
                for (int lane = 0; lane < 16; lane++) begin
                    if ($isunknown({core.tsbg.mem_rsp_weight[bank][lane],
                            core.tsbg.bridge_effective_weight[bank][lane]}))
                        $fatal(1, "M2056 TSBG mapped payload X/Z bank=%0d lane=%0d",
                            bank, lane);
                end
            end
            for (int lane = 0; lane < 16; lane++) begin
                if ($isunknown(core.tsbg.commit_accumulator[lane]))
                    $fatal(1, "M2056 TSBG mapped accumulator X/Z lane=%0d", lane);
            end
            if (core.tsbg.protocol_error || core.tsbg.stale_response_seen
                    || core.tsbg.numeric_overflow)
                $fatal(1, "M2056 TSBG mapped fault in clean SAIF window");
`endif
        end
    endtask

    task automatic check_selected_completion;
        integer measured_cycles;
        begin
`ifdef M2056_AXIS_ORDINARY
            measured_cycles = core.base_done_cycle
                - core.full_execute_start_cycle;
            if (measured_cycles != FROZEN_BASE_CYCLES
                    || core.base.row_access_count != FROZEN_ROWS
                    || core.base.issue_count != FROZEN_ISSUES
                    || core.base.product_count != FROZEN_PRODUCTS
                    || core.base.cache_miss_count != 149
                    || core.base.cache_hit_count != 0
                    || core.base.cache_eviction_count != 145
                    || core.base.weight_bundle_beat_count != FROZEN_BASE_BUNDLES
                    || core.base.scalar_bank_request_count != FROZEN_BASE_SCALAR
                    || core.base.scalar_bank_response_count != FROZEN_BASE_SCALAR)
                $fatal(1, "M2056 ordinary mapped completion ledger drift");
`elsif M2056_AXIS_TSBG
            measured_cycles = core.tsbg_done_cycle
                - core.full_execute_start_cycle;
            if (measured_cycles != FROZEN_TSBG_CYCLES
                    || core.tsbg.row_access_count != FROZEN_ROWS
                    || core.tsbg.issue_count != FROZEN_ISSUES
                    || core.tsbg.product_count != FROZEN_PRODUCTS
                    || core.tsbg.cache_miss_count != 48
                    || core.tsbg.cache_hit_count != 101
                    || core.tsbg.cache_eviction_count != 44
                    || core.tsbg.weight_bundle_beat_count != FROZEN_TSBG_BUNDLES
                    || core.tsbg.scalar_bank_request_count != FROZEN_TSBG_SCALAR
                    || core.tsbg.scalar_bank_response_count != FROZEN_TSBG_SCALAR)
                $fatal(1, "M2056 TSBG mapped completion ledger drift");
`endif
        end
    endtask

    // This is deliberately active at every measurement-window rising edge;
    // endpoint-only X/Z checks are insufficient for a production gate SAIF.
    always @(posedge core.clk_core) begin : mapped_window_continuous_xz_monitor
        if (measurement_window_active) begin
            #0.01;
            check_selected_public_known();
        end
    end

    initial begin : m2056_two_stop_window
        wait (core.full_execute_start_cycle >= 0);
        #0.01;
        check_frozen_workload_identity();
        check_selected_public_known();
        measurement_window_active = 1'b1;
        $display("M2056_SAIF_WINDOW_BEGIN global_slot=42 m2047_anchor_slot=0 sample=0 layer=28 is_fc2=0 token_start=0 source_groups=48 preload_cycles=383");
        $stop;
`ifdef M2056_AXIS_ORDINARY
        wait (core.base_done_cycle >= 0);
`elsif M2056_AXIS_TSBG
        wait (core.tsbg_done_cycle >= 0);
`endif
        #0.01;
        check_selected_completion();
        check_selected_public_known();
        measurement_window_active = 1'b0;
`ifdef M2056_AXIS_ORDINARY
        $display("M2056_SAIF_WINDOW_END axis=ordinary_lru4 global_slot=42 measurement_cycles=20292");
`elsif M2056_AXIS_TSBG
        $display("M2056_SAIF_WINDOW_END axis=tsbg_b4 global_slot=42 measurement_cycles=7569");
`endif
        $stop;
    end
endmodule

`default_nettype wire
