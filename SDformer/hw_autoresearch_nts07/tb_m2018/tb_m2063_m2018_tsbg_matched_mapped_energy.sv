`timescale 1ns/1ps
`default_nettype none

// Unsealed M2063 source-only draft.  This additive successor preserves the M2056 slot42
// identity and two-stop window.  All reset releases live in the M2063 base TB;
// no force/release seam exists here.  The runner exact-pins compile-time
// +vcs+initreg+random and runtime +vcs+initreg+0 only for zero-delay
// X-pessimism, not as silicon power-on or delay behavior.  Mapped
// functional outputs are observed at a settled negedge; qualifier/control/
// fault/counter signals are checked unconditionally, while payload sidebands
// are checked only under their owning valid.  This is not a gate-delay fix.
module tb_m2063_m2018_tsbg_matched_mapped_energy;
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
    tb_m2063_ep34_tsbg_full40_cycle core();

    task automatic require_known(
            input logic [4095:0] value,
            input string signal_name);
        begin
            if ($isunknown(value))
                $fatal(1, "M2063 mapped X/Z signal=%s", signal_name);
        end
    endtask

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
                $fatal(1, "M2063 frozen global-slot42 workload identity drift");
            if (core.full_execute_start_cycle != FROZEN_PRELOAD_CYCLES)
                $fatal(1, "M2063 descriptor preload is not exactly 383 cycles");
        end
    endtask

    task automatic check_selected_settled_known;
        begin
`ifdef M2056_AXIS_ORDINARY
            require_known(core.clk_core, "ordinary.clk_core");
            require_known(core.rst_core, "ordinary.rst_core");
            require_known(core.load_valid_base, "ordinary.load_valid");
            require_known(core.base.load_ready, "ordinary.load_ready");
            require_known(core.base.load_accept, "ordinary.load_accept");
            require_known(core.base.mem_req_valid, "ordinary.mem_req_valid");
            require_known(core.base.mem_req_ready, "ordinary.mem_req_ready");
            require_known(core.base.mem_req_accept, "ordinary.mem_req_accept");
            require_known(core.base.mem_rsp_valid, "ordinary.mem_rsp_valid");
            require_known(core.base.mem_rsp_ready, "ordinary.mem_rsp_ready");
            require_known(core.base.mem_rsp_accept, "ordinary.mem_rsp_accept");
            require_known(core.base.bridge_valid, "ordinary.bridge_valid");
            require_known(core.base.bridge_ready, "ordinary.bridge_ready");
            require_known(core.base.bridge_accept, "ordinary.bridge_accept");
            require_known(core.base.commit_valid, "ordinary.commit_valid");
            require_known(core.base.commit_ready, "ordinary.commit_ready");
            require_known(core.base.commit_accept, "ordinary.commit_accept");
            require_known(core.base.bundle_done_valid, "ordinary.bundle_done_valid");
            require_known(core.base.bundle_done_ready, "ordinary.bundle_done_ready");
            require_known(core.base.protocol_error, "ordinary.protocol_error");
            require_known(core.base.stale_response_seen, "ordinary.stale_response_seen");
            require_known(core.base.numeric_overflow, "ordinary.numeric_overflow");
            require_known(core.base.busy, "ordinary.busy");
            require_known(core.base.cycle_count, "ordinary.cycle_count");
            require_known(core.base.row_access_count, "ordinary.row_access_count");
            require_known(core.base.cache_hit_count, "ordinary.cache_hit_count");
            require_known(core.base.cache_miss_count, "ordinary.cache_miss_count");
            require_known(core.base.cache_eviction_count, "ordinary.cache_eviction_count");
            require_known(core.base.weight_bundle_beat_count, "ordinary.weight_bundle_beat_count");
            require_known(core.base.scalar_bank_request_count, "ordinary.scalar_bank_request_count");
            require_known(core.base.scalar_bank_response_count, "ordinary.scalar_bank_response_count");
            require_known(core.base.issue_count, "ordinary.issue_count");
            require_known(core.base.product_count, "ordinary.product_count");
            require_known(core.base.commit_count, "ordinary.commit_count");
            if (core.load_valid_base) begin
                require_known(core.load_context, "ordinary.load_context");
                require_known(core.load_tag, "ordinary.load_tag");
                require_known(core.load_group, "ordinary.load_group");
                require_known(core.load_source_active, "ordinary.load_source_active");
                require_known(core.load_source_sign, "ordinary.load_source_sign");
                require_known(core.load_last, "ordinary.load_last");
            end
            for (int bank = 0; bank < 8; bank++) begin
                if (core.base.mem_req_valid[bank]) begin
                    require_known(core.base.mem_req_epoch[bank], $sformatf("ordinary.mem_req_epoch[%0d]", bank));
                    require_known(core.base.mem_req_slot[bank], $sformatf("ordinary.mem_req_slot[%0d]", bank));
                    require_known(core.base.mem_req_generation[bank], $sformatf("ordinary.mem_req_generation[%0d]", bank));
                    require_known(core.base.mem_req_tag[bank], $sformatf("ordinary.mem_req_tag[%0d]", bank));
                    require_known(core.base.mem_req_output_block[bank], $sformatf("ordinary.mem_req_output_block[%0d]", bank));
                    require_known(core.base.mem_req_slice[bank], $sformatf("ordinary.mem_req_slice[%0d]", bank));
                    require_known(core.base.mem_req_source_channel[bank], $sformatf("ordinary.mem_req_source_channel[%0d]", bank));
                end
                if (core.base.mem_rsp_valid[bank]) begin
                    require_known(core.base.mem_rsp_epoch[bank], $sformatf("ordinary.mem_rsp_epoch[%0d]", bank));
                    require_known(core.base.mem_rsp_slot[bank], $sformatf("ordinary.mem_rsp_slot[%0d]", bank));
                    require_known(core.base.mem_rsp_generation[bank], $sformatf("ordinary.mem_rsp_generation[%0d]", bank));
                    require_known(core.base.mem_rsp_tag[bank], $sformatf("ordinary.mem_rsp_tag[%0d]", bank));
                    for (int lane = 0; lane < 16; lane++) begin
                        require_known(core.base.mem_rsp_weight[bank][lane],
                            $sformatf("ordinary.mem_rsp_weight[%0d][%0d]", bank, lane));
                    end
                end
                if (core.base.bridge_valid && core.base.bridge_bank_valid[bank]) begin
                    require_known(core.base.bridge_source_channel[bank],
                        $sformatf("ordinary.bridge_source_channel[%0d]", bank));
                    require_known(core.base.bridge_source_value[bank],
                        $sformatf("ordinary.bridge_source_value[%0d]", bank));
                    for (int lane = 0; lane < 16; lane++) begin
                        require_known(core.base.bridge_effective_weight[bank][lane],
                            $sformatf("ordinary.bridge_effective_weight[%0d][%0d]", bank, lane));
                    end
                end
            end
            if (core.base.bridge_valid) begin
                require_known(core.base.bridge_bank_valid, "ordinary.bridge_bank_valid");
                require_known(core.base.bridge_context, "ordinary.bridge_context");
                require_known(core.base.bridge_group, "ordinary.bridge_group");
                require_known(core.base.bridge_half, "ordinary.bridge_half");
                require_known(core.base.bridge_slice, "ordinary.bridge_slice");
            end
            if (core.base.commit_valid) begin
                require_known(core.base.commit_context, "ordinary.commit_context");
                require_known(core.base.commit_tag, "ordinary.commit_tag");
                require_known(core.base.commit_slice, "ordinary.commit_slice");
                require_known(core.base.commit_terminal, "ordinary.commit_terminal");
                for (int lane = 0; lane < 16; lane++) begin
                    require_known(core.base.commit_accumulator[lane],
                        $sformatf("ordinary.commit_accumulator[%0d]", lane));
                end
            end
            if (core.base.protocol_error)
                $fatal(1, "M2063 ordinary protocol_error in clean SAIF window");
            if (core.base.stale_response_seen)
                $fatal(1, "M2063 ordinary stale_response_seen in clean SAIF window");
            if (core.base.numeric_overflow)
                $fatal(1, "M2063 ordinary numeric_overflow in clean SAIF window");
`elsif M2056_AXIS_TSBG
            require_known(core.clk_core, "tsbg.clk_core");
            require_known(core.rst_core, "tsbg.rst_core");
            require_known(core.load_valid_tsbg, "tsbg.load_valid");
            require_known(core.tsbg.load_ready, "tsbg.load_ready");
            require_known(core.tsbg.load_accept, "tsbg.load_accept");
            require_known(core.tsbg.mem_req_valid, "tsbg.mem_req_valid");
            require_known(core.tsbg.mem_req_ready, "tsbg.mem_req_ready");
            require_known(core.tsbg.mem_req_accept, "tsbg.mem_req_accept");
            require_known(core.tsbg.mem_rsp_valid, "tsbg.mem_rsp_valid");
            require_known(core.tsbg.mem_rsp_ready, "tsbg.mem_rsp_ready");
            require_known(core.tsbg.mem_rsp_accept, "tsbg.mem_rsp_accept");
            require_known(core.tsbg.bridge_valid, "tsbg.bridge_valid");
            require_known(core.tsbg.bridge_ready, "tsbg.bridge_ready");
            require_known(core.tsbg.bridge_accept, "tsbg.bridge_accept");
            require_known(core.tsbg.commit_valid, "tsbg.commit_valid");
            require_known(core.tsbg.commit_ready, "tsbg.commit_ready");
            require_known(core.tsbg.commit_accept, "tsbg.commit_accept");
            require_known(core.tsbg.bundle_done_valid, "tsbg.bundle_done_valid");
            require_known(core.tsbg.bundle_done_ready, "tsbg.bundle_done_ready");
            require_known(core.tsbg.protocol_error, "tsbg.protocol_error");
            require_known(core.tsbg.stale_response_seen, "tsbg.stale_response_seen");
            require_known(core.tsbg.numeric_overflow, "tsbg.numeric_overflow");
            require_known(core.tsbg.busy, "tsbg.busy");
            require_known(core.tsbg.cycle_count, "tsbg.cycle_count");
            require_known(core.tsbg.row_access_count, "tsbg.row_access_count");
            require_known(core.tsbg.cache_hit_count, "tsbg.cache_hit_count");
            require_known(core.tsbg.cache_miss_count, "tsbg.cache_miss_count");
            require_known(core.tsbg.cache_eviction_count, "tsbg.cache_eviction_count");
            require_known(core.tsbg.weight_bundle_beat_count, "tsbg.weight_bundle_beat_count");
            require_known(core.tsbg.scalar_bank_request_count, "tsbg.scalar_bank_request_count");
            require_known(core.tsbg.scalar_bank_response_count, "tsbg.scalar_bank_response_count");
            require_known(core.tsbg.issue_count, "tsbg.issue_count");
            require_known(core.tsbg.product_count, "tsbg.product_count");
            require_known(core.tsbg.commit_count, "tsbg.commit_count");
            if (core.load_valid_tsbg) begin
                require_known(core.load_context, "tsbg.load_context");
                require_known(core.load_tag, "tsbg.load_tag");
                require_known(core.load_group, "tsbg.load_group");
                require_known(core.load_source_active, "tsbg.load_source_active");
                require_known(core.load_source_sign, "tsbg.load_source_sign");
                require_known(core.load_last, "tsbg.load_last");
            end
            for (int bank = 0; bank < 8; bank++) begin
                if (core.tsbg.mem_req_valid[bank]) begin
                    require_known(core.tsbg.mem_req_epoch[bank], $sformatf("tsbg.mem_req_epoch[%0d]", bank));
                    require_known(core.tsbg.mem_req_slot[bank], $sformatf("tsbg.mem_req_slot[%0d]", bank));
                    require_known(core.tsbg.mem_req_generation[bank], $sformatf("tsbg.mem_req_generation[%0d]", bank));
                    require_known(core.tsbg.mem_req_tag[bank], $sformatf("tsbg.mem_req_tag[%0d]", bank));
                    require_known(core.tsbg.mem_req_output_block[bank], $sformatf("tsbg.mem_req_output_block[%0d]", bank));
                    require_known(core.tsbg.mem_req_slice[bank], $sformatf("tsbg.mem_req_slice[%0d]", bank));
                    require_known(core.tsbg.mem_req_source_channel[bank], $sformatf("tsbg.mem_req_source_channel[%0d]", bank));
                end
                if (core.tsbg.mem_rsp_valid[bank]) begin
                    require_known(core.tsbg.mem_rsp_epoch[bank], $sformatf("tsbg.mem_rsp_epoch[%0d]", bank));
                    require_known(core.tsbg.mem_rsp_slot[bank], $sformatf("tsbg.mem_rsp_slot[%0d]", bank));
                    require_known(core.tsbg.mem_rsp_generation[bank], $sformatf("tsbg.mem_rsp_generation[%0d]", bank));
                    require_known(core.tsbg.mem_rsp_tag[bank], $sformatf("tsbg.mem_rsp_tag[%0d]", bank));
                    for (int lane = 0; lane < 16; lane++) begin
                        require_known(core.tsbg.mem_rsp_weight[bank][lane],
                            $sformatf("tsbg.mem_rsp_weight[%0d][%0d]", bank, lane));
                    end
                end
                if (core.tsbg.bridge_valid && core.tsbg.bridge_bank_valid[bank]) begin
                    require_known(core.tsbg.bridge_source_channel[bank],
                        $sformatf("tsbg.bridge_source_channel[%0d]", bank));
                    require_known(core.tsbg.bridge_source_value[bank],
                        $sformatf("tsbg.bridge_source_value[%0d]", bank));
                    for (int lane = 0; lane < 16; lane++) begin
                        require_known(core.tsbg.bridge_effective_weight[bank][lane],
                            $sformatf("tsbg.bridge_effective_weight[%0d][%0d]", bank, lane));
                    end
                end
            end
            if (core.tsbg.bridge_valid) begin
                require_known(core.tsbg.bridge_bank_valid, "tsbg.bridge_bank_valid");
                require_known(core.tsbg.bridge_context, "tsbg.bridge_context");
                require_known(core.tsbg.bridge_group, "tsbg.bridge_group");
                require_known(core.tsbg.bridge_half, "tsbg.bridge_half");
                require_known(core.tsbg.bridge_slice, "tsbg.bridge_slice");
            end
            if (core.tsbg.commit_valid) begin
                require_known(core.tsbg.commit_context, "tsbg.commit_context");
                require_known(core.tsbg.commit_tag, "tsbg.commit_tag");
                require_known(core.tsbg.commit_slice, "tsbg.commit_slice");
                require_known(core.tsbg.commit_terminal, "tsbg.commit_terminal");
                for (int lane = 0; lane < 16; lane++) begin
                    require_known(core.tsbg.commit_accumulator[lane],
                        $sformatf("tsbg.commit_accumulator[%0d]", lane));
                end
            end
            if (core.tsbg.protocol_error)
                $fatal(1, "M2063 TSBG protocol_error in clean SAIF window");
            if (core.tsbg.stale_response_seen)
                $fatal(1, "M2063 TSBG stale_response_seen in clean SAIF window");
            if (core.tsbg.numeric_overflow)
                $fatal(1, "M2063 TSBG numeric_overflow in clean SAIF window");
`endif
        end
    endtask

    task automatic check_selected_completion;
        integer measured_cycles;
        begin
`ifdef M2056_AXIS_ORDINARY
            measured_cycles = core.base_done_cycle - core.full_execute_start_cycle;
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
                $fatal(1, "M2063 ordinary mapped completion ledger drift");
`elsif M2056_AXIS_TSBG
            measured_cycles = core.tsbg_done_cycle - core.full_execute_start_cycle;
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
                $fatal(1, "M2063 TSBG mapped completion ledger drift");
`endif
        end
    endtask

    always @(negedge core.clk_core) begin : mapped_window_settled_xz_monitor
        if (measurement_window_active) begin
            #0.01;
            check_selected_settled_known();
        end
    end

    initial begin : m2063_two_stop_window
        wait (core.full_execute_start_cycle >= 0);
        @(negedge core.clk_core);
        #0.01;
        check_frozen_workload_identity();
        check_selected_settled_known();
        measurement_window_active = 1'b1;
        $display("M2063_SAIF_WINDOW_BEGIN sampling=settled_negedge global_slot=42 m2047_anchor_slot=0 sample=0 layer=28 is_fc2=0 token_start=0 source_groups=48 preload_cycles=383");
        $stop;
`ifdef M2056_AXIS_ORDINARY
        wait (core.base_done_cycle >= 0);
`elsif M2056_AXIS_TSBG
        wait (core.tsbg_done_cycle >= 0);
`endif
        @(negedge core.clk_core);
        #0.01;
        check_selected_completion();
        check_selected_settled_known();
        measurement_window_active = 1'b0;
`ifdef M2056_AXIS_ORDINARY
        $display("M2063_SAIF_WINDOW_END axis=ordinary_lru4 sampling=settled_negedge global_slot=42 measurement_cycles=20292");
`elsif M2056_AXIS_TSBG
        $display("M2063_SAIF_WINDOW_END axis=tsbg_b4 sampling=settled_negedge global_slot=42 measurement_cycles=7569");
`endif
        $stop;
    end
endmodule

`default_nettype wire
