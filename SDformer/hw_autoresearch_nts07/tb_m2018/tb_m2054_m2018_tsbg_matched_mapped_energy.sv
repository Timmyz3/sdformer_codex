`timescale 1ns/1ps
`default_nettype none

// Two-stop wrapper around the frozen M2051 slot-0 scoreboard.  The first stop
// occurs after both sides have loaded the identical 383 descriptor cycles and
// immediately before useful execution.  The second stop follows completion of
// the selected mapped axis.  UCLI records only the mapped child between them;
// it then resumes the M2051 test to its functional PASS token.
module tb_m2054_m2018_tsbg_matched_mapped_energy;
    tb_m2051_ep34_tsbg_full40_cycle core();

    task automatic check_selected_public_known;
        begin
`ifdef M2054_AXIS_ORDINARY
            if ($isunknown({core.base.load_ready, core.base.load_accept,
                    core.base.mem_req_valid, core.base.mem_req_ready,
                    core.base.mem_req_accept, core.base.mem_rsp_valid,
                    core.base.mem_rsp_ready, core.base.mem_rsp_accept,
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
                    core.base.busy, core.base.cycle_count,
                    core.base.row_access_count, core.base.cache_hit_count,
                    core.base.cache_miss_count, core.base.cache_eviction_count,
                    core.base.weight_bundle_beat_count,
                    core.base.scalar_bank_request_count,
                    core.base.scalar_bank_response_count,
                    core.base.issue_count, core.base.product_count,
                    core.base.commit_count}))
                $fatal(1, "M2054 ordinary public scalar/vector X/Z before SAIF");
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
                    $fatal(1, "M2054 ordinary bank metadata X/Z before SAIF");
                for (int lane = 0; lane < 16; lane++) begin
                    if ($isunknown({core.base.mem_rsp_weight[bank][lane],
                            core.base.bridge_effective_weight[bank][lane]}))
                        $fatal(1, "M2054 ordinary bank payload X/Z before SAIF");
                end
            end
            for (int lane = 0; lane < 16; lane++)
                if ($isunknown(core.base.commit_accumulator[lane]))
                    $fatal(1, "M2054 ordinary accumulator X/Z before SAIF");
            if (core.base.protocol_error || core.base.stale_response_seen
                    || core.base.numeric_overflow)
                $fatal(1, "M2054 ordinary fault before SAIF");
`elsif M2054_AXIS_TSBG
            if ($isunknown({core.tsbg.load_ready, core.tsbg.load_accept,
                    core.tsbg.mem_req_valid, core.tsbg.mem_req_ready,
                    core.tsbg.mem_req_accept, core.tsbg.mem_rsp_valid,
                    core.tsbg.mem_rsp_ready, core.tsbg.mem_rsp_accept,
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
                    core.tsbg.busy, core.tsbg.cycle_count,
                    core.tsbg.row_access_count, core.tsbg.cache_hit_count,
                    core.tsbg.cache_miss_count, core.tsbg.cache_eviction_count,
                    core.tsbg.weight_bundle_beat_count,
                    core.tsbg.scalar_bank_request_count,
                    core.tsbg.scalar_bank_response_count,
                    core.tsbg.issue_count, core.tsbg.product_count,
                    core.tsbg.commit_count}))
                $fatal(1, "M2054 TSBG public scalar/vector X/Z before SAIF");
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
                    $fatal(1, "M2054 TSBG bank metadata X/Z before SAIF");
                for (int lane = 0; lane < 16; lane++) begin
                    if ($isunknown({core.tsbg.mem_rsp_weight[bank][lane],
                            core.tsbg.bridge_effective_weight[bank][lane]}))
                        $fatal(1, "M2054 TSBG bank payload X/Z before SAIF");
                end
            end
            for (int lane = 0; lane < 16; lane++)
                if ($isunknown(core.tsbg.commit_accumulator[lane]))
                    $fatal(1, "M2054 TSBG accumulator X/Z before SAIF");
            if (core.tsbg.protocol_error || core.tsbg.stale_response_seen
                    || core.tsbg.numeric_overflow)
                $fatal(1, "M2054 TSBG fault before SAIF");
`endif
        end
    endtask

    initial begin : m2054_two_stop_window
        wait (core.full_execute_start_cycle >= 0);
        #0.01;
        check_selected_public_known();
        $display("M2054_SAIF_WINDOW_BEGIN workload_slot=0 preload_cycles=383");
        $stop;
`ifdef M2054_AXIS_ORDINARY
        wait (core.base_done_cycle >= 0);
        if (core.full_base_exec_cycles <= 0 && core.base_done_cycle <= 0)
            $fatal(1, "M2054 ordinary completion invalid");
`elsif M2054_AXIS_TSBG
        wait (core.tsbg_done_cycle >= 0);
        if (core.full_tsbg_exec_cycles <= 0 && core.tsbg_done_cycle <= 0)
            $fatal(1, "M2054 TSBG completion invalid");
`endif
        check_selected_public_known();
        $display("M2054_SAIF_WINDOW_END workload_slot=0");
        $stop;
    end
endmodule

`default_nettype wire
