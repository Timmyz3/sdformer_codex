`timescale 1ns/1ps
`default_nettype none

module m1332_c2_production_activity_assertions (
    input logic clk_core, input logic rst_core,
    input logic header_valid, input logic header_ready,
    input logic header_accept, input logic raw_valid,
    input logic raw_ready, input logic raw_accept,
    input logic [3:0] raw_lane_valid, input logic [19:0] raw_beat_index,
    input logic [383:0] raw_bitmap, input logic raw_last,
    input logic [7:0] mem_req_valid, input logic [7:0] mem_req_ready,
    input logic [7:0] mem_req_accept, input logic [127:0] mem_req_epoch,
    input logic [23:0] mem_req_slot,
    input logic [255:0] mem_req_generation,
    input logic [191:0] mem_req_tag,
    input logic [23:0] mem_req_output_block,
    input logic [23:0] mem_req_slice,
    input logic [95:0] mem_req_source_channel,
    input logic [7:0] mem_rsp_valid, input logic [7:0] mem_rsp_ready,
    input logic [7:0] mem_rsp_accept, input logic [127:0] mem_rsp_epoch,
    input logic [23:0] mem_rsp_slot,
    input logic [255:0] mem_rsp_generation,
    input logic [191:0] mem_rsp_tag,
    input logic [1023:0] mem_rsp_weight,
    input logic result_valid, input logic result_ready,
    input logic result_accept, input logic [23:0] result_tag,
    input logic [2:0] result_output_block, input logic [2:0] result_slice,
    input logic [383:0] result_accumulator, input logic result_last,
    input logic token_done_valid, input logic token_done_ready,
    input logic token_done_accept, input logic [23:0] token_done_tag,
    input logic token_done_had_event,
    input logic protocol_error, input logic numeric_overflow,
    input logic stale_response_seen, input logic [7:0] endpoint_fault
);
    integer case_id;
    integer header_count, source_count, endpoint_count, commit_count;
    integer stall_count, done_count, unknown_count, fault_count;
    logic check_pending;

    initial begin
        if (!$value$plusargs("M979_CASE=%d", case_id))
            $fatal(1, "M1332 requires M979_CASE");
    end

`ifdef SVA_RUNTIME_ENABLED
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_header_accept_exact:
        assert property (header_accept == (header_valid && header_ready));
    ap_raw_accept_exact:
        assert property (raw_accept == (raw_valid && raw_ready));
    ap_request_accept_exact:
        assert property (mem_req_accept == (mem_req_valid & mem_req_ready));
    ap_response_accept_exact:
        assert property (mem_rsp_accept == (mem_rsp_valid & mem_rsp_ready));
    ap_result_accept_exact:
        assert property (result_accept == (result_valid && result_ready));
    ap_done_accept_exact:
        assert property (token_done_accept
            == (token_done_valid && token_done_ready));
    ap_raw_payload_known:
        assert property (raw_valid |-> !$isunknown({raw_lane_valid,
            raw_beat_index, raw_bitmap, raw_last}));
    ap_request_payload_known:
        assert property ((|mem_req_valid) |-> !$isunknown({mem_req_epoch,
            mem_req_slot, mem_req_generation, mem_req_tag,
            mem_req_output_block, mem_req_slice, mem_req_source_channel}));
    ap_response_payload_known:
        assert property ((|mem_rsp_valid) |-> !$isunknown({mem_rsp_epoch,
            mem_rsp_slot, mem_rsp_generation, mem_rsp_tag, mem_rsp_weight}));
    ap_result_payload_known:
        assert property (result_valid |-> !$isunknown({result_tag,
            result_output_block, result_slice, result_accumulator,
            result_last}));
    ap_done_payload_known:
        assert property (token_done_valid |-> !$isunknown({token_done_tag,
            token_done_had_event}));
    ap_result_stable_under_stall:
        assert property (result_valid && !result_ready
            |=> result_valid && $stable({result_tag, result_output_block,
                result_slice, result_accumulator, result_last}));
    ap_done_stable_under_stall:
        assert property (token_done_valid && !token_done_ready
            |=> token_done_valid && $stable({token_done_tag,
                token_done_had_event}));
    ap_no_endpoint_fault:
        assert property (!(|endpoint_fault));
    ap_no_protocol_fault:
        assert property (!(protocol_error || numeric_overflow
            || stale_response_seen));

    cp_source: cover property (raw_accept);
    cp_endpoint: cover property (|mem_req_accept);
    cp_commit: cover property (result_accept);
    cp_stall: cover property ((raw_valid && !raw_ready)
        || (|(mem_req_valid & ~mem_req_ready))
        || (|(mem_rsp_valid & ~mem_rsp_ready))
        || (result_valid && !result_ready)
        || (token_done_valid && !token_done_ready));
    cp_done: cover property (token_done_accept);
`endif

    always @(posedge clk_core) begin
        if (rst_core) begin
            header_count <= 0;
            source_count <= 0;
            endpoint_count <= 0;
            commit_count <= 0;
            stall_count <= 0;
            done_count <= 0;
            unknown_count <= 0;
            fault_count <= 0;
            check_pending <= 1'b0;
        end else begin
            if ($isunknown({header_valid, header_ready, header_accept,
                    raw_valid, raw_ready, raw_accept, mem_req_valid,
                    mem_req_ready, mem_req_accept, mem_rsp_valid,
                    mem_rsp_ready, mem_rsp_accept, result_valid,
                    result_ready, result_accept, token_done_valid,
                    token_done_ready, token_done_accept}))
                unknown_count <= unknown_count + 1;
            if (|endpoint_fault || protocol_error || numeric_overflow
                    || stale_response_seen)
                fault_count <= fault_count + 1;
            if (header_accept) header_count <= header_count + 1;
            if (raw_accept) source_count <= source_count + 1;
            endpoint_count <= endpoint_count + $countones(mem_req_accept);
            if (result_accept) commit_count <= commit_count + 1;
            if ((raw_valid && !raw_ready)
                    || (|(mem_req_valid & ~mem_req_ready))
                    || (|(mem_rsp_valid & ~mem_rsp_ready))
                    || (result_valid && !result_ready)
                    || (token_done_valid && !token_done_ready))
                stall_count <= stall_count + 1;
            if (token_done_accept) begin
                done_count <= done_count + 1;
                check_pending <= 1'b1;
            end
        end
    end

    always @(negedge clk_core) begin
        if (!rst_core && check_pending) begin
            if (header_count != 1 || source_count == 0 || commit_count == 0
                    || stall_count == 0 || done_count != 1
                    || unknown_count != 0 || fault_count != 0)
                $fatal(1, "M1332 coverage/fault gate failed case=%0d header=%0d source=%0d endpoint=%0d commit=%0d stall=%0d done=%0d unknown=%0d fault=%0d",
                    case_id, header_count, source_count, endpoint_count,
                    commit_count, stall_count, done_count, unknown_count,
                    fault_count);
            if (case_id < 4 && endpoint_count == 0)
                $fatal(1, "M1332 nonzero case lacks endpoint activity");
            if (case_id == 4 && endpoint_count != 0)
                $fatal(1, "M1332 zero-event case manufactured endpoint activity");
            $display("PASS M1332 coverage case=%0d source=%0d endpoint=%0d commit=%0d stall=%0d done=%0d unknown=0 fault=0",
                case_id, source_count, endpoint_count, commit_count,
                stall_count, done_count);
            check_pending <= 1'b0;
        end
    end

    initial begin
        #1000000;
        $fatal(1, "M1332 assertion absolute watchdog");
    end
endmodule

`default_nettype wire
