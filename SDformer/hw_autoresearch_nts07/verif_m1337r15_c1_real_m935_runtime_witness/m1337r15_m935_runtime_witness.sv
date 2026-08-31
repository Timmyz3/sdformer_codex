`timescale 1ns/1ps
`default_nettype none

// Additive R15 verification-only witness.  The M528/M935/M1162 design,
// M1168/R3 SVA and M1270/R13 TB are frozen.  Only the two documented
// same-edge groups are legal: first weight+psum request, and each
// response+core accept.  Every later milestone consumes a registered stage.
module m1337r15_m935_runtime_witness (
    input logic clk_core,
    input logic reset_n,
    input logic issue_request_valid,
    input logic issue_request_first,
    input logic issue_request_last,
    input logic issue_request_source_valid,
    input logic [3:0] issue_request_source_index,
    input logic weight_request_fire,
    input logic psum_request_fire,
    input logic response_accept,
    input logic core_accept,
    input logic psum_commit_fire,
    input logic [5:0] psum_commit_address,
    input logic row_complete_fire,
    input logic [5:0] row_complete_id,
    input logic task_done_fire,
    input logic [15:0] task_done_epoch,
    input logic request_hold_attack_mode,
    input logic weight_service_attack_mode,
    input logic psum_service_attack_mode,
    input logic protocol_error,
    input logic boundary_fault,
    input logic core_fault,
    input logic m935_fault,
    input logic weight_service_fault,
    input logic psum_service_fault,
    input logic [63:0] design_issue_accepts,
    input logic [63:0] design_psum_commits,
    input logic [63:0] design_row_completions
);
    localparam logic [3:0] W_RESET          = 4'd0;
    localparam logic [3:0] W_FIRST_REQUEST  = 4'd1;
    localparam logic [3:0] W_FIRST_ACCEPT   = 4'd2;
    localparam logic [3:0] W_SECOND_REQUEST = 4'd3;
    localparam logic [3:0] W_SECOND_ACCEPT  = 4'd4;
    localparam logic [3:0] W_PSUM_COMMIT    = 4'd5;
    localparam logic [3:0] W_ROW_DONE       = 4'd6;
    localparam logic [3:0] W_TASK_DONE      = 4'd7;

    logic [3:0] stage_q;
    logic [3:0] weight_requests_q;
    logic [3:0] psum_requests_q;
    logic [3:0] responses_q;
    logic [3:0] core_accepts_q;
    logic [3:0] psum_commits_q;
    logic [3:0] row_completions_q;
    logic [3:0] task_completions_q;
    logic witness_fault_q;
    logic saw_reset_q;

    wire attack_mask_active = request_hold_attack_mode
        || weight_service_attack_mode || psum_service_attack_mode;
    wire any_design_fault = protocol_error || boundary_fault || core_fault
        || m935_fault || weight_service_fault || psum_service_fault;
    wire any_milestone_event = weight_request_fire || psum_request_fire
        || response_accept || core_accept || psum_commit_fire
        || row_complete_fire || task_done_fire;

    always_ff @(posedge clk_core or negedge reset_n) begin : witness_progress
        logic control_unknown;
        logic identity_unknown;
        if (!reset_n) begin
            stage_q <= W_RESET;
            weight_requests_q <= '0;
            psum_requests_q <= '0;
            responses_q <= '0;
            core_accepts_q <= '0;
            psum_commits_q <= '0;
            row_completions_q <= '0;
            task_completions_q <= '0;
            witness_fault_q <= 1'b0;
            saw_reset_q <= 1'b1;
        end else begin
            control_unknown = $isunknown({weight_request_fire,
                psum_request_fire, response_accept, core_accept,
                psum_commit_fire, row_complete_fire, task_done_fire,
                request_hold_attack_mode, weight_service_attack_mode,
                psum_service_attack_mode, protocol_error, boundary_fault,
                core_fault, m935_fault, weight_service_fault,
                psum_service_fault});
            identity_unknown = 1'b0;
            if ((weight_request_fire === 1'b1)
                    || (psum_request_fire === 1'b1))
                identity_unknown = identity_unknown
                    || $isunknown({issue_request_valid,
                        issue_request_first, issue_request_last,
                        issue_request_source_valid,
                        issue_request_source_index});
            if (psum_commit_fire === 1'b1)
                identity_unknown = identity_unknown
                    || $isunknown(psum_commit_address);
            if (row_complete_fire === 1'b1)
                identity_unknown = identity_unknown
                    || $isunknown(row_complete_id);
            if (task_done_fire === 1'b1)
                identity_unknown = identity_unknown
                    || $isunknown(task_done_epoch);

            if (control_unknown || identity_unknown
                    || $isunknown(stage_q) || $isunknown(saw_reset_q)
                    || $isunknown({design_issue_accepts,
                        design_psum_commits, design_row_completions})) begin
                witness_fault_q <= 1'b1;
            end else begin
                case (stage_q)
                    W_RESET: begin
                        if (any_milestone_event === 1'b1) begin
                            if ((weight_request_fire === 1'b1)
                                    && (psum_request_fire === 1'b1)
                                    && (response_accept === 1'b0)
                                    && (core_accept === 1'b0)
                                    && (psum_commit_fire === 1'b0)
                                    && (row_complete_fire === 1'b0)
                                    && (task_done_fire === 1'b0)
                                    && (issue_request_valid === 1'b1)
                                    && (issue_request_first === 1'b1)
                                    && (issue_request_last === 1'b0)
                                    && (issue_request_source_valid === 1'b1)
                                    && (issue_request_source_index === 4'd0)) begin
                                weight_requests_q <= 4'd1;
                                psum_requests_q <= 4'd1;
                                stage_q <= W_FIRST_REQUEST;
                            end else begin
                                witness_fault_q <= 1'b1;
                            end
                        end
                    end
                    W_FIRST_REQUEST: begin
                        if (any_milestone_event === 1'b1) begin
                            if ((response_accept === 1'b1)
                                    && (core_accept === 1'b1)
                                    && (weight_request_fire === 1'b0)
                                    && (psum_request_fire === 1'b0)
                                    && (psum_commit_fire === 1'b0)
                                    && (row_complete_fire === 1'b0)
                                    && (task_done_fire === 1'b0)) begin
                                responses_q <= 4'd1;
                                core_accepts_q <= 4'd1;
                                stage_q <= W_FIRST_ACCEPT;
                            end else begin
                                witness_fault_q <= 1'b1;
                            end
                        end
                    end
                    W_FIRST_ACCEPT: begin
                        if (any_milestone_event === 1'b1) begin
                            if ((weight_request_fire === 1'b1)
                                    && (psum_request_fire === 1'b0)
                                    && (response_accept === 1'b0)
                                    && (core_accept === 1'b0)
                                    && (psum_commit_fire === 1'b0)
                                    && (row_complete_fire === 1'b0)
                                    && (task_done_fire === 1'b0)
                                    && (issue_request_valid === 1'b1)
                                    && (issue_request_first === 1'b0)
                                    && (issue_request_last === 1'b1)
                                    && (issue_request_source_valid === 1'b1)
                                    && (issue_request_source_index === 4'd1)) begin
                                weight_requests_q <= 4'd2;
                                stage_q <= W_SECOND_REQUEST;
                            end else begin
                                witness_fault_q <= 1'b1;
                            end
                        end
                    end
                    W_SECOND_REQUEST: begin
                        if (any_milestone_event === 1'b1) begin
                            if ((response_accept === 1'b1)
                                    && (core_accept === 1'b1)
                                    && (weight_request_fire === 1'b0)
                                    && (psum_request_fire === 1'b0)
                                    && (psum_commit_fire === 1'b0)
                                    && (row_complete_fire === 1'b0)
                                    && (task_done_fire === 1'b0)) begin
                                responses_q <= 4'd2;
                                core_accepts_q <= 4'd2;
                                stage_q <= W_SECOND_ACCEPT;
                            end else begin
                                witness_fault_q <= 1'b1;
                            end
                        end
                    end
                    W_SECOND_ACCEPT: begin
                        if (any_milestone_event === 1'b1) begin
                            if ((psum_commit_fire === 1'b1)
                                    && (weight_request_fire === 1'b0)
                                    && (psum_request_fire === 1'b0)
                                    && (response_accept === 1'b0)
                                    && (core_accept === 1'b0)
                                    && (row_complete_fire === 1'b0)
                                    && (task_done_fire === 1'b0)
                                    && (psum_commit_address === 6'd0)) begin
                                psum_commits_q <= 4'd1;
                                stage_q <= W_PSUM_COMMIT;
                            end else begin
                                witness_fault_q <= 1'b1;
                            end
                        end
                    end
                    W_PSUM_COMMIT: begin
                        if (any_milestone_event === 1'b1) begin
                            if ((row_complete_fire === 1'b1)
                                    && (weight_request_fire === 1'b0)
                                    && (psum_request_fire === 1'b0)
                                    && (response_accept === 1'b0)
                                    && (core_accept === 1'b0)
                                    && (psum_commit_fire === 1'b0)
                                    && (task_done_fire === 1'b0)
                                    && (row_complete_id === 6'd0)) begin
                                row_completions_q <= 4'd1;
                                stage_q <= W_ROW_DONE;
                            end else begin
                                witness_fault_q <= 1'b1;
                            end
                        end
                    end
                    W_ROW_DONE: begin
                        if (any_milestone_event === 1'b1) begin
                            if ((task_done_fire === 1'b1)
                                    && (weight_request_fire === 1'b0)
                                    && (psum_request_fire === 1'b0)
                                    && (response_accept === 1'b0)
                                    && (core_accept === 1'b0)
                                    && (psum_commit_fire === 1'b0)
                                    && (row_complete_fire === 1'b0)
                                    && (task_done_epoch === 16'h9001)) begin
                                task_completions_q <= 4'd1;
                                stage_q <= W_TASK_DONE;
                            end else begin
                                witness_fault_q <= 1'b1;
                            end
                        end
                    end
                    W_TASK_DONE: begin
                        if (any_milestone_event === 1'b1)
                            witness_fault_q <= 1'b1;
                    end
                    default: witness_fault_q <= 1'b1;
                endcase
            end

            if ((saw_reset_q !== 1'b1)
                    || (attack_mask_active !== 1'b0)
                    || (any_design_fault !== 1'b0))
                witness_fault_q <= 1'b1;
        end
    end

    final begin : witness_final_oracle
        logic pass;
        pass = (saw_reset_q === 1'b1)
            && (witness_fault_q === 1'b0)
            && (stage_q === W_TASK_DONE)
            && (weight_requests_q === 4'd2)
            && (psum_requests_q === 4'd1)
            && (responses_q === 4'd2)
            && (core_accepts_q === 4'd2)
            && (psum_commits_q === 4'd1)
            && (row_completions_q === 4'd1)
            && (task_completions_q === 4'd1)
            && (design_issue_accepts === 64'd2)
            && (design_psum_commits === 64'd1)
            && (design_row_completions === 64'd1)
            && (attack_mask_active === 1'b0)
            && (any_design_fault === 1'b0);
        $display("M1337R15_WITNESS_OPERANDS pass=%0d stage=%0d weight_req=%0d psum_req=%0d responses=%0d core_accepts=%0d psum_commits=%0d rows=%0d tasks=%0d design_issue=%0d design_commit=%0d design_rows=%0d masks=%0d faults=%0d",
            pass, stage_q, weight_requests_q, psum_requests_q, responses_q,
            core_accepts_q, psum_commits_q, row_completions_q,
            task_completions_q, design_issue_accepts, design_psum_commits,
            design_row_completions, attack_mask_active, any_design_fault);
        $fflush();
        if (pass === 1'b1) begin
            $display("PASS_M1337R15_REAL_M935_RUNTIME_WITNESS wrapper_functional_candidate=true strict_registered_stages=true unknown_fail_closed=true structural_bind=true ledger_bytes=214912 functional_vcs=false timing_verified=false cycles_measured=false speedup=false ppa=false energy=false headline=false");
        end else begin
            $fatal(1, "M1337R15 runtime witness incomplete, unknown, or attacked");
        end
    end
endmodule

bind tb_m1270r13_m1162_real_m935_protocol_unit_delay_r13
    m1337r15_m935_runtime_witness u_m1337r15_runtime_witness (
        .clk_core(clk_core),
        .reset_n(reset_n),
        .issue_request_valid(issue_request_valid),
        .issue_request_first(issue_request_first),
        .issue_request_last(issue_request_last),
        .issue_request_source_valid(issue_request_source_valid),
        .issue_request_source_index(issue_request_source_index),
        .weight_request_fire(weight_req_valid && weight_req_ready),
        .psum_request_fire(psum_req_valid && psum_req_ready),
        .response_accept(dut.response_accept_w),
        .core_accept(dut.core_issue_data_valid && dut.core_issue_data_ready),
        .psum_commit_fire(psum_write_valid && psum_write_ready),
        .psum_commit_address(psum_write_address),
        .row_complete_fire(row_complete_valid && row_complete_ready),
        .row_complete_id(row_complete_id),
        .task_done_fire(task_done_valid),
        .task_done_epoch(task_done_epoch),
        .request_hold_attack_mode(request_hold_attack_mode),
        .weight_service_attack_mode(weight_service_attack_mode),
        .psum_service_attack_mode(psum_service_attack_mode),
        .protocol_error(protocol_error),
        .boundary_fault(dut.boundary_fault_q),
        .core_fault(dut.core_protocol_error),
        .m935_fault(dut.u_frozen_m935.fault_q),
        .weight_service_fault(weight_service_fault),
        .psum_service_fault(psum_service_fault),
        .design_issue_accepts(count_issue_accepts),
        .design_psum_commits(count_psum_commits),
        .design_row_completions(count_row_completions)
    );

`default_nettype wire
