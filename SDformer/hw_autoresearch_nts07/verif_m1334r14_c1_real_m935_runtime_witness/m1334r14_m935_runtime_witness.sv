`timescale 1ns/1ps
`default_nettype none

// Verification-only monotonic witness bound to the exact M1270/R13 real-M935
// top.  It changes no M528/M935/M1162/R3-SVA design byte.  Its only authority
// is a future wrapper-functional VCS verdict after a separate release.
module m1334r14_m935_runtime_witness (
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

    always_ff @(posedge clk_core or negedge reset_n) begin : witness_progress
        integer weight_after;
        integer psum_after;
        integer response_after;
        integer core_after;
        integer commit_after;
        integer row_after;
        integer task_after;
        logic [3:0] next_stage;
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
            weight_after = weight_requests_q + weight_request_fire;
            psum_after = psum_requests_q + psum_request_fire;
            response_after = responses_q + response_accept;
            core_after = core_accepts_q + core_accept;
            commit_after = psum_commits_q + psum_commit_fire;
            row_after = row_completions_q + row_complete_fire;
            task_after = task_completions_q + task_done_fire;
            next_stage = stage_q;

            // Every counted edge is observed at the real child/service pins.
            // First beat is source 0 and reads weight+psum; non-first is source
            // 1 and reads weight only.  No TB-local proxy can satisfy this.
            if (weight_request_fire) begin
                if (!(issue_request_valid && issue_request_source_valid
                        && issue_request_source_index == weight_requests_q[3:0]
                        && issue_request_first == (weight_requests_q == 0)
                        && issue_request_last == (weight_requests_q == 1)
                        && (weight_requests_q == 0 || core_after >= 1)
                        && weight_requests_q < 2))
                    witness_fault_q <= 1'b1;
                weight_requests_q <= weight_requests_q + 1'b1;
            end
            if (psum_request_fire) begin
                if (!(issue_request_valid && issue_request_first
                        && issue_request_source_valid
                        && issue_request_source_index == 0
                        && psum_requests_q == 0))
                    witness_fault_q <= 1'b1;
                psum_requests_q <= psum_requests_q + 1'b1;
            end
            if (response_accept) begin
                if (!(core_accept && responses_q < 2
                        && weight_after >= responses_q + 1
                        && (responses_q != 0 || psum_after >= 1)))
                    witness_fault_q <= 1'b1;
                responses_q <= responses_q + 1'b1;
            end
            if (core_accept) begin
                if (!(response_accept && core_accepts_q < 2))
                    witness_fault_q <= 1'b1;
                core_accepts_q <= core_accepts_q + 1'b1;
            end
            if (psum_commit_fire) begin
                if (!(core_after == 2 && psum_commits_q == 0
                        && psum_commit_address == 0))
                    witness_fault_q <= 1'b1;
                psum_commits_q <= psum_commits_q + 1'b1;
            end
            if (row_complete_fire) begin
                if (!(commit_after == 1 && row_completions_q == 0
                        && row_complete_id == 0))
                    witness_fault_q <= 1'b1;
                row_completions_q <= row_completions_q + 1'b1;
            end
            if (task_done_fire) begin
                if (!(row_after == 1 && task_completions_q == 0
                        && task_done_epoch == 16'h9001))
                    witness_fault_q <= 1'b1;
                task_completions_q <= task_completions_q + 1'b1;
            end

            // The frontier is a monotonic function of observed cumulative
            // milestones.  Missing or guarded stimulus can never reach DONE.
            if (weight_after >= 1 && psum_after >= 1)
                next_stage = W_FIRST_REQUEST;
            if (response_after >= 1 && core_after >= 1)
                next_stage = W_FIRST_ACCEPT;
            if (weight_after >= 2)
                next_stage = W_SECOND_REQUEST;
            if (response_after >= 2 && core_after >= 2)
                next_stage = W_SECOND_ACCEPT;
            if (commit_after >= 1)
                next_stage = W_PSUM_COMMIT;
            if (row_after >= 1)
                next_stage = W_ROW_DONE;
            if (task_after >= 1)
                next_stage = W_TASK_DONE;
            if (next_stage < stage_q)
                witness_fault_q <= 1'b1;
            stage_q <= next_stage;

            if (!saw_reset_q || attack_mask_active || any_design_fault
                    || weight_after > 2 || psum_after > 1
                    || response_after > 2 || core_after > 2
                    || commit_after > 1 || row_after > 1 || task_after > 1)
                witness_fault_q <= 1'b1;
        end
    end

    // A runtime guard, early $finish, missing beat, child-output seam, count
    // spoof, or active attack mask all fail here.  Operands are printed before
    // the single fatal so a sealed log remains independently diagnosable.
    final begin : witness_final_oracle
        logic pass;
        pass = saw_reset_q && !witness_fault_q && stage_q == W_TASK_DONE
            && weight_requests_q == 2 && psum_requests_q == 1
            && responses_q == 2 && core_accepts_q == 2
            && psum_commits_q == 1 && row_completions_q == 1
            && task_completions_q == 1 && design_issue_accepts == 2
            && design_psum_commits == 1 && design_row_completions == 1
            && !attack_mask_active && !any_design_fault;
        $display("M1334R14_WITNESS_OPERANDS pass=%0d stage=%0d weight_req=%0d psum_req=%0d responses=%0d core_accepts=%0d psum_commits=%0d rows=%0d tasks=%0d design_issue=%0d design_commit=%0d design_rows=%0d masks=%0d faults=%0d",
            pass, stage_q, weight_requests_q, psum_requests_q, responses_q,
            core_accepts_q, psum_commits_q, row_completions_q,
            task_completions_q, design_issue_accepts, design_psum_commits,
            design_row_completions, attack_mask_active, any_design_fault);
        $fflush();
        if (pass !== 1'b1)
            $fatal(1, "M1334R14 runtime witness incomplete or attacked");
        $display("PASS_M1334R14_REAL_M935_RUNTIME_WITNESS wrapper_functional_candidate=true weight_requests=2 psum_requests=1 core_accepts=2 psum_commits=1 row_completions=1 task_completions=1 attack_masks=0 child_outputs=real functional_vcs=false timing_verified=false cycles_measured=false speedup=false ppa=false energy=false headline=false");
    end
endmodule

bind tb_m1270r13_m1162_real_m935_protocol_unit_delay_r13
    m1334r14_m935_runtime_witness u_m1334r14_runtime_witness (
        .clk_core(clk_core), .reset_n(reset_n),
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
        .task_done_fire(task_done_valid), .task_done_epoch(task_done_epoch),
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
