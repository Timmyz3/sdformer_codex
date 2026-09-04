`timescale 1ns/1ps
`default_nettype none

// Executable protocol assertions for the additive M1162 boundary.  Service
// response stability is an environment obligation.  Directed negative tests
// may set service_attack_mode only while demonstrating that the independent TB
// monitor detects a deliberately broken service; no DUT fault is inferred from
// those assumption attacks.
module m1168r2_m1162_common_charge_protocol_assertions_r2 (
    input logic clk_core,
    input logic reset_n,
    input logic service_attack_mode,
    input logic issue_request_valid,
    input logic [15:0] issue_request_epoch,
    input logic [5:0] issue_request_row_id,
    input logic issue_request_first,
    input logic issue_request_last,
    input logic issue_request_source_valid,
    input logic [3:0] issue_request_source_index,
    input logic issue_request_parent_valid,
    input logic [5:0] issue_request_parent_id,
    input logic weight_request_valid,
    input logic weight_request_ready,
    input logic [15:0] weight_request_epoch,
    input logic [5:0] weight_request_row_id,
    input logic [3:0] weight_request_source_index,
    input logic weight_request_source_valid,
    input logic weight_response_valid,
    input logic weight_response_ready,
    input logic [1151:0] weight_response_data,
    input logic psum_request_valid,
    input logic psum_request_ready,
    input logic [15:0] psum_request_epoch,
    input logic [5:0] psum_request_address,
    input logic psum_response_valid,
    input logic psum_response_ready,
    input logic [1823:0] psum_response_data,
    input logic request_active,
    input logic weight_request_accepted,
    input logic psum_request_accepted,
    input logic request_first,
    input logic core_issue_data_valid,
    input logic core_issue_data_ready,
    input logic response_accept,
    input logic boundary_fault
);
    default clocking cb @(posedge clk_core); endclocking

    ap_weight_request_hold: assert property (disable iff (!reset_n)
        weight_request_valid && !weight_request_ready
        |=> weight_request_valid
            && $stable({weight_request_epoch, weight_request_row_id,
                weight_request_source_index, weight_request_source_valid}));
    ap_psum_request_hold: assert property (disable iff (!reset_n)
        psum_request_valid && !psum_request_ready
        |=> psum_request_valid
            && $stable({psum_request_epoch, psum_request_address}));

    // An accepted service request is suppressed on the following cycle while
    // its peer may continue independently.  This is the executable no-reissue
    // check; the TB also scores exact one-fire counts per transaction.
    ap_weight_no_reissue: assert property (disable iff (!reset_n)
        request_active && weight_request_accepted
        |-> !weight_request_valid);
    ap_psum_no_reissue: assert property (disable iff (!reset_n)
        request_active && psum_request_accepted
        |-> !psum_request_valid);
    ap_nonfirst_never_requests_psum: assert property (disable iff (!reset_n)
        request_active && !request_first |-> !psum_request_valid);

    ap_core_valid_requires_requests: assert property (disable iff (!reset_n)
        core_issue_data_valid |-> request_active
            && weight_request_accepted
            && (!request_first || psum_request_accepted)
            && weight_response_valid
            && (!request_first || psum_response_valid));
    ap_weight_ready_is_atomic: assert property (disable iff (!reset_n)
        weight_response_ready |-> weight_response_valid
            && core_issue_data_valid && core_issue_data_ready
            && (!request_first || psum_response_valid));
    ap_psum_ready_is_first_atomic: assert property (disable iff (!reset_n)
        psum_response_ready |-> request_first && psum_response_valid
            && weight_response_valid && weight_response_ready);
    ap_no_lone_weight_consume: assert property (disable iff (!reset_n)
        request_active && request_first
            && weight_response_valid && !psum_response_valid
        |-> !weight_response_ready && !psum_response_ready);
    ap_no_lone_psum_consume: assert property (disable iff (!reset_n)
        request_active && request_first
            && psum_response_valid && !weight_response_valid
        |-> !weight_response_ready && !psum_response_ready);
    ap_core_backpressure_atomic: assert property (disable iff (!reset_n)
        core_issue_data_valid && !core_issue_data_ready
        |-> !weight_response_ready && !psum_response_ready);

    // The frozen service contract is checked as an assertion in every legal
    // test.  It is explicitly masked only for the two assumption-attacks,
    // which must instead trip the TB's separate sticky service monitor.
    ap_weight_response_hold: assert property (disable iff (!reset_n)
        !service_attack_mode && weight_response_valid && !weight_response_ready
        |=> weight_response_valid && $stable(weight_response_data));
    ap_psum_response_hold: assert property (disable iff (!reset_n)
        !service_attack_mode && psum_response_valid && !psum_response_ready
        |=> psum_response_valid && $stable(psum_response_data));

    ap_boundary_fault_sticky: assert property (disable iff (!reset_n)
        boundary_fault |=> boundary_fault);
    ap_reset_clears_transaction: assert property (
        !reset_n |-> !request_active && !weight_request_accepted
            && !psum_request_accepted && !boundary_fault);

    // Even under arbitrary stalls, the depth-one wrapper cannot complete two
    // beats on adjacent edges.  Directed zero-stall stimulus additionally
    // proves that the lower bound is attained, i.e. completed-beat II=2.
    ap_no_consecutive_response_accept: assert property (disable iff (!reset_n)
        response_accept |=> !response_accept);

    cp_weight_first: cover property (disable iff (!reset_n)
        weight_request_valid && weight_request_ready
            && psum_request_valid && !psum_request_ready
        ##1 request_active && weight_request_accepted
            && !psum_request_accepted);
    cp_psum_first: cover property (disable iff (!reset_n)
        psum_request_valid && psum_request_ready
            && weight_request_valid && !weight_request_ready
        ##1 request_active && psum_request_accepted
            && !weight_request_accepted);
    cp_nonfirst: cover property (disable iff (!reset_n)
        request_active && !request_first && weight_request_accepted
            && !psum_request_valid);
    cp_response_skew_weight: cover property (disable iff (!reset_n)
        request_active && request_first && weight_response_valid
            && !psum_response_valid && !weight_response_ready);
    cp_response_skew_psum: cover property (disable iff (!reset_n)
        request_active && request_first && psum_response_valid
            && !weight_response_valid && !psum_response_ready);
    cp_ii2: cover property (disable iff (!reset_n)
        response_accept ##1 !response_accept ##1 response_accept);
endmodule

`default_nettype wire
