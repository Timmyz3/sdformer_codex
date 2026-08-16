`timescale 1ns/1ps
`default_nettype none

module h67_temporal_slot_flow_2s_assertions #(
    parameter int HEAD_DIM = 32,
    parameter int TOKEN_W = 9,
    parameter int GATE_W = 9,
    parameter int THRESHOLD_W = 8,
    parameter int FIFO_OCC_W = 6,
    parameter int SLOT_FIFO_DEPTH = 32,
    parameter int PAIRS = 225,
    parameter int PAIR_ID_W = (PAIRS <= 1) ? 1 : $clog2(PAIRS),
    parameter int PAIR_COUNT_W = $clog2(PAIRS + 1)
) (
    input logic                       clk_core,
    input logic                       rst_core,
    input logic                       window_start,
    input logic                       window_start_accept,
    input logic                       window_start_reject,
    input logic                       window_active_q,
    input logic                       window_done,
    input logic                       pair_ready,
    input logic                       pair_commit,
    input logic                       encoder_pair_commit,
    input logic                       packet_valid,
    input logic                       packet_ready,
    input logic [1:0]                 packet_slot_count,
    input logic [15:0]                packet_slot0,
    input logic [15:0]                packet_slot1,
    input logic                       fifo_valid,
    input logic                       fifo_ready,
    input logic [1:0]                 fifo_count,
    input logic [1:0]                 k_read_req_valid,
    input logic [1:0]                 k_read_resp_valid,
    input logic                       directory_in_valid,
    input logic                       directory_in_ready,
    input logic [1:0]                 slot0_temporal_mask,
    input logic [1:0]                 slot1_temporal_mask,
    input logic                       slot0_pair_last,
    input logic                       pair_open_q,
    input logic                       open_after_slot0,
    input logic                       open_after_slot1,
    input logic [1:0]                 closed_pair_count,
    input logic                       batch_fire,
    input logic [PAIR_COUNT_W-1:0]    decoded_pairs_q,
    input logic [PAIR_ID_W-1:0]       directory_in0_pair_id,
    input logic [PAIR_ID_W-1:0]       directory_in1_pair_id,
    input logic                       directory_seal,
    input logic                       seal_ready,
    input logic                       seal_issued_q,
    input logic [31:0]                perf_original_tokens,
    input logic                       out_valid,
    input logic                       out_ready,
    input logic                       out_last,
    input logic [TOKEN_W-1:0]         out_token_id,
    input logic [HEAD_DIM-1:0]        out_k_bits,
    input logic [GATE_W-1:0]          out_gate_q17,
    input logic [THRESHOLD_W-1:0]     out_threshold_q8,
    input logic [FIFO_OCC_W-1:0]      perf_fifo_occupancy,
    input logic                       protocol_error
);
    property p_output_stable_under_backpressure;
        @(posedge clk_core) disable iff (rst_core || window_start_accept)
        out_valid && !out_ready |=> out_valid
            && $stable({out_last, out_token_id, out_k_bits,
                        out_gate_q17, out_threshold_q8});
    endproperty

    property p_last_requires_valid;
        @(posedge clk_core) disable iff (rst_core)
        out_last |-> out_valid;
    endproperty

    property p_emitted_k_is_active;
        @(posedge clk_core) disable iff (rst_core || window_start_accept)
        out_valid |-> out_k_bits != 0;
    endproperty

    property p_fifo_capacity;
        @(posedge clk_core) disable iff (rst_core)
        32'(perf_fifo_occupancy) <= 32'(SLOT_FIFO_DEPTH);
    endproperty

    property p_start_blocks_external_handshakes;
        @(posedge clk_core) disable iff (rst_core)
        window_start_accept |-> !pair_ready && !pair_commit && !out_valid
            && !(packet_valid && packet_ready) && !fifo_valid;
    endproperty

    property p_start_accept_reject_partition;
        @(posedge clk_core) disable iff (rst_core)
        window_start |-> window_start_accept ^ window_start_reject;
    endproperty

    property p_start_accept_is_legal;
        @(posedge clk_core) disable iff (rst_core)
        window_start_accept |-> !window_active_q || window_done;
    endproperty

    property p_rejected_start_sets_error;
        @(posedge clk_core) disable iff (rst_core)
        window_start_reject |=> protocol_error;
    endproperty

    property p_rejected_start_blocks_new_pair;
        @(posedge clk_core) disable iff (rst_core)
        window_start_reject |-> !pair_ready && !pair_commit
                              && !encoder_pair_commit;
    endproperty

    property p_packet_shape;
        @(posedge clk_core) disable iff (rst_core || window_start_accept)
        packet_valid |->
            (packet_slot_count == 1
             && packet_slot0[12] && packet_slot0[9:8] == 2'b11)
            || (packet_slot_count == 2
                && !packet_slot0[12] && packet_slot0[9:8] == 2'b01
                && packet_slot1[12] && packet_slot1[9:8] == 2'b10);
    endproperty

    property p_packet_stable_under_backpressure;
        @(posedge clk_core) disable iff (rst_core || window_start_accept)
        packet_valid && !packet_ready |=> packet_valid
            && $stable({packet_slot_count, packet_slot0, packet_slot1});
    endproperty

    property p_k_read_is_one_cycle;
        @(posedge clk_core) disable iff (rst_core || window_start_accept)
        !$past(window_start_accept) |-> k_read_resp_valid == $past(k_read_req_valid);
    endproperty

    property p_multiplicity_accumulates;
        @(posedge clk_core) disable iff (rst_core || window_start_accept)
        !$past(window_start_accept) && $past(directory_in_valid && directory_in_ready)
        |-> perf_original_tokens == $past(perf_original_tokens)
            + 32'($past(slot0_temporal_mask[0]))
            + 32'($past(slot0_temporal_mask[1]))
            + ($past(fifo_count) == 2
               ? 32'($past(slot1_temporal_mask[0]))
                 + 32'($past(slot1_temporal_mask[1]))
               : 0);
    endproperty

    property p_multiplicity_holds_without_input;
        @(posedge clk_core) disable iff (rst_core || window_start_accept)
        !$past(window_start_accept) && !$past(directory_in_valid && directory_in_ready)
        |-> perf_original_tokens == $past(perf_original_tokens);
    endproperty

    property p_fifo_count_enq_only;
        @(posedge clk_core) disable iff (rst_core || window_start_accept)
        !$past(window_start_accept)
        && $past(packet_valid && packet_ready)
        && !$past(fifo_valid && fifo_ready)
        |-> 32'(perf_fifo_occupancy) == 32'($past(perf_fifo_occupancy))
            + 32'($past(packet_slot_count));
    endproperty

    property p_fifo_count_deq_only;
        @(posedge clk_core) disable iff (rst_core || window_start_accept)
        !$past(window_start_accept)
        && !$past(packet_valid && packet_ready)
        && $past(fifo_valid && fifo_ready)
        |-> 32'(perf_fifo_occupancy) + 32'($past(fifo_count))
            == 32'($past(perf_fifo_occupancy));
    endproperty

    property p_fifo_count_both;
        @(posedge clk_core) disable iff (rst_core || window_start_accept)
        !$past(window_start_accept)
        && $past(packet_valid && packet_ready)
        && $past(fifo_valid && fifo_ready)
        |-> 32'(perf_fifo_occupancy) + 32'($past(fifo_count))
            == 32'($past(perf_fifo_occupancy))
            + 32'($past(packet_slot_count));
    endproperty

    property p_fifo_count_idle;
        @(posedge clk_core) disable iff (rst_core || window_start_accept)
        !$past(window_start_accept)
        && !$past(packet_valid && packet_ready)
        && !$past(fifo_valid && fifo_ready)
        |-> perf_fifo_occupancy == $past(perf_fifo_occupancy);
    endproperty

    property p_pair_state_transition;
        @(posedge clk_core) disable iff (rst_core || window_start_accept)
        !$past(window_start_accept) && $past(batch_fire)
        |-> decoded_pairs_q == $past(decoded_pairs_q)
                              + PAIR_COUNT_W'($past(closed_pair_count))
            && pair_open_q == ($past(fifo_count) == 2
                               ? $past(open_after_slot1)
                               : $past(open_after_slot0));
    endproperty

    property p_directory_pair_ids;
        @(posedge clk_core) disable iff (rst_core || window_start_accept)
        directory_in_valid |->
            directory_in0_pair_id == PAIR_ID_W'(decoded_pairs_q)
            && (fifo_count != 2
                || directory_in1_pair_id
                   == PAIR_ID_W'(decoded_pairs_q
                                 + PAIR_COUNT_W'(slot0_pair_last)));
    endproperty

    property p_seal_is_one_shot;
        @(posedge clk_core) disable iff (rst_core || window_start_accept)
        seal_issued_q |-> !seal_ready;
    endproperty

    assert property (p_output_stable_under_backpressure)
        else $fatal(1, "slot flow output changed under backpressure");
    assert property (p_last_requires_valid)
        else $fatal(1, "slot flow last without valid");
    assert property (p_emitted_k_is_active)
        else $fatal(1, "slot flow emitted zero K");
    assert property (p_fifo_capacity)
        else $fatal(1, "slot FIFO occupancy exceeded capacity");
    assert property (p_start_blocks_external_handshakes)
        else $fatal(1, "window_start overlapped an external handshake");
    assert property (p_start_accept_reject_partition)
        else $fatal(1, "window_start was not classified exactly once");
    assert property (p_start_accept_is_legal)
        else $fatal(1, "window_start was accepted before prior window completion");
    assert property (p_rejected_start_sets_error)
        else $fatal(1, "rejected window_start did not set protocol_error");
    assert property (p_rejected_start_blocks_new_pair)
        else $fatal(1, "rejected window_start accepted an unhandshaken pair");
    assert property (p_packet_shape)
        else $fatal(1, "slot packet shape is not atomic common/split form");
    assert property (p_packet_stable_under_backpressure)
        else $fatal(1, "slot packet changed under backpressure");
    assert property (p_k_read_is_one_cycle)
        else $fatal(1, "synchronous K response did not match prior request mask");
    assert property (p_multiplicity_accumulates)
        else $fatal(1, "weighted-SCS multiplicity increment mismatch");
    assert property (p_multiplicity_holds_without_input)
        else $fatal(1, "weighted-SCS multiplicity changed without input");
    assert property (p_fifo_count_enq_only)
        else $fatal(1, "slot FIFO enqueue-only conservation mismatch");
    assert property (p_fifo_count_deq_only)
        else $fatal(1, "slot FIFO dequeue-only conservation mismatch");
    assert property (p_fifo_count_both)
        else $fatal(1, "slot FIFO simultaneous conservation mismatch");
    assert property (p_fifo_count_idle)
        else $fatal(1, "slot FIFO occupancy changed while idle");
    assert property (p_pair_state_transition)
        else $fatal(1, "dual-slot cross-pair state transition mismatch");
    assert property (p_directory_pair_ids)
        else $fatal(1, "dual-slot directory pair ID mapping mismatch");
    assert property (p_seal_is_one_shot)
        else $fatal(1, "window seal remained ready after commit");

    cover property (@(posedge clk_core) disable iff (rst_core)
        out_valid && !out_ready |=> out_valid && out_ready);
    cover property (@(posedge clk_core) disable iff (rst_core)
        perf_fifo_occupancy == FIFO_OCC_W'(SLOT_FIFO_DEPTH));
    cover property (@(posedge clk_core) disable iff (rst_core)
        protocol_error);
    cover property (@(posedge clk_core) disable iff (rst_core || window_start_accept)
        batch_fire && fifo_count == 2 && slot0_pair_last);
    cover property (@(posedge clk_core) disable iff (rst_core || window_start_accept)
        directory_seal);
    cover property (@(posedge clk_core) disable iff (rst_core)
        window_start_reject |=> protocol_error);
endmodule

module h67_temporal_directory_2s_assertions #(
    parameter int MAX_SCORE = 162,
    parameter int MAX_DESCRIPTORS = 450,
    parameter int SCORE_W = 16,
    parameter int PAIR_ID_W = 8,
    parameter int COUNT_W = $clog2(2 * MAX_DESCRIPTORS + 1),
    parameter int DESC_COUNT_W = $clog2(MAX_DESCRIPTORS + 1),
    parameter int CLASS_W = $clog2(MAX_SCORE + 1)
) (
    input logic clk_core,
    input logic rst_core,
    input logic window_start,
    input logic in_valid,
    input logic in_ready,
    input logic batch_legal,
    input logic [1:0] in_count,
    input logic [PAIR_ID_W-1:0] in0_pair_id,
    input logic signed [SCORE_W-1:0] in0_score_q7,
    input logic [1:0] in0_temporal_mask,
    input logic [1:0] in0_active_mask,
    input logic [PAIR_ID_W-1:0] in1_pair_id,
    input logic signed [SCORE_W-1:0] in1_score_q7,
    input logic [1:0] in1_temporal_mask,
    input logic [1:0] in1_active_mask,
    input logic [1:0] multiplicity0,
    input logic [1:0] multiplicity1,
    input logic [1:0] active_add,
    input logic [DESC_COUNT_W-1:0] active_count_q,
    input logic [MAX_SCORE:0] class_present_q,
    input logic [COUNT_W-1:0] class_hist [0:MAX_SCORE],
    input logic [PAIR_ID_W-1:0] active_pair_store [0:MAX_DESCRIPTORS-1],
    input logic signed [SCORE_W-1:0] active_score_store [0:MAX_DESCRIPTORS-1],
    input logic [1:0] active_temporal_store [0:MAX_DESCRIPTORS-1],
    input logic [1:0] active_mask_store [0:MAX_DESCRIPTORS-1]
);
    property p_same_class_collision_merge;
        @(posedge clk_core) disable iff (rst_core || window_start)
        !$past(window_start)
        && $past(in_valid && in_ready && batch_legal && in_count == 2
                 && in0_score_q7 == in1_score_q7)
        |-> class_hist[$past(in0_score_q7[CLASS_W-1:0])]
            == ($past(class_present_q[in0_score_q7[CLASS_W-1:0]])
                ? $past(class_hist[in0_score_q7[CLASS_W-1:0]])
                  + COUNT_W'($past(multiplicity0))
                  + COUNT_W'($past(multiplicity1))
                : COUNT_W'($past(multiplicity0))
                  + COUNT_W'($past(multiplicity1)));
    endproperty

    property p_active_count_commit;
        @(posedge clk_core) disable iff (rst_core || window_start)
        !$past(window_start) && $past(in_valid && in_ready && batch_legal)
        |-> active_count_q == $past(active_count_q)
                              + DESC_COUNT_W'($past(active_add));
    endproperty

    property p_active0_append;
        @(posedge clk_core) disable iff (rst_core || window_start)
        !$past(window_start)
        && $past(in_valid && in_ready && batch_legal && in0_active_mask != 0)
        |-> active_pair_store[$past(active_count_q)] == $past(in0_pair_id)
            && active_score_store[$past(active_count_q)] == $past(in0_score_q7)
            && active_temporal_store[$past(active_count_q)]
               == $past(in0_temporal_mask)
            && active_mask_store[$past(active_count_q)] == $past(in0_active_mask);
    endproperty

    property p_active1_append;
        @(posedge clk_core) disable iff (rst_core || window_start)
        !$past(window_start)
        && $past(in_valid && in_ready && batch_legal && in_count == 2
                 && in1_active_mask != 0)
        |-> active_pair_store[
                $past(active_count_q + DESC_COUNT_W'(in0_active_mask != 0))
            ] == $past(in1_pair_id)
            && active_score_store[
                $past(active_count_q + DESC_COUNT_W'(in0_active_mask != 0))
            ] == $past(in1_score_q7)
            && active_temporal_store[
                $past(active_count_q + DESC_COUNT_W'(in0_active_mask != 0))
            ] == $past(in1_temporal_mask)
            && active_mask_store[
                $past(active_count_q + DESC_COUNT_W'(in0_active_mask != 0))
            ] == $past(in1_active_mask);
    endproperty

    assert property (p_same_class_collision_merge)
        else $fatal(1, "2S same-class histogram collision merge mismatch");
    assert property (p_active_count_commit)
        else $fatal(1, "2S active descriptor count mismatch");
    assert property (p_active0_append)
        else $fatal(1, "2S first active append mismatch");
    assert property (p_active1_append)
        else $fatal(1, "2S second active append mismatch");

    cover property (@(posedge clk_core) disable iff (rst_core || window_start)
        in_valid && in_ready && in_count == 2 && in0_score_q7 == in1_score_q7);
    cover property (@(posedge clk_core) disable iff (rst_core || window_start)
        in_valid && in_ready && in_count == 2
        && in0_active_mask != 0 && in1_active_mask != 0);
endmodule

module h67_temporal_fifo_2s_assertions #(
    parameter int DEPTH = 32,
    parameter int PTR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    parameter int OCC_W = $clog2(DEPTH + 1)
) (
    input logic clk_core,
    input logic rst_core,
    input logic window_start,
    input logic enq_valid,
    input logic enq_ready,
    input logic [1:0] enq_count,
    input logic [15:0] enq_slot0,
    input logic [15:0] enq_slot1,
    input logic deq_valid,
    input logic deq_ready,
    input logic [1:0] deq_count,
    input logic [15:0] deq_slot0,
    input logic [15:0] deq_slot1,
    input logic [PTR_W-1:0] write_ptr_q,
    input logic [PTR_W-1:0] read_ptr_q,
    input logic [15:0] slot_mem [0:DEPTH-1]
);
    function automatic logic [PTR_W-1:0] ptr_add(
        input logic [PTR_W-1:0] ptr,
        input int unsigned amount
    );
        int unsigned value;
        begin
            value = 32'(ptr) + amount;
            if (value >= DEPTH)
                value = value - DEPTH;
            ptr_add = PTR_W'(value);
        end
    endfunction

    property p_start_blocks_leaf_handshakes;
        @(posedge clk_core) disable iff (rst_core)
        window_start |-> !enq_ready && !deq_valid;
    endproperty

    property p_write_pointer_advance;
        @(posedge clk_core) disable iff (rst_core || window_start)
        !$past(window_start) && $past(enq_valid && enq_ready)
        |-> write_ptr_q == ptr_add($past(write_ptr_q),
                                  $past(enq_count) == 2 ? 2 : 1);
    endproperty

    property p_read_pointer_advance;
        @(posedge clk_core) disable iff (rst_core || window_start)
        !$past(window_start) && $past(deq_valid && deq_ready)
        |-> read_ptr_q == ptr_add($past(read_ptr_q),
                                 $past(deq_count) == 2 ? 2 : 1);
    endproperty

    property p_slot0_write;
        @(posedge clk_core) disable iff (rst_core || window_start)
        !$past(window_start) && $past(enq_valid && enq_ready)
        |-> slot_mem[$past(write_ptr_q)] == $past(enq_slot0);
    endproperty

    property p_slot1_write;
        @(posedge clk_core) disable iff (rst_core || window_start)
        !$past(window_start)
        && $past(enq_valid && enq_ready && enq_count == 2)
        |-> slot_mem[ptr_add($past(write_ptr_q), 1)] == $past(enq_slot1);
    endproperty

    property p_dequeue_data_matches_head;
        @(posedge clk_core) disable iff (rst_core || window_start)
        deq_valid |-> deq_slot0 == slot_mem[read_ptr_q]
            && (deq_count != 2
                || deq_slot1 == slot_mem[ptr_add(read_ptr_q, 1)]);
    endproperty

    assert property (p_start_blocks_leaf_handshakes)
        else $fatal(1, "2S FIFO accepted traffic during window_start");
    assert property (p_write_pointer_advance)
        else $fatal(1, "2S FIFO write pointer advance mismatch");
    assert property (p_read_pointer_advance)
        else $fatal(1, "2S FIFO read pointer advance mismatch");
    assert property (p_slot0_write)
        else $fatal(1, "2S FIFO first slot write mismatch");
    assert property (p_slot1_write)
        else $fatal(1, "2S FIFO second slot write mismatch");
    assert property (p_dequeue_data_matches_head)
        else $fatal(1, "2S FIFO dequeue data mismatch");

    cover property (@(posedge clk_core) disable iff (rst_core || window_start)
        enq_valid && enq_ready && deq_valid && deq_ready);
endmodule

module h67_sync_dual_bank_k_store_assertions #(
    parameter int HEAD_DIM = 32,
    parameter int PAIRS = 225,
    parameter int ADDR_W = (PAIRS <= 1) ? 1 : $clog2(PAIRS)
) (
    input logic clk_core,
    input logic rst_core,
    input logic window_start,
    input logic [1:0] read_req_valid,
    input logic [ADDR_W-1:0] read_req_addr,
    input logic [1:0] read_resp_valid,
    input logic [HEAD_DIM-1:0] read_resp_k0,
    input logic [HEAD_DIM-1:0] read_resp_k1,
    input logic [HEAD_DIM-1:0] bank0 [0:PAIRS-1],
    input logic [HEAD_DIM-1:0] bank1 [0:PAIRS-1]
);
    property p_bank0_read_data;
        @(posedge clk_core) disable iff (rst_core || window_start)
        !$past(window_start) && $past(read_req_valid[0])
        |-> read_resp_valid[0]
            && read_resp_k0 == $past(bank0[read_req_addr]);
    endproperty

    property p_bank1_read_data;
        @(posedge clk_core) disable iff (rst_core || window_start)
        !$past(window_start) && $past(read_req_valid[1])
        |-> read_resp_valid[1]
            && read_resp_k1 == $past(bank1[read_req_addr]);
    endproperty

    assert property (p_bank0_read_data)
        else $fatal(1, "K bank0 address/data response mismatch");
    assert property (p_bank1_read_data)
        else $fatal(1, "K bank1 address/data response mismatch");
    cover property (@(posedge clk_core) disable iff (rst_core || window_start)
        read_req_valid == 2'b11);
endmodule

bind h67_temporal_weighted_scs_directory_2s
    h67_temporal_directory_2s_assertions #(
        .MAX_SCORE(MAX_SCORE),
        .MAX_DESCRIPTORS(MAX_DESCRIPTORS),
        .SCORE_W(SCORE_W),
        .PAIR_ID_W(PAIR_ID_W),
        .COUNT_W(COUNT_W),
        .DESC_COUNT_W(DESC_COUNT_W),
        .CLASS_W(CLASS_W)
    ) u_h67_temporal_directory_2s_assertions (
        .clk_core(clk_core), .rst_core(rst_core), .window_start(window_start),
        .in_valid(in_valid), .in_ready(in_ready), .batch_legal(batch_legal),
        .in_count(in_count),
        .in0_pair_id(in0_pair_id), .in0_score_q7(in0_score_q7),
        .in0_temporal_mask(in0_temporal_mask), .in0_active_mask(in0_active_mask),
        .in1_pair_id(in1_pair_id), .in1_score_q7(in1_score_q7),
        .in1_temporal_mask(in1_temporal_mask), .in1_active_mask(in1_active_mask),
        .multiplicity0(multiplicity0), .multiplicity1(multiplicity1),
        .active_add(active_add), .active_count_q(active_count_q),
        .class_present_q(class_present_q), .class_hist(class_hist),
        .active_pair_store(active_pair_store),
        .active_score_store(active_score_store),
        .active_temporal_store(active_temporal_store),
        .active_mask_store(active_mask_store)
    );

bind h67_temporal_slot_fifo_2s
    h67_temporal_fifo_2s_assertions #(
        .DEPTH(DEPTH), .PTR_W(PTR_W), .OCC_W(OCC_W)
    ) u_h67_temporal_fifo_2s_assertions (
        .clk_core(clk_core), .rst_core(rst_core), .window_start(window_start),
        .enq_valid(enq_valid), .enq_ready(enq_ready), .enq_count(enq_count),
        .enq_slot0(enq_slot0), .enq_slot1(enq_slot1),
        .deq_valid(deq_valid), .deq_ready(deq_ready), .deq_count(deq_count),
        .deq_slot0(deq_slot0), .deq_slot1(deq_slot1),
        .write_ptr_q(write_ptr_q), .read_ptr_q(read_ptr_q), .slot_mem(slot_mem)
    );

bind h67_sync_dual_bank_k_store
    h67_sync_dual_bank_k_store_assertions #(
        .HEAD_DIM(HEAD_DIM), .PAIRS(PAIRS), .ADDR_W(ADDR_W)
    ) u_h67_sync_dual_bank_k_store_assertions (
        .clk_core(clk_core), .rst_core(rst_core), .window_start(window_start),
        .read_req_valid(read_req_valid), .read_req_addr(read_req_addr),
        .read_resp_valid(read_resp_valid),
        .read_resp_k0(read_resp_k0), .read_resp_k1(read_resp_k1),
        .bank0(bank0), .bank1(bank1)
    );

bind h67_temporal_slot_shiftmax_sync_k_2s_top
    h67_temporal_slot_flow_2s_assertions #(
        .HEAD_DIM(HEAD_DIM),
        .TOKEN_W(TOKEN_W),
        .GATE_W(GATE_W),
        .THRESHOLD_W(THRESHOLD_W),
        .FIFO_OCC_W(FIFO_OCC_W),
        .SLOT_FIFO_DEPTH(SLOT_FIFO_DEPTH),
        .PAIRS(PAIRS),
        .PAIR_ID_W(PAIR_ID_W)
    ) u_h67_temporal_slot_flow_2s_assertions (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start(window_start),
        .window_start_accept(window_start_accept),
        .window_start_reject(window_start_reject),
        .window_active_q(window_active_q),
        .window_done(window_done),
        .pair_ready(pair_ready),
        .pair_commit(pair_commit),
        .encoder_pair_commit(encoder_pair_commit),
        .packet_valid(packet_valid),
        .packet_ready(packet_ready),
        .packet_slot_count(packet_slot_count),
        .packet_slot0(packet_slot0),
        .packet_slot1(packet_slot1),
        .fifo_valid(fifo_valid),
        .fifo_ready(fifo_ready),
        .fifo_count(fifo_count),
        .k_read_req_valid(k_read_req_valid),
        .k_read_resp_valid(k_read_resp_valid),
        .directory_in_valid(directory_in_valid),
        .directory_in_ready(directory_in_ready),
        .slot0_temporal_mask(slot0_temporal_mask),
        .slot1_temporal_mask(slot1_temporal_mask),
        .slot0_pair_last(slot0_pair_last),
        .pair_open_q(pair_open_q),
        .open_after_slot0(open_after_slot0),
        .open_after_slot1(open_after_slot1),
        .closed_pair_count(closed_pair_count),
        .batch_fire(batch_fire),
        .decoded_pairs_q(decoded_pairs_q),
        .directory_in0_pair_id(u_directory.in0_pair_id),
        .directory_in1_pair_id(u_directory.in1_pair_id),
        .directory_seal(directory_seal),
        .seal_ready(seal_ready),
        .seal_issued_q(seal_issued_q),
        .perf_original_tokens(perf_original_tokens),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_last(out_last),
        .out_token_id(out_token_id),
        .out_k_bits(out_k_bits),
        .out_gate_q17(out_gate_q17),
        .out_threshold_q8(out_threshold_q8),
        .perf_fifo_occupancy(perf_fifo_occupancy),
        .protocol_error(protocol_error)
    );

`default_nettype wire
