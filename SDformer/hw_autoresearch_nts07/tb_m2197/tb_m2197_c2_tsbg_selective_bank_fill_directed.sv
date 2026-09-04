`timescale 1ns/1ps
`default_nettype none

package m2197_tb_pkg;
    function automatic integer signed weight_value(
        input integer channel, input integer slice, input integer lane);
        integer raw;
        begin
            if (channel == 0 && slice == 0 && lane == 0)
                weight_value = -128;
            else begin
                raw = (channel * 7 + slice * 5 + lane * 3) % 31;
                weight_value = raw - 15;
            end
        end
    endfunction
endpackage

module m2197_directed_bank_memory #(
    parameter int LANES = 16,
    parameter int CHANNEL_BITS = 12,
    parameter int EPOCH_BITS = 16,
    parameter int GENERATION_BITS = 32,
    parameter int TAG_BITS = 24
) (
    input logic clk_core,
    input logic rst_core,
    input logic [1:0] phase,
    input logic [7:0] req_valid,
    output logic [7:0] req_ready,
    input logic [EPOCH_BITS-1:0] req_epoch [0:7],
    input logic [2:0] req_slot [0:7],
    input logic [GENERATION_BITS-1:0] req_generation [0:7],
    input logic [TAG_BITS-1:0] req_tag [0:7],
    input logic [2:0] req_output_block [0:7],
    input logic [2:0] req_slice [0:7],
    input logic [CHANNEL_BITS-1:0] req_source_channel [0:7],
    input logic [7:0] req_accept,
    output logic [7:0] rsp_valid,
    input logic [7:0] rsp_ready,
    output logic [EPOCH_BITS-1:0] rsp_epoch [0:7],
    output logic [2:0] rsp_slot [0:7],
    output logic [GENERATION_BITS-1:0] rsp_generation [0:7],
    output logic [TAG_BITS-1:0] rsp_tag [0:7],
    output logic signed [7:0] rsp_weight [0:7][0:LANES-1],
    input logic [7:0] rsp_accept,
    output logic [31:0] reorder_count,
    output logic [31:0] request_backpressure_count
);
    import m2197_tb_pkg::*;
    logic pending_q [0:7];
    logic [31:0] due_q [0:7];
    logic [EPOCH_BITS-1:0] epoch_q [0:7];
    logic [2:0] slot_q [0:7];
    logic [GENERATION_BITS-1:0] generation_q [0:7];
    logic [TAG_BITS-1:0] tag_q [0:7];
    logic [2:0] slice_q [0:7];
    logic [CHANNEL_BITS-1:0] channel_q [0:7];
    logic [31:0] cycle_q;
    logic response_seen_q;
    logic [TAG_BITS-1:0] response_tag_q;
    logic [3:0] last_response_bank_q;
    integer request_count [0:2][0:95];

    always_comb begin
        for (int bank = 0; bank < 8; bank++) begin
            req_ready[bank] = !pending_q[bank]
                && ((cycle_q + bank * 3 + 1) % 7 != 0);
            rsp_valid[bank] = pending_q[bank] && cycle_q >= due_q[bank];
            rsp_epoch[bank] = epoch_q[bank];
            rsp_slot[bank] = slot_q[bank];
            rsp_generation[bank] = generation_q[bank];
            rsp_tag[bank] = tag_q[bank];
            for (int lane = 0; lane < LANES; lane++)
                rsp_weight[bank][lane] = 8'(weight_value(
                    int'(channel_q[bank]), int'(slice_q[bank]), lane));
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            cycle_q <= 0;
            reorder_count <= 0;
            request_backpressure_count <= 0;
            response_seen_q <= 0;
            response_tag_q <= 0;
            last_response_bank_q <= 0;
            for (int bank = 0; bank < 8; bank++) begin
                pending_q[bank] <= 0;
                due_q[bank] <= 0;
                epoch_q[bank] <= 0;
                slot_q[bank] <= 0;
                generation_q[bank] <= 0;
                tag_q[bank] <= 0;
                slice_q[bank] <= 0;
                channel_q[bank] <= 0;
            end
            for (int p = 0; p < 3; p++)
                for (int channel = 0; channel < 96; channel++)
                    request_count[p][channel] <= 0;
        end else begin
            cycle_q <= cycle_q + 1'b1;
            if ((req_valid & ~req_ready) != 0)
                request_backpressure_count <= request_backpressure_count + 1'b1;
            for (int bank = 0; bank < 8; bank++) begin
                if (req_accept[bank]) begin
                    if (pending_q[bank])
                        $fatal(1, "M2197 memory accepted into occupied bank");
                    pending_q[bank] <= 1;
                    due_q[bank] <= cycle_q + (8 - bank);
                    epoch_q[bank] <= req_epoch[bank];
                    slot_q[bank] <= req_slot[bank];
                    generation_q[bank] <= req_generation[bank];
                    tag_q[bank] <= req_tag[bank];
                    slice_q[bank] <= req_slice[bank];
                    channel_q[bank] <= req_source_channel[bank];
                    request_count[phase][req_source_channel[bank]]
                        <= request_count[phase][req_source_channel[bank]] + 1;
                end
                if (rsp_accept[bank]) begin
                    pending_q[bank] <= 0;
                    if (response_seen_q && response_tag_q == rsp_tag[bank]
                            && bank < last_response_bank_q)
                        reorder_count <= reorder_count + 1'b1;
                    response_seen_q <= 1;
                    response_tag_q <= rsp_tag[bank];
                    last_response_bank_q <= bank;
                end
            end
        end
    end
endmodule

module m2197_directed_side #(
    parameter int SCHEDULE_MODE = 0,
    parameter int SOURCE_GROUPS = 6,
    parameter int LANES = 16
) (
    input logic clk_core,
    input logic rst_core,
    input logic [1:0] phase,
    input logic load_valid,
    output logic load_ready,
    input logic [2:0] load_context,
    input logic [23:0] load_tag,
    input logic [5:0] load_group,
    input logic [15:0] load_source_active,
    input logic [15:0] load_source_sign,
    input logic load_last,
    output logic load_accept,
    output logic commit_valid,
    output logic commit_ready,
    output logic [2:0] commit_context,
    output logic [23:0] commit_tag,
    output logic [2:0] commit_slice,
    output logic signed [23:0] commit_accumulator [0:LANES-1],
    output logic commit_terminal,
    output logic commit_accept,
    output logic bundle_done_valid,
    output logic protocol_error,
    output logic stale_response_seen,
    output logic numeric_overflow,
    output logic [31:0] partial_hit_count,
    output logic [31:0] eviction_count,
    output logic [31:0] refill_bank_request_count,
    output logic [31:0] scalar_bank_request_count,
    output logic [31:0] zero_descriptor_skip_count,
    output logic [31:0] reorder_count,
    output logic [31:0] request_backpressure_count,
    output logic [31:0] bridge_backpressure_count,
    output logic [31:0] commit_backpressure_count,
    output logic [31:0] signed_product_count,
    output logic [31:0] commit_count
);
    logic [31:0] side_cycle_q;
    logic [7:0] mem_req_valid, mem_req_ready, mem_req_accept;
    logic [15:0] mem_req_epoch [0:7];
    logic [2:0] mem_req_slot [0:7];
    logic [31:0] mem_req_generation [0:7];
    logic [23:0] mem_req_tag [0:7];
    logic [2:0] mem_req_output_block [0:7];
    logic [2:0] mem_req_slice [0:7];
    logic [11:0] mem_req_source_channel [0:7];
    logic [7:0] mem_rsp_valid, mem_rsp_ready, mem_rsp_accept;
    logic [15:0] mem_rsp_epoch [0:7];
    logic [2:0] mem_rsp_slot [0:7];
    logic [31:0] mem_rsp_generation [0:7];
    logic [23:0] mem_rsp_tag [0:7];
    logic signed [7:0] mem_rsp_weight [0:7][0:LANES-1];
    logic bridge_valid, bridge_ready, bridge_accept;
    logic [2:0] bridge_context;
    logic [5:0] bridge_group;
    logic bridge_half;
    logic [2:0] bridge_slice;
    logic [7:0] bridge_bank_valid;
    logic [11:0] bridge_source_channel [0:7];
    logic signed [1:0] bridge_source_value [0:7];
    logic signed [8:0] bridge_effective_weight [0:7][0:LANES-1];
    logic bundle_done_ready;
    logic busy;
    logic [31:0] cycle_count, row_access_count, cache_hit_count;
    logic [31:0] cache_miss_count, weight_bundle_beat_count;
    logic [31:0] scalar_bank_response_count, issue_count;

    assign bridge_ready = (side_cycle_q % 5) != 2;
    assign commit_ready = (side_cycle_q % 7) != 3;
    assign bundle_done_ready = 1'b1;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            side_cycle_q <= 0;
            bridge_backpressure_count <= 0;
            commit_backpressure_count <= 0;
        end else begin
            side_cycle_q <= side_cycle_q + 1'b1;
            if (bridge_valid && !bridge_ready)
                bridge_backpressure_count <= bridge_backpressure_count + 1'b1;
            if (commit_valid && !commit_ready)
                commit_backpressure_count <= commit_backpressure_count + 1'b1;
        end
    end

    m2193_c2_tsbg_b4_selective_bank_fill_frontend #(
        .SCHEDULE_MODE(SCHEDULE_MODE), .SOURCE_GROUPS(SOURCE_GROUPS)
    ) dut (
        .clk_core(clk_core), .rst_core(rst_core),
        .load_valid(load_valid), .load_ready(load_ready),
        .load_context(load_context), .load_tag(load_tag), .load_group(load_group),
        .load_source_active(load_source_active), .load_source_sign(load_source_sign),
        .load_last(load_last), .load_accept(load_accept),
        .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
        .mem_req_epoch(mem_req_epoch), .mem_req_slot(mem_req_slot),
        .mem_req_generation(mem_req_generation), .mem_req_tag(mem_req_tag),
        .mem_req_output_block(mem_req_output_block), .mem_req_slice(mem_req_slice),
        .mem_req_source_channel(mem_req_source_channel), .mem_req_accept(mem_req_accept),
        .mem_rsp_valid(mem_rsp_valid), .mem_rsp_ready(mem_rsp_ready),
        .mem_rsp_epoch(mem_rsp_epoch), .mem_rsp_slot(mem_rsp_slot),
        .mem_rsp_generation(mem_rsp_generation), .mem_rsp_tag(mem_rsp_tag),
        .mem_rsp_weight(mem_rsp_weight), .mem_rsp_accept(mem_rsp_accept),
        .bridge_valid(bridge_valid), .bridge_ready(bridge_ready),
        .bridge_context(bridge_context), .bridge_group(bridge_group),
        .bridge_half(bridge_half), .bridge_slice(bridge_slice),
        .bridge_bank_valid(bridge_bank_valid),
        .bridge_source_channel(bridge_source_channel),
        .bridge_source_value(bridge_source_value),
        .bridge_effective_weight(bridge_effective_weight),
        .bridge_accept(bridge_accept), .commit_valid(commit_valid),
        .commit_ready(commit_ready), .commit_context(commit_context),
        .commit_tag(commit_tag), .commit_slice(commit_slice),
        .commit_accumulator(commit_accumulator), .commit_terminal(commit_terminal),
        .commit_accept(commit_accept), .bundle_done_valid(bundle_done_valid),
        .bundle_done_ready(bundle_done_ready), .protocol_error(protocol_error),
        .stale_response_seen(stale_response_seen), .numeric_overflow(numeric_overflow),
        .busy(busy), .debug_cycle_count(cycle_count),
        .debug_row_access_count(row_access_count),
        .debug_cache_hit_count(cache_hit_count),
        .debug_cache_miss_count(cache_miss_count),
        .debug_cache_eviction_count(eviction_count),
        .debug_weight_bundle_beat_count(weight_bundle_beat_count),
        .debug_scalar_bank_request_count(scalar_bank_request_count),
        .debug_scalar_bank_response_count(scalar_bank_response_count),
        .debug_issue_count(issue_count),
        .debug_signed_product_count(signed_product_count),
        .debug_commit_count(commit_count),
        .debug_partial_hit_count(partial_hit_count),
        .debug_refill_bank_request_count(refill_bank_request_count),
        .debug_zero_descriptor_skip_count(zero_descriptor_skip_count));

    m2197_directed_bank_memory memory (
        .clk_core(clk_core), .rst_core(rst_core), .phase(phase),
        .req_valid(mem_req_valid), .req_ready(mem_req_ready),
        .req_epoch(mem_req_epoch), .req_slot(mem_req_slot),
        .req_generation(mem_req_generation), .req_tag(mem_req_tag),
        .req_output_block(mem_req_output_block), .req_slice(mem_req_slice),
        .req_source_channel(mem_req_source_channel), .req_accept(mem_req_accept),
        .rsp_valid(mem_rsp_valid), .rsp_ready(mem_rsp_ready),
        .rsp_epoch(mem_rsp_epoch), .rsp_slot(mem_rsp_slot),
        .rsp_generation(mem_rsp_generation), .rsp_tag(mem_rsp_tag),
        .rsp_weight(mem_rsp_weight), .rsp_accept(mem_rsp_accept),
        .reorder_count(reorder_count),
        .request_backpressure_count(request_backpressure_count));

    m2197_c2_tsbg_selective_bank_fill_assertions assertions (
        .clk_core(clk_core), .rst_core(rst_core),
        .load_accept(load_accept), .load_source_active(load_source_active),
        .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
        .mem_req_accept(mem_req_accept), .mem_req_epoch(mem_req_epoch),
        .mem_req_slot(mem_req_slot), .mem_req_generation(mem_req_generation),
        .mem_req_tag(mem_req_tag), .mem_req_output_block(mem_req_output_block),
        .mem_req_slice(mem_req_slice),
        .mem_req_source_channel(mem_req_source_channel),
        .mem_rsp_valid(mem_rsp_valid), .mem_rsp_ready(mem_rsp_ready),
        .mem_rsp_accept(mem_rsp_accept), .bridge_valid(bridge_valid),
        .bridge_ready(bridge_ready), .bridge_bank_valid(bridge_bank_valid),
        .bridge_source_value(bridge_source_value),
        .bridge_effective_weight(bridge_effective_weight),
        .commit_valid(commit_valid), .commit_ready(commit_ready),
        .commit_context(commit_context), .commit_tag(commit_tag),
        .commit_slice(commit_slice),
        .commit_accumulator(commit_accumulator), .commit_terminal(commit_terminal),
        .protocol_error(protocol_error), .stale_response_seen(stale_response_seen),
        .numeric_overflow(numeric_overflow),
        .debug_partial_hit_count(partial_hit_count),
        .debug_cache_eviction_count(eviction_count),
        .debug_refill_bank_request_count(refill_bank_request_count),
        .debug_scalar_bank_request_count(scalar_bank_request_count),
        .debug_zero_descriptor_skip_count(zero_descriptor_skip_count));
endmodule

module tb_m2197_c2_tsbg_selective_bank_fill_directed;
    import m2197_tb_pkg::*;
    localparam int LANES = 16;
    logic clk_core = 0;
    logic rst_core = 1;
    logic [1:0] phase;
    logic load_valid;
    logic load_ready_o, load_ready_t;
    logic [2:0] load_context;
    logic [23:0] load_tag;
    logic [5:0] load_group;
    logic [15:0] load_source_active, load_source_sign;
    logic load_last;
    logic load_accept_o, load_accept_t;

    logic commit_valid_o, commit_ready_o, commit_terminal_o, commit_accept_o;
    logic [2:0] commit_context_o, commit_slice_o;
    logic [23:0] commit_tag_o;
    logic signed [23:0] commit_accumulator_o [0:LANES-1];
    logic done_o, error_o, stale_o, overflow_o;
    logic [31:0] partial_o, evict_o, refill_o, scalar_o, zero_o;
    logic [31:0] reorder_o, req_stall_o, bridge_stall_o, commit_stall_o;
    logic [31:0] products_o, commits_o;

    logic commit_valid_t, commit_ready_t, commit_terminal_t, commit_accept_t;
    logic [2:0] commit_context_t, commit_slice_t;
    logic [23:0] commit_tag_t;
    logic signed [23:0] commit_accumulator_t [0:LANES-1];
    logic done_t, error_t, stale_t, overflow_t;
    logic [31:0] partial_t, evict_t, refill_t, scalar_t, zero_t;
    logic [31:0] reorder_t, req_stall_t, bridge_stall_t, commit_stall_t;
    logic [31:0] products_t, commits_t;

    longint signed expected [0:3][0:5][0:LANES-1];
    logic [23:0] golden_tag_o [0:3];
    logic [23:0] golden_tag_t [0:3];
    logic observed_o [0:3][0:5];
    logic observed_t [0:3][0:5];
    logic done_seen_o, done_seen_t;
    integer bundle_count;
    integer commit_seq_o, commit_seq_t;
    integer identity_checks_o, identity_checks_t;

    always #1.5 clk_core = ~clk_core;

    m2197_directed_side #(.SCHEDULE_MODE(0)) ordinary (
        .clk_core(clk_core), .rst_core(rst_core), .phase(phase),
        .load_valid(load_valid), .load_ready(load_ready_o),
        .load_context(load_context), .load_tag(load_tag), .load_group(load_group),
        .load_source_active(load_source_active), .load_source_sign(load_source_sign),
        .load_last(load_last), .load_accept(load_accept_o),
        .commit_valid(commit_valid_o), .commit_ready(commit_ready_o),
        .commit_context(commit_context_o), .commit_tag(commit_tag_o),
        .commit_slice(commit_slice_o), .commit_accumulator(commit_accumulator_o),
        .commit_terminal(commit_terminal_o), .commit_accept(commit_accept_o),
        .bundle_done_valid(done_o), .protocol_error(error_o),
        .stale_response_seen(stale_o), .numeric_overflow(overflow_o),
        .partial_hit_count(partial_o), .eviction_count(evict_o),
        .refill_bank_request_count(refill_o),
        .scalar_bank_request_count(scalar_o), .zero_descriptor_skip_count(zero_o),
        .reorder_count(reorder_o), .request_backpressure_count(req_stall_o),
        .bridge_backpressure_count(bridge_stall_o),
        .commit_backpressure_count(commit_stall_o),
        .signed_product_count(products_o), .commit_count(commits_o));

    m2197_directed_side #(.SCHEDULE_MODE(1)) tsbg (
        .clk_core(clk_core), .rst_core(rst_core), .phase(phase),
        .load_valid(load_valid), .load_ready(load_ready_t),
        .load_context(load_context), .load_tag(load_tag), .load_group(load_group),
        .load_source_active(load_source_active), .load_source_sign(load_source_sign),
        .load_last(load_last), .load_accept(load_accept_t),
        .commit_valid(commit_valid_t), .commit_ready(commit_ready_t),
        .commit_context(commit_context_t), .commit_tag(commit_tag_t),
        .commit_slice(commit_slice_t), .commit_accumulator(commit_accumulator_t),
        .commit_terminal(commit_terminal_t), .commit_accept(commit_accept_t),
        .bundle_done_valid(done_t), .protocol_error(error_t),
        .stale_response_seen(stale_t), .numeric_overflow(overflow_t),
        .partial_hit_count(partial_t), .eviction_count(evict_t),
        .refill_bank_request_count(refill_t),
        .scalar_bank_request_count(scalar_t), .zero_descriptor_skip_count(zero_t),
        .reorder_count(reorder_t), .request_backpressure_count(req_stall_t),
        .bridge_backpressure_count(bridge_stall_t),
        .commit_backpressure_count(commit_stall_t),
        .signed_product_count(products_t), .commit_count(commits_t));

    task automatic clear_scoreboard;
        for (int ctx = 0; ctx < 4; ctx++)
            for (int slice = 0; slice < 6; slice++) begin
                observed_o[ctx][slice] = 0;
                observed_t[ctx][slice] = 0;
                for (int lane = 0; lane < LANES; lane++)
                    expected[ctx][slice][lane] = 0;
            end
        done_seen_o = 0;
        done_seen_t = 0;
        commit_seq_o = 0;
        commit_seq_t = 0;
    endtask

    task automatic account_row(input int ctx, input int group,
                               input logic [15:0] active,
                               input logic [15:0] sign);
        for (int source = 0; source < 16; source++) begin
            if (active[source]) begin
                for (int slice = 0; slice < 6; slice++)
                    for (int lane = 0; lane < LANES; lane++) begin
                        if (sign[source])
                            expected[ctx][slice][lane] -= weight_value(
                                group * 16 + source, slice, lane);
                        else
                            expected[ctx][slice][lane] += weight_value(
                                group * 16 + source, slice, lane);
                    end
            end
        end
    endtask

    task automatic send_row(input int ctx, input int group,
                            input logic [15:0] active,
                            input logic [15:0] sign, input logic last);
        wait (load_ready_o && load_ready_t);
        @(negedge clk_core);
        load_context = 3'(ctx);
        load_tag = 24'h530000 + bundle_count * 16 + ctx;
        load_group = 6'(group);
        load_source_active = active;
        load_source_sign = active & sign;
        load_last = last;
        load_valid = 1;
        wait (load_accept_o && load_accept_t);
        @(posedge clk_core);
        account_row(ctx, group, active, sign);
        @(negedge clk_core);
        load_valid = 0;
    endtask

    task automatic send_bundle(input int which);
        logic [15:0] active;
        logic [15:0] sign;
        phase = 2'(which);
        bundle_count = which;
        clear_scoreboard();
        for (int ctx = 0; ctx < 4; ctx++) begin
            golden_tag_o[ctx] = 24'h530000 + which * 16 + ctx;
            golden_tag_t[ctx] = 24'h530000 + which * 16 + ctx;
        end
        if (which == 0) begin
            for (int ctx = 0; ctx < 4; ctx++) begin
                active = 0; sign = 0;
                case (ctx)
                    0: active[0] = 1;
                    1: begin active[2] = 1; sign[2] = 1; end
                    2: active[9] = 1;
                    3: begin active[0] = 1; sign[0] = 1; end
                endcase
                send_row(ctx, 0, active, sign, 0);
                send_row(ctx, 1, 0, 0, 1);
            end
        end else if (which == 1) begin
            for (int ctx = 0; ctx < 4; ctx++) begin
                active = 0; sign = 0;
                case (ctx)
                    0: begin active[0] = 1; active[4] = 1; end
                    1: begin active[9] = 1; active[11] = 1; sign[9] = 1; end
                    2: begin active[2] = 1; sign[2] = 1; end
                    3: begin active[11] = 1; sign[11] = 1; end
                endcase
                send_row(ctx, 0, active, sign, 0);
                send_row(ctx, 1, 0, 0, 1);
            end
        end else begin
            for (int ctx = 0; ctx < 4; ctx++) begin
                for (int group = 0; group < 6; group++) begin
                    active = 0; sign = 0;
                    active[(group * 3 + ctx * 2) % 16] = 1;
                    if ((group + ctx) % 2)
                        sign[(group * 3 + ctx * 2) % 16] = 1;
                    send_row(ctx, group, active, sign, group == 5);
                end
            end
        end
        fork : wait_bundle
            begin
                wait (done_seen_o && done_seen_t);
            end
            begin
                repeat (100000) @(posedge clk_core);
                $fatal(1, "M2197 bundle timeout phase=%0d", which);
            end
        join_any
        disable wait_bundle;
        for (int ctx = 0; ctx < 4; ctx++)
            for (int slice = 0; slice < 6; slice++)
                if (!observed_o[ctx][slice] || !observed_t[ctx][slice])
                    $fatal(1, "M2197 missing commit phase=%0d ctx=%0d slice=%0d",
                           which, ctx, slice);
        if (commit_seq_o != 24 || commit_seq_t != 24)
            $fatal(1, "M2197 per-bundle identity count mismatch ordinary=%0d TSBG=%0d",
                   commit_seq_o, commit_seq_t);
        @(posedge clk_core);
    endtask

    // Testbench scoreboard state is also cleared by the stimulus task between
    // bundles, so this is deliberately a verification process rather than an
    // always_ff single-writer process.
    always @(posedge clk_core) begin
        if (!rst_core) begin
            if (done_o) done_seen_o <= 1;
            if (done_t) done_seen_t <= 1;
            if (commit_accept_o) begin
                if (commit_context_o !== 3'(commit_seq_o / 6))
                    $fatal(1, "M2197 ordinary context mismatch got=%0d expected=%0d",
                           commit_context_o, commit_seq_o / 6);
                if (commit_slice_o !== 3'(commit_seq_o % 6))
                    $fatal(1, "M2197 ordinary slice mismatch got=%0d expected=%0d",
                           commit_slice_o, commit_seq_o % 6);
                if (commit_tag_o !== golden_tag_o[commit_context_o])
                    $fatal(1, "M2197 ordinary golden-tag mismatch context=%0d got=%h expected=%h",
                           commit_context_o, commit_tag_o,
                           golden_tag_o[commit_context_o]);
                if (commit_terminal_o !== (commit_slice_o == 5))
                    $fatal(1, "M2197 ordinary terminal mismatch");
                commit_seq_o <= commit_seq_o + 1;
                identity_checks_o <= identity_checks_o + 1;
                if (observed_o[commit_context_o][commit_slice_o])
                    $fatal(1, "M2197 duplicate ordinary commit");
                observed_o[commit_context_o][commit_slice_o] <= 1;
                for (int lane = 0; lane < LANES; lane++)
                    if (commit_accumulator_o[lane] !==
                            expected[commit_context_o][commit_slice_o][lane])
                        $fatal(1, "M2197 ordinary Acc24 mismatch phase=%0d ctx=%0d slice=%0d lane=%0d got=%0d expected=%0d",
                               phase, commit_context_o, commit_slice_o, lane,
                               commit_accumulator_o[lane],
                               expected[commit_context_o][commit_slice_o][lane]);
            end
            if (commit_accept_t) begin
                if (commit_context_t !== 3'(commit_seq_t / 6))
                    $fatal(1, "M2197 TSBG context mismatch got=%0d expected=%0d",
                           commit_context_t, commit_seq_t / 6);
                if (commit_slice_t !== 3'(commit_seq_t % 6))
                    $fatal(1, "M2197 TSBG slice mismatch got=%0d expected=%0d",
                           commit_slice_t, commit_seq_t % 6);
                if (commit_tag_t !== golden_tag_t[commit_context_t])
                    $fatal(1, "M2197 TSBG golden-tag mismatch context=%0d got=%h expected=%h",
                           commit_context_t, commit_tag_t,
                           golden_tag_t[commit_context_t]);
                if (commit_terminal_t !== (commit_slice_t == 5))
                    $fatal(1, "M2197 TSBG terminal mismatch");
                commit_seq_t <= commit_seq_t + 1;
                identity_checks_t <= identity_checks_t + 1;
                if (observed_t[commit_context_t][commit_slice_t])
                    $fatal(1, "M2197 duplicate TSBG commit");
                observed_t[commit_context_t][commit_slice_t] <= 1;
                for (int lane = 0; lane < LANES; lane++)
                    if (commit_accumulator_t[lane] !==
                            expected[commit_context_t][commit_slice_t][lane])
                        $fatal(1, "M2197 TSBG Acc24 mismatch phase=%0d ctx=%0d slice=%0d lane=%0d got=%0d expected=%0d",
                               phase, commit_context_t, commit_slice_t, lane,
                               commit_accumulator_t[lane],
                               expected[commit_context_t][commit_slice_t][lane]);
            end
        end
    end

    initial begin
        phase = 0;
        load_valid = 0;
        load_context = 0;
        load_tag = 0;
        load_group = 0;
        load_source_active = 0;
        load_source_sign = 0;
        load_last = 0;
        bundle_count = 0;
        identity_checks_o = 0;
        identity_checks_t = 0;
        clear_scoreboard();
        repeat (5) @(posedge clk_core);
        rst_core = 0;
        send_bundle(0);
        send_bundle(1);
        send_bundle(2);
        repeat (10) @(posedge clk_core);

        if (error_o || error_t || stale_o || stale_t || overflow_o || overflow_t)
            $fatal(1, "M2197 protocol/numeric failure");
        if (partial_o < 1 || partial_t < 1)
            $fatal(1, "M2197 partial refill not covered");
        if (evict_o < 1 || evict_t < 1)
            $fatal(1, "M2197 eviction not covered");
        if (reorder_o < 1 || reorder_t < 1)
            $fatal(1, "M2197 response reorder not covered");
        if (req_stall_o < 1 || req_stall_t < 1
                || bridge_stall_o < 1 || bridge_stall_t < 1
                || commit_stall_o < 1 || commit_stall_t < 1)
            $fatal(1, "M2197 backpressure not covered");
        if (zero_o < 8 || zero_t < 8)
            $fatal(1, "M2197 zero descriptor skip not covered");
        if (refill_o != scalar_o || refill_t != scalar_t)
            $fatal(1, "M2197 refill/source-count mismatch");
        if (commits_o != 72 || commits_t != 72)
            $fatal(1, "M2197 commit ledger mismatch");
        for (int channel = 0; channel < 96; channel++) begin
            if (ordinary.memory.request_count[0][channel]
                    != ((channel == 0 || channel == 2 || channel == 9) ? 6 : 0))
                $fatal(1, "M2197 phase0 ordinary selective mask mismatch channel=%0d", channel);
            if (tsbg.memory.request_count[0][channel]
                    != ((channel == 0 || channel == 2 || channel == 9) ? 6 : 0))
                $fatal(1, "M2197 phase0 TSBG selective mask mismatch channel=%0d", channel);
            if (ordinary.memory.request_count[1][channel]
                    != ((channel == 4 || channel == 11) ? 6 : 0))
                $fatal(1, "M2197 phase1 ordinary missing-only mask mismatch channel=%0d", channel);
            if (tsbg.memory.request_count[1][channel]
                    != ((channel == 4 || channel == 11) ? 6 : 0))
                $fatal(1, "M2197 phase1 TSBG missing-only mask mismatch channel=%0d", channel);
        end
        $display("M2197_COVER partial_o=%0d partial_t=%0d eviction_o=%0d eviction_t=%0d reorder_o=%0d reorder_t=%0d reqstall_o=%0d reqstall_t=%0d bridgestall_o=%0d bridgestall_t=%0d commitstall_o=%0d commitstall_t=%0d zero_o=%0d zero_t=%0d",
                 partial_o, partial_t, evict_o, evict_t, reorder_o, reorder_t,
                 req_stall_o, req_stall_t, bridge_stall_o, bridge_stall_t,
                 commit_stall_o, commit_stall_t, zero_o, zero_t);
        $display("PASS_M2197_C2_TSBG_SELECTIVE_BANK_FILL_DIRECTED bundles=3 commits_o=%0d commits_t=%0d identity_o=%0d identity_t=%0d partial_o=%0d partial_t=%0d refills_o=%0d refills_t=%0d scalar_o=%0d scalar_t=%0d products_o=%0d products_t=%0d",
                 commits_o, commits_t, identity_checks_o, identity_checks_t,
                 partial_o, partial_t, refill_o, refill_t, scalar_o, scalar_t,
                 products_o, products_t);
        $finish;
    end
endmodule

`default_nettype wire
