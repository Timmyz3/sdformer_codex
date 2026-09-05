`timescale 1ns/1ps
`default_nettype none
// Actual M803 adapter + borrowed-payload consumer, not a stub SRAM wrapper.
// A deterministic two-slot bank model provides unequal request readiness and
// response skew. All numerical comparisons use a separate integer reference.
module tb_m2243_borrowed_weight_consumers;
    localparam int N = 256;
    logic clk_core = 0, rst_core = 1;
    always #1.5 clk_core = ~clk_core;
    int cycle = 0, sent = 0, retired = 0, updates = 0;
    int empty_beats = 0, four_consumers = 0, stalls = 0;
    int concurrent_refills = 0, positive_128 = 0;
    int masked_mode = 0, expected_bank_reads = 0;
    bit slot_busy [0:1];
    bit outstanding [0:7][0:1];
    int due [0:7][0:1], transaction [0:7][0:1];
    bit delivered [0:N-1][0:3];
    bit retired_seen [0:N-1];
    int delivered_count [0:N-1];
    int expected_updates = 0;

    logic core_req_valid, core_req_ready, core_req_accept;
    logic [15:0] core_req_epoch = 16'h2243;
    logic [2:0] core_req_slot, core_req_output_block, core_req_slice;
    logic [31:0] core_req_generation;
    logic [23:0] core_req_tag;
    logic [3:0] core_req_source_count;
    logic [7:0] core_req_bank_valid;
    logic [11:0] core_req_source_channel [0:7];
    logic [7:0] bank_req_valid, bank_req_ready, bank_req_accept;
    logic [15:0] bank_req_epoch [0:7];
    logic [2:0] bank_req_slot [0:7], bank_req_output_block [0:7], bank_req_slice [0:7];
    logic [31:0] bank_req_generation [0:7];
    logic [23:0] bank_req_tag [0:7];
    logic [11:0] bank_req_source_channel [0:7];
    logic [7:0] bank_rsp_valid, bank_rsp_ready, bank_rsp_accept;
    logic [15:0] bank_rsp_epoch [0:7];
    logic [2:0] bank_rsp_slot [0:7];
    logic [31:0] bank_rsp_generation [0:7];
    logic [23:0] bank_rsp_tag [0:7];
    logic signed [7:0] bank_rsp_weight [0:7][0:15];
    logic core_rsp_valid, core_rsp_ready, core_rsp_accept;
    logic [15:0] core_rsp_epoch;
    logic [2:0] core_rsp_slot;
    logic [31:0] core_rsp_generation;
    logic [23:0] core_rsp_tag;
    logic [7:0] core_rsp_bank_valid;
    logic signed [7:0] core_rsp_weight [0:7][0:15];
    logic protocol_error, stale_response_seen, busy;
    logic [3:0] debug_live_slots;
    logic [31:0] debug_bundle_request_count, debug_bank_request_count;
    logic [31:0] debug_bank_response_count, debug_bundle_response_count;
    m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter adapter (.*);

    logic meta_ready, bridge_valid, bridge_ready, beat_done, consumer_error;
    logic [7:0] active [0:3], negative [0:3];
    logic [1:0] bridge_context;
    logic [5:0] bridge_group;
    logic bridge_half;
    logic [2:0] bridge_slice;
    logic [23:0] bridge_tag;
    logic [7:0] bridge_bank_valid;
    logic signed [8:0] bridge_effective_weight [0:7][0:15];
    m2243_c2_borrowed_weight_consumers consumer (
        .clk_core, .rst_core,
        .meta_valid(core_rsp_valid), .meta_ready,
        .meta_epoch(core_rsp_epoch), .meta_generation(core_rsp_generation),
        .meta_tag(core_rsp_tag), .meta_group(6'(core_rsp_tag % 48)),
        .meta_half(core_rsp_tag[0]), .meta_slice(3'(core_rsp_tag % 6)),
        .meta_active(active), .meta_negative(negative),
        .rsp_valid(core_rsp_valid), .rsp_ready(core_rsp_ready),
        .rsp_epoch(core_rsp_epoch), .rsp_generation(core_rsp_generation),
        .rsp_tag(core_rsp_tag), .rsp_bank_valid(core_rsp_bank_valid),
        .rsp_weight(core_rsp_weight), .bridge_valid, .bridge_ready,
        .bridge_context, .bridge_group, .bridge_half, .bridge_slice,
        .bridge_tag, .bridge_bank_valid, .bridge_effective_weight,
        .beat_done, .protocol_error(consumer_error));

    function automatic int weight_value(input int id, b, l);
        return ((id * 19 + b * 37 + l * 11) % 256) - 128;
    endfunction
    function automatic logic [7:0] mask_value(input int id, c);
        if (id % 17 == 0) return 0;
        if (id % 4 == 0) return 8'hff;
        if (c > id % 4) return 0;
        return 8'((id * 53 + c * 73) | 1);
    endfunction
    function automatic logic [7:0] sign_value(input int id, c);
        return 8'(id * 11 + c * 59 + 1);
    endfunction

    always_comb begin
        core_req_valid = !rst_core && sent < N && !slot_busy[sent % 2];
        core_req_slot = 3'(sent % 2);
        core_req_generation = sent + 1;
        core_req_tag = 24'(sent);
        core_req_output_block = 3'(sent % 2);
        core_req_slice = 3'(sent % 6);
        core_req_bank_valid = 0;
        for (int c = 0; c < 4; c++)
            core_req_bank_valid |= mask_value(sent, c);
        if (!masked_mode) core_req_bank_valid = 8'hff;
        // Directed empty-response drainage test. Production empty-union work
        // would issue no request at all; this one-bank probe is not a benefit.
        if (core_req_bank_valid == 0) core_req_bank_valid = 1;
        core_req_source_count = 4'($countones(core_req_bank_valid));
        bridge_ready = cycle % 11 != 3 && cycle % 11 != 4;
        for (int c = 0; c < 4; c++) begin
            active[c] = mask_value(int'(core_rsp_tag), c);
            negative[c] = sign_value(int'(core_rsp_tag), c);
        end
        for (int b = 0; b < 8; b++) begin
            core_req_source_channel[b] = 12'((sent % 48) * 16 + b);
            bank_req_ready[b] = !rst_core && cycle % 7 != (2 * b) % 7;
        end
    end

    // Drive responses on falling edges; preserve a response until accepted.
    // No late sampling of combinational accept after the DUT's NBA updates.
    always @(negedge clk_core) begin
        if (rst_core) begin
            bank_rsp_valid = 0;
        end else begin
            for (int b = 0; b < 8; b++) begin
                if (!bank_rsp_valid[b]) begin
                    for (int s = 1; s >= 0; s--) begin
                        if (!bank_rsp_valid[b] && outstanding[b][s] && due[b][s] <= cycle) begin
                            bank_rsp_valid[b] = 1;
                            bank_rsp_epoch[b] = 16'h2243;
                            bank_rsp_slot[b] = 3'(s);
                            bank_rsp_tag[b] = 24'(transaction[b][s]);
                            bank_rsp_generation[b] = transaction[b][s] + 1;
                            for (int l = 0; l < 16; l++)
                                bank_rsp_weight[b][l] = 8'(weight_value(transaction[b][s], b, l));
                        end
                    end
                end
            end
        end
    end

    always @(posedge clk_core) begin : score
        int id, c, wanted, expected, actual, value;
        logic [7:0] m, signs;
        if (!rst_core) begin
            cycle <= cycle + 1;
            if (cycle > 30000) $fatal(1, "M2243 timeout sent=%0d retired=%0d", sent, retired);
            if (protocol_error || stale_response_seen || consumer_error)
                $fatal(1, "M2243 unexpected protocol error");
            if (core_req_accept) begin
                slot_busy[sent % 2] <= 1;
                expected_bank_reads += $countones(core_req_bank_valid);
                sent <= sent + 1;
            end
            for (int b = 0; b < 8; b++) begin
                if (bank_req_accept[b]) begin
                    if (outstanding[b][bank_req_slot[b]]) $fatal(1, "duplicate bank request");
                    outstanding[b][bank_req_slot[b]] = 1;
                    transaction[b][bank_req_slot[b]] = int'(bank_req_tag[b]);
                    due[b][bank_req_slot[b]] = cycle + 2 + ((b * 3 + int'(bank_req_tag[b])) % 9);
                end
                if (bank_rsp_accept[b]) begin
                    outstanding[b][bank_rsp_slot[b]] = 0;
                    bank_rsp_valid[b] <= 0;
                end
            end
            if (bridge_valid && !bridge_ready) stalls++;
            if (core_rsp_valid && !core_rsp_ready && bank_req_accept != 0) concurrent_refills++;
            if (bridge_valid && bridge_ready) begin
                id = int'(bridge_tag);
                c = int'(bridge_context);
                if (id >= N || delivered[id][c]) $fatal(1, "duplicate/invalid consumer");
                m = mask_value(id, c);
                signs = sign_value(id, c);
                if (m == 0 || bridge_bank_valid != m || bridge_group != id % 48
                    || bridge_half != id % 2 || bridge_slice != id % 6)
                    $fatal(1, "M2243 metadata mismatch");
                for (int l = 0; l < 16; l++) begin
                    expected = 0;
                    actual = 0;
                    for (int b = 0; b < 8; b++) begin
                        value = m[b] ? (signs[b] ? -weight_value(id,b,l) : weight_value(id,b,l)) : 0;
                        if (int'(bridge_effective_weight[b][l]) != value)
                            $fatal(1, "M2243 signed product mismatch");
                        if (value == 128) positive_128++;
                        expected += value;
                        actual += int'(bridge_effective_weight[b][l]);
                    end
                    if (actual != expected) $fatal(1, "M2243 independent Acc24 update mismatch");
                end
                delivered[id][c] = 1;
                delivered_count[id]++;
                updates++;
            end
            if (core_rsp_accept) begin
                id = int'(core_rsp_tag);
                if (id >= N || retired_seen[id]) $fatal(1, "duplicate/invalid retirement");
                retired_seen[id] = 1;
                wanted = 0;
                for (int k = 0; k < 4; k++) if (mask_value(id,k) != 0) wanted++;
                if (delivered_count[id] != wanted || !beat_done)
                    $fatal(1, "M2243 premature payload release");
                if (wanted == 0) empty_beats++;
                if (wanted == 4) four_consumers++;
                slot_busy[core_rsp_slot] <= 0;
                retired <= retired + 1;
            end
            if (retired == N) begin
                for (int k = 0; k < N; k++)
                    if (!retired_seen[k]) $fatal(1, "missing retirement");
                if (sent != N || updates != expected_updates
                    || debug_bank_request_count != expected_bank_reads || debug_bank_response_count != expected_bank_reads
                    || debug_bundle_response_count != N || stalls == 0 || empty_beats == 0
                    || four_consumers == 0 || positive_128 == 0 || concurrent_refills == 0)
                    $fatal(1, "M2243 coverage/count gap");
                $display("PASS_M2243_M803_BORROWED_CONSUMERS masked=%0d beats=%0d reads=%0d updates=%0d empty=%0d full4=%0d stalls=%0d overlap=%0d positive128=%0d", masked_mode, retired, expected_bank_reads, updates, empty_beats, four_consumers, stalls, concurrent_refills, positive_128);
                $finish;
            end
        end
    end

    assert property (@(posedge clk_core) disable iff (rst_core)
        bridge_valid && !bridge_ready |=> bridge_valid &&
        $stable({bridge_context, bridge_group, bridge_half, bridge_slice, bridge_tag, bridge_bank_valid}));
    for (genvar b = 0; b < 8; b++) begin
        for (genvar l = 0; l < 16; l++) begin
            assert property (@(posedge clk_core) disable iff (rst_core)
                bridge_valid && !bridge_ready |=> $stable(bridge_effective_weight[b][l]));
        end
    end
    initial begin
        if ($value$plusargs("MASKED=%d", masked_mode)) begin end
        for (int s = 0; s < 2; s++) begin
            slot_busy[s] = 0;
            for (int b = 0; b < 8; b++) outstanding[b][s] = 0;
        end
        for (int id = 0; id < N; id++) begin
            delivered_count[id] = 0;
            for (int c = 0; c < 4; c++) if (mask_value(id,c) != 0) expected_updates++;
        end
        bank_rsp_valid = 0;
        repeat (5) @(negedge clk_core);
        rst_core = 0;
    end
endmodule
`default_nettype wire
