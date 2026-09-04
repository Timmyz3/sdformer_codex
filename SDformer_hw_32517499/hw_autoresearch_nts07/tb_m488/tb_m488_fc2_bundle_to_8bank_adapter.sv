`timescale 1ns/1ps
`default_nettype none

module tb_m488_fc2_bundle_to_8bank_adapter;
    localparam int REQUESTS = 96;
    localparam int QUEUE_DEPTH = 128;

    logic clk_core, rst_core;
    logic core_req_valid, core_req_ready, core_req_accept;
    logic [15:0] core_req_epoch;
    logic [2:0] core_req_slot;
    logic [31:0] core_req_generation;
    logic [23:0] core_req_tag;
    logic [2:0] core_req_output_block, core_req_slice;
    logic [3:0] core_req_source_count;
    logic [7:0] core_req_bank_valid;
    logic [11:0] core_req_source_channel [0:7];

    logic [7:0] bank_req_valid, bank_req_ready, bank_req_accept;
    logic [15:0] bank_req_epoch [0:7];
    logic [2:0] bank_req_slot [0:7];
    logic [31:0] bank_req_generation [0:7];
    logic [23:0] bank_req_tag [0:7];
    logic [2:0] bank_req_output_block [0:7], bank_req_slice [0:7];
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

    logic [15:0] q_epoch [0:7][0:QUEUE_DEPTH-1];
    logic [2:0] q_slot [0:7][0:QUEUE_DEPTH-1];
    logic [31:0] q_generation [0:7][0:QUEUE_DEPTH-1];
    logic [23:0] q_tag [0:7][0:QUEUE_DEPTH-1];
    integer q_due [0:7][0:QUEUE_DEPTH-1];
    integer q_head [0:7], q_tail [0:7];

    logic ref_valid [0:7];
    logic [15:0] ref_epoch [0:7];
    logic [31:0] ref_generation [0:7];
    logic [23:0] ref_tag [0:7];
    logic [7:0] ref_mask [0:7];

    integer cycle_count, issued, completed, expected_bank_beats;
    integer partial_dispatches, request_stalls, response_stalls;
    integer out_of_order_responses, last_response_generation;
    logic attack_mode;
    logic force_all_ready, force_equal_due;

    function automatic logic signed [7:0] expected_weight(
        input logic [23:0] tag,
        input logic [31:0] generation,
        input int bank,
        input int lane
    );
        integer value;
        begin
            value = (tag[7:0] + generation[7:0]
                + bank * 13 + lane * 7) % 127;
            expected_weight = value - 63;
        end
    endfunction

    function automatic logic [3:0] popcount8(input logic [7:0] value);
        logic [3:0] count;
        begin
            count = 0;
            for (int index = 0; index < 8; index++) count += value[index];
            return count;
        end
    endfunction

    always #1.5 clk_core = ~clk_core;

    always_comb begin
        for (int bank = 0; bank < 8; bank++) begin
            bank_req_ready[bank] = !attack_mode
                && (force_all_ready
                    || (((cycle_count + bank) % 7 != 0)
                        && ((cycle_count * 3 + bank) % 11 != 0)));
            bank_rsp_valid[bank] = 0;
            bank_rsp_epoch[bank] = 0;
            bank_rsp_slot[bank] = 0;
            bank_rsp_generation[bank] = 0;
            bank_rsp_tag[bank] = 0;
            for (int lane = 0; lane < 16; lane++)
                bank_rsp_weight[bank][lane] = 0;
            if (!attack_mode && q_head[bank] < q_tail[bank]
                    && q_due[bank][q_head[bank]] <= cycle_count) begin
                bank_rsp_valid[bank] = 1;
                bank_rsp_epoch[bank] = q_epoch[bank][q_head[bank]];
                bank_rsp_slot[bank] = q_slot[bank][q_head[bank]];
                bank_rsp_generation[bank]
                    = q_generation[bank][q_head[bank]];
                bank_rsp_tag[bank] = q_tag[bank][q_head[bank]];
                for (int lane = 0; lane < 16; lane++)
                    bank_rsp_weight[bank][lane] = expected_weight(
                        q_tag[bank][q_head[bank]],
                        q_generation[bank][q_head[bank]], bank, lane);
            end
        end
        if (attack_mode) begin
            bank_rsp_valid[3] = 1;
            bank_rsp_epoch[3] = 16'hdead;
            bank_rsp_slot[3] = 3'd7;
            bank_rsp_generation[3] = 32'hbad0_0001;
            bank_rsp_tag[3] = 24'hbadbad;
            for (int lane = 0; lane < 16; lane++)
                bank_rsp_weight[3][lane] = lane;
        end
        core_rsp_ready = !attack_mode && (cycle_count % 9 != 3)
            && (cycle_count % 13 != 4);
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
            issued <= 0;
            completed <= 0;
            expected_bank_beats <= 0;
            partial_dispatches <= 0;
            request_stalls <= 0;
            response_stalls <= 0;
            out_of_order_responses <= 0;
            last_response_generation <= -1;
            for (int bank = 0; bank < 8; bank++) begin
                q_head[bank] <= 0;
                q_tail[bank] <= 0;
                ref_valid[bank] <= 0;
                ref_epoch[bank] <= 0;
                ref_generation[bank] <= 0;
                ref_tag[bank] <= 0;
                ref_mask[bank] <= 0;
            end
        end else begin
            cycle_count <= cycle_count + 1;
            if (core_req_valid && !core_req_ready) request_stalls <= request_stalls + 1;
            if (core_rsp_valid && !core_rsp_ready) response_stalls <= response_stalls + 1;
            if (core_req_accept) begin
                if (ref_valid[core_req_slot]
                        && !(core_rsp_accept
                            && core_rsp_slot == core_req_slot))
                    $fatal(1, "reference slot reused while live");
                ref_valid[core_req_slot] <= 1;
                ref_epoch[core_req_slot] <= core_req_epoch;
                ref_generation[core_req_slot] <= core_req_generation;
                ref_tag[core_req_slot] <= core_req_tag;
                ref_mask[core_req_slot] <= core_req_bank_valid;
                issued <= issued + 1;
                expected_bank_beats <= expected_bank_beats
                    + popcount8(core_req_bank_valid);
                if (bank_req_accept != 0
                        && bank_req_accept != core_req_bank_valid)
                    partial_dispatches <= partial_dispatches + 1;
            end

            for (int bank = 0; bank < 8; bank++) begin
                if (bank_req_accept[bank]) begin
                    if (q_tail[bank] >= QUEUE_DEPTH)
                        $fatal(1, "bank queue overflow bank=%0d", bank);
                    q_epoch[bank][q_tail[bank]] <= bank_req_epoch[bank];
                    q_slot[bank][q_tail[bank]] <= bank_req_slot[bank];
                    q_generation[bank][q_tail[bank]]
                        <= bank_req_generation[bank];
                    q_tag[bank][q_tail[bank]] <= bank_req_tag[bank];
                    q_due[bank][q_tail[bank]] <= force_equal_due
                        ? cycle_count + 5
                        : cycle_count + 1
                            + ((bank_req_generation[bank] + bank * 3) % 7);
                    q_tail[bank] <= q_tail[bank] + 1;
                    if (bank_req_source_channel[bank][2:0] != bank[2:0])
                        $fatal(1, "bank/channel mismatch");
                end
                if (bank_rsp_accept[bank]) q_head[bank] <= q_head[bank] + 1;
            end

            if (core_rsp_accept) begin
                if (!ref_valid[core_rsp_slot])
                    $fatal(1, "response to non-live slot=%0d", core_rsp_slot);
                if (core_rsp_epoch != ref_epoch[core_rsp_slot]
                        || core_rsp_generation != ref_generation[core_rsp_slot]
                        || core_rsp_tag != ref_tag[core_rsp_slot]
                        || core_rsp_bank_valid != ref_mask[core_rsp_slot])
                    $fatal(1, "response identity mismatch slot=%0d", core_rsp_slot);
                for (int bank = 0; bank < 8; bank++) begin
                    for (int lane = 0; lane < 16; lane++) begin
                        if (core_rsp_bank_valid[bank]
                                && core_rsp_weight[bank][lane]
                                !== expected_weight(core_rsp_tag,
                                    core_rsp_generation, bank, lane))
                            $fatal(1, "weight mismatch slot=%0d bank=%0d lane=%0d",
                                core_rsp_slot, bank, lane);
                    end
                end
                if (last_response_generation >= 0
                        && core_rsp_generation < last_response_generation)
                    out_of_order_responses <= out_of_order_responses + 1;
                last_response_generation <= core_rsp_generation;
                if (!(core_req_accept && core_req_slot == core_rsp_slot))
                    ref_valid[core_rsp_slot] <= 0;
                completed <= completed + 1;
            end
        end
    end

    task automatic send_request(input integer number);
        logic [7:0] mask;
        integer slot;
        begin
            slot = number % 8;
            wait (!ref_valid[slot]);
            case (number % 7)
                0: mask = 8'hff;
                1: mask = 8'h81;
                2: mask = 8'h24;
                3: mask = 8'h10;
                4: mask = 8'h5a;
                5: mask = 8'h03;
                default: mask = 8'hc7;
            endcase
            @(negedge clk_core);
            core_req_valid = 1;
            core_req_epoch = 16'h1200 + number / 32;
            core_req_slot = slot;
            core_req_generation = number;
            core_req_tag = 24'h410000 + number;
            core_req_output_block = number % 8;
            core_req_slice = number % 6;
            core_req_bank_valid = mask;
            core_req_source_count = popcount8(mask);
            for (int bank = 0; bank < 8; bank++)
                core_req_source_channel[bank]
                    = {number[8:0], bank[2:0]};
            do @(posedge clk_core); while (!core_req_accept);
            @(negedge clk_core);
            core_req_valid = 0;
        end
    endtask

    m488_fc2_bundle_to_8bank_adapter dut (.*);

    initial begin
        clk_core = 0;
        rst_core = 1;
        core_req_valid = 0;
        core_req_epoch = 0;
        core_req_slot = 0;
        core_req_generation = 0;
        core_req_tag = 0;
        core_req_output_block = 0;
        core_req_slice = 0;
        core_req_source_count = 0;
        core_req_bank_valid = 0;
        attack_mode = 0;
        force_all_ready = 0;
        force_equal_due = 0;
        for (int bank = 0; bank < 8; bank++)
            core_req_source_channel[bank] = 0;

        repeat (5) @(posedge clk_core);
        rst_core = 0;
        for (int number = 0; number < REQUESTS; number++)
            send_request(number);

        wait (completed == REQUESTS);
        wait (!busy);

        // A full-eight response is made simultaneous across all banks.  A new
        // request for the retiring slot is held valid: it must stall on the
        // retirement edge and be accepted exactly one edge later.  This is the
        // loop-free replacement for the former combinational same-edge reuse.
        force_all_ready = 1;
        force_equal_due = 1;
        send_request(112);
        force_all_ready = 0;
        force_equal_due = 0;
        while (1) begin
            @(negedge clk_core);
            if (core_rsp_valid && core_rsp_ready && core_rsp_slot == 0) begin
                core_req_valid = 1;
                core_req_epoch = 16'h1300;
                core_req_slot = 0;
                core_req_generation = 32'd1000;
                core_req_tag = 24'h421000;
                core_req_output_block = 3;
                core_req_slice = 4;
                core_req_bank_valid = 8'h5a;
                core_req_source_count = popcount8(8'h5a);
                for (int bank = 0; bank < 8; bank++)
                    core_req_source_channel[bank] = {9'h155, bank[2:0]};
                #0.1;
                if (!core_rsp_accept || core_req_accept || protocol_error)
                    $fatal(1, "retiring slot did not produce a clean one-cycle stall");
                @(posedge clk_core);
                @(negedge clk_core);
                if (!core_req_accept || protocol_error)
                    $fatal(1, "retired slot was not reusable on next edge");
                @(posedge clk_core);
                @(negedge clk_core);
                core_req_valid = 0;
                break;
            end
        end

        wait (completed == REQUESTS + 2);
        wait (!busy);
        repeat (3) @(posedge clk_core);
        if (debug_bundle_request_count != REQUESTS + 2
                || debug_bundle_response_count != REQUESTS + 2)
            $fatal(1, "bundle counter mismatch req=%0d rsp=%0d",
                debug_bundle_request_count, debug_bundle_response_count);
        if (debug_bank_request_count != expected_bank_beats
                || debug_bank_response_count != expected_bank_beats)
            $fatal(1, "bank counter mismatch req=%0d rsp=%0d exp=%0d",
                debug_bank_request_count, debug_bank_response_count,
                expected_bank_beats);
        if (partial_dispatches == 0 || request_stalls == 0
                || response_stalls == 0 || out_of_order_responses == 0)
            $fatal(1, "coverage hole partial=%0d reqstall=%0d rspstall=%0d ooo=%0d",
                partial_dispatches, request_stalls, response_stalls,
                out_of_order_responses);

        @(negedge clk_core);
        attack_mode = 1;
        @(posedge clk_core);
        #0.1;
        if (!protocol_error || bank_rsp_accept != 0
                || core_req_accept || core_rsp_accept)
            $fatal(1, "attack was not quarantined atomically");
        @(negedge clk_core);
        attack_mode = 0;
        repeat (2) @(posedge clk_core);
        if (!protocol_error || !stale_response_seen)
            $fatal(1, "fault was not sticky");

        $display("PASS M488 bundle-to-8bank adapter requests=%0d bank_beats=%0d partial=%0d request_stalls=%0d response_stalls=%0d out_of_order=%0d attack=1 cycles=%0d headline=false system_speedup=false",
            issued, expected_bank_beats, partial_dispatches, request_stalls,
            response_stalls, out_of_order_responses, cycle_count);
        $finish;
    end

    initial begin
        repeat (20000) @(posedge clk_core);
        $fatal(1, "watchdog timeout");
    end
endmodule

`default_nettype wire
