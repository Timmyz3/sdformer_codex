`timescale 1ns/1ps
`default_nettype none

`include "vector_contract.svh"

module tb_checkpoint_atlif_dptme;
    localparam int LANES = 32;
    localparam int SLOTS = 10;
    localparam int PACK_GROUPS = 5;
    localparam int X_W = 8;
    localparam int W_W = 8;
    localparam int ACC_W = 24;
    localparam int TAG_W = 16;
    localparam int COMMANDS = `ATLIF_COMMANDS;
    localparam int TOTAL_CYCLES = `ATLIF_TOTAL_CYCLES;

    logic clk_core = 1'b0;
    logic rst_core;
    logic step_valid;
    logic step_ready;
    logic mode_t2;
    logic step_first;
    logic step_last;
    logic [PACK_GROUPS-1:0] group_valid;
    logic [(PACK_GROUPS*LANES*X_W)-1:0] x_groups;
    logic [(SLOTS*W_W)-1:0] weight_slots;
    logic [(SLOTS*ACC_W)-1:0] bias_slots;
    logic [(SLOTS*ACC_W)-1:0] threshold_slots;
    logic [TAG_W-1:0] step_tag;
    logic out_valid;
    logic out_ready;
    logic [(SLOTS*LANES)-1:0] out_events;
    logic [(SLOTS*LANES*ACC_W)-1:0] out_hidden;
    logic [SLOTS-1:0] out_slot_valid;
    logic [TAG_W-1:0] out_tag;
    logic protocol_error;

    logic [47:0] meta_mem [0:COMMANDS-1];
    logic [(PACK_GROUPS*LANES*X_W)-1:0] x_mem [0:TOTAL_CYCLES-1];
    logic [(SLOTS*W_W)-1:0] weight_mem [0:TOTAL_CYCLES-1];
    logic [(SLOTS*ACC_W)-1:0] bias_mem [0:COMMANDS-1];
    logic [(SLOTS*ACC_W)-1:0] threshold_mem [0:COMMANDS-1];
    logic [(SLOTS*LANES*ACC_W)-1:0] expected_hidden_mem [0:COMMANDS-1];
    logic [(SLOTS*LANES)-1:0] expected_event_mem [0:COMMANDS-1];

    integer cycle_base;
    integer temporal_steps;
    integer hidden_mismatches;
    integer event_mismatches;
    integer compared_hidden;
    integer compared_events;
    integer sampled_protocol_errors;

    always #1 clk_core = ~clk_core;

    // protocol_error is a combinational reject indication. Count only values
    // visible at the synchronous interface sampling edge.
    always @(posedge clk_core) begin
        if (!rst_core && protocol_error)
            sampled_protocol_errors = sampled_protocol_errors + 1;
    end

    hitflow_dptme_array #(
        .LANES(LANES), .SLOTS(SLOTS), .PACK_GROUPS(PACK_GROUPS),
        .X_W(X_W), .W_W(W_W), .ACC_W(ACC_W), .TAG_W(TAG_W)
    ) dut (.*);

    task automatic drive_step(input integer cycle_index, input logic first, input logic last);
        begin
            repeat ((cycle_index % 3)) @(posedge clk_core);
            @(negedge clk_core);
            x_groups = x_mem[cycle_index];
            weight_slots = weight_mem[cycle_index];
            step_first = first;
            step_last = last;
            step_valid = 1'b1;
            #0.1;
`ifdef ATLIF_TB_PROGRESS
            if (cycle_index < 3)
                $display("ATLIF_TB_PROGRESS drive cycle=%0d first=%0b last=%0b pre ready=%0b busy=%0b seen=%0d err=%0b",
                         cycle_index, first, last, step_ready, dut.busy_q,
                         dut.steps_seen_q, protocol_error);
`endif
            // Drive only on the opposite edge and sample ready in the same phase.
            // This prevents both active/NBA races and accidental same-edge accepts.
            while (!step_ready) begin
                @(negedge clk_core);
                #0.1;
`ifdef ATLIF_TB_PROGRESS
                if (cycle_index < 3)
                    $display("ATLIF_TB_PROGRESS drive cycle=%0d neg ready=%0b busy=%0b seen=%0d err=%0b",
                             cycle_index, step_ready, dut.busy_q,
                             dut.steps_seen_q, protocol_error);
`endif
            end
            @(posedge clk_core);
            #0.1;
`ifdef ATLIF_TB_PROGRESS
            if (cycle_index < 3)
                $display("ATLIF_TB_PROGRESS drive cycle=%0d accepted ready=%0b busy=%0b seen=%0d err=%0b",
                         cycle_index, step_ready, dut.busy_q,
                         dut.steps_seen_q, protocol_error);
`endif
            step_valid = 1'b0;
        end
    endtask

    initial begin
`ifdef SIMULATOR_ICARUS
        $display("SIMULATOR=icarus");
`elsif SIMULATOR_VERILATOR
        $display("SIMULATOR=verilator");
`else
        $fatal(1, "simulator identity define missing");
`endif
`ifdef SVA_RUNTIME_ENABLED
        $display("ASSERTIONS=enabled");
`else
        $display("ASSERTIONS=not_bound");
`endif
        $readmemh("meta.mem", meta_mem);
`ifdef ATLIF_TB_PROGRESS
        $display("ATLIF_TB_PROGRESS meta_loaded");
`endif
        $readmemh("x.mem", x_mem);
`ifdef ATLIF_TB_PROGRESS
        $display("ATLIF_TB_PROGRESS x_loaded");
`endif
        $readmemh("weight.mem", weight_mem);
`ifdef ATLIF_TB_PROGRESS
        $display("ATLIF_TB_PROGRESS weight_loaded");
`endif
        $readmemh("bias.mem", bias_mem);
        $readmemh("threshold.mem", threshold_mem);
        $readmemh("expected_hidden.mem", expected_hidden_mem);
        $readmemh("expected_event.mem", expected_event_mem);
`ifdef ATLIF_TB_PROGRESS
        $display("ATLIF_TB_PROGRESS all_vectors_loaded");
`endif

        rst_core = 1'b1;
        step_valid = 1'b0;
        mode_t2 = 1'b0;
        step_first = 1'b0;
        step_last = 1'b0;
        group_valid = '1;
        x_groups = '0;
        weight_slots = '0;
        bias_slots = '0;
        threshold_slots = '0;
        step_tag = '0;
        out_ready = 1'b0;
        cycle_base = 0;
        hidden_mismatches = 0;
        event_mismatches = 0;
        compared_hidden = 0;
        compared_events = 0;
        sampled_protocol_errors = 0;
        repeat (3) @(posedge clk_core);
        #0.1 rst_core = 1'b0;

        for (int command = 0; command < COMMANDS; command++) begin
`ifdef ATLIF_TB_PROGRESS
            if ((command % 8) == 0)
                $display("ATLIF_TB_PROGRESS command=%0d cycle_base=%0d", command, cycle_base);
`endif
            mode_t2 = meta_mem[command][0];
            temporal_steps = meta_mem[command][15:8];
            step_tag = meta_mem[command][31:16];
            group_valid = '1;
            bias_slots = bias_mem[command];
            threshold_slots = threshold_mem[command];
            for (int step = 0; step < temporal_steps; step++) begin
                drive_step(cycle_base + step, step == 0, step == temporal_steps - 1);
            end
            if (!out_valid) $fatal(1, "command %0d missing out_valid", command);
            if (out_tag != step_tag) $fatal(1, "command %0d tag mismatch", command);
            if (out_slot_valid != {SLOTS{1'b1}}) $fatal(1, "command %0d slot-valid mismatch", command);
            for (int hold = 0; hold < (command % 4); hold++) begin
                logic [(SLOTS*LANES)-1:0] held_events;
                logic [(SLOTS*LANES*ACC_W)-1:0] held_hidden;
                held_events = out_events;
                held_hidden = out_hidden;
                @(posedge clk_core);
                #0.1;
                if (!out_valid || out_events !== held_events || out_hidden !== held_hidden)
                    $fatal(1, "command %0d output changed under backpressure", command);
            end
            for (int slot = 0; slot < SLOTS; slot++) begin
                for (int lane = 0; lane < LANES; lane++) begin
                    int index;
                    index = slot * LANES + lane;
                    compared_hidden = compared_hidden + 1;
                    compared_events = compared_events + 1;
                    if (out_hidden[(index*ACC_W) +: ACC_W] !==
                        expected_hidden_mem[command][(index*ACC_W) +: ACC_W]) begin
                        hidden_mismatches = hidden_mismatches + 1;
                        if (hidden_mismatches <= 4)
                            $display("HIDDEN_MISMATCH command=%0d slot=%0d lane=%0d got=%h expected=%h",
                                     command, slot, lane,
                                     out_hidden[(index*ACC_W) +: ACC_W],
                                     expected_hidden_mem[command][(index*ACC_W) +: ACC_W]);
                    end
                    if (out_events[index] !== expected_event_mem[command][index]) begin
                        event_mismatches = event_mismatches + 1;
                        if (event_mismatches <= 4)
                            $display("EVENT_MISMATCH command=%0d slot=%0d lane=%0d got=%b expected=%b",
                                     command, slot, lane, out_events[index], expected_event_mem[command][index]);
                    end
                end
            end
            out_ready = 1'b1;
            @(posedge clk_core);
            #0.1 out_ready = 1'b0;
            cycle_base = cycle_base + temporal_steps;
        end
        if (protocol_error) $fatal(1, "protocol_error asserted at completion");
        if (sampled_protocol_errors != 0)
            $fatal(1, "legal checkpoint stream sampled %0d protocol errors",
                   sampled_protocol_errors);
        $display("ATLIF_DPTME_RESULT commands=%0d hidden=%0d hidden_mismatches=%0d events=%0d event_mismatches=%0d sampled_protocol_errors=%0d",
                 COMMANDS, compared_hidden, hidden_mismatches, compared_events,
                 event_mismatches, sampled_protocol_errors);
        if (hidden_mismatches != 0 || event_mismatches != 0) $fatal(1, "checkpoint ATLIF DP-TME mismatch");
        $display("PASS: checkpoint-bound ATLIF DP-TME RTL exact");
        $finish;
    end
endmodule

`default_nettype wire
