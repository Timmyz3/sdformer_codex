`timescale 1ns/1ps
`default_nettype none

`include "vector_contract.svh"

// Checkpoint-bound proof for the M7 candidate: a 16-lane, ten-slot ATLIF
// slice.  The frozen vectors contain 32 lanes, so each checkpoint command is
// executed as two physical lane segments.  T=2 commands keep all ten slots
// occupied by packing five independent neuron groups; T=10 commands use one
// group and accumulate all temporal output slots in parallel.
module tb_checkpoint_atlif_dptme_l16_segmented;
    localparam int VECTOR_LANES = 32;
    localparam int LANES = 16;
    localparam int LANE_SEGMENTS = VECTOR_LANES / LANES;
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
    logic [(PACK_GROUPS*VECTOR_LANES*X_W)-1:0] x_mem [0:TOTAL_CYCLES-1];
    logic [(SLOTS*W_W)-1:0] weight_mem [0:TOTAL_CYCLES-1];
    logic [(SLOTS*ACC_W)-1:0] bias_mem [0:COMMANDS-1];
    logic [(SLOTS*ACC_W)-1:0] threshold_mem [0:COMMANDS-1];
    logic [(SLOTS*VECTOR_LANES*ACC_W)-1:0] expected_hidden_mem [0:COMMANDS-1];
    logic [(SLOTS*VECTOR_LANES)-1:0] expected_event_mem [0:COMMANDS-1];

    integer cycle_base;
    integer temporal_steps;
    integer current_segment;
    integer hidden_mismatches;
    integer event_mismatches;
    integer compared_hidden;
    integer compared_events;
    integer sampled_protocol_errors;
    integer t2_segment_commands;
    integer t10_segment_commands;

    always #1 clk_core = ~clk_core;

    always @(posedge clk_core) begin
        if (!rst_core && protocol_error)
            sampled_protocol_errors = sampled_protocol_errors + 1;
    end

    hitflow_dptme_array #(
        .LANES(LANES), .SLOTS(SLOTS), .PACK_GROUPS(PACK_GROUPS),
        .X_W(X_W), .W_W(W_W), .ACC_W(ACC_W), .TAG_W(TAG_W)
    ) dut (.*);

    task automatic map_lane_segment(input integer cycle_index, input integer segment);
        begin
            x_groups = '0;
            for (int group = 0; group < PACK_GROUPS; group++) begin
                for (int lane = 0; lane < LANES; lane++) begin
                    x_groups[((group*LANES + lane)*X_W) +: X_W] =
                        x_mem[cycle_index][((group*VECTOR_LANES + segment*LANES + lane)*X_W) +: X_W];
                end
            end
        end
    endtask

    task automatic drive_step(
        input integer cycle_index,
        input integer segment,
        input logic first,
        input logic last
    );
        begin
            repeat ((cycle_index + segment) % 3) @(posedge clk_core);
            @(negedge clk_core);
            map_lane_segment(cycle_index, segment);
            weight_slots = weight_mem[cycle_index];
            step_first = first;
            step_last = last;
            step_valid = 1'b1;
            #0.1;
            while (!step_ready) begin
                @(negedge clk_core);
                #0.1;
            end
            @(posedge clk_core);
            #0.1;
            step_valid = 1'b0;
        end
    endtask

    initial begin
`ifdef SIMULATOR_VCS
        $display("SIMULATOR=Synopsys VCS");
`else
        $fatal(1, "M7 L16 checkpoint regression requires Synopsys VCS identity");
`endif
`ifdef SVA_RUNTIME_ENABLED
        $display("ASSERTIONS=enabled");
`else
        $fatal(1, "M7 L16 checkpoint regression requires bound SVA");
`endif
        if (VECTOR_LANES % LANES != 0)
            $fatal(1, "checkpoint vector lanes must divide physical lanes");
        $readmemh("meta.mem", meta_mem);
        $readmemh("x.mem", x_mem);
        $readmemh("weight.mem", weight_mem);
        $readmemh("bias.mem", bias_mem);
        $readmemh("threshold.mem", threshold_mem);
        $readmemh("expected_hidden.mem", expected_hidden_mem);
        $readmemh("expected_event.mem", expected_event_mem);

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
        t2_segment_commands = 0;
        t10_segment_commands = 0;
        repeat (3) @(posedge clk_core);
        #0.1 rst_core = 1'b0;

        for (int command = 0; command < COMMANDS; command++) begin
            mode_t2 = meta_mem[command][0];
            temporal_steps = meta_mem[command][15:8];
            group_valid = '1;
            bias_slots = bias_mem[command];
            threshold_slots = threshold_mem[command];
            for (current_segment = 0; current_segment < LANE_SEGMENTS; current_segment++) begin
                step_tag = meta_mem[command][31:16] ^ TAG_W'(current_segment << 14);
                for (int step = 0; step < temporal_steps; step++) begin
                    drive_step(cycle_base + step, current_segment,
                               step == 0, step == temporal_steps - 1);
                end
                if (!out_valid) $fatal(1, "command %0d segment %0d missing out_valid", command, current_segment);
                if (out_tag != step_tag) $fatal(1, "command %0d segment %0d tag mismatch", command, current_segment);
                if (out_slot_valid != {SLOTS{1'b1}})
                    $fatal(1, "command %0d segment %0d slot-valid mismatch", command, current_segment);
                for (int hold = 0; hold < ((command + current_segment) % 4); hold++) begin
                    logic [(SLOTS*LANES)-1:0] held_events;
                    logic [(SLOTS*LANES*ACC_W)-1:0] held_hidden;
                    held_events = out_events;
                    held_hidden = out_hidden;
                    @(posedge clk_core);
                    #0.1;
                    if (!out_valid || out_events !== held_events || out_hidden !== held_hidden)
                        $fatal(1, "command %0d segment %0d output changed under backpressure",
                               command, current_segment);
                end
                for (int slot = 0; slot < SLOTS; slot++) begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        int vector_index;
                        int dut_index;
                        vector_index = slot * VECTOR_LANES + current_segment * LANES + lane;
                        dut_index = slot * LANES + lane;
                        compared_hidden = compared_hidden + 1;
                        compared_events = compared_events + 1;
                        if (out_hidden[(dut_index*ACC_W) +: ACC_W] !==
                            expected_hidden_mem[command][(vector_index*ACC_W) +: ACC_W])
                            hidden_mismatches = hidden_mismatches + 1;
                        if (out_events[dut_index] !== expected_event_mem[command][vector_index])
                            event_mismatches = event_mismatches + 1;
                    end
                end
                if (mode_t2) t2_segment_commands = t2_segment_commands + 1;
                else t10_segment_commands = t10_segment_commands + 1;
                out_ready = 1'b1;
                @(posedge clk_core);
                #0.1 out_ready = 1'b0;
            end
            cycle_base = cycle_base + temporal_steps;
        end
        if (protocol_error || sampled_protocol_errors != 0)
            $fatal(1, "legal stream sampled protocol errors=%0d", sampled_protocol_errors);
        $display("ATLIF_L16_SEGMENTED_RESULT commands=%0d lane_segments=%0d t10_segment_commands=%0d t2_segment_commands=%0d hidden=%0d hidden_mismatches=%0d events=%0d event_mismatches=%0d sampled_protocol_errors=%0d",
                 COMMANDS, LANE_SEGMENTS, t10_segment_commands, t2_segment_commands,
                 compared_hidden, hidden_mismatches, compared_events,
                 event_mismatches, sampled_protocol_errors);
        if (hidden_mismatches != 0 || event_mismatches != 0)
            $fatal(1, "checkpoint ATLIF L16 segmented mismatch");
        $display("PASS: Synopsys VCS checkpoint-bound ATLIF L16 S10/S2 exact");
        $finish;
    end
endmodule

`default_nettype wire
