`timescale 1ns/1ps
`default_nettype none

module tb_hitflow_dptme_array;
    localparam int LANES = 4;
    localparam int SLOTS = 10;
    localparam int PACK_GROUPS = 5;
    localparam int X_W = 8;
    localparam int W_W = 8;
    localparam int ACC_W = 24;
    localparam int TAG_W = 16;

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

    integer expected;
    integer observed;
    integer sampled_protocol_errors;
    integer tag_rejects;
    integer early_last_rejects;
    integer single_step_rejects;
    integer state_advance_errors;

    logic snapshot_busy;
    logic snapshot_mode;
    logic [3:0] snapshot_steps_seen;
    logic [PACK_GROUPS-1:0] snapshot_group_valid;
    logic [TAG_W-1:0] snapshot_tag;
    logic snapshot_out_valid;
    logic [(SLOTS*LANES)-1:0] snapshot_events;
    logic [(SLOTS*LANES*ACC_W)-1:0] snapshot_hidden;
    logic [SLOTS-1:0] snapshot_slot_valid;
    logic [TAG_W-1:0] snapshot_out_tag;

    always #1 clk_core = ~clk_core;

    always @(posedge clk_core) begin
        if (!rst_core && protocol_error)
            sampled_protocol_errors = sampled_protocol_errors + 1;
    end

    hitflow_dptme_array #(
        .LANES(LANES), .SLOTS(SLOTS), .PACK_GROUPS(PACK_GROUPS),
        .X_W(X_W), .W_W(W_W), .ACC_W(ACC_W), .TAG_W(TAG_W)
    ) dut (.*);

    task automatic drive_step(input logic first, input logic last);
        begin
            step_first = first;
            step_last = last;
            step_valid = 1'b1;
            do @(posedge clk_core); while (!step_ready);
            #0.1;
            step_valid = 1'b0;
        end
    endtask

    task automatic check(input logic condition, input string message);
        if (!condition) $fatal(1, "%s", message);
    endtask

    task automatic capture_state;
        begin
            snapshot_busy = dut.busy_q;
            snapshot_mode = dut.mode_q;
            snapshot_steps_seen = dut.steps_seen_q;
            snapshot_group_valid = dut.group_valid_q;
            snapshot_tag = dut.tag_q;
            snapshot_out_valid = out_valid;
            snapshot_events = out_events;
            snapshot_hidden = out_hidden;
            snapshot_slot_valid = out_slot_valid;
            snapshot_out_tag = out_tag;
        end
    endtask

    task automatic account_state_change;
        begin
            if (dut.busy_q !== snapshot_busy || dut.mode_q !== snapshot_mode ||
                dut.steps_seen_q !== snapshot_steps_seen ||
                dut.group_valid_q !== snapshot_group_valid ||
                dut.tag_q !== snapshot_tag || out_valid !== snapshot_out_valid ||
                out_events !== snapshot_events || out_hidden !== snapshot_hidden ||
                out_slot_valid !== snapshot_slot_valid || out_tag !== snapshot_out_tag)
                state_advance_errors = state_advance_errors + 1;
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
        sampled_protocol_errors = 0;
        tag_rejects = 0;
        early_last_rejects = 0;
        single_step_rejects = 0;
        state_advance_errors = 0;
        repeat (3) @(posedge clk_core);
        #0.1 rst_core = 1'b0;

        $display("阶段1：T10广播输入和多拍累加");
        mode_t2 = 1'b0;
        step_tag = 16'h1010;
        for (int lane = 0; lane < LANES; lane++) begin
            x_groups[(lane*X_W) +: X_W] = lane + 1;
        end
        for (int slot = 0; slot < SLOTS; slot++) begin
            weight_slots[(slot*W_W) +: W_W] = slot - 4;
            bias_slots[(slot*ACC_W) +: ACC_W] = slot;
            threshold_slots[(slot*ACC_W) +: ACC_W] = 0;
        end
        drive_step(1'b1, 1'b0);
        for (int lane = 0; lane < LANES; lane++) begin
            x_groups[(lane*X_W) +: X_W] = lane + 2;
        end
        drive_step(1'b0, 1'b0);
        weight_slots = '0;
        repeat (7) drive_step(1'b0, 1'b0);
        drive_step(1'b0, 1'b1);
        check(out_valid, "T10最后一步后必须输出valid");
        check(out_tag == 16'h1010, "T10输出tag错误");
        check(out_slot_valid == 10'h3ff, "T10十个slot都应有效");
        for (int slot = 0; slot < SLOTS; slot++) begin
            for (int lane = 0; lane < LANES; lane++) begin
                expected = slot + (slot - 4) * ((lane + 1) + (lane + 2));
                observed = $signed(out_hidden[((slot*LANES+lane)*ACC_W) +: ACC_W]);
                check(observed == expected, "T10 hidden累加错误");
                check(out_events[slot*LANES+lane] == (expected >= 0), "T10 event比较错误");
            end
        end
        out_ready = 1'b1;
        @(posedge clk_core);
        #0.1 out_ready = 1'b0;

        $display("阶段2：T2五组独立输入和尾组mask");
        mode_t2 = 1'b1;
        group_valid = 5'b00111;
        step_tag = 16'h2020;
        for (int group = 0; group < PACK_GROUPS; group++) begin
            for (int lane = 0; lane < LANES; lane++) begin
                x_groups[((group*LANES+lane)*X_W) +: X_W] = group + lane + 1;
            end
        end
        for (int slot = 0; slot < SLOTS; slot++) begin
            weight_slots[(slot*W_W) +: W_W] = (slot & 1) ? -2 : 3;
            bias_slots[(slot*ACC_W) +: ACC_W] = 1;
            threshold_slots[(slot*ACC_W) +: ACC_W] = 0;
        end
        drive_step(1'b1, 1'b0);
        for (int group = 0; group < PACK_GROUPS; group++) begin
            for (int lane = 0; lane < LANES; lane++) begin
                x_groups[((group*LANES+lane)*X_W) +: X_W] = group + lane + 2;
            end
        end
        drive_step(1'b0, 1'b1);
        check(out_slot_valid == 10'b00_00_11_11_11, "T2 slot mask错误");
        for (int slot = 0; slot < 6; slot++) begin
            for (int lane = 0; lane < LANES; lane++) begin
                expected = 1 + ((slot & 1) ? -2 : 3) *
                           (((slot/2) + lane + 1) + ((slot/2) + lane + 2));
                observed = $signed(out_hidden[((slot*LANES+lane)*ACC_W) +: ACC_W]);
                check(observed == expected, "T2 hidden累加错误");
            end
        end
        for (int bit_index = 6*LANES; bit_index < SLOTS*LANES; bit_index++) begin
            check(!out_events[bit_index], "无效T2组event必须为0");
        end
        out_ready = 1'b1;
        @(posedge clk_core);
        #0.1 out_ready = 1'b0;

        $display("阶段3：协议错配拒绝");
        mode_t2 = 1'b1;
        group_valid = '1;
        step_tag = 16'h3030;
        drive_step(1'b1, 1'b0);
        capture_state();
        step_tag = 16'h3031;
        step_first = 1'b0;
        step_last = 1'b1;
        step_valid = 1'b1;
        #0.1;
        check(protocol_error, "进行中命令tag变化必须报错");
        check(!step_ready, "协议错误步骤不得接收");
        @(posedge clk_core);
        #0.1;
        check(sampled_protocol_errors == 1, "tag错误必须在采样沿被协议监视器捕获");
        account_state_change();
        check(state_advance_errors == 0, "tag错误不得推进命令状态");
        tag_rejects = tag_rejects + 1;
        step_valid = 1'b0;
        step_tag = 16'h3030;
        drive_step(1'b0, 1'b1);
        out_ready = 1'b1;
        @(posedge clk_core);
        #0.1 out_ready = 1'b0;

        $display("阶段4：T10提前last和单步命令拒绝");
        mode_t2 = 1'b0;
        group_valid = '1;
        step_tag = 16'h4040;
        drive_step(1'b1, 1'b0);
        capture_state();
        step_first = 1'b0;
        step_last = 1'b1;
        step_valid = 1'b1;
        #0.1;
        check(protocol_error, "T10第二步提前last必须报错");
        check(!step_ready, "T10提前last不得接收");
        @(posedge clk_core);
        #0.1;
        check(sampled_protocol_errors == 2, "提前last必须在采样沿被协议监视器捕获");
        account_state_change();
        check(state_advance_errors == 0, "提前last不得推进命令状态");
        early_last_rejects = early_last_rejects + 1;
        step_valid = 1'b0;
        repeat (8) drive_step(1'b0, 1'b0);
        drive_step(1'b0, 1'b1);
        out_ready = 1'b1;
        @(posedge clk_core);
        #0.1 out_ready = 1'b0;

        capture_state();
        step_first = 1'b1;
        step_last = 1'b1;
        step_valid = 1'b1;
        #0.1;
        check(protocol_error, "first和last同拍的单步命令必须报错");
        check(!step_ready, "单步命令不得接收");
        @(posedge clk_core);
        #0.1;
        check(sampled_protocol_errors == 3, "单步命令必须在采样沿被协议监视器捕获");
        account_state_change();
        check(state_advance_errors == 0, "非法单步命令不得启动或改变内部命令");
        single_step_rejects = single_step_rejects + 1;
        step_valid = 1'b0;

        $display("DPTME_PROTOCOL_RESULT sampled_protocol_errors=%0d tag_reject=%0d early_last_reject=%0d single_step_reject=%0d state_advance_errors=%0d",
                 sampled_protocol_errors, tag_rejects, early_last_rejects,
                 single_step_rejects, state_advance_errors);
        $display("PASS: HIT-Flow DP-TME array");
        $finish;
    end
endmodule

`default_nettype wire
