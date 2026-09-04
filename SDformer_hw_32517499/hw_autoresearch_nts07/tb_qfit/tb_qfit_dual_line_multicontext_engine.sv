`timescale 1ns/1ps
`default_nettype none

module tb_qfit_dual_line_multicontext_engine;
    localparam int TILE_BITS = 256;
    localparam int ISSUE_WIDTH = 16;
    localparam int CONTEXTS = 4;
    localparam int OUT_LANES = 96;
    localparam int TAG_W = 32;
    localparam int OBJECT_W = 64;
    localparam int W_W = 8;
    localparam int ACC_W = 32;
    localparam int BANK_ADDR_W = 4;
    localparam int CTX_W = 2;
    localparam int COUNT_W = 9;

    logic clk_core = 1'b0;
    logic rst_core = 1'b1;
    always #1.5 clk_core = ~clk_core;

    logic command_valid;
    logic command_ready;
    logic [TAG_W-1:0] command_tag;
    logic [OBJECT_W-1:0] command_object_tag;
    logic command_batch_last;
    logic command_use_motion;
    logic [TILE_BITS-1:0] command_source_bits;
    logic [TILE_BITS-1:0] command_negative_bits;
    logic [OUT_LANES*ACC_W-1:0] command_seed_acc;
    logic weight_request_valid;
    logic weight_request_ready;
    logic [OBJECT_W-1:0] weight_request_object_tag;
    logic [ISSUE_WIDTH-1:0] weight_request_bank_valid;
    logic [ISSUE_WIDTH*BANK_ADDR_W-1:0] weight_request_bank_addr;
    logic [ISSUE_WIDTH*CTX_W-1:0] weight_request_bank_context;
    logic [ISSUE_WIDTH-1:0] weight_request_bank_negative;
    logic weight_response_valid;
    logic weight_response_ready;
    logic [ISSUE_WIDTH-1:0] weight_response_bank_valid;
    logic [ISSUE_WIDTH*OUT_LANES*W_W-1:0] weight_response_data;
    logic output_valid;
    logic output_ready;
    logic [TAG_W-1:0] output_tag;
    logic [OBJECT_W-1:0] output_object_tag;
    logic output_use_motion;
    logic [COUNT_W-1:0] output_source_count;
    logic [OUT_LANES*ACC_W-1:0] output_acc;
    logic [CONTEXTS-1:0] context_active;
    logic protocol_error;

    logic mem_pending_q;
    logic [OBJECT_W-1:0] mem_object_q;
    logic [ISSUE_WIDTH-1:0] mem_bank_valid_q;
    logic [ISSUE_WIDTH*BANK_ADDR_W-1:0] mem_bank_addr_q;
    logic [ISSUE_WIDTH*CTX_W-1:0] mem_bank_context_q;
    logic [ISSUE_WIDTH-1:0] mem_bank_negative_q;
    logic [31:0] lfsr_q;

    logic [CONTEXTS-1:0] sb_occupied_q;
    logic [TAG_W-1:0] sb_tag_q [0:CONTEXTS-1];
    logic [OBJECT_W-1:0] sb_object_q [0:CONTEXTS-1];
    logic sb_motion_q [0:CONTEXTS-1];
    logic [TILE_BITS-1:0] sb_expected_bits_q [0:CONTEXTS-1];
    logic [TILE_BITS-1:0] sb_seen_bits_q [0:CONTEXTS-1];
    logic [TILE_BITS-1:0] sb_negative_bits_q [0:CONTEXTS-1];
    logic signed [ACC_W-1:0] sb_expected_acc_q [0:CONTEXTS-1][0:OUT_LANES-1];

    integer commands_sent;
    integer outputs_seen;
    integer request_beats;
    integer bank_reads;
    integer output_stall_cycles;
    integer trace_fd;
    integer trace_scan;
    integer real_commands;
    reg [4095:0] trace_path;
    logic [OBJECT_W-1:0] trace_object;
    integer trace_tag;
    integer trace_batch_last;
    integer trace_use_motion;
    logic [TILE_BITS-1:0] trace_bits;
    logic [TILE_BITS-1:0] trace_negative;

    logic command_fire;
    logic request_fire;
    logic response_fire;
    logic output_fire;

    function automatic logic signed [W_W-1:0] model_weight(
        input logic [OBJECT_W-1:0] object_tag,
        input integer source,
        input integer lane
    );
        integer value;
        begin
            value = (
                object_tag[15:0] * 13 + source * 73 + lane * 151
                + ((source * source) % 251) * 19 + source * lane * 7 + 911
            ) % 255 - 127;
            model_weight = W_W'(value);
        end
    endfunction

    function automatic integer decode_source(
        input logic [BANK_ADDR_W-1:0] address,
        input integer bank
    );
        decode_source = $unsigned(address) * ISSUE_WIDTH + bank;
    endfunction

    qfit_dual_line_multicontext_engine #(
        .TILE_BITS(TILE_BITS), .ISSUE_WIDTH(ISSUE_WIDTH), .CONTEXTS(CONTEXTS),
        .OUT_LANES(OUT_LANES), .TAG_W(TAG_W), .OBJECT_W(OBJECT_W),
        .W_W(W_W), .ACC_W(ACC_W)
    ) dut (
        .clk_core, .rst_core,
        .command_valid, .command_ready, .command_tag, .command_object_tag,
        .command_batch_last,
        .command_use_motion, .command_source_bits, .command_negative_bits,
        .command_seed_acc,
        .weight_request_valid, .weight_request_ready, .weight_request_object_tag,
        .weight_request_bank_valid, .weight_request_bank_addr,
        .weight_request_bank_context, .weight_request_bank_negative,
        .weight_response_valid, .weight_response_ready,
        .weight_response_bank_valid, .weight_response_data,
        .output_valid, .output_ready, .output_tag, .output_object_tag,
        .output_use_motion, .output_source_count, .output_acc,
        .context_active, .protocol_error
    );

    assign command_fire = command_valid && command_ready;
    assign request_fire = weight_request_valid && weight_request_ready;
    assign response_fire = weight_response_valid && weight_response_ready;
    assign output_fire = output_valid && output_ready;
    assign weight_request_ready = (!mem_pending_q || response_fire) && lfsr_q[0];
    assign weight_response_valid = mem_pending_q && lfsr_q[3];
    assign weight_response_bank_valid = mem_bank_valid_q;
    assign output_ready = lfsr_q[5];

    always_comb begin
        weight_response_data = '0;
        for (int bank = 0; bank < ISSUE_WIDTH; bank = bank + 1) begin
            for (int lane = 0; lane < OUT_LANES; lane = lane + 1) begin
                weight_response_data[(bank*OUT_LANES + lane)*W_W +: W_W]
                    = model_weight(
                        mem_object_q,
                        decode_source(
                            mem_bank_addr_q[bank*BANK_ADDR_W +: BANK_ADDR_W], bank
                        ),
                        lane
                    );
            end
        end
    end

    always_ff @(posedge clk_core) begin : scoreboard
        integer free_ctx;
        integer out_ctx;
        integer source;
        integer ctx;
        logic found;
        logic signed [ACC_W-1:0] expected_value;
        if (rst_core) begin
            mem_pending_q <= 1'b0;
            mem_object_q <= '0;
            mem_bank_valid_q <= '0;
            mem_bank_addr_q <= '0;
            mem_bank_context_q <= '0;
            mem_bank_negative_q <= '0;
            lfsr_q <= 32'h83d1_74ab;
            sb_occupied_q <= '0;
            commands_sent <= 0;
            outputs_seen <= 0;
            request_beats <= 0;
            bank_reads <= 0;
            output_stall_cycles <= 0;
            for (int c = 0; c < CONTEXTS; c = c + 1) begin
                sb_tag_q[c] <= '0;
                sb_object_q[c] <= '0;
                sb_motion_q[c] <= 1'b0;
                sb_expected_bits_q[c] <= '0;
                sb_seen_bits_q[c] <= '0;
                sb_negative_bits_q[c] <= '0;
                for (int lane = 0; lane < OUT_LANES; lane = lane + 1)
                    sb_expected_acc_q[c][lane] <= '0;
            end
        end else begin
            lfsr_q <= {lfsr_q[30:0], lfsr_q[31] ^ lfsr_q[21] ^ lfsr_q[1] ^ lfsr_q[0]};
            if (output_valid && !output_ready)
                output_stall_cycles <= output_stall_cycles + 1;
            if (response_fire)
                mem_pending_q <= 1'b0;
            if (request_fire) begin
                mem_pending_q <= 1'b1;
                mem_object_q <= weight_request_object_tag;
                mem_bank_valid_q <= weight_request_bank_valid;
                mem_bank_addr_q <= weight_request_bank_addr;
                mem_bank_context_q <= weight_request_bank_context;
                mem_bank_negative_q <= weight_request_bank_negative;
                request_beats <= request_beats + 1;
                bank_reads <= bank_reads + $countones(weight_request_bank_valid);
                for (int bank = 0; bank < ISSUE_WIDTH; bank = bank + 1) begin
                    if (weight_request_bank_valid[bank]) begin
                        ctx = $unsigned(
                            weight_request_bank_context[bank*CTX_W +: CTX_W]
                        );
                        source = decode_source(
                            weight_request_bank_addr[bank*BANK_ADDR_W +: BANK_ADDR_W], bank
                        );
                        if (ctx >= CONTEXTS || !sb_occupied_q[ctx])
                            $fatal(1, "request used invalid context=%0d", ctx);
                        if (weight_request_object_tag !== sb_object_q[ctx])
                            $fatal(1, "weight object mismatch context=%0d", ctx);
                        if (!sb_expected_bits_q[ctx][source])
                            $fatal(1, "inactive source context=%0d source=%0d", ctx, source);
                        if (sb_seen_bits_q[ctx][source])
                            $fatal(1, "duplicate source context=%0d source=%0d", ctx, source);
                        if (weight_request_bank_negative[bank]
                                !== sb_negative_bits_q[ctx][source])
                            $fatal(1, "source sign mismatch context=%0d source=%0d", ctx, source);
                        sb_seen_bits_q[ctx][source] <= 1'b1;
                    end
                end
            end

            if (command_fire) begin
                free_ctx = -1;
                for (int c = 0; c < CONTEXTS; c = c + 1) begin
                    if (free_ctx < 0 && !context_active[c])
                        free_ctx = c;
                end
                if (free_ctx < 0)
                    $fatal(1, "command handshake without free context");
                sb_occupied_q[free_ctx] <= 1'b1;
                sb_tag_q[free_ctx] <= command_tag;
                sb_object_q[free_ctx] <= command_object_tag;
                sb_motion_q[free_ctx] <= command_use_motion;
                sb_expected_bits_q[free_ctx] <= command_source_bits;
                sb_seen_bits_q[free_ctx] <= '0;
                sb_negative_bits_q[free_ctx] <= command_negative_bits;
                commands_sent <= commands_sent + 1;
                for (int lane = 0; lane < OUT_LANES; lane = lane + 1) begin
                    expected_value = command_seed_acc[lane*ACC_W +: ACC_W];
                    for (int s = 0; s < TILE_BITS; s = s + 1) begin
                        if (command_source_bits[s]) begin
                            if (command_negative_bits[s])
                                expected_value = expected_value
                                    - $signed(model_weight(command_object_tag, s, lane));
                            else
                                expected_value = expected_value
                                    + $signed(model_weight(command_object_tag, s, lane));
                        end
                    end
                    sb_expected_acc_q[free_ctx][lane] <= expected_value;
                end
            end

            if (output_fire) begin
                out_ctx = -1;
                found = 1'b0;
                for (int c = 0; c < CONTEXTS; c = c + 1) begin
                    if (!found && sb_occupied_q[c] && sb_tag_q[c] == output_tag) begin
                        found = 1'b1;
                        out_ctx = c;
                    end
                end
                if (out_ctx < 0)
                    $fatal(1, "unknown output tag=%0d", output_tag);
                if (output_object_tag !== sb_object_q[out_ctx]
                        || output_use_motion !== sb_motion_q[out_ctx])
                    $fatal(1, "output identity mismatch tag=%0d", output_tag);
                if (output_source_count !== $countones(sb_expected_bits_q[out_ctx]))
                    $fatal(1, "output source count mismatch tag=%0d got=%0d expected=%0d",
                        output_tag, output_source_count, $countones(sb_expected_bits_q[out_ctx]));
                if (sb_seen_bits_q[out_ctx] !== sb_expected_bits_q[out_ctx])
                    $fatal(1, "source coverage mismatch tag=%0d", output_tag);
                for (int lane = 0; lane < OUT_LANES; lane = lane + 1) begin
                    if ($signed(output_acc[lane*ACC_W +: ACC_W])
                            !== sb_expected_acc_q[out_ctx][lane])
                        $fatal(1, "acc mismatch tag=%0d lane=%0d got=%0d expected=%0d",
                            output_tag, lane,
                            $signed(output_acc[lane*ACC_W +: ACC_W]),
                            sb_expected_acc_q[out_ctx][lane]);
                end
                sb_occupied_q[out_ctx] <= 1'b0;
                outputs_seen <= outputs_seen + 1;
            end
        end
    end

    task automatic drive_command(
        input logic [OBJECT_W-1:0] object_tag,
        input logic [TAG_W-1:0] tag,
        input logic batch_last,
        input logic use_motion,
        input logic [TILE_BITS-1:0] bits,
        input logic [TILE_BITS-1:0] negative
    );
        begin
            @(negedge clk_core);
            command_valid = 1'b1;
            command_object_tag = object_tag;
            command_tag = tag;
            command_batch_last = batch_last;
            command_use_motion = use_motion;
            command_source_bits = bits;
            command_negative_bits = negative;
            command_seed_acc = '0;
            for (int lane = 0; lane < OUT_LANES; lane = lane + 1)
                command_seed_acc[lane*ACC_W +: ACC_W]
                    = ACC_W'($signed((tag * 11 + lane * 7) % 257 - 128));
            do @(posedge clk_core); while (!command_ready);
            @(negedge clk_core);
            command_valid = 1'b0;
        end
    endtask

    task automatic wait_outputs(input integer target);
        integer timeout;
        begin
            timeout = 0;
            while (outputs_seen < target && timeout < 200000) begin
                @(posedge clk_core);
                timeout = timeout + 1;
            end
            if (outputs_seen != target)
                $fatal(1, "timeout outputs=%0d target=%0d", outputs_seen, target);
        end
    endtask

    initial begin : stimulus
        logic [TILE_BITS-1:0] bits;
        logic [TILE_BITS-1:0] negative;
        integer expected_outputs;
        integer issue_start;
        command_valid = 1'b0;
        command_tag = '0;
        command_object_tag = '0;
        command_batch_last = 1'b0;
        command_use_motion = 1'b0;
        command_source_bits = '0;
        command_negative_bits = '0;
        command_seed_acc = '0;
        repeat (8) @(posedge clk_core);
        rst_core = 1'b0;

        // Deterministic bank-complement case: four contexts complete 16
        // source updates in four issue beats instead of sixteen p1 cycles.
        issue_start = request_beats;
        for (int c = 0; c < CONTEXTS; c = c + 1) begin
            bits = '0;
            negative = '0;
            for (int address = 0; address < 4; address = address + 1)
                bits[address*ISSUE_WIDTH + c] = 1'b1;
            drive_command(
                64'h100, TAG_W'(c + 1), c == CONTEXTS-1, 1'b0, bits, negative
            );
        end
        // Standard ready/valid permits the producer to present and hold the
        // next object's payload while ready is low.  Start it before the
        // current batch drains and require eventual acceptance without a
        // false protocol fault.
        bits = '0;
        negative = '0;
        bits[3] = 1'b1;
        bits[19] = 1'b1;
        fork
            drive_command(64'h200, TAG_W'(90), 1'b1, 1'b0, bits, negative);
            begin
                @(negedge clk_core);
                while (!command_valid || command_ready)
                    @(negedge clk_core);
                repeat (3) @(posedge clk_core);
                if (protocol_error)
                    $fatal(1, "held next-object valid caused protocol_error");
            end
            wait_outputs(CONTEXTS);
        join
        if (request_beats - issue_start != 4)
            $fatal(1, "context-fill case expected 4 beats got=%0d",
                request_beats - issue_start);
        wait_outputs(CONTEXTS + 1);

        expected_outputs = outputs_seen;
        real_commands = 0;
        if ($value$plusargs("REAL_TRACE=%s", trace_path)) begin
            trace_fd = $fopen(trace_path, "r");
            if (trace_fd == 0)
                $fatal(1, "cannot open REAL_TRACE=%s", trace_path);
            while (!$feof(trace_fd)) begin
                trace_scan = $fscanf(
                    trace_fd, "%h %d %d %d %h %h\n",
                    trace_object, trace_tag, trace_batch_last,
                    trace_use_motion, trace_bits, trace_negative
                );
                if (trace_scan == 6) begin
                    drive_command(
                        trace_object, TAG_W'(trace_tag), trace_batch_last != 0,
                        trace_use_motion != 0, trace_bits, trace_negative
                    );
                    expected_outputs = expected_outputs + 1;
                    real_commands = real_commands + 1;
                    if (trace_batch_last != 0)
                        wait_outputs(expected_outputs);
                end else if (!$feof(trace_fd)) begin
                    $fatal(1, "malformed REAL_TRACE after commands=%0d scan=%0d",
                        real_commands, trace_scan);
                end
            end
            $fclose(trace_fd);
            wait_outputs(expected_outputs);
        end else begin
            for (int batch = 0; batch < 200; batch = batch + 1) begin
                for (int c = 0; c < CONTEXTS; c = c + 1) begin
                    bits = '0;
                    negative = '0;
                    for (int source = 0; source < TILE_BITS; source = source + 1) begin
                        if (($urandom % 100) < (5 + (batch % 55))) begin
                            bits[source] = 1'b1;
                            if ((batch[0] || c[0]) && ($urandom % 4) == 0)
                                negative[source] = 1'b1;
                        end
                    end
                    drive_command(
                        OBJECT_W'(64'h1000 + batch),
                        TAG_W'(100 + batch*CONTEXTS + c),
                        c == CONTEXTS-1, batch[0] || c[0], bits, negative
                    );
                end
                expected_outputs = expected_outputs + CONTEXTS;
                wait_outputs(expected_outputs);
                if (protocol_error)
                    $fatal(1, "unexpected protocol_error batch=%0d", batch);
            end
        end

        if (commands_sent != outputs_seen)
            $fatal(1, "command/output mismatch sent=%0d seen=%0d",
                commands_sent, outputs_seen);
        if (output_stall_cycles == 0)
            $fatal(1, "output backpressure was not exercised");
        $display(
            "PASS_M3_P16C4_DUAL_LINE commands=%0d outputs=%0d issue_beats=%0d bank_reads=%0d output_stall_cycles=%0d real_commands=%0d",
            commands_sent, outputs_seen, request_beats, bank_reads,
            output_stall_cycles, real_commands
        );
        $finish;
    end
endmodule

`default_nettype wire
