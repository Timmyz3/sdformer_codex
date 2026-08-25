`timescale 1ns/1ps
`default_nettype none

`ifndef M2_ISSUE_WIDTH
`define M2_ISSUE_WIDTH 4
`endif
`ifndef M2_OUT_LANES
`define M2_OUT_LANES 16
`endif

module tb_qfit_local_banked_multisource_engine;
    localparam int TILE_BITS = 256;
    localparam int WORD_BITS = 32;
    localparam int ISSUE_WIDTH = `M2_ISSUE_WIDTH;
    localparam int OUT_LANES = `M2_OUT_LANES;
    localparam int TAG_W = 32;
    localparam int W_W = 8;
    localparam int ACC_W = 32;
    localparam int INDEX_W = $clog2(TILE_BITS);
    localparam int BANK_BITS = (ISSUE_WIDTH <= 1) ? 0 : $clog2(ISSUE_WIDTH);
    localparam int BANK_ADDR_W = INDEX_W - BANK_BITS;
    localparam int COUNT_W = $clog2(TILE_BITS + 1);

    logic clk_core = 1'b0;
    logic rst_core = 1'b1;
    always #1.5 clk_core = ~clk_core;

    logic command_valid;
    logic command_ready;
    logic [TAG_W-1:0] command_tag;
    logic [TILE_BITS-1:0] command_current_bits;
    logic [OUT_LANES*ACC_W-1:0] command_seed_acc;
    logic weight_request_valid;
    logic weight_request_ready;
    logic [ISSUE_WIDTH-1:0] weight_request_bank_valid;
    logic [ISSUE_WIDTH*BANK_ADDR_W-1:0] weight_request_bank_addr;
    logic weight_request_last;
    logic weight_response_valid;
    logic weight_response_ready;
    logic [ISSUE_WIDTH-1:0] weight_response_bank_valid;
    logic [ISSUE_WIDTH*OUT_LANES*W_W-1:0] weight_response_data;
    logic output_valid;
    logic output_ready;
    logic [TAG_W-1:0] output_tag;
    logic [COUNT_W-1:0] output_source_count;
    logic [OUT_LANES*ACC_W-1:0] output_acc;
    logic protocol_error;
    logic [63:0] monitor_commands;
    logic [63:0] monitor_issue_beats;
    logic [63:0] monitor_weight_bank_reads;
    logic [63:0] monitor_accumulator_updates;

    logic mem_pending_q;
    logic [ISSUE_WIDTH-1:0] mem_bank_valid_q;
    logic [ISSUE_WIDTH*BANK_ADDR_W-1:0] mem_bank_addr_q;
    logic [2:0] mem_delay_q;
    logic [31:0] lfsr_q;
    logic inject_bad_response;
    logic inject_unsolicited_response;
    logic request_fire;
    logic response_fire;
    logic [TILE_BITS-1:0] request_source_mask;
    logic [TILE_BITS-1:0] scoreboard_expected_q;
    logic [TILE_BITS-1:0] scoreboard_seen_q;

    integer total_cases;
    integer total_sources;
    integer empty_cases;
    integer trace_fd;
    integer scan_status;
    integer max_cases;
    integer timeout_cycles;
    integer real_issue_beats;
    integer real_cases;
    integer real_sources;
    integer real_empty_cases;
    longint unsigned real_ideal_ready_cycles;
    longint unsigned real_latency_cycles;
    longint unsigned real_full_wall_cycles;
    longint unsigned total_ideal_ready_cycles;
    longint unsigned total_latency_cycles;
    longint unsigned total_full_wall_cycles;
    reg [4095:0] trace_path;
    logic [TILE_BITS-1:0] trace_bits;

    function automatic logic signed [W_W-1:0] model_weight(
        input int source,
        input int lane
    );
        integer value;
        begin
            value = (
                source * 73 + lane * 151 + ((source * source) % 251) * 19
                + source * lane * 7 + (source >> (lane % 5)) * 31 + 911
            ) % 255 - 127;
            model_weight = W_W'(value);
        end
    endfunction

    function automatic integer bitmap_popcount(input logic [TILE_BITS-1:0] bits);
        integer count;
        begin
            count = 0;
            for (int source = 0; source < TILE_BITS; source = source + 1)
                count = count + bits[source];
            bitmap_popcount = count;
        end
    endfunction

    function automatic integer bitmap_issue_beats(input logic [TILE_BITS-1:0] bits);
        integer bank_counts [0:ISSUE_WIDTH-1];
        integer maximum;
        begin
            for (int bank = 0; bank < ISSUE_WIDTH; bank = bank + 1)
                bank_counts[bank] = 0;
            for (int source = 0; source < TILE_BITS; source = source + 1) begin
                if (bits[source])
                    bank_counts[source % ISSUE_WIDTH] = bank_counts[source % ISSUE_WIDTH] + 1;
            end
            maximum = 0;
            for (int bank = 0; bank < ISSUE_WIDTH; bank = bank + 1) begin
                if (bank_counts[bank] > maximum)
                    maximum = bank_counts[bank];
            end
            bitmap_issue_beats = maximum;
        end
    endfunction

    function automatic integer decode_source(
        input logic [BANK_ADDR_W-1:0] bank_addr,
        input integer bank
    );
        decode_source = $unsigned(bank_addr) * ISSUE_WIDTH + bank;
    endfunction

    qfit_local_banked_multisource_engine #(
        .TILE_BITS(TILE_BITS), .WORD_BITS(WORD_BITS), .ISSUE_WIDTH(ISSUE_WIDTH),
        .OUT_LANES(OUT_LANES), .TAG_W(TAG_W), .W_W(W_W), .ACC_W(ACC_W)
    ) dut (
        .clk_core, .rst_core,
        .command_valid, .command_ready, .command_tag,
        .command_current_bits, .command_seed_acc,
        .weight_request_valid, .weight_request_ready,
        .weight_request_bank_valid, .weight_request_bank_addr, .weight_request_last,
        .weight_response_valid, .weight_response_ready,
        .weight_response_bank_valid, .weight_response_data,
        .output_valid, .output_ready, .output_tag, .output_source_count, .output_acc,
        .protocol_error
    );

    assign request_fire = weight_request_valid && weight_request_ready;
    assign response_fire = weight_response_valid && weight_response_ready;
    assign weight_response_valid = inject_unsolicited_response
        || (mem_pending_q && mem_delay_q == 0 && lfsr_q[3]);
    assign weight_response_bank_valid = inject_unsolicited_response
        ? {{(ISSUE_WIDTH-1){1'b0}}, 1'b1}
        : (inject_bad_response
            ? (mem_bank_valid_q ^ {{(ISSUE_WIDTH-1){1'b0}}, 1'b1}) : mem_bank_valid_q);
    assign weight_request_ready = (!mem_pending_q || response_fire) && lfsr_q[0];
    assign output_ready = lfsr_q[4];

    always_comb begin
        weight_response_data = '0;
        request_source_mask = '0;
        for (int bank = 0; bank < ISSUE_WIDTH; bank = bank + 1) begin
            if (weight_request_bank_valid[bank]) begin
                request_source_mask[decode_source(
                    weight_request_bank_addr[bank*BANK_ADDR_W +: BANK_ADDR_W], bank
                )] = 1'b1;
            end
            for (int lane = 0; lane < OUT_LANES; lane = lane + 1) begin
                weight_response_data[(bank*OUT_LANES + lane)*W_W +: W_W]
                    = model_weight(decode_source(
                        mem_bank_addr_q[bank*BANK_ADDR_W +: BANK_ADDR_W], bank
                    ), lane);
            end
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            mem_pending_q <= 1'b0;
            mem_bank_valid_q <= '0;
            mem_bank_addr_q <= '0;
            mem_delay_q <= '0;
            lfsr_q <= 32'h1d87_2b41;
            scoreboard_expected_q <= '0;
            scoreboard_seen_q <= '0;
            monitor_commands <= '0;
            monitor_issue_beats <= '0;
            monitor_weight_bank_reads <= '0;
            monitor_accumulator_updates <= '0;
        end else begin
            lfsr_q <= {lfsr_q[30:0], lfsr_q[31] ^ lfsr_q[21] ^ lfsr_q[1] ^ lfsr_q[0]};
            if (mem_pending_q && mem_delay_q != 0)
                mem_delay_q <= mem_delay_q - 3'd1;
            if (response_fire)
                mem_pending_q <= 1'b0;
            if (request_fire) begin
                monitor_issue_beats <= monitor_issue_beats + 64'd1;
                if (weight_request_last !==
                        ((scoreboard_seen_q | request_source_mask) == scoreboard_expected_q))
                    $fatal(1, "last oracle mismatch last=%0b seen_after=%h expected=%h",
                        weight_request_last, scoreboard_seen_q | request_source_mask,
                        scoreboard_expected_q);
                mem_pending_q <= 1'b1;
                mem_bank_valid_q <= weight_request_bank_valid;
                mem_bank_addr_q <= weight_request_bank_addr;
                mem_delay_q <= {1'b0, lfsr_q[2:1]};
                for (int bank = 0; bank < ISSUE_WIDTH; bank = bank + 1) begin
                    if (weight_request_bank_valid[bank]) begin : check_request_source
                        integer source;
                        source = decode_source(
                            weight_request_bank_addr[bank*BANK_ADDR_W +: BANK_ADDR_W], bank
                        );
                        if (!scoreboard_expected_q[source])
                            $fatal(1, "request emitted inactive source=%0d bank=%0d expected=%h",
                                source, bank, scoreboard_expected_q);
                        if (scoreboard_seen_q[source])
                            $fatal(1, "request emitted duplicate source=%0d bank=%0d", source, bank);
                        scoreboard_seen_q[source] <= 1'b1;
                    end
                end
            end
            if (response_fire) begin
                monitor_weight_bank_reads <= monitor_weight_bank_reads
                    + $countones(weight_response_bank_valid);
                monitor_accumulator_updates <= monitor_accumulator_updates
                    + $countones(weight_response_bank_valid) * OUT_LANES;
            end
            if (output_valid && output_ready)
                monitor_commands <= monitor_commands + 64'd1;
            if (command_valid && command_ready) begin
                scoreboard_expected_q <= command_current_bits;
                scoreboard_seen_q <= '0;
            end
        end
    end

    task automatic run_case(
        input logic [TILE_BITS-1:0] bits,
        input integer case_id,
        input logic wrap_seed
    );
        integer expected [0:OUT_LANES-1];
        integer source_count;
        integer case_issue_beats;
        integer case_latency_cycles;
        integer case_full_wall_cycles;
        begin
            source_count = bitmap_popcount(bits);
            case_issue_beats = bitmap_issue_beats(bits);
            for (int lane = 0; lane < OUT_LANES; lane = lane + 1) begin
                expected[lane] = wrap_seed
                    ? (32'sh7fff_ff00 + lane * 13)
                    : (((case_id * 19 + lane * 7) % 2049) - 1024);
                for (int source = 0; source < TILE_BITS; source = source + 1) begin
                    if (bits[source])
                        expected[lane] = expected[lane] + $signed(model_weight(source, lane));
                end
            end

            while (!command_ready)
                @(negedge clk_core);
            command_tag = TAG_W'(case_id);
            command_current_bits = bits;
            for (int lane = 0; lane < OUT_LANES; lane = lane + 1)
                command_seed_acc[lane*ACC_W +: ACC_W] = ACC_W'(
                    wrap_seed
                        ? (32'sh7fff_ff00 + lane * 13)
                        : (((case_id * 19 + lane * 7) % 2049) - 1024)
                );
            command_valid = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            command_valid = 1'b0;

            timeout_cycles = 0;
            case_latency_cycles = 0;
            while (!output_valid) begin
                @(negedge clk_core);
                case_latency_cycles = case_latency_cycles + 1;
                timeout_cycles = timeout_cycles + 1;
                if (timeout_cycles > 10000)
                    $fatal(1, "timeout case=%0d bits=%h", case_id, bits);
            end
            if (output_tag !== TAG_W'(case_id))
                $fatal(1, "tag mismatch case=%0d got=%0d", case_id, output_tag);
            if (output_source_count !== COUNT_W'(source_count))
                $fatal(1, "source-count mismatch case=%0d expected=%0d got=%0d",
                    case_id, source_count, output_source_count);
            if (scoreboard_seen_q !== bits)
                $fatal(1, "source scoreboard mismatch case=%0d missing_or_extra=%h",
                    case_id, scoreboard_seen_q ^ bits);
            for (int lane = 0; lane < OUT_LANES; lane = lane + 1) begin
                if ($signed(output_acc[lane*ACC_W +: ACC_W]) !== expected[lane])
                    $fatal(1, "acc mismatch case=%0d lane=%0d expected=%0d got=%0d",
                        case_id, lane, expected[lane], $signed(output_acc[lane*ACC_W +: ACC_W]));
            end
            case_full_wall_cycles = case_latency_cycles;
            while (!(output_valid && output_ready))
            begin
                @(negedge clk_core);
                case_full_wall_cycles = case_full_wall_cycles + 1;
            end
            @(posedge clk_core);
            @(negedge clk_core);
            // The accepting output edge is one cycle after the final observed
            // ready/valid negedge.  Count it explicitly.
            case_full_wall_cycles = case_full_wall_cycles + 1;
            total_cases = total_cases + 1;
            total_sources = total_sources + source_count;
            empty_cases = empty_cases + (source_count == 0);
            total_ideal_ready_cycles = total_ideal_ready_cycles
                + ((source_count == 0) ? 1 : case_issue_beats + 2);
            total_latency_cycles = total_latency_cycles + case_latency_cycles;
            total_full_wall_cycles = total_full_wall_cycles + case_full_wall_cycles;
        end
    endtask

    task automatic pulse_reset;
        begin
            command_valid = 1'b0;
            inject_bad_response = 1'b0;
            inject_unsolicited_response = 1'b0;
            rst_core = 1'b1;
            repeat (3) @(negedge clk_core);
            rst_core = 1'b0;
            @(negedge clk_core);
            if (protocol_error || output_valid || !command_ready)
                $fatal(1, "reset did not restore idle command-ready state");
        end
    endtask

    initial begin
        command_valid = 1'b0;
        command_tag = '0;
        command_current_bits = '0;
        command_seed_acc = '0;
        inject_bad_response = 1'b0;
        inject_unsolicited_response = 1'b0;
        total_cases = 0;
        total_sources = 0;
        empty_cases = 0;
        total_ideal_ready_cycles = 0;
        total_latency_cycles = 0;
        total_full_wall_cycles = 0;
        if (!$value$plusargs("TRACE_FILE=%s", trace_path))
            $fatal(1, "TRACE_FILE plusarg is required");
        if (!$value$plusargs("MAX_CASES=%d", max_cases))
            max_cases = 20000;
        repeat (8) @(negedge clk_core);
        rst_core = 1'b0;

        trace_fd = $fopen(trace_path, "r");
        if (trace_fd == 0)
            $fatal(1, "cannot open real-tile trace %0s", trace_path);
        while (!$feof(trace_fd) && total_cases < max_cases) begin
            scan_status = $fscanf(trace_fd, "%h\n", trace_bits);
            if (scan_status == 1)
                run_case(trace_bits, total_cases, 1'b0);
        end
        $fclose(trace_fd);
        if (total_cases != max_cases)
            $fatal(1, "trace cardinality mismatch expected=%0d got=%0d", max_cases, total_cases);
        if (monitor_commands != 64'(total_cases)
                || monitor_weight_bank_reads != 64'(total_sources)
                || monitor_accumulator_updates != 64'(total_sources * OUT_LANES))
            $fatal(1, "performance counter mismatch commands=%0d reads=%0d updates=%0d",
                monitor_commands, monitor_weight_bank_reads, monitor_accumulator_updates);
        real_issue_beats = monitor_issue_beats;
        real_cases = total_cases;
        real_sources = total_sources;
        real_empty_cases = empty_cases;
        real_ideal_ready_cycles = total_ideal_ready_cycles;
        real_latency_cycles = total_latency_cycles;
        real_full_wall_cycles = total_full_wall_cycles;

        // An unsolicited response must fault closed, and reset must recover.
        while (!command_ready)
            @(negedge clk_core);
        inject_unsolicited_response = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        inject_unsolicited_response = 1'b0;
        if (!protocol_error || command_ready || output_valid)
            $fatal(1, "unsolicited response did not fault closed");
        pulse_reset();

        // Reset a transaction after its first accepted request and prove no
        // stale output or protocol fault survives.  This also exercises the
        // maximum-popcount request frontier.
        command_tag = 32'hffff_fe00;
        command_current_bits = '1;
        command_seed_acc = '0;
        command_valid = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        command_valid = 1'b0;
        while (!request_fire)
            @(negedge clk_core);
        @(posedge clk_core);
        @(negedge clk_core);
        pulse_reset();

        // Complete a maximum-popcount transaction from near INT32_MAX.  The
        // independent integer oracle therefore checks two's-complement wrap.
        run_case({TILE_BITS{1'b1}}, 32'h7fff_0000, 1'b1);

        // End-of-run fail-safe injection: corrupt the first in-order bank-valid
        // response and require sticky rejection with no architectural output.
        while (!command_ready)
            @(negedge clk_core);
        command_tag = 32'hffff_ff00;
        command_current_bits = {{(TILE_BITS-1){1'b0}}, 1'b1};
        command_seed_acc = '0;
        command_valid = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        command_valid = 1'b0;
        inject_bad_response = 1'b1;
        while (!protocol_error)
            @(negedge clk_core);
        inject_bad_response = 1'b0;
        repeat (5) @(negedge clk_core);
        if (output_valid || command_ready)
            $fatal(1, "faulted engine did not remain fail-closed");

        // A protocol fault is recoverable only through reset; prove that the
        // next architectural command is exact after recovery.
        pulse_reset();
        run_case({{(TILE_BITS-1){1'b0}}, 1'b1}, 32'h7fff_0001, 1'b0);

        $display("PASS M2B banked multi-source issue_width=%0d out_lanes=%0d real_cases=%0d sources=%0d issue_beats=%0d ideal_ready_cycles=%0d latency_cycles=%0d full_wall_cycles=%0d empty=%0d protocol_injections=2 directed_reset_cases=2 wrap_cases=1",
            ISSUE_WIDTH, OUT_LANES, real_cases, real_sources, real_issue_beats,
            real_ideal_ready_cycles, real_latency_cycles, real_full_wall_cycles,
            real_empty_cases);
        $finish;
    end
endmodule

`default_nettype wire
