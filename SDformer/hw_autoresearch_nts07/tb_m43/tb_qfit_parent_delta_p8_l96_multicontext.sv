`timescale 1ns/1ps
`default_nettype none

module tb_qfit_parent_delta_p8_l96_multicontext;
    localparam int TILE_BITS = 256;
    localparam int ISSUE_WIDTH = 8;
    localparam int OUT_LANES = 96;
    localparam int ACC_W = 19;
    localparam int TAG_W = 48;
    localparam int PACKETS = 128;

    logic clk_core;
    logic rst_core;
    logic command_valid;
    logic command_ready;
    logic [TAG_W-1:0] command_tag;
    logic [TILE_BITS-1:0] command_add_bits;
    logic [TILE_BITS-1:0] command_subtract_bits;
    logic [OUT_LANES*ACC_W-1:0] command_seed_acc;
    logic weight_request_valid;
    logic weight_request_ready;
    logic [ISSUE_WIDTH-1:0] weight_request_bank_valid;
    logic [ISSUE_WIDTH*5-1:0] weight_request_bank_addr;
    logic [ISSUE_WIDTH-1:0] weight_request_bank_subtract;
    logic weight_request_last;
    logic [1:0] weight_request_context;
    logic weight_response_valid;
    logic weight_response_ready;
    logic [1:0] weight_response_context;
    logic [ISSUE_WIDTH-1:0] weight_response_bank_valid;
    logic [ISSUE_WIDTH*OUT_LANES*8-1:0] weight_response_data;
    logic output_valid;
    logic output_ready;
    logic [TAG_W-1:0] output_tag;
    logic [8:0] output_source_count;
    logic [OUT_LANES*ACC_W-1:0] output_acc;
    logic protocol_error;
    logic busy;
    logic [2:0] context_occupancy;
    logic [4:0] response_metadata_occupancy;
    logic command_accept;
    logic request_accept;
    logic response_accept;
    logic output_accept;

    logic signed [ACC_W-1:0] expected_acc [0:PACKETS-1][0:OUT_LANES-1];
    integer expected_count [0:PACKETS-1];
    logic seen [0:PACKETS-1];
    integer outputs_seen;
    integer requests_seen;
    integer responses_seen;
    integer simultaneous_request_response;
    integer maximum_context_occupancy;
    integer request_stalls;
    integer output_stalls;
    logic automatic_responses;
    logic automatic_backpressure;
    logic output_was_stalled;
    logic [TAG_W-1:0] stalled_output_tag;
    logic [8:0] stalled_output_source_count;
    logic [OUT_LANES*ACC_W-1:0] stalled_output_acc;
    logic request_was_stalled;
    logic [ISSUE_WIDTH-1:0] stalled_request_bank_valid;
    logic [ISSUE_WIDTH*5-1:0] stalled_request_bank_addr;
    logic [ISSUE_WIDTH-1:0] stalled_request_bank_subtract;
    logic stalled_request_last;
    logic [1:0] stalled_request_context;
    logic [1:0] manual_response_context;
    logic [ISSUE_WIDTH-1:0] manual_response_bank_valid;

    qfit_parent_delta_p8_l96_multicontext dut (.*);

    function automatic logic signed [7:0] model_weight(
        input integer source,
        input integer lane
    );
        integer raw;
        begin
            if (source < ISSUE_WIDTH) begin
                model_weight = -8'sd128;
            end else begin
                raw = ((source * 37 + lane * 13 + 19) % 255) - 127;
                model_weight = raw[7:0];
            end
        end
    endfunction

    function automatic integer popcount256(input logic [TILE_BITS-1:0] value);
        integer count;
        begin
            count = 0;
            for (int bit_index = 0; bit_index < TILE_BITS; bit_index++)
                count = count + value[bit_index];
            popcount256 = count;
        end
    endfunction

    task automatic prepare_packet(input integer packet);
        integer seed;
        integer total;
        integer random_value;
        begin
            command_tag = TAG_W'(packet);
            command_add_bits = '0;
            command_subtract_bits = '0;
            for (int source = 0; source < TILE_BITS; source++) begin
                random_value = $urandom_range(0, 99);
                if (packet == 0) begin
                    command_add_bits[source] = 1'b0;
                    command_subtract_bits[source] = 1'b0;
                end else if (packet == 1 && source % ISSUE_WIDTH == 0
                        && source < 80) begin
                    command_add_bits[source] = 1'b1;
                end else if (packet == 2 && source < ISSUE_WIDTH) begin
                    command_subtract_bits[source] = 1'b1;
                end else if (packet == 3 && source < ISSUE_WIDTH) begin
                    command_add_bits[source] = 1'b1;
                end else if (random_value < 12) begin
                    command_add_bits[source] = 1'b1;
                end else if (random_value < 23) begin
                    command_subtract_bits[source] = 1'b1;
                end
            end
            expected_count[packet] = popcount256(
                command_add_bits | command_subtract_bits);
            for (int lane = 0; lane < OUT_LANES; lane++) begin
                seed = ((packet * 17 + lane * 5) % 101) - 50;
                command_seed_acc[lane*ACC_W +: ACC_W] = ACC_W'(seed);
                total = seed;
                for (int source = 0; source < TILE_BITS; source++) begin
                    if (command_add_bits[source])
                        total = total + $signed(model_weight(source, lane));
                    if (command_subtract_bits[source])
                        total = total - $signed(model_weight(source, lane));
                end
                expected_acc[packet][lane] = ACC_W'(total);
            end
        end
    endtask

    always #1.5 clk_core = ~clk_core;

    always @(negedge clk_core) begin
        if (rst_core) begin
            weight_request_ready <= 1'b0;
            output_ready <= 1'b0;
        end else if (automatic_backpressure) begin
            weight_request_ready <= $urandom_range(0, 7) != 0;
            output_ready <= $urandom_range(0, 5) != 0;
        end
    end

    always @(posedge clk_core) begin
        if (rst_core || !automatic_responses) begin
            if (automatic_responses) begin
                weight_response_valid <= 1'b0;
                weight_response_context <= '0;
                weight_response_bank_valid <= '0;
                weight_response_data <= '0;
            end
        end else begin
            weight_response_valid <= request_accept;
            weight_response_context <= weight_request_context;
            weight_response_bank_valid <= weight_request_bank_valid;
            for (int bank = 0; bank < ISSUE_WIDTH; bank++) begin
                integer source;
                source = (weight_request_bank_addr[bank*5 +: 5]
                          * ISSUE_WIDTH) + bank;
                for (int lane = 0; lane < OUT_LANES; lane++)
                    weight_response_data[(bank*OUT_LANES+lane)*8 +: 8]
                        <= model_weight(source, lane);
            end
        end
    end

    always @(posedge clk_core) begin
        if (rst_core) begin
            output_was_stalled <= 1'b0;
            request_was_stalled <= 1'b0;
        end else begin
            if (output_was_stalled && !protocol_error) begin
                if (!output_valid || output_tag !== stalled_output_tag
                        || output_source_count !== stalled_output_source_count
                        || output_acc !== stalled_output_acc)
                    $fatal(1, "M43 output changed under backpressure");
            end
            output_was_stalled <= output_valid && !output_ready;
            if (output_valid && !output_ready) begin
                stalled_output_tag <= output_tag;
                stalled_output_source_count <= output_source_count;
                stalled_output_acc <= output_acc;
            end
            if (request_was_stalled && !protocol_error) begin
                if (!weight_request_valid
                        || weight_request_bank_valid
                            !== stalled_request_bank_valid
                        || weight_request_bank_addr
                            !== stalled_request_bank_addr
                        || weight_request_bank_subtract
                            !== stalled_request_bank_subtract
                        || weight_request_last !== stalled_request_last
                        || weight_request_context
                            !== stalled_request_context)
                    $fatal(1, "M43 weight request changed under backpressure");
            end
            request_was_stalled
                <= weight_request_valid && !weight_request_ready;
            if (weight_request_valid && !weight_request_ready) begin
                stalled_request_bank_valid <= weight_request_bank_valid;
                stalled_request_bank_addr <= weight_request_bank_addr;
                stalled_request_bank_subtract
                    <= weight_request_bank_subtract;
                stalled_request_last <= weight_request_last;
                stalled_request_context <= weight_request_context;
            end
            if (request_accept)
                requests_seen <= requests_seen + 1;
            if (response_accept)
                responses_seen <= responses_seen + 1;
            if (request_accept && response_accept)
                simultaneous_request_response <= simultaneous_request_response + 1;
            if (weight_request_valid && !weight_request_ready)
                request_stalls <= request_stalls + 1;
            if (output_valid && !output_ready)
                output_stalls <= output_stalls + 1;
            if (context_occupancy > maximum_context_occupancy)
                maximum_context_occupancy <= context_occupancy;
            if (output_accept) begin
                integer packet;
                packet = output_tag;
                if (packet < 0 || packet >= PACKETS)
                    $fatal(1, "M43 output tag out of range: %0d", packet);
                if (seen[packet])
                    $fatal(1, "M43 duplicate output tag: %0d", packet);
                if (output_source_count !== expected_count[packet][8:0])
                    $fatal(1, "M43 source count mismatch packet=%0d got=%0d exp=%0d",
                           packet, output_source_count, expected_count[packet]);
                for (int lane = 0; lane < OUT_LANES; lane++) begin
                    if ($signed(output_acc[lane*ACC_W +: ACC_W])
                            !== expected_acc[packet][lane])
                        $fatal(1,
                            "M43 accumulator mismatch packet=%0d lane=%0d got=%0d exp=%0d",
                            packet, lane,
                            $signed(output_acc[lane*ACC_W +: ACC_W]),
                            expected_acc[packet][lane]);
                end
                seen[packet] <= 1'b1;
                outputs_seen <= outputs_seen + 1;
            end
        end
    end

    initial begin
        integer timeout;
        integer random_seed;
        clk_core = 1'b0;
        rst_core = 1'b1;
        command_valid = 1'b0;
        command_tag = '0;
        command_add_bits = '0;
        command_subtract_bits = '0;
        command_seed_acc = '0;
        weight_request_ready = 1'b0;
        weight_response_valid = 1'b0;
        weight_response_context = '0;
        weight_response_bank_valid = '0;
        weight_response_data = '0;
        output_ready = 1'b0;
        outputs_seen = 0;
        requests_seen = 0;
        responses_seen = 0;
        simultaneous_request_response = 0;
        maximum_context_occupancy = 0;
        request_stalls = 0;
        output_stalls = 0;
        automatic_responses = 1'b1;
        automatic_backpressure = 1'b1;
        output_was_stalled = 1'b0;
        request_was_stalled = 1'b0;
        random_seed = 32'h43a5c19d;
        random_seed = $urandom(random_seed);
        for (int packet = 0; packet < PACKETS; packet++)
            seen[packet] = 1'b0;
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;

        for (int packet = 0; packet < PACKETS; packet++) begin
            prepare_packet(packet);
            do @(negedge clk_core); while (!command_ready);
            command_valid = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            command_valid = 1'b0;
        end

        timeout = 0;
        while (outputs_seen != PACKETS && timeout < 200000) begin
            @(posedge clk_core);
            timeout = timeout + 1;
        end
        if (outputs_seen != PACKETS)
            $fatal(1, "M43 timeout outputs=%0d/%0d", outputs_seen, PACKETS);
        if (protocol_error)
            $fatal(1, "M43 unexpected protocol_error in legal traffic");
        if (requests_seen != responses_seen)
            $fatal(1, "M43 request/response conservation mismatch %0d/%0d",
                   requests_seen, responses_seen);
        if (maximum_context_occupancy != 4)
            $fatal(1, "M43 failed to cover four occupied contexts: %0d",
                   maximum_context_occupancy);
        if (simultaneous_request_response == 0 || request_stalls == 0
                || output_stalls == 0)
            $fatal(1, "M43 missing pipeline/stall coverage req_rsp=%0d req_stall=%0d out_stall=%0d",
                   simultaneous_request_response, request_stalls, output_stalls);

        // Fill the response metadata FIFO exactly, then reset it under stall.
        automatic_responses = 1'b0;
        automatic_backpressure = 1'b0;
        @(negedge clk_core);
        weight_request_ready = 1'b1;
        output_ready = 1'b1;
        weight_response_valid = 1'b0;
        for (int fill_context = 0; fill_context < 4; fill_context++) begin
            command_add_bits = {TILE_BITS{1'b1}};
            command_subtract_bits = '0;
            command_seed_acc = '0;
            command_tag = TAG_W'(48'h4d43_1000 + fill_context);
            do @(negedge clk_core); while (!command_ready);
            command_valid = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            command_valid = 1'b0;
        end
        timeout = 0;
        while (response_metadata_occupancy != 16 && timeout < 100) begin
            @(posedge clk_core);
            timeout = timeout + 1;
        end
        if (response_metadata_occupancy != 16)
            $fatal(1, "M43 metadata FIFO did not reach saturation");
        repeat (3) @(posedge clk_core);
        if (response_metadata_occupancy != 16 || weight_request_valid)
            $fatal(1, "M43 metadata FIFO did not backpressure at saturation");
        weight_request_ready = 1'b0;
        rst_core = 1'b1;
        repeat (2) @(posedge clk_core);
        rst_core = 1'b0;
        weight_request_ready = 1'b1;
        if (busy || context_occupancy != 0
                || response_metadata_occupancy != 0 || protocol_error)
            $fatal(1, "M43 reset did not clear saturated/stalled state");

        // A request held under memory backpressure must also reset cleanly.
        command_add_bits = '0;
        command_subtract_bits = '0;
        command_add_bits[11] = 1'b1;
        command_seed_acc = '0;
        command_tag = 48'h4d43_1004;
        do @(negedge clk_core); while (!command_ready);
        command_valid = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        command_valid = 1'b0;
        weight_request_ready = 1'b0;
        do @(posedge clk_core); while (!weight_request_valid);
        repeat (2) @(posedge clk_core);
        rst_core = 1'b1;
        repeat (2) @(posedge clk_core);
        rst_core = 1'b0;
        weight_request_ready = 1'b1;
        if (busy || weight_request_valid || output_valid || protocol_error)
            $fatal(1, "M43 reset-under-request-stall recovery failed");

        // Unexpected response must fail closed and reset must recover.
        @(negedge clk_core);
        weight_response_valid = 1'b1;
        weight_response_context = 2'b00;
        weight_response_bank_valid = 8'h01;
        @(posedge clk_core);
        @(negedge clk_core);
        weight_response_valid = 1'b0;
        if (!protocol_error)
            $fatal(1, "M43 unexpected response did not fail closed");
        @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b1;
        repeat (2) @(posedge clk_core);
        rst_core = 1'b0;
        weight_request_ready = 1'b1;
        output_ready = 1'b1;
        if (protocol_error)
            $fatal(1, "M43 reset did not clear protocol_error");

        // A response-context mismatch with live metadata must fail closed.
        command_add_bits = '0;
        command_subtract_bits = '0;
        command_add_bits[3] = 1'b1;
        command_seed_acc = '0;
        command_tag = 48'h4d43_0001;
        do @(negedge clk_core); while (!command_ready);
        command_valid = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        command_valid = 1'b0;
        do @(posedge clk_core); while (!request_accept);
        manual_response_context = weight_request_context;
        manual_response_bank_valid = weight_request_bank_valid;
        @(negedge clk_core);
        weight_response_valid = 1'b1;
        weight_response_context = manual_response_context ^ 2'b01;
        weight_response_bank_valid = manual_response_bank_valid;
        @(posedge clk_core);
        @(negedge clk_core);
        weight_response_valid = 1'b0;
        if (!protocol_error)
            $fatal(1, "M43 response-context mismatch did not fail closed");
        @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b1;
        repeat (2) @(posedge clk_core);
        rst_core = 1'b0;
        weight_request_ready = 1'b1;
        output_ready = 1'b1;
        if (protocol_error)
            $fatal(1, "M43 second reset did not clear protocol_error");

        // A response bank-valid mismatch with live metadata must fail closed.
        command_add_bits = '0;
        command_subtract_bits = '0;
        command_add_bits[5] = 1'b1;
        command_seed_acc = '0;
        command_tag = 48'h4d43_0003;
        do @(negedge clk_core); while (!command_ready);
        command_valid = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        command_valid = 1'b0;
        do @(posedge clk_core); while (!request_accept);
        manual_response_context = weight_request_context;
        manual_response_bank_valid = weight_request_bank_valid;
        @(negedge clk_core);
        weight_response_valid = 1'b1;
        weight_response_context = manual_response_context;
        weight_response_bank_valid = manual_response_bank_valid ^ 8'h80;
        @(posedge clk_core);
        @(negedge clk_core);
        weight_response_valid = 1'b0;
        if (!protocol_error)
            $fatal(1, "M43 response-bank mismatch did not fail closed");
        @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b1;
        repeat (2) @(posedge clk_core);
        rst_core = 1'b0;
        weight_request_ready = 1'b1;
        output_ready = 1'b1;
        if (protocol_error)
            $fatal(1, "M43 third reset did not clear protocol_error");

        // Overlapping add/sub descriptors must fail closed.
        command_add_bits = '0;
        command_subtract_bits = '0;
        command_add_bits[17] = 1'b1;
        command_subtract_bits[17] = 1'b1;
        command_seed_acc = '0;
        command_tag = '1;
        do @(negedge clk_core); while (!command_ready);
        command_valid = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        command_valid = 1'b0;
        if (!protocol_error)
            $fatal(1, "M43 overlapping signed masks did not fail closed");
        @(posedge clk_core);
        @(negedge clk_core);

        // Overflow from a legal descriptor/response is a fail-closed fault.
        rst_core = 1'b1;
        repeat (2) @(posedge clk_core);
        rst_core = 1'b0;
        weight_request_ready = 1'b1;
        output_ready = 1'b1;
        command_add_bits = '0;
        command_subtract_bits = '0;
        command_add_bits[0] = 1'b1;
        command_seed_acc = '0;
        for (int lane = 0; lane < OUT_LANES; lane++)
            command_seed_acc[lane*ACC_W +: ACC_W]
                = {1'b0, {(ACC_W-1){1'b1}}};
        command_tag = 48'h4d43_0002;
        do @(negedge clk_core); while (!command_ready);
        command_valid = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        command_valid = 1'b0;
        do @(posedge clk_core); while (!request_accept);
        manual_response_context = weight_request_context;
        manual_response_bank_valid = weight_request_bank_valid;
        @(negedge clk_core);
        weight_response_valid = 1'b1;
        weight_response_context = manual_response_context;
        weight_response_bank_valid = manual_response_bank_valid;
        weight_response_data = '0;
        for (int lane = 0; lane < OUT_LANES; lane++)
            weight_response_data[lane*8 +: 8] = 8'sd1;
        @(posedge clk_core);
        @(negedge clk_core);
        weight_response_valid = 1'b0;
        if (!protocol_error)
            $fatal(1, "M43 accumulator overflow did not fail closed");
        @(posedge clk_core);
        @(negedge clk_core);

`ifdef SVA_RUNTIME_ENABLED
        $display("M43_SVA_BOUND=1");
`else
        $display("M43_SVA_BOUND=0");
`endif
        $display("M43_ATTACKS metadata_fifo_saturation=1 reset_saturated=1 reset_request_stall=1 unexpected_response=1 response_context_mismatch=1 response_bank_mismatch=1 overlapping_masks=1 accumulator_overflow=1 request_stability=1 output_stability=1");
        $display("PASS M43 multicontext packets=%0d outputs=%0d requests=%0d simultaneous=%0d max_contexts=%0d request_stalls=%0d output_stalls=%0d",
                 PACKETS, outputs_seen, requests_seen,
                 simultaneous_request_response, maximum_context_occupancy,
                 request_stalls, output_stalls);
        $finish;
    end
endmodule

`default_nettype wire
