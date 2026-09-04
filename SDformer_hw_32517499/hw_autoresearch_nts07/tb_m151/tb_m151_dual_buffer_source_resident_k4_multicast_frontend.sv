`timescale 1ns/1ps
`default_nettype none

module tb_m151_dual_buffer_source_resident_k4_multicast_frontend;
    localparam int LANES = 96;
    localparam int KEYS = 48;
    localparam int DESCRIPTORS = KEYS * 2;
    localparam int QUEUE_DEPTH = 160;

    logic clk_core = 1'b0;
    logic rst_core;
    always #1.5 clk_core = ~clk_core;

    logic load_valid, load_ready, load_accept;
    logic [31:0] load_sequence;
    logic [8:0] load_row;
    logic [3:0] load_source;
    logic signed [10:0] load_vector [0:LANES-1];
    logic descriptor_valid, descriptor_ready, descriptor_accept;
    logic [31:0] descriptor_sequence;
    logic [8:0] descriptor_row;
    logic [3:0] descriptor_source;
    logic [3:0] descriptor_destination_valid;
    logic [2:0] descriptor_destination [0:3];
    logic [3:0] descriptor_negate;
    logic descriptor_last_for_source;
    logic multicast_valid, multicast_ready, multicast_accept;
    logic [31:0] multicast_sequence;
    logic [8:0] multicast_row;
    logic [3:0] multicast_source;
    logic [3:0] multicast_destination_valid;
    logic [2:0] multicast_destination [0:3];
    logic [3:0] multicast_negate;
    logic multicast_last_for_source;
    logic signed [10:0] multicast_vector [0:LANES-1];
    logic release_valid;
    logic [31:0] release_sequence;
    logic [8:0] release_row;
    logic [3:0] release_source;
    logic [1:0] resident_valid, resident_retiring;
    logic protocol_error, busy;

    logic force_multicast_ready;
    logic attack_mode;
    int unsigned cycle_count;
    int unsigned accepted_loads, accepted_descriptors, accepted_outputs;
    int unsigned releases, output_stalls, overlap_accepts, ii1_pairs;
    int unsigned protocol_attacks, prior_descriptor_cycle;
    int unsigned loader_progress;
    bit saw_both_slots, saw_tail1, saw_tail2, saw_tail3, saw_full4;
    bit saw_pos1023, saw_neg1024;

    logic [31:0] expected_sequence [0:QUEUE_DEPTH-1];
    logic [8:0] expected_row [0:QUEUE_DEPTH-1];
    logic [3:0] expected_source [0:QUEUE_DEPTH-1];
    logic [3:0] expected_valid [0:QUEUE_DEPTH-1];
    logic [2:0] expected_destination [0:QUEUE_DEPTH-1][0:3];
    logic [3:0] expected_negate [0:QUEUE_DEPTH-1];
    logic expected_last [0:QUEUE_DEPTH-1];
    logic signed [10:0] expected_vector [0:QUEUE_DEPTH-1][0:LANES-1];
    int unsigned queue_head, queue_tail;

    m151_dual_buffer_source_resident_k4_multicast_frontend dut (.*);
    m151_dual_buffer_source_resident_k4_multicast_frontend_assertions sva (.*);

    function automatic logic signed [10:0] generated_value(
        input int unsigned key,
        input int unsigned lane
    );
        int signed raw;
        if (key == 0 && lane == 0)
            raw = 1023;
        else if (key == 1 && lane == 0)
            raw = -1024;
        else begin
            raw = (key * 137 + lane * 61 + lane / 5) % 2048;
            raw -= 1024;
        end
        return raw[10:0];
    endfunction

    task automatic clear_load;
        load_valid = 1'b0;
        load_sequence = '0;
        load_row = '0;
        load_source = '0;
        for (int lane = 0; lane < LANES; lane++)
            load_vector[lane] = '0;
    endtask

    task automatic clear_descriptor;
        descriptor_valid = 1'b0;
        descriptor_sequence = '0;
        descriptor_row = '0;
        descriptor_source = '0;
        descriptor_destination_valid = 4'b0001;
        descriptor_negate = '0;
        descriptor_last_for_source = 1'b0;
        for (int destination = 0; destination < 4; destination++)
            descriptor_destination[destination] = '0;
    endtask

    task automatic apply_reset;
        @(negedge clk_core);
        rst_core = 1'b1;
        clear_load();
        clear_descriptor();
        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        repeat (2) @(posedge clk_core);
        @(negedge clk_core);
    endtask

    task automatic drive_load(input int unsigned key);
        load_valid = 1'b1;
        load_sequence = key;
        load_row = key[8:0];
        load_source = key[3:0];
        for (int lane = 0; lane < LANES; lane++)
            load_vector[lane] = generated_value(key, lane);
        do @(posedge clk_core); while (!load_accept);
        loader_progress = key + 1;
        @(negedge clk_core);
    endtask

    task automatic drive_descriptor(
        input int unsigned key,
        input int unsigned descriptor_index
    );
        logic [3:0] shape;
        while (loader_progress <= key)
            @(negedge clk_core);
        descriptor_valid = 1'b1;
        descriptor_sequence = key;
        descriptor_row = key[8:0];
        descriptor_source = key[3:0];
        if (key == 0 && descriptor_index == 0)
            shape = 4'b0001;
        else if (key == 1 && descriptor_index == 0)
            shape = 4'b0011;
        else if (key == 2 && descriptor_index == 0)
            shape = 4'b0111;
        else
            shape = 4'b1111;
        descriptor_destination_valid = shape;
        descriptor_negate = (key + descriptor_index * 5) & 4'hf;
        descriptor_last_for_source = descriptor_index == 1;
        for (int destination = 0; destination < 4; destination++)
            descriptor_destination[destination]
                = descriptor_index * 4 + destination;
        do @(posedge clk_core); while (!descriptor_accept);
        @(negedge clk_core);
    endtask

    always_comb begin
        if (rst_core)
            multicast_ready = 1'b0;
        else if (force_multicast_ready)
            multicast_ready = 1'b1;
        else
            multicast_ready = (cycle_count % 11) != 4
                           && (cycle_count % 11) != 5;
    end

    always @(posedge clk_core) begin
        if (rst_core)
            cycle_count <= 0;
        else
            cycle_count <= cycle_count + 1;
    end

    always @(posedge clk_core) begin : exact_scoreboard
        if (rst_core) begin
            queue_head = 0;
            queue_tail = 0;
        end else if (!attack_mode) begin
            if (resident_valid == 2'b11)
                saw_both_slots <= 1'b1;
            if (load_accept) begin
                accepted_loads <= accepted_loads + 1;
                if (load_accept && descriptor_accept)
                    overlap_accepts <= overlap_accepts + 1;
            end
            if (descriptor_accept) begin
                accepted_descriptors <= accepted_descriptors + 1;
                if (accepted_descriptors != 0
                        && cycle_count == prior_descriptor_cycle + 1)
                    ii1_pairs <= ii1_pairs + 1;
                prior_descriptor_cycle <= cycle_count;
                expected_sequence[queue_tail] = descriptor_sequence;
                expected_row[queue_tail] = descriptor_row;
                expected_source[queue_tail] = descriptor_source;
                expected_valid[queue_tail]
                    = descriptor_destination_valid;
                expected_negate[queue_tail] = descriptor_negate;
                expected_last[queue_tail] = descriptor_last_for_source;
                for (int destination = 0; destination < 4; destination++)
                    expected_destination[queue_tail][destination]
                        = descriptor_destination[destination];
                for (int lane = 0; lane < LANES; lane++)
                    expected_vector[queue_tail][lane]
                        = generated_value(descriptor_sequence, lane);
                queue_tail = queue_tail + 1;
            end
            if (multicast_valid) begin
                if (queue_head == queue_tail)
                    $fatal(1, "M151 multicast without expectation");
                if (multicast_sequence !== expected_sequence[queue_head]
                        || multicast_row !== expected_row[queue_head]
                        || multicast_source !== expected_source[queue_head]
                        || multicast_destination_valid
                           !== expected_valid[queue_head]
                        || multicast_negate !== expected_negate[queue_head]
                        || multicast_last_for_source
                           !== expected_last[queue_head])
                    $fatal(1, "M151 multicast metadata mismatch");
                for (int destination = 0; destination < 4; destination++) begin
                    if (multicast_destination[destination]
                            !== expected_destination[queue_head][destination])
                        $fatal(1, "M151 destination mismatch");
                end
                for (int lane = 0; lane < LANES; lane++) begin
                    if (multicast_vector[lane]
                            !== expected_vector[queue_head][lane])
                        $fatal(1, "M151 resident vector mismatch lane=%0d",
                               lane);
                end
                if (!multicast_ready)
                    output_stalls <= output_stalls + 1;
            end
            if (multicast_accept) begin
                accepted_outputs <= accepted_outputs + 1;
                case (multicast_destination_valid)
                    4'b0001: saw_tail1 <= 1'b1;
                    4'b0011: saw_tail2 <= 1'b1;
                    4'b0111: saw_tail3 <= 1'b1;
                    4'b1111: saw_full4 <= 1'b1;
                    default: $fatal(1, "M151 illegal output shape");
                endcase
                if (multicast_vector[0] == 11'sd1023)
                    saw_pos1023 <= 1'b1;
                if (multicast_vector[0] == -11'sd1024)
                    saw_neg1024 <= 1'b1;
                queue_head = queue_head + 1;
            end
            if (release_valid) begin
                if (!multicast_accept || !multicast_last_for_source
                        || release_sequence != multicast_sequence
                        || release_row != multicast_row
                        || release_source != multicast_source)
                    $fatal(1, "M151 release mismatch");
                releases <= releases + 1;
            end
        end
    end

    initial begin : stimulus
        int watchdog;

        rst_core = 1'b1;
        clear_load();
        clear_descriptor();
        force_multicast_ready = 1'b0;
        attack_mode = 1'b0;
        cycle_count = 0;
        accepted_loads = 0;
        accepted_descriptors = 0;
        accepted_outputs = 0;
        releases = 0;
        output_stalls = 0;
        overlap_accepts = 0;
        ii1_pairs = 0;
        protocol_attacks = 0;
        prior_descriptor_cycle = 0;
        loader_progress = 0;
        saw_both_slots = 1'b0;
        saw_tail1 = 1'b0;
        saw_tail2 = 1'b0;
        saw_tail3 = 1'b0;
        saw_full4 = 1'b0;
        saw_pos1023 = 1'b0;
        saw_neg1024 = 1'b0;

        apply_reset();
        fork
            begin : loader
                for (int key = 0; key < KEYS; key++)
                    drive_load(key);
                clear_load();
            end
            begin : descriptor_sender
                for (int key = 0; key < KEYS; key++) begin
                    drive_descriptor(key, 0);
                    drive_descriptor(key, 1);
                end
                clear_descriptor();
            end
        join

        watchdog = 0;
        while ((accepted_outputs != DESCRIPTORS
                || releases != KEYS || busy) && watchdog < 3000) begin
            @(posedge clk_core);
            watchdog++;
        end
        if (watchdog == 3000)
            $fatal(1, "M151 normal traffic failed to drain");
        repeat (3) @(posedge clk_core);
        if (protocol_error || accepted_loads != KEYS
                || accepted_descriptors != DESCRIPTORS
                || accepted_outputs != DESCRIPTORS || releases != KEYS
                || queue_head != queue_tail || output_stalls == 0
                || overlap_accepts == 0 || ii1_pairs == 0
                || !saw_both_slots || !saw_tail1 || !saw_tail2
                || !saw_tail3 || !saw_full4
                || !saw_pos1023 || !saw_neg1024)
            $fatal(1, "M151 normal contract incomplete");

        attack_mode = 1'b1;
        force_multicast_ready = 1'b1;
        apply_reset();
        load_valid = 1'b1;
        load_sequence = 32'h100;
        load_row = 9'h100;
        load_source = 4'h3;
        for (int lane = 0; lane < LANES; lane++)
            load_vector[lane] = lane;
        do @(posedge clk_core); while (!load_accept);
        @(negedge clk_core);
        // Exact duplicate resident identity is a fail-closed protocol attack.
        @(posedge clk_core);
        #0.1;
        if (!protocol_error || load_accept || multicast_valid)
            $fatal(1, "M151 duplicate load not quarantined");
        protocol_attacks++;

        apply_reset();
        descriptor_valid = 1'b1;
        descriptor_sequence = 32'h200;
        descriptor_row = 9'h001;
        descriptor_source = 4'h2;
        descriptor_destination_valid = 4'b1111;
        for (int destination = 0; destination < 4; destination++)
            descriptor_destination[destination] = destination;
        @(posedge clk_core);
        #0.1;
        if (!protocol_error || descriptor_accept || multicast_valid)
            $fatal(1, "M151 missing resident event not quarantined");
        protocol_attacks++;

        apply_reset();
        load_valid = 1'b1;
        load_sequence = 32'h300;
        load_row = 9'h002;
        load_source = 4'h4;
        do @(posedge clk_core); while (!load_accept);
        @(negedge clk_core);
        load_valid = 1'b0;
        descriptor_valid = 1'b1;
        descriptor_sequence = 32'h300;
        descriptor_row = 9'h002;
        descriptor_source = 4'h4;
        descriptor_destination_valid = 4'b0101;
        descriptor_destination[0] = 0;
        descriptor_destination[1] = 1;
        descriptor_destination[2] = 2;
        descriptor_destination[3] = 3;
        @(posedge clk_core);
        #0.1;
        if (!protocol_error || descriptor_accept)
            $fatal(1, "M151 gapped destination shape not quarantined");
        protocol_attacks++;

        apply_reset();
        load_valid = 1'b1;
        load_sequence = 32'h400;
        load_row = 9'h003;
        load_source = 4'h5;
        do @(posedge clk_core); while (!load_accept);
        @(negedge clk_core);
        load_valid = 1'b0;
        descriptor_valid = 1'b1;
        descriptor_sequence = 32'h400;
        descriptor_row = 9'h003;
        descriptor_source = 4'h5;
        descriptor_destination_valid = 4'b1111;
        descriptor_destination[0] = 0;
        descriptor_destination[1] = 1;
        descriptor_destination[2] = 1;
        descriptor_destination[3] = 3;
        @(posedge clk_core);
        #0.1;
        if (!protocol_error || descriptor_accept)
            $fatal(1, "M151 duplicate destination not quarantined");
        protocol_attacks++;

        $display("PASS M151 dual-buffer source-resident K4 multicast frontend VCS keys=%0d descriptors=%0d outputs=%0d releases=%0d stalls=%0d load_descriptor_overlap=%0d ii1_pairs=%0d protocol_attacks=%0d lanes=96 resident_slots=2 source_vector_bits=1056 destination_ports=4 stable_source_vector_once=true independent_negate_metadata=true accumulator_write_ports=false pwp_storage=false sram_macro=false physical_speedup=false system_speedup=false headline=false",
                 KEYS, DESCRIPTORS, accepted_outputs, releases,
                 output_stalls, overlap_accepts, ii1_pairs,
                 protocol_attacks);
        $finish;
    end

    initial begin : watchdog
        #500000;
        $fatal(1, "M151 watchdog timeout");
    end
endmodule

`default_nettype wire
