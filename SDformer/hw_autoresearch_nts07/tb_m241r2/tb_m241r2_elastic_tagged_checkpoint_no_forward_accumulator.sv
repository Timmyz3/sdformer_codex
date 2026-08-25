`timescale 1ns/1ps
`default_nettype none

module tb_m241r2_elastic_tagged_checkpoint_no_forward_accumulator;
    localparam int LANES = 8;
    localparam int ROWS = 384;
    localparam int ACC_BITS = 19;
    localparam int ACC_MIN = -(1 << (ACC_BITS - 1));
    localparam int ACC_MAX = (1 << (ACC_BITS - 1)) - 1;
    localparam int ROW_BITS = 9;
    localparam int ACC_ADDR_BITS = 10;
    localparam int VECTOR_BITS = LANES * ACC_BITS;
    localparam int MAX_DESCRIPTORS = 126;

    logic clk_core, rst_core;
    logic loader_binding_valid;
    logic [1:0] loader_binding_operator;
    logic [8:0] loader_binding_partition;
    logic [15:0] loader_binding_weight_epoch;
    logic [31:0] loader_binding_payload_id;
    logic context_open_valid, context_open_ready, context_open_accept;
    logic [31:0] context_open_sequence;
    logic [1:0] context_open_operator;
    logic [8:0] context_open_partition;
    logic [15:0] context_open_window;
    logic [15:0] context_open_weight_epoch;
    logic [31:0] context_open_payload_id;
    logic descriptor_valid, descriptor_ready, descriptor_accept;
    logic [31:0] descriptor_sequence;
    logic [1:0] descriptor_operator;
    logic [8:0] descriptor_partition;
    logic [15:0] descriptor_window;
    logic [15:0] descriptor_weight_epoch;
    logic [31:0] descriptor_payload_id;
    logic [15:0] descriptor_order;
    logic [ROW_BITS-1:0] descriptor_row;
    logic [3:0] descriptor_source;
    logic [3:0] descriptor_destination_valid;
    logic [2:0] descriptor_destination [0:3];
    logic [3:0] descriptor_negate;
    logic descriptor_last;
    logic context_close_valid, context_close_ready, context_close_accept;
    logic window_done;

    logic weight_req_valid, weight_req_ready, weight_req_accept;
    logic [3:0] weight_req_bank_valid;
    logic [4:0] weight_req_addr [0:3];
    logic [31:0] weight_req_sequence;
    logic [1:0] weight_req_operator;
    logic [8:0] weight_req_partition;
    logic [15:0] weight_req_window;
    logic [15:0] weight_req_weight_epoch;
    logic [31:0] weight_req_payload_id;
    logic [15:0] weight_req_order;
    logic [3:0] weight_req_source;
    logic weight_req_half;
    logic weight_rsp_valid, weight_rsp_ready, weight_rsp_accept;
    logic [3:0] weight_rsp_bank_valid;
    logic [31:0] weight_rsp_sequence;
    logic [1:0] weight_rsp_operator;
    logic [8:0] weight_rsp_partition;
    logic [15:0] weight_rsp_window;
    logic [15:0] weight_rsp_weight_epoch;
    logic [31:0] weight_rsp_payload_id;
    logic [15:0] weight_rsp_order;
    logic [3:0] weight_rsp_source;
    logic weight_rsp_half;
    logic signed [7:0] weight_rsp_data [0:3][0:LANES-1];
    logic [3:0] weight_cache_hit, weight_cache_miss;

    logic acc_req_valid, acc_req_ready, acc_req_accept;
    logic [3:0] acc_req_bank_valid;
    logic [ACC_ADDR_BITS-1:0] acc_req_addr [0:3];
    logic [31:0] acc_req_sequence;
    logic [15:0] acc_req_window;
    logic [15:0] acc_req_weight_epoch;
    logic [31:0] acc_req_payload_id;
    logic [15:0] acc_req_order;
    logic acc_rsp_valid, acc_rsp_ready, acc_rsp_accept;
    logic [3:0] acc_rsp_bank_valid;
    logic [31:0] acc_rsp_sequence;
    logic [15:0] acc_rsp_window;
    logic [15:0] acc_rsp_weight_epoch;
    logic [31:0] acc_rsp_payload_id;
    logic [15:0] acc_rsp_order;
    logic signed [ACC_BITS-1:0] acc_rsp_data [0:3][0:LANES-1];
    logic [3:0] acc_wr_en;
    logic [ACC_ADDR_BITS-1:0] acc_wr_addr [0:3];
    logic signed [ACC_BITS-1:0] acc_wr_data [0:3][0:LANES-1];

    logic commit_valid, commit_ready, commit_accept;
    logic [15:0] commit_order, commit_window;
    logic [ROW_BITS-1:0] commit_row;
    logic [3:0] commit_bank_valid;
    logic [2:0] commit_destination [0:3];
    logic commit_last;
    logic abort_valid, abort_ready, abort_accept;
    logic [15:0] abort_order, abort_window;
    logic [1:0] abort_discarded_tokens;
    logic context_abort;
    logic rmw_alias_stall;
    logic [15:0] next_descriptor_order;
    logic context_active, protocol_error, overflow_error, busy;

    logic [63:0] descriptor_memory [0:MAX_DESCRIPTORS-1];
    logic [31:0] meta_memory [0:15];
    logic [7:0] weight_flat [0:4*32*LANES-1];
    logic signed [ACC_BITS-1:0] acc_memory
        [0:3][0:2*ROWS-1][0:LANES-1];
    integer signed reference_memory
        [0:3][0:2*ROWS-1][0:LANES-1];

    typedef struct packed {
        logic [ACC_ADDR_BITS-1:0] address;
        logic [VECTOR_BITS-1:0] data;
    } expected_write_t;
    expected_write_t expected_write_q [0:3][$];

    logic weight_pending_q;
    int weight_countdown_q;
    int weight_latency_q;
    logic [3:0] weight_pending_bank_valid_q;
    logic [4:0] weight_pending_addr_q [0:3];
    logic [31:0] weight_pending_sequence_q;
    logic [1:0] weight_pending_operator_q;
    logic [8:0] weight_pending_partition_q;
    logic [15:0] weight_pending_window_q;
    logic [15:0] weight_pending_epoch_q;
    logic [31:0] weight_pending_payload_q;
    logic [15:0] weight_pending_order_q;
    logic [3:0] weight_pending_source_q;
    logic weight_pending_half_q;

    logic acc_pending_q;
    int acc_countdown_q;
    int acc_latency_q;
    logic [3:0] acc_pending_bank_valid_q;
    logic [ACC_ADDR_BITS-1:0] acc_pending_addr_q [0:3];
    logic [31:0] acc_pending_sequence_q;
    logic [15:0] acc_pending_window_q;
    logic [15:0] acc_pending_epoch_q;
    logic [31:0] acc_pending_payload_q;
    logic [15:0] acc_pending_order_q;

    string vector_dir;
    logic random_macro_mode;
    logic auto_commit_stall;
    logic force_commit_stall;
    logic force_abort_stall;
    logic corrupt_next_weight_response;
    logic corrupt_next_acc_response;
    int configured_weight_latency;
    int configured_acc_latency;
    int cycle_count;
    int weight_request_accepts;
    int weight_request_groups;
    int weight_response_accepts;
    int weight_request_stalls;
    int weight_response_stalls;
    int acc_request_accepts;
    int acc_request_groups;
    int acc_response_accepts;
    int acc_request_stalls;
    int acc_response_stalls;
    int commit_accepts;
    int commit_stalls;
    int abort_accepts;
    int abort_stalls;
    int exact_write_checks;
    int exact_lane_checks;
    int exact_mismatches;
    int accepted_descriptors;
    int accepted_groups;
    int cache_hits;
    int expected_overflows;
    int stale_weight_response_attacks;
    int stale_acc_response_attacks;
    int loader_binding_attacks;
    int overflow_younger_attacks;
    int overflow_success_commits;
    int overflow_writes;
    int recovery_commits;
    int scenario_passes;
    int alias_stalls;

    m241r2_elastic_tagged_checkpoint_no_forward_accumulator #(
        .LANES(LANES), .ROWS(ROWS), .ACC_BITS(ACC_BITS)
    ) dut (.*);

    m241r2_elastic_tagged_checkpoint_no_forward_accumulator_assertions #(
        .LANES(LANES), .ROWS(ROWS), .ACC_BITS(ACC_BITS)
    ) checks (.*);

    always #1.5 clk_core = ~clk_core;
    assign weight_req_ready = !rst_core && !weight_pending_q
                            && !weight_rsp_valid
                            && (!random_macro_mode || cycle_count % 5 != 1);
    assign acc_req_ready = !rst_core && !acc_pending_q && !acc_rsp_valid
                         && (!random_macro_mode || cycle_count % 7 != 2);
    assign commit_ready = !force_commit_stall
                        && (!auto_commit_stall || cycle_count % 11 != 4);
    assign abort_ready = !force_abort_stall;

    always @(posedge clk_core) begin : monitor
        if (!rst_core) begin
            cycle_count <= cycle_count + 1;
            if (weight_req_accept) begin
                weight_request_accepts <= weight_request_accepts + 1;
                weight_request_groups <= weight_request_groups
                                       + $countones(weight_req_bank_valid);
            end
            if (weight_rsp_accept)
                weight_response_accepts <= weight_response_accepts + 1;
            if (weight_req_valid && !weight_req_ready)
                weight_request_stalls <= weight_request_stalls + 1;
            if (weight_rsp_valid && !weight_rsp_ready)
                weight_response_stalls <= weight_response_stalls + 1;
            if (acc_req_accept) begin
                acc_request_accepts <= acc_request_accepts + 1;
                acc_request_groups <= acc_request_groups
                                   + $countones(acc_req_bank_valid);
            end
            if (acc_rsp_accept)
                acc_response_accepts <= acc_response_accepts + 1;
            if (acc_req_valid && !acc_req_ready)
                acc_request_stalls <= acc_request_stalls + 1;
            if (acc_rsp_valid && !acc_rsp_ready)
                acc_response_stalls <= acc_response_stalls + 1;
            if (commit_accept)
                commit_accepts <= commit_accepts + 1;
            if (commit_valid && !commit_ready)
                commit_stalls <= commit_stalls + 1;
            if (abort_accept)
                abort_accepts <= abort_accepts + 1;
            if (abort_valid && !abort_ready)
                abort_stalls <= abort_stalls + 1;
            if (descriptor_accept) begin
                accepted_descriptors <= accepted_descriptors + 1;
                accepted_groups <= accepted_groups
                                 + $countones(descriptor_destination_valid);
                cache_hits <= cache_hits + $countones(weight_cache_hit);
            end
            if (rmw_alias_stall)
                alias_stalls <= alias_stalls + 1;
            if (overflow_error && commit_accept)
                overflow_success_commits <= overflow_success_commits + 1;
            if (overflow_error && |acc_wr_en)
                overflow_writes <= overflow_writes + 1;
        end
    end

    always @(posedge clk_core) begin : elastic_weight_macro
        if (rst_core) begin
            weight_pending_q <= 1'b0;
            weight_rsp_valid <= 1'b0;
            weight_countdown_q <= 0;
        end else begin
            if (weight_req_accept) begin
                weight_pending_q <= 1'b1;
                weight_countdown_q <= random_macro_mode
                    ? 1 + (weight_request_accepts % 3)
                    : configured_weight_latency;
                weight_pending_bank_valid_q <= weight_req_bank_valid;
                weight_pending_sequence_q <= weight_req_sequence;
                weight_pending_operator_q <= weight_req_operator;
                weight_pending_partition_q <= weight_req_partition;
                weight_pending_window_q <= weight_req_window;
                weight_pending_epoch_q <= weight_req_weight_epoch;
                weight_pending_payload_q <= weight_req_payload_id;
                weight_pending_order_q <= weight_req_order;
                weight_pending_source_q <= weight_req_source;
                weight_pending_half_q <= weight_req_half;
                for (int bank = 0; bank < 4; bank++)
                    weight_pending_addr_q[bank] <= weight_req_addr[bank];
            end
            if (weight_pending_q) begin
                if (weight_countdown_q > 1) begin
                    weight_countdown_q <= weight_countdown_q - 1;
                end else if (!random_macro_mode || cycle_count % 6 != 3) begin
                    weight_pending_q <= 1'b0;
                    weight_rsp_valid <= 1'b1;
                    weight_rsp_bank_valid <= weight_pending_bank_valid_q;
                    weight_rsp_sequence <= weight_pending_sequence_q;
                    weight_rsp_operator <= weight_pending_operator_q;
                    weight_rsp_partition <= weight_pending_partition_q;
                    weight_rsp_window <= weight_pending_window_q;
                    weight_rsp_weight_epoch <= corrupt_next_weight_response
                        ? weight_pending_epoch_q ^ 16'h0001
                        : weight_pending_epoch_q;
                    weight_rsp_payload_id <= weight_pending_payload_q;
                    weight_rsp_order <= weight_pending_order_q;
                    weight_rsp_source <= weight_pending_source_q;
                    weight_rsp_half <= weight_pending_half_q;
                    for (int bank = 0; bank < 4; bank++) begin
                        for (int lane = 0; lane < LANES; lane++) begin
                            weight_rsp_data[bank][lane] <= $signed(weight_flat[
                                (bank * 32 + weight_pending_addr_q[bank])
                                * LANES + lane]);
                        end
                    end
                    if (corrupt_next_weight_response)
                        corrupt_next_weight_response <= 1'b0;
                end
            end
            if (weight_rsp_accept)
                weight_rsp_valid <= 1'b0;
        end
    end

    always @(posedge clk_core) begin : elastic_accumulator_macro
        if (rst_core) begin
            acc_pending_q <= 1'b0;
            acc_rsp_valid <= 1'b0;
            acc_countdown_q <= 0;
        end else begin
            if (acc_req_accept) begin
                acc_pending_q <= 1'b1;
                acc_countdown_q <= random_macro_mode
                    ? 1 + (acc_request_accepts % 3)
                    : configured_acc_latency;
                acc_pending_bank_valid_q <= acc_req_bank_valid;
                acc_pending_sequence_q <= acc_req_sequence;
                acc_pending_window_q <= acc_req_window;
                acc_pending_epoch_q <= acc_req_weight_epoch;
                acc_pending_payload_q <= acc_req_payload_id;
                acc_pending_order_q <= acc_req_order;
                for (int bank = 0; bank < 4; bank++)
                    acc_pending_addr_q[bank] <= acc_req_addr[bank];
            end
            if (acc_pending_q) begin
                if (acc_countdown_q > 1) begin
                    acc_countdown_q <= acc_countdown_q - 1;
                end else if (!random_macro_mode || cycle_count % 8 != 5) begin
                    acc_pending_q <= 1'b0;
                    acc_rsp_valid <= 1'b1;
                    acc_rsp_bank_valid <= acc_pending_bank_valid_q;
                    acc_rsp_sequence <= acc_pending_sequence_q;
                    acc_rsp_window <= acc_pending_window_q;
                    acc_rsp_weight_epoch <= acc_pending_epoch_q;
                    acc_rsp_payload_id <= corrupt_next_acc_response
                        ? acc_pending_payload_q ^ 32'h0000_0001
                        : acc_pending_payload_q;
                    acc_rsp_order <= acc_pending_order_q;
                    for (int bank = 0; bank < 4; bank++) begin
                        for (int lane = 0; lane < LANES; lane++) begin
                            acc_rsp_data[bank][lane] <= acc_memory[bank]
                                [acc_pending_addr_q[bank]][lane];
                        end
                    end
                    if (corrupt_next_acc_response)
                        corrupt_next_acc_response <= 1'b0;
                end
            end
            if (acc_rsp_accept)
                acc_rsp_valid <= 1'b0;
            for (int bank = 0; bank < 4; bank++) begin
                if (acc_wr_en[bank]) begin
                    for (int lane = 0; lane < LANES; lane++)
                        acc_memory[bank][acc_wr_addr[bank]][lane]
                            <= acc_wr_data[bank][lane];
                end
            end
        end
    end

    always @(posedge clk_core) begin : exact_scoreboard
        expected_write_t expected;
        expected_write_t produced;
        int bank;
        int address;
        int weight_address;
        int signed weight_value;
        int signed contribution;
        int signed new_value;
        bit descriptor_overflow;
        if (rst_core || context_open_accept) begin
            for (int bank_index = 0; bank_index < 4; bank_index++) begin
                expected_write_q[bank_index].delete();
                for (int address_index = 0; address_index < 2 * ROWS;
                        address_index++) begin
                    for (int lane = 0; lane < LANES; lane++)
                        reference_memory[bank_index][address_index][lane] = 0;
                end
            end
        end else begin
            for (int bank_index = 0; bank_index < 4; bank_index++) begin
                if (acc_wr_en[bank_index]) begin
                    if (expected_write_q[bank_index].size() == 0)
                        $fatal(1, "M241r2 unexpected write bank=%0d",
                               bank_index);
                    expected = expected_write_q[bank_index].pop_front();
                    if (acc_wr_addr[bank_index] !== expected.address)
                        $fatal(1, "M241r2 address mismatch bank=%0d",
                               bank_index);
                    for (int lane = 0; lane < LANES; lane++) begin
                        if (acc_wr_data[bank_index][lane]
                                !== $signed(expected.data[
                                    lane * ACC_BITS +: ACC_BITS])) begin
                            exact_mismatches = exact_mismatches + 1;
                            $fatal(1,
                                "M241r2 integer mismatch bank=%0d lane=%0d expected=%0d actual=%0d",
                                bank_index, lane,
                                $signed(expected.data[
                                    lane * ACC_BITS +: ACC_BITS]),
                                $signed(acc_wr_data[bank_index][lane]));
                        end
                        exact_lane_checks = exact_lane_checks + 1;
                    end
                    exact_write_checks = exact_write_checks + 1;
                end
            end
            if (descriptor_accept) begin
                descriptor_overflow = 1'b0;
                for (int tuple = 0; tuple < 4; tuple++) begin
                    if (descriptor_destination_valid[tuple]) begin
                        bank = descriptor_destination[tuple][1:0];
                        address = (descriptor_destination[tuple][2]
                                   ? ROWS : 0) + descriptor_row;
                        weight_address = {
                            descriptor_destination[tuple][2],
                            descriptor_source};
                        for (int lane = 0; lane < LANES; lane++) begin
                            weight_value = $signed(weight_flat[
                                (bank * 32 + weight_address) * LANES
                                + lane]);
                            contribution = descriptor_negate[tuple]
                                ? -weight_value : weight_value;
                            new_value = reference_memory[bank][address][lane]
                                      + contribution;
                            if (new_value < ACC_MIN || new_value > ACC_MAX)
                                descriptor_overflow = 1'b1;
                        end
                    end
                end
                if (descriptor_overflow) begin
                    expected_overflows = expected_overflows + 1;
                end else begin
                    for (int tuple = 0; tuple < 4; tuple++) begin
                        if (descriptor_destination_valid[tuple]) begin
                            bank = descriptor_destination[tuple][1:0];
                            address = (descriptor_destination[tuple][2]
                                       ? ROWS : 0) + descriptor_row;
                            weight_address = {
                                descriptor_destination[tuple][2],
                                descriptor_source};
                            produced.address = address;
                            produced.data = '0;
                            for (int lane = 0; lane < LANES; lane++) begin
                                weight_value = $signed(weight_flat[
                                    (bank * 32 + weight_address) * LANES
                                    + lane]);
                                contribution = descriptor_negate[tuple]
                                    ? -weight_value : weight_value;
                                reference_memory[bank][address][lane]
                                    = reference_memory[bank][address][lane]
                                      + contribution;
                                produced.data[
                                    lane * ACC_BITS +: ACC_BITS]
                                    = reference_memory[bank][address][lane]
                                      [ACC_BITS-1:0];
                            end
                            expected_write_q[bank].push_back(produced);
                        end
                    end
                end
            end
            if (abort_accept) begin
                for (int bank_index = 0; bank_index < 4; bank_index++)
                    expected_write_q[bank_index].delete();
            end
        end
    end

    task automatic clear_drivers;
        begin
            context_open_valid = 1'b0;
            descriptor_valid = 1'b0;
            context_close_valid = 1'b0;
            force_commit_stall = 1'b0;
            force_abort_stall = 1'b0;
        end
    endtask

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            clear_drivers();
            loader_binding_valid = 1'b0;
            auto_commit_stall = 1'b0;
            random_macro_mode = 1'b0;
            corrupt_next_weight_response = 1'b0;
            corrupt_next_acc_response = 1'b0;
            rst_core = 1'b1;
            repeat (3) @(negedge clk_core);
            rst_core = 1'b0;
            repeat (1) @(negedge clk_core);
        end
    endtask

    task automatic bind_loader;
        begin
            loader_binding_valid = 1'b1;
            loader_binding_operator = meta_memory[2][1:0];
            loader_binding_partition = meta_memory[3][8:0];
            loader_binding_weight_epoch = meta_memory[4][15:0];
            loader_binding_payload_id = {meta_memory[4][15:0],
                                         7'd0, meta_memory[3][8:0]};
        end
    endtask

    task automatic open_context;
        begin
            bind_loader();
            context_open_sequence = meta_memory[1];
            context_open_operator = meta_memory[2][1:0];
            context_open_partition = meta_memory[3][8:0];
            context_open_window = meta_memory[5][15:0];
            context_open_weight_epoch = meta_memory[4][15:0];
            context_open_payload_id = loader_binding_payload_id;
            context_open_valid = 1'b1;
            do @(posedge clk_core); while (!context_open_accept);
            @(negedge clk_core);
            context_open_valid = 1'b0;
        end
    endtask

    task automatic drive_descriptor(
        input logic [15:0] order_id,
        input logic [ROW_BITS-1:0] row_id,
        input logic [3:0] source_id,
        input logic [3:0] valid_mask,
        input logic [2:0] destination0,
        input logic [2:0] destination1,
        input logic [2:0] destination2,
        input logic [2:0] destination3,
        input logic [3:0] negate_mask,
        input logic last_flag);
        begin
            descriptor_sequence = meta_memory[1];
            descriptor_operator = meta_memory[2][1:0];
            descriptor_partition = meta_memory[3][8:0];
            descriptor_window = meta_memory[5][15:0];
            descriptor_weight_epoch = meta_memory[4][15:0];
            descriptor_payload_id = loader_binding_payload_id;
            descriptor_order = order_id;
            descriptor_row = row_id;
            descriptor_source = source_id;
            descriptor_destination_valid = valid_mask;
            descriptor_destination[0] = destination0;
            descriptor_destination[1] = destination1;
            descriptor_destination[2] = destination2;
            descriptor_destination[3] = destination3;
            descriptor_negate = negate_mask;
            descriptor_last = last_flag;
            descriptor_valid = 1'b1;
            do @(posedge clk_core); while (!descriptor_accept);
            @(negedge clk_core);
            descriptor_valid = 1'b0;
        end
    endtask

    task automatic drive_packed(input logic [63:0] packed_word);
        begin
            drive_descriptor(
                packed_word[15:0], packed_word[24:16],
                packed_word[28:25], packed_word[32:29],
                packed_word[35:33], packed_word[38:36],
                packed_word[41:39], packed_word[44:42],
                packed_word[48:45], packed_word[49]);
        end
    endtask

    task automatic wait_scoreboard_empty;
        begin
            wait (expected_write_q[0].size() == 0
                  && expected_write_q[1].size() == 0
                  && expected_write_q[2].size() == 0
                  && expected_write_q[3].size() == 0
                  && !commit_valid && !abort_valid);
            @(negedge clk_core);
        end
    endtask

    task automatic close_context;
        begin
            wait_scoreboard_empty();
            context_close_valid = 1'b1;
            do @(posedge clk_core); while (!context_close_accept);
            @(negedge clk_core);
            context_close_valid = 1'b0;
        end
    endtask

    task automatic run_real_scenario(
        input int weight_latency,
        input int acc_latency,
        input logic random_mode);
        int writes_before;
        int lanes_before;
        int weight_groups_before;
        int weight_responses_before;
        int hits_before;
        int acc_groups_before;
        int acc_responses_before;
        begin
            reset_dut();
            configured_weight_latency = weight_latency;
            configured_acc_latency = acc_latency;
            random_macro_mode = random_mode;
            auto_commit_stall = random_mode;
            open_context();
            writes_before = exact_write_checks;
            lanes_before = exact_lane_checks;
            weight_groups_before = weight_request_groups;
            weight_responses_before = weight_response_accepts;
            hits_before = cache_hits;
            acc_groups_before = acc_request_groups;
            acc_responses_before = acc_response_accepts;
            for (int index = 0; index < MAX_DESCRIPTORS; index++)
                drive_packed(descriptor_memory[index]);
            close_context();
            if (exact_write_checks - writes_before != 504
                    || exact_lane_checks - lanes_before != 4032
                    || weight_request_groups - weight_groups_before != 56
                    || weight_response_accepts - weight_responses_before != 14
                    || cache_hits - hits_before != 448
                    || acc_request_groups - acc_groups_before != 40
                    || acc_response_accepts - acc_responses_before == 0)
                $fatal(1,
                    "M241r2 scenario failed wl=%0d al=%0d random=%0d writes=%0d lanes=%0d wgroups=%0d wrsp=%0d hits=%0d agroups=%0d arsp=%0d",
                    weight_latency, acc_latency, random_mode,
                    exact_write_checks - writes_before,
                    exact_lane_checks - lanes_before,
                    weight_request_groups - weight_groups_before,
                    weight_response_accepts - weight_responses_before,
                    cache_hits - hits_before,
                    acc_request_groups - acc_groups_before,
                    acc_response_accepts - acc_responses_before);
            scenario_passes = scenario_passes + 1;
        end
    endtask

    // Construct deterministic downstream backpressure instead of relying on
    // random coincidence.  A stalled clean commit fills s2, a non-aliasing
    // accumulator response fills the s1 boundary, and a third miss holds a
    // tagged weight response at s0.  Releasing the commit permits both held
    // responses to retire on the immediately following edge.
    task automatic run_response_backpressure_attack;
        begin
            reset_dut();
            configured_weight_latency = 1;
            configured_acc_latency = 1;
            open_context();
            drive_descriptor(16'd0, 9'd70, 4'd0, 4'b0001,
                             3'd0, 3'd0, 3'd0, 3'd0, 4'b0, 1'b0);
            wait_scoreboard_empty();
            force_commit_stall = 1'b1;
            drive_descriptor(16'd1, 9'd71, 4'd1, 4'b0001,
                             3'd0, 3'd0, 3'd0, 3'd0, 4'b0, 1'b0);
            drive_descriptor(16'd2, 9'd70, 4'd2, 4'b0001,
                             3'd0, 3'd0, 3'd0, 3'd0, 4'b0, 1'b0);
            drive_descriptor(16'd3, 9'd72, 4'd3, 4'b0001,
                             3'd0, 3'd0, 3'd0, 3'd0, 4'b0, 1'b1);
            wait (weight_rsp_valid && !weight_rsp_ready
                  && acc_rsp_valid && !acc_rsp_ready);
            // Keep the two responses blocked across a sampled edge so SVA
            // observes the hold, then release before the next edge.
            @(posedge clk_core);
            @(negedge clk_core);
            force_commit_stall = 1'b0;
            wait_scoreboard_empty();
            close_context();
        end
    endtask

    // A separate same-address sequence makes the no-forwarding RAW fence
    // observable while preserving exact external accumulator semantics.
    task automatic run_alias_interlock_attack;
        begin
            reset_dut();
            configured_weight_latency = 1;
            configured_acc_latency = 1;
            open_context();
            drive_descriptor(16'd0, 9'd80, 4'd0, 4'b0001,
                             3'd0, 3'd0, 3'd0, 3'd0, 4'b0, 1'b0);
            wait_scoreboard_empty();
            force_commit_stall = 1'b1;
            drive_descriptor(16'd1, 9'd80, 4'd1, 4'b0001,
                             3'd0, 3'd0, 3'd0, 3'd0, 4'b0, 1'b0);
            drive_descriptor(16'd2, 9'd80, 4'd1, 4'b0001,
                             3'd0, 3'd0, 3'd0, 3'd0, 4'b0, 1'b1);
            wait (rmw_alias_stall);
            @(negedge clk_core);
            force_commit_stall = 1'b0;
            wait_scoreboard_empty();
            close_context();
        end
    endtask

    initial begin : test
        int writes_before;
        int commits_before;
        int best_abs;
        int best_bank;
        int best_address;
        int best_lane;
        int signed candidate;
        int signed contribution;
        int target_address;
        logic [2:0] target_destination;
        logic target_negate;

        clk_core = 1'b0;
        rst_core = 1'b1;
        loader_binding_valid = 1'b0;
        context_open_valid = 1'b0;
        descriptor_valid = 1'b0;
        context_close_valid = 1'b0;
        weight_rsp_valid = 1'b0;
        acc_rsp_valid = 1'b0;
        random_macro_mode = 1'b0;
        auto_commit_stall = 1'b0;
        force_commit_stall = 1'b0;
        force_abort_stall = 1'b0;
        corrupt_next_weight_response = 1'b0;
        corrupt_next_acc_response = 1'b0;
        configured_weight_latency = 1;
        configured_acc_latency = 1;
        cycle_count = 0;
        weight_request_accepts = 0;
        weight_request_groups = 0;
        weight_response_accepts = 0;
        weight_request_stalls = 0;
        weight_response_stalls = 0;
        acc_request_accepts = 0;
        acc_request_groups = 0;
        acc_response_accepts = 0;
        acc_request_stalls = 0;
        acc_response_stalls = 0;
        commit_accepts = 0;
        commit_stalls = 0;
        abort_accepts = 0;
        abort_stalls = 0;
        exact_write_checks = 0;
        exact_lane_checks = 0;
        exact_mismatches = 0;
        accepted_descriptors = 0;
        accepted_groups = 0;
        cache_hits = 0;
        expected_overflows = 0;
        stale_weight_response_attacks = 0;
        stale_acc_response_attacks = 0;
        loader_binding_attacks = 0;
        overflow_younger_attacks = 0;
        overflow_success_commits = 0;
        overflow_writes = 0;
        recovery_commits = 0;
        scenario_passes = 0;
        alias_stalls = 0;
        clear_drivers();
        for (int tuple = 0; tuple < 4; tuple++)
            descriptor_destination[tuple] = '0;
        for (int bank = 0; bank < 4; bank++) begin
            weight_rsp_bank_valid = '0;
            acc_rsp_bank_valid = '0;
            for (int lane = 0; lane < LANES; lane++) begin
                weight_rsp_data[bank][lane] = '0;
                acc_rsp_data[bank][lane] = '0;
            end
            for (int address = 0; address < 2 * ROWS; address++) begin
                for (int lane = 0; lane < LANES; lane++) begin
                    acc_memory[bank][address][lane]
                        = ((bank + 1) * 113 + address * 7 + lane * 3)
                          & ((1 << ACC_BITS) - 1);
                    reference_memory[bank][address][lane] = 0;
                end
            end
        end
        if (!$value$plusargs("VECTOR_DIR=%s", vector_dir))
            $fatal(1, "M241r2 VECTOR_DIR plusarg missing");
        $readmemh({vector_dir, "/descriptor.mem"}, descriptor_memory);
        $readmemh({vector_dir, "/weight.mem"}, weight_flat);
        $readmemh({vector_dir, "/meta.mem"}, meta_memory);
        if (meta_memory[0] != MAX_DESCRIPTORS
                || meta_memory[7] != LANES || meta_memory[8] != ROWS)
            $fatal(1, "M241r2 vector metadata drift");

        run_real_scenario(1, 1, 1'b0);
        run_real_scenario(2, 2, 1'b0);
        run_real_scenario(3, 3, 1'b0);
        run_real_scenario(1, 1, 1'b1);
        run_response_backpressure_attack();
        run_alias_interlock_attack();

        // Loader payload and cache epoch are bound before context admission.
        reset_dut();
        bind_loader();
        loader_binding_payload_id = loader_binding_payload_id ^ 32'h1;
        context_open_sequence = meta_memory[1];
        context_open_operator = meta_memory[2][1:0];
        context_open_partition = meta_memory[3][8:0];
        context_open_window = meta_memory[5][15:0];
        context_open_weight_epoch = meta_memory[4][15:0];
        context_open_payload_id = {meta_memory[4][15:0],
                                   7'd0, meta_memory[3][8:0]};
        context_open_valid = 1'b1;
        #0.1;
        if (!protocol_error || context_open_ready)
            $fatal(1, "M241r2 wrong loader binding escaped");
        @(posedge clk_core);
        @(negedge clk_core);
        context_open_valid = 1'b0;
        loader_binding_attacks = loader_binding_attacks + 1;

        // Correct request, stale/mis-epoch macro response: ready stays low and
        // no response or payload is consumed.
        reset_dut();
        configured_weight_latency = 2;
        configured_acc_latency = 2;
        open_context();
        corrupt_next_weight_response = 1'b1;
        drive_descriptor(16'd0, 9'd2, 4'd0, 4'b1111,
                         3'd0, 3'd1, 3'd2, 3'd3, 4'b0, 1'b0);
        wait (protocol_error && weight_rsp_valid);
        #0.1;
        if (weight_rsp_accept || |acc_wr_en)
            $fatal(1, "M241r2 stale weight response consumed");
        stale_weight_response_attacks = stale_weight_response_attacks + 1;
        @(posedge clk_core);

        // Correct weight response followed by stale accumulator payload tag.
        reset_dut();
        configured_weight_latency = 1;
        configured_acc_latency = 2;
        open_context();
        drive_descriptor(16'd0, 9'd3, 4'd0, 4'b0001,
                         3'd0, 3'd0, 3'd0, 3'd0, 4'b0, 1'b0);
        wait_scoreboard_empty();
        corrupt_next_acc_response = 1'b1;
        drive_descriptor(16'd1, 9'd3, 4'd1, 4'b0001,
                         3'd0, 3'd0, 3'd0, 3'd0, 4'b0, 1'b0);
        wait (protocol_error && acc_rsp_valid);
        #0.1;
        if (acc_rsp_accept || |acc_wr_en)
            $fatal(1, "M241r2 stale accumulator response consumed");
        stale_acc_response_attacks = stale_acc_response_attacks + 1;
        @(posedge clk_core);

        // Find a strong real checkpoint byte for the overflow/younger attack.
        best_abs = 0;
        best_bank = 0;
        best_address = 0;
        best_lane = 0;
        for (int bank = 0; bank < 4; bank++) begin
            for (int address = 0; address < 32; address++) begin
                for (int lane = 0; lane < LANES; lane++) begin
                    candidate = $signed(weight_flat[
                        (bank * 32 + address) * LANES + lane]);
                    if (candidate < 0)
                        candidate = -candidate;
                    if (candidate > best_abs) begin
                        best_abs = candidate;
                        best_bank = bank;
                        best_address = address;
                        best_lane = lane;
                    end
                end
            end
        end
        if (best_abs < 64)
            $fatal(1, "M241r2 overflow checkpoint byte too weak");
        target_destination = best_bank + 4 * (best_address >> 4);
        target_negate = $signed(weight_flat[
            (best_bank * 32 + best_address) * LANES + best_lane]) < 0;
        target_address = (target_destination[2] ? ROWS : 0) + 9'd31;

        reset_dut();
        configured_weight_latency = 1;
        configured_acc_latency = 3;
        open_context();
        drive_descriptor(16'd0, 9'd31, best_address[3:0], 4'b0001,
                         target_destination, 3'd0, 3'd0, 3'd0,
                         {3'b0, target_negate}, 1'b0);
        wait_scoreboard_empty();
        for (int lane = 0; lane < LANES; lane++) begin
            contribution = $signed(weight_flat[
                (best_bank * 32 + best_address) * LANES + lane]);
            if (target_negate)
                contribution = -contribution;
            if (contribution > 0)
                reference_memory[best_bank][target_address][lane]
                    = ACC_MAX - contribution + 1;
            else if (contribution < 0)
                reference_memory[best_bank][target_address][lane]
                    = ACC_MIN - contribution - 1;
            else
                reference_memory[best_bank][target_address][lane] = 0;
            acc_memory[best_bank][target_address][lane]
                = reference_memory[best_bank][target_address][lane];
        end
        writes_before = exact_write_checks;
        commits_before = commit_accepts;
        force_abort_stall = 1'b1;
        drive_descriptor(16'd1, 9'd31, best_address[3:0], 4'b0001,
                         target_destination, 3'd0, 3'd0, 3'd0,
                         {3'b0, target_negate}, 1'b0);
        drive_descriptor(16'd2, 9'd32, best_address[3:0], 4'b0001,
                         target_destination, 3'd0, 3'd0, 3'd0,
                         4'b0, 1'b0);
        drive_descriptor(16'd3, 9'd33, best_address[3:0], 4'b0001,
                         target_destination, 3'd0, 3'd0, 3'd0,
                         4'b0, 1'b0);
        wait (abort_valid);
        #0.1;
        if (commit_valid || commit_accept || |acc_wr_en
                || abort_discarded_tokens != 2)
            $fatal(1,
                "M241r2 overflow success/quarantine failure cv=%0d ca=%0d wr=%b discarded=%0d",
                commit_valid, commit_accept, acc_wr_en,
                abort_discarded_tokens);
        repeat (2) @(posedge clk_core);
        force_abort_stall = 1'b0;
        wait (abort_accept);
        @(posedge clk_core);
        #0.1;
        if (!context_abort || exact_write_checks != writes_before
                || commit_accepts != commits_before
                || overflow_success_commits != 0 || overflow_writes != 0)
            $fatal(1,
                "M241r2 overflow abort accounting failed abort=%0d writes=%0d commits=%0d success=%0d overflow_writes=%0d",
                context_abort, exact_write_checks - writes_before,
                commit_accepts - commits_before,
                overflow_success_commits, overflow_writes);
        overflow_younger_attacks = overflow_younger_attacks + 1;
        @(posedge clk_core);

        // Reset recovers queues and identities after the explicit abort.
        reset_dut();
        configured_weight_latency = 2;
        configured_acc_latency = 2;
        open_context();
        commits_before = commit_accepts;
        drive_descriptor(16'd0, 9'd40, 4'd2, 4'b0001,
                         3'd0, 3'd0, 3'd0, 3'd0, 4'b0, 1'b1);
        close_context();
        if (commit_accepts != commits_before + 1)
            $fatal(1, "M241r2 post-abort recovery failed");
        recovery_commits = recovery_commits + 1;

        if (scenario_passes != 4 || exact_mismatches != 0
                || stale_weight_response_attacks != 1
                || stale_acc_response_attacks != 1
                || loader_binding_attacks != 1
                || overflow_younger_attacks != 1
                || overflow_success_commits != 0 || overflow_writes != 0
                || recovery_commits != 1
                || weight_request_stalls == 0
                || acc_request_stalls == 0
                || weight_response_stalls == 0
                || acc_response_stalls == 0
                || commit_stalls == 0 || abort_stalls == 0)
            $fatal(1,
                "M241r2 final coverage mismatch scenarios=%0d mismatches=%0d stale_w=%0d stale_a=%0d loader=%0d overflow=%0d success=%0d ow=%0d recovery=%0d wrqs=%0d arqs=%0d wrss=%0d arss=%0d cs=%0d as=%0d",
                scenario_passes, exact_mismatches,
                stale_weight_response_attacks,
                stale_acc_response_attacks, loader_binding_attacks,
                overflow_younger_attacks, overflow_success_commits,
                overflow_writes, recovery_commits,
                weight_request_stalls, acc_request_stalls,
                weight_response_stalls, acc_response_stalls,
                commit_stalls, abort_stalls);
        $display("PASS M241r2 scenarios=4 latency_modes=1_2_3_random real_descriptors_each=126 real_writes_each=504 real_lane_checks_each=4032 total_mismatches=0 weight_request_stalls=%0d weight_response_stalls=%0d acc_request_stalls=%0d acc_response_stalls=%0d commit_stalls=%0d stale_weight_responses=1 stale_acc_responses=1 stale_response_accepts=0 loader_binding_attacks=1 overflow_aborts=1 overflow_success_commits=0 overflow_writes=0 accepted_younger_discarded=2 abort_stalls=%0d recovery_commits=1 window_identity=true payload_epoch_binding=true lazy_valid=true overflow_guard=true forwarding_payload_bits=0 m149_instantiated=false real_full_trace=false m238_target_speedup=1.687018 physical_speedup=false system_speedup=false headline=false",
                 weight_request_stalls, weight_response_stalls,
                 acc_request_stalls, acc_response_stalls, commit_stalls,
                 abort_stalls);
        $finish;
    end

    initial begin
        #4000000;
        $fatal(1, "M241r2 watchdog timeout");
    end
endmodule

`default_nettype wire
