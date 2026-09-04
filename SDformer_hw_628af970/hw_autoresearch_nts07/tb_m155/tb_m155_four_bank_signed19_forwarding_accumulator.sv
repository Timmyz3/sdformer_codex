`timescale 1ns/1ps
`default_nettype none

module tb_m155_four_bank_signed19_forwarding_accumulator;
    localparam int LANES = 96;
    localparam int ROWS = 384;
    localparam int ACC_BITS = 19;
    localparam int VECTOR_BITS = LANES * ACC_BITS;

    logic clk_core;
    logic rst_core;
    logic window_start_valid;
    logic window_start_ready;
    logic window_start_accept;
    logic update_valid;
    logic update_ready;
    logic [8:0] update_row;
    logic [3:0] update_group_valid;
    logic [2:0] update_destination [0:3];
    logic [3:0] update_negate;
    logic signed [10:0] update_vector [0:3][0:LANES-1];
    logic update_accept;
    logic window_end_valid;
    logic window_end_ready;
    logic window_end_accept;
    logic window_done;
    logic [3:0] acc_rd_en;
    logic [9:0] acc_rd_addr [0:3];
    logic signed [ACC_BITS-1:0] acc_rd_data [0:3][0:LANES-1];
    logic [3:0] acc_wr_en;
    logic [9:0] acc_wr_addr [0:3];
    logic signed [ACC_BITS-1:0] acc_wr_data [0:3][0:LANES-1];
    logic [3:0] same_address_forward;
    logic window_active;
    logic protocol_error;
    logic overflow_error;
    logic busy;

    logic signed [ACC_BITS-1:0] acc_memory [0:3][0:767][0:LANES-1];
    integer signed reference_memory [0:3][0:767][0:LANES-1];
    bit reference_touched [0:3][0:767];

    typedef struct packed {
        logic [9:0] address;
        logic [VECTOR_BITS-1:0] data;
    } expected_write_t;
    expected_write_t expected_write_q [0:3][$];

    int descriptor_count;
    int accepted_group_count;
    int full4_count;
    int tail_count;
    int write_group_count;
    int write_lane_checks;
    int forward_count;
    int macro_read_count;
    int protocol_attack_count;
    int overflow_attack_count;
    int overflow_descriptor_count;
    int window_count;

    m155_four_bank_signed19_forwarding_accumulator dut (.*);
    m155_four_bank_signed19_forwarding_accumulator_assertions checks (.*);

    always #1.5 clk_core = ~clk_core;

    always @(posedge clk_core) begin : behavioral_accumulator_sram
        if (rst_core) begin
            for (int bank = 0; bank < 4; bank++)
                for (int lane = 0; lane < LANES; lane++)
                    acc_rd_data[bank][lane] <= '0;
        end else begin
            for (int bank = 0; bank < 4; bank++) begin
                if (acc_rd_en[bank]) begin
                    for (int lane = 0; lane < LANES; lane++)
                        acc_rd_data[bank][lane]
                            <= acc_memory[bank][acc_rd_addr[bank]][lane];
                end
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
        int signed contribution;
        int signed new_value;
        int address;
        int bank;
        bit descriptor_will_overflow;
        if (!rst_core) begin
            for (int bank_index = 0; bank_index < 4; bank_index++) begin
                if (acc_wr_en[bank_index]) begin
                    if (expected_write_q[bank_index].size() == 0)
                        $fatal(1, "M155 unexpected bank write");
                    expected = expected_write_q[bank_index].pop_front();
                    if (acc_wr_addr[bank_index] !== expected.address)
                        $fatal(1, "M155 write address mismatch");
                    for (int lane = 0; lane < LANES; lane++) begin
                        if (acc_wr_data[bank_index][lane]
                                !== $signed(expected.data[
                                    lane * ACC_BITS +: ACC_BITS]))
                            $fatal(1,
                                "M155 write data mismatch bank=%0d lane=%0d",
                                bank_index, lane);
                    end
                    write_group_count = write_group_count + 1;
                    write_lane_checks = write_lane_checks + LANES;
                end
                if (acc_rd_en[bank_index])
                    macro_read_count = macro_read_count + 1;
            end
            if (update_accept) begin
                descriptor_count = descriptor_count + 1;
                accepted_group_count = accepted_group_count
                    + $countones(update_group_valid);
                if (update_group_valid == 4'b1111)
                    full4_count = full4_count + 1;
                else
                    tail_count = tail_count + 1;
                forward_count = forward_count
                    + $countones(same_address_forward);
                descriptor_will_overflow = 1'b0;
                for (int tuple = 0; tuple < 4; tuple++) begin
                    if (update_group_valid[tuple]) begin
                        bank = update_destination[tuple][1:0];
                        address = {update_destination[tuple][2], update_row};
                        produced.address = address;
                        produced.data = '0;
                        for (int lane = 0; lane < LANES; lane++) begin
                            contribution = $signed(update_vector[tuple][lane]);
                            if (update_negate[tuple])
                                contribution = -contribution;
                            new_value = reference_memory[bank][address][lane]
                                      + contribution;
                            if (new_value < -(1 << (ACC_BITS - 1))
                                    || new_value
                                       > (1 << (ACC_BITS - 1)) - 1)
                                descriptor_will_overflow = 1'b1;
                            produced.data[lane * ACC_BITS +: ACC_BITS]
                                = new_value[ACC_BITS-1:0];
                        end
                        if (!descriptor_will_overflow) begin
                            for (int lane = 0; lane < LANES; lane++) begin
                                contribution
                                    = $signed(update_vector[tuple][lane]);
                                if (update_negate[tuple])
                                    contribution = -contribution;
                                reference_memory[bank][address][lane]
                                    = reference_memory[bank][address][lane]
                                      + contribution;
                            end
                            reference_touched[bank][address] = 1'b1;
                            expected_write_q[bank].push_back(produced);
                        end
                    end
                end
                if (descriptor_will_overflow)
                    overflow_descriptor_count
                        = overflow_descriptor_count + 1;
            end
            if (window_start_accept) begin
                window_count = window_count + 1;
                for (int bank_index = 0; bank_index < 4; bank_index++) begin
                    if (expected_write_q[bank_index].size() != 0)
                        $fatal(1, "M155 stale expected write at window start");
                    for (int address_index = 0; address_index < 768;
                            address_index++) begin
                        reference_touched[bank_index][address_index] = 1'b0;
                        for (int lane = 0; lane < LANES; lane++)
                            reference_memory[bank_index][address_index][lane]
                                = 0;
                    end
                end
            end
        end
    end

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            rst_core = 1'b1;
            window_start_valid = 1'b0;
            update_valid = 1'b0;
            window_end_valid = 1'b0;
            repeat (3) @(negedge clk_core);
            rst_core = 1'b0;
        end
    endtask

    task automatic start_window;
        begin
            @(negedge clk_core);
            window_start_valid = 1'b1;
            do @(posedge clk_core); while (!window_start_accept);
            @(negedge clk_core);
            window_start_valid = 1'b0;
        end
    endtask

    task automatic send_update(
        input logic [8:0] row_id,
        input logic [3:0] valid_mask,
        input int destination_base,
        input int seed,
        input logic overflow_pattern);
        int signed value;
        begin
            update_row = row_id;
            update_group_valid = valid_mask;
            for (int tuple = 0; tuple < 4; tuple++) begin
                update_destination[tuple] = destination_base + tuple;
                update_negate[tuple] = overflow_pattern ? 1'b1
                    : ((seed + tuple) % 5 == 0);
                for (int lane = 0; lane < LANES; lane++) begin
                    if (overflow_pattern)
                        value = -1024;
                    else begin
                        value = (seed * 17 + tuple * 31 + lane * 7) % 255;
                        value = value - 127;
                    end
                    update_vector[tuple][lane] = value;
                end
            end
            update_valid = 1'b1;
            do @(posedge clk_core); while (!update_accept);
            @(negedge clk_core);
            update_valid = 1'b0;
        end
    endtask

    task automatic end_window;
        begin
            @(negedge clk_core);
            window_end_valid = 1'b1;
            do @(posedge clk_core); while (!window_end_accept);
            @(negedge clk_core);
            window_end_valid = 1'b0;
            for (int bank = 0; bank < 4; bank++) begin
                if (expected_write_q[bank].size() != 0)
                    $fatal(1, "M155 expected write remained after end");
            end
        end
    endtask

    task automatic inject_bank_conflict_while_pending;
        int writes_before;
        begin
            // Called on the negedge immediately after a legal descriptor;
            // that older descriptor is in the write pipeline.
            writes_before = write_group_count;
            update_row = 9'd23;
            update_group_valid = 4'b0011;
            update_destination[0] = 3'd0;
            update_destination[1] = 3'd4;
            update_destination[2] = 3'd2;
            update_destination[3] = 3'd3;
            update_negate = '0;
            for (int tuple = 0; tuple < 4; tuple++)
                for (int lane = 0; lane < LANES; lane++)
                    update_vector[tuple][lane] = 11'sd1;
            update_valid = 1'b1;
            #0.1;
            if (!protocol_error || update_ready)
                $fatal(1, "M155 bank conflict did not fail closed");
            @(posedge clk_core);
            @(negedge clk_core);
            update_valid = 1'b0;
            if (write_group_count != writes_before + 4)
                $fatal(1, "M155 younger fault erased older four writes");
            protocol_attack_count = protocol_attack_count + 1;
        end
    endtask

    initial begin : test
        clk_core = 1'b0;
        rst_core = 1'b1;
        window_start_valid = 1'b0;
        update_valid = 1'b0;
        update_row = '0;
        update_group_valid = '0;
        update_negate = '0;
        window_end_valid = 1'b0;
        descriptor_count = 0;
        accepted_group_count = 0;
        full4_count = 0;
        tail_count = 0;
        write_group_count = 0;
        write_lane_checks = 0;
        forward_count = 0;
        macro_read_count = 0;
        protocol_attack_count = 0;
        overflow_attack_count = 0;
        overflow_descriptor_count = 0;
        window_count = 0;
        for (int tuple = 0; tuple < 4; tuple++) begin
            update_destination[tuple] = '0;
            for (int lane = 0; lane < LANES; lane++)
                update_vector[tuple][lane] = '0;
        end
        for (int bank = 0; bank < 4; bank++) begin
            expected_write_q[bank].delete();
            for (int address = 0; address < 768; address++) begin
                reference_touched[bank][address] = 1'b0;
                for (int lane = 0; lane < LANES; lane++) begin
                    acc_memory[bank][address][lane] = '0;
                    reference_memory[bank][address][lane] = 0;
                end
            end
        end

        reset_dut();
        start_window();
        for (int source = 0; source < 16; source++)
            send_update(9'd5, 4'b1111, 0, source, 1'b0);
        for (int source = 0; source < 16; source++)
            send_update(9'd5, 4'b1111, 4, source + 20, 1'b0);
        send_update(9'd5, 4'b1111, 0, 50, 1'b0);
        send_update(9'd17, 4'b0001, 0, 61, 1'b0);
        send_update(9'd18, 4'b0011, 0, 62, 1'b0);
        send_update(9'd19, 4'b0111, 0, 63, 1'b0);
        end_window();

        // Lazy valid reset must overwrite stale macro contents without a read.
        start_window();
        send_update(9'd5, 4'b1111, 0, 70, 1'b0);
        send_update(9'd5, 4'b1111, 0, 71, 1'b0);
        end_window();

        // Fault atomicity: older accepted full4 write survives a malformed
        // same-bank younger request.
        reset_dut();
        start_window();
        send_update(9'd23, 4'b1111, 0, 80, 1'b0);
        inject_bank_conflict_while_pending();

        // Exact positive boundary attack: -(-1024) is +1024.  The 256th
        // accepted update would produce +262144 and must suppress the write.
        reset_dut();
        start_window();
        for (int item = 0; item < 256; item++)
            send_update(9'd31, 4'b0001, 0, item, 1'b1);
        #0.1;
        if (!overflow_error || acc_wr_en != 4'b0000)
            $fatal(1, "M155 overflow was not atomically suppressed");
        @(posedge clk_core);
        overflow_attack_count = overflow_attack_count + 1;
        repeat (2) @(posedge clk_core);

        if (descriptor_count != 295 || accepted_group_count != 406
                || full4_count != 36 || tail_count != 259
                || write_group_count != 405
                || write_lane_checks != 38880
                || forward_count != 379 || macro_read_count != 4
                || protocol_attack_count != 1
                || overflow_attack_count != 1
                || overflow_descriptor_count != 1
                || window_count != 4)
            $fatal(1,
                "M155 count mismatch d=%0d g=%0d f=%0d t=%0d wr=%0d lane=%0d fwd=%0d rd=%0d pa=%0d oa=%0d od=%0d win=%0d",
                descriptor_count, accepted_group_count, full4_count,
                tail_count, write_group_count, write_lane_checks,
                forward_count, macro_read_count, protocol_attack_count,
                overflow_attack_count, overflow_descriptor_count,
                window_count);
        $display("PASS M155 four-bank signed19 forwarding accumulator VCS windows=%0d descriptors=%0d accepted_groups=%0d writes=%0d write_lane_checks=%0d full4=%0d tails=%0d same_address_forwards=%0d macro_reads=%0d protocol_attacks=%0d overflow_attacks=%0d banks=4 lanes=96 input_bits=11 negate_bits=12 accumulator_bits=19 address_depth_per_bank=768 lazy_valid=true same_address_rdw_mode_independent=true external_sram_macro=true commit_scan=false physical_speedup=false system_speedup=false headline=false",
                 window_count, descriptor_count, accepted_group_count,
                 write_group_count, write_lane_checks, full4_count,
                 tail_count, forward_count, macro_read_count,
                 protocol_attack_count, overflow_attack_count);
        $finish;
    end

    initial begin
        #500000;
        $fatal(1, "M155 watchdog timeout");
    end
endmodule

`default_nettype wire
