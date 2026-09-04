`timescale 1ns/1ps
`default_nettype none

module tb_m408_h67_q32_static_codec_full_vcs;
    localparam int TAG_BITS = 24;
    localparam int BLOCKS = 442368;
    localparam int EXPECTED_NARROW = 112167;
    localparam int EXPECTED_CONTRIBUTIONS = 772569;

    logic clk_core, reset_n, config_reload;
    logic low_valid, low_ready, low_accept;
    logic [TAG_BITS-1:0] low_tag;
    logic low_tile;
    logic [4:0] low_center_id;
    logic [2:0] low_output_block;
    logic low_narrow;
    logic [767:0] low_data;
    logic high_valid, high_ready, high_accept;
    logic [TAG_BITS-1:0] high_tag;
    logic high_tile;
    logic [4:0] high_center_id;
    logic [2:0] high_output_block;
    logic [511:0] high_data;
    logic contribution_valid, contribution_ready, contribution_accept;
    logic [TAG_BITS-1:0] contribution_tag;
    logic contribution_tile;
    logic [4:0] contribution_center_id;
    logic [2:0] contribution_output_block;
    logic contribution_narrow, contribution_part_high, contribution_last;
    logic [1151:0] contribution_data;
    logic protocol_error, busy;
    logic [1:0] debug_completed_fifo_count;
    logic [31:0] debug_low_accepts, debug_high_accepts;
    logic [31:0] debug_narrow_blocks, debug_wide_blocks;
    logic [31:0] debug_contributions;

    logic [1280:0] records [0:BLOCKS-1];
    string stimulus_path;
    integer expected_block, expected_part;
    integer checked_contributions, checked_lanes;
    integer narrow_blocks, wide_blocks;
    integer semantic_narrow_mismatches, padding_mismatches;
    integer metadata_mismatches, arithmetic_mismatches;

    m405_exact_elastic_pwp_issue_adapter #(.TAG_BITS(TAG_BITS)) dut (.*);
    m405_exact_elastic_pwp_issue_adapter_assertions #(.TAG_BITS(TAG_BITS))
        m408_sva (.*);

    always #1.5 clk_core = ~clk_core;

    function automatic logic [1151:0] expected_vector(
        input integer block_index,
        input integer part_index
    );
        logic [1151:0] value;
        logic [7:0] low_lane;
        logic [3:0] high_lane;
        begin
            value = '0;
            for (int lane = 0; lane < 96; lane++) begin
                low_lane = records[block_index][lane*8 +: 8];
                high_lane = records[block_index][768+lane*4 +: 4];
                if (records[block_index][1280])
                    value[lane*12 +: 12] = {{4{low_lane[7]}},low_lane};
                else if (part_index == 0)
                    value[lane*12 +: 12] = {4'b0000,low_lane};
                else
                    value[lane*12 +: 12] = {high_lane,8'b0};
            end
            expected_vector = value;
        end
    endfunction

    task automatic drive_low_block(input integer block_index);
        integer output_block, center_id;
        begin
            output_block = block_index % 8;
            center_id = (block_index / 8) % 32;
            if (clk_core !== 1'b0) @(negedge clk_core);
            low_valid = 1;
            low_tag = block_index[TAG_BITS-1:0];
            low_tile = output_block >= 4;
            low_center_id = center_id[4:0];
            low_output_block = output_block[2:0];
            low_narrow = records[block_index][1280];
            low_data = records[block_index][767:0];
            do @(posedge clk_core); while (!low_accept && !protocol_error);
            @(negedge clk_core); low_valid = 0;
        end
    endtask

    task automatic drive_high_block(input integer block_index);
        integer output_block, center_id;
        begin
            output_block = block_index % 8;
            center_id = (block_index / 8) % 32;
            if (clk_core !== 1'b0) @(negedge clk_core);
            high_valid = 1;
            high_tag = block_index[TAG_BITS-1:0];
            high_tile = output_block >= 4;
            high_center_id = center_id[4:0];
            high_output_block = output_block[2:0];
            high_data = records[block_index][1279:768];
            do @(posedge clk_core); while (!high_accept && !protocol_error);
            @(negedge clk_core); high_valid = 0;
        end
    endtask

    always @(posedge clk_core) begin
        integer expected_output_block, expected_center_id;
        logic expected_narrow, expected_high, expected_last, expected_tile;
        logic [1151:0] vector;
        if (contribution_accept) begin
            if (expected_block >= BLOCKS)
                $fatal(1,"M408 contribution overrun");
            expected_output_block = expected_block % 8;
            expected_center_id = (expected_block / 8) % 32;
            expected_narrow = records[expected_block][1280];
            expected_high = !expected_narrow && expected_part == 1;
            expected_last = expected_narrow || expected_high;
            expected_tile = expected_output_block >= 4;
            vector = expected_vector(expected_block,expected_part);
            if (contribution_tag !== expected_block[TAG_BITS-1:0]
                    || contribution_tile !== expected_tile
                    || contribution_center_id !== expected_center_id[4:0]
                    || contribution_output_block !==
                       expected_output_block[2:0]
                    || contribution_narrow !== expected_narrow
                    || contribution_part_high !== expected_high
                    || contribution_last !== expected_last) begin
                metadata_mismatches++;
                $fatal(1,"M408 metadata/order mismatch block=%0d part=%0d tag=%0d center=%0d output=%0d",
                       expected_block,expected_part,contribution_tag,
                       contribution_center_id,contribution_output_block);
            end
            if (contribution_data !== vector) begin
                arithmetic_mismatches++;
                $fatal(1,"M408 arithmetic mismatch block=%0d part=%0d xor=%h",
                       expected_block,expected_part,
                       contribution_data ^ vector);
            end
            checked_contributions++;
            checked_lanes += 96;
            if (!expected_narrow && expected_part == 0)
                expected_part = 1;
            else begin
                expected_part = 0;
                expected_block++;
            end
        end
    end

    initial begin
        logic [7:0] low_lane;
        logic [3:0] high_lane;
        clk_core = 0;
        reset_n = 0;
        config_reload = 0;
        low_valid = 0;
        low_tag = 0;
        low_tile = 0;
        low_center_id = 0;
        low_output_block = 0;
        low_narrow = 0;
        low_data = 0;
        high_valid = 0;
        high_tag = 0;
        high_tile = 0;
        high_center_id = 0;
        high_output_block = 0;
        high_data = 0;
        contribution_ready = 1;
        expected_block = 0;
        expected_part = 0;
        checked_contributions = 0;
        checked_lanes = 0;
        narrow_blocks = 0;
        wide_blocks = 0;
        semantic_narrow_mismatches = 0;
        padding_mismatches = 0;
        metadata_mismatches = 0;
        arithmetic_mismatches = 0;
        if (!$value$plusargs("M408_STIMULUS=%s",stimulus_path))
            $fatal(1,"M408 missing +M408_STIMULUS");
        $readmemh(stimulus_path,records);
        if (^records[0] === 1'bx || ^records[BLOCKS-1] === 1'bx)
            $fatal(1,"M408 stimulus extent/read failure");
        for (int block_index = 0; block_index < BLOCKS; block_index++) begin
            if (records[block_index][1279:1152] != 0)
                padding_mismatches++;
            if (records[block_index][1280]) begin
                narrow_blocks++;
                for (int lane = 0; lane < 96; lane++) begin
                    low_lane = records[block_index][lane*8 +: 8];
                    high_lane = records[block_index][768+lane*4 +: 4];
                    if (high_lane !== {4{low_lane[7]}})
                        semantic_narrow_mismatches++;
                end
            end else begin
                wide_blocks++;
            end
        end
        if (narrow_blocks != EXPECTED_NARROW
                || semantic_narrow_mismatches != 0
                || padding_mismatches != 0)
            $fatal(1,"M408 static semantic gate narrow=%0d semantic=%0d padding=%0d",
                   narrow_blocks,semantic_narrow_mismatches,
                   padding_mismatches);

        repeat (5) @(posedge clk_core);
        @(negedge clk_core); reset_n = 1;
        for (int block_index = 0; block_index < BLOCKS; block_index++) begin
            drive_low_block(block_index);
            if (!records[block_index][1280])
                drive_high_block(block_index);
        end
        wait (expected_block == BLOCKS && !busy);
        if (protocol_error || checked_contributions != EXPECTED_CONTRIBUTIONS
                || checked_lanes != EXPECTED_CONTRIBUTIONS*96
                || debug_low_accepts != BLOCKS
                || debug_high_accepts != wide_blocks
                || debug_narrow_blocks != narrow_blocks
                || debug_wide_blocks != wide_blocks
                || debug_contributions != EXPECTED_CONTRIBUTIONS
                || metadata_mismatches != 0 || arithmetic_mismatches != 0)
            $fatal(1,"M408 final ledger failure blocks=%0d/%0d high=%0d/%0d contributions=%0d/%0d lanes=%0d protocol=%0d metadata=%0d arithmetic=%0d",
                   debug_low_accepts,BLOCKS,debug_high_accepts,wide_blocks,
                   checked_contributions,EXPECTED_CONTRIBUTIONS,
                   checked_lanes,protocol_error,metadata_mismatches,
                   arithmetic_mismatches);
        $display("PASS M408 full static codec blocks=442368 lanes=42467328 narrow=112167 wide=330201 contributions=772569 metadata_mismatches=0 arithmetic_mismatches=0 semantic_narrow_mismatches=0 padding_mismatches=0 exact_low8_high4=true single_shared96=true system_speedup=false headline=false");
        $finish;
    end

    initial begin
        #20000000;
        $fatal(1,"M408 full static codec watchdog block=%0d contributions=%0d busy=%0d error=%0d",
               expected_block,checked_contributions,busy,protocol_error);
    end
endmodule

`default_nettype wire
