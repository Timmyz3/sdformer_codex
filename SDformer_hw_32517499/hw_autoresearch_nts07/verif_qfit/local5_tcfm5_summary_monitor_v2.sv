`timescale 1ns/1ps
`default_nettype none

// Passive lower-level observer for public terms and physical TCFM5 bank updates.
module local5_tcfm5_summary_monitor_v2 #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int HEAD_DIM = 32,
    parameter int LANE_W = 5,
    parameter int Y_W = 4,
    parameter int X_W = 4,
    parameter int PLANE_W = 1,
    parameter int BANK_ADDR_W = 7
) (
    input logic clk_core,
    input logic rst_core,
    input logic term_commit,
    input logic [PLANE_W-1:0] term_source_plane,
    input logic [Y_W-1:0] term_source_y,
    input logic [X_W-1:0] term_source_x,
    input logic [LANE_W-1:0] term_lane,
    input logic [4:0] term_destination_mask,
    input logic [4:0] actual_bank_mask,
    input logic [5*BANK_ADDR_W-1:0] actual_bank_addr_flat
);
    localparam int RESOURCE_CODE = 6;
    localparam logic [63:0] DIGEST_SEED0 = 64'hcbf29ce484222325;
    localparam logic [63:0] DIGEST_SEED1 = 64'h00001505d3c4b2a1;
    localparam logic [63:0] DIGEST_PRIME = 64'h00000100000001b3;

    integer summary_fd;
    longint unsigned cycle_q;
    longint unsigned count_q;
    longint unsigned update_count_q;
    longint unsigned mismatch_count_q;
    logic [63:0] digest0_q;
    logic [63:0] digest1_q;
    longint unsigned first_anchor_q [0:11];
    longint unsigned last_anchor_q [0:11];
    bit first_valid_q;
    string output_prefix;
    string output_path;
    string observer_instance_path;
    string target_instance_path;

    function automatic logic [4:0] topology_mask(
        input integer y,
        input integer x,
        input logic [4:0] destination_mask
    );
        integer role_y;
        integer role_x;
        integer bank;
        begin
            topology_mask = '0;
            for (integer role = 0; role < 5; role = role + 1) begin
                role_y = y;
                role_x = x;
                case (role)
                    1: role_y = y + 1;
                    2: role_y = y - 1;
                    3: role_x = x + 1;
                    4: role_x = x - 1;
                    default: begin end
                endcase
                if (destination_mask[role]
                    && role_y >= 0 && role_y < HEIGHT
                    && role_x >= 0 && role_x < WIDTH) begin
                    bank = (role_x + 2*role_y) % 5;
                    topology_mask[bank] = 1'b1;
                end
            end
        end
    endfunction

    task automatic hash_byte(input logic [7:0] value);
        begin
            digest0_q = (digest0_q ^ {56'd0, value}) * DIGEST_PRIME;
            digest1_q = ((digest1_q << 5) + digest1_q) ^ {56'd0, value};
        end
    endtask

    task automatic hash_u16(input logic [15:0] value);
        begin
            hash_byte(value[7:0]);
            hash_byte(value[15:8]);
        end
    endtask

    task automatic hash_u64(input longint unsigned value);
        begin
            for (integer byte_index = 0; byte_index < 8; byte_index++)
                hash_byte(value[byte_index*8 +: 8]);
        end
    endtask

    task automatic hash_string(input string value);
        begin
            for (integer index = 0; index < value.len(); index++)
                hash_byte(value.getc(index));
        end
    endtask

    task automatic record_term;
        longint unsigned values [0:11];
        string domain_tag;
        logic [4:0] expected_mask;
        integer source_id;
        begin
            expected_mask = topology_mask(32'(term_source_y),
                32'(term_source_x), term_destination_mask);
            source_id = 32'(term_source_plane)*225
                      + 32'(term_source_y)*15 + 32'(term_source_x);
            values[0] = count_q;
            values[1] = cycle_q;
            values[2] = 64'(source_id);
            values[3] = 64'(term_lane);
            values[4] = 64'(expected_mask);
            values[5] = 64'(actual_bank_mask);
            for (integer bank = 0; bank < 5; bank++)
                values[6+bank] = 64'(actual_bank_addr_flat[
                    bank*BANK_ADDR_W +: BANK_ADDR_W]);
            values[11] = 0;

            domain_tag = "LOCAL5_PHASE_SUMMARY_V2";
            hash_u16(16'(domain_tag.len()));
            hash_string(domain_tag);
            hash_u16(16'd2);
            hash_u16(16'(RESOURCE_CODE));
            hash_u16(16'(target_instance_path.len()));
            hash_string(target_instance_path);
            hash_u64(values[0]);
            hash_u64(values[1]);
            hash_u16(16'd80);
            for (integer index = 2; index < 12; index++)
                hash_u64(values[index]);

            for (integer index = 0; index < 12; index++) begin
                last_anchor_q[index] = values[index];
                if (!first_valid_q)
                    first_anchor_q[index] = values[index];
            end
            first_valid_q = 1'b1;
            count_q = count_q + 1;
            for (integer bank = 0; bank < 5; bank++)
                if (actual_bank_mask[bank])
                    update_count_q = update_count_q + 1;
            if (expected_mask != actual_bank_mask)
                mismatch_count_q = mismatch_count_q + 1;
        end
    endtask

    task automatic write_anchor(input string kind, input bit first);
        begin
            $fwrite(summary_fd,
                "A,TCFM5_TERM_COMMIT,%s,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                kind,
                first ? first_anchor_q[0] : last_anchor_q[0],
                first ? first_anchor_q[1] : last_anchor_q[1],
                first ? first_anchor_q[2] : last_anchor_q[2],
                first ? first_anchor_q[3] : last_anchor_q[3],
                first ? first_anchor_q[4] : last_anchor_q[4],
                first ? first_anchor_q[5] : last_anchor_q[5],
                first ? first_anchor_q[6] : last_anchor_q[6],
                first ? first_anchor_q[7] : last_anchor_q[7],
                first ? first_anchor_q[8] : last_anchor_q[8],
                first ? first_anchor_q[9] : last_anchor_q[9],
                first ? first_anchor_q[10] : last_anchor_q[10],
                first ? first_anchor_q[11] : last_anchor_q[11]);
        end
    endtask

    initial begin
        observer_instance_path = $sformatf("%m");
        if (HEIGHT != 15 || WIDTH != 15 || TIME_PLANES != 2
            || HEAD_DIM != 32 || BANK_ADDR_W != 7)
            $fatal(1, "TCFM5 observer bound to a non-target configuration");
        if (!$value$plusargs("TCFM5_SUMMARY_PREFIX_V2=%s", output_prefix)
            || !$value$plusargs(
                "TCFM5_TARGET_INSTANCE=%s", target_instance_path))
            $fatal(1, "TCFM5 summary v2 plusargs are mandatory");
        output_path = $sformatf("%s.%s.csv", output_prefix,
            observer_instance_path);
        summary_fd = $fopen(output_path, "w");
        if (summary_fd == 0)
            $fatal(1, "cannot open TCFM5 summary v2 output");
        cycle_q = 0;
        count_q = 0;
        update_count_q = 0;
        mismatch_count_q = 0;
        digest0_q = DIGEST_SEED0;
        digest1_q = DIGEST_SEED1;
        first_valid_q = 1'b0;
        $fwrite(summary_fd, "SCHEMA,local5_tcfm5_summary_v2\n");
        $fwrite(summary_fd, "ORIGIN,RTL_LOWER_BANKS\n");
        $fwrite(summary_fd, "OBSERVER_INSTANCE,%s\n",
            observer_instance_path);
        $fwrite(summary_fd, "TARGET_INSTANCE,%s\n", target_instance_path);
        $fwrite(summary_fd,
            "DIGEST,FNV1A64_AND_DJB2XOR64_V1,%016h,%016h\n",
            DIGEST_SEED0, DIGEST_SEED1);
        $fwrite(summary_fd, "RESOURCE_CODE,TCFM5_TERM_COMMIT,%0d\n",
            RESOURCE_CODE);
        $fwrite(summary_fd, "PAYLOAD_U64_COUNT,10\n");
    end

    always @(posedge clk_core) begin
        if (rst_core) begin
            cycle_q = 0;
            count_q = 0;
            update_count_q = 0;
            mismatch_count_q = 0;
            digest0_q = DIGEST_SEED0;
            digest1_q = DIGEST_SEED1;
            first_valid_q = 1'b0;
        end else begin
            if (term_commit)
                record_term();
            cycle_q = cycle_q + 1;
        end
    end

    final begin
        if (summary_fd != 0) begin
            if (first_valid_q) begin
                write_anchor("FIRST", 1'b1);
                write_anchor("LAST", 1'b0);
            end
            $fwrite(summary_fd, "S,TCFM5_TERM_COMMIT,%0d,%016h,%016h\n",
                count_q, digest0_q, digest1_q);
            $fwrite(summary_fd, "L,TCFM5_TERM_LEDGER,%0d,%0d,%0d\n",
                count_q, update_count_q, mismatch_count_q);
            $fwrite(summary_fd, "END,%0d,RTL_LOWER_BANKS\n", cycle_q);
            $fclose(summary_fd);
        end
    end
endmodule

`default_nettype wire
