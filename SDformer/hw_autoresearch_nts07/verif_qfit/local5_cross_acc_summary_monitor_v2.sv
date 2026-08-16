`timescale 1ns/1ps
`default_nettype none

// Passive lower-level observer for the physical cross-head 1RW memory port.
module local5_cross_acc_summary_monitor_v2 #(
    parameter int DEPTH = 14400,
    parameter int VEC_W = 32,
    parameter int ADDR_W = 14
) (
    input logic clk_core,
    input logic rst_core,
    input logic command_valid,
    input logic command_write,
    input logic [ADDR_W-1:0] command_addr,
    input logic [VEC_W-1:0] command_write_data
);
    localparam int RESOURCE_CODE = 5;
    localparam logic [63:0] DIGEST_SEED0 = 64'hcbf29ce484222325;
    localparam logic [63:0] DIGEST_SEED1 = 64'h00001505d3c4b2a1;
    localparam logic [63:0] DIGEST_PRIME = 64'h00000100000001b3;

    integer summary_fd;
    longint unsigned cycle_q;
    longint unsigned count_q;
    longint unsigned read_count_q;
    longint unsigned write_count_q;
    logic [63:0] digest0_q;
    logic [63:0] digest1_q;
    longint unsigned first_anchor_q [0:11];
    longint unsigned last_anchor_q [0:11];
    bit first_valid_q;
    string output_prefix;
    string output_path;
    string observer_instance_path;
    string target_instance_path;

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

    task automatic record_command;
        longint unsigned values [0:11];
        string domain_tag;
        begin
            values[0] = count_q;
            values[1] = cycle_q;
            values[2] = 64'(command_write);
            values[3] = 64'(command_addr);
            values[4] = command_write ? 64'(command_write_data) : 0;
            for (integer index = 5; index < 12; index++)
                values[index] = 0;

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
            if (command_write)
                write_count_q = write_count_q + 1;
            else
                read_count_q = read_count_q + 1;
        end
    endtask

    task automatic write_anchor(input string kind, input bit first);
        begin
            $fwrite(summary_fd,
                "A,CROSS_ACC_COMMAND,%s,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
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
        if (DEPTH != 14400 || VEC_W != 32)
            $fatal(1, "cross observer bound to a non-target memory shape");
        if (!$value$plusargs("CROSS_SUMMARY_PREFIX_V2=%s", output_prefix)
            || !$value$plusargs(
                "CROSS_ACC_TARGET_INSTANCE=%s", target_instance_path))
            $fatal(1, "cross summary v2 plusargs are mandatory");
        output_path = $sformatf("%s.%s.csv", output_prefix,
            observer_instance_path);
        summary_fd = $fopen(output_path, "w");
        if (summary_fd == 0)
            $fatal(1, "cannot open cross summary v2 output");
        cycle_q = 0;
        count_q = 0;
        read_count_q = 0;
        write_count_q = 0;
        digest0_q = DIGEST_SEED0;
        digest1_q = DIGEST_SEED1;
        first_valid_q = 1'b0;
        $fwrite(summary_fd, "SCHEMA,local5_cross_acc_summary_v2\n");
        $fwrite(summary_fd, "ORIGIN,RTL_LOWER_PORT\n");
        $fwrite(summary_fd, "OBSERVER_INSTANCE,%s\n",
            observer_instance_path);
        $fwrite(summary_fd, "TARGET_INSTANCE,%s\n", target_instance_path);
        $fwrite(summary_fd,
            "DIGEST,FNV1A64_AND_DJB2XOR64_V1,%016h,%016h\n",
            DIGEST_SEED0, DIGEST_SEED1);
        $fwrite(summary_fd, "RESOURCE_CODE,CROSS_ACC_COMMAND,%0d\n",
            RESOURCE_CODE);
        $fwrite(summary_fd, "PAYLOAD_U64_COUNT,10\n");
    end

    always @(posedge clk_core) begin
        if (rst_core) begin
            cycle_q = 0;
            count_q = 0;
            read_count_q = 0;
            write_count_q = 0;
            digest0_q = DIGEST_SEED0;
            digest1_q = DIGEST_SEED1;
            first_valid_q = 1'b0;
        end else begin
            if (command_valid)
                record_command();
            cycle_q = cycle_q + 1;
        end
    end

    final begin
        if (summary_fd != 0) begin
            if (first_valid_q) begin
                write_anchor("FIRST", 1'b1);
                write_anchor("LAST", 1'b0);
            end
            $fwrite(summary_fd, "S,CROSS_ACC_COMMAND,%0d,%016h,%016h\n",
                count_q, digest0_q, digest1_q);
            $fwrite(summary_fd, "L,CROSS_ACC_PROTOCOL_LEDGER,%0d,%0d,%0d\n",
                count_q, read_count_q, write_count_q);
            $fwrite(summary_fd, "END,%0d,RTL_LOWER_PORT\n", cycle_q);
            $fclose(summary_fd);
        end
    end
endmodule

`default_nettype wire
