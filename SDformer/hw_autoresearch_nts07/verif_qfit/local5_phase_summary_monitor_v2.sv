`timescale 1ns/1ps
`default_nettype none

// Passive simulation-only monitor. It has no outputs and drives no DUT signal.
module local5_phase_summary_monitor_v2 #(
    parameter int H = 24,
    parameter int HEAD_W = 5,
    parameter int OUTPUT_TILE_W = 5,
    parameter int TOKEN_ID_W = 9,
    parameter int RESULT_ADDR_W = 14,
    parameter int OUT_W = 5,
    parameter int LANE_W = 5,
    parameter int ACC_W = 32,
    parameter int TCFM_BANK_ADDR_W = 7
) (
    input logic clk_core,
    input logic rst_core,
    input logic group_valid,
    input logic group_ready,
    input logic group_done_valid,
    input logic group_done_ready,
    input logic tile_start_valid,
    input logic tile_start_ready,
    input logic [1:0] tile_start_stage,
    input logic [2:0] tile_start_block,
    input logic [8:0] tile_start_window,
    input logic [OUTPUT_TILE_W-1:0] tile_start_output_tile,
    input logic tile_done_valid,
    input logic tile_done_ready,
    input logic head_job_valid,
    input logic head_job_ready,
    input logic [HEAD_W-1:0] head_job_input_head,
    input logic [OUTPUT_TILE_W-1:0] head_job_output_tile,
    input logic head_done_valid,
    input logic head_done_ready,
    input logic [HEAD_W-1:0] head_done_input_head,
    input logic token_req_valid,
    input logic token_req_ready,
    input logic [HEAD_W-1:0] token_req_input_head,
    input logic [TOKEN_ID_W-1:0] token_req_token_id,
    input logic token_rsp_valid,
    input logic token_rsp_ready,
    input logic [HEAD_W-1:0] token_rsp_input_head,
    input logic [TOKEN_ID_W-1:0] token_rsp_token_id,
    input logic weight_req_valid,
    input logic weight_req_ready,
    input logic [HEAD_W-1:0] weight_req_input_head,
    input logic [OUTPUT_TILE_W-1:0] weight_req_output_tile,
    input logic [LANE_W-1:0] weight_req_lane,
    input logic [OUT_W-1:0] weight_req_out,
    input logic weight_rsp_valid,
    input logic weight_rsp_ready,
    input logic [HEAD_W-1:0] weight_rsp_input_head,
    input logic [OUTPUT_TILE_W-1:0] weight_rsp_output_tile,
    input logic [LANE_W-1:0] weight_rsp_lane,
    input logic [OUT_W-1:0] weight_rsp_out,
    input logic tile_result_valid,
    input logic tile_result_ready,
    input logic [OUTPUT_TILE_W-1:0] tile_result_output_tile,
    input logic tile_result_plane,
    input logic [3:0] tile_result_y,
    input logic [3:0] tile_result_x,
    input logic [OUT_W-1:0] tile_result_out,
    input logic [3:0] tx_state,
    input logic [4:0] head_state,
    input logic [HEAD_W-1:0] active_head,
    input logic memory_command_valid,
    input logic memory_command_write,
    input logic [RESULT_ADDR_W-1:0] memory_command_addr,
    input logic [ACC_W-1:0] memory_command_write_data,
    input logic tcfm_term_commit,
    input logic tcfm_term_source_plane,
    input logic [3:0] tcfm_term_source_y,
    input logic [3:0] tcfm_term_source_x,
    input logic [LANE_W-1:0] tcfm_term_lane,
    input logic [4:0] tcfm_term_destination_mask,
    input logic protocol_error,
    input logic scheduler_error
);
    localparam int R_REL_REQ = 0;
    localparam int R_REL_RSP = 1;
    localparam int R_WEIGHT_REQ = 2;
    localparam int R_WEIGHT_RSP = 3;
    localparam int R_FINAL = 4;
    localparam int R_CROSS = 5;
    localparam int R_TCFM5 = 6;
    localparam int RESOURCE_COUNT = 7;
    localparam logic [63:0] DIGEST_SEED0 = 64'hcbf29ce484222325;
    localparam logic [63:0] DIGEST_SEED1 = 64'h00001505d3c4b2a1;
    localparam logic [63:0] DIGEST_PRIME = 64'h00000100000001b3;

    integer phase_fd;
    integer summary_fd;
    longint unsigned cycle_q;
    longint unsigned phase_sequence_q;
    longint unsigned event_count_q [0:RESOURCE_COUNT-1];
    logic [63:0] digest0_q [0:RESOURCE_COUNT-1];
    logic [63:0] digest1_q [0:RESOURCE_COUNT-1];
    longint unsigned first_anchor_q [0:RESOURCE_COUNT-1][0:11];
    longint unsigned last_anchor_q [0:RESOURCE_COUNT-1][0:11];
    bit first_valid_q [0:RESOURCE_COUNT-1];
    longint unsigned cross_protocol_count_q;
    longint unsigned cross_read_count_q;
    longint unsigned cross_write_count_q;
    logic [63:0] cross_protocol_digest0_q;
    logic [63:0] cross_protocol_digest1_q;
    longint unsigned tcfm_update_count_q;
    longint unsigned tcfm_mask_mismatch_count_q;
    integer active_head_phase_q;
    integer active_head_phase_start_q;
    integer active_head_phase_tile_q;
    integer active_head_phase_head_q;
    integer active_head_transaction_start_q;
    integer active_head_transaction_tile_q;
    integer active_head_transaction_head_q;
    integer active_tile_phase_start_q;
    integer active_tile_q;
    integer active_group_phase_start_q;
    integer active_drain_phase_start_q;
    integer active_drain_tile_q;
    integer expected_stage_q;
    integer expected_block_q;
    integer expected_window_q;
    integer observed_stage_q;
    integer observed_block_q;
    integer observed_window_q;
    string phase_path;
    string summary_path;
    string main_resource_instance;
    string cross_resource_instance;
    string tcfm_resource_instance;
    string monitor_instance_path;
    string resource_name [0:RESOURCE_COUNT-1];
    string resource_instance [0:RESOURCE_COUNT-1];
    bit closed_q;

    function automatic integer classify_head_phase(input logic [4:0] state);
        begin
            case (state)
                5'd1, 5'd2: classify_head_phase = 1;
                5'd3, 5'd4, 5'd5, 5'd6, 5'd7, 5'd8, 5'd9:
                    classify_head_phase = 2;
                5'd10, 5'd11, 5'd12: classify_head_phase = 3;
                5'd13, 5'd14: classify_head_phase = 4;
                5'd15: classify_head_phase = 5;
                default: classify_head_phase = 0;
            endcase
        end
    endfunction

    function automatic string head_phase_name(input integer phase);
        begin
            case (phase)
                1: head_phase_name = "HEAD_WEIGHT";
                2: head_phase_name = "HEAD_FRONTEND";
                3: head_phase_name = "HEAD_READOUT";
                4: head_phase_name = "HEAD_RELEASE";
                5: head_phase_name = "HEAD_ERROR";
                default: head_phase_name = "INVALID";
            endcase
        end
    endfunction

    function automatic logic [4:0] expected_tcfm_mask(
        input integer y,
        input integer x,
        input logic [4:0] destination_mask
    );
        integer role_y;
        integer role_x;
        integer bank;
        begin
            expected_tcfm_mask = '0;
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
                    && role_y >= 0 && role_y < 15
                    && role_x >= 0 && role_x < 15) begin
                    bank = (role_x + 2 * role_y) % 5;
                    expected_tcfm_mask[bank] = 1'b1;
                end
            end
        end
    endfunction

    function automatic logic [5*TCFM_BANK_ADDR_W-1:0]
        expected_tcfm_bank_addr_flat(
            input integer plane,
            input integer y,
            input integer x
        );
        integer role_y;
        integer role_x;
        integer bank;
        integer source_color;
        integer bank_offset;
        integer address;
        begin
            expected_tcfm_bank_addr_flat = '0;
            source_color = (x + 2*y) % 5;
            for (integer role = 0; role < 5; role = role + 1) begin
                role_y = y;
                role_x = x;
                case (role)
                    1: if (y < 14) role_y = y + 1;
                    2: if (y > 0) role_y = y - 1;
                    3: if (x < 14) role_x = x + 1;
                    4: if (x > 0) role_x = x - 1;
                    default: begin end
                endcase
                case (role)
                    1: bank_offset = 2;
                    2: bank_offset = 3;
                    3: bank_offset = 1;
                    4: bank_offset = 4;
                    default: bank_offset = 0;
                endcase
                bank = (source_color + bank_offset) % 5;
                address = plane*45 + role_y*3 + role_x/5;
                expected_tcfm_bank_addr_flat[
                    bank*TCFM_BANK_ADDR_W +: TCFM_BANK_ADDR_W
                ] = TCFM_BANK_ADDR_W'(address);
            end
        end
    endfunction

    task automatic hash_byte(
        input integer resource,
        input logic [7:0] value
    );
        begin
            digest0_q[resource] =
                (digest0_q[resource] ^ {56'd0, value}) * DIGEST_PRIME;
            digest1_q[resource] =
                ((digest1_q[resource] << 5) + digest1_q[resource])
                ^ {56'd0, value};
        end
    endtask

    task automatic hash_u16(
        input integer resource,
        input logic [15:0] value
    );
        begin
            hash_byte(resource, value[7:0]);
            hash_byte(resource, value[15:8]);
        end
    endtask

    task automatic hash_u64(
        input integer resource,
        input longint unsigned value
    );
        begin
            for (integer byte_index = 0; byte_index < 8; byte_index++)
                hash_byte(resource, value[byte_index*8 +: 8]);
        end
    endtask

    task automatic hash_string(
        input integer resource,
        input string value
    );
        begin
            for (integer index = 0; index < value.len(); index++)
                hash_byte(resource, value.getc(index));
        end
    endtask

    task automatic init_resource(
        input integer resource,
        input string name,
        input string instance_path
    );
        begin
            event_count_q[resource] = 0;
            digest0_q[resource] = DIGEST_SEED0;
            digest1_q[resource] = DIGEST_SEED1;
            first_valid_q[resource] = 1'b0;
        end
    endtask

    task automatic record_event(
        input integer resource,
        input longint unsigned event_cycle,
        input longint unsigned field0,
        input longint unsigned field1,
        input longint unsigned field2,
        input longint unsigned field3,
        input longint unsigned field4,
        input longint unsigned field5,
        input longint unsigned field6,
        input longint unsigned field7,
        input longint unsigned field8,
        input longint unsigned field9
    );
        longint unsigned values [0:11];
        string domain_tag;
        begin
            values[0] = event_count_q[resource];
            values[1] = event_cycle;
            values[2] = field0;
            values[3] = field1;
            values[4] = field2;
            values[5] = field3;
            values[6] = field4;
            values[7] = field5;
            values[8] = field6;
            values[9] = field7;
            values[10] = field8;
            values[11] = field9;
            domain_tag = "LOCAL5_PHASE_SUMMARY_V2";
            hash_u16(resource, 16'(domain_tag.len()));
            hash_string(resource, domain_tag);
            hash_u16(resource, 16'd2);
            hash_u16(resource, 16'(resource));
            hash_u16(resource, 16'(resource_instance[resource].len()));
            hash_string(resource, resource_instance[resource]);
            hash_u64(resource, values[0]);
            hash_u64(resource, values[1]);
            hash_u16(resource, 16'd80);
            for (integer index = 2; index < 12; index++) begin
                hash_u64(resource, values[index]);
            end
            for (integer index = 0; index < 12; index++) begin
                last_anchor_q[resource][index] = values[index];
                if (!first_valid_q[resource])
                    first_anchor_q[resource][index] = values[index];
            end
            first_valid_q[resource] = 1'b1;
            event_count_q[resource] = event_count_q[resource] + 1;
        end
    endtask

    task automatic hash_protocol_byte(input logic [7:0] value);
        begin
            cross_protocol_digest0_q =
                (cross_protocol_digest0_q ^ {56'd0, value}) * DIGEST_PRIME;
            cross_protocol_digest1_q =
                ((cross_protocol_digest1_q << 5)
                 + cross_protocol_digest1_q) ^ {56'd0, value};
        end
    endtask

    task automatic hash_protocol_u64(input longint unsigned value);
        begin
            for (integer byte_index = 0; byte_index < 8; byte_index++)
                hash_protocol_byte(value[byte_index*8 +: 8]);
        end
    endtask

    task automatic hash_protocol_string(input string value);
        begin
            hash_protocol_byte(8'(value.len()));
            hash_protocol_byte(8'(value.len() >> 8));
            for (integer index = 0; index < value.len(); index++)
                hash_protocol_byte(value.getc(index));
        end
    endtask

    task automatic record_cross_protocol(
        input longint unsigned rw,
        input longint unsigned addr
    );
        begin
            hash_protocol_u64(cross_protocol_count_q);
            hash_protocol_u64(rw);
            hash_protocol_u64(addr);
            cross_protocol_count_q = cross_protocol_count_q + 1;
            if (rw == 0)
                cross_read_count_q = cross_read_count_q + 1;
            else
                cross_write_count_q = cross_write_count_q + 1;
        end
    endtask

    task automatic write_phase(
        input integer tile,
        input integer head,
        input string role,
        input integer start_cycle,
        input integer end_cycle
    );
        begin
            if (end_cycle < start_cycle)
                $fatal(1, "phase interval duration is negative");
            $fwrite(phase_fd,
                "P,%0d,%0d,%0d,%0d,%0d,%0d,%s,%0d,%0d,%0d,RTL_DIRECT\n",
                phase_sequence_q, observed_stage_q, observed_block_q,
                observed_window_q, tile, head, role, start_cycle, end_cycle,
                end_cycle - start_cycle + 1);
            phase_sequence_q = phase_sequence_q + 1;
        end
    endtask

    task automatic write_anchor(
        input integer resource,
        input string kind,
        input bit first
    );
        begin
            $fwrite(summary_fd,
                "A,%s,%s,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                resource_name[resource], kind,
                first ? first_anchor_q[resource][0] : last_anchor_q[resource][0],
                first ? first_anchor_q[resource][1] : last_anchor_q[resource][1],
                first ? first_anchor_q[resource][2] : last_anchor_q[resource][2],
                first ? first_anchor_q[resource][3] : last_anchor_q[resource][3],
                first ? first_anchor_q[resource][4] : last_anchor_q[resource][4],
                first ? first_anchor_q[resource][5] : last_anchor_q[resource][5],
                first ? first_anchor_q[resource][6] : last_anchor_q[resource][6],
                first ? first_anchor_q[resource][7] : last_anchor_q[resource][7],
                first ? first_anchor_q[resource][8] : last_anchor_q[resource][8],
                first ? first_anchor_q[resource][9] : last_anchor_q[resource][9],
                first ? first_anchor_q[resource][10] : last_anchor_q[resource][10],
                first ? first_anchor_q[resource][11] : last_anchor_q[resource][11]);
        end
    endtask

    task automatic close_outputs;
        begin
            if (!closed_q) begin
                for (integer resource = 0; resource < RESOURCE_COUNT; resource++) begin
                    if (first_valid_q[resource]) begin
                        write_anchor(resource, "FIRST", 1'b1);
                        write_anchor(resource, "LAST", 1'b0);
                    end
                    $fwrite(summary_fd, "S,%s,%0d,%016h,%016h\n",
                        resource_name[resource], event_count_q[resource],
                        digest0_q[resource], digest1_q[resource]);
                end
                $fwrite(summary_fd,
                    "L,CROSS_ACC_PROTOCOL_LEDGER,%0d,%0d,%0d,%016h,%016h\n",
                    cross_protocol_count_q, cross_read_count_q,
                    cross_write_count_q, cross_protocol_digest0_q,
                    cross_protocol_digest1_q);
                $fwrite(summary_fd,
                    "L,TCFM5_TERM_LEDGER,%0d,%0d,%0d\n",
                    event_count_q[R_TCFM5], tcfm_update_count_q,
                    tcfm_mask_mismatch_count_q);
                $fwrite(summary_fd, "END,%0d,RTL_DIRECT\n", cycle_q);
                $fwrite(phase_fd, "END,%0d,%0d,RTL_DIRECT\n",
                    cycle_q, phase_sequence_q);
                $fclose(summary_fd);
                $fclose(phase_fd);
                summary_fd = 0;
                phase_fd = 0;
                closed_q = 1'b1;
            end
        end
    endtask

    initial begin
        monitor_instance_path = $sformatf("%m");
        closed_q = 1'b0;
        phase_fd = 0;
        summary_fd = 0;
        if (H < 1 || H > 32 || ACC_W > 64)
            $fatal(1, "unsupported H or ACC_W in phase summary monitor");
        if (!$value$plusargs("PHASE_INTERVALS_V2=%s", phase_path)
            || !$value$plusargs("MAIN_SUMMARY_V2=%s", summary_path)
            || !$value$plusargs("TELEMETRY_STAGE=%d", expected_stage_q)
            || !$value$plusargs("TELEMETRY_BLOCK=%d", expected_block_q)
            || !$value$plusargs("TELEMETRY_WINDOW=%d", expected_window_q)
            || !$value$plusargs(
                "MAIN_RESOURCE_INSTANCE=%s", main_resource_instance)
            || !$value$plusargs(
                "CROSS_ACC_TARGET_INSTANCE=%s", cross_resource_instance)
            || !$value$plusargs(
                "TCFM5_TARGET_INSTANCE=%s", tcfm_resource_instance))
            $fatal(1, "phase summary v2 plusargs are mandatory");
        phase_fd = $fopen(phase_path, "w");
        summary_fd = $fopen(summary_path, "w");
        if (phase_fd == 0 || summary_fd == 0)
            $fatal(1, "cannot open phase summary v2 output");
        resource_name[R_REL_REQ] = "RELATION_REQ_ACCEPT";
        resource_name[R_REL_RSP] = "RELATION_RSP_ACCEPT";
        resource_name[R_WEIGHT_REQ] = "WEIGHT_REQ_ACCEPT";
        resource_name[R_WEIGHT_RSP] = "WEIGHT_RSP_ACCEPT";
        resource_name[R_FINAL] = "FINAL_ACCEPT";
        resource_name[R_CROSS] = "CROSS_ACC_COMMAND";
        resource_name[R_TCFM5] = "TCFM5_TERM_COMMIT";
        for (integer index = 0; index < 5; index++)
            resource_instance[index] = main_resource_instance;
        resource_instance[R_CROSS] = cross_resource_instance;
        resource_instance[R_TCFM5] = tcfm_resource_instance;
        for (integer resource = 0; resource < RESOURCE_COUNT; resource++)
            init_resource(resource, resource_name[resource],
                resource_instance[resource]);
        cross_protocol_count_q = 0;
        cross_read_count_q = 0;
        cross_write_count_q = 0;
        cross_protocol_digest0_q = DIGEST_SEED0;
        cross_protocol_digest1_q = DIGEST_SEED1;
        hash_protocol_string("LOCAL5_PHASE_SUMMARY_V2");
        hash_protocol_string("local5_cross_acc_protocol_ledger_v2");
        hash_protocol_string("CROSS_ACC_PROTOCOL_LEDGER");
        hash_protocol_string(cross_resource_instance);
        tcfm_update_count_q = 0;
        tcfm_mask_mismatch_count_q = 0;
        $fwrite(phase_fd, "SCHEMA,local5_phase_interval_ledger_v2\n");
        $fwrite(phase_fd, "ORIGIN,RTL_DIRECT\n");
        $fwrite(phase_fd, "H,%0d\n", H);
        $fwrite(phase_fd,
            "COLUMNS,record,sequence,stage,block,window,tile,head,role,start_cycle,end_cycle,duration,origin\n");
        $fwrite(summary_fd, "SCHEMA,local5_ordered_summary_v2\n");
        $fwrite(summary_fd, "ORIGIN,RTL_DIRECT\n");
        $fwrite(summary_fd, "MONITOR_INSTANCE,%s\n", monitor_instance_path);
        $fwrite(summary_fd, "H,%0d\n", H);
        $fwrite(summary_fd,
            "DIGEST,FNV1A64_AND_DJB2XOR64_V1,%016h,%016h,%016h\n",
            DIGEST_SEED0, DIGEST_SEED1, DIGEST_PRIME);
        $fwrite(summary_fd, "BYTE_ORDER,LITTLE_ENDIAN\n");
        $fwrite(summary_fd,
            "SERIALIZATION,domain_u16le_ascii_schema_u16le_resource_u16le_instance_u16le_utf8_sequence_u64le_cycle_u64le_payload_len_u16le_payload_10xu64le\n");
        $fwrite(summary_fd, "PAYLOAD_U64_COUNT,10\n");
        $fwrite(summary_fd,
            "SAME_CYCLE_ORDER,RELATION_REQ,RELATION_RSP,WEIGHT_REQ,WEIGHT_RSP,FINAL,CROSS_ACC,TCFM5\n");
        $fwrite(summary_fd,
            "EMPTY_STREAM,raw_seed_without_event_frame\n");
        for (integer resource = 0; resource < RESOURCE_COUNT; resource++)
            $fwrite(summary_fd, "R,%s,%s\n",
                resource_name[resource], resource_instance[resource]);
        $fwrite(summary_fd,
            "F,RELATION_REQ_ACCEPT,tile,head,source,reserved0,reserved1,reserved2,reserved3,reserved4,reserved5,reserved6\n");
        $fwrite(summary_fd,
            "F,RELATION_RSP_ACCEPT,tile,head,source,reserved0,reserved1,reserved2,reserved3,reserved4,reserved5,reserved6\n");
        $fwrite(summary_fd,
            "F,WEIGHT_REQ_ACCEPT,tile,head,lane,out,reserved0,reserved1,reserved2,reserved3,reserved4,reserved5\n");
        $fwrite(summary_fd,
            "F,WEIGHT_RSP_ACCEPT,tile,head,lane,out,reserved0,reserved1,reserved2,reserved3,reserved4,reserved5\n");
        $fwrite(summary_fd,
            "F,FINAL_ACCEPT,tile,source,out,reserved0,reserved1,reserved2,reserved3,reserved4,reserved5,reserved6\n");
        $fwrite(summary_fd,
            "F,CROSS_ACC_COMMAND,rw,addr,write_data,reserved0,reserved1,reserved2,reserved3,reserved4,reserved5,reserved6\n");
        $fwrite(summary_fd,
            "F,TCFM5_TERM_COMMIT,source,lane,expected_mask,actual_mask,bank_addr0,bank_addr1,bank_addr2,bank_addr3,bank_addr4,reserved0\n");
        $fwrite(summary_fd,
            "P,CROSS_ACC_PROTOCOL_LEDGER,sequence_u64le,rw_u64le,addr_u64le\n");
    end

    always @(posedge clk_core) begin : p_passive_summary
        integer observed_head_phase;
        integer source_id;
        logic [4:0] expected_mask;
        logic [5*TCFM_BANK_ADDR_W-1:0] expected_bank_addr_flat;
        longint unsigned write_data_projection;
        if (rst_core) begin
            cycle_q = 0;
            phase_sequence_q = 0;
            active_head_phase_q = 0;
            active_head_phase_start_q = -1;
            active_head_phase_tile_q = -1;
            active_head_phase_head_q = -1;
            active_head_transaction_start_q = -1;
            active_head_transaction_tile_q = -1;
            active_head_transaction_head_q = -1;
            active_tile_phase_start_q = -1;
            active_tile_q = -1;
            active_group_phase_start_q = -1;
            active_drain_phase_start_q = -1;
            active_drain_tile_q = -1;
            observed_stage_q = -1;
            observed_block_q = -1;
            observed_window_q = -1;
        end else if (!closed_q) begin
            if (protocol_error || scheduler_error)
                $fatal(1, "observed DUT error during phase summary v2");
            if (group_valid && group_ready)
                active_group_phase_start_q = cycle_q;
            if (tile_start_valid && tile_start_ready) begin
                if (32'(tile_start_stage) != expected_stage_q
                    || 32'(tile_start_block) != expected_block_q
                    || 32'(tile_start_window) != expected_window_q)
                    $fatal(1, "phase summary identity mismatch");
                observed_stage_q = 32'(tile_start_stage);
                observed_block_q = 32'(tile_start_block);
                observed_window_q = 32'(tile_start_window);
                active_tile_phase_start_q = cycle_q;
                active_tile_q = 32'(tile_start_output_tile);
            end
            observed_head_phase = classify_head_phase(head_state);
            if (head_job_valid && head_job_ready) begin
                if (active_head_transaction_start_q >= 0)
                    $fatal(1, "overlapping head interval");
                active_head_transaction_start_q = cycle_q;
                active_head_transaction_tile_q = 32'(head_job_output_tile);
                active_head_transaction_head_q = 32'(head_job_input_head);
            end
            if (observed_head_phase != active_head_phase_q) begin
                if (active_head_phase_q != 0)
                    write_phase(active_head_phase_tile_q,
                        active_head_phase_head_q,
                        head_phase_name(active_head_phase_q),
                        active_head_phase_start_q, cycle_q - 1);
                active_head_phase_q = observed_head_phase;
                if (observed_head_phase != 0) begin
                    active_head_phase_start_q = cycle_q;
                    active_head_phase_tile_q = active_tile_q;
                    active_head_phase_head_q = 32'(active_head);
                end
            end
            if (head_done_valid && head_done_ready) begin
                if (active_head_transaction_start_q < 0
                    || active_head_transaction_head_q
                       != 32'(head_done_input_head))
                    $fatal(1, "head interval close mismatch");
                write_phase(active_head_transaction_tile_q,
                    active_head_transaction_head_q, "HEAD_TRANSACTION",
                    active_head_transaction_start_q, cycle_q);
                active_head_transaction_start_q = -1;
            end
            if (tx_state >= 4 && tx_state <= 6
                && active_drain_phase_start_q < 0) begin
                active_drain_phase_start_q = cycle_q;
                active_drain_tile_q = active_tile_q;
            end else if (!(tx_state >= 4 && tx_state <= 6)
                         && active_drain_phase_start_q >= 0) begin
                write_phase(active_drain_tile_q, -1, "TILE_DRAIN",
                    active_drain_phase_start_q, cycle_q - 1);
                active_drain_phase_start_q = -1;
            end
            if (token_req_valid && token_req_ready)
                record_event(R_REL_REQ, cycle_q, active_tile_q,
                    32'(token_req_input_head), 32'(token_req_token_id),
                    0, 0, 0, 0, 0, 0, 0);
            if (token_rsp_valid && token_rsp_ready)
                record_event(R_REL_RSP, cycle_q, active_tile_q,
                    32'(token_rsp_input_head), 32'(token_rsp_token_id),
                    0, 0, 0, 0, 0, 0, 0);
            if (weight_req_valid && weight_req_ready)
                record_event(R_WEIGHT_REQ, cycle_q,
                    32'(weight_req_output_tile), 32'(weight_req_input_head),
                    32'(weight_req_lane), 32'(weight_req_out),
                    0, 0, 0, 0, 0, 0);
            if (weight_rsp_valid && weight_rsp_ready)
                record_event(R_WEIGHT_RSP, cycle_q,
                    32'(weight_rsp_output_tile), 32'(weight_rsp_input_head),
                    32'(weight_rsp_lane), 32'(weight_rsp_out),
                    0, 0, 0, 0, 0, 0);
            if (tile_result_valid && tile_result_ready) begin
                source_id = 32'(tile_result_plane) * 225
                          + 32'(tile_result_y) * 15
                          + 32'(tile_result_x);
                record_event(R_FINAL, cycle_q,
                    32'(tile_result_output_tile), source_id,
                    32'(tile_result_out), 0, 0, 0, 0, 0, 0, 0);
            end
            if (memory_command_valid) begin
                write_data_projection = memory_command_write
                    ? 64'(memory_command_write_data) : 0;
                record_event(R_CROSS, cycle_q, memory_command_write,
                    32'(memory_command_addr), write_data_projection,
                    0, 0, 0, 0, 0, 0, 0);
                record_cross_protocol(memory_command_write,
                    32'(memory_command_addr));
            end
            if (tcfm_term_commit) begin
                source_id = 32'(tcfm_term_source_plane) * 225
                          + 32'(tcfm_term_source_y) * 15
                          + 32'(tcfm_term_source_x);
                expected_mask = expected_tcfm_mask(
                    32'(tcfm_term_source_y), 32'(tcfm_term_source_x),
                    tcfm_term_destination_mask);
                expected_bank_addr_flat = expected_tcfm_bank_addr_flat(
                    32'(tcfm_term_source_plane),
                    32'(tcfm_term_source_y), 32'(tcfm_term_source_x));
                for (integer bank = 0; bank < 5; bank++)
                    if (expected_mask[bank])
                        tcfm_update_count_q++;
                record_event(R_TCFM5, cycle_q, source_id,
                    32'(tcfm_term_lane), 32'(expected_mask),
                    32'(expected_mask),
                    32'(expected_bank_addr_flat[0*TCFM_BANK_ADDR_W
                                             +: TCFM_BANK_ADDR_W]),
                    32'(expected_bank_addr_flat[1*TCFM_BANK_ADDR_W
                                             +: TCFM_BANK_ADDR_W]),
                    32'(expected_bank_addr_flat[2*TCFM_BANK_ADDR_W
                                             +: TCFM_BANK_ADDR_W]),
                    32'(expected_bank_addr_flat[3*TCFM_BANK_ADDR_W
                                             +: TCFM_BANK_ADDR_W]),
                    32'(expected_bank_addr_flat[4*TCFM_BANK_ADDR_W
                                             +: TCFM_BANK_ADDR_W]), 0);
            end
            if (tile_done_valid && tile_done_ready) begin
                if (active_drain_phase_start_q >= 0) begin
                    write_phase(active_drain_tile_q, -1, "TILE_DRAIN",
                        active_drain_phase_start_q, cycle_q);
                    active_drain_phase_start_q = -1;
                end
                if (active_tile_phase_start_q < 0)
                    $fatal(1, "tile interval close without start");
                write_phase(active_tile_q, -1, "TILE_TRANSACTION",
                    active_tile_phase_start_q, cycle_q);
                active_tile_phase_start_q = -1;
            end
            if (group_done_valid && group_done_ready) begin
                if (active_head_phase_q != 0
                    || active_group_phase_start_q < 0)
                    $fatal(1, "group interval close with open phase");
                write_phase(-1, -1, "GROUP_TRANSACTION",
                    active_group_phase_start_q, cycle_q);
                if (phase_sequence_q != 1 + 2*H + 5*H*H)
                    $fatal(1, "phase interval count does not match H contract");
                close_outputs();
            end
            cycle_q = cycle_q + 1;
        end
    end
endmodule

`default_nettype wire
