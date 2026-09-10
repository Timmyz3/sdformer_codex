// Single physical PE, isolated functional implementation, no SRAM/PPA claim.
// Common source NR4 window: zero-W rows are masked within the supplied window;
// this does NOT implement GustavSNN's upstream private zero-W compaction.
// Physical address = (f*P+p)*7+r, with P8/F1 or P4/F2. R is 1..7.
// Accepted source packages remain stable internally until every F/p/r is done.
module gp_temporal_pe (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         cancel,
    input  logic         start_valid,
    output logic         start_ready,
    input  logic         start_p4f2,
    input  logic         start_reduce,
    input  logic [2:0]   start_rank,
    input  logic [55:0]  start_decode, // code*7+r, binary incidence, no theta=1
    input  logic [15:0]  start_tag,
    input  logic         packet_valid,
    output logic         packet_ready,
    input  logic [95:0]  packet_codes, // (source*8+p)*3, four source rows
    input  logic [63:0]  packet_weights, // (f*4+source)*8, signed INT8
    input  logic [2:0]   packet_sources, // 0..4; unused rows/p/F ignored
    input  logic         packet_last,
    output logic         busy,
    output logic         state_complete,
    input  logic         rd_valid,
    output logic         rd_ready,
    input  logic [5:0]   rd_address,
    output logic         rsp_valid,
    input  logic         rsp_ready,
    output logic [5:0]   rsp_address,
    output logic signed [14:0] rsp_value,
    output logic         rsp_present,
    output logic [15:0]  rsp_tag,
    input  logic         consumer_done_valid,
    output logic         consumer_done_ready,
    output logic         overflow,
    // Combinational execution observation, no counters or additional state.
    output logic         execute_valid,
    output logic         commit_valid,
    output logic [5:0]   commit_address,
    output logic [3:0]   execute_members,
    output logic signed [9:0] commit_delta
);
    typedef enum logic [1:0] {IDLE, WAIT_PACKET, EXECUTE, HOLD_STATE} phase_t;
    phase_t phase;
    logic p4f2_q, reduce_q, last_q;
    logic [2:0] rank_q, count_q;
    logic [55:0] decode_q;
    logic [15:0] tag_q;
    logic [95:0] codes_q;
    logic [63:0] weights_q;

    // One register-array read/modify/write port. No reset of the wide data.
    logic signed [14:0] state_words [0:55];
    logic [55:0] state_valid;
    logic [55:0] pending_q, incoming_pending, pending_after;

    logic scalar_active;
    logic [5:0] scalar_address;
    logic [3:0] scalar_remaining;
    logic signed [9:0] scalar_partial;

    logic rsp_valid_q;
    logic [7:0] position_pending;
    logic position_found, row_found, source_found;
    logic [2:0] selected_position, selected_row;
    logic [5:0] selected_address;
    logic [3:0] members, remaining_after_scalar;
    logic [1:0] selected_source;
    logic [2:0] local_position;
    logic selected_f;
    logic signed [9:0] operands [0:3];
    logic [9:0] csa_s1, csa_c1, csa_s2, csa_c2;
    logic signed [9:0] reduced_sum, scalar_sum, delta;
    logic signed [15:0] old_extended, sum_extended;
    logic [5:0] state_read_address;
    logic state_read_enable, state_read_present;
    logic signed [14:0] state_read_value;

    assign busy = phase != IDLE;
    assign start_ready = phase == IDLE && !cancel;
    assign packet_ready = phase == WAIT_PACKET && !cancel;
    assign state_complete = phase == HOLD_STATE && !cancel;
    assign rd_ready = state_complete && (!rsp_valid_q || rsp_ready);
    assign rsp_valid = rsp_valid_q && !cancel;
    // Consumer release is explicit and cannot race a pending/read response.
    assign consumer_done_ready = state_complete && !rsp_valid_q && !rd_valid;

    always_comb begin
        incoming_pending = '0;
        for (int fp = 0; fp < 8; fp++) begin
            for (int r = 0; r < 7; r++) begin
                for (int s = 0; s < 4; s++) begin
                    if (s < int'(packet_sources) && r < int'(rank_q)
                        && packet_weights[((p4f2_q ? fp/4 : 0)*4+s)*8 +: 8] != 8'b0
                        && decode_q[int'(packet_codes[(s*8+(p4f2_q ? fp%4 : fp))*3 +: 3])*7+r])
                        incoming_pending[fp*7+r] = 1'b1;
                end
            end
        end
    end

    // Hierarchical P8 then R7 selection, not a flat 56-way event merger.
    always_comb begin
        position_pending = '0;
        for (int p = 0; p < 8; p++)
            position_pending[p] = |pending_q[p*7 +: 7];
        position_found = 1'b0;
        row_found = 1'b0;
        selected_position = '0;
        selected_row = '0;
        for (int p = 0; p < 8; p++) begin
            if (!position_found && position_pending[p]) begin
                position_found = 1'b1;
                selected_position = 3'(p);
            end
        end
        for (int r = 0; r < 7; r++) begin
            if (!row_found && pending_q[int'(selected_position)*7+r]) begin
                row_found = 1'b1;
                selected_row = 3'(r);
            end
        end
        selected_address = scalar_active ? scalar_address
                              : 6'(int'(selected_position)*7+int'(selected_row));
        selected_f = p4f2_q && selected_address >= 6'd28;
        local_position = 3'(int'(selected_address)/7-(selected_f ? 4 : 0));
        members = '0;
        for (int s = 0; s < 4; s++) begin
            if (s < int'(count_q)
                && weights_q[(int'(selected_f)*4+s)*8 +: 8] != 8'b0
                && decode_q[int'(codes_q[(s*8+int'(local_position))*3 +: 3])*7
                             +int'(selected_address)%7])
                members[s] = 1'b1;
        end
        if (scalar_active) members = scalar_remaining;
        source_found = 1'b0;
        selected_source = '0;
        for (int s = 0; s < 4; s++) begin
            if (!source_found && members[s]) begin
                source_found = 1'b1;
                selected_source = 2'(s);
            end
            operands[s] = members[s]
                ? $signed({{2{weights_q[(int'(selected_f)*4+s)*8+7]}},
                              weights_q[(int'(selected_f)*4+s)*8 +: 8]}) : 10'sd0;
        end
        remaining_after_scalar = members & ~(4'b0001 << selected_source);
        pending_after = pending_q & ~(56'b1 << selected_address);
    end

    // Actual two-level 3:2 compression followed by carry propagation.
    // Four signed W8 sum exactly in W10: [-512,508]. Truncating compressor
    // carries here implements modulo-2^10 arithmetic; the final sum fits W10.
    assign csa_s1 = operands[0] ^ operands[1] ^ operands[2];
    assign csa_c1 = ((operands[0]&operands[1]) | (operands[0]&operands[2])
                    | (operands[1]&operands[2])) << 1;
    assign csa_s2 = csa_s1 ^ csa_c1 ^ operands[3];
    assign csa_c2 = ((csa_s1&csa_c1) | (csa_s1&operands[3])
                    | (csa_c1&operands[3])) << 1;
    assign reduced_sum = $signed(csa_s2+csa_c2);
    assign scalar_sum = (scalar_active ? scalar_partial : 10'sd0)
                            + operands[selected_source];
    assign delta = reduce_q ? reduced_sum : scalar_sum;
    // Consumer reads and accumulation explicitly share ONE read-address mux.
    assign state_read_address = phase == HOLD_STATE ? rd_address : selected_address;
    assign state_read_enable = commit_valid || (rd_valid && rd_ready);
    assign state_read_present = state_read_enable && state_read_address < 6'd56
                                 && state_valid[state_read_address];
    assign state_read_value = state_read_present ? state_words[state_read_address] : 15'sd0;
    assign old_extended = {state_read_value[14],state_read_value};
    assign sum_extended = old_extended + {{6{delta[9]}},delta};
    assign execute_valid = phase == EXECUTE && (|pending_q) && !cancel;
    assign commit_valid = execute_valid && (reduce_q || !(|remaining_after_scalar));
    assign execute_members = !execute_valid ? 4'b0
                             : reduce_q ? members : (4'b0001 << selected_source);
    assign commit_address = selected_address;
    assign commit_delta = delta;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase <= IDLE;
            state_valid <= '0;
            pending_q <= '0;
            scalar_active <= 1'b0;
            scalar_address <= '0;
            scalar_remaining <= '0;
            scalar_partial <= '0;
            rsp_valid_q <= 1'b0;
            rsp_address <= '0;
            rsp_value <= '0;
            rsp_present <= 1'b0;
            rsp_tag <= '0;
            p4f2_q <= 1'b0;
            reduce_q <= 1'b0;
            last_q <= 1'b0;
            rank_q <= '0;
            count_q <= '0;
            decode_q <= '0;
            tag_q <= '0;
            codes_q <= '0;
            weights_q <= '0;
            overflow <= 1'b0;
        end else if (cancel) begin
            // Explicit task cancellation also withdraws an unread response.
            phase <= IDLE;
            state_valid <= '0;
            pending_q <= '0;
            scalar_active <= 1'b0;
            rsp_valid_q <= 1'b0;
            overflow <= 1'b0;
        end else begin
            if (rsp_valid_q && rsp_ready) rsp_valid_q <= 1'b0;
            case (phase)
                IDLE: if (start_valid) begin
                    p4f2_q <= start_p4f2;
                    reduce_q <= start_reduce;
                    rank_q <= start_rank;
                    decode_q <= start_decode;
                    tag_q <= start_tag;
                    overflow <= 1'b0;
                    phase <= WAIT_PACKET;
                end
                WAIT_PACKET: if (packet_valid) begin
                    codes_q <= packet_codes;
                    weights_q <= packet_weights;
                    count_q <= packet_sources;
                    last_q <= packet_last;
                    pending_q <= incoming_pending;
                    scalar_active <= 1'b0;
                    scalar_partial <= '0;
                    phase <= EXECUTE;
                end
                EXECUTE: begin
                    if (!(|pending_q)) begin
                        phase <= last_q ? HOLD_STATE : WAIT_PACKET;
                    end else if (commit_valid) begin
                        state_words[selected_address] <= sum_extended[14:0];
                        state_valid[selected_address] <= 1'b1;
                        overflow <= overflow | (sum_extended[15] != sum_extended[14]);
                        pending_q <= pending_after;
                        scalar_active <= 1'b0;
                        scalar_partial <= '0;
                        if (!(|pending_after)) phase <= last_q ? HOLD_STATE : WAIT_PACKET;
                    end else begin
                        scalar_active <= 1'b1;
                        scalar_address <= selected_address;
                        scalar_remaining <= remaining_after_scalar;
                        scalar_partial <= scalar_sum;
                    end
                end
                HOLD_STATE: begin
                    if (rd_valid && rd_ready) begin
                        rsp_valid_q <= 1'b1;
                        rsp_address <= rd_address;
                        rsp_tag <= tag_q;
                        rsp_present <= state_read_present;
                        rsp_value <= state_read_value;
                    end
                    if (consumer_done_valid && consumer_done_ready) begin
                        state_valid <= '0;
                        phase <= IDLE;
                    end
                end
                default: phase <= IDLE;
            endcase
        end
    end
endmodule
