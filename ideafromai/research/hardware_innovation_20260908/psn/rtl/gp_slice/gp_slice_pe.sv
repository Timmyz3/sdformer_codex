// Bounded P4/F1 PE. Private W!=0 compaction, two NR4 packets, 56 physical S15
// words (28 active), and a scalar noncausal T10 consumer. No SRAM/PPA claim.
// Adapted arithmetic/selection from gp_temporal_pe and shared_fc1_psn; those
// standalone leaves are unchanged. A W10 partial adder and optional member
// CSA accompany ONE Acc24 add/subtract datapath shared by S commits and PSN.
module gp_slice_pe (
    input logic clk, rst_n, start,
    input logic reduce_mode,
    input logic [2:0] rank,
    input logic [55:0] decode,
    input logic [8:0] program_length,
    input logic row_valid,
    output logic row_ready,
    input logic [11:0] row_codes,
    input logic signed [7:0] row_weight,
    input logic end_valid,
    output logic end_ready,
    output logic syn_done,
    input logic consumer_go,
    output logic [7:0] program_address,
    input logic [15:0] program_data,
    output logic [3:0] tau_address,
    input logic signed [23:0] tau_data,
    output logic out_valid,
    input logic out_ready,
    output logic [1:0] out_position,
    output logic [9:0] out_gates,
    output logic finished,
    output logic overflow,
    output logic [1:0] dbg_occupancy,
    output logic dbg_packet,
    output logic [2:0] dbg_packet_sources,
    output logic dbg_zero_weight,
    output logic [3:0] dbg_members,
    output logic dbg_commit,
    output logic [4:0] dbg_s_address,
    output logic signed [14:0] dbg_s_value,
    output logic dbg_psn
);
    typedef enum logic [2:0] {IDLE, SYN, LOAD_S, RUN_PSN, OUTPUT_P, FINISHED} phase_t;
    phase_t phase;
    logic [55:0] decode_q;
    logic [2:0] rank_q;
    logic [8:0] length_q;
    logic reduce_q;
    logic [11:0] packet_codes [0:1][0:3];
    logic signed [7:0] packet_weight [0:1][0:3];
    logic [2:0] packet_count [0:1];
    logic [1:0] packet_ready;
    logic wr_ptr, rd_ptr, sealed, executing;
    logic [27:0] pending, incoming_pending;
    logic scalar_active;
    logic [3:0] scalar_remaining;
    logic signed [9:0] scalar_partial;
    logic [4:0] scalar_address;
    logic signed [14:0] state_words [0:55];
    logic [55:0] state_valid;
    logic signed [14:0] source_cache [0:6];
    logic signed [23:0] u;
    logic [2:0] load_r;
    logic [1:0] position;
    logic [7:0] pc;
    logic [9:0] gates;

    logic [3:0] position_pending, members, remaining_after;
    logic found_p, found_r, found_s;
    logic [1:0] selected_p, selected_s;
    logic [2:0] selected_r;
    logic [4:0] address;
    logic signed [9:0] operands [0:3];
    logic [9:0] csa_s1, csa_c1, csa_s2, csa_c2;
    logic signed [9:0] delta, scalar_sum;
    logic [5:0] rf_address;
    logic signed [14:0] rf_value;
    logic signed [23:0] alu_a, alu_b;
    logic alu_sub;
    logic [25:0] alu_guarded;
    logic signed [24:0] result_guarded;
    logic signed [23:0] result;
    logic commit;
    logic [27:0] pending_after;

    assign row_ready = phase==SYN && !sealed && !packet_ready[wr_ptr]
                       && !(executing && wr_ptr==rd_ptr);
    assign end_ready = phase==SYN && !sealed;
    assign syn_done = phase==SYN && sealed && !executing && packet_ready==0;
    assign program_address = pc;
    assign tau_address = program_data[14:11];
    assign out_valid = phase==OUTPUT_P;
    assign out_position = position;
    assign out_gates = gates;
    assign finished = phase==FINISHED;
    assign dbg_packet = phase==SYN && !executing && packet_ready[rd_ptr];
    assign dbg_packet_sources = packet_count[rd_ptr];
    assign dbg_zero_weight = row_valid && row_ready && row_weight==0;
    assign dbg_psn = phase==RUN_PSN;
    assign dbg_commit = commit;
    assign dbg_s_address = address;
    assign dbg_s_value = result[14:0];
    always_comb begin
        dbg_occupancy = 0;
        for (int b=0;b<2;b++)
            if (packet_count[b]!=0 || packet_ready[b] || (executing && int'(rd_ptr)==b))
                dbg_occupancy = dbg_occupancy+1'b1;
        incoming_pending = 0;
        for (int s=0;s<4;s++) for (int p=0;p<4;p++) for (int r=0;r<7;r++)
            if (s<int'(packet_count[rd_ptr]) && r<int'(rank_q)
                && decode_q[int'(packet_codes[rd_ptr][s][p*3 +: 3])*7+r])
                incoming_pending[p*7+r]=1'b1;
    end
    // P4 then R7 first destination, then L1D over the four private source rows.
    always_comb begin
        position_pending=0;found_p=0;found_r=0;found_s=0;
        selected_p=0;selected_r=0;selected_s=0;
        for (int p=0;p<4;p++) position_pending[p]=|pending[p*7 +: 7];
        for (int p=0;p<4;p++) if (!found_p && position_pending[p]) begin
            selected_p=2'(p);found_p=1;
        end
        for (int r=0;r<7;r++) if (!found_r && pending[int'(selected_p)*7+r]) begin
            selected_r=3'(r);found_r=1;
        end
        address=scalar_active ? scalar_address : 5'(int'(selected_p)*7+int'(selected_r));
        members=0;
        for (int s=0;s<4;s++)
            if (s<int'(packet_count[rd_ptr])
                && decode_q[int'(packet_codes[rd_ptr][s][(int'(address)/7)*3 +: 3])*7
                             +int'(address)%7]) members[s]=1;
        if (scalar_active) members=scalar_remaining;
        for (int s=0;s<4;s++) begin
            if (!found_s && members[s]) begin selected_s=2'(s);found_s=1;end
            operands[s]=members[s] ? {{2{packet_weight[rd_ptr][s][7]}},packet_weight[rd_ptr][s]} : 10'sd0;
        end
        remaining_after=members & ~(4'b1<<selected_s);
        pending_after=pending & ~(28'b1<<address);
    end
    assign csa_s1=operands[0]^operands[1]^operands[2];
    assign csa_c1=((operands[0]&operands[1])|(operands[0]&operands[2])|(operands[1]&operands[2]))<<1;
    assign csa_s2=csa_s1^csa_c1^operands[3];
    assign csa_c2=((csa_s1&csa_c1)|(csa_s1&operands[3])|(csa_c1&operands[3]))<<1;
    assign scalar_sum=(scalar_active ? scalar_partial : 10'sd0)+operands[selected_s];
    assign delta=reduce_q ? $signed(csa_s2+csa_c2) : scalar_sum;
    assign commit=phase==SYN && executing && (|pending) && (reduce_q || remaining_after==0);
    assign dbg_members=phase!=SYN || !executing || pending==0 ? 4'd0
                       : reduce_q ? members : (4'b1<<selected_s);
    // Exactly one S read mux and one S write port. LOAD_S uses the same mux.
    assign rf_address=phase==LOAD_S ? 6'(int'(position)*7+int'(load_r)) : {1'b0,address};
    assign rf_value=state_valid[rf_address] ? state_words[rf_address] : 15'sd0;
    always_comb begin
        alu_a=0;alu_b=0;alu_sub=0;
        if (commit) begin
            alu_a={{9{rf_value[14]}},rf_value};alu_b={{14{delta[9]}},delta};
        end else if (phase==RUN_PSN) begin
            if (program_data[9]) begin
                alu_a=0;alu_b=tau_data;alu_sub=1; // explicit -tau instruction
            end else begin
                alu_a=u;
                if (!program_data[15]) begin
                    alu_b=$signed({{9{source_cache[program_data[2:0]][14]}},
                                    source_cache[program_data[2:0]]}) <<< program_data[7:3];
                    alu_sub=program_data[8];
                end
            end
        end
    end
    // One carry chain with sign guard; xor/carry-in also implements subtraction.
    assign alu_guarded={alu_a[23],alu_a,1'b1}
                       + {({alu_b[23],alu_b}^{25{alu_sub}}),alu_sub};
    assign result_guarded=$signed(alu_guarded[25:1]);
    assign result=result_guarded[23:0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase<=IDLE;packet_ready<=0;packet_count[0]<=0;packet_count[1]<=0;
            wr_ptr<=0;rd_ptr<=0;sealed<=0;executing<=0;pending<=0;
            scalar_active<=0;scalar_partial<=0;scalar_remaining<=0;scalar_address<=0;
            state_valid<=0;position<=0;load_r<=0;pc<=0;gates<=0;u<=0;
            rank_q<=0;decode_q<=0;length_q<=0;reduce_q<=0;overflow<=0;
        end else if (start) begin
            phase<=SYN;packet_ready<=0;packet_count[0]<=0;packet_count[1]<=0;
            wr_ptr<=0;rd_ptr<=0;sealed<=0;executing<=0;pending<=0;
            scalar_active<=0;scalar_partial<=0;scalar_remaining<=0;
            state_valid<=0;position<=0;load_r<=0;pc<=0;gates<=0;u<=0;
            rank_q<=rank;decode_q<=decode;length_q<=program_length;reduce_q<=reduce_mode;overflow<=0;
        end else begin
            if (phase==SYN) begin
                if (row_valid && row_ready && row_weight!=0) begin
                    packet_codes[wr_ptr][packet_count[wr_ptr][1:0]]<=row_codes;
                    packet_weight[wr_ptr][packet_count[wr_ptr][1:0]]<=row_weight;
                    packet_count[wr_ptr]<=packet_count[wr_ptr]+1'b1;
                    if (packet_count[wr_ptr]==3) begin packet_ready[wr_ptr]<=1;wr_ptr<=~wr_ptr;end
                end
                if (end_valid && end_ready) begin
                    sealed<=1;
                    if (packet_count[wr_ptr]!=0 && !packet_ready[wr_ptr]
                        && !(executing && wr_ptr==rd_ptr)) begin
                        packet_ready[wr_ptr]<=1;wr_ptr<=~wr_ptr;
                    end
                end
                if (!executing && packet_ready[rd_ptr]) begin
                    executing<=1;packet_ready[rd_ptr]<=0;pending<=incoming_pending;
                    scalar_active<=0;scalar_partial<=0;
                end else if (executing) begin
                    if (pending==0 || (commit && pending_after==0)) begin
                        executing<=0;packet_count[rd_ptr]<=0;rd_ptr<=~rd_ptr;
                    end
                    if (commit) begin
                        state_words[rf_address]<=result[14:0];state_valid[rf_address]<=1;
                        overflow<=overflow | (result_guarded[24:14]!={11{result_guarded[14]}});
                        pending<=pending_after;scalar_active<=0;scalar_partial<=0;
                    end else if (pending!=0) begin
                        scalar_active<=1;scalar_address<=address;
                        scalar_remaining<=remaining_after;scalar_partial<=scalar_sum;
                    end
                end
                if (consumer_go && syn_done) begin
                    phase<=LOAD_S;position<=0;load_r<=0;gates<=0;
                end
            end else if (phase==LOAD_S) begin
                source_cache[load_r]<=rf_value;
                if (load_r+1'b1==rank_q) begin pc<=0;phase<=RUN_PSN;end
                else load_r<=load_r+1'b1;
            end else if (phase==RUN_PSN) begin
                u<=result;
                overflow<=overflow | (result_guarded[24]!=result_guarded[23]);
                if (program_data[10]) gates[program_data[14:11]]<=!result[23];
                if ({1'b0,pc}+9'd1==length_q) phase<=OUTPUT_P;
                else pc<=pc+1'b1;
            end else if (phase==OUTPUT_P && out_ready) begin
                if (position==3) begin phase<=FINISHED;state_valid<=0;end
                else begin position<=position+1'b1;load_r<=0;gates<=0;phase<=LOAD_S;end
            end
        end
    end
endmodule
