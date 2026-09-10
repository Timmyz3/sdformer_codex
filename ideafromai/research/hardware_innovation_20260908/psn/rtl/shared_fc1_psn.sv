// Isolated functional prototype: register/RF ports, no SRAM macro claim.
// FC1 and PSN use the same BANKS x LANES ACC-bit add/subtract carry chains.
// The caller supplies integer W, code decode, legal uops, and admitted ranges.
// In particular all PSN prefixes INCLUDING -tau must fit signed ACC bits.
module shared_fc1_psn #(
    parameter int LANES = 96,
    parameter int P     = 32,
    parameter int R     = 7,
    parameter int BANKS = 7,
    parameter int ACC   = 24,
    parameter int SBITS = 15
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    cfg_decode_valid,
    input  logic [2:0]              cfg_code,
    input  logic [R*2-1:0]          cfg_decode,
    input  logic                    cfg_uop_valid,
    input  logic [7:0]              cfg_uop_addr,
    input  logic [15:0]             cfg_uop,
    input  logic                    cfg_tau_valid,
    input  logic [3:0]              cfg_tau_row,
    input  logic [LANES*ACC-1:0]     cfg_tau,
    input  logic                    start,
    input  logic [5:0]              positions,
    input  logic [8:0]              program_length,
    input  logic                    src_valid,
    output logic                    src_ready,
    input  logic [P*3-1:0]          src_codes,
    input  logic [LANES*8-1:0]      src_weight,
    input  logic                    src_last,
    output logic                    out_valid,
    input  logic                    out_ready,
    output logic [5:0]              out_position,
    output logic [LANES*10-1:0]     out_spikes,
    output logic                    busy,
    output logic                    done
);
    localparam int WORD_W  = (P < 2) ? 1 : $clog2(P);
    localparam int ROW_W   = (R < 2) ? 1 : $clog2(R);
    localparam int GROUP_W = (BANKS < 2) ? 1 : $clog2(BANKS);
    localparam int COUNT_W = $clog2(BANKS+1);

    typedef enum logic [3:0] {
        IDLE, ACCEPT_SOURCE, UPDATE_SOURCE, PREPARE_GROUP,
        LOAD_GROUP, FETCH_FIRST, EXECUTE_UOP, DRAIN_GROUP
    } phase_t;
    phase_t phase;

    logic [R*2-1:0] decode_table [0:7];
    logic [15:0] uop_mem [0:255];
    logic [LANES-1:0][ACC-1:0] negative_tau [0:9];

    // Seven physical banks each have P words. The R*P logical words occupy
    // addr=r*P+p; bank=addr%BANKS; word=addr/BANKS. Unused words stay invalid.
    logic [LANES-1:0][SBITS-1:0] state_rf [0:BANKS-1][0:P-1];
    logic [P-1:0] state_valid [0:BANKS-1];
    logic [P-1:0] pending [0:BANKS-1];
    logic [P-1:0] pending_negative [0:BANKS-1];
    logic [P-1:0] decoded_pending [0:BANKS-1];
    logic [P-1:0] decoded_negative [0:BANKS-1];
    logic [P-1:0] pending_after_issue [0:BANKS-1];
    logic pending_after_any;
    logic bank_issue [0:BANKS-1];
    logic [WORD_W-1:0] issue_word [0:BANKS-1];

    logic [LANES*8-1:0] weight_q;
    logic last_source_q;
    logic [5:0] positions_q, group_base_q;
    logic [COUNT_W-1:0] group_count_q;
    logic [GROUP_W-1:0] drain_q;
    logic [ROW_W-1:0] load_row_q;
    logic [8:0] length_q, pc_q;
    logic [15:0] uop_q;
    logic [8:0] next_pc;
    // This bounded prototype admits signed-8 CSD coefficients: shift 0..7.
    // The common interface retains bits [7:6], which the caller keeps zero.
    logic [1:0] unused_uop_shift_reserved;

    logic [LANES-1:0][SBITS-1:0] group_source [0:BANKS-1][0:R-1];
    logic [LANES-1:0][ACC-1:0] u_state [0:BANKS-1];
    // Packet bit t*LANES+lane is the output gate at time t for that lane.
    logic [LANES*10-1:0] packets [0:BANKS-1];

    // One actual read address per physical RF bank. During LOAD_GROUP the
    // bank-to-group-position permutation is fixed by the current r and base.
    logic rf_read_enable [0:BANKS-1];
    logic [WORD_W-1:0] rf_read_word [0:BANKS-1];
    logic [LANES-1:0][SBITS-1:0] rf_read_value [0:BANKS-1];
    integer load_group_index [0:BANKS-1];

    logic signed [ACC-1:0] alu_base [0:BANKS-1][0:LANES-1];
    logic signed [ACC-1:0] alu_operand [0:BANKS-1][0:LANES-1];
    logic alu_subtract [0:BANKS-1];
    logic [ACC:0] alu_extended_sum [0:BANKS-1][0:LANES-1];
    logic signed [ACC-1:0] alu_sum [0:BANKS-1][0:LANES-1];
    logic [LANES-1:0][ACC-1:0] sum_vector [0:BANKS-1];
    logic [LANES-1:0][SBITS-1:0] source_sum_vector [0:BANKS-1];
    logic [LANES-1:0] gate_vector [0:BANKS-1];

    assign next_pc = pc_q + 9'd1;
    assign unused_uop_shift_reserved = uop_q[7:6];
    assign busy = (phase != IDLE);
    assign src_ready = (phase == ACCEPT_SOURCE);
    assign out_valid = (phase == DRAIN_GROUP);
    assign out_position = group_base_q + 6'(drain_q);
    assign out_spikes = packets[drain_q];

    // Decode one source column. The selected coefficient vector is retained
    // until every physical-bank bitmap is empty; there is no hidden queue.
    always_comb begin
        for (int b = 0; b < BANKS; b++) begin
            decoded_pending[b] = '0;
            decoded_negative[b] = '0;
        end
        for (int r = 0; r < R; r++) begin
            for (int p = 0; p < P; p++) begin
                if (p < int'(positions_q)) begin
                    decoded_pending[(r*P+p)%BANKS][(r*P+p)/BANKS] =
                        (decode_table[src_codes[p*3 +: 3]][r*2 +: 2] != 2'b00);
                    decoded_negative[(r*P+p)%BANKS][(r*P+p)/BANKS] =
                        decode_table[src_codes[p*3 +: 3]][r*2+1];
                end
            end
        end
    end

    always_comb begin
        pending_after_any = 1'b0;
        for (int b = 0; b < BANKS; b++) begin
            bank_issue[b] = 1'b0;
            issue_word[b] = '0;
            pending_after_issue[b] = pending[b];
            for (int word = 0; word < P; word++) begin
                if (!bank_issue[b] && pending[b][word]) begin
                    bank_issue[b] = 1'b1;
                    issue_word[b] = WORD_W'(word);
                    pending_after_issue[b][word] = 1'b0;
                end
            end
            pending_after_any = pending_after_any | (|pending_after_issue[b]);
        end
    end

    always_comb begin
        for (int b = 0; b < BANKS; b++) begin
            load_group_index[b] =
                (b+BANKS-((int'(load_row_q)*P+int'(group_base_q))%BANKS))%BANKS;
            rf_read_word[b] = '0;
            rf_read_enable[b] = 1'b0;
            if (phase == UPDATE_SOURCE) begin
                rf_read_word[b] = issue_word[b];
                rf_read_enable[b] = bank_issue[b] && state_valid[b][issue_word[b]];
            end else if (phase == LOAD_GROUP) begin
                rf_read_word[b] = WORD_W'((int'(load_row_q)*P+int'(group_base_q)
                                          +load_group_index[b])/BANKS);
                rf_read_enable[b] = (load_group_index[b] < int'(group_count_q))
                                    && state_valid[b][rf_read_word[b]];
            end
            for (int lane = 0; lane < LANES; lane++) begin
                rf_read_value[b][lane] = '0;
                if (rf_read_enable[b])
                    rf_read_value[b][lane] = state_rf[b][rf_read_word[b]][lane];
            end
        end
    end

    always_comb begin
        for (int b = 0; b < BANKS; b++) begin
            alu_subtract[b] = 1'b0;
            for (int lane = 0; lane < LANES; lane++) begin
                alu_base[b][lane] = '0;
                alu_operand[b][lane] = '0;
            end
            if (phase == UPDATE_SOURCE && bank_issue[b]) begin
                alu_subtract[b] = pending_negative[b][issue_word[b]];
                for (int lane = 0; lane < LANES; lane++) begin
                    alu_base[b][lane] =
                        {{(ACC-SBITS){rf_read_value[b][lane][SBITS-1]}}, rf_read_value[b][lane]};
                    alu_operand[b][lane] =
                        {{(ACC-8){weight_q[lane*8+7]}}, weight_q[lane*8 +: 8]};
                end
            end else if (phase == EXECUTE_UOP && b < int'(group_count_q)) begin
                alu_subtract[b] = uop_q[8] && !uop_q[15];
                for (int lane = 0; lane < LANES; lane++) begin
                    alu_base[b][lane] = uop_q[9] ? negative_tau[uop_q[14:11]][lane]
                                                              : u_state[b][lane];
                    if (!uop_q[15])
                        alu_operand[b][lane] =
                            $signed({{(ACC-SBITS){group_source[b][uop_q[2:0]][lane][SBITS-1]}},
                                      group_source[b][uop_q[2:0]][lane]}) <<< uop_q[5:3];
                end
            end else if (phase == IDLE && cfg_tau_valid && b == 0) begin
                // Configuration also reuses the same lane adders for -tau.
                alu_subtract[b] = 1'b1;
                for (int lane = 0; lane < LANES; lane++)
                    alu_operand[b][lane] = $signed(cfg_tau[lane*ACC +: ACC]);
            end
        end
    end

    generate
        for (genvar b = 0; b < BANKS; b++) begin : g_bank_add
            for (genvar lane = 0; lane < LANES; lane++) begin : g_lane_add
                // The extra low bit implements carry-in: 1+subtract carries
                // exactly subtract into the ACC-bit sum. Thus this ONE plus
                // expression is base+operand or base-operand, without a
                // separate vector negation adder. The low result bit is unused.
                assign alu_extended_sum[b][lane] =
                    {alu_base[b][lane], 1'b1}
                    + {(alu_operand[b][lane] ^ {ACC{alu_subtract[b]}}), alu_subtract[b]};
                assign alu_sum[b][lane] = alu_extended_sum[b][lane][ACC:1];
                assign sum_vector[b][lane] = alu_sum[b][lane];
                assign source_sum_vector[b][lane] = alu_sum[b][lane][SBITS-1:0];
                assign gate_vector[b][lane] = !alu_sum[b][lane][ACC-1];
            end
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase <= IDLE;
            done <= 1'b0;
            positions_q <= '0;
            group_base_q <= '0;
            group_count_q <= '0;
            drain_q <= '0;
            load_row_q <= '0;
            length_q <= '0;
            pc_q <= '0;
            uop_q <= '0;
            weight_q <= '0;
            last_source_q <= 1'b0;
            for (int b = 0; b < BANKS; b++) begin
                state_valid[b] <= '0;
                pending[b] <= '0;
                pending_negative[b] <= '0;
            end
        end else begin
            done <= 1'b0;
            if (phase == IDLE) begin
                if (cfg_decode_valid)
                    decode_table[cfg_code] <= cfg_decode;
                if (cfg_uop_valid)
                    uop_mem[cfg_uop_addr] <= cfg_uop;
                if (cfg_tau_valid)
                    negative_tau[cfg_tau_row] <= sum_vector[0];
            end
            case (phase)
                IDLE: begin
                    if (start) begin
                        positions_q <= positions;
                        length_q <= program_length;
                        group_base_q <= '0;
                        drain_q <= '0;
                        for (int b = 0; b < BANKS; b++) begin
                            state_valid[b] <= '0;
                            pending[b] <= '0;
                        end
                        if (positions == 0) done <= 1'b1;
                        else phase <= ACCEPT_SOURCE;
                    end
                end
                ACCEPT_SOURCE: begin
                    if (src_valid) begin
                        weight_q <= src_weight;
                        last_source_q <= src_last;
                        for (int b = 0; b < BANKS; b++) begin
                            pending[b] <= decoded_pending[b];
                            pending_negative[b] <= decoded_negative[b];
                        end
                        phase <= UPDATE_SOURCE;
                    end
                end
                UPDATE_SOURCE: begin
                    for (int b = 0; b < BANKS; b++) begin
                        pending[b] <= pending_after_issue[b];
                        if (bank_issue[b]) begin
                            state_valid[b][issue_word[b]] <= 1'b1;
                            state_rf[b][issue_word[b]] <= source_sum_vector[b];
                        end
                    end
                    if (!pending_after_any)
                        phase <= last_source_q ? PREPARE_GROUP : ACCEPT_SOURCE;
                end
                PREPARE_GROUP: begin
                    if (int'(positions_q)-int'(group_base_q) < BANKS)
                        group_count_q <= COUNT_W'(int'(positions_q)-int'(group_base_q));
                    else group_count_q <= COUNT_W'(BANKS);
                    load_row_q <= '0;
                    phase <= LOAD_GROUP;
                end
                LOAD_GROUP: begin
                    // R clock edges read R source planes, one word per bank
                    // per edge. Invalid words are explicitly loaded as zero.
                    for (int b = 0; b < BANKS; b++)
                        if (load_group_index[b] < int'(group_count_q))
                            group_source[load_group_index[b]][load_row_q] <= rf_read_value[b];
                    if (int'(load_row_q) == R-1) phase <= FETCH_FIRST;
                    else load_row_q <= load_row_q + 1'b1;
                end
                FETCH_FIRST: begin
                    pc_q <= '0;
                    uop_q <= uop_mem[0];
                    phase <= EXECUTE_UOP;
                end
                EXECUTE_UOP: begin
                    for (int b = 0; b < BANKS; b++) begin
                        if (b < int'(group_count_q)) begin
                            u_state[b] <= sum_vector[b];
                            if (uop_q[10])
                                packets[b][int'(uop_q[14:11])*LANES +: LANES] <= gate_vector[b];
                        end
                    end
                    if (uop_q[10] && uop_q[14:11] == 4'd9) begin
                        drain_q <= '0;
                        phase <= DRAIN_GROUP;
                    end else if (next_pc < length_q) begin
                        pc_q <= next_pc;
                        uop_q <= uop_mem[next_pc[7:0]];
                    end
                    // Programs supplied by the caller end with t9/last;
                    // no dynamic program repair or skip scheduler is hidden.
                end
                DRAIN_GROUP: begin
                    if (out_ready) begin
                        if (int'(group_base_q)+int'(drain_q)+1 >= int'(positions_q)) begin
                            phase <= IDLE;
                            done <= 1'b1;
                        end else if (int'(drain_q)+1 == int'(group_count_q)) begin
                            group_base_q <= group_base_q + 6'(BANKS);
                            phase <= PREPARE_GROUP;
                        end else drain_q <= drain_q + 1'b1;
                    end
                end
                default: phase <= IDLE;
            endcase
        end
    end
endmodule
