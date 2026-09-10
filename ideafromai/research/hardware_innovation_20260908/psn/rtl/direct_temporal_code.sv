// Standard integer source-encoding leaf; no GP layer or SRAM-macro claim.
// Input: ten signed12 samples, sample t in in_values[12*t +: 12].
// CSD uop: [3:0] input index, [6:4] shift 0..7, [7] subtract.
// Config is written only while idle. A row may be a constant predicate.
// One shared add/sub chain performs -tau initialization and all CSD terms.
module direct_temporal_code (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         cfg_uop_valid,
    input  logic [6:0]   cfg_uop_address,
    input  logic [7:0]   cfg_uop,
    input  logic         cfg_row_valid,
    input  logic [1:0]   cfg_row,
    input  logic [6:0]   cfg_row_start,
    input  logic [5:0]   cfg_row_length,
    input  logic [23:0]  cfg_tau,
    input  logic         cfg_variable,
    input  logic         cfg_constant,
    input  logic         cfg_map_valid,
    input  logic [2:0]   cfg_map_address,
    input  logic [2:0]   cfg_map_value,
    input  logic         in_valid,
    output logic         in_ready,
    input  logic [119:0] in_values,
    output logic         out_valid,
    input  logic         out_ready,
    output logic [2:0]   out_code,
    output logic         busy
);
    typedef enum logic [2:0] {IDLE, INITIALIZE, EXECUTE, LOOKUP, OUTPUT_CODE} state_t;
    state_t state;
    logic [7:0] program_mem [0:127];
    logic [6:0] row_start [0:2];
    logic [5:0] row_length [0:2];
    logic [23:0] row_tau [0:2];
    logic row_variable [0:2];
    logic row_constant [0:2];
    logic [2:0] class_map [0:7];

    logic [119:0] samples;
    logic [23:0] accumulator;
    logic [1:0] row_index;
    logic [6:0] pc;
    logic [5:0] remaining;
    logic [7:0] uop;
    logic [2:0] predicates;
    logic [2:0] code_register;

    logic [11:0] selected_sample;
    logic [23:0] shifted_sample;
    logic [23:0] add_base, add_operand;
    logic subtract;
    logic unused_carry_lsb;
    logic [23:0] result;

    always_comb begin
        in_ready = (state == IDLE);
        out_valid = (state == OUTPUT_CODE);
        out_code = code_register;
        busy = (state != IDLE);
        selected_sample = samples[12*int'(uop[3:0]) +: 12];
        shifted_sample = {{12{selected_sample[11]}}, selected_sample} << uop[6:4];
        add_base = accumulator;
        add_operand = shifted_sample;
        subtract = uop[7];
        if (state == INITIALIZE) begin
            add_base = 24'b0;
            add_operand = row_tau[row_index];
            subtract = 1'b1;
        end
        // The low bit supplies carry-in to the one physical add/sub chain.
        {result, unused_carry_lsb} = {add_base, 1'b1}
                                  + {add_operand ^ {24{subtract}}, subtract};
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            row_index <= 2'b0;
            predicates <= 3'b0;
            code_register <= 3'b0;
        end else begin
            if (state == IDLE) begin
                if (cfg_uop_valid)
                    program_mem[cfg_uop_address] <= cfg_uop;
                if (cfg_row_valid) begin
                    row_start[cfg_row] <= cfg_row_start;
                    row_length[cfg_row] <= cfg_row_length;
                    row_tau[cfg_row] <= cfg_tau;
                    row_variable[cfg_row] <= cfg_variable;
                    row_constant[cfg_row] <= cfg_constant;
                end
                if (cfg_map_valid)
                    class_map[cfg_map_address] <= cfg_map_value;
            end
            case (state)
                IDLE: begin
                    if (in_valid) begin
                        samples <= in_values;
                        row_index <= 2'b0;
                        predicates <= 3'b0;
                        state <= INITIALIZE;
                    end
                end
                INITIALIZE: begin
                    accumulator <= result;
                    if (!row_variable[row_index] || row_length[row_index] == 0) begin
                        predicates[row_index] <= row_variable[row_index] ? !result[23]
                                                                        : row_constant[row_index];
                        if (row_index == 2)
                            state <= LOOKUP;
                        else
                            row_index <= row_index + 2'd1;
                    end else begin
                        uop <= program_mem[row_start[row_index]];
                        pc <= row_start[row_index] + 7'd1;
                        remaining <= row_length[row_index];
                        state <= EXECUTE;
                    end
                end
                EXECUTE: begin
                    accumulator <= result;
                    if (remaining == 1) begin
                        predicates[row_index] <= !result[23];
                        if (row_index == 2)
                            state <= LOOKUP;
                        else begin
                            row_index <= row_index + 2'd1;
                            state <= INITIALIZE;
                        end
                    end else begin
                        uop <= program_mem[pc];
                        pc <= pc + 7'd1;
                        remaining <= remaining - 6'd1;
                    end
                end
                LOOKUP: begin
                    code_register <= class_map[predicates];
                    state <= OUTPUT_CODE;
                end
                OUTPUT_CODE: begin
                    if (out_ready)
                        state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
