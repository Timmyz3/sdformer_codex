// One-outstanding-operation ping-pong reduction address controller.
// Consumer must acknowledge ONLY after all reads and the destination write retire.
// No SRAM, arithmetic, DMA, load/export engine or numerical BN implementation here.
module bn_reduce_addr (
    input  logic clk,
    input  logic rst_n,
    input  logic cfg_valid,
    output logic cfg_ready,
    input  logic [12:0] cfg_samples,
    input  logic [12:0] cfg_channels,
    output logic cmd_valid,
    input  logic cmd_ready,
    output logic cmd_carry,
    output logic [12:0] cmd_src0,
    output logic [12:0] cmd_src1,
    output logic [12:0] cmd_dst,
    output logic [3:0] cmd_src0_bank,
    output logic [3:0] cmd_src1_bank,
    output logic [3:0] cmd_dst_bank,
    output logic [8:0] cmd_src0_row,
    output logic [8:0] cmd_src1_row,
    output logic [8:0] cmd_dst_row,
    output logic [3:0] cmd_level,
    output logic cmd_level_end,
    input  logic retire_valid,
    output logic retire_ready,
    output logic finish_valid,
    input  logic finish_ready,
    output logic finish_fault,
    output logic [12:0] result_base,
    output logic [12:0] result_channels
);
    typedef enum logic [1:0] {IDLE, ISSUE, WAIT_WRITE, FINISH} state_t;
    state_t state;
    logic [12:0] n, channels, group_idx, channel_idx;
    logic [12:0] src0, src1, dst;
    logic src_region;
    logic [3:0] level;
    logic [25:0] cfg_words;
    logic [12:0] next_n;
    logic last_channel, last_group;

    assign cfg_words = cfg_samples * cfg_channels;
    assign next_n = (n + 13'd1) >> 1;
    assign last_channel = channel_idx == channels - 13'd1;
    assign last_group = group_idx == next_n - 13'd1;
    assign cfg_ready = state == IDLE;
    assign cmd_valid = state == ISSUE;
    assign retire_ready = state == WAIT_WRITE;
    assign finish_valid = state == FINISH;
    assign cmd_carry = n[0] && (group_idx == (n >> 1));
    assign cmd_src0 = src0;
    // A carry has no second read. Zero the unused address explicitly.
    assign cmd_src1 = cmd_carry ? 13'd0 : src1;
    assign cmd_dst = dst;
    assign cmd_src0_bank = cmd_src0[3:0];
    assign cmd_src1_bank = cmd_src1[3:0];
    assign cmd_dst_bank = cmd_dst[3:0];
    assign cmd_src0_row = cmd_src0[12:4];
    assign cmd_src1_row = cmd_src1[12:4];
    assign cmd_dst_row = cmd_dst[12:4];
    assign cmd_level = level;
    assign cmd_level_end = last_channel && last_group;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            n <= 0; channels <= 0; group_idx <= 0; channel_idx <= 0;
            src0 <= 0; src1 <= 0; dst <= 0; src_region <= 0; level <= 0;
            finish_fault <= 0; result_base <= 0; result_channels <= 0;
        end else begin
            case (state)
                IDLE: if (cfg_valid) begin
                    n <= cfg_samples; channels <= cfg_channels;
                    group_idx <= 0; channel_idx <= 0; level <= 0; src_region <= 0;
                    src0 <= 0; src1 <= cfg_channels; dst <= 13'd4096;
                    result_base <= 0; result_channels <= cfg_channels;
                    finish_fault <= 0;
                    if (cfg_samples == 0 || cfg_channels == 0 || cfg_words > 26'd4096) begin
                        finish_fault <= 1;
                        state <= FINISH;
                    end else if (cfg_samples == 1) begin
                        state <= FINISH;
                    end else begin
                        state <= ISSUE;
                    end
                end
                ISSUE: if (cmd_ready) state <= WAIT_WRITE;
                WAIT_WRITE: if (retire_valid) begin
                    if (last_channel && last_group) begin
                        if (n <= 2) begin
                            result_base <= src_region ? 13'd0 : 13'd4096;
                            state <= FINISH;
                        end else begin
                            // Previous level's last write has retired. Reuse is now legal.
                            n <= next_n; src_region <= ~src_region; level <= level + 4'd1;
                            group_idx <= 0; channel_idx <= 0;
                            src0 <= src_region ? 13'd0 : 13'd4096;
                            src1 <= (src_region ? 13'd0 : 13'd4096) + channels;
                            dst <= src_region ? 13'd4096 : 13'd0;
                            state <= ISSUE;
                        end
                    end else begin
                        dst <= dst + 13'd1;
                        if (last_channel) begin
                            channel_idx <= 0; group_idx <= group_idx + 13'd1;
                            src0 <= src0 + channels + 13'd1;
                            src1 <= src1 + channels + 13'd1;
                        end else begin
                            channel_idx <= channel_idx + 13'd1;
                            src0 <= src0 + 13'd1; src1 <= src1 + 13'd1;
                        end
                        state <= ISSUE;
                    end
                end
                FINISH: if (finish_ready) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
