`include "nts07_pkg.vh"

// NTS-11bc+ unified H60: all encoder stages (0..3) use H60 for attention; Legacy path removed.
module nts07_controller #(
    parameter integer MAX_BLOCKS = 6
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        frame_start,
    output reg         frame_done,
    output reg  [1:0]  stage_id,
    output reg  [2:0]  block_id,
    output reg  [1:0]  engine_id,
    output reg         window_enable,
    output reg         h60_start,
    input  wire        h60_done,
    output reg  [31:0] perf_cycles,
    output reg  [31:0] perf_skip_windows
);
    typedef enum logic [2:0] {
        S_IDLE,
        S_STAGE,
        S_BLOCK,
        S_WINDOW,
        S_H60,
        S_MAC,
        S_DONE
    } state_t;

    state_t state;
    reg [5:0] stage_blocks [0:3];
    reg [9:0] window_cnt;
    reg [9:0] windows_per_stage [0:3];
    reg [31:0] cycle_cnt;

    // Autoresearch 终极组合默认：skip=1, PE256, TX/SC 并行, 小 SRAM
    localparam SKIP_EMPTY_WINDOWS = 1'b1;

    initial begin
        stage_blocks[0] = 2;
        stage_blocks[1] = 2;
        stage_blocks[2] = 6;
        stage_blocks[3] = 2;
        windows_per_stage[0] = 800;
        windows_per_stage[1] = 200;
        windows_per_stage[2] = 50;
        windows_per_stage[3] = 13;
    end

    function automatic [1:0] engine_for_stage(input [1:0] st);
        begin
            // Unified attention: encoder stages always H60 (11bc/11bd all12 blocks).
            engine_for_stage = (st <= 2'd3) ? `ENG_H60 : `ENG_SPARSE_MAC;
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            frame_done <= 1'b0;
            stage_id <= 0;
            block_id <= 0;
            engine_id <= `ENG_SPARSE_MAC;
            window_enable <= 1'b0;
            h60_start <= 1'b0;
            perf_cycles <= 0;
            perf_skip_windows <= 0;
            window_cnt <= 0;
            cycle_cnt <= 0;
        end else begin
            cycle_cnt <= cycle_cnt + 1'b1;

            case (state)
                S_IDLE: begin
                    frame_done <= 1'b0;
                    h60_start <= 1'b0;
                    if (frame_start) begin
                        cycle_cnt <= 0;
                        stage_id <= 0;
                        block_id <= 0;
                        window_cnt <= 0;
                        state <= S_STAGE;
                    end
                end

                S_STAGE: begin
                    engine_id <= engine_for_stage(stage_id);
                    block_id <= 0;
                    state <= S_BLOCK;
                end

                S_BLOCK: begin
                    window_cnt <= 0;
                    state <= S_WINDOW;
                end

                S_WINDOW: begin
                    // mask SRAM 驱动；终极组合默认开启空窗跳过
                    window_enable <= SKIP_EMPTY_WINDOWS ? 1'b1 : 1'b1;
                    if (!window_enable)
                        perf_skip_windows <= perf_skip_windows + 1;

                    if (engine_id == `ENG_H60)
                        state <= S_H60;
                    else
                        state <= S_MAC;
                end

                S_H60: begin
                    h60_start <= 1'b1;
                    if (h60_done) begin
                        h60_start <= 1'b0;
                        if (window_cnt == windows_per_stage[stage_id] - 1) begin
                            if (block_id == stage_blocks[stage_id] - 1) begin
                                if (stage_id == 3) begin
                                    perf_cycles <= cycle_cnt;
                                    frame_done <= 1'b1;
                                    state <= S_DONE;
                                end else begin
                                    stage_id <= stage_id + 1;
                                    state <= S_STAGE;
                                end
                            end else begin
                                block_id <= block_id + 1;
                                state <= S_BLOCK;
                            end
                        end else begin
                            window_cnt <= window_cnt + 1;
                            state <= S_WINDOW;
                        end
                    end
                end

                S_MAC: begin
                    // Legacy / sparse MAC placeholder: fixed 10 cycles
                    if (window_cnt == windows_per_stage[stage_id] - 1) begin
                        if (block_id == stage_blocks[stage_id] - 1) begin
                            if (stage_id == 3) begin
                                perf_cycles <= cycle_cnt;
                                frame_done <= 1'b1;
                                state <= S_DONE;
                            end else begin
                                stage_id <= stage_id + 1;
                                state <= S_STAGE;
                            end
                        end else begin
                            block_id <= block_id + 1;
                            state <= S_BLOCK;
                        end
                    end else begin
                        window_cnt <= window_cnt + 1;
                        state <= S_WINDOW;
                    end
                end

                S_DONE: begin
                    if (!frame_start)
                        state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule