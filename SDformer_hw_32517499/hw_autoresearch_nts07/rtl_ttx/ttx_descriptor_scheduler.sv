`default_nettype none

module ttx_descriptor_scheduler #(
    parameter int TOKEN_W = 8,
    parameter int WINDOW_W = 10,
    parameter int HEAD_W = 5
)(
    input  logic                 clk,
    input  logic                 rst_n,
    input  logic                 start_frame,

    output logic                 row_req_valid,
    input  logic                 row_req_ready,
    output logic [1:0]           row_stage,
    output logic [2:0]           row_block,
    output logic [HEAD_W-1:0]    row_head,
    output logic [WINDOW_W-1:0]  row_window,
    output logic [TOKEN_W-1:0]   row_n_tokens,

    input  logic                 row_done,
    output logic                 busy,
    output logic                 done,
    output logic [15:0]          perf_rows_issued
);
    typedef enum logic [1:0] {
        ST_IDLE,
        ST_ISSUE,
        ST_WAIT,
        ST_DONE
    } state_t;

    state_t state_q;
    logic [3:0] descriptor_idx_q;
    logic [HEAD_W-1:0] head_idx_q;
    logic [WINDOW_W-1:0] window_idx_q;
    logic [HEAD_W-1:0] descriptor_heads_w;
    logic [WINDOW_W-1:0] descriptor_windows_w;
    logic [15:0] rows_issued_q;

    always_comb begin
        row_stage = 2'd0;
        row_block = 3'd0;
        descriptor_heads_w = HEAD_W'(3);
        descriptor_windows_w = WINDOW_W'(440);

        unique case (descriptor_idx_q)
            4'd0: begin row_stage = 2'd0; row_block = 3'd0; descriptor_heads_w = HEAD_W'(3);  descriptor_windows_w = WINDOW_W'(440); end
            4'd1: begin row_stage = 2'd0; row_block = 3'd1; descriptor_heads_w = HEAD_W'(3);  descriptor_windows_w = WINDOW_W'(440); end
            4'd2: begin row_stage = 2'd1; row_block = 3'd0; descriptor_heads_w = HEAD_W'(6);  descriptor_windows_w = WINDOW_W'(120); end
            4'd3: begin row_stage = 2'd1; row_block = 3'd1; descriptor_heads_w = HEAD_W'(6);  descriptor_windows_w = WINDOW_W'(120); end
            4'd4: begin row_stage = 2'd2; row_block = 3'd0; descriptor_heads_w = HEAD_W'(12); descriptor_windows_w = WINDOW_W'(30);  end
            4'd5: begin row_stage = 2'd2; row_block = 3'd1; descriptor_heads_w = HEAD_W'(12); descriptor_windows_w = WINDOW_W'(30);  end
            4'd6: begin row_stage = 2'd2; row_block = 3'd2; descriptor_heads_w = HEAD_W'(12); descriptor_windows_w = WINDOW_W'(30);  end
            4'd7: begin row_stage = 2'd2; row_block = 3'd3; descriptor_heads_w = HEAD_W'(12); descriptor_windows_w = WINDOW_W'(30);  end
            4'd8: begin row_stage = 2'd2; row_block = 3'd4; descriptor_heads_w = HEAD_W'(12); descriptor_windows_w = WINDOW_W'(30);  end
            4'd9: begin row_stage = 2'd2; row_block = 3'd5; descriptor_heads_w = HEAD_W'(12); descriptor_windows_w = WINDOW_W'(30);  end
            4'd10: begin row_stage = 2'd3; row_block = 3'd0; descriptor_heads_w = HEAD_W'(24); descriptor_windows_w = WINDOW_W'(10); end
            default: begin row_stage = 2'd3; row_block = 3'd1; descriptor_heads_w = HEAD_W'(24); descriptor_windows_w = WINDOW_W'(10); end
        endcase
    end

    always_comb begin
        row_req_valid = (state_q == ST_ISSUE);
        row_head = head_idx_q;
        row_window = window_idx_q;
        row_n_tokens = TOKEN_W'(162);
        busy = (state_q != ST_IDLE);
        done = (state_q == ST_DONE);
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            descriptor_idx_q <= '0;
            head_idx_q <= '0;
            window_idx_q <= '0;
            rows_issued_q <= '0;
        end else begin
            unique case (state_q)
                ST_IDLE: begin
                    if (start_frame) begin
                        descriptor_idx_q <= '0;
                        head_idx_q <= '0;
                        window_idx_q <= '0;
                        rows_issued_q <= '0;
                        state_q <= ST_ISSUE;
                    end
                end

                ST_ISSUE: begin
                    if (row_req_valid && row_req_ready) begin
                        rows_issued_q <= rows_issued_q + 1'b1;
                        state_q <= ST_WAIT;
                    end
                end

                ST_WAIT: begin
                    if (row_done) begin
                        if (window_idx_q + 1'b1 < descriptor_windows_w) begin
                            window_idx_q <= window_idx_q + 1'b1;
                            state_q <= ST_ISSUE;
                        end else if (head_idx_q + 1'b1 < descriptor_heads_w) begin
                            window_idx_q <= '0;
                            head_idx_q <= head_idx_q + 1'b1;
                            state_q <= ST_ISSUE;
                        end else if (descriptor_idx_q != 4'd11) begin
                            descriptor_idx_q <= descriptor_idx_q + 1'b1;
                            head_idx_q <= '0;
                            window_idx_q <= '0;
                            state_q <= ST_ISSUE;
                        end else begin
                            state_q <= ST_DONE;
                        end
                    end
                end

                ST_DONE: begin
                    state_q <= ST_IDLE;
                end

                default: begin
                    state_q <= ST_IDLE;
                end
            endcase
        end
    end

    assign perf_rows_issued = rows_issued_q;
endmodule

`default_nettype wire
