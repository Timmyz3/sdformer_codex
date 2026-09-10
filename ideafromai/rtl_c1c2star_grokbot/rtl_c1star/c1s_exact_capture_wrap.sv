// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-05
// Module: c1s_exact_capture_wrap — Exact-product capture gated by OGEC×PRRC
// INNOVATION: Exact hits latch/count ONLY when OGEC exact_en AND PRRC allow_exact;
//             finite CAPACITY ends the window (done). Not flat zero-skip capture.
// Protocol: capture_en↑ → busy; valid beats accumulate gated hits; CAPACITY or
//           capture_en↓ → done; capture_en must drop to re-arm (IDLE).

`timescale 1ns/1ps
`default_nettype none

module c1s_exact_capture_wrap #(
  parameter int N_TILE   = 8,
  parameter int CNT_W    = 16,
  parameter int CAPACITY = 16
) (
  input  logic                 clk,
  input  logic                 rst_n,
  input  logic                 valid_i,
  input  logic                 capture_en,
  input  logic [N_TILE-1:0]    exact_en,     // OGEC exact path
  input  logic                 allow_exact,  // PRRC budget allow
  output logic [N_TILE-1:0]    hit_bitmap,
  output logic [CNT_W-1:0]     capture_cnt,
  output logic                 busy,
  output logic                 done
);

  localparam logic [1:0] ST_IDLE = 2'd0;
  localparam logic [1:0] ST_BUSY = 2'd1;
  localparam logic [1:0] ST_DONE = 2'd2;

  logic [1:0]        state, state_n;
  logic [N_TILE-1:0] gated_hits;
  logic [CNT_W-1:0]  pop;
  logic [CNT_W-1:0]  add_cnt;
  integer            i;
  localparam logic [CNT_W-1:0] CAP_C = CAPACITY;

  assign gated_hits = allow_exact ? exact_en : {N_TILE{1'b0}};
  assign busy = (state == ST_BUSY);
  assign done = (state == ST_DONE);

  always @(*) begin
    pop = {CNT_W{1'b0}};
    for (i = 0; i < N_TILE; i = i + 1) begin
      if (gated_hits[i])
        pop = pop + {{(CNT_W-1){1'b0}}, 1'b1};
    end
  end

  always @(*) begin
    if ((capture_cnt + pop) >= CAP_C)
      add_cnt = CAP_C;
    else
      add_cnt = capture_cnt + pop;
  end

  always @(*) begin
    state_n = state;
    case (state)
      ST_IDLE: begin
        if (capture_en)
          state_n = ST_BUSY;
      end
      ST_BUSY: begin
        if (!capture_en)
          state_n = ST_DONE;
        else if (valid_i && (add_cnt >= CAP_C))
          state_n = ST_DONE;
      end
      ST_DONE: begin
        if (!capture_en)
          state_n = ST_IDLE;
      end
      default: state_n = ST_IDLE;
    endcase
  end

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state       <= ST_IDLE;
      hit_bitmap  <= {N_TILE{1'b0}};
      capture_cnt <= {CNT_W{1'b0}};
    end else begin
      if (state == ST_IDLE && state_n == ST_BUSY) begin
        hit_bitmap  <= {N_TILE{1'b0}};
        capture_cnt <= {CNT_W{1'b0}};
      end else if (state == ST_BUSY && valid_i) begin
        hit_bitmap  <= hit_bitmap | gated_hits;
        capture_cnt <= add_cnt;
      end
      state <= state_n;
    end
  end

endmodule

`default_nettype wire
