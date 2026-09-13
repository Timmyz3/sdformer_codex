module pair_execution #(
  parameter integer K=864, O=16, P=20, ACC=32
)(
  input logic clk, reset_n, start,
  input logic [1:0] mode,
  input logic map_we,
  input logic [$clog2(K/2)-1:0] map_addr,
  input logic [2*$clog2(K)-1:0] map_data,
  input logic source_valid,
  output logic source_ready,
  input logic [P-1:0] source_data,
  output logic weight_req,
  input logic weight_ready,
  output logic [$clog2(O*K/2)-1:0] weight_addr,
  input logic weight_valid,
  input logic [31:0] weight_data,
  output logic result_valid,
  input logic result_ready,
  output logic signed [ACC-1:0] result_data,
  output logic [$clog2(O*P)-1:0] result_addr,
  output logic done,
  output logic [31:0] cycles, weight_words, replay_cycles, issue_cycles
);
  localparam integer KW=$clog2(K), PAIRS=K/2;
  typedef enum logic [3:0] {IDLE,LOAD,CLEAR,GET_MASK,CHECK_MASK,REQUEST,
    RESPONSE,ISSUE_FIRST,ISSUE_SECOND,NEXT_PAIR,WRITE_RESULT,FINISH} state_t;
  state_t state;
  logic [P-1:0] source_mem [K];
  logic [2*KW-1:0] map_mem [PAIRS];
  logic [P-1:0] mask_a,mask_b;
  logic signed [15:0] weight_a,weight_b;
  logic signed [ACC-1:0] pair_sum;
  logic signed [ACC-1:0] accum [P];
  logic signed [ACC-1:0] add_a [P],add_b [P],add_y [P];
  integer source_index,pair_index,output_index,write_index;
  integer lane;
  logic [1:0] selected_mode;
  always_comb begin
    source_ready=(state==LOAD);
    weight_req=(state==REQUEST);
    weight_addr=$clog2(O*K/2)'(output_index*PAIRS+pair_index);
    result_valid=(state==WRITE_RESULT);
    result_data=accum[write_index];
    result_addr=$clog2(O*P)'(output_index*P+write_index);
    done=(state==FINISH);
    for(integer j=0;j<P;j=j+1) begin
      add_a[j]=accum[j]; add_b[j]='0;
      if(state==ISSUE_FIRST) begin
        if(selected_mode==0) begin
          if(|mask_a) begin if(mask_a[j]) add_b[j]=ACC'($signed(weight_a)); end
          else if(mask_b[j]) add_b[j]=ACC'($signed(weight_b));
        end else if(selected_mode==1) begin
          if(mask_a[j]) add_b[j]=ACC'($signed(weight_a));
          else if(mask_b[j]) add_b[j]=ACC'($signed(weight_b));
        end else begin
          if(mask_a[j] && mask_b[j]) add_b[j]=pair_sum;
          else if(mask_a[j]) add_b[j]=ACC'($signed(weight_a));
          else if(mask_b[j]) add_b[j]=ACC'($signed(weight_b));
        end
      end else if(state==ISSUE_SECOND) begin
        if(mask_b[j] && (selected_mode==0 || mask_a[j]))
          add_b[j]=ACC'($signed(weight_b));
      end
    end
    // During coefficient arrival, the first existing accumulator adder is idle.
    // Reuse it for the two-source product; no twenty-first adder is assumed.
    if(state==RESPONSE && selected_mode==2) begin
      add_a[0]=ACC'($signed(weight_data[15:0]));
      add_b[0]=ACC'($signed(weight_data[31:16]));
    end
    for(integer j=0;j<P;j=j+1) add_y[j]=add_a[j]+add_b[j];
  end
  always_ff @(posedge clk) begin
    if(!reset_n) begin
      state<=IDLE;cycles<=0;weight_words<=0;replay_cycles<=0;issue_cycles<=0;
      source_index<=0;pair_index<=0;output_index<=0;write_index<=0;
      selected_mode<=0;mask_a<=0;mask_b<=0;weight_a<=0;weight_b<=0;pair_sum<=0;
      for(lane=0;lane<P;lane=lane+1) accum[lane]<=0;
    end else begin
      if(map_we) map_mem[map_addr]<=map_data;
      if(state!=IDLE && state!=FINISH) cycles<=cycles+1;
      case(state)
        IDLE: if(start) begin state<=LOAD;selected_mode<=mode;cycles<=0;
          weight_words<=0;replay_cycles<=0;issue_cycles<=0;source_index<=0;
          output_index<=0;pair_index<=0;end
        LOAD: if(source_valid) begin source_mem[source_index]<=source_data;
          if(source_index==K-1) state<=CLEAR; else source_index<=source_index+1;end
        CLEAR: begin
          for(lane=0;lane<P;lane=lane+1) accum[lane]<=0;
          pair_index<=0;state<=GET_MASK;
        end
        GET_MASK: begin
          mask_a<=source_mem[map_mem[pair_index][KW-1:0]];
          mask_b<=source_mem[map_mem[pair_index][2*KW-1:KW]];
          state<=CHECK_MASK;
        end
        CHECK_MASK: state<=((mask_a|mask_b)=='0)?NEXT_PAIR:REQUEST;
        REQUEST: if(weight_ready) begin state<=RESPONSE;weight_words<=weight_words+1;end
        RESPONSE: if(weight_valid) begin
          weight_a<=$signed(weight_data[15:0]);weight_b<=$signed(weight_data[31:16]);
          if(selected_mode==2) pair_sum<=add_y[0];state<=ISSUE_FIRST;
        end
        ISSUE_FIRST: begin
          for(lane=0;lane<P;lane=lane+1) accum[lane]<=add_y[lane];
          issue_cycles<=issue_cycles+1;
          if((selected_mode==0 && (|mask_a) && (|mask_b)) ||
             (selected_mode==1 && (|(mask_a&mask_b)))) state<=ISSUE_SECOND;
          else state<=NEXT_PAIR;
        end
        ISSUE_SECOND: begin
          for(lane=0;lane<P;lane=lane+1) accum[lane]<=add_y[lane];
          replay_cycles<=replay_cycles+1;issue_cycles<=issue_cycles+1;state<=NEXT_PAIR;
        end
        NEXT_PAIR: if(pair_index==PAIRS-1) begin write_index<=0;state<=WRITE_RESULT;end
          else begin pair_index<=pair_index+1;state<=GET_MASK;end
        WRITE_RESULT: if(result_ready) begin
          if(write_index==P-1) begin
            if(output_index==O-1) state<=FINISH;
            else begin output_index<=output_index+1;state<=CLEAR;end
          end else write_index<=write_index+1;
        end
        FINISH: if(!start) state<=IDLE;
        default: state<=IDLE;
      endcase
    end
  end
endmodule
