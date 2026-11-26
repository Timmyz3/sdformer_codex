module count_tree (
 input logic clk, rst_n,
 input logic cfg_we, input logic [3:0] cfg_addr, input logic [31:0] cfg_data,
 output logic cfg_ready,
 input logic frame_start, output logic frame_ready,
 input logic in_valid, output logic in_ready,
 input logic [511:0] in_data, input logic in_last,
 output logic out_valid, input logic out_ready,
 output logic [2:0] out_action, output logic error,
 input logic [4:0] debug_count_index, output logic [31:0] debug_count
);
 typedef enum logic [4:0] {IDLE,CLEAR,WAIT_INPUT,CLASSIFY,READ_P,POP_P,ADD_P,WRITE_P,
  READ_N,POP_N,ADD_N,WRITE_N,ADVANCE,ROOT,N0,N1,N2,N3,NCOUNT,NCMP,NBRANCH,EMIT} state_t;
 state_t state;
 logic [31:0] rf[0:47];
 logic [4:0] clear_index, feature_index;
 logic [1:0] cell_x,cell_y,node_index;
 logic [3:0] sub_x;
 logic [6:0] sub_y;
 logic [17:0] beat_count;
 logic [511:0] data_latch;
 logic last_latch;
 logic [15:0] positive,negative;
 logic [31:0] operand,total,threshold,left_node,right_node;
 logic [4:0] population;
 logic compare_result;
 logic [1:0] depth;
 wire [4:0] positive_index={cell_y,cell_x,1'b0};
 wire [31:0] chosen=compare_result?left_node:right_node;
 function automatic logic [4:0] pop16(input logic [15:0] bits);
  logic [4:0] s;
  begin s=0;for(integer j=0;j<16;j=j+1)s=s+{4'b0,bits[j]};pop16=s;end
 endfunction
 assign cfg_ready=state==IDLE;
 assign frame_ready=state==IDLE&&!cfg_we;
 assign in_ready=state==WAIT_INPUT;
 assign out_valid=state==EMIT;
 // Validation-only combinational visibility, also synthesized and charged.
 logic [5:0] rf_read_address;
 logic [31:0] rf_read_data;
 always_comb begin
  case(state)
   IDLE:rf_read_address={1'b0,debug_count_index};
   READ_P:rf_read_address={1'b0,positive_index};
   READ_N:rf_read_address={1'b0,positive_index}+6'd1;
   ROOT:rf_read_address=6'd44;
   N0:rf_read_address=6'd32+{2'b00,node_index,2'b00};
   N1:rf_read_address=6'd33+{2'b00,node_index,2'b00};
   N2:rf_read_address=6'd34+{2'b00,node_index,2'b00};
   N3:rf_read_address=6'd35+{2'b00,node_index,2'b00};
   NCOUNT:rf_read_address={1'b0,feature_index};
   default:rf_read_address=0;
  endcase
 end
 assign rf_read_data=rf[rf_read_address];
 assign debug_count=(state==IDLE&&!cfg_we&&!frame_start)?rf_read_data:32'b0;
 always_ff @(posedge clk) begin
  if(!rst_n)begin
   state<=IDLE;out_action<=0;error<=0;clear_index<=0;cell_x<=0;cell_y<=0;sub_x<=0;sub_y<=0;
   beat_count<=0;data_latch<=0;last_latch<=0;positive<=0;negative<=0;operand<=0;total<=0;
   population<=0;node_index<=0;feature_index<=0;threshold<=0;left_node<=0;right_node<=0;
   compare_result<=0;depth<=0;
  end else begin
   case(state)
    IDLE:begin
     if(cfg_we)rf[6'd32+{2'b00,cfg_addr}]<=cfg_data;
     else if(frame_start)begin rf[0]<=0;state<=CLEAR;clear_index<=1;cell_x<=0;cell_y<=0;sub_x<=0;sub_y<=0;beat_count<=0;error<=0;end
    end
    CLEAR:begin rf[{1'b0,clear_index}]<=0;if(clear_index==31)state<=WAIT_INPUT;else clear_index<=clear_index+1'b1;end
    WAIT_INPUT:if(in_valid)begin
     if(in_last!=(beat_count==191999))begin error<=1;out_action<=7;state<=EMIT;end
     else begin data_latch<=in_data;last_latch<=in_last;state<=CLASSIFY;end
    end
    CLASSIFY:begin
     for(integer j=0;j<16;j=j+1)begin
      positive[j]<=(|data_latch[32*j+:31])&&!data_latch[32*j+31];
      negative[j]<=(|data_latch[32*j+:31])&&data_latch[32*j+31];
     end
     state<=READ_P;
    end
    READ_P:begin operand<=rf_read_data;state<=POP_P;end
    POP_P:begin population<=pop16(positive);state<=ADD_P;end
    ADD_P:begin total<=operand+{27'b0,population};state<=WRITE_P;end
    WRITE_P:begin rf[{1'b0,positive_index}]<=total;state<=READ_N;end
    READ_N:begin operand<=rf_read_data;state<=POP_N;end
    POP_N:begin population<=pop16(negative);state<=ADD_N;end
    ADD_N:begin total<=operand+{27'b0,population};state<=WRITE_N;end
    WRITE_N:begin rf[{1'b0,positive_index}+6'd1]<=total;state<=ADVANCE;end
    ADVANCE:begin
     beat_count<=beat_count+1'b1;
     if(sub_x==9)begin
      sub_x<=0;
      if(cell_x==3)begin cell_x<=0;if(sub_y==119)begin sub_y<=0;cell_y<=cell_y+1'b1;end else sub_y<=sub_y+1'b1;end
      else cell_x<=cell_x+1'b1;
     end else sub_x<=sub_x+1'b1;
     if(last_latch)begin state<=ROOT;depth<=0;end else state<=WAIT_INPUT;
    end
    ROOT:begin
     if(rf_read_data[31])begin out_action<=rf_read_data[2:0];state<=EMIT;end
     else begin node_index<=rf_read_data[1:0];state<=N0;end
    end
    N0:begin feature_index<=rf_read_data[4:0];state<=N1;end
    N1:begin threshold<=rf_read_data;state<=N2;end
    N2:begin left_node<=rf_read_data;state<=N3;end
    N3:begin right_node<=rf_read_data;state<=NCOUNT;end
    NCOUNT:begin operand<=rf_read_data;state<=NCMP;end
    NCMP:begin compare_result<=operand<=threshold;state<=NBRANCH;end
    NBRANCH:begin
     depth<=depth+1'b1;
     if(chosen[31])begin
      if(chosen[30:3]!=0||chosen[2:0]>6)begin error<=1;out_action<=7;end else out_action<=chosen[2:0];
      state<=EMIT;
     end
     else if(depth==1)begin error<=1;out_action<=7;state<=EMIT;end
     else begin node_index<=chosen[1:0];state<=N0;end
    end
    EMIT:if(out_ready)state<=IDLE;
    default:state<=IDLE;
   endcase
  end
 end
endmodule
