// Representative signed int8/int32 core, NOT an accuracy-validated SNN format.
// One outstanding memory request. Response computes one vector MAC. No PE/port
// can retire another dot until its output write is accepted. Arithmetic wraps
// modulo 2^32 (no saturation). Partial output lanes are masked, not padded dots.
module dot_residual #(parameter integer LANES=4) (
 input logic clk,rst,
 input logic cmd_valid, output logic cmd_ready,
 input logic cmd_keep,
 input logic [15:0] cmd_count,cmd_tag,
 input logic [LANES-1:0] cmd_lane_mask,
 input logic [LANES*32-1:0] cmd_bias,cmd_residual,
 output logic read_valid,input logic read_ready,
 output logic [15:0] read_index,
 input logic response_valid,output logic response_ready,
 input logic signed [7:0] response_scalar,
 input logic [LANES*8-1:0] response_vector,
 output logic write_valid,input logic write_ready,
 output logic [15:0] write_tag,
 output logic [LANES-1:0] write_lane_mask,
 output logic [LANES*32-1:0] write_data
);
 typedef enum logic[1:0] {IDLE,REQUEST,RESPONSE,WRITEBACK} state_t;
 state_t state;
 logic [15:0] count_q,index_q,tag_q;
 logic [LANES-1:0] mask_q;
 logic signed [31:0] accum[LANES];
 logic signed [31:0] residual[LANES];
 logic signed [15:0] product[LANES];
 integer i;
 always_comb begin
   cmd_ready=(state==IDLE)&&!rst;
   read_valid=(state==REQUEST)&&!rst;
   response_ready=(state==RESPONSE)&&!rst;
   read_index=index_q;
   write_valid=(state==WRITEBACK)&&!rst;
   write_tag=tag_q;
   write_lane_mask=mask_q;
   for(integer j=0;j<LANES;j=j+1) begin
     product[j]=response_scalar*$signed(response_vector[j*8+:8]);
     write_data[j*32+:32]=mask_q[j] ? accum[j]+residual[j] : 32'b0;
   end
 end
 always_ff @(posedge clk) begin
   if(rst) begin
     state<=IDLE;count_q<=0;index_q<=0;tag_q<=0;mask_q<='0;
     for(i=0;i<LANES;i=i+1) begin accum[i]<=0;residual[i]<=0;end
   end else case(state)
     IDLE: if(cmd_valid&&cmd_ready) begin
       count_q<=cmd_count;index_q<=0;tag_q<=cmd_tag;mask_q<=cmd_lane_mask;
       for(i=0;i<LANES;i=i+1) begin
         accum[i]<=cmd_keep?$signed(cmd_bias[i*32+:32]):32'sd0;
         residual[i]<=$signed(cmd_residual[i*32+:32]);
       end
       state<=cmd_keep&&cmd_count!=0?REQUEST:WRITEBACK;
     end
     REQUEST: if(read_valid&&read_ready) state<=RESPONSE;
     RESPONSE: if(response_valid&&response_ready) begin
       for(i=0;i<LANES;i=i+1) if(mask_q[i]) accum[i]<=accum[i]+{{16{product[i][15]}},product[i]};
       if(index_q+16'd1==count_q) state<=WRITEBACK;
       else begin index_q<=index_q+16'd1;state<=REQUEST;end
     end
     WRITEBACK: if(write_valid&&write_ready) state<=IDLE;
     default: state<=IDLE;
   endcase
 end
endmodule
