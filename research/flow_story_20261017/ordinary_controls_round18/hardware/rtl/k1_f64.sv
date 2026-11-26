// Exact finite IEEE-754 binary64 ordering; no FP arithmetic/score generator.
// 12 inputs in fixed row-major order: 3 rows x 4 full tiles, grid stride 5.
module k1_f64(
 input logic clk, rst_n,
 input logic start_valid, output logic start_ready,
 input logic score_valid, output logic score_ready,
 input logic [63:0] score_bits,
 output logic result_valid, input logic result_ready,
 output logic [3:0] winner,
 output logic [19:0] fine_mask,
 output logic fault
);
 typedef enum logic [1:0] {IDLE, READ_SCORES, RESULT} state_t;
 state_t state;
 logic [3:0] count;
 logic [63:0] best;
 logic [4:0] mask_index;
 function automatic logic greater_finite(input logic [63:0] a,b);
   if (a[62:0]==0 && b[62:0]==0) greater_finite=0;
   else if (a[63]!=b[63]) greater_finite=~a[63];
   else if (a[63]) greater_finite=(a[62:0]<b[62:0]);
   else greater_finite=(a[62:0]>b[62:0]);
 endfunction
 assign start_ready=(state==IDLE);
 assign score_ready=(state==READ_SCORES);
 assign result_valid=(state==RESULT);
 assign mask_index={1'b0,winner}+{3'b0,winner[3:2]};
 assign fine_mask=(fault || state!=RESULT)?20'b0:(20'b1<<mask_index);
 always_ff @(posedge clk or negedge rst_n) begin
   if(!rst_n) begin
     state<=IDLE;count<=0;best<=0;winner<=0;fault<=0;
   end else begin
     case(state)
       IDLE: if(start_valid) begin
         state<=READ_SCORES;count<=0;best<=0;winner<=0;fault<=0;
       end
       READ_SCORES: if(score_valid) begin
         if(score_bits[62:52]==11'h7ff) fault<=1;
         if(count==0 || greater_finite(score_bits,best)) begin
           best<=score_bits;winner<=count;
         end
         if(count==11) state<=RESULT;
         else count<=count+1'b1;
       end
       RESULT: if(result_ready) state<=IDLE;
       default: begin state<=IDLE;fault<=1;end
     endcase
   end
 end
endmodule
