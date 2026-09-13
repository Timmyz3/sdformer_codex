// Isolated 96x24 post-factor. Shared 8x signed(32*16+48) MAC datapath.
// Flop/mux local arrays: no SRAM/PPA implementation claim. One 128b CR word.
module ped_kron(
 input logic clk, rst,
 input logic [1:0] mode,
 input logic cfg_valid, input logic [8:0] cfg_addr,
 input logic [127:0] cfg_data,
 input logic bias_valid,input logic [6:0] bias_addr,input logic signed [23:0] bias_data,
 input logic start,
 input logic in_valid,output logic in_ready,input logic [191:0] in_data,
 output logic out_valid,input logic out_ready,output logic [191:0] out_data,
 output logic done,output logic [2:0] debug_state,
 output logic debug_phase
);
 localparam IDLE=0,LOAD=1,CLEAR=2,MAC=3,COMMIT=4,OUTPUT=5;
 logic [2:0] state;
 logic [127:0] coeff [0:287];
 logic signed [23:0] bias [0:95];
 logic signed [31:0] src [0:23];
 logic signed [31:0] latent [0:15];
 logic signed [47:0] result [0:95];
 logic signed [47:0] acc [0:7];
 logic signed [31:0] mul_x [0:7];
 logic signed [15:0] mul_w [0:7];
 wire signed [47:0] product [0:7];
 logic [127:0] cw;
 integer group_idx,k,term_idx,beat;
 logic [8:0] word_addr;
 logic phase;
 logic [1:0] active_mode;
 genvar lane;
 generate for(lane=0;lane<8;lane=lane+1)begin: mac_lane
   assign product[lane]=mul_x[lane]*mul_w[lane];
 end endgenerate
 function automatic signed [23:0] finish(input signed [47:0] x,input signed [23:0] b);
   logic signed [47:0] q,t;
   begin
    q=x>>>15;
    if ((x[14:0]>15'd16384)||((x[14:0]==15'd16384)&&q[0]))q=q+1;
    if(q>48'sd8388607)q=48'sd8388607;
    if(q< -48'sd8388608)q= -48'sd8388608;
    t=q+$signed({{24{b[23]}},b});
    if(t>48'sd8388607)t=48'sd8388607;
    if(t< -48'sd8388608)t= -48'sd8388608;
    finish=t[23:0];
   end
 endfunction
 always_comb begin
   word_addr=0;
   if(active_mode==0)word_addr=9'(group_idx*24+k);
   else if(!phase)word_addr=9'(term_idx*18+k);
   else word_addr=9'(term_idx*18+6+(group_idx%3)*4+k);
   cw=coeff[word_addr];
   for(integer i=0;i<8;i=i+1)begin
     mul_x[i]='0;mul_w[i]='0;
     if(state==MAC)begin
       if(active_mode==0)begin mul_x[i]=src[k];mul_w[i]=$signed(cw[i*16+:16]);end
       else if(!phase)begin
         mul_x[i]=src[k*4+(i%4)];
         mul_w[i]=$signed(cw[(group_idx*2+i/4)*16+:16]);
       end else begin
         mul_x[i]=latent[(group_idx/3)*4+k];mul_w[i]=$signed(cw[i*16+:16]);
       end
     end
   end
   in_ready=(state==LOAD);out_valid=(state==OUTPUT);out_data='0;
   if(state==OUTPUT)for(integer i=0;i<8;i=i+1)
      out_data[i*24+:24]=finish(result[group_idx*8+i],bias[group_idx*8+i]);
   debug_state=state;debug_phase=phase;
 end
 always_ff @(posedge clk)begin
   if(rst)begin
     state<=IDLE;done<=0;group_idx<=0;k<=0;term_idx<=0;beat<=0;phase<=0;active_mode<=0;
   end else begin
     done<=0;
     case(state)
       IDLE:begin
         if(cfg_valid)coeff[cfg_addr]<=cfg_data;
         if(bias_valid)bias[bias_addr]<=bias_data;
         if(start)begin state<=LOAD;beat<=0;active_mode<=mode;end
       end
       LOAD:if(in_valid)begin
         for(integer i=0;i<8;i=i+1)src[beat*8+i]<=$signed({{8{in_data[i*24+23]}},in_data[i*24+:24]});
         if(beat==2)begin group_idx<=0;k<=0;term_idx<=0;phase<=0;state<=CLEAR;end
         else beat<=beat+1;
       end
       CLEAR:begin
         for(integer i=0;i<8;i=i+1)
           if(active_mode!=0 && phase && term_idx!=0)acc[i]<=result[group_idx*8+i];
           else acc[i]<=0;
         k<=0;state<=MAC;
       end
       MAC:begin
         for(integer i=0;i<8;i=i+1)acc[i]<=acc[i]+product[i];
         if((active_mode==0 && k==23)||(active_mode!=0 && !phase && k==5)||(active_mode!=0 && phase && k==3))state<=COMMIT;
         else k<=k+1;
       end
       COMMIT:begin
         for(integer i=0;i<8;i=i+1)
           if(active_mode!=0 && !phase)latent[group_idx*8+i]<=acc[i][31:0];
           else result[group_idx*8+i]<=acc[i];
         if(active_mode!=0 && !phase && group_idx==1)begin phase<=1;group_idx<=0;state<=CLEAR;end
         else if((active_mode==0 || phase) && group_idx==11)begin
           group_idx<=0;
           if(active_mode==2 && term_idx==0)begin term_idx<=1;phase<=0;state<=CLEAR;end
           else state<=OUTPUT;
         end else begin group_idx<=group_idx+1;state<=CLEAR;end
       end
       OUTPUT:if(out_ready)begin
         if(group_idx==11)begin state<=IDLE;group_idx<=0;done<=1;end
         else group_idx<=group_idx+1;
       end
       default:state<=IDLE;
     endcase
   end
 end
endmodule
