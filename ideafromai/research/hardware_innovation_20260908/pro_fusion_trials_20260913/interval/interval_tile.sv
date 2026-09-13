// Bounded complete K864,T10,N16,P2 operator. No inferred extra prefix adders.
// Eight explicit 32-bit add/sub carry chains are time-multiplexed for AC,
// temporal prefix and RNE. Both modes reserve the same dual-read state array.
module interval_tile(
 input logic clk, rst, start,
 input logic [1:0] mode, // 0 direct, 1 endpoint, 2 paid full-tile count/select
 output logic source_req,
 output logic [9:0] source_k,
 input logic source_valid,
 input logic [19:0] source_mask,
 output logic weight_req,
 output logic [9:0] weight_k,
 input logic weight_valid,
 input logic [255:0] weight_data,
 output logic out_valid,
 input logic out_ready,
 output logic [5:0] out_index,
 output logic [191:0] out_data,
 output logic done, busy, selected_endpoint,
 output integer cycles, clear_cycles, source_reads, source_stalls,
 output integer weight_reads, weight_stalls, event_add_cycles,
 output integer prefix_cycles, round_cycles, output_beats, output_stalls,
 output integer selection_cycles
);
 typedef enum logic [3:0] {IDLE, CLEAR, STATS, SELECT, SCAN, WEIGHT,
                          APPLY, PREFIX, ROUND, SEND, FINISH} state_t;
 state_t state;
 logic signed [31:0] accum [0:39][0:7];
 logic [255:0] wreg;
 logic [19:0] active_mask, negative_mask;
 logic [19:0] delta_mask, delta_neg;
 logic [1:0] mode_reg;
 integer k_index, group_index, hg, event_index;
 integer count_spikes, count_endpoints;
 logic found;
 logic signed [31:0] add_a[0:7], add_b[0:7], add_y[0:7];
 logic [31:0] xor_b[0:7];
 logic subtract;

 always_comb begin
   delta_mask='0; delta_neg='0;
   for (integer pp=0;pp<2;pp=pp+1) begin
     delta_mask[pp*10]=source_mask[pp*10];
     for (integer tt=1;tt<10;tt=tt+1) begin
       delta_mask[pp*10+tt]=source_mask[pp*10+tt]^source_mask[pp*10+tt-1];
       delta_neg[pp*10+tt]=!source_mask[pp*10+tt] && source_mask[pp*10+tt-1];
     end
   end
   event_index=0; found=0;
   for (integer jj=0;jj<20;jj=jj+1) begin
     if(active_mask[jj] && !found) begin event_index=jj; found=1; end
   end
   source_req=(state==SCAN || state==STATS);
   source_k=10'(k_index);
   weight_req=(state==WEIGHT); weight_k=10'(k_index);
   out_valid=(state==SEND); out_index=6'(group_index);
   subtract=(state==APPLY && negative_mask[event_index]);
   for(integer ll=0;ll<8;ll=ll+1) begin
     add_a[ll]=0; add_b[ll]=0;
     if(state==APPLY) begin
       add_a[ll]=accum[event_index*2+hg][ll];
       add_b[ll]={{16{wreg[(hg*8+ll)*16+15]}},wreg[(hg*8+ll)*16+:16]};
     end else if(state==PREFIX) begin
       add_a[ll]=accum[group_index][ll]; add_b[ll]=accum[group_index-2][ll];
     end else if(state==ROUND) begin
       add_a[ll]=accum[group_index][ll] >>> 3;
       add_b[ll]=32'((accum[group_index][ll][2:0]>3'd4) ||
                    (accum[group_index][ll][2:0]==3'd4 && accum[group_index][ll][3]));
     end
   end
 end
 // Exactly one explicit carry chain per lane; add/sub selection uses XOR/carry-in.
 genvar lane,bitno;
 generate for(lane=0;lane<8;lane=lane+1) begin: ADDER
   assign xor_b[lane]=add_b[lane]^{32{subtract}};
   for(bitno=0;bitno<32;bitno=bitno+1) begin: BIT
     wire carry_in, carry_out;
     if(bitno==0) assign carry_in=subtract;
     else assign carry_in=BIT[bitno-1].carry_out;
     assign add_y[lane][bitno]=add_a[lane][bitno]^xor_b[lane][bitno]^carry_in;
     assign carry_out=(add_a[lane][bitno]&xor_b[lane][bitno]) |
       ((add_a[lane][bitno]^xor_b[lane][bitno])&carry_in);
   end
 end endgenerate

 always_ff @(posedge clk) begin
   if(rst) begin
     state<=IDLE; done<=0; busy<=0; selected_endpoint<=0;
     mode_reg<=0; k_index<=0; group_index<=0; hg<=0;
     wreg<=0; active_mask<=0; negative_mask<=0; out_data<=0;
     cycles<=0; clear_cycles<=0; source_reads<=0; source_stalls<=0;
     weight_reads<=0; weight_stalls<=0; event_add_cycles<=0;
     prefix_cycles<=0; round_cycles<=0; output_beats<=0; output_stalls<=0;
     selection_cycles<=0; count_spikes<=0; count_endpoints<=0;
     // Accumulators deliberately not reset here: each command pays CLEAR.
   end else begin
     done<=0;
     if(busy) cycles<=cycles+1;
     case(state)
       IDLE: if(start) begin
         state<=CLEAR; busy<=1; mode_reg<=mode; selected_endpoint<=(mode==1);
         k_index<=0; group_index<=0; hg<=0; count_spikes<=0; count_endpoints<=0;
         active_mask<=0; negative_mask<=0;
         cycles<=0; clear_cycles<=0; source_reads<=0; source_stalls<=0;
         weight_reads<=0; weight_stalls<=0; event_add_cycles<=0;
         prefix_cycles<=0; round_cycles<=0; output_beats<=0; output_stalls<=0;
         selection_cycles<=0;
       end
       CLEAR: begin
         for(integer ll=0;ll<8;ll=ll+1) accum[group_index][ll]<=0;
         clear_cycles<=clear_cycles+1;
         if(group_index==39) begin group_index<=0; state<=(mode_reg==2)?STATS:SCAN; end
         else group_index<=group_index+1;
       end
       STATS: begin
         selection_cycles<=selection_cycles+1;
         if(source_valid) begin
           source_reads<=source_reads+1;
           count_spikes<=count_spikes+32'($countones(source_mask));
           count_endpoints<=count_endpoints+32'($countones(delta_mask));
           if(k_index==863) begin state<=SELECT; k_index<=0; end
           else k_index<=k_index+1;
         end else source_stalls<=source_stalls+1;
       end
       SELECT: begin
         // The same two vector groups per event; 36 extra prefix vector adds.
         selected_endpoint<=(count_endpoints*2+36 < count_spikes*2);
         selection_cycles<=selection_cycles+1; state<=SCAN;
       end
       SCAN: begin
         if(source_valid) begin
           source_reads<=source_reads+1;
           active_mask<=selected_endpoint?delta_mask:source_mask;
           negative_mask<=selected_endpoint?delta_neg:20'd0;
           if(source_mask!=0) state<=WEIGHT;
           else if(k_index==863) begin
             group_index<=selected_endpoint?2:0;
             state<=selected_endpoint?PREFIX:ROUND;
           end else k_index<=k_index+1;
         end else source_stalls<=source_stalls+1;
       end
       WEIGHT: begin
         if(weight_valid) begin wreg<=weight_data; weight_reads<=weight_reads+1; hg<=0; state<=APPLY; end
         else weight_stalls<=weight_stalls+1;
       end
       APPLY: begin
         for(integer ll=0;ll<8;ll=ll+1) accum[event_index*2+hg][ll]<=add_y[ll];
         event_add_cycles<=event_add_cycles+1;
         if(hg==0) hg<=1;
         else begin
           hg<=0; active_mask[event_index]<=0;
           if((active_mask & (active_mask-20'd1))==0) begin
             if(k_index==863) begin
               group_index<=selected_endpoint?2:0;
               state<=selected_endpoint?PREFIX:ROUND;
             end else begin k_index<=k_index+1; state<=SCAN; end
           end
         end
       end
       PREFIX: begin
         for(integer ll=0;ll<8;ll=ll+1) accum[group_index][ll]<=add_y[ll];
         prefix_cycles<=prefix_cycles+1;
         if(group_index==39) begin group_index<=0; state<=ROUND; end
         else if(group_index==19) group_index<=22; // P1,T0 is already delta=Z0.
         else group_index<=group_index+1;
       end
       ROUND: begin
         for(integer ll=0;ll<8;ll=ll+1) begin
           if(add_y[ll]>32'sd8388607) out_data[ll*24+:24]<=24'h7fffff;
           else if(add_y[ll]< -32'sd8388608) out_data[ll*24+:24]<=24'h800000;
           else out_data[ll*24+:24]<=add_y[ll][23:0];
         end
         round_cycles<=round_cycles+1; state<=SEND;
       end
       SEND: begin
         if(out_ready) begin
           output_beats<=output_beats+1;
           if(group_index==39) state<=FINISH;
           else begin group_index<=group_index+1; state<=ROUND; end
         end else output_stalls<=output_stalls+1;
       end
       FINISH: begin state<=IDLE; busy<=0; done<=1; end
       default: state<=IDLE;
     endcase
   end
 end
endmodule
