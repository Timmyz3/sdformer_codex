`timescale 1ns/1ps
`default_nettype none
module c2_ftp_shared (
 input logic clk_core, rst_core, share_enable,
 input logic start_valid, input logic [23:0] start_tag, output logic start_ready,
 input logic source_valid, input logic [8:0] source_index,
 input logic [9:0] source_mask, output logic source_ready,
 output logic [7:0] mem_req_valid, input logic [7:0] mem_req_ready,
 output logic [15:0] mem_req_epoch [0:7],
 output logic [2:0] mem_req_slot [0:7],
 output logic [31:0] mem_req_generation [0:7],
 output logic [23:0] mem_req_tag [0:7],
 output logic [2:0] mem_req_output_block [0:7],
 output logic [2:0] mem_req_slice [0:7],
 output logic [11:0] mem_req_source_channel [0:7],
 output logic [7:0] mem_req_accept,
 input logic [7:0] mem_rsp_valid, output logic [7:0] mem_rsp_ready,
 input logic [15:0] mem_rsp_epoch [0:7],
 input logic [2:0] mem_rsp_slot [0:7],
 input logic [31:0] mem_rsp_generation [0:7],
 input logic [23:0] mem_rsp_tag [0:7],
 input logic signed [7:0] mem_rsp_weight [0:7][0:15],
 output logic [7:0] mem_rsp_accept,
 input logic bridge_ready,
 output logic commit_valid, input logic commit_ready,
 output logic [3:0] commit_context, output logic [23:0] commit_tag,
 output logic [2:0] commit_slice,
 output logic signed [23:0] commit_accumulator [0:15],
 output logic commit_terminal, commit_accept,
 output logic bundle_done_valid, input logic bundle_done_ready,
 output logic protocol_error, numeric_overflow,
 output logic [31:0] debug_additions, debug_source_loads
);
 localparam logic [9:0] MODE = 10'h140;
 typedef enum logic [3:0] {IDLE, LOAD, SCAN, REQUEST, RESPONSE, APPLY,
                          SCATTER, NEXT_SLICE, OUTPUTS, DONE} state_t;
 state_t state_q;
 logic [23:0] tag_q;
 logic share_q;
 logic [8:0] expected_source_q, channel_q;
 logic [2:0] slice_q, out_slice_q;
 logic [3:0] out_time_q;
 logic [9:0] masks_q [0:383];
 logic [5:0] remaining_q [0:7];
 logic live_q [0:7];
 logic signed [23:0] partial_q [0:7][0:5][0:15];
 logic signed [23:0] y_q [0:9][0:5][0:15];
 logic y_live_q [0:9][0:5];
 logic signed [23:0] weight_q [0:15], work_q [0:15];
 logic [9:0] scatter_q;
 logic [2:0] return_bank, physical_bank;
 logic selected;
 logic [31:0] serial_q;
 logic [3:0] next_time;
 logic next_time_valid;
 assign return_bank=channel_q[2:0];
 // Captured layer base/row_bytes=64 and output tile is zero: group modulo8.
 assign physical_bank=channel_q[6:4];
 assign selected=share_q && masks_q[channel_q]==MODE;
 assign start_ready=state_q==IDLE && !protocol_error && !numeric_overflow;
 assign source_ready=state_q==LOAD && !protocol_error && !numeric_overflow;
 assign mem_req_accept=mem_req_valid & mem_req_ready;
 assign mem_rsp_accept=mem_rsp_valid & mem_rsp_ready;
 assign commit_valid=state_q==OUTPUTS && !protocol_error && !numeric_overflow;
 assign commit_accept=commit_valid && commit_ready;
 assign commit_context=out_time_q;
 assign commit_slice=out_slice_q;
 assign commit_tag=tag_q;
 assign commit_terminal=out_slice_q==5;
 assign bundle_done_valid=state_q==DONE && !protocol_error && !numeric_overflow;
 always_comb begin
  mem_req_valid=0; mem_rsp_ready=0;
  for (int b=0;b<8;b++) begin
   mem_req_epoch[b]=0; mem_req_slot[b]=serial_q[2:0];
   mem_req_generation[b]=serial_q; mem_req_tag[b]=tag_q;
   mem_req_output_block[b]={2'b0,channel_q[3]};
   mem_req_slice[b]=slice_q; mem_req_source_channel[b]={3'b0,channel_q};
  end
  if (!protocol_error && !numeric_overflow) begin
   if (state_q==REQUEST) mem_req_valid[return_bank]=1;
   if (state_q==RESPONSE) mem_rsp_ready[return_bank]=1;
  end
  for (int l=0;l<16;l++)
   commit_accumulator[l]=y_live_q[out_time_q][out_slice_q]
     ? y_q[out_time_q][out_slice_q][l] : 24'sd0;
  next_time=0; next_time_valid=0;
  for (int t=0;t<10;t++)
   if (scatter_q[t] && !next_time_valid) begin
    next_time=4'(t); next_time_valid=1;
   end
 end
 always_ff @(posedge clk_core) begin
  if (rst_core) begin
   state_q<=IDLE; tag_q<=0; share_q<=0; expected_source_q<=0;
   channel_q<=0; slice_q<=0; out_time_q<=0; out_slice_q<=0;
   scatter_q<=0; serial_q<=0; protocol_error<=0; numeric_overflow<=0;
   debug_additions<=0; debug_source_loads<=0;
   for (int b=0;b<8;b++) begin remaining_q[b]<=0;live_q[b]<=0;end
   for (int t=0;t<10;t++) for(int s=0;s<6;s++) y_live_q[t][s]<=0;
  end else if (!protocol_error && !numeric_overflow) begin
   case(state_q)
    IDLE: if(start_valid && start_ready) begin
     tag_q<=start_tag; share_q<=share_enable; expected_source_q<=0;
     channel_q<=0; slice_q<=0; out_time_q<=0;out_slice_q<=0;
     state_q<=LOAD;
     for(int b=0;b<8;b++) begin remaining_q[b]<=0;live_q[b]<=0;end
     for(int t=0;t<10;t++) for(int s=0;s<6;s++) y_live_q[t][s]<=0;
    end
    LOAD: if(source_valid && source_ready) begin
     if(source_index!=expected_source_q || source_index>=384)
      protocol_error<=1;
     else begin
      masks_q[source_index]<=source_mask;
      debug_source_loads<=debug_source_loads+1;
      if(share_q && source_mask==MODE) begin
       if(remaining_q[source_index[6:4]]==48) protocol_error<=1;
       else remaining_q[source_index[6:4]]<=remaining_q[source_index[6:4]]+1;
      end
      if(source_index==383) state_q<=SCAN;
      else expected_source_q<=expected_source_q+1;
     end
    end
    SCAN: begin
     slice_q<=0;
     if(masks_q[channel_q]!=0) state_q<=REQUEST;
     else if(channel_q==383) state_q<=OUTPUTS;
     else channel_q<=channel_q+1;
    end
    REQUEST: if(mem_req_accept[return_bank]) state_q<=RESPONSE;
    RESPONSE: if(mem_rsp_accept[return_bank]) begin
     if(mem_rsp_epoch[return_bank]!=0 || mem_rsp_slot[return_bank]!=serial_q[2:0]
       || mem_rsp_generation[return_bank]!=serial_q
       || mem_rsp_tag[return_bank]!=tag_q) protocol_error<=1;
     for(int l=0;l<16;l++) weight_q[l]<={{16{mem_rsp_weight[return_bank][l][7]}},mem_rsp_weight[return_bank][l]};
     serial_q<=serial_q+1;
     state_q<=APPLY;
    end
    APPLY: if(bridge_ready) begin
     for(int l=0;l<16;l++) begin
      logic signed [24:0] sum;
      sum={weight_q[l][23],weight_q[l]};
      if(selected && live_q[physical_bank])
       sum=sum+{partial_q[physical_bank][slice_q][l][23],partial_q[physical_bank][slice_q][l]};
      if(sum>25'sd8388607 || sum< -25'sd8388608) numeric_overflow<=1;
      work_q[l]<=sum[23:0];
      if(selected) partial_q[physical_bank][slice_q][l]<=sum[23:0];
     end
     if(selected && live_q[physical_bank]) debug_additions<=debug_additions+16;
     if(selected && remaining_q[physical_bank]==0) protocol_error<=1;
     if(selected && remaining_q[physical_bank]>1) state_q<=NEXT_SLICE;
     else begin scatter_q<=masks_q[channel_q]; state_q<=SCATTER;end
    end
    SCATTER: if(bridge_ready) begin
     if(!next_time_valid) protocol_error<=1;
     for(int l=0;l<16;l++) begin
      logic signed [24:0] sum;
      sum={work_q[l][23],work_q[l]};
      if(y_live_q[next_time][slice_q]) sum=sum+{y_q[next_time][slice_q][l][23],y_q[next_time][slice_q][l]};
      if(sum>25'sd8388607 || sum< -25'sd8388608) numeric_overflow<=1;
      y_q[next_time][slice_q][l]<=sum[23:0];
     end
     if(y_live_q[next_time][slice_q]) debug_additions<=debug_additions+16;
     y_live_q[next_time][slice_q]<=1;
     scatter_q[next_time]<=0;
     if((scatter_q & (scatter_q-1))==0) state_q<=NEXT_SLICE;
    end
    NEXT_SLICE: begin
     if(slice_q!=5) begin slice_q<=slice_q+1;state_q<=REQUEST;end
     else begin
      if(selected) begin
       remaining_q[physical_bank]<=remaining_q[physical_bank]-1;
       live_q[physical_bank]<=remaining_q[physical_bank]!=1;
      end
      if(channel_q==383) state_q<=OUTPUTS;
      else begin channel_q<=channel_q+1;state_q<=SCAN;end
     end
    end
    OUTPUTS: if(commit_accept) begin
     if(out_slice_q!=5) out_slice_q<=out_slice_q+1;
     else begin
      out_slice_q<=0;
      if(out_time_q==9) begin
       for(int b=0;b<8;b++) if(remaining_q[b]!=0 || live_q[b]) protocol_error<=1;
       state_q<=DONE;
      end else out_time_q<=out_time_q+1;
     end
    end
    DONE: if(bundle_done_ready) state_q<=IDLE;
    default: protocol_error<=1;
   endcase
  end
 end
endmodule
`default_nettype wire
