`timescale 1ns/1ps
`default_nettype none
module c2_ftp_bank_parallel (
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
 output logic [31:0] debug_additions, debug_source_loads,
 output logic [31:0] debug_reduction_additions,
 output logic [7:0] debug_bank_update_mask,
 output logic [9:0] debug_time_write_mask [0:7]
);
 localparam logic [9:0] MODE=10'h140;
 typedef enum logic [2:0] {IDLE, LOAD, RUN_BANKS, REDUCE_BANK, OUTPUTS, DONE} state_t;
 typedef enum logic [2:0] {B_SCAN, B_REQUEST, B_RESPONSE, B_APPLY,
                           B_FLUSH, B_NEXT_SLICE, B_FINISHED} bank_state_t;
 state_t state_q;
 bank_state_t bank_state_q [0:7];
 logic [23:0] tag_q;
 logic share_q;
 logic [8:0] expected_source_q;
 logic [9:0] masks_q [0:383];
 logic [5:0] ordinal_q [0:7];
 logic [8:0] channel [0:7];
 logic [2:0] slice_q [0:7];
 logic [5:0] remaining_q [0:7];
 logic partial_live_q [0:7];
 logic signed [23:0] partial_q [0:7][0:5][0:15];
 // Separate bank and time dimensions are actual independent writable state.
 // All ten active temporal vectors update on the same bank APPLY/FLUSH edge.
 logic signed [23:0] y_q [0:7][0:9][0:5][0:15];
 logic y_live_q [0:7][0:9][0:5];
 logic signed [23:0] weight_q [0:7][0:15], work_q [0:7][0:15];
 logic [31:0] serial_q [0:7], bank_additions_q [0:7];
 logic selected [0:7], all_finished;
 logic [3:0] out_time_q;
 logic [2:0] out_slice_q, reduce_bank_q;
 logic reduction_live_q;
 logic signed [23:0] reduction_q [0:15];
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
  all_finished=1;
  debug_additions=debug_reduction_additions;
  debug_bank_update_mask=0;
  for(int b=0;b<8;b++) begin
   channel[b]={ordinal_q[b][5:4],3'(b),ordinal_q[b][3:0]};
   selected[b]=share_q && masks_q[channel[b]]==MODE;
   debug_additions=debug_additions+bank_additions_q[b];
   if(bank_state_q[b]!=B_FINISHED) all_finished=0;
   mem_req_epoch[b]=0;mem_req_slot[b]=serial_q[b][2:0];
   mem_req_generation[b]=serial_q[b];mem_req_tag[b]=tag_q;
   mem_req_output_block[b]={2'b0,channel[b][3]};
   mem_req_slice[b]=slice_q[b];mem_req_source_channel[b]={3'b0,channel[b]};
   debug_time_write_mask[b]=0;
   if(state_q==RUN_BANKS && !protocol_error && !numeric_overflow) begin
    if(bank_state_q[b]==B_REQUEST) mem_req_valid[b]=1;
    if(bank_state_q[b]==B_RESPONSE) mem_rsp_ready[b]=1;
    if(bridge_ready && (bank_state_q[b]==B_APPLY || bank_state_q[b]==B_FLUSH))
     debug_bank_update_mask[b]=1;
    if(bridge_ready && ((bank_state_q[b]==B_APPLY && !selected[b]) || bank_state_q[b]==B_FLUSH))
     debug_time_write_mask[b]=masks_q[channel[b]];
   end
  end
  for(int l=0;l<16;l++)
   commit_accumulator[l]=reduction_live_q ? reduction_q[l] : 24'sd0;
 end
 always_ff @(posedge clk_core) begin
  if(rst_core) begin
   state_q<=IDLE;tag_q<=0;share_q<=0;expected_source_q<=0;
   out_time_q<=0;out_slice_q<=0;reduce_bank_q<=0;reduction_live_q<=0;
   protocol_error<=0;numeric_overflow<=0;
   debug_source_loads<=0;debug_reduction_additions<=0;
   for(int b=0;b<8;b++) begin
    bank_state_q[b]<=B_FINISHED;ordinal_q[b]<=0;slice_q[b]<=0;
    remaining_q[b]<=0;partial_live_q[b]<=0;serial_q[b]<=0;bank_additions_q[b]<=0;
    for(int t=0;t<10;t++) for(int s=0;s<6;s++) y_live_q[b][t][s]<=0;
   end
  end else if(!protocol_error && !numeric_overflow) begin
   case(state_q)
    IDLE: if(start_valid && start_ready) begin
     tag_q<=start_tag;share_q<=share_enable;expected_source_q<=0;
     out_time_q<=0;out_slice_q<=0;reduce_bank_q<=0;reduction_live_q<=0;
     state_q<=LOAD;
     for(int b=0;b<8;b++) begin
      ordinal_q[b]<=0;slice_q[b]<=0;remaining_q[b]<=0;partial_live_q[b]<=0;
      bank_state_q[b]<=B_FINISHED;
      for(int t=0;t<10;t++) for(int s=0;s<6;s++) y_live_q[b][t][s]<=0;
     end
    end
    LOAD: if(source_valid && source_ready) begin
     if(source_index!=expected_source_q || source_index>=384) protocol_error<=1;
     else begin
      masks_q[source_index]<=source_mask;
      debug_source_loads<=debug_source_loads+1;
      if(share_q && source_mask==MODE) begin
       if(remaining_q[source_index[6:4]]==48) protocol_error<=1;
       else remaining_q[source_index[6:4]]<=remaining_q[source_index[6:4]]+1;
      end
      if(source_index==383) begin
       state_q<=RUN_BANKS;
       for(int b=0;b<8;b++) bank_state_q[b]<=B_SCAN;
      end else expected_source_q<=expected_source_q+1;
     end
    end
    RUN_BANKS: begin
     for(int b=0;b<8;b++) begin
      case(bank_state_q[b])
       B_SCAN: begin
        slice_q[b]<=0;
        if(masks_q[channel[b]]!=0) bank_state_q[b]<=B_REQUEST;
        else if(ordinal_q[b]==47) bank_state_q[b]<=B_FINISHED;
        else ordinal_q[b]<=ordinal_q[b]+1;
       end
       B_REQUEST: if(mem_req_accept[b]) bank_state_q[b]<=B_RESPONSE;
       B_RESPONSE: if(mem_rsp_accept[b]) begin
        if(mem_rsp_epoch[b]!=0 || mem_rsp_slot[b]!=serial_q[b][2:0]
          || mem_rsp_generation[b]!=serial_q[b] || mem_rsp_tag[b]!=tag_q)
         protocol_error<=1;
        for(int l=0;l<16;l++) weight_q[b][l]<={{16{mem_rsp_weight[b][l][7]}},mem_rsp_weight[b][l]};
        serial_q[b]<=serial_q[b]+1;
        bank_state_q[b]<=B_APPLY;
       end
       B_APPLY: if(bridge_ready) begin
        if(selected[b]) begin
         if(remaining_q[b]==0) protocol_error<=1;
         for(int l=0;l<16;l++) begin
          logic signed [24:0] sum;
          sum={weight_q[b][l][23],weight_q[b][l]};
          if(partial_live_q[b]) sum=sum+{partial_q[b][slice_q[b]][l][23],partial_q[b][slice_q[b]][l]};
          if(sum>25'sd8388607 || sum< -25'sd8388608) numeric_overflow<=1;
          partial_q[b][slice_q[b]][l]<=sum[23:0];work_q[b][l]<=sum[23:0];
         end
         if(partial_live_q[b]) bank_additions_q[b]<=bank_additions_q[b]+16;
         // A final shared partial is registered before a separately charged broadcast.
         if(remaining_q[b]==1) bank_state_q[b]<=B_FLUSH;
         else bank_state_q[b]<=B_NEXT_SLICE;
        end else begin
         logic [31:0] increments;
         increments=0;
         for(int t=0;t<10;t++) if(masks_q[channel[b]][t]) begin
          for(int l=0;l<16;l++) begin
           logic signed [24:0] sum;
           sum={weight_q[b][l][23],weight_q[b][l]};
           if(y_live_q[b][t][slice_q[b]]) sum=sum+{y_q[b][t][slice_q[b]][l][23],y_q[b][t][slice_q[b]][l]};
           if(sum>25'sd8388607 || sum< -25'sd8388608) numeric_overflow<=1;
           y_q[b][t][slice_q[b]][l]<=sum[23:0];
          end
          if(y_live_q[b][t][slice_q[b]]) increments=increments+16;
          y_live_q[b][t][slice_q[b]]<=1;
         end
         bank_additions_q[b]<=bank_additions_q[b]+increments;
         bank_state_q[b]<=B_NEXT_SLICE;
        end
       end
       B_FLUSH: if(bridge_ready) begin
        logic [31:0] increments;
        increments=0;
        for(int t=0;t<10;t++) if(masks_q[channel[b]][t]) begin
         for(int l=0;l<16;l++) begin
          logic signed [24:0] sum;
          sum={work_q[b][l][23],work_q[b][l]};
          if(y_live_q[b][t][slice_q[b]]) sum=sum+{y_q[b][t][slice_q[b]][l][23],y_q[b][t][slice_q[b]][l]};
          if(sum>25'sd8388607 || sum< -25'sd8388608) numeric_overflow<=1;
          y_q[b][t][slice_q[b]][l]<=sum[23:0];
         end
         if(y_live_q[b][t][slice_q[b]]) increments=increments+16;
         y_live_q[b][t][slice_q[b]]<=1;
        end
        bank_additions_q[b]<=bank_additions_q[b]+increments;
        bank_state_q[b]<=B_NEXT_SLICE;
       end
       B_NEXT_SLICE: begin
        if(slice_q[b]!=5) begin slice_q[b]<=slice_q[b]+1;bank_state_q[b]<=B_REQUEST;end
        else begin
         if(selected[b]) begin
          remaining_q[b]<=remaining_q[b]-1;
          partial_live_q[b]<=remaining_q[b]!=1;
         end
         if(ordinal_q[b]==47) bank_state_q[b]<=B_FINISHED;
         else begin ordinal_q[b]<=ordinal_q[b]+1;bank_state_q[b]<=B_SCAN;end
        end
       end
       B_FINISHED: begin end
       default: protocol_error<=1;
      endcase
     end
     if(all_finished) begin
      for(int b=0;b<8;b++) if(remaining_q[b]!=0 || partial_live_q[b]) protocol_error<=1;
      reduce_bank_q<=0;reduction_live_q<=0;state_q<=REDUCE_BANK;
     end
    end
    REDUCE_BANK: if(bridge_ready) begin
     if(y_live_q[reduce_bank_q][out_time_q][out_slice_q]) begin
      for(int l=0;l<16;l++) begin
       logic signed [24:0] sum;
       sum={y_q[reduce_bank_q][out_time_q][out_slice_q][l][23],y_q[reduce_bank_q][out_time_q][out_slice_q][l]};
       if(reduction_live_q) sum=sum+{reduction_q[l][23],reduction_q[l]};
       if(sum>25'sd8388607 || sum< -25'sd8388608) numeric_overflow<=1;
       reduction_q[l]<=sum[23:0];
      end
      if(reduction_live_q) debug_reduction_additions<=debug_reduction_additions+16;
      reduction_live_q<=1;
     end
     if(reduce_bank_q==7) state_q<=OUTPUTS;
     else reduce_bank_q<=reduce_bank_q+1;
    end
    OUTPUTS: if(commit_accept) begin
     if(out_time_q==9 && out_slice_q==5) state_q<=DONE;
     else begin
      if(out_slice_q!=5) out_slice_q<=out_slice_q+1;
      else begin out_slice_q<=0;out_time_q<=out_time_q+1;end
      reduce_bank_q<=0;reduction_live_q<=0;state_q<=REDUCE_BANK;
     end
    end
    DONE: if(bundle_done_ready) state_q<=IDLE;
    default: protocol_error<=1;
   endcase
  end
 end
endmodule
`default_nettype wire
