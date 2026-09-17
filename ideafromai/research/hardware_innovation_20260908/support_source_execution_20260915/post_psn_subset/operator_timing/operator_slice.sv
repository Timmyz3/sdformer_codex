module operator_slice #(parameter integer NORMALIZED=0)(
 input logic clk_core,rst_core,
 input logic cfg_we,cfg_half,input logic[4:0] cfg_addr,input logic signed[15:0] cfg_data,
 input logic context_we,codes_we,group_go,step_go,
 input logic signed[47:0] prefix_in,tau_in,p_in,n_in,
 input logic[4:0] lo_code_in,hi_code_in,m_in,
 input logic positive_in,constant_in,constant_gate_in,
 output logic gate_out,locked_out,lower_hit_out,upper_hit_out
);
 logic signed[15:0] lut_lo[32],lut_hi[32];
 logic signed[47:0] prefix_r,tau_r,p_r,n_r;
 logic[4:0] lo_r,hi_r,m_r;
 logic positive_r,constant_r,constant_gate_r;
 logic signed[47:0] lo_value,hi_value,first_sum,nv;
 logic lower_hit,upper_hit;
 always_comb begin
  lo_value=48'($signed(lut_lo[lo_r]));hi_value=48'($signed(lut_hi[hi_r]));
  first_sum=(prefix_r<<<1)+lo_value;nv=first_sum+hi_value;
 end
 always_ff @(posedge clk_core)begin
  if(cfg_we)begin
   if(cfg_half)lut_hi[cfg_addr]<=cfg_data;else lut_lo[cfg_addr]<=cfg_data;
  end
  if(rst_core)begin
   prefix_r<=0;tau_r<=0;p_r<=0;n_r<=0;lo_r<=0;hi_r<=0;m_r<=0;
   positive_r<=0;constant_r<=0;constant_gate_r<=0;
   gate_out<=0;locked_out<=0;lower_hit_out<=0;upper_hit_out<=0;
  end else if(context_we)begin
   prefix_r<=prefix_in;tau_r<=tau_in;p_r<=p_in;n_r<=n_in;
   lo_r<=lo_code_in;hi_r<=hi_code_in;m_r<=m_in;
   positive_r<=positive_in;constant_r<=constant_in;constant_gate_r<=constant_gate_in;
   gate_out<=0;locked_out<=0;lower_hit_out<=0;upper_hit_out<=0;
  end else begin
   if(codes_we)begin lo_r<=lo_code_in;hi_r<=hi_code_in;end
   if(group_go)begin
    gate_out<=constant_r?constant_gate_r:1'b0;locked_out<=constant_r;
    lower_hit_out<=0;upper_hit_out<=0;
   end else if(step_go)begin
    prefix_r<=nv;if(m_r!=0)m_r<=m_r-1'b1;
    lower_hit_out<=lower_hit;upper_hit_out<=upper_hit;
    if(!locked_out)begin
     if(lower_hit)begin gate_out<=positive_r;locked_out<=1;end
     else if(upper_hit)begin gate_out<=!positive_r;locked_out<=1;end
    end
   end
  end
 end
 generate if(NORMALIZED==0)begin:original
  logic signed[47:0] tail_p_r,tail_n_r,tail_p_delta,tail_n_delta;
  logic signed[47:0] bound_base,bound_lo,bound_hi,tail_p_a,tail_n_a;
  always_comb begin
   tail_p_a=group_go?(p_r<<<m_r):tail_p_r;
   tail_n_a=group_go?(n_r<<<m_r):tail_n_r;
   tail_p_delta=tail_p_a-p_r;tail_n_delta=tail_n_a-n_r;
   bound_base=nv<<<m_r;bound_lo=bound_base+tail_n_r;bound_hi=bound_base+tail_p_r;
   lower_hit=positive_r?(bound_lo>=tau_r):(bound_lo>tau_r);
   upper_hit=positive_r?(bound_hi<tau_r):(bound_hi<=tau_r);
  end
  always_ff @(posedge clk_core)begin
   if(rst_core)begin tail_p_r<=0;tail_n_r<=0;end
   else if(!context_we)begin
    if(group_go)begin tail_p_r<=tail_p_delta;tail_n_r<=tail_n_delta;end
    else if(step_go&&m_r!=0)begin tail_p_r<=$signed(tail_p_delta)>>>1;tail_n_r<=$signed(tail_n_delta)>>>1;end
   end
  end
 end else begin:normalized
  logic signed[48:0] q_p_r,q_n_r,pre_p,pre_n,sum_p,sum_n,rhs_p,rhs_n;
  logic signed[48:0] delta;
  always_comb begin
   delta=positive_r?49'sd1:49'sd0;
   // Same two add/sub positions are multiplexed between GROUP and PLANE.
   pre_p=(group_go?49'($signed(tau_r)):49'($signed(nv)))+49'($signed(p_r));
   pre_n=(group_go?49'($signed(tau_r)):49'($signed(nv)))+49'($signed(n_r));
   sum_p=pre_p-delta;sum_n=pre_n-delta;
   rhs_p=$signed(q_p_r)>>>m_r;rhs_n=$signed(q_n_r)>>>m_r;
   lower_hit=pre_n>rhs_n;upper_hit=pre_p<=rhs_p;
  end
  always_ff @(posedge clk_core)begin
   if(rst_core)begin q_p_r<=0;q_n_r<=0;end
   else if(!context_we&&group_go)begin q_p_r<=sum_p;q_n_r<=sum_n;end
  end
 end endgenerate
endmodule

