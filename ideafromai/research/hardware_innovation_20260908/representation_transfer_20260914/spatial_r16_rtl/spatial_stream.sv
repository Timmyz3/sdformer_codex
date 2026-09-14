module spatial_stream(
 input logic clk,reset_n,cfg_valid,start,
 input logic [2:0] cfg_kind,input logic [10:0] cfg_addr,input logic [255:0] cfg_data,
 input logic source_allow,weight_allow,raw_allow,result_ready,
 output logic result_valid,done,error,output logic [8:0] result_addr,output logic [255:0] result_data,
 output logic identity_request_valid,output logic [8:0] identity_address,
 input logic identity_valid,input logic [255:0] identity_data,
 output logic raw_monitor_valid,output logic [8:0] raw_monitor_addr,output logic [255:0] raw_monitor_data,
 output logic j_monitor_valid,output logic [8:0] j_monitor_address,output logic [255:0] j_monitor_data,
 output logic wide_monitor_valid,output logic [8:0] wide_monitor_address,output logic [511:0] wide_monitor_data,
 output logic [5:0] debug_state,
 output logic z_monitor_valid,z_monitor_stripe,output logic [5:0] z_monitor_addr,output logic [255:0] z_monitor_data,
 output logic [31:0] cycles, source_words, q1_words, q2_words, local_gathers, q1_issues, q2_issues, z_vector_reads, z_scalar_reads, z_writes, psum_reads, psum_writes, cache_writes, source_stalls, weight_stalls, output_stalls, c_cycles, c_raw_words, c_identity_words, c_coefficient_words, c_mul_issues, c_add_issues, c_round_issues, c_output_words, c_identity_stalls, c_raw_wait_cycles, c_join_wait_cycles, c_output_stalls, c_saturations, c_conversion_issues, c_conversion_saturations, c_wide_waits
);
 logic raw_valid,raw_ready,join_ready,core_done,add_req;
 logic [8:0] raw_addr;logic [255:0] raw_data;
 logic [511:0] add_lhs,add_rhs,add_y;
 assign raw_ready=join_ready&&raw_allow;
 assign raw_monitor_valid=raw_valid&&raw_ready;
 assign raw_monitor_addr=raw_addr;assign raw_monitor_data=raw_data;
 spatial_core producer(.clk(clk),.reset_n(reset_n),.cfg_valid(cfg_valid&&cfg_kind!=6),.start(start),
 .cfg_kind(cfg_kind),.cfg_addr(cfg_addr),.cfg_data(cfg_data),.source_allow(source_allow),.weight_allow(weight_allow),
 .result_ready(raw_ready),.result_valid(raw_valid),.result_addr(raw_addr),.result_data(raw_data),.done(core_done),
 .debug_state(debug_state),.z_monitor_valid(z_monitor_valid),.z_monitor_stripe(z_monitor_stripe),.z_monitor_addr(z_monitor_addr),.z_monitor_data(z_monitor_data),
 .cycles(cycles),
 .source_words(source_words),
 .q1_words(q1_words),
 .q2_words(q2_words),
 .local_gathers(local_gathers),
 .q1_issues(q1_issues),
 .q2_issues(q2_issues),
 .z_vector_reads(z_vector_reads),
 .z_scalar_reads(z_scalar_reads),
 .z_writes(z_writes),
 .psum_reads(psum_reads),
 .psum_writes(psum_writes),
 .cache_writes(cache_writes),
 .source_stalls(source_stalls),
 .weight_stalls(weight_stalls),
 .output_stalls(output_stalls));
 wide_phase_alu wide(.split_fields(1'b0),.lhs(add_lhs),.rhs(add_rhs),.y(add_y));
 i24_consumer consumer(.clk(clk),.reset_n(reset_n),.start(start),.add_req(add_req),.add_grant(add_req),
 .add_lhs_bus(add_lhs),.add_rhs_bus(add_rhs),.add_y_bus(add_y),
 .cfg_valid(cfg_valid&&cfg_kind==6),.cfg_addr(cfg_addr[4:0]),.cfg_data(cfg_data),
 .raw_valid(raw_valid&&raw_allow),.raw_ready(join_ready),.raw_addr(raw_addr),.raw_data(raw_data),
 .identity_request_valid(identity_request_valid),.identity_address(identity_address),.identity_valid(identity_valid),.identity_data(identity_data),
 .j_monitor_valid(j_monitor_valid),.j_monitor_address(j_monitor_address),.j_monitor_data(j_monitor_data),
 .wide_monitor_valid(wide_monitor_valid),.wide_monitor_address(wide_monitor_address),.wide_monitor_data(wide_monitor_data),
 .result_valid(result_valid),.result_ready(result_ready),.result_addr(result_addr),.result_data(result_data),.done(done),.error(error),
 .cycles(c_cycles),
 .raw_words(c_raw_words),
 .identity_words(c_identity_words),
 .coefficient_words(c_coefficient_words),
 .mul_issues(c_mul_issues),
 .add_issues(c_add_issues),
 .round_issues(c_round_issues),
 .output_words(c_output_words),
 .identity_stalls(c_identity_stalls),
 .raw_wait_cycles(c_raw_wait_cycles),
 .join_wait_cycles(c_join_wait_cycles),
 .output_stalls(c_output_stalls),
 .saturations(c_saturations),
 .conversion_issues(c_conversion_issues),
 .conversion_saturations(c_conversion_saturations),
 .wide_waits(c_wide_waits));
 `ifdef VERILATOR
 logic busy,seen_core_done;
 always_ff @(posedge clk)begin
  if(!reset_n)begin busy<=0;seen_core_done<=0;end
  else begin
   if(busy&&!done&&(cfg_valid||start))$fatal(1,"live configuration/restart");
   if(start)begin busy<=1;seen_core_done<=0;end
   if(core_done)seen_core_done<=1;
   if(done)begin
    if(!seen_core_done)$fatal(1,"consumer completed before producer");
    busy<=0;
   end
  end
 end
 `endif
endmodule
