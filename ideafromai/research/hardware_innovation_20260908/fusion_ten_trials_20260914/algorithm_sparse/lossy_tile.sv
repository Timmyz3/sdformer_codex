module lossy_tile(
 input logic clk,reset_n,cfg_valid,
 input logic [3:0] cfg_kind,input logic [10:0] cfg_addr,input logic [255:0] cfg_data,
 input logic start,input logic [3:0] mode,input logic source_allow,weight_allow,
 output logic result_valid,input logic result_ready,
 output logic [8:0] result_addr,output logic [255:0] result_data,output logic done,output logic error,
 output logic [31:0] cycles,source_words,weight_words,second_weight_words,local_source_reads,
 output logic [31:0] z_vector_reads,z_scalar_reads,z_writes,first_issues,dual_updates,
 output logic [31:0] psum_reads,psum_writes,mac_issues,
 output logic [31:0] source_stalls,weight_stalls,output_stalls,
 output logic [31:0] encoder_cycles,prototype_reads,held_vectors,
 output logic [5:0] debug_state
,
 input logic identity_valid,input logic [255:0] identity_data,output logic identity_request_valid,output logic [8:0] identity_address,
 output logic raw_monitor_valid,output logic [8:0] raw_monitor_addr,output logic [255:0] raw_monitor_data,
 output logic j_monitor_valid,output logic [8:0] j_monitor_address,output logic [255:0] j_monitor_data,
 output logic [31:0] total_cycles,consumer_cycles,consumer_raw_words,consumer_identity_words,consumer_mul_issues,consumer_add_issues,consumer_round_issues,consumer_output_words,consumer_conversion_issues,consumer_coefficient_words,consumer_join_wait,consumer_output_stalls
);
logic raw_valid,raw_ready,core_done,cons_done,core_seen,cons_seen,active;
logic [8:0] raw_addr;logic [255:0] raw_data;
logic [31:0] unused_id_stall,unused_raw_wait,unused_sat,unused_conv_sat;
assign raw_monitor_valid=raw_valid&&raw_ready;assign raw_monitor_addr=raw_addr;assign raw_monitor_data=raw_data;
lossy_r8 core(.clk(clk),.reset_n(reset_n),.cfg_valid(cfg_valid&&cfg_kind<10),.cfg_kind(cfg_kind),.cfg_addr(cfg_addr),.cfg_data(cfg_data),.start(start),.mode(mode),.source_allow(source_allow),.weight_allow(weight_allow),.result_valid(raw_valid),.result_ready(raw_ready),.result_addr(raw_addr),.result_data(raw_data),.done(core_done),
.cycles(cycles),
.source_words(source_words),
.weight_words(weight_words),
.second_weight_words(second_weight_words),
.local_source_reads(local_source_reads),
.z_vector_reads(z_vector_reads),
.z_scalar_reads(z_scalar_reads),
.z_writes(z_writes),
.first_issues(first_issues),
.dual_updates(dual_updates),
.psum_reads(psum_reads),
.psum_writes(psum_writes),
.mac_issues(mac_issues),
.source_stalls(source_stalls),
.weight_stalls(weight_stalls),
.output_stalls(output_stalls),
.encoder_cycles(encoder_cycles),
.prototype_reads(prototype_reads),
.held_vectors(held_vectors),
.debug_state(debug_state));
i24_consumer consumer(.clk(clk),.reset_n(reset_n),.start(start),.cfg_valid(cfg_valid&&cfg_kind==10),.cfg_addr(cfg_addr[4:0]),.cfg_data(cfg_data),
.raw_valid(raw_valid),.raw_ready(raw_ready),.raw_addr(raw_addr),.raw_data(raw_data),
.identity_request_valid(identity_request_valid),.identity_address(identity_address),.identity_valid(identity_valid),.identity_data(identity_data),
.j_monitor_valid(j_monitor_valid),.j_monitor_address(j_monitor_address),.j_monitor_data(j_monitor_data),
.result_valid(result_valid),.result_ready(result_ready),.result_addr(result_addr),.result_data(result_data),.done(cons_done),.error(error),
.cycles(consumer_cycles),.raw_words(consumer_raw_words),.identity_words(consumer_identity_words),.coefficient_words(consumer_coefficient_words),
.mul_issues(consumer_mul_issues),.add_issues(consumer_add_issues),.round_issues(consumer_round_issues),.output_words(consumer_output_words),
.identity_stalls(unused_id_stall),.raw_wait_cycles(unused_raw_wait),.join_wait_cycles(consumer_join_wait),.output_stalls(consumer_output_stalls),.saturations(unused_sat),
.conversion_issues(consumer_conversion_issues),.conversion_saturations(unused_conv_sat));
always_ff @(posedge clk)begin
 if(!reset_n)begin done<=0;active<=0;core_seen<=0;cons_seen<=0;total_cycles<=0;end
 else begin
 done<=0;
 if(start&&!active)begin active<=1;core_seen<=0;cons_seen<=0;total_cycles<=0;end
 if(active)begin
 total_cycles<=total_cycles+1;
 if(core_done)core_seen<=1;if(cons_done)cons_seen<=1;
 if((core_seen||core_done)&&(cons_seen||cons_done))begin done<=1;active<=0;end
 end
 end
end
endmodule
