module interleave_stream(
 input logic clk,reset_n,go,borrow_enable,
 input logic [14:0] first_tile,tile_count,output logic busy,done,error,
 output logic parameter_request_valid,output logic [3:0] parameter_kind,
 output logic [13:0] parameter_address,input logic parameter_valid,input logic [255:0] parameter_data,
 output logic source_request_valid,output logic [14:0] source_tile,output logic [10:0] source_address,
 input logic source_valid,input logic [9:0] source_data,
 output logic origin_request_valid,output logic [14:0] origin_tile,input logic origin_valid,input logic [31:0] origin_data,
 output logic identity_request_valid,output logic [14:0] identity_tile,output logic [8:0] identity_address,
 input logic identity_valid,input logic [255:0] identity_data,
 input logic compute_source_allow,compute_weight_allow,
 output logic raw_monitor_valid,output logic [8:0] raw_monitor_address,output logic [255:0] raw_monitor_data,
 output logic j_monitor_valid,output logic [8:0] j_monitor_address,output logic [255:0] j_monitor_data,
 output logic wide_monitor_valid,output logic [8:0] wide_monitor_address,output logic [511:0] wide_monitor_data,
 output logic [1:0] z_monitor_valid,z_monitor_stripe,d_monitor_valid,
 output logic [11:0] z_monitor_address,d_monitor_address,
 output logic [511:0] z_monitor_data,d_monitor_data,
 output logic [29:0] context_tile,
 output logic result_valid,input logic result_ready,output logic [14:0] result_tile,
 output logic [8:0] result_address,output logic [255:0] result_data,
 output logic result_tile_last,result_job_last,output logic [31:0] retired_tiles,
 output logic [3:0] debug_state,output logic [11:0] debug_core_state,debug_request,
 output logic [1:0] debug_grant,
 output logic [63:0] total_cycles,
 output logic [63:0] static_words,
 output logic [63:0] parameter_stalls,
 output logic [63:0] source_load_words,
 output logic [63:0] source_load_stalls,
 output logic [63:0] origin_words,
 output logic [63:0] origin_stalls,
 output logic [63:0] output_beats,
 output logic [63:0] window_cycles,
 output logic [63:0] launch_cycles,
 output logic [63:0] batches,
 output logic [63:0] conflict_cycles,
 output logic [63:0] both_compute_cycles,
 output logic [63:0] shared_source_grants,
 output logic [63:0] shared_weight_grants,
 output logic [63:0] shared_z_grants,
 output logic [63:0] shared_psum_grants,
 output logic [63:0] shared_alu_grants,
 output logic [63:0] shared_wide_grants,
 output logic [63:0] borrow_grants,
 output logic [63:0] wide_conflict_cycles,
 output logic [63:0] core_cycles,
 output logic [63:0] core_source_words,
 output logic [63:0] core_q1_words,
 output logic [63:0] core_q2_words,
 output logic [63:0] core_local_gathers,
 output logic [63:0] core_q1_issues,
 output logic [63:0] core_q2_issues,
 output logic [63:0] core_z_vector_reads,
 output logic [63:0] core_z_scalar_reads,
 output logic [63:0] core_z_writes,
 output logic [63:0] core_psum_reads,
 output logic [63:0] core_psum_writes,
 output logic [63:0] core_cache_writes,
 output logic [63:0] core_source_stalls,
 output logic [63:0] core_weight_stalls,
 output logic [63:0] core_output_stalls,
 output logic [63:0] core_transform_issues,
 output logic [63:0] core_transform_reads,
 output logic [63:0] core_transform_writes,
 output logic [63:0] core_reconstruction_issues,
 output logic [63:0] core_stripe_add_issues,
 output logic [63:0] core_cache_reads,
 output logic [63:0] core_exact_halves,
 output logic [63:0] core_arbitration_stalls,
 output logic [63:0] consumer_cycles,
 output logic [63:0] consumer_raw_words,
 output logic [63:0] consumer_identity_words,
 output logic [63:0] consumer_coefficient_words,
 output logic [63:0] consumer_mul_issues,
 output logic [63:0] consumer_add_issues,
 output logic [63:0] consumer_round_issues,
 output logic [63:0] consumer_output_words,
 output logic [63:0] consumer_identity_stalls,
 output logic [63:0] consumer_raw_wait_cycles,
 output logic [63:0] consumer_join_wait_cycles,
 output logic [63:0] consumer_output_stalls,
 output logic [63:0] consumer_saturations,
 output logic [63:0] consumer_conversion_issues,
 output logic [63:0] consumer_conversion_saturations,
 output logic [63:0] consumer_wide_waits
);
 typedef enum logic [3:0] {IDLE,PARAMETERS,SOURCE,ORIGIN,LAUNCH0,RUN,NEXT_CONSUMER,FINISH,FAILED} state_t;
 state_t state;
 logic resident,borrow_q,load_context,consumer_select,has_second,rr,wide_rr;
 logic [1:0] core_seen,owned;
 logic [14:0] batch_tile,last_tile,load_tile;
 logic [3:0] param_kind;logic [13:0] param_row;logic [10:0] load_index;logic [8:0] accepted;
 logic [1:0] core_start,core_cfg_valid,raw_valid,raw_ready,leaf_done,compute_done;
 logic [2:0] core_cfg_kind;logic [10:0] core_cfg_addr;logic [31:0] core_cfg_data;
 logic [8:0] raw_addr[0:1];logic [255:0] raw_data[0:1];
 logic [5:0] request[0:1],core_state[0:1];logic [1:0] grant,eligible;
 logic [1:0] weight_kind,child_mac,child_sub,child_split;
 logic [9:0] weight_addr[0:1];logic [255:0] shared_weight_data;
 logic [255:0] child_lhs[0:1],child_rhs[0:1],shared_alu_result,product_bus,borrow_result;
 logic [151:0] child_mul_lhs[0:1];logic [103:0] child_mul_rhs[0:1];
 logic [511:0] child_wide_lhs[0:1],child_wide_rhs[0:1],cons_lhs,cons_rhs,wide_lhs,wide_rhs,wide_y;
 logic cons_add_req,cons_add_grant,borrow_owner,borrow_active,weight_owner,alu_owner,alu_active,alu_sub,alu_split;
 logic signed [7:0] q1_mem[0:7][0:575];logic signed [12:0] q2_mem[0:7][0:575];
 logic [575:0] q1_live,q2_live;logic cfg_q1_live,cfg_q2_live;
 logic signed [18:0] mul_lhs[0:7];logic signed [12:0] mul_rhs[0:7];
 logic signed [31:0] product[0:7],lhs[0:7],rhs[0:7];
 logic consumer_start,consumer_raw_ready,consumer_done,consumer_error,consumer_valid,consumer_identity_request;
 logic [8:0] cons_identity_addr;
 logic [31:0] l_cycles[0:1];
 logic [31:0] l_source_words[0:1];
 logic [31:0] l_q1_words[0:1];
 logic [31:0] l_q2_words[0:1];
 logic [31:0] l_local_gathers[0:1];
 logic [31:0] l_q1_issues[0:1];
 logic [31:0] l_q2_issues[0:1];
 logic [31:0] l_z_vector_reads[0:1];
 logic [31:0] l_z_scalar_reads[0:1];
 logic [31:0] l_z_writes[0:1];
 logic [31:0] l_psum_reads[0:1];
 logic [31:0] l_psum_writes[0:1];
 logic [31:0] l_cache_writes[0:1];
 logic [31:0] l_source_stalls[0:1];
 logic [31:0] l_weight_stalls[0:1];
 logic [31:0] l_output_stalls[0:1];
 logic [31:0] l_transform_issues[0:1];
 logic [31:0] l_transform_reads[0:1];
 logic [31:0] l_transform_writes[0:1];
 logic [31:0] l_reconstruction_issues[0:1];
 logic [31:0] l_stripe_add_issues[0:1];
 logic [31:0] l_cache_reads[0:1];
 logic [31:0] l_exact_halves[0:1];
 logic [31:0] l_arbitration_stalls[0:1];
 logic [31:0] c_cycles;
 logic [31:0] c_raw_words;
 logic [31:0] c_identity_words;
 logic [31:0] c_coefficient_words;
 logic [31:0] c_mul_issues;
 logic [31:0] c_add_issues;
 logic [31:0] c_round_issues;
 logic [31:0] c_output_words;
 logic [31:0] c_identity_stalls;
 logic [31:0] c_raw_wait_cycles;
 logic [31:0] c_join_wait_cycles;
 logic [31:0] c_output_stalls;
 logic [31:0] c_saturations;
 logic [31:0] c_conversion_issues;
 logic [31:0] c_conversion_saturations;
 logic [31:0] c_wide_waits;
 // Old RR semantics: every required resource is granted atomically.
 assign cons_add_grant=cons_add_req&&!(borrow_q&&wide_rr&&(request[0][5]||request[1][5]));
 assign borrow_owner=grant[1]&&request[1][5];
 assign borrow_active=(grant[0]&&request[0][5])||(grant[1]&&request[1][5]);
 assign wide_lhs=cons_add_grant?cons_lhs:(borrow_active?child_wide_lhs[borrow_owner]:512'd0);
 assign wide_rhs=cons_add_grant?cons_rhs:(borrow_active?child_wide_rhs[borrow_owner]:512'd0);
 wide_phase_alu wide_alu(.split_fields(!cons_add_grant),.lhs(wide_lhs),.rhs(wide_rhs),.y(wide_y));
 always_comb begin
  eligible=0;grant=0;
  for(integer c=0;c<2;c=c+1)
   eligible[c]=request[c]!=0&&(!request[c][0]||compute_source_allow)&&(!request[c][1]||compute_weight_allow)&&(!request[c][5]||!cons_add_grant);
  if(eligible==3)begin
   if((request[0]&request[1])==0)grant=3;else grant=rr?2:1;
  end else grant=eligible;
 end
 always_comb begin



  shared_weight_data=0;cfg_q1_live=0;cfg_q2_live=0;
  for(integer l=0;l<8;l=l+1)begin
   cfg_q1_live=cfg_q1_live||(parameter_data[l*32+:8]!=0);
   cfg_q2_live=cfg_q2_live||(parameter_data[l*32+:13]!=0);
   if((grant[0]&&request[0][1])||(grant[1]&&request[1][1]))begin
    if(weight_kind[weight_owner])shared_weight_data[l*32+:13]=q2_mem[l][weight_addr[weight_owner]];
    else shared_weight_data[l*32+:8]=q1_mem[l][weight_addr[weight_owner]];
   end



   lhs[l]=alu_active?$signed(child_lhs[alu_owner][l*32+:32]):32'sd0;
   rhs[l]=alu_active?$signed(child_rhs[alu_owner][l*32+:32]):32'sd0;
   borrow_result[l*32+:32]=wide_y[l*64+:32];
  end
 end
 assign weight_owner=grant[1]&&request[1][1];
 assign alu_owner=grant[1]&&request[1][4];
 assign alu_active=(grant[0]&&request[0][4])||(grant[1]&&request[1][4]);
 assign alu_sub=alu_active&&child_sub[alu_owner];
 assign alu_split=alu_active&&child_split[alu_owner];
 always_comb begin
  for(integer m=0;m<8;m=m+1)begin
   mul_lhs[m]=(alu_active&&child_mac[alu_owner])?$signed(child_mul_lhs[alu_owner][m*19+:19]):19'sd0;
   mul_rhs[m]=(alu_active&&child_mac[alu_owner])?$signed(child_mul_rhs[alu_owner][m*13+:13]):13'sd0;
   product[m]=$signed(mul_lhs[m])*$signed(mul_rhs[m]);product_bus[m*32+:32]=product[m];
  end
 end
 genvar gl,gb;
 generate for(gl=0;gl<8;gl=gl+1)begin:ALU
  for(gb=0;gb<32;gb=gb+1)begin:BIT
   wire cin;
   if(gb==0)assign cin=alu_sub;
   else if(gb==15)assign cin=alu_split?1'b0:BIT[gb-1].CARRY.cout;
   else assign cin=BIT[gb-1].CARRY.cout;
   assign shared_alu_result[gl*32+gb]=lhs[gl][gb]^(rhs[gl][gb]^alu_sub)^cin;
   if(gb<31)begin:CARRY
    wire cout;assign cout=(lhs[gl][gb]&(rhs[gl][gb]^alu_sub))|((lhs[gl][gb]^(rhs[gl][gb]^alu_sub))&cin);
   end
  end
 end endgenerate
 always_comb begin
  busy=state!=IDLE;debug_state=state;debug_core_state={core_state[1],core_state[0]};debug_request={request[1],request[0]};debug_grant=grant;
  context_tile={batch_tile+15'd1,batch_tile};load_tile=batch_tile+15'(load_context);
  parameter_request_valid=state==PARAMETERS;parameter_kind=param_kind;parameter_address=param_row;
  source_request_valid=state==SOURCE;source_tile=load_tile;source_address=load_index;
  origin_request_valid=state==ORIGIN;origin_tile=load_tile;
  core_cfg_valid=0;core_cfg_kind=0;core_cfg_addr=0;core_cfg_data=0;
  if(state==SOURCE&&source_valid)begin
   core_cfg_valid[load_context]=1;core_cfg_kind=0;core_cfg_addr=load_index;core_cfg_data[9:0]=source_data;
  end else if(state==ORIGIN&&origin_valid)begin
   core_cfg_valid[load_context]=1;core_cfg_kind=3;core_cfg_data[31:0]=origin_data;
  end
  core_start=0;core_start[0]=state==LAUNCH0;core_start[1]=has_second&&state==LAUNCH0;
  consumer_start=state==LAUNCH0||state==NEXT_CONSUMER;
  raw_ready=0;raw_ready[consumer_select]=consumer_raw_ready&&state==RUN;
  raw_monitor_valid=raw_valid[consumer_select]&&raw_ready[consumer_select];
  raw_monitor_address=raw_addr[consumer_select];raw_monitor_data=raw_data[consumer_select];
  identity_request_valid=state==RUN&&consumer_identity_request;identity_tile=batch_tile+15'(consumer_select);identity_address=cons_identity_addr;
  result_valid=state==RUN&&consumer_valid;result_tile=batch_tile+15'(consumer_select);
  result_tile_last=result_address==479;result_job_last=result_tile==last_tile&&result_tile_last;
 end
 genvar gc;
 generate for(gc=0;gc<2;gc=gc+1)begin:CONTEXT
 rr_context core(
  .clk(clk),.reset_n(reset_n),.start(core_start[gc]),.borrow_enable(borrow_q),
  .cfg_valid(core_cfg_valid[gc]),.cfg_kind(core_cfg_kind),.cfg_addr(core_cfg_addr),.cfg_data(core_cfg_data),
  .source_allow(compute_source_allow),.weight_allow(compute_weight_allow),.q1_live(q1_live),.q2_live(q2_live),
  .resource_request(request[gc]),.resource_grant(grant[gc]),.weight_kind(weight_kind[gc]),.weight_address(weight_addr[gc]),.weight_data(shared_weight_data),
  .alu_lhs(child_lhs[gc]),.alu_rhs(child_rhs[gc]),.alu_result(shared_alu_result),.product_data(product_bus),.borrow_result(borrow_result),
  .mul_lhs(child_mul_lhs[gc]),.mul_rhs(child_mul_rhs[gc]),.alu_mac(child_mac[gc]),.alu_sub(child_sub[gc]),.alu_split15(child_split[gc]),
  .borrow_lhs(child_wide_lhs[gc]),.borrow_rhs(child_wide_rhs[gc]),
  .result_valid(raw_valid[gc]),.result_ready(raw_ready[gc]),.result_addr(raw_addr[gc]),.result_data(raw_data[gc]),.done(leaf_done[gc]),.compute_done(compute_done[gc]),.debug_state(core_state[gc]),
  .z_monitor_valid(z_monitor_valid[gc]),.z_monitor_stripe(z_monitor_stripe[gc]),.z_monitor_addr(z_monitor_address[gc*6+:6]),.z_monitor_data(z_monitor_data[gc*256+:256]),
  .d_monitor_valid(d_monitor_valid[gc]),.d_monitor_addr(d_monitor_address[gc*6+:6]),.d_monitor_data(d_monitor_data[gc*256+:256]),
  .cycles(l_cycles[gc]),
  .source_words(l_source_words[gc]),
  .q1_words(l_q1_words[gc]),
  .q2_words(l_q2_words[gc]),
  .local_gathers(l_local_gathers[gc]),
  .q1_issues(l_q1_issues[gc]),
  .q2_issues(l_q2_issues[gc]),
  .z_vector_reads(l_z_vector_reads[gc]),
  .z_scalar_reads(l_z_scalar_reads[gc]),
  .z_writes(l_z_writes[gc]),
  .psum_reads(l_psum_reads[gc]),
  .psum_writes(l_psum_writes[gc]),
  .cache_writes(l_cache_writes[gc]),
  .source_stalls(l_source_stalls[gc]),
  .weight_stalls(l_weight_stalls[gc]),
  .output_stalls(l_output_stalls[gc]),
  .transform_issues(l_transform_issues[gc]),
  .transform_reads(l_transform_reads[gc]),
  .transform_writes(l_transform_writes[gc]),
  .reconstruction_issues(l_reconstruction_issues[gc]),
  .stripe_add_issues(l_stripe_add_issues[gc]),
  .cache_reads(l_cache_reads[gc]),
  .exact_halves(l_exact_halves[gc]),
  .arbitration_stalls(l_arbitration_stalls[gc])
 );
 end endgenerate
 i24_consumer consumer(
  .clk(clk),.reset_n(reset_n),.start(consumer_start),
  .add_req(cons_add_req),.add_grant(cons_add_grant),.add_lhs_bus(cons_lhs),.add_rhs_bus(cons_rhs),.add_y_bus(wide_y),
  .cfg_valid(state==PARAMETERS&&parameter_valid&&param_kind==7),.cfg_addr(param_row[4:0]),.cfg_data(parameter_data),
  .raw_valid(state==RUN&&raw_valid[consumer_select]),.raw_ready(consumer_raw_ready),.raw_addr(raw_addr[consumer_select]),.raw_data(raw_data[consumer_select]),
  .identity_request_valid(consumer_identity_request),.identity_address(cons_identity_addr),.identity_valid(state==RUN&&identity_valid),.identity_data(identity_data),
  .j_monitor_valid(j_monitor_valid),.j_monitor_address(j_monitor_address),.j_monitor_data(j_monitor_data),
  .wide_monitor_valid(wide_monitor_valid),.wide_monitor_address(wide_monitor_address),.wide_monitor_data(wide_monitor_data),
  .result_valid(consumer_valid),.result_ready(state==RUN&&result_ready),.result_addr(result_address),.result_data(result_data),.done(consumer_done),.error(consumer_error),
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
  .wide_waits(c_wide_waits)
 );
 task clear_counters;
 begin
 retired_tiles<=0;
 total_cycles<=0;
 static_words<=0;
 parameter_stalls<=0;
 source_load_words<=0;
 source_load_stalls<=0;
 origin_words<=0;
 origin_stalls<=0;
 output_beats<=0;
 window_cycles<=0;
 launch_cycles<=0;
 batches<=0;
 conflict_cycles<=0;
 both_compute_cycles<=0;
 shared_source_grants<=0;
 shared_weight_grants<=0;
 shared_z_grants<=0;
 shared_psum_grants<=0;
 shared_alu_grants<=0;
 shared_wide_grants<=0;
 borrow_grants<=0;
 wide_conflict_cycles<=0;
 core_cycles<=0;
 core_source_words<=0;
 core_q1_words<=0;
 core_q2_words<=0;
 core_local_gathers<=0;
 core_q1_issues<=0;
 core_q2_issues<=0;
 core_z_vector_reads<=0;
 core_z_scalar_reads<=0;
 core_z_writes<=0;
 core_psum_reads<=0;
 core_psum_writes<=0;
 core_cache_writes<=0;
 core_source_stalls<=0;
 core_weight_stalls<=0;
 core_output_stalls<=0;
 core_transform_issues<=0;
 core_transform_reads<=0;
 core_transform_writes<=0;
 core_reconstruction_issues<=0;
 core_stripe_add_issues<=0;
 core_cache_reads<=0;
 core_exact_halves<=0;
 core_arbitration_stalls<=0;
 consumer_cycles<=0;
 consumer_raw_words<=0;
 consumer_identity_words<=0;
 consumer_coefficient_words<=0;
 consumer_mul_issues<=0;
 consumer_add_issues<=0;
 consumer_round_issues<=0;
 consumer_output_words<=0;
 consumer_identity_stalls<=0;
 consumer_raw_wait_cycles<=0;
 consumer_join_wait_cycles<=0;
 consumer_output_stalls<=0;
 consumer_saturations<=0;
 consumer_conversion_issues<=0;
 consumer_conversion_saturations<=0;
 consumer_wide_waits<=0;
 end endtask
 always_ff @(posedge clk)begin
  if(!reset_n)begin
   state<=IDLE;resident<=0;borrow_q<=0;load_context<=0;consumer_select<=0;has_second<=0;rr<=0;wide_rr<=0;core_seen<=0;owned<=0;
   batch_tile<=0;last_tile<=0;param_kind<=4;param_row<=0;load_index<=0;accepted<=0;done<=0;error<=0;
   q1_live<=0;q2_live<=0;clear_counters();
  end else begin
   done<=0;if(state!=IDLE)total_cycles<=total_cycles+1;
   if(eligible==3&&(request[0]&request[1])!=0)begin rr<=!rr;conflict_cycles<=conflict_cycles+1;end
   if(cons_add_req&&(request[0][5]||request[1][5]))begin wide_rr<=!wide_rr;wide_conflict_cycles<=wide_conflict_cycles+1;end
   if((grant[0]&&request[0][0])||(grant[1]&&request[1][0]))shared_source_grants<=shared_source_grants+1;
   if((grant[0]&&request[0][1])||(grant[1]&&request[1][1]))shared_weight_grants<=shared_weight_grants+1;
   if((grant[0]&&request[0][2])||(grant[1]&&request[1][2]))shared_z_grants<=shared_z_grants+1;
   if((grant[0]&&request[0][3])||(grant[1]&&request[1][3]))shared_psum_grants<=shared_psum_grants+1;
   if((grant[0]&&request[0][4])||(grant[1]&&request[1][4]))shared_alu_grants<=shared_alu_grants+1;
   if(cons_add_grant||borrow_active)shared_wide_grants<=shared_wide_grants+1;
   if(borrow_active)borrow_grants<=borrow_grants+1;
   if(state==RUN||state==NEXT_CONSUMER)begin
    window_cycles<=window_cycles+1;
    if(has_second&&!compute_done[0]&&!compute_done[1])both_compute_cycles<=both_compute_cycles+1;
    for(integer c=0;c<2;c=c+1)if(leaf_done[c])core_seen[c]<=1;
    if(|leaf_done)core_cycles<=core_cycles+(leaf_done[0]?64'(l_cycles[0]):64'd0)+(leaf_done[1]?64'(l_cycles[1]):64'd0);
    if(|leaf_done)core_source_words<=core_source_words+(leaf_done[0]?64'(l_source_words[0]):64'd0)+(leaf_done[1]?64'(l_source_words[1]):64'd0);
    if(|leaf_done)core_q1_words<=core_q1_words+(leaf_done[0]?64'(l_q1_words[0]):64'd0)+(leaf_done[1]?64'(l_q1_words[1]):64'd0);
    if(|leaf_done)core_q2_words<=core_q2_words+(leaf_done[0]?64'(l_q2_words[0]):64'd0)+(leaf_done[1]?64'(l_q2_words[1]):64'd0);
    if(|leaf_done)core_local_gathers<=core_local_gathers+(leaf_done[0]?64'(l_local_gathers[0]):64'd0)+(leaf_done[1]?64'(l_local_gathers[1]):64'd0);
    if(|leaf_done)core_q1_issues<=core_q1_issues+(leaf_done[0]?64'(l_q1_issues[0]):64'd0)+(leaf_done[1]?64'(l_q1_issues[1]):64'd0);
    if(|leaf_done)core_q2_issues<=core_q2_issues+(leaf_done[0]?64'(l_q2_issues[0]):64'd0)+(leaf_done[1]?64'(l_q2_issues[1]):64'd0);
    if(|leaf_done)core_z_vector_reads<=core_z_vector_reads+(leaf_done[0]?64'(l_z_vector_reads[0]):64'd0)+(leaf_done[1]?64'(l_z_vector_reads[1]):64'd0);
    if(|leaf_done)core_z_scalar_reads<=core_z_scalar_reads+(leaf_done[0]?64'(l_z_scalar_reads[0]):64'd0)+(leaf_done[1]?64'(l_z_scalar_reads[1]):64'd0);
    if(|leaf_done)core_z_writes<=core_z_writes+(leaf_done[0]?64'(l_z_writes[0]):64'd0)+(leaf_done[1]?64'(l_z_writes[1]):64'd0);
    if(|leaf_done)core_psum_reads<=core_psum_reads+(leaf_done[0]?64'(l_psum_reads[0]):64'd0)+(leaf_done[1]?64'(l_psum_reads[1]):64'd0);
    if(|leaf_done)core_psum_writes<=core_psum_writes+(leaf_done[0]?64'(l_psum_writes[0]):64'd0)+(leaf_done[1]?64'(l_psum_writes[1]):64'd0);
    if(|leaf_done)core_cache_writes<=core_cache_writes+(leaf_done[0]?64'(l_cache_writes[0]):64'd0)+(leaf_done[1]?64'(l_cache_writes[1]):64'd0);
    if(|leaf_done)core_source_stalls<=core_source_stalls+(leaf_done[0]?64'(l_source_stalls[0]):64'd0)+(leaf_done[1]?64'(l_source_stalls[1]):64'd0);
    if(|leaf_done)core_weight_stalls<=core_weight_stalls+(leaf_done[0]?64'(l_weight_stalls[0]):64'd0)+(leaf_done[1]?64'(l_weight_stalls[1]):64'd0);
    if(|leaf_done)core_output_stalls<=core_output_stalls+(leaf_done[0]?64'(l_output_stalls[0]):64'd0)+(leaf_done[1]?64'(l_output_stalls[1]):64'd0);
    if(|leaf_done)core_transform_issues<=core_transform_issues+(leaf_done[0]?64'(l_transform_issues[0]):64'd0)+(leaf_done[1]?64'(l_transform_issues[1]):64'd0);
    if(|leaf_done)core_transform_reads<=core_transform_reads+(leaf_done[0]?64'(l_transform_reads[0]):64'd0)+(leaf_done[1]?64'(l_transform_reads[1]):64'd0);
    if(|leaf_done)core_transform_writes<=core_transform_writes+(leaf_done[0]?64'(l_transform_writes[0]):64'd0)+(leaf_done[1]?64'(l_transform_writes[1]):64'd0);
    if(|leaf_done)core_reconstruction_issues<=core_reconstruction_issues+(leaf_done[0]?64'(l_reconstruction_issues[0]):64'd0)+(leaf_done[1]?64'(l_reconstruction_issues[1]):64'd0);
    if(|leaf_done)core_stripe_add_issues<=core_stripe_add_issues+(leaf_done[0]?64'(l_stripe_add_issues[0]):64'd0)+(leaf_done[1]?64'(l_stripe_add_issues[1]):64'd0);
    if(|leaf_done)core_cache_reads<=core_cache_reads+(leaf_done[0]?64'(l_cache_reads[0]):64'd0)+(leaf_done[1]?64'(l_cache_reads[1]):64'd0);
    if(|leaf_done)core_exact_halves<=core_exact_halves+(leaf_done[0]?64'(l_exact_halves[0]):64'd0)+(leaf_done[1]?64'(l_exact_halves[1]):64'd0);
    if(|leaf_done)core_arbitration_stalls<=core_arbitration_stalls+(leaf_done[0]?64'(l_arbitration_stalls[0]):64'd0)+(leaf_done[1]?64'(l_arbitration_stalls[1]):64'd0);
   end
   if(state==PARAMETERS&&parameter_valid)begin
    if(param_kind==4)begin
     for(integer l=0;l<8;l=l+1)q1_mem[l][param_row[9:0]]<=parameter_data[l*32+:8];
     q1_live[param_row[9:0]]<=cfg_q1_live;
    end
    if(param_kind==5)begin
     for(integer l=0;l<8;l=l+1)q2_mem[l][param_row[9:0]]<=parameter_data[l*32+:13];
     q2_live[param_row[9:0]]<=cfg_q2_live;
    end
   end
   case(state)
    IDLE:if(go)begin
     borrow_q<=borrow_enable;batch_tile<=first_tile;last_tile<=first_tile+tile_count-1;has_second<=tile_count>1;
     load_context<=0;consumer_select<=0;rr<=0;wide_rr<=0;load_index<=0;param_kind<=4;param_row<=0;accepted<=0;core_seen<=0;error<=0;clear_counters();
     if(tile_count==0||int'(first_tile)+int'(tile_count)>32767)begin error<=1;state<=FAILED;end
     else state<=resident?SOURCE:PARAMETERS;
    end
    PARAMETERS:if(parameter_valid)begin
     static_words<=static_words+1;
     if(((param_kind==4||param_kind==5)&&param_row==575)||(param_kind==7&&param_row==23))begin
      param_row<=0;
      if(param_kind==4)param_kind<=5;
      else if(param_kind==5)param_kind<=7;
      else begin resident<=1;state<=SOURCE;end
     end else param_row<=param_row+1;
    end else parameter_stalls<=parameter_stalls+1;
    SOURCE:if(source_valid)begin
     source_load_words<=source_load_words+1;
     if(load_index==1535)state<=ORIGIN;else load_index<=load_index+1;
    end else source_load_stalls<=source_load_stalls+1;
    ORIGIN:if(origin_valid)begin
     origin_words<=origin_words+1;
     if(has_second&&!load_context)begin load_context<=1;load_index<=0;state<=SOURCE;end else state<=LAUNCH0;
    end else origin_stalls<=origin_stalls+1;
    LAUNCH0:begin
     batches<=batches+1;launch_cycles<=launch_cycles+1;consumer_select<=0;accepted<=0;core_seen<=0;owned<=has_second?2'b11:2'b01;state<=RUN;
    end
    RUN:begin
     if(consumer_error)error<=1;
     if(result_valid&&result_ready)begin
      if(result_address!=accepted)error<=1;
      accepted<=accepted+1;output_beats<=output_beats+1;
     end
     if(consumer_done)begin
      if(accepted!=480||!(core_seen[consumer_select]||leaf_done[consumer_select]))begin error<=1;state<=FAILED;end
      else begin
       owned[consumer_select]<=0;retired_tiles<=retired_tiles+1;
       consumer_cycles<=consumer_cycles+64'(c_cycles);
       consumer_raw_words<=consumer_raw_words+64'(c_raw_words);
       consumer_identity_words<=consumer_identity_words+64'(c_identity_words);
       consumer_coefficient_words<=consumer_coefficient_words+64'(c_coefficient_words);
       consumer_mul_issues<=consumer_mul_issues+64'(c_mul_issues);
       consumer_add_issues<=consumer_add_issues+64'(c_add_issues);
       consumer_round_issues<=consumer_round_issues+64'(c_round_issues);
       consumer_output_words<=consumer_output_words+64'(c_output_words);
       consumer_identity_stalls<=consumer_identity_stalls+64'(c_identity_stalls);
       consumer_raw_wait_cycles<=consumer_raw_wait_cycles+64'(c_raw_wait_cycles);
       consumer_join_wait_cycles<=consumer_join_wait_cycles+64'(c_join_wait_cycles);
       consumer_output_stalls<=consumer_output_stalls+64'(c_output_stalls);
       consumer_saturations<=consumer_saturations+64'(c_saturations);
       consumer_conversion_issues<=consumer_conversion_issues+64'(c_conversion_issues);
       consumer_conversion_saturations<=consumer_conversion_saturations+64'(c_conversion_saturations);
       consumer_wide_waits<=consumer_wide_waits+64'(c_wide_waits);
       if(has_second&&!consumer_select)begin consumer_select<=1;accepted<=0;state<=NEXT_CONSUMER;end
       else if(batch_tile+15'(has_second)==last_tile)state<=FINISH;
       else begin
        batch_tile<=batch_tile+2;has_second<=batch_tile+3<=last_tile;
        load_context<=0;load_index<=0;state<=SOURCE;
       end
      end
     end
    end
    NEXT_CONSUMER:state<=RUN;
    FINISH:begin done<=1;state<=IDLE;end
    FAILED:begin done<=1;state<=IDLE;end
    default:state<=FAILED;
   endcase
  end
 end
 `ifdef VERILATOR
 always_ff @(posedge clk)if(reset_n)begin
  for(integer k=0;k<6;k=k+1)if(grant[0]&&request[0][k]&&grant[1]&&request[1][k])$fatal(1,"duplicate resource owner");
  if(borrow_active&&cons_add_grant)$fatal(1,"wide double owner");
  if((state==SOURCE||state==ORIGIN)&&owned[load_context])$fatal(1,"context reloaded before I24 retirement");
  if(state==LAUNCH0&&owned!=0)$fatal(1,"context launch before retirement");
  if(result_valid&&!owned[consumer_select])$fatal(1,"I24 without owned context");
  if(state==IDLE&&owned!=0)$fatal(1,"early job completion");
 end
 `endif
endmodule
