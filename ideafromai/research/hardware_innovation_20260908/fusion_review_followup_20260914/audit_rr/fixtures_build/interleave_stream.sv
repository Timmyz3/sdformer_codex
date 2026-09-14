module interleave_stream(
 input logic clk,reset_n,go,
 input logic [3:0] mode,
 input logic [14:0] first_tile,tile_count,
 output logic busy,done,error,
 output logic parameter_request_valid,
 output logic [3:0] parameter_kind,
 output logic [13:0] parameter_address,
 input logic parameter_valid,input logic [255:0] parameter_data,
 output logic source_request_valid,output logic [22:0] source_address,
 input logic source_valid,input logic [9:0] source_data,
 output logic identity_request_valid,output logic [14:0] identity_tile,
 output logic [8:0] identity_address,input logic identity_valid,input logic [255:0] identity_data,
 input logic compute_source_allow,compute_weight_allow,
 output logic raw_monitor_valid,output logic [8:0] raw_monitor_address,output logic [255:0] raw_monitor_data,
 output logic j_monitor_valid,output logic [8:0] j_monitor_address,output logic [255:0] j_monitor_data,
 output logic result_valid,input logic result_ready,output logic [14:0] result_tile,
 output logic [8:0] result_address,output logic [255:0] result_data,
 output logic result_tile_last,result_job_last,
 output logic [31:0] retired_tiles,
 output logic [63:0] total_cycles,static_words,parameter_stalls,source_load_words,external_source_words,
 output logic [63:0] padding_words,origin_words,source_load_stalls,output_beats,
 output logic [63:0] window_cycles,launch_cycles,batches,conflict_cycles,both_compute_cycles,
 output logic [63:0] shared_source_grants,shared_weight_grants,shared_z_grants,shared_psum_grants,shared_alu_grants,
 output logic [63:0] proof_issues,range_fallback_tiles,core_repair_issues,core_repair_fields,core_normalization_issues,
 output logic [63:0] core_repair_arbitration_stalls,core_normalization_arbitration_stalls,
 output logic proof_range_ok,output logic [103:0] proof_positive_bounds,proof_negative_bounds,
 output logic [63:0] core_cycles,
 output logic [63:0] core_source_words,
 output logic [63:0] core_weight_words,
 output logic [63:0] core_second_weight_words,
 output logic [63:0] core_local_source_reads,
 output logic [63:0] core_z_vector_reads,
 output logic [63:0] core_z_scalar_reads,
 output logic [63:0] core_z_writes,
 output logic [63:0] core_first_issues,
 output logic [63:0] core_merged_updates,
 output logic [63:0] core_psum_reads,
 output logic [63:0] core_psum_writes,
 output logic [63:0] core_mac_issues,
 output logic [63:0] core_source_stalls,
 output logic [63:0] core_weight_stalls,
 output logic [63:0] core_output_stalls,
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
 output logic [63:0] consumer_conversion_saturations
);
 typedef enum logic [3:0] {IDLE,PARAMETERS,SOURCE,ORIGIN,LAUNCH0,RUN,NEXT_CONSUMER,FINISH,FAILED} state_t;
 state_t state;
 logic [1:0] mode_q;
 logic resident,load_context,consumer_select,has_second,second_started,rr;
 logic [1:0] core_seen;
 logic [14:0] batch_tile,last_tile,load_tile;
 logic [3:0] param_kind;logic [13:0] param_row;logic [10:0] load_index;
 logic [8:0] accepted;
 logic signed [15:0] origin_y,origin_x;
 integer native_y,native_x,native_c;logic in_bounds;
 logic [1:0] core_start,core_cfg_valid;
 logic [2:0] core_cfg_kind;logic [10:0] core_cfg_addr;logic [255:0] core_cfg_data;
 logic [1:0] raw_valid,raw_ready,leaf_done,compute_done,fallback_used;
 logic [8:0] raw_addr[0:1];logic [255:0] raw_data[0:1];
 logic [4:0] request[0:1];logic [1:0] grant,eligible;
 logic [1:0] weight_is_q2;logic [9:0] weight_addr[0:1];logic [255:0] shared_weight_data;
 logic [255:0] child_lhs[0:1],child_rhs[0:1];logic [151:0] child_coefficient[0:1];
 logic signed [12:0] child_scalar[0:1];logic [1:0] child_mac;logic [2:0] child_format[0:1];
 logic [255:0] shared_alu_result;
 logic [5:0] unused_debug[0:1];
 logic signed [2:0] q_mem[0:7][0:863];logic signed [15:0] v_mem[0:7][0:95];
 logic weight_owner,alu_owner,alu_active,alu_mac,proof_active;
 logic [2:0] alu_format;
 logic signed [12:0] positive_bound[0:7],negative_bound[0:7];
 logic [31:0] l_repair_issues[0:1],l_repair_fields[0:1],l_normalization_issues[0:1];
 logic [31:0] l_repair_arbitration_stalls[0:1],l_normalization_arbitration_stalls[0:1];
 logic signed [18:0] coefficient[0:7];logic signed [12:0] scalar;
 logic signed [31:0] product[0:7],lhs[0:7],rhs[0:7];
 logic consumer_start,consumer_raw_ready,consumer_done,consumer_error,consumer_valid,consumer_identity_request;
 logic [8:0] cons_identity_addr;
 logic [31:0] l_cycles[0:1];
 logic [31:0] l_source_words[0:1];
 logic [31:0] l_weight_words[0:1];
 logic [31:0] l_second_weight_words[0:1];
 logic [31:0] l_local_source_reads[0:1];
 logic [31:0] l_z_vector_reads[0:1];
 logic [31:0] l_z_scalar_reads[0:1];
 logic [31:0] l_z_writes[0:1];
 logic [31:0] l_first_issues[0:1];
 logic [31:0] l_merged_updates[0:1];
 logic [31:0] l_psum_reads[0:1];
 logic [31:0] l_psum_writes[0:1];
 logic [31:0] l_mac_issues[0:1];
 logic [31:0] l_source_stalls[0:1];
 logic [31:0] l_weight_stalls[0:1];
 logic [31:0] l_output_stalls[0:1];
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
 always_comb begin
  eligible=0;grant=0;
  for(integer c=0;c<2;c=c+1)
   eligible[c]=request[c]!=0 && (!request[c][0] || compute_source_allow) && (!request[c][1] || compute_weight_allow);
  if(eligible==2'b11)begin
   if((request[0]&request[1])==0)grant=2'b11;
   else grant=rr?2'b10:2'b01;
  end else grant=eligible;
 end
 always_comb begin
  weight_owner=grant[1]&&request[1][1];alu_owner=grant[1]&&request[1][4];
  proof_active=(state==PARAMETERS&&parameter_valid&&param_kind==4);
  alu_active=proof_active||(grant[0]&&request[0][4])||(grant[1]&&request[1][4]);
  alu_mac=alu_active&&!proof_active&&child_mac[alu_owner];
  alu_format=proof_active?3'd1:(alu_active?child_format[alu_owner]:3'd0);
  proof_range_ok=1;
  for(integer i=0;i<8;i=i+1)begin
   proof_positive_bounds[i*13+:13]=positive_bound[i];proof_negative_bounds[i*13+:13]=negative_bound[i];
   if(positive_bound[i]>13'sd511 || negative_bound[i]<-13'sd512)proof_range_ok=0;
  end
  scalar=alu_mac?child_scalar[alu_owner]:13'sd0;shared_weight_data=0;
  for(integer l=0;l<8;l=l+1)begin
   if((grant[0]&&request[0][1])||(grant[1]&&request[1][1]))begin
    if(weight_is_q2[weight_owner])shared_weight_data[l*32+:16]=v_mem[l][weight_addr[weight_owner][6:0]];
    else shared_weight_data[l*32+:3]=q_mem[l][weight_addr[weight_owner]];
   end
   coefficient[l]=alu_mac?$signed(child_coefficient[alu_owner][l*19+:19]):19'sd0;
   product[l]=$signed(coefficient[l])*$signed(scalar);
   lhs[l]=alu_active?$signed(child_lhs[alu_owner][l*32+:32]):32'sd0;
   rhs[l]=alu_mac?product[l]:(alu_active?$signed(child_rhs[alu_owner][l*32+:32]):32'sd0);
   if(proof_active)begin
    lhs[l]=(param_row==0)?32'd0:{6'd0,negative_bound[l],positive_bound[l]};
    rhs[l]={6'd0,(parameter_data[l*32+2]?{{10{parameter_data[l*32+2]}},parameter_data[l*32+:3]}:13'd0),
                    (parameter_data[l*32+2]?13'd0:{10'd0,parameter_data[l*32+:3]})};
   end
  end
 end
 genvar gl,gb;
 generate for(gl=0;gl<8;gl=gl+1)begin:ALU
  for(gb=0;gb<32;gb=gb+1)begin:BIT
   wire cin;
   if(gb==0)assign cin=1'b0;
   else assign cin=(((gb==13 || gb==26)&&alu_format==1) || ((gb%10==0)&&alu_format==2) ||
                    ((gb%8==0)&&alu_format==3) || ((gb%5==0)&&alu_format==4))?1'b0:BIT[gb-1].CARRY.cout;
   assign shared_alu_result[gl*32+gb]=lhs[gl][gb]^rhs[gl][gb]^cin;
   if(gb<31)begin:CARRY
    wire cout;assign cout=(lhs[gl][gb]&rhs[gl][gb])|((lhs[gl][gb]^rhs[gl][gb])&cin);
   end
  end
 end endgenerate
 always_comb begin
  busy=(state!=IDLE);parameter_request_valid=(state==PARAMETERS);parameter_kind=param_kind;parameter_address=param_row;
  load_tile=batch_tile+15'(load_context);
  origin_y=16'(2*(int'(load_tile)/160)-1);origin_x=16'(2*(int'(load_tile)%160)-1);
  native_y=int'(origin_y)+(int'(load_index)%16)/4;native_x=int'(origin_x)+int'(load_index)%4;native_c=int'(load_index)/16;
  in_bounds=native_y>=0&&native_y<240&&native_x>=0&&native_x<320;
  source_request_valid=(state==SOURCE&&in_bounds);source_address=in_bounds?23'(native_c*76800+native_y*320+native_x):23'd0;
  core_cfg_valid=0;core_cfg_kind=0;core_cfg_addr=0;core_cfg_data=0;
  if(state==PARAMETERS && parameter_valid && param_kind<7)begin
   core_cfg_valid=3;core_cfg_kind=param_kind[2:0];core_cfg_addr=param_row[10:0];core_cfg_data=parameter_data;
  end else if(state==SOURCE&&(!in_bounds||source_valid))begin
   core_cfg_valid[load_context]=1;core_cfg_kind=0;core_cfg_addr=load_index;core_cfg_data[9:0]=in_bounds?source_data:10'd0;
  end else if(state==ORIGIN)begin
   core_cfg_valid[load_context]=1;core_cfg_kind=3;core_cfg_data[31:0]={origin_x,origin_y};
  end
  core_start=0;core_start[0]=(state==LAUNCH0);
  core_start[1]=has_second && state==LAUNCH0;
  consumer_start=(state==LAUNCH0 || state==NEXT_CONSUMER);
  raw_ready=0;raw_ready[consumer_select]=consumer_raw_ready&&(state==RUN);
  raw_monitor_valid=raw_valid[consumer_select]&&raw_ready[consumer_select];
  raw_monitor_address=raw_addr[consumer_select];raw_monitor_data=raw_data[consumer_select];
  identity_request_valid=(state==RUN&&consumer_identity_request);identity_tile=batch_tile+15'(consumer_select);identity_address=cons_identity_addr;
  result_valid=(state==RUN&&consumer_valid);result_tile=batch_tile+15'(consumer_select);
  result_tile_last=(result_address==479);result_job_last=(result_tile==last_tile&&result_tile_last);
 end
 genvar c;
 generate for(c=0;c<2;c=c+1)begin:CONTEXT
  rr_context core(
   .clk(clk),.reset_n(reset_n),.cfg_valid(core_cfg_valid[c]),.cfg_kind(core_cfg_kind),.cfg_addr(core_cfg_addr),.cfg_data(core_cfg_data),
   .start(core_start[c]),.mode({2'd0,mode_q}),.range_ok(proof_range_ok),.fallback_used(fallback_used[c]),.source_allow(compute_source_allow),.weight_allow(compute_weight_allow),
   .resource_grant(grant[c]),.resource_request(request[c]),.weight_is_q2(weight_is_q2[c]),.weight_address(weight_addr[c]),.weight_data(shared_weight_data),
   .alu_lhs(child_lhs[c]),.alu_rhs(child_rhs[c]),.alu_coefficient(child_coefficient[c]),.alu_scalar(child_scalar[c]),.alu_mac(child_mac[c]),.alu_format(child_format[c]),.alu_result(shared_alu_result),
   .compute_done(compute_done[c]),.result_valid(raw_valid[c]),.result_ready(raw_ready[c]),.result_addr(raw_addr[c]),.result_data(raw_data[c]),.done(leaf_done[c]),.debug_state(unused_debug[c]),
   .cycles(l_cycles[c]),
   .source_words(l_source_words[c]),
   .weight_words(l_weight_words[c]),
   .second_weight_words(l_second_weight_words[c]),
   .local_source_reads(l_local_source_reads[c]),
   .z_vector_reads(l_z_vector_reads[c]),
   .z_scalar_reads(l_z_scalar_reads[c]),
   .z_writes(l_z_writes[c]),
   .first_issues(l_first_issues[c]),
   .merged_updates(l_merged_updates[c]),
   .psum_reads(l_psum_reads[c]),
   .psum_writes(l_psum_writes[c]),
   .mac_issues(l_mac_issues[c]),
   .source_stalls(l_source_stalls[c]),
   .weight_stalls(l_weight_stalls[c]),
   .output_stalls(l_output_stalls[c]),
   .arbitration_stalls(l_arbitration_stalls[c]),
   .repair_issues(l_repair_issues[c]),.repair_fields(l_repair_fields[c]),.normalization_issues(l_normalization_issues[c]),
   .repair_arbitration_stalls(l_repair_arbitration_stalls[c]),.normalization_arbitration_stalls(l_normalization_arbitration_stalls[c])
  );
 end endgenerate
 i24_consumer consumer(
  .clk(clk),.reset_n(reset_n),.start(consumer_start),.cfg_valid(state==PARAMETERS&&param_kind==7&&parameter_valid),.cfg_addr(param_row[4:0]),.cfg_data(parameter_data),
  .raw_valid(state==RUN&&raw_valid[consumer_select]),.raw_ready(consumer_raw_ready),.raw_addr(raw_addr[consumer_select]),.raw_data(raw_data[consumer_select]),
  .identity_request_valid(consumer_identity_request),.identity_address(cons_identity_addr),.identity_valid(state==RUN&&identity_valid),.identity_data(identity_data),
  .j_monitor_valid(j_monitor_valid),.j_monitor_address(j_monitor_address),.j_monitor_data(j_monitor_data),
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
  .conversion_saturations(c_conversion_saturations)
 );
 always_ff @(posedge clk)begin
  if(!reset_n)begin
   for(integer i=0;i<8;i=i+1)begin positive_bound[i]<=0;negative_bound[i]<=0;end
   state<=IDLE;resident<=0;mode_q<=0;batch_tile<=0;last_tile<=0;load_context<=0;consumer_select<=0;has_second<=0;second_started<=0;core_seen<=0;rr<=0;
   param_kind<=4;param_row<=0;load_index<=0;accepted<=0;done<=0;error<=0;
   retired_tiles<=0;total_cycles<=0;static_words<=0;parameter_stalls<=0;source_load_words<=0;external_source_words<=0;padding_words<=0;origin_words<=0;source_load_stalls<=0;output_beats<=0;
   window_cycles<=0;launch_cycles<=0;batches<=0;conflict_cycles<=0;both_compute_cycles<=0;
   shared_source_grants<=0;shared_weight_grants<=0;shared_z_grants<=0;shared_psum_grants<=0;shared_alu_grants<=0;proof_issues<=0;range_fallback_tiles<=0;
   core_repair_issues<=0;core_repair_fields<=0;core_normalization_issues<=0;
   core_repair_arbitration_stalls<=0;core_normalization_arbitration_stalls<=0;
   core_cycles<=0;
   core_source_words<=0;
   core_weight_words<=0;
   core_second_weight_words<=0;
   core_local_source_reads<=0;
   core_z_vector_reads<=0;
   core_z_scalar_reads<=0;
   core_z_writes<=0;
   core_first_issues<=0;
   core_merged_updates<=0;
   core_psum_reads<=0;
   core_psum_writes<=0;
   core_mac_issues<=0;
   core_source_stalls<=0;
   core_weight_stalls<=0;
   core_output_stalls<=0;
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
  end else begin
   done<=0;if(state!=IDLE)total_cycles<=total_cycles+1;
   if(eligible==3 && (request[0]&request[1])!=0)begin rr<=!rr;conflict_cycles<=conflict_cycles+1;end
   if((grant[0]&&request[0][0])||(grant[1]&&request[1][0]))shared_source_grants<=shared_source_grants+1;
   if((grant[0]&&request[0][1])||(grant[1]&&request[1][1]))shared_weight_grants<=shared_weight_grants+1;
   if((grant[0]&&request[0][2])||(grant[1]&&request[1][2]))shared_z_grants<=shared_z_grants+1;
   if((grant[0]&&request[0][3])||(grant[1]&&request[1][3]))shared_psum_grants<=shared_psum_grants+1;
   if(proof_active||(grant[0]&&request[0][4])||(grant[1]&&request[1][4]))shared_alu_grants<=shared_alu_grants+1;
   if(proof_active)begin
    proof_issues<=proof_issues+1;
    for(integer i=0;i<8;i=i+1)begin positive_bound[i]<=shared_alu_result[i*32+:13];negative_bound[i]<=shared_alu_result[i*32+13+:13];end
   end
   if(state==RUN||state==NEXT_CONSUMER)begin
    window_cycles<=window_cycles+1;
    if(second_started&&!compute_done[0]&&!compute_done[1])both_compute_cycles<=both_compute_cycles+1;
    if(core_start[1])second_started<=1;
    for(integer t=0;t<2;t=t+1)if(leaf_done[t])core_seen[t]<=1;
    if(|leaf_done)range_fallback_tiles<=range_fallback_tiles+(leaf_done[0]?64'(fallback_used[0]):64'd0)+(leaf_done[1]?64'(fallback_used[1]):64'd0);
    if(|leaf_done)core_repair_issues<=core_repair_issues+(leaf_done[0]?64'(l_repair_issues[0]):64'd0)+(leaf_done[1]?64'(l_repair_issues[1]):64'd0);
    if(|leaf_done)core_repair_fields<=core_repair_fields+(leaf_done[0]?64'(l_repair_fields[0]):64'd0)+(leaf_done[1]?64'(l_repair_fields[1]):64'd0);
    if(|leaf_done)core_normalization_issues<=core_normalization_issues+(leaf_done[0]?64'(l_normalization_issues[0]):64'd0)+(leaf_done[1]?64'(l_normalization_issues[1]):64'd0);
    if(|leaf_done)core_repair_arbitration_stalls<=core_repair_arbitration_stalls+(leaf_done[0]?64'(l_repair_arbitration_stalls[0]):64'd0)+(leaf_done[1]?64'(l_repair_arbitration_stalls[1]):64'd0);
    if(|leaf_done)core_normalization_arbitration_stalls<=core_normalization_arbitration_stalls+(leaf_done[0]?64'(l_normalization_arbitration_stalls[0]):64'd0)+(leaf_done[1]?64'(l_normalization_arbitration_stalls[1]):64'd0);
    if(|leaf_done)core_cycles<=core_cycles+(leaf_done[0]?64'(l_cycles[0]):64'd0)+(leaf_done[1]?64'(l_cycles[1]):64'd0);
    if(|leaf_done)core_source_words<=core_source_words+(leaf_done[0]?64'(l_source_words[0]):64'd0)+(leaf_done[1]?64'(l_source_words[1]):64'd0);
    if(|leaf_done)core_weight_words<=core_weight_words+(leaf_done[0]?64'(l_weight_words[0]):64'd0)+(leaf_done[1]?64'(l_weight_words[1]):64'd0);
    if(|leaf_done)core_second_weight_words<=core_second_weight_words+(leaf_done[0]?64'(l_second_weight_words[0]):64'd0)+(leaf_done[1]?64'(l_second_weight_words[1]):64'd0);
    if(|leaf_done)core_local_source_reads<=core_local_source_reads+(leaf_done[0]?64'(l_local_source_reads[0]):64'd0)+(leaf_done[1]?64'(l_local_source_reads[1]):64'd0);
    if(|leaf_done)core_z_vector_reads<=core_z_vector_reads+(leaf_done[0]?64'(l_z_vector_reads[0]):64'd0)+(leaf_done[1]?64'(l_z_vector_reads[1]):64'd0);
    if(|leaf_done)core_z_scalar_reads<=core_z_scalar_reads+(leaf_done[0]?64'(l_z_scalar_reads[0]):64'd0)+(leaf_done[1]?64'(l_z_scalar_reads[1]):64'd0);
    if(|leaf_done)core_z_writes<=core_z_writes+(leaf_done[0]?64'(l_z_writes[0]):64'd0)+(leaf_done[1]?64'(l_z_writes[1]):64'd0);
    if(|leaf_done)core_first_issues<=core_first_issues+(leaf_done[0]?64'(l_first_issues[0]):64'd0)+(leaf_done[1]?64'(l_first_issues[1]):64'd0);
    if(|leaf_done)core_merged_updates<=core_merged_updates+(leaf_done[0]?64'(l_merged_updates[0]):64'd0)+(leaf_done[1]?64'(l_merged_updates[1]):64'd0);
    if(|leaf_done)core_psum_reads<=core_psum_reads+(leaf_done[0]?64'(l_psum_reads[0]):64'd0)+(leaf_done[1]?64'(l_psum_reads[1]):64'd0);
    if(|leaf_done)core_psum_writes<=core_psum_writes+(leaf_done[0]?64'(l_psum_writes[0]):64'd0)+(leaf_done[1]?64'(l_psum_writes[1]):64'd0);
    if(|leaf_done)core_mac_issues<=core_mac_issues+(leaf_done[0]?64'(l_mac_issues[0]):64'd0)+(leaf_done[1]?64'(l_mac_issues[1]):64'd0);
    if(|leaf_done)core_source_stalls<=core_source_stalls+(leaf_done[0]?64'(l_source_stalls[0]):64'd0)+(leaf_done[1]?64'(l_source_stalls[1]):64'd0);
    if(|leaf_done)core_weight_stalls<=core_weight_stalls+(leaf_done[0]?64'(l_weight_stalls[0]):64'd0)+(leaf_done[1]?64'(l_weight_stalls[1]):64'd0);
    if(|leaf_done)core_output_stalls<=core_output_stalls+(leaf_done[0]?64'(l_output_stalls[0]):64'd0)+(leaf_done[1]?64'(l_output_stalls[1]):64'd0);
    if(|leaf_done)core_arbitration_stalls<=core_arbitration_stalls+(leaf_done[0]?64'(l_arbitration_stalls[0]):64'd0)+(leaf_done[1]?64'(l_arbitration_stalls[1]):64'd0);
   end
   if(state==PARAMETERS&&parameter_valid)begin
    if(param_kind==4)for(integer l=0;l<8;l=l+1)q_mem[l][param_row[9:0]]<=parameter_data[l*32+:3];
    if(param_kind==5)for(integer l=0;l<8;l=l+1)v_mem[l][param_row[6:0]]<=parameter_data[l*32+:16];
   end
   case(state)
    IDLE:if(go)begin
     mode_q<=mode[1:0];batch_tile<=first_tile;last_tile<=first_tile+tile_count-15'd1;has_second<=tile_count>1;
     load_context<=0;consumer_select<=0;second_started<=0;core_seen<=0;rr<=0;load_index<=0;param_kind<=4;param_row<=0;accepted<=0;error<=0;
   retired_tiles<=0;total_cycles<=0;static_words<=0;parameter_stalls<=0;source_load_words<=0;external_source_words<=0;padding_words<=0;origin_words<=0;source_load_stalls<=0;output_beats<=0;
   window_cycles<=0;launch_cycles<=0;batches<=0;conflict_cycles<=0;both_compute_cycles<=0;
   shared_source_grants<=0;shared_weight_grants<=0;shared_z_grants<=0;shared_psum_grants<=0;shared_alu_grants<=0;proof_issues<=0;range_fallback_tiles<=0;
   core_repair_issues<=0;core_repair_fields<=0;core_normalization_issues<=0;
   core_repair_arbitration_stalls<=0;core_normalization_arbitration_stalls<=0;
   core_cycles<=0;
   core_source_words<=0;
   core_weight_words<=0;
   core_second_weight_words<=0;
   core_local_source_reads<=0;
   core_z_vector_reads<=0;
   core_z_scalar_reads<=0;
   core_z_writes<=0;
   core_first_issues<=0;
   core_merged_updates<=0;
   core_psum_reads<=0;
   core_psum_writes<=0;
   core_mac_issues<=0;
   core_source_stalls<=0;
   core_weight_stalls<=0;
   core_output_stalls<=0;
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
     if(tile_count==0||int'(first_tile)+int'(tile_count)>19200||(mode!=1&&mode!=2))begin error<=1;state<=FAILED;end
     else state<=resident?SOURCE:PARAMETERS;
    end
    PARAMETERS:if(parameter_valid)begin
     static_words<=static_words+1;
     if(((param_kind==4||param_kind==6)&&param_row==863)||(param_kind==5&&param_row==95)||(param_kind==7&&param_row==23))begin
      param_row<=0;
      case(param_kind)4:param_kind<=5;5:param_kind<=6;6:param_kind<=7;default:begin resident<=1;state<=SOURCE;end endcase
     end else param_row<=param_row+1;
    end else parameter_stalls<=parameter_stalls+1;
    SOURCE:if(!in_bounds||source_valid)begin
     source_load_words<=source_load_words+1;if(in_bounds)external_source_words<=external_source_words+1;else padding_words<=padding_words+1;
     if(load_index==1535)state<=ORIGIN;else load_index<=load_index+1;
    end else source_load_stalls<=source_load_stalls+1;
    ORIGIN:begin
     origin_words<=origin_words+1;
     if(has_second&&!load_context)begin load_context<=1;load_index<=0;state<=SOURCE;end else state<=LAUNCH0;
    end
    LAUNCH0:begin
     batches<=batches+1;launch_cycles<=launch_cycles+1;consumer_select<=0;accepted<=0;core_seen<=0;second_started<=has_second;state<=RUN;
    end
    RUN:begin
     if(consumer_error)error<=1;
     if(result_valid&&result_ready)begin
      if(result_address!=accepted)error<=1;accepted<=accepted+1;output_beats<=output_beats+1;
     end
     if(consumer_done)begin
      if(accepted!=480 || !(core_seen[consumer_select]||leaf_done[consumer_select]))begin error<=1;state<=FAILED;end
      else begin
       retired_tiles<=retired_tiles+1;
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
       if(!consumer_select&&has_second)begin consumer_select<=1;state<=NEXT_CONSUMER;end
       else if(batch_tile+15'(has_second)==last_tile)state<=FINISH;
       else begin
        batch_tile<=batch_tile+15'd2;has_second<=(int'(batch_tile)+2<int'(last_tile));
        load_context<=0;load_index<=0;state<=SOURCE;
       end
      end
     end
    end
    NEXT_CONSUMER:begin accepted<=0;state<=RUN;end
    FINISH:begin done<=1;state<=IDLE;end
    FAILED:begin done<=1;state<=IDLE;end
    default:begin error<=1;state<=FAILED;end
   endcase
  end
 end

 logic [1:0] audit_owed;
 always_ff @(posedge clk)begin
  if(!reset_n) audit_owed<=0;
  else begin
   if((grant & ~eligible)!=0)$fatal(1,"grant to ineligible context");
   if(grant==3 && (request[0]&request[1])!=0)$fatal(1,"overlapping grant");
   if(proof_active && grant!=0)$fatal(1,"proof and context backend overlap");
   for(integer ac=0;ac<2;ac=ac+1)begin
    if(audit_owed[ac] && eligible[ac] && !grant[ac])$fatal(1,"second eligible denial");
    if(grant[ac] || request[ac]==0)audit_owed[ac]<=0;
    else if(eligible[ac])audit_owed[ac]<=1;
   end
  end
 end

endmodule
