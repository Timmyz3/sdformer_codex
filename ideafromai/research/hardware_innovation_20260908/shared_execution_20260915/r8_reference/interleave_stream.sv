module interleave_stream(
 input logic clk,reset_n,go,
 input logic [4:0] mode,
 input logic [14:0] first_tile,tile_count,
 output logic busy,done,error,
 output logic parameter_request_valid,
 output logic [3:0] parameter_kind,
 output logic [13:0] parameter_address,
 input logic parameter_valid,input logic [255:0] parameter_data,
 output logic source_request_valid,output logic [14:0] source_tile,output logic [10:0] source_address,
 output logic origin_request_valid,output logic [14:0] origin_tile,input logic origin_valid,input logic [31:0] origin_data,
 input logic source_valid,input logic [9:0] source_data,
 output logic identity_request_valid,output logic [14:0] identity_tile,
 output logic [8:0] identity_address,input logic identity_valid,input logic [255:0] identity_data,
 input logic compute_source_allow,compute_weight_allow,
 output logic raw_monitor_valid,output logic [8:0] raw_monitor_address,output logic [255:0] raw_monitor_data,
 output logic j_monitor_valid,output logic [8:0] j_monitor_address,output logic [255:0] j_monitor_data,
 output logic wide_monitor_valid,output logic [8:0] wide_monitor_address,output logic [511:0] wide_monitor_data,
 output logic result_valid,input logic result_ready,output logic [14:0] result_tile,
 output logic [8:0] result_address,output logic [255:0] result_data,
 output logic result_tile_last,result_job_last,
 output logic [31:0] retired_tiles,
 output logic [63:0] total_cycles,static_words,parameter_stalls,source_load_words,external_source_words,
 output logic [63:0] padding_words,origin_words,origin_stalls,source_load_stalls,output_beats,
 output logic [63:0] window_cycles,launch_cycles,batches,conflict_cycles,both_compute_cycles,
 output logic [63:0] shared_source_grants,shared_weight_grants,shared_z_grants,shared_psum_grants,shared_alu_grants,
 output logic [63:0] shared_wide_grants,borrow_grants,borrow_consumer_stalls,borrow_rr_stalls,
 output logic [63:0] wide_conflict_cycles,consumer_wide_waits,
 output logic [63:0] proof_issues,range_fallback_tiles,core_repair_issues,core_repair_fields,core_normalization_issues,
 output logic [63:0] core_repair_arbitration_stalls,core_normalization_arbitration_stalls,
 output logic proof_range_ok,output logic [103:0] proof_positive_bounds,proof_negative_bounds,
 output logic [63:0] core_bitmap_native_reads,
 output logic [63:0] core_bitmap_native_issues,
 output logic [63:0] core_cache_reads,
 output logic [63:0] core_cache_writes,
 output logic [63:0] core_bitmap_z_arbitration_stalls,
 output logic [63:0] core_bitmap_alu_arbitration_stalls,
 output logic [63:0] core_bitmap_weight_arbitration_stalls,
 output logic [63:0] core_bitmap_hold_writes,
 output logic [63:0] core_bitmap_hold_reads,
 output logic [63:0] core_bitmap_row_z_arbitration_stalls,
 output logic [63:0] core_aux_reads,
 output logic [63:0] core_aux_writes,
 output logic [63:0] core_aux_issues,
 output logic [63:0] core_aux_weight_words,
 output logic [63:0] core_aux_events,
 output logic [63:0] core_metadata_reads,
 output logic [63:0] core_count_checks,
 output logic [63:0] core_count_bank_reads,
 output logic [63:0] core_count_bank_writes,
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
 logic [4:0] mode_q;logic wide_rr;
 logic class_resident,permutation_resident;logic [39:0] time_permutation;logic [5:0] ngroups;
 logic [23:0] class_mem[0:863];logic [23:0] representative[0:31];
 logic resident,load_context,consumer_select,has_second,second_started,rr;
 logic [1:0] core_seen;
 logic [14:0] batch_tile,last_tile,load_tile;
 logic [3:0] param_kind;logic [13:0] param_row;logic [10:0] load_index;
 logic [8:0] accepted;
 logic [1:0] core_start,core_cfg_valid;
 logic [2:0] core_cfg_kind;logic [10:0] core_cfg_addr;logic [255:0] core_cfg_data;
 logic [1:0] raw_valid,raw_ready,leaf_done,compute_done,fallback_used;
 logic [8:0] raw_addr[0:1];logic [255:0] raw_data[0:1];
 logic [5:0] request[0:1];logic [1:0] grant,eligible;
 logic [2:0] weight_kind[0:1];logic [9:0] weight_addr[0:1];logic [255:0] shared_weight_data;
 logic [255:0] child_lhs[0:1],child_rhs[0:1];logic [151:0] child_coefficient[0:1];
 logic [103:0] child_scalar[0:1];logic [7:0] child_correction[0:1];logic [1:0] child_mac;logic [2:0] child_format[0:1];
 logic [255:0] shared_alu_result;
 logic [511:0] borrow_lhs[0:1],borrow_rhs[0:1],cons_lhs,cons_rhs,wide_lhs,wide_rhs,wide_y;
 logic [415:0] borrow_y;
 logic cons_add_req,cons_add_grant,borrow_owner,borrow_active;
 logic [31:0] c_wide_waits;
 // Mode3 retains fixed consumer priority; mode4 alternates colliding groups. A producer
 // ZADD receives its z and wide permission atomically through resource_grant.
 assign cons_add_grant=cons_add_req && !(mode_q==4 && wide_rr && (request[0][5]||request[1][5]));
 assign borrow_owner=grant[1]&&request[1][5];
 assign borrow_active=(grant[0]&&request[0][5])||(grant[1]&&request[1][5]);
 assign wide_lhs=cons_add_grant?cons_lhs:(borrow_active?borrow_lhs[borrow_owner]:512'd0);
 assign wide_rhs=cons_add_grant?cons_rhs:(borrow_active?borrow_rhs[borrow_owner]:512'd0);
 wide_phase_alu wide_alu(.split_fields(!cons_add_grant),.lhs(wide_lhs),.rhs(wide_rhs),.y(wide_y));
 genvar wl;
 generate for(wl=0;wl<8;wl=wl+1)begin:WIDE_RESULT
  assign borrow_y[wl*52+:52]=wide_y[wl*64+:52];
 end endgenerate
 logic [5:0] unused_debug[0:1];
 logic signed [2:0] q_mem[0:7][0:863];logic signed [15:0] v_mem[0:7][0:95];
 logic [15:0] bp_q[0:2][0:7][0:53];logic [161:0] plane_live;logic [2:0] cfg_plane_live;
 logic [1:0] child_pop;logic [15:0] child_pop_mask[0:1];logic [127:0] child_pop_plane[0:1];logic [1:0] child_pop_shift[0:1];
 logic alu_pop,alu_sub;logic [4:0] pop_count[0:7];logic [15:0] pop_input[0:7];
 function automatic [4:0] pop16(input logic [15:0] a);
 logic [1:0] x[0:7];logic [2:0] y[0:3];logic [3:0] z[0:1];
 begin
 for(integer i=0;i<8;i=i+1)x[i]={1'b0,a[2*i]}+{1'b0,a[2*i+1]};
 for(integer i=0;i<4;i=i+1)y[i]={1'b0,x[2*i]}+{1'b0,x[2*i+1]};
 for(integer i=0;i<2;i=i+1)z[i]={1'b0,y[2*i]}+{1'b0,y[2*i+1]};
 pop16={1'b0,z[0]}+{1'b0,z[1]};
 end endfunction
 logic weight_owner,alu_owner,alu_active,alu_mac,proof_active;
 logic [2:0] alu_format;
 logic signed [12:0] positive_bound[0:7],negative_bound[0:7];
 logic [31:0] l_repair_issues[0:1],l_repair_fields[0:1],l_normalization_issues[0:1];
 logic [31:0] l_repair_arbitration_stalls[0:1],l_normalization_arbitration_stalls[0:1];
 logic signed [18:0] coefficient[0:7];logic signed [12:0] scalar[0:7];
 logic signed [31:0] product[0:7],lhs[0:7],rhs[0:7];
 logic consumer_start,consumer_raw_ready,consumer_done,consumer_error,consumer_valid,consumer_identity_request;
 logic [8:0] cons_identity_addr;
 logic [31:0] l_bitmap_native_reads[0:1];
 logic [31:0] l_bitmap_native_issues[0:1];
 logic [31:0] l_cache_reads[0:1];
 logic [31:0] l_cache_writes[0:1];
 logic [31:0] l_bitmap_z_arbitration_stalls[0:1];
 logic [31:0] l_bitmap_alu_arbitration_stalls[0:1];
 logic [31:0] l_bitmap_weight_arbitration_stalls[0:1];
 logic [31:0] l_bitmap_hold_writes[0:1];
 logic [31:0] l_bitmap_hold_reads[0:1];
 logic [31:0] l_bitmap_row_z_arbitration_stalls[0:1];
 logic [31:0] l_aux_reads[0:1];
 logic [31:0] l_aux_writes[0:1];
 logic [31:0] l_aux_issues[0:1];
 logic [31:0] l_aux_weight_words[0:1];
 logic [31:0] l_aux_events[0:1];
 logic [31:0] l_metadata_reads[0:1];
 logic [31:0] l_count_checks[0:1];
 logic [31:0] l_count_bank_reads[0:1];
 logic [31:0] l_count_bank_writes[0:1];
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
   eligible[c]=request[c]!=0 && (!request[c][0] || compute_source_allow) && (!request[c][1] || compute_weight_allow) && (!request[c][5] || !cons_add_grant);
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
  alu_pop=alu_active&&!proof_active&&child_pop[alu_owner];alu_sub=alu_active&&!proof_active&&(alu_format==6);
  cfg_plane_live=0;
  for(integer b=0;b<3;b=b+1)for(integer l=0;l<8;l=l+1)cfg_plane_live[b]=cfg_plane_live[b]|parameter_data[l*32+b];
  proof_range_ok=1;
  for(integer i=0;i<8;i=i+1)begin
   proof_positive_bounds[i*13+:13]=positive_bound[i];proof_negative_bounds[i*13+:13]=negative_bound[i];
   if(positive_bound[i]>13'sd511 || negative_bound[i]<-13'sd512)proof_range_ok=0;
  end
  shared_weight_data=0;
  for(integer l=0;l<8;l=l+1)begin
   if((grant[0]&&request[0][1])||(grant[1]&&request[1][1]))begin
    if(weight_kind[weight_owner]==1)shared_weight_data[l*32+:16]=v_mem[l][weight_addr[weight_owner][6:0]];
    else if(weight_kind[weight_owner]==0)shared_weight_data[l*32+:3]=q_mem[l][weight_addr[weight_owner]];
    else if(weight_kind[weight_owner]==3)shared_weight_data[l*32+:3]=representative[weight_addr[weight_owner][4:0]][l*3+:3];
    else if(weight_kind[weight_owner]==4)shared_weight_data[l*32+:16]=bp_q[weight_addr[weight_owner]/54][l][weight_addr[weight_owner]%54];
   end
   coefficient[l]=alu_mac?$signed(child_coefficient[alu_owner][l*19+:19]):19'sd0;
   scalar[l]=alu_mac?$signed(child_scalar[alu_owner][l*13+:13]):13'sd0;
   product[l]=$signed(coefficient[l])*$signed(scalar[l]);
   lhs[l]=alu_active?$signed(child_lhs[alu_owner][l*32+:32]):32'sd0;
   rhs[l]=alu_mac?product[l]:(alu_active?$signed(child_rhs[alu_owner][l*32+:32]):32'sd0);
   pop_input[l]=alu_pop?(child_pop_mask[alu_owner]&child_pop_plane[alu_owner][l*16+:16]):16'd0;
   pop_count[l]=pop16(pop_input[l]);
   if(alu_pop)rhs[l]=$signed({27'd0,pop_count[l]})<<<child_pop_shift[alu_owner];
   if(alu_mac&&alu_format==5)rhs[l]={6'd0,product[l][23:11],{2{product[l][10]}},product[l][10:0]};
   if(proof_active)begin
    lhs[l]=(param_row==0)?32'd0:{6'd0,negative_bound[l],positive_bound[l]};
    rhs[l]={6'd0,(parameter_data[l*32+2]?{{10{parameter_data[l*32+2]}},parameter_data[l*32+:3]}:13'd0),
                    (parameter_data[l*32+2]?13'd0:{10'd0,parameter_data[l*32+:3]})};
   end
  end
  if(((grant[0]&&request[0][1])||(grant[1]&&request[1][1]))&&weight_kind[weight_owner]==2)
   shared_weight_data[23:0]=class_mem[weight_addr[weight_owner]];
 end
 genvar gl,gb;
 generate for(gl=0;gl<8;gl=gl+1)begin:ALU
  for(gb=0;gb<32;gb=gb+1)begin:BIT
   wire cin;
   if(gb==0)assign cin=alu_sub;
   else if(gb==13)assign cin=(alu_format==5)?child_correction[alu_owner][gl]:((alu_format==1)?1'b0:BIT[gb-1].CARRY.cout);
   else assign cin=(((gb==13 || gb==26)&&alu_format==1) || ((gb%10==0)&&alu_format==2) ||
                    ((gb%8==0)&&alu_format==3) || ((gb%5==0)&&alu_format==4))?1'b0:BIT[gb-1].CARRY.cout;
   assign shared_alu_result[gl*32+gb]=lhs[gl][gb]^(rhs[gl][gb]^alu_sub)^cin;
   if(gb<31)begin:CARRY
    wire cout;assign cout=(lhs[gl][gb]&(rhs[gl][gb]^alu_sub))|((lhs[gl][gb]^(rhs[gl][gb]^alu_sub))&cin);
   end
  end
 end endgenerate
 always_comb begin
  busy=(state!=IDLE);parameter_request_valid=(state==PARAMETERS);parameter_kind=param_kind;parameter_address=param_row;
  load_tile=batch_tile+15'(load_context);
  source_request_valid=(state==SOURCE);source_tile=load_tile;source_address=load_index;
  origin_request_valid=(state==ORIGIN);origin_tile=load_tile;
  core_cfg_valid=0;core_cfg_kind=0;core_cfg_addr=0;core_cfg_data=0;
  if(state==PARAMETERS && parameter_valid && param_kind<7)begin
   core_cfg_valid=3;core_cfg_kind=param_kind[2:0];core_cfg_addr=param_row[10:0];core_cfg_data=parameter_data;
  end else if(state==SOURCE&&source_valid)begin
   core_cfg_valid[load_context]=1;core_cfg_kind=0;core_cfg_addr=load_index;core_cfg_data[9:0]=source_data;
  end else if(state==ORIGIN&&origin_valid)begin
   core_cfg_valid[load_context]=1;core_cfg_kind=3;core_cfg_data[31:0]=origin_data;
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
   .plane_live(plane_live),.alu_pop(child_pop[c]),.pop_mask(child_pop_mask[c]),.pop_plane(child_pop_plane[c]),.pop_shift(child_pop_shift[c]),
   .bitmap_native_reads(l_bitmap_native_reads[c]),
   .bitmap_native_issues(l_bitmap_native_issues[c]),
   .cache_reads(l_cache_reads[c]),
   .cache_writes(l_cache_writes[c]),
   .bitmap_z_arbitration_stalls(l_bitmap_z_arbitration_stalls[c]),
   .bitmap_alu_arbitration_stalls(l_bitmap_alu_arbitration_stalls[c]),
   .bitmap_weight_arbitration_stalls(l_bitmap_weight_arbitration_stalls[c]),
   .bitmap_hold_writes(l_bitmap_hold_writes[c]),
   .bitmap_hold_reads(l_bitmap_hold_reads[c]),
   .bitmap_row_z_arbitration_stalls(l_bitmap_row_z_arbitration_stalls[c]),
   .time_permutation(time_permutation),.ngroups(ngroups),
   .aux_reads(l_aux_reads[c]),
   .aux_writes(l_aux_writes[c]),
   .aux_issues(l_aux_issues[c]),
   .aux_weight_words(l_aux_weight_words[c]),
   .aux_events(l_aux_events[c]),
   .metadata_reads(l_metadata_reads[c]),
   .count_checks(l_count_checks[c]),
   .count_bank_reads(l_count_bank_reads[c]),
   .count_bank_writes(l_count_bank_writes[c]),
   .clk(clk),.reset_n(reset_n),.cfg_valid(core_cfg_valid[c]),.cfg_kind(core_cfg_kind),.cfg_addr(core_cfg_addr),.cfg_data(core_cfg_data),
   .start(core_start[c]),.mode(mode_q==4?5'd3:mode_q),.range_ok(proof_range_ok),.fallback_used(fallback_used[c]),.source_allow(compute_source_allow),.weight_allow(compute_weight_allow),
   .borrow_lhs(borrow_lhs[c]),.borrow_rhs(borrow_rhs[c]),.borrow_y(borrow_y),
   .resource_grant(grant[c]),.resource_request(request[c]),.weight_kind(weight_kind[c]),.weight_address(weight_addr[c]),.weight_data(shared_weight_data),
   .alu_lhs(child_lhs[c]),.alu_rhs(child_rhs[c]),.alu_coefficient(child_coefficient[c]),.alu_scalar(child_scalar[c]),.alu_correction(child_correction[c]),.alu_mac(child_mac[c]),.alu_format(child_format[c]),.alu_result(shared_alu_result),
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
  .add_req(cons_add_req),.add_grant(cons_add_grant),.add_lhs_bus(cons_lhs),.add_rhs_bus(cons_rhs),.add_y_bus(wide_y),.wide_waits(c_wide_waits),
  .clk(clk),.reset_n(reset_n),.start(consumer_start),.cfg_valid(state==PARAMETERS&&param_kind==7&&parameter_valid),.cfg_addr(param_row[4:0]),.cfg_data(parameter_data),
  .raw_valid(state==RUN&&raw_valid[consumer_select]),.raw_ready(consumer_raw_ready),.raw_addr(raw_addr[consumer_select]),.raw_data(raw_data[consumer_select]),
  .identity_request_valid(consumer_identity_request),.identity_address(cons_identity_addr),.identity_valid(state==RUN&&identity_valid),.identity_data(identity_data),
  .wide_monitor_valid(wide_monitor_valid),.wide_monitor_address(wide_monitor_address),.wide_monitor_data(wide_monitor_data),
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
   state<=IDLE;resident<=0;class_resident<=0;permutation_resident<=0;ngroups<=0;time_permutation<=40'h9876543210;mode_q<=0;batch_tile<=0;last_tile<=0;load_context<=0;consumer_select<=0;has_second<=0;second_started<=0;core_seen<=0;rr<=0;wide_rr<=0;
   param_kind<=4;param_row<=0;load_index<=0;accepted<=0;done<=0;error<=0;
   retired_tiles<=0;total_cycles<=0;static_words<=0;parameter_stalls<=0;source_load_words<=0;external_source_words<=0;padding_words<=0;origin_words<=0;origin_stalls<=0;source_load_stalls<=0;output_beats<=0;
   window_cycles<=0;launch_cycles<=0;batches<=0;conflict_cycles<=0;both_compute_cycles<=0;
   shared_wide_grants<=0;borrow_grants<=0;borrow_consumer_stalls<=0;borrow_rr_stalls<=0;wide_conflict_cycles<=0;consumer_wide_waits<=0;
   shared_source_grants<=0;shared_weight_grants<=0;shared_z_grants<=0;shared_psum_grants<=0;shared_alu_grants<=0;proof_issues<=0;range_fallback_tiles<=0;
   core_repair_issues<=0;core_repair_fields<=0;core_normalization_issues<=0;
   core_repair_arbitration_stalls<=0;core_normalization_arbitration_stalls<=0;
   core_bitmap_native_reads<=0;
   core_bitmap_native_issues<=0;
   core_cache_reads<=0;
   core_cache_writes<=0;
   core_bitmap_z_arbitration_stalls<=0;
   core_bitmap_alu_arbitration_stalls<=0;
   core_bitmap_weight_arbitration_stalls<=0;
   core_bitmap_hold_writes<=0;
   core_bitmap_hold_reads<=0;
   core_bitmap_row_z_arbitration_stalls<=0;
   core_aux_reads<=0;
   core_aux_writes<=0;
   core_aux_issues<=0;
   core_aux_weight_words<=0;
   core_aux_events<=0;
   core_metadata_reads<=0;
   core_count_checks<=0;
   core_count_bank_reads<=0;
   core_count_bank_writes<=0;
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
   if(cons_add_grant || borrow_active)shared_wide_grants<=shared_wide_grants+1;
   if(borrow_active)borrow_grants<=borrow_grants+1;
   if(cons_add_req && (request[0][5]||request[1][5]))begin
    wide_conflict_cycles<=wide_conflict_cycles+1;
    if(mode_q==4)wide_rr<=!wide_rr;
   end
   borrow_consumer_stalls<=borrow_consumer_stalls+64'(cons_add_grant&&request[0][5])+64'(cons_add_grant&&request[1][5]);
   borrow_rr_stalls<=borrow_rr_stalls+64'(request[0][5]&&!grant[0]&&!cons_add_grant)+64'(request[1][5]&&!grant[1]&&!cons_add_grant);
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
    if(|leaf_done)core_bitmap_native_reads<=core_bitmap_native_reads+(leaf_done[0]?64'(l_bitmap_native_reads[0]):64'd0)+(leaf_done[1]?64'(l_bitmap_native_reads[1]):64'd0);
    if(|leaf_done)core_bitmap_native_issues<=core_bitmap_native_issues+(leaf_done[0]?64'(l_bitmap_native_issues[0]):64'd0)+(leaf_done[1]?64'(l_bitmap_native_issues[1]):64'd0);
    if(|leaf_done)core_cache_reads<=core_cache_reads+(leaf_done[0]?64'(l_cache_reads[0]):64'd0)+(leaf_done[1]?64'(l_cache_reads[1]):64'd0);
    if(|leaf_done)core_cache_writes<=core_cache_writes+(leaf_done[0]?64'(l_cache_writes[0]):64'd0)+(leaf_done[1]?64'(l_cache_writes[1]):64'd0);
    if(|leaf_done)core_bitmap_z_arbitration_stalls<=core_bitmap_z_arbitration_stalls+(leaf_done[0]?64'(l_bitmap_z_arbitration_stalls[0]):64'd0)+(leaf_done[1]?64'(l_bitmap_z_arbitration_stalls[1]):64'd0);
    if(|leaf_done)core_bitmap_alu_arbitration_stalls<=core_bitmap_alu_arbitration_stalls+(leaf_done[0]?64'(l_bitmap_alu_arbitration_stalls[0]):64'd0)+(leaf_done[1]?64'(l_bitmap_alu_arbitration_stalls[1]):64'd0);
    if(|leaf_done)core_bitmap_weight_arbitration_stalls<=core_bitmap_weight_arbitration_stalls+(leaf_done[0]?64'(l_bitmap_weight_arbitration_stalls[0]):64'd0)+(leaf_done[1]?64'(l_bitmap_weight_arbitration_stalls[1]):64'd0);
    if(|leaf_done)core_bitmap_hold_writes<=core_bitmap_hold_writes+(leaf_done[0]?64'(l_bitmap_hold_writes[0]):64'd0)+(leaf_done[1]?64'(l_bitmap_hold_writes[1]):64'd0);
    if(|leaf_done)core_bitmap_hold_reads<=core_bitmap_hold_reads+(leaf_done[0]?64'(l_bitmap_hold_reads[0]):64'd0)+(leaf_done[1]?64'(l_bitmap_hold_reads[1]):64'd0);
    if(|leaf_done)core_bitmap_row_z_arbitration_stalls<=core_bitmap_row_z_arbitration_stalls+(leaf_done[0]?64'(l_bitmap_row_z_arbitration_stalls[0]):64'd0)+(leaf_done[1]?64'(l_bitmap_row_z_arbitration_stalls[1]):64'd0);
    if(|leaf_done)core_aux_reads<=core_aux_reads+(leaf_done[0]?64'(l_aux_reads[0]):64'd0)+(leaf_done[1]?64'(l_aux_reads[1]):64'd0);
    if(|leaf_done)core_aux_writes<=core_aux_writes+(leaf_done[0]?64'(l_aux_writes[0]):64'd0)+(leaf_done[1]?64'(l_aux_writes[1]):64'd0);
    if(|leaf_done)core_aux_issues<=core_aux_issues+(leaf_done[0]?64'(l_aux_issues[0]):64'd0)+(leaf_done[1]?64'(l_aux_issues[1]):64'd0);
    if(|leaf_done)core_aux_weight_words<=core_aux_weight_words+(leaf_done[0]?64'(l_aux_weight_words[0]):64'd0)+(leaf_done[1]?64'(l_aux_weight_words[1]):64'd0);
    if(|leaf_done)core_aux_events<=core_aux_events+(leaf_done[0]?64'(l_aux_events[0]):64'd0)+(leaf_done[1]?64'(l_aux_events[1]):64'd0);
    if(|leaf_done)core_metadata_reads<=core_metadata_reads+(leaf_done[0]?64'(l_metadata_reads[0]):64'd0)+(leaf_done[1]?64'(l_metadata_reads[1]):64'd0);
    if(|leaf_done)core_count_checks<=core_count_checks+(leaf_done[0]?64'(l_count_checks[0]):64'd0)+(leaf_done[1]?64'(l_count_checks[1]):64'd0);
    if(|leaf_done)core_count_bank_reads<=core_count_bank_reads+(leaf_done[0]?64'(l_count_bank_reads[0]):64'd0)+(leaf_done[1]?64'(l_count_bank_reads[1]):64'd0);
    if(|leaf_done)core_count_bank_writes<=core_count_bank_writes+(leaf_done[0]?64'(l_count_bank_writes[0]):64'd0)+(leaf_done[1]?64'(l_count_bank_writes[1]):64'd0);
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
    if(param_kind==8)class_mem[param_row[9:0]]<=parameter_data[23:0];
    if(param_kind==9)for(integer l=0;l<8;l=l+1)representative[param_row[4:0]][l*3+:3]<=parameter_data[l*32+:3];
    if(param_kind==10)ngroups<=parameter_data[5:0];
    if(param_kind==11)time_permutation<=parameter_data[39:0];
    if(param_kind==4)begin
     for(integer l=0;l<8;l=l+1)begin
      q_mem[l][param_row[9:0]]<=parameter_data[l*32+:3];
      for(integer b=0;b<3;b=b+1)bp_q[b][l][param_row[9:4]][param_row[3:0]]<=parameter_data[l*32+b];
     end
     for(integer b=0;b<3;b=b+1)
      if(param_row[3:0]==0)plane_live[b*54+int'(param_row[9:4])]<=cfg_plane_live[b];
      else plane_live[b*54+int'(param_row[9:4])]<=plane_live[b*54+int'(param_row[9:4])]|cfg_plane_live[b];
    end
    if(param_kind==5)for(integer l=0;l<8;l=l+1)v_mem[l][param_row[6:0]]<=parameter_data[l*32+:16];
   end
   case(state)
    IDLE:if(go)begin
     mode_q<=mode;batch_tile<=first_tile;last_tile<=first_tile+tile_count-15'd1;has_second<=tile_count>1;
     load_context<=0;consumer_select<=0;second_started<=0;core_seen<=0;rr<=0;wide_rr<=0;load_index<=0;param_kind<=4;param_row<=0;accepted<=0;error<=0;
   retired_tiles<=0;total_cycles<=0;static_words<=0;parameter_stalls<=0;source_load_words<=0;external_source_words<=0;padding_words<=0;origin_words<=0;origin_stalls<=0;source_load_stalls<=0;output_beats<=0;
   window_cycles<=0;launch_cycles<=0;batches<=0;conflict_cycles<=0;both_compute_cycles<=0;
   shared_wide_grants<=0;borrow_grants<=0;borrow_consumer_stalls<=0;borrow_rr_stalls<=0;wide_conflict_cycles<=0;consumer_wide_waits<=0;
   shared_source_grants<=0;shared_weight_grants<=0;shared_z_grants<=0;shared_psum_grants<=0;shared_alu_grants<=0;proof_issues<=0;range_fallback_tiles<=0;
   core_repair_issues<=0;core_repair_fields<=0;core_normalization_issues<=0;
   core_repair_arbitration_stalls<=0;core_normalization_arbitration_stalls<=0;
   core_bitmap_native_reads<=0;
   core_bitmap_native_issues<=0;
   core_cache_reads<=0;
   core_cache_writes<=0;
   core_bitmap_z_arbitration_stalls<=0;
   core_bitmap_alu_arbitration_stalls<=0;
   core_bitmap_weight_arbitration_stalls<=0;
   core_bitmap_hold_writes<=0;
   core_bitmap_hold_reads<=0;
   core_bitmap_row_z_arbitration_stalls<=0;
   core_aux_reads<=0;
   core_aux_writes<=0;
   core_aux_issues<=0;
   core_aux_weight_words<=0;
   core_aux_events<=0;
   core_metadata_reads<=0;
   core_count_checks<=0;
   core_count_bank_reads<=0;
   core_count_bank_writes<=0;
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
     if(tile_count==0||int'(first_tile)+int'(tile_count)>19200||(mode!=0&&mode!=1&&mode!=2&&mode!=3&&mode!=4&&mode!=7&&mode!=8&&mode!=20&&mode!=21))begin error<=1;state<=FAILED;end
     else if(!resident)state<=PARAMETERS;
     else if(mode>=20&&!class_resident)begin param_kind<=8;state<=PARAMETERS;end
     else if(mode==21&&!permutation_resident)begin param_kind<=11;state<=PARAMETERS;end
     else state<=SOURCE;
    end
    PARAMETERS:if(parameter_valid)begin
     static_words<=static_words+1;
     if(((param_kind==4||param_kind==6||param_kind==8)&&param_row==863)||(param_kind==5&&param_row==95)||(param_kind==7&&param_row==23)||(param_kind==9&&param_row==31)||param_kind==10||param_kind==11)begin
      param_row<=0;
      case(param_kind)
       4:param_kind<=5;5:param_kind<=6;6:param_kind<=7;
       7:begin resident<=1;if(mode_q>=20)param_kind<=8;else state<=SOURCE;end
       8:param_kind<=9;9:param_kind<=10;
       10:begin class_resident<=1;if(mode_q==21)param_kind<=11;else state<=SOURCE;end
       default:begin permutation_resident<=1;state<=SOURCE;end
      endcase
     end else param_row<=param_row+1;
    end else parameter_stalls<=parameter_stalls+1;
    SOURCE:if(source_valid)begin
     source_load_words<=source_load_words+1;external_source_words<=external_source_words+1;
     if(load_index==1535)state<=ORIGIN;else load_index<=load_index+1;
    end else source_load_stalls<=source_load_stalls+1;
    ORIGIN:if(origin_valid)begin
     origin_words<=origin_words+1;
     if(has_second&&!load_context)begin load_context<=1;load_index<=0;state<=SOURCE;end else state<=LAUNCH0;
    end else origin_stalls<=origin_stalls+1;
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
       consumer_wide_waits<=consumer_wide_waits+64'(c_wide_waits);
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
`ifdef VERILATOR
 logic [1:0] check_owned;
 always_ff @(posedge clk)begin
  if(!reset_n)check_owned<=0;
  else begin
   for(integer check_ctx=0;check_ctx<2;check_ctx=check_ctx+1)begin
    if(core_start[check_ctx])begin
     if(check_owned[check_ctx])$fatal(1,"restarting context before consumer retirement");
     check_owned[check_ctx]<=1;
    end
    if(check_owned[check_ctx]&&core_cfg_valid[check_ctx])$fatal(1,"reconfiguring live psum owner");
    if(grant[check_ctx]&&!eligible[check_ctx])$fatal(1,"grant without eligibility");
    if(grant[check_ctx]&&request[check_ctx][5]&&!request[check_ctx][2])$fatal(1,"borrow without z permission");
   end
   if(state==RUN&&consumer_done)begin
    if(!check_owned[consumer_select])$fatal(1,"consumer without context ownership");
    check_owned[consumer_select]<=0;
   end
   if(grant==3&&((request[0]&request[1])!=0))$fatal(1,"shared resource double grant");
   if(cons_add_grant&&borrow_active)$fatal(1,"wide chain double use");
   if(proof_active&&((grant[0]&&request[0][4])||(grant[1]&&request[1][4])))$fatal(1,"proof double use");
  end
 end
`endif
endmodule
