module consumer_stream (
 input logic clk, reset_n, go,
 input logic [3:0] mode,
 input logic [14:0] first_tile, tile_count,
 output logic busy, done, error,
 output logic parameter_request_valid,
 output logic [3:0] parameter_kind,
 output logic [13:0] parameter_address,
 input logic parameter_valid,
 input logic [255:0] parameter_data,
 output logic source_request_valid,
 output logic [22:0] source_address,
 input logic source_valid,
 input logic [9:0] source_data,
 output logic identity_request_valid,
 output logic [14:0] identity_tile,
 output logic [8:0] identity_address,
 input logic identity_valid,
 input logic [255:0] identity_data,
 input logic compute_source_allow, compute_weight_allow,
 output logic raw_monitor_valid,
 output logic [8:0] raw_monitor_address,
 output logic [255:0] raw_monitor_data,
 output logic result_valid,
 input logic result_ready,
 output logic [14:0] result_tile,
 output logic [8:0] result_address,
 output logic [255:0] result_data,
 output logic result_tile_last, result_job_last,
 output logic [31:0] retired_tiles,
 output logic [63:0] total_cycles, static_words, parameter_stalls,
 output logic [63:0] source_load_words, external_source_words, padding_words,
 output logic [63:0] origin_words, source_load_stalls, output_beats,
 output logic [63:0] core_cycles,
 output logic [63:0] core_source_words,
 output logic [63:0] core_weight_words,
 output logic [63:0] core_psum_reads,
 output logic [63:0] core_psum_writes,
 output logic [63:0] core_sum_issues,
 output logic [63:0] core_update_issues,
 output logic [63:0] core_source_stalls,
 output logic [63:0] core_weight_stalls,
 output logic [63:0] core_output_stalls,
 output logic [63:0] core_masked_contexts,
 output logic [63:0] core_zero_contexts,
 output logic [63:0] core_pattern_copies,
 output logic [63:0] core_merge_issues,
 output logic [63:0] core_z_reads,
 output logic [63:0] core_z_writes,
 output logic [63:0] core_first_issues,
 output logic [63:0] core_mac_issues,
 output logic [63:0] core_second_weight_words,
 output logic [63:0] core_support_reads,
 output logic [63:0] core_delta_issues,
 output logic [63:0] core_prefix_issues,
 output logic [63:0] core_local_source_reads,
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
 output logic [63:0] consumer_saturations
);
 typedef enum logic [2:0] {IDLE, PARAMETERS, SOURCE, ORIGIN, LAUNCH, RUN, FINISH, FAILED} state_t;
 state_t state;
 logic resident, core_seen, consumer_seen;
 logic [3:0] mode_q, param_kind;
 logic [13:0] param_row;
 logic [14:0] tile_id, last_tile;
 logic [10:0] load_index;
 logic [8:0] accepted;
 logic signed [15:0] origin_y, origin_x;
 integer native_y, native_x, native_c;
 logic in_bounds, core_cfg_valid, core_start, raw_valid, raw_ready, leaf_done;
 logic [2:0] core_cfg_kind;
 logic [13:0] core_cfg_addr;
 logic [255:0] core_cfg_data, raw_data;
 logic [8:0] raw_addr;
 logic cons_done, cons_error, cons_identity_request, cons_result_valid;
 logic [8:0] cons_identity_addr;
 logic [31:0] l_cycles;
 logic [31:0] l_source_words;
 logic [31:0] l_weight_words;
 logic [31:0] l_psum_reads;
 logic [31:0] l_psum_writes;
 logic [31:0] l_sum_issues;
 logic [31:0] l_update_issues;
 logic [31:0] l_source_stalls;
 logic [31:0] l_weight_stalls;
 logic [31:0] l_output_stalls;
 logic [31:0] l_masked_contexts;
 logic [31:0] l_zero_contexts;
 logic [31:0] l_pattern_copies;
 logic [31:0] l_merge_issues;
 logic [31:0] l_z_reads;
 logic [31:0] l_z_writes;
 logic [31:0] l_first_issues;
 logic [31:0] l_mac_issues;
 logic [31:0] l_second_weight_words;
 logic [31:0] l_support_reads;
 logic [31:0] l_delta_issues;
 logic [31:0] l_prefix_issues;
 logic [31:0] l_local_source_reads;
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
 always_comb begin
  busy=(state!=IDLE);
  parameter_request_valid=(state==PARAMETERS);
  parameter_kind=param_kind;parameter_address=param_row;
  origin_y=16'(2*(int'(tile_id)/160)-1);origin_x=16'(2*(int'(tile_id)%160)-1);
  native_y=int'(origin_y)+(int'(load_index)%16)/4;
  native_x=int'(origin_x)+(int'(load_index)%16)%4;native_c=int'(load_index)/16;
  in_bounds=(native_y>=0 && native_y<240 && native_x>=0 && native_x<320);
  source_request_valid=(state==SOURCE && in_bounds);
  source_address=in_bounds?23'(native_c*76800+native_y*320+native_x):23'd0;
  core_cfg_valid=0;core_cfg_kind=0;core_cfg_addr=0;core_cfg_data=0;
  if(state==PARAMETERS && param_kind<7 && parameter_valid) begin
   core_cfg_valid=1;core_cfg_kind=param_kind[2:0];core_cfg_addr=param_row;core_cfg_data=parameter_data;
  end else if(state==SOURCE && (!in_bounds || source_valid)) begin
   core_cfg_valid=1;core_cfg_kind=0;core_cfg_addr={3'd0,load_index};core_cfg_data[9:0]=in_bounds?source_data:10'd0;
  end else if(state==ORIGIN) begin
   core_cfg_valid=1;core_cfg_kind=3;core_cfg_data[31:0]={origin_x,origin_y};
  end
  core_start=(state==LAUNCH);
  identity_request_valid=(state==RUN && cons_identity_request);
  identity_tile=tile_id;identity_address=cons_identity_addr;
  raw_monitor_valid=(state==RUN && raw_valid && raw_ready);
  raw_monitor_address=raw_addr;raw_monitor_data=raw_data;
  result_valid=(state==RUN && cons_result_valid);result_tile=tile_id;
  result_tile_last=(result_address==479);result_job_last=(tile_id==last_tile && result_tile_last);
 end
 r8_fusion leaf (
  .clk(clk),.reset_n(reset_n),.cfg_valid(core_cfg_valid),.cfg_kind(core_cfg_kind),
  .cfg_addr(core_cfg_addr),.cfg_data(core_cfg_data),.start(core_start),.mode(mode_q),
  .source_allow(compute_source_allow),.weight_allow(compute_weight_allow),
  .result_valid(raw_valid),.result_ready(raw_ready),.result_addr(raw_addr),.result_data(raw_data),.done(leaf_done),
  .cycles(l_cycles),
  .source_words(l_source_words),
  .weight_words(l_weight_words),
  .psum_reads(l_psum_reads),
  .psum_writes(l_psum_writes),
  .sum_issues(l_sum_issues),
  .update_issues(l_update_issues),
  .source_stalls(l_source_stalls),
  .weight_stalls(l_weight_stalls),
  .output_stalls(l_output_stalls),
  .masked_contexts(l_masked_contexts),
  .zero_contexts(l_zero_contexts),
  .pattern_copies(l_pattern_copies),
  .merge_issues(l_merge_issues),
  .z_reads(l_z_reads),
  .z_writes(l_z_writes),
  .first_issues(l_first_issues),
  .mac_issues(l_mac_issues),
  .second_weight_words(l_second_weight_words),
  .support_reads(l_support_reads),
  .delta_issues(l_delta_issues),
  .prefix_issues(l_prefix_issues),
  .local_source_reads(l_local_source_reads)
 );
 i24_consumer consumer (
  .clk(clk),.reset_n(reset_n),.start(core_start),
  .cfg_valid(state==PARAMETERS && param_kind==7 && parameter_valid),
  .cfg_addr(param_row[4:0]),.cfg_data(parameter_data),
  .raw_valid(raw_valid),.raw_ready(raw_ready),.raw_addr(raw_addr),.raw_data(raw_data),
  .identity_request_valid(cons_identity_request),.identity_address(cons_identity_addr),
  .identity_valid(state==RUN && identity_valid),.identity_data(identity_data),
  .result_valid(cons_result_valid),.result_ready(state==RUN && result_ready),
  .result_addr(result_address),.result_data(result_data),.done(cons_done),.error(cons_error),
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
  .saturations(c_saturations)
 );
 always_ff @(posedge clk) begin
  if(!reset_n) begin
   state<=IDLE;resident<=0;mode_q<=6;tile_id<=0;last_tile<=0;param_kind<=1;param_row<=0;
   load_index<=0;accepted<=0;done<=0;error<=0;core_seen<=0;consumer_seen<=0;
   retired_tiles<=0;total_cycles<=0;static_words<=0;parameter_stalls<=0;
   source_load_words<=0;external_source_words<=0;padding_words<=0;origin_words<=0;source_load_stalls<=0;output_beats<=0;
   core_cycles<=0;
   core_source_words<=0;
   core_weight_words<=0;
   core_psum_reads<=0;
   core_psum_writes<=0;
   core_sum_issues<=0;
   core_update_issues<=0;
   core_source_stalls<=0;
   core_weight_stalls<=0;
   core_output_stalls<=0;
   core_masked_contexts<=0;
   core_zero_contexts<=0;
   core_pattern_copies<=0;
   core_merge_issues<=0;
   core_z_reads<=0;
   core_z_writes<=0;
   core_first_issues<=0;
   core_mac_issues<=0;
   core_second_weight_words<=0;
   core_support_reads<=0;
   core_delta_issues<=0;
   core_prefix_issues<=0;
   core_local_source_reads<=0;
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
  end else begin
   done<=0;if(state!=IDLE) total_cycles<=total_cycles+1;
   case(state)
    IDLE: if(go) begin
     error<=0;mode_q<=mode;tile_id<=first_tile;last_tile<=first_tile+tile_count-15'd1;
     param_kind<=1;param_row<=0;load_index<=0;accepted<=0;core_seen<=0;consumer_seen<=0;
   retired_tiles<=0;total_cycles<=0;static_words<=0;parameter_stalls<=0;
   source_load_words<=0;external_source_words<=0;padding_words<=0;origin_words<=0;source_load_stalls<=0;output_beats<=0;
   core_cycles<=0;
   core_source_words<=0;
   core_weight_words<=0;
   core_psum_reads<=0;
   core_psum_writes<=0;
   core_sum_issues<=0;
   core_update_issues<=0;
   core_source_stalls<=0;
   core_weight_stalls<=0;
   core_output_stalls<=0;
   core_masked_contexts<=0;
   core_zero_contexts<=0;
   core_pattern_copies<=0;
   core_merge_issues<=0;
   core_z_reads<=0;
   core_z_writes<=0;
   core_first_issues<=0;
   core_mac_issues<=0;
   core_second_weight_words<=0;
   core_support_reads<=0;
   core_delta_issues<=0;
   core_prefix_issues<=0;
   core_local_source_reads<=0;
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
     if(tile_count==0 || int'(first_tile)+int'(tile_count)>19200 || (mode!=6 && mode!=7 && mode!=9 && mode!=11)) begin error<=1;state<=FAILED;end
     else state<=resident?SOURCE:PARAMETERS;
    end
    PARAMETERS: if(parameter_valid) begin
     static_words<=static_words+1;
     if((param_kind==1 && param_row==10367) || (param_kind==2 && param_row==287) ||
        ((param_kind==4 || param_kind==6) && param_row==863) || (param_kind==5 && param_row==95) ||
        (param_kind==7 && param_row==23)) begin
      param_row<=0;
      case(param_kind)
       1:param_kind<=2;2:param_kind<=4;4:param_kind<=5;5:param_kind<=6;6:param_kind<=7;
       default:begin resident<=1;state<=SOURCE;end
      endcase
     end else param_row<=param_row+1;
    end else parameter_stalls<=parameter_stalls+1;
    SOURCE: if(!in_bounds || source_valid) begin
     source_load_words<=source_load_words+1;
     if(in_bounds) external_source_words<=external_source_words+1;else padding_words<=padding_words+1;
     if(load_index==1535) state<=ORIGIN;else load_index<=load_index+1;
    end else source_load_stalls<=source_load_stalls+1;
    ORIGIN:begin origin_words<=origin_words+1;state<=LAUNCH;end
    LAUNCH:begin accepted<=0;core_seen<=0;consumer_seen<=0;state<=RUN;end
    RUN:begin
     if(cons_error) error<=1;
     if(result_valid && result_ready) begin
      if(result_address!=accepted) error<=1;
      accepted<=accepted+1;output_beats<=output_beats+1;
     end
     if(leaf_done) begin
      core_seen<=1;
      core_cycles<=core_cycles+64'(l_cycles);
      core_source_words<=core_source_words+64'(l_source_words);
      core_weight_words<=core_weight_words+64'(l_weight_words);
      core_psum_reads<=core_psum_reads+64'(l_psum_reads);
      core_psum_writes<=core_psum_writes+64'(l_psum_writes);
      core_sum_issues<=core_sum_issues+64'(l_sum_issues);
      core_update_issues<=core_update_issues+64'(l_update_issues);
      core_source_stalls<=core_source_stalls+64'(l_source_stalls);
      core_weight_stalls<=core_weight_stalls+64'(l_weight_stalls);
      core_output_stalls<=core_output_stalls+64'(l_output_stalls);
      core_masked_contexts<=core_masked_contexts+64'(l_masked_contexts);
      core_zero_contexts<=core_zero_contexts+64'(l_zero_contexts);
      core_pattern_copies<=core_pattern_copies+64'(l_pattern_copies);
      core_merge_issues<=core_merge_issues+64'(l_merge_issues);
      core_z_reads<=core_z_reads+64'(l_z_reads);
      core_z_writes<=core_z_writes+64'(l_z_writes);
      core_first_issues<=core_first_issues+64'(l_first_issues);
      core_mac_issues<=core_mac_issues+64'(l_mac_issues);
      core_second_weight_words<=core_second_weight_words+64'(l_second_weight_words);
      core_support_reads<=core_support_reads+64'(l_support_reads);
      core_delta_issues<=core_delta_issues+64'(l_delta_issues);
      core_prefix_issues<=core_prefix_issues+64'(l_prefix_issues);
      core_local_source_reads<=core_local_source_reads+64'(l_local_source_reads);
     end
     if(cons_done) begin
      consumer_seen<=1;
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
     end
     if((core_seen || leaf_done) && (consumer_seen || cons_done)) begin
      if(accepted!=480) begin error<=1;state<=FAILED;end
      else begin
       retired_tiles<=retired_tiles+1;
       if(tile_id==last_tile) state<=FINISH;
       else begin tile_id<=tile_id+1;load_index<=0;state<=SOURCE;end
      end
     end
    end
    FINISH:begin done<=1;state<=IDLE;end
    FAILED:begin done<=1;state<=IDLE;end
    default:begin error<=1;state<=FAILED;end
   endcase
  end
 end
endmodule
