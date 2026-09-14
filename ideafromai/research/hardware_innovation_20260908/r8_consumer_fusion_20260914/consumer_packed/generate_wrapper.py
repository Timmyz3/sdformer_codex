from pathlib import Path
H=Path(__file__).resolve().parent
core=['cycles', 'source_words', 'weight_words', 'second_weight_words', 'local_source_reads', 'z_vector_reads', 'z_scalar_reads', 'z_writes', 'first_issues', 'dual_updates', 'psum_reads', 'psum_writes', 'mac_issues', 'source_stalls', 'weight_stalls', 'output_stalls']
cons=['cycles','raw_words','identity_words','coefficient_words','mul_issues','add_issues','round_issues','output_words','identity_stalls','raw_wait_cycles','join_wait_cycles','output_stalls','saturations','conversion_issues','conversion_saturations']
text='''module consumer_stream (
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
 output logic j_monitor_valid,
 output logic [8:0] j_monitor_address,
 output logic [255:0] j_monitor_data,
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
'''
text+=',\n'.join(' output logic [63:0] '+prefix+x for prefix,xs in [('core_',core),('consumer_',cons)] for x in xs)+'\n);\n'
text+=''' typedef enum logic [2:0] {IDLE, PARAMETERS, SOURCE, ORIGIN, LAUNCH, RUN, FINISH, FAILED} state_t;
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
 logic [10:0] core_cfg_addr;
 logic [255:0] core_cfg_data, raw_data;
 logic [8:0] raw_addr;
 logic cons_done, cons_error, cons_identity_request, cons_result_valid;
 logic [5:0] unused_debug_state;
 logic [8:0] cons_identity_addr;
'''
text+='\n'.join(' logic [31:0] '+prefix+x+';' for prefix,xs in [('l_',core),('c_',cons)] for x in xs)+'\n'
text+=''' always_comb begin
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
   core_cfg_valid=1;core_cfg_kind=param_kind[2:0];core_cfg_addr=param_row[10:0];core_cfg_data=parameter_data;
  end else if(state==SOURCE && (!in_bounds || source_valid)) begin
   core_cfg_valid=1;core_cfg_kind=0;core_cfg_addr=load_index;core_cfg_data[9:0]=in_bounds?source_data:10'd0;
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
 packed_r8 leaf (
  .clk(clk),.reset_n(reset_n),.cfg_valid(core_cfg_valid),.cfg_kind(core_cfg_kind),
  .cfg_addr(core_cfg_addr[10:0]),.cfg_data(core_cfg_data),.start(core_start),.mode(mode_q),
  .source_allow(compute_source_allow),.weight_allow(compute_weight_allow),
  .result_valid(raw_valid),.result_ready(raw_ready),.result_addr(raw_addr),.result_data(raw_data),.done(leaf_done),.debug_state(unused_debug_state),
'''
text+=',\n'.join('  .'+x+'(l_'+x+')' for x in core)+'\n );\n'
text+=''' i24_consumer consumer (
  .clk(clk),.reset_n(reset_n),.start(core_start),
  .cfg_valid(state==PARAMETERS && param_kind==7 && parameter_valid),
  .cfg_addr(param_row[4:0]),.cfg_data(parameter_data),
  .raw_valid(raw_valid),.raw_ready(raw_ready),.raw_addr(raw_addr),.raw_data(raw_data),
  .identity_request_valid(cons_identity_request),.identity_address(cons_identity_addr),
  .identity_valid(state==RUN && identity_valid),.identity_data(identity_data),
  .j_monitor_valid(j_monitor_valid),.j_monitor_address(j_monitor_address),.j_monitor_data(j_monitor_data),
  .result_valid(cons_result_valid),.result_ready(state==RUN && result_ready),
  .result_addr(result_address),.result_data(result_data),.done(cons_done),.error(cons_error),
'''
text+=',\n'.join('  .'+x+'(c_'+x+')' for x in cons)+'\n );\n'
reset='''   retired_tiles<=0;total_cycles<=0;static_words<=0;parameter_stalls<=0;
   source_load_words<=0;external_source_words<=0;padding_words<=0;origin_words<=0;source_load_stalls<=0;output_beats<=0;
'''
reset+='\n'.join('   '+prefix+x+'<=0;' for prefix,xs in [('core_',core),('consumer_',cons)] for x in xs)+'\n'
text+=''' always_ff @(posedge clk) begin
  if(!reset_n) begin
   state<=IDLE;resident<=0;mode_q<=14;tile_id<=0;last_tile<=0;param_kind<=4;param_row<=0;
   load_index<=0;accepted<=0;done<=0;error<=0;core_seen<=0;consumer_seen<=0;
'''+reset+'''  end else begin
   done<=0;if(state!=IDLE) total_cycles<=total_cycles+1;
   case(state)
    IDLE: if(go) begin
     error<=0;mode_q<=mode;tile_id<=first_tile;last_tile<=first_tile+tile_count-15'd1;
     param_kind<=4;param_row<=0;load_index<=0;accepted<=0;core_seen<=0;consumer_seen<=0;
'''+reset+'''     if(tile_count==0 || int'(first_tile)+int'(tile_count)>19200 || (mode!=14 && mode!=15)) begin error<=1;state<=FAILED;end
     else state<=resident?SOURCE:PARAMETERS;
    end
    PARAMETERS: if(parameter_valid) begin
     static_words<=static_words+1;
     if(((param_kind==4 || param_kind==6) && param_row==863) || (param_kind==5 && param_row==95) ||
        (param_kind==7 && param_row==23)) begin
      param_row<=0;
      case(param_kind)
       4:param_kind<=5;5:param_kind<=6;6:param_kind<=7;
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
'''
text+='\n'.join('      core_'+x+'<=core_'+x+"+64'(l_"+x+');' for x in core)+'\n     end\n     if(cons_done) begin\n      consumer_seen<=1;\n'
text+='\n'.join('      consumer_'+x+'<=consumer_'+x+"+64'(c_"+x+');' for x in cons)+'\n'
text+='''     end
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
'''
(H/'consumer_stream.sv').write_text(text)
(H/'counter_fields.json').write_text(__import__('json').dumps({'core':core,'consumer':cons},indent=2)+'\n')
