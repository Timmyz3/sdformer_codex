module stream_wrapper (
  input logic clk, reset_n, go,
  input logic [2:0] mode,
  input logic [14:0] first_tile, tile_count,
  output logic busy, done, error,
  output logic parameter_request_valid,
  output logic parameter_request_mask,
  output logic [13:0] parameter_address,
  input logic parameter_valid,
  input logic [127:0] parameter_data,
  output logic source_request_valid,
  output logic [22:0] source_address,
  input logic source_valid,
  input logic [9:0] source_data,
  input logic compute_source_allow, compute_weight_allow,
  output logic result_valid,
  input logic result_ready,
  output logic [14:0] result_tile,
  output logic [8:0] result_address,
  output logic [255:0] result_data,
  output logic result_tile_last, result_job_last,
  output logic [31:0] retired_tiles,
  output logic [63:0] total_cycles, static_weight_words, static_mask_words,
  output logic [63:0] parameter_stalls, source_load_words, external_source_words,
  output logic [63:0] padding_words, origin_words, source_load_stalls, output_beats,
  output logic [63:0] core_cycles, core_source_words, core_weight_words,
  output logic [63:0] core_psum_reads, core_psum_writes, core_sum_issues,
  output logic [63:0] core_update_issues, core_merge_issues,
  output logic [63:0] core_source_stalls, core_weight_stalls, core_output_stalls
);
  typedef enum logic [3:0] {IDLE, LOAD_W, LOAD_MASK, LOAD_SOURCE, LOAD_ORIGIN,
                             LAUNCH, RUN_TILE, FINISH, FAILED} stream_state_t;
  stream_state_t state;
  logic resident;
  logic [2:0] mode_q;
  logic [14:0] tile_id, last_tile;
  logic [13:0] parameter_row;
  logic [10:0] load_index;
  logic [8:0] accepted_beats;
  integer native_y, native_x, native_c;
  logic in_bounds;
  logic signed [15:0] origin_y, origin_x;
  logic core_cfg_valid, core_start, leaf_result_valid, leaf_ready, leaf_done;
  logic [1:0] core_cfg_kind;
  logic [13:0] core_cfg_address;
  logic [127:0] core_cfg_data;
  logic [8:0] leaf_result_address;
  logic [255:0] leaf_result_data;
  logic [31:0] leaf_cycles, leaf_source_words, leaf_weight_words, leaf_psum_reads,
               leaf_psum_writes, leaf_sum_issues, leaf_update_issues, leaf_merge_issues,
               leaf_source_stalls, leaf_weight_stalls, leaf_output_stalls;
  // Unused leaf diagnostics are deliberately not separate wrapper resources.
  logic [31:0] unused_masked, unused_zero, unused_copies;

  always_comb begin
    busy=(state!=IDLE);
    parameter_request_valid=(state==LOAD_W || state==LOAD_MASK);
    parameter_request_mask=(state==LOAD_MASK);
    parameter_address=parameter_row;
    origin_y=16'(2*(int'(tile_id)/160)-1);
    origin_x=16'(2*(int'(tile_id)%160)-1);
    native_y=int'(origin_y)+(int'(load_index)%16)/4;
    native_x=int'(origin_x)+(int'(load_index)%16)%4;
    native_c=int'(load_index)/16;
    in_bounds=(native_y>=0 && native_y<240 && native_x>=0 && native_x<320);
    source_request_valid=(state==LOAD_SOURCE && in_bounds);
    source_address=in_bounds ? 23'(native_c*76800+native_y*320+native_x) : 23'd0;
    core_cfg_valid=0;core_cfg_kind=0;core_cfg_address=0;core_cfg_data=0;
    if(parameter_request_valid && parameter_valid) begin
      core_cfg_valid=1;core_cfg_kind=(state==LOAD_W)?2'd1:2'd2;
      core_cfg_address=parameter_row;core_cfg_data=parameter_data;
    end else if(state==LOAD_SOURCE && (!in_bounds || source_valid)) begin
      core_cfg_valid=1;core_cfg_kind=0;core_cfg_address={3'd0,load_index};
      core_cfg_data[9:0]=in_bounds?source_data:10'd0;
    end else if(state==LOAD_ORIGIN) begin
      core_cfg_valid=1;core_cfg_kind=3;core_cfg_data[31:0]={origin_x,origin_y};
    end
    core_start=(state==LAUNCH);
    result_valid=(state==RUN_TILE && leaf_result_valid);
    leaf_ready=(state==RUN_TILE && result_ready);
    result_tile=tile_id;result_address=leaf_result_address;result_data=leaf_result_data;
    result_tile_last=(leaf_result_address==479);
    result_job_last=(tile_id==last_tile && result_tile_last);
  end

  pair_parent_merge leaf (
    .clk(clk),.reset_n(reset_n),.cfg_valid(core_cfg_valid),.cfg_kind(core_cfg_kind),
    .cfg_addr(core_cfg_address),.cfg_data(core_cfg_data),.start(core_start),.mode(mode_q),
    .source_allow(compute_source_allow),.weight_allow(compute_weight_allow),
    .result_valid(leaf_result_valid),.result_ready(leaf_ready),.result_addr(leaf_result_address),
    .result_data(leaf_result_data),.done(leaf_done),.cycles(leaf_cycles),
    .source_words(leaf_source_words),.weight_words(leaf_weight_words),.psum_reads(leaf_psum_reads),
    .psum_writes(leaf_psum_writes),.sum_issues(leaf_sum_issues),.update_issues(leaf_update_issues),
    .source_stalls(leaf_source_stalls),.weight_stalls(leaf_weight_stalls),.output_stalls(leaf_output_stalls),
    .masked_contexts(unused_masked),.zero_contexts(unused_zero),.pattern_copies(unused_copies),
    .merge_issues(leaf_merge_issues)
  );

  always_ff @(posedge clk) begin
    if(!reset_n) begin
      state<=IDLE;resident<=0;mode_q<=5;tile_id<=0;last_tile<=0;
      parameter_row<=0;load_index<=0;accepted_beats<=0;done<=0;error<=0;
      retired_tiles<=0;total_cycles<=0;static_weight_words<=0;static_mask_words<=0;
      parameter_stalls<=0;source_load_words<=0;external_source_words<=0;padding_words<=0;
      origin_words<=0;source_load_stalls<=0;output_beats<=0;
      core_cycles<=0;core_source_words<=0;core_weight_words<=0;core_psum_reads<=0;core_psum_writes<=0;
      core_sum_issues<=0;core_update_issues<=0;core_merge_issues<=0;
      core_source_stalls<=0;core_weight_stalls<=0;core_output_stalls<=0;
    end else begin
      done<=0;
      if(state!=IDLE) total_cycles<=total_cycles+1;
      case(state)
        IDLE: if(go) begin
          error<=0;mode_q<=mode;tile_id<=first_tile;last_tile<=first_tile+tile_count-15'd1;
          parameter_row<=0;load_index<=0;accepted_beats<=0;
          retired_tiles<=0;total_cycles<=0;static_weight_words<=0;static_mask_words<=0;
          parameter_stalls<=0;source_load_words<=0;external_source_words<=0;padding_words<=0;
          origin_words<=0;source_load_stalls<=0;output_beats<=0;
          core_cycles<=0;core_source_words<=0;core_weight_words<=0;core_psum_reads<=0;core_psum_writes<=0;
          core_sum_issues<=0;core_update_issues<=0;core_merge_issues<=0;
          core_source_stalls<=0;core_weight_stalls<=0;core_output_stalls<=0;
          if(tile_count==0 || int'(first_tile)+int'(tile_count)>19200 || (mode!=5 && mode!=6))
            begin error<=1;state<=FAILED;end
          else state<=resident?LOAD_SOURCE:LOAD_W;
        end
        LOAD_W: if(parameter_valid) begin
          static_weight_words<=static_weight_words+1;
          if(parameter_row==10367) begin parameter_row<=0;state<=LOAD_MASK;end
          else parameter_row<=parameter_row+1;
        end else parameter_stalls<=parameter_stalls+1;
        LOAD_MASK: if(parameter_valid) begin
          static_mask_words<=static_mask_words+1;
          if(parameter_row==287) begin resident<=1;load_index<=0;state<=LOAD_SOURCE;end
          else parameter_row<=parameter_row+1;
        end else parameter_stalls<=parameter_stalls+1;
        LOAD_SOURCE: if(!in_bounds || source_valid) begin
          source_load_words<=source_load_words+1;
          if(in_bounds) external_source_words<=external_source_words+1;
          else padding_words<=padding_words+1;
          if(load_index==1535) state<=LOAD_ORIGIN;
          else load_index<=load_index+1;
        end else source_load_stalls<=source_load_stalls+1;
        LOAD_ORIGIN: begin origin_words<=origin_words+1;state<=LAUNCH;end
        LAUNCH: begin accepted_beats<=0;state<=RUN_TILE;end
        RUN_TILE: begin
          if(result_valid && result_ready) begin
            output_beats<=output_beats+1;accepted_beats<=accepted_beats+1;
            if(leaf_result_address!=accepted_beats) error<=1;
          end
          if(leaf_done) begin
            if(accepted_beats!=480) begin error<=1;state<=FAILED;end
            else begin
              retired_tiles<=retired_tiles+1;
              core_cycles<=core_cycles+64'(leaf_cycles);
              core_source_words<=core_source_words+64'(leaf_source_words);
              core_weight_words<=core_weight_words+64'(leaf_weight_words);
              core_psum_reads<=core_psum_reads+64'(leaf_psum_reads);
              core_psum_writes<=core_psum_writes+64'(leaf_psum_writes);
              core_sum_issues<=core_sum_issues+64'(leaf_sum_issues);
              core_update_issues<=core_update_issues+64'(leaf_update_issues);
              core_merge_issues<=core_merge_issues+64'(leaf_merge_issues);
              core_source_stalls<=core_source_stalls+64'(leaf_source_stalls);
              core_weight_stalls<=core_weight_stalls+64'(leaf_weight_stalls);
              core_output_stalls<=core_output_stalls+64'(leaf_output_stalls);
              if(tile_id==last_tile) state<=FINISH;
              else begin tile_id<=tile_id+1;load_index<=0;state<=LOAD_SOURCE;end
            end
          end
        end
        FINISH: begin done<=1;state<=IDLE;end
        FAILED: begin done<=1;state<=IDLE;end
        default: begin error<=1;state<=FAILED;end
      endcase
    end
  end
endmodule
