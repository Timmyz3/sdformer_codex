`timescale 1ns/1ps
`default_nettype none
module tb_gatestack_replay_lifecycle_manager;
 logic clk_core,rst_core,session_valid,session_ready,session_context_id;
 logic [4:0] session_head_id; logic [31:0] session_tag;
 logic session_cache_owned,session_last_output_tile;
 logic decoder_done_valid,decoder_done_ready; logic [31:0] decoder_done_tag;
 logic decoder_done_error,backend_done_valid,backend_done_ready;
 logic [31:0] backend_done_tag; logic backend_done_error;
 logic slot_release_valid,slot_release_ready,slot_release_context_id;
 logic [4:0] slot_release_head_id; logic cache_release_valid;
 logic cache_release_ready,cache_release_context_id;
 logic [4:0] cache_release_head_id; logic session_done_valid;
 logic session_done_ready; logic [31:0] session_done_tag;
 logic session_done_error,protocol_error; logic [31:0] count_sessions;
 logic [31:0] count_final_tile_releases,count_cache_releases,count_session_errors;
 int slot_releases,cache_releases;
 gatestack_replay_lifecycle_manager dut(.*);
 always #5 clk_core<=~clk_core;
 always_ff @(posedge clk_core) begin
  if(rst_core) begin slot_releases<=0; cache_releases<=0; end
  else begin
   if(slot_release_valid&&slot_release_ready) begin
    if(slot_release_context_id!=session_context_id||
       slot_release_head_id!=session_head_id) $fatal(1,"slot release id");
    slot_releases<=slot_releases+1;
   end
   if(cache_release_valid&&cache_release_ready) begin
    if(cache_release_context_id!=session_context_id||
       cache_release_head_id!=session_head_id) $fatal(1,"cache release id");
    cache_releases<=cache_releases+1;
   end
  end
 end
 task automatic begin_session(input int id,input logic cache_owned,
                              input logic last_tile);
  begin
   @(negedge clk_core); session_context_id=id[0]; session_head_id=5'(id);
   session_tag=32'h8100_0000+32'(id); session_cache_owned=cache_owned;
   session_last_output_tile=last_tile; session_valid=1;
   do @(posedge clk_core); while(!session_ready);
   @(negedge clk_core); session_valid=0;
  end
 endtask
 task automatic send_decoder(input int id,input logic error_flag);
  begin @(negedge clk_core); decoder_done_tag=32'h8100_0000+32'(id);
   decoder_done_error=error_flag; decoder_done_valid=1;
   do @(posedge clk_core); while(!decoder_done_ready);
   @(negedge clk_core); decoder_done_valid=0; decoder_done_error=0; end
 endtask
 task automatic send_backend(input int id,input logic bad_tag);
  begin @(negedge clk_core); backend_done_tag=32'h8100_0000+32'(id)+32'(bad_tag);
   backend_done_error=0; backend_done_valid=1;
   do @(posedge clk_core); while(!backend_done_ready);
   @(negedge clk_core); backend_done_valid=0; end
 endtask
 task automatic accept_done(input int id,input logic expected_error);
  begin wait(session_done_valid);
   if(session_done_tag!=32'h8100_0000+32'(id)||
      session_done_error!=expected_error) $fatal(1,"session done mismatch");
   @(negedge clk_core); session_done_ready=1; @(posedge clk_core);
   @(negedge clk_core); session_done_ready=0; end
 endtask
 initial begin
  clk_core=0;rst_core=1;session_valid=0;session_context_id=0;
  session_head_id=0;session_tag=0;session_cache_owned=0;
  session_last_output_tile=0;decoder_done_valid=0;decoder_done_tag=0;
  decoder_done_error=0;backend_done_valid=0;backend_done_tag=0;
  backend_done_error=0;slot_release_ready=0;cache_release_ready=0;
  session_done_ready=0;repeat(5)@(posedge clk_core);rst_core=0;

  begin_session(0,1,0);
  fork send_backend(0,0); begin repeat(3)@(posedge clk_core);send_decoder(0,0);end join
  accept_done(0,0);
  if(slot_releases!=0||cache_releases!=0)$fatal(1,"nonfinal released");

  begin_session(1,1,1);
  fork send_decoder(1,0); begin repeat(2)@(posedge clk_core);send_backend(1,0);end join
  wait(slot_release_valid&&cache_release_valid);
  repeat(2)@(posedge clk_core);
  @(negedge clk_core);cache_release_ready=1;@(posedge clk_core);
  @(negedge clk_core);cache_release_ready=0;repeat(2)@(posedge clk_core);
  if(!slot_release_valid||cache_release_valid)$fatal(1,"release pending bits");
  @(negedge clk_core);slot_release_ready=1;@(posedge clk_core);
  @(negedge clk_core);slot_release_ready=0;accept_done(1,0);

  begin_session(2,0,1);
  fork send_decoder(2,0);send_backend(2,0);join
  wait(slot_release_valid);
  if(cache_release_valid)$fatal(1,"IPD requested cache release");
  @(negedge clk_core);slot_release_ready=1;@(posedge clk_core);
  @(negedge clk_core);slot_release_ready=0;accept_done(2,0);

  begin_session(3,0,0);
  fork send_decoder(3,0);send_backend(3,1);join
  accept_done(3,1);
  if(!protocol_error||count_sessions!=4||count_final_tile_releases!=2||
     count_cache_releases!=1||count_session_errors!=1||
     slot_releases!=2||cache_releases!=1)$fatal(1,"lifecycle counters");
  $display("PASS: lifecycle sessions=%0d final=%0d slot_release=%0d cache_release=%0d errors=%0d",
   count_sessions,count_final_tile_releases,slot_releases,cache_releases,
   count_session_errors);$finish;
 end
 initial begin repeat(5000)@(posedge clk_core);$fatal(1,"lifecycle timeout");end
endmodule
`default_nettype wire
