module joined_core(
 input logic clk,rst_n,start_valid,output logic start_ready,
 input logic[1:0] start_mode,input logic[1:0] start_last_hblock,input logic start_backend_dedup,
 output logic[7:0] req_valid,input logic[7:0] req_ready,output logic[13:0] req_addr[8],
 input logic[7:0] rsp_valid,output logic[7:0] rsp_ready,input logic[127:0] rsp_data[8],
 output logic out_valid,input logic out_ready,output logic[1:0] out_hblock,output logic[8:0] out_row,
 output logic[95:0] out_gate,output logic[2303:0] out_y,output logic[4607:0] out_u,
 output logic done_valid,input logic done_ready,
 output logic producer_valid,output logic[4:0] producer_p,output logic[6:0] producer_channel,
 output logic[9:0] producer_gate,output logic[479:0] producer_u,
 output logic code_valid,output logic[2:0] code_group,output logic[39:0] code_data,
 output logic bridge_write,output logic[8:0] bridge_row,output logic[2:0] bridge_group,output logic[15:0] bridge_data,
 output logic bridge_read,output logic[8:0] bridge_read_row,output logic[95:0] bridge_read_data,
 output logic[31:0] count_dictionary_words,count_source_config_words,count_source_x_words,count_source_graph_words,
 output logic[31:0] count_backend_config_words,count_backend_coeff_words,count_source_mac,count_source_channels,
 output logic[31:0] count_backend_mac,count_backend_updates,count_bridge_writes,count_bridge_reads,
 output logic[31:0] count_source_prefetch_words,
 output logic[3:0] debug_state,output logic[4:0] debug_source_state,output logic[3:0] debug_backend_state
);
 typedef enum logic[3:0]{IDLE,DREQ,DRSP,PSTART,PRUN,EXPAND,BSTART,BFEED,BRUN,DONE} state_t;
 state_t st;logic[1:0] mode,hblock,last_hblock;logic dedup;logic[4:0] point;logic[3:0] dict_word,expand_t;
 logic[2:0] expand_group;logic[39:0] expand_codes;logic[8:0] read_row;
 logic[15:0] dictionary[96];logic[15:0] gates[6][320];
 logic[13:0] daddr;logic source_phase,backend_phase;
 logic sstart_valid,sstart_ready,sout_valid,sout_ready,sdone_valid,sdone_ready;
 logic[1:0] smode;logic[7:0] sreq_valid,sreq_ready,srsp_valid,srsp_ready;logic[12:0] sreq_addr[8];
 logic[2:0] sout_group;logic[39:0] sout_code;
 logic[31:0] schannels,smac,sgraph,shits,sxwords,sprefetch;
 logic bstart_valid,bstart_ready,bsource_valid,bsource_ready,bdone_valid,bdone_ready;
 logic[2:0] bmode;logic[95:0] bsource_data;logic[7:0] breq_valid,breq_ready,brsp_valid,brsp_ready;logic[12:0] breq_addr[8];
 logic[31:0] bupdates,bmac,bcoeff,bjobs,bsrows,bquery,bzero,bwait,bpeakw,bpeakd,bskip;
 source_classifier source_core(
  .clk,.rst_n,.start_valid(sstart_valid),.start_ready(sstart_ready),.start_mode(smode),
  .start_pack32(1'b1),.start_prefetch(1'b1),.req_valid(sreq_valid),.req_ready(sreq_ready),.req_addr(sreq_addr),
  .rsp_valid(srsp_valid),.rsp_ready(srsp_ready),.rsp_data,
  .out_valid(sout_valid),.out_ready(sout_ready),.out_group(sout_group),.out_code(sout_code),
  .producer_valid,.producer_channel,.producer_gate,.producer_u,.done_valid(sdone_valid),.done_ready(sdone_ready),
  .count_channels(schannels),.count_mac(smac),.count_graph_words(sgraph),.count_graph_hits(shits),
  .count_source_words(sxwords),.count_prefetch_words(sprefetch),.debug_state(debug_source_state));
 support_fc1_zero backend_core(
  .clk,.rst_n,.start_valid(bstart_valid),.start_ready(bstart_ready),.start_mode(bmode),.start_hblock(hblock),
  .source_valid(bsource_valid),.source_ready(bsource_ready),.source_data(bsource_data),
  .mem_req_valid(breq_valid),.mem_req_ready(breq_ready),.mem_req_addr(breq_addr),
  .mem_rsp_valid(brsp_valid),.mem_rsp_ready(brsp_ready),.mem_rsp_data(rsp_data),
  .out_valid,.out_ready,.out_row,.out_gate,.out_y,.out_u,.done_valid(bdone_valid),.done_ready(bdone_ready),
  .dbg_updates(bupdates),.dbg_mac(bmac),.dbg_coeff_words(bcoeff),.dbg_jobs(bjobs),.dbg_source_rows(bsrows),.dbg_query(bquery),
  .dbg_zero_jobs(bzero),.dbg_psn_wait(bwait),.dbg_peak_words(bpeakw),.dbg_peak_desc(bpeakd),.dbg_bank_skips(bskip),
  .dbg_state(debug_backend_state));
 assign start_ready=st==IDLE;
 assign done_valid=st==DONE;
 assign debug_state=st;
 assign source_phase=st==PSTART||st==PRUN||st==EXPAND;
 assign backend_phase=st==BSTART||st==BFEED||st==BRUN;
 assign daddr=14'(6161+int'(dict_word));
 assign sreq_ready=source_phase?req_ready:8'b0;
 assign srsp_valid=source_phase?rsp_valid:8'b0;
 assign breq_ready=backend_phase?req_ready:8'b0;
 assign brsp_valid=backend_phase?rsp_valid:8'b0;
 assign sstart_valid=st==PSTART;
 assign smode=mode<2?2'd1:mode;
 assign sout_ready=st==PRUN;
 assign sdone_ready=st==PRUN;
 assign bstart_valid=st==BSTART;
 assign bmode=mode==0?3'd0:(dedup?3'd4:3'd2);
 assign bsource_valid=st==BFEED;
 assign bdone_ready=st==BRUN;
 for(genvar g=0;g<6;g++)assign bsource_data[g*16+:16]=gates[g][read_row];
 assign out_hblock=hblock;
 assign producer_p=point;
 assign code_valid=sout_valid&&sout_ready;
 assign code_group=sout_group;
 assign code_data=sout_code;
 assign bridge_write=st==EXPAND;
 assign bridge_row=9'(int'(point)*10+int'(expand_t));
 assign bridge_group=expand_group;
 assign bridge_data=dictionary[int'(expand_group)*16+int'(expand_codes[int'(expand_t)*4+:4])];
 assign bridge_read=bsource_valid&&bsource_ready;
 assign bridge_read_row=read_row;
 assign bridge_read_data=bsource_data;
 always_comb begin
  req_valid=0;rsp_ready=0;
  for(integer b=0;b<8;b++)begin
   req_addr[b]=0;
   if(source_phase)begin
    req_valid[b]=sreq_valid[b];rsp_ready[b]=srsp_ready[b];
    req_addr[b]=sreq_addr[b]<192?14'(int'(point)*192+int'(sreq_addr[b])):14'(5952+int'(sreq_addr[b]));
   end
   if(backend_phase)begin
    req_valid[b]=breq_valid[b];rsp_ready[b]=brsp_ready[b];
    req_addr[b]=14'(8192+int'(breq_addr[b]));
   end
  end
  if(st==DREQ)begin req_valid[daddr[2:0]]=1;req_addr[daddr[2:0]]=daddr;end
  if(st==DRSP)rsp_ready[daddr[2:0]]=1;
 end
 always_ff @(posedge clk or negedge rst_n)begin
  if(!rst_n)begin
   st<=IDLE;mode<=0;dedup<=0;hblock<=0;last_hblock<=0;point<=0;dict_word<=0;expand_t<=0;expand_group<=0;expand_codes<=0;read_row<=0;
   count_dictionary_words<=0;count_source_config_words<=0;count_source_x_words<=0;count_source_graph_words<=0;
   count_backend_config_words<=0;count_backend_coeff_words<=0;count_source_mac<=0;count_source_channels<=0;
   count_backend_mac<=0;count_backend_updates<=0;count_bridge_writes<=0;count_bridge_reads<=0;count_source_prefetch_words<=0;
  end else begin
   integer dx,dc,dg,bc,bw,dd;dx=0;dc=0;dg=0;bc=0;bw=0;dd=0;
   for(integer b=0;b<8;b++)if(req_valid[b]&&req_ready[b])begin
    if(st==DREQ)dd++;
    if(source_phase)begin
     if(sreq_addr[b]<192)dx++;else if(sreq_addr[b]>=256)dg++;else dc++;
    end
    if(backend_phase)begin if(debug_backend_state==4)bw++;else bc++;end
   end
   count_dictionary_words<=count_dictionary_words+32'(dd);
   count_source_x_words<=count_source_x_words+32'(dx);count_source_config_words<=count_source_config_words+32'(dc);
   count_source_graph_words<=count_source_graph_words+32'(dg);
   count_backend_config_words<=count_backend_config_words+32'(bc);count_backend_coeff_words<=count_backend_coeff_words+32'(bw);
   if(bridge_write)count_bridge_writes<=count_bridge_writes+1;
   if(bridge_read)count_bridge_reads<=count_bridge_reads+1;
   case(st)
    IDLE:if(start_valid)begin
     mode<=start_mode;dedup<=start_backend_dedup;last_hblock<=start_last_hblock;hblock<=0;point<=0;dict_word<=0;st<=DREQ;
     count_dictionary_words<=0;count_source_config_words<=0;count_source_x_words<=0;count_source_graph_words<=0;
     count_backend_config_words<=0;count_backend_coeff_words<=0;count_source_mac<=0;count_source_channels<=0;
     count_backend_mac<=0;count_backend_updates<=0;count_bridge_writes<=0;count_bridge_reads<=0;count_source_prefetch_words<=0;
    end
    DREQ:if(req_valid[daddr[2:0]]&&req_ready[daddr[2:0]])st<=DRSP;
    DRSP:if(rsp_valid[daddr[2:0]])begin
     for(integer j=0;j<8;j++)dictionary[int'(dict_word)*8+j]<=rsp_data[daddr[2:0]][j*16+:16];
     if(dict_word==11)st<=PSTART;else begin dict_word<=dict_word+1'b1;st<=DREQ;end
    end
    PSTART:if(sstart_ready)st<=PRUN;
    PRUN:begin
     if(sout_valid)begin expand_codes<=sout_code;expand_group<=sout_group;expand_t<=0;st<=EXPAND;end
     if(sdone_valid)begin
      count_source_mac<=count_source_mac+smac;count_source_channels<=count_source_channels+schannels;
      count_source_prefetch_words<=count_source_prefetch_words+sprefetch;
      if(point==31)begin read_row<=0;st<=BSTART;end else begin point<=point+1'b1;st<=PSTART;end
     end
    end
    EXPAND:begin
     gates[expand_group][bridge_row]<=bridge_data;
     if(expand_t==9)st<=PRUN;else expand_t<=expand_t+1'b1;
    end
    BSTART:if(bstart_ready)begin read_row<=0;st<=BFEED;end
    BFEED:if(bsource_ready)begin if(read_row==319)st<=BRUN;else read_row<=read_row+1'b1;end
    BRUN:if(bdone_valid)begin
     count_backend_mac<=count_backend_mac+bmac;count_backend_updates<=count_backend_updates+bupdates;
     if(hblock==last_hblock)st<=DONE;else begin hblock<=hblock+1'b1;read_row<=0;st<=BSTART;end
    end
    DONE:if(done_ready)st<=IDLE;
    default:st<=IDLE;
   endcase
  end
 end
endmodule
