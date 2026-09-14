module rr_context(
 input logic clk,reset_n,cfg_valid,
 input logic [2:0] cfg_kind,input logic [10:0] cfg_addr,input logic [255:0] cfg_data,
 input logic start,input logic [3:0] mode,input logic source_allow,weight_allow,
 input logic range_ok,resource_grant,
 output logic fallback_used,compute_done,
 output logic [4:0] resource_request,
 output logic weight_is_q2,output logic [9:0] weight_address,input logic [255:0] weight_data,
 output logic [255:0] alu_lhs,alu_rhs,output logic [151:0] alu_coefficient,
 output logic signed [12:0] alu_scalar,output logic alu_mac,output logic [2:0] alu_format,
 input logic [255:0] alu_result,
 output logic [31:0] arbitration_stalls,repair_arbitration_stalls,normalization_arbitration_stalls,
 output logic result_valid,input logic result_ready,
 output logic [8:0] result_addr,output logic [255:0] result_data,output logic done,
 output logic [31:0] cycles,source_words,weight_words,second_weight_words,local_source_reads,
 output logic [31:0] z_vector_reads,z_scalar_reads,z_writes,first_issues,merged_updates,
 output logic [31:0] repair_issues,repair_fields,normalization_issues,
 output logic [31:0] psum_reads,psum_writes,mac_issues,
 output logic [31:0] source_stalls,weight_stalls,output_stalls,
 output logic [5:0] debug_state
);
 typedef enum logic [5:0] {IDLE,ZCLEAR,L_START,L_LOAD,L_GATHER,CHECK,QREAD,TIMESEL,
 ZREAD,ZADD,REPAIR_READ,REPAIR_ADD,KNEXT,NORMALIZE_READ,NORMALIZE_ADD,
 ZSCAN,VLOAD,POSLOAD,BASE_MAC,STORE,DRAIN_READ,DRAIN_SEND,FINISH} state_t;
 state_t state;logic [1:0] mode_q;
 logic [9:0] source_mem[0:1535],local_source[0:15],src_masks[0:3];
 logic signed [2:0] q_hold[0:7];
 logic signed [15:0] qblock[0:7][0:7];
 logic k_live[0:863],v_live[0:95];
 logic [51:0] z_mem[0:7][0:9],z_hold[0:7];
 logic signed [31:0] p_mem[0:7][0:479],acc[0:7];
 logic signed [1:0] correction[0:7][0:3],next_correction[0:7][0:3];
 logic [7:0] position_live[0:39],rank_live,block_live,remaining,scan_mask;
 logic [39:0] pending;logic signed [15:0] oy,ox;
 integer k,fp,zrow,og,qfill,row,load_xy,selected_time;
 logic [10:0] source_addr;logic [3:0] read_zrow;
 logic in_bounds,cfg_v_live,pair_sel;logic [3:0] active;
 logic [2:0] selected_rank;logic signed [12:0] scalar;
 logic signed [18:0] multiply_coefficient[0:7];
 logic signed [31:0] lhs[0:7],rhs[0:7],add_y[0:7];
 logic [415:0] z_read_bus;
 logic [3:0] z_read_address;
 logic repair_needed;
 logic [5:0] correction_count;

 task clear_counters;
 begin
 cycles<=0;arbitration_stalls<=0;repair_arbitration_stalls<=0;normalization_arbitration_stalls<=0;source_words<=0;weight_words<=0;second_weight_words<=0;local_source_reads<=0;
 z_vector_reads<=0;z_scalar_reads<=0;z_writes<=0;first_issues<=0;merged_updates<=0;
 repair_issues<=0;repair_fields<=0;normalization_issues<=0;
 psum_reads<=0;psum_writes<=0;mac_issues<=0;
 source_stalls<=0;weight_stalls<=0;output_stalls<=0;
 end
 endtask

 // Each bank has one explicit shared read address/expression. Scalar MAC reads
 // only its selected bank; scans and updates use the common 416-bit vector.
 always_comb begin
 source_addr=11'((k/9)*16+load_xy);
 in_bounds=(int'(oy)+load_xy/4>=0 && int'(oy)+load_xy/4<240 &&
 int'(ox)+load_xy%4>=0 && int'(ox)+load_xy%4<320);
 cfg_v_live=0;for(integer i=0;i<8;i=i+1)cfg_v_live=cfg_v_live||(cfg_data[i*32+:16]!=0);
 selected_time=0;for(integer i=39;i>=0;i=i-1)if(pending[i])selected_time=i;
 selected_rank=0;for(integer i=7;i>=0;i=i-1)if(remaining[i])selected_rank=3'(i);
 read_zrow=4'(fp%10);
 z_read_address=(state==ZSCAN || state==BASE_MAC)?read_zrow:4'(zrow);
 resource_request=0;
 case(state)
  ZCLEAR,ZREAD,ZSCAN,REPAIR_READ,NORMALIZE_READ:resource_request=5'b00100;
  L_LOAD:if(in_bounds)resource_request=5'b00001;
  QREAD:resource_request=5'b00010;
  VLOAD:if(rank_live[qfill]&&v_live[og*8+qfill])resource_request=5'b00010;
  ZADD,REPAIR_ADD,NORMALIZE_ADD,BASE_MAC:resource_request=5'b10100;
  STORE,DRAIN_READ:resource_request=5'b01000;
  default:begin end
 endcase
 weight_is_q2=(state==VLOAD);weight_address=weight_is_q2?10'(og*8+qfill):10'(k);
 alu_mac=(state==BASE_MAC);alu_format=0;
 if(state==REPAIR_ADD || state==NORMALIZE_ADD)alu_format=4;
 else if(state==ZADD)alu_format={1'b0,mode_q}+3'd1;
 end
 always_comb begin
 z_read_bus=0;
 for(integer i=0;i<8;i=i+1)
  if(resource_grant && (state==ZSCAN || state==ZREAD || state==REPAIR_READ || state==NORMALIZE_READ ||
     (state==BASE_MAC && selected_rank==3'(i))))
   z_read_bus[i*52+:52]=z_mem[i][z_read_address];
 scan_mask=0;
 if(state==ZSCAN)for(integer i=0;i<8;i=i+1)
  scan_mask[i]=(z_read_bus[i*52+13*(fp/10)+:13]!=0);
 scalar=(state==BASE_MAC)?$signed(z_read_bus[int'(selected_rank)*52+13*(fp/10)+:13]):13'sd0;
 alu_scalar=scalar;
 for(integer l=0;l<8;l=l+1)begin
  multiply_coefficient[l]={{3{qblock[selected_rank][l][15]}},qblock[selected_rank][l]};
  lhs[l]=acc[l];rhs[l]=0;
  if(state==REPAIR_ADD || state==NORMALIZE_ADD)begin
   lhs[l]=0;rhs[l]=0;
   for(integer p=0;p<4;p=p+1)begin
    lhs[l][p*5+:5]=z_hold[l][p*13+8+:5];
    if(state==REPAIR_ADD)rhs[l][p*5+:5]={{3{correction[l][p][1]}},correction[l][p]};
    else rhs[l][p*5+:5]=z_hold[l][p*13+7]?5'b11111:5'd0;
   end
  end else if(state==ZADD)begin
   lhs[l]=0;rhs[l]=0;
   if(mode_q==0)begin
    lhs[l]={6'b0,z_hold[l][26*int'(pair_sel)+:26]};
    rhs[l]={6'b0,(active[2*int'(pair_sel)+1]?{{10{q_hold[l][2]}},q_hold[l]}:13'd0),
                     (active[2*int'(pair_sel)]?{{10{q_hold[l][2]}},q_hold[l]}:13'd0)};
   end else if(mode_q==1)begin
    if(pair_sel)begin
     lhs[l][9:0]=z_hold[l][39+:10];
     rhs[l][9:0]=active[3]?{{7{q_hold[l][2]}},q_hold[l]}:10'd0;
    end else for(integer p=0;p<3;p=p+1)begin
     lhs[l][p*10+:10]=z_hold[l][p*13+:10];
     rhs[l][p*10+:10]=active[p]?{{7{q_hold[l][2]}},q_hold[l]}:10'd0;
    end
   end else for(integer p=0;p<4;p=p+1)begin
    lhs[l][p*8+:8]=z_hold[l][p*13+:8];
    rhs[l][p*8+:8]=active[p]?{{5{q_hold[l][2]}},q_hold[l]}:8'd0;
   end
  end
  alu_lhs[l*32+:32]=lhs[l];alu_rhs[l*32+:32]=rhs[l];
  alu_coefficient[l*19+:19]=multiply_coefficient[l];
 end
 result_valid=(state==DRAIN_SEND);result_addr=9'(row);debug_state=state;
 end

 always_comb begin
  for(integer lane=0;lane<8;lane=lane+1)add_y[lane]=alu_result[lane*32+:32];
 end
 always_comb begin
  correction_count=0;repair_needed=0;
  for(integer lane=0;lane<8;lane=lane+1)for(integer p=0;p<4;p=p+1)begin
   next_correction[lane][p]=0;
   if(state==ZADD && mode_q==2 && active[p] &&
      (z_hold[lane][p*13+7]==q_hold[lane][2]) &&
      (z_hold[lane][p*13+7]!=add_y[lane][p*8+7]))begin
    next_correction[lane][p]=z_hold[lane][p*13+7]?-2'sd1:2'sd1;
    repair_needed=1;
    correction_count=correction_count+6'd1;
   end
  end
 end

 always_ff @(posedge clk)begin
 if(!reset_n)begin
  state<=IDLE;mode_q<=0;done<=0;compute_done<=0;fallback_used<=0;oy<=0;ox<=0;
  k<=0;fp<=0;zrow<=0;og<=0;qfill<=0;row<=0;load_xy<=0;
  rank_live<=0;block_live<=0;remaining<=0;pending<=0;result_data<=0;active<=0;pair_sel<=0;clear_counters();
  for(integer i=0;i<4;i=i+1)src_masks[i]<=0;
  for(integer i=0;i<8;i=i+1)begin
   q_hold[i]<=0;z_hold[i]<=0;acc[i]<=0;
   for(integer p=0;p<4;p=p+1)correction[i][p]<=0;
  end
 end else begin
  done<=0;
  if(cfg_valid&&state==IDLE)case(cfg_kind)
   0:source_mem[cfg_addr[10:0]]<=cfg_data[9:0];
   3:begin oy<=cfg_data[15:0];ox<=cfg_data[31:16];end
   4:begin end
   5:begin
    v_live[cfg_addr[6:0]]<=cfg_v_live;
   end
   6:k_live[cfg_addr[9:0]]<=cfg_data[0];
   default:begin end
  endcase
  if(state!=IDLE)cycles<=cycles+1;
  if(resource_request!=0 && !resource_grant)begin
   if(resource_request[0]&&!source_allow)source_stalls<=source_stalls+1;
   else if(resource_request[1]&&!weight_allow)weight_stalls<=weight_stalls+1;
   else arbitration_stalls<=arbitration_stalls+1;
   if(state==REPAIR_READ || state==REPAIR_ADD)repair_arbitration_stalls<=repair_arbitration_stalls+1;
   if(state==NORMALIZE_READ || state==NORMALIZE_ADD)normalization_arbitration_stalls<=normalization_arbitration_stalls+1;
  end
  if(resource_request==0 || resource_grant)case(state)
   IDLE:if(start)begin
    compute_done<=0;
    mode_q<=(mode==1&&!range_ok)?2'd0:mode[1:0];fallback_used<=(mode==1&&!range_ok);
    zrow<=0;rank_live<=0;block_live<=0;state<=ZCLEAR;clear_counters();
   end
   ZCLEAR:begin
    for(integer i=0;i<8;i=i+1)z_mem[i][zrow]<=0;
    z_writes<=z_writes+1;
    if(zrow==9)begin k<=0;state<=L_START;end else zrow<=zrow+1;
   end
   L_START:begin load_xy<=0;state<=L_LOAD;end
   L_LOAD:if(!in_bounds||source_allow)begin
    local_source[load_xy]<=in_bounds?source_mem[source_addr]:10'd0;
    if(in_bounds)source_words<=source_words+1;
    if(load_xy==15)state<=L_GATHER;else load_xy<=load_xy+1;
   end else source_stalls<=source_stalls+1;
   L_GATHER:if(!k_live[k])state<=KNEXT;
   else begin
    for(integer i=0;i<4;i=i+1)src_masks[i]<=local_source[(i/2+(k%9)/3)*4+i%2+k%3];
    local_source_reads<=local_source_reads+1;state<=CHECK;
   end
   CHECK:begin
    if(mode_q==2)pending<={30'd0,(src_masks[0]|src_masks[1]|src_masks[2]|src_masks[3])};
    else if(mode_q==1)pending<={20'd0,src_masks[3],(src_masks[0]|src_masks[1]|src_masks[2])};
    else pending<={20'd0,(src_masks[2]|src_masks[3]),(src_masks[0]|src_masks[1])};
    state<=((src_masks[0]|src_masks[1]|src_masks[2]|src_masks[3])==0)?KNEXT:QREAD;
   end
   QREAD:if(weight_allow)begin
    for(integer i=0;i<8;i=i+1)q_hold[i]<=weight_data[i*32+:3];
    weight_words<=weight_words+1;state<=TIMESEL;
   end else weight_stalls<=weight_stalls+1;
   TIMESEL:if(pending!=0)begin
    zrow<=selected_time%10;pair_sel<=(selected_time>=10);
    for(integer p=0;p<4;p=p+1)
     active[p]<=src_masks[p][selected_time%10] &&
       (mode_q==2 || (mode_q==1 ? (p==3)==(selected_time>=10) : p/2==selected_time/10));
    state<=ZREAD;
   end else state<=KNEXT;
   ZREAD:begin
    for(integer i=0;i<8;i=i+1)z_hold[i]<=z_read_bus[i*52+:52];
    z_vector_reads<=z_vector_reads+1;state<=ZADD;
   end
   ZADD:begin
    for(integer i=0;i<8;i=i+1)begin
     if(mode_q==0)begin
      if(pair_sel)z_mem[i][zrow]<={add_y[i][25:0],z_hold[i][25:0]};
      else z_mem[i][zrow]<={z_hold[i][51:26],add_y[i][25:0]};
     end else if(mode_q==1)begin
      if(pair_sel)z_mem[i][zrow]<={{{3{add_y[i][9]}},add_y[i][9:0]},z_hold[i][38:0]};
      else z_mem[i][zrow]<={z_hold[i][51:39],{{3{add_y[i][29]}},add_y[i][29:20]},
                          {{3{add_y[i][19]}},add_y[i][19:10]},{{3{add_y[i][9]}},add_y[i][9:0]}};
     end else begin
      z_mem[i][zrow]<={z_hold[i][51:47],add_y[i][31:24],z_hold[i][38:34],add_y[i][23:16],
                      z_hold[i][25:21],add_y[i][15:8],z_hold[i][12:8],add_y[i][7:0]};
      for(integer p=0;p<4;p=p+1)correction[i][p]<=next_correction[i][p];
     end
    end
    first_issues<=first_issues+1;z_writes<=z_writes+1;
    merged_updates<=merged_updates+32'(active[0])+32'(active[1])+32'(active[2])+32'(active[3])-32'd1;
    pending<=pending&(pending-40'd1);
    if(mode_q==2 && repair_needed)begin
     repair_fields<=repair_fields+32'(correction_count);state<=REPAIR_READ;
    end else state<=TIMESEL;
   end
   REPAIR_READ,NORMALIZE_READ:begin
    for(integer i=0;i<8;i=i+1)z_hold[i]<=z_read_bus[i*52+:52];
    z_vector_reads<=z_vector_reads+1;
    state<=(state==REPAIR_READ)?REPAIR_ADD:NORMALIZE_ADD;
   end
   REPAIR_ADD,NORMALIZE_ADD:begin
    for(integer i=0;i<8;i=i+1)
     z_mem[i][zrow]<={add_y[i][19:15],z_hold[i][46:39],add_y[i][14:10],z_hold[i][33:26],
                     add_y[i][9:5],z_hold[i][20:13],add_y[i][4:0],z_hold[i][7:0]};
    z_writes<=z_writes+1;
    if(state==REPAIR_ADD)begin repair_issues<=repair_issues+1;state<=TIMESEL;end
    else begin
     normalization_issues<=normalization_issues+1;
     if(zrow==9)begin fp<=0;state<=ZSCAN;end else begin zrow<=zrow+1;state<=NORMALIZE_READ;end
    end
   end
   KNEXT:if(k==863)begin
    if(mode_q==2)begin zrow<=0;state<=NORMALIZE_READ;end else begin fp<=0;state<=ZSCAN;end
   end else begin k<=k+1;state<=(k%9==8)?L_START:L_GATHER;end
   ZSCAN:begin
    z_vector_reads<=z_vector_reads+1;position_live[fp]<=scan_mask;rank_live<=rank_live|scan_mask;
    if(fp==39)begin og<=0;qfill<=0;state<=VLOAD;end else fp<=fp+1;
   end
   VLOAD:begin
    if(!rank_live[qfill]||!v_live[og*8+qfill]||weight_allow)begin
     block_live[qfill]<=rank_live[qfill]&&v_live[og*8+qfill];
     for(integer i=0;i<8;i=i+1)qblock[qfill][i]<=(rank_live[qfill]&&v_live[og*8+qfill])?weight_data[i*32+:16]:16'sd0;
     if(rank_live[qfill]&&v_live[og*8+qfill])begin weight_words<=weight_words+1;second_weight_words<=second_weight_words+1;end
     if(qfill==7)begin fp<=0;state<=POSLOAD;end else qfill<=qfill+1;
    end else weight_stalls<=weight_stalls+1;
   end
   POSLOAD:begin
    for(integer i=0;i<8;i=i+1)acc[i]<=0;
    remaining<=position_live[fp]&block_live;
    state<=((position_live[fp]&block_live)==0)?STORE:BASE_MAC;
   end
   BASE_MAC:begin
    for(integer i=0;i<8;i=i+1)acc[i]<=add_y[i];
    mac_issues<=mac_issues+1;z_scalar_reads<=z_scalar_reads+1;
    remaining<=remaining&(remaining-8'd1);
    if((remaining&(remaining-8'd1))==0)state<=STORE;
   end
   STORE:begin
    for(integer i=0;i<8;i=i+1)p_mem[i][og*40+fp]<=acc[i];
    psum_writes<=psum_writes+1;
    if(fp<39)begin fp<=fp+1;state<=POSLOAD;end
    else if(og<11)begin og<=og+1;qfill<=0;state<=VLOAD;end
    else begin compute_done<=1;row<=0;state<=DRAIN_READ;end
   end
   DRAIN_READ:begin
    for(integer i=0;i<8;i=i+1)result_data[i*32+:32]<=p_mem[i][row];
    psum_reads<=psum_reads+1;state<=DRAIN_SEND;
   end
   DRAIN_SEND:if(result_ready)begin
    if(row==479)state<=FINISH;else begin row<=row+1;state<=DRAIN_READ;end
   end else output_stalls<=output_stalls+1;
   FINISH:begin done<=1;state<=IDLE;end
   default:state<=IDLE;
  endcase
 end
 end

 logic audit_blocked;
 logic [5:0] audit_state;
 logic [255:0] audit_acc;
 logic [415:0] audit_hold;
 logic [63:0] audit_correction;
 logic [39:0] audit_pending;
 logic [51:0] audit_z[0:7][0:9];
 logic [31:0] audit_p[0:7];
 integer audit_k,audit_fp,audit_zrow,audit_og,audit_qfill,audit_row,audit_load_xy;
 logic [8:0] audit_p_index;
 always_ff @(posedge clk)begin
  if(!reset_n) audit_blocked<=0;
  else begin
   if(audit_blocked)begin
    if(state!=audit_state || k!=audit_k || fp!=audit_fp || zrow!=audit_zrow || og!=audit_og ||
       qfill!=audit_qfill || row!=audit_row || load_xy!=audit_load_xy || pending!=audit_pending)
       $fatal(1,"denied context state advanced");
    for(integer al=0;al<8;al=al+1)begin
     if(acc[al]!=audit_acc[al*32+:32] || z_hold[al]!=audit_hold[al*52+:52])$fatal(1,"denied context data advanced");
     if(p_mem[al][audit_p_index]!=audit_p[al])$fatal(1,"denied psum committed");
     for(integer ar=0;ar<10;ar=ar+1)if(z_mem[al][ar]!=audit_z[al][ar])$fatal(1,"denied z committed");
     for(integer ap=0;ap<4;ap=ap+1)if(correction[al][ap]!=audit_correction[al*8+ap*2+:2])$fatal(1,"denied repair changed");
    end
   end
   audit_blocked<=resource_request!=0 && !resource_grant;
   audit_state<=state;audit_k<=k;audit_fp<=fp;audit_zrow<=zrow;audit_og<=og;audit_qfill<=qfill;
   audit_row<=row;audit_load_xy<=load_xy;audit_pending<=pending;
   audit_p_index<=9'((state==DRAIN_READ)?row:og*40+fp);
   for(integer al=0;al<8;al=al+1)begin
    audit_acc[al*32+:32]<=acc[al];audit_hold[al*52+:52]<=z_hold[al];
    audit_p[al]<=p_mem[al][(state==DRAIN_READ)?row:og*40+fp];
    for(integer ar=0;ar<10;ar=ar+1)audit_z[al][ar]<=z_mem[al][ar];
    for(integer ap=0;ap<4;ap=ap+1)audit_correction[al*8+ap*2+:2]<=correction[al][ap];
   end
  end
 end

endmodule
