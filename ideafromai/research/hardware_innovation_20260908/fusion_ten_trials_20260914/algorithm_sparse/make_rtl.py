from pathlib import Path
H=Path(__file__).resolve().parent;old=H.parents[1]/'r8_consumer_fusion_20260914/packed_rtl/packed_r8.sv'
s=old.read_text().replace('module packed_r8(','module lossy_r8(').replace('input logic [2:0] cfg_kind','input logic [3:0] cfg_kind')
s=s.replace('output logic [5:0] debug_state','output logic [31:0] encoder_cycles,prototype_reads,held_vectors,\n output logic [5:0] debug_state')
s=s.replace('DRAIN_SEND,FINISH} state_t','DRAIN_SEND,FINISH,EREAD,EDIFF,EABS,ERED1,ERED2,ERED3,ESCORE,EMAX,EWRITE,PROTO_READ} state_t')
s=s.replace('logic signed [12:0] scalar;','logic signed [13:0] scalar;')
s=s.replace('logic signed [18:0] multiply_coefficient','logic signed [15:0] multiply_coefficient')
pos=s.index(' task clear_counters;')
s=s[:pos]+''' // All modes receive the union resources: one encoder state bank and static K4 table.
 logic signed [12:0] current_z[0:7],reference_z[0:7],codebook[0:3][0:7];
 logic signed [13:0] difference[0:7];
 logic signed [31:0] work[0:7];
 logic [2:0] score_shift[0:7];
 logic [31:0] rank_threshold[0:7],time_rank_threshold[0:7],group_threshold[0:8],time_threshold;
 logic signed [31:0] prototype_mem[0:7][0:47];
 logic [1:0] codes[0:39],enc_code,best_code;
 logic [2:0] residual_rank[0:39];
 logic signed [13:0] residual[0:39];
 logic refresh[0:39],encode_residual,drop_or_hold;
 logic [31:0] best_score;
 logic [3:0] current_nnz;
 logic [2:0] max_rank;
 logic signed [12:0] approximate[0:7];
 logic changed;
 logic alu_cin[0:7];
''' +s[pos:]
s=s.replace('source_stalls<=0;weight_stalls<=0;output_stalls<=0;','source_stalls<=0;weight_stalls<=0;output_stalls<=0;encoder_cycles<=0;prototype_reads<=0;held_vectors<=0;')
s=s.replace('scalar=read_half?', 'scalar=read_half?') # sign extension is explicit by expression width context
s=s.replace(' for(integer l=0;l<8;l=l+1)begin\n multiply_coefficient', ''' if(state==ZSCAN && (mode_q==3 || mode_q==4)) begin
  scan_mask=0;if(residual[fp]!=0)scan_mask[residual_rank[fp]]=1;
 end
 if(mode_q==3 || mode_q==4)scalar=residual[fp];
 max_rank=0;for(integer i=1;i<8;i=i+1)if(work[i]>work[max_rank])max_rank=3'(i);
 current_nnz=0;changed=0;
 for(integer i=0;i<8;i=i+1)begin
  if(current_z[i]!=0)current_nnz=current_nnz+1;
  approximate[i]=current_z[i];
  if(mode_q==1 && drop_or_hold)approximate[i]=0;
  if(mode_q==2 && $unsigned(work[i]>>score_shift[i])<=rank_threshold[i])approximate[i]=0;
  if(mode_q==3 || mode_q==4)approximate[i]=(i==int'(max_rank))?current_z[i]:codebook[enc_code][i];
  if(mode_q==5 && fp%10!=0 && drop_or_hold)approximate[i]=reference_z[i];
  if(mode_q==6 && fp%10!=0 && $unsigned(work[i]>>score_shift[i])<=time_rank_threshold[i])approximate[i]=reference_z[i];
  changed=changed || approximate[i]!=reference_z[i];
 end
 for(integer l=0;l<8;l=l+1)begin
 multiply_coefficient''')
s=s.replace('multiply_coefficient[l]={{3{qblock[selected_rank][l][15]}},qblock[selected_rank][l]};','multiply_coefficient[l]=qblock[selected_rank][l];')
s=s.replace('lhs[l]=acc[l];rhs[l]=product[l];','''lhs[l]=acc[l];rhs[l]=product[l];alu_cin[l]=0;
 if(state==EDIFF)begin
 lhs[l]=32'(current_z[l]);
 if(mode_q==3 || mode_q==4)rhs[l]=~32'(codebook[enc_code][l]);
 else if(mode_q==5 || mode_q==6)rhs[l]=~32'(reference_z[l]);
 else rhs[l]=-32'sd1;
 alu_cin[l]=1;
 end
 if(state==EABS)begin
 lhs[l]=0;rhs[l]=difference[l][13]?~32'(difference[l]):32'(difference[l]);alu_cin[l]=difference[l][13];
 end
 if(state==ERED1)begin lhs[l]=0;rhs[l]=0;if(l<4)begin lhs[l]=work[2*l];rhs[l]=work[2*l+1];end end
 if(state==ERED2)begin lhs[l]=0;rhs[l]=0;if(l<2)begin lhs[l]=work[2*l];rhs[l]=work[2*l+1];end end
 if(state==ERED3)begin lhs[l]=0;rhs[l]=0;if(l==0)begin lhs[l]=work[0];rhs[l]=work[1];end end''')
s=s.replace("if(b==0)assign cin=1'b0;","if(b==0)assign cin=alu_cin[l];")
s=s.replace('state<=IDLE;mode_q<=14;','state<=IDLE;mode_q<=0;')
s=s.replace('rank_live<=0;block_live<=0;remaining<=0;pending<=0;', 'enc_code<=0;best_code<=0;best_score<=0;encode_residual<=0;drop_or_hold<=0;rank_live<=0;block_live<=0;remaining<=0;pending<=0;',1)
s=s.replace(' 6:k_live[cfg_addr[9:0]]<=cfg_data[0];',''' 6:k_live[cfg_addr[9:0]]<=cfg_data[0];
 7:begin
  if(cfg_addr==0)for(integer i=0;i<8;i=i+1)score_shift[i]<=cfg_data[i*32+:3];
  if(cfg_addr==1)for(integer i=0;i<8;i=i+1)rank_threshold[i]<=cfg_data[i*32+:32];
  if(cfg_addr==2)for(integer i=0;i<8;i=i+1)time_rank_threshold[i]<=cfg_data[i*32+:32];
  if(cfg_addr==3)time_threshold<=cfg_data[31:0];
  if(cfg_addr>=4 && cfg_addr<=12)group_threshold[cfg_addr-4]<=cfg_data[31:0];
 end
 8:for(integer i=0;i<8;i=i+1)codebook[cfg_addr[1:0]][i]<=cfg_data[i*32+:13];
 9:for(integer i=0;i<8;i=i+1)prototype_mem[i][cfg_addr[5:0]]<=cfg_data[i*32+:32];''')
# avoid repeated decode: first phase common packed15 for all modes
s=s.replace('if(mode_q==15)',"if(1'b1)")
s=s.replace('if(state!=IDLE)cycles<=cycles+1;', 'if(state!=IDLE)cycles<=cycles+1;\n if(state>=EREAD && state<=EWRITE)encoder_cycles<=encoder_cycles+1;')
s=s.replace('KNEXT:if(k==863)begin fp<=0;state<=ZSCAN;end','''KNEXT:if(k==863)begin fp<=0;state<=(mode_q==0)?ZSCAN:EREAD;end''')
s=s.replace(' POSLOAD:begin\n for(integer i=0;i<8;i=i+1)acc[i]<=0;\n remaining<=position_live[fp]&block_live;\n state<=((position_live[fp]&block_live)==0)?STORE:BASE_MAC;\n end',''' POSLOAD:begin
 remaining<=position_live[fp]&block_live;
 if((mode_q==5 || mode_q==6) && !refresh[fp])begin state<=STORE;held_vectors<=held_vectors+1;end
 else begin
 for(integer i=0;i<8;i=i+1)acc[i]<=0;
 if(mode_q==3 && codes[fp]!=0)state<=PROTO_READ;
 else state<=((position_live[fp]&block_live)==0)?STORE:BASE_MAC;
 end
 end
 PROTO_READ:if(weight_allow)begin
 for(integer i=0;i<8;i=i+1)acc[i]<=prototype_mem[i][og*4+int'(codes[fp])];
 weight_words<=weight_words+1;prototype_reads<=prototype_reads+1;
 state<=(remaining==0)?STORE:BASE_MAC;
 end else weight_stalls<=weight_stalls+1;''')
marker=' ZSCAN:begin'
insert=''' EREAD:begin
 for(integer i=0;i<8;i=i+1)begin
 z_hold[i]<=z_mem[i][read_zrow];
 current_z[i]<=read_half?$signed(z_mem[i][read_zrow][25:13]):$signed(z_mem[i][read_zrow][12:0]);
 end
 z_vector_reads<=z_vector_reads+1;enc_code<=0;best_code<=0;best_score<=32'hffffffff;
 encode_residual<=(mode_q==4);drop_or_hold<=0;
 state<=((mode_q==5 || mode_q==6) && fp%10==0)?EWRITE:EDIFF;
 end
 EDIFF:begin
 for(integer i=0;i<8;i=i+1)difference[i]<=add_y[i][13:0];state<=EABS;
 end
 EABS:begin
 for(integer i=0;i<8;i=i+1)work[i]<=add_y[i]<<<score_shift[i];
 if(mode_q==2 || mode_q==6)state<=EWRITE;
 else if(encode_residual)state<=EMAX;
 else state<=ERED1;
 end
 ERED1:begin for(integer i=0;i<4;i=i+1)work[i]<=add_y[i];state<=ERED2;end
 ERED2:begin for(integer i=0;i<2;i=i+1)work[i]<=add_y[i];state<=ERED3;end
 ERED3:begin work[0]<=add_y[0];state<=ESCORE;end
 ESCORE:begin
 if(mode_q==3)begin
 if($unsigned(work[0])<best_score)begin best_score<=32'(work[0]);best_code<=enc_code;end
 if(enc_code==3)begin
 enc_code<=($unsigned(work[0])<best_score)?enc_code:best_code;
 encode_residual<=1;state<=EDIFF;
 end else begin enc_code<=enc_code+1;state<=EDIFF;end
 end else begin
 drop_or_hold<=(mode_q==1)?($unsigned(work[0])<=group_threshold[current_nnz]):($unsigned(work[0])<=time_threshold);
 state<=EWRITE;
 end
 end
 EMAX:begin
 codes[fp]<=enc_code;residual_rank[fp]<=max_rank;residual[fp]<=difference[max_rank];state<=EWRITE;
 end
 EWRITE:begin
 for(integer i=0;i<8;i=i+1)begin
 z_mem[i][read_zrow]<=read_half?{approximate[i],z_hold[i][12:0]}:{z_hold[i][25:13],approximate[i]};
 reference_z[i]<=approximate[i];
 end
 refresh[fp]<=((mode_q!=5 && mode_q!=6) || fp%10==0 || changed);
 z_writes<=z_writes+1;
 if(fp==39)begin fp<=0;state<=ZSCAN;end else begin fp<=fp+1;state<=EREAD;end
 end
'''
s=s.replace(marker,insert+marker)
(H/'lossy_r8.sv').write_text(s)
