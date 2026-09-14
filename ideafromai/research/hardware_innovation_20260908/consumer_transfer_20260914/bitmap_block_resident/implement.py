from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent/'bitmap_pipeline'
s=(P/'decomp_core.sv').read_text()
s=s.replace('BM_NATIVE_ADD} state_t','BM_NATIVE_ADD,BM_GROUP} state_t')
s=s.replace('bitmap[0:39][0:53]','bitmap[0:39]')
s=s.replace('logic [53:0] bm_live[0:39],bm_pending,bm_after;','logic [39:0] bitmap_live,bm_pending,bm_after;')
a=s.index(' task bm_next;');b=s.index(' task plane_next;',a)
s=s[:a]+''' task advance_block;
 begin
 if(k==863)begin fp<=0;state<=ZSCAN;end
 else begin k<=k+1;state<=(k%9==8)?L_START:L_GATHER;end
 end endtask
 task bm_next;
 begin state<=BM_STORE;end endtask
'''+s[b:]
s=s.replace('bitmap[fp][bm_block]','bitmap[fp]')
s=s.replace("bm_pending-54'd1","bm_pending-40'd1")
s=s.replace('for(integer j=53;j>=0;j=j-1)begin\n  if(bm_live[fp][j])first_live=j;','for(integer j=39;j>=0;j=j-1)begin\n  if(bitmap_live[j])first_live=j;')
a=s.index(' L_GATHER:');b=s.index(' BM_SCAN:',a)
s=s[:a]+''' L_GATHER:if(mode_q!=14)begin
 if(k_live[k])local_source_reads<=local_source_reads+1;
 for(integer i=0;i<40;i=i+1)begin
  bitmap[i][k%16]<=k_live[k]&&local_source[(i/20+(k%9)/3)*4+(i/10)%2+k%3][i%10];
  if(k%16==0)bitmap_live[i]<=k_live[k]&&local_source[(i/20+(k%9)/3)*4+(i/10)%2+k%3][i%10];
  else bitmap_live[i]<=bitmap_live[i]||(k_live[k]&&local_source[(i/20+(k%9)/3)*4+(i/10)%2+k%3][i%10]);
 end
 aux_writes<=aux_writes+1;state<=KNEXT;
 end else if(!k_live[k])state<=KNEXT;
 else begin
 for(integer i=0;i<4;i=i+1)src_masks[i]<=local_source[(i/2+(k%9)/3)*4+i%2+k%3];
 local_source_reads<=local_source_reads+1;state<=CHECK;
 end
 BM_GROUP:begin
 bm_pending<=bitmap_live;bm_block<=k/16;fp<=first_live;
 if(bitmap_live==0)advance_block();else state<=BM_POS;
 end
 BM_POS:begin
 // One actual 208bit read; retain the opposite half for whole-word writeback.
 for(integer i=0;i<8;i=i+1)begin
  z_hold[i]<=z_mem[i][read_zrow];
  bm_acc[i]<=read_half?$signed(z_mem[i][read_zrow][25:13]):$signed(z_mem[i][read_zrow][12:0]);
 end
 z_vector_reads<=z_vector_reads+1;state<=BM_SCAN;
 end
'''+s[b:]
s=s.replace('state<=(mode_q==10&&bitmap_onehot)?BM_NATIVE_QREAD:((mode_q==12||(mode_q==13||mode_q==10))?BM_PIPE:BM_QREAD);','state<=bitmap_onehot?BM_NATIVE_QREAD:BM_PIPE;')
a=s.index(' BM_STORE:');b=s.index(' CHECK:',a)
s=s[:a]+''' BM_STORE:begin
 for(integer i=0;i<8;i=i+1)
 z_mem[i][read_zrow]<=read_half?{bm_acc[i][12:0],z_hold[i][12:0]}:{z_hold[i][25:13],bm_acc[i][12:0]};
 z_writes<=z_writes+1;bm_pending<=bm_after;
 if(bm_after==0)advance_block();else begin fp<=next_live;state<=BM_POS;end
 end
'''+s[b:]
a=s.index(' KNEXT:');b=s.index(' ZSCAN:',a)
s=s[:a]+''' KNEXT:if(mode_q!=14&&k%16==15)state<=BM_GROUP;
 else advance_block();
'''+s[b:]
# The single read expression supplies both the retained full z word and its selected field.
s=s.replace('logic [25:0] z_mem[0:7][0:19],z_hold[0:7];','logic [25:0] z_mem[0:7][0:19],z_hold[0:7],bm_z_word[0:7];')
s=s.replace('q_read_data[l]=q_read_enable?',"bm_z_word[l]=(state==BM_POS)?z_mem[l][read_zrow]:26'd0;\n q_read_data[l]=q_read_enable?")
s=s.replace('z_hold[i]<=z_mem[i][read_zrow];','z_hold[i]<=bm_z_word[i];')
s=s.replace('$signed(z_mem[i][read_zrow][25:13]):$signed(z_mem[i][read_zrow][12:0])','$signed(bm_z_word[i][25:13]):$signed(bm_z_word[i][12:0])')
s=s.replace('$signed(bm_z_word[i][25:13]):$signed(bm_z_word[i][12:0])',"32'($signed(bm_z_word[i][25:13])):32'($signed(bm_z_word[i][12:0]))")
# Unify the one z read port across every mutually exclusive use.
s=s.replace('bm_z_word[0:7]','z_read_word[0:7]')
s=s.replace('logic [10:0] source_addr;logic [4:0] read_zrow;', 'logic [10:0] source_addr;logic [4:0] read_zrow,z_read_addr;logic z_read_enable;')
s=s.replace(' scan_mask=0;', ''' z_read_addr=(state==ZREAD)?5'(zrow):read_zrow;
 z_read_enable=(state==ZREAD||state==ZSCAN||state==BASE_MAC||state==BM_POS);
 for(integer zi=0;zi<8;zi=zi+1)z_read_word[zi]=(z_read_enable&&(state!=BASE_MAC||selected_rank==3'(zi)))?z_mem[zi][z_read_addr]:26'd0;
 scan_mask=0;''')
s=s.replace('z_mem[i][read_zrow][25:13]','z_read_word[i][25:13]').replace('z_mem[i][read_zrow][12:0]','z_read_word[i][12:0]')
s=s.replace('z_mem[selected_rank][read_zrow]','z_read_word[selected_rank]')
s=s.replace("bm_z_word[l]=(state==BM_POS)?z_mem[l][read_zrow]:26'd0;\n ",'')
s=s.replace('bm_z_word[i]','z_read_word[i]').replace('z_hold[i]<=z_mem[i][zrow]','z_hold[i]<=z_read_word[i]')
# Same K16 cache interface; borrow the existing Q2 register tile before Q2 starts.
s=s.replace('bitmap_native_reads,bitmap_native_issues,','bitmap_native_reads,bitmap_native_issues,cache_reads,cache_writes,')
s=s.replace('BM_GROUP} state_t','BM_GROUP,BM_WFILL,BM_CACHED_POP} state_t')
s=s.replace('logic pop_execute;', 'logic [2:0] qblock_read_index;logic [15:0] qblock_read_word[0:7];logic qblock_read_enable;\n logic pop_execute;')
s=s.replace('bitmap_native_reads<=0;', 'cache_reads<=0;cache_writes<=0;bitmap_native_reads<=0;')
s=s.replace('(state==BM_QREAD||state==BM_PIPE)', '(state==BM_QREAD||state==BM_PIPE||state==BM_WFILL)')
s=s.replace('pop_execute=(state==BM_POP)||', 'pop_execute=(state==BM_CACHED_POP&&bp_live[bm_plane][bm_block])||(state==BM_POP)||')
s=s.replace(' for(integer l=0;l<8;l=l+1)begin\n q_read_data', " qblock_read_index=(state==BM_CACHED_POP)?3'(bm_plane):selected_rank;\n qblock_read_enable=(state==BASE_MAC)||(state==BM_CACHED_POP&&bp_live[bm_plane][bm_block]);\n for(integer l=0;l<8;l=l+1)begin\n qblock_read_word[l]=qblock_read_enable?qblock[qblock_read_index][l]:16'd0;\n q_read_data")
s=s.replace('qblock[selected_rank][l][15]','qblock_read_word[l][15]').replace('qblock[selected_rank][l]','qblock_read_word[l]')
s=s.replace('pop16(bm_hold&bpq_hold[l])', 'pop16(bm_hold&((state==BM_CACHED_POP)?qblock_read_word[l]:bpq_hold[l]))')
s=s.replace('bm_pending<=bitmap_live;bm_block<=k/16;fp<=first_live;', 'bm_pending<=bitmap_live;bm_block<=k/16;fp<=first_live;bm_plane<=0;')
s=s.replace('if(bitmap_live==0)advance_block();else state<=BM_POS;', 'if(bitmap_live==0)advance_block();else state<=(mode_q==8)?BM_WFILL:BM_POS;')
s=s.replace(' BM_POS:begin', ''' BM_WFILL:if(!bp_live[bm_plane][bm_block]||weight_allow)begin
 for(integer i=0;i<8;i=i+1)qblock[bm_plane][i]<=bp_live[bm_plane][bm_block]?$signed(bp_read_data[i]):16'sd0;
 cache_writes<=cache_writes+1;
 if(bp_live[bm_plane][bm_block])begin weight_words<=weight_words+1;aux_weight_words<=aux_weight_words+1;end
 if(bm_plane==2)begin bm_plane<=0;state<=BM_POS;end else bm_plane<=bm_plane+1;
 end else weight_stalls<=weight_stalls+1;
 BM_CACHED_POP:begin
 if(bp_live[bm_plane][bm_block])begin
 for(integer i=0;i<8;i=i+1)bm_acc[i]<=add_y[i];
 first_issues<=first_issues+1;aux_issues<=aux_issues+1;cache_reads<=cache_reads+1;
 end
 if(bm_plane==2)bm_next();else bm_plane<=bm_plane+1;
 end
 BM_POS:begin''')
s=s.replace('state<=bitmap_onehot?BM_NATIVE_QREAD:BM_PIPE;', 'state<=bitmap_onehot?BM_NATIVE_QREAD:((mode_q==8)?BM_CACHED_POP:BM_PIPE);')
(H/'decomp_core.sv').write_text(s)
t=(P/'tb.cpp').read_text().replace('<<",\\\"state_cycles\\\":["', '<<",\\\"cache_reads\\\":"<<d.cache_reads<<",\\\"cache_writes\\\":"<<d.cache_writes<<",\\\"state_cycles\\\":["')
(H/'tb.cpp').write_text(t)
r=(P/'run.py').read_text().replace('[14,15,11,12,13,10]','[14,9,8]')
(H/'run.py').write_text(r)
r=(P/'run_stream.py').read_text().replace('[14,15,13,10]','[14,9,8]')
(H/'run_stream.py').write_text(r)
