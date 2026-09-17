"""Derive finite PDE from the frozen ordinary resident/frontier source producer."""
from pathlib import Path
P=Path(__file__).resolve().parent
s=(P.parent/'frontier_joined/retained_config/frontier_source.sv').read_text()
def sub(a,b):
 global s
 assert a in s,a[:100]
 s=s.replace(a,b)
def section(a,b,c):
 global s
 i=s.index(a);j=s.index(b,i);s=s[:i]+c+s[j:]
sub(' output logic[4:0] debug_state',' output logic[31:0] count_bound_cycles,count_popcount_ops,count_dominance_tests,count_reduce_cycles,\n output logic[4:0] debug_state')
sub('HAM,OUT,DONE}', 'HAM,OUT,DONE,B_DIST,B_DOM,B_REDUCE}')
sub('dict[96],info_mask[6],roots[12]','dict[96],info_mask[6]')
sub('logic[15:0] node_id[10],node_lo[10],node_hi[10];logic[3:0] node_var[10];logic[9:0] node_ready;', '''logic[4:0] node_id[10];logic[3:0] node_var[10];logic[9:0] node_ready;
 logic[15:0] known[10],survivors;logic[9:0] bound_pending;
 logic[3:0] canonical[96],bound_t,witness;
 logic[4:0] distance_hold[16],pc_value[16];logic[15:0] pc_input[16];
 logic[15:0] survivor_or,survivor_and,unresolved_bits;
 logic common_label;logic[3:0] survivor_label;integer eval_t,first_survivor,next_variable;''')
sub('logic[12:0] graph_addr[10],boot_addr;', 'logic[12:0] boot_addr;')
section(' function automatic logic[35:0] decode_node',' function automatic integer pop16','')
sub('   all_ready=&node_ready;all_terminal=1;next_channel=-1;', '   node_ready=10\'h3ff;all_ready=1;all_terminal=1;next_channel=-1;')
sub("     graph_addr[t]=13'((mode==3?1536:256)+(pack32?int'(node_id[t])/4:int'(node_id[t])/2));\n",'')
section('   if(st==LOOK)for(integer t=0;t<10;t++)begin','   frontier_channels=0;','')
section('     logic[15:0] child;integer b;', '     if(prefetch&&(st==MAC||st==DRAIN)', '''     integer b;
     pf_addr[i]=13'((int'(group_id)*16+next_static)*2+(i%2));b=int'(pf_addr[i][2:0]);
     pf_relevant[i]=mode<2&&i<2&&next_static>=0;
''')
section('   nearest=0;nearest_dist=17;', '   producer_valid=st==DECIDE;', '''   // Exactly sixteen physical popcount16 units, shared by static HAM,
   // the known-distance pass and the one-witness-per-cycle dominance pass.
   eval_t=-1;for(integer t=9;t>=0;t--)if(bound_pending[t])eval_t=t;
   for(integer k=0;k<16;k++)begin
     pc_input[k]=0;
     if(st==HAM)pc_input[k]=raw_word[ham_t]^dict[int'(group_id)*16+k];
     if(st==B_DIST&&eval_t>=0)pc_input[k]=(raw_word[eval_t]^dict[int'(group_id)*16+k])&known[eval_t];
     if(st==B_DOM)pc_input[k]=(dict[int'(group_id)*16+k]^dict[int'(group_id)*16+int'(witness)])&~known[bound_t];
     pc_value[k]=5'(pop16(pc_input[k]));
   end
   nearest=0;nearest_dist=17;
   for(integer k=0;k<16;k++)if(int'(pc_value[k])<nearest_dist)begin nearest_dist=int'(pc_value[k]);nearest=4'(k);end
   survivor_or=0;survivor_and=16'hffff;first_survivor=-1;common_label=1;survivor_label=0;
   for(integer k=0;k<16;k++)if(survivors[k])begin
     survivor_or|=dict[int'(group_id)*16+k];survivor_and&=dict[int'(group_id)*16+k];
     if(first_survivor<0)begin first_survivor=k;survivor_label=mode==3?canonical[int'(group_id)*16+k]:4'(k);end
     else if(survivor_label!=(mode==3?canonical[int'(group_id)*16+k]:4'(k)))common_label=0;
   end
   unresolved_bits=(survivor_or^survivor_and)&~known[bound_t];next_variable=-1;
   for(integer c=0;c<16;c++)if(unresolved_bits[c] &&
     (next_variable<0||order_rank[int'(group_id)*16+c]<order_rank[int'(group_id)*16+next_variable]))next_variable=c;
''')
sub('codes<=0;node_ready<=0;', 'codes<=0;bound_pending<=0;bound_t<=0;witness<=0;survivors<=0;')
sub('node_lo[t]<=0;node_hi[t]<=0;', 'known[t]<=0;')
sub('count_prefetch_words<=0;', 'count_prefetch_words<=0;count_bound_cycles<=0;count_popcount_ops<=0;count_dominance_tests<=0;count_reduce_cycles<=0;')
section('         if(pending_addr[b]>=256', '       end\n       if(req_valid[b]&&req_ready[b])begin', '''         if(pending_addr[b]<192&&mode<2&&prefetch)begin
           cache_addr[b]<=pending_addr[b];cache_word[b]<=rsp_data[b];cache_valid[b]<=1;
         end
''')
sub('         if(boot==30)for(integer j=0;j<6;j++)roots[(mode==3?6:0)+j]<=wd[j*16+:16];', '')
sub('if(boot>=32)for(integer j=0;j<32;j++)', 'if(boot>=32&&boot<35)for(integer j=0;j<32;j++)')
sub('         if(boot==34||(mode==0&&boot==28)||(mode==1&&boot==29))st<=GSTART;\n         else begin boot<=boot==16&&mode>=2?6\'d30:(boot==30?6\'d32:boot+1\'b1);st<=BREQ;end', '''         if(boot>=35&&boot<=37)for(integer j=0;j<32;j++)canonical[(int'(boot)-35)*32+j]<=wd[j*4+:4];
         if((mode==2&&boot==34)||(mode==3&&boot==37)||(mode==0&&boot==28)||(mode==1&&boot==29))st<=GSTART;
         else begin boot<=boot==28&&mode>=2?6'd32:boot+1'b1;st<=BREQ;end''')
section('       GSTART:begin','       SELECT:begin', '''       GSTART:begin
         channel_cursor<=0;codes<=0;bound_pending<=10'h3ff;
         for(integer t=0;t<10;t++)begin raw_word[t]<=0;known[t]<=0;node_id[t]<=16;node_var[t]<=0;end
         st<=mode>=2?B_DIST:SELECT;
       end
       B_DIST:begin
         count_bound_cycles<=count_bound_cycles+1;
         if(eval_t<0)st<=LOOK;
         else begin
           bound_t<=4'(eval_t);witness<=0;survivors<=16'hffff;
           for(integer k=0;k<16;k++)distance_hold[k]<=pc_value[k];
           count_popcount_ops<=count_popcount_ops+16;st<=B_DOM;
         end
       end
       B_DOM:begin
         count_bound_cycles<=count_bound_cycles+1;count_popcount_ops<=count_popcount_ops+16;count_dominance_tests<=count_dominance_tests+16;
         for(integer k=0;k<16;k++)begin
           logic[5:0] dk,dj_plus_span;
           dk={1'b0,distance_hold[k]};
           dj_plus_span={1'b0,distance_hold[witness]}+{1'b0,pc_value[k]};
           // Strictly better full Hamming cost, or same cost and smaller original index.
           // Six bits are required: d_j + span can equal 32.
           if(dk>dj_plus_span||(dk==dj_plus_span&&int'(witness)<k))survivors[k]<=0;
         end
         if(witness==15)st<=B_REDUCE;else witness<=witness+1'b1;
       end
       B_REDUCE:begin
         count_bound_cycles<=count_bound_cycles+1;count_reduce_cycles<=count_reduce_cycles+1;
         assert(first_survivor>=0)else $fatal(1,"winner eliminated");
         if(common_label)node_id[bound_t]<={1'b0,survivor_label};
         else begin
           assert(next_variable>=0)else $fatal(1,"no distinguishing unknown channel");
           node_id[bound_t]<=16;node_var[bound_t]<=4'(next_variable);
         end
         bound_pending[bound_t]<=0;st<=B_DIST;
       end
       LOOK:begin
         if(pending==0)begin
           if(all_terminal)begin
             for(integer t=0;t<10;t++)codes[t*4+:4]<=node_id[t][3:0];
             st<=OUT;
           end else begin
             channel<=4'(next_channel);xissued<=allocated_received;xreceived<=allocated_received;st<=XFETCH;
             batch_count<=3'(selected_count);active_mask<=selected_active;
             batch_valid<=allocated_refs;slot_valid<=allocated_valid;fifo<=allocated_fifo;count_cache_hits<=count_cache_hits+32'(allocated_hits);
             for(integer i=0;i<4;i++)begin batch_channel[i]<=allocated_channel[i];slot_tag[i]<=allocated_tag[i];end
             for(integer t=0;t<10;t++)lane_slot[t]<=selected_slot[t];
           end
         end
       end
''')
section('           if(mode>=2&&active_mask[t]', '         end\n         channel_cursor', '''           if(mode>=2&&active_mask[t])known[t][batch_channel[lane_slot[t]]]<=1;
''')
sub("channel_cursor<=5'(int'(channel)+1);st<=mode>=2?LOOK:SELECT;", "channel_cursor<=5'(int'(channel)+1);bound_pending<=active_mask;st<=mode>=2?B_DIST:SELECT;")
sub('       HAM:begin codes', '       HAM:begin count_popcount_ops<=count_popcount_ops+16;codes')
(P/'distance_source.sv').write_text(s)
