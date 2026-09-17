from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'frontier_source.sv').read_text()
s=s.replace('start_prefetch,start_frontier,','start_prefetch,start_frontier,start_resident,')
s=s.replace('count_channels,count_batches,count_pairs,count_mac,','count_channels,count_batches,count_pairs,count_mac,count_cache_hits,count_refetches,count_peak_slots,')
s=s.replace('logic pack32,prefetch,frontier;', 'logic pack32,prefetch,frontier,resident;')
s=s.replace('logic[15:0] frontier_channels;integer selected_count,active_count;logic[7:0] xneed;', '''logic[15:0] frontier_channels;integer selected_count,active_count;logic[7:0] xneed;
 logic[3:0] slot_valid,batch_valid,allocated_refs,allocated_valid;
 logic[6:0] slot_tag[4],allocated_tag[4];logic[1:0] fifo,allocated_fifo;
 logic[3:0] allocated_channel[4];logic[1:0] selected_physical[4];logic[7:0] allocated_received;
 logic[95:0] source_seen;integer allocated_hits,selected_pass,selected_victim,peak_next;''')
s=s.replace("   xneed=8'((1<<(2*int'(batch_count)))-1);", '   xneed=0;for(integer i=0;i<4;i++)xneed[i*2+:2]={2{batch_valid[i]}};')
s=s.replace("if(i<int'(batch_count) && !xissued[2*i+w]", "if(batch_valid[i] && !xissued[2*i+w]")
start=s.index('   frontier_channels=0;');end=s.index('   active_count=0;',start)
s=s[:start]+'''   frontier_channels=0;selected_count=0;selected_active=0;selected_pass=0;
   for(integer t=0;t<10;t++)if(node_id[t]>=16 && node_ready[t])frontier_channels[node_var[t]]=1;
   for(integer i=0;i<4;i++)selected_channel[i]=0;
   if(mode<2 || !frontier)begin
     if(next_channel>=0)begin selected_count=1;selected_channel[0]=4'(next_channel);end
   end else begin
     // Ordinary resident-first frontier scheduling, fixed D-rank within each pass.
     for(integer pass_id=0;pass_id<2;pass_id++)for(integer r=0;r<16;r++)for(integer c=0;c<16;c++)begin
       logic cached;cached=0;
       for(integer i=0;i<4;i++)if(resident && slot_valid[i] && slot_tag[i]==7'(int'(group_id)*16+c))cached=1;
       if(frontier_channels[c] && int'(order_rank[int'(group_id)*16+c])==r && selected_count<4 &&
          ((pass_id==0 && cached)||(pass_id==1&&!cached)))begin selected_channel[selected_count]=4'(c);selected_count++;end
     end
   end
   allocated_refs=0;allocated_valid=slot_valid;allocated_fifo=fifo;allocated_received=0;allocated_hits=0;selected_victim=-1;
   for(integer i=0;i<4;i++)begin allocated_tag[i]=slot_tag[i];allocated_channel[i]=batch_channel[i];selected_physical[i]=0;end
   // Reserve every selected resident slot before choosing any replacement.
   for(integer j=0;j<4;j++)if(j<selected_count)for(integer i=0;i<4;i++)
     if(resident && slot_valid[i] && slot_tag[i]==7'(int'(group_id)*16+int'(selected_channel[j])))begin
       selected_physical[j]=2'(i);allocated_refs[i]=1;allocated_received[i*2+:2]=2'b11;allocated_hits++;
     end
   for(integer j=0;j<4;j++)if(j<selected_count)begin
     logic hit;hit=0;
     for(integer i=0;i<4;i++)if(resident && slot_valid[i] && slot_tag[i]==7'(int'(group_id)*16+int'(selected_channel[j])))hit=1;
     if(!hit)begin
       selected_victim=-1;
       if(!resident)selected_victim=j;
       else begin
         for(integer i=3;i>=0;i--)if(!allocated_refs[i]&&!allocated_valid[i])selected_victim=i;
         if(selected_victim<0)for(integer k=3;k>=0;k--)if(!allocated_refs[(int'(allocated_fifo)+k)%4])selected_victim=(int'(allocated_fifo)+k)%4;
       end
       selected_physical[j]=2'(selected_victim);allocated_refs[selected_victim]=1;allocated_valid[selected_victim]=0;
       allocated_tag[selected_victim]=7'(int'(group_id)*16+int'(selected_channel[j]));allocated_fifo=2'(selected_victim+1);
     end
     allocated_channel[selected_physical[j]]=selected_channel[j];
   end
   for(integer t=0;t<10;t++)begin
     selected_slot[t]=selected_physical[0];
     for(integer i=0;i<4;i++)if(i<selected_count && node_id[t]>=16 && node_var[t]==selected_channel[i])begin selected_slot[t]=selected_physical[i];selected_active[t]=1;end
   end
   peak_next=0;for(integer i=0;i<4;i++)peak_next+=int'(slot_valid[i]|batch_valid[i]);
'''+s[end:]
s=s.replace('frontier<=0;batch_count<=0;', 'frontier<=0;resident<=0;slot_valid<=0;batch_valid<=0;fifo<=0;source_seen<=0;batch_count<=0;')
s=s.replace('count_channels<=0;count_batches<=0;', 'count_cache_hits<=0;count_refetches<=0;count_peak_slots<=0;count_channels<=0;count_batches<=0;')
s=s.replace("if(i<int'(batch_count) && pending_addr[b]/2", "if(batch_valid[i] && pending_addr[b]/2")
s=s.replace("if(i<int'(batch_count) && req_addr[b]/2", "if(batch_valid[i] && req_addr[b]/2")
s=s.replace('         if(req_addr[b]<192)source_read_count++;', '''         if(req_addr[b]<192)begin
           source_read_count++;
           if(!req_addr[b][0])begin source_seen[req_addr[b]/2]<=1;end
         end''')
# Count up to four repeated source-channel reads accepted in the same clock.
s=s.replace('     read_count=0;source_read_count=0;', '     read_count=0;source_read_count=0;\n     begin integer repeats;repeats=0;for(integer b=0;b<8;b++)if(req_valid[b]&&req_ready[b]&&req_addr[b]<192&&!req_addr[b][0]&&source_seen[req_addr[b]/2])repeats++;count_refetches<=count_refetches+32\'(repeats);end')
s=s.replace('frontier<=start_frontier;boot<=0;', 'frontier<=start_frontier;resident<=start_resident;slot_valid<=0;batch_valid<=0;fifo<=0;source_seen<=0;boot<=0;')
a="""             channel<=4'(next_channel);xissued<=0;xreceived<=0;st<=XFETCH;
             batch_count<=frontier?3'(selected_count):3'd1;active_mask<=frontier?selected_active:10'h3ff;
             for(integer i=0;i<4;i++)batch_channel[i]<=frontier?selected_channel[i]:(i==0?4'(next_channel):4'd0);
             for(integer t=0;t<10;t++)lane_slot[t]<=frontier?selected_slot[t]:2'd0;"""
b="""             channel<=4'(next_channel);xissued<=allocated_received;xreceived<=allocated_received;st<=XFETCH;
             batch_count<=3'(selected_count);active_mask<=frontier?selected_active:10'h3ff;
             batch_valid<=allocated_refs;slot_valid<=allocated_valid;fifo<=allocated_fifo;count_cache_hits<=count_cache_hits+32'(allocated_hits);
             for(integer i=0;i<4;i++)begin batch_channel[i]<=allocated_channel[i];slot_tag[i]<=allocated_tag[i];end
             for(integer t=0;t<10;t++)lane_slot[t]<=frontier?selected_slot[t]:selected_physical[0];"""
assert a in s;s=s.replace(a,b)
a="""           channel<=4'(next_channel);batch_count<=1;batch_channel[0]<=4'(next_channel);active_mask<=10'h3ff;
           for(integer t=0;t<10;t++)lane_slot[t]<=0;
           xissued<=0;xreceived<=0;st<=XFETCH;"""
b="""           channel<=4'(next_channel);batch_count<=1;active_mask<=10'h3ff;
           batch_valid<=allocated_refs;slot_valid<=allocated_valid;fifo<=allocated_fifo;count_cache_hits<=count_cache_hits+32'(allocated_hits);
           for(integer i=0;i<4;i++)begin batch_channel[i]<=allocated_channel[i];slot_tag[i]<=allocated_tag[i];end
           for(integer t=0;t<10;t++)lane_slot[t]<=selected_physical[0];
           xissued<=allocated_received;xreceived<=allocated_received;st<=XFETCH;"""
assert a in s;s=s.replace(a,b)
s=s.replace('         for(integer w=0;w<2;w++)begin\n           integer addr,b;addr=', '         for(integer i=0;i<4;i++)for(integer w=0;w<2;w++)begin\n           integer addr,b;addr=')
s=s.replace("addr=(int'(group_id)*16+int'(channel))*2+w;b=addr%8;\n           if(mode<2", "addr=(int'(group_id)*16+int'(batch_channel[i]))*2+w;b=addr%8;\n           if(batch_valid[i]&&mode<2")
s=s.replace('xword[0][w]<=cache_word[b];xreceived[w]<=1;xissued[w]<=1;', 'xword[i][w]<=cache_word[b];xreceived[i*2+w]<=1;xissued[i*2+w]<=1;')
s=s.replace('if(xreceived==xneed && pending==0)st<=CLEAR;', "if(xreceived==xneed && pending==0)begin st<=CLEAR;slot_valid<=resident?(slot_valid|batch_valid):4'b0;if(32'(peak_next)>count_peak_slots)count_peak_slots<=32'(peak_next);end")
(p/'frontier_source.sv').write_text(s)
