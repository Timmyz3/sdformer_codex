from pathlib import Path
p=Path(__file__).resolve().parent
s=(p.parent/'source_classifier.sv').read_text().replace('module source_classifier(','module frontier_source(')
s=s.replace('input logic start_pack32,start_prefetch,','input logic start_pack32,start_prefetch,start_frontier,')
s=s.replace('output logic producer_valid,output logic[6:0] producer_channel,output logic[9:0] producer_gate,','output logic producer_valid,output logic[6:0] producer_channel[10],output logic[9:0] producer_active,producer_gate,')
s=s.replace('output logic[31:0] count_channels,count_mac,','output logic[31:0] count_channels,count_batches,count_pairs,count_mac,')
s=s.replace('logic pack32,prefetch;', 'logic pack32,prefetch,frontier;')
s=s.replace('logic[1:0] xissued,xreceived;logic[127:0] xword[2];','logic[7:0] xissued,xreceived;logic[127:0] xword[4][2];\n logic[3:0] batch_channel[4],selected_channel[4];logic[2:0] batch_count;\n logic[1:0] lane_slot[10],selected_slot[10];logic[9:0] active_mask,selected_active;\n logic[15:0] frontier_channels;integer selected_count,active_count;logic[7:0] xneed;')
s=s.replace('logic signed[23:0] xscalar;', 'logic signed[23:0] xscalar[10];')
a='''   if(st==XFETCH)for(integer w=0;w<2;w++)begin
     integer addr,b;addr=(int'(group_id)*16+int'(channel))*2+w;b=addr%8;
     if(!xissued[w]&&!pending[b]&&!(mode<2&&prefetch&&cache_valid[b]&&cache_addr[b]==13'(addr)))begin req_valid[b]=1;req_addr[b]=13'(addr);end
   end'''
b='''   xneed=8'((1<<(2*int'(batch_count)))-1);
   if(st==XFETCH)for(integer i=0;i<4;i++)for(integer w=0;w<2;w++)begin
     integer addr,b;addr=(int'(group_id)*16+int'(batch_channel[i]))*2+w;b=addr%8;
     if(i<int'(batch_count) && !xissued[2*i+w]&&!pending[b]&&!req_valid[b]&&!(mode<2&&prefetch&&cache_valid[b]&&cache_addr[b]==13'(addr)))begin req_valid[b]=1;req_addr[b]=13'(addr);end
   end'''
assert a in s;s=s.replace(a,b)
s=s.replace('   next_static=-1;', '''   frontier_channels=0;selected_count=0;selected_active=0;
   for(integer t=0;t<10;t++)if(node_id[t]>=16 && node_ready[t])frontier_channels[node_var[t]]=1;
   for(integer i=0;i<4;i++)selected_channel[i]=0;
   for(integer r=0;r<16;r++)for(integer c=0;c<16;c++)
     if(frontier_channels[c] && int'(order_rank[int'(group_id)*16+c])==r && selected_count<4)begin
       selected_channel[selected_count]=4'(c);selected_count++;
     end
   for(integer t=0;t<10;t++)begin
     selected_slot[t]=0;
     for(integer i=0;i<4;i++)if(i<selected_count && node_id[t]>=16 && node_var[t]==selected_channel[i])begin selected_slot[t]=2'(i);selected_active[t]=1;end
   end
   active_count=0;for(integer t=0;t<10;t++)active_count+=int'(active_mask[t]);
   next_static=-1;''')
s=s.replace('pf_relevant[i]=node_id[i/2]>=16&&node_var[i/2]==channel;', 'pf_relevant[i]=node_id[i/2]>=16&&(frontier?active_mask[i/2]:(node_var[i/2]==channel));')
s=s.replace("   xscalar=$signed(xword[int'(source_t)/5][(int'(source_t)%5)*24+:24]);", "   for(integer t=0;t<10;t++)xscalar[t]=$signed(xword[lane_slot[t]][int'(source_t)/5][(int'(source_t)%5)*24+:24]);")
s=s.replace("   producer_valid=st==DECIDE;producer_channel=7'(int'(group_id)*16+int'(channel));producer_gate=0;producer_u=0;", "   producer_valid=st==DECIDE;producer_active=active_mask;producer_gate=0;producer_u=0;\n   for(integer t=0;t<10;t++)producer_channel[t]=7'(int'(group_id)*16+int'(batch_channel[lane_slot[t]]));")
s=s.replace('pack32<=0;prefetch<=0;', 'pack32<=0;prefetch<=0;frontier<=0;batch_count<=0;active_mask<=0;')
s=s.replace('count_channels<=0;count_mac<=0;', 'count_channels<=0;count_batches<=0;count_pairs<=0;count_mac<=0;')
s=s.replace('       if(pv[1])u[t]<=u[t]+', '       if(pv[1]&&active_mask[t])u[t]<=u[t]+')
a='''         if(st==XFETCH&&pending_addr[b]/2==13'(int'(group_id)*16+int'(channel)))begin
           xword[pending_addr[b][0]]<=rsp_data[b];xreceived[pending_addr[b][0]]<=1;xissued[pending_addr[b][0]]<=1;
         end'''
b='''         if(st==XFETCH)for(integer i=0;i<4;i++)if(i<int'(batch_count) && pending_addr[b]/2==13'(int'(group_id)*16+int'(batch_channel[i])))begin
           xword[i][pending_addr[b][0]]<=rsp_data[b];xreceived[i*2+int'(pending_addr[b][0])]<=1;xissued[i*2+int'(pending_addr[b][0])]<=1;
         end'''
assert a in s;s=s.replace(a,b)
s=s.replace("         if(st==XFETCH)xissued[req_addr[b][0]]<=1;", "         if(st==XFETCH)for(integer i=0;i<4;i++)if(i<int'(batch_count) && req_addr[b]/2==13'(int'(group_id)*16+int'(batch_channel[i])))xissued[i*2+int'(req_addr[b][0])]<=1;")
s=s.replace('prefetch<=start_prefetch;boot<=0;', 'prefetch<=start_prefetch;frontier<=start_frontier;boot<=0;')
s=s.replace("           end else begin channel<=4'(next_channel);xissued<=0;xreceived<=0;st<=XFETCH;end", """           end else begin
             channel<=4'(next_channel);xissued<=0;xreceived<=0;st<=XFETCH;
             batch_count<=frontier?3'(selected_count):3'd1;active_mask<=frontier?selected_active:10'h3ff;
             for(integer i=0;i<4;i++)batch_channel[i]<=frontier?selected_channel[i]:(i==0?4'(next_channel):4'd0);
             for(integer t=0;t<10;t++)lane_slot[t]<=frontier?selected_slot[t]:2'd0;
           end""")
s=s.replace("         else begin channel<=4'(next_channel);xissued<=0;xreceived<=0;st<=XFETCH;end", """         else begin
           channel<=4'(next_channel);batch_count<=1;batch_channel[0]<=4'(next_channel);active_mask<=10'h3ff;
           for(integer t=0;t<10;t++)lane_slot[t]<=0;
           xissued<=0;xreceived<=0;st<=XFETCH;
         end""")
s=s.replace('             xword[w]<=cache_word[b];xreceived[w]<=1;xissued[w]<=1;', '             xword[0][w]<=cache_word[b];xreceived[w]<=1;xissued[w]<=1;')
s=s.replace('if(xreceived==3 && pending==0)', 'if(xreceived==xneed && pending==0)')
s=s.replace('for(integer t=0;t<10;t++)product[0][t]<=$signed(a[t][source_t])*$signed(xscalar);\n         count_mac<=count_mac+10;', "for(integer t=0;t<10;t++)if(active_mask[t])product[0][t]<=$signed(a[t][source_t])*$signed(xscalar[t]);\n         count_mac<=count_mac+32'(active_count);")
s=s.replace('         count_channels<=count_channels+1;', "         count_channels<=count_channels+32'(batch_count);count_batches<=count_batches+1;count_pairs<=count_pairs+32'(active_count);")
s=s.replace('           raw_word[t][channel]<=producer_gate[t];', '           if(active_mask[t])raw_word[t][batch_channel[lane_slot[t]]]<=producer_gate[t];')
s=s.replace('if(mode>=2&&node_id[t]>=16&&node_var[t]==channel)', 'if(mode>=2&&active_mask[t]&&node_id[t]>=16&&node_var[t]==batch_channel[lane_slot[t]])')
assert s.count('$signed(a[t][source_t])*$signed(xscalar[t])')==1
(p/'frontier_source.sv').write_text(s)
