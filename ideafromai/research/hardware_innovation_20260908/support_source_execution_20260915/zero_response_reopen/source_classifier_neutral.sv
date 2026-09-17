module source_classifier(
 input logic clk,rst_n,start_valid,output logic start_ready,input logic[1:0] start_mode,input logic start_pack32,start_prefetch,
 output logic[7:0] req_valid,input logic[7:0] req_ready,output logic[12:0] req_addr[8],
 input logic[7:0] rsp_valid,output logic[7:0] rsp_ready,input logic[127:0] rsp_data[8],
 output logic out_valid,input logic out_ready,output logic[2:0] out_group,output logic[39:0] out_code,
 output logic producer_valid,output logic[6:0] producer_channel,output logic[9:0] producer_gate,
 output logic[479:0] producer_u,
 output logic done_valid,input logic done_ready,
 output logic[31:0] count_channels,count_mac,count_graph_words,count_graph_hits,count_source_words,count_prefetch_words,
 output logic[4:0] debug_state
);
 typedef enum logic[4:0]{IDLE,BREQ,BRSP,GSTART,LOOK,SELECT,XFETCH,CLEAR,MAC,DRAIN,DECIDE,HAM,OUT,DONE} state_t;
 state_t st;logic[1:0] mode;logic[5:0] boot;logic pack32,prefetch;logic[19:0] pf_done;
 logic signed[15:0] a[10][10];logic signed[47:0] tau[10];logic[15:0] dict[96],info_mask[6],roots[12];
 logic[3:0] order_rank[96];
 logic[2:0] group_id;logic[4:0] channel_cursor;logic[3:0] channel,source_t,ham_t;
 logic[15:0] raw_word[10];logic[39:0] codes;
 logic[15:0] node_id[10],node_lo[10],node_hi[10];logic[3:0] node_var[10];logic[9:0] node_ready;
 logic[7:0] pending,cache_valid;logic[12:0] pending_addr[8],cache_addr[8];logic[127:0] cache_word[8];
 logic[1:0] xissued,xreceived;logic[127:0] xword[2];
 logic signed[23:0] xscalar;logic signed[47:0] u[10];logic signed[39:0] product[2][10];logic[1:0] pv;
 logic all_ready,all_terminal;integer next_channel,next_static;logic[12:0] graph_addr[10],boot_addr;
 integer read_count,source_read_count;logic[3:0] nearest;integer nearest_dist;
 logic[12:0] pf_addr[20];logic[19:0] pf_relevant;
 function automatic logic[35:0] decode_node(input logic[127:0] word_d,input logic[15:0] id,input logic packed_mode);
   logic[63:0] v;begin
     if(packed_mode)begin v=64'(word_d[int'(id[1:0])*32+:32]);return {v[27:24],4'b0,v[23:12],4'b0,v[11:0]};end
     else begin v=word_d[int'(id[0])*64+:64];return v[35:0];end
   end
 endfunction
 function automatic integer pop16(input logic[15:0] x);
   integer n;begin n=0;for(integer i=0;i<16;i++)n+=int'(x[i]);return n;end
 endfunction
 always_comb begin
   start_ready=st==IDLE;done_valid=st==DONE;debug_state=st;
   boot_addr=13'(192+int'(boot)); // Both modes load their own root from the same physical word.
   req_valid=0;rsp_ready=0;
   for(integer b=0;b<8;b++)begin req_addr[b]=0;rsp_ready[b]=pending[b];end
   if(st==BREQ)begin req_valid[boot_addr[2:0]]=1;req_addr[boot_addr[2:0]]=boot_addr;end
   if(st==XFETCH)for(integer w=0;w<2;w++)begin
     integer addr,b;addr=(int'(group_id)*16+int'(channel))*2+w;b=addr%8;
     if(!xissued[w]&&!pending[b]&&!(mode<2&&prefetch&&cache_valid[b]&&cache_addr[b]==13'(addr)))begin req_valid[b]=1;req_addr[b]=13'(addr);end
   end
   all_ready=&node_ready;all_terminal=1;next_channel=-1;
   for(integer t=9;t>=0;t--)begin
     graph_addr[t]=13'((mode==3?1536:256)+(pack32?int'(node_id[t])/4:int'(node_id[t])/2));
     if(node_id[t]>=16)all_terminal=0;
   end
   if(mode>=2)begin
     for(integer t=0;t<10;t++)if(node_id[t]>=16 && node_ready[t] &&
         (next_channel<0||order_rank[int'(group_id)*16+int'(node_var[t])]<order_rank[int'(group_id)*16+next_channel]))next_channel=int'(node_var[t]);
   end else begin
     for(integer c=15;c>=0;c--)if(c>=int'(channel_cursor) && (mode==0||info_mask[group_id][c]))next_channel=c;
   end
   if(st==LOOK)for(integer t=0;t<10;t++)begin
     integer b;b=int'(graph_addr[t][2:0]);
     if(!node_ready[t]&&node_id[t]>=16 && !(cache_valid[b]&&cache_addr[b]==graph_addr[t]) && !pending[b]&&!req_valid[b])begin
       req_valid[b]=1;req_addr[b]=graph_addr[t];
     end
   end
   next_static=-1;
   for(integer c=15;c>=0;c--)if(c>int'(channel)&&(mode==0||info_mask[group_id][c]))next_static=c;
   pf_relevant=0;
   for(integer i=0;i<20;i++)begin
     logic[15:0] child;integer b;child=i%2==0?node_lo[i/2]:node_hi[i/2];
     pf_addr[i]=13'((mode==3?1536:256)+(pack32?int'(child)/4:int'(child)/2));b=int'(pf_addr[i][2:0]);
     pf_relevant[i]=node_id[i/2]>=16&&node_var[i/2]==channel&&child>=16;
     if(mode<2)begin
       pf_addr[i]=13'((int'(group_id)*16+next_static)*2+(i%2));b=int'(pf_addr[i][2:0]);
       pf_relevant[i]=i<2&&next_static>=0;
     end
     if(prefetch&&(st==MAC||st==DRAIN)&&pf_relevant[i]&&!pf_done[i]&&
        !(cache_valid[b]&&cache_addr[b]==pf_addr[i])&&!pending[b]&&!req_valid[b])begin
       req_valid[b]=1;req_addr[b]=pf_addr[i];
     end
   end
   xscalar=$signed(xword[int'(source_t)/5][(int'(source_t)%5)*24+:24]);
   nearest=0;nearest_dist=17;
   for(integer k=0;k<16;k++)begin
     integer d;d=pop16(raw_word[ham_t]^dict[int'(group_id)*16+k]);
     if(d<nearest_dist)begin nearest_dist=d;nearest=4'(k);end
   end
   producer_valid=st==DECIDE;producer_channel=7'(int'(group_id)*16+int'(channel));producer_gate=0;producer_u=0;
   for(integer t=0;t<10;t++)begin producer_gate[t]=u[t]>=tau[t];producer_u[t*48+:48]=u[t];end
   out_valid=st==OUT;out_group=group_id;out_code=codes;
 end
 always_ff @(posedge clk or negedge rst_n)begin
   if(!rst_n)begin
     st<=IDLE;mode<=0;pack32<=0;prefetch<=0;pf_done<=0;boot<=0;group_id<=0;channel_cursor<=0;channel<=0;source_t<=0;ham_t<=0;
     codes<=0;node_ready<=0;pending<=0;cache_valid<=0;xissued<=0;xreceived<=0;pv<=0;
     count_channels<=0;count_mac<=0;count_graph_words<=0;count_graph_hits<=0;count_source_words<=0;count_prefetch_words<=0;
     for(integer b=0;b<8;b++)begin pending_addr[b]<=0;cache_addr[b]<=0;end
     for(integer t=0;t<10;t++)begin raw_word[t]<=0;node_id[t]<=0;node_lo[t]<=0;node_hi[t]<=0;node_var[t]<=0;end
   end else begin
     pv<={pv[0],1'b0};
     for(integer t=0;t<10;t++)begin
       product[1][t]<=product[0][t];
       if(pv[1])u[t]<=u[t]+48'($signed(product[1][t]));
     end
     read_count=0;source_read_count=0;
     if(prefetch&&(st==MAC||st==DRAIN))for(integer i=0;i<20;i++)begin
       integer b;b=int'(pf_addr[i][2:0]);
       if(!pf_relevant[i]||(cache_valid[b]&&cache_addr[b]==pf_addr[i])||
          (req_valid[b]&&req_ready[b]&&req_addr[b]==pf_addr[i]))pf_done[i]<=1;
     end
     for(integer b=0;b<8;b++)begin
       if(rsp_valid[b]&&rsp_ready[b])begin
         pending[b]<=0;
         if(st==XFETCH&&pending_addr[b]/2==13'(int'(group_id)*16+int'(channel)))begin
           xword[pending_addr[b][0]]<=rsp_data[b];xreceived[pending_addr[b][0]]<=1;xissued[pending_addr[b][0]]<=1;
         end
         if(pending_addr[b]>=256||(pending_addr[b]<192&&mode<2&&prefetch))begin
           cache_addr[b]<=pending_addr[b];cache_word[b]<=rsp_data[b];cache_valid[b]<=1;
           for(integer t=0;t<10;t++)if(st==LOOK&&!node_ready[t]&&graph_addr[t]==pending_addr[b])begin
             logic[35:0] nd;nd=decode_node(rsp_data[b],node_id[t],pack32);
             node_lo[t]<=nd[15:0];node_hi[t]<=nd[31:16];node_var[t]<=nd[35:32];node_ready[t]<=1;
           end
         end
       end
       if(req_valid[b]&&req_ready[b])begin
         pending[b]<=1;pending_addr[b]<=req_addr[b];
         if(st==XFETCH)xissued[req_addr[b][0]]<=1;
         if(req_addr[b]<192)source_read_count++;
         if(req_addr[b]>=256)read_count++;
       end
     end
     count_graph_words<=count_graph_words+32'(read_count);count_source_words<=count_source_words+32'(source_read_count);
     if(st==MAC||st==DRAIN)count_prefetch_words<=count_prefetch_words+32'(read_count+source_read_count);
     case(st)
       IDLE:if(start_valid)begin
         st<=BREQ;mode<=start_mode;pack32<=start_pack32;prefetch<=start_prefetch;boot<=0;group_id<=0;pending<=0;cache_valid<=0;pv<=0;pf_done<=0;
         count_channels<=0;count_mac<=0;count_graph_words<=0;count_graph_hits<=0;count_source_words<=0;count_prefetch_words<=0;
       end
       BREQ:if(req_ready[boot_addr[2:0]])st<=BRSP;
       BRSP:if(rsp_valid[boot_addr[2:0]])begin
         logic[127:0] wd;wd=rsp_data[boot_addr[2:0]];
         if(boot<13)for(integer j=0;j<8;j++)if(int'(boot)*8+j<100)
           a[(int'(boot)*8+j)/10][(int'(boot)*8+j)%10]<=$signed(wd[j*16+:16]);
         if(boot>=13&&boot<17)for(integer j=0;j<16;j++)begin
           integer bi;bi=(int'(boot)-13)*16+j;
           if(bi<60)tau[bi/6][(bi%6)*8+:8]<=wd[j*8+:8];
         end
         if(boot>=17&&boot<29)for(integer j=0;j<8;j++)dict[(int'(boot)-17)*8+j]<=wd[j*16+:16];
         if(boot==29)for(integer j=0;j<6;j++)info_mask[j]<=wd[j*16+:16];
         if(boot==30)for(integer j=0;j<6;j++)roots[(mode==3?6:0)+j]<=wd[j*16+:16];
         if(boot>=32)for(integer j=0;j<32;j++)order_rank[(int'(boot)-32)*32+j]<=wd[j*4+:4];
         if(boot==34||(mode==0&&boot==28)||(mode==1&&boot==29))st<=GSTART;
         else begin boot<=boot==16&&mode>=2?6'd30:(boot==30?6'd32:boot+1'b1);st<=BREQ;end
       end
       GSTART:begin
         channel_cursor<=0;codes<=0;
         for(integer t=0;t<10;t++)begin
           raw_word[t]<=0;node_id[t]<=roots[(mode==3?6:0)+int'(group_id)];
           node_ready[t]<=roots[(mode==3?6:0)+int'(group_id)]<16;
         end
         st<=mode>=2?LOOK:SELECT;
       end
       LOOK:begin
         integer hits;hits=0;
         for(integer t=0;t<10;t++)if(!node_ready[t])begin
           integer b;b=int'(graph_addr[t][2:0]);
           if(node_id[t]<16)node_ready[t]<=1;
           else if(cache_valid[b]&&cache_addr[b]==graph_addr[t])begin
             logic[35:0] nd;nd=decode_node(cache_word[b],node_id[t],pack32);
             node_lo[t]<=nd[15:0];node_hi[t]<=nd[31:16];node_var[t]<=nd[35:32];node_ready[t]<=1;hits++;
           end
         end
         count_graph_hits<=count_graph_hits+32'(hits);
         if(all_ready&&pending==0)begin
           if(all_terminal)begin
             for(integer t=0;t<10;t++)codes[t*4+:4]<=node_id[t][3:0];
             st<=OUT;
           end else begin channel<=4'(next_channel);xissued<=0;xreceived<=0;st<=XFETCH;end
         end
       end
       SELECT:begin
         if(next_channel<0)begin ham_t<=0;st<=HAM;end
         else begin channel<=4'(next_channel);xissued<=0;xreceived<=0;st<=XFETCH;end
       end
       XFETCH:begin
         for(integer w=0;w<2;w++)begin
           integer addr,b;addr=(int'(group_id)*16+int'(channel))*2+w;b=addr%8;
           if(mode<2&&prefetch&&cache_valid[b]&&cache_addr[b]==13'(addr))begin
             xword[w]<=cache_word[b];xreceived[w]<=1;xissued[w]<=1;
           end
         end
         if(xreceived==3 && pending==0)st<=CLEAR;
       end
       CLEAR:begin for(integer t=0;t<10;t++)u[t]<=0;source_t<=0;pf_done<=0;st<=MAC;end
       MAC:begin
         pv[0]<=1;
         for(integer t=0;t<10;t++)product[0][t]<=$signed(a[t][source_t])*$signed(xscalar);
         count_mac<=count_mac+10;
         if(source_t==9)st<=DRAIN;else source_t<=source_t+1'b1;
       end
       DRAIN:if(pv==0 && !(|req_valid))st<=DECIDE;
       DECIDE:begin
         count_channels<=count_channels+1;
         for(integer t=0;t<10;t++)begin
           raw_word[t][channel]<=producer_gate[t];
           if(mode>=2&&node_id[t]>=16&&node_var[t]==channel)begin
             node_id[t]<=producer_gate[t]?node_hi[t]:node_lo[t];
             node_ready[t]<=(producer_gate[t]?node_hi[t]:node_lo[t])<16;
           end
         end
         channel_cursor<=5'(int'(channel)+1);st<=mode>=2?LOOK:SELECT;
       end
       HAM:begin codes[int'(ham_t)*4+:4]<=nearest;if(ham_t==9)st<=OUT;else ham_t<=ham_t+1'b1;end
       OUT:if(out_ready)begin if(group_id==5)st<=DONE;else begin group_id<=group_id+1'b1;st<=GSTART;end end
       DONE:if(done_ready)st<=IDLE;
       default:st<=IDLE;
     endcase
   end
 end
endmodule
