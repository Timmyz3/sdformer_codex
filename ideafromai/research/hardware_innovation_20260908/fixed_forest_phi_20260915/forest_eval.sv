module forest_eval(
 input logic clk,rst_n,
 input logic cfg_valid,input logic [5:0] cfg_row,cfg_parent,input logic [15:0] cfg_mask,
 input logic start_valid,output logic start_ready,input logic [2:0] start_mode,input logic start_lazy,
 output logic mem_req_valid,input logic mem_req_ready,output logic [2:0] mem_req_addr,
 input logic mem_rsp_valid,output logic mem_rsp_ready,input logic [255:0] mem_rsp_data,
 output logic out_valid,input logic out_ready,output logic [5:0] out_row,output logic [255:0] out_data,
 output logic done_valid,input logic done_ready,
 output logic [31:0] dbg_query,dbg_alu,dbg_fields,dbg_build,dbg_parent_reads,dbg_joins,
 output logic [31:0] dbg_neg_fields,dbg_pwp_rows,dbg_native_rows,
 output logic [3:0] dbg_state
);
 typedef enum logic[3:0] {IDLE,MREQ,MRSP,PICK,PREP,QUERY,APPLY,NEED,BUILD,JOIN,RUN,COMMIT,DONE} state_t;
 state_t st,after_mem;
 logic [2:0] mode,ma;
 logic lazy,center_loaded;logic [7:0] query_seen,query_eligible;logic [2:0] query_addr;
 logic [15:0] masks[40],centers[8];logic [5:0] parents[40];
 logic signed[7:0] weights[16][8],pwp[8][8],result[40][8];
 logic [7:0] pvalid;logic [39:0] issued,finished;
 logic [2:0] count;logic [1:0] prep_index,commit_index;
 logic [5:0] rid[4];logic [15:0] u[4],posmask[4],negmask[4];
 logic signed[7:0] acc[4][8];logic [3:0] chosen[4],best[4];logic [5:0] bestcost[4];
 logic [2:0] query_index;logic [2:0] build_id;logic [15:0] build_mask;logic signed[7:0] build_acc[8];
 logic [255:0] first_word;
 logic [7:0] alu_a[4][8],alu_b[4][8],alu_y[4][8];logic [3:0] alu_sub;
 integer selected_count,select_id[4],missing,bitpos[4],build_bit;
 logic [39:0] selected_mask;logic use_forest,use_phi,any_work,any_pattern;
 function automatic integer pop16(input logic[15:0] v);
   integer n;begin n=0;for(integer b=0;b<16;b++)n=n+int'(v[b]);return n;end
 endfunction
 function automatic integer first16(input logic[15:0] v);
   integer n;begin n=0;for(integer b=15;b>=0;b--)if(v[b])n=b;return n;end
 endfunction
 always_comb begin
   use_forest=(mode!=0 && mode!=2 && mode!=7);use_phi=(mode>=2);
   query_eligible=0;
   for(integer q=0;q<8;q++)if(centers[q]!=0 && !query_seen[q] && (mode!=5 || q<2))query_eligible[q]=1;
   query_addr=lazy?3'(first16({8'b0,query_eligible})):query_index;
   selected_count=0;selected_mask=0;
   for(integer j=0;j<4;j++)select_id[j]=0;
   for(integer j=0;j<40;j++)if(!issued[j] && (!use_forest || parents[j]==63 || finished[parents[j]]))begin
     if(selected_count<4)begin select_id[selected_count]=j;selected_mask[j]=1;selected_count++;end
   end
   missing=-1;any_work=0;any_pattern=0;
   for(integer b=0;b<4;b++)begin
     bitpos[b]=first16(posmask[b]!=0?posmask[b]:negmask[b]);
     if(b<int'(count))begin
       any_work|=(posmask[b]!=0 || negmask[b]!=0);
       if(chosen[b]<8)begin
         any_pattern=1;
         if(!pvalid[chosen[b][2:0]] && missing<0)missing=int'(chosen[b]);
       end
     end
   end
   build_bit=first16(build_mask);
   alu_sub=0;
   for(integer b=0;b<4;b++)begin
     for(integer n=0;n<8;n++)begin
       alu_a[b][n]=0;alu_b[b][n]=0;
       if(st==BUILD && b==0)begin alu_a[b][n]=build_acc[n];alu_b[b][n]=weights[build_bit][n];end
       if(st==JOIN && b<int'(count) && chosen[b]<8)begin alu_a[b][n]=acc[b][n];alu_b[b][n]=pwp[chosen[b][2:0]][n];end
       if(st==RUN && b<int'(count))begin
         alu_a[b][n]=acc[b][n];alu_b[b][n]=weights[bitpos[b]][n];alu_sub[b]=(posmask[b]==0);
       end
       // Eight lanes, each with four independent 8-bit fields and carry-ins.
       alu_y[b][n]=alu_a[b][n]+(alu_b[b][n]^{8{alu_sub[b]}})+8'(alu_sub[b]);
     end
   end
   start_ready=(st==IDLE);mem_req_valid=(st==MREQ);mem_req_addr=ma;mem_rsp_ready=(st==MRSP);
   out_valid=(st==COMMIT);out_row=rid[commit_index];out_data=0;
   for(integer n=0;n<8;n++)out_data[n*32+:32]=32'($signed(acc[commit_index][n]));
   done_valid=(st==DONE);dbg_state=st;
 end
 always_ff @(posedge clk or negedge rst_n)begin
   if(!rst_n)begin
     st<=IDLE;mode<=0;ma<=0;after_mem<=IDLE;issued<=0;finished<=0;pvalid<=0;
     lazy<=0;center_loaded<=0;query_seen<=0;
     count<=0;prep_index<=0;commit_index<=0;query_index<=0;build_id<=0;build_mask<=0;first_word<=0;
     dbg_query<=0;dbg_alu<=0;dbg_fields<=0;dbg_build<=0;dbg_parent_reads<=0;dbg_joins<=0;
     dbg_neg_fields<=0;dbg_pwp_rows<=0;dbg_native_rows<=0;
     for(integer b=0;b<4;b++)begin
       rid[b]<=0;u[b]<=0;posmask[b]<=0;negmask[b]<=0;chosen[b]<=8;best[b]<=8;bestcost[b]<=0;
       for(integer n=0;n<8;n++)acc[b][n]<=0;
     end
     for(integer n=0;n<8;n++)build_acc[n]<=0;
   end else begin
     if(cfg_valid && st==IDLE)begin masks[cfg_row]<=cfg_mask;parents[cfg_row]<=cfg_parent;end
     case(st)
       IDLE:if(start_valid)begin
         mode<=start_mode;issued<=0;finished<=0;pvalid<=0;ma<=0;after_mem<=PICK;st<=MREQ;
         lazy<=start_lazy;center_loaded<=0;query_seen<=0;
         dbg_query<=0;dbg_alu<=0;dbg_fields<=0;dbg_build<=0;dbg_parent_reads<=0;dbg_joins<=0;
         dbg_neg_fields<=0;dbg_pwp_rows<=0;dbg_native_rows<=0;
       end
       MREQ:if(mem_req_ready)st<=MRSP;
       MRSP:if(mem_rsp_valid)begin
         if(ma==0)begin first_word<=mem_rsp_data;ma<=1;st<=MREQ;end
         else if(ma==1)begin
           for(integer k=0;k<16;k++)for(integer n=0;n<8;n++)begin
             // W is a packed signed3 stream, 384 bits total across two words.
             if((k*8+n)*3<256)begin
               if((k*8+n)*3==255)weights[k][n]<=8'($signed({mem_rsp_data[1:0],first_word[255]}));
               else weights[k][n]<=8'($signed(first_word[(k*8+n)*3+:3]));
             end else weights[k][n]<=8'($signed(mem_rsp_data[(k*8+n)*3-256+:3]));
           end
           if(mode>=2 && !lazy)begin ma<=2;st<=MREQ;end else st<=PICK;
         end else if(ma==2)begin
           for(integer q=0;q<8;q++)centers[q]<=mem_rsp_data[q*16+:16];center_loaded<=1;st<=after_mem;
         end else begin
           for(integer q=0;q<4;q++)begin
             for(integer n=0;n<8;n++)pwp[(ma==3?0:4)+q][n]<=$signed(mem_rsp_data[q*64+n*8+:8]);
             pvalid[(ma==3?0:4)+q]<=1;
           end
           st<=after_mem;
         end
       end
       PICK:begin
         if(&finished)st<=DONE;
         else if(selected_count>0)begin
           count<=3'(selected_count);issued<=issued|selected_mask;prep_index<=0;
           for(integer b=0;b<4;b++)begin
             rid[b]<=6'(select_id[b]);chosen[b]<=8;best[b]<=8;posmask[b]<=0;negmask[b]<=0;
           end
           st<=PREP;
         end
       end
       PREP:begin
         logic[15:0] delta;delta=masks[rid[prep_index]];
         if(use_forest && parents[rid[prep_index]]!=63)begin
           delta=delta&~masks[parents[rid[prep_index]]];dbg_parent_reads<=dbg_parent_reads+1;
           for(integer n=0;n<8;n++)acc[prep_index][n]<=result[parents[rid[prep_index]]][n];
         end else for(integer n=0;n<8;n++)acc[prep_index][n]<=0;
         u[prep_index]<=delta;posmask[prep_index]<=delta;negmask[prep_index]<=0;bestcost[prep_index]<=6'(pop16(delta));
         if({1'b0,prep_index}+3'd1==count)begin
           logic complex_row;complex_row=pop16(delta)>1;
           for(integer b=0;b<4;b++)if(b<int'(prep_index) && pop16(u[b])>1)complex_row=1;
           query_index<=0;query_seen<=0;
           if(use_phi && (!lazy || complex_row))begin
             if(lazy && !center_loaded)begin ma<=2;after_mem<=QUERY;st<=MREQ;end
             else st<=QUERY;
           end else st<=RUN;
         end
         else prep_index<=prep_index+1;
       end
       QUERY:begin
         logic need_query;need_query=0;
         for(integer b=0;b<4;b++)if(b<int'(count) && pop16(u[b])>1)need_query=1;
         if(!need_query || (lazy && query_eligible==0))st<=RUN;
         else begin
           dbg_query<=dbg_query+1;
           for(integer b=0;b<4;b++)if(b<int'(count) && pop16(u[b])>1)begin
             integer cost;logic legal;
             cost=pop16(u[b]^centers[query_addr]);legal=centers[query_addr]!=0;
             if(mode==5)begin legal=legal && ((centers[query_addr]&~u[b])==0);cost=cost+1;end
             if(mode==6 || mode==7)cost=cost+1+(pvalid[query_addr]?0:2);
             if(legal && cost<int'(bestcost[b]))begin best[b]<={1'b0,query_addr};bestcost[b]<=6'(cost);end
           end
           if(lazy)begin
             query_seen[query_addr]<=1;
             if((query_eligible & ~(8'b1<<query_addr))==0)st<=APPLY;
           end else if(query_index==(mode==5?1:7))st<=APPLY;else query_index<=query_index+1;
         end
       end
       APPLY:begin
         integer uses;uses=0;
         for(integer b=0;b<4;b++)if(b<int'(count))begin
           chosen[b]<=best[b];
           if(best[b]<8)begin
             posmask[b]<=u[b]&~centers[best[b][2:0]];negmask[b]<=centers[best[b][2:0]]&~u[b];uses++;
           end
         end
         dbg_pwp_rows<=dbg_pwp_rows+32'(uses);dbg_native_rows<=dbg_native_rows+32'(int'(count)-uses);st<=NEED;
       end
       NEED:begin
         if(missing>=0)begin
           if(mode==4)begin
             build_id<=3'(missing);build_mask<=centers[missing];for(integer n=0;n<8;n++)build_acc[n]<=0;st<=BUILD;
           end else begin ma<=missing<4?3:4;after_mem<=NEED;st<=MREQ;end
         end else st<=any_pattern?JOIN:RUN;
       end
       BUILD:begin
         if(build_mask!=0)begin
           for(integer n=0;n<8;n++)build_acc[n]<=alu_y[0][n];
           build_mask<=build_mask&(build_mask-1'b1);dbg_build<=dbg_build+1;dbg_alu<=dbg_alu+1;dbg_fields<=dbg_fields+1;
         end else begin
           for(integer n=0;n<8;n++)pwp[build_id][n]<=build_acc[n];pvalid[build_id]<=1;st<=NEED;
         end
       end
       JOIN:begin
         integer used;used=0;
         for(integer b=0;b<4;b++)if(b<int'(count) && chosen[b]<8)begin
           for(integer n=0;n<8;n++)acc[b][n]<=alu_y[b][n];used++;
         end
         dbg_alu<=dbg_alu+1;dbg_joins<=dbg_joins+1;dbg_fields<=dbg_fields+32'(used);st<=RUN;
       end
       RUN:begin
         integer used,minus;used=0;minus=0;
         if(any_work)begin
           for(integer b=0;b<4;b++)if(b<int'(count))begin
             if(posmask[b]!=0)begin
               for(integer n=0;n<8;n++)acc[b][n]<=alu_y[b][n];
               posmask[b]<=posmask[b]&(posmask[b]-1'b1);used++;
             end else if(negmask[b]!=0)begin
               for(integer n=0;n<8;n++)acc[b][n]<=alu_y[b][n];
               negmask[b]<=negmask[b]&(negmask[b]-1'b1);used++;minus++;
             end
           end
           dbg_alu<=dbg_alu+1;dbg_fields<=dbg_fields+32'(used);dbg_neg_fields<=dbg_neg_fields+32'(minus);
         end else begin commit_index<=0;st<=COMMIT;end
       end
       COMMIT:if(out_ready)begin
         for(integer n=0;n<8;n++)result[rid[commit_index]][n]<=acc[commit_index][n];
         finished[rid[commit_index]]<=1;
         if({1'b0,commit_index}+3'd1==count)st<=PICK;else commit_index<=commit_index+1;
       end
       DONE:if(done_ready)st<=IDLE;
       default:st<=IDLE;
     endcase
   end
 end
endmodule
