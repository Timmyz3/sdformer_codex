module support_fc1(
 input logic clk,rst_n,
 input logic start_valid,output logic start_ready,input logic[2:0] start_mode,input logic[1:0] start_hblock,
 input logic source_valid,output logic source_ready,input logic[95:0] source_data,
 output logic[7:0] mem_req_valid,input logic[7:0] mem_req_ready,output logic[12:0] mem_req_addr[8],
 input logic[7:0] mem_rsp_valid,output logic[7:0] mem_rsp_ready,input logic[127:0] mem_rsp_data[8],
 output logic out_valid,input logic out_ready,output logic[8:0] out_row,
 output logic[95:0] out_gate,output logic[2303:0] out_y,output logic[4607:0] out_u,
 output logic done_valid,input logic done_ready,
 output logic[31:0] dbg_updates,dbg_mac,dbg_coeff_words,dbg_jobs,dbg_source_rows,dbg_query,
 output logic[31:0] dbg_zero_jobs,dbg_psn_wait,dbg_peak_words,dbg_peak_desc,dbg_bank_skips,
 output logic[3:0] dbg_state
);
 typedef enum logic[3:0]{IDLE,BREQ,BRSP,SOURCE,FC,PCLEAR,PREAD,PMAC,PDRAIN,POUT,DONE} state_t;
 state_t st;
 logic[2:0] mode;logic[1:0] hb;logic[2:0] section;logic[8:0] boot_word,source_row;
 logic[12:0] boot_addr;integer boot_limit;
 logic signed[15:0] a[10][10];logic signed[47:0] tau[10][96];
 logic[15:0] patterns[96];logic[6:0] lut_index[96],num_lut;
 logic[95:0] positive,constant_ch;logic[959:0] constant_gate;
 logic[7:0] directory[96];
 logic[3:0] class_map[96];
 logic[319:0] routes[192];logic[191:0] jobs_left,new_routes;
 logic signed[23:0] y[320][96];logic[319:0] yvalid[8];
 logic[127:0] payload[24];
 logic[4:0] alloc_tail;logic[5:0] used_words;logic[1:0] qhead,qtail;logic[2:0] qcount;
 logic[7:0] qjob[4];logic[4:0] qbase[4];logic[3:0] qwords[4];logic[12:0] qaddr[4];
 logic[11:0] qneed[4],qissued[4],qrecv[4];
 logic[7:0] pending;logic[1:0] pending_q[8];logic[3:0] pending_w[8];
 logic active;logic[319:0] consumers;logic signed[15:0] coeff[96];logic[7:0] active_banks;
 logic[1:0] fv;logic[8:0] fr[2];logic[7:0] fb[2];logic signed[23:0] fvalue[2][96];
 logic[4:0] p;logic[3:0] s,clear_t,output_t;logic[9:0] targets;
 logic signed[23:0] yhold[96];logic signed[47:0] u[10][96];
 logic[3:0] pv;logic[3:0] pt[4];logic signed[39:0] product[4][96];logic[2:0] busy[10];
 logic signed[47:0] add_a[96],add_b[96],add_y[96];
 logic[8:0] y_read_addr;logic[3:0] u_read_addr;
 logic signed[23:0] y_read_data[96];logic signed[47:0] u_read_data[96];
 integer next_job,active_row,next_t,new_nw,new_addr,req_q[8],req_w[8];
 logic[11:0] new_need;logic[7:0] loaded_banks;logic signed[15:0] loaded_coeff[96];
 logic push_job,pop_job,skip_job,fc_issue,psn_issue,any_y;
 function automatic integer pop16(input logic[15:0] x);
   integer z;begin z=0;for(integer k=0;k<16;k++)z+=int'(x[k]);return z;end
 endfunction
 function automatic integer ring24(input integer x);return x>=24?x-24:x;endfunction
 always_comb begin
   dbg_state=st;start_ready=st==IDLE;source_ready=st==SOURCE;done_valid=st==DONE;
   case(section)
    0:begin boot_addr=13'(8064+int'(boot_word));boot_limit=13;end
    1:begin boot_addr=13'(8077+int'(boot_word));boot_limit=12;end
    2:begin boot_addr=13'(6624+int'(hb)*360+int'(boot_word));boot_limit=360;end
    3:begin boot_addr=13'(8089+int'(hb)*9+int'(boot_word));boot_limit=9;end
    4:begin boot_addr=13'(8125+int'(hb)*6+int'(boot_word));boot_limit=6;end
    default:begin boot_addr=13'(8150+int'(hb)*3+int'(boot_word));boot_limit=3;end
   endcase
   new_routes=0;
   for(integer g=0;g<6;g++)begin
     integer match_id;logic[15:0] word_s;match_id=-1;word_s=source_data[g*16+:16];
     for(integer k=15;k>=0;k--)if(word_s==patterns[g*16+k])match_id=k;
     if(mode!=0 && match_id>=0 && pop16(word_s)>=2)
       new_routes[g*32+16+(mode==4?int'(class_map[g*16+match_id]):match_id)]=1;
     else for(integer k=0;k<16;k++)new_routes[g*32+k]=word_s[k];
   end
   next_job=-1;for(integer j=191;j>=0;j--)if(jobs_left[j])next_job=j;
   new_nw=0;new_addr=0;new_need=0;
   if(next_job>=0)begin
     if(next_job%32<16)begin
       new_nw=6;new_addr=(int'(hb)*96+(next_job/32)*16+next_job%32)*6;
     end else begin
       new_nw=mode==1?12:8;
       new_addr=2304+(int'(hb)*int'(num_lut)+int'(lut_index[(next_job/32)*16+next_job%32-16]))*new_nw;
     end
     new_need=12'((1<<new_nw)-1);
     if(mode==3 && next_job%32>=16)new_need={4'b0,directory[lut_index[(next_job/32)*16+next_job%32-16]]};
   end
   active_row=0;for(integer r=319;r>=0;r--)if(consumers[r])active_row=r;
   fc_issue=st==FC && active && consumers!=0 && active_banks!=0;
   pop_job=st==FC && qcount!=0 && qrecv[qhead]==qneed[qhead] &&
           (!active || active_banks==0 || (fc_issue && (consumers&(consumers-1'b1))==0));
   skip_job=st==FC && next_job>=0 && new_need==0;
   push_job=st==FC && next_job>=0 && !skip_job && (qcount<4 || pop_job) &&
            int'(used_words)+new_nw-(pop_job?int'(qwords[qhead]):0)<=24;
   loaded_banks=0;
   for(integer h=0;h<96;h++)begin
     integer wi,bi;wi=0;bi=0;loaded_coeff[h]=0;
     if(qjob[qhead]%32<16)begin wi=h/16;bi=(h%16)*8;
       loaded_coeff[h]=16'($signed(payload[ring24(int'(qbase[qhead])+wi)][bi+:8]));
     end else if(mode==1)begin wi=h/8;bi=(h%8)*16;
       loaded_coeff[h]=$signed(payload[ring24(int'(qbase[qhead])+wi)][bi+:16]);
     end else begin wi=h/12;bi=(h%12)*10;
       if(qneed[qhead][wi])loaded_coeff[h]=16'($signed(payload[ring24(int'(qbase[qhead])+wi)][bi+:10]));
     end
     if(loaded_coeff[h]!=0)loaded_banks[h/12]=1;
   end
   mem_req_valid=0;mem_rsp_ready=0;
   for(integer b=0;b<8;b++)begin
     mem_req_addr[b]=0;req_q[b]=-1;req_w[b]=0;
     if(st==FC)begin
       mem_rsp_ready[b]=pending[b];
       if(!pending[b] || (mem_rsp_valid[b] && mem_rsp_ready[b]))begin
         for(integer i=0;i<4;i++)if(i<int'(qcount))begin
           integer qi;qi=(int'(qhead)+i)%4;
           for(integer w=0;w<12;w++)if(req_q[b]<0 && qneed[qi][w] && !qissued[qi][w] &&
                  (int'(qaddr[qi])+w)%8==b)begin
             req_q[b]=qi;req_w[b]=w;mem_req_valid[b]=1;mem_req_addr[b]=13'(int'(qaddr[qi])+w);
           end
         end
       end
     end
   end
   if(st==BREQ)begin mem_req_valid[boot_addr[2:0]]=1;mem_req_addr[boot_addr[2:0]]=boot_addr;end
   if(st==BRSP)mem_rsp_ready[boot_addr[2:0]]=pending[boot_addr[2:0]];
   any_y=0;for(integer b=0;b<8;b++)any_y|=yvalid[b][int'(p)*10+int'(s)];
   next_t=-1;for(integer t=9;t>=0;t--)if(targets[t] && busy[t]==0)next_t=t;
   psn_issue=st==PMAC && next_t>=0;
   y_read_addr=st==FC?9'(active_row):(st==PREAD?9'(int'(p)*10+int'(s)):9'(int'(p)*10+int'(output_t)));
   u_read_addr=pv[3]?pt[3]:output_t;
   for(integer h=0;h<96;h++)begin
     y_read_data[h]=yvalid[h/12][y_read_addr]?y[y_read_addr][h]:24'sd0;
     u_read_data[h]=u[u_read_addr][h];
     add_a[h]=0;add_b[h]=0;
     if(fc_issue)begin
       add_a[h]=48'($signed(y_read_data[h]));
       if(fv[1] && int'(fr[1])==active_row && fb[1][h/12])add_a[h]=48'($signed(fvalue[1][h]));
       if(fv[0] && int'(fr[0])==active_row && fb[0][h/12])add_a[h]=48'($signed(fvalue[0][h]));
       add_b[h]=48'($signed(coeff[h]));
     end else if(pv[3])begin add_a[h]=u_read_data[h];add_b[h]=48'($signed(product[3][h]));end
     add_y[h]=add_a[h]+add_b[h];
   end
   out_valid=st==POUT;out_row=9'(int'(p)*10+int'(output_t));out_y=0;out_u=0;out_gate=0;
   for(integer h=0;h<96;h++)begin
     if(st==POUT)out_y[h*24+:24]=y_read_data[h];
     if(st==POUT)out_u[h*48+:48]=u_read_data[h];
     out_gate[h]=constant_ch[h]?constant_gate[int'(output_t)*96+h]:
           (positive[h]?(u_read_data[h]>=tau[output_t][h]):(u_read_data[h]<=tau[output_t][h]));
   end
 end
 always_ff @(posedge clk or negedge rst_n)begin
   if(!rst_n)begin
     st<=IDLE;mode<=0;hb<=0;section<=0;boot_word<=0;source_row<=0;num_lut<=0;
     pending<=0;jobs_left<=0;qhead<=0;qtail<=0;qcount<=0;alloc_tail<=0;used_words<=0;
     active<=0;consumers<=0;active_banks<=0;fv<=0;pv<=0;p<=0;s<=0;clear_t<=0;output_t<=0;targets<=0;
     dbg_updates<=0;dbg_mac<=0;dbg_coeff_words<=0;dbg_jobs<=0;dbg_source_rows<=0;dbg_query<=0;
     dbg_zero_jobs<=0;dbg_psn_wait<=0;dbg_peak_words<=0;dbg_peak_desc<=0;dbg_bank_skips<=0;
     for(integer b=0;b<8;b++)begin yvalid[b]<=0;pending_q[b]<=0;pending_w[b]<=0;end
     for(integer t=0;t<10;t++)busy[t]<=0;
     for(integer q=0;q<4;q++)begin qjob[q]<=0;qbase[q]<=0;qwords[q]<=0;qaddr[q]<=0;qneed[q]<=0;qissued[q]<=0;qrecv[q]<=0;pt[q]<=0;end
     for(integer z=0;z<2;z++)begin fr[z]<=0;fb[z]<=0;end
   end else begin
     fv<={fv[0],1'b0};pv<={pv[2:0],1'b0};
     fr[1]<=fr[0];fb[1]<=fb[0];
     for(integer h=0;h<96;h++)begin
       fvalue[1][h]<=fvalue[0][h];
       if(fv[1] && fb[1][h/12])y[fr[1]][h]<=fvalue[1][h];
       if(pv[3])u[pt[3]][h]<=add_y[h];
       for(integer k=1;k<4;k++)product[k][h]<=product[k-1][h];
     end
     for(integer b=0;b<8;b++)if(fv[1] && fb[1][b])yvalid[b][fr[1]]<=1;
     for(integer k=1;k<4;k++)pt[k]<=pt[k-1];
     for(integer t=0;t<10;t++)if(busy[t]!=0)busy[t]<=busy[t]-1'b1;
     if(st==FC)begin
       integer read_count;read_count=0;
       for(integer b=0;b<8;b++)begin
         if(mem_rsp_valid[b] && mem_rsp_ready[b])begin
           payload[ring24(int'(qbase[pending_q[b]])+int'(pending_w[b]))]<=mem_rsp_data[b];
           qrecv[pending_q[b]][pending_w[b]]<=1;pending[b]<=0;
         end
         if(mem_req_valid[b] && mem_req_ready[b])begin
           pending[b]<=1;pending_q[b]<=2'(req_q[b]);pending_w[b]<=4'(req_w[b]);
           qissued[req_q[b]][req_w[b]]<=1;read_count++;
         end
       end
       dbg_coeff_words<=dbg_coeff_words+32'(read_count);
       if(fc_issue)begin
         fv[0]<=1;fr[0]<=9'(active_row);fb[0]<=active_banks;
         for(integer h=0;h<96;h++)fvalue[0][h]<=add_y[h][23:0];
         consumers<=consumers&(consumers-1'b1);dbg_updates<=dbg_updates+1;
         if((consumers&(consumers-1'b1))==0)active<=0;
       end else if(active && active_banks==0)active<=0;
       if(pop_job)begin
         for(integer h=0;h<96;h++)coeff[h]<=loaded_coeff[h];
         active_banks<=loaded_banks;consumers<=routes[qjob[qhead]];active<=1;qhead<=qhead+1'b1;
       end
       if(push_job)begin
         qjob[qtail]<=8'(next_job);qbase[qtail]<=alloc_tail;qwords[qtail]<=4'(new_nw);qaddr[qtail]<=13'(new_addr);
         qneed[qtail]<=new_need;qissued[qtail]<=0;qrecv[qtail]<=0;qtail<=qtail+1'b1;
         alloc_tail<=5'(ring24(int'(alloc_tail)+new_nw));jobs_left[next_job]<=0;dbg_jobs<=dbg_jobs+1;
         if(mode==3 && next_job%32>=16)dbg_bank_skips<=dbg_bank_skips+32'(8-pop16({4'b0,new_need}));
       end
       if(skip_job)begin jobs_left[next_job]<=0;dbg_zero_jobs<=dbg_zero_jobs+1;end
       used_words<=6'(int'(used_words)+(push_job?new_nw:0)-(pop_job?int'(qwords[qhead]):0));
       qcount<=3'(int'(qcount)+int'(push_job)-int'(pop_job));
       if(int'(used_words)+(push_job?new_nw:0)-(pop_job?int'(qwords[qhead]):0)>int'(dbg_peak_words))
         dbg_peak_words<=32'(int'(used_words)+(push_job?new_nw:0)-(pop_job?int'(qwords[qhead]):0));
       if(int'(qcount)+int'(push_job)-int'(pop_job)>int'(dbg_peak_desc))dbg_peak_desc<=32'(int'(qcount)+int'(push_job)-int'(pop_job));
     end
     case(st)
      IDLE:if(start_valid)begin
       st<=BREQ;mode<=start_mode;hb<=start_hblock;section<=0;boot_word<=0;source_row<=0;num_lut<=0;
       pending<=0;jobs_left<=0;qhead<=0;qtail<=0;qcount<=0;alloc_tail<=0;used_words<=0;active<=0;fv<=0;pv<=0;
       for(integer b=0;b<8;b++)yvalid[b]<=0;
       dbg_updates<=0;dbg_mac<=0;dbg_coeff_words<=0;dbg_jobs<=0;dbg_source_rows<=0;dbg_query<=0;
       dbg_zero_jobs<=0;dbg_psn_wait<=0;dbg_peak_words<=0;dbg_peak_desc<=0;dbg_bank_skips<=0;
      end
      BREQ:if(mem_req_ready[boot_addr[2:0]])begin pending[boot_addr[2:0]]<=1;st<=BRSP;end
      BRSP:if(mem_rsp_valid[boot_addr[2:0]])begin
       logic[127:0] word_d;word_d=mem_rsp_data[boot_addr[2:0]];pending[boot_addr[2:0]]<=0;
       if(section==0)for(integer i=0;i<8;i++)if(int'(boot_word)*8+i<100)
         a[(int'(boot_word)*8+i)/10][(int'(boot_word)*8+i)%10]<=$signed(word_d[i*16+:16]);
       if(section==1)begin
         integer lid;lid=int'(num_lut);
         for(integer i=0;i<8;i++)begin
           patterns[int'(boot_word)*8+i]<=word_d[i*16+:16];lut_index[int'(boot_word)*8+i]<=7'(lid);
           if(word_d[i*16+:16]!=0)lid++;
         end
         num_lut<=7'(lid);
       end
       if(section==2)for(integer i=0;i<16;i++)begin
         integer bi,vi;bi=int'(boot_word)*16+i;vi=bi/6;
         tau[vi/96][vi%96][(bi%6)*8+:8]<=word_d[i*8+:8];
       end
       if(section==3)for(integer i=0;i<16;i++)begin
         integer bi;bi=int'(boot_word)*16+i;
         if(bi<12)positive[bi*8+:8]<=word_d[i*8+:8];
         else if(bi<24)constant_ch[(bi-12)*8+:8]<=word_d[i*8+:8];
         else constant_gate[(bi-24)*8+:8]<=word_d[i*8+:8];
       end
       if(section==4)for(integer i=0;i<16;i++)directory[int'(boot_word)*16+i]<=word_d[i*8+:8];
       if(section==5)for(integer i=0;i<32;i++)class_map[int'(boot_word)*32+i]<=word_d[i*4+:4];
       if(int'(boot_word)+1==boot_limit)begin
         boot_word<=0;
         if(section==5 || section==4 || (section==3 && mode<3))st<=SOURCE;
         else begin
           section<=section==0 && mode==0?3'd2:(section==3 && mode==4?3'd5:section+1'b1);
           st<=BREQ;
         end
       end else begin boot_word<=boot_word+1'b1;st<=BREQ;end
      end
      SOURCE:if(source_valid)begin
       for(integer j=0;j<192;j++)routes[j][source_row]<=new_routes[j];
       jobs_left<=jobs_left|new_routes;dbg_source_rows<=dbg_source_rows+1;
       if(mode!=0)dbg_query<=dbg_query+96;
       if(source_row==319)st<=FC;else source_row<=source_row+1'b1;
      end
      FC:if(jobs_left==0 && qcount==0 && !active && fv==0 && pending==0)begin p<=0;clear_t<=0;st<=PCLEAR;end
      PCLEAR:begin
       for(integer h=0;h<96;h++)u[clear_t][h]<=0;
       if(clear_t==9)begin s<=0;st<=PREAD;end else clear_t<=clear_t+1'b1;
      end
      PREAD:begin
       if(any_y)begin
         for(integer h=0;h<96;h++)yhold[h]<=y_read_data[h];
         for(integer t=0;t<10;t++)targets[t]<=a[t][s]!=0;
         st<=PMAC;
       end else if(s==9)st<=PDRAIN;else s<=s+1'b1;
      end
      PMAC:begin
       if(psn_issue)begin
         pv[0]<=1;pt[0]<=4'(next_t);busy[next_t]<=4;targets[next_t]<=0;dbg_mac<=dbg_mac+1;
         for(integer h=0;h<96;h++)product[0][h]<=$signed(a[next_t][s])*$signed(yhold[h]);
         if((targets & ~(10'b1<<next_t))==0)begin
           if(s==9)st<=PDRAIN;else begin s<=s+1'b1;st<=PREAD;end
         end
       end else if(targets==0)begin
         if(s==9)st<=PDRAIN;else begin s<=s+1'b1;st<=PREAD;end
       end else dbg_psn_wait<=dbg_psn_wait+1;
      end
      PDRAIN:if(pv==0)begin output_t<=0;st<=POUT;end
      POUT:if(out_ready)begin
       if(output_t==9)begin
         if(p==31)st<=DONE;else begin p<=p+1'b1;clear_t<=0;st<=PCLEAR;end
       end else output_t<=output_t+1'b1;
      end
      DONE:if(done_ready)st<=IDLE;
      default:st<=IDLE;
     endcase
   end
 end
endmodule
