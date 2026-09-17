module subset_psn(
 input logic clk,rst_n,
 input logic start_valid,output logic start_ready,input logic start_cert,start_reload_a,
 output logic param_req_valid,input logic param_req_ready,output logic[8:0] param_req_addr,
 input logic param_rsp_valid,output logic param_rsp_ready,input logic[127:0] param_rsp_data,
 input logic y_valid,output logic y_ready,input logic[2303:0] y_data,
 output logic out_valid,input logic out_ready,output logic[4:0] out_p,output logic[3:0] out_hgroup,
 output logic[79:0] out_gate,output logic[3839:0] out_u,
 output logic done_valid,input logic done_ready,
 output logic[4:0] dbg_state,output logic[4:0] dbg_m,output logic[4:0] dbg_exponent,
 output logic[31:0] dbg_planes,dbg_early,dbg_y_reads,dbg_table_writes,
 output logic mon_lower,mon_upper,output logic[3839:0] mon_bound,
 output logic[2303:0] mon_y,output logic[8:0] mon_yrow,
 output logic[159:0] mon_table,output logic[5:0] mon_table_addr
);
 typedef enum logic[4:0]{IDLE,PREQ,PRSP,TABLE,PN_INIT,PN_POS,PN_NEG,YLOAD,PINIT,YREAD,
  GROUP,SIGN_SUM,SIGN_NEG,PLANE_SUM,PREFIX,BOUND_POS,BOUND_NEG,LOWER,UPPER,OUTPUT,DONE} state_t;
 state_t st;
 logic cert,reload_a;logic[8:0] pa,yr;
 logic signed[15:0] a[10][10];
 // One physical register table. Each of 20 banks has eight explicit read muxes.
 logic signed[15:0] lut[2][32][10];
 logic signed[47:0] tau[10][96];
 logic[95:0] positive,constant_ch;logic[959:0] constant_gate;
 logic signed[23:0] y_mem[320][96];
 logic signed[23:0] y_read_data[96],ybuf[10][96];
 logic[4:0] exponents[12],exp_next[12];
 logic[4:0] p,m;logic[3:0] hgroup,s,col;logic half;logic[4:0] code;
 logic signed[47:0] pos[10],neg[10],tail_pos[10],tail_neg[10];
 logic signed[47:0] v[80],dot_hold[80];
 logic[79:0] locked,gate,lower_hit,lower_gate;
 logic[4:0] lo_code[8],hi_code[8];
 logic signed[15:0] lookup_lo[80],lookup_hi[80];
 logic signed[47:0] add_a[80],add_b[80],add_y[80];logic[79:0] add_sub;
 logic[79:0] upper_hit,upper_gate,locked_next,gate_next,cmp_less,cmp_equal;
 integer lsb,previous;
 function automatic logic[4:0] magnitude_bits(input logic signed[23:0] x);
   logic[4:0] e;logic signed[24:0] xx,lim;
   begin
    e=0;xx=25'(x);
    // Fixed constant comparisons/priority encoder: no absolute-value adder.
    for(integer b=0;b<24;b++)begin lim=25'sd1<<<b;if(xx>=lim || xx<=-lim)e=5'(b+1);end
    return e;
   end
 endfunction
 always_comb begin
   start_ready=st==IDLE;param_req_valid=st==PREQ;param_req_addr=pa;param_rsp_ready=st==PRSP;
   y_ready=st==YLOAD;out_valid=st==OUTPUT;done_valid=st==DONE;
   out_p=p;out_hgroup=hgroup;out_gate=gate;out_u=0;mon_bound=0;mon_y=0;mon_table=0;mon_table_addr={half,code};mon_yrow=9'(int'(p)*10+int'(s));
   dbg_state=st;dbg_m=m;dbg_exponent=exponents[hgroup];mon_lower=st==LOWER;mon_upper=st==UPPER;
   lsb=0;for(integer b=4;b>=0;b--)if(code[b])lsb=b;
   previous=int'(code)&~(1<<lsb);
   for(integer h=0;h<96;h++)y_read_data[h]=st==YREAD?y_mem[int'(p)*10+int'(s)][h]:24'sd0;
   for(integer h=0;h<96;h++)mon_y[h*24+:24]=y_read_data[h];
   for(integer g=0;g<12;g++)begin
     exp_next[g]=exponents[g];
     for(integer j=0;j<8;j++)if(magnitude_bits(y_read_data[g*8+j])>exp_next[g])exp_next[g]=magnitude_bits(y_read_data[g*8+j]);
   end
   for(integer i=0;i<8;i++)begin
     lo_code[i]=0;hi_code[i]=0;
     for(integer j=0;j<5;j++)begin
       lo_code[i][j]=(st==SIGN_SUM)?ybuf[j][int'(hgroup)*8+i][23]:ybuf[j][int'(hgroup)*8+i][m];
       hi_code[i][j]=(st==SIGN_SUM)?ybuf[5+j][int'(hgroup)*8+i][23]:ybuf[5+j][int'(hgroup)*8+i][m];
     end
   end
   for(integer i=0;i<80;i++)begin
     lookup_lo[i]=0;lookup_hi[i]=0;
     if(st==TABLE || st==SIGN_SUM || st==PLANE_SUM)begin
       lookup_lo[i]=lut[0][st==TABLE?5'(previous):lo_code[i%8]][i/8];
       lookup_hi[i]=lut[1][st==TABLE?5'(previous):hi_code[i%8]][i/8];
     end
   end
   upper_hit=0;upper_gate=0;locked_next=locked;gate_next=gate;add_sub=0;
   for(integer i=0;i<80;i++)begin
     integer t,h;t=i/8;h=int'(hgroup)*8+i%8;
     add_a[i]=0;add_b[i]=0;
     if(st==TABLE && i<10 && code!=0)begin
       add_a[i]=half?48'($signed(lookup_hi[i*8])):48'($signed(lookup_lo[i*8]));add_b[i]=48'($signed(a[i][int'(half)*5+lsb]));
     end else if(st==PN_POS && i<10)begin add_a[i]=pos[i];add_b[i]=a[i][col]>0?48'($signed(a[i][col])):48'sd0;end
     else if(st==PN_NEG && i<10)begin add_a[i]=neg[i];add_b[i]=a[i][col]<0?48'($signed(a[i][col])):48'sd0;end
     else if(st==SIGN_SUM || st==PLANE_SUM)begin add_a[i]=48'($signed(lookup_lo[i]));add_b[i]=48'($signed(lookup_hi[i]));end
     else if(st==SIGN_NEG)begin add_a[i]=0;add_b[i]=dot_hold[i];add_sub[i]=1;end
     else if(st==PREFIX)begin add_a[i]=v[i]<<<1;add_b[i]=dot_hold[i];end
     else if(st==BOUND_POS && i<10)begin add_a[i]=pos[i]<<<m;add_b[i]=pos[i];add_sub[i]=1;end
     else if(st==BOUND_NEG && i<10)begin add_a[i]=neg[i]<<<m;add_b[i]=neg[i];add_sub[i]=1;end
     else if(st==LOWER)begin add_a[i]=v[i]<<<m;add_b[i]=tail_neg[t];end
     else if(st==UPPER)begin add_a[i]=v[i]<<<m;add_b[i]=tail_pos[t];end
     // The sole data add/sub resource. SUB maps to complemented B plus carry-in.
     add_y[i]=add_a[i]+(add_b[i]^{48{add_sub[i]}})+48'(add_sub[i]);
     out_u[i*48+:48]=v[i];mon_bound[i*48+:48]=add_y[i];
     if(i<10)mon_table[i*16+:16]=code==0?16'd0:16'(add_y[i]);
     cmp_less[i]=add_y[i]<tau[t][h];cmp_equal[i]=add_y[i]==tau[t][h];
     upper_hit[i]=positive[h]?cmp_less[i]:(cmp_less[i]||cmp_equal[i]);
     upper_gate[i]=!positive[h];
     if(!locked[i])begin
       if(lower_hit[i])begin locked_next[i]=1;gate_next[i]=lower_gate[i];end
       else if(upper_hit[i])begin locked_next[i]=1;gate_next[i]=upper_gate[i];end
     end
   end
 end
 always_ff @(posedge clk or negedge rst_n)begin
   if(!rst_n)begin
     st<=IDLE;cert<=0;reload_a<=0;pa<=0;yr<=0;p<=0;s<=0;hgroup<=0;m<=0;half<=0;code<=0;col<=0;
     locked<=0;gate<=0;lower_hit<=0;lower_gate<=0;
     dbg_planes<=0;dbg_early<=0;dbg_y_reads<=0;dbg_table_writes<=0;
   end else begin
     case(st)
     IDLE:if(start_valid)begin
       cert<=start_cert;reload_a<=start_reload_a;pa<=start_reload_a?9'd0:9'd13;st<=PREQ;
       dbg_planes<=0;dbg_early<=0;dbg_y_reads<=0;dbg_table_writes<=0;
     end
     PREQ:if(param_req_ready)st<=PRSP;
     PRSP:if(param_rsp_valid)begin
       if(pa<13)begin for(integer j=0;j<8;j++)if(int'(pa)*8+j<100)a[(int'(pa)*8+j)/10][(int'(pa)*8+j)%10]<=param_rsp_data[j*16+:16];end
       else if(pa<493)begin for(integer j=0;j<2;j++)tau[((int'(pa)-13)*2+j)/96][((int'(pa)-13)*2+j)%96]<=param_rsp_data[j*48+:48];end
       else if(pa==493)positive<=param_rsp_data[95:0];
       else if(pa==494)constant_ch<=param_rsp_data[95:0];
       else for(integer b=0;b<128;b++)if((int'(pa)-495)*128+b<960)constant_gate[(int'(pa)-495)*128+b]<=param_rsp_data[b];
       if(pa==502)begin
         if(reload_a)begin half<=0;code<=0;st<=TABLE;end
         else begin yr<=0;st<=YLOAD;end
       end else begin pa<=pa+1'b1;st<=PREQ;end
     end
     TABLE:begin
       for(integer t=0;t<10;t++)lut[half][code][t]<=code==0?16'sd0:16'(add_y[t]);
       dbg_table_writes<=dbg_table_writes+1'b1;
       if(code==31)begin
         if(half)st<=PN_INIT;else begin half<=1;code<=0;end
       end else code<=code+1'b1;
     end
     PN_INIT:begin for(integer t=0;t<10;t++)begin pos[t]<=0;neg[t]<=0;end col<=0;st<=PN_POS;end
     PN_POS:begin for(integer t=0;t<10;t++)pos[t]<=add_y[t];st<=PN_NEG;end
     PN_NEG:begin
       for(integer t=0;t<10;t++)neg[t]<=add_y[t];
       if(col==9)begin yr<=0;st<=YLOAD;end else begin col<=col+1'b1;st<=PN_POS;end
     end
     YLOAD:if(y_valid)begin
       for(integer h=0;h<96;h++)y_mem[yr][h]<=y_data[h*24+:24];
       if(yr==319)begin p<=0;st<=PINIT;end else yr<=yr+1'b1;
     end
     PINIT:begin for(integer g=0;g<12;g++)exponents[g]<=0;s<=0;st<=YREAD;end
     YREAD:begin
       for(integer h=0;h<96;h++)ybuf[s][h]<=y_read_data[h];
       for(integer g=0;g<12;g++)exponents[g]<=exp_next[g];
       dbg_y_reads<=dbg_y_reads+1'b1;
       if(s==9)begin hgroup<=0;st<=GROUP;end else s<=s+1'b1;
     end
     GROUP:begin
       for(integer i=0;i<80;i++)begin
         integer t,h;t=i/8;h=int'(hgroup)*8+i%8;
         locked[i]<=constant_ch[h];gate[i]<=constant_ch[h]?constant_gate[t*96+h]:1'b0;
       end
       lower_hit<=0;lower_gate<=0;m<=exponents[hgroup]==0?5'd0:exponents[hgroup]-1'b1;st<=SIGN_SUM;
     end
     SIGN_SUM:begin for(integer i=0;i<80;i++)dot_hold[i]<=add_y[i];st<=SIGN_NEG;end
     SIGN_NEG:begin for(integer i=0;i<80;i++)v[i]<=add_y[i];st<=PLANE_SUM;end
     PLANE_SUM:begin for(integer i=0;i<80;i++)dot_hold[i]<=add_y[i];st<=PREFIX;end
     PREFIX:begin
       for(integer i=0;i<80;i++)v[i]<=add_y[i];dbg_planes<=dbg_planes+1'b1;
       if(m==0)begin
         for(integer i=0;i<80;i++)begin
           integer t,h;t=i/8;h=int'(hgroup)*8+i%8;
           if(!locked[i])gate[i]<=positive[h]?!cmp_less[i]:(cmp_less[i]||cmp_equal[i]);
         end
         locked<='1;st<=OUTPUT;
       end else if(cert)st<=BOUND_POS;
       else begin m<=m-1'b1;st<=PLANE_SUM;end
     end
     BOUND_POS:begin for(integer t=0;t<10;t++)tail_pos[t]<=add_y[t];st<=BOUND_NEG;end
     BOUND_NEG:begin for(integer t=0;t<10;t++)tail_neg[t]<=add_y[t];st<=LOWER;end
     LOWER:begin
       for(integer i=0;i<80;i++)begin
         integer t,h;t=i/8;h=int'(hgroup)*8+i%8;
         lower_hit[i]<=positive[h]?!cmp_less[i]:(!cmp_less[i]&&!cmp_equal[i]);
         lower_gate[i]<=positive[h];
       end
       st<=UPPER;
     end
     UPPER:begin
       locked<=locked_next;gate<=gate_next;
       if(locked_next[31:0]==32'hffffffff && locked_next[63:32]==32'hffffffff && locked_next[79:64]==16'hffff)begin dbg_early<=dbg_early+1'b1;st<=OUTPUT;end
       else begin m<=m-1'b1;st<=PLANE_SUM;end
     end
     OUTPUT:if(out_ready)begin
       if(hgroup==11)begin if(p==31)st<=DONE;else begin p<=p+1'b1;st<=PINIT;end end
       else begin hgroup<=hgroup+1'b1;st<=GROUP;end
     end
     DONE:if(done_ready)st<=IDLE;
     default:st<=IDLE;
     endcase
   end
 end
endmodule
