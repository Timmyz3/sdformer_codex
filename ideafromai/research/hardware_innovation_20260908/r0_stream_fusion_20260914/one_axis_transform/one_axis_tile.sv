// Complete C96,N96,T10: horizontal F(2,3), vertical direct. Shared signed48 chains.
// Static W16/U2 signed18 tight bit streams; input and inverse arithmetic in RTL.
module one_axis_tile(
 input logic clk,rst,start,input logic [1:0] mode, //0 direct,1 one-axis
 input logic signed [15:0] origin_y,origin_x,
 input logic meta_valid,input logic [9:0] meta_addr,input logic [127:0] meta_data,
 input logic in_valid,output logic in_ready,input logic [127:0] in_data,
 output logic weight_req,input logic weight_ready,output logic [31:0] weight_addr,
 input logic weight_valid,input logic [255:0] weight_data,
 output logic out_valid,input logic out_ready,output logic [255:0] out_data,
 output logic [8:0] out_addr,output logic done,output logic [4:0] debug_state,
 output logic debug_cache_hit,debug_add,debug_half_zero,parity_error
);
 localparam IDLE=0,LOAD=1,TR1=2,SCAN2=3,CLEAR=4,SCAN=5,LOOKUP=6,
   REQUEST=7,RESPONSE=8,DECODE=9,PRE3=10,AAC=11,INV1A=12,INV1B=13,
   INV2A=14,INV2B=15,ROUND=16,SEND=17,FINISH=18;
 logic [4:0] state;
 logic [1:0] active_mode;
 logic [15:0] source [0:959];
 logic signed [2:0] vmem [0:15359],vreg [0:19],vnow [0:9];
 logic signed [47:0] mem [0:1279],work [0:7];
 logic [1:0] support [0:6911];
 logic signed [31:0] weights [0:15],three [0:15],decoded [0:15];
 logic [255:0] cache [0:1];logic [31:0] tags [0:1];logic [1:0] cache_valid;
 logic victim;
 logic [511:0] assembled;
 logic [39:0] events,events_now,events_join;
 logic signed [15:0] origin_y_q,origin_x_q;logic [15:0] boundary_mask;
 logic [127:0] masked_input;
 logic [1:0] half_live;
 logic any_three,three_now,source_any,zero_fast,transform_any;
 integer input_beat,transform_group,transform_pos,group_idx,key_idx,clear_idx;
 integer half_idx,inv_idx,output_idx,keys,c_idx,tap_idx,xi,phase_idx;
 logic [3:0] src_idx;
 integer vector_idx,coef_bits,vector_base,first_line,last_line,current_line,segment;
 integer ev_idx,bit_offset,transform_t,transform_c,transform_row,transform_col;
 integer next_key,selector_key;logic next_found;
 integer inv_xi,inv_t,inv_half,inv_row,inv_col,inv_src1,inv_src2,inv_src3;
 logic event_found,cache_found,cache_slot,subtract;
 logic signed [3:0] factor;
 integer magnitude;
 logic signed [47:0] add_a [0:7],add_b [0:7],add_y [0:7];
 logic [47:0] xor_b [0:7];
 function automatic integer bfirst(input integer r);
   case(r)0:bfirst=0;1:bfirst=1;2:bfirst=2;default:bfirst=1;endcase
 endfunction
 function automatic integer bsecond(input integer r);
   case(r)0:bsecond=2;1:bsecond=2;2:bsecond=1;default:bsecond=3;endcase
 endfunction
 function automatic integer maddr(input integer coord,input integer t,input integer h,input integer l);
   maddr=(coord*10+t)*16+h*8+l;
 endfunction
 always_comb begin
   keys=(active_mode==0)?864:1152;
   coef_bits=(active_mode==0)?16:18;
   vector_idx=group_idx*keys+key_idx;
   vector_base=vector_idx*16*coef_bits;
   // Common finite 64-entry static-support successor. At most one empty block
   // is advanced per SCAN cycle; no unlimited sparse list or offline cycle oracle.
   next_key=((key_idx/64)+1)*64;
   if(next_key>keys)next_key=keys;
   next_found=0;
   for(integer j=0;j<64;j=j+1)begin
     selector_key=(key_idx/64)*64+j;
     if(selector_key>key_idx && selector_key<keys && !next_found &&
       support[group_idx*keys+selector_key]!=0)begin next_key=selector_key;next_found=1;end
   end
   c_idx=(active_mode==0)?key_idx/9:key_idx/12;
   tap_idx=(active_mode==0)?key_idx%9:(key_idx%12)/4;xi=key_idx%4;phase_idx=0;src_idx=0;
   events_now='0;three_now=0;
   for(integer t=0;t<10;t=t+1)begin
     vnow[t]=0;
     if(active_mode==1 && source[c_idx*10+t]!=0)vnow[t]=vmem[(c_idx*10+t)*16+(tap_idx+((state==SCAN2)?1:0))*4+xi];
     if(active_mode==0)begin
       for(integer p=0;p<4;p=p+1)
         events_now[p*10+t]=source[c_idx*10+t][(p/2+tap_idx/3)*4+p%2+tap_idx%3];
     end else if(active_mode==1)begin
       events_now[t]=(vnow[t]!=0);three_now=three_now||(vnow[t]==3);
     end else events_now[phase_idx*10+t]=source[c_idx*10+t][src_idx];
   end
   events_join=events | (events_now<<10);
   ev_idx=0;event_found=0;
   for(integer e=0;e<40;e=e+1)if(events[e]&&!event_found)begin ev_idx=e;event_found=1;end
   factor=1;
   if(active_mode==1 && ev_idx<20)factor=4'(vreg[ev_idx]);
   magnitude=(factor<0)?-32'(factor):32'(factor);
   cache_found=0;cache_slot=0;
   if(cache_valid[0] && tags[0]==32'(current_line))begin cache_found=1;cache_slot=0;end
   else if(cache_valid[1] && tags[1]==32'(current_line))begin cache_found=1;cache_slot=1;end
   for(integer q=0;q<16;q=q+1)begin
     decoded[q]=0;
     bit_offset=vector_base+q*coef_bits-first_line*256;
     if(half_live[q/8])begin
       if(active_mode==0)decoded[q]={{16{assembled[bit_offset+15]}},assembled[bit_offset+:16]};
       else decoded[q]={{14{assembled[bit_offset+17]}},assembled[bit_offset+:18]};
     end
   end
   transform_t=transform_group/12;transform_c=(transform_group%12)*8;
   transform_row=transform_pos/4;transform_col=transform_pos%4;
   transform_any=0;
   for(integer l=0;l<8;l=l+1)transform_any=transform_any||(|source[(transform_c+l)*10+transform_t]);
   inv_xi=inv_idx/20;inv_t=(inv_idx/2)%10;inv_half=inv_idx%2;
   inv_row=inv_xi/2;inv_col=inv_xi%2;
   inv_src1=inv_row*4+((inv_col==0)?0:1);
   inv_src2=inv_row*4+((inv_col==0)?1:2);
   inv_src3=inv_row*4+((inv_col==0)?2:3);
   boundary_mask=0;
   for(integer z=0;z<16;z=z+1)boundary_mask[z]=
     (int'(origin_y_q)+z/4>=0 && int'(origin_y_q)+z/4<240 &&
      int'(origin_x_q)+z%4>=0 && int'(origin_x_q)+z%4<320);
   for(integer l=0;l<8;l=l+1)masked_input[l*16+:16]=in_data[l*16+:16]&boundary_mask;
   subtract=0;
   if(state==TR1)subtract=(transform_col!=1);
   else if(state==AAC)subtract=(factor<0);
   else if(state==INV1A || state==INV1B)subtract=(inv_col==1);
   else if(state==INV2A || state==INV2B)subtract=(inv_col==1);
   for(integer l=0;l<8;l=l+1)begin
     add_a[l]=0;add_b[l]=0;
     if(state==TR1)begin
       add_a[l]=48'(source[(transform_c+l)*10+transform_t][transform_row*4+bfirst(transform_col)]);
       add_b[l]=48'(source[(transform_c+l)*10+transform_t][transform_row*4+bsecond(transform_col)]);
     end else if(state==PRE3)begin
       add_a[l]=48'(weights[half_idx*8+l]);add_b[l]=48'(weights[half_idx*8+l])<<<1;
     end else if(state==AAC)begin
       if(active_mode==1)add_a[l]=mem[maddr((ev_idx/10)*4+xi,ev_idx%10,half_idx,l)];
       else add_a[l]=mem[ev_idx*16+half_idx*8+l];
       if(magnitude==3)add_b[l]=48'(three[half_idx*8+l]);
       else if(magnitude==4)add_b[l]=48'(weights[half_idx*8+l])<<<2;
       else if(magnitude==2)add_b[l]=48'(weights[half_idx*8+l])<<<1;
       else add_b[l]=48'(weights[half_idx*8+l]);
     end else if(state==INV1A || state==INV2A)begin
       add_a[l]=mem[maddr(inv_src1,inv_t,inv_half,l)];
       add_b[l]=mem[maddr(inv_src2,inv_t,inv_half,l)];
     end else if(state==INV1B || state==INV2B)begin
       add_a[l]=work[l];add_b[l]=mem[maddr(inv_src3,inv_t,inv_half,l)];
     end else if(state==ROUND)begin
       if(active_mode==0)begin add_a[l]=mem[(output_idx/2)*16+(output_idx%2)*8+l];end
       else begin
         add_a[l]=mem[(output_idx/2)*16+(output_idx%2)*8+l]>>>1;
         add_b[l]=0;
       end
     end
   end
   in_ready=(state==LOAD);weight_req=(state==REQUEST);weight_addr=32'(current_line);
   out_valid=(state==SEND);out_addr=9'(group_idx*80+output_idx);done=(state==FINISH);
   debug_state=state;debug_cache_hit=(state==LOOKUP && cache_found);
   debug_half_zero=(state==SCAN && events_now!=0 && support[vector_idx]!=3);
   debug_add=(state==TR1 && (transform_pos!=0 || transform_any)) ||
     state==PRE3 || state==AAC || state==INV1A || state==INV1B ||
     state==INV2A || state==INV2B || state==ROUND;
 end
 genvar lane,bitno;
 generate for(lane=0;lane<8;lane=lane+1)begin:ALU
   assign xor_b[lane]=add_b[lane]^{48{subtract}};
   for(bitno=0;bitno<48;bitno=bitno+1)begin:BIT
     wire cin;
     if(bitno==0)assign cin=subtract;else assign cin=BIT[bitno-1].CARRY.cout;
     assign add_y[lane][bitno]=add_a[lane][bitno]^xor_b[lane][bitno]^cin;
     if(bitno<47)begin:CARRY
       wire cout;
       assign cout=(add_a[lane][bitno]&xor_b[lane][bitno])|((add_a[lane][bitno]^xor_b[lane][bitno])&cin);
     end
   end
 end endgenerate
 always_ff @(posedge clk)begin
   if(rst)begin
     state<=IDLE;active_mode<=0;input_beat<=0;transform_group<=0;transform_pos<=0;
     group_idx<=0;key_idx<=0;clear_idx<=0;half_idx<=0;inv_idx<=0;output_idx<=0;
     first_line<=0;last_line<=0;current_line<=0;segment<=0;assembled<=0;
     cache_valid<=0;victim<=0;events<=0;half_live<=0;any_three<=0;
     source_any<=0;zero_fast<=0;out_data<=0;origin_y_q<=0;origin_x_q<=0;parity_error<=0;
   end else begin
     if(state==IDLE && meta_valid)for(integer j=0;j<64;j=j+1)support[32'(meta_addr)*64+j]<=meta_data[j*2+:2];
     case(state)
       IDLE:if(start)begin
         state<=LOAD;active_mode<=mode;input_beat<=0;source_any<=0;cache_valid<=0;
         origin_y_q<=origin_y;origin_x_q<=origin_x;parity_error<=0;
         group_idx<=0;key_idx<=0;clear_idx<=0;transform_group<=0;transform_pos<=0;output_idx<=0;
       end
       LOAD:if(in_valid)begin
         for(integer l=0;l<8;l=l+1)source[((input_beat%12)*8+l)*10+input_beat/12]<=masked_input[l*16+:16];
         source_any<=source_any||(|masked_input);
         if(input_beat==119)begin
           zero_fast<=!(source_any||(|masked_input));
           state<=(active_mode==1 && (source_any||(|masked_input)))?TR1:CLEAR;
         end else input_beat<=input_beat+1;
       end
       TR1:begin
         if(transform_pos==0 && !transform_any)begin
           if(transform_group==119)begin state<=CLEAR;clear_idx<=0;end
           else transform_group<=transform_group+1;
         end else begin
           for(integer l=0;l<8;l=l+1)vmem[((transform_c+l)*10+transform_t)*16+transform_pos]<=add_y[l][2:0];
           if(transform_pos==15)begin
             transform_pos<=0;
             if(transform_group==119)begin state<=CLEAR;clear_idx<=0;end
             else transform_group<=transform_group+1;
           end
           else transform_pos<=transform_pos+1;
         end
       end
       CLEAR:begin
         for(integer l=0;l<8;l=l+1)mem[clear_idx*8+l]<=0;
         if(clear_idx==((active_mode==1 && !zero_fast)?159:79))begin
           clear_idx<=0;key_idx<=0;output_idx<=0;state<=zero_fast?ROUND:SCAN;
         end else clear_idx<=clear_idx+1;
       end
       SCAN:begin
         if(active_mode==1 && support[vector_idx]!=0)begin
           events<=events_now;
           for(integer t=0;t<10;t=t+1)vreg[t]<=vnow[t];
           state<=SCAN2;
         end else if(events_now!=0 && support[vector_idx]!=0)begin
           events<=events_now;half_live<=support[vector_idx];any_three<=three_now;
           for(integer t=0;t<10;t=t+1)vreg[t]<=vnow[t];
           first_line<=(vector_base+(support[vector_idx][0]?0:8*coef_bits))/256;
           current_line<=(vector_base+(support[vector_idx][0]?0:8*coef_bits))/256;
           last_line<=(vector_base+(support[vector_idx][1]?16:8)*coef_bits-1)/256;
           segment<=0;assembled<=0;state<=LOOKUP;
         end else if(next_key>=keys)begin
           inv_idx<=0;output_idx<=0;state<=(active_mode==1)?INV1A:ROUND;
         end else key_idx<=next_key;
       end
       SCAN2:begin
         for(integer t=0;t<10;t=t+1)vreg[10+t]<=vnow[t];
         if(events_join!=0)begin
           events<=events_join;half_live<=support[vector_idx];any_three<=0;
           first_line<=(vector_base+(support[vector_idx][0]?0:8*coef_bits))/256;
           current_line<=(vector_base+(support[vector_idx][0]?0:8*coef_bits))/256;
           last_line<=(vector_base+(support[vector_idx][1]?16:8)*coef_bits-1)/256;
           segment<=0;assembled<=0;state<=LOOKUP;
         end else if(next_key>=keys)begin inv_idx<=0;output_idx<=0;state<=INV1A;end
         else begin key_idx<=next_key;state<=SCAN;end
       end
       LOOKUP:begin
         if(cache_found)begin
           assembled[segment*256+:256]<=cache[cache_slot];
           if(current_line==last_line)state<=DECODE;
           else begin current_line<=current_line+1;segment<=segment+1;end
         end else state<=REQUEST;
       end
       REQUEST:if(weight_ready)state<=RESPONSE;
       RESPONSE:if(weight_valid)begin
         assembled[segment*256+:256]<=weight_data;cache[victim]<=weight_data;
         tags[victim]<=32'(current_line);cache_valid[victim]<=1;victim<=!victim;
         if(current_line==last_line)state<=DECODE;
         else begin current_line<=current_line+1;segment<=segment+1;state<=LOOKUP;end
       end
       DECODE:begin
         for(integer q=0;q<16;q=q+1)weights[q]<=decoded[q];
         half_idx<=half_live[0]?0:1;
         state<=(active_mode==1 && any_three)?PRE3:AAC;
       end
       PRE3:begin
         for(integer l=0;l<8;l=l+1)three[half_idx*8+l]<=add_y[l][31:0];
         if(half_idx==0 && half_live[1])half_idx<=1;
         else begin half_idx<=half_live[0]?0:1;state<=AAC;end
       end
       AAC:begin
         for(integer l=0;l<8;l=l+1)
           if(active_mode==1)mem[maddr((ev_idx/10)*4+xi,ev_idx%10,half_idx,l)]<=add_y[l];
           else mem[ev_idx*16+half_idx*8+l]<=add_y[l];
         if(half_idx==0 && half_live[1])half_idx<=1;
         else begin
           half_idx<=half_live[0]?0:1;events[ev_idx]<=0;
           if((events & (events-40'd1))==0)begin
             if(next_key>=keys)begin inv_idx<=0;output_idx<=0;state<=(active_mode==1)?INV1A:ROUND;end
             else begin key_idx<=next_key;state<=SCAN;end
           end
         end
       end
       INV1A:begin for(integer l=0;l<8;l=l+1)work[l]<=add_y[l];state<=INV1B;end
       INV1B:begin
         for(integer l=0;l<8;l=l+1)mem[maddr(inv_xi,inv_t,inv_half,l)]<=add_y[l];
         if(inv_idx==79)begin output_idx<=0;state<=ROUND;end
         else begin inv_idx<=inv_idx+1;state<=INV1A;end
       end
       INV2A:begin for(integer l=0;l<8;l=l+1)work[l]<=add_y[l];state<=INV2B;end
       INV2B:begin
         for(integer l=0;l<8;l=l+1)mem[maddr(inv_xi,inv_t,inv_half,l)]<=add_y[l];
         if(inv_idx==79)begin output_idx<=0;state<=ROUND;end
         else begin inv_idx<=inv_idx+1;state<=INV2A;end
       end
       ROUND:begin
         for(integer l=0;l<8;l=l+1)begin
           out_data[l*32+:32]<=add_y[l][31:0];
           if(active_mode==1 && mem[(output_idx/2)*16+(output_idx%2)*8+l][0])parity_error<=1;
         end
         state<=SEND;
       end
       SEND:if(out_ready)begin
         if(output_idx==79)begin
           if(group_idx==5)state<=FINISH;
           else begin group_idx<=group_idx+1;clear_idx<=0;state<=CLEAR;end
         end else begin output_idx<=output_idx+1;state<=ROUND;end
       end
       FINISH:state<=IDLE;
       default:state<=IDLE;
     endcase
   end
 end
endmodule
