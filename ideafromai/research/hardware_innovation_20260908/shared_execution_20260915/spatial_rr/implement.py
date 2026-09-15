from pathlib import Path
H=Path(__file__).resolve().parent;B=H.parents[1];N=B/'representation_transfer_20260914';A=N/'spatial_moment3'
s=(A/'spatial_core.sv').read_text().replace('module spatial_core(','module rr_context(')
s=s.replace('input logic clk,reset_n,cfg_valid,start,mode,','''input logic clk,reset_n,cfg_valid,start,borrow_enable,
 input logic resource_grant,input logic [575:0] q1_live,q2_live,
 output logic [5:0] resource_request,
 output logic weight_kind,output logic [9:0] weight_address,input logic [255:0] weight_data,
 output logic [255:0] alu_lhs,alu_rhs,input logic [255:0] alu_result,product_data,borrow_result,
 output logic [151:0] mul_lhs,output logic [103:0] mul_rhs,output logic alu_mac,alu_sub,alu_split15,
 output logic [511:0] borrow_lhs,borrow_rhs,
 output logic [31:0] arbitration_stalls,output logic compute_done,''')
s=s.replace('logic stripe,mode_q;','logic stripe;wire mode_q=1\'b1;')
s=s.replace('logic signed [7:0] q1_mem[0:7][0:575],q1_hold[0:7];','logic signed [7:0] q1_hold[0:7];')
s=s.replace('logic signed [12:0] q2_mem[0:7][0:575],qcache[0:7][0:23];','logic signed [12:0] qcache[0:7][0:23];')
s=s.replace(' logic q1_live[0:575],q2_live[0:575];\n','')
s=s.replace('logic [31:0] z_mem[0:7][0:39],z_hold[0:7],z_read_word[0:7];','logic [51:0] z_mem[0:7][0:39];logic [31:0] z_hold[0:7],z_read_word[0:7];')
s=s.replace(' logic cfg_q1_live,cfg_q2_live;','')
s=s.replace(' source_stalls<=0;weight_stalls<=0;output_stalls<=0;',' source_stalls<=0;weight_stalls<=0;output_stalls<=0;arbitration_stalls<=0;')
a=s.index('  cfg_q1_live=0');b=s.index('  selected_time=0',a);s=s[:a]+s[b:]
s=s.replace('z_read_enable=(state==ZREAD','z_read_enable=resource_grant&&(state==ZREAD')
s=s.replace('p_read_enable=(state==DRAIN_READ)||(state==POSLOAD&&stripe&&!mode_q)||state==W_PREAD0||state==W_PREAD1;','p_read_enable=resource_grant&&((state==DRAIN_READ)||(state==POSLOAD&&stripe&&!mode_q)||state==W_PREAD0||state==W_PREAD1);')
s=s.replace('p_write_enable=(state==STORE||state==W_STORE0||state==W_STORE1);','p_write_enable=resource_grant&&(state==STORE||state==W_STORE0||state==W_STORE1);')
s=s.replace('z_mem[l][z_addr]:32\'d0','z_mem[l][z_addr][31:0]:32\'d0')
s=s.replace('(state==QREAD&&weight_allow)?q1_mem[l][weight_addr]',"(state==QREAD&&resource_grant)?$signed(weight_data[l*32+:8])")
s=s.replace('(state==VLOAD&&weight_allow&&rank_live[qfill/3]&&q2_live[weight_addr])?q2_mem[l][weight_addr]',"(state==VLOAD&&resource_grant&&rank_live[qfill/3]&&q2_live[weight_addr])?$signed(weight_data[l*32+:13])")
s=s.replace("multiply_rhs[l]=(state==BASE_MAC)?qcache[l][selected_term]","multiply_rhs[l]=(state==BASE_MAC&&resource_grant)?qcache[l][selected_term]")
s=s.replace('product[l]=$signed(multiply_lhs[l])*$signed(multiply_rhs[l]);','product[l]=$signed(product_data[l*32+:32]);')
s=s.replace('d_monitor_valid=state==XD1||state==XD2;','d_monitor_valid=resource_grant&&(state==XD1||state==XD2);')
s=s.replace('z_monitor_valid=state==ZSCAN;','z_monitor_valid=resource_grant&&state==ZSCAN;')
a=s.index(' genvar l,b;');b=s.index(' always_ff @(posedge clk)begin',a)
s=s[:a]+''' always_comb begin
  resource_request=0;
  case(state)
   L_LOAD:if(in_bounds)resource_request=6'b000001;
   QREAD:resource_request=6'b000010;
   VLOAD:if(rank_live[qfill/3]&&q2_live[weight_addr])resource_request=6'b000010;
   ZCLEAR,ZREAD,ZSCAN,XDREAD0,XDREAD1:resource_request=6'b000100;
   ZADD:resource_request=borrow_enable?6'b100100:6'b010100;
   BASE_MAC,XD1,XD2:resource_request=6'b010100;
   XD0,W_INV0A,W_INV1A,W_PADD0,W_PADD1:resource_request=6'b010000;
   STORE,DRAIN_READ,W_PREAD0,W_PREAD1,W_STORE0,W_STORE1:resource_request=6'b001000;
   default:begin end
  endcase
  weight_kind=(state==VLOAD);weight_address=10'(weight_addr);
  alu_mac=state==BASE_MAC;alu_sub=sub_alu;alu_split15=state==ZADD;
  for(integer l=0;l<8;l=l+1)begin
   alu_lhs[l*32+:32]=lhs[l];alu_rhs[l*32+:32]=rhs[l];
   mul_lhs[l*19+:19]=multiply_lhs[l];mul_rhs[l*13+:13]=multiply_rhs[l];
   borrow_lhs[l*64+:64]={32'd0,lhs[l]};borrow_rhs[l*64+:64]={32'd0,rhs[l]};
   add_y[l]=$signed((state==ZADD&&borrow_enable)?borrow_result[l*32+:32]:alu_result[l*32+:32]);
  end
 end
'''+s[b:]
s=s.replace('stripe<=0;mode_q<=0;tx<=0;done<=0;','stripe<=0;tx<=0;done<=0;compute_done<=0;')
a=s.index('    4:begin\n');b=s.index('    default:begin end',a);s=s[:a]+s[b:]
s=s.replace('   case(state)\n    IDLE:if(start)begin mode_q<=mode;','''   if(resource_request!=0&&!resource_grant)begin
    if(resource_request[0]&&!source_allow)source_stalls<=source_stalls+1;
    else if(resource_request[1]&&!weight_allow)weight_stalls<=weight_stalls+1;
    else arbitration_stalls<=arbitration_stalls+1;
   end
   if(resource_request==0||resource_grant)case(state)
    IDLE:if(start)begin compute_done<=0;''')
s=s.replace("z_mem[i][zrow]<={2'd0,add_y[i][29:0]}","z_mem[i][zrow]<={22'd0,add_y[i][29:0]}")
s=s.replace("(state==XD2)?{16'd0,add_y[i][15:0]}:{add_y[i][15:0],acc[i][15:0]};", "(state==XD2)?{36'd0,add_y[i][15:0]}:{20'd0,add_y[i][15:0],acc[i][15:0]};")
s=s.replace('else begin row<=0;state<=DRAIN_READ;end','else begin compute_done<=1;row<=0;state<=DRAIN_READ;end')
# Assertions are architectural retire/compute checks only on successful service.
s=s.replace('if(state==STORE||state==W_STORE0||state==W_STORE1)begin','if(resource_grant&&(state==STORE||state==W_STORE0||state==W_STORE1))begin')
s=s.replace('if((state==W_PREAD0||state==W_PREAD1)&&!checked_first[p_addr])','if(resource_grant&&(state==W_PREAD0||state==W_PREAD1)&&!checked_first[p_addr])')
s=s.replace('if(state==XD0||state==XD1||state==XD2)for','if(resource_grant&&(state==XD0||state==XD1||state==XD2))for')
s=s.replace('if(state==BASE_MAC||state==W_INV0A||state==W_INV1A||state==W_PADD0||state==W_PADD1)','if(resource_grant&&(state==BASE_MAC||state==W_INV0A||state==W_INV1A||state==W_PADD0||state==W_PADD1))')
s=s.replace('if(state==ZADD)for(integer i=0;i<8;i=i+1)begin','if(resource_grant&&state==ZADD)for(integer i=0;i<8;i=i+1)begin')
s=s.replace('   if(p_read_enable&&p_write_enable)', '   if(resource_request!=0&&!resource_grant&&(z_read_enable||p_read_enable||p_write_enable))$fatal(1,"ungranted memory access");\n   if(p_read_enable&&p_write_enable)')
s=s.replace("d_monitor_data[i*32+:32]=(state==XD2)?{36'd0,add_y[i][15:0]}:{20'd0,add_y[i][15:0],acc[i][15:0]}","d_monitor_data[i*32+:32]=(state==XD2)?{16'd0,add_y[i][15:0]}:{add_y[i][15:0],acc[i][15:0]}")
s=s.replace('input logic [255:0] cfg_data','input logic [31:0] cfg_data')
s=s.replace("  weight_addr=state==QREAD?int'(stripe)*288+k:int'(stripe)*288+og*24+qfill;",'')
s=s.replace(' always_comb begin\n  in_bounds='," assign weight_addr=state==QREAD?int'(stripe)*288+k:int'(stripe)*288+og*24+qfill;\n always_comb begin\n  in_bounds=")
(H/'rr_context.sv').write_text(s)
(H/'i24_consumer.sv').write_text((A/'i24_consumer.sv').read_text())
w=(A/'wide_phase_alu.sv').read_text().replace('b==13 || b==26 || b==39','b==15 || b==30 || b==45 || b==60')
(H/'wide_phase_alu.sv').write_text(w)
print('Extracted state-only contexts with physical Z52, shared datapath interface; consumer copied.')
