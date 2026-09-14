from pathlib import Path
H=Path(__file__).resolve().parent
B=H.parents[1]/'consumer_transfer_20260914/count_rr'
s=(B/'i24_consumer.sv').read_text().replace('  output logic done, error,','  output logic wide_monitor_valid, output logic [8:0] wide_monitor_address, output logic [511:0] wide_monitor_data,\n  output logic done, error,')
s=s.replace('    add_req=(state==ADD_BIAS || state==ADD_IDENTITY);','    wide_monitor_valid=(state==ROUND);wide_monitor_address=row;\n    for(integer l=0;l<8;l=l+1)wide_monitor_data[l*64+:64]=wide_hold[l];\n    add_req=(state==ADD_BIAS || state==ADD_IDENTITY);')
(H/'i24_consumer.sv').write_text(s)
(H/'wide_phase_alu.sv').write_text((B/'wide_phase_alu.sv').read_text())
base=['cycles','source_words','q1_words','q2_words','local_gathers','q1_issues','q2_issues','z_vector_reads','z_scalar_reads','z_writes','psum_reads','psum_writes','cache_writes','source_stalls','weight_stalls','output_stalls']
consumer=['cycles','raw_words','identity_words','coefficient_words','mul_issues','add_issues','round_issues','output_words','identity_stalls','raw_wait_cycles','join_wait_cycles','output_stalls','saturations','conversion_issues','conversion_saturations','wide_waits']
s='''module spatial_stream(
 input logic clk,reset_n,cfg_valid,start,
 input logic [2:0] cfg_kind,input logic [10:0] cfg_addr,input logic [255:0] cfg_data,
 input logic source_allow,weight_allow,raw_allow,result_ready,
 output logic result_valid,done,error,output logic [8:0] result_addr,output logic [255:0] result_data,
 output logic identity_request_valid,output logic [8:0] identity_address,
 input logic identity_valid,input logic [255:0] identity_data,
 output logic raw_monitor_valid,output logic [8:0] raw_monitor_addr,output logic [255:0] raw_monitor_data,
 output logic j_monitor_valid,output logic [8:0] j_monitor_address,output logic [255:0] j_monitor_data,
 output logic wide_monitor_valid,output logic [8:0] wide_monitor_address,output logic [511:0] wide_monitor_data,
 output logic [5:0] debug_state,
 output logic z_monitor_valid,z_monitor_stripe,output logic [5:0] z_monitor_addr,output logic [255:0] z_monitor_data,
 output logic [31:0] '''+', '.join(base+['c_'+x for x in consumer])+'''
);
 logic raw_valid,raw_ready,join_ready,core_done,add_req;
 logic [8:0] raw_addr;logic [255:0] raw_data;
 logic [511:0] add_lhs,add_rhs,add_y;
 assign raw_ready=join_ready&&raw_allow;
 assign raw_monitor_valid=raw_valid&&raw_ready;
 assign raw_monitor_addr=raw_addr;assign raw_monitor_data=raw_data;
 spatial_core producer(.clk(clk),.reset_n(reset_n),.cfg_valid(cfg_valid&&cfg_kind!=6),.start(start),
 .cfg_kind(cfg_kind),.cfg_addr(cfg_addr),.cfg_data(cfg_data),.source_allow(source_allow),.weight_allow(weight_allow),
 .result_ready(raw_ready),.result_valid(raw_valid),.result_addr(raw_addr),.result_data(raw_data),.done(core_done),
 .debug_state(debug_state),.z_monitor_valid(z_monitor_valid),.z_monitor_stripe(z_monitor_stripe),.z_monitor_addr(z_monitor_addr),.z_monitor_data(z_monitor_data),
 '''+',\n '.join('.'+x+'('+x+')' for x in base)+''');
 wide_phase_alu wide(.split_fields(1'b0),.lhs(add_lhs),.rhs(add_rhs),.y(add_y));
 i24_consumer consumer(.clk(clk),.reset_n(reset_n),.start(start),.add_req(add_req),.add_grant(add_req),
 .add_lhs_bus(add_lhs),.add_rhs_bus(add_rhs),.add_y_bus(add_y),
 .cfg_valid(cfg_valid&&cfg_kind==6),.cfg_addr(cfg_addr[4:0]),.cfg_data(cfg_data),
 .raw_valid(raw_valid&&raw_allow),.raw_ready(join_ready),.raw_addr(raw_addr),.raw_data(raw_data),
 .identity_request_valid(identity_request_valid),.identity_address(identity_address),.identity_valid(identity_valid),.identity_data(identity_data),
 .j_monitor_valid(j_monitor_valid),.j_monitor_address(j_monitor_address),.j_monitor_data(j_monitor_data),
 .wide_monitor_valid(wide_monitor_valid),.wide_monitor_address(wide_monitor_address),.wide_monitor_data(wide_monitor_data),
 .result_valid(result_valid),.result_ready(result_ready),.result_addr(result_addr),.result_data(result_data),.done(done),.error(error),
 '''+',\n '.join('.'+x+'(c_'+x+')' for x in consumer)+''');
 `ifdef VERILATOR
 logic busy,seen_core_done;
 always_ff @(posedge clk)begin
  if(!reset_n)begin busy<=0;seen_core_done<=0;end
  else begin
   if(busy&&!done&&(cfg_valid||start))$fatal(1,"live configuration/restart");
   if(start)begin busy<=1;seen_core_done<=0;end
   if(core_done)seen_core_done<=1;
   if(done)begin
    if(!seen_core_done)$fatal(1,"consumer completed before producer");
    busy<=0;
   end
  end
 end
 `endif
endmodule
'''
(H/'spatial_stream.sv').write_text(s)
# Exact consumer oracle uses source-derived P. The real fixtures must match supplied J/I24.
import numpy as np
f=np.load(H.parent/'spatial_r16_integer/factors.npz')
a=np.repeat(f['a_q40'].reshape(12,1,8),40,axis=1).reshape(480,8).astype(np.int64)
b=np.repeat(f['b_q20'].reshape(12,1,8),40,axis=1).reshape(480,8).astype(np.int64)
def rd(p):return np.array([int(x,16) for x in p.read_text().split()],np.uint32)
def wr(p,v):p.write_text(''.join(f'{int(x)&0xffffffff:08x}\n' for x in v.reshape(-1)))
for d in sorted((H/'fixtures').iterdir()):
 p=rd(d/'gold.hex').view(np.int32).astype(np.int64).reshape(480,8)
 if (d/'identity.hex').exists():bits=rd(d/'identity.hex')
 else:bits=np.zeros(3840,np.uint32);wr(d/'identity.hex',bits)
 j=np.clip(np.rint(bits.view(np.float32).astype(np.float64)*2**20),-2**31,2**31-1).astype(np.int64).reshape(480,8)
 wide=p*a+(b+j)*2**20
 q=wide>>26;rem=wide-(q<<26);q+=((rem>2**25)|((rem==2**25)&((q&1)!=0)))
 i24=np.clip(q,-2**23,2**23-1)
 for name,x in [('j',j),('i24',i24)]:
  if (d/f'{name}.hex').exists():assert np.array_equal(rd(d/f'{name}.hex').view(np.int32).reshape(480,8),x),(d,name)
  else:wr(d/f'{name}.hex',x)
 wr(d/'wide.hex',np.stack([wide&0xffffffff,(wide>>32)&0xffffffff],axis=-1))
print('Consumer wrapper and independent exact oracles ready')
