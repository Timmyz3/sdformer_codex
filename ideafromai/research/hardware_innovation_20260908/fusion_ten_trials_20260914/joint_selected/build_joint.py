from pathlib import Path
H=Path(__file__).resolve().parent
B=H.parent/'phase_borrow'
for f in ['prepare.py','run.py','run_stream.py','tb.cpp','stream_tb.cpp','wide_phase_alu.sv','i24_consumer.sv']:
 (H/f).write_text((B/f).read_text())
c=(B/'phase_core.sv').read_text()
c=c.replace('input logic start,input logic [3:0] mode,','input logic start,input logic source_phase,input logic [3:0] mode,')
c=c.replace("source_addr=11'((k/9)*16+load_xy);", "source_addr=11'((k/9)*16+load_xy)^(source_phase?11'd2:11'd0);")
c=c.replace('DRAIN_SEND,FINISH}', 'DRAIN_SEND,FINISH,DIRECT_SEND}')
c=c.replace('result_valid=(state==DRAIN_SEND);result_addr=9\'(row);', "result_valid=(state==DRAIN_SEND || state==DIRECT_SEND);result_addr=(state==DIRECT_SEND)?9'(og*40+fp):9'(row);")
a=c.index(' STORE:begin');b=c.index(' DRAIN_READ:',a)
old=c[a:b]
old=old.replace(' STORE:begin',' STORE:if(mode_q==4)begin',1)
old=old.rstrip()+''' else begin
 for(integer i=0;i<8;i=i+1)result_data[i*32+:32]<=acc[i];
 state<=DIRECT_SEND;
 end
 DIRECT_SEND:if(result_ready)begin
 if(fp<39)begin fp<=fp+1;state<=POSLOAD;end
 else if(og<11)begin og<=og+1;qfill<=0;state<=VLOAD;end
 else state<=FINISH;
 end else output_stalls<=output_stalls+1;
'''
c=c[:a]+old+c[b:]
(H/'phase_core.sv').write_text(c)
c=(B/'consumer_stream.sv').read_text()
c=c.replace('logic resident, core_seen, consumer_seen;', 'logic resident, core_seen, consumer_seen, phase_q, reuse_halo;')
c=c.replace('core_cfg_addr=load_index;', "core_cfg_addr=load_index^(phase_q?11'd2:11'd0);")
c=c.replace('.start(core_start),.mode(mode_q),','.start(core_start),.mode(mode_q),.source_phase(phase_q),')
c=c.replace('load_index<=0;accepted<=0;', 'phase_q<=0;reuse_halo<=0;load_index<=0;accepted<=0;')
c=c.replace('if(load_index==1535) state<=ORIGIN;else load_index<=load_index+1;', "if(load_index==1535) state<=ORIGIN;else load_index<=load_index+((reuse_halo && load_index[1:0]==3)?11'd3:11'd1);")
c=c.replace('else begin tile_id<=tile_id+1;load_index<=0;state<=SOURCE;end', '''else begin tile_id<=tile_id+1;state<=SOURCE;
        if(int'(tile_id)%160==159)begin phase_q<=0;reuse_halo<=0;load_index<=0;end
        else begin phase_q<=!phase_q;reuse_halo<=1;load_index<=2;end
       end''')
(H/'consumer_stream.sv').write_text(c)
