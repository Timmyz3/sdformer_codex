from pathlib import Path
H=Path(__file__).resolve().parent
OLD=H.parents[1]/'fusion_ten_trials_20260914/decompositions/q2_da'
s=(OLD/'decomp_core.sv').read_text()
s=s.replace('output logic [5:0] debug_state','output logic [31:0] hybrid_evaluations,hybrid_selected,hybrid_fallback,hybrid_bypass,\n output logic [31:0] hybrid_saved_mac,hybrid_blocks,\n output logic [5:0] debug_state')
s=s.replace('logic [3:0] da_selmask;logic sub_alu,da_newlive;','logic [3:0] da_selmask;logic sub_alu,da_newlive;\n logic da_ready,da_return_pos,hybrid_mode;\n integer rank_cost,position_cost,da_cost;\n logic [7:0] block_live_next;')
s=s.replace('aux_reads<=0;aux_writes<=0;', 'hybrid_evaluations<=0;hybrid_selected<=0;hybrid_fallback<=0;hybrid_bypass<=0;hybrid_saved_mac<=0;hybrid_blocks<=0;\n aux_reads<=0;aux_writes<=0;',1)
s=s.replace('da_encoded=0;','hybrid_mode=(mode_q==11||mode_q==12||mode_q==13);\n rank_cost=0;position_cost=0;\n for(integer i=0;i<8;i=i+1)begin\n rank_cost=rank_cost+int\'(remaining[i]);\n position_cost=position_cost+int\'(position_live[fp][i]&&block_live[i]);\n end\n block_live_next=block_live;\n block_live_next[qfill]=rank_live[qfill]&&v_live[og*8+qfill];\n da_encoded=0;',1)
s=s.replace('da_selected=0;', 'da_cost=0;for(integer i=0;i<26;i=i+1)da_cost=da_cost+int\'(da_encoded[i]);\n da_selected=0;',1)
s=s.replace('da_group<=0;da_mask<=0;da_pending<=0;da_width_hold<=1;', 'da_ready<=0;da_return_pos<=0;\n da_group<=0;da_mask<=0;da_pending<=0;da_width_hold<=1;',1)
s=s.replace('mode_q<=mode;zrow<=0;', 'da_ready<=0;da_return_pos<=0;\n mode_q<=mode;zrow<=0;',1)
s=s.replace('if(qfill==7)begin fp<=0;da_group<=0;da_mask<=0;state<=(mode_q==15)?DA_BUILD:POSLOAD;end else qfill<=qfill+1;', '''if(qfill==7)begin
 fp<=0;da_group<=0;da_mask<=0;da_ready<=0;da_return_pos<=0;
 state<=((mode_q==15)||(mode_q==13&&block_live_next!=0))?DA_BUILD:POSLOAD;
 end else qfill<=qfill+1;''')
s=s.replace('state<=((position_live[fp]&block_live)==0)?STORE:((mode_q==15)?DA_POS:BASE_MAC);', '''if((position_live[fp]&block_live)==0)state<=STORE;
 else if(mode_q==15)state<=DA_POS;
 else if(hybrid_mode&&position_cost>2)begin
 if(!da_ready)begin da_group<=0;da_mask<=0;da_return_pos<=1;state<=DA_BUILD;end
 else state<=DA_POS;
 end else begin
 if(hybrid_mode)hybrid_bypass<=hybrid_bypass+1;
 state<=BASE_MAC;
 end''')
s=s.replace('if(da_group==1)state<=POSLOAD;else begin da_group<=1;da_mask<=0;end', '''if(da_group==1)begin
 da_ready<=1;state<=da_return_pos?DA_POS:POSLOAD;
 if(hybrid_mode)hybrid_blocks<=hybrid_blocks+1;
 end else begin da_group<=1;da_mask<=0;end''')
s=s.replace('state<=(da_encoded==0)?STORE:DA_MAC;', '''if(hybrid_mode)begin
 hybrid_evaluations<=hybrid_evaluations+1;
 if(((mode_q==11)?da_cost:(da_cost+2))<rank_cost)begin
 hybrid_selected<=hybrid_selected+1;hybrid_saved_mac<=hybrid_saved_mac+32'(rank_cost);
 state<=(da_encoded==0)?STORE:DA_MAC;
 end else begin hybrid_fallback<=hybrid_fallback+1;state<=BASE_MAC;end
 end else state<=(da_encoded==0)?STORE:DA_MAC;''')
(H/'decomp_core.sv').write_text(s)
