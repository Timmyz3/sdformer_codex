"""One targeted change: RR between consumers and the producer borrow group."""
from pathlib import Path

H=Path(__file__).resolve().parent
A=H/'adapt_rr'
A.mkdir(exist_ok=True)
for name in ['rr_context.sv','wide_phase_alu.sv','i24_consumer.sv']:
    (A/name).write_text((H/name).read_text())
s=(H/'interleave_stream.sv').read_text()
s=s.replace('logic [1:0] mode_q;', 'logic [2:0] mode_q;logic wide_rr;')
s=s.replace('assign cons_add_grant=cons_add_req;', '''assign cons_add_grant=cons_add_req && !(mode_q==4 && wide_rr && (request[0][5]||request[1][5]));''')
s=s.replace('cons_add_req?cons_lhs', 'cons_add_grant?cons_lhs').replace('cons_add_req?cons_rhs','cons_add_grant?cons_rhs')
s=s.replace('.split_fields(!cons_add_req)', '.split_fields(!cons_add_grant)')
s=s.replace('(!request[c][5] || !cons_add_req)', '(!request[c][5] || !cons_add_grant)')
s=s.replace('.mode({2\'d0,mode_q})', ".mode(mode_q==4?4'd3:{1'd0,mode_q})")
s=s.replace('rr<=0;', 'rr<=0;wide_rr<=0;')
s=s.replace('mode_q<=mode[1:0]', 'mode_q<=mode[2:0]')
s=s.replace('(mode!=1&&mode!=2&&mode!=3)', '(mode!=1&&mode!=2&&mode!=3&&mode!=4)')
s=s.replace("64'(cons_add_req&&request[0][5])+64'(cons_add_req&&request[1][5])", "64'(cons_add_grant&&request[0][5])+64'(cons_add_grant&&request[1][5])")
s=s.replace('&&!cons_add_req)', '&&!cons_add_grant)')
s=s.replace('if(cons_add_req && (request[0][5]||request[1][5]))wide_conflict_cycles<=wide_conflict_cycles+1;', '''if(cons_add_req && (request[0][5]||request[1][5]))begin
    wide_conflict_cycles<=wide_conflict_cycles+1;
    if(mode_q==4)wide_rr<=!wide_rr;
   end''')
s=s.replace('// Consumer priority is exactly the original phase-borrow policy.', '// Mode3 retains fixed consumer priority; mode4 alternates colliding groups.')
(A/'interleave_stream.sv').write_text(s)
for name in ['tb.cpp','stream_tb.cpp']:
    s=(H/name).read_text().replace('active_mode==3?', 'active_mode>=3?')
    s=s.replace(' || d.consumer_wide_waits)', ')')
    (A/name).write_text(s)
for name in ['run.py','run_stream.py','verify.py','audit.py']:
    s=(H/name).read_text().replace('H.parents[1]', 'H.parents[2]')
    s=s.replace('subprocess.run(["/opt/anaconda3/bin/python3.12", str(H / "build_from_sources.py")], check=True)', '')
    s=s.replace('for m in [1, 2, 3]', 'for m in [4]').replace('for mode in [1, 2, 3]', 'for mode in [4]')
    if name=='run_stream.py':
        s=s.replace('assert checks["passed"] and checks["gate_64"]', 'assert checks["passed"] and json.loads((H / "audit_summary.json").read_text())["passed"]')
    if name=='verify.py':
        s=s.replace('U3=u2,','U3=u2, U4=u2,')
        s=s.replace('mode == 3', 'mode >= 3')
        s=s.replace('assert r["consumer_wide_waits"] == 0', 'assert r["consumer_wide_waits"] <= r["wide_conflict_cycles"]')
        s=s.replace('3385 * n + r["consumer_join_wait_cycles"] + r["consumer_output_stalls"]', '3385 * n + r["consumer_join_wait_cycles"] + r["consumer_output_stalls"] + r["consumer_wide_waits"]')
        s=s.replace('if (H / "results_64.json").exists():', 'if (H / "results_controls.json").exists():\n    stream_rows += json.loads((H / "results_controls.json").read_text())\nif (H / "results_64.json").exists():')
    if name=='audit.py':
        s=s.replace('&& cons_add_req)$fatal', '&& cons_add_grant)$fatal')
        s=s.replace('[(1,3),(3,2),(2,3),(3,1)]','[(1,4),(4,3),(2,4),(4,2)]')
        s=s.replace("s = (H / name).read_text()\n    if key:", '''s = (H / name).read_text()
    if name=='i24_consumer.sv':
        s=s.replace('endmodule', ''' + repr('''
 logic audit_wide_blocked;logic [3:0] audit_wide_state;logic [511:0] audit_wide_hold;
 always_ff @(posedge clk)begin
  if(!reset_n)audit_wide_blocked<=0;
  else begin
   if(audit_wide_blocked)begin
    if(state!=audit_wide_state)$fatal(1,"denied consumer advanced");
    for(integer l=0;l<8;l=l+1)if(wide_hold[l]!=audit_wide_hold[l*64+:64])$fatal(1,"denied consumer sum changed");
   end
   audit_wide_blocked<=add_req&&!add_grant;audit_wide_state<=state;
   for(integer l=0;l<8;l=l+1)audit_wide_hold[l*64+:64]<=wide_hold[l];
  end
 end
endmodule
''') + ''')
    if key:''')
        s=s.replace("borrow_consumer_stalls=sum", "consumer_wide_waits=sum(r['consumer_wide_waits'] for r in rows),\n             borrow_consumer_stalls=sum")
    (A/name).write_text(s)
print(A)
