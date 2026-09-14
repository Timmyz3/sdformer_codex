"""Add actual existing-consumer-ALU borrowing to the unchanged RR baseline."""
from pathlib import Path
import shutil

H = Path(__file__).resolve().parent
B = H.parents[1]
R = B / 'fusion_review_followup_20260914/rr_modular'
P = B / 'fusion_ten_trials_20260914/phase_borrow'

def replace(s, a, b):
    assert a in s, a
    return s.replace(a, b)

s = (R / 'rr_context.sv').read_text()
s = replace(s, 'output logic [4:0] resource_request,', '''output logic [5:0] resource_request,
 output logic [511:0] borrow_lhs,borrow_rhs,input logic [415:0] borrow_y,''')
s = replace(s, 'ZADD,REPAIR_ADD,NORMALIZE_ADD,BASE_MAC:resource_request=5\'b10100;', '''ZADD:resource_request=(mode_q==3)?6'b100100:6'b010100;
  REPAIR_ADD,NORMALIZE_ADD,BASE_MAC:resource_request=6'b010100;''')
# Explicitly extend the five original request masks; no grant semantics change.
for v in ['00100','00001','00010','01000']:
    s = s.replace("5'b" + v, "6'b0" + v)
s = replace(s, 'for(integer l=0;l<8;l=l+1)begin\n  multiply_coefficient', '''for(integer l=0;l<8;l=l+1)begin
  borrow_lhs[l*64+:64]={12'd0,z_hold[l]};borrow_rhs[l*64+:64]=0;
  for(integer p=0;p<4;p=p+1)
   borrow_rhs[l*64+p*13+:13]=active[p]?{{10{q_hold[l][2]}},q_hold[l]}:13'd0;
  multiply_coefficient''')
s = replace(s, 'if(mode_q==2)pending', 'if(mode_q>=2)pending')
s = replace(s, '(mode_q==2 || (mode_q==1', '(mode_q>=2 || (mode_q==1')
s = replace(s, 'if(mode_q==0)begin\n      if(pair_sel)', '''if(mode_q==3)z_mem[i][zrow]<=borrow_y[i*52+:52];
     else if(mode_q==0)begin
      if(pair_sel)''')
(H / 'rr_context.sv').write_text(s)

s = (R / 'interleave_stream.sv').read_text()
s = replace(s, 'output logic [63:0] proof_issues,', '''output logic [63:0] shared_wide_grants,borrow_grants,borrow_consumer_stalls,borrow_rr_stalls,
 output logic [63:0] wide_conflict_cycles,consumer_wide_waits,
 output logic [63:0] proof_issues,''')
s = replace(s, 'logic [4:0] request[0:1];', 'logic [5:0] request[0:1];')
s = replace(s, 'logic [255:0] shared_alu_result;', '''logic [255:0] shared_alu_result;
 logic [511:0] borrow_lhs[0:1],borrow_rhs[0:1],cons_lhs,cons_rhs,wide_lhs,wide_rhs,wide_y;
 logic [415:0] borrow_y;
 logic cons_add_req,cons_add_grant,borrow_owner,borrow_active;
 logic [31:0] c_wide_waits;
 // Consumer priority is exactly the original phase-borrow policy. A producer
 // ZADD receives its z and wide permission atomically through resource_grant.
 assign cons_add_grant=cons_add_req;
 assign borrow_owner=grant[1]&&request[1][5];
 assign borrow_active=(grant[0]&&request[0][5])||(grant[1]&&request[1][5]);
 assign wide_lhs=cons_add_req?cons_lhs:(borrow_active?borrow_lhs[borrow_owner]:512'd0);
 assign wide_rhs=cons_add_req?cons_rhs:(borrow_active?borrow_rhs[borrow_owner]:512'd0);
 wide_phase_alu wide_alu(.split_fields(!cons_add_req),.lhs(wide_lhs),.rhs(wide_rhs),.y(wide_y));
 genvar wl;
 generate for(wl=0;wl<8;wl=wl+1)begin:WIDE_RESULT
  assign borrow_y[wl*52+:52]=wide_y[wl*64+:52];
 end endgenerate''')
s = replace(s, '(!request[c][1] || compute_weight_allow);', '(!request[c][1] || compute_weight_allow) && (!request[c][5] || !cons_add_req);')
s = replace(s, '.resource_grant(grant[c]),', '.borrow_lhs(borrow_lhs[c]),.borrow_rhs(borrow_rhs[c]),.borrow_y(borrow_y),\n   .resource_grant(grant[c]),')
s = replace(s, 'i24_consumer consumer(\n', '''i24_consumer consumer(
  .add_req(cons_add_req),.add_grant(cons_add_grant),.add_lhs_bus(cons_lhs),.add_rhs_bus(cons_rhs),.add_y_bus(wide_y),.wide_waits(c_wide_waits),
''')
s = replace(s, 'shared_source_grants<=0;', '''shared_wide_grants<=0;borrow_grants<=0;borrow_consumer_stalls<=0;borrow_rr_stalls<=0;wide_conflict_cycles<=0;consumer_wide_waits<=0;
   shared_source_grants<=0;''')
s = replace(s, 'if(proof_active)begin\n    proof_issues', '''if(cons_add_grant || borrow_active)shared_wide_grants<=shared_wide_grants+1;
   if(borrow_active)borrow_grants<=borrow_grants+1;
   if(cons_add_req && (request[0][5]||request[1][5]))wide_conflict_cycles<=wide_conflict_cycles+1;
   borrow_consumer_stalls<=borrow_consumer_stalls+64'(cons_add_req&&request[0][5])+64'(cons_add_req&&request[1][5]);
   borrow_rr_stalls<=borrow_rr_stalls+64'(request[0][5]&&!grant[0]&&!cons_add_req)+64'(request[1][5]&&!grant[1]&&!cons_add_req);
   if(proof_active)begin
    proof_issues''')
s = replace(s, 'consumer_cycles<=consumer_cycles+64\'(c_cycles);', "consumer_wide_waits<=consumer_wide_waits+64'(c_wide_waits);\n       consumer_cycles<=consumer_cycles+64'(c_cycles);")
s = replace(s, '(mode!=1&&mode!=2)', '(mode!=1&&mode!=2&&mode!=3)')
(H / 'interleave_stream.sv').write_text(s)
for f in ['i24_consumer.sv', 'wide_phase_alu.sv']:
    shutil.copyfile(P / f, H / f)

for name in ['tb.cpp','stream_tb.cpp']:
    s = (R / name).read_text()
    s = replace(s, 'if(argc!=6)' if name=='tb.cpp' else 'if(argc!=13)',
                'if(argc!=6 && argc!=7)' if name=='tb.cpp' else 'if(argc!=13 && argc!=14)')
    s = replace(s, 'd.mode=mode;', '''unsigned active_mode=command && argc>''' + ('6' if name=='tb.cpp' else '13') + '''?std::stoul(argv[''' + ('6' if name=='tb.cpp' else '13') + ''']):mode;
  d.mode=active_mode;''')
    # All command-specific expectations use the mode actually submitted.
    s = s.replace('(mode==1', '(active_mode==1').replace('(mode==2', '(active_mode==2').replace('if(mode!=2', 'if(active_mode!=2')
    s = replace(s, 'd.proof_issues+d.core_first_issues+d.core_mac_issues', 'd.proof_issues+(active_mode==3?0:d.core_first_issues)+d.core_mac_issues')
    s = replace(s, 'd.conflict_cycles!=d.core_arbitration_stalls', 'd.conflict_cycles+d.borrow_consumer_stalls!=d.core_arbitration_stalls')
    s = replace(s, 'if(d.shared_z_grants', '''if(d.borrow_grants!=(active_mode==3?d.core_first_issues:0))return 41;
    if(d.shared_wide_grants!=d.borrow_grants+d.consumer_add_issues || d.consumer_wide_waits)return 42;
    if(d.shared_z_grants''')
    s = replace(s, '<<mode<<', '<<active_mode<<')
    s = replace(s, 'SHOW(proof_issues);', '''SHOW(shared_wide_grants);SHOW(borrow_grants);SHOW(borrow_consumer_stalls);SHOW(borrow_rr_stalls);SHOW(wide_conflict_cycles);SHOW(consumer_wide_waits);SHOW(proof_issues);''')
    (H / name).write_text(s)

for name in ['run.py','run_stream.py','verify.py']:
    s = (R / name).read_text().replace('"python3.12"', '"/opt/anaconda3/bin/python3.12"')
    s = s.replace('"i24_consumer.sv",', '"i24_consumer.sv", "wide_phase_alu.sv",')
    s = s.replace('for m in [1, 2]', 'for m in [1, 2, 3]').replace('for mode in [1, 2]', 'for mode in [1, 2, 3]')
    if name=='verify.py':
        s = s.replace('if (H / "results_64.json").exists():', 'if (H / "results_cross.json").exists():\n    stream_rows += json.loads((H / "results_cross.json").read_text())\nif (H / "results_64.json").exists():')
        s = s.replace('U1=u1, U2=u2,', 'U1=u1, U2=u2, U3=u2,')
        s = s.replace('r["proof_issues"] + u + p["M"]', 'r["proof_issues"] + (0 if mode == 3 else u) + p["M"]')
        s = s.replace('r["conflict_cycles"] == r["core_arbitration_stalls"]', 'r["conflict_cycles"] + r["borrow_consumer_stalls"] == r["core_arbitration_stalls"]')
        s = s.replace('assert r["shared_z_grants"]', '''assert r["borrow_grants"] == (u if mode == 3 else 0)
    assert r["shared_wide_grants"] == r["borrow_grants"] + r["consumer_add_issues"]
    assert r["consumer_wide_waits"] == 0
    assert r["shared_z_grants"]''')
        s = s.replace('gate = all(small[first, 2]["total_cycles"] < small[first, 1]["total_cycles"] for first in [159, 19197])', 'gate = all((first, mode) in small for first in [159, 19197] for mode in [1, 2, 3])')
    (H / name).write_text(s)
