"""Create isolated wrapper/TBs from the old phase experiment; never edit OLD."""
from pathlib import Path

H = Path(__file__).resolve().parent
OLD = H.parents[1] / "fusion_ten_trials_20260914"
P = OLD / "phase_borrow"
s = (P / "consumer_stream.sv").read_text()
s = s.replace("compute_source_allow, compute_weight_allow, compute_wide_allow", "compute_source_allow, compute_weight_allow")
s = s.replace(" output logic [63:0] core_borrow_waits, consumer_wide_waits,", " output logic [63:0] core_repair_issues,core_repair_fields,core_normalization_issues,proof_issues,range_fallback_tiles,\n output logic proof_range_ok,output logic [103:0] proof_positive_bounds,proof_negative_bounds,")
a = s.index(" logic borrow_req,")
b = s.index(" typedef enum", a)
s = s[:a] + " logic [31:0] l_repair_issues,l_repair_fields,l_normalization_issues;\n logic l_fallback;\n" + s[b:]
s = s.replace(" phase_core leaf (", " modular_core leaf (")
s = s.replace("  .borrow_req(borrow_req),.borrow_grant(borrow_grant),.borrow_lhs(borrow_lhs),.borrow_rhs(borrow_rhs),.borrow_y(borrow_y),.borrow_waits(l_borrow_waits),", "  .range_ok(proof_range_ok),.positive_bounds(proof_positive_bounds),.negative_bounds(proof_negative_bounds),.fallback_used(l_fallback),\n  .repair_issues(l_repair_issues),.repair_fields(l_repair_fields),.normalization_issues(l_normalization_issues),")
s = s.replace("  .add_req(cons_add_req),.add_grant(cons_add_grant),.add_lhs_bus(cons_lhs),.add_rhs_bus(cons_rhs),.add_y_bus(wide_y),.wide_waits(c_wide_waits),\n", "")
s = s.replace("core_borrow_waits<=0;consumer_wide_waits<=0;", "core_repair_issues<=0;core_repair_fields<=0;core_normalization_issues<=0;proof_issues<=0;range_fallback_tiles<=0;")
s = s.replace("core_borrow_waits<=core_borrow_waits+64'(l_borrow_waits);", "core_repair_issues<=core_repair_issues+64'(l_repair_issues);\n      core_repair_fields<=core_repair_fields+64'(l_repair_fields);\n      core_normalization_issues<=core_normalization_issues+64'(l_normalization_issues);\n      range_fallback_tiles<=range_fallback_tiles+64'(l_fallback);")
s = s.replace("      consumer_wide_waits<=consumer_wide_waits+64'(c_wide_waits);\n", "")
s = s.replace("static_words<=static_words+1;", "static_words<=static_words+1;\n     if(param_kind==4)proof_issues<=proof_issues+1;")
s = s.replace("(mode!=2 && mode!=3)", "mode>2").replace("mode_q<=2", "mode_q<=0")
assert "borrow" not in s and "wide_wait" not in s and "wide_alu" not in s
(H / "consumer_stream.sv").write_text(s)
(H / "i24_consumer.sv").write_text((OLD / "dataflow/d3_interleave/i24_consumer.sv").read_text())

for name in ["tb.cpp", "stream_tb.cpp"]:
    s = (P / name).read_text()
    s = s.replace("d.compute_wide_allow=1;", "")
    s = s.replace("   d.compute_wide_allow=(!stall || n%17!=4);\n", "")
    s = s.replace("SHOW(core_borrow_waits);SHOW(consumer_wide_waits);", "SHOW(core_repair_issues);SHOW(core_repair_fields);SHOW(core_normalization_issues);SHOW(proof_issues);SHOW(range_fallback_tiles);SHOW(proof_range_ok);")
    s = s.replace(' #x "\\\":"<<d.x', ' #x "\\\":"<<uint64_t(d.x)')
    check = '''
    if(d.proof_issues!=(command?0:864))return 30;
    bool expected_range=true;
    for(int lane=0;lane<8;++lane){
     int pos=0,neg=0;
     for(int k=0;k<864;++k){int q=int32_t(param[4][k*8+lane]);if(q>=0)pos+=q;else neg+=q;}
     auto field=[&](const auto& bus){unsigned bit=lane*13,word=bit/32;uint64_t v=bus[word];if(word<3)v|=uint64_t(bus[word+1])<<32;int z=(v>>(bit%32))&8191;return z&4096?z-8192:z;};
     if(field(d.proof_positive_bounds)!=pos || field(d.proof_negative_bounds)!=neg)return 31;
     expected_range &= pos<=511 && neg>=-512;
    }
    if(bool(d.proof_range_ok)!=expected_range)return 32;
    if(d.range_fallback_tiles!=((mode==1&&!expected_range)?d.retired_tiles:0))return 33;
    if(d.core_normalization_issues!=(mode==2?10*d.retired_tiles:0))return 34;
    if(mode!=2&&(d.core_repair_issues||d.core_repair_fields))return 35;
'''
    s = s.replace("   if(d.done) {", "   if(d.done) {" + check)
    assert "borrow" not in s and "compute_wide" not in s
    (H / name).write_text(s)
print("Isolated wrapper, unchanged complete consumer, and checking TBs prepared")
