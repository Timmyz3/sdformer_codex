"""Build the isolated shared-resource integration; inputs are read-only."""
from pathlib import Path

H = Path(__file__).resolve().parent
MOD = H.parent / "modular_packing"
D3 = H.parents[1] / "fusion_ten_trials_20260914/dataflow/d3_interleave"

def replace(s, old, new):
    assert old in s, old[:100]
    return s.replace(old, new)

s = (MOD / "modular_core.sv").read_text().replace("module modular_core(", "module rr_context(")
s = replace(s, " output logic range_ok,fallback_used,\n output logic [103:0] positive_bounds,negative_bounds,", """ input logic range_ok,resource_grant,
 output logic fallback_used,compute_done,
 output logic [4:0] resource_request,
 output logic weight_is_q2,output logic [9:0] weight_address,input logic [255:0] weight_data,
 output logic [255:0] alu_lhs,alu_rhs,output logic [151:0] alu_coefficient,
 output logic signed [12:0] alu_scalar,output logic alu_mac,output logic [2:0] alu_format,
 input logic [255:0] alu_result,
 output logic [31:0] arbitration_stalls,repair_arbitration_stalls,normalization_arbitration_stalls,""")
s = replace(s, "q_mem[0:7][0:863],q_hold", "q_hold")
s = replace(s, "v_mem[0:7][0:95],qblock", "qblock")
s = replace(s, " logic signed [12:0] positive_bound[0:7],negative_bound[0:7];\n", "")
s = replace(s, "product[0:7],lhs", "lhs")
s = replace(s, "logic proof_active,split5,split8,split10,split13,repair_needed;", "logic repair_needed;")
s = replace(s, " cycles<=0;source_words", " cycles<=0;arbitration_stalls<=0;repair_arbitration_stalls<=0;normalization_arbitration_stalls<=0;source_words")
s = replace(s, "  if(state==ZSCAN || state==ZREAD || state==REPAIR_READ || state==NORMALIZE_READ ||\n     (state==BASE_MAC && selected_rank==3'(i)))", "  if(resource_grant && (state==ZSCAN || state==ZREAD || state==REPAIR_READ || state==NORMALIZE_READ ||\n     (state==BASE_MAC && selected_rank==3'(i))))")
a = s.index(" proof_active=cfg_valid")
b = s.index(" for(integer l=0;l<8;l=l+1)begin", a)
s = s[:a] + """ resource_request=0;
 case(state)
  ZCLEAR,ZREAD,ZSCAN,REPAIR_READ,NORMALIZE_READ:resource_request=5'b00100;
  L_LOAD:if(in_bounds)resource_request=5'b00001;
  QREAD:resource_request=5'b00010;
  VLOAD:if(rank_live[qfill]&&v_live[og*8+qfill])resource_request=5'b00010;
  ZADD,REPAIR_ADD,NORMALIZE_ADD,BASE_MAC:resource_request=5'b10100;
  STORE,DRAIN_READ:resource_request=5'b01000;
  default:begin end
 endcase
 weight_is_q2=(state==VLOAD);weight_address=weight_is_q2?10'(og*8+qfill):10'(k);
 alu_mac=(state==BASE_MAC);alu_scalar=scalar;alu_format=0;
 if(state==REPAIR_ADD || state==NORMALIZE_ADD)alu_format=4;
 else if(state==ZADD)alu_format={1'b0,mode_q}+3'd1;
""" + s[b:]
s = replace(s, "  positive_bounds[l*13+:13]=positive_bound[l];negative_bounds[l*13+:13]=negative_bound[l];\n  if(positive_bound[l]>13'sd511 || negative_bound[l]<-13'sd512)range_ok=0;\n", "")
s = replace(s, "  product[l]=$signed(multiply_coefficient[l])*$signed(scalar);\n  lhs[l]=acc[l];rhs[l]=product[l];", "  lhs[l]=acc[l];rhs[l]=0;")
a = s.index("  if(proof_active)begin")
b = s.index("state==REPAIR_ADD || state==NORMALIZE_ADD", a)
s = s[:a] + "  if(" + s[b:]
s = replace(s, " end\n result_valid=(state==DRAIN_SEND)", "  alu_lhs[l*32+:32]=lhs[l];alu_rhs[l*32+:32]=rhs[l];\n  alu_coefficient[l*19+:19]=multiply_coefficient[l];\n end\n result_valid=(state==DRAIN_SEND)")
a = s.index(" // The only producer data adders:")
b = s.index(" always_comb begin\n  correction_count", a)
s = s[:a] + " always_comb begin\n  for(integer lane=0;lane<8;lane=lane+1)add_y[lane]=alu_result[lane*32+:32];\n end\n" + s[b:]
s = replace(s, "done<=0;fallback_used<=0;", "done<=0;compute_done<=0;fallback_used<=0;")
s = replace(s, "q_hold[i]<=0;z_hold[i]<=0;acc[i]<=0;positive_bound[i]<=0;negative_bound[i]<=0;", "q_hold[i]<=0;z_hold[i]<=0;acc[i]<=0;")
a = s.index("   4:for(integer i=0;i<8;i=i+1)begin")
b = s.index("    v_live[cfg_addr[6:0]]<=cfg_v_live;", a)
s = s[:a] + "   4:begin end\n   5:begin\n" + s[b:]
s = replace(s, "  case(state)\n   IDLE:if(start)begin", """  if(resource_request!=0 && !resource_grant)begin
   if(resource_request[0]&&!source_allow)source_stalls<=source_stalls+1;
   else if(resource_request[1]&&!weight_allow)weight_stalls<=weight_stalls+1;
   else arbitration_stalls<=arbitration_stalls+1;
   if(state==REPAIR_READ || state==REPAIR_ADD)repair_arbitration_stalls<=repair_arbitration_stalls+1;
   if(state==NORMALIZE_READ || state==NORMALIZE_ADD)normalization_arbitration_stalls<=normalization_arbitration_stalls+1;
  end
  if(resource_request==0 || resource_grant)case(state)
   IDLE:if(start)begin
    compute_done<=0;""")
s = replace(s, "q_hold[i]<=q_mem[i][k];", "q_hold[i]<=weight_data[i*32+:3];")
s = replace(s, "?v_mem[i][og*8+qfill]:16'sd0;", "?weight_data[i*32+:16]:16'sd0;")
s = replace(s, "else begin row<=0;state<=DRAIN_READ;end", "else begin compute_done<=1;row<=0;state<=DRAIN_READ;end")
# Keep grant-independent requests in a separate combinational process from
# grant-gated memory reads and arithmetic operands, as in the original D3.
a = s.index(" z_read_bus=0;")
b = s.index(" resource_request=0;", a)
read_logic = s[a:b]
s = s[:a] + s[b:]
s = replace(s, "alu_mac=(state==BASE_MAC);alu_scalar=scalar;alu_format=0;", "alu_mac=(state==BASE_MAC);alu_format=0;")
a = s.index(" for(integer l=0;l<8;l=l+1)begin")
s = s[:a] + " end\n always_comb begin\n" + read_logic + " alu_scalar=scalar;\n" + s[a:]
assert all(term not in s for term in ["q_mem", "v_mem", "product[", "positive_bound", "proof_active", "genvar"])
(H / "rr_context.sv").write_text(s)

s = (D3 / "interleave_stream.sv").read_text().replace("core_dual_updates", "core_merged_updates").replace("l_dual_updates", "l_merged_updates")
s = replace(s, " output logic [63:0] core_cycles,", """ output logic [63:0] proof_issues,range_fallback_tiles,core_repair_issues,core_repair_fields,core_normalization_issues,
 output logic [63:0] core_repair_arbitration_stalls,core_normalization_arbitration_stalls,
 output logic proof_range_ok,output logic [103:0] proof_positive_bounds,proof_negative_bounds,
 output logic [63:0] core_cycles,""")
s = replace(s, "leaf_done,unused_first_done,compute_done", "leaf_done,compute_done,fallback_used")
s = replace(s, "logic [1:0] child_mac,child_split;", "logic [1:0] child_mac;logic [2:0] child_format[0:1];")
s = replace(s, "logic weight_owner,alu_owner,alu_active,alu_mac,alu_split;", """logic weight_owner,alu_owner,alu_active,alu_mac,proof_active;
 logic [2:0] alu_format;
 logic signed [12:0] positive_bound[0:7],negative_bound[0:7];
 logic [31:0] l_repair_issues[0:1],l_repair_fields[0:1],l_normalization_issues[0:1];
 logic [31:0] l_repair_arbitration_stalls[0:1],l_normalization_arbitration_stalls[0:1];""")
s = replace(s, "  alu_active=(grant[0]&&request[0][4])||(grant[1]&&request[1][4]);\n  alu_mac=alu_active&&child_mac[alu_owner];alu_split=alu_active&&child_split[alu_owner];", """  proof_active=(state==PARAMETERS&&parameter_valid&&param_kind==4);
  alu_active=proof_active||(grant[0]&&request[0][4])||(grant[1]&&request[1][4]);
  alu_mac=alu_active&&!proof_active&&child_mac[alu_owner];
  alu_format=proof_active?3'd1:(alu_active?child_format[alu_owner]:3'd0);
  proof_range_ok=1;
  for(integer i=0;i<8;i=i+1)begin
   proof_positive_bounds[i*13+:13]=positive_bound[i];proof_negative_bounds[i*13+:13]=negative_bound[i];
   if(positive_bound[i]>13'sd511 || negative_bound[i]<-13'sd512)proof_range_ok=0;
  end""")
s = replace(s, "   rhs[l]=alu_mac?product[l]:(alu_active?$signed(child_rhs[alu_owner][l*32+:32]):32'sd0);", """   rhs[l]=alu_mac?product[l]:(alu_active?$signed(child_rhs[alu_owner][l*32+:32]):32'sd0);
   if(proof_active)begin
    lhs[l]=(param_row==0)?32'd0:{6'd0,negative_bound[l],positive_bound[l]};
    rhs[l]={6'd0,(parameter_data[l*32+2]?{{10{parameter_data[l*32+2]}},parameter_data[l*32+:3]}:13'd0),
                    (parameter_data[l*32+2]?13'd0:{10'd0,parameter_data[l*32+:3]})};
   end""")
s = replace(s, "   else if(gb==13)assign cin=alu_split?1'b0:BIT[gb-1].CARRY.cout;\n   else assign cin=BIT[gb-1].CARRY.cout;", """   else assign cin=(((gb==13 || gb==26)&&alu_format==1) || ((gb%10==0)&&alu_format==2) ||
                    ((gb%8==0)&&alu_format==3) || ((gb%5==0)&&alu_format==4))?1'b0:BIT[gb-1].CARRY.cout;""")
s = replace(s, "  core_start[1]=has_second && ((state==LAUNCH0 && mode_q==2) ||\n   (state==RUN && !second_started && mode_q!=2 && (mode_q==1?unused_first_done[0]:compute_done[0])));", "  core_start[1]=has_second && state==LAUNCH0;")
s = replace(s, "thread_context core(", "rr_context core(")
s = replace(s, ".start(core_start[c]),.mode(4'd15)", ".start(core_start[c]),.mode({2'd0,mode_q}),.range_ok(proof_range_ok),.fallback_used(fallback_used[c])")
s = replace(s, ".alu_split(child_split[c])", ".alu_format(child_format[c])")
s = replace(s, ".first_done(unused_first_done[c]),", "")
s = replace(s, ".dual_updates(l_merged_updates[c])", ".merged_updates(l_merged_updates[c])")
s = replace(s, "   .arbitration_stalls(l_arbitration_stalls[c])", """   .arbitration_stalls(l_arbitration_stalls[c]),
   .repair_issues(l_repair_issues[c]),.repair_fields(l_repair_fields[c]),.normalization_issues(l_normalization_issues[c]),
   .repair_arbitration_stalls(l_repair_arbitration_stalls[c]),.normalization_arbitration_stalls(l_normalization_arbitration_stalls[c])""")
s = replace(s, "shared_alu_grants<=0;", "shared_alu_grants<=0;proof_issues<=0;range_fallback_tiles<=0;\n   core_repair_issues<=0;core_repair_fields<=0;core_normalization_issues<=0;\n   core_repair_arbitration_stalls<=0;core_normalization_arbitration_stalls<=0;")
s = replace(s, "  if(!reset_n)begin", "  if(!reset_n)begin\n   for(integer i=0;i<8;i=i+1)begin positive_bound[i]<=0;negative_bound[i]<=0;end")
s = replace(s, "if((grant[0]&&request[0][4])||(grant[1]&&request[1][4]))shared_alu_grants<=shared_alu_grants+1;", """if(proof_active||(grant[0]&&request[0][4])||(grant[1]&&request[1][4]))shared_alu_grants<=shared_alu_grants+1;
   if(proof_active)begin
    proof_issues<=proof_issues+1;
    for(integer i=0;i<8;i=i+1)begin positive_bound[i]<=shared_alu_result[i*32+:13];negative_bound[i]<=shared_alu_result[i*32+13+:13];end
   end""")
marker = "    if(|leaf_done)core_cycles<="
at = s.index(marker)
extra = "    if(|leaf_done)range_fallback_tiles<=range_fallback_tiles+(leaf_done[0]?64'(fallback_used[0]):64'd0)+(leaf_done[1]?64'(fallback_used[1]):64'd0);\n"
for name in ["repair_issues", "repair_fields", "normalization_issues", "repair_arbitration_stalls", "normalization_arbitration_stalls"]:
    extra += f"    if(|leaf_done)core_{name}<=core_{name}+(leaf_done[0]?64'(l_{name}[0]):64'd0)+(leaf_done[1]?64'(l_{name}[1]):64'd0);\n"
s = s[:at] + extra + s[at:]
s = replace(s, "||mode>2)", "||(mode!=1&&mode!=2))")
s = replace(s, "second_started<=has_second&&mode_q==2;", "second_started<=has_second;")
assert all(term not in s for term in ["unused_first_done", "child_split", "alu_split", "thread_context"])
(H / "interleave_stream.sv").write_text(s)
(H / "i24_consumer.sv").write_text((D3 / "i24_consumer.sv").read_text())

# Test harnesses preserve the old native source/identity input contract.
proof_check = (MOD / "stream_tb.cpp").read_text()
a = proof_check.index("    if(d.proof_issues!=")
b = proof_check.index("    if(outputs!=", a)
proof_check = proof_check[a:b]
shared_check = """
    if(d.shared_alu_grants!=d.proof_issues+d.core_first_issues+d.core_mac_issues+d.core_repair_issues+d.core_normalization_issues)return 36;
    if(d.shared_z_grants!=d.core_z_vector_reads+d.core_z_scalar_reads+d.core_z_writes)return 37;
    if(d.shared_source_grants!=d.core_source_words || d.shared_weight_grants!=d.core_weight_words)return 38;
    if(d.shared_psum_grants!=d.core_psum_reads+d.core_psum_writes || d.conflict_cycles!=d.core_arbitration_stalls)return 39;
"""
new_fields = "SHOW(proof_issues);SHOW(range_fallback_tiles);SHOW(proof_range_ok);SHOW(core_repair_issues);SHOW(core_repair_fields);SHOW(core_normalization_issues);SHOW(core_repair_arbitration_stalls);SHOW(core_normalization_arbitration_stalls);"
for name in ["tb.cpp", "stream_tb.cpp"]:
    s = (D3 / name).read_text().replace("core_dual_updates", "core_merged_updates")
    s = replace(s, ' #x "\\\":"<<d.x', ' #x "\\\":"<<uint64_t(d.x)')
    s = replace(s, "   if(d.done) {", "   if(d.done) {\n" + proof_check + shared_check)
    s = replace(s, "    SHOW(total_cycles)", "    " + new_fields + "SHOW(total_cycles)")
    if name == "tb.cpp":
        s = replace(s, "if(argc!=5)return 2;", "if(argc!=6)return 2;")
        s = replace(s, "tile=std::stoi(argv[4]);", "tile=std::stoi(argv[4]),count=std::stoi(argv[5]);")
        s = replace(s, "d.tile_count=1;", "d.tile_count=count;")
        s = replace(s, "    if(c>=96 || ly<0 || ly>=4 || lx<0 || lx>=4)return 17;\n    d.source_data=src.at(c*16+ly*4+lx);", """    if(c>=96 || ly<0 || ly>=4 || lx<0 || lx>=4+2*(count-1))return 17;
    if(count==1)d.source_data=src.at(c*16+ly*4+lx);
    else {for(int pos=1;pos<16;++pos)if(src.at(c*16+pos)!=src.at(c*16))return 40;d.source_data=src.at(c*16);}""")
        s = replace(s, "if(d.identity_tile!=tile)return 18;", "if(d.identity_tile<tile || d.identity_tile>=tile+count)return 18;")
        s = replace(s, "if(d.j_monitor_address!=js)return 27;", "if(d.j_monitor_address!=js%480)return 27;")
        s = replace(s, "jgold.at(js*8+l)", "jgold.at((js%480)*8+l)")
        s = replace(s, "if(d.raw_monitor_address!=raws)return 19;", "if(d.raw_monitor_address!=raws%480)return 19;")
        s = replace(s, "raw.at(raws*8+l)", "raw.at((raws%480)*8+l)")
        s = replace(s, "if(d.result_tile!=tile || d.result_address!=outputs || bool(d.result_tile_last)!=(outputs==479) || bool(d.result_job_last)!=(outputs==479))return 4;", "if(d.result_tile!=tile+outputs/480 || d.result_address!=outputs%480 || bool(d.result_tile_last)!=(outputs%480==479) || bool(d.result_job_last)!=(outputs==480*count-1))return 4;")
        s = replace(s, "gold.at(outputs*8+l)", "gold.at((outputs%480)*8+l)")
        s = replace(s, "outputs!=480 || raws!=480 || js!=480 || d.retired_tiles!=1", "outputs!=480*count || raws!=480*count || js!=480*count || d.retired_tiles!=count")
        s = replace(s, "identity_count!=480", "identity_count!=480*count")
        s = replace(s, "    SHOW(proof_issues)", "    SHOW(proof_issues)")
    (H / name).write_text(s)
print("Shared RR top, arithmetic-free contexts and complete checking harnesses generated")
