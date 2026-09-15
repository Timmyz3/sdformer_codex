from pathlib import Path

H=Path(__file__).resolve().parent
OLD=H.parents[1]/'representation_transfer_20260914/bitmap_rr'
s=(OLD/'interleave_stream.sv').read_text()
s=s.replace('output logic source_request_valid,output logic [22:0] source_address,',
'''output logic source_request_valid,output logic [14:0] source_tile,output logic [10:0] source_address,
 output logic origin_request_valid,output logic [14:0] origin_tile,input logic origin_valid,input logic [31:0] origin_data,''')
s=s.replace('output logic [63:0] padding_words,origin_words,source_load_stalls,output_beats,',
            'output logic [63:0] padding_words,origin_words,origin_stalls,source_load_stalls,output_beats,')
s=s.replace(' output logic result_valid,input logic result_ready,',
''' output logic wide_monitor_valid,output logic [8:0] wide_monitor_address,output logic [511:0] wide_monitor_data,
 output logic result_valid,input logic result_ready,''')
s=s.replace(' logic signed [15:0] origin_y,origin_x;\n integer native_y,native_x,native_c;logic in_bounds;\n','')
start=s.index("  origin_y=16'")
end=s.index('  core_cfg_valid=0;',start)
s=s[:start]+'''  source_request_valid=(state==SOURCE);source_tile=load_tile;source_address=load_index;
  origin_request_valid=(state==ORIGIN);origin_tile=load_tile;
'''+s[end:]
s=s.replace('state==SOURCE&&(!in_bounds||source_valid)','state==SOURCE&&source_valid')
s=s.replace('core_cfg_data[9:0]=in_bounds?source_data:10\'d0;','core_cfg_data[9:0]=source_data;')
s=s.replace('else if(state==ORIGIN)begin','else if(state==ORIGIN&&origin_valid)begin')
s=s.replace('core_cfg_data[31:0]={origin_x,origin_y};','core_cfg_data[31:0]=origin_data;')
s=s.replace('origin_words<=0;','origin_words<=0;origin_stalls<=0;')
s=s.replace('SOURCE:if(!in_bounds||source_valid)begin','SOURCE:if(source_valid)begin')
s=s.replace('source_load_words<=source_load_words+1;if(in_bounds)external_source_words<=external_source_words+1;else padding_words<=padding_words+1;',
            'source_load_words<=source_load_words+1;external_source_words<=external_source_words+1;')
s=s.replace('    ORIGIN:begin','    ORIGIN:if(origin_valid)begin')
s=s.replace('    end\n    LAUNCH0:begin','    end else origin_stalls<=origin_stalls+1;\n    LAUNCH0:begin')
s=s.replace('  .j_monitor_valid(j_monitor_valid)',
'''  .wide_monitor_valid(wide_monitor_valid),.wide_monitor_address(wide_monitor_address),.wide_monitor_data(wide_monitor_data),
  .j_monitor_valid(j_monitor_valid)''')
assert 'in_bounds' not in s
(H/'interleave_stream.sv').write_text(s)
c=(OLD/'rr_context.sv').read_text().replace('qblock[0:7][0:7]','qblock[0:23][0:7]')
c=c.replace('z_mem[0:7][0:9]','z_mem[0:7][0:39]')
c=c.replace('logic [2:0] qblock_read_index','logic [4:0] qblock_read_index')
c=c.replace("qblock_read_index=(state==BM_CACHED_POP)?3'(bm_plane):selected_rank;", "qblock_read_index=(state==BM_CACHED_POP)?5'(bm_plane):{2'd0,selected_rank};")
c=c.replace('logic [3:0] read_zrow','logic [5:0] read_zrow').replace('logic [3:0] z_read_address','logic [5:0] z_read_address')
c=c.replace('logic [3:0] check_bm_row','logic [5:0] check_bm_row')
c=c.replace("read_zrow=4'(fp%10);","read_zrow=6'(fp%10);").replace(":4'(zrow);",":6'(zrow);")
(H/'rr_context.sv').write_text(c)
u=(OLD/'i24_consumer.sv').read_text()
u=u.replace('  output logic done, error,',
'''  output logic wide_monitor_valid, output logic [8:0] wide_monitor_address, output logic [511:0] wide_monitor_data,
  output logic done, error,''')
u=u.replace('    add_req=(state==ADD_BIAS || state==ADD_IDENTITY);',
'''    wide_monitor_valid=(state==ROUND);wide_monitor_address=row;
    for(integer l=0;l<8;l=l+1)wide_monitor_data[l*64+:64]=wide_hold[l];
    add_req=(state==ADD_BIAS || state==ADD_IDENTITY);''')
(H/'i24_consumer.sv').write_text(u)
(H/'wide_phase_alu.sv').write_text((OLD/'wide_phase_alu.sv').read_text())

t=(OLD/'stream_tb.cpp').read_text()
start=t.index('struct NpyMap {');end=t.index('double sc_time_stamp()',start)
t=t[:start]+'''struct Fixture {
 std::vector<uint16_t> source;
 std::vector<uint32_t> raw,id,j,gold,wide,origin;
 Fixture(const std::string& p):source(readbin<uint16_t>(p+"/source.bin")),raw(readbin<uint32_t>(p+"/raw.bin")),id(readbin<uint32_t>(p+"/identity_fp32.bin")),j(readbin<uint32_t>(p+"/identity.bin")),gold(readbin<uint32_t>(p+"/gold.bin")),wide(readbin<uint32_t>(p+"/wide.bin")),origin(readbin<uint32_t>(p+"/origin.bin")) {
  if(source.size()!=1536||raw.size()!=3840||id.size()!=3840||j.size()!=3840||gold.size()!=3840||wide.size()!=7680||origin.size()!=2)throw std::runtime_error("fixture size");
 }
};
'''+t[end:]
start=t.index(' Verilated::commandArgs');end=t.index(' std::vector<uint32_t> param[12]',start)
t=t[:start]+''' Verilated::commandArgs(argc,argv);if(argc!=5 && argc!=6)return 2;
 std::string dir=argv[1];std::ifstream manifest(argv[2]);std::string p;std::vector<Fixture> fixtures;
 while(manifest>>p)fixtures.emplace_back(p);
 unsigned mode=std::stoul(argv[3]),stall=std::stoul(argv[4]),first=0,count=fixtures.size(),repeats=2;
 uint64_t limit=12000000;if(!count)return 3;
'''+t[end:]
t=t.replace('d.source_valid=0;d.identity_valid=0;','d.source_valid=0;d.identity_valid=0;d.origin_valid=0;')
t=t.replace('command && argc>13?std::stoul(argv[13]):mode','command && argc>5?std::stoul(argv[5]):mode')
t=t.replace('  if(command && argc>14)first=std::stoul(argv[14]);\n','')
t=t.replace('outputs=0,raws=0,js=0;','outputs=0,raws=0,js=0,wides=0;')
t=t.replace('heldi=false;','heldi=false,heldo=false;')
t=t.replace('sa=0,ia=0,it=0;','sa=0,st=0,ia=0,it=0,ot=0;')
t=t.replace('source_count=0,identity_count=0;','source_count=0,identity_count=0,origin_count=0;')
t=t.replace('   d.parameter_valid=', '   d.origin_valid=(!stall || (n%17!=4 && n%17!=5));\n   d.parameter_valid=')
t=t.replace('d.source_address!=sa))return 15;','(d.source_address!=sa||d.source_tile!=st)))return 15;')
t=t.replace('   if(d.parameter_request_valid)for', '   if(heldo&&(!d.origin_request_valid||d.origin_tile!=ot))return 43;\n   if(d.origin_request_valid){const auto& f=fixtures.at(d.origin_tile);d.origin_data=(f.origin[0]&65535)|((f.origin[1]&65535)<<16);}\n   if(d.parameter_request_valid)for')
start=t.index('   if(d.source_request_valid) {');end=t.index('\n   d.eval();',start)
t=t[:start]+'''   if(d.source_request_valid)d.source_data=fixtures.at(d.source_tile).source.at(d.source_address);
   if(d.identity_request_valid)for(int l=0;l<8;++l)d.identity_data[l]=fixtures.at(d.identity_tile).id.at(d.identity_address*8+l);
'''+t[end:]
t=t.replace('helds=d.source_request_valid&&!d.source_valid;sa=d.source_address;',
'''helds=d.source_request_valid&&!d.source_valid;sa=d.source_address;st=d.source_tile;
   heldo=d.origin_request_valid&&!d.origin_valid;ot=d.origin_tile;''')
t=t.replace('   identity_count+=','   origin_count+=d.origin_request_valid&&d.origin_valid;\n   identity_count+=')
t=t.replace('jgold[index(tile,row,l)]','fixtures.at(tile).j.at(row*8+l)')
t=t.replace('raw[index(tile,row,l)]','fixtures.at(tile).raw.at(row*8+l)')
t=t.replace('gold[index(tile,row,l)]','fixtures.at(tile).gold.at(row*8+l)')
needle='   if(d.result_valid&&d.result_ready) {'
t=t.replace(needle,'''   if(d.wide_monitor_valid) {
    unsigned tile=first+wides/480,row=wides%480;
    if(d.result_tile!=tile||d.wide_monitor_address!=row)return 44;
    for(int l=0;l<16;++l)if(d.wide_monitor_data[l]!=fixtures.at(tile).wide.at(row*16+l))return 45;
    ++wides;
   }
'''+needle)
t=t.replace('js!=480ULL*count || d.retired_tiles!=count','js!=480ULL*count || wides!=480ULL*count || d.retired_tiles!=count')
t=t.replace('identity_count!=480ULL*count)return 22;','identity_count!=480ULL*count || origin_count!=count || source_count!=1536ULL*count)return 22;')
t=t.replace('d.source_load_stalls+1)return 24;', 'd.source_load_stalls+d.origin_stalls+1)return 24;')
t=t.replace('    SHOW(shared_wide_grants);', '    SHOW(origin_stalls);\n    std::cout<<",\\\"wide_outputs\\\":"<<wides*8;\n    SHOW(shared_wide_grants);')
(H/'tb.cpp').write_text(t)
print('Derived R8 capacity/manifest interface; datapath and RR unchanged.')
