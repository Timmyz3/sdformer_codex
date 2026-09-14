from pathlib import Path
H=Path(__file__).resolve().parent;A=H.parent/'spatial_r16_rtl'
text=(A/'spatial_stream.sv').read_text().replace('module spatial_stream(', 'module os_stream(').replace('input logic [10:0] cfg_addr','input logic [13:0] cfg_addr')
text=text.replace(' output logic z_monitor_valid,z_monitor_stripe,output logic [5:0] z_monitor_addr,output logic [255:0] z_monitor_data,\n','')
old='cycles, source_words, q1_words, q2_words, local_gathers, q1_issues, q2_issues, z_vector_reads, z_scalar_reads, z_writes, psum_reads, psum_writes, cache_writes'
new='cycles, source_words, weight_words, local_gathers, add_issues, bitmap_reads, bitmap_writes, psum_reads, psum_writes, psum_clears'
text=text.replace(old,new).replace('spatial_core producer','os_core producer')
text=text.replace('.debug_state(debug_state),.z_monitor_valid(z_monitor_valid),.z_monitor_stripe(z_monitor_stripe),.z_monitor_addr(z_monitor_addr),.z_monitor_data(z_monitor_data),','.debug_state(debug_state),')
for name in ['q1_words','q2_words','q1_issues','q2_issues','z_vector_reads','z_scalar_reads','z_writes','cache_writes']:
 text=text.replace(f' .{name}({name}),\n','')
text=text.replace(' .source_words(source_words),',' .source_words(source_words),\n .weight_words(weight_words),.add_issues(add_issues),.bitmap_reads(bitmap_reads),.bitmap_writes(bitmap_writes),.psum_clears(psum_clears),')
(H/'os_stream.sv').write_text(text)
tb=(A/'stream_tb.cpp').read_text().replace('Vspatial_stream','Vos_stream')
tb=tb.replace('auto q1=readhex(root+"/parameters/q1.hex"),q2=readhex(root+"/parameters/q2.hex"),ab=', 'auto w=readhex(root+"/parameters/weight.hex"),ab=')
tb=tb.replace('q1.size()!=4608||q2.size()!=4608','w.size()!=82944')
tb=tb.replace('for(unsigned a=0;a<576;a++){for(int i=0;i<8;i++)data[i]=q1[a*8+i];cfg(4,a);}', 'for(unsigned a=0;a<10368;a++){for(int i=0;i<8;i++)data[i]=w[a*8+i];cfg(4,a);}')
tb=tb.replace(' for(unsigned a=0;a<576;a++){for(int i=0;i<8;i++)data[i]=q2[a*8+i];cfg(5,a);}\n','')
tb=tb.replace(',zgold=readhex(dir+"/z.hex")','').replace('||zgold.size()!=640','').replace(',zoutputs=0','').replace('||zoutputs!=80','').replace('\\"z_values\\":1280,','')
tb='\n'.join(line for line in tb.splitlines() if 'if(d.z_monitor_valid)' not in line)+'\n'
for name in ['q1_words','q2_words','q1_issues','q2_issues','z_vector_reads','z_scalar_reads','z_writes','cache_writes']:
 tb=tb.replace(f'F({name});','')
tb=tb.replace('F(source_words);','F(source_words);F(weight_words);F(add_issues);F(bitmap_reads);F(bitmap_writes);F(psum_clears);')
(H/'stream_tb.cpp').write_text(tb)
# Q13 closure uses exactly the same executable and existing real fixtures.
(H/'parameters/consumer.hex').write_text((A/'parameters/consumer.hex').read_text())
print('OS full-consumer wrapper and harness generated; original consumer/ALU compiled read-only')
