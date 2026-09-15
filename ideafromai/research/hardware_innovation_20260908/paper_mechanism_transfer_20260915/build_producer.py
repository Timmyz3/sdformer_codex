from pathlib import Path

H = Path(__file__).resolve().parent
SRC = H.parent / 'consumer_transfer_20260914/gustav_intersection'
D = H / 'producer_rtl'
D.mkdir(exist_ok=True)

s = (SRC / 'gp_slice_lazy.sv').read_text()
s = s.replace('// External interfaces are four 64-bit source banks and two W8 ports/tile.',
              '// Source storage is four 1RW behavioral banks; two external W8 ports/tile.')
for decl in ['logic [3:0] src_req_ready', 'logic [3:0] src_rsp_valid', 'logic [255:0] src_rsp_data']:
    s = s.replace('input ' + decl, 'output ' + decl)
s = s.replace('    input logic clk,rst_n,', '''    input logic clk,rst_n,
    input logic [3:0] prod_valid,
    output logic [3:0] prod_ready,
    input logic [255:0] prod_data,
    output logic [27:0] dbg_produced_words,
    output logic [31:0] dbg_seen_codes,
    output logic [3:0] dbg_producer_live,
    output logic [3:0] dbg_source_write,''')
s = s.replace('frontend_q!=4', 'frontend_q<4').replace('frontend_q==4', 'frontend_q>=4')
s = s.replace('        meta_enabled=(frontend_q<4);dbg_rank_catchup=0;', '''        meta_enabled=(frontend_q<4);dbg_rank_catchup=0;
        for(int k=0;k<4;k++)begin
            if(frontend_q==5 && produced_words[k]==72 && |(seen_codes[k]&live_decode))meta_enabled=1;
            if(frontend_q==6 && produced_words[k]==72 && producer_live[k])meta_enabled=1;
        end''')
bank = '''
    // Each bank owns one read/write port. An accepted write has priority;
    // read data enters one response-holding register on the next edge.
    // start_fire begins a new exclusive generation. Contents are not cleared;
    // committed-length checks prevent reads of the previous generation.
    logic [63:0] source_ram [0:3][0:71];
    logic [6:0] produced_words [0:3];
    logic [7:0] seen_codes [0:3],seen_next [0:3],live_decode;
    logic [3:0] producer_live,source_response,word_has_live;
    logic [63:0] source_return [0:3];
    logic [1:0] carry_bits [0:3],carry_value [0:3];
    logic [65:0] joined [0:3];
    integer code_count [0:3];
    always_comb begin
        live_decode=0;
        for(int c=0;c<8;c++)live_decode[c]=|decode_q[c*7+:7];
        prod_ready=0;src_req_ready=0;src_rsp_valid=source_response;src_rsp_data=0;word_has_live=0;
        dbg_produced_words=0;dbg_seen_codes=0;dbg_producer_live=producer_live;
        for(int k=0;k<4;k++)begin
            prod_ready[k]=busy && produced_words[k]<72;
            src_req_ready[k]=busy && !source_response[k]
                && src_req_address[k*7+:7]<produced_words[k] && !prod_valid[k];
            src_rsp_data[k*64+:64]=source_return[k];
            dbg_produced_words[k*7+:7]=produced_words[k];
            dbg_seen_codes[k*8+:8]=seen_codes[k];
            joined[k]=(66'(prod_data[k*64+:64])<<carry_bits[k])|66'(carry_value[k]);
            code_count[k]=(64+int'(carry_bits[k]))/3;
            seen_next[k]=seen_codes[k];
            for(int j=0;j<22;j++)if(j<code_count[k])begin
                seen_next[k][joined[k][j*3+:3]]=1;
                word_has_live[k]|=live_decode[joined[k][j*3+:3]];
            end
        end
        dbg_source_write=prod_valid&prod_ready;
    end
    always_ff @(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            source_response<=0;producer_live<=0;
            for(int k=0;k<4;k++)begin
                produced_words[k]<=0;seen_codes[k]<=0;carry_bits[k]<=0;carry_value[k]<=0;source_return[k]<=0;
            end
        end else if(start_fire)begin
            source_response<=0;producer_live<=0;
            for(int k=0;k<4;k++)begin
                produced_words[k]<=0;seen_codes[k]<=0;carry_bits[k]<=0;carry_value[k]<=0;
            end
        end else begin
            for(int k=0;k<4;k++)begin
                if(source_response[k] && src_rsp_ready[k])source_response[k]<=0;
                if(prod_valid[k] && prod_ready[k])begin
                    source_ram[k][produced_words[k]]<=prod_data[k*64+:64];
                    produced_words[k]<=produced_words[k]+1'b1;
                    seen_codes[k]<=seen_next[k];
                    producer_live[k]<=producer_live[k]|word_has_live[k];
                    carry_bits[k]<=2'(64+int'(carry_bits[k])-3*code_count[k]);
                    carry_value[k]<=2'(joined[k]>>(3*code_count[k]));
                end else if(src_req_valid[k] && src_req_ready[k])begin
                    source_return[k]<=source_ram[k][src_req_address[k*7+:7]];
                    source_response[k]<=1;
                end
            end
        end
    end
'''
s = s.replace('    // Four program replicas,', bank + '\n    // Four program replicas,')
(D / 'gp_slice.sv').write_text(s)

s = (SRC / 'lazy_scoreboard.cpp').read_text()
s = s.replace('static uint32_t rng=0x19283047;', 'static uint32_t rng=0x19283047;\nstatic int producer_calendar=0;')
s = s.replace('d.src_req_ready=0;d.src_rsp_valid=0;', 'd.prod_valid=0;')
s = s.replace('for(int w=0;w<8;w++)d.src_rsp_data[w]=0;', 'for(int w=0;w<8;w++)d.prod_data[w]=0;')
s = s.replace('    U cycles=0,last_gate=0,', '    U source_writes=0,first_write=0,last_write=0,write_wait=0;\n    U cycles=0,last_gate=0,')
s = s.replace('    unsigned src_word[4]{};', '    unsigned produced[4]{},seen_written[4]{},src_word[4]{};')
s = s.replace('intersection==4&&!any_source', 'intersection>=4&&!any_source')
s = s.replace('intersection!=4 || d.dbg_meta_trigger', 'intersection<4 || d.dbg_meta_trigger')
start = s.index('            if(!src[k].pending && (!stress')
end = s.index('            if(!weight[k].pending', start)
s = s[:start]+'''            bool offer=producer_calendar==0 || (step%8==unsigned(k));
            if(produced[k]<72 && offer) {
                d.prod_valid|=1u<<k;
                U word=e.source[k][produced[k]];
                d.prod_data[k*2]=uint32_t(word);d.prod_data[k*2+1]=uint32_t(word>>32);
            }
'''+s[end:]
s = s.replace('        unsigned w_accept=0,src_accept=0;', '''        unsigned w_accept=0,src_accept=0;
        for(int k=0;k<4;k++)if((d.prod_valid&d.prod_ready)&(1u<<k)) {
            if(s.source_writes==0)s.first_write=cycle-begin+1;
            s.last_write=cycle-begin+1;s.source_writes++;produced[k]++;
            // The reference consumes the actual committed bitstream, including
            // code3 values crossing a 64-bit bank word.
            unsigned complete_codes=produced[k]*64/3;
            for(unsigned j=0;j<complete_codes;j++) {
                unsigned bit=j*3,w=bit/64,offset=bit%64;
                U v=e.source[k][w]>>offset;
                if(offset>61)v|=e.source[k][w+1]<<(64-offset);
                seen_written[k]|=1u<<(v&7);
            }
        }
        s.write_wait+=pc(d.prod_valid&~d.prod_ready);''')
s = s.replace('src[q]={true,cycle+1+unsigned(stress?((rand>>(q+1))&3):0),e.source[q][addr],addr};',
'''need(addr<produced[q],"read before actual producer commit");
                src[q]={true,cycle+1,e.source[q][addr],addr};''')
s = s.replace('need(src[q].pending,"unowned source response");src[q].pending=false;',
'''need(src[q].pending,"unowned source response");
                need((U(d.src_rsp_data[q*2])|(U(d.src_rsp_data[q*2+1])<<32))==src[q].value,
                     "RTL bank read differs from committed producer data");src[q].pending=false;''')
s = s.replace('need(src_word[k]==72 && !src[k].pending', '''need(produced[k]==72 && ((d.dbg_produced_words>>(k*7))&127)==72,
                     "producer bank incomplete at retirement");
                need(((d.dbg_seen_codes>>(k*8))&255)==seen_written[k],"code3 cross-word summary");
                bool expected_live=false;for(int z=0;z<8;z++)if(seen_written[k]&(1u<<z))expected_live|=c.decode[z]!=0;
                need(bool((d.dbg_producer_live>>k)&1)==expected_live,"producer decode support");
                need(src_word[k]==72 && !src[k].pending''')
s = s.replace('for(int intersection=3;intersection<5;intersection++)',
              'for(producer_calendar=0;producer_calendar<2;producer_calendar++)for(int intersection=3;intersection<7;intersection++)')
s = s.replace('rank_catchup_pe_beats\\n";', 'rank_catchup_pe_beats\\tproducer_calendar\\tsource_writes\\tfirst_write\\tlast_write\\tproducer_write_wait\\tservice_from_first_write\\n";')
s = s.replace("<<s.rank_catchup_pe_beats<<'\\n';", "<<s.rank_catchup_pe_beats<<'\\t'<<producer_calendar<<'\\t'<<s.source_writes<<'\\t'<<s.first_write<<'\\t'<<s.last_write<<'\\t'<<s.write_wait<<'\\t'<<(s.last_gate-s.first_write+1)<<'\\n';")
s = s.replace('eight lazy comparison configurations', '32 producer/source/consumer configurations')
(D / 'scoreboard.cpp').write_text(s)

print('wrote', D)
