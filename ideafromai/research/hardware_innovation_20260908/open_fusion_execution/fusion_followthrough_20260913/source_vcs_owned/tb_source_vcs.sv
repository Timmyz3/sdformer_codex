`timescale 1ns/1ps
// Direct SV adaptation of the existing C++ source TB. DUT is unchanged.
module tb_source_vcs;
    localparam integer MAX_TILES=1452;
    logic clk, rst_n, cfg_we, start;
    logic [8:0] cfg_addr;
    logic [127:0] cfg_data;
    logic [31:0] input_base, output_base;
    logic [15:0] input_stride;
    wire idle, done, rd_valid, wr_valid, wb_valid;
    logic rd_ready, rsp_valid, wr_ready;
    wire [31:0] rd_addr, wr_addr;
    logic [63:0] rsp_data;
    wire [63:0] wr_data;
    wire [6:0] wb_dst;
    wire [383:0] wb_data;
    temporal_source dut(.*);

    logic [127:0] program_words [0:511];
    logic [31:0] input_words [0:MAX_TILES*80-1];
    logic [15:0] gold_words [0:MAX_TILES*8-1];
    logic [7:0] memory_bytes [0:2879];
    longint signed reference_rf [0:95][0:7];
    longint signed expected_value [0:511][0:7];
    integer expected_dst [0:511];
    logic [15:0] reference_gate [0:7];
    integer expected_count;
    string program_path, input_path, gold_path, mode;
    integer program_length, tiles;
    longint unsigned cycles, reads, writes, rd_stall, wr_stall, wb_count;
    longint signed due;
    logic [63:0] response, held_data;
    logic [31:0] held_addr;
    bit held;

    function automatic void require(input bit condition, input string explanation);
        if (!condition) $fatal(1, "%s", explanation);
    endfunction

    task automatic tick_clock;
        #5 clk=1;
        #5 clk=0;
        #1;
    endtask

    function automatic longint signed exact48(input logic signed [95:0] value);
        if (value < -(96'sd1 <<< 47) || value >= (96'sd1 <<< 47))
            $fatal(1, "reference signed48 overflow");
        return longint'(value);
    endfunction

    function automatic longint signed norm24(input longint signed value, input integer shift);
        longint signed divisor, quo, remn;
        require(shift >= 0 && shift < 47, "reference normalization shift");
        divisor=64'sd1 <<< shift;
        quo=value/divisor;
        remn=value%divisor;
        if (remn<0) begin quo=quo-1; remn=remn+divisor; end
        if (shift!=0 && ((remn>divisor/2) || ((remn==divisor/2) && quo[0]))) quo=quo+1;
        if (quo < -8388608) return -8388608;
        if (quo > 8388607) return 8388607;
        return quo;
    endfunction

    task automatic reference_tile(input integer tile);
        logic [127:0] ins;
        integer kind, dst, src_a, src_b, sh_a, sh_b, rne, tick, constant_code;
        bit neg_a, neg_b, direction_negative;
        longint signed threshold, a, b, value;
        logic signed [95:0] wide;
        for (integer r=0;r<96;r=r+1)
            for (integer lane=0;lane<8;lane=lane+1) reference_rf[r][lane]=0;
        for (integer lane=0;lane<8;lane=lane+1) reference_gate[lane]=0;
        expected_count=0;
        for (integer pc=0;pc<program_length;pc=pc+1) begin
            ins=program_words[pc];
            kind=int'(ins[2:0]); dst=int'(ins[9:3]);
            src_a=int'(ins[16:10]); src_b=int'(ins[23:17]);
            sh_a=int'(ins[29:24]); sh_b=int'(ins[35:30]);
            neg_a=ins[36]; neg_b=ins[37]; rne=int'(ins[43:38]); tick=int'(ins[47:44]);
            threshold=longint'($signed(ins[95:48])); direction_negative=ins[96]; constant_code=int'(ins[98:97]);
            if (kind!=0 && kind!=5) begin
                for (integer lane=0;lane<8;lane=lane+1) begin
                    if (kind==1) value=longint'($signed(input_words[tile*80+tick*8+lane]));
                    else begin
                        wide=96'(reference_rf[src_a][lane]); wide=wide <<< sh_a;
                        if (neg_a) wide=-wide;
                        a=exact48(wide);
                        if (kind==2) begin
                            wide=96'(reference_rf[src_b][lane]); wide=wide <<< sh_b;
                            if (neg_b) wide=-wide;
                            b=exact48(wide);
                            wide=96'(a); wide=wide+96'(b); value=exact48(wide);
                        end else if (kind==3) value=norm24(a,rne);
                        else if (kind==4) begin
                            if (constant_code!=0) value=(constant_code==2)?64'sd1:64'sd0;
                            else if (direction_negative) value=(a<=threshold)?64'sd1:64'sd0;
                            else value=(a>=threshold)?64'sd1:64'sd0;
                        end else $fatal(1,"unknown instruction kind=%0d",kind);
                    end
                    reference_rf[dst][lane]=value;
                    expected_value[expected_count][lane]=value;
                    if (kind==4) reference_gate[lane][tick]=value[0];
                end
                expected_dst[expected_count]=dst;
                expected_count=expected_count+1;
            end
        end
        for (integer lane=0;lane<8;lane=lane+1)
            require(reference_gate[lane]===gold_words[tile*8+lane],
                    $sformatf("interpreter/captured gate tile=%0d lane=%0d",tile,lane));
    endtask

    initial begin : run
        integer wi, stores;
        longint unsigned begin_cycle;
        longint signed actual_lane;
        require($value$plusargs("PROGRAM=%s",program_path), "PROGRAM argument");
        require($value$plusargs("INPUT=%s",input_path), "INPUT argument");
        require($value$plusargs("GOLD=%s",gold_path), "GOLD argument");
        require($value$plusargs("MODE=%s",mode), "MODE argument");
        require($value$plusargs("PROGRAM_LENGTH=%d",program_length), "PROGRAM_LENGTH argument");
        require($value$plusargs("TILES=%d",tiles), "TILES argument");
        require(program_length>0 && program_length<=512, "program length");
        require(tiles>0 && tiles<=MAX_TILES, "tile count");
        require(mode=="ready" || mode=="stress", "fixed mode");
        $readmemh(program_path,program_words,0,program_length-1);
        $readmemh(input_path,input_words,0,tiles*80-1);
        $readmemh(gold_path,gold_words,0,tiles*8-1);
        clk=0; rst_n=0; cfg_we=0; cfg_addr=0; cfg_data=0; start=0;
        rd_ready=0; rsp_valid=0; rsp_data=0; wr_ready=0;
        input_base=0; output_base=4096; input_stride=288;
        tick_clock(); tick_clock(); rst_n=1; tick_clock();
        for (integer pc=0;pc<program_length;pc=pc+1) begin
            cfg_we=1; cfg_addr=9'(pc); cfg_data=program_words[pc]; tick_clock();
        end
        cfg_we=0;
        cycles=0; reads=0; writes=0; rd_stall=0; wr_stall=0; wb_count=0;
        due=-1; response=0; held=0; held_data=0; held_addr=0;
        for (integer tile=0;tile<tiles;tile=tile+1) begin
            reference_tile(tile);
            for (integer j=0;j<2880;j=j+1) memory_bytes[j]=0;
            for (integer t=0;t<10;t=t+1)
                for (integer lane=0;lane<8;lane=lane+1)
                    for (integer j=0;j<3;j=j+1)
                        memory_bytes[t*288+lane*3+j]=8'(input_words[tile*80+t*8+lane] >> (8*j));
            wi=0; stores=0; begin_cycle=cycles;
            require(idle===1'b1, "start when busy");
            start=1; tick_clock(); cycles=cycles+1; start=0;
            while (1) begin
                require(cycles-begin_cycle<20000, "tile timeout");
                rsp_valid=(longint'(cycles)==due); rsp_data=response;
                rd_ready=(mode!="stress" || cycles%32<24);
                wr_ready=(mode!="stress" || cycles%32<28);
                #1; // Settle DUT combinational outputs before sampling, as d.eval().
                if (wb_valid===1'b1) begin
                    require(wi<expected_count, "extra RF writeback");
                    require(wb_dst===7'(expected_dst[wi]), "writeback destination");
                    for (integer lane=0;lane<8;lane=lane+1) begin
                        actual_lane=longint'($signed(wb_data[lane*48+:48]));
                        require(actual_lane===expected_value[wi][lane],
                            $sformatf("WB tile=%0d step=%0d lane=%0d actual=%0d expected=%0d",
                                      tile,wi,lane,actual_lane,expected_value[wi][lane]));
                    end
                    wi=wi+1; wb_count=wb_count+1;
                end
                if (rd_valid===1'b1) begin
                    if (!rd_ready) rd_stall=rd_stall+1;
                    else begin
                        require(rd_addr+8<=2880, "read address");
                        response=0;
                        for (integer j=0;j<8;j=j+1) response[j*8+:8]=memory_bytes[rd_addr+j];
                        due=longint'(cycles)+1+((mode=="stress")?longint'(cycles%3):0);
                        reads=reads+1;
                    end
                end
                if (held) require(wr_valid===1'b1 && wr_addr===held_addr && wr_data===held_data,
                                  "blocked write changed");
                held=(wr_valid===1'b1 && !wr_ready);
                if (held) begin
                    held_addr=wr_addr; held_data=wr_data; wr_stall=wr_stall+1;
                end
                if (wr_valid===1'b1 && wr_ready) begin
                    require(stores<2 && wr_addr==4096+stores*8, "gate write address/order");
                    for (integer lane=0;lane<4;lane=lane+1)
                        require(wr_data[lane*16+:16]===gold_words[tile*8+stores*4+lane],
                                $sformatf("RTL/capture gate tile=%0d store=%0d lane=%0d",tile,stores,lane));
                    stores=stores+1; writes=writes+1;
                end
                tick_clock(); cycles=cycles+1;
                if (done===1'b1) begin
                    require(stores==2 && wi==expected_count, "incomplete tile");
                    break;
                end
            end
        end
        $display("SOURCE_VCS_RESULT {\"mode\":\"%s\",\"tiles\":%0d,\"cycles\":%0d,\"SR64_reads\":%0d,\"SW64_writes\":%0d,\"read_stall_cycles\":%0d,\"write_stall_cycles\":%0d,\"RF_vector_writebacks_checked\":%0d,\"input_values\":%0d,\"gate_bits_checked\":%0d,\"differences\":0}",
            mode,tiles,cycles,reads,writes,rd_stall,wr_stall,wb_count,tiles*80,tiles*80);
        $finish;
    end
endmodule
