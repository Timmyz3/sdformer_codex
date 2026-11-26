// Native binary AAC and G4 subset-table execution. Common 8 x Acc24 backend.
// Scope: 80 rows = T10 x P8, C<=384; complete linear Y, no BN/PSN.
module native_fc (
    input logic clk,rst_n,
    input logic cfg_valid,cfg_kind,
    input logic [15:0] cfg_addr,
    input logic [127:0] cfg_data,
    output logic cfg_ready,
    input logic start,
    input logic [8:0] c_count,
    input logic [1:0] mode,
    input logic src_enable,coef_enable,
    output logic busy,done,
    output logic out_valid,
    input logic out_ready,
    output logic [6:0] out_row,
    output logic [191:0] out_data,
    output logic [31:0] dbg_source_reads,dbg_coef_reads,dbg_add_cycles,dbg_build_cycles
);
    logic [127:0] source_mem [0:239];
    logic [63:0] weight_mem [0:383];
    logic [79:0] table_mem [0:1535];
    logic [191:0] y_mem [0:79];
    logic [127:0] mask;
    logic [63:0] read_weight;
    logic [79:0] read_table;
    logic signed [23:0] acc [0:7];
    logic read_pending;
    logic [8:0] columns,table_columns;
    logic [1:0] mode_q,word_id;
    logic [6:0] row_id;
    logic table_valid;
    logic [6:0] build_group;
    logic [2:0] build_weight;
    logic [4:0] build_step;
    logic [63:0] four_weights [0:3];
    logic signed [9:0] build_acc [0:7];
    logic [3:0] gray;
    logic [1:0] flip;
    logic [79:0] build_value;
    integer bit_pick,nibble_pick;
    logic found;
    logic [3:0] code;
    logic [191:0] acc_word;
    logic [23:0] alu_a [0:7],alu_b [0:7],alu_result [0:7];
    logic alu_sub;
    typedef enum logic [3:0] {IDLE,BW,BUILD,SRC,RUN,STORE,OUT_FETCH,OUT_HOLD} phase_t;
    phase_t phase;

    assign cfg_ready=(phase==IDLE && !start);
    assign out_valid=(phase==OUT_HOLD);
    assign out_row=row_id;
    always_comb begin
        bit_pick=0;nibble_pick=0;found=0;code=0;
        if(mode_q==0)begin
            for(int b=0;b<128;b++)if(!found && mask[b])begin bit_pick=b;found=1;end
        end else begin
            for(int g=0;g<32;g++)if(!found && |mask[g*4+:4])begin
                nibble_pick=g;code=mask[g*4+:4];found=1;
            end
        end
        gray=4'(build_step ^ (build_step>>1));
        flip=0;
        if(build_step[0])flip=0;
        else if(build_step[1])flip=1;
        else if(build_step[2])flip=2;
        else flip=3;
        build_value=0;acc_word=0;
        alu_sub=(phase==BUILD && !gray[flip]);
        for(int h=0;h<8;h++)begin
            acc_word[h*24+:24]=acc[h];
            if(phase==BUILD)begin
                alu_a[h]=24'($signed(build_acc[h]));
                alu_b[h]=24'($signed(four_weights[flip][h*8+:8]));
            end else begin
                alu_a[h]=acc[h];
                if(mode_q==0)alu_b[h]=24'($signed(read_weight[h*8+:8]));
                else alu_b[h]=24'($signed(read_table[h*10+:10]));
            end
            // One selected add/sub datapath per output lane, reused during table build.
            alu_result[h]=alu_a[h]+(alu_b[h]^{24{alu_sub}})+24'(alu_sub);
            if(build_step==0)build_value[h*10+:10]=0;
            else build_value[h*10+:10]=alu_result[h][9:0];
        end
    end

    always_ff @(posedge clk)begin
        if(!rst_n)begin
            phase<=IDLE;busy<=0;done<=0;table_valid<=0;table_columns<=0;
            row_id<=0;word_id<=0;columns<=0;mode_q<=0;read_pending<=0;
            mask<=0;read_weight<=0;read_table<=0;out_data<=0;
            build_group<=0;build_weight<=0;build_step<=0;
            dbg_source_reads<=0;dbg_coef_reads<=0;dbg_add_cycles<=0;dbg_build_cycles<=0;
            for(int h=0;h<8;h++)begin acc[h]<=0;build_acc[h]<=0;end
        end else begin
            done<=0;
            if(cfg_valid && cfg_ready)begin
                if(cfg_kind)begin weight_mem[cfg_addr]<=cfg_data[63:0];table_valid<=0;end
                else source_mem[cfg_addr]<=cfg_data;
            end
            case(phase)
            IDLE: if(start)begin
                busy<=1;columns<=c_count;mode_q<=mode;row_id<=0;word_id<=0;read_pending<=0;
                dbg_source_reads<=0;dbg_coef_reads<=0;dbg_add_cycles<=0;dbg_build_cycles<=0;
                for(int h=0;h<8;h++)acc[h]<=0;
                if(mode==1 && (!table_valid || table_columns!=c_count))begin
                    build_group<=0;build_weight<=0;phase<=BW;
                end else phase<=SRC;
            end
            BW:begin
                dbg_build_cycles<=dbg_build_cycles+1;
                if(coef_enable)begin
                    four_weights[build_weight]<=weight_mem[build_group*4+build_weight];
                    dbg_coef_reads<=dbg_coef_reads+1;
                    if(build_weight==3)begin
                        build_step<=0;phase<=BUILD;
                        for(int h=0;h<8;h++)build_acc[h]<=0;
                    end else build_weight<=build_weight+1;
                end
            end
            BUILD:begin
                dbg_build_cycles<=dbg_build_cycles+1;
                // A write uses the same finite coefficient service calendar as reads.
                if(coef_enable)begin
                    table_mem[build_group*16+gray]<=build_value;
                    for(int h=0;h<8;h++)build_acc[h]<=$signed(build_value[h*10+:10]);
                    if(build_step!=0)dbg_add_cycles<=dbg_add_cycles+1;
                    if(build_step==15)begin
                        if((build_group+1)*4==columns)begin
                            table_valid<=1;table_columns<=columns;phase<=SRC;
                        end else begin build_group<=build_group+1;build_weight<=0;phase<=BW;end
                    end else build_step<=build_step+1;
                end
            end
            SRC: if(src_enable)begin
                mask<=source_mem[row_id*3+word_id];
                dbg_source_reads<=dbg_source_reads+1;
                read_pending<=0;phase<=RUN;
            end
            RUN:begin
                if(read_pending)begin
                    for(int h=0;h<8;h++)begin
                        acc[h]<=alu_result[h];
                    end
                    dbg_add_cycles<=dbg_add_cycles+1;
                end
                read_pending<=0;
                if(found && coef_enable)begin
                    if(mode_q==0)begin
                        read_weight<=weight_mem[word_id*128+bit_pick];
                        mask[bit_pick]<=0;
                    end else begin
                        read_table<=table_mem[(word_id*32+nibble_pick)*16+code];
                        mask[nibble_pick*4+:4]<=0;
                    end
                    dbg_coef_reads<=dbg_coef_reads+1;read_pending<=1;
                end
                if(!found && !read_pending)begin
                    if((word_id+1)*128>=columns)phase<=STORE;
                    else begin word_id<=word_id+1;phase<=SRC;end
                end
            end
            STORE:begin
                y_mem[row_id]<=acc_word;
                if(row_id==79)begin row_id<=0;phase<=OUT_FETCH;end
                else begin
                    row_id<=row_id+1;word_id<=0;phase<=SRC;
                    for(int h=0;h<8;h++)acc[h]<=0;
                end
            end
            OUT_FETCH:begin out_data<=y_mem[row_id];phase<=OUT_HOLD;end
            OUT_HOLD:if(out_ready)begin
                if(row_id==79)begin busy<=0;done<=1;phase<=IDLE;end
                else begin row_id<=row_id+1;out_data<=y_mem[row_id+1];end
            end
            default:phase<=IDLE;
            endcase
        end
    end
endmodule
