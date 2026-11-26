// G4 subset accumulation fused with Gustav-inspired P8 record reuse.
// Exact integer slice only; neither full GustavSNN nor a novelty claim.
// One synchronous source-word read and one synchronous coefficient-word read
// per cycle. Two NR4 packets overlap coefficient fetch with serial position
// merge execution. Eight 24-bit ALUs update one P8 position per cycle.
module native_gp4 (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         cfg_valid,
    input  logic         cfg_kind,
    input  logic [15:0]  cfg_addr,
    input  logic [127:0] cfg_data,
    output logic         cfg_ready,
    input  logic         start,
    input  logic [8:0]   c_count,
    input  logic [1:0]   mode,
    input  logic         src_enable,
    input  logic         coef_enable,
    output logic         busy,
    output logic         done,
    output logic         out_valid,
    input  logic         out_ready,
    output logic [6:0]   out_row,
    output logic [191:0] out_data,
    output logic [31:0]  dbg_source_reads,
    output logic [31:0]  dbg_coef_reads,
    output logic [31:0]  dbg_add_cycles,
    output logic [31:0]  dbg_build_cycles
);
    localparam [3:0] IDLE=0, CLEAR_ACC=1, LOAD_SRC=2, MAKE_BITS=3,
                     RUN=4, COMMIT=5, DRAIN=6, TABLE_W=7, TABLE_BUILD=8;
    logic [3:0] state;
    logic [127:0] source_mem [0:239];
    logic [63:0] coefficient_mem [0:383];
    logic [79:0] table_mem [0:1535];
    logic table_valid;
    logic [8:0] table_columns;
    logic [6:0] table_group;
    logic [2:0] table_weight;
    logic [4:0] table_step;
    logic [63:0] four_weights [0:3];
    logic signed [9:0] table_acc [0:7];
    logic [3:0] gray;
    logic [1:0] flip;
    logic [79:0] table_value;
    logic [23:0] alu_a [0:7], alu_b [0:7], alu_result [0:7];
    logic alu_sub;
    logic [191:0] y_mem [0:79];
    logic [127:0] source_block [0:7];
    logic [191:0] acc [0:7];
    logic [8:0] channels;
    logic [3:0] tick_group;
    logic [1:0] word_index;
    logic [3:0] source_read_index;
    logic [2:0] local_row;
    logic [6:0] drain_next;

    // Each record is (G4 group, nonzero code, matching P8 positions).
    // One table vector is fetched once, then scattered to every matching position.
    logic [1:0] packet_valid;
    logic [2:0] packet_len [0:1];
    logic [2:0] fetch_count [0:1];
    logic [3:0] weight_ready [0:1];
    logic [10:0] record_address [0:1][0:3];
    logic [7:0] record_mask [0:1][0:3];
    logic [79:0] record_value [0:1][0:3];
    logic fill_packet, fetch_packet, execute_packet;

    logic [127:0] source_union;
    logic [127:0] after_build [0:7];
    logic [2:0] build_len;
    logic [10:0] build_address [0:3];
    logic [7:0] build_mask [0:3];
    logic [3:0] merge_position;
    logic [1:0] merge_record;
    logic execute_has_work;
    logic work_after_issue;
    integer i, j, k;
    integer pick;
    logic [3:0] pick_code;
    integer pos;

    assign cfg_ready = (state == IDLE) && !start;

    // Transposition is real: source_block only becomes valid after eight
    // synchronous source reads. The NRV priority logic consumes this buffer.
    always_comb begin
        source_union = 128'd0;
        for (integer p=0; p<8; p=p+1)
            source_union = source_union | source_block[p];
        for (integer c=0; c<128; c=c+1)
            if ((integer'(word_index)*128+c) >= integer'(channels))
                source_union[c] = 1'b0;
    end

    // Up to four (group,code) records are generated in one charged build
    // cycle. Select one nonempty group, select its first nonzero P8 code,
    // gather all equal codes, and remove those positions from the cached block.
    // This is combinational priority/equality logic, not a free pre-encoded input.
    always_comb begin
        for (integer p=0; p<8; p=p+1) after_build[p] = source_block[p];
        build_len = 0;
        pick = -1;
        pick_code = 0;
        for (integer r=0; r<4; r=r+1) begin
            build_address[r] = 0;
            build_mask[r] = 0;
            pick = -1;
            pick_code = 0;
            for (integer g=0; g<32; g=g+1)
                for (integer p=0; p<8; p=p+1)
                    if (pick < 0 && after_build[p][g*4 +: 4] != 0 &&
                        (integer'(word_index)*128+g*4) < integer'(channels)) begin
                        pick = g;
                        pick_code = after_build[p][g*4 +: 4];
                    end
            if (pick >= 0) begin
                build_address[r] = 11'((integer'(word_index)*32+pick)*16) + 11'(pick_code);
                for (integer p=0; p<8; p=p+1)
                    if (after_build[p][pick*4 +: 4] == pick_code) begin
                        build_mask[r][p] = 1'b1;
                        after_build[p][pick*4 +: 4] = 0;
                    end
                build_len = build_len + 1'b1;
            end
        end
    end

    // Minimum position merger. Ties issue one record at a time: combining
    // multiple W vectors in one cycle would exceed the common eight-ALU budget.
    always_comb begin
        merge_position = 4'd8;
        merge_record = 2'd0;
        execute_has_work = 1'b0;
        pos = 8;
        for (integer r=0; r<4; r=r+1) begin
            pos = 8;
            for (integer p=0; p<8; p=p+1)
                if (record_mask[execute_packet][r][p] && pos == 8) pos = p;
            if (r < integer'(packet_len[execute_packet]) && pos < integer'(merge_position)) begin
                merge_position = 4'(pos);
                merge_record = 2'(r);
                execute_has_work = 1'b1;
            end
        end
        work_after_issue = 1'b0;
        for (integer r=0; r<4; r=r+1)
            if (r < integer'(packet_len[execute_packet])) begin
                if (r == integer'(merge_record))
                    work_after_issue = work_after_issue |
                        (|(record_mask[execute_packet][r] & ~(8'd1 << merge_position)));
                else work_after_issue = work_after_issue | (|record_mask[execute_packet][r]);
            end
    end

    // The same eight 24-bit add/sub paths serve table construction and
    // spatial accumulation; the phases are mutually exclusive.
    always_comb begin
        gray = 4'(table_step ^ (table_step >> 1));
        if (table_step[0]) flip=0;
        else if (table_step[1]) flip=1;
        else if (table_step[2]) flip=2;
        else flip=3;
        alu_sub = (state == TABLE_BUILD) && !gray[flip];
        table_value = 0;
        for (integer h=0; h<8; h=h+1) begin
            if (state == TABLE_BUILD) begin
                alu_a[h] = 24'($signed(table_acc[h]));
                alu_b[h] = 24'($signed(four_weights[flip][h*8 +: 8]));
            end else begin
                alu_a[h] = acc[merge_position[2:0]][h*24 +: 24];
                alu_b[h] = 24'($signed(record_value[execute_packet][merge_record][h*10 +: 10]));
            end
            alu_result[h] = alu_a[h] + (alu_b[h] ^ {24{alu_sub}}) + 24'(alu_sub);
            if (table_step != 0) table_value[h*10 +: 10] = alu_result[h][9:0];
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            out_valid <= 1'b0;
            out_row <= 7'd0;
            out_data <= 192'd0;
            channels <= 9'd0;
            tick_group <= 4'd0;
            word_index <= 2'd0;
            source_read_index <= 4'd0;
            local_row <= 3'd0;
            drain_next <= 7'd0;
            table_valid <= 0;
            table_columns <= 0;
            table_group <= 0;
            table_weight <= 0;
            table_step <= 0;
            for (i=0; i<8; i=i+1) table_acc[i] <= 0;
            packet_valid <= 2'd0;
            fill_packet <= 1'b0;
            fetch_packet <= 1'b0;
            execute_packet <= 1'b0;
            dbg_source_reads <= 0;
            dbg_coef_reads <= 0;
            dbg_add_cycles <= 0;
            dbg_build_cycles <= 0;
            for (i=0; i<2; i=i+1) begin
                packet_len[i] <= 0;
                fetch_count[i] <= 0;
                weight_ready[i] <= 0;
                for (j=0; j<4; j=j+1) begin
                    record_address[i][j] <= 0;
                    record_mask[i][j] <= 0;
                    record_value[i][j] <= 0;
                end
            end
        end else begin
            done <= 1'b0;
            if (cfg_valid && cfg_ready) begin
                if (cfg_kind) begin
                    coefficient_mem[cfg_addr[8:0]] <= cfg_data[63:0];
                    table_valid <= 0;
                end
                else source_mem[cfg_addr[7:0]] <= cfg_data;
            end
            case (state)
                IDLE: if (start) begin
                    channels <= c_count;
                    tick_group <= 0;
                    local_row <= 0;
                    word_index <= 0;
                    if (mode == 0 || !table_valid || table_columns != c_count) begin
                        table_group <= 0;
                        table_weight <= 0;
                        state <= TABLE_W;
                    end else state <= CLEAR_ACC;
                    busy <= 1'b1;
                    out_valid <= 1'b0;
                    dbg_source_reads <= 0;
                    dbg_coef_reads <= 0;
                    dbg_add_cycles <= 0;
                    dbg_build_cycles <= 0;
                end
                TABLE_W: begin
                    dbg_build_cycles <= dbg_build_cycles + 1'b1;
                    if (coef_enable) begin
                        four_weights[table_weight[1:0]] <=
                            coefficient_mem[integer'(table_group)*4+integer'(table_weight)];
                        dbg_coef_reads <= dbg_coef_reads + 1'b1;
                        if (table_weight == 3) begin
                            table_step <= 0;
                            for (k=0; k<8; k=k+1) table_acc[k] <= 0;
                            state <= TABLE_BUILD;
                        end else table_weight <= table_weight + 1'b1;
                    end
                end
                TABLE_BUILD: begin
                    dbg_build_cycles <= dbg_build_cycles + 1'b1;
                    if (coef_enable) begin
                        table_mem[integer'(table_group)*16+integer'(gray)] <= table_value;
                        for (k=0; k<8; k=k+1) table_acc[k] <= $signed(table_value[k*10 +: 10]);
                        if (table_step != 0) dbg_add_cycles <= dbg_add_cycles + 1'b1;
                        if (table_step == 15) begin
                            if ((integer'(table_group)+1)*4 == integer'(channels)) begin
                                table_valid <= 1'b1;
                                table_columns <= channels;
                                state <= CLEAR_ACC;
                            end else begin
                                table_group <= table_group + 1'b1;
                                table_weight <= 0;
                                state <= TABLE_W;
                            end
                        end else table_step <= table_step + 1'b1;
                    end
                end
                CLEAR_ACC: begin
                    // One P8 row is cleared per cycle; initialization is timed.
                    acc[local_row] <= 0;
                    if (local_row == 7) begin
                        source_read_index <= 0;
                        word_index <= 0;
                        state <= LOAD_SRC;
                    end else local_row <= local_row + 1'b1;
                end
                LOAD_SRC: if (src_enable) begin
                    source_block[source_read_index[2:0]] <=
                        source_mem[(integer'(tick_group)*8+integer'(source_read_index))*3+integer'(word_index)];
                    dbg_source_reads <= dbg_source_reads + 1'b1;
                    if (source_read_index == 7) state <= MAKE_BITS;
                    else source_read_index <= source_read_index + 1'b1;
                end
                MAKE_BITS: begin
                    packet_valid <= 0;
                    fill_packet <= 0;
                    fetch_packet <= 0;
                    execute_packet <= 0;
                    state <= RUN;
                end
                RUN: begin
                    if (source_union != 0 && !packet_valid[fill_packet]) begin
                        packet_valid[fill_packet] <= 1'b1;
                        packet_len[fill_packet] <= build_len;
                        fetch_count[fill_packet] <= 0;
                        weight_ready[fill_packet] <= 0;
                        for (k=0; k<4; k=k+1) begin
                            record_address[fill_packet][k] <= build_address[k];
                            record_mask[fill_packet][k] <= build_mask[k];
                        end
                        for (k=0; k<8; k=k+1) source_block[k] <= after_build[k];
                        fill_packet <= ~fill_packet;
                        dbg_build_cycles <= dbg_build_cycles + 1'b1;
                    end
                    // Registered RAM return: execution sees weight_ready only
                    // on the following cycle; no combinational coefficient port.
                    if (coef_enable && packet_valid[fetch_packet] &&
                        fetch_count[fetch_packet] < packet_len[fetch_packet]) begin
                        record_value[fetch_packet][fetch_count[fetch_packet][1:0]] <=
                            table_mem[record_address[fetch_packet][fetch_count[fetch_packet][1:0]]];
                        weight_ready[fetch_packet][fetch_count[fetch_packet][1:0]] <= 1'b1;
                        fetch_count[fetch_packet] <= fetch_count[fetch_packet] + 1'b1;
                        dbg_coef_reads <= dbg_coef_reads + 1'b1;
                        if (fetch_count[fetch_packet] + 1'b1 == packet_len[fetch_packet])
                            fetch_packet <= ~fetch_packet;
                    end
                    if (packet_valid[execute_packet]) begin
                        if (execute_has_work) begin
                            if (weight_ready[execute_packet][merge_record[1:0]]) begin
                                for (k=0; k<8; k=k+1)
                                    acc[merge_position[2:0]][k*24 +: 24] <= alu_result[k];
                                record_mask[execute_packet][merge_record[1:0]][merge_position[2:0]] <= 1'b0;
                                dbg_add_cycles <= dbg_add_cycles + 1'b1;
                                if (!work_after_issue) begin
                                    packet_valid[execute_packet] <= 1'b0;
                                    execute_packet <= ~execute_packet;
                                end
                            end
                        end else begin
                            packet_valid[execute_packet] <= 1'b0;
                            execute_packet <= ~execute_packet;
                        end
                    end
                    if (source_union == 0 && packet_valid == 0) begin
                        if ((integer'(word_index)+1)*128 < integer'(channels)) begin
                            word_index <= word_index + 1'b1;
                            source_read_index <= 0;
                            state <= LOAD_SRC;
                        end else begin
                            local_row <= 0;
                            state <= COMMIT;
                        end
                    end
                end
                COMMIT: begin
                    // One 192-bit Y write per cycle. No free bulk commit.
                    y_mem[integer'(tick_group)*8+integer'(local_row)] <= acc[local_row];
                    if (local_row == 7) begin
                        if (tick_group == 9) begin
                            drain_next <= 0;
                            out_valid <= 0;
                            state <= DRAIN;
                        end else begin
                            tick_group <= tick_group + 1'b1;
                            local_row <= 0;
                            state <= CLEAR_ACC;
                        end
                    end else local_row <= local_row + 1'b1;
                end
                DRAIN: if (!out_valid || out_ready) begin
                    if (drain_next < 80) begin
                        out_data <= y_mem[drain_next];
                        out_row <= drain_next;
                        out_valid <= 1'b1;
                        drain_next <= drain_next + 1'b1;
                    end else begin
                        out_valid <= 1'b0;
                        busy <= 1'b0;
                        done <= 1'b1;
                        state <= IDLE;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
    // mode=0 forces a cold table build; mode=1 reuses a valid same-C table.
    // Weight configuration invalidates the table in either mode.
endmodule
