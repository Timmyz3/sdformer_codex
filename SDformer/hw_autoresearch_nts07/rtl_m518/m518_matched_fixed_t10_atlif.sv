`timescale 1ns/1ps
`default_nettype none

// M518 matched Fixed T10 ATLIF production candidate.
//
// Boundary: five 256-bit configuration beats, five 256-bit raw beats per
// tile, 48-bit tags, two ordered raw banks, and a 16-entry registered 48-bit
// result FIFO. Arithmetic: 100 dynamic signed-INT8 weights, ten signed-Q24
// biases, one signed-Q24 threshold, and exactly 96 signed-INT8 multiplier
// slots. Each tile is closed in 17 issue cycles; cycles 12..16 directly push
// result beats 0..4. There is no intermediate bank or product register.
module m518_matched_fixed_t10_atlif #(
    parameter int TAG_W = 48,
    parameter int FIFO_DEPTH = 16,
    localparam int T = 10,
    localparam int LANES = 16,
    localparam int IN_W = 8,
    localparam int ACC_W = 25,
    localparam int Q24_W = 24,
    localparam int MULTIPLIERS = 96,
    localparam int CONFIG_BITS = 1280,
    localparam int PAYLOAD_BITS = 1064,
    localparam int FIFO_PTR_W = $clog2(FIFO_DEPTH),
    localparam int FIFO_COUNT_W = $clog2(FIFO_DEPTH+1)
) (
    input  logic                     clk_core,
    input  logic                     rst_core,

    input  logic                     config_valid,
    output logic                     config_ready,
    output logic                     config_accept,
    input  logic [255:0]             config_data,
    input  logic                     config_last,

    input  logic                     raw_valid,
    output logic                     raw_ready,
    output logic                     raw_accept,
    input  logic [255:0]             raw_data,
    input  logic                     raw_last,
    input  logic [TAG_W-1:0]         raw_tag,

    output logic                     result_valid,
    input  logic                     result_ready,
    output logic                     result_accept,
    output logic [TAG_W-1:0]         result_tag,
    output logic [2:0]               result_beat,
    output logic [47:0]              result_valid_bits,
    output logic [47:0]              result_data,

    input  logic                     release_valid,
    output logic                     release_ready,
    output logic                     release_accept,

    output logic                     tile_done_valid,
    output logic [TAG_W-1:0]         tile_done_tag,
    output logic                     context_retire_valid,
    output logic [31:0]              context_retire_cycles,
    output logic                     config_loaded,
    output logic                     protocol_error,
    output logic                     busy,

    output logic                     stage1_issue,
    output logic                     stage2_issue,
    output logic                     product_push,
    output logic                     product_replace,
    output logic                     fifo_push,
    output logic                     fifo_pop,
    output logic [FIFO_COUNT_W-1:0]  result_fifo_occupancy,
    output logic [1:0]               raw_bank_occupancy,
    output logic [1:0]               intermediate_bank_occupancy,
    output logic [31:0]              debug_config_beats,
    output logic [31:0]              debug_raw_beats,
    output logic [31:0]              debug_tiles_loaded,
    output logic [31:0]              debug_stage1_issues,
    output logic [31:0]              debug_stage1_done,
    output logic [31:0]              debug_stage2_issues,
    output logic [31:0]              debug_stage2_done,
    output logic [31:0]              debug_product_pushes,
    output logic [31:0]              debug_result_departures,
    output logic [31:0]              debug_product_replacements,
    output logic [31:0]              debug_context_cycles
`ifdef M518_VCS_V06_HARNESS
    ,input logic                     v06_hold_dense_issue
    ,input logic                     v06_first_empty_fill_bank1
`endif
);
    localparam int WEIGHT_OFFSET = 0;
    localparam int BIAS_OFFSET = 800;
    localparam int THRESHOLD_OFFSET = 1040;
    localparam int OUTPUTS = T*LANES;

    logic [CONFIG_BITS-1:0] config_frame_q,config_candidate;
    logic [2:0] config_beat_q;
    logic config_loaded_q,protocol_error_q;
    logic candidate_padding_legal,config_frame_error,raw_frame_error;
    logic zero_tile_release_error,fault_event;
    logic signed [IN_W-1:0] weight_q [0:T*T-1];
    logic signed [Q24_W-1:0] bias_q [0:T-1];
    logic signed [Q24_W-1:0] threshold_q;

    logic [T*LANES*IN_W-1:0] raw_bank0_q,raw_bank1_q;
    logic [TAG_W-1:0] raw_tag0_q,raw_tag1_q;
    logic [31:0] raw_order0_q,raw_order1_q;
    logic [1:0] raw_owned_q,raw_ready_q;
    logic fill_active_q,fill_bank_q;
    logic [2:0] fill_beat_q;
    logic [TAG_W-1:0] fill_tag_q;
    logic raw_target_bank,raw_expected_last;

    logic dense_active_q,dense_raw_bank_q;
    logic [4:0] dense_cycle_q,dense_selected_cycle;
    logic dense_selected_raw_bank,dense_source_valid,dense_issue;
    logic [T*LANES*IN_W-1:0] dense_raw_data;
    logic [TAG_W-1:0] dense_tag;
    logic signed [ACC_W-1:0] acc_q [0:OUTPUTS-1];
    logic signed [ACC_W-1:0] acc_next_comb [0:OUTPUTS-1];
    logic [OUTPUTS-1:0] acc_update_valid_comb;
    // Bind-visible packed image used only by independent atomic-stall SVA.
    // It is a pure view of the architectural accumulator registers.
    logic [(OUTPUTS*ACC_W)-1:0] acc_state_observe;
    logic overflow_safe_comb;

    logic signed [IN_W-1:0] multiplier_a [0:MULTIPLIERS-1];
    logic signed [IN_W-1:0] multiplier_b [0:MULTIPLIERS-1];
    wire signed [(2*IN_W)-1:0] multiplier_product [0:MULTIPLIERS-1];
    logic [MULTIPLIERS-1:0] multiplier_active_mask;
    logic [MULTIPLIERS-1:0] issue_tuple_valid;
    logic [(MULTIPLIERS*4)-1:0] issue_tuple_row;
    logic [(MULTIPLIERS*4)-1:0] issue_tuple_lane;
    logic [(MULTIPLIERS*4)-1:0] issue_tuple_time;

    logic [2:0] dense_result_beat_comb;
    logic [47:0] dense_result_data_comb;
    logic [TAG_W-1:0] fifo_tag_q [0:FIFO_DEPTH-1];
    logic [2:0] fifo_beat_q [0:FIFO_DEPTH-1];
    logic [47:0] fifo_valid_bits_q [0:FIFO_DEPTH-1];
    logic [47:0] fifo_data_q [0:FIFO_DEPTH-1];
    logic [FIFO_PTR_W-1:0] fifo_read_pointer_q,fifo_write_pointer_q;
    logic [FIFO_COUNT_W-1:0] fifo_count_q;
    logic fifo_credit,result_fire;

    logic [31:0] config_beats_q,raw_beats_q,tiles_loaded_q;
    logic [31:0] dense_issues_q,dense_done_q;
    logic [31:0] product_pushes_q,result_departures_q;
    logic context_counting_q;
    logic [31:0] context_cycles_q;
    logic context_retire_valid_q;
    logic [31:0] context_retire_cycles_q;
    logic tile_done_valid_q;
    logic [TAG_W-1:0] tile_done_tag_q;
    logic work_empty;

    function automatic logic signed [Q24_W-1:0] sat_q25_to_q24(
        input logic signed [ACC_W-1:0] value
    );
        logic signed [ACC_W-1:0] maximum,minimum;
        begin
            maximum=25'sd8388607;
            minimum=-25'sd8388608;
            if(value>maximum)sat_q25_to_q24=24'sh7fffff;
            else if(value<minimum)sat_q25_to_q24=24'sh800000;
            else sat_q25_to_q24=value[Q24_W-1:0];
        end
    endfunction

    initial begin
        if(TAG_W!=48||FIFO_DEPTH!=16||T!=10||LANES!=16||IN_W!=8
                ||ACC_W!=25||Q24_W!=24||MULTIPLIERS!=96
                ||CONFIG_BITS!=1280||PAYLOAD_BITS!=1064)
            $fatal(1,"M518 frozen matched Fixed geometry drift");
    end

    always_comb begin:assemble_config
        config_candidate=config_frame_q;
        config_candidate[(config_beat_q*256)+:256]=config_data;
        candidate_padding_legal=
            config_candidate[CONFIG_BITS-1:PAYLOAD_BITS]=='0;
    end

    always_comb begin:raw_framing
        raw_target_bank=fill_active_q?fill_bank_q:
            ((!raw_owned_q[0])?1'b0:1'b1);
`ifdef M518_VCS_V06_HARNESS
        if(!fill_active_q&&raw_owned_q==0&&v06_first_empty_fill_bank1)
            raw_target_bank=1'b1;
`endif
        raw_expected_last=fill_active_q&&(fill_beat_q==4);
    end

    always_comb begin:dense_select
        dense_source_valid=1'b0;
        dense_selected_raw_bank=1'b0;
        dense_selected_cycle='0;
        if(!protocol_error_q)begin
            if(dense_active_q)begin
                dense_source_valid=1'b1;
                dense_selected_raw_bank=dense_raw_bank_q;
                dense_selected_cycle=dense_cycle_q;
            end else if(raw_ready_q!=0)begin
                dense_source_valid=1'b1;
                if(raw_ready_q==2'b11)
                    dense_selected_raw_bank=(raw_order1_q<raw_order0_q);
                else dense_selected_raw_bank=raw_ready_q[0]?1'b0:1'b1;
                dense_selected_cycle=5'd0;
            end
        end
        dense_raw_data=dense_selected_raw_bank?raw_bank1_q:raw_bank0_q;
        dense_tag=dense_selected_raw_bank?raw_tag1_q:raw_tag0_q;
    end

    always_comb begin:ports_and_control
        work_empty=!fill_active_q&&raw_owned_q==0&&!dense_active_q
            &&fifo_count_q==0;
        config_ready=!rst_core&&!config_loaded_q&&!protocol_error_q&&work_empty;
        config_accept=config_valid&&config_ready;
        raw_ready=!rst_core&&config_loaded_q&&!protocol_error_q
            &&(fill_active_q||raw_owned_q!=2'b11);
        raw_accept=raw_valid&&raw_ready;
        config_frame_error=config_accept&&(
            config_last!=(config_beat_q==4)
            ||((config_beat_q==4)&&!candidate_padding_legal));
        raw_frame_error=raw_accept&&(
            raw_last!=raw_expected_last
            ||(fill_active_q&&raw_tag!=fill_tag_q));
        zero_tile_release_error=release_valid&&config_loaded_q
            &&tiles_loaded_q==0&&work_empty&&!raw_valid;
        fault_event=config_frame_error||raw_frame_error
            ||zero_tile_release_error;

        result_valid=fifo_count_q!=0&&!protocol_error_q;
        result_tag=fifo_tag_q[fifo_read_pointer_q];
        result_beat=fifo_beat_q[fifo_read_pointer_q];
        result_valid_bits=fifo_valid_bits_q[fifo_read_pointer_q];
        result_data=fifo_data_q[fifo_read_pointer_q];
        result_accept=result_valid&&result_ready;
        result_fire=result_accept;
        fifo_credit=fifo_count_q<FIFO_DEPTH||result_fire;

        dense_issue=dense_source_valid
            &&(dense_selected_cycle<12||fifo_credit)&&!protocol_error_q
`ifdef M518_VCS_V06_HARNESS
            &&!v06_hold_dense_issue
`endif
            ;
        stage1_issue=dense_issue;
        stage2_issue=1'b0;
        fifo_push=dense_issue&&(dense_selected_cycle>=12);
        fifo_pop=result_fire;
        product_push=fifo_push;
        product_replace=1'b0;

        release_ready=!rst_core&&config_loaded_q&&!protocol_error_q
            &&tiles_loaded_q!=0&&work_empty&&!raw_valid;
        release_accept=release_valid&&release_ready;
        protocol_error=protocol_error_q;
        config_loaded=config_loaded_q;
        busy=!work_empty;
        tile_done_valid=tile_done_valid_q;
        tile_done_tag=tile_done_tag_q;
        context_retire_valid=context_retire_valid_q;
        context_retire_cycles=context_retire_cycles_q;
        result_fifo_occupancy=fifo_count_q;
        raw_bank_occupancy={1'b0,raw_owned_q[0]}
            +{1'b0,raw_owned_q[1]};
        intermediate_bank_occupancy=2'd0;
        debug_config_beats=config_beats_q;
        debug_raw_beats=raw_beats_q;
        debug_tiles_loaded=tiles_loaded_q;
        debug_stage1_issues=dense_issues_q;
        debug_stage1_done=dense_done_q;
        debug_stage2_issues=32'd0;
        debug_stage2_done=32'd0;
        debug_product_pushes=product_pushes_q;
        debug_result_departures=result_departures_q;
        debug_product_replacements=32'd0;
        debug_context_cycles=context_cycles_q;
    end

    always_comb begin:map_multiplier_slots
        multiplier_active_mask='0;
        issue_tuple_valid='0;
        issue_tuple_row='0;
        issue_tuple_lane='0;
        issue_tuple_time='0;
        for(int slot=0;slot<MULTIPLIERS;slot++)begin
            multiplier_a[slot]='0;
            multiplier_b[slot]='0;
        end
        if(dense_issue)begin
            for(int slot=0;slot<MULTIPLIERS;slot++)begin
                int beat,sub,scalar,tap_within,row,lane,time_index;
                logic slot_active;
                beat=0;sub=0;scalar=0;tap_within=0;
                row=0;lane=0;time_index=0;slot_active=1'b1;
                if(dense_selected_cycle<=11)begin
                    beat=dense_selected_cycle/3;
                    sub=dense_selected_cycle%3;
                    scalar=slot/3;
                    tap_within=slot%3;
                    row=(beat*2)+(scalar/16);
                    lane=scalar%16;
                    time_index=(sub*3)+tap_within;
                end else if(dense_selected_cycle<=15)begin
                    if(slot<32)begin
                        beat=dense_selected_cycle-12;
                        scalar=slot;
                        row=(beat*2)+(scalar/16);
                        lane=scalar%16;
                        time_index=9;
                    end else begin
                        scalar=(slot-32)/2;
                        tap_within=(slot-32)%2;
                        row=8+(scalar/16);
                        lane=scalar%16;
                        time_index=((dense_selected_cycle-12)*2)+tap_within;
                    end
                end else begin
                    if(slot<32)begin
                        scalar=slot;
                        row=8+(scalar/16);
                        lane=scalar%16;
                        time_index=8;
                    end else if(slot<64)begin
                        scalar=slot-32;
                        row=8+(scalar/16);
                        lane=scalar%16;
                        time_index=9;
                    end else slot_active=1'b0;
                end
                if(slot_active)begin
                    multiplier_active_mask[slot]=1'b1;
                    issue_tuple_valid[slot]=1'b1;
                    issue_tuple_row[(slot*4)+:4]=row;
                    issue_tuple_lane[(slot*4)+:4]=lane;
                    issue_tuple_time[(slot*4)+:4]=time_index;
                    multiplier_a[slot]=$signed(dense_raw_data[
                        (((time_index*LANES)+lane)*IN_W)+:IN_W]);
                    multiplier_b[slot]=weight_q[(row*T)+time_index];
                end
            end
        end
    end

    generate
        for(genvar multiplier=0;multiplier<MULTIPLIERS;multiplier++)begin:mul96
            assign multiplier_product[multiplier]=
                multiplier_a[multiplier]*multiplier_b[multiplier];
        end
        for(genvar accumulator_view=0;accumulator_view<OUTPUTS;
                accumulator_view++)begin:pack_acc_state
            assign acc_state_observe[(accumulator_view*ACC_W)+:ACC_W]=
                acc_q[accumulator_view];
        end
    endgenerate

    always_comb begin:reduce_and_close
        acc_update_valid_comb='0;
        overflow_safe_comb=1'b1;
        dense_result_beat_comb=(dense_selected_cycle>=12)?
            dense_selected_cycle-12:3'd0;
        dense_result_data_comb='0;
        for(int accumulator=0;accumulator<OUTPUTS;accumulator++)
            acc_next_comb[accumulator]=acc_q[accumulator];

        if(dense_issue&&dense_selected_cycle<=11)begin
            for(int scalar=0;scalar<32;scalar++)begin
                logic signed [25:0] base_value,product0,product1,product2;
                logic signed [25:0] wide_sum;
                int beat,sub,row,lane,accumulator;
                beat=dense_selected_cycle/3;
                sub=dense_selected_cycle%3;
                row=(beat*2)+(scalar/16);
                lane=scalar%16;
                accumulator=(row*LANES)+lane;
                if(sub==0)
                    base_value=$signed({{2{bias_q[row][Q24_W-1]}},bias_q[row]});
                else base_value=$signed({acc_q[accumulator][ACC_W-1],
                    acc_q[accumulator]});
                product0=$signed({{10{multiplier_product[(scalar*3)+0][15]}},
                    multiplier_product[(scalar*3)+0]});
                product1=$signed({{10{multiplier_product[(scalar*3)+1][15]}},
                    multiplier_product[(scalar*3)+1]});
                product2=$signed({{10{multiplier_product[(scalar*3)+2][15]}},
                    multiplier_product[(scalar*3)+2]});
                wide_sum=base_value+product0+product1+product2;
                if(wide_sum[25]!=wide_sum[24])overflow_safe_comb=1'b0;
                acc_update_valid_comb[accumulator]=1'b1;
                acc_next_comb[accumulator]=wide_sum[ACC_W-1:0];
            end
        end else if(dense_issue&&dense_selected_cycle<=15)begin
            for(int scalar=0;scalar<32;scalar++)begin
                logic signed [25:0] base_value,product0,product1,wide_sum;
                logic signed [Q24_W-1:0] saturated_output;
                int beat,row,lane,accumulator;
                beat=dense_selected_cycle-12;
                row=(beat*2)+(scalar/16);
                lane=scalar%16;
                accumulator=(row*LANES)+lane;
                base_value=$signed({acc_q[accumulator][ACC_W-1],
                    acc_q[accumulator]});
                product0=$signed({{10{multiplier_product[scalar][15]}},
                    multiplier_product[scalar]});
                wide_sum=base_value+product0;
                if(wide_sum[25]!=wide_sum[24])overflow_safe_comb=1'b0;
                saturated_output=sat_q25_to_q24(wide_sum[ACC_W-1:0]);
                dense_result_data_comb[scalar]=
                    $signed(saturated_output)>=$signed(threshold_q);

                row=8+(scalar/16);
                accumulator=(row*LANES)+lane;
                if(dense_selected_cycle==12)
                    base_value=$signed({{2{bias_q[row][Q24_W-1]}},bias_q[row]});
                else base_value=$signed({acc_q[accumulator][ACC_W-1],
                    acc_q[accumulator]});
                product0=$signed({{10{multiplier_product[32+(scalar*2)][15]}},
                    multiplier_product[32+(scalar*2)]});
                product1=$signed({{10{
                    multiplier_product[33+(scalar*2)][15]}},
                    multiplier_product[33+(scalar*2)]});
                wide_sum=base_value+product0+product1;
                if(wide_sum[25]!=wide_sum[24])overflow_safe_comb=1'b0;
                acc_update_valid_comb[accumulator]=1'b1;
                acc_next_comb[accumulator]=wide_sum[ACC_W-1:0];
            end
        end else if(dense_issue)begin
            for(int scalar=0;scalar<32;scalar++)begin
                logic signed [25:0] base_value,product0,product1,wide_sum;
                logic signed [Q24_W-1:0] saturated_output;
                int row,lane,accumulator;
                row=8+(scalar/16);
                lane=scalar%16;
                accumulator=(row*LANES)+lane;
                base_value=$signed({acc_q[accumulator][ACC_W-1],
                    acc_q[accumulator]});
                product0=$signed({{10{multiplier_product[scalar][15]}},
                    multiplier_product[scalar]});
                product1=$signed({{10{multiplier_product[32+scalar][15]}},
                    multiplier_product[32+scalar]});
                wide_sum=base_value+product0+product1;
                if(wide_sum[25]!=wide_sum[24])overflow_safe_comb=1'b0;
                saturated_output=sat_q25_to_q24(wide_sum[ACC_W-1:0]);
                dense_result_data_comb[scalar]=
                    $signed(saturated_output)>=$signed(threshold_q);
            end
        end
    end

    always_ff @(posedge clk_core)begin:state
        if(rst_core)begin
            config_frame_q<='0;
            config_beat_q<='0;
            config_loaded_q<=1'b0;
            protocol_error_q<=1'b0;
            for(int coefficient=0;coefficient<T*T;coefficient++)
                weight_q[coefficient]<='0;
            for(int row=0;row<T;row++)bias_q[row]<='0;
            threshold_q<='0;

            raw_bank0_q<='0;raw_bank1_q<='0;
            raw_tag0_q<='0;raw_tag1_q<='0;
            raw_order0_q<='0;raw_order1_q<='0;
            raw_owned_q<='0;raw_ready_q<='0;
            fill_active_q<=1'b0;fill_bank_q<=1'b0;
            fill_beat_q<='0;fill_tag_q<='0;

            dense_active_q<=1'b0;dense_raw_bank_q<=1'b0;
            dense_cycle_q<='0;
            for(int accumulator=0;accumulator<OUTPUTS;accumulator++)
                acc_q[accumulator]<='0;

            for(int entry=0;entry<FIFO_DEPTH;entry++)begin
                fifo_tag_q[entry]<='0;fifo_beat_q[entry]<='0;
                fifo_valid_bits_q[entry]<='0;fifo_data_q[entry]<='0;
            end
            fifo_read_pointer_q<='0;fifo_write_pointer_q<='0;
            fifo_count_q<='0;

            config_beats_q<='0;raw_beats_q<='0;tiles_loaded_q<='0;
            dense_issues_q<='0;dense_done_q<='0;
            product_pushes_q<='0;result_departures_q<='0;
            context_counting_q<=1'b0;context_cycles_q<='0;
            context_retire_valid_q<=1'b0;context_retire_cycles_q<='0;
            tile_done_valid_q<=1'b0;tile_done_tag_q<='0;
        end else begin
            context_retire_valid_q<=1'b0;
            tile_done_valid_q<=1'b0;
            if(fault_event)protocol_error_q<=1'b1;
            if(!protocol_error_q)begin
                if(context_counting_q)begin
                    if(release_accept)begin
                        context_retire_valid_q<=1'b1;
                        context_retire_cycles_q<=context_cycles_q+1'b1;
                        context_counting_q<=1'b0;
                    end else context_cycles_q<=context_cycles_q+1'b1;
                end

                if(config_accept&&!config_frame_error)begin
                    if(config_beat_q==0)begin
                        context_counting_q<=1'b1;context_cycles_q<=1;
                        config_beats_q<=1;raw_beats_q<='0;tiles_loaded_q<='0;
                        dense_issues_q<='0;dense_done_q<='0;
                        product_pushes_q<='0;result_departures_q<='0;
                    end else config_beats_q<=config_beats_q+1'b1;
                    config_frame_q<=config_candidate;
                    if(config_beat_q==4)begin
                        config_beat_q<='0;config_loaded_q<=1'b1;
                        for(int coefficient=0;coefficient<T*T;coefficient++)
                            weight_q[coefficient]<=$signed(config_candidate[
                                WEIGHT_OFFSET+(coefficient*IN_W)+:IN_W]);
                        for(int row=0;row<T;row++)
                            bias_q[row]<=$signed(config_candidate[
                                BIAS_OFFSET+(row*Q24_W)+:Q24_W]);
                        threshold_q<=$signed(config_candidate[
                            THRESHOLD_OFFSET+:Q24_W]);
                    end else config_beat_q<=config_beat_q+1'b1;
                end

                if(raw_accept&&!raw_frame_error)begin
                    raw_beats_q<=raw_beats_q+1'b1;
                    if(!fill_active_q)begin
                        raw_owned_q[raw_target_bank]<=1'b1;
                        if(raw_target_bank)begin
                            raw_bank1_q[0+:256]<=raw_data;raw_tag1_q<=raw_tag;
                        end else begin
                            raw_bank0_q[0+:256]<=raw_data;raw_tag0_q<=raw_tag;
                        end
                        fill_active_q<=1'b1;fill_bank_q<=raw_target_bank;
                        fill_beat_q<=1;fill_tag_q<=raw_tag;
                    end else begin
                        if(fill_bank_q)raw_bank1_q[fill_beat_q*256+:256]<=raw_data;
                        else raw_bank0_q[fill_beat_q*256+:256]<=raw_data;
                        if(fill_beat_q==4)begin
                            raw_ready_q[fill_bank_q]<=1'b1;
                            fill_active_q<=1'b0;fill_beat_q<='0;
                            tiles_loaded_q<=tiles_loaded_q+1'b1;
                            if(fill_bank_q)raw_order1_q<=tiles_loaded_q;
                            else raw_order0_q<=tiles_loaded_q;
                        end else fill_beat_q<=fill_beat_q+1'b1;
                    end
                end

                if(dense_issue)begin
                    dense_issues_q<=dense_issues_q+1'b1;
                    if(!dense_active_q)begin
                        dense_raw_bank_q<=dense_selected_raw_bank;
                        raw_ready_q[dense_selected_raw_bank]<=1'b0;
                    end
                    for(int accumulator=0;accumulator<OUTPUTS;accumulator++)
                        if(acc_update_valid_comb[accumulator])
                            acc_q[accumulator]<=acc_next_comb[accumulator];
                    if(dense_selected_cycle==16)begin
                        dense_active_q<=1'b0;dense_cycle_q<='0;
                        raw_owned_q[dense_selected_raw_bank]<=1'b0;
                        dense_done_q<=dense_done_q+1'b1;
                    end else begin
                        dense_active_q<=1'b1;
                        dense_cycle_q<=dense_selected_cycle+1'b1;
                    end
                end

                if(fifo_push)begin
                    fifo_tag_q[fifo_write_pointer_q]<=dense_tag;
                    fifo_beat_q[fifo_write_pointer_q]<=dense_result_beat_comb;
                    fifo_valid_bits_q[fifo_write_pointer_q]
                        <=48'h0000ffffffff;
                    fifo_data_q[fifo_write_pointer_q]<=dense_result_data_comb;
                    fifo_write_pointer_q<=fifo_write_pointer_q+1'b1;
                    product_pushes_q<=product_pushes_q+1'b1;
                    if(dense_result_beat_comb==4)begin
                        tile_done_valid_q<=1'b1;tile_done_tag_q<=dense_tag;
                    end
                end
                if(result_fire)begin
                    fifo_read_pointer_q<=fifo_read_pointer_q+1'b1;
                    result_departures_q<=result_departures_q+1'b1;
                end
                case({fifo_push,result_fire})
                    2'b10:fifo_count_q<=fifo_count_q+1'b1;
                    2'b01:fifo_count_q<=fifo_count_q-1'b1;
                    default:fifo_count_q<=fifo_count_q;
                endcase
                if(release_accept)config_loaded_q<=1'b0;
            end
        end
    end
endmodule
`default_nettype wire
