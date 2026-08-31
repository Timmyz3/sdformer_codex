`timescale 1ns/1ps
`default_nettype none

// M273 integrates the complete rank-3 T10 ATLIF module boundary used by M265:
// six 256-bit configuration beats, five 256-bit raw beats per tile, a 96-way
// INT8 stage1, an M37-class CSD stage2, one elastic product register and one
// registered 48-bit result FIFO.  M285/M273r2 registers fault observation at
// the offending handshake edge: legal sustained-valid traffic cannot create a
// post-edge combinational fault pulse or transiently retract issue/result_valid.
// Fixed is deliberately not instantiated here; the M31/M265 exact96 baseline
// is not yet area matched.
module m273_integrated_rank3_atlif #(
    parameter int TAG_W = 48,
    parameter int FIFO_DEPTH = 16,
    localparam int T = 10,
    localparam int RANK = 3,
    localparam int LANES = 16,
    localparam int IN_W = 8,
    localparam int ACC_W = 24,
    localparam int TERMS = 4,
    localparam int CONFIG_BITS = 1536,
    localparam int PAYLOAD_BITS = 1349,
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
);
    localparam int RIGHT_OFFSET = 0;
    localparam int REQUANT_OFFSET = 240;
    localparam int LEFT_OFFSET = 245;
    localparam int VALID_OFFSET = 485;
    localparam int NEGATIVE_OFFSET = 605;
    localparam int SHIFT_OFFSET = 725;
    localparam int BIAS_OFFSET = 1085;
    localparam int THRESHOLD_OFFSET = 1325;

    logic [CONFIG_BITS-1:0] config_frame_q,config_candidate;
    logic [2:0] config_beat_q;
    logic config_loaded_q,protocol_error_q;
    logic candidate_descriptor_legal,candidate_padding_legal;
    logic candidate_requant_legal,config_frame_error,raw_frame_error;
    logic zero_tile_release_error;
    logic fault_event;

    logic [239:0] right_factor_q;
    logic [4:0] requant_shift_q;
    logic [119:0] term_valid_q,term_negative_q;
    logic [359:0] term_shift_q;
    logic [239:0] bias_q;
    logic signed [ACC_W-1:0] threshold_q;

    logic [1279:0] raw_bank0_q,raw_bank1_q;
    logic [TAG_W-1:0] raw_tag0_q,raw_tag1_q;
    logic [31:0] raw_order0_q,raw_order1_q;
    logic [1:0] raw_owned_q,raw_ready_q;
    logic fill_active_q,fill_bank_q;
    logic [2:0] fill_beat_q;
    logic [TAG_W-1:0] fill_tag_q;
    logic raw_target_bank,raw_expected_last;

    logic stage1_active_q,stage1_raw_bank_q,stage1_inter_bank_q;
    logic [2:0] stage1_phase_q,stage1_selected_phase;
    logic stage1_selected_raw_bank,stage1_selected_inter_bank;
    logic stage1_source_valid;
    logic [1279:0] stage1_raw_data;
    logic signed [ACC_W-1:0] stage1_acc_q [0:RANK*LANES-1];
    logic signed [ACC_W-1:0] stage1_sum_comb [0:RANK*LANES-1];
    logic [RANK*LANES*IN_W-1:0] stage1_requant_comb;

    logic [RANK*LANES*IN_W-1:0] inter_bank0_q,inter_bank1_q;
    logic [TAG_W-1:0] inter_tag0_q,inter_tag1_q;
    logic [31:0] inter_order0_q,inter_order1_q;
    logic [1:0] inter_reserved_q,inter_valid_q;
    logic stage2_active_q,stage2_bank_q;
    logic [2:0] stage2_phase_q,stage2_selected_phase;
    logic stage2_selected_bank,stage2_source_valid;
    logic [RANK*LANES*IN_W-1:0] stage2_intermediate;
    logic [TAG_W-1:0] stage2_tag;
    logic [31:0] stage2_event_bits;
    logic [47:0] stage2_data_comb,stage2_valid_bits_comb;

    logic product_valid_q;
    logic [TAG_W-1:0] product_tag_q;
    logic [2:0] product_beat_q;
    logic [47:0] product_valid_bits_q,product_data_q;
    logic product_stage_ready;

    logic [TAG_W-1:0] fifo_tag_q [0:FIFO_DEPTH-1];
    logic [2:0] fifo_beat_q [0:FIFO_DEPTH-1];
    logic [47:0] fifo_valid_bits_q [0:FIFO_DEPTH-1];
    logic [47:0] fifo_data_q [0:FIFO_DEPTH-1];
    logic [FIFO_PTR_W-1:0] fifo_read_pointer_q,fifo_write_pointer_q;
    logic [FIFO_COUNT_W-1:0] fifo_count_q;
    logic fifo_credit,result_fire;

    logic [31:0] config_beats_q,raw_beats_q,tiles_loaded_q;
    logic [31:0] stage1_issues_q,stage1_done_q;
    logic [31:0] stage2_issues_q,stage2_done_q;
    logic [31:0] product_pushes_q,result_departures_q,replacements_q;
    logic context_counting_q;
    logic [31:0] context_cycles_q;
    logic context_retire_valid_q;
    logic [31:0] context_retire_cycles_q;
    logic tile_done_valid_q;
    logic [TAG_W-1:0] tile_done_tag_q;
    logic work_empty;

    function automatic logic signed [IN_W-1:0] rne_sat_q24_to_q8(
        input logic signed [ACC_W-1:0] value,
        input logic [4:0] shift
    );
        logic negative;
        logic [ACC_W-1:0] magnitude,quotient,remainder,remainder_mask,half;
        logic round_up;
        logic [ACC_W:0] rounded_magnitude;
        begin
            if(shift>23)rne_sat_q24_to_q8='0;
            else begin
                negative=value[ACC_W-1];
                magnitude=negative?(~$unsigned(value)+1'b1):$unsigned(value);
                if(shift==0)begin
                    quotient=magnitude;remainder='0;half='0;round_up=1'b0;
                end else begin
                    remainder_mask=({ACC_W{1'b1}}>>(ACC_W-shift));
                    remainder=magnitude&remainder_mask;
                    half={{(ACC_W-1){1'b0}},1'b1}<<(shift-1'b1);
                    quotient=magnitude>>shift;
                    round_up=(remainder>half)||((remainder==half)&&quotient[0]);
                end
                rounded_magnitude={1'b0,quotient}+round_up;
                if(!negative&&rounded_magnitude>127)rne_sat_q24_to_q8=8'sd127;
                else if(negative&&rounded_magnitude>128)
                    rne_sat_q24_to_q8=-8'sd128;
                else if(negative)
                    rne_sat_q24_to_q8=-$signed(rounded_magnitude[IN_W-1:0]);
                else rne_sat_q24_to_q8=$signed(rounded_magnitude[IN_W-1:0]);
            end
        end
    endfunction

    function automatic logic signed [17:0] csd_product(
        input logic signed [IN_W-1:0] value,
        input logic [TERMS-1:0] valid,
        input logic [TERMS-1:0] negative,
        input logic [TERMS*3-1:0] shifts
    );
        logic signed [17:0] value_wide,term_value,total;
        begin
            value_wide={{(18-IN_W){value[IN_W-1]}},value};
            total='0;
            for(int term=0;term<TERMS;term++)begin
                case(shifts[(term*3)+:3])
                    3'd0:term_value=value_wide;
                    3'd1:term_value=value_wide<<<1;
                    3'd2:term_value=value_wide<<<2;
                    3'd3:term_value=value_wide<<<3;
                    3'd4:term_value=value_wide<<<4;
                    3'd5:term_value=value_wide<<<5;
                    3'd6:term_value=value_wide<<<6;
                    default:term_value=value_wide<<<7;
                endcase
                if(valid[term])begin
                    if(negative[term])total=total-term_value;
                    else total=total+term_value;
                end
            end
            csd_product=total;
        end
    endfunction

    function automatic logic signed [ACC_W-1:0] sat_q26_to_q24(
        input logic signed [ACC_W+1:0] value
    );
        logic signed [ACC_W+1:0] maximum,minimum;
        begin
            maximum=({{(ACC_W+1){1'b0}},1'b1}<<(ACC_W-1))-1'b1;
            minimum=-({{(ACC_W+1){1'b0}},1'b1}<<(ACC_W-1));
            if(value>maximum)sat_q26_to_q24={1'b0,{(ACC_W-1){1'b1}}};
            else if(value<minimum)sat_q26_to_q24={1'b1,{(ACC_W-1){1'b0}}};
            else sat_q26_to_q24=value[ACC_W-1:0];
        end
    endfunction

    initial begin
        if(TAG_W!=48||FIFO_DEPTH!=16||T!=10||RANK!=3||LANES!=16
                ||IN_W!=8||ACC_W!=24||TERMS!=4)
            $fatal(1,"M273 frozen integrated geometry drift");
    end

    always_comb begin:assemble_and_validate_config
        config_candidate=config_frame_q;
        config_candidate[(config_beat_q*256)+:256]=config_data;
        candidate_padding_legal=config_candidate[CONFIG_BITS-1:PAYLOAD_BITS]=='0;
        candidate_requant_legal=config_candidate[REQUANT_OFFSET+:5]<=23;
        candidate_descriptor_legal=1'b1;
        for(int coefficient=0;coefficient<T*RANK;coefficient++)begin
            logic signed[10:0] descriptor_sum;
            logic seen_invalid,previous_valid;
            logic[2:0] previous_shift;
            descriptor_sum='0;seen_invalid=1'b0;previous_valid=1'b0;
            previous_shift='0;
            for(int term=0;term<TERMS;term++)begin
                logic term_valid,term_negative;
                logic[2:0] term_shift;
                term_valid=config_candidate[VALID_OFFSET+(coefficient*TERMS)+term];
                term_negative=config_candidate[
                    NEGATIVE_OFFSET+(coefficient*TERMS)+term];
                term_shift=config_candidate[
                    SHIFT_OFFSET+(((coefficient*TERMS)+term)*3)+:3];
                if(term_valid)begin
                    if(seen_invalid)candidate_descriptor_legal=1'b0;
                    if(previous_valid&&({1'b0,term_shift}
                            <=({1'b0,previous_shift}+4'd1)))
                        candidate_descriptor_legal=1'b0;
                    if(term_negative)descriptor_sum=descriptor_sum-(11'sd1<<<term_shift);
                    else descriptor_sum=descriptor_sum+(11'sd1<<<term_shift);
                    previous_valid=1'b1;previous_shift=term_shift;
                end else begin
                    seen_invalid=1'b1;
                    if(term_negative||term_shift!=0)candidate_descriptor_legal=1'b0;
                end
            end
            if(descriptor_sum!=$signed({{3{config_candidate[
                    LEFT_OFFSET+(coefficient*IN_W)+IN_W-1]}},
                    config_candidate[LEFT_OFFSET+(coefficient*IN_W)+:IN_W]}))
                candidate_descriptor_legal=1'b0;
        end
    end

    always_comb begin:raw_framing
        raw_target_bank=fill_active_q?fill_bank_q:
            ((!raw_owned_q[0])?1'b0:1'b1);
        raw_expected_last=fill_active_q&&(fill_beat_q==4);
    end

    always_comb begin:stage1_select_and_compute
        stage1_source_valid=1'b0;stage1_selected_raw_bank=1'b0;
        stage1_selected_inter_bank=1'b0;stage1_selected_phase='0;
        if(!protocol_error_q)begin
            if(stage1_active_q)begin
                stage1_source_valid=1'b1;
                stage1_selected_raw_bank=stage1_raw_bank_q;
                stage1_selected_inter_bank=stage1_inter_bank_q;
                stage1_selected_phase=stage1_phase_q;
            end else if(raw_ready_q!=0&&inter_reserved_q!=2'b11)begin
                stage1_source_valid=1'b1;
                if(raw_ready_q==2'b11)
                    stage1_selected_raw_bank=(raw_order1_q<raw_order0_q);
                else stage1_selected_raw_bank=raw_ready_q[0]?1'b0:1'b1;
                stage1_selected_inter_bank=!inter_reserved_q[0]?1'b0:1'b1;
                stage1_selected_phase=3'd0;
            end
        end
        stage1_raw_data=stage1_selected_raw_bank?raw_bank1_q:raw_bank0_q;
        stage1_requant_comb='0;
        for(int rank=0;rank<RANK;rank++)begin
            for(int lane=0;lane<LANES;lane++)begin
                logic signed[IN_W-1:0] x0,x1,w0,w1;
                logic signed[(2*IN_W)-1:0] product0,product1;
                logic signed[ACC_W:0] base_value;
                logic signed[ACC_W:0] sum_ext;
                int accumulator,time0,time1;
                accumulator=(rank*LANES)+lane;
                time0=stage1_selected_phase*2;
                time1=time0+1;
                x0=$signed(stage1_raw_data[((time0*LANES)+lane)*IN_W+:IN_W]);
                x1=$signed(stage1_raw_data[((time1*LANES)+lane)*IN_W+:IN_W]);
                w0=$signed(right_factor_q[((rank*T)+time0)*IN_W+:IN_W]);
                w1=$signed(right_factor_q[((rank*T)+time1)*IN_W+:IN_W]);
                product0=x0*w0;product1=x1*w1;
                if(stage1_selected_phase==0)base_value='0;
                else base_value={stage1_acc_q[accumulator][ACC_W-1],
                    stage1_acc_q[accumulator]};
                sum_ext=base_value
                    +$signed({{(ACC_W+1-2*IN_W){product0[(2*IN_W)-1]}},product0})
                    +$signed({{(ACC_W+1-2*IN_W){product1[(2*IN_W)-1]}},product1});
                stage1_sum_comb[accumulator]=sum_ext[ACC_W-1:0];
                stage1_requant_comb[accumulator*IN_W+:IN_W]=rne_sat_q24_to_q8(
                    sum_ext[ACC_W-1:0],requant_shift_q);
            end
        end
    end

    always_comb begin:stage2_select_and_compute
        stage2_source_valid=1'b0;stage2_selected_bank=1'b0;
        stage2_selected_phase='0;
        if(!protocol_error_q)begin
            if(stage2_active_q)begin
                stage2_source_valid=1'b1;stage2_selected_bank=stage2_bank_q;
                stage2_selected_phase=stage2_phase_q;
            end else if(inter_valid_q!=0)begin
                stage2_source_valid=1'b1;
                if(inter_valid_q==2'b11)
                    stage2_selected_bank=(inter_order1_q<inter_order0_q);
                else stage2_selected_bank=inter_valid_q[0]?1'b0:1'b1;
                stage2_selected_phase=3'd0;
            end
        end
        stage2_intermediate=stage2_selected_bank?inter_bank1_q:inter_bank0_q;
        stage2_tag=stage2_selected_bank?inter_tag1_q:inter_tag0_q;
        stage2_event_bits='0;
        for(int row_in_beat=0;row_in_beat<2;row_in_beat++)begin
            for(int lane=0;lane<LANES;lane++)begin
                logic signed[ACC_W+1:0] output_sum;
                logic signed[ACC_W-1:0] saturated_output;
                int output_row,output_bit;
                output_row=(stage2_selected_phase*2)+row_in_beat;
                output_bit=(row_in_beat*LANES)+lane;
                output_sum=$signed({{2{bias_q[(output_row*ACC_W)+ACC_W-1]}},
                                   bias_q[(output_row*ACC_W)+:ACC_W]});
                for(int rank=0;rank<RANK;rank++)begin
                    logic signed[17:0] product_value;
                    int descriptor,intermediate;
                    descriptor=(output_row*RANK)+rank;
                    intermediate=(rank*LANES)+lane;
                    product_value=csd_product(
                        $signed(stage2_intermediate[intermediate*IN_W+:IN_W]),
                        term_valid_q[descriptor*TERMS+:TERMS],
                        term_negative_q[descriptor*TERMS+:TERMS],
                        term_shift_q[descriptor*TERMS*3+:TERMS*3]);
                    output_sum=output_sum+$signed({{(ACC_W+2-18){
                        product_value[17]}},product_value});
                end
                saturated_output=sat_q26_to_q24(output_sum);
                stage2_event_bits[output_bit]=
                    $signed(saturated_output)>=$signed(threshold_q);
            end
        end
        stage2_data_comb={{16{1'b0}},stage2_event_bits};
        stage2_valid_bits_comb={{16{1'b0}},{32{1'b1}}};
    end

    always_comb begin:ports_and_control
        work_empty=!fill_active_q&&raw_owned_q==0&&!stage1_active_q
            &&inter_reserved_q==0&&!stage2_active_q&&!product_valid_q
            &&fifo_count_q==0;
        config_ready=!rst_core&&!config_loaded_q&&!protocol_error_q&&work_empty;
        config_accept=config_valid&&config_ready;
        raw_ready=!rst_core&&config_loaded_q&&!protocol_error_q
            &&(fill_active_q||raw_owned_q!=2'b11);
        raw_accept=raw_valid&&raw_ready;
        config_frame_error=config_accept&&(
            config_last!=(config_beat_q==5)
            ||((config_beat_q==5)&&(!candidate_padding_legal
               ||!candidate_requant_legal||!candidate_descriptor_legal)));
        raw_frame_error=raw_accept&&(
            raw_last!=raw_expected_last
            ||(fill_active_q&&raw_tag!=fill_tag_q));
        // Fault causes are evaluated from the transaction presented to the
        // current handshake state and sampled only by integrated_state.  They
        // are deliberately not exposed as combinational output qualifiers.
        // Releasing an empty configured context is an illegal N=0 operation;
        // release never handshakes and the attempt enters sticky quarantine.
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
        product_push=product_valid_q&&fifo_credit&&!protocol_error_q;
        product_stage_ready=!product_valid_q||fifo_credit;
        stage1_issue=stage1_source_valid;
        stage2_issue=stage2_source_valid&&product_stage_ready;
        product_replace=product_push&&stage2_issue;
        fifo_push=product_push;fifo_pop=result_fire;

        release_ready=!rst_core&&config_loaded_q&&!protocol_error_q
            &&tiles_loaded_q!=0&&work_empty&&!raw_valid;
        release_accept=release_valid&&release_ready;
        protocol_error=protocol_error_q;
        config_loaded=config_loaded_q;
        busy=!work_empty;
        tile_done_valid=tile_done_valid_q;tile_done_tag=tile_done_tag_q;
        context_retire_valid=context_retire_valid_q;
        context_retire_cycles=context_retire_cycles_q;
        result_fifo_occupancy=fifo_count_q;
        raw_bank_occupancy=raw_owned_q[0]+raw_owned_q[1];
        intermediate_bank_occupancy=inter_reserved_q[0]+inter_reserved_q[1];
        debug_config_beats=config_beats_q;debug_raw_beats=raw_beats_q;
        debug_tiles_loaded=tiles_loaded_q;
        debug_stage1_issues=stage1_issues_q;debug_stage1_done=stage1_done_q;
        debug_stage2_issues=stage2_issues_q;debug_stage2_done=stage2_done_q;
        debug_product_pushes=product_pushes_q;
        debug_result_departures=result_departures_q;
        debug_product_replacements=replacements_q;
        debug_context_cycles=context_cycles_q;
    end

    always_ff @(posedge clk_core)begin:integrated_state
        if(rst_core)begin
            config_frame_q<='0;config_beat_q<='0;config_loaded_q<=1'b0;
            protocol_error_q<=1'b0;right_factor_q<='0;requant_shift_q<='0;
            term_valid_q<='0;term_negative_q<='0;term_shift_q<='0;
            bias_q<='0;threshold_q<='0;
            raw_bank0_q<='0;raw_bank1_q<='0;raw_tag0_q<='0;raw_tag1_q<='0;
            raw_order0_q<='0;raw_order1_q<='0;
            raw_owned_q<='0;raw_ready_q<='0;fill_active_q<=1'b0;
            fill_bank_q<=1'b0;fill_beat_q<='0;fill_tag_q<='0;
            stage1_active_q<=1'b0;stage1_raw_bank_q<=1'b0;
            stage1_inter_bank_q<=1'b0;stage1_phase_q<='0;
            for(int accumulator=0;accumulator<RANK*LANES;accumulator++)
                stage1_acc_q[accumulator]<='0;
            inter_bank0_q<='0;inter_bank1_q<='0;inter_tag0_q<='0;inter_tag1_q<='0;
            inter_order0_q<='0;inter_order1_q<='0;
            inter_reserved_q<='0;inter_valid_q<='0;
            stage2_active_q<=1'b0;stage2_bank_q<=1'b0;stage2_phase_q<='0;
            product_valid_q<=1'b0;product_tag_q<='0;product_beat_q<='0;
            product_valid_bits_q<='0;product_data_q<='0;
            fifo_read_pointer_q<='0;fifo_write_pointer_q<='0;fifo_count_q<='0;
            config_beats_q<='0;raw_beats_q<='0;tiles_loaded_q<='0;
            stage1_issues_q<='0;stage1_done_q<='0;stage2_issues_q<='0;
            stage2_done_q<='0;product_pushes_q<='0;result_departures_q<='0;
            replacements_q<='0;context_counting_q<=1'b0;context_cycles_q<='0;
            context_retire_valid_q<=1'b0;context_retire_cycles_q<='0;
            tile_done_valid_q<=1'b0;tile_done_tag_q<='0;
        end else begin
            context_retire_valid_q<=1'b0;tile_done_valid_q<=1'b0;
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
                        stage1_issues_q<='0;stage1_done_q<='0;
                        stage2_issues_q<='0;stage2_done_q<='0;
                        product_pushes_q<='0;result_departures_q<='0;
                        replacements_q<='0;
                    end else config_beats_q<=config_beats_q+1'b1;
                    config_frame_q<=config_candidate;
                    if(config_beat_q==5)begin
                        config_beat_q<='0;config_loaded_q<=1'b1;
                        right_factor_q<=config_candidate[RIGHT_OFFSET+:240];
                        requant_shift_q<=config_candidate[REQUANT_OFFSET+:5];
                        term_valid_q<=config_candidate[VALID_OFFSET+:120];
                        term_negative_q<=config_candidate[NEGATIVE_OFFSET+:120];
                        term_shift_q<=config_candidate[SHIFT_OFFSET+:360];
                        bias_q<=config_candidate[BIAS_OFFSET+:240];
                        threshold_q<=config_candidate[THRESHOLD_OFFSET+:24];
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
                            raw_ready_q[fill_bank_q]<=1'b1;fill_active_q<=1'b0;
                            fill_beat_q<='0;tiles_loaded_q<=tiles_loaded_q+1'b1;
                            if(fill_bank_q)raw_order1_q<=tiles_loaded_q;
                            else raw_order0_q<=tiles_loaded_q;
                        end else fill_beat_q<=fill_beat_q+1'b1;
                    end
                end

                if(stage1_issue)begin
                    stage1_issues_q<=stage1_issues_q+1'b1;
                    if(!stage1_active_q)begin
                        stage1_raw_bank_q<=stage1_selected_raw_bank;
                        stage1_inter_bank_q<=stage1_selected_inter_bank;
                        raw_ready_q[stage1_selected_raw_bank]<=1'b0;
                        inter_reserved_q[stage1_selected_inter_bank]<=1'b1;
                    end
                    for(int accumulator=0;accumulator<RANK*LANES;accumulator++)begin
                        if(stage1_selected_phase==4)begin
                            if(stage1_selected_inter_bank)
                                inter_bank1_q[accumulator*IN_W+:IN_W]
                                    <=stage1_requant_comb[accumulator*IN_W+:IN_W];
                            else inter_bank0_q[accumulator*IN_W+:IN_W]
                                    <=stage1_requant_comb[accumulator*IN_W+:IN_W];
                        end else stage1_acc_q[accumulator]
                            <=stage1_sum_comb[accumulator];
                    end
                    if(stage1_selected_phase==4)begin
                        stage1_active_q<=1'b0;stage1_phase_q<='0;
                        raw_owned_q[stage1_selected_raw_bank]<=1'b0;
                        inter_valid_q[stage1_selected_inter_bank]<=1'b1;
                        if(stage1_selected_inter_bank)
                            begin
                                inter_tag1_q<=stage1_selected_raw_bank?raw_tag1_q:raw_tag0_q;
                                inter_order1_q<=stage1_selected_raw_bank?raw_order1_q:raw_order0_q;
                            end
                        else begin
                            inter_tag0_q<=stage1_selected_raw_bank?raw_tag1_q:raw_tag0_q;
                            inter_order0_q<=stage1_selected_raw_bank?raw_order1_q:raw_order0_q;
                        end
                        stage1_done_q<=stage1_done_q+1'b1;
                    end else begin
                        stage1_active_q<=1'b1;
                        stage1_phase_q<=stage1_selected_phase+1'b1;
                    end
                end

                if(stage2_issue)begin
                    stage2_issues_q<=stage2_issues_q+1'b1;
                    if(!stage2_active_q)begin
                        stage2_bank_q<=stage2_selected_bank;
                        inter_valid_q[stage2_selected_bank]<=1'b0;
                    end
                    if(stage2_selected_phase==4)begin
                        stage2_active_q<=1'b0;stage2_phase_q<='0;
                        inter_reserved_q[stage2_selected_bank]<=1'b0;
                        stage2_done_q<=stage2_done_q+1'b1;
                    end else begin
                        stage2_active_q<=1'b1;
                        stage2_phase_q<=stage2_selected_phase+1'b1;
                    end
                end

                if(product_stage_ready)begin
                    product_valid_q<=stage2_issue;
                    if(stage2_issue)begin
                        product_tag_q<=stage2_tag;
                        product_beat_q<=stage2_selected_phase;
                        product_valid_bits_q<=stage2_valid_bits_comb;
                        product_data_q<=stage2_data_comb;
                    end
                end
                if(product_push)begin
                    fifo_tag_q[fifo_write_pointer_q]<=product_tag_q;
                    fifo_beat_q[fifo_write_pointer_q]<=product_beat_q;
                    fifo_valid_bits_q[fifo_write_pointer_q]<=product_valid_bits_q;
                    fifo_data_q[fifo_write_pointer_q]<=product_data_q;
                    fifo_write_pointer_q<=fifo_write_pointer_q+1'b1;
                    product_pushes_q<=product_pushes_q+1'b1;
                    if(product_beat_q==4)begin
                        tile_done_valid_q<=1'b1;tile_done_tag_q<=product_tag_q;
                    end
                end
                if(result_fire)begin
                    fifo_read_pointer_q<=fifo_read_pointer_q+1'b1;
                    result_departures_q<=result_departures_q+1'b1;
                end
                case({product_push,result_fire})
                    2'b10:fifo_count_q<=fifo_count_q+1'b1;
                    2'b01:fifo_count_q<=fifo_count_q-1'b1;
                    default:fifo_count_q<=fifo_count_q;
                endcase
                if(product_replace)replacements_q<=replacements_q+1'b1;
                if(release_accept)config_loaded_q<=1'b0;
            end
        end
    end
endmodule
`default_nettype wire
