`timescale 1ns/1ps
`default_nettype none
module tb_m235_dynamic_bn_segmented_lut_newton_coefficient_engine;
    localparam int TAG_BITS=24;
    logic clk_core=0,rst_core;always #1.5 clk_core=~clk_core;
    logic request_valid,request_ready,request_accept;
    logic[TAG_BITS-1:0]request_tag;
    logic[21:0]variance_plus_epsilon_uq6p16;
    logic signed[17:0]mean_sq3p14;
    logic signed[15:0]gamma_sq1p14,beta_sq1p14;
    logic result_valid,result_ready,result_accept;
    logic[TAG_BITS-1:0]result_tag;logic[19:0]invstd_uq4p16;
    logic signed[19:0]alpha_sq3p16,offset_sq3p16;
    logic protocol_error,busy;logic[3:0]debug_state;
    logic[31:0]debug_request_count,debug_result_count;
    integer cycle_count,vectors_checked,mismatches,max_latency,max_accept_ii;
    integer result_stall_cycles,attacks,last_accept_cycle;
    integer expected_invstd,expected_alpha,expected_offset,expected_tag;

    m235_dynamic_bn_segmented_lut_newton_coefficient_engine dut(.*);
    m235_dynamic_bn_segmented_lut_newton_coefficient_engine_assertions sva(.*);

    always @(posedge clk_core)begin
        if(rst_core)cycle_count=0;
        else begin
            cycle_count++;
            if(result_valid&&!result_ready)result_stall_cycles++;
            if(result_accept)begin
                if(result_tag!==expected_tag[TAG_BITS-1:0]
                        ||invstd_uq4p16!==expected_invstd[19:0]
                        ||alpha_sq3p16!==expected_alpha[19:0]
                        ||offset_sq3p16!==expected_offset[19:0])begin
                    mismatches++;
                    $fatal(1,"M235 mismatch tag=%0d got=%0d inv=%0d/%0d alpha=%0d/%0d offset=%0d/%0d",
                        expected_tag,result_tag,expected_invstd,invstd_uq4p16,
                        expected_alpha,$signed(alpha_sq3p16),expected_offset,
                        $signed(offset_sq3p16));
                end
                vectors_checked++;
            end
        end
    end

    task automatic reset;begin
        @(negedge clk_core);rst_core=1;request_valid=0;result_ready=0;
        repeat(4)@(posedge clk_core);@(negedge clk_core);rst_core=0;
        result_ready=1;
    end endtask

    task automatic run_vector(input integer vector_id,input integer variance_q,
            input integer mean_q,input integer gamma_q,input integer beta_q,
            input integer inv_q,input integer alpha_q,input integer offset_q,
            input bit stall_result);
        integer accept_cycle,latency;
        begin
            expected_tag=vector_id+1;expected_invstd=inv_q;
            expected_alpha=alpha_q;expected_offset=offset_q;
            @(negedge clk_core);request_tag=expected_tag;
            variance_plus_epsilon_uq6p16=variance_q;
            mean_sq3p14=mean_q;gamma_sq1p14=gamma_q;beta_sq1p14=beta_q;
            request_valid=1;
            do @(posedge clk_core);while(!request_accept);
            accept_cycle=cycle_count;
            if(last_accept_cycle>=0&&accept_cycle-last_accept_cycle>max_accept_ii)
                max_accept_ii=accept_cycle-last_accept_cycle;
            last_accept_cycle=accept_cycle;
            @(negedge clk_core);request_valid=0;
            if(stall_result)result_ready=0;
            wait(result_valid);latency=cycle_count-accept_cycle;
            if(latency>max_latency)max_latency=latency;
            if(stall_result)begin
                repeat(5)@(posedge clk_core);
                @(negedge clk_core);result_ready=1;
            end
            do @(posedge clk_core);while(!result_accept);
            // Leave the global expected payload unchanged until the posedge
            // scoreboard has observed the accepted result.
            @(negedge clk_core);
        end
    endtask

    task automatic fault_with_pending_result(input integer variance_q,
            input integer mean_q,input integer gamma_q,input integer beta_q,
            input integer inv_q,input integer alpha_q,input integer offset_q);
        integer requests_before,results_before;
        begin
            expected_tag=24'h235bad;expected_invstd=inv_q;
            expected_alpha=alpha_q;expected_offset=offset_q;
            @(negedge clk_core);request_tag=expected_tag;
            variance_plus_epsilon_uq6p16=variance_q;
            mean_sq3p14=mean_q;gamma_sq1p14=gamma_q;beta_sq1p14=beta_q;
            request_valid=1;result_ready=0;
            do @(posedge clk_core);while(!request_accept);
            @(negedge clk_core);request_valid=0;
            wait(result_valid);requests_before=debug_request_count;
            results_before=debug_result_count;
            @(negedge clk_core);request_valid=1;
            variance_plus_epsilon_uq6p16=0;result_ready=1;#0.1;
            if(!protocol_error||request_accept||result_accept||request_ready
                    ||result_valid)$fatal(1,"M235 fault-cycle atomicity failed");
            @(posedge clk_core);#0.2;
            if(debug_request_count!=requests_before
                    ||debug_result_count!=results_before||!protocol_error)
                $fatal(1,"M235 fault cycle state commit");
            attacks++;
            @(negedge clk_core);request_valid=0;
        end
    endtask

    initial begin #2000000;$fatal(1,"M235 watchdog");end
    initial begin
        integer file_descriptor,scan_status,vector_id,flat_index;
        integer variance_q,mean_q,gamma_q,beta_q,even_exp,mantissa_q,lut_index;
        integer inv_q,alpha_q,offset_q;real ref_inv,ref_alpha,ref_offset,error_bound;
        reg[4095:0]line;
        rst_core=1;request_valid=0;result_ready=0;request_tag=0;
        variance_plus_epsilon_uq6p16=0;mean_sq3p14=0;
        gamma_sq1p14=0;beta_sq1p14=0;cycle_count=0;vectors_checked=0;
        mismatches=0;max_latency=0;max_accept_ii=0;result_stall_cycles=0;
        attacks=0;last_accept_cycle=-1;
        repeat(4)@(posedge clk_core);@(negedge clk_core);rst_core=0;result_ready=1;
        file_descriptor=$fopen("results/m234_h67_dynamic_bn_lut_newton_coefficient_dse_r1_20260825/m234_selected_coefficient_vectors.csv","r");
        if(file_descriptor==0)$fatal(1,"M235 vector file missing");
        scan_status=$fgets(line,file_descriptor);
        while(!$feof(file_descriptor))begin
            scan_status=$fscanf(file_descriptor,
                "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%f,%f,%f,%f\n",
                vector_id,flat_index,variance_q,mean_q,gamma_q,beta_q,
                even_exp,mantissa_q,lut_index,inv_q,alpha_q,offset_q,
                ref_inv,ref_alpha,ref_offset,error_bound);
            if(scan_status==16)
                run_vector(vector_id,variance_q,mean_q,gamma_q,beta_q,
                    inv_q,alpha_q,offset_q,vector_id==511);
            else if(!$feof(file_descriptor))$fatal(1,"M235 vector parse failed %0d",scan_status);
        end
        $fclose(file_descriptor);
        if(vectors_checked!=1024||mismatches!=0||max_latency>16
                ||max_accept_ii>16||result_stall_cycles<5)
            $fatal(1,"M235 coverage/count failure vectors=%0d mismatch=%0d lat=%0d ii=%0d stall=%0d",
                vectors_checked,mismatches,max_latency,max_accept_ii,result_stall_cycles);
        fault_with_pending_result(9517,-149,16707,-123,171973,175363,1103);
        if(attacks!=1)$fatal(1,"M235 protocol attack missing");
        $display("PASS M235 checkpoint vectors=1024 mismatches=0 max_latency=%0d max_accept_ii=%0d result_stalls=%0d protocol_attacks=1 shared_multiplier_slots=1 lut_entries=64 moment_finalizer=false event_equivalence=false system_speedup=false headline=false",
            max_latency,max_accept_ii,result_stall_cycles);
        $finish;
    end
endmodule
`default_nettype wire
