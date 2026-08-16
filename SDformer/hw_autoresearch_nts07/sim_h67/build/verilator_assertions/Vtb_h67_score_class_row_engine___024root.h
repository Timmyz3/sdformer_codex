// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vtb_h67_score_class_row_engine.h for the primary calling header

#ifndef VERILATED_VTB_H67_SCORE_CLASS_ROW_ENGINE___024ROOT_H_
#define VERILATED_VTB_H67_SCORE_CLASS_ROW_ENGINE___024ROOT_H_  // guard

#include "verilated.h"
#include "verilated_timing.h"


class Vtb_h67_score_class_row_engine__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vtb_h67_score_class_row_engine___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    // Anonymous structures to workaround compiler member-count bugs
    struct {
        CData/*0:0*/ tb_h67_score_class_row_engine__DOT__clk;
        CData/*0:0*/ tb_h67_score_class_row_engine__DOT__rst_n;
        CData/*0:0*/ tb_h67_score_class_row_engine__DOT__cfg_start;
        CData/*3:0*/ tb_h67_score_class_row_engine__DOT__cfg_n_tokens;
        CData/*0:0*/ tb_h67_score_class_row_engine__DOT__cfg_preserve_mean;
        CData/*0:0*/ tb_h67_score_class_row_engine__DOT__cfg_enable_score_fold;
        CData/*7:0*/ tb_h67_score_class_row_engine__DOT__cfg_threshold_q8;
        CData/*0:0*/ tb_h67_score_class_row_engine__DOT__in_valid;
        CData/*0:0*/ tb_h67_score_class_row_engine__DOT__in_ready;
        CData/*0:0*/ tb_h67_score_class_row_engine__DOT__in_last;
        CData/*0:0*/ tb_h67_score_class_row_engine__DOT__in_time_sel;
        CData/*0:0*/ tb_h67_score_class_row_engine__DOT__out_valid;
        CData/*0:0*/ tb_h67_score_class_row_engine__DOT__out_ready;
        CData/*0:0*/ tb_h67_score_class_row_engine__DOT__out_last;
        CData/*2:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__state_q;
        CData/*3:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__n_tokens_q;
        CData/*0:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__preserve_mean_q;
        CData/*0:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__enable_score_fold_q;
        CData/*7:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__threshold_q8_q;
        CData/*3:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__load_idx_q;
        CData/*3:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__active_count_q;
        CData/*3:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__scan_idx_q;
        CData/*3:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__emit_idx_q;
        CData/*5:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__clear_all_idx_q;
        CData/*0:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__hist_initialized_q;
        CData/*5:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__classes_remaining_q;
        CData/*3:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__tokens_loaded_q;
        CData/*3:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__folded_tokens_q;
        CData/*3:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__emitted_entries_q;
        CData/*5:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__fold_classes_q;
        CData/*0:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__score_range_error_q;
        CData/*0:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_in_range_w;
        CData/*0:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__fold_input_w;
        CData/*5:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w;
        CData/*0:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w;
        CData/*5:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_q;
        CData/*3:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__class_hist_count_q;
        CData/*3:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w;
        CData/*3:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w;
        CData/*3:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT____Vlvbound_h68cafb07__0;
        CData/*3:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT____Vlvbound_h6b3d7c2f__0;
        CData/*0:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT____Vlvbound_h60725e1c__0;
        CData/*3:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT____Vlvbound_h0c4876d2__0;
        CData/*0:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT____Vlvbound_h8b805d45__0;
        CData/*0:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT___Vpast_0_0;
        CData/*0:0*/ __VstlFirstIteration;
        CData/*0:0*/ __Vtrigprevexpr___TOP__tb_h67_score_class_row_engine__DOT__clk__0;
        CData/*0:0*/ __VactContinue;
        CData/*0:0*/ __Vsampled__TOP__tb_h67_score_class_row_engine__DOT__rst_n;
        CData/*0:0*/ __Vsampled__TOP__tb_h67_score_class_row_engine__DOT__out_last;
        CData/*0:0*/ __Vsampled__TOP__tb_h67_score_class_row_engine__DOT__out_valid;
        CData/*2:0*/ __Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__state_q;
        CData/*0:0*/ __Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__score_range_error_q;
        CData/*0:0*/ __Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT___Vpast_0_0;
        CData/*7:0*/ __Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__threshold_q8_q;
        CData/*5:0*/ __Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__classes_remaining_q;
        CData/*0:0*/ __Vsampled__TOP__tb_h67_score_class_row_engine__DOT__out_ready;
        SData/*15:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__row_max_q;
        SData/*15:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__exp_transactions_q;
        SData/*15:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w;
        SData/*15:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__active_exp_w;
        SData/*15:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__class_exp_w;
        SData/*8:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__gate_w;
        SData/*8:0*/ __Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__gate_w;
    };
    struct {
        IData/*31:0*/ tb_h67_score_class_row_engine__DOT__in_q_bits;
        IData/*31:0*/ tb_h67_score_class_row_engine__DOT__token_idx;
        IData/*31:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__row_sum_q;
        IData/*31:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w;
        IData/*31:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__class_sum_term_w;
        IData/*31:0*/ __VactIterCount;
        QData/*63:0*/ tb_h67_score_class_row_engine__DOT__in_k_pair_bits;
        QData/*34:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q;
        QData/*51:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w;
        QData/*53:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT___Vpast_1_0;
        QData/*53:0*/ __Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT___Vpast_1_0;
        QData/*51:0*/ __Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w;
        QData/*34:0*/ __Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q;
        VlUnpacked<IData/*31:0*/, 8> tb_h67_score_class_row_engine__DOT__q_vector;
        VlUnpacked<IData/*31:0*/, 8> tb_h67_score_class_row_engine__DOT__k_current_vector;
        VlUnpacked<IData/*31:0*/, 8> tb_h67_score_class_row_engine__DOT__k_peer_vector;
        VlUnpacked<CData/*0:0*/, 8> tb_h67_score_class_row_engine__DOT__time_vector;
        VlUnpacked<IData/*31:0*/, 8> tb_h67_score_class_row_engine__DOT__expected_score;
        VlUnpacked<IData/*31:0*/, 8> tb_h67_score_class_row_engine__DOT__expected_gate;
        VlUnpacked<QData/*51:0*/, 8> tb_h67_score_class_row_engine__DOT__dut__DOT__active_entry_mem;
        VlUnpacked<CData/*3:0*/, 35> tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist;
    };
    VlDelayScheduler __VdlySched;
    VlTriggerScheduler __VtrigSched_h9aed8dba__0;
    VlTriggerScheduler __VtrigSched_h9aed8d87__0;
    VlTriggerVec<1> __VstlTriggered;
    VlTriggerVec<3> __VactTriggered;
    VlTriggerVec<3> __VnbaTriggered;

    // INTERNAL VARIABLES
    Vtb_h67_score_class_row_engine__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vtb_h67_score_class_row_engine___024root(Vtb_h67_score_class_row_engine__Syms* symsp, const char* v__name);
    ~Vtb_h67_score_class_row_engine___024root();
    VL_UNCOPYABLE(Vtb_h67_score_class_row_engine___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
