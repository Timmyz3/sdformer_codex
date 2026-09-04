// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb_h67_score_class_row_engine.h for the primary calling header

#include "Vtb_h67_score_class_row_engine__pch.h"
#include "Vtb_h67_score_class_row_engine___024root.h"

VL_ATTR_COLD void Vtb_h67_score_class_row_engine___024root___eval_static(Vtb_h67_score_class_row_engine___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_h67_score_class_row_engine__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_h67_score_class_row_engine___024root___eval_static\n"); );
}

VL_ATTR_COLD void Vtb_h67_score_class_row_engine___024root___eval_final(Vtb_h67_score_class_row_engine___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_h67_score_class_row_engine__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_h67_score_class_row_engine___024root___eval_final\n"); );
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_h67_score_class_row_engine___024root___dump_triggers__stl(Vtb_h67_score_class_row_engine___024root* vlSelf);
#endif  // VL_DEBUG
VL_ATTR_COLD bool Vtb_h67_score_class_row_engine___024root___eval_phase__stl(Vtb_h67_score_class_row_engine___024root* vlSelf);

VL_ATTR_COLD void Vtb_h67_score_class_row_engine___024root___eval_settle(Vtb_h67_score_class_row_engine___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_h67_score_class_row_engine__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_h67_score_class_row_engine___024root___eval_settle\n"); );
    // Init
    IData/*31:0*/ __VstlIterCount;
    CData/*0:0*/ __VstlContinue;
    // Body
    __VstlIterCount = 0U;
    vlSelf->__VstlFirstIteration = 1U;
    __VstlContinue = 1U;
    while (__VstlContinue) {
        if (VL_UNLIKELY((0x64U < __VstlIterCount))) {
#ifdef VL_DEBUG
            Vtb_h67_score_class_row_engine___024root___dump_triggers__stl(vlSelf);
#endif
            VL_FATAL_MT("tb_h67/tb_h67_score_class_row_engine.sv", 4, "", "Settle region did not converge.");
        }
        __VstlIterCount = ((IData)(1U) + __VstlIterCount);
        __VstlContinue = 0U;
        if (Vtb_h67_score_class_row_engine___024root___eval_phase__stl(vlSelf)) {
            __VstlContinue = 1U;
        }
        vlSelf->__VstlFirstIteration = 0U;
    }
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_h67_score_class_row_engine___024root___dump_triggers__stl(Vtb_h67_score_class_row_engine___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_h67_score_class_row_engine__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_h67_score_class_row_engine___024root___dump_triggers__stl\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VstlTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VstlTriggered.word(0U))) {
        VL_DBG_MSGF("         'stl' region trigger index 0 is active: Internal 'stl' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vtb_h67_score_class_row_engine___024root___stl_sequent__TOP__0(Vtb_h67_score_class_row_engine___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_h67_score_class_row_engine__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_h67_score_class_row_engine___024root___stl_sequent__TOP__0\n"); );
    // Init
    SData/*15:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__active_delta_w;
    tb_h67_score_class_row_engine__DOT__dut__DOT__active_delta_w = 0;
    SData/*15:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__class_delta_w;
    tb_h67_score_class_row_engine__DOT__dut__DOT__class_delta_w = 0;
    CData/*2:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_addr_w;
    tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_addr_w = 0;
    CData/*3:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w;
    tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w = 0;
    CData/*5:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count = 0;
    CData/*5:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count = 0;
    CData/*5:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count = 0;
    CData/*5:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__same_zero_count;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__same_zero_count = 0;
    CData/*5:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__silence_integer;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__silence_integer = 0;
    CData/*3:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__silence_remainder;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__silence_remainder = 0;
    CData/*0:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__silence_increment;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__silence_increment = 0;
    SData/*8:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__score_integer;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__score_integer = 0;
    SData/*8:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__score_unsigned;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__score_unsigned = 0;
    SData/*15:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__abs_delta;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__abs_delta = 0;
    SData/*8:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__integer_shift;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__integer_shift = 0;
    CData/*3:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_index;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_index = 0;
    CData/*4:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_round;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_round = 0;
    SData/*15:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_value;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_value = 0;
    CData/*7:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__shift_amount;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__shift_amount = 0;
    SData/*15:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__abs_delta;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__abs_delta = 0;
    SData/*8:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__integer_shift;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__integer_shift = 0;
    CData/*3:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_index;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_index = 0;
    CData/*4:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_round;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_round = 0;
    SData/*15:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_value;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_value = 0;
    CData/*7:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__shift_amount;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__shift_amount = 0;
    CData/*5:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 0;
    CData/*3:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__token_scale_w;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__token_scale_w = 0;
    IData/*19:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__exp_token_product_w;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__exp_token_product_w = 0;
    IData/*31:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__scaled_w;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__scaled_w = 0;
    IData/*31:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__quotient_w;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__quotient_w = 0;
    IData/*31:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__remainder_w;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__remainder_w = 0;
    IData/*31:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__half_w;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__half_w = 0;
    IData/*31:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__rounded_w;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__rounded_w = 0;
    IData/*31:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe = 0;
    // Body
    vlSelf->tb_h67_score_class_row_engine__DOT__in_ready 
        = (2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q));
    vlSelf->tb_h67_score_class_row_engine__DOT__out_valid 
        = ((6U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
           & (0U != (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_count_q)));
    vlSelf->tb_h67_score_class_row_engine__DOT__out_last 
        = ((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__out_valid) 
           & ((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__emit_idx_q) 
              == (0xfU & ((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_count_q) 
                          - (IData)(1U)))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 0U;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe 
        = ((1U >= vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__row_sum_q)
            ? 1U : (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__row_sum_q 
                    - (IData)(1U)));
    if ((1U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 1U;
    }
    if ((2U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 2U;
    }
    if ((4U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 3U;
    }
    if ((8U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 4U;
    }
    if ((0x10U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 5U;
    }
    if ((0x20U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 6U;
    }
    if ((0x40U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 7U;
    }
    if ((0x80U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 8U;
    }
    if ((0x100U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 9U;
    }
    if ((0x200U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 0xaU;
    }
    if ((0x400U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 0xbU;
    }
    if ((0x800U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 0xcU;
    }
    if ((0x1000U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 0xdU;
    }
    if ((0x2000U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 0xeU;
    }
    if ((0x4000U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 0xfU;
    }
    if ((0x8000U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 0x10U;
    }
    if ((0x10000U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 0x11U;
    }
    if ((0x20000U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 0x12U;
    }
    if ((0x40000U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 0x13U;
    }
    if ((0x80000U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 0x14U;
    }
    if ((0x100000U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 0x15U;
    }
    if ((0x200000U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 0x16U;
    }
    if ((0x400000U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 0x17U;
    }
    if ((0x800000U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 0x18U;
    }
    if ((0x1000000U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 0x19U;
    }
    if ((0x2000000U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 0x1aU;
    }
    if ((0x4000000U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 0x1bU;
    }
    if ((0x8000000U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 0x1cU;
    }
    if ((0x10000000U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 0x1dU;
    }
    if ((0x20000000U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 0x1eU;
    }
    if ((0x40000000U & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 0x1fU;
    }
    if ((tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__u_normalization_shift__DOT__probe 
         >> 0x1fU)) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w = 0x20U;
    }
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 0U;
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & ((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q) 
            >> 1U))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 1U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & ((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q) 
            >> 2U))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 2U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w = 0U;
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0U];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (1U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [1U];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [2U];
    }
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
        = ((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__in_time_sel)
            ? (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__in_k_pair_bits 
                       >> 0x20U)) : (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__in_k_pair_bits));
    tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_addr_w 
        = (7U & ((6U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))
                  ? (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__emit_idx_q)
                  : (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__scan_idx_q)));
    tb_h67_score_class_row_engine__DOT__dut__DOT__class_delta_w 
        = (((5U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
            | (4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)))
            ? (0xffffU & ((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w) 
                          - (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__row_max_q)))
            : 0U);
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (1U & vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits);
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (1U & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w);
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                 & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 1U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 1U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 1U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 2U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 2U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 2U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 3U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 3U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 3U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 4U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 4U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 4U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 5U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 5U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 5U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 6U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 6U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 6U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 7U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 7U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 7U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 8U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 8U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 8U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 9U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 9U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 9U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 0xaU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 0xaU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 0xaU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 0xbU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 0xbU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 0xbU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 0xcU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 0xcU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 0xcU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 0xdU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 0xdU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 0xdU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 0xeU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 0xeU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 0xeU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 0xfU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 0xfU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 0xfU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 0x10U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 0x10U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 0x10U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 0x11U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 0x11U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 0x11U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 0x12U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 0x12U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 0x12U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 0x13U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 0x13U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 0x13U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 0x14U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 0x14U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 0x14U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 0x15U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 0x15U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 0x15U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 0x16U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 0x16U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 0x16U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 0x17U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 0x17U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 0x17U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 0x18U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 0x18U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 0x18U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 0x19U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 0x19U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 0x19U))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 0x1aU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 0x1aU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 0x1aU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 0x1bU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 0x1bU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 0x1bU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 0x1cU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 0x1cU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 0x1cU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 0x1dU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 0x1dU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 0x1dU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                             >> 0x1eU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                             >> 0x1eU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                              & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                             >> 0x1eU))));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count) 
                    + (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                       >> 0x1fU)));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count) 
                    + (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                       >> 0x1fU)));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count) 
                    + ((vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                        & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w) 
                       >> 0x1fU)));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__same_zero_count 
        = (0x3fU & ((((IData)(0x20U) - (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count)) 
                     - (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count)) 
                    + (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count)));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__silence_integer 
        = (0x3fU & VL_SHIFTR_III(6,6,32, (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__same_zero_count), 4U));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__silence_remainder 
        = (0xfU & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__same_zero_count));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__score_integer 
        = (0x1ffU & (VL_SHIFTL_III(9,9,32, (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count), 2U) 
                     + (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__silence_integer)));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__silence_increment 
        = ((8U < (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__silence_remainder)) 
           | ((8U == (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__silence_remainder)) 
              & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__score_integer)));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__score_unsigned 
        = (0x1ffU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__score_integer) 
                     + (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__silence_increment)));
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w 
        = tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__score_unsigned;
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w = 0ULL;
    if ((((3U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          | (6U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))) 
         & (0U == (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_addr_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_entry_mem
            [0U];
    }
    if ((((3U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          | (6U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))) 
         & (1U == (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_addr_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_entry_mem
            [1U];
    }
    if ((((3U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          | (6U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))) 
         & (2U == (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_addr_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_entry_mem
            [2U];
    }
    if ((((3U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          | (6U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))) 
         & (3U == (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_addr_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_entry_mem
            [3U];
    }
    if ((((3U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          | (6U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))) 
         & (4U == (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_addr_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_entry_mem
            [4U];
    }
    if ((((3U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          | (6U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))) 
         & (5U == (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_addr_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_entry_mem
            [5U];
    }
    if ((((3U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          | (6U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))) 
         & (6U == (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_addr_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_entry_mem
            [6U];
    }
    if ((((3U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          | (6U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))) 
         & (7U == (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_addr_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_entry_mem
            [7U];
    }
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__abs_delta = 0U;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__integer_shift = 0U;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_index = 0U;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_round = 0U;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_value = 0x100U;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__shift_amount = 0U;
    if (VL_LTES_III(32, 0U, VL_EXTENDS_II(32,16, (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__class_delta_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_exp_w = 0x100U;
    } else {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__abs_delta 
            = (0xffffU & (- (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__class_delta_w)));
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__integer_shift 
            = (0x1ffU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__abs_delta) 
                         >> 7U));
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_round 
            = (0x1fU & ((0xfU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__abs_delta) 
                                 >> 3U)) + (0U != (7U 
                                                   & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__abs_delta)))));
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_index 
            = ((0x10U & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_round))
                ? 0xfU : (0xfU & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_round)));
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_value 
            = ((8U & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_index))
                ? ((4U & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_index))
                    ? ((2U & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_index))
                        ? ((1U & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_index))
                            ? 0x85U : 0x8bU) : ((1U 
                                                 & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_index))
                                                 ? 0x91U
                                                 : 0x98U))
                    : ((2U & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_index))
                        ? ((1U & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_index))
                            ? 0x9eU : 0xa5U) : ((1U 
                                                 & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_index))
                                                 ? 0xadU
                                                 : 0xb5U)))
                : ((4U & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_index))
                    ? ((2U & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_index))
                        ? ((1U & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_index))
                            ? 0xbcU : 0xc4U) : ((1U 
                                                 & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_index))
                                                 ? 0xcdU
                                                 : 0xd7U))
                    : ((2U & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_index))
                        ? ((1U & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_index))
                            ? 0xe0U : 0xeaU) : ((1U 
                                                 & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_index))
                                                 ? 0xf5U
                                                 : 0x100U))));
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__shift_amount 
            = ((8U < (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__integer_shift))
                ? 8U : (0xffU & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__integer_shift)));
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_exp_w 
            = (0xffffU & VL_SHIFTR_III(16,16,8, (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__fraction_value), (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_class_exp__DOT__shift_amount)));
    }
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w = 0U;
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0U == (3U & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0U];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (1U == (3U & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [1U];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (2U == (3U & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [2U];
    }
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_in_range_w 
        = ((~ ((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w) 
               >> 0xfU)) & VL_GTES_III(16, 2U, (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w)));
    tb_h67_score_class_row_engine__DOT__dut__DOT__active_delta_w 
        = (((3U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
            | (6U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)))
            ? (0xffffU & ((IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w 
                                   >> 0x24U)) - (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__row_max_q)))
            : 0U);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_sum_term_w 
        = ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w) 
           * (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_exp_w));
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__fold_input_w 
        = ((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__enable_score_fold_q) 
           & ((~ (IData)((0U != vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w))) 
              & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_in_range_w)));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__abs_delta = 0U;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__integer_shift = 0U;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_index = 0U;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_round = 0U;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_value = 0x100U;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__shift_amount = 0U;
    if (VL_LTES_III(32, 0U, VL_EXTENDS_II(32,16, (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__active_delta_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_exp_w = 0x100U;
    } else {
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__abs_delta 
            = (0xffffU & (- (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__active_delta_w)));
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__integer_shift 
            = (0x1ffU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__abs_delta) 
                         >> 7U));
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_round 
            = (0x1fU & ((0xfU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__abs_delta) 
                                 >> 3U)) + (0U != (7U 
                                                   & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__abs_delta)))));
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_index 
            = ((0x10U & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_round))
                ? 0xfU : (0xfU & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_round)));
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_value 
            = ((8U & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_index))
                ? ((4U & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_index))
                    ? ((2U & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_index))
                        ? ((1U & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_index))
                            ? 0x85U : 0x8bU) : ((1U 
                                                 & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_index))
                                                 ? 0x91U
                                                 : 0x98U))
                    : ((2U & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_index))
                        ? ((1U & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_index))
                            ? 0x9eU : 0xa5U) : ((1U 
                                                 & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_index))
                                                 ? 0xadU
                                                 : 0xb5U)))
                : ((4U & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_index))
                    ? ((2U & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_index))
                        ? ((1U & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_index))
                            ? 0xbcU : 0xc4U) : ((1U 
                                                 & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_index))
                                                 ? 0xcdU
                                                 : 0xd7U))
                    : ((2U & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_index))
                        ? ((1U & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_index))
                            ? 0xe0U : 0xeaU) : ((1U 
                                                 & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_index))
                                                 ? 0xf5U
                                                 : 0x100U))));
        tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__shift_amount 
            = ((8U < (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__integer_shift))
                ? 8U : (0xffU & (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__integer_shift)));
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_exp_w 
            = (0xffffU & VL_SHIFTR_III(16,16,8, (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__fraction_value), (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_active_exp__DOT__shift_amount)));
    }
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__token_scale_w 
        = ((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__preserve_mean_q)
            ? (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__n_tokens_q)
            : 1U);
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__exp_token_product_w 
        = (0xfffffU & ((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_exp_w) 
                       * (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__token_scale_w)));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__scaled_w 
        = VL_SHIFTL_III(32,32,32, tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__exp_token_product_w, 7U);
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__quotient_w = 0U;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__remainder_w = 0U;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__half_w = 0U;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__rounded_w = 0U;
    if ((0U != vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__row_sum_q)) {
        if ((0U == (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w))) {
            tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__rounded_w 
                = tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__scaled_w;
        } else {
            tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__quotient_w 
                = VL_SHIFTR_III(32,32,6, tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__scaled_w, (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w));
            tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__remainder_w 
                = (tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__scaled_w 
                   - VL_SHIFTL_III(32,32,6, tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__quotient_w, (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w)));
            tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__half_w 
                = VL_SHIFTL_III(32,32,6, (IData)(1U), 
                                (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__denominator_shift_w) 
                                          - (IData)(1U))));
            tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__rounded_w 
                = tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__quotient_w;
            if (((tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__remainder_w 
                  > tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__half_w) 
                 | ((tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__remainder_w 
                     == tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__half_w) 
                    & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__quotient_w))) {
                tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__rounded_w 
                    = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__quotient_w);
            }
        }
    }
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__gate_w 
        = ((0x100U < tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__rounded_w)
            ? 0x100U : (0x1ffU & tb_h67_score_class_row_engine__DOT__dut__DOT__u_gate_quant__DOT__rounded_w));
}

VL_ATTR_COLD void Vtb_h67_score_class_row_engine___024root___eval_stl(Vtb_h67_score_class_row_engine___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_h67_score_class_row_engine__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_h67_score_class_row_engine___024root___eval_stl\n"); );
    // Body
    if ((1ULL & vlSelf->__VstlTriggered.word(0U))) {
        Vtb_h67_score_class_row_engine___024root___stl_sequent__TOP__0(vlSelf);
    }
}

VL_ATTR_COLD void Vtb_h67_score_class_row_engine___024root___eval_triggers__stl(Vtb_h67_score_class_row_engine___024root* vlSelf);

VL_ATTR_COLD bool Vtb_h67_score_class_row_engine___024root___eval_phase__stl(Vtb_h67_score_class_row_engine___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_h67_score_class_row_engine__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_h67_score_class_row_engine___024root___eval_phase__stl\n"); );
    // Init
    CData/*0:0*/ __VstlExecute;
    // Body
    Vtb_h67_score_class_row_engine___024root___eval_triggers__stl(vlSelf);
    __VstlExecute = vlSelf->__VstlTriggered.any();
    if (__VstlExecute) {
        Vtb_h67_score_class_row_engine___024root___eval_stl(vlSelf);
    }
    return (__VstlExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_h67_score_class_row_engine___024root___dump_triggers__act(Vtb_h67_score_class_row_engine___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_h67_score_class_row_engine__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_h67_score_class_row_engine___024root___dump_triggers__act\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VactTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 0 is active: @(posedge tb_h67_score_class_row_engine.clk)\n");
    }
    if ((2ULL & vlSelf->__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 1 is active: @(negedge tb_h67_score_class_row_engine.clk)\n");
    }
    if ((4ULL & vlSelf->__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 2 is active: @([true] __VdlySched.awaitingCurrentTime())\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_h67_score_class_row_engine___024root___dump_triggers__nba(Vtb_h67_score_class_row_engine___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_h67_score_class_row_engine__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_h67_score_class_row_engine___024root___dump_triggers__nba\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VnbaTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 0 is active: @(posedge tb_h67_score_class_row_engine.clk)\n");
    }
    if ((2ULL & vlSelf->__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 1 is active: @(negedge tb_h67_score_class_row_engine.clk)\n");
    }
    if ((4ULL & vlSelf->__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 2 is active: @([true] __VdlySched.awaitingCurrentTime())\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vtb_h67_score_class_row_engine___024root___ctor_var_reset(Vtb_h67_score_class_row_engine___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_h67_score_class_row_engine__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_h67_score_class_row_engine___024root___ctor_var_reset\n"); );
    // Body
    vlSelf->tb_h67_score_class_row_engine__DOT__clk = VL_RAND_RESET_I(1);
    vlSelf->tb_h67_score_class_row_engine__DOT__rst_n = VL_RAND_RESET_I(1);
    vlSelf->tb_h67_score_class_row_engine__DOT__cfg_start = VL_RAND_RESET_I(1);
    vlSelf->tb_h67_score_class_row_engine__DOT__cfg_n_tokens = VL_RAND_RESET_I(4);
    vlSelf->tb_h67_score_class_row_engine__DOT__cfg_preserve_mean = VL_RAND_RESET_I(1);
    vlSelf->tb_h67_score_class_row_engine__DOT__cfg_enable_score_fold = VL_RAND_RESET_I(1);
    vlSelf->tb_h67_score_class_row_engine__DOT__cfg_threshold_q8 = VL_RAND_RESET_I(8);
    vlSelf->tb_h67_score_class_row_engine__DOT__in_valid = VL_RAND_RESET_I(1);
    vlSelf->tb_h67_score_class_row_engine__DOT__in_ready = VL_RAND_RESET_I(1);
    vlSelf->tb_h67_score_class_row_engine__DOT__in_last = VL_RAND_RESET_I(1);
    vlSelf->tb_h67_score_class_row_engine__DOT__in_time_sel = VL_RAND_RESET_I(1);
    vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits = VL_RAND_RESET_I(32);
    vlSelf->tb_h67_score_class_row_engine__DOT__in_k_pair_bits = VL_RAND_RESET_Q(64);
    vlSelf->tb_h67_score_class_row_engine__DOT__out_valid = VL_RAND_RESET_I(1);
    vlSelf->tb_h67_score_class_row_engine__DOT__out_ready = VL_RAND_RESET_I(1);
    vlSelf->tb_h67_score_class_row_engine__DOT__out_last = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        vlSelf->tb_h67_score_class_row_engine__DOT__q_vector[__Vi0] = VL_RAND_RESET_I(32);
    }
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector[__Vi0] = VL_RAND_RESET_I(32);
    }
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector[__Vi0] = VL_RAND_RESET_I(32);
    }
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        vlSelf->tb_h67_score_class_row_engine__DOT__time_vector[__Vi0] = VL_RAND_RESET_I(1);
    }
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        vlSelf->tb_h67_score_class_row_engine__DOT__expected_score[__Vi0] = VL_RAND_RESET_I(32);
    }
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        vlSelf->tb_h67_score_class_row_engine__DOT__expected_gate[__Vi0] = VL_RAND_RESET_I(32);
    }
    vlSelf->tb_h67_score_class_row_engine__DOT__token_idx = VL_RAND_RESET_I(32);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q = VL_RAND_RESET_I(3);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__n_tokens_q = VL_RAND_RESET_I(4);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__preserve_mean_q = VL_RAND_RESET_I(1);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__enable_score_fold_q = VL_RAND_RESET_I(1);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__threshold_q8_q = VL_RAND_RESET_I(8);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__load_idx_q = VL_RAND_RESET_I(4);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_count_q = VL_RAND_RESET_I(4);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__scan_idx_q = VL_RAND_RESET_I(4);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__emit_idx_q = VL_RAND_RESET_I(4);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__clear_all_idx_q = VL_RAND_RESET_I(2);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_initialized_q = VL_RAND_RESET_I(1);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q = VL_RAND_RESET_I(3);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__classes_remaining_q = VL_RAND_RESET_I(2);
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_entry_mem[__Vi0] = VL_RAND_RESET_Q(52);
    }
    for (int __Vi0 = 0; __Vi0 < 3; ++__Vi0) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist[__Vi0] = VL_RAND_RESET_I(4);
    }
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__row_max_q = VL_RAND_RESET_I(16);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__row_sum_q = VL_RAND_RESET_I(32);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__tokens_loaded_q = VL_RAND_RESET_I(4);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__folded_tokens_q = VL_RAND_RESET_I(4);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__emitted_entries_q = VL_RAND_RESET_I(4);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__fold_classes_q = VL_RAND_RESET_I(2);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__exp_transactions_q = VL_RAND_RESET_I(16);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_range_error_q = VL_RAND_RESET_I(1);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w = VL_RAND_RESET_I(32);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w = VL_RAND_RESET_I(16);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_in_range_w = VL_RAND_RESET_I(1);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__fold_input_w = VL_RAND_RESET_I(1);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_exp_w = VL_RAND_RESET_I(16);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = VL_RAND_RESET_I(2);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = VL_RAND_RESET_I(1);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_exp_w = VL_RAND_RESET_I(16);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_sum_term_w = VL_RAND_RESET_I(32);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__gate_w = VL_RAND_RESET_I(9);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w = VL_RAND_RESET_Q(52);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w = VL_RAND_RESET_I(4);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT____Vlvbound_he2560073__0 = VL_RAND_RESET_I(4);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT____Vlvbound_he3f1879c__0 = VL_RAND_RESET_I(4);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT____Vlvbound_h92d13a42__0 = VL_RAND_RESET_I(1);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT____Vlvbound_h7c73bd9a__0 = VL_RAND_RESET_I(4);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT____Vlvbound_h73d2f73b__0 = VL_RAND_RESET_I(1);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT___Vpast_0_0 = VL_RAND_RESET_I(1);
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT___Vpast_1_0 = VL_RAND_RESET_Q(54);
    vlSelf->__Vtrigprevexpr___TOP__tb_h67_score_class_row_engine__DOT__clk__0 = VL_RAND_RESET_I(1);
}
