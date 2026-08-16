// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb_h67_score_class_row_engine.h for the primary calling header

#include "Vtb_h67_score_class_row_engine__pch.h"
#include "Vtb_h67_score_class_row_engine___024root.h"

VlCoroutine Vtb_h67_score_class_row_engine___024root___eval_initial__TOP__Vtiming__0(Vtb_h67_score_class_row_engine___024root* vlSelf);
VlCoroutine Vtb_h67_score_class_row_engine___024root___eval_initial__TOP__Vtiming__1(Vtb_h67_score_class_row_engine___024root* vlSelf);
VlCoroutine Vtb_h67_score_class_row_engine___024root___eval_initial__TOP__Vtiming__2(Vtb_h67_score_class_row_engine___024root* vlSelf);

void Vtb_h67_score_class_row_engine___024root___eval_initial(Vtb_h67_score_class_row_engine___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_h67_score_class_row_engine__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_h67_score_class_row_engine___024root___eval_initial\n"); );
    // Body
    Vtb_h67_score_class_row_engine___024root___eval_initial__TOP__Vtiming__0(vlSelf);
    Vtb_h67_score_class_row_engine___024root___eval_initial__TOP__Vtiming__1(vlSelf);
    Vtb_h67_score_class_row_engine___024root___eval_initial__TOP__Vtiming__2(vlSelf);
    vlSelf->__Vtrigprevexpr___TOP__tb_h67_score_class_row_engine__DOT__clk__0 
        = vlSelf->tb_h67_score_class_row_engine__DOT__clk;
}

VL_INLINE_OPT VlCoroutine Vtb_h67_score_class_row_engine___024root___eval_initial__TOP__Vtiming__2(Vtb_h67_score_class_row_engine___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_h67_score_class_row_engine__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_h67_score_class_row_engine___024root___eval_initial__TOP__Vtiming__2\n"); );
    // Body
    while (1U) {
        co_await vlSelf->__VdlySched.delay(0x1388ULL, 
                                           nullptr, 
                                           "tb_h67/tb_h67_score_class_row_engine.sv", 
                                           98);
        vlSelf->tb_h67_score_class_row_engine__DOT__clk 
            = (1U & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__clk)));
    }
}

VL_INLINE_OPT void Vtb_h67_score_class_row_engine___024root___act_comb__TOP__0(Vtb_h67_score_class_row_engine___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_h67_score_class_row_engine__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_h67_score_class_row_engine___024root___act_comb__TOP__0\n"); );
    // Init
    IData/*31:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w;
    tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w = 0;
    CData/*5:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count = 0;
    CData/*5:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count = 0;
    CData/*5:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count = 0;
    CData/*5:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count;
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count = 0;
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
    // Body
    if (vlSelf->tb_h67_score_class_row_engine__DOT__in_time_sel) {
        tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w 
            = (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__in_k_pair_bits);
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
            = (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__in_k_pair_bits 
                       >> 0x20U));
    } else {
        tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w 
            = (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__in_k_pair_bits 
                       >> 0x20U));
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
            = (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__in_k_pair_bits);
    }
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__q_count 
        = (1U & vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits);
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__k_count 
        = (1U & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w);
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count 
        = (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                 & vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w));
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (1U & (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                 ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w));
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + (1U & ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                              ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
    tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count 
        = (0x3fU & ((IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count) 
                    + ((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w 
                        ^ tb_h67_score_class_row_engine__DOT__dut__DOT__peer_k_w) 
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
        = (0x1ffU & ((VL_SHIFTL_III(9,9,32, (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__overlap_count), 2U) 
                      + (IData)(tb_h67_score_class_row_engine__DOT__dut__DOT__u_score__DOT__motion_count)) 
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
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w = 0U;
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0U == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0U];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (1U == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [1U];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (2U == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [2U];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (3U == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [3U];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (4U == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [4U];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (5U == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [5U];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (6U == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [6U];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (7U == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [7U];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (8U == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [8U];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (9U == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [9U];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0xaU == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0xaU];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0xbU == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0xbU];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0xcU == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0xcU];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0xdU == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0xdU];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0xeU == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0xeU];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0xfU == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0xfU];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x10U == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x10U];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x11U == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x11U];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x12U == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x12U];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x13U == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x13U];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x14U == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x14U];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x15U == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x15U];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x16U == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x16U];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x17U == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x17U];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x18U == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x18U];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x19U == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x19U];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x1aU == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x1aU];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x1bU == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x1bU];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x1cU == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x1cU];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x1dU == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x1dU];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x1eU == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x1eU];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x1fU == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x1fU];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x20U == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x20U];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x21U == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x21U];
    }
    if (((2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x22U == (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x22U];
    }
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_in_range_w 
        = ((~ ((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w) 
               >> 0xfU)) & VL_GTES_III(16, 0x22U, (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w)));
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__fold_input_w 
        = ((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__enable_score_fold_q) 
           & ((~ (IData)((0U != vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w))) 
              & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_in_range_w)));
}

void Vtb_h67_score_class_row_engine___024root___eval_act(Vtb_h67_score_class_row_engine___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_h67_score_class_row_engine__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_h67_score_class_row_engine___024root___eval_act\n"); );
    // Body
    if ((3ULL & vlSelf->__VactTriggered.word(0U))) {
        Vtb_h67_score_class_row_engine___024root___act_comb__TOP__0(vlSelf);
    }
}

void Vtb_h67_score_class_row_engine___024root___nba_sequent__TOP__0(Vtb_h67_score_class_row_engine___024root* vlSelf);

void Vtb_h67_score_class_row_engine___024root___eval_nba(Vtb_h67_score_class_row_engine___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_h67_score_class_row_engine__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_h67_score_class_row_engine___024root___eval_nba\n"); );
    // Body
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vtb_h67_score_class_row_engine___024root___nba_sequent__TOP__0(vlSelf);
    }
    if ((3ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vtb_h67_score_class_row_engine___024root___act_comb__TOP__0(vlSelf);
    }
}

void Vtb_h67_score_class_row_engine___024root___timing_resume(Vtb_h67_score_class_row_engine___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_h67_score_class_row_engine__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_h67_score_class_row_engine___024root___timing_resume\n"); );
    // Body
    if ((1ULL & vlSelf->__VactTriggered.word(0U))) {
        vlSelf->__VtrigSched_h9aed8dba__0.resume("@(posedge tb_h67_score_class_row_engine.clk)");
    }
    if ((2ULL & vlSelf->__VactTriggered.word(0U))) {
        vlSelf->__VtrigSched_h9aed8d87__0.resume("@(negedge tb_h67_score_class_row_engine.clk)");
    }
    if ((4ULL & vlSelf->__VactTriggered.word(0U))) {
        vlSelf->__VdlySched.resume();
    }
}

void Vtb_h67_score_class_row_engine___024root___timing_commit(Vtb_h67_score_class_row_engine___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_h67_score_class_row_engine__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_h67_score_class_row_engine___024root___timing_commit\n"); );
    // Body
    if ((! (1ULL & vlSelf->__VactTriggered.word(0U)))) {
        vlSelf->__VtrigSched_h9aed8dba__0.commit("@(posedge tb_h67_score_class_row_engine.clk)");
    }
    if ((! (2ULL & vlSelf->__VactTriggered.word(0U)))) {
        vlSelf->__VtrigSched_h9aed8d87__0.commit("@(negedge tb_h67_score_class_row_engine.clk)");
    }
}

void Vtb_h67_score_class_row_engine___024root___eval_triggers__act(Vtb_h67_score_class_row_engine___024root* vlSelf);

bool Vtb_h67_score_class_row_engine___024root___eval_phase__act(Vtb_h67_score_class_row_engine___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_h67_score_class_row_engine__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_h67_score_class_row_engine___024root___eval_phase__act\n"); );
    // Init
    VlTriggerVec<3> __VpreTriggered;
    CData/*0:0*/ __VactExecute;
    // Body
    Vtb_h67_score_class_row_engine___024root___eval_triggers__act(vlSelf);
    Vtb_h67_score_class_row_engine___024root___timing_commit(vlSelf);
    __VactExecute = vlSelf->__VactTriggered.any();
    if (__VactExecute) {
        __VpreTriggered.andNot(vlSelf->__VactTriggered, vlSelf->__VnbaTriggered);
        vlSelf->__VnbaTriggered.thisOr(vlSelf->__VactTriggered);
        Vtb_h67_score_class_row_engine___024root___timing_resume(vlSelf);
        Vtb_h67_score_class_row_engine___024root___eval_act(vlSelf);
    }
    return (__VactExecute);
}

bool Vtb_h67_score_class_row_engine___024root___eval_phase__nba(Vtb_h67_score_class_row_engine___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_h67_score_class_row_engine__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_h67_score_class_row_engine___024root___eval_phase__nba\n"); );
    // Init
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = vlSelf->__VnbaTriggered.any();
    if (__VnbaExecute) {
        Vtb_h67_score_class_row_engine___024root___eval_nba(vlSelf);
        vlSelf->__VnbaTriggered.clear();
    }
    return (__VnbaExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_h67_score_class_row_engine___024root___dump_triggers__nba(Vtb_h67_score_class_row_engine___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_h67_score_class_row_engine___024root___dump_triggers__act(Vtb_h67_score_class_row_engine___024root* vlSelf);
#endif  // VL_DEBUG

void Vtb_h67_score_class_row_engine___024root___eval(Vtb_h67_score_class_row_engine___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_h67_score_class_row_engine__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_h67_score_class_row_engine___024root___eval\n"); );
    // Init
    vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__rst_n 
        = vlSelf->tb_h67_score_class_row_engine__DOT__rst_n;
    vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__out_last 
        = vlSelf->tb_h67_score_class_row_engine__DOT__out_last;
    vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__out_valid 
        = vlSelf->tb_h67_score_class_row_engine__DOT__out_valid;
    vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__state_q 
        = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q;
    vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__score_range_error_q 
        = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_range_error_q;
    vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT___Vpast_0_0 
        = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT___Vpast_0_0;
    vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT___Vpast_1_0 
        = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT___Vpast_1_0;
    vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w 
        = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w;
    vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__gate_w 
        = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__gate_w;
    vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__threshold_q8_q 
        = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__threshold_q8_q;
    vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__classes_remaining_q 
        = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__classes_remaining_q;
    vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
        = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q;
    vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__out_ready 
        = vlSelf->tb_h67_score_class_row_engine__DOT__out_ready;
    IData/*31:0*/ __VnbaIterCount;
    CData/*0:0*/ __VnbaContinue;
    // Body
    __VnbaIterCount = 0U;
    __VnbaContinue = 1U;
    while (__VnbaContinue) {
        if (VL_UNLIKELY((0x64U < __VnbaIterCount))) {
#ifdef VL_DEBUG
            Vtb_h67_score_class_row_engine___024root___dump_triggers__nba(vlSelf);
#endif
            VL_FATAL_MT("tb_h67/tb_h67_score_class_row_engine.sv", 4, "", "NBA region did not converge.");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        __VnbaContinue = 0U;
        vlSelf->__VactIterCount = 0U;
        vlSelf->__VactContinue = 1U;
        while (vlSelf->__VactContinue) {
            if (VL_UNLIKELY((0x64U < vlSelf->__VactIterCount))) {
#ifdef VL_DEBUG
                Vtb_h67_score_class_row_engine___024root___dump_triggers__act(vlSelf);
#endif
                VL_FATAL_MT("tb_h67/tb_h67_score_class_row_engine.sv", 4, "", "Active region did not converge.");
            }
            vlSelf->__VactIterCount = ((IData)(1U) 
                                       + vlSelf->__VactIterCount);
            vlSelf->__VactContinue = 0U;
            if (Vtb_h67_score_class_row_engine___024root___eval_phase__act(vlSelf)) {
                vlSelf->__VactContinue = 1U;
            }
        }
        if (Vtb_h67_score_class_row_engine___024root___eval_phase__nba(vlSelf)) {
            __VnbaContinue = 1U;
        }
    }
}

#ifdef VL_DEBUG
void Vtb_h67_score_class_row_engine___024root___eval_debug_assertions(Vtb_h67_score_class_row_engine___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_h67_score_class_row_engine__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_h67_score_class_row_engine___024root___eval_debug_assertions\n"); );
}
#endif  // VL_DEBUG
