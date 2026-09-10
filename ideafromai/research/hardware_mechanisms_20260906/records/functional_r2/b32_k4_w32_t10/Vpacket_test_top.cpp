// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vpacket_test_top.h for the primary calling header

#include "Vpacket_test_top.h"
#include "Vpacket_test_top__Syms.h"

//==========

VL_CTOR_IMP(Vpacket_test_top) {
    Vpacket_test_top__Syms* __restrict vlSymsp = __VlSymsp = new Vpacket_test_top__Syms(this, name());
    Vpacket_test_top* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Reset internal values
    
    // Reset structure values
    _ctor_var_reset();
}

void Vpacket_test_top::__Vconfigure(Vpacket_test_top__Syms* vlSymsp, bool first) {
    if (0 && first) {}  // Prevent unused
    this->__VlSymsp = vlSymsp;
}

Vpacket_test_top::~Vpacket_test_top() {
    delete __VlSymsp; __VlSymsp=NULL;
}

void Vpacket_test_top::eval() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vpacket_test_top::eval\n"); );
    Vpacket_test_top__Syms* __restrict vlSymsp = this->__VlSymsp;  // Setup global symbol table
    Vpacket_test_top* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
#ifdef VL_DEBUG
    // Debug assertions
    _eval_debug_assertions();
#endif  // VL_DEBUG
    // Initialize
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) _eval_initial_loop(vlSymsp);
    // Evaluate till stable
    int __VclockLoop = 0;
    QData __Vchange = 1;
    do {
        VL_DEBUG_IF(VL_DBG_MSGF("+ Clock loop\n"););
        _eval(vlSymsp);
        if (VL_UNLIKELY(++__VclockLoop > 100)) {
            // About to fail, so enable debug to see what's not settling.
            // Note you must run make with OPT=-DVL_DEBUG for debug prints.
            int __Vsaved_debug = Verilated::debug();
            Verilated::debug(1);
            __Vchange = _change_request(vlSymsp);
            Verilated::debug(__Vsaved_debug);
            VL_FATAL_MT("/home/zhumd/work/ideafromai/research/hardware_mechanisms_20260906/sim/packet_test_top.sv", 3, "",
                "Verilated model didn't converge\n"
                "- See DIDNOTCONVERGE in the Verilator manual");
        } else {
            __Vchange = _change_request(vlSymsp);
        }
    } while (VL_UNLIKELY(__Vchange));
}

void Vpacket_test_top::_eval_initial_loop(Vpacket_test_top__Syms* __restrict vlSymsp) {
    vlSymsp->__Vm_didInit = true;
    _eval_initial(vlSymsp);
    // Evaluate till stable
    int __VclockLoop = 0;
    QData __Vchange = 1;
    do {
        _eval_settle(vlSymsp);
        _eval(vlSymsp);
        if (VL_UNLIKELY(++__VclockLoop > 100)) {
            // About to fail, so enable debug to see what's not settling.
            // Note you must run make with OPT=-DVL_DEBUG for debug prints.
            int __Vsaved_debug = Verilated::debug();
            Verilated::debug(1);
            __Vchange = _change_request(vlSymsp);
            Verilated::debug(__Vsaved_debug);
            VL_FATAL_MT("/home/zhumd/work/ideafromai/research/hardware_mechanisms_20260906/sim/packet_test_top.sv", 3, "",
                "Verilated model didn't DC converge\n"
                "- See DIDNOTCONVERGE in the Verilator manual");
        } else {
            __Vchange = _change_request(vlSymsp);
        }
    } while (VL_UNLIKELY(__Vchange));
}

VL_INLINE_OPT void Vpacket_test_top::_sequent__TOP__1(Vpacket_test_top__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vpacket_test_top::_sequent__TOP__1\n"); );
    Vpacket_test_top* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Variables
    CData/*1:0*/ __Vdly__packet_test_top__DOT__encoder__DOT__state_q;
    CData/*5:0*/ __Vdly__packet_test_top__DOT__encoder__DOT__count_q;
    CData/*0:0*/ __Vdly__packet_test_top__DOT__encoder__DOT__a_valid_q;
    CData/*0:0*/ __Vdly__packet_test_top__DOT__encoder__DOT__b_valid_q;
    CData/*0:0*/ __Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v0;
    CData/*1:0*/ __Vdlyvdim0__packet_test_top__DOT__encoder__DOT__kept_lo_q__v4;
    CData/*0:0*/ __Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v4;
    CData/*1:0*/ __Vdlyvdim0__packet_test_top__DOT__encoder__DOT__kept_hi_q__v4;
    CData/*1:0*/ __Vdlyvdim0__packet_test_top__DOT__encoder__DOT__kept_index_q__v4;
    CData/*4:0*/ __Vdlyvval__packet_test_top__DOT__encoder__DOT__kept_index_q__v4;
    CData/*1:0*/ __Vdlyvdim0__packet_test_top__DOT__encoder__DOT__kept_distance_q__v4;
    CData/*0:0*/ __Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v5;
    CData/*2:0*/ __Vdly__packet_test_top__DOT__group_commit__DOT__state_q;
    CData/*0:0*/ __Vdly__packet_test_top__DOT__group_commit__DOT__error_q;
    QData/*32:0*/ __Vdly__packet_test_top__DOT__encoder__DOT__a_q;
    QData/*32:0*/ __Vdly__packet_test_top__DOT__encoder__DOT__b_q;
    QData/*32:0*/ __Vdlyvval__packet_test_top__DOT__encoder__DOT__kept_lo_q__v4;
    QData/*32:0*/ __Vdlyvval__packet_test_top__DOT__encoder__DOT__kept_hi_q__v4;
    QData/*33:0*/ __Vdlyvval__packet_test_top__DOT__encoder__DOT__kept_distance_q__v4;
    // Body
    __Vdly__packet_test_top__DOT__group_commit__DOT__error_q 
        = vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q;
    __Vdly__packet_test_top__DOT__group_commit__DOT__state_q 
        = vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q;
    __Vdly__packet_test_top__DOT__encoder__DOT__count_q 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__count_q;
    __Vdly__packet_test_top__DOT__encoder__DOT__state_q 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__state_q;
    __Vdly__packet_test_top__DOT__encoder__DOT__b_valid_q 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__b_valid_q;
    __Vdly__packet_test_top__DOT__encoder__DOT__a_valid_q 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__a_valid_q;
    __Vdly__packet_test_top__DOT__encoder__DOT__b_q 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__b_q;
    __Vdly__packet_test_top__DOT__encoder__DOT__a_q 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__a_q;
    __Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v0 = 0U;
    __Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v4 = 0U;
    __Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v5 = 0U;
    if (vlTOPp->rst_n) {
        if (vlTOPp->v_ready) {
            if (vlTOPp->v_valid) {
                vlTOPp->v_out_epoch = (0xffffU & ((
                                                   vlTOPp->v_packet[0xeU] 
                                                   << 0x1cU) 
                                                  | (vlTOPp->v_packet[0xdU] 
                                                     >> 4U)));
            }
        }
    } else {
        vlTOPp->v_out_epoch = 0U;
    }
    if (vlTOPp->rst_n) {
        if (vlTOPp->v_ready) {
            if (vlTOPp->v_valid) {
                vlTOPp->v_theta = ((vlTOPp->v_packet[0xeU] 
                                    << 0xcU) | (vlTOPp->v_packet[0xdU] 
                                                >> 0x14U));
            }
        }
    } else {
        vlTOPp->v_theta = 0U;
    }
    if (vlTOPp->rst_n) {
        if (vlTOPp->v_ready) {
            if (vlTOPp->v_valid) {
                vlTOPp->v_out_context = ((vlTOPp->v_packet[0xdU] 
                                          << 0x1cU) 
                                         | (vlTOPp->v_packet[0xcU] 
                                            >> 4U));
            }
        }
    } else {
        vlTOPp->v_out_context = 0U;
    }
    if (vlTOPp->rst_n) {
        if (vlTOPp->v_ready) {
            vlTOPp->v_result_valid = vlTOPp->v_valid;
        }
    } else {
        vlTOPp->v_result_valid = 0U;
    }
    if (vlTOPp->rst_n) {
        if (vlTOPp->v_ready) {
            if (vlTOPp->v_valid) {
                vlTOPp->v_status = vlTOPp->packet_test_top__DOT__verifier__DOT__status_next;
            }
        }
    } else {
        vlTOPp->v_status = 2U;
    }
    if (vlTOPp->rst_n) {
        if (vlTOPp->v_ready) {
            if (vlTOPp->v_valid) {
                vlTOPp->v_bits = ((0U == (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__status_next))
                                   ? vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits
                                   : 0U);
            }
        }
    } else {
        vlTOPp->v_bits = 0U;
    }
    if (vlTOPp->rst_n) {
        if ((4U & (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q))) {
            if ((2U & (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q))) {
                __Vdly__packet_test_top__DOT__group_commit__DOT__state_q = 0U;
            } else {
                if ((1U & (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q))) {
                    if (vlTOPp->g_result_ready) {
                        __Vdly__packet_test_top__DOT__group_commit__DOT__state_q = 0U;
                    }
                } else {
                    if (vlTOPp->g_repair_valid) {
                        __Vdly__packet_test_top__DOT__group_commit__DOT__error_q 
                            = (((vlTOPp->g_repair_context 
                                 != vlTOPp->g_out_context) 
                                | ((IData)(vlTOPp->g_repair_epoch) 
                                   != (IData)(vlTOPp->g_out_epoch))) 
                               | (vlTOPp->g_repair_theta 
                                  != vlTOPp->g_out_theta));
                        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[0U] 
                            = vlTOPp->g_repair_bits[0U];
                        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[1U] 
                            = vlTOPp->g_repair_bits[1U];
                        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[2U] 
                            = vlTOPp->g_repair_bits[2U];
                        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[3U] 
                            = vlTOPp->g_repair_bits[3U];
                        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[4U] 
                            = vlTOPp->g_repair_bits[4U];
                        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[5U] 
                            = vlTOPp->g_repair_bits[5U];
                        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[6U] 
                            = vlTOPp->g_repair_bits[6U];
                        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[7U] 
                            = vlTOPp->g_repair_bits[7U];
                        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[8U] 
                            = vlTOPp->g_repair_bits[8U];
                        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[9U] 
                            = vlTOPp->g_repair_bits[9U];
                        __Vdly__packet_test_top__DOT__group_commit__DOT__state_q = 5U;
                    }
                }
            }
        } else {
            if ((2U & (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q))) {
                if ((1U & (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q))) {
                    if (vlTOPp->g_replay_ready) {
                        __Vdly__packet_test_top__DOT__group_commit__DOT__state_q = 4U;
                    }
                } else {
                    __Vdly__packet_test_top__DOT__group_commit__DOT__state_q 
                        = ((IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q)
                            ? 5U : ((0U != (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__replay_q))
                                     ? 3U : 5U));
                }
            } else {
                if ((1U & (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q))) {
                    if (vlTOPp->g_packet_valid) {
                        if (((((vlTOPp->g_packet_context 
                                != (vlTOPp->g_out_context 
                                    | (0xfU & (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__count_q)))) 
                               | ((IData)(vlTOPp->g_packet_epoch) 
                                  != (IData)(vlTOPp->g_out_epoch))) 
                              | (vlTOPp->g_packet_theta 
                                 != vlTOPp->g_out_theta)) 
                             | (2U <= (IData)(vlTOPp->g_packet_status)))) {
                            __Vdly__packet_test_top__DOT__group_commit__DOT__error_q = 1U;
                        }
                        if ((1U == (IData)(vlTOPp->g_packet_status))) {
                            vlTOPp->packet_test_top__DOT__group_commit__DOT____Vlvbound1 = 1U;
                            if ((9U >= (0xfU & (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__count_q)))) {
                                vlTOPp->packet_test_top__DOT__group_commit__DOT__replay_q 
                                    = (((~ ((IData)(1U) 
                                            << (0xfU 
                                                & (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__count_q)))) 
                                        & (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__replay_q)) 
                                       | ((IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT____Vlvbound1) 
                                          << (0xfU 
                                              & (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__count_q))));
                            }
                        }
                        if ((0U == (IData)(vlTOPp->g_packet_status))) {
                            vlTOPp->packet_test_top__DOT__group_commit__DOT____Vlvbound2 
                                = vlTOPp->g_packet_bits;
                            if ((0x13fU >= (0x1ffU 
                                            & VL_MULS_III(9,32,32, (IData)(0x20U), (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__count_q))))) {
                                VL_ASSIGNSEL_WIII(32,
                                                  (0x1ffU 
                                                   & VL_MULS_III(9,32,32, (IData)(0x20U), (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__count_q))), vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q, vlTOPp->packet_test_top__DOT__group_commit__DOT____Vlvbound2);
                            }
                        }
                        if ((9U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__count_q))) {
                            __Vdly__packet_test_top__DOT__group_commit__DOT__state_q = 2U;
                        } else {
                            vlTOPp->packet_test_top__DOT__group_commit__DOT__count_q 
                                = (0x1fU & ((IData)(1U) 
                                            + (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__count_q)));
                        }
                    }
                } else {
                    if (vlTOPp->g_start_valid) {
                        __Vdly__packet_test_top__DOT__group_commit__DOT__state_q = 1U;
                        vlTOPp->packet_test_top__DOT__group_commit__DOT__count_q = 0U;
                        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[0U] = 0U;
                        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[1U] = 0U;
                        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[2U] = 0U;
                        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[3U] = 0U;
                        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[4U] = 0U;
                        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[5U] = 0U;
                        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[6U] = 0U;
                        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[7U] = 0U;
                        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[8U] = 0U;
                        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[9U] = 0U;
                        vlTOPp->packet_test_top__DOT__group_commit__DOT__replay_q = 0U;
                        __Vdly__packet_test_top__DOT__group_commit__DOT__error_q 
                            = (0U != (0xfU & vlTOPp->g_base));
                        vlTOPp->g_out_context = vlTOPp->g_base;
                        vlTOPp->g_out_epoch = vlTOPp->g_epoch;
                        vlTOPp->g_out_theta = vlTOPp->g_theta;
                    }
                }
            }
        }
    } else {
        __Vdly__packet_test_top__DOT__group_commit__DOT__state_q = 0U;
        vlTOPp->packet_test_top__DOT__group_commit__DOT__count_q = 0U;
        __Vdly__packet_test_top__DOT__group_commit__DOT__error_q = 0U;
        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[0U] = 0U;
        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[1U] = 0U;
        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[2U] = 0U;
        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[3U] = 0U;
        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[4U] = 0U;
        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[5U] = 0U;
        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[6U] = 0U;
        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[7U] = 0U;
        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[8U] = 0U;
        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[9U] = 0U;
        vlTOPp->packet_test_top__DOT__group_commit__DOT__replay_q = 0U;
        vlTOPp->g_out_context = 0U;
        vlTOPp->g_out_epoch = 0U;
        vlTOPp->g_out_theta = 0U;
    }
    if (vlTOPp->rst_n) {
        if ((0U == (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__state_q))) {
            if (vlTOPp->e_start_valid) {
                __Vdly__packet_test_top__DOT__encoder__DOT__state_q = 1U;
                __Vdly__packet_test_top__DOT__encoder__DOT__count_q = 0U;
                vlTOPp->packet_test_top__DOT__encoder__DOT__context_q 
                    = vlTOPp->e_context;
                vlTOPp->packet_test_top__DOT__encoder__DOT__epoch_q 
                    = vlTOPp->e_epoch;
                vlTOPp->packet_test_top__DOT__encoder__DOT__theta_q 
                    = vlTOPp->e_theta;
                vlTOPp->packet_test_top__DOT__encoder__DOT__reverse_q 
                    = vlTOPp->e_reverse;
                vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q 
                    = (VL_ULL(0x1ffffffff) & ((IData)(vlTOPp->e_reverse)
                                               ? VL_NEGATE_Q(
                                                             (((QData)((IData)(
                                                                               (1U 
                                                                                & (vlTOPp->e_prediction 
                                                                                >> 0x1fU)))) 
                                                               << 0x20U) 
                                                              | (QData)((IData)(vlTOPp->e_prediction))))
                                               : (((QData)((IData)(
                                                                   (1U 
                                                                    & (vlTOPp->e_prediction 
                                                                       >> 0x1fU)))) 
                                                   << 0x20U) 
                                                  | (QData)((IData)(vlTOPp->e_prediction)))));
                vlTOPp->packet_test_top__DOT__encoder__DOT__error_q = 0U;
                vlTOPp->packet_test_top__DOT__encoder__DOT__bits_q = 0U;
                __Vdly__packet_test_top__DOT__encoder__DOT__a_q = VL_ULL(0);
                __Vdly__packet_test_top__DOT__encoder__DOT__b_q = VL_ULL(0);
                __Vdly__packet_test_top__DOT__encoder__DOT__a_valid_q = 0U;
                __Vdly__packet_test_top__DOT__encoder__DOT__b_valid_q = 0U;
                vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q = 0U;
                __Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v0 = 1U;
            }
        } else {
            if ((1U == (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__state_q))) {
                if (vlTOPp->e_in_valid) {
                    if (vlTOPp->packet_test_top__DOT__encoder__DOT__do_discard) {
                        if (vlTOPp->packet_test_top__DOT__encoder__DOT__discard_bit) {
                            if ((1U & ((~ (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__b_valid_q)) 
                                       | VL_LTS_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__encoder__DOT__discard_lo, vlTOPp->packet_test_top__DOT__encoder__DOT__b_q)))) {
                                __Vdly__packet_test_top__DOT__encoder__DOT__b_q 
                                    = vlTOPp->packet_test_top__DOT__encoder__DOT__discard_lo;
                            }
                            __Vdly__packet_test_top__DOT__encoder__DOT__b_valid_q = 1U;
                        } else {
                            if ((1U & ((~ (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__a_valid_q)) 
                                       | VL_GTS_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__encoder__DOT__discard_hi, vlTOPp->packet_test_top__DOT__encoder__DOT__a_q)))) {
                                __Vdly__packet_test_top__DOT__encoder__DOT__a_q 
                                    = vlTOPp->packet_test_top__DOT__encoder__DOT__discard_hi;
                            }
                            __Vdly__packet_test_top__DOT__encoder__DOT__a_valid_q = 1U;
                        }
                    }
                    if (VL_GTS_III(1,32,32, vlTOPp->e_lo, vlTOPp->e_hi)) {
                        vlTOPp->packet_test_top__DOT__encoder__DOT__error_q = 1U;
                    }
                    vlTOPp->packet_test_top__DOT__encoder__DOT__bits_q 
                        = (((~ ((IData)(1U) << (0x1fU 
                                                & (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__count_q)))) 
                            & vlTOPp->packet_test_top__DOT__encoder__DOT__bits_q) 
                           | ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__bit_now) 
                              << (0x1fU & (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__count_q))));
                    if (vlTOPp->packet_test_top__DOT__encoder__DOT__do_keep) {
                        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q 
                            = ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q) 
                               | ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__chosen)));
                        __Vdlyvval__packet_test_top__DOT__encoder__DOT__kept_lo_q__v4 
                            = vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo;
                        __Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v4 = 1U;
                        __Vdlyvdim0__packet_test_top__DOT__encoder__DOT__kept_lo_q__v4 
                            = vlTOPp->packet_test_top__DOT__encoder__DOT__chosen;
                        __Vdlyvval__packet_test_top__DOT__encoder__DOT__kept_hi_q__v4 
                            = vlTOPp->packet_test_top__DOT__encoder__DOT__norm_hi;
                        __Vdlyvdim0__packet_test_top__DOT__encoder__DOT__kept_hi_q__v4 
                            = vlTOPp->packet_test_top__DOT__encoder__DOT__chosen;
                        __Vdlyvval__packet_test_top__DOT__encoder__DOT__kept_index_q__v4 
                            = (0x1fU & (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__count_q));
                        __Vdlyvdim0__packet_test_top__DOT__encoder__DOT__kept_index_q__v4 
                            = vlTOPp->packet_test_top__DOT__encoder__DOT__chosen;
                        __Vdlyvval__packet_test_top__DOT__encoder__DOT__kept_distance_q__v4 
                            = vlTOPp->packet_test_top__DOT__encoder__DOT__distance_now;
                        __Vdlyvdim0__packet_test_top__DOT__encoder__DOT__kept_distance_q__v4 
                            = vlTOPp->packet_test_top__DOT__encoder__DOT__chosen;
                    }
                    if ((0x1fU == (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__count_q))) {
                        __Vdly__packet_test_top__DOT__encoder__DOT__state_q = 2U;
                    } else {
                        __Vdly__packet_test_top__DOT__encoder__DOT__count_q 
                            = (0x3fU & ((IData)(1U) 
                                        + (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__count_q)));
                    }
                }
            } else {
                if ((2U == (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__state_q))) {
                    if (vlTOPp->e_packet_ready) {
                        __Vdly__packet_test_top__DOT__encoder__DOT__state_q = 0U;
                    }
                } else {
                    __Vdly__packet_test_top__DOT__encoder__DOT__state_q = 0U;
                }
            }
        }
    } else {
        __Vdly__packet_test_top__DOT__encoder__DOT__state_q = 0U;
        __Vdly__packet_test_top__DOT__encoder__DOT__count_q = 0U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__context_q = 0U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__epoch_q = 0U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__theta_q = 0U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__reverse_q = 0U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__error_q = 0U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q = VL_ULL(0);
        vlTOPp->packet_test_top__DOT__encoder__DOT__bits_q = 0U;
        __Vdly__packet_test_top__DOT__encoder__DOT__a_q = VL_ULL(0);
        __Vdly__packet_test_top__DOT__encoder__DOT__b_q = VL_ULL(0);
        __Vdly__packet_test_top__DOT__encoder__DOT__a_valid_q = 0U;
        __Vdly__packet_test_top__DOT__encoder__DOT__b_valid_q = 0U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q = 0U;
        __Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v5 = 1U;
    }
    vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q 
        = __Vdly__packet_test_top__DOT__group_commit__DOT__error_q;
    vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q 
        = __Vdly__packet_test_top__DOT__group_commit__DOT__state_q;
    vlTOPp->packet_test_top__DOT__encoder__DOT__count_q 
        = __Vdly__packet_test_top__DOT__encoder__DOT__count_q;
    vlTOPp->packet_test_top__DOT__encoder__DOT__state_q 
        = __Vdly__packet_test_top__DOT__encoder__DOT__state_q;
    vlTOPp->packet_test_top__DOT__encoder__DOT__a_q 
        = __Vdly__packet_test_top__DOT__encoder__DOT__a_q;
    vlTOPp->packet_test_top__DOT__encoder__DOT__b_q 
        = __Vdly__packet_test_top__DOT__encoder__DOT__b_q;
    vlTOPp->packet_test_top__DOT__encoder__DOT__a_valid_q 
        = __Vdly__packet_test_top__DOT__encoder__DOT__a_valid_q;
    vlTOPp->packet_test_top__DOT__encoder__DOT__b_valid_q 
        = __Vdly__packet_test_top__DOT__encoder__DOT__b_valid_q;
    if (__Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v0) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q[0U] = VL_ULL(0);
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q[1U] = VL_ULL(0);
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q[2U] = VL_ULL(0);
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q[3U] = VL_ULL(0);
    }
    if (__Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v4) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q[__Vdlyvdim0__packet_test_top__DOT__encoder__DOT__kept_distance_q__v4] 
            = __Vdlyvval__packet_test_top__DOT__encoder__DOT__kept_distance_q__v4;
    }
    if (__Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v5) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q[0U] = VL_ULL(0);
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q[1U] = VL_ULL(0);
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q[2U] = VL_ULL(0);
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q[3U] = VL_ULL(0);
    }
    if (__Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v0) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q[0U] = 0U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q[1U] = 0U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q[2U] = 0U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q[3U] = 0U;
    }
    if (__Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v4) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q[__Vdlyvdim0__packet_test_top__DOT__encoder__DOT__kept_index_q__v4] 
            = __Vdlyvval__packet_test_top__DOT__encoder__DOT__kept_index_q__v4;
    }
    if (__Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v5) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q[0U] = 0U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q[1U] = 0U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q[2U] = 0U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q[3U] = 0U;
    }
    if (__Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v0) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q[0U] = VL_ULL(0);
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q[1U] = VL_ULL(0);
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q[2U] = VL_ULL(0);
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q[3U] = VL_ULL(0);
    }
    if (__Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v4) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q[__Vdlyvdim0__packet_test_top__DOT__encoder__DOT__kept_hi_q__v4] 
            = __Vdlyvval__packet_test_top__DOT__encoder__DOT__kept_hi_q__v4;
    }
    if (__Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v5) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q[0U] = VL_ULL(0);
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q[1U] = VL_ULL(0);
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q[2U] = VL_ULL(0);
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q[3U] = VL_ULL(0);
    }
    if (__Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v0) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q[0U] = VL_ULL(0);
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q[1U] = VL_ULL(0);
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q[2U] = VL_ULL(0);
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q[3U] = VL_ULL(0);
    }
    if (__Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v4) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q[__Vdlyvdim0__packet_test_top__DOT__encoder__DOT__kept_lo_q__v4] 
            = __Vdlyvval__packet_test_top__DOT__encoder__DOT__kept_lo_q__v4;
    }
    if (__Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v5) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q[0U] = VL_ULL(0);
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q[1U] = VL_ULL(0);
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q[2U] = VL_ULL(0);
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q[3U] = VL_ULL(0);
    }
    vlTOPp->g_replay_mask = vlTOPp->packet_test_top__DOT__group_commit__DOT__replay_q;
    vlTOPp->g_status = ((IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q)
                         ? 2U : 0U);
    vlTOPp->g_start_ready = (0U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q));
    vlTOPp->g_packet_ready = (1U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q));
    vlTOPp->g_replay_valid = (3U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q));
    vlTOPp->g_repair_ready = (4U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q));
    vlTOPp->g_result_valid = (5U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q));
    vlTOPp->g_bits[0U] = (((5U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q)) 
                           & (~ (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q)))
                           ? vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[0U]
                           : 0U);
    vlTOPp->g_bits[1U] = (((5U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q)) 
                           & (~ (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q)))
                           ? vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[1U]
                           : 0U);
    vlTOPp->g_bits[2U] = (((5U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q)) 
                           & (~ (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q)))
                           ? vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[2U]
                           : 0U);
    vlTOPp->g_bits[3U] = (((5U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q)) 
                           & (~ (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q)))
                           ? vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[3U]
                           : 0U);
    vlTOPp->g_bits[4U] = (((5U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q)) 
                           & (~ (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q)))
                           ? vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[4U]
                           : 0U);
    vlTOPp->g_bits[5U] = (((5U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q)) 
                           & (~ (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q)))
                           ? vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[5U]
                           : 0U);
    vlTOPp->g_bits[6U] = (((5U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q)) 
                           & (~ (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q)))
                           ? vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[6U]
                           : 0U);
    vlTOPp->g_bits[7U] = (((5U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q)) 
                           & (~ (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q)))
                           ? vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[7U]
                           : 0U);
    vlTOPp->g_bits[8U] = (((5U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q)) 
                           & (~ (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q)))
                           ? vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[8U]
                           : 0U);
    vlTOPp->g_bits[9U] = (((5U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q)) 
                           & (~ (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q)))
                           ? vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[9U]
                           : 0U);
    vlTOPp->e_start_ready = (0U == (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__state_q));
    vlTOPp->e_in_ready = (1U == (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__state_q));
    vlTOPp->e_packet_valid = (2U == (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__state_q));
    vlTOPp->e_packet[0U] = 0U;
    vlTOPp->e_packet[1U] = 0U;
    vlTOPp->e_packet[2U] = 0U;
    vlTOPp->e_packet[3U] = 0U;
    vlTOPp->e_packet[4U] = 0U;
    vlTOPp->e_packet[5U] = 0U;
    vlTOPp->e_packet[6U] = 0U;
    vlTOPp->e_packet[7U] = 0U;
    vlTOPp->e_packet[8U] = 0U;
    vlTOPp->e_packet[9U] = 0U;
    vlTOPp->e_packet[0xaU] = 0U;
    vlTOPp->e_packet[0xbU] = 0U;
    vlTOPp->e_packet[0xcU] = 0U;
    vlTOPp->e_packet[0xdU] = 0U;
    vlTOPp->e_packet[0xeU] = 0U;
    vlTOPp->e_packet[0U] = vlTOPp->packet_test_top__DOT__encoder__DOT__bits_q;
    vlTOPp->e_packet[1U] = (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__a_q);
    vlTOPp->e_packet[2U] = ((0xfffffffeU & vlTOPp->e_packet[2U]) 
                            | (IData)((vlTOPp->packet_test_top__DOT__encoder__DOT__a_q 
                                       >> 0x20U)));
    vlTOPp->e_packet[2U] = ((1U & vlTOPp->e_packet[2U]) 
                            | (0xfffffffeU & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__b_q) 
                                              << 1U)));
    vlTOPp->e_packet[3U] = ((0xfffffffcU & vlTOPp->e_packet[3U]) 
                            | ((1U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__b_q) 
                                      >> 0x1fU)) | 
                               (0xfffffffeU & ((IData)(
                                                       (vlTOPp->packet_test_top__DOT__encoder__DOT__b_q 
                                                        >> 0x20U)) 
                                               << 1U))));
    vlTOPp->e_packet[3U] = ((0xfffffffbU & vlTOPp->e_packet[3U]) 
                            | (0xfffffffcU & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__a_valid_q) 
                                              << 2U)));
    vlTOPp->e_packet[3U] = ((0xfffffff7U & vlTOPp->e_packet[3U]) 
                            | (0xfffffff8U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__b_valid_q) 
                                              << 3U)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q
        [0U];
    vlTOPp->e_packet[3U] = ((0xfU & vlTOPp->e_packet[3U]) 
                            | (0xfffffff0U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1) 
                                              << 4U)));
    vlTOPp->e_packet[4U] = ((0xffffffe0U & vlTOPp->e_packet[4U]) 
                            | ((0xfU & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1) 
                                        >> 0x1cU)) 
                               | (0xfffffff0U & ((IData)(
                                                         (vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1 
                                                          >> 0x20U)) 
                                                 << 4U))));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q
        [0U];
    vlTOPp->e_packet[4U] = ((0x1fU & vlTOPp->e_packet[4U]) 
                            | (0xffffffe0U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2) 
                                              << 5U)));
    vlTOPp->e_packet[5U] = ((0xffffffc0U & vlTOPp->e_packet[5U]) 
                            | ((0x1fU & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2) 
                                         >> 0x1bU)) 
                               | (0xffffffe0U & ((IData)(
                                                         (vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2 
                                                          >> 0x20U)) 
                                                 << 5U))));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound3 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
        [0U];
    vlTOPp->e_packet[5U] = ((0xfffff83fU & vlTOPp->e_packet[5U]) 
                            | (0xffffffc0U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound3) 
                                              << 6U)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound4 
        = (1U & (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q));
    vlTOPp->e_packet[5U] = ((0xfffff7ffU & vlTOPp->e_packet[5U]) 
                            | (0xfffff800U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound4) 
                                              << 0xbU)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q
        [1U];
    vlTOPp->e_packet[5U] = ((0xfffU & vlTOPp->e_packet[5U]) 
                            | (0xfffff000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1) 
                                              << 0xcU)));
    vlTOPp->e_packet[6U] = ((0xffffe000U & vlTOPp->e_packet[6U]) 
                            | ((0xfffU & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1) 
                                          >> 0x14U)) 
                               | (0xfffff000U & ((IData)(
                                                         (vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1 
                                                          >> 0x20U)) 
                                                 << 0xcU))));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q
        [1U];
    vlTOPp->e_packet[6U] = ((0x1fffU & vlTOPp->e_packet[6U]) 
                            | (0xffffe000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2) 
                                              << 0xdU)));
    vlTOPp->e_packet[7U] = ((0xffffc000U & vlTOPp->e_packet[7U]) 
                            | ((0x1fffU & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2) 
                                           >> 0x13U)) 
                               | (0xffffe000U & ((IData)(
                                                         (vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2 
                                                          >> 0x20U)) 
                                                 << 0xdU))));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound3 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
        [1U];
    vlTOPp->e_packet[7U] = ((0xfff83fffU & vlTOPp->e_packet[7U]) 
                            | (0xffffc000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound3) 
                                              << 0xeU)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound4 
        = (1U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q) 
                 >> 1U));
    vlTOPp->e_packet[7U] = ((0xfff7ffffU & vlTOPp->e_packet[7U]) 
                            | (0xfff80000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound4) 
                                              << 0x13U)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q
        [2U];
    vlTOPp->e_packet[7U] = ((0xfffffU & vlTOPp->e_packet[7U]) 
                            | (0xfff00000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1) 
                                              << 0x14U)));
    vlTOPp->e_packet[8U] = ((0xffe00000U & vlTOPp->e_packet[8U]) 
                            | ((0xfffffU & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1) 
                                            >> 0xcU)) 
                               | (0xfff00000U & ((IData)(
                                                         (vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1 
                                                          >> 0x20U)) 
                                                 << 0x14U))));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q
        [2U];
    vlTOPp->e_packet[8U] = ((0x1fffffU & vlTOPp->e_packet[8U]) 
                            | (0xffe00000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2) 
                                              << 0x15U)));
    vlTOPp->e_packet[9U] = ((0xffc00000U & vlTOPp->e_packet[9U]) 
                            | ((0x1fffffU & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2) 
                                             >> 0xbU)) 
                               | (0xffe00000U & ((IData)(
                                                         (vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2 
                                                          >> 0x20U)) 
                                                 << 0x15U))));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound3 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
        [2U];
    vlTOPp->e_packet[9U] = ((0xf83fffffU & vlTOPp->e_packet[9U]) 
                            | (0xffc00000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound3) 
                                              << 0x16U)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound4 
        = (1U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q) 
                 >> 2U));
    vlTOPp->e_packet[9U] = ((0xf7ffffffU & vlTOPp->e_packet[9U]) 
                            | (0xf8000000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound4) 
                                              << 0x1bU)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q
        [3U];
    vlTOPp->e_packet[9U] = ((0xfffffffU & vlTOPp->e_packet[9U]) 
                            | (0xf0000000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1) 
                                              << 0x1cU)));
    vlTOPp->e_packet[0xaU] = ((0xe0000000U & vlTOPp->e_packet[0xaU]) 
                              | ((0xfffffffU & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1) 
                                                >> 4U)) 
                                 | (0xf0000000U & ((IData)(
                                                           (vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1 
                                                            >> 0x20U)) 
                                                   << 0x1cU))));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q
        [3U];
    vlTOPp->e_packet[0xaU] = ((0x1fffffffU & vlTOPp->e_packet[0xaU]) 
                              | (0xe0000000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2) 
                                                << 0x1dU)));
    vlTOPp->e_packet[0xbU] = ((0xc0000000U & vlTOPp->e_packet[0xbU]) 
                              | ((0x1fffffffU & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2) 
                                                 >> 3U)) 
                                 | (0xe0000000U & ((IData)(
                                                           (vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2 
                                                            >> 0x20U)) 
                                                   << 0x1dU))));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound3 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
        [3U];
    vlTOPp->e_packet[0xbU] = ((0x3fffffffU & vlTOPp->e_packet[0xbU]) 
                              | (0xc0000000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound3) 
                                                << 0x1eU)));
    vlTOPp->e_packet[0xcU] = ((0xfffffff8U & vlTOPp->e_packet[0xcU]) 
                              | (0x3fffffffU & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound3) 
                                                >> 2U)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound4 
        = (1U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q) 
                 >> 3U));
    vlTOPp->e_packet[0xcU] = ((0xfffffff7U & vlTOPp->e_packet[0xcU]) 
                              | (0xfffffff8U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound4) 
                                                << 3U)));
    vlTOPp->e_packet[0xcU] = ((0xfU & vlTOPp->e_packet[0xcU]) 
                              | (0xfffffff0U & (vlTOPp->packet_test_top__DOT__encoder__DOT__context_q 
                                                << 4U)));
    vlTOPp->e_packet[0xdU] = ((0xfffffff0U & vlTOPp->e_packet[0xdU]) 
                              | (0xfU & (vlTOPp->packet_test_top__DOT__encoder__DOT__context_q 
                                         >> 0x1cU)));
    vlTOPp->e_packet[0xdU] = ((0xfff0000fU & vlTOPp->e_packet[0xdU]) 
                              | (0xfffffff0U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__epoch_q) 
                                                << 4U)));
    vlTOPp->e_packet[0xdU] = ((0xfffffU & vlTOPp->e_packet[0xdU]) 
                              | (0xfff00000U & (vlTOPp->packet_test_top__DOT__encoder__DOT__theta_q 
                                                << 0x14U)));
    vlTOPp->e_packet[0xeU] = ((0x300000U & vlTOPp->e_packet[0xeU]) 
                              | (0xfffffU & (vlTOPp->packet_test_top__DOT__encoder__DOT__theta_q 
                                             >> 0xcU)));
    vlTOPp->e_packet[0xeU] = ((0x2fffffU & vlTOPp->e_packet[0xeU]) 
                              | (0xfff00000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__reverse_q) 
                                                << 0x14U)));
    vlTOPp->e_packet[0xeU] = ((0x1fffffU & vlTOPp->e_packet[0xeU]) 
                              | (0xffe00000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__error_q) 
                                                << 0x15U)));
}

void Vpacket_test_top::_settle__TOP__2(Vpacket_test_top__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vpacket_test_top::_settle__TOP__2\n"); );
    Vpacket_test_top* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->packet_test_top__DOT__verifier__DOT__tau_lo 
        = (((QData)((IData)((1U & (vlTOPp->v_tau_lo 
                                   >> 0x1fU)))) << 0x20U) 
           | (QData)((IData)(vlTOPp->v_tau_lo)));
    vlTOPp->packet_test_top__DOT__verifier__DOT__tau_hi 
        = (((QData)((IData)((1U & (vlTOPp->v_tau_hi 
                                   >> 0x1fU)))) << 0x20U) 
           | (QData)((IData)(vlTOPp->v_tau_hi)));
    if ((0x100000U & vlTOPp->v_packet[0xeU])) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__tau_lo 
            = (VL_ULL(0x1ffffffff) & VL_NEGATE_Q((((QData)((IData)(
                                                                   (1U 
                                                                    & (vlTOPp->v_tau_hi 
                                                                       >> 0x1fU)))) 
                                                   << 0x20U) 
                                                  | (QData)((IData)(vlTOPp->v_tau_hi)))));
        vlTOPp->packet_test_top__DOT__verifier__DOT__tau_hi 
            = (VL_ULL(0x1ffffffff) & VL_NEGATE_Q((((QData)((IData)(
                                                                   (1U 
                                                                    & (vlTOPp->v_tau_lo 
                                                                       >> 0x1fU)))) 
                                                   << 0x20U) 
                                                  | (QData)((IData)(vlTOPp->v_tau_lo)))));
    }
    vlTOPp->packet_test_top__DOT__verifier__DOT__bad 
        = (1U & ((((vlTOPp->v_packet[0xeU] >> 0x15U) 
                   | (((vlTOPp->v_packet[0xdU] << 0x1cU) 
                       | (vlTOPp->v_packet[0xcU] >> 4U)) 
                      != vlTOPp->v_context)) | ((0xffffU 
                                                 & ((vlTOPp->v_packet[0xeU] 
                                                     << 0x1cU) 
                                                    | (vlTOPp->v_packet[0xdU] 
                                                       >> 4U))) 
                                                != (IData)(vlTOPp->v_epoch))) 
                 | VL_GTS_III(1,32,32, vlTOPp->v_tau_lo, vlTOPp->v_tau_hi)));
    vlTOPp->packet_test_top__DOT__verifier__DOT__a 
        = (VL_ULL(0x1ffffffff) & (((QData)((IData)(
                                                   vlTOPp->v_packet[2U])) 
                                   << 0x20U) | (QData)((IData)(
                                                               vlTOPp->v_packet[1U]))));
    vlTOPp->packet_test_top__DOT__verifier__DOT__b 
        = (VL_ULL(0x1ffffffff) & (((QData)((IData)(
                                                   vlTOPp->v_packet[4U])) 
                                   << 0x3fU) | (((QData)((IData)(
                                                                 vlTOPp->v_packet[3U])) 
                                                 << 0x1fU) 
                                                | ((QData)((IData)(
                                                                   vlTOPp->v_packet[2U])) 
                                                   >> 1U))));
    vlTOPp->packet_test_top__DOT__verifier__DOT__uncertain 
        = (((vlTOPp->v_packet[3U] >> 2U) & VL_GTES_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__a, vlTOPp->packet_test_top__DOT__verifier__DOT__tau_lo)) 
           | ((vlTOPp->v_packet[3U] >> 3U) & VL_GTS_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__tau_hi, vlTOPp->packet_test_top__DOT__verifier__DOT__b)));
    vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
        = vlTOPp->v_packet[0U];
    vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index = 0U;
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo 
        = (VL_ULL(0x1ffffffff) & (((QData)((IData)(
                                                   vlTOPp->v_packet[5U])) 
                                   << 0x3cU) | (((QData)((IData)(
                                                                 vlTOPp->v_packet[4U])) 
                                                 << 0x1cU) 
                                                | ((QData)((IData)(
                                                                   vlTOPp->v_packet[3U])) 
                                                   >> 4U))));
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi 
        = (VL_ULL(0x1ffffffff) & (((QData)((IData)(
                                                   vlTOPp->v_packet[6U])) 
                                   << 0x3bU) | (((QData)((IData)(
                                                                 vlTOPp->v_packet[5U])) 
                                                 << 0x1bU) 
                                                | ((QData)((IData)(
                                                                   vlTOPp->v_packet[4U])) 
                                                   >> 5U))));
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_index 
        = (0x1fU & ((vlTOPp->v_packet[6U] << 0x1aU) 
                    | (vlTOPp->v_packet[5U] >> 6U)));
    if ((1U & (~ (vlTOPp->v_packet[5U] >> 0xbU)))) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
    }
    if ((VL_LTES_III(1,32,32, 0x20U, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)) 
         | VL_GTS_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo, vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi))) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
    } else {
        if ((1U & (vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
                   >> (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)))) {
            vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
        }
        vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
            = (vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
               | ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)));
        if (VL_GTES_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo, vlTOPp->packet_test_top__DOT__verifier__DOT__tau_hi)) {
            vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                = (vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                   | ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)));
        } else {
            if (VL_LTS_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi, vlTOPp->packet_test_top__DOT__verifier__DOT__tau_lo)) {
                vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                    = ((~ ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index))) 
                       & vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits);
            } else {
                vlTOPp->packet_test_top__DOT__verifier__DOT__uncertain = 1U;
            }
        }
    }
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo 
        = (VL_ULL(0x1ffffffff) & (((QData)((IData)(
                                                   vlTOPp->v_packet[7U])) 
                                   << 0x34U) | (((QData)((IData)(
                                                                 vlTOPp->v_packet[6U])) 
                                                 << 0x14U) 
                                                | ((QData)((IData)(
                                                                   vlTOPp->v_packet[5U])) 
                                                   >> 0xcU))));
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi 
        = (VL_ULL(0x1ffffffff) & (((QData)((IData)(
                                                   vlTOPp->v_packet[8U])) 
                                   << 0x33U) | (((QData)((IData)(
                                                                 vlTOPp->v_packet[7U])) 
                                                 << 0x13U) 
                                                | ((QData)((IData)(
                                                                   vlTOPp->v_packet[6U])) 
                                                   >> 0xdU))));
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_index 
        = (0x1fU & ((vlTOPp->v_packet[8U] << 0x12U) 
                    | (vlTOPp->v_packet[7U] >> 0xeU)));
    if ((1U & (~ (vlTOPp->v_packet[7U] >> 0x13U)))) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
    }
    if ((VL_LTES_III(1,32,32, 0x20U, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)) 
         | VL_GTS_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo, vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi))) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
    } else {
        if ((1U & (vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
                   >> (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)))) {
            vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
        }
        vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
            = (vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
               | ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)));
        if (VL_GTES_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo, vlTOPp->packet_test_top__DOT__verifier__DOT__tau_hi)) {
            vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                = (vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                   | ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)));
        } else {
            if (VL_LTS_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi, vlTOPp->packet_test_top__DOT__verifier__DOT__tau_lo)) {
                vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                    = ((~ ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index))) 
                       & vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits);
            } else {
                vlTOPp->packet_test_top__DOT__verifier__DOT__uncertain = 1U;
            }
        }
    }
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo 
        = (VL_ULL(0x1ffffffff) & (((QData)((IData)(
                                                   vlTOPp->v_packet[9U])) 
                                   << 0x2cU) | (((QData)((IData)(
                                                                 vlTOPp->v_packet[8U])) 
                                                 << 0xcU) 
                                                | ((QData)((IData)(
                                                                   vlTOPp->v_packet[7U])) 
                                                   >> 0x14U))));
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi 
        = (VL_ULL(0x1ffffffff) & (((QData)((IData)(
                                                   vlTOPp->v_packet[0xaU])) 
                                   << 0x2bU) | (((QData)((IData)(
                                                                 vlTOPp->v_packet[9U])) 
                                                 << 0xbU) 
                                                | ((QData)((IData)(
                                                                   vlTOPp->v_packet[8U])) 
                                                   >> 0x15U))));
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_index 
        = (0x1fU & ((vlTOPp->v_packet[0xaU] << 0xaU) 
                    | (vlTOPp->v_packet[9U] >> 0x16U)));
    if ((1U & (~ (vlTOPp->v_packet[9U] >> 0x1bU)))) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
    }
    if ((VL_LTES_III(1,32,32, 0x20U, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)) 
         | VL_GTS_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo, vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi))) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
    } else {
        if ((1U & (vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
                   >> (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)))) {
            vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
        }
        vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
            = (vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
               | ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)));
        if (VL_GTES_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo, vlTOPp->packet_test_top__DOT__verifier__DOT__tau_hi)) {
            vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                = (vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                   | ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)));
        } else {
            if (VL_LTS_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi, vlTOPp->packet_test_top__DOT__verifier__DOT__tau_lo)) {
                vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                    = ((~ ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index))) 
                       & vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits);
            } else {
                vlTOPp->packet_test_top__DOT__verifier__DOT__uncertain = 1U;
            }
        }
    }
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo 
        = (VL_ULL(0x1ffffffff) & (((QData)((IData)(
                                                   vlTOPp->v_packet[0xbU])) 
                                   << 0x24U) | (((QData)((IData)(
                                                                 vlTOPp->v_packet[0xaU])) 
                                                 << 4U) 
                                                | ((QData)((IData)(
                                                                   vlTOPp->v_packet[9U])) 
                                                   >> 0x1cU))));
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi 
        = (VL_ULL(0x1ffffffff) & (((QData)((IData)(
                                                   vlTOPp->v_packet[0xcU])) 
                                   << 0x23U) | (((QData)((IData)(
                                                                 vlTOPp->v_packet[0xbU])) 
                                                 << 3U) 
                                                | ((QData)((IData)(
                                                                   vlTOPp->v_packet[0xaU])) 
                                                   >> 0x1dU))));
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_index 
        = (0x1fU & ((vlTOPp->v_packet[0xcU] << 2U) 
                    | (vlTOPp->v_packet[0xbU] >> 0x1eU)));
    if ((1U & (~ (vlTOPp->v_packet[0xcU] >> 3U)))) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
    }
    if ((VL_LTES_III(1,32,32, 0x20U, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)) 
         | VL_GTS_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo, vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi))) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
    } else {
        if ((1U & (vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
                   >> (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)))) {
            vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
        }
        vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
            = (vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
               | ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)));
        if (VL_GTES_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo, vlTOPp->packet_test_top__DOT__verifier__DOT__tau_hi)) {
            vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                = (vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                   | ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)));
        } else {
            if (VL_LTS_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi, vlTOPp->packet_test_top__DOT__verifier__DOT__tau_lo)) {
                vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                    = ((~ ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index))) 
                       & vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits);
            } else {
                vlTOPp->packet_test_top__DOT__verifier__DOT__uncertain = 1U;
            }
        }
    }
    vlTOPp->packet_test_top__DOT__verifier__DOT__status_next 
        = ((IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__bad)
            ? 2U : ((IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__uncertain)
                     ? 1U : 0U));
    vlTOPp->e_start_ready = (0U == (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__state_q));
    vlTOPp->e_in_ready = (1U == (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__state_q));
    vlTOPp->e_packet_valid = (2U == (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__state_q));
    vlTOPp->g_start_ready = (0U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q));
    vlTOPp->g_packet_ready = (1U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q));
    vlTOPp->g_replay_valid = (3U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q));
    vlTOPp->g_replay_mask = vlTOPp->packet_test_top__DOT__group_commit__DOT__replay_q;
    vlTOPp->g_repair_ready = (4U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q));
    vlTOPp->g_result_valid = (5U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q));
    vlTOPp->g_status = ((IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q)
                         ? 2U : 0U);
    vlTOPp->v_ready = (1U & ((~ (IData)(vlTOPp->v_result_valid)) 
                             | (IData)(vlTOPp->v_result_ready)));
    vlTOPp->g_bits[0U] = (((5U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q)) 
                           & (~ (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q)))
                           ? vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[0U]
                           : 0U);
    vlTOPp->g_bits[1U] = (((5U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q)) 
                           & (~ (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q)))
                           ? vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[1U]
                           : 0U);
    vlTOPp->g_bits[2U] = (((5U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q)) 
                           & (~ (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q)))
                           ? vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[2U]
                           : 0U);
    vlTOPp->g_bits[3U] = (((5U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q)) 
                           & (~ (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q)))
                           ? vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[3U]
                           : 0U);
    vlTOPp->g_bits[4U] = (((5U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q)) 
                           & (~ (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q)))
                           ? vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[4U]
                           : 0U);
    vlTOPp->g_bits[5U] = (((5U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q)) 
                           & (~ (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q)))
                           ? vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[5U]
                           : 0U);
    vlTOPp->g_bits[6U] = (((5U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q)) 
                           & (~ (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q)))
                           ? vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[6U]
                           : 0U);
    vlTOPp->g_bits[7U] = (((5U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q)) 
                           & (~ (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q)))
                           ? vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[7U]
                           : 0U);
    vlTOPp->g_bits[8U] = (((5U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q)) 
                           & (~ (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q)))
                           ? vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[8U]
                           : 0U);
    vlTOPp->g_bits[9U] = (((5U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q)) 
                           & (~ (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q)))
                           ? vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q[9U]
                           : 0U);
    vlTOPp->e_packet[0U] = 0U;
    vlTOPp->e_packet[1U] = 0U;
    vlTOPp->e_packet[2U] = 0U;
    vlTOPp->e_packet[3U] = 0U;
    vlTOPp->e_packet[4U] = 0U;
    vlTOPp->e_packet[5U] = 0U;
    vlTOPp->e_packet[6U] = 0U;
    vlTOPp->e_packet[7U] = 0U;
    vlTOPp->e_packet[8U] = 0U;
    vlTOPp->e_packet[9U] = 0U;
    vlTOPp->e_packet[0xaU] = 0U;
    vlTOPp->e_packet[0xbU] = 0U;
    vlTOPp->e_packet[0xcU] = 0U;
    vlTOPp->e_packet[0xdU] = 0U;
    vlTOPp->e_packet[0xeU] = 0U;
    vlTOPp->e_packet[0U] = vlTOPp->packet_test_top__DOT__encoder__DOT__bits_q;
    vlTOPp->e_packet[1U] = (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__a_q);
    vlTOPp->e_packet[2U] = ((0xfffffffeU & vlTOPp->e_packet[2U]) 
                            | (IData)((vlTOPp->packet_test_top__DOT__encoder__DOT__a_q 
                                       >> 0x20U)));
    vlTOPp->e_packet[2U] = ((1U & vlTOPp->e_packet[2U]) 
                            | (0xfffffffeU & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__b_q) 
                                              << 1U)));
    vlTOPp->e_packet[3U] = ((0xfffffffcU & vlTOPp->e_packet[3U]) 
                            | ((1U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__b_q) 
                                      >> 0x1fU)) | 
                               (0xfffffffeU & ((IData)(
                                                       (vlTOPp->packet_test_top__DOT__encoder__DOT__b_q 
                                                        >> 0x20U)) 
                                               << 1U))));
    vlTOPp->e_packet[3U] = ((0xfffffffbU & vlTOPp->e_packet[3U]) 
                            | (0xfffffffcU & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__a_valid_q) 
                                              << 2U)));
    vlTOPp->e_packet[3U] = ((0xfffffff7U & vlTOPp->e_packet[3U]) 
                            | (0xfffffff8U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__b_valid_q) 
                                              << 3U)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q
        [0U];
    vlTOPp->e_packet[3U] = ((0xfU & vlTOPp->e_packet[3U]) 
                            | (0xfffffff0U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1) 
                                              << 4U)));
    vlTOPp->e_packet[4U] = ((0xffffffe0U & vlTOPp->e_packet[4U]) 
                            | ((0xfU & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1) 
                                        >> 0x1cU)) 
                               | (0xfffffff0U & ((IData)(
                                                         (vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1 
                                                          >> 0x20U)) 
                                                 << 4U))));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q
        [0U];
    vlTOPp->e_packet[4U] = ((0x1fU & vlTOPp->e_packet[4U]) 
                            | (0xffffffe0U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2) 
                                              << 5U)));
    vlTOPp->e_packet[5U] = ((0xffffffc0U & vlTOPp->e_packet[5U]) 
                            | ((0x1fU & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2) 
                                         >> 0x1bU)) 
                               | (0xffffffe0U & ((IData)(
                                                         (vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2 
                                                          >> 0x20U)) 
                                                 << 5U))));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound3 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
        [0U];
    vlTOPp->e_packet[5U] = ((0xfffff83fU & vlTOPp->e_packet[5U]) 
                            | (0xffffffc0U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound3) 
                                              << 6U)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound4 
        = (1U & (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q));
    vlTOPp->e_packet[5U] = ((0xfffff7ffU & vlTOPp->e_packet[5U]) 
                            | (0xfffff800U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound4) 
                                              << 0xbU)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q
        [1U];
    vlTOPp->e_packet[5U] = ((0xfffU & vlTOPp->e_packet[5U]) 
                            | (0xfffff000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1) 
                                              << 0xcU)));
    vlTOPp->e_packet[6U] = ((0xffffe000U & vlTOPp->e_packet[6U]) 
                            | ((0xfffU & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1) 
                                          >> 0x14U)) 
                               | (0xfffff000U & ((IData)(
                                                         (vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1 
                                                          >> 0x20U)) 
                                                 << 0xcU))));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q
        [1U];
    vlTOPp->e_packet[6U] = ((0x1fffU & vlTOPp->e_packet[6U]) 
                            | (0xffffe000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2) 
                                              << 0xdU)));
    vlTOPp->e_packet[7U] = ((0xffffc000U & vlTOPp->e_packet[7U]) 
                            | ((0x1fffU & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2) 
                                           >> 0x13U)) 
                               | (0xffffe000U & ((IData)(
                                                         (vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2 
                                                          >> 0x20U)) 
                                                 << 0xdU))));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound3 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
        [1U];
    vlTOPp->e_packet[7U] = ((0xfff83fffU & vlTOPp->e_packet[7U]) 
                            | (0xffffc000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound3) 
                                              << 0xeU)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound4 
        = (1U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q) 
                 >> 1U));
    vlTOPp->e_packet[7U] = ((0xfff7ffffU & vlTOPp->e_packet[7U]) 
                            | (0xfff80000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound4) 
                                              << 0x13U)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q
        [2U];
    vlTOPp->e_packet[7U] = ((0xfffffU & vlTOPp->e_packet[7U]) 
                            | (0xfff00000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1) 
                                              << 0x14U)));
    vlTOPp->e_packet[8U] = ((0xffe00000U & vlTOPp->e_packet[8U]) 
                            | ((0xfffffU & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1) 
                                            >> 0xcU)) 
                               | (0xfff00000U & ((IData)(
                                                         (vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1 
                                                          >> 0x20U)) 
                                                 << 0x14U))));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q
        [2U];
    vlTOPp->e_packet[8U] = ((0x1fffffU & vlTOPp->e_packet[8U]) 
                            | (0xffe00000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2) 
                                              << 0x15U)));
    vlTOPp->e_packet[9U] = ((0xffc00000U & vlTOPp->e_packet[9U]) 
                            | ((0x1fffffU & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2) 
                                             >> 0xbU)) 
                               | (0xffe00000U & ((IData)(
                                                         (vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2 
                                                          >> 0x20U)) 
                                                 << 0x15U))));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound3 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
        [2U];
    vlTOPp->e_packet[9U] = ((0xf83fffffU & vlTOPp->e_packet[9U]) 
                            | (0xffc00000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound3) 
                                              << 0x16U)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound4 
        = (1U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q) 
                 >> 2U));
    vlTOPp->e_packet[9U] = ((0xf7ffffffU & vlTOPp->e_packet[9U]) 
                            | (0xf8000000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound4) 
                                              << 0x1bU)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q
        [3U];
    vlTOPp->e_packet[9U] = ((0xfffffffU & vlTOPp->e_packet[9U]) 
                            | (0xf0000000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1) 
                                              << 0x1cU)));
    vlTOPp->e_packet[0xaU] = ((0xe0000000U & vlTOPp->e_packet[0xaU]) 
                              | ((0xfffffffU & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1) 
                                                >> 4U)) 
                                 | (0xf0000000U & ((IData)(
                                                           (vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1 
                                                            >> 0x20U)) 
                                                   << 0x1cU))));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q
        [3U];
    vlTOPp->e_packet[0xaU] = ((0x1fffffffU & vlTOPp->e_packet[0xaU]) 
                              | (0xe0000000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2) 
                                                << 0x1dU)));
    vlTOPp->e_packet[0xbU] = ((0xc0000000U & vlTOPp->e_packet[0xbU]) 
                              | ((0x1fffffffU & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2) 
                                                 >> 3U)) 
                                 | (0xe0000000U & ((IData)(
                                                           (vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2 
                                                            >> 0x20U)) 
                                                   << 0x1dU))));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound3 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
        [3U];
    vlTOPp->e_packet[0xbU] = ((0x3fffffffU & vlTOPp->e_packet[0xbU]) 
                              | (0xc0000000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound3) 
                                                << 0x1eU)));
    vlTOPp->e_packet[0xcU] = ((0xfffffff8U & vlTOPp->e_packet[0xcU]) 
                              | (0x3fffffffU & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound3) 
                                                >> 2U)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound4 
        = (1U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q) 
                 >> 3U));
    vlTOPp->e_packet[0xcU] = ((0xfffffff7U & vlTOPp->e_packet[0xcU]) 
                              | (0xfffffff8U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound4) 
                                                << 3U)));
    vlTOPp->e_packet[0xcU] = ((0xfU & vlTOPp->e_packet[0xcU]) 
                              | (0xfffffff0U & (vlTOPp->packet_test_top__DOT__encoder__DOT__context_q 
                                                << 4U)));
    vlTOPp->e_packet[0xdU] = ((0xfffffff0U & vlTOPp->e_packet[0xdU]) 
                              | (0xfU & (vlTOPp->packet_test_top__DOT__encoder__DOT__context_q 
                                         >> 0x1cU)));
    vlTOPp->e_packet[0xdU] = ((0xfff0000fU & vlTOPp->e_packet[0xdU]) 
                              | (0xfffffff0U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__epoch_q) 
                                                << 4U)));
    vlTOPp->e_packet[0xdU] = ((0xfffffU & vlTOPp->e_packet[0xdU]) 
                              | (0xfff00000U & (vlTOPp->packet_test_top__DOT__encoder__DOT__theta_q 
                                                << 0x14U)));
    vlTOPp->e_packet[0xeU] = ((0x300000U & vlTOPp->e_packet[0xeU]) 
                              | (0xfffffU & (vlTOPp->packet_test_top__DOT__encoder__DOT__theta_q 
                                             >> 0xcU)));
    vlTOPp->e_packet[0xeU] = ((0x2fffffU & vlTOPp->e_packet[0xeU]) 
                              | (0xfff00000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__reverse_q) 
                                                << 0x14U)));
    vlTOPp->e_packet[0xeU] = ((0x1fffffU & vlTOPp->e_packet[0xeU]) 
                              | (0xffe00000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__error_q) 
                                                << 0x15U)));
    vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo 
        = (((QData)((IData)((1U & (vlTOPp->e_lo >> 0x1fU)))) 
            << 0x20U) | (QData)((IData)(vlTOPp->e_lo)));
    vlTOPp->packet_test_top__DOT__encoder__DOT__norm_hi 
        = (((QData)((IData)((1U & (vlTOPp->e_hi >> 0x1fU)))) 
            << 0x20U) | (QData)((IData)(vlTOPp->e_hi)));
    if (vlTOPp->packet_test_top__DOT__encoder__DOT__reverse_q) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo 
            = (VL_ULL(0x1ffffffff) & VL_NEGATE_Q((((QData)((IData)(
                                                                   (1U 
                                                                    & (vlTOPp->e_hi 
                                                                       >> 0x1fU)))) 
                                                   << 0x20U) 
                                                  | (QData)((IData)(vlTOPp->e_hi)))));
        vlTOPp->packet_test_top__DOT__encoder__DOT__norm_hi 
            = (VL_ULL(0x1ffffffff) & VL_NEGATE_Q((((QData)((IData)(
                                                                   (1U 
                                                                    & (vlTOPp->e_lo 
                                                                       >> 0x1fU)))) 
                                                   << 0x20U) 
                                                  | (QData)((IData)(vlTOPp->e_lo)))));
    }
    vlTOPp->packet_test_top__DOT__encoder__DOT__bit_now 
        = VL_GTES_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo, vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q);
    vlTOPp->packet_test_top__DOT__encoder__DOT__distance_wide = VL_ULL(0);
    if (VL_GTS_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo, vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q)) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__distance_wide 
            = (VL_ULL(0x3ffffffff) & ((((QData)((IData)(
                                                        (1U 
                                                         & (IData)(
                                                                   (vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo 
                                                                    >> 0x20U))))) 
                                        << 0x21U) | vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo) 
                                      - (((QData)((IData)(
                                                          (1U 
                                                           & (IData)(
                                                                     (vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q 
                                                                      >> 0x20U))))) 
                                          << 0x21U) 
                                         | vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q)));
    } else {
        if (VL_LTS_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__encoder__DOT__norm_hi, vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q)) {
            vlTOPp->packet_test_top__DOT__encoder__DOT__distance_wide 
                = (VL_ULL(0x3ffffffff) & ((((QData)((IData)(
                                                            (1U 
                                                             & (IData)(
                                                                       (vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q 
                                                                        >> 0x20U))))) 
                                            << 0x21U) 
                                           | vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q) 
                                          - (((QData)((IData)(
                                                              (1U 
                                                               & (IData)(
                                                                         (vlTOPp->packet_test_top__DOT__encoder__DOT__norm_hi 
                                                                          >> 0x20U))))) 
                                              << 0x21U) 
                                             | vlTOPp->packet_test_top__DOT__encoder__DOT__norm_hi)));
        }
    }
    vlTOPp->packet_test_top__DOT__encoder__DOT__distance_now 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__distance_wide;
    vlTOPp->packet_test_top__DOT__encoder__DOT__vacant = 0xffffffffU;
    vlTOPp->packet_test_top__DOT__encoder__DOT__worst = 0U;
    if ((1U & (~ (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q)))) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__vacant = 0U;
    }
    if (((vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
          [0U] > vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
          [0U]) | ((vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
                    [0U] == vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
                    [0U]) & (vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
                             [0U] > vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
                             [0U])))) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__worst = 0U;
    }
    if (((~ ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q) 
             >> 1U)) & VL_GTS_III(1,32,32, 0U, vlTOPp->packet_test_top__DOT__encoder__DOT__vacant))) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__vacant = 1U;
    }
    if (((vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
          [1U] > vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
          [vlTOPp->packet_test_top__DOT__encoder__DOT__worst]) 
         | ((vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
             [1U] == vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
             [vlTOPp->packet_test_top__DOT__encoder__DOT__worst]) 
            & (vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
               [1U] > vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
               [vlTOPp->packet_test_top__DOT__encoder__DOT__worst])))) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__worst = 1U;
    }
    if (((~ ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q) 
             >> 2U)) & VL_GTS_III(1,32,32, 0U, vlTOPp->packet_test_top__DOT__encoder__DOT__vacant))) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__vacant = 2U;
    }
    if (((vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
          [2U] > vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
          [vlTOPp->packet_test_top__DOT__encoder__DOT__worst]) 
         | ((vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
             [2U] == vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
             [vlTOPp->packet_test_top__DOT__encoder__DOT__worst]) 
            & (vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
               [2U] > vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
               [vlTOPp->packet_test_top__DOT__encoder__DOT__worst])))) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__worst = 2U;
    }
    if (((~ ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q) 
             >> 3U)) & VL_GTS_III(1,32,32, 0U, vlTOPp->packet_test_top__DOT__encoder__DOT__vacant))) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__vacant = 3U;
    }
    if (((vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
          [3U] > vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
          [vlTOPp->packet_test_top__DOT__encoder__DOT__worst]) 
         | ((vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
             [3U] == vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
             [vlTOPp->packet_test_top__DOT__encoder__DOT__worst]) 
            & (vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
               [3U] > vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
               [vlTOPp->packet_test_top__DOT__encoder__DOT__worst])))) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__worst = 3U;
    }
    vlTOPp->packet_test_top__DOT__encoder__DOT__chosen = 0U;
    vlTOPp->packet_test_top__DOT__encoder__DOT__do_keep = 0U;
    vlTOPp->packet_test_top__DOT__encoder__DOT__do_discard = 1U;
    vlTOPp->packet_test_top__DOT__encoder__DOT__discard_lo 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo;
    vlTOPp->packet_test_top__DOT__encoder__DOT__discard_hi 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__norm_hi;
    vlTOPp->packet_test_top__DOT__encoder__DOT__discard_bit 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__bit_now;
    if (VL_LTES_III(1,32,32, 0U, vlTOPp->packet_test_top__DOT__encoder__DOT__vacant)) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__chosen 
            = (3U & vlTOPp->packet_test_top__DOT__encoder__DOT__vacant);
        vlTOPp->packet_test_top__DOT__encoder__DOT__do_keep = 1U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__do_discard = 0U;
    } else {
        if ((vlTOPp->packet_test_top__DOT__encoder__DOT__distance_now 
             < vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
             [vlTOPp->packet_test_top__DOT__encoder__DOT__worst])) {
            vlTOPp->packet_test_top__DOT__encoder__DOT__chosen 
                = vlTOPp->packet_test_top__DOT__encoder__DOT__worst;
            vlTOPp->packet_test_top__DOT__encoder__DOT__do_keep = 1U;
            vlTOPp->packet_test_top__DOT__encoder__DOT__discard_lo 
                = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q
                [vlTOPp->packet_test_top__DOT__encoder__DOT__worst];
            vlTOPp->packet_test_top__DOT__encoder__DOT__discard_hi 
                = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q
                [vlTOPp->packet_test_top__DOT__encoder__DOT__worst];
            vlTOPp->packet_test_top__DOT__encoder__DOT__discard_bit 
                = (1U & (vlTOPp->packet_test_top__DOT__encoder__DOT__bits_q 
                         >> vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
                         [vlTOPp->packet_test_top__DOT__encoder__DOT__worst]));
        }
    }
}

VL_INLINE_OPT void Vpacket_test_top::_combo__TOP__3(Vpacket_test_top__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vpacket_test_top::_combo__TOP__3\n"); );
    Vpacket_test_top* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->packet_test_top__DOT__verifier__DOT__tau_lo 
        = (((QData)((IData)((1U & (vlTOPp->v_tau_lo 
                                   >> 0x1fU)))) << 0x20U) 
           | (QData)((IData)(vlTOPp->v_tau_lo)));
    vlTOPp->packet_test_top__DOT__verifier__DOT__tau_hi 
        = (((QData)((IData)((1U & (vlTOPp->v_tau_hi 
                                   >> 0x1fU)))) << 0x20U) 
           | (QData)((IData)(vlTOPp->v_tau_hi)));
    if ((0x100000U & vlTOPp->v_packet[0xeU])) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__tau_lo 
            = (VL_ULL(0x1ffffffff) & VL_NEGATE_Q((((QData)((IData)(
                                                                   (1U 
                                                                    & (vlTOPp->v_tau_hi 
                                                                       >> 0x1fU)))) 
                                                   << 0x20U) 
                                                  | (QData)((IData)(vlTOPp->v_tau_hi)))));
        vlTOPp->packet_test_top__DOT__verifier__DOT__tau_hi 
            = (VL_ULL(0x1ffffffff) & VL_NEGATE_Q((((QData)((IData)(
                                                                   (1U 
                                                                    & (vlTOPp->v_tau_lo 
                                                                       >> 0x1fU)))) 
                                                   << 0x20U) 
                                                  | (QData)((IData)(vlTOPp->v_tau_lo)))));
    }
    vlTOPp->packet_test_top__DOT__verifier__DOT__bad 
        = (1U & ((((vlTOPp->v_packet[0xeU] >> 0x15U) 
                   | (((vlTOPp->v_packet[0xdU] << 0x1cU) 
                       | (vlTOPp->v_packet[0xcU] >> 4U)) 
                      != vlTOPp->v_context)) | ((0xffffU 
                                                 & ((vlTOPp->v_packet[0xeU] 
                                                     << 0x1cU) 
                                                    | (vlTOPp->v_packet[0xdU] 
                                                       >> 4U))) 
                                                != (IData)(vlTOPp->v_epoch))) 
                 | VL_GTS_III(1,32,32, vlTOPp->v_tau_lo, vlTOPp->v_tau_hi)));
    vlTOPp->packet_test_top__DOT__verifier__DOT__a 
        = (VL_ULL(0x1ffffffff) & (((QData)((IData)(
                                                   vlTOPp->v_packet[2U])) 
                                   << 0x20U) | (QData)((IData)(
                                                               vlTOPp->v_packet[1U]))));
    vlTOPp->packet_test_top__DOT__verifier__DOT__b 
        = (VL_ULL(0x1ffffffff) & (((QData)((IData)(
                                                   vlTOPp->v_packet[4U])) 
                                   << 0x3fU) | (((QData)((IData)(
                                                                 vlTOPp->v_packet[3U])) 
                                                 << 0x1fU) 
                                                | ((QData)((IData)(
                                                                   vlTOPp->v_packet[2U])) 
                                                   >> 1U))));
    vlTOPp->packet_test_top__DOT__verifier__DOT__uncertain 
        = (((vlTOPp->v_packet[3U] >> 2U) & VL_GTES_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__a, vlTOPp->packet_test_top__DOT__verifier__DOT__tau_lo)) 
           | ((vlTOPp->v_packet[3U] >> 3U) & VL_GTS_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__tau_hi, vlTOPp->packet_test_top__DOT__verifier__DOT__b)));
    vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
        = vlTOPp->v_packet[0U];
    vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index = 0U;
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo 
        = (VL_ULL(0x1ffffffff) & (((QData)((IData)(
                                                   vlTOPp->v_packet[5U])) 
                                   << 0x3cU) | (((QData)((IData)(
                                                                 vlTOPp->v_packet[4U])) 
                                                 << 0x1cU) 
                                                | ((QData)((IData)(
                                                                   vlTOPp->v_packet[3U])) 
                                                   >> 4U))));
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi 
        = (VL_ULL(0x1ffffffff) & (((QData)((IData)(
                                                   vlTOPp->v_packet[6U])) 
                                   << 0x3bU) | (((QData)((IData)(
                                                                 vlTOPp->v_packet[5U])) 
                                                 << 0x1bU) 
                                                | ((QData)((IData)(
                                                                   vlTOPp->v_packet[4U])) 
                                                   >> 5U))));
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_index 
        = (0x1fU & ((vlTOPp->v_packet[6U] << 0x1aU) 
                    | (vlTOPp->v_packet[5U] >> 6U)));
    if ((1U & (~ (vlTOPp->v_packet[5U] >> 0xbU)))) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
    }
    if ((VL_LTES_III(1,32,32, 0x20U, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)) 
         | VL_GTS_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo, vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi))) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
    } else {
        if ((1U & (vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
                   >> (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)))) {
            vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
        }
        vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
            = (vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
               | ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)));
        if (VL_GTES_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo, vlTOPp->packet_test_top__DOT__verifier__DOT__tau_hi)) {
            vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                = (vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                   | ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)));
        } else {
            if (VL_LTS_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi, vlTOPp->packet_test_top__DOT__verifier__DOT__tau_lo)) {
                vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                    = ((~ ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index))) 
                       & vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits);
            } else {
                vlTOPp->packet_test_top__DOT__verifier__DOT__uncertain = 1U;
            }
        }
    }
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo 
        = (VL_ULL(0x1ffffffff) & (((QData)((IData)(
                                                   vlTOPp->v_packet[7U])) 
                                   << 0x34U) | (((QData)((IData)(
                                                                 vlTOPp->v_packet[6U])) 
                                                 << 0x14U) 
                                                | ((QData)((IData)(
                                                                   vlTOPp->v_packet[5U])) 
                                                   >> 0xcU))));
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi 
        = (VL_ULL(0x1ffffffff) & (((QData)((IData)(
                                                   vlTOPp->v_packet[8U])) 
                                   << 0x33U) | (((QData)((IData)(
                                                                 vlTOPp->v_packet[7U])) 
                                                 << 0x13U) 
                                                | ((QData)((IData)(
                                                                   vlTOPp->v_packet[6U])) 
                                                   >> 0xdU))));
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_index 
        = (0x1fU & ((vlTOPp->v_packet[8U] << 0x12U) 
                    | (vlTOPp->v_packet[7U] >> 0xeU)));
    if ((1U & (~ (vlTOPp->v_packet[7U] >> 0x13U)))) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
    }
    if ((VL_LTES_III(1,32,32, 0x20U, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)) 
         | VL_GTS_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo, vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi))) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
    } else {
        if ((1U & (vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
                   >> (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)))) {
            vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
        }
        vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
            = (vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
               | ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)));
        if (VL_GTES_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo, vlTOPp->packet_test_top__DOT__verifier__DOT__tau_hi)) {
            vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                = (vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                   | ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)));
        } else {
            if (VL_LTS_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi, vlTOPp->packet_test_top__DOT__verifier__DOT__tau_lo)) {
                vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                    = ((~ ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index))) 
                       & vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits);
            } else {
                vlTOPp->packet_test_top__DOT__verifier__DOT__uncertain = 1U;
            }
        }
    }
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo 
        = (VL_ULL(0x1ffffffff) & (((QData)((IData)(
                                                   vlTOPp->v_packet[9U])) 
                                   << 0x2cU) | (((QData)((IData)(
                                                                 vlTOPp->v_packet[8U])) 
                                                 << 0xcU) 
                                                | ((QData)((IData)(
                                                                   vlTOPp->v_packet[7U])) 
                                                   >> 0x14U))));
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi 
        = (VL_ULL(0x1ffffffff) & (((QData)((IData)(
                                                   vlTOPp->v_packet[0xaU])) 
                                   << 0x2bU) | (((QData)((IData)(
                                                                 vlTOPp->v_packet[9U])) 
                                                 << 0xbU) 
                                                | ((QData)((IData)(
                                                                   vlTOPp->v_packet[8U])) 
                                                   >> 0x15U))));
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_index 
        = (0x1fU & ((vlTOPp->v_packet[0xaU] << 0xaU) 
                    | (vlTOPp->v_packet[9U] >> 0x16U)));
    if ((1U & (~ (vlTOPp->v_packet[9U] >> 0x1bU)))) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
    }
    if ((VL_LTES_III(1,32,32, 0x20U, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)) 
         | VL_GTS_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo, vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi))) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
    } else {
        if ((1U & (vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
                   >> (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)))) {
            vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
        }
        vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
            = (vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
               | ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)));
        if (VL_GTES_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo, vlTOPp->packet_test_top__DOT__verifier__DOT__tau_hi)) {
            vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                = (vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                   | ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)));
        } else {
            if (VL_LTS_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi, vlTOPp->packet_test_top__DOT__verifier__DOT__tau_lo)) {
                vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                    = ((~ ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index))) 
                       & vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits);
            } else {
                vlTOPp->packet_test_top__DOT__verifier__DOT__uncertain = 1U;
            }
        }
    }
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo 
        = (VL_ULL(0x1ffffffff) & (((QData)((IData)(
                                                   vlTOPp->v_packet[0xbU])) 
                                   << 0x24U) | (((QData)((IData)(
                                                                 vlTOPp->v_packet[0xaU])) 
                                                 << 4U) 
                                                | ((QData)((IData)(
                                                                   vlTOPp->v_packet[9U])) 
                                                   >> 0x1cU))));
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi 
        = (VL_ULL(0x1ffffffff) & (((QData)((IData)(
                                                   vlTOPp->v_packet[0xcU])) 
                                   << 0x23U) | (((QData)((IData)(
                                                                 vlTOPp->v_packet[0xbU])) 
                                                 << 3U) 
                                                | ((QData)((IData)(
                                                                   vlTOPp->v_packet[0xaU])) 
                                                   >> 0x1dU))));
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_index 
        = (0x1fU & ((vlTOPp->v_packet[0xcU] << 2U) 
                    | (vlTOPp->v_packet[0xbU] >> 0x1eU)));
    if ((1U & (~ (vlTOPp->v_packet[0xcU] >> 3U)))) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
    }
    if ((VL_LTES_III(1,32,32, 0x20U, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)) 
         | VL_GTS_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo, vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi))) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
    } else {
        if ((1U & (vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
                   >> (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)))) {
            vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
        }
        vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
            = (vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
               | ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)));
        if (VL_GTES_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo, vlTOPp->packet_test_top__DOT__verifier__DOT__tau_hi)) {
            vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                = (vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                   | ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)));
        } else {
            if (VL_LTS_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi, vlTOPp->packet_test_top__DOT__verifier__DOT__tau_lo)) {
                vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                    = ((~ ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index))) 
                       & vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits);
            } else {
                vlTOPp->packet_test_top__DOT__verifier__DOT__uncertain = 1U;
            }
        }
    }
    vlTOPp->packet_test_top__DOT__verifier__DOT__status_next 
        = ((IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__bad)
            ? 2U : ((IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__uncertain)
                     ? 1U : 0U));
    vlTOPp->v_ready = (1U & ((~ (IData)(vlTOPp->v_result_valid)) 
                             | (IData)(vlTOPp->v_result_ready)));
    vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo 
        = (((QData)((IData)((1U & (vlTOPp->e_lo >> 0x1fU)))) 
            << 0x20U) | (QData)((IData)(vlTOPp->e_lo)));
    vlTOPp->packet_test_top__DOT__encoder__DOT__norm_hi 
        = (((QData)((IData)((1U & (vlTOPp->e_hi >> 0x1fU)))) 
            << 0x20U) | (QData)((IData)(vlTOPp->e_hi)));
    if (vlTOPp->packet_test_top__DOT__encoder__DOT__reverse_q) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo 
            = (VL_ULL(0x1ffffffff) & VL_NEGATE_Q((((QData)((IData)(
                                                                   (1U 
                                                                    & (vlTOPp->e_hi 
                                                                       >> 0x1fU)))) 
                                                   << 0x20U) 
                                                  | (QData)((IData)(vlTOPp->e_hi)))));
        vlTOPp->packet_test_top__DOT__encoder__DOT__norm_hi 
            = (VL_ULL(0x1ffffffff) & VL_NEGATE_Q((((QData)((IData)(
                                                                   (1U 
                                                                    & (vlTOPp->e_lo 
                                                                       >> 0x1fU)))) 
                                                   << 0x20U) 
                                                  | (QData)((IData)(vlTOPp->e_lo)))));
    }
    vlTOPp->packet_test_top__DOT__encoder__DOT__bit_now 
        = VL_GTES_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo, vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q);
    vlTOPp->packet_test_top__DOT__encoder__DOT__distance_wide = VL_ULL(0);
    if (VL_GTS_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo, vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q)) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__distance_wide 
            = (VL_ULL(0x3ffffffff) & ((((QData)((IData)(
                                                        (1U 
                                                         & (IData)(
                                                                   (vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo 
                                                                    >> 0x20U))))) 
                                        << 0x21U) | vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo) 
                                      - (((QData)((IData)(
                                                          (1U 
                                                           & (IData)(
                                                                     (vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q 
                                                                      >> 0x20U))))) 
                                          << 0x21U) 
                                         | vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q)));
    } else {
        if (VL_LTS_IQQ(1,33,33, vlTOPp->packet_test_top__DOT__encoder__DOT__norm_hi, vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q)) {
            vlTOPp->packet_test_top__DOT__encoder__DOT__distance_wide 
                = (VL_ULL(0x3ffffffff) & ((((QData)((IData)(
                                                            (1U 
                                                             & (IData)(
                                                                       (vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q 
                                                                        >> 0x20U))))) 
                                            << 0x21U) 
                                           | vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q) 
                                          - (((QData)((IData)(
                                                              (1U 
                                                               & (IData)(
                                                                         (vlTOPp->packet_test_top__DOT__encoder__DOT__norm_hi 
                                                                          >> 0x20U))))) 
                                              << 0x21U) 
                                             | vlTOPp->packet_test_top__DOT__encoder__DOT__norm_hi)));
        }
    }
    vlTOPp->packet_test_top__DOT__encoder__DOT__distance_now 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__distance_wide;
    vlTOPp->packet_test_top__DOT__encoder__DOT__vacant = 0xffffffffU;
    vlTOPp->packet_test_top__DOT__encoder__DOT__worst = 0U;
    if ((1U & (~ (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q)))) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__vacant = 0U;
    }
    if (((vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
          [0U] > vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
          [0U]) | ((vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
                    [0U] == vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
                    [0U]) & (vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
                             [0U] > vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
                             [0U])))) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__worst = 0U;
    }
    if (((~ ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q) 
             >> 1U)) & VL_GTS_III(1,32,32, 0U, vlTOPp->packet_test_top__DOT__encoder__DOT__vacant))) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__vacant = 1U;
    }
    if (((vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
          [1U] > vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
          [vlTOPp->packet_test_top__DOT__encoder__DOT__worst]) 
         | ((vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
             [1U] == vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
             [vlTOPp->packet_test_top__DOT__encoder__DOT__worst]) 
            & (vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
               [1U] > vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
               [vlTOPp->packet_test_top__DOT__encoder__DOT__worst])))) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__worst = 1U;
    }
    if (((~ ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q) 
             >> 2U)) & VL_GTS_III(1,32,32, 0U, vlTOPp->packet_test_top__DOT__encoder__DOT__vacant))) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__vacant = 2U;
    }
    if (((vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
          [2U] > vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
          [vlTOPp->packet_test_top__DOT__encoder__DOT__worst]) 
         | ((vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
             [2U] == vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
             [vlTOPp->packet_test_top__DOT__encoder__DOT__worst]) 
            & (vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
               [2U] > vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
               [vlTOPp->packet_test_top__DOT__encoder__DOT__worst])))) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__worst = 2U;
    }
    if (((~ ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q) 
             >> 3U)) & VL_GTS_III(1,32,32, 0U, vlTOPp->packet_test_top__DOT__encoder__DOT__vacant))) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__vacant = 3U;
    }
    if (((vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
          [3U] > vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
          [vlTOPp->packet_test_top__DOT__encoder__DOT__worst]) 
         | ((vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
             [3U] == vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
             [vlTOPp->packet_test_top__DOT__encoder__DOT__worst]) 
            & (vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
               [3U] > vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
               [vlTOPp->packet_test_top__DOT__encoder__DOT__worst])))) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__worst = 3U;
    }
    vlTOPp->packet_test_top__DOT__encoder__DOT__chosen = 0U;
    vlTOPp->packet_test_top__DOT__encoder__DOT__do_keep = 0U;
    vlTOPp->packet_test_top__DOT__encoder__DOT__do_discard = 1U;
    vlTOPp->packet_test_top__DOT__encoder__DOT__discard_lo 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo;
    vlTOPp->packet_test_top__DOT__encoder__DOT__discard_hi 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__norm_hi;
    vlTOPp->packet_test_top__DOT__encoder__DOT__discard_bit 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__bit_now;
    if (VL_LTES_III(1,32,32, 0U, vlTOPp->packet_test_top__DOT__encoder__DOT__vacant)) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__chosen 
            = (3U & vlTOPp->packet_test_top__DOT__encoder__DOT__vacant);
        vlTOPp->packet_test_top__DOT__encoder__DOT__do_keep = 1U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__do_discard = 0U;
    } else {
        if ((vlTOPp->packet_test_top__DOT__encoder__DOT__distance_now 
             < vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q
             [vlTOPp->packet_test_top__DOT__encoder__DOT__worst])) {
            vlTOPp->packet_test_top__DOT__encoder__DOT__chosen 
                = vlTOPp->packet_test_top__DOT__encoder__DOT__worst;
            vlTOPp->packet_test_top__DOT__encoder__DOT__do_keep = 1U;
            vlTOPp->packet_test_top__DOT__encoder__DOT__discard_lo 
                = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q
                [vlTOPp->packet_test_top__DOT__encoder__DOT__worst];
            vlTOPp->packet_test_top__DOT__encoder__DOT__discard_hi 
                = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q
                [vlTOPp->packet_test_top__DOT__encoder__DOT__worst];
            vlTOPp->packet_test_top__DOT__encoder__DOT__discard_bit 
                = (1U & (vlTOPp->packet_test_top__DOT__encoder__DOT__bits_q 
                         >> vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
                         [vlTOPp->packet_test_top__DOT__encoder__DOT__worst]));
        }
    }
}

void Vpacket_test_top::_eval(Vpacket_test_top__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vpacket_test_top::_eval\n"); );
    Vpacket_test_top* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    if (((IData)(vlTOPp->clk) & (~ (IData)(vlTOPp->__Vclklast__TOP__clk)))) {
        vlTOPp->_sequent__TOP__1(vlSymsp);
    }
    vlTOPp->_combo__TOP__3(vlSymsp);
    // Final
    vlTOPp->__Vclklast__TOP__clk = vlTOPp->clk;
}

void Vpacket_test_top::_eval_initial(Vpacket_test_top__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vpacket_test_top::_eval_initial\n"); );
    Vpacket_test_top* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->__Vclklast__TOP__clk = vlTOPp->clk;
}

void Vpacket_test_top::final() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vpacket_test_top::final\n"); );
    // Variables
    Vpacket_test_top__Syms* __restrict vlSymsp = this->__VlSymsp;
    Vpacket_test_top* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

void Vpacket_test_top::_eval_settle(Vpacket_test_top__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vpacket_test_top::_eval_settle\n"); );
    Vpacket_test_top* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->_settle__TOP__2(vlSymsp);
}

VL_INLINE_OPT QData Vpacket_test_top::_change_request(Vpacket_test_top__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vpacket_test_top::_change_request\n"); );
    Vpacket_test_top* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // Change detection
    QData __req = false;  // Logically a bool
    return __req;
}

#ifdef VL_DEBUG
void Vpacket_test_top::_eval_debug_assertions() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vpacket_test_top::_eval_debug_assertions\n"); );
    // Body
    if (VL_UNLIKELY((clk & 0xfeU))) {
        Verilated::overWidthError("clk");}
    if (VL_UNLIKELY((rst_n & 0xfeU))) {
        Verilated::overWidthError("rst_n");}
    if (VL_UNLIKELY((e_start_valid & 0xfeU))) {
        Verilated::overWidthError("e_start_valid");}
    if (VL_UNLIKELY((e_reverse & 0xfeU))) {
        Verilated::overWidthError("e_reverse");}
    if (VL_UNLIKELY((e_in_valid & 0xfeU))) {
        Verilated::overWidthError("e_in_valid");}
    if (VL_UNLIKELY((e_packet_ready & 0xfeU))) {
        Verilated::overWidthError("e_packet_ready");}
    if (VL_UNLIKELY((v_valid & 0xfeU))) {
        Verilated::overWidthError("v_valid");}
    if (VL_UNLIKELY((v_packet[0xeU] & 0xffc00000U))) {
        Verilated::overWidthError("v_packet");}
    if (VL_UNLIKELY((v_result_ready & 0xfeU))) {
        Verilated::overWidthError("v_result_ready");}
    if (VL_UNLIKELY((g_start_valid & 0xfeU))) {
        Verilated::overWidthError("g_start_valid");}
    if (VL_UNLIKELY((g_packet_valid & 0xfeU))) {
        Verilated::overWidthError("g_packet_valid");}
    if (VL_UNLIKELY((g_packet_status & 0xfcU))) {
        Verilated::overWidthError("g_packet_status");}
    if (VL_UNLIKELY((g_replay_ready & 0xfeU))) {
        Verilated::overWidthError("g_replay_ready");}
    if (VL_UNLIKELY((g_repair_valid & 0xfeU))) {
        Verilated::overWidthError("g_repair_valid");}
    if (VL_UNLIKELY((g_result_ready & 0xfeU))) {
        Verilated::overWidthError("g_result_ready");}
}
#endif  // VL_DEBUG

void Vpacket_test_top::_ctor_var_reset() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vpacket_test_top::_ctor_var_reset\n"); );
    // Body
    clk = VL_RAND_RESET_I(1);
    rst_n = VL_RAND_RESET_I(1);
    e_start_valid = VL_RAND_RESET_I(1);
    e_start_ready = VL_RAND_RESET_I(1);
    e_context = VL_RAND_RESET_I(32);
    e_theta = VL_RAND_RESET_I(32);
    e_epoch = VL_RAND_RESET_I(16);
    e_reverse = VL_RAND_RESET_I(1);
    e_prediction = VL_RAND_RESET_I(32);
    e_in_valid = VL_RAND_RESET_I(1);
    e_in_ready = VL_RAND_RESET_I(1);
    e_lo = VL_RAND_RESET_I(32);
    e_hi = VL_RAND_RESET_I(32);
    e_packet_valid = VL_RAND_RESET_I(1);
    e_packet_ready = VL_RAND_RESET_I(1);
    VL_RAND_RESET_W(470, e_packet);
    v_valid = VL_RAND_RESET_I(1);
    v_ready = VL_RAND_RESET_I(1);
    VL_RAND_RESET_W(470, v_packet);
    v_context = VL_RAND_RESET_I(32);
    v_epoch = VL_RAND_RESET_I(16);
    v_tau_lo = VL_RAND_RESET_I(32);
    v_tau_hi = VL_RAND_RESET_I(32);
    v_result_valid = VL_RAND_RESET_I(1);
    v_result_ready = VL_RAND_RESET_I(1);
    v_status = VL_RAND_RESET_I(2);
    v_bits = VL_RAND_RESET_I(32);
    v_theta = VL_RAND_RESET_I(32);
    v_out_context = VL_RAND_RESET_I(32);
    v_out_epoch = VL_RAND_RESET_I(16);
    g_start_valid = VL_RAND_RESET_I(1);
    g_start_ready = VL_RAND_RESET_I(1);
    g_base = VL_RAND_RESET_I(32);
    g_theta = VL_RAND_RESET_I(32);
    g_epoch = VL_RAND_RESET_I(16);
    g_packet_valid = VL_RAND_RESET_I(1);
    g_packet_ready = VL_RAND_RESET_I(1);
    g_packet_status = VL_RAND_RESET_I(2);
    g_packet_bits = VL_RAND_RESET_I(32);
    g_packet_context = VL_RAND_RESET_I(32);
    g_packet_theta = VL_RAND_RESET_I(32);
    g_packet_epoch = VL_RAND_RESET_I(16);
    g_replay_valid = VL_RAND_RESET_I(1);
    g_replay_ready = VL_RAND_RESET_I(1);
    g_replay_mask = VL_RAND_RESET_I(10);
    g_repair_valid = VL_RAND_RESET_I(1);
    g_repair_ready = VL_RAND_RESET_I(1);
    VL_RAND_RESET_W(320, g_repair_bits);
    g_repair_context = VL_RAND_RESET_I(32);
    g_repair_theta = VL_RAND_RESET_I(32);
    g_repair_epoch = VL_RAND_RESET_I(16);
    g_result_valid = VL_RAND_RESET_I(1);
    g_result_ready = VL_RAND_RESET_I(1);
    g_status = VL_RAND_RESET_I(2);
    VL_RAND_RESET_W(320, g_bits);
    g_out_context = VL_RAND_RESET_I(32);
    g_out_theta = VL_RAND_RESET_I(32);
    g_out_epoch = VL_RAND_RESET_I(16);
    packet_test_top__DOT__encoder__DOT__state_q = VL_RAND_RESET_I(2);
    packet_test_top__DOT__encoder__DOT__count_q = VL_RAND_RESET_I(6);
    packet_test_top__DOT__encoder__DOT__context_q = VL_RAND_RESET_I(32);
    packet_test_top__DOT__encoder__DOT__theta_q = VL_RAND_RESET_I(32);
    packet_test_top__DOT__encoder__DOT__epoch_q = VL_RAND_RESET_I(16);
    packet_test_top__DOT__encoder__DOT__reverse_q = VL_RAND_RESET_I(1);
    packet_test_top__DOT__encoder__DOT__error_q = VL_RAND_RESET_I(1);
    packet_test_top__DOT__encoder__DOT__prediction_q = VL_RAND_RESET_Q(33);
    packet_test_top__DOT__encoder__DOT__bits_q = VL_RAND_RESET_I(32);
    packet_test_top__DOT__encoder__DOT__a_q = VL_RAND_RESET_Q(33);
    packet_test_top__DOT__encoder__DOT__b_q = VL_RAND_RESET_Q(33);
    packet_test_top__DOT__encoder__DOT__a_valid_q = VL_RAND_RESET_I(1);
    packet_test_top__DOT__encoder__DOT__b_valid_q = VL_RAND_RESET_I(1);
    packet_test_top__DOT__encoder__DOT__kept_valid_q = VL_RAND_RESET_I(4);
    { int __Vi0=0; for (; __Vi0<4; ++__Vi0) {
            packet_test_top__DOT__encoder__DOT__kept_lo_q[__Vi0] = VL_RAND_RESET_Q(33);
    }}
    { int __Vi0=0; for (; __Vi0<4; ++__Vi0) {
            packet_test_top__DOT__encoder__DOT__kept_hi_q[__Vi0] = VL_RAND_RESET_Q(33);
    }}
    { int __Vi0=0; for (; __Vi0<4; ++__Vi0) {
            packet_test_top__DOT__encoder__DOT__kept_index_q[__Vi0] = VL_RAND_RESET_I(5);
    }}
    { int __Vi0=0; for (; __Vi0<4; ++__Vi0) {
            packet_test_top__DOT__encoder__DOT__kept_distance_q[__Vi0] = VL_RAND_RESET_Q(34);
    }}
    packet_test_top__DOT__encoder__DOT__norm_lo = VL_RAND_RESET_Q(33);
    packet_test_top__DOT__encoder__DOT__norm_hi = VL_RAND_RESET_Q(33);
    packet_test_top__DOT__encoder__DOT__distance_wide = VL_RAND_RESET_Q(34);
    packet_test_top__DOT__encoder__DOT__distance_now = VL_RAND_RESET_Q(34);
    packet_test_top__DOT__encoder__DOT__bit_now = VL_RAND_RESET_I(1);
    packet_test_top__DOT__encoder__DOT__do_keep = VL_RAND_RESET_I(1);
    packet_test_top__DOT__encoder__DOT__do_discard = VL_RAND_RESET_I(1);
    packet_test_top__DOT__encoder__DOT__discard_bit = VL_RAND_RESET_I(1);
    packet_test_top__DOT__encoder__DOT__discard_lo = VL_RAND_RESET_Q(33);
    packet_test_top__DOT__encoder__DOT__discard_hi = VL_RAND_RESET_Q(33);
    packet_test_top__DOT__encoder__DOT__chosen = VL_RAND_RESET_I(2);
    packet_test_top__DOT__encoder__DOT__worst = VL_RAND_RESET_I(2);
    packet_test_top__DOT__encoder__DOT__vacant = VL_RAND_RESET_I(32);
    packet_test_top__DOT__encoder__DOT____Vlvbound1 = VL_RAND_RESET_Q(33);
    packet_test_top__DOT__encoder__DOT____Vlvbound2 = VL_RAND_RESET_Q(33);
    packet_test_top__DOT__encoder__DOT____Vlvbound3 = VL_RAND_RESET_I(5);
    packet_test_top__DOT__encoder__DOT____Vlvbound4 = VL_RAND_RESET_I(1);
    packet_test_top__DOT__verifier__DOT__bad = VL_RAND_RESET_I(1);
    packet_test_top__DOT__verifier__DOT__uncertain = VL_RAND_RESET_I(1);
    packet_test_top__DOT__verifier__DOT__final_bits = VL_RAND_RESET_I(32);
    packet_test_top__DOT__verifier__DOT__seen_index = VL_RAND_RESET_I(32);
    packet_test_top__DOT__verifier__DOT__tau_lo = VL_RAND_RESET_Q(33);
    packet_test_top__DOT__verifier__DOT__tau_hi = VL_RAND_RESET_Q(33);
    packet_test_top__DOT__verifier__DOT__a = VL_RAND_RESET_Q(33);
    packet_test_top__DOT__verifier__DOT__b = VL_RAND_RESET_Q(33);
    packet_test_top__DOT__verifier__DOT__value_lo = VL_RAND_RESET_Q(33);
    packet_test_top__DOT__verifier__DOT__value_hi = VL_RAND_RESET_Q(33);
    packet_test_top__DOT__verifier__DOT__value_index = VL_RAND_RESET_I(5);
    packet_test_top__DOT__verifier__DOT__status_next = VL_RAND_RESET_I(2);
    packet_test_top__DOT__group_commit__DOT__state_q = VL_RAND_RESET_I(3);
    packet_test_top__DOT__group_commit__DOT__count_q = VL_RAND_RESET_I(5);
    packet_test_top__DOT__group_commit__DOT__error_q = VL_RAND_RESET_I(1);
    VL_RAND_RESET_W(320, packet_test_top__DOT__group_commit__DOT__bits_q);
    packet_test_top__DOT__group_commit__DOT__replay_q = VL_RAND_RESET_I(10);
    packet_test_top__DOT__group_commit__DOT____Vlvbound1 = VL_RAND_RESET_I(1);
    packet_test_top__DOT__group_commit__DOT____Vlvbound2 = VL_RAND_RESET_I(32);
}
