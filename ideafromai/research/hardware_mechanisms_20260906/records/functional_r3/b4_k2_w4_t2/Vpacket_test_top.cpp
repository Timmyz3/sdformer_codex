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
    CData/*2:0*/ __Vdly__packet_test_top__DOT__encoder__DOT__count_q;
    CData/*4:0*/ __Vdly__packet_test_top__DOT__encoder__DOT__a_q;
    CData/*4:0*/ __Vdly__packet_test_top__DOT__encoder__DOT__b_q;
    CData/*0:0*/ __Vdly__packet_test_top__DOT__encoder__DOT__a_valid_q;
    CData/*0:0*/ __Vdly__packet_test_top__DOT__encoder__DOT__b_valid_q;
    CData/*0:0*/ __Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v0;
    CData/*0:0*/ __Vdlyvdim0__packet_test_top__DOT__encoder__DOT__kept_lo_q__v2;
    CData/*4:0*/ __Vdlyvval__packet_test_top__DOT__encoder__DOT__kept_lo_q__v2;
    CData/*0:0*/ __Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v2;
    CData/*0:0*/ __Vdlyvdim0__packet_test_top__DOT__encoder__DOT__kept_hi_q__v2;
    CData/*4:0*/ __Vdlyvval__packet_test_top__DOT__encoder__DOT__kept_hi_q__v2;
    CData/*0:0*/ __Vdlyvdim0__packet_test_top__DOT__encoder__DOT__kept_index_q__v2;
    CData/*1:0*/ __Vdlyvval__packet_test_top__DOT__encoder__DOT__kept_index_q__v2;
    CData/*0:0*/ __Vdlyvdim0__packet_test_top__DOT__encoder__DOT__kept_distance_q__v2;
    CData/*5:0*/ __Vdlyvval__packet_test_top__DOT__encoder__DOT__kept_distance_q__v2;
    CData/*0:0*/ __Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v3;
    CData/*2:0*/ __Vdly__packet_test_top__DOT__group_commit__DOT__state_q;
    CData/*0:0*/ __Vdly__packet_test_top__DOT__group_commit__DOT__error_q;
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
    __Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v2 = 0U;
    __Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v3 = 0U;
    if (vlTOPp->rst_n) {
        if (vlTOPp->v_ready) {
            if (vlTOPp->v_valid) {
                vlTOPp->v_out_epoch = (0xffffU & ((
                                                   vlTOPp->v_packet[3U] 
                                                   << 0x16U) 
                                                  | (vlTOPp->v_packet[2U] 
                                                     >> 0xaU)));
            }
        }
    } else {
        vlTOPp->v_out_epoch = 0U;
    }
    if (vlTOPp->rst_n) {
        if (vlTOPp->v_ready) {
            if (vlTOPp->v_valid) {
                vlTOPp->v_theta = ((vlTOPp->v_packet[3U] 
                                    << 6U) | (vlTOPp->v_packet[2U] 
                                              >> 0x1aU));
            }
        }
    } else {
        vlTOPp->v_theta = 0U;
    }
    if (vlTOPp->rst_n) {
        if (vlTOPp->v_ready) {
            if (vlTOPp->v_valid) {
                vlTOPp->v_out_context = ((vlTOPp->v_packet[2U] 
                                          << 0x16U) 
                                         | (vlTOPp->v_packet[1U] 
                                            >> 0xaU));
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
                                   ? (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits)
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
                        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q 
                            = vlTOPp->g_repair_bits;
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
                            vlTOPp->packet_test_top__DOT__group_commit__DOT__replay_q 
                                = ((IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__replay_q) 
                                   | ((IData)(1U) << 
                                      (1U & (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__count_q))));
                        }
                        if ((0U == (IData)(vlTOPp->g_packet_status))) {
                            vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q 
                                = (((~ ((IData)(0xfU) 
                                        << (7U & VL_MULS_III(3,32,32, (IData)(4U), (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__count_q))))) 
                                    & (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q)) 
                                   | ((IData)(vlTOPp->g_packet_bits) 
                                      << (7U & VL_MULS_III(3,32,32, (IData)(4U), (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__count_q)))));
                        }
                        if ((1U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__count_q))) {
                            __Vdly__packet_test_top__DOT__group_commit__DOT__state_q = 2U;
                        } else {
                            vlTOPp->packet_test_top__DOT__group_commit__DOT__count_q 
                                = (0x1fU & ((IData)(1U) 
                                            + (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__count_q)));
                        }
                    }
                } else {
                    if (vlTOPp->g_start_valid) {
                        vlTOPp->packet_test_top__DOT__group_commit__DOT__count_q = 0U;
                        __Vdly__packet_test_top__DOT__group_commit__DOT__state_q = 1U;
                        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q = 0U;
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
        vlTOPp->packet_test_top__DOT__group_commit__DOT__count_q = 0U;
        __Vdly__packet_test_top__DOT__group_commit__DOT__state_q = 0U;
        __Vdly__packet_test_top__DOT__group_commit__DOT__error_q = 0U;
        vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q = 0U;
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
                    = (0x1fU & ((IData)(vlTOPp->e_reverse)
                                 ? VL_NEGATE_I(((0x10U 
                                                 & ((IData)(vlTOPp->e_prediction) 
                                                    << 1U)) 
                                                | (IData)(vlTOPp->e_prediction)))
                                 : ((0x10U & ((IData)(vlTOPp->e_prediction) 
                                              << 1U)) 
                                    | (IData)(vlTOPp->e_prediction))));
                vlTOPp->packet_test_top__DOT__encoder__DOT__error_q = 0U;
                vlTOPp->packet_test_top__DOT__encoder__DOT__bits_q = 0U;
                __Vdly__packet_test_top__DOT__encoder__DOT__a_q = 0U;
                __Vdly__packet_test_top__DOT__encoder__DOT__b_q = 0U;
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
                                       | VL_LTS_III(1,5,5, (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__discard_lo), (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__b_q))))) {
                                __Vdly__packet_test_top__DOT__encoder__DOT__b_q 
                                    = vlTOPp->packet_test_top__DOT__encoder__DOT__discard_lo;
                            }
                            __Vdly__packet_test_top__DOT__encoder__DOT__b_valid_q = 1U;
                        } else {
                            if ((1U & ((~ (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__a_valid_q)) 
                                       | VL_GTS_III(1,5,5, (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__discard_hi), (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__a_q))))) {
                                __Vdly__packet_test_top__DOT__encoder__DOT__a_q 
                                    = vlTOPp->packet_test_top__DOT__encoder__DOT__discard_hi;
                            }
                            __Vdly__packet_test_top__DOT__encoder__DOT__a_valid_q = 1U;
                        }
                    }
                    if (VL_GTS_III(1,4,4, (IData)(vlTOPp->e_lo), (IData)(vlTOPp->e_hi))) {
                        vlTOPp->packet_test_top__DOT__encoder__DOT__error_q = 1U;
                    }
                    vlTOPp->packet_test_top__DOT__encoder__DOT__bits_q 
                        = (((~ ((IData)(1U) << (3U 
                                                & (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__count_q)))) 
                            & (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__bits_q)) 
                           | ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__bit_now) 
                              << (3U & (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__count_q))));
                    if (vlTOPp->packet_test_top__DOT__encoder__DOT__do_keep) {
                        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q 
                            = ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q) 
                               | ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__chosen)));
                        __Vdlyvval__packet_test_top__DOT__encoder__DOT__kept_lo_q__v2 
                            = vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo;
                        __Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v2 = 1U;
                        __Vdlyvdim0__packet_test_top__DOT__encoder__DOT__kept_lo_q__v2 
                            = vlTOPp->packet_test_top__DOT__encoder__DOT__chosen;
                        __Vdlyvval__packet_test_top__DOT__encoder__DOT__kept_hi_q__v2 
                            = vlTOPp->packet_test_top__DOT__encoder__DOT__norm_hi;
                        __Vdlyvdim0__packet_test_top__DOT__encoder__DOT__kept_hi_q__v2 
                            = vlTOPp->packet_test_top__DOT__encoder__DOT__chosen;
                        __Vdlyvval__packet_test_top__DOT__encoder__DOT__kept_index_q__v2 
                            = (3U & (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__count_q));
                        __Vdlyvdim0__packet_test_top__DOT__encoder__DOT__kept_index_q__v2 
                            = vlTOPp->packet_test_top__DOT__encoder__DOT__chosen;
                        __Vdlyvval__packet_test_top__DOT__encoder__DOT__kept_distance_q__v2 
                            = vlTOPp->packet_test_top__DOT__encoder__DOT__distance_now;
                        __Vdlyvdim0__packet_test_top__DOT__encoder__DOT__kept_distance_q__v2 
                            = vlTOPp->packet_test_top__DOT__encoder__DOT__chosen;
                    }
                    if ((3U == (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__count_q))) {
                        __Vdly__packet_test_top__DOT__encoder__DOT__state_q = 2U;
                    } else {
                        __Vdly__packet_test_top__DOT__encoder__DOT__count_q 
                            = (7U & ((IData)(1U) + (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__count_q)));
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
        vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q = 0U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__bits_q = 0U;
        __Vdly__packet_test_top__DOT__encoder__DOT__a_q = 0U;
        __Vdly__packet_test_top__DOT__encoder__DOT__b_q = 0U;
        __Vdly__packet_test_top__DOT__encoder__DOT__a_valid_q = 0U;
        __Vdly__packet_test_top__DOT__encoder__DOT__b_valid_q = 0U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q = 0U;
        __Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v3 = 1U;
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
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q[0U] = 0U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q[1U] = 0U;
    }
    if (__Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v2) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q[__Vdlyvdim0__packet_test_top__DOT__encoder__DOT__kept_distance_q__v2] 
            = __Vdlyvval__packet_test_top__DOT__encoder__DOT__kept_distance_q__v2;
    }
    if (__Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v3) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q[0U] = 0U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_distance_q[1U] = 0U;
    }
    if (__Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v0) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q[0U] = 0U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q[1U] = 0U;
    }
    if (__Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v2) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q[__Vdlyvdim0__packet_test_top__DOT__encoder__DOT__kept_index_q__v2] 
            = __Vdlyvval__packet_test_top__DOT__encoder__DOT__kept_index_q__v2;
    }
    if (__Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v3) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q[0U] = 0U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q[1U] = 0U;
    }
    if (__Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v0) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q[0U] = 0U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q[1U] = 0U;
    }
    if (__Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v2) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q[__Vdlyvdim0__packet_test_top__DOT__encoder__DOT__kept_hi_q__v2] 
            = __Vdlyvval__packet_test_top__DOT__encoder__DOT__kept_hi_q__v2;
    }
    if (__Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v3) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q[0U] = 0U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q[1U] = 0U;
    }
    if (__Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v0) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q[0U] = 0U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q[1U] = 0U;
    }
    if (__Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v2) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q[__Vdlyvdim0__packet_test_top__DOT__encoder__DOT__kept_lo_q__v2] 
            = __Vdlyvval__packet_test_top__DOT__encoder__DOT__kept_lo_q__v2;
    }
    if (__Vdlyvset__packet_test_top__DOT__encoder__DOT__kept_lo_q__v3) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q[0U] = 0U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q[1U] = 0U;
    }
    vlTOPp->g_replay_mask = vlTOPp->packet_test_top__DOT__group_commit__DOT__replay_q;
    vlTOPp->g_status = ((IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q)
                         ? 2U : 0U);
    vlTOPp->g_start_ready = (0U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q));
    vlTOPp->g_packet_ready = (1U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q));
    vlTOPp->g_replay_valid = (3U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q));
    vlTOPp->g_repair_ready = (4U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q));
    vlTOPp->g_result_valid = (5U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q));
    vlTOPp->g_bits = (((5U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q)) 
                       & (~ (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q)))
                       ? (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q)
                       : 0U);
    vlTOPp->e_start_ready = (0U == (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__state_q));
    vlTOPp->e_in_ready = (1U == (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__state_q));
    vlTOPp->e_packet_valid = (2U == (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__state_q));
    vlTOPp->e_packet[0U] = 0U;
    vlTOPp->e_packet[1U] = 0U;
    vlTOPp->e_packet[2U] = 0U;
    vlTOPp->e_packet[3U] = 0U;
    vlTOPp->e_packet[0U] = ((0xfffffff0U & vlTOPp->e_packet[0U]) 
                            | (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__bits_q));
    vlTOPp->e_packet[0U] = ((0xfffffe0fU & vlTOPp->e_packet[0U]) 
                            | (0xfffffff0U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__a_q) 
                                              << 4U)));
    vlTOPp->e_packet[0U] = ((0xffffc1ffU & vlTOPp->e_packet[0U]) 
                            | (0xfffffe00U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__b_q) 
                                              << 9U)));
    vlTOPp->e_packet[0U] = ((0xffffbfffU & vlTOPp->e_packet[0U]) 
                            | (0xffffc000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__a_valid_q) 
                                              << 0xeU)));
    vlTOPp->e_packet[0U] = ((0xffff7fffU & vlTOPp->e_packet[0U]) 
                            | (0xffff8000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__b_valid_q) 
                                              << 0xfU)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q
        [0U];
    vlTOPp->e_packet[0U] = ((0xffe0ffffU & vlTOPp->e_packet[0U]) 
                            | (0xffff0000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1) 
                                              << 0x10U)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q
        [0U];
    vlTOPp->e_packet[0U] = ((0xfc1fffffU & vlTOPp->e_packet[0U]) 
                            | (0xffe00000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2) 
                                              << 0x15U)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound3 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
        [0U];
    vlTOPp->e_packet[0U] = ((0xf3ffffffU & vlTOPp->e_packet[0U]) 
                            | (0xfc000000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound3) 
                                              << 0x1aU)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound4 
        = (1U & (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q));
    vlTOPp->e_packet[0U] = ((0xefffffffU & vlTOPp->e_packet[0U]) 
                            | (0xf0000000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound4) 
                                              << 0x1cU)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q
        [1U];
    vlTOPp->e_packet[0U] = ((0x1fffffffU & vlTOPp->e_packet[0U]) 
                            | (0xe0000000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1) 
                                              << 0x1dU)));
    vlTOPp->e_packet[1U] = ((0xfffffffcU & vlTOPp->e_packet[1U]) 
                            | (0x1fffffffU & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1) 
                                              >> 3U)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q
        [1U];
    vlTOPp->e_packet[1U] = ((0xffffff83U & vlTOPp->e_packet[1U]) 
                            | (0xfffffffcU & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2) 
                                              << 2U)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound3 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
        [1U];
    vlTOPp->e_packet[1U] = ((0xfffffe7fU & vlTOPp->e_packet[1U]) 
                            | (0xffffff80U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound3) 
                                              << 7U)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound4 
        = (1U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q) 
                 >> 1U));
    vlTOPp->e_packet[1U] = ((0xfffffdffU & vlTOPp->e_packet[1U]) 
                            | (0xfffffe00U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound4) 
                                              << 9U)));
    vlTOPp->e_packet[1U] = ((0x3ffU & vlTOPp->e_packet[1U]) 
                            | (0xfffffc00U & (vlTOPp->packet_test_top__DOT__encoder__DOT__context_q 
                                              << 0xaU)));
    vlTOPp->e_packet[2U] = ((0xfffffc00U & vlTOPp->e_packet[2U]) 
                            | (0x3ffU & (vlTOPp->packet_test_top__DOT__encoder__DOT__context_q 
                                         >> 0x16U)));
    vlTOPp->e_packet[2U] = ((0xfc0003ffU & vlTOPp->e_packet[2U]) 
                            | (0xfffffc00U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__epoch_q) 
                                              << 0xaU)));
    vlTOPp->e_packet[2U] = ((0x3ffffffU & vlTOPp->e_packet[2U]) 
                            | (0xfc000000U & (vlTOPp->packet_test_top__DOT__encoder__DOT__theta_q 
                                              << 0x1aU)));
    vlTOPp->e_packet[3U] = ((0xc000000U & vlTOPp->e_packet[3U]) 
                            | (0x3ffffffU & (vlTOPp->packet_test_top__DOT__encoder__DOT__theta_q 
                                             >> 6U)));
    vlTOPp->e_packet[3U] = ((0xbffffffU & vlTOPp->e_packet[3U]) 
                            | (0xfc000000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__reverse_q) 
                                              << 0x1aU)));
    vlTOPp->e_packet[3U] = ((0x7ffffffU & vlTOPp->e_packet[3U]) 
                            | (0xf8000000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__error_q) 
                                              << 0x1bU)));
}

void Vpacket_test_top::_settle__TOP__2(Vpacket_test_top__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vpacket_test_top::_settle__TOP__2\n"); );
    Vpacket_test_top* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->packet_test_top__DOT__verifier__DOT__tau_lo 
        = ((0x10U & ((IData)(vlTOPp->v_tau_lo) << 1U)) 
           | (IData)(vlTOPp->v_tau_lo));
    vlTOPp->packet_test_top__DOT__verifier__DOT__tau_hi 
        = ((0x10U & ((IData)(vlTOPp->v_tau_hi) << 1U)) 
           | (IData)(vlTOPp->v_tau_hi));
    if ((0x4000000U & vlTOPp->v_packet[3U])) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__tau_lo 
            = (0x1fU & VL_NEGATE_I(((0x10U & ((IData)(vlTOPp->v_tau_hi) 
                                              << 1U)) 
                                    | (IData)(vlTOPp->v_tau_hi))));
        vlTOPp->packet_test_top__DOT__verifier__DOT__tau_hi 
            = (0x1fU & VL_NEGATE_I(((0x10U & ((IData)(vlTOPp->v_tau_lo) 
                                              << 1U)) 
                                    | (IData)(vlTOPp->v_tau_lo))));
    }
    vlTOPp->packet_test_top__DOT__verifier__DOT__bad 
        = (1U & ((((vlTOPp->v_packet[3U] >> 0x1bU) 
                   | (((vlTOPp->v_packet[2U] << 0x16U) 
                       | (vlTOPp->v_packet[1U] >> 0xaU)) 
                      != vlTOPp->v_context)) | ((0xffffU 
                                                 & ((vlTOPp->v_packet[3U] 
                                                     << 0x16U) 
                                                    | (vlTOPp->v_packet[2U] 
                                                       >> 0xaU))) 
                                                != (IData)(vlTOPp->v_epoch))) 
                 | VL_GTS_III(1,4,4, (IData)(vlTOPp->v_tau_lo), (IData)(vlTOPp->v_tau_hi))));
    vlTOPp->packet_test_top__DOT__verifier__DOT__a 
        = (0x1fU & ((vlTOPp->v_packet[1U] << 0x1cU) 
                    | (vlTOPp->v_packet[0U] >> 4U)));
    vlTOPp->packet_test_top__DOT__verifier__DOT__b 
        = (0x1fU & ((vlTOPp->v_packet[1U] << 0x17U) 
                    | (vlTOPp->v_packet[0U] >> 9U)));
    vlTOPp->packet_test_top__DOT__verifier__DOT__uncertain 
        = (((vlTOPp->v_packet[0U] >> 0xeU) & VL_GTES_III(1,5,5, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__a), (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__tau_lo))) 
           | ((vlTOPp->v_packet[0U] >> 0xfU) & VL_GTS_III(1,5,5, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__tau_hi), (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__b))));
    vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
        = (0xfU & vlTOPp->v_packet[0U]);
    vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index = 0U;
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo 
        = (0x1fU & ((vlTOPp->v_packet[1U] << 0x10U) 
                    | (vlTOPp->v_packet[0U] >> 0x10U)));
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi 
        = (0x1fU & ((vlTOPp->v_packet[1U] << 0xbU) 
                    | (vlTOPp->v_packet[0U] >> 0x15U)));
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_index 
        = (3U & ((vlTOPp->v_packet[1U] << 6U) | (vlTOPp->v_packet[0U] 
                                                 >> 0x1aU)));
    if ((1U & (~ (vlTOPp->v_packet[0U] >> 0x1cU)))) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
    }
    if ((VL_LTES_III(1,32,32, 4U, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)) 
         | VL_GTS_III(1,5,5, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo), (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi)))) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
    } else {
        if ((1U & ((IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index) 
                   >> (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)))) {
            vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
        }
        vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
            = ((IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index) 
               | ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)));
        if (VL_GTES_III(1,5,5, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo), (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__tau_hi))) {
            vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                = ((IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits) 
                   | ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)));
        } else {
            if (VL_LTS_III(1,5,5, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi), (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__tau_lo))) {
                vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                    = ((~ ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index))) 
                       & (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits));
            } else {
                vlTOPp->packet_test_top__DOT__verifier__DOT__uncertain = 1U;
            }
        }
    }
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo 
        = (0x1fU & ((vlTOPp->v_packet[1U] << 3U) | 
                    (vlTOPp->v_packet[0U] >> 0x1dU)));
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi 
        = (0x1fU & ((vlTOPp->v_packet[2U] << 0x1eU) 
                    | (vlTOPp->v_packet[1U] >> 2U)));
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_index 
        = (3U & ((vlTOPp->v_packet[2U] << 0x19U) | 
                 (vlTOPp->v_packet[1U] >> 7U)));
    if ((1U & (~ (vlTOPp->v_packet[1U] >> 9U)))) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
    }
    if ((VL_LTES_III(1,32,32, 4U, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)) 
         | VL_GTS_III(1,5,5, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo), (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi)))) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
    } else {
        if ((1U & ((IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index) 
                   >> (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)))) {
            vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
        }
        vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
            = ((IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index) 
               | ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)));
        if (VL_GTES_III(1,5,5, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo), (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__tau_hi))) {
            vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                = ((IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits) 
                   | ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)));
        } else {
            if (VL_LTS_III(1,5,5, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi), (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__tau_lo))) {
                vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                    = ((~ ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index))) 
                       & (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits));
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
    vlTOPp->g_bits = (((5U == (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__state_q)) 
                       & (~ (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__error_q)))
                       ? (IData)(vlTOPp->packet_test_top__DOT__group_commit__DOT__bits_q)
                       : 0U);
    vlTOPp->e_packet[0U] = 0U;
    vlTOPp->e_packet[1U] = 0U;
    vlTOPp->e_packet[2U] = 0U;
    vlTOPp->e_packet[3U] = 0U;
    vlTOPp->e_packet[0U] = ((0xfffffff0U & vlTOPp->e_packet[0U]) 
                            | (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__bits_q));
    vlTOPp->e_packet[0U] = ((0xfffffe0fU & vlTOPp->e_packet[0U]) 
                            | (0xfffffff0U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__a_q) 
                                              << 4U)));
    vlTOPp->e_packet[0U] = ((0xffffc1ffU & vlTOPp->e_packet[0U]) 
                            | (0xfffffe00U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__b_q) 
                                              << 9U)));
    vlTOPp->e_packet[0U] = ((0xffffbfffU & vlTOPp->e_packet[0U]) 
                            | (0xffffc000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__a_valid_q) 
                                              << 0xeU)));
    vlTOPp->e_packet[0U] = ((0xffff7fffU & vlTOPp->e_packet[0U]) 
                            | (0xffff8000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__b_valid_q) 
                                              << 0xfU)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q
        [0U];
    vlTOPp->e_packet[0U] = ((0xffe0ffffU & vlTOPp->e_packet[0U]) 
                            | (0xffff0000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1) 
                                              << 0x10U)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q
        [0U];
    vlTOPp->e_packet[0U] = ((0xfc1fffffU & vlTOPp->e_packet[0U]) 
                            | (0xffe00000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2) 
                                              << 0x15U)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound3 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
        [0U];
    vlTOPp->e_packet[0U] = ((0xf3ffffffU & vlTOPp->e_packet[0U]) 
                            | (0xfc000000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound3) 
                                              << 0x1aU)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound4 
        = (1U & (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q));
    vlTOPp->e_packet[0U] = ((0xefffffffU & vlTOPp->e_packet[0U]) 
                            | (0xf0000000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound4) 
                                              << 0x1cU)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_lo_q
        [1U];
    vlTOPp->e_packet[0U] = ((0x1fffffffU & vlTOPp->e_packet[0U]) 
                            | (0xe0000000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1) 
                                              << 0x1dU)));
    vlTOPp->e_packet[1U] = ((0xfffffffcU & vlTOPp->e_packet[1U]) 
                            | (0x1fffffffU & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound1) 
                                              >> 3U)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_hi_q
        [1U];
    vlTOPp->e_packet[1U] = ((0xffffff83U & vlTOPp->e_packet[1U]) 
                            | (0xfffffffcU & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound2) 
                                              << 2U)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound3 
        = vlTOPp->packet_test_top__DOT__encoder__DOT__kept_index_q
        [1U];
    vlTOPp->e_packet[1U] = ((0xfffffe7fU & vlTOPp->e_packet[1U]) 
                            | (0xffffff80U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound3) 
                                              << 7U)));
    vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound4 
        = (1U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__kept_valid_q) 
                 >> 1U));
    vlTOPp->e_packet[1U] = ((0xfffffdffU & vlTOPp->e_packet[1U]) 
                            | (0xfffffe00U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT____Vlvbound4) 
                                              << 9U)));
    vlTOPp->e_packet[1U] = ((0x3ffU & vlTOPp->e_packet[1U]) 
                            | (0xfffffc00U & (vlTOPp->packet_test_top__DOT__encoder__DOT__context_q 
                                              << 0xaU)));
    vlTOPp->e_packet[2U] = ((0xfffffc00U & vlTOPp->e_packet[2U]) 
                            | (0x3ffU & (vlTOPp->packet_test_top__DOT__encoder__DOT__context_q 
                                         >> 0x16U)));
    vlTOPp->e_packet[2U] = ((0xfc0003ffU & vlTOPp->e_packet[2U]) 
                            | (0xfffffc00U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__epoch_q) 
                                              << 0xaU)));
    vlTOPp->e_packet[2U] = ((0x3ffffffU & vlTOPp->e_packet[2U]) 
                            | (0xfc000000U & (vlTOPp->packet_test_top__DOT__encoder__DOT__theta_q 
                                              << 0x1aU)));
    vlTOPp->e_packet[3U] = ((0xc000000U & vlTOPp->e_packet[3U]) 
                            | (0x3ffffffU & (vlTOPp->packet_test_top__DOT__encoder__DOT__theta_q 
                                             >> 6U)));
    vlTOPp->e_packet[3U] = ((0xbffffffU & vlTOPp->e_packet[3U]) 
                            | (0xfc000000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__reverse_q) 
                                              << 0x1aU)));
    vlTOPp->e_packet[3U] = ((0x7ffffffU & vlTOPp->e_packet[3U]) 
                            | (0xf8000000U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__error_q) 
                                              << 0x1bU)));
    vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo 
        = ((0x10U & ((IData)(vlTOPp->e_lo) << 1U)) 
           | (IData)(vlTOPp->e_lo));
    vlTOPp->packet_test_top__DOT__encoder__DOT__norm_hi 
        = ((0x10U & ((IData)(vlTOPp->e_hi) << 1U)) 
           | (IData)(vlTOPp->e_hi));
    if (vlTOPp->packet_test_top__DOT__encoder__DOT__reverse_q) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo 
            = (0x1fU & VL_NEGATE_I(((0x10U & ((IData)(vlTOPp->e_hi) 
                                              << 1U)) 
                                    | (IData)(vlTOPp->e_hi))));
        vlTOPp->packet_test_top__DOT__encoder__DOT__norm_hi 
            = (0x1fU & VL_NEGATE_I(((0x10U & ((IData)(vlTOPp->e_lo) 
                                              << 1U)) 
                                    | (IData)(vlTOPp->e_lo))));
    }
    vlTOPp->packet_test_top__DOT__encoder__DOT__bit_now 
        = VL_GTES_III(1,5,5, (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo), (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q));
    vlTOPp->packet_test_top__DOT__encoder__DOT__distance_wide = 0U;
    if (VL_GTS_III(1,5,5, (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo), (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q))) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__distance_wide 
            = (0x3fU & (((0x20U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo) 
                                   << 1U)) | (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo)) 
                        - ((0x20U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q) 
                                     << 1U)) | (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q))));
    } else {
        if (VL_LTS_III(1,5,5, (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__norm_hi), (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q))) {
            vlTOPp->packet_test_top__DOT__encoder__DOT__distance_wide 
                = (0x3fU & (((0x20U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q) 
                                       << 1U)) | (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q)) 
                            - ((0x20U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__norm_hi) 
                                         << 1U)) | (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__norm_hi))));
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
            = (1U & vlTOPp->packet_test_top__DOT__encoder__DOT__vacant);
        vlTOPp->packet_test_top__DOT__encoder__DOT__do_keep = 1U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__do_discard = 0U;
    } else {
        if (((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__distance_now) 
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
                = (1U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__bits_q) 
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
        = ((0x10U & ((IData)(vlTOPp->v_tau_lo) << 1U)) 
           | (IData)(vlTOPp->v_tau_lo));
    vlTOPp->packet_test_top__DOT__verifier__DOT__tau_hi 
        = ((0x10U & ((IData)(vlTOPp->v_tau_hi) << 1U)) 
           | (IData)(vlTOPp->v_tau_hi));
    if ((0x4000000U & vlTOPp->v_packet[3U])) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__tau_lo 
            = (0x1fU & VL_NEGATE_I(((0x10U & ((IData)(vlTOPp->v_tau_hi) 
                                              << 1U)) 
                                    | (IData)(vlTOPp->v_tau_hi))));
        vlTOPp->packet_test_top__DOT__verifier__DOT__tau_hi 
            = (0x1fU & VL_NEGATE_I(((0x10U & ((IData)(vlTOPp->v_tau_lo) 
                                              << 1U)) 
                                    | (IData)(vlTOPp->v_tau_lo))));
    }
    vlTOPp->packet_test_top__DOT__verifier__DOT__bad 
        = (1U & ((((vlTOPp->v_packet[3U] >> 0x1bU) 
                   | (((vlTOPp->v_packet[2U] << 0x16U) 
                       | (vlTOPp->v_packet[1U] >> 0xaU)) 
                      != vlTOPp->v_context)) | ((0xffffU 
                                                 & ((vlTOPp->v_packet[3U] 
                                                     << 0x16U) 
                                                    | (vlTOPp->v_packet[2U] 
                                                       >> 0xaU))) 
                                                != (IData)(vlTOPp->v_epoch))) 
                 | VL_GTS_III(1,4,4, (IData)(vlTOPp->v_tau_lo), (IData)(vlTOPp->v_tau_hi))));
    vlTOPp->packet_test_top__DOT__verifier__DOT__a 
        = (0x1fU & ((vlTOPp->v_packet[1U] << 0x1cU) 
                    | (vlTOPp->v_packet[0U] >> 4U)));
    vlTOPp->packet_test_top__DOT__verifier__DOT__b 
        = (0x1fU & ((vlTOPp->v_packet[1U] << 0x17U) 
                    | (vlTOPp->v_packet[0U] >> 9U)));
    vlTOPp->packet_test_top__DOT__verifier__DOT__uncertain 
        = (((vlTOPp->v_packet[0U] >> 0xeU) & VL_GTES_III(1,5,5, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__a), (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__tau_lo))) 
           | ((vlTOPp->v_packet[0U] >> 0xfU) & VL_GTS_III(1,5,5, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__tau_hi), (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__b))));
    vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
        = (0xfU & vlTOPp->v_packet[0U]);
    vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index = 0U;
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo 
        = (0x1fU & ((vlTOPp->v_packet[1U] << 0x10U) 
                    | (vlTOPp->v_packet[0U] >> 0x10U)));
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi 
        = (0x1fU & ((vlTOPp->v_packet[1U] << 0xbU) 
                    | (vlTOPp->v_packet[0U] >> 0x15U)));
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_index 
        = (3U & ((vlTOPp->v_packet[1U] << 6U) | (vlTOPp->v_packet[0U] 
                                                 >> 0x1aU)));
    if ((1U & (~ (vlTOPp->v_packet[0U] >> 0x1cU)))) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
    }
    if ((VL_LTES_III(1,32,32, 4U, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)) 
         | VL_GTS_III(1,5,5, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo), (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi)))) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
    } else {
        if ((1U & ((IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index) 
                   >> (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)))) {
            vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
        }
        vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
            = ((IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index) 
               | ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)));
        if (VL_GTES_III(1,5,5, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo), (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__tau_hi))) {
            vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                = ((IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits) 
                   | ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)));
        } else {
            if (VL_LTS_III(1,5,5, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi), (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__tau_lo))) {
                vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                    = ((~ ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index))) 
                       & (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits));
            } else {
                vlTOPp->packet_test_top__DOT__verifier__DOT__uncertain = 1U;
            }
        }
    }
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo 
        = (0x1fU & ((vlTOPp->v_packet[1U] << 3U) | 
                    (vlTOPp->v_packet[0U] >> 0x1dU)));
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi 
        = (0x1fU & ((vlTOPp->v_packet[2U] << 0x1eU) 
                    | (vlTOPp->v_packet[1U] >> 2U)));
    vlTOPp->packet_test_top__DOT__verifier__DOT__value_index 
        = (3U & ((vlTOPp->v_packet[2U] << 0x19U) | 
                 (vlTOPp->v_packet[1U] >> 7U)));
    if ((1U & (~ (vlTOPp->v_packet[1U] >> 9U)))) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
    }
    if ((VL_LTES_III(1,32,32, 4U, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)) 
         | VL_GTS_III(1,5,5, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo), (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi)))) {
        vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
    } else {
        if ((1U & ((IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index) 
                   >> (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)))) {
            vlTOPp->packet_test_top__DOT__verifier__DOT__bad = 1U;
        }
        vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index 
            = ((IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__seen_index) 
               | ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)));
        if (VL_GTES_III(1,5,5, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_lo), (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__tau_hi))) {
            vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                = ((IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits) 
                   | ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index)));
        } else {
            if (VL_LTS_III(1,5,5, (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_hi), (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__tau_lo))) {
                vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits 
                    = ((~ ((IData)(1U) << (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__value_index))) 
                       & (IData)(vlTOPp->packet_test_top__DOT__verifier__DOT__final_bits));
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
        = ((0x10U & ((IData)(vlTOPp->e_lo) << 1U)) 
           | (IData)(vlTOPp->e_lo));
    vlTOPp->packet_test_top__DOT__encoder__DOT__norm_hi 
        = ((0x10U & ((IData)(vlTOPp->e_hi) << 1U)) 
           | (IData)(vlTOPp->e_hi));
    if (vlTOPp->packet_test_top__DOT__encoder__DOT__reverse_q) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo 
            = (0x1fU & VL_NEGATE_I(((0x10U & ((IData)(vlTOPp->e_hi) 
                                              << 1U)) 
                                    | (IData)(vlTOPp->e_hi))));
        vlTOPp->packet_test_top__DOT__encoder__DOT__norm_hi 
            = (0x1fU & VL_NEGATE_I(((0x10U & ((IData)(vlTOPp->e_lo) 
                                              << 1U)) 
                                    | (IData)(vlTOPp->e_lo))));
    }
    vlTOPp->packet_test_top__DOT__encoder__DOT__bit_now 
        = VL_GTES_III(1,5,5, (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo), (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q));
    vlTOPp->packet_test_top__DOT__encoder__DOT__distance_wide = 0U;
    if (VL_GTS_III(1,5,5, (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo), (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q))) {
        vlTOPp->packet_test_top__DOT__encoder__DOT__distance_wide 
            = (0x3fU & (((0x20U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo) 
                                   << 1U)) | (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__norm_lo)) 
                        - ((0x20U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q) 
                                     << 1U)) | (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q))));
    } else {
        if (VL_LTS_III(1,5,5, (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__norm_hi), (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q))) {
            vlTOPp->packet_test_top__DOT__encoder__DOT__distance_wide 
                = (0x3fU & (((0x20U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q) 
                                       << 1U)) | (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__prediction_q)) 
                            - ((0x20U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__norm_hi) 
                                         << 1U)) | (IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__norm_hi))));
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
            = (1U & vlTOPp->packet_test_top__DOT__encoder__DOT__vacant);
        vlTOPp->packet_test_top__DOT__encoder__DOT__do_keep = 1U;
        vlTOPp->packet_test_top__DOT__encoder__DOT__do_discard = 0U;
    } else {
        if (((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__distance_now) 
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
                = (1U & ((IData)(vlTOPp->packet_test_top__DOT__encoder__DOT__bits_q) 
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
    if (VL_UNLIKELY((e_prediction & 0xf0U))) {
        Verilated::overWidthError("e_prediction");}
    if (VL_UNLIKELY((e_in_valid & 0xfeU))) {
        Verilated::overWidthError("e_in_valid");}
    if (VL_UNLIKELY((e_lo & 0xf0U))) {
        Verilated::overWidthError("e_lo");}
    if (VL_UNLIKELY((e_hi & 0xf0U))) {
        Verilated::overWidthError("e_hi");}
    if (VL_UNLIKELY((e_packet_ready & 0xfeU))) {
        Verilated::overWidthError("e_packet_ready");}
    if (VL_UNLIKELY((v_valid & 0xfeU))) {
        Verilated::overWidthError("v_valid");}
    if (VL_UNLIKELY((v_packet[3U] & 0xf0000000U))) {
        Verilated::overWidthError("v_packet");}
    if (VL_UNLIKELY((v_tau_lo & 0xf0U))) {
        Verilated::overWidthError("v_tau_lo");}
    if (VL_UNLIKELY((v_tau_hi & 0xf0U))) {
        Verilated::overWidthError("v_tau_hi");}
    if (VL_UNLIKELY((v_result_ready & 0xfeU))) {
        Verilated::overWidthError("v_result_ready");}
    if (VL_UNLIKELY((g_start_valid & 0xfeU))) {
        Verilated::overWidthError("g_start_valid");}
    if (VL_UNLIKELY((g_packet_valid & 0xfeU))) {
        Verilated::overWidthError("g_packet_valid");}
    if (VL_UNLIKELY((g_packet_status & 0xfcU))) {
        Verilated::overWidthError("g_packet_status");}
    if (VL_UNLIKELY((g_packet_bits & 0xf0U))) {
        Verilated::overWidthError("g_packet_bits");}
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
    e_prediction = VL_RAND_RESET_I(4);
    e_in_valid = VL_RAND_RESET_I(1);
    e_in_ready = VL_RAND_RESET_I(1);
    e_lo = VL_RAND_RESET_I(4);
    e_hi = VL_RAND_RESET_I(4);
    e_packet_valid = VL_RAND_RESET_I(1);
    e_packet_ready = VL_RAND_RESET_I(1);
    VL_RAND_RESET_W(124, e_packet);
    v_valid = VL_RAND_RESET_I(1);
    v_ready = VL_RAND_RESET_I(1);
    VL_RAND_RESET_W(124, v_packet);
    v_context = VL_RAND_RESET_I(32);
    v_epoch = VL_RAND_RESET_I(16);
    v_tau_lo = VL_RAND_RESET_I(4);
    v_tau_hi = VL_RAND_RESET_I(4);
    v_result_valid = VL_RAND_RESET_I(1);
    v_result_ready = VL_RAND_RESET_I(1);
    v_status = VL_RAND_RESET_I(2);
    v_bits = VL_RAND_RESET_I(4);
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
    g_packet_bits = VL_RAND_RESET_I(4);
    g_packet_context = VL_RAND_RESET_I(32);
    g_packet_theta = VL_RAND_RESET_I(32);
    g_packet_epoch = VL_RAND_RESET_I(16);
    g_replay_valid = VL_RAND_RESET_I(1);
    g_replay_ready = VL_RAND_RESET_I(1);
    g_replay_mask = VL_RAND_RESET_I(2);
    g_repair_valid = VL_RAND_RESET_I(1);
    g_repair_ready = VL_RAND_RESET_I(1);
    g_repair_bits = VL_RAND_RESET_I(8);
    g_repair_context = VL_RAND_RESET_I(32);
    g_repair_theta = VL_RAND_RESET_I(32);
    g_repair_epoch = VL_RAND_RESET_I(16);
    g_result_valid = VL_RAND_RESET_I(1);
    g_result_ready = VL_RAND_RESET_I(1);
    g_status = VL_RAND_RESET_I(2);
    g_bits = VL_RAND_RESET_I(8);
    g_out_context = VL_RAND_RESET_I(32);
    g_out_theta = VL_RAND_RESET_I(32);
    g_out_epoch = VL_RAND_RESET_I(16);
    packet_test_top__DOT__encoder__DOT__state_q = VL_RAND_RESET_I(2);
    packet_test_top__DOT__encoder__DOT__count_q = VL_RAND_RESET_I(3);
    packet_test_top__DOT__encoder__DOT__context_q = VL_RAND_RESET_I(32);
    packet_test_top__DOT__encoder__DOT__theta_q = VL_RAND_RESET_I(32);
    packet_test_top__DOT__encoder__DOT__epoch_q = VL_RAND_RESET_I(16);
    packet_test_top__DOT__encoder__DOT__reverse_q = VL_RAND_RESET_I(1);
    packet_test_top__DOT__encoder__DOT__error_q = VL_RAND_RESET_I(1);
    packet_test_top__DOT__encoder__DOT__prediction_q = VL_RAND_RESET_I(5);
    packet_test_top__DOT__encoder__DOT__bits_q = VL_RAND_RESET_I(4);
    packet_test_top__DOT__encoder__DOT__a_q = VL_RAND_RESET_I(5);
    packet_test_top__DOT__encoder__DOT__b_q = VL_RAND_RESET_I(5);
    packet_test_top__DOT__encoder__DOT__a_valid_q = VL_RAND_RESET_I(1);
    packet_test_top__DOT__encoder__DOT__b_valid_q = VL_RAND_RESET_I(1);
    packet_test_top__DOT__encoder__DOT__kept_valid_q = VL_RAND_RESET_I(2);
    { int __Vi0=0; for (; __Vi0<2; ++__Vi0) {
            packet_test_top__DOT__encoder__DOT__kept_lo_q[__Vi0] = VL_RAND_RESET_I(5);
    }}
    { int __Vi0=0; for (; __Vi0<2; ++__Vi0) {
            packet_test_top__DOT__encoder__DOT__kept_hi_q[__Vi0] = VL_RAND_RESET_I(5);
    }}
    { int __Vi0=0; for (; __Vi0<2; ++__Vi0) {
            packet_test_top__DOT__encoder__DOT__kept_index_q[__Vi0] = VL_RAND_RESET_I(2);
    }}
    { int __Vi0=0; for (; __Vi0<2; ++__Vi0) {
            packet_test_top__DOT__encoder__DOT__kept_distance_q[__Vi0] = VL_RAND_RESET_I(6);
    }}
    packet_test_top__DOT__encoder__DOT__norm_lo = VL_RAND_RESET_I(5);
    packet_test_top__DOT__encoder__DOT__norm_hi = VL_RAND_RESET_I(5);
    packet_test_top__DOT__encoder__DOT__distance_wide = VL_RAND_RESET_I(6);
    packet_test_top__DOT__encoder__DOT__distance_now = VL_RAND_RESET_I(6);
    packet_test_top__DOT__encoder__DOT__bit_now = VL_RAND_RESET_I(1);
    packet_test_top__DOT__encoder__DOT__do_keep = VL_RAND_RESET_I(1);
    packet_test_top__DOT__encoder__DOT__do_discard = VL_RAND_RESET_I(1);
    packet_test_top__DOT__encoder__DOT__discard_bit = VL_RAND_RESET_I(1);
    packet_test_top__DOT__encoder__DOT__discard_lo = VL_RAND_RESET_I(5);
    packet_test_top__DOT__encoder__DOT__discard_hi = VL_RAND_RESET_I(5);
    packet_test_top__DOT__encoder__DOT__chosen = VL_RAND_RESET_I(1);
    packet_test_top__DOT__encoder__DOT__worst = VL_RAND_RESET_I(1);
    packet_test_top__DOT__encoder__DOT__vacant = VL_RAND_RESET_I(32);
    packet_test_top__DOT__encoder__DOT____Vlvbound1 = VL_RAND_RESET_I(5);
    packet_test_top__DOT__encoder__DOT____Vlvbound2 = VL_RAND_RESET_I(5);
    packet_test_top__DOT__encoder__DOT____Vlvbound3 = VL_RAND_RESET_I(2);
    packet_test_top__DOT__encoder__DOT____Vlvbound4 = VL_RAND_RESET_I(1);
    packet_test_top__DOT__verifier__DOT__bad = VL_RAND_RESET_I(1);
    packet_test_top__DOT__verifier__DOT__uncertain = VL_RAND_RESET_I(1);
    packet_test_top__DOT__verifier__DOT__final_bits = VL_RAND_RESET_I(4);
    packet_test_top__DOT__verifier__DOT__seen_index = VL_RAND_RESET_I(4);
    packet_test_top__DOT__verifier__DOT__tau_lo = VL_RAND_RESET_I(5);
    packet_test_top__DOT__verifier__DOT__tau_hi = VL_RAND_RESET_I(5);
    packet_test_top__DOT__verifier__DOT__a = VL_RAND_RESET_I(5);
    packet_test_top__DOT__verifier__DOT__b = VL_RAND_RESET_I(5);
    packet_test_top__DOT__verifier__DOT__value_lo = VL_RAND_RESET_I(5);
    packet_test_top__DOT__verifier__DOT__value_hi = VL_RAND_RESET_I(5);
    packet_test_top__DOT__verifier__DOT__value_index = VL_RAND_RESET_I(2);
    packet_test_top__DOT__verifier__DOT__status_next = VL_RAND_RESET_I(2);
    packet_test_top__DOT__group_commit__DOT__state_q = VL_RAND_RESET_I(3);
    packet_test_top__DOT__group_commit__DOT__count_q = VL_RAND_RESET_I(5);
    packet_test_top__DOT__group_commit__DOT__error_q = VL_RAND_RESET_I(1);
    packet_test_top__DOT__group_commit__DOT__bits_q = VL_RAND_RESET_I(8);
    packet_test_top__DOT__group_commit__DOT__replay_q = VL_RAND_RESET_I(2);
}
