// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Primary design header
//
// This header should be included by all source files instantiating the design.
// The class here is then constructed to instantiate the design.
// See the Verilator manual for examples.

#ifndef _VPACKET_TEST_TOP_H_
#define _VPACKET_TEST_TOP_H_  // guard

#include "verilated.h"

//==========

class Vpacket_test_top__Syms;

//----------

VL_MODULE(Vpacket_test_top) {
  public:
    
    // PORTS
    // The application code writes and reads these signals to
    // propagate new values into/out from the Verilated model.
    VL_IN8(clk,0,0);
    VL_IN8(rst_n,0,0);
    VL_IN8(e_start_valid,0,0);
    VL_OUT8(e_start_ready,0,0);
    VL_IN8(e_reverse,0,0);
    VL_IN8(e_prediction,3,0);
    VL_IN8(e_in_valid,0,0);
    VL_OUT8(e_in_ready,0,0);
    VL_IN8(e_lo,3,0);
    VL_IN8(e_hi,3,0);
    VL_OUT8(e_packet_valid,0,0);
    VL_IN8(e_packet_ready,0,0);
    VL_IN8(v_valid,0,0);
    VL_OUT8(v_ready,0,0);
    VL_IN8(v_tau_lo,3,0);
    VL_IN8(v_tau_hi,3,0);
    VL_OUT8(v_result_valid,0,0);
    VL_IN8(v_result_ready,0,0);
    VL_OUT8(v_status,1,0);
    VL_OUT8(v_bits,3,0);
    VL_IN8(g_start_valid,0,0);
    VL_OUT8(g_start_ready,0,0);
    VL_IN8(g_packet_valid,0,0);
    VL_OUT8(g_packet_ready,0,0);
    VL_IN8(g_packet_status,1,0);
    VL_IN8(g_packet_bits,3,0);
    VL_OUT8(g_replay_valid,0,0);
    VL_IN8(g_replay_ready,0,0);
    VL_OUT8(g_replay_mask,1,0);
    VL_IN8(g_repair_valid,0,0);
    VL_OUT8(g_repair_ready,0,0);
    VL_IN8(g_repair_bits,7,0);
    VL_OUT8(g_result_valid,0,0);
    VL_IN8(g_result_ready,0,0);
    VL_OUT8(g_status,1,0);
    VL_OUT8(g_bits,7,0);
    VL_IN16(e_epoch,15,0);
    VL_IN16(v_epoch,15,0);
    VL_OUT16(v_out_epoch,15,0);
    VL_IN16(g_epoch,15,0);
    VL_IN16(g_packet_epoch,15,0);
    VL_IN16(g_repair_epoch,15,0);
    VL_OUT16(g_out_epoch,15,0);
    VL_IN(e_context,31,0);
    VL_IN(e_theta,31,0);
    VL_OUTW(e_packet,97,0,4);
    VL_INW(v_packet,97,0,4);
    VL_IN(v_context,31,0);
    VL_OUT(v_theta,31,0);
    VL_OUT(v_out_context,31,0);
    VL_IN(g_base,31,0);
    VL_IN(g_theta,31,0);
    VL_IN(g_packet_context,31,0);
    VL_IN(g_packet_theta,31,0);
    VL_IN(g_repair_context,31,0);
    VL_IN(g_repair_theta,31,0);
    VL_OUT(g_out_context,31,0);
    VL_OUT(g_out_theta,31,0);
    
    // LOCAL SIGNALS
    // Internals; generally not touched by application code
    CData/*1:0*/ packet_test_top__DOT__encoder__DOT__state_q;
    CData/*2:0*/ packet_test_top__DOT__encoder__DOT__count_q;
    CData/*0:0*/ packet_test_top__DOT__encoder__DOT__reverse_q;
    CData/*0:0*/ packet_test_top__DOT__encoder__DOT__error_q;
    CData/*4:0*/ packet_test_top__DOT__encoder__DOT__prediction_q;
    CData/*3:0*/ packet_test_top__DOT__encoder__DOT__bits_q;
    CData/*4:0*/ packet_test_top__DOT__encoder__DOT__a_q;
    CData/*4:0*/ packet_test_top__DOT__encoder__DOT__b_q;
    CData/*0:0*/ packet_test_top__DOT__encoder__DOT__a_valid_q;
    CData/*0:0*/ packet_test_top__DOT__encoder__DOT__b_valid_q;
    CData/*4:0*/ packet_test_top__DOT__encoder__DOT__norm_lo;
    CData/*4:0*/ packet_test_top__DOT__encoder__DOT__norm_hi;
    CData/*0:0*/ packet_test_top__DOT__encoder__DOT__bit_now;
    CData/*0:0*/ packet_test_top__DOT__encoder__DOT__discard_bit;
    CData/*4:0*/ packet_test_top__DOT__encoder__DOT__discard_lo;
    CData/*4:0*/ packet_test_top__DOT__encoder__DOT__discard_hi;
    CData/*0:0*/ packet_test_top__DOT__verifier__DOT__bad;
    CData/*0:0*/ packet_test_top__DOT__verifier__DOT__uncertain;
    CData/*4:0*/ packet_test_top__DOT__verifier__DOT__tau_lo;
    CData/*4:0*/ packet_test_top__DOT__verifier__DOT__tau_hi;
    CData/*4:0*/ packet_test_top__DOT__verifier__DOT__a;
    CData/*4:0*/ packet_test_top__DOT__verifier__DOT__b;
    CData/*1:0*/ packet_test_top__DOT__verifier__DOT__status_next;
    CData/*2:0*/ packet_test_top__DOT__group_commit__DOT__state_q;
    CData/*4:0*/ packet_test_top__DOT__group_commit__DOT__count_q;
    CData/*0:0*/ packet_test_top__DOT__group_commit__DOT__error_q;
    CData/*7:0*/ packet_test_top__DOT__group_commit__DOT__bits_q;
    CData/*1:0*/ packet_test_top__DOT__group_commit__DOT__replay_q;
    SData/*15:0*/ packet_test_top__DOT__encoder__DOT__epoch_q;
    IData/*31:0*/ packet_test_top__DOT__encoder__DOT__context_q;
    IData/*31:0*/ packet_test_top__DOT__encoder__DOT__theta_q;
    CData/*4:0*/ packet_test_top__DOT__encoder__DOT__kept_lo_q[1];
    CData/*4:0*/ packet_test_top__DOT__encoder__DOT__kept_hi_q[1];
    CData/*1:0*/ packet_test_top__DOT__encoder__DOT__kept_index_q[1];
    CData/*5:0*/ packet_test_top__DOT__encoder__DOT__kept_distance_q[1];
    
    // LOCAL VARIABLES
    // Internals; generally not touched by application code
    CData/*0:0*/ __Vclklast__TOP__clk;
    
    // INTERNAL VARIABLES
    // Internals; generally not touched by application code
    Vpacket_test_top__Syms* __VlSymsp;  // Symbol table
    
    // CONSTRUCTORS
  private:
    VL_UNCOPYABLE(Vpacket_test_top);  ///< Copying not allowed
  public:
    /// Construct the model; called by application code
    /// The special name  may be used to make a wrapper with a
    /// single model invisible with respect to DPI scope names.
    Vpacket_test_top(const char* name = "TOP");
    /// Destroy the model; called (often implicitly) by application code
    ~Vpacket_test_top();
    
    // API METHODS
    /// Evaluate the model.  Application must call when inputs change.
    void eval();
    /// Simulation complete, run final blocks.  Application must call on completion.
    void final();
    
    // INTERNAL METHODS
  private:
    static void _eval_initial_loop(Vpacket_test_top__Syms* __restrict vlSymsp);
  public:
    void __Vconfigure(Vpacket_test_top__Syms* symsp, bool first);
  private:
    static QData _change_request(Vpacket_test_top__Syms* __restrict vlSymsp);
  public:
    static void _combo__TOP__3(Vpacket_test_top__Syms* __restrict vlSymsp);
  private:
    void _ctor_var_reset() VL_ATTR_COLD;
  public:
    static void _eval(Vpacket_test_top__Syms* __restrict vlSymsp);
  private:
#ifdef VL_DEBUG
    void _eval_debug_assertions();
#endif  // VL_DEBUG
  public:
    static void _eval_initial(Vpacket_test_top__Syms* __restrict vlSymsp) VL_ATTR_COLD;
    static void _eval_settle(Vpacket_test_top__Syms* __restrict vlSymsp) VL_ATTR_COLD;
    static void _sequent__TOP__1(Vpacket_test_top__Syms* __restrict vlSymsp);
    static void _settle__TOP__2(Vpacket_test_top__Syms* __restrict vlSymsp) VL_ATTR_COLD;
} VL_ATTR_ALIGNED(VL_CACHE_LINE_BYTES);

//----------


#endif  // guard
