// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb_h67_score_class_row_engine.h for the primary calling header

#include "Vtb_h67_score_class_row_engine__pch.h"
#include "Vtb_h67_score_class_row_engine__Syms.h"
#include "Vtb_h67_score_class_row_engine___024root.h"

VL_INLINE_OPT VlCoroutine Vtb_h67_score_class_row_engine___024root___eval_initial__TOP__Vtiming__0(Vtb_h67_score_class_row_engine___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_h67_score_class_row_engine__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_h67_score_class_row_engine___024root___eval_initial__TOP__Vtiming__0\n"); );
    // Init
    IData/*31:0*/ tb_h67_score_class_row_engine__DOT__errors;
    tb_h67_score_class_row_engine__DOT__errors = 0;
    IData/*31:0*/ tb_h67_score_class_row_engine__DOT__output_count;
    tb_h67_score_class_row_engine__DOT__output_count = 0;
    IData/*31:0*/ tb_h67_score_class_row_engine__DOT__expected_outputs;
    tb_h67_score_class_row_engine__DOT__expected_outputs = 0;
    IData/*31:0*/ tb_h67_score_class_row_engine__DOT__expected_folded;
    tb_h67_score_class_row_engine__DOT__expected_folded = 0;
    IData/*31:0*/ tb_h67_score_class_row_engine__DOT__row_max;
    tb_h67_score_class_row_engine__DOT__row_max = 0;
    IData/*31:0*/ tb_h67_score_class_row_engine__DOT__row_sum;
    tb_h67_score_class_row_engine__DOT__row_sum = 0;
    IData/*31:0*/ tb_h67_score_class_row_engine__DOT__denominator_shift;
    tb_h67_score_class_row_engine__DOT__denominator_shift = 0;
    IData/*31:0*/ tb_h67_score_class_row_engine__DOT__unnamedblk1__DOT__all_class_count;
    tb_h67_score_class_row_engine__DOT__unnamedblk1__DOT__all_class_count = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__n_tokens;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__n_tokens = 0;
    CData/*0:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__enable_fold;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__enable_fold = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__idx;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__idx = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__n_tokens;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__n_tokens = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__idx;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__idx = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__exp_value;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__exp_value = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__scaled;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__scaled = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__q;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__q = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__k;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__k = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__peer;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__peer = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__overlap_count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__overlap_count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__same_zero_count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__same_zero_count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__motion_count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__motion_count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__6__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__6__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__6__count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__6__count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__6__integer_base;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__6__integer_base = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__6__quotient;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__6__quotient = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__6__remainder;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__6__remainder = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__delta_q7;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__delta_q7 = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__abs_delta;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__abs_delta = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__integer_shift;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__integer_shift = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__8__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__8__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__8__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__8__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__8__probe;
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__8__probe = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__8__shift_value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__8__shift_value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__delta_q7;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__delta_q7 = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__abs_delta;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__abs_delta = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__integer_shift;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__integer_shift = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__shift_value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__shift_value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__quotient;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__quotient = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__remainder;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__remainder = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__half;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__half = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__n_tokens;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__n_tokens = 0;
    CData/*0:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__enable_fold;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__enable_fold = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__idx;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__idx = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__n_tokens;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__n_tokens = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__idx;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__idx = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__exp_value;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__exp_value = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__scaled;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__scaled = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__q;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__q = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__k;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__k = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__peer;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__peer = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__overlap_count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__overlap_count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__same_zero_count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__same_zero_count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__motion_count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__motion_count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__17__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__17__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__17__count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__17__count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__17__integer_base;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__17__integer_base = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__17__quotient;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__17__quotient = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__17__remainder;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__17__remainder = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__delta_q7;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__delta_q7 = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__abs_delta;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__abs_delta = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__integer_shift;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__integer_shift = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__19__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__19__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__19__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__19__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__19__probe;
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__19__probe = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__19__shift_value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__19__shift_value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__delta_q7;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__delta_q7 = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__abs_delta;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__abs_delta = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__integer_shift;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__integer_shift = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__shift_value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__shift_value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__quotient;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__quotient = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__remainder;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__remainder = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__half;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__half = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_active__22__n_tokens;
    __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_active__22__n_tokens = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_active__22__idx;
    __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_active__22__idx = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__n_tokens;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__n_tokens = 0;
    CData/*0:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__enable_fold;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__enable_fold = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__idx;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__idx = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__n_tokens;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__n_tokens = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__idx;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__idx = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__exp_value;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__exp_value = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__scaled;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__scaled = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__q;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__q = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__k;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__k = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__peer;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__peer = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__overlap_count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__overlap_count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__same_zero_count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__same_zero_count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__motion_count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__motion_count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__29__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__29__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__29__count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__29__count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__29__integer_base;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__29__integer_base = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__29__quotient;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__29__quotient = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__29__remainder;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__29__remainder = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__delta_q7;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__delta_q7 = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__abs_delta;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__abs_delta = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__integer_shift;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__integer_shift = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__31__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__31__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__31__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__31__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__31__probe;
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__31__probe = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__31__shift_value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__31__shift_value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__delta_q7;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__delta_q7 = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__abs_delta;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__abs_delta = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__integer_shift;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__integer_shift = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__shift_value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__shift_value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__quotient;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__quotient = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__remainder;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__remainder = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__half;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__half = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__n_tokens;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__n_tokens = 0;
    CData/*0:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__enable_fold;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__enable_fold = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__idx;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__idx = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__n_tokens;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__n_tokens = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__idx;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__idx = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__exp_value;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__exp_value = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__scaled;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__scaled = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__q;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__q = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__k;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__k = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__peer;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__peer = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__overlap_count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__overlap_count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__same_zero_count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__same_zero_count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__motion_count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__motion_count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__40__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__40__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__40__count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__40__count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__40__integer_base;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__40__integer_base = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__40__quotient;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__40__quotient = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__40__remainder;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__40__remainder = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__delta_q7;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__delta_q7 = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__abs_delta;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__abs_delta = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__integer_shift;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__integer_shift = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__42__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__42__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__42__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__42__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__42__probe;
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__42__probe = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__42__shift_value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__42__shift_value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__delta_q7;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__delta_q7 = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__abs_delta;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__abs_delta = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__integer_shift;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__integer_shift = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__shift_value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__shift_value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__quotient;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__quotient = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__remainder;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__remainder = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__half;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__half = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__class_count;
    __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__class_count = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__target;
    __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__target = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__q_active;
    __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__q_active = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__peer_active;
    __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__peer_active = 0;
    CData/*0:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__found;
    __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__found = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__q;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__q = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__k;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__k = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__peer;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__peer = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__overlap_count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__overlap_count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__same_zero_count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__same_zero_count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__motion_count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__motion_count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__47__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__47__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__47__count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__47__count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__48__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__48__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__48__count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__48__count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__52__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__52__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__52__count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__52__count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__52__integer_base;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__52__integer_base = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__52__quotient;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__52__quotient = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__52__remainder;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__52__remainder = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__53__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__53__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__53__count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__53__count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__54__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__54__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__54__count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__54__count = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__n_tokens;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__n_tokens = 0;
    CData/*0:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__enable_fold;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__enable_fold = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__idx;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__idx = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__n_tokens;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__n_tokens = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__idx;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__idx = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__exp_value;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__exp_value = 0;
    IData/*31:0*/ __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__scaled;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__scaled = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__q;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__q = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__k;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__k = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__peer;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__peer = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__overlap_count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__overlap_count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__same_zero_count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__same_zero_count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__motion_count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__motion_count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__61__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__61__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__61__count;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__61__count = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__61__integer_base;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__61__integer_base = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__61__quotient;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__61__quotient = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__61__remainder;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__61__remainder = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__delta_q7;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__delta_q7 = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__abs_delta;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__abs_delta = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__integer_shift;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__integer_shift = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__63__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__63__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__63__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__63__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__63__probe;
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__63__probe = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__63__shift_value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__63__shift_value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__delta_q7;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__delta_q7 = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__abs_delta;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__abs_delta = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__integer_shift;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__integer_shift = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__Vfuncout = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__shift_value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__shift_value = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__quotient;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__quotient = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__remainder;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__remainder = 0;
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__half;
    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__half = 0;
    IData/*31:0*/ __Vtemp_99;
    IData/*31:0*/ __Vtemp_217;
    IData/*31:0*/ __Vtemp_336;
    IData/*31:0*/ __Vtemp_454;
    IData/*31:0*/ __Vtemp_671;
    // Body
    vlSelf->tb_h67_score_class_row_engine__DOT__clk = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__rst_n = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__cfg_start = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__cfg_n_tokens = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__cfg_preserve_mean = 1U;
    vlSelf->tb_h67_score_class_row_engine__DOT__cfg_enable_score_fold = 1U;
    vlSelf->tb_h67_score_class_row_engine__DOT__cfg_threshold_q8 = 0x40U;
    vlSelf->tb_h67_score_class_row_engine__DOT__in_valid = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__in_last = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__in_time_sel = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__in_k_pair_bits = 0ULL;
    vlSelf->tb_h67_score_class_row_engine__DOT__out_ready = 1U;
    tb_h67_score_class_row_engine__DOT__errors = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__q_vector[0U] = 0x9e3779b9U;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector[0U] = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector[0U] = 0xd1b54a35U;
    vlSelf->tb_h67_score_class_row_engine__DOT__time_vector[0U] = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__q_vector[1U] = 0x3c6ef372U;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector[1U] = 0x7e4b7d14U;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector[1U] = 0xc1950ab4U;
    vlSelf->tb_h67_score_class_row_engine__DOT__time_vector[1U] = 1U;
    vlSelf->tb_h67_score_class_row_engine__DOT__q_vector[2U] = 0xdaa66d2bU;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector[2U] = 0x7d487e17U;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector[2U] = 0xf1f5cb37U;
    vlSelf->tb_h67_score_class_row_engine__DOT__time_vector[2U] = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__q_vector[3U] = 0x78dde6e4U;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector[3U] = 0x7c497f16U;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector[3U] = 0xe1d58bb6U;
    vlSelf->tb_h67_score_class_row_engine__DOT__time_vector[3U] = 1U;
    vlSelf->tb_h67_score_class_row_engine__DOT__q_vector[4U] = 0x1715609dU;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector[4U] = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector[4U] = 0x91344831U;
    vlSelf->tb_h67_score_class_row_engine__DOT__time_vector[4U] = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__q_vector[5U] = 0xb54cda56U;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector[5U] = 0x7a4f7910U;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector[5U] = 0x811408b0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__time_vector[5U] = 1U;
    vlSelf->tb_h67_score_class_row_engine__DOT__q_vector[6U] = 0x5384540fU;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector[6U] = 0x794c7a13U;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector[6U] = 0xb174c933U;
    vlSelf->tb_h67_score_class_row_engine__DOT__time_vector[6U] = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__q_vector[7U] = 0xf1bbcdc8U;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector[7U] = 0x784d7b12U;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector[7U] = 0xa15489b2U;
    vlSelf->tb_h67_score_class_row_engine__DOT__time_vector[7U] = 1U;
    vlSelf->tb_h67_score_class_row_engine__DOT__q_vector[0U] = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector[0U] = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector[0U] = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__time_vector[0U] = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__q_vector[1U] = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector[1U] = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector[1U] = 0xffffffffU;
    vlSelf->tb_h67_score_class_row_engine__DOT__time_vector[1U] = 1U;
    vlSelf->tb_h67_score_class_row_engine__DOT__q_vector[2U] = 0xffffU;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector[2U] = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector[2U] = 0xfU;
    vlSelf->tb_h67_score_class_row_engine__DOT__time_vector[2U] = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__q_vector[3U] = 0xffffU;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector[3U] = 0xfU;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector[3U] = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__time_vector[3U] = 1U;
    vlSelf->tb_h67_score_class_row_engine__DOT__q_vector[4U] = 0xff00ffU;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector[4U] = 3U;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector[4U] = 1U;
    vlSelf->tb_h67_score_class_row_engine__DOT__time_vector[4U] = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__q_vector[5U] = 0xffffffffU;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector[5U] = 0xffffffffU;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector[5U] = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__time_vector[5U] = 1U;
    vlSelf->tb_h67_score_class_row_engine__DOT__q_vector[6U] = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector[6U] = 0x80000000U;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector[6U] = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__time_vector[6U] = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__q_vector[7U] = 0xa5a5a5a5U;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector[7U] = 0x5a5a5a5aU;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector[7U] = 0xa5a5a5a5U;
    vlSelf->tb_h67_score_class_row_engine__DOT__time_vector[7U] = 1U;
    co_await vlSelf->__VtrigSched_h9aed8dba__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_h67_score_class_row_engine.clk)", 
                                                       "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                       404);
    co_await vlSelf->__VtrigSched_h9aed8dba__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_h67_score_class_row_engine.clk)", 
                                                       "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                       404);
    co_await vlSelf->__VtrigSched_h9aed8dba__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_h67_score_class_row_engine.clk)", 
                                                       "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                       404);
    vlSelf->tb_h67_score_class_row_engine__DOT__rst_n = 1U;
    co_await vlSelf->__VtrigSched_h9aed8dba__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_h67_score_class_row_engine.clk)", 
                                                       "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                       406);
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__enable_fold = 1U;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__n_tokens = 8U;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__n_tokens 
        = __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__n_tokens;
    tb_h67_score_class_row_engine__DOT__row_max = 0xffff8000U;
    tb_h67_score_class_row_engine__DOT__row_sum = 0U;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__idx = 0U;
    while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__idx, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__n_tokens)) {
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__peer 
            = vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector
            [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__idx)];
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__k 
            = vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector
            [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__idx)];
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__q 
            = vlSelf->tb_h67_score_class_row_engine__DOT__q_vector
            [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__idx)];
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__q 
               & __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__k);
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout = 0U;
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 1U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 2U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 3U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 4U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 5U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 6U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 7U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 8U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 9U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 0xaU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 0xbU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 0xcU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 0xdU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 0xeU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 0xfU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 0x10U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 0x11U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 0x12U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 0x13U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 0x14U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 0x15U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 0x16U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 0x17U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 0x18U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 0x19U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 0x1aU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 0x1bU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 0x1cU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 0x1dU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                        >> 0x1eU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout 
               + (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__value 
                  >> 0x1fU));
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__overlap_count 
            = __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__3__Vfuncout;
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
            = ((~ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__q) 
               & (~ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__k));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout = 0U;
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 1U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 2U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 3U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 4U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 5U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 6U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 7U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 8U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 9U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 0xaU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 0xbU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 0xcU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 0xdU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 0xeU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 0xfU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 0x10U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 0x11U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 0x12U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 0x13U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 0x14U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 0x15U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 0x16U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 0x17U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 0x18U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 0x19U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 0x1aU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 0x1bU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 0x1cU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 0x1dU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                        >> 0x1eU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout 
               + (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__value 
                  >> 0x1fU));
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__same_zero_count 
            = __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__4__Vfuncout;
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__k 
               ^ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__peer);
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout = 0U;
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 1U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 2U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 3U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 4U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 5U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 6U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 7U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 8U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 9U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 0xaU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 0xbU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 0xcU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 0xdU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 0xeU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 0xfU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 0x10U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 0x11U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 0x12U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 0x13U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 0x14U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 0x15U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 0x16U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 0x17U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 0x18U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 0x19U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 0x1aU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 0x1bU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 0x1cU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 0x1dU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                        >> 0x1eU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout 
               + (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__value 
                  >> 0x1fU));
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__motion_count 
            = __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__5__Vfuncout;
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__Vfuncout 
            = ((VL_MULS_III(32, (IData)(4U), __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__overlap_count) 
                + __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__motion_count) 
               + ([&]() {
                    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__6__integer_base 
                        = (VL_MULS_III(32, (IData)(4U), __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__overlap_count) 
                           + __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__motion_count);
                    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__6__count 
                        = __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__same_zero_count;
                    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__6__quotient 
                        = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__6__count, 4U);
                    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__6__remainder 
                        = (0xfU & __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__6__count);
                    if ((VL_LTS_III(32, 8U, __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__6__remainder) 
                         | ((8U == __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__6__remainder) 
                            & (0U != VL_MODDIVS_III(32, 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__6__integer_base 
                                                     + __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__6__quotient), (IData)(2U)))))) {
                        __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__6__quotient 
                            = ((IData)(1U) + __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__6__quotient);
                    }
                    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__6__Vfuncout 
                        = __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__6__quotient;
                }(), __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__6__Vfuncout));
        vlSelf->tb_h67_score_class_row_engine__DOT__expected_score[(7U 
                                                                    & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__idx)] 
            = __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__2__Vfuncout;
        if (VL_GTS_III(32, vlSelf->tb_h67_score_class_row_engine__DOT__expected_score
                       [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__idx)], tb_h67_score_class_row_engine__DOT__row_max)) {
            tb_h67_score_class_row_engine__DOT__row_max 
                = vlSelf->tb_h67_score_class_row_engine__DOT__expected_score
                [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__idx)];
        }
        __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__idx 
            = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__idx);
    }
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__idx = 0U;
    while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__idx, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__n_tokens)) {
        __Vtemp_99 = (tb_h67_score_class_row_engine__DOT__row_sum 
                      + ([&]() {
                    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__delta_q7 
                        = (vlSelf->tb_h67_score_class_row_engine__DOT__expected_score
                           [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__idx)] 
                           - tb_h67_score_class_row_engine__DOT__row_max);
                    if (VL_LTES_III(32, 0U, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__delta_q7)) {
                        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__Vfuncout = 0x100U;
                    } else {
                        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__abs_delta 
                            = (- __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__delta_q7);
                        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__integer_shift 
                            = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__abs_delta, 7U);
                        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index 
                            = (0xfU & VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__abs_delta, 3U));
                        if ((0U != (7U & __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__abs_delta))) {
                            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index 
                                = ((IData)(1U) + __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index);
                        }
                        if (VL_LTS_III(32, 0xfU, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index)) {
                            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index = 0xfU;
                        }
                        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_value 
                            = (((((((((0U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index) 
                                      | (1U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index)) 
                                     | (2U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index)) 
                                    | (3U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index)) 
                                   | (4U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index)) 
                                  | (5U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index)) 
                                 | (6U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index)) 
                                | (7U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index))
                                ? ((0U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index)
                                    ? 0x100U : ((1U 
                                                 == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index)
                                                 ? 0xf5U
                                                 : 
                                                ((2U 
                                                  == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index)
                                                  ? 0xeaU
                                                  : 
                                                 ((3U 
                                                   == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index)
                                                   ? 0xe0U
                                                   : 
                                                  ((4U 
                                                    == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index)
                                                    ? 0xd7U
                                                    : 
                                                   ((5U 
                                                     == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index)
                                                     ? 0xcdU
                                                     : 
                                                    ((6U 
                                                      == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index)
                                                      ? 0xc4U
                                                      : 0xbcU)))))))
                                : ((8U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index)
                                    ? 0xb5U : ((9U 
                                                == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index)
                                                ? 0xadU
                                                : (
                                                   (0xaU 
                                                    == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index)
                                                    ? 0xa5U
                                                    : 
                                                   ((0xbU 
                                                     == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index)
                                                     ? 0x9eU
                                                     : 
                                                    ((0xcU 
                                                      == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index)
                                                      ? 0x98U
                                                      : 
                                                     ((0xdU 
                                                       == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index)
                                                       ? 0x91U
                                                       : 
                                                      ((0xeU 
                                                        == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_index)
                                                        ? 0x8bU
                                                        : 0x85U))))))));
                        if (VL_LTS_III(32, 8U, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__integer_shift)) {
                            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__integer_shift = 8U;
                        }
                        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__Vfuncout 
                            = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__fraction_value, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__integer_shift);
                    }
                }(), __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__7__Vfuncout));
        tb_h67_score_class_row_engine__DOT__row_sum 
            = __Vtemp_99;
        __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__idx 
            = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__idx);
    }
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__8__value 
        = tb_h67_score_class_row_engine__DOT__row_sum;
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__8__probe 
        = (VL_GTES_III(32, 1U, __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__8__value)
            ? 1U : (__Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__8__value 
                    - (IData)(1U)));
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__8__shift_value = 0U;
    while (VL_LTS_III(32, 0U, __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__8__probe)) {
        __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__8__probe 
            = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__8__probe, 1U);
        __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__8__shift_value 
            = ((IData)(1U) + __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__8__shift_value);
    }
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__8__Vfuncout 
        = __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__8__shift_value;
    tb_h67_score_class_row_engine__DOT__denominator_shift 
        = __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__8__Vfuncout;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__idx = 0U;
    while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__idx, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__n_tokens)) {
        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__delta_q7 
            = (vlSelf->tb_h67_score_class_row_engine__DOT__expected_score
               [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__idx)] 
               - tb_h67_score_class_row_engine__DOT__row_max);
        if (VL_LTES_III(32, 0U, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__delta_q7)) {
            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__Vfuncout = 0x100U;
        } else {
            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__abs_delta 
                = (- __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__delta_q7);
            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__integer_shift 
                = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__abs_delta, 7U);
            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index 
                = (0xfU & VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__abs_delta, 3U));
            if ((0U != (7U & __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__abs_delta))) {
                __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index 
                    = ((IData)(1U) + __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index);
            }
            if (VL_LTS_III(32, 0xfU, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index)) {
                __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index = 0xfU;
            }
            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_value 
                = (((((((((0U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index) 
                          | (1U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index)) 
                         | (2U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index)) 
                        | (3U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index)) 
                       | (4U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index)) 
                      | (5U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index)) 
                     | (6U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index)) 
                    | (7U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index))
                    ? ((0U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index)
                        ? 0x100U : ((1U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index)
                                     ? 0xf5U : ((2U 
                                                 == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index)
                                                 ? 0xeaU
                                                 : 
                                                ((3U 
                                                  == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index)
                                                  ? 0xe0U
                                                  : 
                                                 ((4U 
                                                   == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index)
                                                   ? 0xd7U
                                                   : 
                                                  ((5U 
                                                    == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index)
                                                    ? 0xcdU
                                                    : 
                                                   ((6U 
                                                     == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index)
                                                     ? 0xc4U
                                                     : 0xbcU)))))))
                    : ((8U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index)
                        ? 0xb5U : ((9U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index)
                                    ? 0xadU : ((0xaU 
                                                == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index)
                                                ? 0xa5U
                                                : (
                                                   (0xbU 
                                                    == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index)
                                                    ? 0x9eU
                                                    : 
                                                   ((0xcU 
                                                     == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index)
                                                     ? 0x98U
                                                     : 
                                                    ((0xdU 
                                                      == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index)
                                                      ? 0x91U
                                                      : 
                                                     ((0xeU 
                                                       == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_index)
                                                       ? 0x8bU
                                                       : 0x85U))))))));
            if (VL_LTS_III(32, 8U, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__integer_shift)) {
                __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__integer_shift = 8U;
            }
            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__Vfuncout 
                = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__fraction_value, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__integer_shift);
        }
        __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__exp_value 
            = __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__9__Vfuncout;
        __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__scaled 
            = VL_MULS_III(32, (IData)(0x80U), VL_MULS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__exp_value, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__n_tokens));
        __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__shift_value 
            = tb_h67_score_class_row_engine__DOT__denominator_shift;
        __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__value 
            = __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__scaled;
        if ((0U == __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__shift_value)) {
            __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__Vfuncout 
                = __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__value;
        } else {
            __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__quotient 
                = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__value, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__shift_value);
            __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__remainder 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__value 
                   - VL_SHIFTL_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__quotient, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__shift_value));
            __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__half 
                = VL_SHIFTL_III(32,32,32, (IData)(1U), 
                                (__Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__shift_value 
                                 - (IData)(1U)));
            if ((VL_GTS_III(32, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__remainder, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__half) 
                 | ((__Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__remainder 
                     == __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__half) 
                    & __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__quotient))) {
                __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__quotient 
                    = ((IData)(1U) + __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__quotient);
            }
            __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__Vfuncout 
                = __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__quotient;
        }
        vlSelf->tb_h67_score_class_row_engine__DOT__expected_gate[(7U 
                                                                   & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__idx)] 
            = __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__10__Vfuncout;
        if (VL_LTS_III(32, 0x100U, vlSelf->tb_h67_score_class_row_engine__DOT__expected_gate
                       [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__idx)])) {
            vlSelf->tb_h67_score_class_row_engine__DOT__expected_gate[(7U 
                                                                       & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__idx)] = 0x100U;
        }
        __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__idx 
            = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__1__idx);
    }
    vlSelf->tb_h67_score_class_row_engine__DOT__cfg_n_tokens 
        = (0xfU & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__n_tokens);
    vlSelf->tb_h67_score_class_row_engine__DOT__cfg_enable_score_fold 
        = __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__enable_fold;
    co_await vlSelf->__VtrigSched_h9aed8d87__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(negedge tb_h67_score_class_row_engine.clk)", 
                                                       "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                       257);
    vlSelf->tb_h67_score_class_row_engine__DOT__cfg_start = 1U;
    co_await vlSelf->__VtrigSched_h9aed8dba__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_h67_score_class_row_engine.clk)", 
                                                       "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                       259);
    co_await vlSelf->__VtrigSched_h9aed8d87__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(negedge tb_h67_score_class_row_engine.clk)", 
                                                       "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                       260);
    vlSelf->tb_h67_score_class_row_engine__DOT__cfg_start = 0U;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__idx = 0U;
    while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__idx, __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__n_tokens)) {
        while ((2U != (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))) {
            co_await vlSelf->__VtrigSched_h9aed8d87__0.trigger(0U, 
                                                               nullptr, 
                                                               "@(negedge tb_h67_score_class_row_engine.clk)", 
                                                               "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                               264);
        }
        vlSelf->tb_h67_score_class_row_engine__DOT__in_valid = 1U;
        vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
            = vlSelf->tb_h67_score_class_row_engine__DOT__q_vector
            [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__idx)];
        if (vlSelf->tb_h67_score_class_row_engine__DOT__time_vector
            [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__idx)]) {
            vlSelf->tb_h67_score_class_row_engine__DOT__in_time_sel = 1U;
            vlSelf->tb_h67_score_class_row_engine__DOT__in_k_pair_bits 
                = (((QData)((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector
                                    [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__idx)])) 
                    << 0x20U) | (QData)((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector
                                                [(7U 
                                                  & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__idx)])));
        } else {
            vlSelf->tb_h67_score_class_row_engine__DOT__in_time_sel = 0U;
            vlSelf->tb_h67_score_class_row_engine__DOT__in_k_pair_bits 
                = (((QData)((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector
                                    [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__idx)])) 
                    << 0x20U) | (QData)((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector
                                                [(7U 
                                                  & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__idx)])));
        }
        vlSelf->tb_h67_score_class_row_engine__DOT__in_last 
            = (__Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__idx 
               == (__Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__n_tokens 
                   - (IData)(1U)));
        co_await vlSelf->__VtrigSched_h9aed8dba__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_h67_score_class_row_engine.clk)", 
                                                           "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                           274);
        co_await vlSelf->__VtrigSched_h9aed8d87__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(negedge tb_h67_score_class_row_engine.clk)", 
                                                           "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                           275);
        vlSelf->tb_h67_score_class_row_engine__DOT__in_valid = 0U;
        vlSelf->tb_h67_score_class_row_engine__DOT__in_last = 0U;
        __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__idx 
            = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__idx);
    }
    tb_h67_score_class_row_engine__DOT__output_count = 0U;
    tb_h67_score_class_row_engine__DOT__expected_outputs = 0U;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__idx = 0U;
    while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__idx, __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__n_tokens)) {
        if ((1U & ((~ (IData)(__Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__enable_fold)) 
                   | (0U != vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector
                      [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__idx)])))) {
            tb_h67_score_class_row_engine__DOT__expected_outputs 
                = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__expected_outputs);
        }
        __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__idx 
            = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__idx);
    }
    tb_h67_score_class_row_engine__DOT__expected_folded 
        = ((IData)(__Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__enable_fold)
            ? (__Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__n_tokens 
               - tb_h67_score_class_row_engine__DOT__expected_outputs)
            : 0U);
    while ((7U != (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))) {
        co_await vlSelf->__VtrigSched_h9aed8d87__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(negedge tb_h67_score_class_row_engine.clk)", 
                                                           "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                           288);
        vlSelf->tb_h67_score_class_row_engine__DOT__out_ready 
            = (1U & ((~ tb_h67_score_class_row_engine__DOT__output_count) 
                     | (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__out_ready))));
        if (((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__out_valid) 
             & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__out_ready))) {
            vlSelf->tb_h67_score_class_row_engine__DOT__token_idx 
                = (0xfU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w));
            if (VL_UNLIKELY(((IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w 
                                      >> 4U)) != vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector
                             [(7U & vlSelf->tb_h67_score_class_row_engine__DOT__token_idx)]))) {
                VL_WRITEF("ERROR token %0d K mismatch\n",
                          32,vlSelf->tb_h67_score_class_row_engine__DOT__token_idx);
                tb_h67_score_class_row_engine__DOT__errors 
                    = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
            }
            if (VL_UNLIKELY(((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__gate_w) 
                             != (0x1ffU & vlSelf->tb_h67_score_class_row_engine__DOT__expected_gate
                                 [(7U & vlSelf->tb_h67_score_class_row_engine__DOT__token_idx)])))) {
                VL_WRITEF("ERROR token %0d gate got=%0# expected=%0d score=%0d\n",
                          32,vlSelf->tb_h67_score_class_row_engine__DOT__token_idx,
                          9,(IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__gate_w),
                          32,vlSelf->tb_h67_score_class_row_engine__DOT__expected_gate
                          [(7U & vlSelf->tb_h67_score_class_row_engine__DOT__token_idx)],
                          32,vlSelf->tb_h67_score_class_row_engine__DOT__expected_score
                          [(7U & vlSelf->tb_h67_score_class_row_engine__DOT__token_idx)]);
                tb_h67_score_class_row_engine__DOT__errors 
                    = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
            }
            tb_h67_score_class_row_engine__DOT__output_count 
                = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__output_count);
        }
    }
    vlSelf->tb_h67_score_class_row_engine__DOT__out_ready = 1U;
    if (VL_UNLIKELY((tb_h67_score_class_row_engine__DOT__output_count 
                     != tb_h67_score_class_row_engine__DOT__expected_outputs))) {
        VL_WRITEF("ERROR output count got=%0d expected=%0d\n",
                  32,tb_h67_score_class_row_engine__DOT__output_count,
                  32,tb_h67_score_class_row_engine__DOT__expected_outputs);
        tb_h67_score_class_row_engine__DOT__errors 
            = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
    }
    if (VL_UNLIKELY(((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__tokens_loaded_q) 
                     != (0xfU & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__n_tokens)))) {
        VL_WRITEF("ERROR loaded count got=%0# expected=%0d\n",
                  4,vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__tokens_loaded_q,
                  32,__Vtask_tb_h67_score_class_row_engine__DOT__run_row__0__n_tokens);
        tb_h67_score_class_row_engine__DOT__errors 
            = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
    }
    if (VL_UNLIKELY(((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__folded_tokens_q) 
                     != (0xfU & tb_h67_score_class_row_engine__DOT__expected_folded)))) {
        VL_WRITEF("ERROR folded count got=%0# expected=%0d\n",
                  4,vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__folded_tokens_q,
                  32,tb_h67_score_class_row_engine__DOT__expected_folded);
        tb_h67_score_class_row_engine__DOT__errors 
            = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
    }
    if (VL_UNLIKELY(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_range_error_q)) {
        VL_WRITEF("ERROR unexpected score range error\n");
        tb_h67_score_class_row_engine__DOT__errors 
            = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
    }
    co_await vlSelf->__VtrigSched_h9aed8dba__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_h67_score_class_row_engine.clk)", 
                                                       "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                       320);
    if (VL_UNLIKELY((3U != (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__fold_classes_q)))) {
        VL_WRITEF("ERROR fold class count got=%0# expected=3\n",
                  6,vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__fold_classes_q);
        tb_h67_score_class_row_engine__DOT__errors 
            = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
    }
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__enable_fold = 0U;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__n_tokens = 8U;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__n_tokens 
        = __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__n_tokens;
    tb_h67_score_class_row_engine__DOT__row_max = 0xffff8000U;
    tb_h67_score_class_row_engine__DOT__row_sum = 0U;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__idx = 0U;
    while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__idx, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__n_tokens)) {
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__peer 
            = vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector
            [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__idx)];
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__k 
            = vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector
            [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__idx)];
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__q 
            = vlSelf->tb_h67_score_class_row_engine__DOT__q_vector
            [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__idx)];
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__q 
               & __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__k);
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout = 0U;
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 1U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 2U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 3U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 4U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 5U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 6U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 7U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 8U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 9U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 0xaU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 0xbU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 0xcU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 0xdU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 0xeU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 0xfU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 0x10U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 0x11U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 0x12U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 0x13U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 0x14U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 0x15U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 0x16U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 0x17U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 0x18U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 0x19U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 0x1aU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 0x1bU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 0x1cU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 0x1dU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                        >> 0x1eU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout 
               + (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__value 
                  >> 0x1fU));
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__overlap_count 
            = __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__14__Vfuncout;
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
            = ((~ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__q) 
               & (~ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__k));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout = 0U;
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 1U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 2U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 3U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 4U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 5U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 6U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 7U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 8U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 9U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 0xaU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 0xbU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 0xcU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 0xdU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 0xeU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 0xfU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 0x10U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 0x11U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 0x12U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 0x13U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 0x14U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 0x15U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 0x16U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 0x17U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 0x18U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 0x19U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 0x1aU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 0x1bU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 0x1cU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 0x1dU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                        >> 0x1eU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout 
               + (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__value 
                  >> 0x1fU));
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__same_zero_count 
            = __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__15__Vfuncout;
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__k 
               ^ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__peer);
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout = 0U;
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 1U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 2U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 3U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 4U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 5U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 6U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 7U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 8U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 9U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 0xaU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 0xbU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 0xcU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 0xdU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 0xeU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 0xfU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 0x10U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 0x11U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 0x12U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 0x13U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 0x14U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 0x15U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 0x16U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 0x17U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 0x18U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 0x19U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 0x1aU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 0x1bU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 0x1cU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 0x1dU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                        >> 0x1eU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout 
               + (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__value 
                  >> 0x1fU));
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__motion_count 
            = __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__16__Vfuncout;
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__Vfuncout 
            = ((VL_MULS_III(32, (IData)(4U), __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__overlap_count) 
                + __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__motion_count) 
               + ([&]() {
                    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__17__integer_base 
                        = (VL_MULS_III(32, (IData)(4U), __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__overlap_count) 
                           + __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__motion_count);
                    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__17__count 
                        = __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__same_zero_count;
                    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__17__quotient 
                        = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__17__count, 4U);
                    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__17__remainder 
                        = (0xfU & __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__17__count);
                    if ((VL_LTS_III(32, 8U, __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__17__remainder) 
                         | ((8U == __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__17__remainder) 
                            & (0U != VL_MODDIVS_III(32, 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__17__integer_base 
                                                     + __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__17__quotient), (IData)(2U)))))) {
                        __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__17__quotient 
                            = ((IData)(1U) + __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__17__quotient);
                    }
                    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__17__Vfuncout 
                        = __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__17__quotient;
                }(), __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__17__Vfuncout));
        vlSelf->tb_h67_score_class_row_engine__DOT__expected_score[(7U 
                                                                    & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__idx)] 
            = __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__13__Vfuncout;
        if (VL_GTS_III(32, vlSelf->tb_h67_score_class_row_engine__DOT__expected_score
                       [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__idx)], tb_h67_score_class_row_engine__DOT__row_max)) {
            tb_h67_score_class_row_engine__DOT__row_max 
                = vlSelf->tb_h67_score_class_row_engine__DOT__expected_score
                [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__idx)];
        }
        __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__idx 
            = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__idx);
    }
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__idx = 0U;
    while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__idx, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__n_tokens)) {
        __Vtemp_217 = (tb_h67_score_class_row_engine__DOT__row_sum 
                       + ([&]() {
                    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__delta_q7 
                        = (vlSelf->tb_h67_score_class_row_engine__DOT__expected_score
                           [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__idx)] 
                           - tb_h67_score_class_row_engine__DOT__row_max);
                    if (VL_LTES_III(32, 0U, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__delta_q7)) {
                        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__Vfuncout = 0x100U;
                    } else {
                        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__abs_delta 
                            = (- __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__delta_q7);
                        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__integer_shift 
                            = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__abs_delta, 7U);
                        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index 
                            = (0xfU & VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__abs_delta, 3U));
                        if ((0U != (7U & __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__abs_delta))) {
                            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index 
                                = ((IData)(1U) + __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index);
                        }
                        if (VL_LTS_III(32, 0xfU, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index)) {
                            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index = 0xfU;
                        }
                        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_value 
                            = (((((((((0U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index) 
                                      | (1U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index)) 
                                     | (2U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index)) 
                                    | (3U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index)) 
                                   | (4U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index)) 
                                  | (5U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index)) 
                                 | (6U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index)) 
                                | (7U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index))
                                ? ((0U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index)
                                    ? 0x100U : ((1U 
                                                 == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index)
                                                 ? 0xf5U
                                                 : 
                                                ((2U 
                                                  == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index)
                                                  ? 0xeaU
                                                  : 
                                                 ((3U 
                                                   == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index)
                                                   ? 0xe0U
                                                   : 
                                                  ((4U 
                                                    == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index)
                                                    ? 0xd7U
                                                    : 
                                                   ((5U 
                                                     == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index)
                                                     ? 0xcdU
                                                     : 
                                                    ((6U 
                                                      == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index)
                                                      ? 0xc4U
                                                      : 0xbcU)))))))
                                : ((8U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index)
                                    ? 0xb5U : ((9U 
                                                == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index)
                                                ? 0xadU
                                                : (
                                                   (0xaU 
                                                    == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index)
                                                    ? 0xa5U
                                                    : 
                                                   ((0xbU 
                                                     == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index)
                                                     ? 0x9eU
                                                     : 
                                                    ((0xcU 
                                                      == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index)
                                                      ? 0x98U
                                                      : 
                                                     ((0xdU 
                                                       == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index)
                                                       ? 0x91U
                                                       : 
                                                      ((0xeU 
                                                        == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_index)
                                                        ? 0x8bU
                                                        : 0x85U))))))));
                        if (VL_LTS_III(32, 8U, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__integer_shift)) {
                            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__integer_shift = 8U;
                        }
                        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__Vfuncout 
                            = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__fraction_value, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__integer_shift);
                    }
                }(), __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__18__Vfuncout));
        tb_h67_score_class_row_engine__DOT__row_sum 
            = __Vtemp_217;
        __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__idx 
            = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__idx);
    }
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__19__value 
        = tb_h67_score_class_row_engine__DOT__row_sum;
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__19__probe 
        = (VL_GTES_III(32, 1U, __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__19__value)
            ? 1U : (__Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__19__value 
                    - (IData)(1U)));
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__19__shift_value = 0U;
    while (VL_LTS_III(32, 0U, __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__19__probe)) {
        __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__19__probe 
            = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__19__probe, 1U);
        __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__19__shift_value 
            = ((IData)(1U) + __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__19__shift_value);
    }
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__19__Vfuncout 
        = __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__19__shift_value;
    tb_h67_score_class_row_engine__DOT__denominator_shift 
        = __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__19__Vfuncout;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__idx = 0U;
    while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__idx, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__n_tokens)) {
        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__delta_q7 
            = (vlSelf->tb_h67_score_class_row_engine__DOT__expected_score
               [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__idx)] 
               - tb_h67_score_class_row_engine__DOT__row_max);
        if (VL_LTES_III(32, 0U, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__delta_q7)) {
            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__Vfuncout = 0x100U;
        } else {
            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__abs_delta 
                = (- __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__delta_q7);
            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__integer_shift 
                = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__abs_delta, 7U);
            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index 
                = (0xfU & VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__abs_delta, 3U));
            if ((0U != (7U & __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__abs_delta))) {
                __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index 
                    = ((IData)(1U) + __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index);
            }
            if (VL_LTS_III(32, 0xfU, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index)) {
                __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index = 0xfU;
            }
            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_value 
                = (((((((((0U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index) 
                          | (1U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index)) 
                         | (2U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index)) 
                        | (3U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index)) 
                       | (4U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index)) 
                      | (5U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index)) 
                     | (6U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index)) 
                    | (7U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index))
                    ? ((0U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index)
                        ? 0x100U : ((1U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index)
                                     ? 0xf5U : ((2U 
                                                 == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index)
                                                 ? 0xeaU
                                                 : 
                                                ((3U 
                                                  == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index)
                                                  ? 0xe0U
                                                  : 
                                                 ((4U 
                                                   == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index)
                                                   ? 0xd7U
                                                   : 
                                                  ((5U 
                                                    == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index)
                                                    ? 0xcdU
                                                    : 
                                                   ((6U 
                                                     == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index)
                                                     ? 0xc4U
                                                     : 0xbcU)))))))
                    : ((8U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index)
                        ? 0xb5U : ((9U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index)
                                    ? 0xadU : ((0xaU 
                                                == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index)
                                                ? 0xa5U
                                                : (
                                                   (0xbU 
                                                    == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index)
                                                    ? 0x9eU
                                                    : 
                                                   ((0xcU 
                                                     == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index)
                                                     ? 0x98U
                                                     : 
                                                    ((0xdU 
                                                      == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index)
                                                      ? 0x91U
                                                      : 
                                                     ((0xeU 
                                                       == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_index)
                                                       ? 0x8bU
                                                       : 0x85U))))))));
            if (VL_LTS_III(32, 8U, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__integer_shift)) {
                __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__integer_shift = 8U;
            }
            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__Vfuncout 
                = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__fraction_value, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__integer_shift);
        }
        __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__exp_value 
            = __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__20__Vfuncout;
        __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__scaled 
            = VL_MULS_III(32, (IData)(0x80U), VL_MULS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__exp_value, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__n_tokens));
        __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__shift_value 
            = tb_h67_score_class_row_engine__DOT__denominator_shift;
        __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__value 
            = __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__scaled;
        if ((0U == __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__shift_value)) {
            __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__Vfuncout 
                = __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__value;
        } else {
            __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__quotient 
                = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__value, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__shift_value);
            __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__remainder 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__value 
                   - VL_SHIFTL_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__quotient, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__shift_value));
            __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__half 
                = VL_SHIFTL_III(32,32,32, (IData)(1U), 
                                (__Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__shift_value 
                                 - (IData)(1U)));
            if ((VL_GTS_III(32, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__remainder, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__half) 
                 | ((__Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__remainder 
                     == __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__half) 
                    & __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__quotient))) {
                __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__quotient 
                    = ((IData)(1U) + __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__quotient);
            }
            __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__Vfuncout 
                = __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__quotient;
        }
        vlSelf->tb_h67_score_class_row_engine__DOT__expected_gate[(7U 
                                                                   & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__idx)] 
            = __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__21__Vfuncout;
        if (VL_LTS_III(32, 0x100U, vlSelf->tb_h67_score_class_row_engine__DOT__expected_gate
                       [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__idx)])) {
            vlSelf->tb_h67_score_class_row_engine__DOT__expected_gate[(7U 
                                                                       & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__idx)] = 0x100U;
        }
        __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__idx 
            = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__12__idx);
    }
    vlSelf->tb_h67_score_class_row_engine__DOT__cfg_n_tokens 
        = (0xfU & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__n_tokens);
    vlSelf->tb_h67_score_class_row_engine__DOT__cfg_enable_score_fold 
        = __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__enable_fold;
    co_await vlSelf->__VtrigSched_h9aed8d87__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(negedge tb_h67_score_class_row_engine.clk)", 
                                                       "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                       257);
    vlSelf->tb_h67_score_class_row_engine__DOT__cfg_start = 1U;
    co_await vlSelf->__VtrigSched_h9aed8dba__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_h67_score_class_row_engine.clk)", 
                                                       "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                       259);
    co_await vlSelf->__VtrigSched_h9aed8d87__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(negedge tb_h67_score_class_row_engine.clk)", 
                                                       "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                       260);
    vlSelf->tb_h67_score_class_row_engine__DOT__cfg_start = 0U;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__idx = 0U;
    while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__idx, __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__n_tokens)) {
        while ((2U != (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))) {
            co_await vlSelf->__VtrigSched_h9aed8d87__0.trigger(0U, 
                                                               nullptr, 
                                                               "@(negedge tb_h67_score_class_row_engine.clk)", 
                                                               "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                               264);
        }
        vlSelf->tb_h67_score_class_row_engine__DOT__in_valid = 1U;
        vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
            = vlSelf->tb_h67_score_class_row_engine__DOT__q_vector
            [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__idx)];
        if (vlSelf->tb_h67_score_class_row_engine__DOT__time_vector
            [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__idx)]) {
            vlSelf->tb_h67_score_class_row_engine__DOT__in_time_sel = 1U;
            vlSelf->tb_h67_score_class_row_engine__DOT__in_k_pair_bits 
                = (((QData)((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector
                                    [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__idx)])) 
                    << 0x20U) | (QData)((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector
                                                [(7U 
                                                  & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__idx)])));
        } else {
            vlSelf->tb_h67_score_class_row_engine__DOT__in_time_sel = 0U;
            vlSelf->tb_h67_score_class_row_engine__DOT__in_k_pair_bits 
                = (((QData)((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector
                                    [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__idx)])) 
                    << 0x20U) | (QData)((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector
                                                [(7U 
                                                  & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__idx)])));
        }
        vlSelf->tb_h67_score_class_row_engine__DOT__in_last 
            = (__Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__idx 
               == (__Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__n_tokens 
                   - (IData)(1U)));
        co_await vlSelf->__VtrigSched_h9aed8dba__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_h67_score_class_row_engine.clk)", 
                                                           "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                           274);
        co_await vlSelf->__VtrigSched_h9aed8d87__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(negedge tb_h67_score_class_row_engine.clk)", 
                                                           "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                           275);
        vlSelf->tb_h67_score_class_row_engine__DOT__in_valid = 0U;
        vlSelf->tb_h67_score_class_row_engine__DOT__in_last = 0U;
        __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__idx 
            = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__idx);
    }
    tb_h67_score_class_row_engine__DOT__output_count = 0U;
    tb_h67_score_class_row_engine__DOT__expected_outputs = 0U;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__idx = 0U;
    while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__idx, __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__n_tokens)) {
        if ((1U & ((~ (IData)(__Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__enable_fold)) 
                   | (0U != vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector
                      [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__idx)])))) {
            tb_h67_score_class_row_engine__DOT__expected_outputs 
                = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__expected_outputs);
        }
        __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__idx 
            = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__idx);
    }
    tb_h67_score_class_row_engine__DOT__expected_folded 
        = ((IData)(__Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__enable_fold)
            ? (__Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__n_tokens 
               - tb_h67_score_class_row_engine__DOT__expected_outputs)
            : 0U);
    while ((7U != (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))) {
        co_await vlSelf->__VtrigSched_h9aed8d87__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(negedge tb_h67_score_class_row_engine.clk)", 
                                                           "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                           288);
        vlSelf->tb_h67_score_class_row_engine__DOT__out_ready 
            = (1U & ((~ tb_h67_score_class_row_engine__DOT__output_count) 
                     | (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__out_ready))));
        if (((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__out_valid) 
             & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__out_ready))) {
            vlSelf->tb_h67_score_class_row_engine__DOT__token_idx 
                = (0xfU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w));
            if (VL_UNLIKELY(((IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w 
                                      >> 4U)) != vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector
                             [(7U & vlSelf->tb_h67_score_class_row_engine__DOT__token_idx)]))) {
                VL_WRITEF("ERROR token %0d K mismatch\n",
                          32,vlSelf->tb_h67_score_class_row_engine__DOT__token_idx);
                tb_h67_score_class_row_engine__DOT__errors 
                    = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
            }
            if (VL_UNLIKELY(((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__gate_w) 
                             != (0x1ffU & vlSelf->tb_h67_score_class_row_engine__DOT__expected_gate
                                 [(7U & vlSelf->tb_h67_score_class_row_engine__DOT__token_idx)])))) {
                VL_WRITEF("ERROR token %0d gate got=%0# expected=%0d score=%0d\n",
                          32,vlSelf->tb_h67_score_class_row_engine__DOT__token_idx,
                          9,(IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__gate_w),
                          32,vlSelf->tb_h67_score_class_row_engine__DOT__expected_gate
                          [(7U & vlSelf->tb_h67_score_class_row_engine__DOT__token_idx)],
                          32,vlSelf->tb_h67_score_class_row_engine__DOT__expected_score
                          [(7U & vlSelf->tb_h67_score_class_row_engine__DOT__token_idx)]);
                tb_h67_score_class_row_engine__DOT__errors 
                    = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
            }
            tb_h67_score_class_row_engine__DOT__output_count 
                = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__output_count);
        }
    }
    vlSelf->tb_h67_score_class_row_engine__DOT__out_ready = 1U;
    if (VL_UNLIKELY((tb_h67_score_class_row_engine__DOT__output_count 
                     != tb_h67_score_class_row_engine__DOT__expected_outputs))) {
        VL_WRITEF("ERROR output count got=%0d expected=%0d\n",
                  32,tb_h67_score_class_row_engine__DOT__output_count,
                  32,tb_h67_score_class_row_engine__DOT__expected_outputs);
        tb_h67_score_class_row_engine__DOT__errors 
            = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
    }
    if (VL_UNLIKELY(((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__tokens_loaded_q) 
                     != (0xfU & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__n_tokens)))) {
        VL_WRITEF("ERROR loaded count got=%0# expected=%0d\n",
                  4,vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__tokens_loaded_q,
                  32,__Vtask_tb_h67_score_class_row_engine__DOT__run_row__11__n_tokens);
        tb_h67_score_class_row_engine__DOT__errors 
            = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
    }
    if (VL_UNLIKELY(((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__folded_tokens_q) 
                     != (0xfU & tb_h67_score_class_row_engine__DOT__expected_folded)))) {
        VL_WRITEF("ERROR folded count got=%0# expected=%0d\n",
                  4,vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__folded_tokens_q,
                  32,tb_h67_score_class_row_engine__DOT__expected_folded);
        tb_h67_score_class_row_engine__DOT__errors 
            = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
    }
    if (VL_UNLIKELY(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_range_error_q)) {
        VL_WRITEF("ERROR unexpected score range error\n");
        tb_h67_score_class_row_engine__DOT__errors 
            = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
    }
    co_await vlSelf->__VtrigSched_h9aed8dba__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_h67_score_class_row_engine.clk)", 
                                                       "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                       320);
    if (VL_UNLIKELY((0U != (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__folded_tokens_q)))) {
        VL_WRITEF("ERROR fold disabled but folded=%0#\n",
                  4,vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__folded_tokens_q);
        tb_h67_score_class_row_engine__DOT__errors 
            = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
    }
    __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_active__22__n_tokens = 8U;
    __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_active__22__idx = 0U;
    while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_active__22__idx, __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_active__22__n_tokens)) {
        vlSelf->tb_h67_score_class_row_engine__DOT__q_vector[(7U 
                                                              & __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_active__22__idx)] 
            = (0x9e3779b9U ^ __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_active__22__idx);
        vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector[(7U 
                                                                      & __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_active__22__idx)] 
            = VL_SHIFTL_III(32,32,32, (IData)(1U), 
                            VL_MODDIVS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_active__22__idx, (IData)(0x20U)));
        vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector[(7U 
                                                                   & __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_active__22__idx)] 
            = VL_SHIFTR_III(32,32,32, 0x80000000U, 
                            VL_MODDIVS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_active__22__idx, (IData)(0x20U)));
        vlSelf->tb_h67_score_class_row_engine__DOT__time_vector[(7U 
                                                                 & __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_active__22__idx)] 
            = (0U != VL_MODDIVS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_active__22__idx, (IData)(2U)));
        __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_active__22__idx 
            = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_active__22__idx);
    }
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__enable_fold = 1U;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__n_tokens = 8U;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__n_tokens 
        = __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__n_tokens;
    tb_h67_score_class_row_engine__DOT__row_max = 0xffff8000U;
    tb_h67_score_class_row_engine__DOT__row_sum = 0U;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__idx = 0U;
    while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__idx, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__n_tokens)) {
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__peer 
            = vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector
            [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__idx)];
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__k 
            = vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector
            [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__idx)];
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__q 
            = vlSelf->tb_h67_score_class_row_engine__DOT__q_vector
            [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__idx)];
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__q 
               & __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__k);
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout = 0U;
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 1U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 2U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 3U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 4U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 5U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 6U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 7U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 8U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 9U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 0xaU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 0xbU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 0xcU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 0xdU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 0xeU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 0xfU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 0x10U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 0x11U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 0x12U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 0x13U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 0x14U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 0x15U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 0x16U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 0x17U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 0x18U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 0x19U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 0x1aU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 0x1bU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 0x1cU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 0x1dU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                        >> 0x1eU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout 
               + (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__value 
                  >> 0x1fU));
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__overlap_count 
            = __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__26__Vfuncout;
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
            = ((~ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__q) 
               & (~ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__k));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout = 0U;
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 1U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 2U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 3U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 4U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 5U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 6U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 7U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 8U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 9U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 0xaU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 0xbU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 0xcU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 0xdU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 0xeU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 0xfU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 0x10U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 0x11U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 0x12U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 0x13U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 0x14U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 0x15U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 0x16U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 0x17U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 0x18U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 0x19U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 0x1aU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 0x1bU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 0x1cU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 0x1dU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                        >> 0x1eU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout 
               + (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__value 
                  >> 0x1fU));
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__same_zero_count 
            = __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__27__Vfuncout;
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__k 
               ^ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__peer);
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout = 0U;
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 1U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 2U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 3U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 4U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 5U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 6U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 7U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 8U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 9U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 0xaU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 0xbU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 0xcU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 0xdU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 0xeU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 0xfU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 0x10U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 0x11U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 0x12U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 0x13U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 0x14U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 0x15U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 0x16U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 0x17U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 0x18U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 0x19U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 0x1aU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 0x1bU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 0x1cU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 0x1dU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                        >> 0x1eU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout 
               + (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__value 
                  >> 0x1fU));
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__motion_count 
            = __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__28__Vfuncout;
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__Vfuncout 
            = ((VL_MULS_III(32, (IData)(4U), __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__overlap_count) 
                + __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__motion_count) 
               + ([&]() {
                    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__29__integer_base 
                        = (VL_MULS_III(32, (IData)(4U), __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__overlap_count) 
                           + __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__motion_count);
                    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__29__count 
                        = __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__same_zero_count;
                    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__29__quotient 
                        = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__29__count, 4U);
                    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__29__remainder 
                        = (0xfU & __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__29__count);
                    if ((VL_LTS_III(32, 8U, __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__29__remainder) 
                         | ((8U == __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__29__remainder) 
                            & (0U != VL_MODDIVS_III(32, 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__29__integer_base 
                                                     + __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__29__quotient), (IData)(2U)))))) {
                        __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__29__quotient 
                            = ((IData)(1U) + __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__29__quotient);
                    }
                    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__29__Vfuncout 
                        = __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__29__quotient;
                }(), __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__29__Vfuncout));
        vlSelf->tb_h67_score_class_row_engine__DOT__expected_score[(7U 
                                                                    & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__idx)] 
            = __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__25__Vfuncout;
        if (VL_GTS_III(32, vlSelf->tb_h67_score_class_row_engine__DOT__expected_score
                       [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__idx)], tb_h67_score_class_row_engine__DOT__row_max)) {
            tb_h67_score_class_row_engine__DOT__row_max 
                = vlSelf->tb_h67_score_class_row_engine__DOT__expected_score
                [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__idx)];
        }
        __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__idx 
            = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__idx);
    }
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__idx = 0U;
    while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__idx, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__n_tokens)) {
        __Vtemp_336 = (tb_h67_score_class_row_engine__DOT__row_sum 
                       + ([&]() {
                    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__delta_q7 
                        = (vlSelf->tb_h67_score_class_row_engine__DOT__expected_score
                           [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__idx)] 
                           - tb_h67_score_class_row_engine__DOT__row_max);
                    if (VL_LTES_III(32, 0U, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__delta_q7)) {
                        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__Vfuncout = 0x100U;
                    } else {
                        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__abs_delta 
                            = (- __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__delta_q7);
                        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__integer_shift 
                            = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__abs_delta, 7U);
                        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index 
                            = (0xfU & VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__abs_delta, 3U));
                        if ((0U != (7U & __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__abs_delta))) {
                            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index 
                                = ((IData)(1U) + __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index);
                        }
                        if (VL_LTS_III(32, 0xfU, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index)) {
                            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index = 0xfU;
                        }
                        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_value 
                            = (((((((((0U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index) 
                                      | (1U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index)) 
                                     | (2U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index)) 
                                    | (3U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index)) 
                                   | (4U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index)) 
                                  | (5U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index)) 
                                 | (6U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index)) 
                                | (7U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index))
                                ? ((0U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index)
                                    ? 0x100U : ((1U 
                                                 == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index)
                                                 ? 0xf5U
                                                 : 
                                                ((2U 
                                                  == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index)
                                                  ? 0xeaU
                                                  : 
                                                 ((3U 
                                                   == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index)
                                                   ? 0xe0U
                                                   : 
                                                  ((4U 
                                                    == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index)
                                                    ? 0xd7U
                                                    : 
                                                   ((5U 
                                                     == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index)
                                                     ? 0xcdU
                                                     : 
                                                    ((6U 
                                                      == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index)
                                                      ? 0xc4U
                                                      : 0xbcU)))))))
                                : ((8U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index)
                                    ? 0xb5U : ((9U 
                                                == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index)
                                                ? 0xadU
                                                : (
                                                   (0xaU 
                                                    == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index)
                                                    ? 0xa5U
                                                    : 
                                                   ((0xbU 
                                                     == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index)
                                                     ? 0x9eU
                                                     : 
                                                    ((0xcU 
                                                      == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index)
                                                      ? 0x98U
                                                      : 
                                                     ((0xdU 
                                                       == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index)
                                                       ? 0x91U
                                                       : 
                                                      ((0xeU 
                                                        == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_index)
                                                        ? 0x8bU
                                                        : 0x85U))))))));
                        if (VL_LTS_III(32, 8U, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__integer_shift)) {
                            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__integer_shift = 8U;
                        }
                        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__Vfuncout 
                            = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__fraction_value, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__integer_shift);
                    }
                }(), __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__30__Vfuncout));
        tb_h67_score_class_row_engine__DOT__row_sum 
            = __Vtemp_336;
        __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__idx 
            = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__idx);
    }
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__31__value 
        = tb_h67_score_class_row_engine__DOT__row_sum;
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__31__probe 
        = (VL_GTES_III(32, 1U, __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__31__value)
            ? 1U : (__Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__31__value 
                    - (IData)(1U)));
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__31__shift_value = 0U;
    while (VL_LTS_III(32, 0U, __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__31__probe)) {
        __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__31__probe 
            = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__31__probe, 1U);
        __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__31__shift_value 
            = ((IData)(1U) + __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__31__shift_value);
    }
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__31__Vfuncout 
        = __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__31__shift_value;
    tb_h67_score_class_row_engine__DOT__denominator_shift 
        = __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__31__Vfuncout;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__idx = 0U;
    while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__idx, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__n_tokens)) {
        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__delta_q7 
            = (vlSelf->tb_h67_score_class_row_engine__DOT__expected_score
               [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__idx)] 
               - tb_h67_score_class_row_engine__DOT__row_max);
        if (VL_LTES_III(32, 0U, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__delta_q7)) {
            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__Vfuncout = 0x100U;
        } else {
            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__abs_delta 
                = (- __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__delta_q7);
            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__integer_shift 
                = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__abs_delta, 7U);
            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index 
                = (0xfU & VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__abs_delta, 3U));
            if ((0U != (7U & __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__abs_delta))) {
                __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index 
                    = ((IData)(1U) + __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index);
            }
            if (VL_LTS_III(32, 0xfU, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index)) {
                __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index = 0xfU;
            }
            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_value 
                = (((((((((0U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index) 
                          | (1U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index)) 
                         | (2U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index)) 
                        | (3U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index)) 
                       | (4U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index)) 
                      | (5U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index)) 
                     | (6U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index)) 
                    | (7U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index))
                    ? ((0U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index)
                        ? 0x100U : ((1U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index)
                                     ? 0xf5U : ((2U 
                                                 == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index)
                                                 ? 0xeaU
                                                 : 
                                                ((3U 
                                                  == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index)
                                                  ? 0xe0U
                                                  : 
                                                 ((4U 
                                                   == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index)
                                                   ? 0xd7U
                                                   : 
                                                  ((5U 
                                                    == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index)
                                                    ? 0xcdU
                                                    : 
                                                   ((6U 
                                                     == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index)
                                                     ? 0xc4U
                                                     : 0xbcU)))))))
                    : ((8U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index)
                        ? 0xb5U : ((9U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index)
                                    ? 0xadU : ((0xaU 
                                                == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index)
                                                ? 0xa5U
                                                : (
                                                   (0xbU 
                                                    == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index)
                                                    ? 0x9eU
                                                    : 
                                                   ((0xcU 
                                                     == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index)
                                                     ? 0x98U
                                                     : 
                                                    ((0xdU 
                                                      == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index)
                                                      ? 0x91U
                                                      : 
                                                     ((0xeU 
                                                       == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_index)
                                                       ? 0x8bU
                                                       : 0x85U))))))));
            if (VL_LTS_III(32, 8U, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__integer_shift)) {
                __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__integer_shift = 8U;
            }
            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__Vfuncout 
                = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__fraction_value, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__integer_shift);
        }
        __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__exp_value 
            = __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__32__Vfuncout;
        __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__scaled 
            = VL_MULS_III(32, (IData)(0x80U), VL_MULS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__exp_value, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__n_tokens));
        __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__shift_value 
            = tb_h67_score_class_row_engine__DOT__denominator_shift;
        __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__value 
            = __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__scaled;
        if ((0U == __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__shift_value)) {
            __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__Vfuncout 
                = __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__value;
        } else {
            __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__quotient 
                = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__value, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__shift_value);
            __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__remainder 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__value 
                   - VL_SHIFTL_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__quotient, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__shift_value));
            __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__half 
                = VL_SHIFTL_III(32,32,32, (IData)(1U), 
                                (__Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__shift_value 
                                 - (IData)(1U)));
            if ((VL_GTS_III(32, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__remainder, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__half) 
                 | ((__Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__remainder 
                     == __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__half) 
                    & __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__quotient))) {
                __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__quotient 
                    = ((IData)(1U) + __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__quotient);
            }
            __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__Vfuncout 
                = __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__quotient;
        }
        vlSelf->tb_h67_score_class_row_engine__DOT__expected_gate[(7U 
                                                                   & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__idx)] 
            = __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__33__Vfuncout;
        if (VL_LTS_III(32, 0x100U, vlSelf->tb_h67_score_class_row_engine__DOT__expected_gate
                       [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__idx)])) {
            vlSelf->tb_h67_score_class_row_engine__DOT__expected_gate[(7U 
                                                                       & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__idx)] = 0x100U;
        }
        __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__idx 
            = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__24__idx);
    }
    vlSelf->tb_h67_score_class_row_engine__DOT__cfg_n_tokens 
        = (0xfU & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__n_tokens);
    vlSelf->tb_h67_score_class_row_engine__DOT__cfg_enable_score_fold 
        = __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__enable_fold;
    co_await vlSelf->__VtrigSched_h9aed8d87__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(negedge tb_h67_score_class_row_engine.clk)", 
                                                       "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                       257);
    vlSelf->tb_h67_score_class_row_engine__DOT__cfg_start = 1U;
    co_await vlSelf->__VtrigSched_h9aed8dba__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_h67_score_class_row_engine.clk)", 
                                                       "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                       259);
    co_await vlSelf->__VtrigSched_h9aed8d87__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(negedge tb_h67_score_class_row_engine.clk)", 
                                                       "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                       260);
    vlSelf->tb_h67_score_class_row_engine__DOT__cfg_start = 0U;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__idx = 0U;
    while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__idx, __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__n_tokens)) {
        while ((2U != (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))) {
            co_await vlSelf->__VtrigSched_h9aed8d87__0.trigger(0U, 
                                                               nullptr, 
                                                               "@(negedge tb_h67_score_class_row_engine.clk)", 
                                                               "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                               264);
        }
        vlSelf->tb_h67_score_class_row_engine__DOT__in_valid = 1U;
        vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
            = vlSelf->tb_h67_score_class_row_engine__DOT__q_vector
            [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__idx)];
        if (vlSelf->tb_h67_score_class_row_engine__DOT__time_vector
            [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__idx)]) {
            vlSelf->tb_h67_score_class_row_engine__DOT__in_time_sel = 1U;
            vlSelf->tb_h67_score_class_row_engine__DOT__in_k_pair_bits 
                = (((QData)((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector
                                    [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__idx)])) 
                    << 0x20U) | (QData)((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector
                                                [(7U 
                                                  & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__idx)])));
        } else {
            vlSelf->tb_h67_score_class_row_engine__DOT__in_time_sel = 0U;
            vlSelf->tb_h67_score_class_row_engine__DOT__in_k_pair_bits 
                = (((QData)((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector
                                    [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__idx)])) 
                    << 0x20U) | (QData)((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector
                                                [(7U 
                                                  & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__idx)])));
        }
        vlSelf->tb_h67_score_class_row_engine__DOT__in_last 
            = (__Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__idx 
               == (__Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__n_tokens 
                   - (IData)(1U)));
        co_await vlSelf->__VtrigSched_h9aed8dba__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_h67_score_class_row_engine.clk)", 
                                                           "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                           274);
        co_await vlSelf->__VtrigSched_h9aed8d87__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(negedge tb_h67_score_class_row_engine.clk)", 
                                                           "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                           275);
        vlSelf->tb_h67_score_class_row_engine__DOT__in_valid = 0U;
        vlSelf->tb_h67_score_class_row_engine__DOT__in_last = 0U;
        __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__idx 
            = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__idx);
    }
    tb_h67_score_class_row_engine__DOT__output_count = 0U;
    tb_h67_score_class_row_engine__DOT__expected_outputs = 0U;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__idx = 0U;
    while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__idx, __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__n_tokens)) {
        if ((1U & ((~ (IData)(__Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__enable_fold)) 
                   | (0U != vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector
                      [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__idx)])))) {
            tb_h67_score_class_row_engine__DOT__expected_outputs 
                = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__expected_outputs);
        }
        __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__idx 
            = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__idx);
    }
    tb_h67_score_class_row_engine__DOT__expected_folded 
        = ((IData)(__Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__enable_fold)
            ? (__Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__n_tokens 
               - tb_h67_score_class_row_engine__DOT__expected_outputs)
            : 0U);
    while ((7U != (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))) {
        co_await vlSelf->__VtrigSched_h9aed8d87__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(negedge tb_h67_score_class_row_engine.clk)", 
                                                           "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                           288);
        vlSelf->tb_h67_score_class_row_engine__DOT__out_ready 
            = (1U & ((~ tb_h67_score_class_row_engine__DOT__output_count) 
                     | (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__out_ready))));
        if (((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__out_valid) 
             & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__out_ready))) {
            vlSelf->tb_h67_score_class_row_engine__DOT__token_idx 
                = (0xfU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w));
            if (VL_UNLIKELY(((IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w 
                                      >> 4U)) != vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector
                             [(7U & vlSelf->tb_h67_score_class_row_engine__DOT__token_idx)]))) {
                VL_WRITEF("ERROR token %0d K mismatch\n",
                          32,vlSelf->tb_h67_score_class_row_engine__DOT__token_idx);
                tb_h67_score_class_row_engine__DOT__errors 
                    = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
            }
            if (VL_UNLIKELY(((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__gate_w) 
                             != (0x1ffU & vlSelf->tb_h67_score_class_row_engine__DOT__expected_gate
                                 [(7U & vlSelf->tb_h67_score_class_row_engine__DOT__token_idx)])))) {
                VL_WRITEF("ERROR token %0d gate got=%0# expected=%0d score=%0d\n",
                          32,vlSelf->tb_h67_score_class_row_engine__DOT__token_idx,
                          9,(IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__gate_w),
                          32,vlSelf->tb_h67_score_class_row_engine__DOT__expected_gate
                          [(7U & vlSelf->tb_h67_score_class_row_engine__DOT__token_idx)],
                          32,vlSelf->tb_h67_score_class_row_engine__DOT__expected_score
                          [(7U & vlSelf->tb_h67_score_class_row_engine__DOT__token_idx)]);
                tb_h67_score_class_row_engine__DOT__errors 
                    = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
            }
            tb_h67_score_class_row_engine__DOT__output_count 
                = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__output_count);
        }
    }
    vlSelf->tb_h67_score_class_row_engine__DOT__out_ready = 1U;
    if (VL_UNLIKELY((tb_h67_score_class_row_engine__DOT__output_count 
                     != tb_h67_score_class_row_engine__DOT__expected_outputs))) {
        VL_WRITEF("ERROR output count got=%0d expected=%0d\n",
                  32,tb_h67_score_class_row_engine__DOT__output_count,
                  32,tb_h67_score_class_row_engine__DOT__expected_outputs);
        tb_h67_score_class_row_engine__DOT__errors 
            = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
    }
    if (VL_UNLIKELY(((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__tokens_loaded_q) 
                     != (0xfU & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__n_tokens)))) {
        VL_WRITEF("ERROR loaded count got=%0# expected=%0d\n",
                  4,vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__tokens_loaded_q,
                  32,__Vtask_tb_h67_score_class_row_engine__DOT__run_row__23__n_tokens);
        tb_h67_score_class_row_engine__DOT__errors 
            = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
    }
    if (VL_UNLIKELY(((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__folded_tokens_q) 
                     != (0xfU & tb_h67_score_class_row_engine__DOT__expected_folded)))) {
        VL_WRITEF("ERROR folded count got=%0# expected=%0d\n",
                  4,vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__folded_tokens_q,
                  32,tb_h67_score_class_row_engine__DOT__expected_folded);
        tb_h67_score_class_row_engine__DOT__errors 
            = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
    }
    if (VL_UNLIKELY(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_range_error_q)) {
        VL_WRITEF("ERROR unexpected score range error\n");
        tb_h67_score_class_row_engine__DOT__errors 
            = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
    }
    co_await vlSelf->__VtrigSched_h9aed8dba__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_h67_score_class_row_engine.clk)", 
                                                       "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                       320);
    if (VL_UNLIKELY(((0U != (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__folded_tokens_q)) 
                     | (0U != (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__fold_classes_q))))) {
        VL_WRITEF("ERROR all-active row unexpectedly folded tokens/classes\n");
        tb_h67_score_class_row_engine__DOT__errors 
            = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
    }
    vlSelf->tb_h67_score_class_row_engine__DOT__q_vector[0U] = 1U;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector[0U] = 1U;
    vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector[0U] = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__time_vector[0U] = 0U;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__enable_fold = 1U;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__n_tokens = 1U;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__n_tokens 
        = __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__n_tokens;
    tb_h67_score_class_row_engine__DOT__row_max = 0xffff8000U;
    tb_h67_score_class_row_engine__DOT__row_sum = 0U;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__idx = 0U;
    while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__idx, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__n_tokens)) {
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__peer 
            = vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector
            [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__idx)];
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__k 
            = vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector
            [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__idx)];
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__q 
            = vlSelf->tb_h67_score_class_row_engine__DOT__q_vector
            [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__idx)];
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__q 
               & __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__k);
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout = 0U;
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 1U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 2U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 3U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 4U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 5U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 6U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 7U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 8U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 9U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 0xaU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 0xbU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 0xcU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 0xdU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 0xeU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 0xfU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 0x10U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 0x11U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 0x12U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 0x13U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 0x14U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 0x15U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 0x16U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 0x17U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 0x18U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 0x19U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 0x1aU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 0x1bU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 0x1cU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 0x1dU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                        >> 0x1eU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout 
               + (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__value 
                  >> 0x1fU));
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__overlap_count 
            = __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__37__Vfuncout;
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
            = ((~ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__q) 
               & (~ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__k));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout = 0U;
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 1U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 2U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 3U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 4U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 5U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 6U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 7U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 8U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 9U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 0xaU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 0xbU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 0xcU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 0xdU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 0xeU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 0xfU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 0x10U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 0x11U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 0x12U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 0x13U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 0x14U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 0x15U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 0x16U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 0x17U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 0x18U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 0x19U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 0x1aU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 0x1bU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 0x1cU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 0x1dU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                        >> 0x1eU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout 
               + (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__value 
                  >> 0x1fU));
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__same_zero_count 
            = __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__38__Vfuncout;
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__k 
               ^ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__peer);
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout = 0U;
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 1U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 2U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 3U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 4U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 5U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 6U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 7U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 8U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 9U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 0xaU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 0xbU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 0xcU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 0xdU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 0xeU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 0xfU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 0x10U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 0x11U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 0x12U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 0x13U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 0x14U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 0x15U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 0x16U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 0x17U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 0x18U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 0x19U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 0x1aU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 0x1bU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 0x1cU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 0x1dU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                        >> 0x1eU)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout 
               + (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__value 
                  >> 0x1fU));
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__motion_count 
            = __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__39__Vfuncout;
        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__Vfuncout 
            = ((VL_MULS_III(32, (IData)(4U), __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__overlap_count) 
                + __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__motion_count) 
               + ([&]() {
                    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__40__integer_base 
                        = (VL_MULS_III(32, (IData)(4U), __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__overlap_count) 
                           + __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__motion_count);
                    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__40__count 
                        = __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__same_zero_count;
                    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__40__quotient 
                        = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__40__count, 4U);
                    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__40__remainder 
                        = (0xfU & __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__40__count);
                    if ((VL_LTS_III(32, 8U, __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__40__remainder) 
                         | ((8U == __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__40__remainder) 
                            & (0U != VL_MODDIVS_III(32, 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__40__integer_base 
                                                     + __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__40__quotient), (IData)(2U)))))) {
                        __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__40__quotient 
                            = ((IData)(1U) + __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__40__quotient);
                    }
                    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__40__Vfuncout 
                        = __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__40__quotient;
                }(), __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__40__Vfuncout));
        vlSelf->tb_h67_score_class_row_engine__DOT__expected_score[(7U 
                                                                    & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__idx)] 
            = __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__36__Vfuncout;
        if (VL_GTS_III(32, vlSelf->tb_h67_score_class_row_engine__DOT__expected_score
                       [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__idx)], tb_h67_score_class_row_engine__DOT__row_max)) {
            tb_h67_score_class_row_engine__DOT__row_max 
                = vlSelf->tb_h67_score_class_row_engine__DOT__expected_score
                [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__idx)];
        }
        __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__idx 
            = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__idx);
    }
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__idx = 0U;
    while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__idx, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__n_tokens)) {
        __Vtemp_454 = (tb_h67_score_class_row_engine__DOT__row_sum 
                       + ([&]() {
                    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__delta_q7 
                        = (vlSelf->tb_h67_score_class_row_engine__DOT__expected_score
                           [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__idx)] 
                           - tb_h67_score_class_row_engine__DOT__row_max);
                    if (VL_LTES_III(32, 0U, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__delta_q7)) {
                        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__Vfuncout = 0x100U;
                    } else {
                        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__abs_delta 
                            = (- __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__delta_q7);
                        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__integer_shift 
                            = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__abs_delta, 7U);
                        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index 
                            = (0xfU & VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__abs_delta, 3U));
                        if ((0U != (7U & __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__abs_delta))) {
                            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index 
                                = ((IData)(1U) + __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index);
                        }
                        if (VL_LTS_III(32, 0xfU, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index)) {
                            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index = 0xfU;
                        }
                        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_value 
                            = (((((((((0U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index) 
                                      | (1U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index)) 
                                     | (2U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index)) 
                                    | (3U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index)) 
                                   | (4U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index)) 
                                  | (5U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index)) 
                                 | (6U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index)) 
                                | (7U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index))
                                ? ((0U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index)
                                    ? 0x100U : ((1U 
                                                 == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index)
                                                 ? 0xf5U
                                                 : 
                                                ((2U 
                                                  == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index)
                                                  ? 0xeaU
                                                  : 
                                                 ((3U 
                                                   == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index)
                                                   ? 0xe0U
                                                   : 
                                                  ((4U 
                                                    == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index)
                                                    ? 0xd7U
                                                    : 
                                                   ((5U 
                                                     == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index)
                                                     ? 0xcdU
                                                     : 
                                                    ((6U 
                                                      == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index)
                                                      ? 0xc4U
                                                      : 0xbcU)))))))
                                : ((8U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index)
                                    ? 0xb5U : ((9U 
                                                == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index)
                                                ? 0xadU
                                                : (
                                                   (0xaU 
                                                    == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index)
                                                    ? 0xa5U
                                                    : 
                                                   ((0xbU 
                                                     == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index)
                                                     ? 0x9eU
                                                     : 
                                                    ((0xcU 
                                                      == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index)
                                                      ? 0x98U
                                                      : 
                                                     ((0xdU 
                                                       == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index)
                                                       ? 0x91U
                                                       : 
                                                      ((0xeU 
                                                        == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_index)
                                                        ? 0x8bU
                                                        : 0x85U))))))));
                        if (VL_LTS_III(32, 8U, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__integer_shift)) {
                            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__integer_shift = 8U;
                        }
                        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__Vfuncout 
                            = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__fraction_value, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__integer_shift);
                    }
                }(), __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__41__Vfuncout));
        tb_h67_score_class_row_engine__DOT__row_sum 
            = __Vtemp_454;
        __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__idx 
            = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__idx);
    }
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__42__value 
        = tb_h67_score_class_row_engine__DOT__row_sum;
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__42__probe 
        = (VL_GTES_III(32, 1U, __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__42__value)
            ? 1U : (__Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__42__value 
                    - (IData)(1U)));
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__42__shift_value = 0U;
    while (VL_LTS_III(32, 0U, __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__42__probe)) {
        __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__42__probe 
            = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__42__probe, 1U);
        __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__42__shift_value 
            = ((IData)(1U) + __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__42__shift_value);
    }
    __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__42__Vfuncout 
        = __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__42__shift_value;
    tb_h67_score_class_row_engine__DOT__denominator_shift 
        = __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__42__Vfuncout;
    __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__idx = 0U;
    while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__idx, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__n_tokens)) {
        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__delta_q7 
            = (vlSelf->tb_h67_score_class_row_engine__DOT__expected_score
               [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__idx)] 
               - tb_h67_score_class_row_engine__DOT__row_max);
        if (VL_LTES_III(32, 0U, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__delta_q7)) {
            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__Vfuncout = 0x100U;
        } else {
            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__abs_delta 
                = (- __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__delta_q7);
            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__integer_shift 
                = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__abs_delta, 7U);
            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index 
                = (0xfU & VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__abs_delta, 3U));
            if ((0U != (7U & __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__abs_delta))) {
                __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index 
                    = ((IData)(1U) + __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index);
            }
            if (VL_LTS_III(32, 0xfU, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index)) {
                __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index = 0xfU;
            }
            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_value 
                = (((((((((0U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index) 
                          | (1U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index)) 
                         | (2U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index)) 
                        | (3U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index)) 
                       | (4U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index)) 
                      | (5U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index)) 
                     | (6U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index)) 
                    | (7U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index))
                    ? ((0U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index)
                        ? 0x100U : ((1U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index)
                                     ? 0xf5U : ((2U 
                                                 == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index)
                                                 ? 0xeaU
                                                 : 
                                                ((3U 
                                                  == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index)
                                                  ? 0xe0U
                                                  : 
                                                 ((4U 
                                                   == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index)
                                                   ? 0xd7U
                                                   : 
                                                  ((5U 
                                                    == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index)
                                                    ? 0xcdU
                                                    : 
                                                   ((6U 
                                                     == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index)
                                                     ? 0xc4U
                                                     : 0xbcU)))))))
                    : ((8U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index)
                        ? 0xb5U : ((9U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index)
                                    ? 0xadU : ((0xaU 
                                                == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index)
                                                ? 0xa5U
                                                : (
                                                   (0xbU 
                                                    == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index)
                                                    ? 0x9eU
                                                    : 
                                                   ((0xcU 
                                                     == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index)
                                                     ? 0x98U
                                                     : 
                                                    ((0xdU 
                                                      == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index)
                                                      ? 0x91U
                                                      : 
                                                     ((0xeU 
                                                       == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_index)
                                                       ? 0x8bU
                                                       : 0x85U))))))));
            if (VL_LTS_III(32, 8U, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__integer_shift)) {
                __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__integer_shift = 8U;
            }
            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__Vfuncout 
                = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__fraction_value, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__integer_shift);
        }
        __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__exp_value 
            = __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__43__Vfuncout;
        __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__scaled 
            = VL_MULS_III(32, (IData)(0x80U), VL_MULS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__exp_value, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__n_tokens));
        __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__shift_value 
            = tb_h67_score_class_row_engine__DOT__denominator_shift;
        __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__value 
            = __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__scaled;
        if ((0U == __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__shift_value)) {
            __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__Vfuncout 
                = __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__value;
        } else {
            __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__quotient 
                = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__value, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__shift_value);
            __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__remainder 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__value 
                   - VL_SHIFTL_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__quotient, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__shift_value));
            __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__half 
                = VL_SHIFTL_III(32,32,32, (IData)(1U), 
                                (__Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__shift_value 
                                 - (IData)(1U)));
            if ((VL_GTS_III(32, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__remainder, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__half) 
                 | ((__Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__remainder 
                     == __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__half) 
                    & __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__quotient))) {
                __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__quotient 
                    = ((IData)(1U) + __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__quotient);
            }
            __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__Vfuncout 
                = __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__quotient;
        }
        vlSelf->tb_h67_score_class_row_engine__DOT__expected_gate[(7U 
                                                                   & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__idx)] 
            = __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__44__Vfuncout;
        if (VL_LTS_III(32, 0x100U, vlSelf->tb_h67_score_class_row_engine__DOT__expected_gate
                       [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__idx)])) {
            vlSelf->tb_h67_score_class_row_engine__DOT__expected_gate[(7U 
                                                                       & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__idx)] = 0x100U;
        }
        __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__idx 
            = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__35__idx);
    }
    vlSelf->tb_h67_score_class_row_engine__DOT__cfg_n_tokens 
        = (0xfU & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__n_tokens);
    vlSelf->tb_h67_score_class_row_engine__DOT__cfg_enable_score_fold 
        = __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__enable_fold;
    co_await vlSelf->__VtrigSched_h9aed8d87__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(negedge tb_h67_score_class_row_engine.clk)", 
                                                       "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                       257);
    vlSelf->tb_h67_score_class_row_engine__DOT__cfg_start = 1U;
    co_await vlSelf->__VtrigSched_h9aed8dba__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_h67_score_class_row_engine.clk)", 
                                                       "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                       259);
    co_await vlSelf->__VtrigSched_h9aed8d87__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(negedge tb_h67_score_class_row_engine.clk)", 
                                                       "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                       260);
    vlSelf->tb_h67_score_class_row_engine__DOT__cfg_start = 0U;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__idx = 0U;
    while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__idx, __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__n_tokens)) {
        while ((2U != (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))) {
            co_await vlSelf->__VtrigSched_h9aed8d87__0.trigger(0U, 
                                                               nullptr, 
                                                               "@(negedge tb_h67_score_class_row_engine.clk)", 
                                                               "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                               264);
        }
        vlSelf->tb_h67_score_class_row_engine__DOT__in_valid = 1U;
        vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
            = vlSelf->tb_h67_score_class_row_engine__DOT__q_vector
            [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__idx)];
        if (vlSelf->tb_h67_score_class_row_engine__DOT__time_vector
            [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__idx)]) {
            vlSelf->tb_h67_score_class_row_engine__DOT__in_time_sel = 1U;
            vlSelf->tb_h67_score_class_row_engine__DOT__in_k_pair_bits 
                = (((QData)((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector
                                    [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__idx)])) 
                    << 0x20U) | (QData)((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector
                                                [(7U 
                                                  & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__idx)])));
        } else {
            vlSelf->tb_h67_score_class_row_engine__DOT__in_time_sel = 0U;
            vlSelf->tb_h67_score_class_row_engine__DOT__in_k_pair_bits 
                = (((QData)((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector
                                    [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__idx)])) 
                    << 0x20U) | (QData)((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector
                                                [(7U 
                                                  & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__idx)])));
        }
        vlSelf->tb_h67_score_class_row_engine__DOT__in_last 
            = (__Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__idx 
               == (__Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__n_tokens 
                   - (IData)(1U)));
        co_await vlSelf->__VtrigSched_h9aed8dba__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_h67_score_class_row_engine.clk)", 
                                                           "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                           274);
        co_await vlSelf->__VtrigSched_h9aed8d87__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(negedge tb_h67_score_class_row_engine.clk)", 
                                                           "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                           275);
        vlSelf->tb_h67_score_class_row_engine__DOT__in_valid = 0U;
        vlSelf->tb_h67_score_class_row_engine__DOT__in_last = 0U;
        __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__idx 
            = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__idx);
    }
    tb_h67_score_class_row_engine__DOT__output_count = 0U;
    tb_h67_score_class_row_engine__DOT__expected_outputs = 0U;
    __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__idx = 0U;
    while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__idx, __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__n_tokens)) {
        if ((1U & ((~ (IData)(__Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__enable_fold)) 
                   | (0U != vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector
                      [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__idx)])))) {
            tb_h67_score_class_row_engine__DOT__expected_outputs 
                = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__expected_outputs);
        }
        __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__idx 
            = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__idx);
    }
    tb_h67_score_class_row_engine__DOT__expected_folded 
        = ((IData)(__Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__enable_fold)
            ? (__Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__n_tokens 
               - tb_h67_score_class_row_engine__DOT__expected_outputs)
            : 0U);
    while ((7U != (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))) {
        co_await vlSelf->__VtrigSched_h9aed8d87__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(negedge tb_h67_score_class_row_engine.clk)", 
                                                           "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                           288);
        vlSelf->tb_h67_score_class_row_engine__DOT__out_ready 
            = (1U & ((~ tb_h67_score_class_row_engine__DOT__output_count) 
                     | (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__out_ready))));
        if (((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__out_valid) 
             & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__out_ready))) {
            vlSelf->tb_h67_score_class_row_engine__DOT__token_idx 
                = (0xfU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w));
            if (VL_UNLIKELY(((IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w 
                                      >> 4U)) != vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector
                             [(7U & vlSelf->tb_h67_score_class_row_engine__DOT__token_idx)]))) {
                VL_WRITEF("ERROR token %0d K mismatch\n",
                          32,vlSelf->tb_h67_score_class_row_engine__DOT__token_idx);
                tb_h67_score_class_row_engine__DOT__errors 
                    = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
            }
            if (VL_UNLIKELY(((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__gate_w) 
                             != (0x1ffU & vlSelf->tb_h67_score_class_row_engine__DOT__expected_gate
                                 [(7U & vlSelf->tb_h67_score_class_row_engine__DOT__token_idx)])))) {
                VL_WRITEF("ERROR token %0d gate got=%0# expected=%0d score=%0d\n",
                          32,vlSelf->tb_h67_score_class_row_engine__DOT__token_idx,
                          9,(IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__gate_w),
                          32,vlSelf->tb_h67_score_class_row_engine__DOT__expected_gate
                          [(7U & vlSelf->tb_h67_score_class_row_engine__DOT__token_idx)],
                          32,vlSelf->tb_h67_score_class_row_engine__DOT__expected_score
                          [(7U & vlSelf->tb_h67_score_class_row_engine__DOT__token_idx)]);
                tb_h67_score_class_row_engine__DOT__errors 
                    = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
            }
            tb_h67_score_class_row_engine__DOT__output_count 
                = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__output_count);
        }
    }
    vlSelf->tb_h67_score_class_row_engine__DOT__out_ready = 1U;
    if (VL_UNLIKELY((tb_h67_score_class_row_engine__DOT__output_count 
                     != tb_h67_score_class_row_engine__DOT__expected_outputs))) {
        VL_WRITEF("ERROR output count got=%0d expected=%0d\n",
                  32,tb_h67_score_class_row_engine__DOT__output_count,
                  32,tb_h67_score_class_row_engine__DOT__expected_outputs);
        tb_h67_score_class_row_engine__DOT__errors 
            = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
    }
    if (VL_UNLIKELY(((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__tokens_loaded_q) 
                     != (0xfU & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__n_tokens)))) {
        VL_WRITEF("ERROR loaded count got=%0# expected=%0d\n",
                  4,vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__tokens_loaded_q,
                  32,__Vtask_tb_h67_score_class_row_engine__DOT__run_row__34__n_tokens);
        tb_h67_score_class_row_engine__DOT__errors 
            = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
    }
    if (VL_UNLIKELY(((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__folded_tokens_q) 
                     != (0xfU & tb_h67_score_class_row_engine__DOT__expected_folded)))) {
        VL_WRITEF("ERROR folded count got=%0# expected=%0d\n",
                  4,vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__folded_tokens_q,
                  32,tb_h67_score_class_row_engine__DOT__expected_folded);
        tb_h67_score_class_row_engine__DOT__errors 
            = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
    }
    if (VL_UNLIKELY(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_range_error_q)) {
        VL_WRITEF("ERROR unexpected score range error\n");
        tb_h67_score_class_row_engine__DOT__errors 
            = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
    }
    co_await vlSelf->__VtrigSched_h9aed8dba__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_h67_score_class_row_engine.clk)", 
                                                       "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                       320);
    tb_h67_score_class_row_engine__DOT__unnamedblk1__DOT__all_class_count = 0x23U;
    if (VL_GTES_III(32, 8U, tb_h67_score_class_row_engine__DOT__unnamedblk1__DOT__all_class_count)) {
        __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__class_count = 0x23U;
        __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__target = 0U;
        while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__target, __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__class_count)) {
            __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__found = 0U;
            __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__q_active = 0U;
            while (VL_GTES_III(32, 0x20U, __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__q_active)) {
                __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__peer_active = 0U;
                while (VL_GTES_III(32, 0x20U, __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__peer_active)) {
                    if (((~ (IData)(__Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__found)) 
                         & (([&]() {
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__47__count 
                                            = __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__peer_active;
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__47__Vfuncout 
                                            = (VL_GTES_III(32, 0U, __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__47__count)
                                                ? 0U
                                                : (
                                                   VL_LTES_III(32, 0x20U, __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__47__count)
                                                    ? 0xffffffffU
                                                    : 
                                                   (VL_SHIFTL_III(32,32,32, (IData)(1U), __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__47__count) 
                                                    - (IData)(1U))));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__peer 
                                            = __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__47__Vfuncout;
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__k = 0U;
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__48__count 
                                            = __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__q_active;
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__48__Vfuncout 
                                            = (VL_GTES_III(32, 0U, __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__48__count)
                                                ? 0U
                                                : (
                                                   VL_LTES_III(32, 0x20U, __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__48__count)
                                                    ? 0xffffffffU
                                                    : 
                                                   (VL_SHIFTL_III(32,32,32, (IData)(1U), __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__48__count) 
                                                    - (IData)(1U))));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__q 
                                            = __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__48__Vfuncout;
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__q 
                                               & __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__k);
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout = 0U;
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 1U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 2U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 3U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 4U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 5U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 6U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 7U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 8U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 9U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 0xaU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 0xbU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 0xcU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 0xdU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 0xeU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 0xfU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 0x10U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 0x11U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 0x12U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 0x13U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 0x14U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 0x15U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 0x16U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 0x17U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 0x18U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 0x19U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 0x1aU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 0x1bU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 0x1cU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 0x1dU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                     >> 0x1eU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout 
                                               + (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__value 
                                                  >> 0x1fU));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__overlap_count 
                                            = __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__49__Vfuncout;
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                            = ((~ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__q) 
                                               & (~ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__k));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout = 0U;
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 1U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 2U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 3U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 4U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 5U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 6U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 7U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 8U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 9U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 0xaU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 0xbU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 0xcU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 0xdU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 0xeU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 0xfU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 0x10U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 0x11U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 0x12U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 0x13U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 0x14U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 0x15U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 0x16U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 0x17U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 0x18U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 0x19U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 0x1aU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 0x1bU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 0x1cU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 0x1dU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                     >> 0x1eU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout 
                                               + (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__value 
                                                  >> 0x1fU));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__same_zero_count 
                                            = __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__50__Vfuncout;
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__k 
                                               ^ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__peer);
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout = 0U;
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 1U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 2U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 3U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 4U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 5U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 6U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 7U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 8U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 9U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 0xaU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 0xbU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 0xcU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 0xdU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 0xeU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 0xfU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 0x10U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 0x11U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 0x12U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 0x13U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 0x14U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 0x15U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 0x16U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 0x17U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 0x18U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 0x19U)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 0x1aU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 0x1bU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 0x1cU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 0x1dU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (1U 
                                                  & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                     >> 0x1eU)));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                            = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout 
                                               + (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__value 
                                                  >> 0x1fU));
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__motion_count 
                                            = __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__51__Vfuncout;
                                        __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__Vfuncout 
                                            = ((VL_MULS_III(32, (IData)(4U), __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__overlap_count) 
                                                + __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__motion_count) 
                                               + ([&]() {
                                                    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__52__integer_base 
                                                        = 
                                                        (VL_MULS_III(32, (IData)(4U), __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__overlap_count) 
                                                         + __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__motion_count);
                                                    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__52__count 
                                                        = __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__same_zero_count;
                                                    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__52__quotient 
                                                        = 
                                                        VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__52__count, 4U);
                                                    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__52__remainder 
                                                        = 
                                                        (0xfU 
                                                         & __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__52__count);
                                                    if (
                                                        (VL_LTS_III(32, 8U, __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__52__remainder) 
                                                         | ((8U 
                                                             == __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__52__remainder) 
                                                            & (0U 
                                                               != 
                                                               VL_MODDIVS_III(32, 
                                                                              (__Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__52__integer_base 
                                                                               + __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__52__quotient), (IData)(2U)))))) {
                                                        __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__52__quotient 
                                                            = 
                                                            ((IData)(1U) 
                                                             + __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__52__quotient);
                                                    }
                                                    __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__52__Vfuncout 
                                                        = __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__52__quotient;
                                                }(), __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__52__Vfuncout));
                                    }(), __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__46__Vfuncout) 
                            == __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__target))) {
                        __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__53__count 
                            = __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__q_active;
                        __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__53__Vfuncout 
                            = (VL_GTES_III(32, 0U, __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__53__count)
                                ? 0U : (VL_LTES_III(32, 0x20U, __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__53__count)
                                         ? 0xffffffffU
                                         : (VL_SHIFTL_III(32,32,32, (IData)(1U), __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__53__count) 
                                            - (IData)(1U))));
                        vlSelf->tb_h67_score_class_row_engine__DOT__q_vector[(7U 
                                                                              & __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__target)] 
                            = __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__53__Vfuncout;
                        vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector[(7U 
                                                                                & __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__target)] = 0U;
                        __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__54__count 
                            = __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__peer_active;
                        __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__54__Vfuncout 
                            = (VL_GTES_III(32, 0U, __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__54__count)
                                ? 0U : (VL_LTES_III(32, 0x20U, __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__54__count)
                                         ? 0xffffffffU
                                         : (VL_SHIFTL_III(32,32,32, (IData)(1U), __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__54__count) 
                                            - (IData)(1U))));
                        vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector[(7U 
                                                                                & __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__target)] 
                            = __Vfunc_tb_h67_score_class_row_engine__DOT__low_mask__54__Vfuncout;
                        vlSelf->tb_h67_score_class_row_engine__DOT__time_vector[(7U 
                                                                                & __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__target)] = 0U;
                        __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__found = 1U;
                    }
                    __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__peer_active 
                        = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__peer_active);
                }
                __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__q_active 
                    = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__q_active);
            }
            if (VL_UNLIKELY((1U & (~ (IData)(__Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__found))))) {
                VL_WRITEF("[%0t] %%Fatal: tb_h67_score_class_row_engine.sv:357: Assertion failed in %Ntb_h67_score_class_row_engine.prepare_all_fold_classes: unable to construct fold score class %0d\n",
                          64,VL_TIME_UNITED_Q(1000),
                          -9,vlSymsp->name(),32,__Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__target);
                VL_STOP_MT("tb_h67/tb_h67_score_class_row_engine.sv", 357, "");
            }
            __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__target 
                = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__target);
        }
        tb_h67_score_class_row_engine__DOT__unnamedblk1__DOT__all_class_count 
            = __Vtask_tb_h67_score_class_row_engine__DOT__prepare_all_fold_classes__45__class_count;
        __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__enable_fold = 1U;
        __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__n_tokens 
            = tb_h67_score_class_row_engine__DOT__unnamedblk1__DOT__all_class_count;
        __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__n_tokens 
            = __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__n_tokens;
        tb_h67_score_class_row_engine__DOT__row_max = 0xffff8000U;
        tb_h67_score_class_row_engine__DOT__row_sum = 0U;
        __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__idx = 0U;
        while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__idx, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__n_tokens)) {
            __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__peer 
                = vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector
                [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__idx)];
            __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__k 
                = vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector
                [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__idx)];
            __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__q 
                = vlSelf->tb_h67_score_class_row_engine__DOT__q_vector
                [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__idx)];
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__q 
                   & __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__k);
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout = 0U;
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 1U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 2U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 3U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 4U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 5U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 6U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 7U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 8U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 9U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 0xaU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 0xbU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 0xcU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 0xdU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 0xeU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 0xfU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 0x10U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 0x11U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 0x12U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 0x13U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 0x14U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 0x15U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 0x16U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 0x17U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 0x18U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 0x19U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 0x1aU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 0x1bU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 0x1cU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 0x1dU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                            >> 0x1eU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout 
                   + (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__value 
                      >> 0x1fU));
            __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__overlap_count 
                = __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__58__Vfuncout;
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                = ((~ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__q) 
                   & (~ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__k));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout = 0U;
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 1U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 2U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 3U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 4U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 5U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 6U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 7U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 8U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 9U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 0xaU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 0xbU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 0xcU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 0xdU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 0xeU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 0xfU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 0x10U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 0x11U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 0x12U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 0x13U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 0x14U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 0x15U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 0x16U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 0x17U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 0x18U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 0x19U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 0x1aU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 0x1bU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 0x1cU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 0x1dU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                            >> 0x1eU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout 
                   + (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__value 
                      >> 0x1fU));
            __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__same_zero_count 
                = __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__59__Vfuncout;
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__k 
                   ^ __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__peer);
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout = 0U;
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 1U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 2U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 3U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 4U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 5U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 6U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 7U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 8U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 9U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 0xaU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 0xbU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 0xcU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 0xdU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 0xeU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 0xfU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 0x10U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 0x11U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 0x12U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 0x13U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 0x14U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 0x15U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 0x16U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 0x17U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 0x18U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 0x19U)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 0x1aU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 0x1bU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 0x1cU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 0x1dU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (1U & (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                            >> 0x1eU)));
            __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                = (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout 
                   + (__Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__value 
                      >> 0x1fU));
            __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__motion_count 
                = __Vfunc_tb_h67_score_class_row_engine__DOT__popcount32__60__Vfuncout;
            __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__Vfuncout 
                = ((VL_MULS_III(32, (IData)(4U), __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__overlap_count) 
                    + __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__motion_count) 
                   + ([&]() {
                        __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__61__integer_base 
                            = (VL_MULS_III(32, (IData)(4U), __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__overlap_count) 
                               + __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__motion_count);
                        __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__61__count 
                            = __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__same_zero_count;
                        __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__61__quotient 
                            = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__61__count, 4U);
                        __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__61__remainder 
                            = (0xfU & __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__61__count);
                        if ((VL_LTS_III(32, 8U, __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__61__remainder) 
                             | ((8U == __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__61__remainder) 
                                & (0U != VL_MODDIVS_III(32, 
                                                        (__Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__61__integer_base 
                                                         + __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__61__quotient), (IData)(2U)))))) {
                            __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__61__quotient 
                                = ((IData)(1U) + __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__61__quotient);
                        }
                        __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__61__Vfuncout 
                            = __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__61__quotient;
                    }(), __Vfunc_tb_h67_score_class_row_engine__DOT__round_even_silence__61__Vfuncout));
            vlSelf->tb_h67_score_class_row_engine__DOT__expected_score[(7U 
                                                                        & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__idx)] 
                = __Vfunc_tb_h67_score_class_row_engine__DOT__h67_score__57__Vfuncout;
            if (VL_GTS_III(32, vlSelf->tb_h67_score_class_row_engine__DOT__expected_score
                           [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__idx)], tb_h67_score_class_row_engine__DOT__row_max)) {
                tb_h67_score_class_row_engine__DOT__row_max 
                    = vlSelf->tb_h67_score_class_row_engine__DOT__expected_score
                    [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__idx)];
            }
            __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__idx 
                = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__idx);
        }
        __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__idx = 0U;
        while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__idx, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__n_tokens)) {
            __Vtemp_671 = (tb_h67_score_class_row_engine__DOT__row_sum 
                           + ([&]() {
                        __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__delta_q7 
                            = (vlSelf->tb_h67_score_class_row_engine__DOT__expected_score
                               [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__idx)] 
                               - tb_h67_score_class_row_engine__DOT__row_max);
                        if (VL_LTES_III(32, 0U, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__delta_q7)) {
                            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__Vfuncout = 0x100U;
                        } else {
                            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__abs_delta 
                                = (- __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__delta_q7);
                            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__integer_shift 
                                = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__abs_delta, 7U);
                            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index 
                                = (0xfU & VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__abs_delta, 3U));
                            if ((0U != (7U & __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__abs_delta))) {
                                __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index 
                                    = ((IData)(1U) 
                                       + __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index);
                            }
                            if (VL_LTS_III(32, 0xfU, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index)) {
                                __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index = 0xfU;
                            }
                            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_value 
                                = (((((((((0U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index) 
                                          | (1U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index)) 
                                         | (2U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index)) 
                                        | (3U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index)) 
                                       | (4U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index)) 
                                      | (5U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index)) 
                                     | (6U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index)) 
                                    | (7U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index))
                                    ? ((0U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index)
                                        ? 0x100U : 
                                       ((1U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index)
                                         ? 0xf5U : 
                                        ((2U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index)
                                          ? 0xeaU : 
                                         ((3U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index)
                                           ? 0xe0U : 
                                          ((4U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index)
                                            ? 0xd7U
                                            : ((5U 
                                                == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index)
                                                ? 0xcdU
                                                : (
                                                   (6U 
                                                    == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index)
                                                    ? 0xc4U
                                                    : 0xbcU)))))))
                                    : ((8U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index)
                                        ? 0xb5U : (
                                                   (9U 
                                                    == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index)
                                                    ? 0xadU
                                                    : 
                                                   ((0xaU 
                                                     == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index)
                                                     ? 0xa5U
                                                     : 
                                                    ((0xbU 
                                                      == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index)
                                                      ? 0x9eU
                                                      : 
                                                     ((0xcU 
                                                       == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index)
                                                       ? 0x98U
                                                       : 
                                                      ((0xdU 
                                                        == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index)
                                                        ? 0x91U
                                                        : 
                                                       ((0xeU 
                                                         == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_index)
                                                         ? 0x8bU
                                                         : 0x85U))))))));
                            if (VL_LTS_III(32, 8U, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__integer_shift)) {
                                __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__integer_shift = 8U;
                            }
                            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__Vfuncout 
                                = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__fraction_value, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__integer_shift);
                        }
                    }(), __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__62__Vfuncout));
            tb_h67_score_class_row_engine__DOT__row_sum 
                = __Vtemp_671;
            __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__idx 
                = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__idx);
        }
        __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__63__value 
            = tb_h67_score_class_row_engine__DOT__row_sum;
        __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__63__probe 
            = (VL_GTES_III(32, 1U, __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__63__value)
                ? 1U : (__Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__63__value 
                        - (IData)(1U)));
        __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__63__shift_value = 0U;
        while (VL_LTS_III(32, 0U, __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__63__probe)) {
            __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__63__probe 
                = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__63__probe, 1U);
            __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__63__shift_value 
                = ((IData)(1U) + __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__63__shift_value);
        }
        __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__63__Vfuncout 
            = __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__63__shift_value;
        tb_h67_score_class_row_engine__DOT__denominator_shift 
            = __Vfunc_tb_h67_score_class_row_engine__DOT__ceil_log2__63__Vfuncout;
        __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__idx = 0U;
        while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__idx, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__n_tokens)) {
            __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__delta_q7 
                = (vlSelf->tb_h67_score_class_row_engine__DOT__expected_score
                   [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__idx)] 
                   - tb_h67_score_class_row_engine__DOT__row_max);
            if (VL_LTES_III(32, 0U, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__delta_q7)) {
                __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__Vfuncout = 0x100U;
            } else {
                __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__abs_delta 
                    = (- __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__delta_q7);
                __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__integer_shift 
                    = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__abs_delta, 7U);
                __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index 
                    = (0xfU & VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__abs_delta, 3U));
                if ((0U != (7U & __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__abs_delta))) {
                    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index 
                        = ((IData)(1U) + __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index);
                }
                if (VL_LTS_III(32, 0xfU, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index)) {
                    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index = 0xfU;
                }
                __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_value 
                    = (((((((((0U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index) 
                              | (1U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index)) 
                             | (2U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index)) 
                            | (3U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index)) 
                           | (4U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index)) 
                          | (5U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index)) 
                         | (6U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index)) 
                        | (7U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index))
                        ? ((0U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index)
                            ? 0x100U : ((1U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index)
                                         ? 0xf5U : 
                                        ((2U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index)
                                          ? 0xeaU : 
                                         ((3U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index)
                                           ? 0xe0U : 
                                          ((4U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index)
                                            ? 0xd7U
                                            : ((5U 
                                                == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index)
                                                ? 0xcdU
                                                : (
                                                   (6U 
                                                    == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index)
                                                    ? 0xc4U
                                                    : 0xbcU)))))))
                        : ((8U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index)
                            ? 0xb5U : ((9U == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index)
                                        ? 0xadU : (
                                                   (0xaU 
                                                    == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index)
                                                    ? 0xa5U
                                                    : 
                                                   ((0xbU 
                                                     == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index)
                                                     ? 0x9eU
                                                     : 
                                                    ((0xcU 
                                                      == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index)
                                                      ? 0x98U
                                                      : 
                                                     ((0xdU 
                                                       == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index)
                                                       ? 0x91U
                                                       : 
                                                      ((0xeU 
                                                        == __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_index)
                                                        ? 0x8bU
                                                        : 0x85U))))))));
                if (VL_LTS_III(32, 8U, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__integer_shift)) {
                    __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__integer_shift = 8U;
                }
                __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__Vfuncout 
                    = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__fraction_value, __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__integer_shift);
            }
            __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__exp_value 
                = __Vfunc_tb_h67_score_class_row_engine__DOT__exp2_q8__64__Vfuncout;
            __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__scaled 
                = VL_MULS_III(32, (IData)(0x80U), VL_MULS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__exp_value, __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__n_tokens));
            __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__shift_value 
                = tb_h67_score_class_row_engine__DOT__denominator_shift;
            __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__value 
                = __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__scaled;
            if ((0U == __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__shift_value)) {
                __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__Vfuncout 
                    = __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__value;
            } else {
                __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__quotient 
                    = VL_SHIFTR_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__value, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__shift_value);
                __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__remainder 
                    = (__Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__value 
                       - VL_SHIFTL_III(32,32,32, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__quotient, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__shift_value));
                __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__half 
                    = VL_SHIFTL_III(32,32,32, (IData)(1U), 
                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__shift_value 
                                     - (IData)(1U)));
                if ((VL_GTS_III(32, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__remainder, __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__half) 
                     | ((__Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__remainder 
                         == __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__half) 
                        & __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__quotient))) {
                    __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__quotient 
                        = ((IData)(1U) + __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__quotient);
                }
                __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__Vfuncout 
                    = __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__quotient;
            }
            vlSelf->tb_h67_score_class_row_engine__DOT__expected_gate[(7U 
                                                                       & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__idx)] 
                = __Vfunc_tb_h67_score_class_row_engine__DOT__round_shift_even__65__Vfuncout;
            if (VL_LTS_III(32, 0x100U, vlSelf->tb_h67_score_class_row_engine__DOT__expected_gate
                           [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__idx)])) {
                vlSelf->tb_h67_score_class_row_engine__DOT__expected_gate[(7U 
                                                                           & __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__idx)] = 0x100U;
            }
            __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__idx 
                = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__build_expected__56__idx);
        }
        vlSelf->tb_h67_score_class_row_engine__DOT__cfg_n_tokens 
            = (0xfU & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__n_tokens);
        vlSelf->tb_h67_score_class_row_engine__DOT__cfg_enable_score_fold 
            = __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__enable_fold;
        co_await vlSelf->__VtrigSched_h9aed8d87__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(negedge tb_h67_score_class_row_engine.clk)", 
                                                           "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                           257);
        vlSelf->tb_h67_score_class_row_engine__DOT__cfg_start = 1U;
        co_await vlSelf->__VtrigSched_h9aed8dba__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_h67_score_class_row_engine.clk)", 
                                                           "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                           259);
        co_await vlSelf->__VtrigSched_h9aed8d87__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(negedge tb_h67_score_class_row_engine.clk)", 
                                                           "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                           260);
        vlSelf->tb_h67_score_class_row_engine__DOT__cfg_start = 0U;
        __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__idx = 0U;
        while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__idx, __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__n_tokens)) {
            while ((2U != (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))) {
                co_await vlSelf->__VtrigSched_h9aed8d87__0.trigger(0U, 
                                                                   nullptr, 
                                                                   "@(negedge tb_h67_score_class_row_engine.clk)", 
                                                                   "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                                   264);
            }
            vlSelf->tb_h67_score_class_row_engine__DOT__in_valid = 1U;
            vlSelf->tb_h67_score_class_row_engine__DOT__in_q_bits 
                = vlSelf->tb_h67_score_class_row_engine__DOT__q_vector
                [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__idx)];
            if (vlSelf->tb_h67_score_class_row_engine__DOT__time_vector
                [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__idx)]) {
                vlSelf->tb_h67_score_class_row_engine__DOT__in_time_sel = 1U;
                vlSelf->tb_h67_score_class_row_engine__DOT__in_k_pair_bits 
                    = (((QData)((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector
                                        [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__idx)])) 
                        << 0x20U) | (QData)((IData)(
                                                    vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector
                                                    [
                                                    (7U 
                                                     & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__idx)])));
            } else {
                vlSelf->tb_h67_score_class_row_engine__DOT__in_time_sel = 0U;
                vlSelf->tb_h67_score_class_row_engine__DOT__in_k_pair_bits 
                    = (((QData)((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__k_peer_vector
                                        [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__idx)])) 
                        << 0x20U) | (QData)((IData)(
                                                    vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector
                                                    [
                                                    (7U 
                                                     & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__idx)])));
            }
            vlSelf->tb_h67_score_class_row_engine__DOT__in_last 
                = (__Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__idx 
                   == (__Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__n_tokens 
                       - (IData)(1U)));
            co_await vlSelf->__VtrigSched_h9aed8dba__0.trigger(0U, 
                                                               nullptr, 
                                                               "@(posedge tb_h67_score_class_row_engine.clk)", 
                                                               "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                               274);
            co_await vlSelf->__VtrigSched_h9aed8d87__0.trigger(0U, 
                                                               nullptr, 
                                                               "@(negedge tb_h67_score_class_row_engine.clk)", 
                                                               "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                               275);
            vlSelf->tb_h67_score_class_row_engine__DOT__in_valid = 0U;
            vlSelf->tb_h67_score_class_row_engine__DOT__in_last = 0U;
            __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__idx 
                = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__idx);
        }
        tb_h67_score_class_row_engine__DOT__output_count = 0U;
        tb_h67_score_class_row_engine__DOT__expected_outputs = 0U;
        __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__idx = 0U;
        while (VL_LTS_III(32, __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__idx, __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__n_tokens)) {
            if ((1U & ((~ (IData)(__Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__enable_fold)) 
                       | (0U != vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector
                          [(7U & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__idx)])))) {
                tb_h67_score_class_row_engine__DOT__expected_outputs 
                    = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__expected_outputs);
            }
            __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__idx 
                = ((IData)(1U) + __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__idx);
        }
        tb_h67_score_class_row_engine__DOT__expected_folded 
            = ((IData)(__Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__enable_fold)
                ? (__Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__n_tokens 
                   - tb_h67_score_class_row_engine__DOT__expected_outputs)
                : 0U);
        while ((7U != (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))) {
            co_await vlSelf->__VtrigSched_h9aed8d87__0.trigger(0U, 
                                                               nullptr, 
                                                               "@(negedge tb_h67_score_class_row_engine.clk)", 
                                                               "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                               288);
            vlSelf->tb_h67_score_class_row_engine__DOT__out_ready 
                = (1U & ((~ tb_h67_score_class_row_engine__DOT__output_count) 
                         | (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__out_ready))));
            if (((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__out_valid) 
                 & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__out_ready))) {
                vlSelf->tb_h67_score_class_row_engine__DOT__token_idx 
                    = (0xfU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w));
                if (VL_UNLIKELY(((IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w 
                                          >> 4U)) != 
                                 vlSelf->tb_h67_score_class_row_engine__DOT__k_current_vector
                                 [(7U & vlSelf->tb_h67_score_class_row_engine__DOT__token_idx)]))) {
                    VL_WRITEF("ERROR token %0d K mismatch\n",
                              32,vlSelf->tb_h67_score_class_row_engine__DOT__token_idx);
                    tb_h67_score_class_row_engine__DOT__errors 
                        = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
                }
                if (VL_UNLIKELY(((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__gate_w) 
                                 != (0x1ffU & vlSelf->tb_h67_score_class_row_engine__DOT__expected_gate
                                     [(7U & vlSelf->tb_h67_score_class_row_engine__DOT__token_idx)])))) {
                    VL_WRITEF("ERROR token %0d gate got=%0# expected=%0d score=%0d\n",
                              32,vlSelf->tb_h67_score_class_row_engine__DOT__token_idx,
                              9,(IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__gate_w),
                              32,vlSelf->tb_h67_score_class_row_engine__DOT__expected_gate
                              [(7U & vlSelf->tb_h67_score_class_row_engine__DOT__token_idx)],
                              32,vlSelf->tb_h67_score_class_row_engine__DOT__expected_score
                              [(7U & vlSelf->tb_h67_score_class_row_engine__DOT__token_idx)]);
                    tb_h67_score_class_row_engine__DOT__errors 
                        = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
                }
                tb_h67_score_class_row_engine__DOT__output_count 
                    = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__output_count);
            }
        }
        vlSelf->tb_h67_score_class_row_engine__DOT__out_ready = 1U;
        if (VL_UNLIKELY((tb_h67_score_class_row_engine__DOT__output_count 
                         != tb_h67_score_class_row_engine__DOT__expected_outputs))) {
            VL_WRITEF("ERROR output count got=%0d expected=%0d\n",
                      32,tb_h67_score_class_row_engine__DOT__output_count,
                      32,tb_h67_score_class_row_engine__DOT__expected_outputs);
            tb_h67_score_class_row_engine__DOT__errors 
                = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
        }
        if (VL_UNLIKELY(((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__tokens_loaded_q) 
                         != (0xfU & __Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__n_tokens)))) {
            VL_WRITEF("ERROR loaded count got=%0# expected=%0d\n",
                      4,vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__tokens_loaded_q,
                      32,__Vtask_tb_h67_score_class_row_engine__DOT__run_row__55__n_tokens);
            tb_h67_score_class_row_engine__DOT__errors 
                = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
        }
        if (VL_UNLIKELY(((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__folded_tokens_q) 
                         != (0xfU & tb_h67_score_class_row_engine__DOT__expected_folded)))) {
            VL_WRITEF("ERROR folded count got=%0# expected=%0d\n",
                      4,vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__folded_tokens_q,
                      32,tb_h67_score_class_row_engine__DOT__expected_folded);
            tb_h67_score_class_row_engine__DOT__errors 
                = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
        }
        if (VL_UNLIKELY(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_range_error_q)) {
            VL_WRITEF("ERROR unexpected score range error\n");
            tb_h67_score_class_row_engine__DOT__errors 
                = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
        }
        co_await vlSelf->__VtrigSched_h9aed8dba__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_h67_score_class_row_engine.clk)", 
                                                           "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                           320);
        if (VL_UNLIKELY(((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__fold_classes_q) 
                         != (0x3fU & tb_h67_score_class_row_engine__DOT__unnamedblk1__DOT__all_class_count)))) {
            VL_WRITEF("ERROR all-class row got=%0# expected=%0d\n",
                      6,vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__fold_classes_q,
                      32,tb_h67_score_class_row_engine__DOT__unnamedblk1__DOT__all_class_count);
            tb_h67_score_class_row_engine__DOT__errors 
                = ((IData)(1U) + tb_h67_score_class_row_engine__DOT__errors);
        }
    }
    if (VL_LIKELY((0U == tb_h67_score_class_row_engine__DOT__errors))) {
        VL_WRITEF("PASS: score-class row engine preserves denominator and gates, motion=1\n");
    } else {
        VL_WRITEF("[%0t] %%Fatal: tb_h67_score_class_row_engine.sv:446: Assertion failed in %Ntb_h67_score_class_row_engine.unnamedblk1: FAIL: H67 row engine errors=%0d\n",
                  64,VL_TIME_UNITED_Q(1000),-9,vlSymsp->name(),
                  32,tb_h67_score_class_row_engine__DOT__errors);
        VL_STOP_MT("tb_h67/tb_h67_score_class_row_engine.sv", 446, "");
    }
    VL_FINISH_MT("tb_h67/tb_h67_score_class_row_engine.sv", 448, "");
}

VL_INLINE_OPT VlCoroutine Vtb_h67_score_class_row_engine___024root___eval_initial__TOP__Vtiming__1(Vtb_h67_score_class_row_engine___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_h67_score_class_row_engine__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_h67_score_class_row_engine___024root___eval_initial__TOP__Vtiming__1\n"); );
    // Init
    IData/*31:0*/ tb_h67_score_class_row_engine__DOT____Vrepeat1;
    tb_h67_score_class_row_engine__DOT____Vrepeat1 = 0;
    // Body
    tb_h67_score_class_row_engine__DOT____Vrepeat1 = 0x30d40U;
    while (VL_LTS_III(32, 0U, tb_h67_score_class_row_engine__DOT____Vrepeat1)) {
        co_await vlSelf->__VtrigSched_h9aed8dba__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_h67_score_class_row_engine.clk)", 
                                                           "tb_h67/tb_h67_score_class_row_engine.sv", 
                                                           452);
        tb_h67_score_class_row_engine__DOT____Vrepeat1 
            = (tb_h67_score_class_row_engine__DOT____Vrepeat1 
               - (IData)(1U));
    }
    VL_WRITEF("[%0t] %%Fatal: tb_h67_score_class_row_engine.sv:453: Assertion failed in %Ntb_h67_score_class_row_engine: FAIL: row engine watchdog timeout\n",
              64,VL_TIME_UNITED_Q(1000),-9,vlSymsp->name());
    VL_STOP_MT("tb_h67/tb_h67_score_class_row_engine.sv", 453, "");
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_h67_score_class_row_engine___024root___dump_triggers__act(Vtb_h67_score_class_row_engine___024root* vlSelf);
#endif  // VL_DEBUG

void Vtb_h67_score_class_row_engine___024root___eval_triggers__act(Vtb_h67_score_class_row_engine___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_h67_score_class_row_engine__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_h67_score_class_row_engine___024root___eval_triggers__act\n"); );
    // Body
    vlSelf->__VactTriggered.set(0U, ((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__clk) 
                                     & (~ (IData)(vlSelf->__Vtrigprevexpr___TOP__tb_h67_score_class_row_engine__DOT__clk__0))));
    vlSelf->__VactTriggered.set(1U, ((~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__clk)) 
                                     & (IData)(vlSelf->__Vtrigprevexpr___TOP__tb_h67_score_class_row_engine__DOT__clk__0)));
    vlSelf->__VactTriggered.set(2U, vlSelf->__VdlySched.awaitingCurrentTime());
    vlSelf->__Vtrigprevexpr___TOP__tb_h67_score_class_row_engine__DOT__clk__0 
        = vlSelf->tb_h67_score_class_row_engine__DOT__clk;
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtb_h67_score_class_row_engine___024root___dump_triggers__act(vlSelf);
    }
#endif
}

VL_INLINE_OPT void Vtb_h67_score_class_row_engine___024root___nba_sequent__TOP__0(Vtb_h67_score_class_row_engine___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_h67_score_class_row_engine__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_h67_score_class_row_engine___024root___nba_sequent__TOP__0\n"); );
    // Init
    SData/*15:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__active_delta_w;
    tb_h67_score_class_row_engine__DOT__dut__DOT__active_delta_w = 0;
    SData/*15:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__class_delta_w;
    tb_h67_score_class_row_engine__DOT__dut__DOT__class_delta_w = 0;
    CData/*2:0*/ tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_addr_w;
    tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_addr_w = 0;
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
    IData/*31:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout;
    __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout = 0;
    QData/*34:0*/ __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value;
    __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value = 0;
    CData/*2:0*/ __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__state_q;
    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__state_q = 0;
    SData/*15:0*/ __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__exp_transactions_q;
    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__exp_transactions_q = 0;
    CData/*3:0*/ __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__scan_idx_q;
    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__scan_idx_q = 0;
    CData/*5:0*/ __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__classes_remaining_q;
    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__classes_remaining_q = 0;
    IData/*31:0*/ __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__row_sum_q;
    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__row_sum_q = 0;
    CData/*5:0*/ __Vdlyvdim0__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v0;
    __Vdlyvdim0__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v0 = 0;
    CData/*3:0*/ __Vdlyvval__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v0;
    __Vdlyvval__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v0 = 0;
    CData/*0:0*/ __Vdlyvset__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v0;
    __Vdlyvset__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v0 = 0;
    QData/*34:0*/ __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q;
    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q = 0;
    CData/*3:0*/ __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__tokens_loaded_q;
    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__tokens_loaded_q = 0;
    CData/*5:0*/ __Vdlyvdim0__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v1;
    __Vdlyvdim0__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v1 = 0;
    CData/*3:0*/ __Vdlyvval__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v1;
    __Vdlyvval__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v1 = 0;
    CData/*0:0*/ __Vdlyvset__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v1;
    __Vdlyvset__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v1 = 0;
    CData/*2:0*/ __Vdlyvdim0__tb_h67_score_class_row_engine__DOT__dut__DOT__active_entry_mem__v0;
    __Vdlyvdim0__tb_h67_score_class_row_engine__DOT__dut__DOT__active_entry_mem__v0 = 0;
    QData/*51:0*/ __Vdlyvval__tb_h67_score_class_row_engine__DOT__dut__DOT__active_entry_mem__v0;
    __Vdlyvval__tb_h67_score_class_row_engine__DOT__dut__DOT__active_entry_mem__v0 = 0;
    CData/*0:0*/ __Vdlyvset__tb_h67_score_class_row_engine__DOT__dut__DOT__active_entry_mem__v0;
    __Vdlyvset__tb_h67_score_class_row_engine__DOT__dut__DOT__active_entry_mem__v0 = 0;
    CData/*3:0*/ __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__load_idx_q;
    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__load_idx_q = 0;
    CData/*5:0*/ __Vdlyvdim0__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v2;
    __Vdlyvdim0__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v2 = 0;
    CData/*3:0*/ __Vdlyvval__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v2;
    __Vdlyvval__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v2 = 0;
    CData/*0:0*/ __Vdlyvset__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v2;
    __Vdlyvset__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v2 = 0;
    CData/*0:0*/ __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__hist_initialized_q;
    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__hist_initialized_q = 0;
    // Body
    if (vlSymsp->_vm_contextp__->assertOn()) {
        if (VL_UNLIKELY((1U & (~ ((~ (IData)(vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__rst_n)) 
                                  | ((~ (IData)(vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__out_last)) 
                                     | (IData)(vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__out_valid))))))) {
            VL_WRITEF("[%0t] %%Error: h67_h68_protocol_assertions.sv:67: Assertion failed in %Ntb_h67_score_class_row_engine.dut.u_protocol_assertions.a_last_requires_valid: 'assert' failed.\n",
                      64,VL_TIME_UNITED_Q(1000),-9,
                      vlSymsp->name());
            VL_STOP_MT("verif_h67_h68/h67_h68_protocol_assertions.sv", 67, "");
        }
    }
    if (vlSymsp->_vm_contextp__->assertOn()) {
        if (VL_UNLIKELY((1U & (~ (1U | (~ (IData)(vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__rst_n))))))) {
            VL_WRITEF("[%0t] %%Error: h67_h68_protocol_assertions.sv:66: Assertion failed in %Ntb_h67_score_class_row_engine.dut.u_protocol_assertions.a_done_implies_busy: 'assert' failed.\n",
                      64,VL_TIME_UNITED_Q(1000),-9,
                      vlSymsp->name());
            VL_STOP_MT("verif_h67_h68/h67_h68_protocol_assertions.sv", 66, "");
        }
    }
    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__hist_initialized_q 
        = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_initialized_q;
    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__load_idx_q 
        = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__load_idx_q;
    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__tokens_loaded_q 
        = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__tokens_loaded_q;
    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__classes_remaining_q 
        = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__classes_remaining_q;
    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__exp_transactions_q 
        = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__exp_transactions_q;
    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
        = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q;
    if (vlSymsp->_vm_contextp__->assertOn()) {
        if (VL_UNLIKELY((1U & (~ ((~ (IData)(vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__rst_n)) 
                                  | (~ (IData)(vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__score_range_error_q))))))) {
            VL_WRITEF("[%0t] %%Error: h67_h68_protocol_assertions.sv:68: Assertion failed in %Ntb_h67_score_class_row_engine.dut.u_protocol_assertions.a_no_frozen_score_overflow: 'assert' failed.\n",
                      64,VL_TIME_UNITED_Q(1000),-9,
                      vlSymsp->name());
            VL_STOP_MT("verif_h67_h68/h67_h68_protocol_assertions.sv", 68, "");
        }
    }
    __Vdlyvset__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v0 = 0U;
    __Vdlyvset__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v1 = 0U;
    __Vdlyvset__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v2 = 0U;
    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__row_sum_q 
        = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__row_sum_q;
    __Vdlyvset__tb_h67_score_class_row_engine__DOT__dut__DOT__active_entry_mem__v0 = 0U;
    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__scan_idx_q 
        = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__scan_idx_q;
    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__state_q 
        = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q;
    if (vlSymsp->_vm_contextp__->assertOn()) {
        if (VL_UNLIKELY((1U & (~ ((~ (IData)(vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__rst_n)) 
                                  | ((~ (IData)(vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT___Vpast_0_0)) 
                                     | ((IData)(vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__out_valid) 
                                        & (vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT___Vpast_1_0 
                                           == (((QData)((IData)(vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__out_last)) 
                                                << 0x35U) 
                                               | (((QData)((IData)(
                                                                   (0xfU 
                                                                    & (IData)(vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w)))) 
                                                   << 0x31U) 
                                                  | (((QData)((IData)(
                                                                      (vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w 
                                                                       >> 4U))) 
                                                      << 0x11U) 
                                                     | (QData)((IData)(
                                                                       (((IData)(vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__gate_w) 
                                                                         << 8U) 
                                                                        | (IData)(vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__threshold_q8_q))))))))))))))) {
            VL_WRITEF("[%0t] %%Error: h67_h68_protocol_assertions.sv:65: Assertion failed in %Ntb_h67_score_class_row_engine.dut.u_protocol_assertions.a_output_stable_under_backpressure: 'assert' failed.\n",
                      64,VL_TIME_UNITED_Q(1000),-9,
                      vlSymsp->name());
            VL_STOP_MT("verif_h67_h68/h67_h68_protocol_assertions.sv", 65, "");
        }
    }
    if (vlSymsp->_vm_contextp__->assertOn()) {
        if (VL_UNLIKELY((1U & (~ ((~ (IData)(vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__rst_n)) 
                                  | ((IData)(vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__classes_remaining_q) 
                                     == (0x3fU & (([&]() {
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                    = vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q;
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (1U 
                                                     & (IData)(__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 1U))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 2U))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 3U))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 4U))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 5U))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 6U))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 7U))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 8U))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 9U))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 0xaU))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 0xbU))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 0xcU))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 0xdU))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 0xeU))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 0xfU))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 0x10U))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 0x11U))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 0x12U))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 0x13U))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 0x14U))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 0x15U))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 0x16U))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 0x17U))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 0x18U))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 0x19U))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 0x1aU))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 0x1bU))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 0x1cU))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 0x1dU))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 0x1eU))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 0x1fU))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 0x20U))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 0x21U))));
                                                __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                    = 
                                                    (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout 
                                                     + 
                                                     (1U 
                                                      & (IData)(
                                                                (__Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__value 
                                                                 >> 0x22U))));
                                            }(), __Vfunc_tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT__popcount_classes__66__Vfuncout) 
                                                  + 
                                                  (5U 
                                                   == (IData)(vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)))))))))) {
            VL_WRITEF("[%0t] %%Error: h67_h68_protocol_assertions.sv:69: Assertion failed in %Ntb_h67_score_class_row_engine.dut.u_protocol_assertions.a_class_count_matches_bitmap: 'assert' failed.\n",
                      64,VL_TIME_UNITED_Q(1000),-9,
                      vlSymsp->name());
            VL_STOP_MT("verif_h67_h68/h67_h68_protocol_assertions.sv", 69, "");
        }
    }
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT___Vpast_0_0 
        = ((IData)(vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__rst_n) 
           & ((IData)(vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__out_valid) 
              & (~ (IData)(vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__out_ready))));
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__u_protocol_assertions__DOT___Vpast_1_0 
        = (((QData)((IData)(vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__out_last)) 
            << 0x35U) | (((QData)((IData)((0xfU & (IData)(vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w)))) 
                          << 0x31U) | (((QData)((IData)(
                                                        (vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w 
                                                         >> 4U))) 
                                        << 0x11U) | (QData)((IData)(
                                                                    (((IData)(vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__gate_w) 
                                                                      << 8U) 
                                                                     | (IData)(vlSelf->__Vsampled__TOP__tb_h67_score_class_row_engine__DOT__dut__DOT__threshold_q8_q)))))));
    if (vlSelf->tb_h67_score_class_row_engine__DOT__rst_n) {
        if ((4U & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))) {
            if ((2U & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))) {
                if ((1U & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))) {
                    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__state_q = 0U;
                } else if (((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__out_valid) 
                            & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__out_ready))) {
                    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__emitted_entries_q 
                        = (0xfU & ((IData)(1U) + (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__emitted_entries_q)));
                    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__exp_transactions_q 
                        = (0xffffU & ((IData)(1U) + (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__exp_transactions_q)));
                    if (vlSelf->tb_h67_score_class_row_engine__DOT__out_last) {
                        __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__state_q = 7U;
                    } else {
                        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__emit_idx_q 
                            = (0xfU & ((IData)(1U) 
                                       + (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__emit_idx_q)));
                        __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__scan_idx_q 
                            = (0xfU & ((IData)(1U) 
                                       + (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__scan_idx_q)));
                    }
                }
            } else if ((1U & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))) {
                __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__classes_remaining_q 
                    = (0x3fU & ((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__classes_remaining_q) 
                                - (IData)(1U)));
                __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__row_sum_q 
                    = (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__row_sum_q 
                       + vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_sum_term_w);
                vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__fold_classes_q 
                    = (0x3fU & ((IData)(1U) + (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__fold_classes_q)));
                __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__exp_transactions_q 
                    = (0xffffU & ((IData)(1U) + (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__exp_transactions_q)));
                if ((1U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__classes_remaining_q))) {
                    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__emit_idx_q = 0U;
                    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__scan_idx_q = 0U;
                    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__state_q 
                        = ((0U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_count_q))
                            ? 7U : 6U);
                } else {
                    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__state_q = 4U;
                }
            } else if ((1U & ((~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w)) 
                              | (0U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__classes_remaining_q))))) {
                vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__emit_idx_q = 0U;
                __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__scan_idx_q = 0U;
                vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_range_error_q = 1U;
                __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__state_q 
                    = ((0U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_count_q))
                        ? 7U : 6U);
            } else {
                vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT____Vlvbound_h0c4876d2__0 = 0U;
                vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT____Vlvbound_h8b805d45__0 = 0U;
                if ((0x22U >= (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w))) {
                    __Vdlyvval__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v0 
                        = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT____Vlvbound_h0c4876d2__0;
                    __Vdlyvset__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v0 = 1U;
                    __Vdlyvdim0__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v0 
                        = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w;
                    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                        = (((~ (1ULL << (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w))) 
                            & __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q) 
                           | (0x7ffffffffULL & ((QData)((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT____Vlvbound_h8b805d45__0)) 
                                                << (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w))));
                }
                vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_q 
                    = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w;
                vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_hist_count_q 
                    = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w;
                __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__state_q = 5U;
            }
        } else if ((2U & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))) {
            if ((1U & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))) {
                if ((0U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_count_q))) {
                    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__state_q 
                        = ((0U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__folded_tokens_q))
                            ? 7U : 4U);
                } else {
                    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__row_sum_q 
                        = (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__row_sum_q 
                           + (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_exp_w));
                    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__exp_transactions_q 
                        = (0xffffU & ((IData)(1U) + (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__exp_transactions_q)));
                    if (((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__scan_idx_q) 
                         == (0xfU & ((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_count_q) 
                                     - (IData)(1U))))) {
                        __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__scan_idx_q = 0U;
                        __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__state_q 
                            = ((0U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__folded_tokens_q))
                                ? 6U : 4U);
                    } else {
                        __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__scan_idx_q 
                            = (0xfU & ((IData)(1U) 
                                       + (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__scan_idx_q)));
                    }
                }
            } else if (((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__in_valid) 
                        & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__in_ready))) {
                __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__tokens_loaded_q 
                    = (0xfU & ((IData)(1U) + (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__tokens_loaded_q)));
                if (((0U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__tokens_loaded_q)) 
                     | VL_GTS_III(16, (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w), (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__row_max_q)))) {
                    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__row_max_q 
                        = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w;
                }
                if ((((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__enable_score_fold_q) 
                      & (~ (IData)((0U != vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w)))) 
                     & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_in_range_w)))) {
                    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_range_error_q = 1U;
                }
                if (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__fold_input_w) {
                    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT____Vlvbound_h6b3d7c2f__0 
                        = (0xfU & ((IData)(1U) + (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_input_count_w)));
                    if ((1U & (~ ((0x22U >= (0x3fU 
                                             & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))) 
                                  && (1U & (IData)(
                                                   (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                                                    >> 
                                                    (0x3fU 
                                                     & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w))))))))) {
                        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT____Vlvbound_h60725e1c__0 = 1U;
                        __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__classes_remaining_q 
                            = (0x3fU & ((IData)(1U) 
                                        + (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__classes_remaining_q)));
                        if ((0x22U >= (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w)))) {
                            __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                                = (((~ (1ULL << (0x3fU 
                                                 & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w)))) 
                                    & __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q) 
                                   | (0x7ffffffffULL 
                                      & ((QData)((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT____Vlvbound_h60725e1c__0)) 
                                         << (0x3fU 
                                             & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w)))));
                        }
                    }
                    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__folded_tokens_q 
                        = (0xfU & ((IData)(1U) + (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__folded_tokens_q)));
                    if ((0x22U >= (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w)))) {
                        __Vdlyvval__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v1 
                            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT____Vlvbound_h6b3d7c2f__0;
                        __Vdlyvset__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v1 = 1U;
                        __Vdlyvdim0__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v1 
                            = (0x3fU & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w));
                    }
                } else {
                    __Vdlyvval__tb_h67_score_class_row_engine__DOT__dut__DOT__active_entry_mem__v0 
                        = (((QData)((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__input_score_w)) 
                            << 0x24U) | (((QData)((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__current_k_w)) 
                                          << 4U) | (QData)((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__load_idx_q))));
                    __Vdlyvset__tb_h67_score_class_row_engine__DOT__dut__DOT__active_entry_mem__v0 = 1U;
                    __Vdlyvdim0__tb_h67_score_class_row_engine__DOT__dut__DOT__active_entry_mem__v0 
                        = (7U & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_count_q));
                    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_count_q 
                        = (0xfU & ((IData)(1U) + (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_count_q)));
                }
                __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__load_idx_q 
                    = (0xfU & ((IData)(1U) + (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__load_idx_q)));
                if (((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__in_last) 
                     | ((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__load_idx_q) 
                        == (0xfU & ((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__n_tokens_q) 
                                    - (IData)(1U)))))) {
                    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__scan_idx_q = 0U;
                    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__row_sum_q = 0U;
                    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__n_tokens_q 
                        = (0xfU & ((IData)(1U) + (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__load_idx_q)));
                    __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__state_q = 3U;
                }
            }
        } else if ((1U & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))) {
            vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT____Vlvbound_h68cafb07__0 = 0U;
            if ((0x22U >= (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__clear_all_idx_q))) {
                __Vdlyvval__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v2 
                    = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT____Vlvbound_h68cafb07__0;
                __Vdlyvset__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v2 = 1U;
                __Vdlyvdim0__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v2 
                    = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__clear_all_idx_q;
            }
            if ((0x22U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__clear_all_idx_q))) {
                __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__hist_initialized_q = 1U;
                __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__state_q = 2U;
            } else {
                vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__clear_all_idx_q 
                    = (0x3fU & ((IData)(1U) + (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__clear_all_idx_q)));
            }
        } else if (vlSelf->tb_h67_score_class_row_engine__DOT__cfg_start) {
            vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_count_q = 0U;
            __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__scan_idx_q = 0U;
            vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__emit_idx_q = 0U;
            __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__row_sum_q = 0U;
            vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__folded_tokens_q = 0U;
            vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__emitted_entries_q = 0U;
            vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__fold_classes_q = 0U;
            __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__exp_transactions_q = 0U;
            if (vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_initialized_q) {
                __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__state_q = 2U;
            } else {
                vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__clear_all_idx_q = 0U;
                __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__state_q = 1U;
            }
            vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__n_tokens_q 
                = (((0U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__cfg_n_tokens)) 
                    | (8U < (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__cfg_n_tokens)))
                    ? 8U : (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__cfg_n_tokens));
            vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__preserve_mean_q 
                = vlSelf->tb_h67_score_class_row_engine__DOT__cfg_preserve_mean;
            vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__enable_score_fold_q 
                = vlSelf->tb_h67_score_class_row_engine__DOT__cfg_enable_score_fold;
            vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__threshold_q8_q 
                = vlSelf->tb_h67_score_class_row_engine__DOT__cfg_threshold_q8;
            __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__load_idx_q = 0U;
            __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q = 0ULL;
            __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__classes_remaining_q = 0U;
            vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_q = 0U;
            vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_hist_count_q = 0U;
            vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__row_max_q = 0x8001U;
            __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__tokens_loaded_q = 0U;
            vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_range_error_q = 0U;
        }
    } else {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_count_q = 0U;
        __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__scan_idx_q = 0U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__emit_idx_q = 0U;
        __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__row_sum_q = 0U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__folded_tokens_q = 0U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__emitted_entries_q = 0U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__fold_classes_q = 0U;
        __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__exp_transactions_q = 0U;
        __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__state_q = 0U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__n_tokens_q = 0U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__preserve_mean_q = 1U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__enable_score_fold_q = 1U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__threshold_q8_q = 0U;
        __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__load_idx_q = 0U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__clear_all_idx_q = 0U;
        __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__hist_initialized_q = 0U;
        __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q = 0ULL;
        __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__classes_remaining_q = 0U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_q = 0U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_hist_count_q = 0U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__row_max_q = 0U;
        __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__tokens_loaded_q = 0U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_range_error_q = 0U;
    }
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__exp_transactions_q 
        = __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__exp_transactions_q;
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__classes_remaining_q 
        = __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__classes_remaining_q;
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__tokens_loaded_q 
        = __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__tokens_loaded_q;
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__load_idx_q 
        = __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__load_idx_q;
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_initialized_q 
        = __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__hist_initialized_q;
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
        = __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q;
    if (__Vdlyvset__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v0) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist[__Vdlyvdim0__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v0] 
            = __Vdlyvval__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v0;
    }
    if (__Vdlyvset__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v1) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist[__Vdlyvdim0__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v1] 
            = __Vdlyvval__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v1;
    }
    if (__Vdlyvset__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v2) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist[__Vdlyvdim0__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v2] 
            = __Vdlyvval__tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist__v2;
    }
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__row_sum_q 
        = __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__row_sum_q;
    if (__Vdlyvset__tb_h67_score_class_row_engine__DOT__dut__DOT__active_entry_mem__v0) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_entry_mem[__Vdlyvdim0__tb_h67_score_class_row_engine__DOT__dut__DOT__active_entry_mem__v0] 
            = __Vdlyvval__tb_h67_score_class_row_engine__DOT__dut__DOT__active_entry_mem__v0;
    }
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__scan_idx_q 
        = __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__scan_idx_q;
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q 
        = __Vdly__tb_h67_score_class_row_engine__DOT__dut__DOT__state_q;
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
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0U;
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 0U;
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 1U)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 1U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 2U)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 2U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 3U)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 3U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 4U)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 4U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 5U)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 5U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 6U)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 6U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 7U)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 7U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 8U)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 8U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 9U)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 9U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 0xaU)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0xaU;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 0xbU)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0xbU;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 0xcU)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0xcU;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 0xdU)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0xdU;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 0xeU)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0xeU;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 0xfU)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0xfU;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 0x10U)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0x10U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 0x11U)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0x11U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 0x12U)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0x12U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 0x13U)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0x13U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 0x14U)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0x14U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 0x15U)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0x15U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 0x16U)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0x16U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 0x17U)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0x17U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 0x18U)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0x18U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 0x19U)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0x19U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 0x1aU)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0x1aU;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 0x1bU)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0x1bU;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 0x1cU)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0x1cU;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 0x1dU)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0x1dU;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 0x1eU)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0x1eU;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 0x1fU)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0x1fU;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 0x20U)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0x20U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 0x21U)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0x21U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    if ((((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
          & (~ (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w))) 
         & (IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_present_q 
                    >> 0x22U)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w = 0x22U;
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_found_w = 1U;
    }
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w = 0U;
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0U];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (1U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [1U];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (2U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [2U];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (3U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [3U];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [4U];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (5U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [5U];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (6U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [6U];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (7U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [7U];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (8U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [8U];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (9U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [9U];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0xaU == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0xaU];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0xbU == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0xbU];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0xcU == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0xcU];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0xdU == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0xdU];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0xeU == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0xeU];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0xfU == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0xfU];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x10U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x10U];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x11U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x11U];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x12U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x12U];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x13U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x13U];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x14U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x14U];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x15U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x15U];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x16U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x16U];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x17U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x17U];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x18U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x18U];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x19U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x19U];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x1aU == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x1aU];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x1bU == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x1bU];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x1cU == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x1cU];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x1dU == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x1dU];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x1eU == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x1eU];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x1fU == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x1fU];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x20U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x20U];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x21U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x21U];
    }
    if (((4U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
         & (0x22U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_w)))) {
        vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__hist_scan_count_w 
            = vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__score_hist
            [0x22U];
    }
    tb_h67_score_class_row_engine__DOT__dut__DOT__class_delta_w 
        = ((5U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))
            ? (0xffffU & ((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_score_code_q) 
                          - (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__row_max_q)))
            : 0U);
    tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_addr_w 
        = (7U & ((6U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q))
                  ? (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__emit_idx_q)
                  : (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__scan_idx_q)));
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
    vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_sum_term_w 
        = ((IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_hist_count_q) 
           * (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__class_exp_w));
    tb_h67_score_class_row_engine__DOT__dut__DOT__active_delta_w 
        = (((3U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)) 
            | (6U == (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__state_q)))
            ? (0xffffU & ((IData)((vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__active_read_entry_w 
                                   >> 0x24U)) - (IData)(vlSelf->tb_h67_score_class_row_engine__DOT__dut__DOT__row_max_q)))
            : 0U);
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
