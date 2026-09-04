// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vtb_h67_score_class_row_engine.h for the primary calling header

#ifndef VERILATED_VTB_H67_SCORE_CLASS_ROW_ENGINE___024UNIT_H_
#define VERILATED_VTB_H67_SCORE_CLASS_ROW_ENGINE___024UNIT_H_  // guard

#include "verilated.h"
#include "verilated_timing.h"


class Vtb_h67_score_class_row_engine__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vtb_h67_score_class_row_engine___024unit final : public VerilatedModule {
  public:

    // INTERNAL VARIABLES
    Vtb_h67_score_class_row_engine__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vtb_h67_score_class_row_engine___024unit(Vtb_h67_score_class_row_engine__Syms* symsp, const char* v__name);
    ~Vtb_h67_score_class_row_engine___024unit();
    VL_UNCOPYABLE(Vtb_h67_score_class_row_engine___024unit);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
