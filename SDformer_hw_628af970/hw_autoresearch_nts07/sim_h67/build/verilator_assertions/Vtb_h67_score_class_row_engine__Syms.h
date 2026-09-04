// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef VERILATED_VTB_H67_SCORE_CLASS_ROW_ENGINE__SYMS_H_
#define VERILATED_VTB_H67_SCORE_CLASS_ROW_ENGINE__SYMS_H_  // guard

#include "verilated.h"

// INCLUDE MODEL CLASS

#include "Vtb_h67_score_class_row_engine.h"

// INCLUDE MODULE CLASSES
#include "Vtb_h67_score_class_row_engine___024root.h"
#include "Vtb_h67_score_class_row_engine___024unit.h"

// SYMS CLASS (contains all model state)
class alignas(VL_CACHE_LINE_BYTES)Vtb_h67_score_class_row_engine__Syms final : public VerilatedSyms {
  public:
    // INTERNAL STATE
    Vtb_h67_score_class_row_engine* const __Vm_modelp;
    VlDeleter __Vm_deleter;
    bool __Vm_didInit = false;

    // MODULE INSTANCE STATE
    Vtb_h67_score_class_row_engine___024root TOP;

    // SCOPE NAMES
    VerilatedScope __Vscope_tb_h67_score_class_row_engine;
    VerilatedScope __Vscope_tb_h67_score_class_row_engine__dut__u_protocol_assertions;
    VerilatedScope __Vscope_tb_h67_score_class_row_engine__dut__u_protocol_assertions__a_class_count_matches_bitmap;
    VerilatedScope __Vscope_tb_h67_score_class_row_engine__dut__u_protocol_assertions__a_done_implies_busy;
    VerilatedScope __Vscope_tb_h67_score_class_row_engine__dut__u_protocol_assertions__a_last_requires_valid;
    VerilatedScope __Vscope_tb_h67_score_class_row_engine__dut__u_protocol_assertions__a_no_frozen_score_overflow;
    VerilatedScope __Vscope_tb_h67_score_class_row_engine__dut__u_protocol_assertions__a_output_stable_under_backpressure;
    VerilatedScope __Vscope_tb_h67_score_class_row_engine__prepare_all_fold_classes;
    VerilatedScope __Vscope_tb_h67_score_class_row_engine__unnamedblk1;

    // CONSTRUCTORS
    Vtb_h67_score_class_row_engine__Syms(VerilatedContext* contextp, const char* namep, Vtb_h67_score_class_row_engine* modelp);
    ~Vtb_h67_score_class_row_engine__Syms();

    // METHODS
    const char* name() { return TOP.name(); }
};

#endif  // guard
