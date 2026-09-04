// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table implementation internals

#include "Vtb_h67_score_class_row_engine__pch.h"
#include "Vtb_h67_score_class_row_engine.h"
#include "Vtb_h67_score_class_row_engine___024root.h"
#include "Vtb_h67_score_class_row_engine___024unit.h"

// FUNCTIONS
Vtb_h67_score_class_row_engine__Syms::~Vtb_h67_score_class_row_engine__Syms()
{
}

Vtb_h67_score_class_row_engine__Syms::Vtb_h67_score_class_row_engine__Syms(VerilatedContext* contextp, const char* namep, Vtb_h67_score_class_row_engine* modelp)
    : VerilatedSyms{contextp}
    // Setup internal state of the Syms class
    , __Vm_modelp{modelp}
    // Setup module instances
    , TOP{this, namep}
{
    // Configure time unit / time precision
    _vm_contextp__->timeunit(-9);
    _vm_contextp__->timeprecision(-12);
    // Setup each module's pointers to their submodules
    // Setup each module's pointer back to symbol table (for public functions)
    TOP.__Vconfigure(true);
    // Setup scopes
    __Vscope_tb_h67_score_class_row_engine.configure(this, name(), "tb_h67_score_class_row_engine", "tb_h67_score_class_row_engine", -9, VerilatedScope::SCOPE_OTHER);
    __Vscope_tb_h67_score_class_row_engine__dut__u_protocol_assertions.configure(this, name(), "tb_h67_score_class_row_engine.dut.u_protocol_assertions", "u_protocol_assertions", -9, VerilatedScope::SCOPE_OTHER);
    __Vscope_tb_h67_score_class_row_engine__dut__u_protocol_assertions__a_class_count_matches_bitmap.configure(this, name(), "tb_h67_score_class_row_engine.dut.u_protocol_assertions.a_class_count_matches_bitmap", "a_class_count_matches_bitmap", -9, VerilatedScope::SCOPE_OTHER);
    __Vscope_tb_h67_score_class_row_engine__dut__u_protocol_assertions__a_done_implies_busy.configure(this, name(), "tb_h67_score_class_row_engine.dut.u_protocol_assertions.a_done_implies_busy", "a_done_implies_busy", -9, VerilatedScope::SCOPE_OTHER);
    __Vscope_tb_h67_score_class_row_engine__dut__u_protocol_assertions__a_last_requires_valid.configure(this, name(), "tb_h67_score_class_row_engine.dut.u_protocol_assertions.a_last_requires_valid", "a_last_requires_valid", -9, VerilatedScope::SCOPE_OTHER);
    __Vscope_tb_h67_score_class_row_engine__dut__u_protocol_assertions__a_no_frozen_score_overflow.configure(this, name(), "tb_h67_score_class_row_engine.dut.u_protocol_assertions.a_no_frozen_score_overflow", "a_no_frozen_score_overflow", -9, VerilatedScope::SCOPE_OTHER);
    __Vscope_tb_h67_score_class_row_engine__dut__u_protocol_assertions__a_output_stable_under_backpressure.configure(this, name(), "tb_h67_score_class_row_engine.dut.u_protocol_assertions.a_output_stable_under_backpressure", "a_output_stable_under_backpressure", -9, VerilatedScope::SCOPE_OTHER);
    __Vscope_tb_h67_score_class_row_engine__prepare_all_fold_classes.configure(this, name(), "tb_h67_score_class_row_engine.prepare_all_fold_classes", "prepare_all_fold_classes", -9, VerilatedScope::SCOPE_OTHER);
    __Vscope_tb_h67_score_class_row_engine__unnamedblk1.configure(this, name(), "tb_h67_score_class_row_engine.unnamedblk1", "unnamedblk1", -9, VerilatedScope::SCOPE_OTHER);
}
