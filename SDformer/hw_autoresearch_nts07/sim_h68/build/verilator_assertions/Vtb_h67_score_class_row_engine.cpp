// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "Vtb_h67_score_class_row_engine__pch.h"

//============================================================
// Constructors

Vtb_h67_score_class_row_engine::Vtb_h67_score_class_row_engine(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new Vtb_h67_score_class_row_engine__Syms(contextp(), _vcname__, this)}
    , rootp{&(vlSymsp->TOP)}
{
    // Register model with the context
    contextp()->addModel(this);
}

Vtb_h67_score_class_row_engine::Vtb_h67_score_class_row_engine(const char* _vcname__)
    : Vtb_h67_score_class_row_engine(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

Vtb_h67_score_class_row_engine::~Vtb_h67_score_class_row_engine() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void Vtb_h67_score_class_row_engine___024root___eval_debug_assertions(Vtb_h67_score_class_row_engine___024root* vlSelf);
#endif  // VL_DEBUG
void Vtb_h67_score_class_row_engine___024root___eval_static(Vtb_h67_score_class_row_engine___024root* vlSelf);
void Vtb_h67_score_class_row_engine___024root___eval_initial(Vtb_h67_score_class_row_engine___024root* vlSelf);
void Vtb_h67_score_class_row_engine___024root___eval_settle(Vtb_h67_score_class_row_engine___024root* vlSelf);
void Vtb_h67_score_class_row_engine___024root___eval(Vtb_h67_score_class_row_engine___024root* vlSelf);

void Vtb_h67_score_class_row_engine::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vtb_h67_score_class_row_engine::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    Vtb_h67_score_class_row_engine___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    vlSymsp->__Vm_deleter.deleteAll();
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) {
        vlSymsp->__Vm_didInit = true;
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial\n"););
        Vtb_h67_score_class_row_engine___024root___eval_static(&(vlSymsp->TOP));
        Vtb_h67_score_class_row_engine___024root___eval_initial(&(vlSymsp->TOP));
        Vtb_h67_score_class_row_engine___024root___eval_settle(&(vlSymsp->TOP));
    }
    VL_DEBUG_IF(VL_DBG_MSGF("+ Eval\n"););
    Vtb_h67_score_class_row_engine___024root___eval(&(vlSymsp->TOP));
    // Evaluate cleanup
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
}

//============================================================
// Events and timing
bool Vtb_h67_score_class_row_engine::eventsPending() { return !vlSymsp->TOP.__VdlySched.empty(); }

uint64_t Vtb_h67_score_class_row_engine::nextTimeSlot() { return vlSymsp->TOP.__VdlySched.nextTimeSlot(); }

//============================================================
// Utilities

const char* Vtb_h67_score_class_row_engine::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

void Vtb_h67_score_class_row_engine___024root___eval_final(Vtb_h67_score_class_row_engine___024root* vlSelf);

VL_ATTR_COLD void Vtb_h67_score_class_row_engine::final() {
    Vtb_h67_score_class_row_engine___024root___eval_final(&(vlSymsp->TOP));
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* Vtb_h67_score_class_row_engine::hierName() const { return vlSymsp->name(); }
const char* Vtb_h67_score_class_row_engine::modelName() const { return "Vtb_h67_score_class_row_engine"; }
unsigned Vtb_h67_score_class_row_engine::threads() const { return 1; }
void Vtb_h67_score_class_row_engine::prepareClone() const { contextp()->prepareClone(); }
void Vtb_h67_score_class_row_engine::atClone() const {
    contextp()->threadPoolpOnClone();
}

//============================================================
// Trace configuration

VL_ATTR_COLD void Vtb_h67_score_class_row_engine::trace(VerilatedVcdC* tfp, int levels, int options) {
    vl_fatal(__FILE__, __LINE__, __FILE__,"'Vtb_h67_score_class_row_engine::trace()' called on model that was Verilated without --trace option");
}
