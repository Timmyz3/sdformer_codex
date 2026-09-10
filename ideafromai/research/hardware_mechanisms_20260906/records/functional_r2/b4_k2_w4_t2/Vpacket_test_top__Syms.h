// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef _VPACKET_TEST_TOP__SYMS_H_
#define _VPACKET_TEST_TOP__SYMS_H_  // guard

#include "verilated.h"

// INCLUDE MODULE CLASSES
#include "Vpacket_test_top.h"

// SYMS CLASS
class Vpacket_test_top__Syms : public VerilatedSyms {
  public:
    
    // LOCAL STATE
    const char* __Vm_namep;
    bool __Vm_didInit;
    
    // SUBCELL STATE
    Vpacket_test_top*              TOPp;
    
    // CREATORS
    Vpacket_test_top__Syms(Vpacket_test_top* topp, const char* namep);
    ~Vpacket_test_top__Syms() {}
    
    // METHODS
    inline const char* name() { return __Vm_namep; }
    
} VL_ATTR_ALIGNED(VL_CACHE_LINE_BYTES);

#endif  // guard
