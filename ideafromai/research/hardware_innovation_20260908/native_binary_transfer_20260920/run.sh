#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
/opt/anaconda3/bin/python3.12 prepare_cases.py
for spec in 'fc:' 'gp:TOP_GP' 'pr:TOP_PR' 'gp4:TOP_GP4'; do
    name=${spec%%:*}
    macro=${spec#*:}
    flags='-O2 -std=c++17'
    if [ -n "$macro" ]; then flags="$flags -D$macro"; fi
    verilator --cc --exe -Wno-fatal --top-module "native_$name" \
        --Mdir "obj_$name" "rtl/native_$name.sv" tb.cpp -CFLAGS "$flags" \
        > "compile_$name.log" 2>&1
    make -C "obj_$name" -f "Vnative_$name.mk" -j4 >> "compile_$name.log" 2>&1
    "obj_$name/Vnative_$name" cases.txt > "${name}_results.csv" 2> "${name}_run.log"
    cat "${name}_run.log"
done
/opt/anaconda3/bin/python3.12 analyze_results.py
