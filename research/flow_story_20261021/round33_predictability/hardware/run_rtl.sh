#!/usr/bin/env bash
set -euo pipefail
r33_hw="$(cd "$(dirname "$0")" && pwd)"
for r33_kind in dynamic static none; do
 r33_obj="$r33_hw/results/rtl_$r33_kind"
 mkdir -p "$r33_obj"
 r33_flags="-O2"
 if [ "$r33_kind" = static ]; then r33_flags="-O2 -DSTATIC_ENGINE"; fi
 if [ "$r33_kind" = none ]; then r33_flags="-O2 -DNONE_ENGINE"; fi
 if ! verilator --cc --exe -Wno-fatal --top-module "bitmap_ctrl_$r33_kind" --Mdir "$r33_obj/obj" -CFLAGS "$r33_flags" "$r33_hw/rtl/bitmap_controller.sv" "$r33_hw/rtl/tb_bitmap.cpp" > "$r33_obj/elaborate.log" 2>&1; then cp "$r33_obj/elaborate.log" "$r33_hw/results/failures/${r33_kind}_elaborate.log"; exit 1; fi
 if ! make -C "$r33_obj/obj" -f "Vbitmap_ctrl_$r33_kind.mk" -j2 > "$r33_obj/compile.log" 2>&1; then cp "$r33_obj/compile.log" "$r33_hw/results/failures/${r33_kind}_compile.log"; exit 1; fi
 if ! "$r33_obj/obj/Vbitmap_ctrl_$r33_kind" "$r33_hw/bitmap_words.txt" > "$r33_obj/RESULT.json" 2> "$r33_obj/run.stderr"; then cp "$r33_obj/run.stderr" "$r33_hw/results/failures/${r33_kind}_run.stderr"; exit 1; fi
 cat "$r33_obj/RESULT.json"
done
