#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
/opt/anaconda3/bin/python3.12 probe.py > probe.log
bash build_source.sh
for variant in zero l2; do
  for scope in small expanded; do
    ./source_obj/Vsource_classifier source_${variant}_${scope}.bin source_${variant}_${scope}.csv >source_${variant}_${scope}.log 2>&1
  done
done
/opt/anaconda3/bin/python3.12 summarize_source.py
bash build_joined.sh
./joined_obj/Vjoined_core 1 1 1 1 2 joined_zero.bin joined_zero_small_cycles.csv zero_response 1 >joined_zero_small.log 2>&1
./joined_obj/Vjoined_core 4 4 2 1 2 joined_zero.bin joined_zero_cycles.csv zero_response 1 >joined_zero.log 2>&1
./joined_obj/Vjoined_core 4 4 2 1 2 joined_l2.bin joined_l2_cycles.csv integer_L2_zero_allowed 1 >joined_l2.log 2>&1
bash build_neutral.sh
./neutral_obj/Vjoined_core 4 4 2 1 2 joined_zero.bin joined_zero_neutral_cycles.csv zero_response 1 >joined_zero_neutral.log 2>&1
./neutral_obj/Vjoined_core 4 4 2 1 2 joined_l2.bin joined_l2_neutral_cycles.csv integer_L2_zero_allowed 1 >joined_l2_neutral.log 2>&1
/opt/anaconda3/bin/python3.12 profile_joined.py
/opt/anaconda3/bin/python3.12 finalize_joined.py

/opt/anaconda3/bin/python3.12 finalize_neutral.py
