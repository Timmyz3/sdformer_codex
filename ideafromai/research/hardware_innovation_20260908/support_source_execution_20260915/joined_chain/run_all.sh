#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bash run.sh > full.log 2>&1
/opt/anaconda3/bin/python3.12 prepare.py --adapt > prepare_adapt.log
./obj_dir/Vjoined_core 4 4 2 1 2 inputs.bin strong_cycles.csv response_class 1 > strong.log 2>&1
./obj_dir/Vjoined_core 4 4 2 1 2 inputs_adapt.bin adapt_strong_cycles.csv adapt_response_class 1 > adapt_strong.log 2>&1
/opt/anaconda3/bin/python3.12 profile.py
/opt/anaconda3/bin/python3.12 finalize.py
