#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
/opt/anaconda3/bin/python3.12 prepare.py --expanded > prepare_expanded.log
/opt/anaconda3/bin/python3.12 prepare.py --expanded --adapt > prepare_expanded_adapt.log
./obj_dir/Vjoined_core 34 4 1 1 1 inputs_expanded.bin expanded_old.csv response_class 1 > expanded_old.log 2>&1
./obj_dir/Vjoined_core 34 4 1 1 1 inputs_adapt_expanded.bin expanded_new.csv adapt_response_class 1 > expanded_new.log 2>&1
/opt/anaconda3/bin/python3.12 summarize.py
