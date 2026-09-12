"""Use the original fixed compiler in its existing da4ml Python environment."""
from pathlib import Path
import importlib.util,json
import numpy as np
HERE=Path(__file__).resolve().parent
BREADTH=HERE.parents[1]
spec=importlib.util.spec_from_file_location('same_fixed_CSD_control',HERE.parent/'dense_low_state_source/run.py')
fixed=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixed)
q=dict(np.load(BREADTH/'algorithm/matched_training/contiguous34/stage320/deployed_constants.npz'))
program,rows=fixed.compile_fixed(q)
fixed.dump(HERE/'contiguous34_low_program.json',program)
fixed.dump(HERE/'contiguous34_low_compilation.json',dict(program_words=len(program),ROM_words=512,work_RF=13,rows=rows,compiler='Unchanged prior deterministic two-chain CSD; CPU full B source checks, not a new RTL claim.'))
assert len(program)<=512
