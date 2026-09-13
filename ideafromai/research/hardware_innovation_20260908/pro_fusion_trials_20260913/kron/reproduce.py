"""Bounded CPU-only rerun; Python3.12 + preinstalled numpy + Verilator + make."""
from pathlib import Path
import json,subprocess,sys
HERE=Path(__file__).resolve().parent
assert sys.version_info[:2]==(3,12)
with (HERE/'fit_stdout.json').open('w') as log:
    subprocess.run([sys.executable,'fit_and_fixture.py'],cwd=HERE,stdout=log,check=True)
with (HERE/'build.log').open('w') as log:
    subprocess.run(['verilator','--cc','--exe','-Wall','--top-module','ped_kron',
        '--Mdir','obj_dir','ped_kron.sv','tb.cpp'],cwd=HERE,stdout=log,stderr=log,check=True)
    subprocess.run(['make','-C','obj_dir','-f','Vped_kron.mk','-j2','CXXFLAGS=-std=c++11'],
        cwd=HERE,stdout=log,stderr=log,check=True)
with (HERE/'run.log').open('w') as log:
    subprocess.run(['./obj_dir/Vped_kron'],cwd=HERE,stdout=log,stderr=log,check=True)
subprocess.run([sys.executable,'summarize.py'],cwd=HERE,check=True)
