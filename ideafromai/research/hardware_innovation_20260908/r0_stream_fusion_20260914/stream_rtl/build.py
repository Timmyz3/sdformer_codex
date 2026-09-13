from pathlib import Path
import subprocess
H=Path(__file__).resolve().parent
with (H/'build.log').open('w') as log:
 for cmd in (['verilator','--cc','--exe','-Wall','-CFLAGS','-O3','--top-module','stream_wrapper','--Mdir',str(H/'obj_dir'),str(H/'stream_wrapper.sv'),str(H/'pair_parent_merge.sv'),str(H/'tb.cpp')],
             ['make','-C',str(H/'obj_dir'),'-f','Vstream_wrapper.mk','-j2']):
  subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
print('BUILD_PASS Verilator4.028 --cc --exe then make; C++ -O3')
