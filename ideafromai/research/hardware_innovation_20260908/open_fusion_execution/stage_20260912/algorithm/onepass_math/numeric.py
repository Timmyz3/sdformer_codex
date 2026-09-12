"""Exact arithmetic only companion; CPU timings never enter service claims."""
from pathlib import Path
import ctypes
import subprocess
import numpy as np
HERE=Path(__file__).resolve().parent

def load():
    binary=HERE/'libonepass_math.so'
    if not binary.exists() or binary.stat().st_mtime<(HERE/'numeric.cpp').stat().st_mtime:
        subprocess.run(['g++','-std=c++17','-O3','-march=native','-ffp-contract=off','-fPIC','-shared',str(HERE/'numeric.cpp'),'-o',str(binary)],check=True)
    lib=ctypes.CDLL(str(binary))
    pointer=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    lib.moments.argtypes=[pointer,ctypes.c_int,pointer,pointer,ctypes.c_float,pointer]
    lib.moments.restype=ctypes.c_int
    lib.centered_moments.argtypes=lib.moments.argtypes
    lib.centered_moments.restype=ctypes.c_int
    lib.normalize.argtypes=[pointer,ctypes.c_int,pointer,pointer]
    lib.normalize.restype=None
    return lib

class Arithmetic:
    def __init__(self,centered=False):
        self.lib=load();self.calculate=self.lib.centered_moments if centered else self.lib.moments
    def statistics(self,x,gamma,beta,eps):
        x=np.ascontiguousarray(x,np.float32).reshape(-1,96)
        stats=np.empty((5,96),np.float32)
        code=self.calculate(x,x.shape[0],np.ascontiguousarray(gamma,np.float32),np.ascontiguousarray(beta,np.float32),float(eps),stats)
        if code:raise ValueError(f'onepass arithmetic rejected input: {code}')
        return stats
    def output(self,x,stats):
        x=np.ascontiguousarray(x,np.float32).reshape(-1,96);y=np.empty_like(x)
        self.lib.normalize(x,x.shape[0],stats,y)
        return y

def difference(a,b):
    a=np.asarray(a,np.float32);b=np.asarray(b,np.float32)
    d=a.astype(np.float64)-b.astype(np.float64)
    return dict(values=a.size,bit_differences=int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32))),max_abs=float(np.max(np.abs(d))))
