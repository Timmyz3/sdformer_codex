"""Retain existing fixed RTL model headers; replace only the dynamic X cases."""
from pathlib import Path
import io
import struct
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
d=np.load(HERE/'validation_tiles.npz')
s=np.load(ROOT/'source_cases.npz')
for dst,src in [('A_fp32','A_fp32'),('bias_fp32','bias_fp32'),
                ('center_fp32','center_fp32'),('theta_fp32','theta_fp32'),('D','D')]:
    assert np.array_equal(d[dst],s[src]),dst
for tag,source in [('adapt',ROOT/'frontier_joined/inputs_adapt_expanded.bin'),
                   ('zero',ROOT/'zero_response_reopen/frontier_retained/inputs_expanded.bin')]:
    data=source.read_bytes()
    f=io.BytesIO(data)
    assert f.read(8)==b'JOIN0001'
    f.seek(1536+200+80+200+30720+384+384+3840+36864*2,1)
    for _ in range(2):
        n,=struct.unpack('<I',f.read(4))
        f.seek(n*8,1)
    f.seek(24+96+96,1)
    header=data[:f.tell()]
    with (HERE/f'validation_{tag}.bin').open('wb') as out:
        out.write(header)
        out.write(struct.pack('<I',len(d['case_name'])))
        for name,x in zip(d['case_name'],d['X_q16']):
            name=str(name).encode()
            assert x.shape==(32,96,10)
            out.write(struct.pack('<II',1,len(name)))
            out.write(name)
            out.write(x.astype('<i4').tobytes())
    print(tag,len(d['case_name']),'dynamic cases; static model header retained')
