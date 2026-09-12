"""Same-function expanded versus genuinely packed whole-frame native W8."""
from pathlib import Path
import argparse, importlib.util, json, subprocess, sys
import numpy as np

HERE=Path(__file__).resolve().parent
WORK=Path('/tmp/native_bn_join_20260912/native_w8')
CAPTURE=Path('/tmp/native_bn_join_20260912/capture/ordinary.npz')
spec=importlib.util.spec_from_file_location('native_base_w8',HERE.parent/'run.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
spec=importlib.util.spec_from_file_location('native_w8_parameters',HERE/'prepare.py')
parameters=importlib.util.module_from_spec(spec);spec.loader.exec_module(parameters)


def headers(mode):
    source=(HERE.parent/'directory_reuse/reuse_native.hpp').read_text()
    if mode=='compact':
        prefix=source.split('static std::vector<float> native(')[0]
        suffix=source[source.index('// No Engine calls,'):]
        source=prefix+(HERE/'compact_native_body.hpp').read_text()+'\n'+suffix
    (HERE/f'{mode}_native.hpp').write_text(source)
    pipeline=(HERE.parent/'pipeline.cpp').read_text().replace('#include "native.hpp"',f'#include "{mode}_native.hpp"')
    if mode=='compact':
        marker='    Engine e(false);'
        pipeline=pipeline.replace(marker,'    CODE_BYTES=read_bytes((dir+"/weights_packed.bin").c_str()); ROW_SCALE=read((dir+"/row_scales.f32").c_str());\n'+marker)
        marker='        <<",\\\"raw_spill_bytes\\\":"<<native_cost.raw_bytes'
        replacement='        <<",\\\"packet_decode_issues\\\":"<<packet_decode_issues<<",\\\"sign8_decode_issues\\\":"<<sign8_decode_issues<<",\\\"restore_scale_issues\\\":"<<restore_scale_issues\n'+marker
        assert marker in pipeline
        pipeline=pipeline.replace(marker,replacement)
    (HERE/'pipeline.cpp').write_text(pipeline)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['expanded','compact'])
    ap.add_argument('--capture',type=Path,default=CAPTURE);a=ap.parse_args()
    parameters.main();headers(a.mode)
    WORK.mkdir(exist_ok=True);work=WORK/a.mode;work.mkdir(exist_ok=True)
    original_prepare=base.prepare
    p=dict(np.load(HERE/'native_w8.npz'))
    def prepare(capture=None):
        result=original_prepare(capture)
        p['expanded_weight_fp32'].astype('<f4').tofile(work/'weights_TF32.f32')
        # Actual physicalCR256 layout: K, H32, lane, four H8 groups.
        packet=p['q'].reshape(3,4,8,864).transpose(3,0,2,1).copy()
        assert packet.size==82944
        packet.tofile(work/'weights_packed.bin');p['scale'].astype('<f4').tofile(work/'row_scales.f32')
        return result
    base.HERE=HERE;base.WORK=work;base.prepare=prepare
    oldargv=sys.argv;sys.argv=['native_w8', '--capture',str(a.capture)]
    base.main();sys.argv=oldargv
    path=HERE/'results_ready.json';r=json.loads(path.read_text())
    r.update(layout=a.mode,quantized_function='Exact declared row dyadicW8; expanded_FP32 and compact same q and scale.',
        borrowed_baseline='Ordinary dyadic quantization, finite LoopTree/Gustav-style source blocking and packed-weight residency; no title X.',
        native_function='Same ascending-K W8 function; compact sums exact int8-valued FP32 then paid row-scale MUL. Complete BN/PED unchanged.',
        evidence='CPU complete-domain payload/port prototype, no RTL/PPA or inherited AEE.')
    r['resource'].update(native_state=dict(directories=[0,110592],headers=[110592,110720],gates=[110720,126272],output_staging=[126272,126304]),
        state_high_water=126304,coefficient_high_water=83328 if a.mode=='compact' else 110592)
    if a.mode=='compact':
        r['resource'].update(native_RF='0..79 accumulator;80 decoded code;81 uniform dyadic scale;93/94 packed uint16 pairs. During directory phase64..72 ordinary H4 cache.',
            packed_code_bytes=82944,scale_bytes=384,
            CR_response_bytes=32,decoder='Fixed same-lane byte select/sign extension; one serialized issue per packet-half packing and per H8 decode plus real RF LOAD/writeback; no arbitrary lane gather.')
    r['limits']=[
        'Existing full ordinary R24+onepass external source-gate/PED boundary. Upstream service and packing not included.',
        'The native W8 function is new and has fresh quality pending; same arithmetic reference does not assert CUDA reduction equivalence.',
        'Only one fixed two4x4 directory layout and ready profile. No altered ALU operation or free decoder issue.',
        'Ordinary fixed format/layout baseline, not a novel mechanism.']
    (HERE/f'{a.mode}_results.json').write_text(json.dumps(r,indent=2)+'\n');path.unlink()
    print('NATIVE_W8_MODE_COMPLETE',a.mode,flush=True)


if __name__=='__main__':main()
