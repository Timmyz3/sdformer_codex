#!/usr/bin/env python3
"""Port-checked, zero-delay mapped VCS activity; no inference about routed timing."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import run_m2233_ep34_tsbg_matched_power_repair_one_shot as cfg

CELL = Path('/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/tcbn28hpcplusbwp35p140.v')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--axis', choices=tuple(cfg.AXES), required=True)
    ap.add_argument('--dc', type=Path, required=True)
    ap.add_argument('--window', choices=tuple(cfg.STRATA), action='append')
    ap.add_argument('--reuse-build', type=Path, help='Completed build for this same axis and mapped directory')
    ap.add_argument('--weights', type=Path, default=cfg.HW/'results/m2251_fc_power_weight_inputs')
    args = ap.parse_args()
    netlist = args.dc.resolve()/'netlist/m2018_axis_mapped.v'
    top = re.search(r'\bmodule\s+(m2018_c2_tsbg_b4\w+)', netlist.read_text()).group(1)
    out = Path(tempfile.mkdtemp(prefix=f'm2256_gate_saif_{args.axis}_', dir=cfg.HW/'results'))
    print('Mapped activity output:', out, flush=True)
    env = {**os.environ, 'PATH':'/usr/bin:/bin', 'LANG':'C', 'LC_ALL':'C',
        'SNPSLMD_LICENSE_FILE':cfg.LICENSE_SERVER, 'LM_LICENSE_FILE':cfg.LICENSE_FILE,
        'VCS_HOME':str(cfg.VCS.parent.parent), 'VCS_ARCH_OVERRIDE':'linux'}
    harness = (cfg.HW/'tb_m2018/tb_m2217_m2018_tsbg_matched_native_saif_power.sv').read_text()
    boundary = harness.index('    m1880_c2_tsbg_b4_real_channel_signed_frontend_assertions')
    dut, rest = harness[:boundary], harness[boundary:]
    # DC's port map places unpacked index zero in the most-significant slice.
    # Concatenations are legal input expressions and output lvalues. Only the
    # mapped instance is flattened; interface, memory and SVA stay unpacked.
    banks = ('mem_req_epoch mem_req_slot mem_req_generation mem_req_tag '
        'mem_req_output_block mem_req_slice mem_req_source_channel mem_rsp_epoch '
        'mem_rsp_slot mem_rsp_generation mem_rsp_tag bridge_source_channel '
        'bridge_source_value').split()
    shapes = {name:(8,) for name in banks}
    shapes.update(mem_rsp_weight=(8,16), bridge_effective_weight=(8,16), commit_accumulator=(16,))
    for name, shape in shapes.items():
        elements = [f'axis.{name}[{i}]' for i in range(shape[0])]
        if len(shape)==2:
            elements = [f'{item}[{j}]' for item in elements for j in range(shape[1])]
        dut = dut.replace(f'.{name}(axis.{name})', f'.{name}({{{", ".join(elements)}}})')
    mapped_harness = out/'mapped_power_tb.sv'
    mapped_harness.write_text(dut+rest)
    sources = [cfg.HW/'verif_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend_assertions.sv',
        cfg.HW/'tb_m2018/tb_m2160_m2018_ordinary_native_saif_report_reset_preflight.sv', mapped_harness]
    simv=out/'simv'
    if args.reuse_build:
        prior=json.loads((args.reuse_build/'result.json').read_text())
        if prior['axis']!=args.axis or Path(prior['dc'])!=args.dc.resolve():
            raise ValueError('Build is for another axis/netlist directory')
        simv=args.reuse_build.resolve()/'simv'
    else:
        with (out/'compile.log').open('w') as log:
            subprocess.run([str(cfg.VCS), '-full64', '-sverilog', '-timescale=1ns/1ps',
                '+vcs+initreg+random', '+M2256_DIRECT_GATE_ACTIVITY',
                f'+define+M2217_SCHEDULE_MODE={cfg.AXES[args.axis]}',
                f'+define+M2256_MAPPED_TOP={top}', '+define+M2253_CAPTURED_WEIGHTS',
                '-debug_access+r', '-assert', 'svaext', '-lca', str(CELL), str(netlist),
                *map(str,sources), '-top', 'tb_m2217_m2018_tsbg_matched_native_saif_power',
                '-o',str(simv)], cwd=out, env=env, stdout=log,
                stderr=subprocess.STDOUT, timeout=1800, check=True)
    rows=[]
    for window in args.window or cfg.STRATA:
        point=out/window
        point.mkdir()
        with (point/'vcs.log').open('w') as log:
            subprocess.run([str(simv), '-no_save', f'+M2217_STRATUM={window}',
                f'+M2253_WEIGHTS={args.weights.resolve()/(window+"_weights.memh")}', '-ucli',
                '-i',str(Path(__file__).with_name('m2256_mapped_power.ucli.tcl'))],
                cwd=point, env={**env,'M2256_OUTPUT':str(point)}, stdout=log,
                stderr=subprocess.STDOUT, timeout=1800, check=True)
        text=(point/'vcs.log').read_text()
        if 'PASS_M2217_SINGLE_DUT_NATIVE_SAIF' not in text:
            raise RuntimeError('Mapped arithmetic/ledger failed: '+str(point))
        cycles=int(re.search(r'M2217_WINDOW_END .*?cycles=(\d+)',text).group(1))
        activity=(point/'activity.saif').read_text()
        scale=re.search(r'\(TIMESCALE\s+([\d.]+)\s+(\w+)\)',activity)
        duration=float(re.search(r'\(DURATION\s+([\d.]+)\)',activity).group(1))
        ns=duration*float(scale.group(1))*{'ps':.001,'ns':1}[scale.group(2)]
        if abs(ns-3*cycles)>.001:
            raise RuntimeError('Mapped activity window differs from the cycle ledger')
        rows.append(dict(window=window,cycles=cycles,activity=str(point/'activity.saif'),numeric_pass=True))
        print('Mapped real-weight VCS PASS:',args.axis,window,flush=True)
    (out/'result.json').write_text(json.dumps(dict(axis=args.axis,dc=str(args.dc.resolve()),rows=rows,
        scope='Zero-delay foundry gate simulation; candidate INT8 FC weights; no routed timing or AEE claim'),indent=2)+'\n')


if __name__=='__main__':
    main()
