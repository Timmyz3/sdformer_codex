#!/usr/bin/env python3
"""One real synthesis point, with no borrowed area/power from older TSBG."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import run_m2233_ep34_tsbg_matched_power_repair_one_shot as cfg

AXES={'ordinary':(0,0),'group_demand':(1,0),'union':(1,1)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--axis',choices=tuple(AXES),required=True)
    args=ap.parse_args()
    cfg.no_same_uid_eda()
    out=Path(tempfile.mkdtemp(prefix=f'm2257_{args.axis}_',dir=cfg.HW/'results'))
    print('Co-fill physical output:',out,flush=True)
    mode,union=AXES[args.axis]
    env={**os.environ,'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C',
        'SNPSLMD_LICENSE_FILE':cfg.LICENSE_SERVER,'LM_LICENSE_FILE':cfg.LICENSE_FILE,
        'M2257_HW':str(cfg.HW),'M2257_OUTPUT':str(out),'M2257_MODE':str(mode),
        'M2257_UNION':str(union),'M2257_SLOW':str(cfg.SLOW_DB),'M2257_FAST':str(cfg.FAST_DB)}
    with (out/'dc.log').open('w') as log:
        subprocess.run([str(cfg.DC),'-f',str(Path(__file__).with_name('m2257_cofill_physical.tcl'))],
            cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=7200,check=True)
    if re.search(r'^Error:',(out/'dc.log').read_text(),re.M):
        raise RuntimeError('DC error; inspect '+str(out/'dc.log'))
    result=dict(axis=args.axis,rtl='rtl_m2249/m2249_c2_consumer_scoped_bank_fill_frontend.sv',
        schedule_mode=mode,union_prefetch=union,scope='3ns prelayout logic + common clock gating; no SRAM bank energy')
    for kind in ('setup','hold'):
        result[kind+'_ns']=float(re.search(r'slack \([^)]*\)\s+([-\d.]+)',
            (out/f'reports/{kind}_after.rpt').read_text()).group(1))
    result['area_um2']=float(re.search(r'Total cell area:\s+([\d.]+)',
        (out/'reports/area.rpt').read_text()).group(1))
    result['setup_and_hold_met']=result['setup_ns']>=0 and result['hold_ns']>=0
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
