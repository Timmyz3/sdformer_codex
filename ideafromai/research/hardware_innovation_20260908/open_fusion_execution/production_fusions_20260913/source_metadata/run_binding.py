from pathlib import Path
import importlib.util,sys
HERE=Path(__file__).resolve().parent
BREADTH=HERE.parents[1]/'breadth_20260912'
sys.path.insert(0,str(BREADTH/'hardware/actual_interleave'))
spec=importlib.util.spec_from_file_location('source_metadata_pair',BREADTH/'hardware/actual_interleave/run.py')
pair=importlib.util.module_from_spec(spec);spec.loader.exec_module(pair)

def inputs(axis,label):
    import json,numpy as np
    structure,q,program,_,live,_,_=pair.inputs(axis)
    original=pair.common.read_npz(pair.common.FULL/'capture/ordinary/000_zurich_city_09_a_0001.npz')
    geo=json.loads(str(original['window_geometry_json']))[label]
    source=pair.matched.source_gold(original[label+'_I24'],q,structure)
    preview,margin=pair.matched.preview_gold(source,live,geo)
    expected,_=pair.matched.oracle.independent_gold(original,q,label,preview['gate'])
    data=dict(original)
    for key,value in [('sn1_gate',source),('preview_Z_shared',preview['z']),('preview_shared_raw',preview['raw']),
        ('preview_BN1_Y',preview['y']),('sn2_gate',preview['gate']),('updated_I24',expected['updated']),
        ('proj_gate',expected['gate']),('continuous_q24',expected['continuous'])]:data[label+'_'+key]=value
    return structure,q,program,data,live
