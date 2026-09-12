from pathlib import Path
import argparse,json
import run_binding as binding
import metadata
import strong_suffix
pair=binding.pair
HERE=Path(__file__).resolve().parent
ORIGINAL_SN2=pair.windows.execute_sn2
ORIGINAL_INTEGER_DIRECTORY=pair.consumer.directory
STRONG_SUFFIX=strong_suffix.install()

def execute(axis,label,mode,stress=False):
    structure,q,program,data,live=binding.inputs(axis,label)
    import source_program
    source_mode=mode.split('_')[0]
    source_program.run=lambda m,d,a,l:metadata.source(m,d,a,l,program,source_mode)
    pair.windows.Machine=metadata.MetadataMachine
    pair.windows.build_nrv=pair.matched.stage.preview_directory if mode=='baseline' else metadata.directory
    pair.windows.execute_sn2=metadata.sn2_with_summary(ORIGINAL_SN2) if mode=='producer_both' else ORIGINAL_SN2
    pair.consumer.directory=metadata.integer_directory if mode.endswith('_both') else ORIGINAL_INTEGER_DIRECTORY
    _,producer,m=pair.windows.window(data,live,label,False,stress,True,structure)
    assert all(x['differences']==0 for x in producer['checks'].values())
    geo=json.loads(str(data['window_geometry_json']))[label]
    if mode=='consumer_both':metadata.build_sn2_summary(m,geo)
    if mode.endswith('_both'):metadata.check_sn2_summary(m,geo)
    producer_end=m.time;m.forward_i24=True
    # The clone has its own globals, so update the selected directory there.
    STRONG_SUFFIX.__globals__['directory']=pair.consumer.directory
    _,consumer=STRONG_SUFFIX(data,q,label,None,stress=stress,machine=m,rank=24)
    assert all(x['differences']==0 for x in consumer['checks'].values())
    assert sum(m.stages.values())==m.time
    return dict(axis=axis,label=label,mode=mode,stress=stress,service_slots=m.time,
        producer_end=producer_end,producer=producer,consumer=consumer,counts=dict(m.count),stages=dict(m.stages),
        metadata_bytes=data[label+'_sn1_gate'].shape[2]*data[label+'_sn1_gate'].shape[3]*8 if mode!='baseline' else 0,
        sn2_metadata_bytes=8*geo['gate_shape'][0]*geo['gate_shape'][1] if mode.endswith('_both') else 0,
        metadata_base=metadata.META,RF=[96,8,48],ROM_words=512,state_bytes=131072,coefficient_bytes=131072,
        evidence='Actual payload same-Machine CPU complete local source/preview/sn2/integer/PED. Not full layer, ASIC PPA, or RTL speed.',
        common_consumer='Immediate ordinary P1 retained Z, original RNE/sat/bias, actual PED/gate egress.',
        new_numeric_function=False,new_AEE=False,new_PPA=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('axis',choices=['dense_twopot','lifting_twopot','contiguous34']);ap.add_argument('label',choices=['corner','interior']);ap.add_argument('--stress',action='store_true');ap.add_argument('--modes',default='baseline,consumer,producer');args=ap.parse_args()
    for mode in args.modes.split(','):
        print('START',args.axis,args.label,mode,args.stress,flush=True)
        result=execute(args.axis,args.label,mode,args.stress)
        name='_'.join([args.axis,args.label,mode])+('_stress' if args.stress else '')
        (HERE/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        print('DONE',name,result['service_slots'],result['producer_end'],flush=True)

if __name__=='__main__':main()
