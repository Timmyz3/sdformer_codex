"""Give every front end the already measured ordinary P1 retained-Z suffix."""
import inspect
import numpy as np
import run_binding as binding
import flows

def install():
    code=inspect.getsource(flows.retained_z)
    code=code.replace('    for ip in range(2):','    observed_U=[]; U_times=[]\n    for ip in range(2):',1)
    code=code.replace('        for h0 in (0,48):',
        '        m.drain(); observed_U.append(m.rf[:30].copy().reshape(10,24)); U_times.append(m.time)\n        for h0 in (0,48):')
    code+='\n    return np.asarray(observed_U,np.int64),max(U_times)\n'
    scope=dict(flows.__dict__);exec(compile(code,'retained_Z_observed_U','exec'),scope)
    retained=scope['retained_z']
    original=binding.pair.consumer.run
    code=inspect.getsource(original)
    old="""                    dense(m,base,'U_ped',96,rank,UPDATED,PED_U,count,int(q['U_ped_exponent']))
                    u_ready=m.time;actual_u=read24(m,PED_U,(count,10,rank))
                    if not late_v:
                        dense(m,base,'V_ped',rank,96,PED_U,PED_V,count,int(q['V_ped_exponent']),bias='PED_bias')
                        actual_ped=read24(m,PED_V,(count,10,96))"""
    new="""                    assert count==2 and rank==24 and not late_v
                    actual_u,u_ready=retained(m,base,q,UPDATED,PED_V)
                    actual_ped=read24(m,PED_V,(count,10,96))"""
    assert old in code
    code=code.replace(old,new)
    scope=dict(original.__globals__);scope['retained']=retained
    exec(compile(code,'common_P1_retained_suffix','exec'),scope)
    return scope['run']
