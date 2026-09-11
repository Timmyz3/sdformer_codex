"""Observer-only full-frame extension of the existing Stage B capture.

Same two students and preselected frame. No training or quantization. Adds
actual full I24/updated I24, three T10 gate-word maps and continuous PED for
whole-layer replay and producer-side zero provenance. Existing local arrays
and complete BN input/output remain available for direct consistency checks.
"""
import json
import sys
import numpy as np
import capture_reference as ref


class FullCapture(ref.WindowCapture):
    def full_integer(self,name,value):
        import torch
        if value.ndim==5:value=value[:,0]
        assert torch.equal(value,value.round())
        assert float(value.min())>=-(1<<23) and float(value.max())<(1<<23)
        self.arrays[name]=value.detach().to(torch.int32).cpu().numpy()

    def full_gate(self,name,output,theta):
        import torch
        value=output[:,0] if output.ndim==5 else output
        words=np.zeros(tuple(value.shape[1:]),np.uint16)
        for t in range(10):
            g=value[t].ne(0)
            assert float(torch.where(g,value[t]-float(theta),value[t]).abs().max())==0
            words|=g.detach().cpu().numpy().astype(np.uint16)<<t
        self.arrays[name]=words

    def source_values(self,module,inputs,output):
        super().source_values(module,inputs,output)
        self.full_integer('full_I24',self.helper.i)
        self.full_gate('full_sn1_words',output,float(module.thresh))

    def sn2_values(self,module,inputs,output):
        super().sn2_values(module,inputs,output)
        self.full_gate('full_sn2_words',output,self.theta_sn2)

    def observe_continuous(self,module,inputs,output):
        super().observe_continuous(module,inputs,output)
        self.full_integer('full_updated_I24',self.helper.updated)
        self.full_integer('full_continuous_q24',output.double()*(1<<14))

    def proj_gate_values(self,module,inputs,output):
        super().proj_gate_values(module,inputs,output)
        self.full_gate('full_proj_words',output,float(module.thresh))


def main():
    if '--output' not in sys.argv:sys.argv+=['--output',str(ref.HERE/'capture_full_producers')]
    ref.WindowCapture=FullCapture
    ref.main()
    output=ref.Path(sys.argv[sys.argv.index('--output')+1])
    path=output/'result.json';record=json.loads(path.read_text())
    record.update(scope=__doc__,capture_scope=__doc__,full_producer_capture=True,
        full_tensor_layout='I24/updated/PED: T,C,H,W int32 containing signed24. Gate words: C,H,W uint16, bit t is original timestep t.',
        hardware_replay=False)
    ref.base.save_json(path,record)


if __name__=='__main__':main()
