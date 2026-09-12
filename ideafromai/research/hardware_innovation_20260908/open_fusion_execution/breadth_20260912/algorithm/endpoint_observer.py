"""Original FullCapture geometry, small outputs and actual onepass stats."""
import numpy as np
from capture_full_producers import FullCapture


class SmallCapture(FullCapture):
    def __init__(self,*args,structure,latest,function_label=None,stop_after_first=False):
        self.structure=structure;self.latest=latest;self.function_label=function_label or structure
        self.stop_after_first=stop_after_first
        super().__init__(*args)
    def full_integer(self,*unused):pass
    def full_gate(self,*unused):pass
    def output_values(self,key,final=False):
        def hook(module,inputs,output):
            self.order.append(key);self.save_value(key,output,'output')
            if key=='proj_norm_fp32':
                self.arrays['proj_bn_full_input_shape']=np.asarray(inputs[0].shape)
                self.arrays['proj_bn_onepass_statistics']=self.latest['stats'].copy()
                self.arrays['proj_bn_function']=np.asarray('Actual full-domain onepass statistics, same separate affine MUL/ADD; no free local statistics.')
            if final:
                self.arrays['trained_structure']=np.asarray(self.structure)
                self.arrays['function_label']=np.asarray(self.function_label)
                self.arrays['cumulative_new_GT_steps']=np.asarray(320)
                self.save_frame()
                if self.stop_after_first:self.restore()
        return hook
