"""No-data audit: frozen coarse statistics cannot encode within-cell orientation."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'algorithm'));import predictors as p
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    assert sha(p.__file__)=='35663dd6b999621652d9d185d72b967e33c380e1064a09ec75540d2759cd61c3'
    z=np.zeros((10,480,640),dtype=np.float32)
    # A signed translating synthetic patch remains inside one120x160 cell.
    for t in range(10):
        y,x=160+2*t,210+3*t
        z[t,y:y+6,x:x+6]=1
        z[t,y:y+6,x-5:x-2]=-1
    grid=z.reshape(10,4,120,4,160)
    cases={'within_cell_horizontal_mirror':grid[:,:,:,:,::-1].reshape(z.shape),
           'within_cell_vertical_mirror':grid[:,:,::-1,:,:].reshape(z.shape),
           'within_cell_both_mirrors':grid[:,:,::-1,:,::-1].reshape(z.shape)}
    x=p.voxel_features(z);records=[]
    vectors={'within_cell_horizontal_mirror':[-3,2],'within_cell_vertical_mirror':[3,-2],'within_cell_both_mirrors':[-3,-2]}
    for name,v in cases.items():
        xx=p.voxel_features(v);assert np.array_equal(x,xx) and not np.array_equal(z,v)
        records.append(dict(case=name,features_bitwise_equal=True,feature_max_abs_difference=float(np.max(np.abs(x-xx))),source_voxel_sha256=p.digest_arrays(z),transformed_voxel_sha256=p.digest_arrays(v),synthetic_translation_pixels_per_bin=vectors[name]))
    out=dict(schema='flow_r33_feature_observability_audit_v1',status='SYNTHETIC_INVARIANCE_CONFIRMED',predictors_sha256=sha(p.__file__),runner_sha256=sha(__file__),
        original_synthetic_translation_pixels_per_bin=[3,2],cases=records,
        reason='Every fixed cell statistic is a spatial sum/count of per-pixel temporal quantities or a temporal mass moment; a time-consistent within-cell pixel permutation preserves each statistic. No spatial derivative/correlation/orientation is encoded inside a cell.',
        scope='A feature information limitation, not a test of actual R33 damage labels, not proof that dataset-level damage prediction cannot work. Cross-cell transport and coarse spatial masses can remain informative.',
        consequences='Keep frozen Round33 unchanged. Failure would not isolate failure of damage information from this input observability limit. If needed, future spatial/temporal derivative or cheap flow proxy must be shared fairly with ordinary heads and its hardware cost paid.',
        real_data_read=False,GT_read=False,model_forwards=0,GPU_used=False,feature_or_model_changed=False)
    dest=HERE/'FEATURE_OBSERVABILITY_V1.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print(out['status'])
if __name__=='__main__':main()
