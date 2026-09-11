"""Exact ordinary/lifting control: retain U32 until native BN completes.

The existing U output already has a signed24 boundary. Delaying V, its RNE,
bias and saturation preserves those operations, unlike moving BN or rounding.
This is a common low-rank retention baseline, not the lifting novelty itself.
"""
from pathlib import Path
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from consumer_service import read_npz, rne24


def store24(values):
    value = np.asarray(values, np.int64)
    assert value.min() >= -(1 << 23) and value.max() < (1 << 23)
    u = value.reshape(-1) & 0xffffff
    return np.stack([u & 255, (u >> 8) & 255, (u >> 16) & 255], axis=1).astype(np.uint8)


def load24(payload, shape):
    b = payload.astype(np.int64)
    u = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
    return ((u ^ (1 << 23))-(1 << 23)).reshape(shape)


def main():
    result = dict(scope=__doc__, axes={}, native_BN_reduction_hardware_closed=False)
    for axis in ('ordinary', 'lifting_raw'):
        p = HERE/'capture'/axis
        q = read_npz(p/'parameters.npz')
        data = read_npz(p/'000_zurich_city_09_a_0001.npz')
        geos = json.loads(str(data['window_geometry_json']))
        rows = {}
        for label in ('corner', 'interior'):
            geo = geos[label]
            updated = data[label+'_updated_I24']
            anchors = np.empty((10, 96, 4, 4), np.int64)
            for y in range(4):
                for x in range(4):
                    sy = 2*(geo['output_origin'][0]+y)-geo['gate_origin'][0]
                    sx = 2*(geo['output_origin'][1]+x)-geo['gate_origin'][1]
                    anchors[:, :, y, x] = updated[:, :, sy, sx]
            matrix = anchors.transpose(1, 0, 2, 3).reshape(96, -1)
            latent = rne24(q['U_ped_q16'].astype(np.int64) @ matrix, int(q['U_ped_exponent']))
            payload = store24(latent)
            retained = load24(payload, latent.shape)
            # Native projection and its complete BN may run between these
            # two operations. V has no consumer-side input other than U24.
            value = rne24(q['V_ped_q16'].astype(np.int64) @ retained, int(q['V_ped_exponent']))
            value = rne24(value+q['PED_bias_q24'][:, None], 0)
            value = value.reshape(96, 10, 4, 4).transpose(1, 0, 2, 3)
            gold = data[label+'_continuous_q24']
            rows[label] = dict(latent_values=int(latent.size), latent_min=int(latent.min()), latent_max=int(latent.max()),
                packed_payload_bytes=int(payload.nbytes),
                payload_roundtrip_differences=int(np.count_nonzero(retained != latent)),
                continuous_values=int(value.size), continuous_differences=int(np.count_nonzero(value != gold)))
        result['axes'][axis] = rows
    values = 10*96*120*160
    early = values*3
    late = values//3*3
    result['full_frame_storage'] = dict(early_V96_bytes=early, late_U32_bytes=late,
        fewer_live_external_bytes=early-late, store_and_reread_bytes_saved=2*(early-late),
        shared_DMA_occupied_slots_released=2*(early-late)//32*5,
        arithmetic_saving=0, common_control_for_both_axes=True,
        V96_work='Moved after BN statistics readiness, never deleted. Same original U24/V24/bias saturation boundaries.',
        coefficient_residency='H32 native projection110592 + PED V6144 + PED bias288 + native BN affine768 =117792 bytes <128KiB. Keeping these weights avoids a hidden V refill; both axes receive the same reservation.')
    result['full_frame_storage']['net_service_saving'] = None
    result['full_frame_storage']['net_service_note'] = 'Bus occupancy released is not net service saved. The delayed V operand writes/gathers, final BN/add epilogue and overlap require a connected schedule; do not subtract this bus number from another model.'
    result['submission_admission'] = False
    result['reason'] = 'Exact delayed integer consumer checked; full frame service still uses a scalar reservation construction and an unclosed FP statistics unit. Generic latent retention is given to ordinary too.'
    (HERE/'consumer_lifetime_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result['full_frame_storage'], ensure_ascii=False))


if __name__ == '__main__':
    main()
