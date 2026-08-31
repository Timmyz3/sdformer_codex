# M501 H67 exact adjacent-overlap fast-kill

Status: `PASS_EXACT_OPPORTUNITY_AUDIT_NO_RTL_ADMISSION`

| Cohort | Axis | Group | Event reduction | Redundant fraction |
|---|---|---:|---:|---:|
| validation_s10 | horizontal | 2 | 1.379629x | 27.516729% |
| validation_s10 | horizontal | 4 | 1.209864x | 17.346085% |
| validation_s10 | horizontal | 8 | 1.045126x | 4.317791% |
| validation_s10 | vertical | 2 | 1.341175x | 25.438498% |
| validation_s10 | vertical | 4 | 1.166157x | 14.248234% |
| validation_s10 | vertical | 8 | 1.044996x | 4.305839% |
| train_calibration_s32 | horizontal | 2 | 1.390349x | 28.075588% |
| train_calibration_s32 | horizontal | 4 | 1.219527x | 18.001026% |
| train_calibration_s32 | horizontal | 8 | 1.051638x | 4.910262% |
| train_calibration_s32 | vertical | 2 | 1.364901x | 26.734634% |
| train_calibration_s32 | vertical | 4 | 1.179093x | 15.189054% |
| train_calibration_s32 | vertical | 8 | 1.039424x | 3.792898% |

Selected validation horizontal G2 event reduction: `1.379629x`.
Ideal four-Conv envelope sensitivity only: `1.036618x`.
Opportunity gate: `True`; next action: `ALLOW_SAME_RESOURCE_CYCLE_FASTKILL_ONLY`.
Selected overlap scratch proxy: `131328 bit` (`16.03125 KiB`, Frozen H67 signed19 resident-psum arithmetic proxy; not an ExSpike-published width and not a generated SRAM macro.).
All frozen records contain only zero plus one operator-constant positive amplitude;
therefore exact-value overlap equals support intersection here and does not activate
a general signed-analog novelty delta.

`new_rtl_admitted=false`: this is exact event-work opportunity, not a same-resource cycle,
energy, PPA, full-network, or system-speedup result. ExSpike APEC is direct prior art.
