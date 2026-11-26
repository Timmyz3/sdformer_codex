#!/usr/bin/env bash
# Dedicated flow host only. Data preparation; never starts GPU training.
set -euo pipefail
exec 9>/data/flow_a/setup/phase1_pipeline.lock
flock -n 9
task_status=/data/flow_a/setup/phase1_pipeline_status.json
write_status() {
  python3 - "$task_status" "$1" "$$" <<'PY'
import datetime,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);t=p.with_suffix('.tmp')
t.write_text(json.dumps({'status':sys.argv[2],'pid':int(sys.argv[3]),'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu_training_started':False},indent=2)+'\n');t.replace(p)
PY
}
trap 'write_status failed' ERR
write_status transferring
python3 /data/flow_a/setup/transfer_dsec_stream.py \
  --audit /data/flow_a/setup/archive_data_audit.json \
  --manifest /data/flow_a/setup/phase1_deployment_manifest.json \
  --source-csv /data/flow_a/setup/archive_v2_transfer_manifest.csv \
  --source-socket /tmp/flow_archive_direct_20261011.sock \
  --target-local --out-dir /data/flow_a/setup/transfer_direct --run
write_status verifying
/data/flow_a/envs/v2/bin/python /data/flow_a/setup/validate_received_data.py \
  --root /data/flow_a/data/DSEC \
  --manifest /data/flow_a/setup/phase1_deployment_manifest.json \
  --report /data/flow_a/setup/received_fullhash.json \
  --mode fullhash --finite all --threads 2 \
  --reference-hashes /data/flow_a/setup/transfer_direct/source_hashes.csv
write_status verified
