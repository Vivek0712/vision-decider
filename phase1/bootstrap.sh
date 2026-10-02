#!/bin/bash
# EC2 user-data for the Phase 1 baseline run. Installs the stack, runs phase1.run,
# streams the log to S3 every 2 minutes, uploads results, then powers off. The
# instance is launched with shutdown behaviour "terminate", so power-off deletes it.
# A hard cap (shutdown +MAX_MIN) stops a stuck run from billing forever.
set -euxo pipefail
BUCKET=__BUCKET__
RUN=__RUN__
MAX_MIN=__MAX_MIN__
ARGS="__ARGS__"
UPSTREAM_SHA=aa92b07

shutdown -h +$MAX_MIN || true
mkdir -p /opt/run && exec > >(tee -a /opt/run/log.txt) 2>&1
export HOME=/root HF_HOME=/opt/hf DEBIAN_FRONTEND=noninteractive

apt-get update -q && apt-get install -yq python3-venv python3-pip git unzip curl
curl -sSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscli.zip
unzip -q /tmp/awscli.zip -d /tmp && /tmp/aws/install
( while true; do aws s3 cp /opt/run/log.txt s3://$BUCKET/runs/$RUN/log.txt --only-show-errors || true; sleep 120; done ) &

finish() {
  status=$?
  echo "exit status $status" | tee /opt/run/STATUS
  aws s3 sync /opt/run s3://$BUCKET/runs/$RUN/ --only-show-errors || true
  shutdown -h now
}
trap finish EXIT

python3 -m venv /opt/vd && . /opt/vd/bin/activate
pip install -q --upgrade pip
# Pinned to the environment the CPU test suite passed in. torchvision is required:
# the Qwen3.5 processor also loads its video processor.
pip install -q "torch==2.14.1" "torchvision==0.29.1" --index-url https://download.pytorch.org/whl/cpu
pip install -q "transformers==5.18.0" "peft==0.21.2" "accelerate==1.15.0" "huggingface_hub==1.33.0" \
  "tokenizers==0.23.2" jinja2 pillow pyyaml fastapi uvicorn safetensors pandas pyarrow
git clone -q https://github.com/strands-labs/strands-decider /opt/strands-decider
git -C /opt/strands-decider checkout -q $UPSTREAM_SHA
pip install -q -e /opt/strands-decider
aws s3 cp s3://$BUCKET/runs/$RUN/code.tar.gz /tmp/code.tar.gz --only-show-errors
mkdir -p /opt/vision-decider && tar -xzf /tmp/code.tar.gz -C /opt/vision-decider
pip install -q -e /opt/vision-decider
python -c "import torch, transformers; print('torch', torch.__version__, 'threads', torch.get_num_threads(), 'transformers', transformers.__version__)"
lscpu | grep -E 'Model name|^CPU\(s\)|amx' | head -5 || true

# Prebuilt Image JevBench preview items (built once locally so renders are identical).
aws s3 sync s3://$BUCKET/data/ijb_preview /opt/ijb --only-show-errors || true
cd /opt/vision-decider
# Preflight: every system on a handful of items, so a load or format error stops the
# run in minutes instead of after hours.
python -m phase1.run --out /opt/run/preflight --nb-groups 1 --pope 2
python -m phase1.run --out /opt/run/results $ARGS
