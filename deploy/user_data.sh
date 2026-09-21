#!/bin/bash
# What a fresh Hohe ASR machine does to itself. No SSH, no image to build.
#
# Everything it needs is in S3: the weights and serve.py. The instance profile
# already grants read on that bucket, so no key travels in this file.
#
# This runs on every replacement the Auto Scaling Group makes, which on spot is
# several times a week. Boot to answering is what matters, so torch comes from
# the CPU wheel index (a tenth of the size of the CUDA one) and the model is
# copied once to local disk.
exec > >(tee -a /var/log/hohe.log) 2>&1
set -x
BUCKET=@@BUCKET@@
MODEL_PREFIX=@@MODEL_PREFIX@@
CODE_KEY=@@CODE_KEY@@

dnf install -y -q python3.11 python3.11-pip
# Amazon Linux ships no ffmpeg at all, in any repository, so a static build is
# kept in our own bucket beside the weights. The first machine learned this the
# hard way: it came up healthy, served /health happily, and returned 500 on
# every clip because ffmpeg was not there.
aws s3 cp "s3://$BUCKET/hohe/ffmpeg" /usr/local/bin/ffmpeg && chmod 755 /usr/local/bin/ffmpeg
/usr/local/bin/ffmpeg -version | head -1 || exit 1
python3.11 -m venv /opt/hohe && . /opt/hohe/bin/activate
pip install -q --upgrade pip
pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -q "numpy<2" "transformers==4.57.6" fastapi==0.115.6 "uvicorn[standard]==0.34.0" \
    python-multipart==0.0.20 soundfile==0.13.1

mkdir -p /model /opt/hohe/app
for f in model.safetensors config.json preprocessor_config.json tokenizer_config.json \
         vocab.json special_tokens_map.json added_tokens.json; do
  aws s3 cp "s3://$BUCKET/$MODEL_PREFIX/$f" /model/ || exit 1
done
aws s3 cp "s3://$BUCKET/$CODE_KEY" /opt/hohe/app/serve.py || exit 1

# The secret is read at boot from SSM, so it is never in this file and never on
# disk in plain form beyond the unit's environment.
SECRET=$(aws ssm get-parameter --name @@SECRET_PARAM@@ --with-decryption --query Parameter.Value --output text 2>/dev/null || echo "")

cat > /etc/systemd/system/hohe.service <<UNIT
[Unit]
Description=Hohe ASR
After=network-online.target

[Service]
Environment=MODEL_DIR=/model
Environment=PORT=8080
Environment=ASR_SECRET=$SECRET
Environment=OMP_NUM_THREADS=@@THREADS@@
Environment=MKL_NUM_THREADS=@@THREADS@@
ExecStart=/opt/hohe/bin/python /opt/hohe/app/serve.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now hohe.service

# The load balancer decides this machine is healthy by asking /health, so say
# clearly in the log when that first succeeds.
for i in $(seq 1 60); do
  sleep 5
  curl -sf -H "x-asr-secret: $SECRET" localhost:8080/health && { echo "HOHE READY"; break; }
done
