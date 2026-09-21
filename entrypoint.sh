#!/bin/sh
# Fetch the weights once, then serve.
#
# Downloading at start rather than baking the model into the image keeps the
# image small, and lets somebody serve their own fine-tune by mounting it at
# /model, which is checked first.
set -e
if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
  echo "fetching $MODEL_ID into $MODEL_DIR"
  python -c "import os; from huggingface_hub import snapshot_download; \
snapshot_download(os.environ['MODEL_ID'], local_dir=os.environ['MODEL_DIR'], \
allow_patterns=['*.json', '*.safetensors', '*.txt'])"
fi
exec python serve.py
