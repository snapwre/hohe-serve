#!/bin/bash
# How fast is v0.3 on ordinary processors? This decides whether the demo runs on
# a $60 CPU machine or a $170 GPU one, so it is measured rather than guessed.
#
# Reports real time factor: seconds of compute per second of audio. Under 1.0 is
# faster than real time. Thread counts are swept so a 2 or 4 vCPU instance can be
# read off the same run instead of renting three boxes.
exec > >(tee -a /var/log/bench.log) 2>&1
set -x
# Your bucket: it needs the model under asr/v0.3 and, if you want the
# results uploaded rather than read from the console, write access.
BUCKET=${BUCKET:-your-bucket}
OUT=s3://$BUCKET/asr-eval/e06-v0.3/cpu-bench/$(date -u +%Y%m%dT%H%M%SZ)
kill_self() { aws s3 cp /var/log/bench.log $OUT/bench.log || true; shutdown -h now; }
trap kill_self EXIT

dnf install -y -q python3.11 python3.11-pip gcc-c++ cmake git >/dev/null 2>&1
python3.11 -m venv /opt/v && . /opt/v/bin/activate
pip install -q --upgrade pip
pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu || exit 1
pip install -q "transformers==4.57.6" numpy soundfile boto3 || exit 1
# The language model decoder is optional: it needs a compiler and can fail on a
# bare image. The encoder numbers are the ones that decide the machine.
pip install -q pyctcdecode kenlm || echo "NO_KENLM"

mkdir -p /root/m && cd /root/m
for f in model.safetensors config.json preprocessor_config.json tokenizer_config.json vocab.json special_tokens_map.json added_tokens.json; do
  aws s3 cp s3://$BUCKET/asr/v0.3/$f . || exit 1
done

cat > /root/bench.py <<'PYEOF'
import json, os, time
import numpy as np, torch
from transformers import AutoModelForCTC, AutoProcessor

torch.set_grad_enabled(False)
proc = AutoProcessor.from_pretrained("/root/m")
base = AutoModelForCTC.from_pretrained("/root/m", torch_dtype=torch.float32).eval()

# int8 on the linear layers. This is the cheap quantisation: no calibration data,
# no export, and it is where nearly all of the time in this model goes.
try:
    q = torch.ao.quantization.quantize_dynamic(base, {torch.nn.Linear}, dtype=torch.qint8)
except Exception as exc:
    print("quantise failed:", exc); q = None

def timed(model, seconds, threads, runs=3):
    torch.set_num_threads(threads)
    audio = np.random.default_rng(0).standard_normal(16000 * seconds).astype("float32") * 0.05
    x = proc(audio, sampling_rate=16000, return_tensors="pt")
    key = "input_features" if "input_features" in x else "input_values"
    kw = {"attention_mask": x["attention_mask"]} if "attention_mask" in x else {}
    model(x[key], **kw)                      # warm the allocator, do not time it
    t = []
    for _ in range(runs):
        s = time.perf_counter(); out = model(x[key], **kw); t.append(time.perf_counter() - s)
    best = min(t)
    return {"seconds_audio": seconds, "threads": threads, "compute_s": round(best, 3),
            "rtf": round(best / seconds, 3), "frames": int(out.logits.shape[1])}

rows = []
cores = os.cpu_count() or 8
for threads in sorted({1, 2, 4, cores}):
    for secs in (5, 15, 30):
        for name, m in (("fp32", base), ("int8", q)):
            if m is None: continue
            r = timed(m, secs, threads); r["precision"] = name; rows.append(r)
            print(json.dumps(r), flush=True)
json.dump({"cpu_count": cores, "rows": rows}, open("/root/bench.json", "w"), indent=1)
print("BENCH DONE")
PYEOF

python /root/bench.py
aws s3 cp /root/bench.json $OUT/bench.json || true
echo "RESULTS AT $OUT"
