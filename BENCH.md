# How fast is Hohe ASR without a GPU

Measured 20 September 2026 on a c7i.2xlarge in eu-central-1, on the v0.3
weights. Real time factor is seconds of compute per second of audio, so 0.25
means a ten second clip takes two and a half seconds. Lower is better and
anything under 1.0 is faster than real time.

| Audio | 1 thread | 2 threads | 4 threads | 8 threads |
|---|---:|---:|---:|---:|
| 5 s, fp32 | 0.495 | 0.277 | 0.146 | 0.163 |
| 5 s, **int8** | 0.231 | 0.135 | **0.078** | 0.084 |
| 15 s, fp32 | 0.716 | 0.376 | 0.219 | 0.199 |
| 15 s, **int8** | 0.477 | 0.260 | **0.155** | 0.139 |
| 30 s, fp32 | 1.014 | 0.560 | 0.319 | 0.278 |
| 30 s, **int8** | 0.793 | 0.447 | **0.258** | 0.220 |

## What it decided

**No GPU.** On four ordinary cores with int8, a five second voice note is
transcribed in four tenths of a second and a thirty second one in under eight.
A GPU would cost roughly three times as much to sit idle between clicks.

**Four cores, not eight.** Eight threads is barely quicker than four and on
short clips it is slower, because the work per thread stops covering the cost
of splitting it. `c7i.xlarge` and its neighbours are the size to rent, at
around $0.09 to $0.10 an hour on spot, which is about $70 a month running
continuously, before the load balancer.

**int8 is worth having.** Between a quarter and a half of the time disappears,
for no export step and no calibration data, so the served weights cannot drift
from the evaluated ones.

**Cut long audio into pieces.** Cost grows faster than length: at four threads,
thirty seconds costs 0.258 against 0.078 for five. Attention is quadratic, so
three ten second pieces genuinely cost less than one thirty second pass. The
chunking in `serve.py` was written to make the text appear while somebody
watches, and it turns out to make the service cheaper as well.

## Reproducing it

The sweep script is `deploy/bench_userdata.sh`. One note if you run it again:
the instance profile could **read** the model bucket but not
write to it, so the box's attempt to upload its results failed silently and the
numbers above were recovered from the EC2 console log. Either read them from
the console again or add a write grant first.
