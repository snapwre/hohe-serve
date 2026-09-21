# hohe-serve

Hohe ASR as a service. Audio in, Amharic out, a piece at a time.

```
POST /transcribe   multipart, field "audio", any format ffmpeg reads
  -> newline delimited JSON:
     {"partial": "ሰላም እን"}
     {"partial": "ሰላም እንደምን ናችሁ"}
     {"final": "ሰላም እንደምን ናችሁ", "seconds": 4.2, "compute_ms": 900}

GET /health        open, because the load balancer asks it and cannot send a header
```

## Streaming, without pretending

This is a CTC model. It reads a piece of audio in one pass instead of
generating word by word, so there is no honest way to emit a letter at a time.
What it does instead is cut the recording at its quietest instants into pieces
of about ten seconds, transcribe them in order, and send the transcript so far
after each one.

Two reasons that is the right design rather than a compromise:

**The cuts land in silence, not mid word**, so nothing is mangled at the seams
and no overlapping windows have to be stitched back together. The method is the
one the TikTok segmenter arrived at: do not look for a threshold, because
processed audio has no true silence, take the quietest instant near where the
cut wants to be.

**Cost grows faster than length.** Attention is quadratic, so one thirty second
pass costs far more than three ten second ones. Chunking makes the service
faster as well as more responsive.

## Run it yourself

Nothing here needs our account. One command, no GPU:

```
docker build -t hohe-asr .
docker run --rm -p 8080:8080 hohe-asr
```

It fetches the weights from Hugging Face on first start, loads them onto the
processor and serves on port 8080. Give it four cores if you can: eight is no
faster, two is about half the speed.

```
curl -F audio=@clip.ogg http://localhost:8080/transcribe
```

Any format ffmpeg reads works, including the .ogg a phone records and the .m4a
an iPhone produces. Set `ASR_SECRET` and send it as an `x-asr-secret` header to
require one; leave it unset and the service is open, which is right on a laptop
and wrong on the internet.

To serve your own fine-tune, mount it: `-v /path/to/model:/model`.

## Which machine

Measured, not guessed. The sweep is in
the bucket under `asr-eval/cpu-bench/`, and it times the model
at several clip lengths and thread counts, in fp32 and int8, so a smaller
instance can be read off the same run.

int8 dynamic quantisation on the linear layers is on by default on CPU. It
needs no calibration data and no export step, so it cannot drift away from the
weights that were evaluated. Set `INT8=0` to turn it off.

The service uses a GPU if it finds one and expects not to.

## Deploying it

```
deploy/launch.py --up        # create or update everything, print the address
deploy/launch.py --status
deploy/launch.py --restart   # after new code or new weights
deploy/launch.py --down
```

**Spot, in an Auto Scaling Group.** Amazon takes the machine back with two
minutes of notice and the group starts another. That group is the watcher;
we write none. An interruption costs a few minutes, which is fine because the
bot queues clips and delivers them afterwards rather than failing.

**Internal load balancer.** The only client is the bot on the app server in the
same VPC, so nothing is exposed to the internet, there is no certificate to
issue and no DNS to own, and audio never leaves Amazon's private network. The
bot gets one stable address across every spot replacement, which is the point
of the balancer.

**No key in the user data.** The instance profile named in `deploy/config.json`
reads the bucket the weights live in, and the shared secret is pulled from SSM
at boot.

`serve.py` travels to the machine through S3 along with the weights, so a
replacement always starts from exactly what was last deployed. Change the code,
then `--up` to upload and `--restart` to roll it.

The browser demo later needs a public listener and a certificate. That is an
addition to this, not a different design.
