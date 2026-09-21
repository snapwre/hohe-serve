"""Hohe ASR as a service: audio in, Amharic out, a piece at a time.

One endpoint that matters.

    POST /transcribe   multipart, field "audio", any format ffmpeg reads
      -> newline delimited JSON, one object per line:
         {"partial": "..."} while it works, then
         {"final": "...", "seconds": 12.3, "compute_ms": 4100}

Streaming without pretending. This is a CTC model, so it reads a whole piece of
audio in one pass rather than generating word by word, and there is no honest
way to emit a letter at a time. What it does instead is cut the recording at
its quietest moments into pieces of about ten seconds, transcribe them in
order, and send the transcript so far after each one. On a short voice note
that is one piece and the text arrives at once. On a two minute one the text
grows in front of the person every few seconds, which is what we wanted, and
because the cuts land in silence rather than mid word, nothing is mangled at
the seams and no overlap has to be stitched back together.

Runs on CPU by default and uses the GPU when there is one. Which machine this
deserves is a measured question, not a taste one: see deploy/README.md.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
from typing import AsyncIterator

import numpy as np
import torch
from fastapi import FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("serve")

MODEL_DIR = os.environ.get("MODEL_DIR", "/model")
SECRET = os.environ.get("ASR_SECRET", "").strip()
MAX_SECONDS = float(os.environ.get("MAX_SECONDS", "180"))
MAX_BYTES = int(os.environ.get("MAX_BYTES", str(25 * 1024 * 1024)))
#: About ten seconds a piece. Long enough that the model has context to work
#: with, short enough that a person watching sees movement.
CHUNK_SECONDS = float(os.environ.get("CHUNK_SECONDS", "10"))
#: How far from the ideal cut we will wander to find a quiet instant.
CHUNK_SLACK = float(os.environ.get("CHUNK_SLACK", "2.5"))
SR = 16000
#: One transcription at a time. Two clips sharing a small machine make both
#: slow and neither correct; the queue is in front, where it belongs.
_lock = asyncio.Lock()

app = FastAPI(title="Hohe ASR")
_model = None
_proc = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def load() -> None:
    global _model, _proc
    from transformers import AutoModelForCTC, AutoProcessor
    t = time.time()
    _proc = AutoProcessor.from_pretrained(MODEL_DIR)
    dtype = torch.float16 if _device == "cuda" else torch.float32
    m = AutoModelForCTC.from_pretrained(MODEL_DIR, torch_dtype=dtype).to(_device).eval()
    if _device == "cpu" and os.environ.get("INT8", "1") != "0":
        # int8 on the linear layers, where nearly all the time goes. No
        # calibration data and no export step, so it cannot drift from the
        # weights that were evaluated.
        try:
            m = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
            log.info("quantised to int8 for cpu")
        except Exception:  # noqa: BLE001
            log.exception("int8 quantisation failed; staying at fp32")
    _model = m
    log.info("model ready on %s in %.1fs", _device, time.time() - t)


def decode_audio(raw: bytes) -> np.ndarray:
    """Whatever arrived into 16 kHz mono float32. ffmpeg knows every format.

    Through a real file, not a pipe. An .ogg voice note streams happily from
    stdin, but mp4 and m4a keep their index in a place ffmpeg has to seek to,
    and a pipe cannot seek: it fails with "partial file" on a file that is
    perfectly intact. Every audio file somebody forwards rather than records
    is one of those, which is how this reached production and failed one clip
    in twenty before anybody noticed.
    """
    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        f.write(raw)
        f.flush()
        p = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", f.name, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
            capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise HTTPException(415, f"could not read that audio: {p.stderr.decode()[:200]}")
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def cut_points(audio: np.ndarray) -> list[tuple[int, int]]:
    """Pieces of about CHUNK_SECONDS, each ending at a quiet instant.

    Loud, processed audio has no true silence, so this does not look for a
    threshold. It takes the quietest window near where the cut wants to be,
    which is the same idea the TikTok segmenter arrived at after a threshold
    found no pauses at all in loudness-processed video.
    """
    n = len(audio)
    step, slack = int(CHUNK_SECONDS * SR), int(CHUNK_SLACK * SR)
    if n <= (CHUNK_SECONDS + CHUNK_SLACK) * SR:
        return [(0, n)]
    win = int(0.05 * SR)
    energy = np.abs(audio)
    pieces, start = [], 0
    while n - start > (CHUNK_SECONDS + CHUNK_SLACK) * SR:
        ideal = start + step
        lo, hi = max(start + win, ideal - slack), min(n - win, ideal + slack)
        window = energy[lo:hi]
        # Mean over a short window rather than a single sample: one quiet
        # sample happens inside every vowel.
        smooth = np.convolve(window, np.ones(win) / win, mode="same") if len(window) > win else window
        cut = lo + int(np.argmin(smooth)) if len(smooth) else ideal
        pieces.append((start, cut))
        start = cut
    pieces.append((start, n))
    return pieces


@torch.inference_mode()
def run_piece(audio: np.ndarray) -> str:
    x = _proc(audio, sampling_rate=SR, return_tensors="pt")
    key = "input_features" if "input_features" in x else "input_values"
    kw = {"attention_mask": x["attention_mask"].to(_device)} if "attention_mask" in x else {}
    dtype = torch.float16 if _device == "cuda" else torch.float32
    logits = _model(x[key].to(_device, dtype), **kw).logits
    text = _proc.batch_decode(logits.argmax(-1).cpu().numpy())[0]
    # The model writes a language tag at the start of its output. Nobody wants
    # to read it and every downstream user would have to strip it themselves.
    return text.replace("[AMH]", "").strip()


async def transcribe_stream(audio: np.ndarray) -> AsyncIterator[bytes]:
    started = time.monotonic()
    seconds = len(audio) / SR
    said: list[str] = []
    for a, b in cut_points(audio):
        piece = await asyncio.to_thread(run_piece, audio[a:b])
        if piece:
            said.append(piece)
        yield json.dumps({"partial": " ".join(said)}, ensure_ascii=False).encode() + b"\n"
    yield json.dumps({"final": " ".join(said), "seconds": round(seconds, 2),
                      "compute_ms": int((time.monotonic() - started) * 1000)},
                     ensure_ascii=False).encode() + b"\n"


def check(secret: str | None) -> None:
    if SECRET and secret != SECRET:
        raise HTTPException(403, "no")


@app.post("/transcribe")
async def transcribe(audio: UploadFile, x_asr_secret: str | None = Header(default=None)):
    check(x_asr_secret)
    raw = await audio.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "too big")
    samples = decode_audio(raw)
    if len(samples) / SR > MAX_SECONDS:
        raise HTTPException(422, f"longer than {MAX_SECONDS:.0f} seconds")
    if len(samples) < SR * 0.2:
        raise HTTPException(422, "too short to be speech")

    async def body():
        # Held for the whole response: one clip at a time on this machine.
        async with _lock:
            async for line in transcribe_stream(samples):
                yield line

    return StreamingResponse(body(), media_type="application/x-ndjson")


@app.get("/health")
async def health():
    """Open on purpose: the load balancer asks this and cannot send a header.

    It gives away nothing except that a model is loaded, and the security group
    already means only the bot can reach this port at all.
    """
    if _model is None:
        raise HTTPException(503, "still loading")
    return {"ok": True, "device": _device, "busy": _lock.locked(),
            "model": os.environ.get("MODEL_NAME", "hohe-asr v1.0")}


@app.on_event("startup")
async def startup() -> None:
    await asyncio.to_thread(load)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), log_level="info")
