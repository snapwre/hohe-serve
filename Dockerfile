FROM python:3.12-slim

# ffmpeg decodes whatever Telegram sends: opus voice notes, m4a, mp4 video
# notes. It is the one system dependency this service has.
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "huggingface_hub>=0.34,<1.0"

COPY serve.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

# MODEL_ID is fetched into MODEL_DIR at start unless something is already
# there, so `docker run` needs no volume, no credentials and no prior steps.
# Mount your own weights at /model to serve those instead.
ENV PYTHONUNBUFFERED=1 MODEL_DIR=/model MODEL_ID=snapwre/hohe-asr-amharic
EXPOSE 8080
CMD ["./entrypoint.sh"]
