FROM python:3.11-slim
WORKDIR /app
ENV PROMETHEUS_MULTIPROC_DIR=/tmp/prom
RUN mkdir -p /tmp/prom
COPY agent/requirements.txt .
COPY shared ./shared
ENV FASTEMBED_CACHE_PATH=/models/fastembed
RUN pip install --no-cache-dir -r requirements.txt ./shared
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8')"
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /app/models && cd /app/models \
 && for v in amy/low/en_US-amy-low ryan/low/en_US-ryan-low lessac/low/en_US-lessac-low; do \
      curl -fsSLO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/$v.onnx \
   && curl -fsSLO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/$v.onnx.json; done
ENV VOICE_DIR=/app/models
RUN python -W ignore -c "from livekit.plugins.turn_detector.english import _EUORunnerEn; _EUORunnerEn._download_files()"
RUN python -c "from voxa_shared.kb.embed import Embedder, Reranker; Embedder(); Reranker()"
COPY agent/voxa_agent ./voxa_agent
CMD ["python", "-m", "voxa_agent.worker", "start"]
