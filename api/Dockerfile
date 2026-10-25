FROM python:3.11-slim
WORKDIR /app
ENV FASTEMBED_CACHE_PATH=/models/fastembed PYTHONUNBUFFERED=1
COPY api/requirements.txt api-requirements.txt
COPY shared ./shared
RUN pip install --no-cache-dir -r api-requirements.txt ./shared \
 && python -c "from voxa_shared.kb.embed import Embedder, Reranker; Embedder(); Reranker()"
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /app/models && cd /app/models \
 && for v in amy/low/en_US-amy-low ryan/low/en_US-ryan-low lessac/low/en_US-lessac-low; do \
      curl -fsSLO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/$v.onnx \
   && curl -fsSLO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/$v.onnx.json; done
ENV VOICE_DIR=/app/models
COPY api/voxa_api ./voxa_api
COPY samples ./samples
CMD ["uvicorn", "voxa_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
