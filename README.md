# sanctra-asr-gpu

GPU-accelerated Whisper large-v3 streaming ASR over WebSocket.
- Accepts raw 16 kHz or 48 kHz mono PCM chunks
- Returns partial and final transcripts with word timings
- VAD gating to reduce latency and false positives
- Containerized for GCE GPU nodes (CUDA 12.1 runtime), exposes port 9000

## Local quickstart (GPU host)
docker build -f Dockerfile.gpu -t sanctra-asr-gpu:dev .
docker run --rm --gpus all -p 9000:9000 sanctra-asr-gpu:dev

## Protocol (WebSocket /ws)
- Client sends binary frames containing raw PCM int16 samples (mono).
- Optional small JSON text messages:
  - {"type":"config","rate":16000}
  - {"type":"eof"} to flush final decode
- Server sends JSON texts:
  - {"type":"partial","text":"...","start":s,"end":e}
  - {"type":"final","text":"...","words":[{"w":"hi","s":0.10,"e":0.24},...],"start":s,"end":e}

Note: This reference server buffers audio and decodes periodically. For true low-latency, tune VAD window and decode cadence.
