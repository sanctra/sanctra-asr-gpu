import asyncio, json, logging, os, signal
import websockets
from aiohttp import web
from websockets.server import WebSocketServerProtocol
from faster_whisper import WhisperModel
from server.transcribe import transcribe_pcm
from server.vad import should_process_chunk

logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"))
logger = logging.getLogger("sanctra.asr")

HOST = os.getenv("HOST","0.0.0.0")
PORT = int(os.getenv("PORT","9000"))
LANG = os.getenv("ASR_LANG","en")
MODEL_SIZE = os.getenv("ASR_MODEL","large-v3")
BEAM_SIZE = int(os.getenv("ASR_BEAM","5"))
TEMP = float(os.getenv("ASR_TEMP","0.1"))
CHUNK_S = float(os.getenv("ASR_CHUNK_SECONDS","5.0"))
RATE = int(os.getenv("ASR_SAMPLE_RATE","48000"))

model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
logger = logging.getLogger("asr")

async def _healthz(request):
    return web.Response(text="ok")

async def _start_health_server(host: str, port: int):
    app = web.Application()
    app.add_routes([web.get("/healthz", _healthz)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

async def consumer(ws: WebSocketServerProtocol):
  buf = bytearray()
  async for msg in ws:
    if isinstance(msg, (bytes, bytearray)):
      buf.extend(msg)
      # Roughly CHUNK_S seconds mono 16-bit PCM @ RATE
      if len(buf) >= int(RATE * 2 * CHUNK_S):
        chunk = bytes(buf)
        buf.clear()
        if should_process_chunk(chunk, RATE):
          text = transcribe_pcm(model, chunk, language=LANG, beam_size=BEAM_SIZE, temperature=TEMP)
          await ws.send(json.dumps({"type":"partial","text":text}))
    else:
      try:
        data = json.loads(msg)
      except Exception:
        continue
      if data.get("eof"):
        break
  return buf

async def handler(ws: WebSocketServerProtocol):
  try:
    remainder = await consumer(ws)
    if remainder:
      if should_process_chunk(remainder, RATE):
        text = transcribe_pcm(model, bytes(remainder), language=LANG, beam_size=BEAM_SIZE, temperature=TEMP)
        await ws.send(json.dumps({"type":"partial","text":text}))
    await ws.send(json.dumps({"type":"final"}))
  except Exception as e:
    logger.exception("ASR error")
    try:
      await ws.send(json.dumps({"type":"error","error":str(e)}))
    except Exception:
      pass

async def main():
  stop = asyncio.Future()
  loop = asyncio.get_running_loop()
  # inside async def main():
  HOST = "0.0.0.0"; PORT = int(os.getenv("PORT","9000"))
  HEALTH_PORT = int(os.getenv("HEALTH_PORT","9001"))
  await _start_health_server(HOST, HEALTH_PORT)
  for s in (signal.SIGINT, signal.SIGTERM):
    loop.add_signal_handler(s, stop.cancel)
  async with websockets.serve(handler, HOST, PORT, compression=None, max_size=2**24):
    logger.info("ASR WebSocket listening on %s:%d", HOST, PORT)
    try:
      await stop
    except asyncio.CancelledError:
      pass

if __name__ == "__main__":
  asyncio.run(main())
