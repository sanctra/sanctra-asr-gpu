from faster_whisper import WhisperModel

def transcribe_pcm(model: WhisperModel, pcm_bytes: bytes, language="en", beam_size=5, temperature=0.1) -> str:
  segments, _ = model.transcribe(audio=pcm_bytes, language=language, beam_size=beam_size, best_of=beam_size, temperature=temperature)
  return "".join(s.text for s in segments).strip()
