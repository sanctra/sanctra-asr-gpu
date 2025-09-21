import array

def rms_level(pcm_bytes: bytes) -> float:
  # 16-bit mono PCM
  if len(pcm_bytes) < 4: return 0.0
  samples = array.array('h', pcm_bytes)
  acc = sum(s*s for s in samples)
  return (acc / max(1, len(samples))) ** 0.5

def should_process_chunk(pcm_bytes: bytes, sample_rate: int) -> bool:
  # crude threshold; tune in prod or plug proper VAD
  return rms_level(pcm_bytes) > 300
