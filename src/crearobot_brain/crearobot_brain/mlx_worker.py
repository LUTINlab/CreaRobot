import sys
import json
import struct
import numpy as np
import mlx_whisper

MODEL = "mlx-community/whisper-small-mlx"

inp = sys.stdin.buffer
out = sys.stdout.buffer
sys.stdout = sys.stderr  # tout print parasite part sur stderr, stdout reste réservé au protocole


def read_exact(n):
    buf = b""
    while len(buf) < n:
        chunk = inp.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


# Préchauffage (charge le modèle, compile les kernels Metal)
mlx_whisper.transcribe(np.zeros(16000, dtype=np.float32), path_or_hf_repo=MODEL, language="fr")
out.write(b'{"ready": true}\n')
out.flush()

while True:
    header = read_exact(4)
    if header is None:
        break
    (n,) = struct.unpack("<I", header)
    data = read_exact(n * 4)
    if data is None:
        break
    audio = np.frombuffer(data, dtype=np.float32)
    try:
        r = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=MODEL,
            language="fr",
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )
        resp = {"text": r["text"], "segments": [{"end": s["end"]} for s in r.get("segments", [])]}
    except Exception as e:
        resp = {"error": str(e)}
    out.write((json.dumps(resp) + "\n").encode())
    out.flush()
