"""CosyVoice /tts/stream server.

Wraps CosyVoice2 inference_zero_shot in the wire contract that
visualllm's pipeline expects:

  POST /tts/stream
  Body (JSON): { "text": "...", "voice": "weather", "sample_rate": 24000 }
  Response:    streaming raw 16-bit PCM mono at sample_rate Hz

The "weather" voice is a fixed female Mandarin zero-shot reference
bundled at asset/zero_shot_prompt.wav. To use a different voice,
mount a wav file and set VOICE_REF_WAV + VOICE_PROMPT_TEXT env vars.

Model weights are downloaded from HuggingFace on first startup if not
already present under pretrained_models/. Set HF_MODEL_ID to switch
between CosyVoice2-0.5B (default) and Fun-CosyVoice3-0.5B-2512.

Run:
  python tts_stream_server.py --port 8001
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "Matcha-TTS"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config from env ────────────────────────────────────────────────────────────
MODEL_DIR   = os.getenv("COSYVOICE_MODEL_DIR", str(ROOT / "pretrained_models" / "CosyVoice2-0.5B"))
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "FunAudioLLM/CosyVoice2-0.5B")
VOICE_REF   = os.getenv("VOICE_REF_WAV", str(ROOT / "asset" / "zero_shot_prompt.wav"))
VOICE_TEXT  = os.getenv("VOICE_PROMPT_TEXT", "希望你以后能够做的比我还好呦。")
USE_VLLM    = os.getenv("USE_VLLM", "1").lower() in ("1", "true")
GPU_UTIL    = float(os.getenv("COSYVOICE_VLLM_GPU_UTIL", "0.5"))

app = FastAPI(title="CosyVoice TTS stream server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

cosyvoice = None
prompt_speech_16k = None


def _download_weights() -> None:
    model_path = Path(MODEL_DIR)
    if (model_path / "cosyvoice.yaml").exists():
        logger.info(f"Model weights found at {MODEL_DIR}")
        return
    logger.info(f"Downloading weights from HuggingFace: {HF_MODEL_ID} → {MODEL_DIR}")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=HF_MODEL_ID, local_dir=MODEL_DIR)
    logger.info("Weights downloaded.")


def _load_model() -> None:
    global cosyvoice, prompt_speech_16k

    _download_weights()

    from cosyvoice.cli.cosyvoice import AutoModel
    from cosyvoice.utils.file_utils import load_wav

    logger.info(f"Loading CosyVoice model (vllm={USE_VLLM}, gpu_util={GPU_UTIL}) ...")
    cosyvoice = AutoModel(
        model_dir=MODEL_DIR,
        load_vllm=USE_VLLM,
        load_jit=False,
        load_trt=False,
        fp16=True,
        vllm_gpu_memory_utilization=GPU_UTIL,
    )

    logger.info(f"Loading voice reference: {VOICE_REF}")
    prompt_speech_16k = load_wav(VOICE_REF, 16000)
    logger.info("Model ready.")


# ── Request schema ─────────────────────────────────────────────────────────────
class TTSRequest(BaseModel):
    text: str
    voice: str = "weather"       # only "weather" supported; field kept for API compat
    sample_rate: int = 24000


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_DIR, "ready": cosyvoice is not None}


# ── Main endpoint ──────────────────────────────────────────────────────────────
@app.post("/tts/stream")
async def tts_stream(req: TTSRequest):
    if cosyvoice is None:
        from fastapi import HTTPException
        raise HTTPException(503, "Model not loaded yet")

    target_sr = req.sample_rate  # pipeline requests 24000

    def generate():
        for chunk in cosyvoice.inference_zero_shot(
            req.text,
            VOICE_TEXT,
            prompt_speech_16k,
            stream=True,
        ):
            audio_np = chunk["tts_speech"].numpy()  # float32, native 24 kHz
            # Resample if caller wants a different rate (rarely needed)
            if target_sr != 24000:
                import librosa
                audio_np = librosa.resample(audio_np, orig_sr=24000, target_sr=target_sr)
            pcm = (audio_np * (2 ** 15)).astype(np.int16).tobytes()
            yield pcm

    return StreamingResponse(generate(), media_type="application/octet-stream")


# ── Entrypoint ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8001")))
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    _load_model()
    uvicorn.run(app, host=args.host, port=args.port)
