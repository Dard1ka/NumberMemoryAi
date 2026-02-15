import io
import os
import time
import json
import subprocess
from typing import List, Optional, Tuple

import uvicorn
import torch
import soundfile as sf
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import WhisperProcessor, WhisperForConditionalGeneration

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "./model_output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEBUG_SAVE_WAV = False
DEBUG_DIR = "debug_audio"
os.makedirs(DEBUG_DIR, exist_ok=True)

print(f"🔄 Loading Model from {MODEL_PATH} to {DEVICE}...")

try:
    processor = WhisperProcessor.from_pretrained(MODEL_PATH, language="English", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"⚠️ Error loading fine-tuned model. Fallback to base 'openai/whisper-tiny'. Error: {e}")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="English", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(DEVICE)

model.eval()

# Utilities
def normalize_text(text: str) -> str:
    return text.lower().strip().replace(".", "").replace(",", "").replace("-", " ")

def webm_to_wav_16k_mono(audio_bytes: bytes) -> bytes:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "wav",
        "-ac", "1",
        "-ar", "16000",
        "pipe:1",
    ]
    try:
        p = subprocess.run(cmd, input=audio_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return p.stdout
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffmpeg not found. Install FFmpeg and add it to PATH.")
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="ignore")
        raise HTTPException(status_code=400, detail=f"ffmpeg decode failed: {err}")

def pad_and_normalize(y: np.ndarray, sr: int = 16000, pad_sec: float = 0.35) -> np.ndarray:
    pad = np.zeros(int(sr * pad_sec), dtype=np.float32)
    y = np.concatenate([pad, y.astype(np.float32), pad])

    peak = float(np.max(np.abs(y)) + 1e-9)
    y = y / peak
    y = np.clip(y, -1.0, 1.0)
    return y

def transcribe_16k(y: np.ndarray) -> Tuple[str, str]:
    input_features = processor(y, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

    with torch.inference_mode():
        predicted_ids = model.generate(
            input_features,
            do_sample=False,
            temperature=0.0,
            num_beams=1,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=32,
        )

    raw = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    clean = normalize_text(raw)
    return raw, clean

from typing import Optional

def extract_number_en(clean_text: str) -> Optional[int]:
    if not clean_text:
        return None

    t = clean_text.lower().replace(",", " ").replace("-", " ").strip()
    tokens = [x for x in t.split() if x]

    if tokens == ["one", "hundred"] or tokens == ["a", "hundred"] or tokens == ["hundred"]:
        return 100

    ones = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    }
    teens = {
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    }
    tens = {
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    }

    out = []
    i = 0
    while i < len(tokens):
        w = tokens[i]

        if w in ("and", "the"):
            i += 1
            continue
        
        if w in teens:
            out.append(teens[w])
            i += 1
            continue

        if w in tens:
            base = tens[w]
            nxt = tokens[i + 1] if (i + 1) < len(tokens) else None
            if nxt and nxt in ones:
                out.append(base + ones[nxt])
                i += 2
            else:
                out.append(base)
                i += 1
            continue
        
        if w in ones:
            out.append(ones[w])
            i += 1
            continue

        i += 1

    if len(out) != 1:
        return None

    n = int(out[0])

    if n < 0 or n > 100:
        return None

    return n

def split_by_weights(y: np.ndarray, weights: List[float]) -> List[np.ndarray]:
    if len(weights) == 0:
        return []

    weights = [max(0.01, float(w)) for w in weights]
    total = sum(weights)

    n = len(y)
    cuts = [0]
    acc = 0.0
    for w in weights[:-1]:
        acc += w
        cuts.append(int(round((acc / total) * n)))
    cuts.append(n)

    segments = []
    for i in range(len(weights)):
        a, b = cuts[i], cuts[i + 1]
        seg = y[a:b]
        if len(seg) < 1600:  # <0.1s
            seg = np.pad(seg, (0, 1600 - len(seg)))
        segments.append(seg)
    return segments

# Endpoints
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "cuda": torch.cuda.is_available()}

@app.post("/judge")
async def judge_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    wav_bytes = webm_to_wav_16k_mono(audio_bytes)

    try:
        y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read wav bytes: {str(e)}")

    if y.ndim > 1:
        y = y.mean(axis=1)

    y = pad_and_normalize(y, sr=16000, pad_sec=0.6)

    audio_len_sec = float(len(y) / 16000.0)
    rms = float(np.sqrt(np.mean(np.square(y))) + 1e-12)

    debug_name = None
    if DEBUG_SAVE_WAV:
        debug_name = f"dbg_{int(time.time()*1000)}.wav"
        sf.write(os.path.join(DEBUG_DIR, debug_name), y, 16000)

    raw, clean = transcribe_16k(y)
    return {
        "raw_text": raw,
        "clean_text": clean,
        "audio_len_sec": audio_len_sec,
        "rms": rms,
        "debug_wav_saved": debug_name if DEBUG_SAVE_WAV else None
    }

@app.post("/judge_level")
async def judge_level(
    file: UploadFile = File(...),
    weights: str = Form(...),          
    expected_len: str = Form(...)      
):
    try:
        w = json.loads(weights)
        if not isinstance(w, list) or any(not isinstance(x, (int, float)) for x in w):
            raise ValueError("weights must be JSON list of numbers")
        exp_len = int(expected_len)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad form fields: {str(e)}")

    if exp_len <= 0 or exp_len > 12:
        raise HTTPException(status_code=400, detail="expected_len out of range (1..12)")

    if len(w) != exp_len:
        raise HTTPException(status_code=400, detail=f"weights length ({len(w)}) != expected_len ({exp_len})")

    audio_bytes = await file.read()
    wav_bytes = webm_to_wav_16k_mono(audio_bytes)

    try:
        y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read wav bytes: {str(e)}")

    if y.ndim > 1:
        y = y.mean(axis=1)

    y = pad_and_normalize(y, sr=16000, pad_sec=0.25)

    audio_len_sec = float(len(y) / 16000.0)
    rms = float(np.sqrt(np.mean(np.square(y))) + 1e-12)

    segments = split_by_weights(y, w)

    results = []
    nums: List[Optional[int]] = []

    # transcribe per segment
    for idx, seg in enumerate(segments):
        # per-seg pad dikit biar ga “kepotong”
        seg = pad_and_normalize(seg, sr=16000, pad_sec=0.20)

        if DEBUG_SAVE_WAV:
            name = f"seg_{int(time.time()*1000)}_{idx+1}.wav"
            sf.write(os.path.join(DEBUG_DIR, name), seg, 16000)

        raw, clean = transcribe_16k(seg)
        n = extract_number_en(clean)

        results.append({
            "idx": idx,
            "raw_text": raw,
            "clean_text": clean,
            "parsed_num": n
        })
        nums.append(n)

    return {
        "mode": "level_split_transcribe",
        "expected_len": exp_len,
        "audio_len_sec": audio_len_sec,
        "rms": rms,
        "nums": nums,
        "segments": results
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
