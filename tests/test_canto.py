"""
Comprehensive Test Suite for Canto Speech Model Clone
Verifies:
1. Acoustic front-end (AGC, noise suppression, VAD)
2. ASR engine & rollout generation
3. GRPO reward scoring & relative advantage math
4. Zero-edit cleaner (self-corrections, fillers, stutters, formatting)
5. End-to-end pipeline execution
6. FastAPI endpoints
"""

import os
import pytest
import numpy as np
from fastapi.testclient import TestClient

from canto_acoustic import CantoAcousticFilter
from canto_asr import CantoASREngine
from canto_grpo import CantoGRPOScorer
from canto_cleaner import CantoZeroEditCleaner
from canto_engine import CantoPipeline
from canto_server import app


# -------------------------------------------------------------
# 1. Acoustic Front-End Tests
# -------------------------------------------------------------
def test_acoustic_filter_gain_control():
    filt = CantoAcousticFilter(enable_whisper_boost=True)
    # Generate very soft signal (simulating whisper)
    t = np.linspace(0, 1, 16000)
    soft_audio = 0.005 * np.sin(2 * np.pi * 300 * t)

    processed, meta = filt.process_audio(soft_audio, sample_rate=16000, is_whisper_hint=True)
    assert meta["boost_applied_db"] > 10.0
    assert np.max(np.abs(processed)) > np.max(np.abs(soft_audio))


def test_acoustic_filter_noise_suppression():
    filt = CantoAcousticFilter(enable_noise_suppression=True)
    t = np.linspace(0, 1, 16000)
    # Tone + stationary background noise
    audio = 0.2 * np.sin(2 * np.pi * 400 * t) + 0.05 * np.random.randn(len(t))

    processed, meta = filt.process_audio(audio, sample_rate=16000)
    assert meta["noise_reduction_applied"] is True
    assert len(processed) > 0


# -------------------------------------------------------------
# 2. GRPO Scorer & Advantage Tests
# -------------------------------------------------------------
def test_grpo_reward_filler_penalty():
    scorer = CantoGRPOScorer()
    clean_rollout = {"text": "Please deploy the application to Kubernetes.", "avg_logprob": -0.2}
    filler_rollout = {"text": "Um like please deploy the application you know to Kubernetes.", "avg_logprob": -0.2}

    r_clean = scorer.compute_reward(clean_rollout)
    r_filler = scorer.compute_reward(filler_rollout)

    assert r_clean["r_total"] > r_filler["r_total"]
    assert r_filler["r_filler"] < 0.0


def test_grpo_relative_advantage_math():
    scorer = CantoGRPOScorer()
    rollouts = [
        {"text": "Um please cancel the meeting.", "avg_logprob": -0.6},
        {"text": "Please cancel the meeting.", "avg_logprob": -0.2},
        {"text": "Cancel meeting.", "avg_logprob": -0.3},
        {"text": "Like cancel meeting maybe.", "avg_logprob": -0.7},
    ]
    best_candidate, scored = scorer.evaluate_group(rollouts)

    # Check advantages sum to approximately zero
    advantages = [r["advantage"] for r in scored]
    assert abs(sum(advantages)) < 1e-3

    # Best candidate must have highest advantage
    assert best_candidate["advantage"] == max(advantages)
    assert best_candidate["text"] == "Please cancel the meeting."


# -------------------------------------------------------------
# 3. Zero-Edit Cleaner & Disfluency Tests
# -------------------------------------------------------------
def test_cleaner_filler_removal():
    cleaner = CantoZeroEditCleaner()
    raw = "Um, so basically we need to update the website, you know."
    cleaned, meta = cleaner.clean_transcript(raw)

    assert "um" not in cleaned.lower()
    assert "basically" not in cleaned.lower()
    assert meta["fillers_removed"] >= 2
    assert cleaned.startswith("We need to update")


def test_cleaner_self_correction_resolution():
    cleaner = CantoZeroEditCleaner()
    # Test mid-sentence preposition correction
    raw1 = "Please send the invoice by Thursday, actually make that Friday morning."
    cleaned1, meta1 = cleaner.clean_transcript(raw1)
    assert "Friday morning" in cleaned1
    assert "Thursday" not in cleaned1
    assert meta1["corrections_resolved"] >= 1

    # Test sentence boundary retraction
    raw2 = "Check the staging logs, actually scratch that, check the database connection."
    cleaned2, meta2 = cleaner.clean_transcript(raw2)
    assert "Check the database connection" in cleaned2
    assert "staging logs" not in cleaned2


def test_cleaner_stutter_deduplication():
    cleaner = CantoZeroEditCleaner()
    raw = "I I want to test the the system."
    cleaned, meta = cleaner.clean_transcript(raw)
    assert "I I" not in cleaned
    assert "the the" not in cleaned
    assert cleaned == "I want to test the system."
    assert meta["stutters_removed"] >= 2


def test_canto_phonetic_and_brand_recognition():
    cleaner = CantoZeroEditCleaner()
    input_text = "Hello, I am currently testing a AI model from a company named Fisper Flu and the name of the model is Kanto."
    cleaned, meta = cleaner.clean_transcript(input_text)

    # Must resolve Fisper Flu -> Wispr Flow
    assert "Wispr Flow" in cleaned
    assert "Fisper Flu" not in cleaned

    # Must resolve Kanto -> Canto
    assert "Canto" in cleaned
    assert "Kanto" not in cleaned

    # Must correct grammar 'a AI' -> 'an AI'
    assert "an AI model" in cleaned


def test_cleaner_tone_formatting():
    cleaner = CantoZeroEditCleaner()
    raw = "We should update the auth token. We must also run the migrations."

    # Formal tone
    cleaned_formal, _ = cleaner.clean_transcript("I can't access it yeah.", tone="formal")
    assert "cannot" in cleaned_formal
    assert "yes" in cleaned_formal

    # Bullet points tone
    cleaned_bullets, _ = cleaner.clean_transcript(raw, tone="bullet points")
    assert cleaned_bullets.startswith("•")


# -------------------------------------------------------------
# 4. End-to-End Pipeline Tests
# -------------------------------------------------------------
def test_pipeline_execution():
    pipeline = CantoPipeline(model_size="tiny.en")
    t = np.linspace(0, 1.0, 16000)
    audio = 0.1 * np.sin(2 * np.pi * 350 * t)

    res = pipeline.transcribe_numpy(audio, sample_rate=16000, rollout_count=2)
    assert "latencies" in res
    assert "total_ms" in res["latencies"]
    assert "metrics" in res
    assert res["latencies"]["total_ms"] > 0.0


# -------------------------------------------------------------
# 5. FastAPI Endpoints Tests
# -------------------------------------------------------------
def test_api_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "memory_rss_mb" in data
    assert data["cost"] == "$0.00"


def test_api_json_transcribe():
    client = TestClient(app)
    # Generate 1 sec empty audio base64
    audio = np.zeros(8000, dtype=np.int16)
    import io, soundfile as sf, base64
    bio = io.BytesIO()
    sf.write(bio, audio, 8000, format="WAV")
    b64_audio = base64.b64encode(bio.getvalue()).decode("utf-8")

    response = client.post("/transcribe/json", json={
        "audio_base64": b64_audio,
        "sample_rate": 8000,
        "rollout_count": 2,
    })
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "latencies" in data
