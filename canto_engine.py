"""
Unified Canto Speech Recognition Engine
Integrates:
1. Acoustic Conditioning (Noise suppression, Whisper AGC, VAD)
2. Ultra-Fast CPU ASR (CTranslate2 int8)
3. GRPO Reinforcement Learning Rollout Scorer & Advantage Selection
4. Zero-Edit Disfluency & Self-Correction Cleaner
100% Local, Zero Cost, Optimized for Intel i5-4570 CPU.
"""

import time
import io
import soundfile as sf
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from canto_acoustic import CantoAcousticFilter
from canto_asr import CantoASREngine
from canto_grpo import CantoGRPOScorer
from canto_cleaner import CantoZeroEditCleaner


class CantoPipeline:
    """
    End-to-end execution pipeline for Canto model clone.
    """

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        custom_vocabulary: Optional[List[str]] = None,
        snippets: Optional[Dict[str, str]] = None,
    ):
        print(f"[Canto] Initializing acoustic front-end...")
        self.acoustic_filter = CantoAcousticFilter(
            target_sample_rate=16000,
            enable_noise_suppression=True,
            enable_whisper_boost=True,
        )

        print(f"[Canto] Initializing ASR core ({model_size}, {compute_type} on {device})...")
        self.asr_engine = CantoASREngine(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
        )

        print(f"[Canto] Initializing GRPO alignment scorer...")
        self.grpo_scorer = CantoGRPOScorer(
            custom_vocabulary=custom_vocabulary,
        )

        print(f"[Canto] Initializing zero-edit formatting engine...")
        dict_map = {w.lower(): w for w in (custom_vocabulary or [])}
        self.cleaner = CantoZeroEditCleaner(
            custom_dictionary=dict_map if dict_map else None,
            snippets=snippets,
        )
        self.initial_prompt = "Wispr Flow, Canto speech model, AI transcription, dictation."
        print(f"[Canto] Pipeline ready for real-time dictation.")

    def decode_audio_bytes(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """
        Decodes arbitrary audio bytes (WebM, Opus, Ogg, MP3, WAV, AAC, M4A)
        into normalized mono float32 numpy array.
        """
        if not audio_bytes:
            return np.zeros(0, dtype=np.float32), 16000

        # 1. Try soundfile (fast for WAV/FLAC/OGG)
        try:
            audio_io = io.BytesIO(audio_bytes)
            data, sr = sf.read(audio_io)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            return data.astype(np.float32), sr
        except Exception:
            pass

        # 2. Try PyAV (handles WebM, Opus, MP4, AAC from browser MediaRecorder)
        try:
            import av
            container = av.open(io.BytesIO(audio_bytes))
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                return np.zeros(0, dtype=np.float32), 16000

            resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
            frames = []
            for frame in container.decode(stream):
                for resampled_frame in resampler.resample(frame):
                    frames.append(resampled_frame.to_ndarray())

            if not frames:
                return np.zeros(0, dtype=np.float32), 16000

            audio_data = np.concatenate(frames, axis=1).flatten().astype(np.float32)
            return audio_data, 16000
        except Exception as e:
            raise ValueError(f"Could not decode audio stream with SoundFile or PyAV: {e}")

    def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        tone: str = "auto",
        enable_grpo: bool = True,
        rollout_count: int = 4,
        is_whisper: bool = False,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribes raw audio bytes (WebM, WAV, OGG, MP3, etc.)."""
        audio_data, sample_rate = self.decode_audio_bytes(audio_bytes)
        return self.transcribe_numpy(
            audio_data=audio_data,
            sample_rate=sample_rate,
            tone=tone,
            enable_grpo=enable_grpo,
            rollout_count=rollout_count,
            is_whisper=is_whisper,
            ground_truth=ground_truth,
        )

    def transcribe_numpy(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        tone: str = "auto",
        enable_grpo: bool = True,
        rollout_count: int = 4,
        is_whisper: bool = False,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs the complete Canto processing sequence.
        """
        total_start = time.perf_counter()
        latencies = {}

        # 1. Acoustic Conditioning
        t0 = time.perf_counter()
        processed_audio, acoustic_meta = self.acoustic_filter.process_audio(
            audio_data=audio_data,
            sample_rate=sample_rate,
            is_whisper_hint=is_whisper,
        )
        latencies["acoustic_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        if len(processed_audio) == 0:
            return {
                "text": "",
                "raw_text": "",
                "rollouts": [],
                "latencies": latencies,
                "metrics": {"word_count": 0, "wpm": 0.0},
            }

        # 2. ASR Decoding & Rollout Generation
        t1 = time.perf_counter()
        prompt = self.initial_prompt
        if enable_grpo and rollout_count > 1:
            raw_rollouts = self.asr_engine.generate_rollouts(
                audio=processed_audio,
                rollout_count=rollout_count,
                initial_prompt=prompt,
            )
            latencies["asr_ms"] = round((time.perf_counter() - t1) * 1000.0, 2)

            # 3. GRPO Scorer & Advantage Selection
            t2 = time.perf_counter()
            best_rollout, scored_rollouts = self.grpo_scorer.evaluate_group(
                rollouts=raw_rollouts,
                ground_truth=ground_truth,
            )
            latencies["grpo_ms"] = round((time.perf_counter() - t2) * 1000.0, 2)
            raw_text = best_rollout.get("text", "")
            words_meta = best_rollout.get("words", [])
            selected_reward = best_rollout.get("reward", 0.0)
            selected_advantage = best_rollout.get("advantage", 0.0)
        else:
            single_res = self.asr_engine.transcribe_single(
                audio=processed_audio,
                temperature=0.0,
                initial_prompt=prompt,
            )
            latencies["asr_ms"] = round((time.perf_counter() - t1) * 1000.0, 2)
            latencies["grpo_ms"] = 0.0
            raw_text = single_res.get("text", "")
            words_meta = single_res.get("words", [])
            scored_rollouts = [single_res]
            selected_reward = 1.0
            selected_advantage = 0.0

        # 4. Zero-Edit Cleaner & Tone Formatter
        t3 = time.perf_counter()
        clean_text, cleaner_meta = self.cleaner.clean_transcript(
            raw_text=raw_text,
            tone=tone,
        )
        latencies["cleaner_ms"] = round((time.perf_counter() - t3) * 1000.0, 2)

        total_elapsed_ms = round((time.perf_counter() - total_start) * 1000.0, 2)
        latencies["total_ms"] = total_elapsed_ms

        # Calculate Speaking Metrics
        duration_sec = len(processed_audio) / 16000.0
        words = clean_text.split()
        word_count = len(words)
        wpm = round((word_count / max(duration_sec, 0.5)) * 60.0, 1) if word_count > 0 else 0.0

        return {
            "text": clean_text,
            "raw_text": raw_text,
            "selected_rollout": {
                "text": raw_text,
                "reward": selected_reward,
                "advantage": selected_advantage,
            },
            "rollouts": [
                {
                    "text": r.get("text", ""),
                    "reward": r.get("reward", 0.0),
                    "advantage": r.get("advantage", 0.0),
                    "temperature": r.get("temperature", 0.0),
                }
                for r in scored_rollouts
            ],
            "latencies": latencies,
            "acoustic_telemetry": acoustic_meta,
            "cleaner_telemetry": cleaner_meta,
            "metrics": {
                "word_count": word_count,
                "audio_duration_sec": round(duration_sec, 2),
                "speaking_wpm": wpm,
                "real_time_factor": round(total_elapsed_ms / (duration_sec * 1000.0 + 1e-6), 3),
            },
            "words": words_meta,
        }
