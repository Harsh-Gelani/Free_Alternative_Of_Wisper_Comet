"""
Canto Acoustic Front-End
Real-world noise suppression, Automatic Gain Control (AGC) for whisper/low-volume speech,
and Voice Activity Detection (VAD).
Zero external paid APIs - 100% local CPU execution.
"""

import numpy as np
import scipy.signal as signal
from typing import Tuple, Optional


class CantoAcousticFilter:
    """
    Real-world acoustic conditioning designed for chaotic environments:
    - High background noise (cafe, traffic, fan noise)
    - Low-volume / whisper speech (dynamic range compression & boost)
    - Short utterance isolation (adaptive energy VAD)
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        enable_noise_suppression: bool = True,
        enable_whisper_boost: bool = True,
        vad_threshold: float = 0.015,
    ):
        self.target_sample_rate = target_sample_rate
        self.enable_noise_suppression = enable_noise_suppression
        self.enable_whisper_boost = enable_whisper_boost
        self.vad_threshold = vad_threshold

    def process_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        is_whisper_hint: bool = False,
    ) -> Tuple[np.ndarray, dict]:
        """
        Processes raw audio into clean 16kHz mono float32 for Canto ASR.
        Returns: (processed_audio, telemetry_metrics)
        """
        metrics = {
            "original_length_sec": len(audio_data) / max(sample_rate, 1),
            "original_rms": float(np.sqrt(np.mean(audio_data**2))) if len(audio_data) > 0 else 0.0,
            "boost_applied_db": 0.0,
            "noise_reduction_applied": False,
            "speech_ratio": 1.0,
        }

        # 1. Convert to mono float32
        audio = self._to_mono_float32(audio_data)
        if len(audio) == 0:
            return np.zeros(0, dtype=np.float32), metrics

        # 2. Resample to 16kHz if needed
        if sample_rate != self.target_sample_rate and sample_rate > 0:
            audio = self._resample(audio, sample_rate, self.target_sample_rate)

        # 3. Bandpass filter for human speech (80 Hz to 7500 Hz)
        audio = self._speech_bandpass(audio, self.target_sample_rate)

        # 4. Whisper / Low-Volume Speech Boost (Automatic Gain Control)
        if self.enable_whisper_boost or is_whisper_hint:
            audio, boost_db = self._apply_automatic_gain_control(audio)
            metrics["boost_applied_db"] = float(boost_db)

        # 5. Real-World Stationary Noise Suppression (Spectral Gating)
        if self.enable_noise_suppression:
            audio = self._suppress_stationary_noise(audio, self.target_sample_rate)
            metrics["noise_reduction_applied"] = True

        # 6. Voice Activity Detection (Trim leading/trailing silence)
        audio, speech_ratio = self._trim_silence_vad(audio, self.target_sample_rate)
        metrics["speech_ratio"] = float(speech_ratio)
        metrics["processed_length_sec"] = len(audio) / self.target_sample_rate
        metrics["final_rms"] = float(np.sqrt(np.mean(audio**2))) if len(audio) > 0 else 0.0

        return audio.astype(np.float32), metrics

    def _to_mono_float32(self, audio: np.ndarray) -> np.ndarray:
        """Converts multi-channel or integer audio to normalized mono float32."""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if np.issubdtype(audio.dtype, np.integer):
            max_val = np.iinfo(audio.dtype).max
            audio = audio.astype(np.float32) / max_val
        else:
            audio = audio.astype(np.float32)

        return audio

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """High-speed polyphase resampling."""
        num_samples = int(len(audio) * target_sr / orig_sr)
        return signal.resample(audio, num_samples).astype(np.float32)

    def _speech_bandpass(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Butterworth bandpass filter keeping frequencies between 80Hz and 7500Hz."""
        nyq = 0.5 * sr
        low = 80.0 / nyq
        high = min(7500.0 / nyq, 0.99)
        b, a = signal.butter(4, [low, high], btype="band")
        return signal.filtfilt(b, a, audio).astype(np.float32)

    def _apply_automatic_gain_control(
        self,
        audio: np.ndarray,
        target_rms: float = 0.15,
        max_gain_db: float = 24.0,
    ) -> Tuple[np.ndarray, float]:
        """
        Dynamic Range Compression & AGC for low-volume whisper dictation.
        Amplifies soft voices without hard clipping loud spikes.
        """
        current_rms = np.sqrt(np.mean(audio**2)) + 1e-9
        if current_rms < 1e-4:
            return audio, 0.0

        gain = target_rms / current_rms
        gain_db = 20.0 * np.log10(gain)

        # Clamp gain
        gain_db = np.clip(gain_db, 0.0, max_gain_db)
        applied_gain = 10.0 ** (gain_db / 20.0)

        boosted = audio * applied_gain

        # Soft limiter (tanh compression) to prevent clipping
        if np.max(np.abs(boosted)) > 0.95:
            boosted = np.tanh(boosted / 0.95) * 0.95

        return boosted.astype(np.float32), float(gain_db)

    def _suppress_stationary_noise(
        self,
        audio: np.ndarray,
        sr: int,
        noise_reduction_strength: float = 0.65,
    ) -> np.ndarray:
        """
        Spectral subtraction filter for stationary background noise
        (fans, air conditioners, street rumble, low hum).
        """
        if len(audio) < sr // 4:
            return audio

        # Estimate noise from lowest energy 10% frames
        frame_len = int(0.032 * sr)  # 32ms frames
        hop_len = int(0.016 * sr)    # 16ms hop

        # STFT
        f, t, zxx = signal.stft(audio, fs=sr, nperseg=frame_len, noverlap=frame_len - hop_len)
        mag = np.abs(zxx)
        phase = np.angle(zxx)

        # Noise profile: bottom 15th percentile energy across time per frequency bin
        noise_profile = np.percentile(mag, 15, axis=-1, keepdims=True)

        # Spectral subtraction with over-subtraction factor
        alpha = 1.5 * noise_reduction_strength
        beta = 0.05  # Spectral floor
        cleaned_mag = np.maximum(mag - alpha * noise_profile, beta * mag)

        # Inverse STFT
        cleaned_zxx = cleaned_mag * np.exp(1j * phase)
        _, cleaned_audio = signal.istft(cleaned_zxx, fs=sr, nperseg=frame_len, noverlap=frame_len - hop_len)

        # Match output length
        if len(cleaned_audio) > len(audio):
            cleaned_audio = cleaned_audio[: len(audio)]
        elif len(cleaned_audio) < len(audio):
            cleaned_audio = np.pad(cleaned_audio, (0, len(audio) - len(cleaned_audio)))

        return cleaned_audio.astype(np.float32)

    def _trim_silence_vad(
        self,
        audio: np.ndarray,
        sr: int,
        padding_sec: float = 0.15,
    ) -> Tuple[np.ndarray, float]:
        """
        Adaptive Voice Activity Detection to trim dead silence around short utterances.
        Preserves natural speech boundaries.
        """
        frame_len = int(0.02 * sr)  # 20ms
        if len(audio) <= frame_len:
            return audio, 1.0

        num_frames = len(audio) // frame_len
        energies = np.array([
            np.sqrt(np.mean(audio[i * frame_len : (i + 1) * frame_len] ** 2))
            for i in range(num_frames)
        ])

        noise_floor = np.percentile(energies, 20) + 1e-6
        threshold = max(noise_floor * 2.2, self.vad_threshold)

        speech_indices = np.where(energies > threshold)[0]
        if len(speech_indices) == 0:
            return audio, 0.0

        pad_frames = int(padding_sec * sr / frame_len)
        start_frame = max(0, speech_indices[0] - pad_frames)
        end_frame = min(num_frames, speech_indices[-1] + pad_frames + 1)

        start_sample = start_frame * frame_len
        end_sample = min(len(audio), end_frame * frame_len)

        trimmed = audio[start_sample:end_sample]
        speech_ratio = len(trimmed) / max(len(audio), 1)

        return trimmed.astype(np.float32), speech_ratio
