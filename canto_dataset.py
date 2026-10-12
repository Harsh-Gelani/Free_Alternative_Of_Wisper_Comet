"""
Canto Synthetic Noisy Audio Augmenter & Dataset Generator
Generates real-world noisy audio conditions from scraped conversational text:
- Cafe & office chatter
- Keyboard clicks and desk taps
- Street traffic low-end rumble
- Low-volume whisper attenuation (-20dB)
- Zero external cost - runs 100% locally on CPU using Windows SAPI native audio.
"""

import os
import json
import random
import numpy as np
import soundfile as sf
import scipy.signal as signal
from typing import List, Dict, Any, Tuple


class CantoAudioSynthesizer:
    """
    Synthesizes clean speech and applies acoustic noise augmentation.
    """

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.audio_dir = os.path.join(data_dir, "audio")
        os.makedirs(self.audio_dir, exist_ok=True)

    def synthesize_clean_speech(self, text: str, filename: str) -> str:
        """Synthesizes text to 16kHz WAV file using native Windows SAPI voice."""
        wav_path = os.path.join(self.audio_dir, filename)
        if not os.path.exists(wav_path):
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            stream = win32com.client.Dispatch("SAPI.SpFileStream")
            # 3 = SSFMCreateForWrite
            stream.Open(wav_path, 3)
            speaker.AudioOutputStream = stream
            speaker.Speak(text)
            stream.Close()

        # Resample to 16kHz mono float32 if needed
        data, sr = sf.read(wav_path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != 16000:
            num_samples = int(len(data) * 16000 / sr)
            data = signal.resample(data, num_samples)
            sr = 16000
            sf.write(wav_path, data.astype(np.float32), sr)

        return wav_path

    def add_environmental_noise(
        self,
        audio: np.ndarray,
        noise_type: str = "cafe",
        snr_db: float = 12.0,
    ) -> np.ndarray:
        """
        Mixes procedural real-world environmental noise into clean speech at specified SNR:
        - 'cafe': Multi-speaker babble + room resonance
        - 'traffic': Low frequency rumble (50-200Hz)
        - 'keyboard': Impulsive clicks and clatter
        - 'whisper': Attenuates speech volume by -20dB + room hiss
        """
        sr = 16000
        n_samples = len(audio)
        if n_samples == 0:
            return audio

        speech_power = np.mean(audio**2) + 1e-9

        # Generate noise waveform
        if noise_type == "cafe":
            # Pink noise with 300-3000Hz speech-band emphasis
            white = np.random.randn(n_samples)
            b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
            a = [1, -2.494956002, 2.017265875, -0.522189400]
            noise = signal.lfilter(b, a, white)

        elif noise_type == "traffic":
            # Low-pass rumble (< 250 Hz)
            white = np.random.randn(n_samples)
            b, a = signal.butter(3, 250.0 / (0.5 * sr), btype="low")
            noise = signal.filtfilt(b, a, white)

        elif noise_type == "keyboard":
            # Impulsive typing clicks
            noise = 0.01 * np.random.randn(n_samples)
            num_clicks = int(n_samples / (sr * 0.25))
            for _ in range(num_clicks):
                pos = random.randint(0, max(1, n_samples - 400))
                click = np.exp(-np.linspace(0, 5, 300)) * np.sin(2 * np.pi * 1200 * np.linspace(0, 0.02, 300))
                noise[pos : pos + 300] += 0.4 * click

        elif noise_type == "whisper":
            # Attenuate speech to simulate quiet whisper speech (-18.4 dB) + mic hiss
            attenuated_speech = audio * 0.12
            hiss = 0.008 * np.random.randn(n_samples)
            return (attenuated_speech + hiss).astype(np.float32)

        else:
            # White noise
            noise = np.random.randn(n_samples)

        # Scale noise to target SNR: SNR = 10 * log10(P_speech / P_noise)
        noise_power = np.mean(noise**2) + 1e-9
        target_noise_power = speech_power / (10.0 ** (snr_db / 10.0))
        scale = np.sqrt(target_noise_power / noise_power)
        scaled_noise = noise * scale

        noisy_audio = audio + scaled_noise
        # Normalize to prevent hard clipping
        max_val = np.max(np.abs(noisy_audio))
        if max_val > 0.95:
            noisy_audio = noisy_audio / max_val * 0.95

        return noisy_audio.astype(np.float32)

    def prepare_augmented_dataset(
        self,
        dataset_json_path: str = "./data/canto_dataset.json",
        max_samples: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Loads scraped text, synthesizes clean audio, creates noisy variants,
        and saves augmented audio files.
        """
        with open(dataset_json_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        augmented_records = []
        noise_types = ["cafe", "traffic", "keyboard", "whisper", "clean"]

        print(f"[Dataset] Synthesizing audio for {min(len(samples), max_samples)} samples...")

        for i, s in enumerate(samples[:max_samples]):
            spoken_text = s["spoken"]
            clean_text = s["clean"]
            noise = noise_types[i % len(noise_types)]

            clean_filename = f"sample_{i:03d}_clean.wav"
            clean_path = self.synthesize_clean_speech(spoken_text, clean_filename)

            # Load clean audio
            audio_data, sr = sf.read(clean_path)

            # Apply noise
            if noise != "clean":
                snr = 10.0 if noise != "whisper" else 20.0
                noisy_audio = self.add_environmental_noise(audio_data, noise_type=noise, snr_db=snr)
                noisy_filename = f"sample_{i:03d}_{noise}.wav"
                noisy_path = os.path.join(self.audio_dir, noisy_filename)
                sf.write(noisy_path, noisy_audio, sr)
                active_audio_path = noisy_path
            else:
                active_audio_path = clean_path

            augmented_records.append({
                "id": i,
                "spoken_text": spoken_text,
                "clean_text": clean_text,
                "audio_path": active_audio_path,
                "noise_type": noise,
                "domain": s.get("domain", "general"),
                "has_correction": s.get("has_correction", False),
                "has_fillers": s.get("has_fillers", False),
            })

        output_manifest = os.path.join(self.data_dir, "augmented_manifest.json")
        with open(output_manifest, "w", encoding="utf-8") as f:
            json.dump(augmented_records, f, indent=2)

        print(f"[Dataset] Prepared {len(augmented_records)} augmented audio samples -> {output_manifest}")
        return augmented_records


if __name__ == "__main__":
    synthesizer = CantoAudioSynthesizer()
    records = synthesizer.prepare_augmented_dataset(max_samples=10)
    print("Done. Samples prepared:", len(records))
