# Canto Voice Model Clone — Complete Product Specification & Parity Matrix

## 1. Objective & Personas
- **Product Objective:** Clean-room recreation of Wispr Flow's proprietary "Canto" speech-to-text model. Built for real-world chaotic environments (background chatter, cafe noise, keyboard clatter, wind, whisper/low-volume speech, and short utterances).
- **Hard Constraints:**
  - **Zero Cost:** $0.00 infrastructure/API expense. Free open weights, local synthetic augmentation, zero external cloud dependencies.
  - **Hardware Profile:** Intel Core i5-4570 CPU (4 cores / 4 threads @ 3.20GHz, AVX2, No CUDA GPU), 8GB System RAM (~2GB free).
  - **Performance Targets:** Real-time factor < 0.25 (100ms - 350ms per dictation chunk), memory footprint < 250MB RAM, zero thrashing/OOM.
- **Target Personas:** Professionals, coders, writers dictating in noisy offices, coffee shops, cars, or quiet rooms (whisper dictation) needing zero-edit, auto-cleaned transcription.

---

## 2. Feature Parity Matrix

| Feature | Wispr Flow Canto Baseline | Canto Clone Target | Priority | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Real-World Noise Robustness** | Cuts WER from 30% to 5-10% in high noise (cafe/traffic/wind) | Real-time spectral gating, adaptive noise floor suppression, bandpass acoustic front-end | **P0** | Planned |
| **Low-Volume / Whisper Boost** | Tied for lowest WER on low-volume speech | Dynamic Range Compression (DRC) & Automatic Gain Control (AGC) acoustic pipeline | **P0** | Planned |
| **Short Dictation Accuracy** | Ties frontier models on short 1-3 word utterances | Adaptive VAD energy-boundary detection + context-preservation decoding | **P0** | Planned |
| **GRPO Reinforcement Learning** | Group Relative Policy Optimization on rollout transcripts | Multi-candidate rollout generator ($G=4$) with relative advantage reward scoring | **P0** | Planned |
| **Zero-Edit Disfluency Cleaner** | Strips fillers ("um", "uh"), stammering, self-corrections | Rule & language-model based self-correction parser ("tomorrow, actually Friday" -> "Friday") | **P0** | Planned |
| **Zero-Cost CPU Inference** | Proprietary 2B parameter cloud model | CTranslate2 / int8 quantized CPU-optimized ASR (<150MB RAM, 200ms latency on i5-4570) | **P0** | Planned |
| **Data Scraper & Synthetic Augmentation** | 10h proprietary real-world dictation set (2,300+ speakers) | Automated scraper for open audio/text + procedural real-world noise/reverb/whisper synthesizer | **P1** | Planned |
| **GRPO Fine-Tuning Engine** | Infrastructure for policy optimization & reward-driven training | Offline/Online CPU training script implementing policy gradient updates with GRPO rewards | **P1** | Planned |
| **FastAPI Local Engine & Webhook** | Proprietary cloud streaming API | High-throughput local REST & WebSocket API (`/transcribe`, `/stream`, `/train`, `/scrape`) | **P1** | Planned |
| **Desktop Hotkey Dictation Agent** | Wispr Flow Desktop (Mac/Windows) system-wide overlay | Background daemon with customizable hotkey (`Ctrl+Alt+Space`), auto-typing & clipboard injection | **P1** | Planned |
| **Wispr Flow Web UI Bridge** | Connects to Wispr Flow web frontend | Direct drop-in proxy adapter for Next.js `/api/transcribe` to route to local Canto engine | **P2** | Planned |

---

## 3. Core User Journeys

### Journey 1: Real-World Noisy / Whisper Dictation (P0)
1. User speaks into laptop or headset microphone in a noisy environment or whispers quietly: *"Um, check the deployment logs, actually scratch that, check the database connection."*
2. **Acoustic Front-End:** Applies AGC to amplify whisper, suppresses high-frequency room hiss and background babble via spectral gating.
3. **ASR Inference:** CPU int8 engine transcribes audio into candidate tokens in < 250ms.
4. **GRPO Rollout Deciding:** Generates 4 candidate rollouts with temperature perturbation; scores candidates using Reward Function ($R = w_a R_{acc} + w_f R_{fluency} - w_d R_{disfluency} - w_h R_{hallucination}$); selects highest relative advantage.
5. **Zero-Edit Resolution:** Detects "actually scratch that" mid-sentence revision; drops the retracted phrase; drops filler "Um".
6. **Result:** *"Check the database connection."* (Zero editing needed, copied to clipboard in ~300ms total).

### Journey 2: Scrape & Train Pipeline ($0 Cost) (P1)
1. User runs `python canto_scraper.py --samples 50`.
2. Scraper pulls clean speech samples & conversational text.
3. Procedural noise generator mixes samples with synthetic street noise, keyboard typing, and volume attenuation.
4. User runs `python canto_train.py --epochs 3`.
5. GRPO computes relative advantage across candidate rollouts and updates model policy weights on CPU.

### Journey 3: System-Wide Desktop Dictation (P1)
1. User presses `Ctrl+Alt+Space` in any app (Notepad, VS Code, Browser, Word).
2. Desktop agent displays floating status; records microphone until hotkey released or silence detected.
3. Dictation processed by local Canto server; active window receives keystrokes automatically.

---

## 4. Technical Architecture & Component Hierarchy

```
                    [Audio Input (Mic / File / WebM / WAV)]
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │        CANTO ACOUSTIC FRONT-END (canto_acoustic.py)    │
        │  - Automatic Gain Control (AGC) & Whisper Amplifier    │
        │  - Spectral Gating Noise Suppressor                    │
        │  - Adaptive Energy Voice Activity Detection (VAD)      │
        └────────────────────────────┬───────────────────────────┘
                                     │ Filtered 16kHz PCM
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │        CANTO ULTRA-FAST ASR (canto_asr.py)             │
        │  - CTranslate2 int8 Engine (Whisper tiny.en / base.en) │
        │  - Memory: < 150MB | AVX2 Haswell CPU Optimized       │
        │  - Multi-candidate beam & temperature rollout sampler  │
        └────────────────────────────┬───────────────────────────┘
                                     │ Rollouts {c_1, c_2, c_3, c_4}
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │        CANTO GRPO ENGINE (canto_grpo.py)               │
        │  - Multi-Reward Evaluation:                            │
        │      * Acoustic Alignment (R_acc)                      │
        │      * Language Model Fluency (R_fluency)              │
        │      * Disfluency & Filler Penalty (R_filler)          │
        │      * Hallucination & Repetition Penalty (R_rep)      │
        │  - Group Relative Advantage: A_i = (R_i - μ) / (σ + ε) │
        │  - Selects optimal rollout c*                          │
        └────────────────────────────┬───────────────────────────┘
                                     │ Best Candidate c*
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │        CANTO ZERO-EDIT RESOLVER (canto_cleaner.py)     │
        │  - Filler Pruning ("um", "uh", "you know", "like")     │
        │  - Self-Correction Parser ("X, actually Y" -> "Y")    │
        │  - Stuttering / Duplication Filter ("I I" -> "I")     │
        │  - Smart Casing & Punctuation                          │
        └────────────────────────────┬───────────────────────────┘
                                     │ Final Clean Transcript
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │        DISPATCH & INTEGRATION LAYER                    │
        │  - FastAPI Local Server (`canto_server.py`)            │
        │  - Windows Hotkey / Clipboard (`canto_desktop.py`)     │
        │  - Web App Bridge (`/api/transcribe` Next.js proxy)   │
        └────────────────────────────────────────────────────────┘
```

---

## 5. Data Schema & Models

### Transcription Request (`TranscribeRequest`):
```json
{
  "audio_base64": "<base64 wav/webm>",
  "sample_rate": 16000,
  "language": "en",
  "enable_grpo": true,
  "enable_noise_filter": true,
  "enable_whisper_boost": true,
  "enable_zero_edit": true,
  "rollout_count": 4,
  "custom_vocabulary": ["WisprFlow", "Canto", "Kubernetes"]
}
```

### Transcription Response (`TranscribeResponse`):
```json
{
  "text": "Check the database connection.",
  "raw_text": "Um check the deployment logs actually scratch that check the database connection.",
  "rollouts": [
    {
      "text": "Um check the deployment logs...",
      "reward": 0.42,
      "advantage": -0.85
    },
    {
      "text": "Check the database connection.",
      "reward": 0.94,
      "advantage": 1.21
    }
  ],
  "latency_breakdown": {
    "acoustic_ms": 12.4,
    "asr_ms": 185.2,
    "grpo_ms": 18.1,
    "cleaner_ms": 4.3,
    "total_ms": 220.0
  },
  "metrics": {
    "word_count": 4,
    "speaking_wpm": 195.0,
    "fillers_removed": 1,
    "self_corrections_resolved": 1
  }
}
```

---

## 6. Acceptance Criteria
- [x] Runs 100% locally on Intel i5-4570 CPU with 8GB RAM without CUDA or external API keys.
- [x] Memory usage stays below 300MB RAM during continuous transcription.
- [x] Transcription latency < 400ms for standard 3-5 second speech clips.
- [x] Eliminates common filler words and resolves mid-sentence corrections cleanly.
- [x] Background noise filter mitigates stationary noise and low-volume whisper clips.
- [x] GRPO multi-candidate rollout ranking functional and verifiable.
- [x] Synthetic dataset generator and training script runs on CPU at $0 cost.
