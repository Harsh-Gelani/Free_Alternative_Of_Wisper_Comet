"""
Canto Desktop System-Wide Dictation Agent
Runs in background on Windows.
- Global Hotkey listener: Ctrl+Alt+Space
- Records microphone input
- Sends audio to Canto Speech Engine
- Automatically writes text to clipboard and types into active window
100% Free, Zero API cost, runs locally on Intel i5-4570 CPU.
"""

import time
import sys
import threading
import numpy as np
import sounddevice as sd
import pyperclip
from pynput import keyboard
from typing import Optional

from canto_engine import CantoPipeline


class CantoDesktopAgent:
    """
    Background desktop dictation service.
    """

    def __init__(self, model_size: str = "tiny.en", sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.is_recording = False
        self.audio_frames = []
        self.pipeline = CantoPipeline(model_size=model_size)
        self.kb_controller = keyboard.Controller()
        self.lock = threading.Lock()

        # Hotkey tracking
        self.current_keys = set()
        print("\n=======================================================")
        print("  CANTO DESKTOP DICTATION AGENT (Wispr Flow Clone)")
        print("  Status: ACTIVE & LISTENING")
        print("  Hotkey: Ctrl + Alt + Space")
        print("  Action: Hold to speak, release to auto-type clean text")
        print("  Hardware: CPU (Zero Cost, Zero Cloud)")
        print("=======================================================\n")

    def start_recording(self):
        with self.lock:
            if self.is_recording:
                return
            self.is_recording = True
            self.audio_frames = []
            print("\r[Canto] 🎙️ LISTENING... (Speak naturally with fillers/corrections)   ", end="", flush=True)

        def audio_callback(indata, frames, time_info, status):
            if self.is_recording:
                self.audio_frames.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=audio_callback,
        )
        self.stream.start()

    def stop_and_transcribe(self):
        with self.lock:
            if not self.is_recording:
                return
            self.is_recording = False
            print("\r[Canto] ⚙️ CANTO GRPO PROCESSING...                             ", end="", flush=True)

        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass

        if not self.audio_frames:
            print("\r[Canto] No audio recorded.                                          ")
            return

        audio_data = np.concatenate(self.audio_frames, axis=0).flatten()
        if len(audio_data) < self.sample_rate * 0.4:
            print("\r[Canto] Clip too short (< 0.4s).                                    ")
            return

        # Run Canto transcription
        res = self.pipeline.transcribe_numpy(
            audio_data=audio_data,
            sample_rate=self.sample_rate,
            enable_grpo=True,
        )

        clean_text = res.get("text", "").strip()
        latency = res.get("latencies", {}).get("total_ms", 0.0)

        if clean_text:
            print(f"\r[Canto] ✨ '{clean_text}' ({latency}ms)                              ")
            # 1. Copy to clipboard
            pyperclip.copy(clean_text)

            # 2. Simulate Ctrl+V paste into active application window
            time.sleep(0.05)
            with self.kb_controller.pressed(keyboard.Key.ctrl):
                self.kb_controller.press("v")
                self.kb_controller.release("v")
        else:
            print("\r[Canto] (Silence or no speech detected)                            ")

        print("\r[Canto] Ready. Press Ctrl+Alt+Space to dictate...                    ", end="", flush=True)

    def on_press(self, key):
        # Add to pressed set
        self.current_keys.add(key)

        # Check for Ctrl + Alt + Space
        has_ctrl = any(k in self.current_keys for k in [keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r])
        has_alt = any(k in self.current_keys for k in [keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr])
        has_space = keyboard.Key.space in self.current_keys

        if has_ctrl and has_alt and has_space and not self.is_recording:
            self.start_recording()

    def on_release(self, key):
        if key in self.current_keys:
            self.current_keys.remove(key)

        # If any of the trigger keys is released while recording, stop & transcribe
        trigger_keys = [
            keyboard.Key.space,
            keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r,
            keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr
        ]
        if key in trigger_keys and self.is_recording:
            self.stop_and_transcribe()

    def run(self):
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()


if __name__ == "__main__":
    agent = CantoDesktopAgent()
    agent.run()
