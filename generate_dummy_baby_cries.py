import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine
import os

# Define cry types and base frequencies
cry_types = {
    'hungry': 440,  # A4
    'pain': 880,  # A5 - higher pitch
    'tired': 220,  # A3 - lower pitch
    'discomfort': 330  # E4
}

duration_ms = 2000  # 2 seconds
output_dir = "baby_cry_dataset"

os.makedirs(output_dir, exist_ok=True)

for cry, freq in cry_types.items():
    cry_folder = os.path.join(output_dir, cry)
    os.makedirs(cry_folder, exist_ok=True)

    for i in range(3):  # generate 3 samples per cry type
        tone = Sine(freq).to_audio_segment(duration=duration_ms)
        # Add some noise
        noise = AudioSegment(
            (np.random.rand(len(tone.get_array_of_samples())) * 200 - 100).astype(np.int16).tobytes(),
            frame_rate=tone.frame_rate,
            sample_width=2,
            channels=1
        )
        audio = tone.overlay(noise)
        filename = os.path.join(cry_folder, f"{cry}_{i + 1}.wav")
        audio.export(filename, format="wav")
        print(f"Saved: {filename}")
