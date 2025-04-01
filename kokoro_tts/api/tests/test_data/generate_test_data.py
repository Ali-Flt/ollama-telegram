import os

import numpy as np


def generate_test_audio():
    """Generate test audio data - 1 second of 440Hz tone"""
    # Create 1 second of silence at 24kHz
    audio = np.zeros(24000, dtype=np.float32)

    # Add a simple sine wave to make it non-zero
    t = np.linspace(0, 1, 24000)
    audio += 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone at half amplitude

    # Create test_data directory if it doesn't exist
    os.makedirs("api/tests/test_data", exist_ok=True)

    # Save the test audio
    np.save("api/tests/test_data/test_audio.npy", audio)


if __name__ == "__main__":
    generate_test_audio()
