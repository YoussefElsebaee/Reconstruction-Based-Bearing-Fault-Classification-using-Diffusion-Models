# Copyright 2025
# Based on LMNT WaveGrad dataset preprocessing script, simplified for segmentation logic.

import os
import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm
from glob import glob
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
# ==============================================================
# Parameters
# ==============================================================
SAMPLE_RATE = 12000      # Sampling frequency (Hz)
SEGMENT_SIZE = 12000     # Segment size in samples (1 second)
HOP_SIZE = 300           # Hop size in samples

# ==============================================================
# Helper: create output directory if it doesn’t exist
# ==============================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ==============================================================
# Core segmentation logic
# ==============================================================
def process_file(filename, base_output_dir):
    """
    Segments a single WAV file into overlapping 1-second windows
    with a hop of 300 samples and saves each window as a new .wav file
    inside a separate folder named after the input file.
    Returns the number of saved segments.
    """
    # Load audio
    audio, sr = torchaudio.load(filename)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Invalid sample rate {sr} in {filename}. Expected {SAMPLE_RATE} Hz.")
    
    # Use mono channel
    #audio = audio[0].clamp(-1.0, 1.0)
    
    total_samples = audio.numel()
    num_segments = int((total_samples - SEGMENT_SIZE) / HOP_SIZE) + 1

    # Create subfolder for this file
    base_name = os.path.splitext(os.path.basename(filename))[0]
    output_dir = os.path.join(base_output_dir, base_name)
    ensure_dir(output_dir)

    for i in range(num_segments):
        start = i * HOP_SIZE
        end = start + SEGMENT_SIZE
        segment = audio[:, start:end]
        if segment.numel() < SEGMENT_SIZE:
            continue

        # reshape to (samples, 1)
        segment_np = segment.cpu().numpy().reshape(-1, 1)

        segment_path = os.path.join(output_dir, f"{base_name}_chunk{i:03d}.wav")
        sf.write(segment_path, segment_np, sr)

    return num_segments

# ==============================================================
# Main function
# ==============================================================
def main(args):
    input_dir = r"C:\Users\yosef\wavegrad\wavegrad\datasets\norm_wav\Normal\raw_audio"
    output_dir = r"C:\Users\yosef\wavegrad\wavegrad\datasets\norm_wav\Normal\processed_audio_1s_(97.5%_overlap)"
    ensure_dir(output_dir)

    filenames = glob(f"{input_dir}/**/*.wav", recursive=True)
    print(f"Found {len(filenames)} .wav files. Starting segmentation...\n")

    total_segments = 0

    for f in tqdm(filenames, desc="Segmenting"):
        num_segments = process_file(f, output_dir)
        total_segments += num_segments
        print(f"→ {os.path.basename(f)} → {num_segments} segments saved.")
    print(f"\n✅ Segmentation complete.")
    print(f"Total number of segments saved: {total_segments}")
    print(f"All segments stored in: {output_dir}")

# ==============================================================
# Entry point
# ==============================================================
if __name__ == "__main__":
    parser = ArgumentParser(description="Segment .wav files using sliding window (1 s window, 300 sample hop).")
    #parser.add_argument("input_dir", help="Directory containing original .wav files")
    #parser.add_argument("output_dir", help="Directory to save segmented .wav files")
    args = parser.parse_args()

    main(args)
