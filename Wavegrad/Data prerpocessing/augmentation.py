import os
import numpy as np
import soundfile as sf
from tqdm import tqdm

# ----------------------------
# Noise function (SNR-based)
# ----------------------------
def add_awgn_snr(signal, snr_db):
    """
    Add additive white Gaussian noise (AWGN) to a signal
    using SNR in dB, following the paper's definition.
    """
    # Signal power
    signal_power = np.mean(signal ** 2)

    # Convert SNR from dB to linear
    snr_linear = 10 ** (snr_db / 10)

    # Noise power
    noise_power = signal_power / snr_linear

    # Generate white Gaussian noise
    noise = np.random.normal(
        loc=0.0,
        scale=np.sqrt(noise_power),
        size=signal.shape
    )

    return signal + noise


# ----------------------------
# Dataset augmentation
# ----------------------------
def create_noisy_datasets_snr(
    input_dir,
    output_dir,
    snr_db_values
):
    """
    Create noisy datasets using SNR (dB), as done in the paper.
    """

    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    # Collect WAV files only
    wav_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, f))

    if len(wav_files) == 0:
        raise RuntimeError("No .wav files found in input directory")

    print(f"Found {len(wav_files)} WAV files")

    for snr_db in snr_db_values:
        snr_name = f"SNR_{snr_db}dB"
        snr_output_root = os.path.join(output_dir, snr_name)

        print(f"\nCreating dataset for {snr_name}")

        for wav_path in tqdm(wav_files):
            # Preserve folder structure
            rel_path = os.path.relpath(wav_path, input_dir)
            out_path = os.path.join(snr_output_root, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # Load audio
            signal, sr = sf.read(wav_path)
            #print(signal.max(), signal.min())
            # Mono or multi-channel handling
            if signal.ndim == 1:
                noisy_signal = add_awgn_snr(signal, snr_db)
            else:
                noisy_signal = np.stack(
                    [add_awgn_snr(signal[:, ch], snr_db)
                     for ch in range(signal.shape[1])],
                    axis=1
                )
            # Optional clipping safeguard
            noisy_signal = np.clip(noisy_signal, -1.0, 1.0)

            # Save noisy file
            sf.write(out_path, noisy_signal, sr)

    print("\nAll SNR-based noisy datasets created successfully.")


input_dataset_dir = r"C:\Users\yosef\wavegrad\wavegrad\final\dataset\norm_wav\outer_race\val_data_padded"
output_dataset_dir = r"C:\Users\yosef\wavegrad\wavegrad\final\dataset\norm_wav\outer_race\noisy_val_data_padded"

snr_db_values = [-4, -2, 0, 2, 4, 6, 8, 10]

create_noisy_datasets_snr(
    input_dir=input_dataset_dir,
    output_dir=output_dataset_dir,
    snr_db_values=snr_db_values
)
