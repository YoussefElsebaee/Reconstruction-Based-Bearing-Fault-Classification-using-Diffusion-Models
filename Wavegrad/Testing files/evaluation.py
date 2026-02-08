from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
import  torch
import numpy as np
import librosa
import torchaudio as T
import torchaudio.transforms as TT
import matplotlib.pyplot as plt
import torchaudio.functional as F
import torch.nn.functional as Fk
import os
import pandas as pd
from pystoi import stoi
import scipy.io as sio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
#from training_files.params import params
from pesq import pesq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Using device: {device}")

#inferences 
sr_o=12000
hop = 300
win = hop * 4
n_fft = 2**((win-1).bit_length())
f_max = sr_o / 2.0
mel= TT.MelSpectrogram(sample_rate=sr_o, n_fft=n_fft, win_length=win, hop_length=hop, f_min=20.0, f_max=f_max, power=1.0, normalized=False, center=False)

def log_mel_mse(x, y, sr=12000):

    pad_left  = 874
    pad_right = 874

    x = Fk.pad(x, (pad_left, pad_right))
    y = Fk.pad(y, (pad_left, pad_right))

    X = torch.log(mel(x) + 1e-5)
    Y = torch.log(mel(y) + 1e-5)
    return torch.mean((X - Y)**2)

def mel_cepstral_distortion(x, y, sr=12000, n_mfcc=13):
    
    if torch.is_tensor(x):
        x = x.squeeze().cpu().numpy()
    if torch.is_tensor(y):
        y = y.squeeze().cpu().numpy()

    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    mfcc_x = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc)
    mfcc_y = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    min_frames = min(mfcc_x.shape[1], mfcc_y.shape[1])
    mfcc_x = mfcc_x[:, :min_frames]
    mfcc_y = mfcc_y[:, :min_frames]

    diff = mfcc_x - mfcc_y
    dist = np.sqrt((diff ** 2).sum(axis=0))  

    mcd = (10.0 / np.log(10)) * np.sqrt(2.0) * np.mean(dist)
    return float(mcd)

def log_spectral_distance(x, y, sr=12000):
    if torch.is_tensor(x):
        x = x.squeeze().cpu().numpy()
    if torch.is_tensor(y):
        y = y.squeeze().cpu().numpy()

    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    X = librosa.stft(x, n_fft=n_fft, hop_length=hop, win_length=win)
    Y = librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win)

    X_mag = np.abs(X)
    Y_mag = np.abs(Y)

    log_X = np.log(X_mag + 1e-8)
    log_Y = np.log(Y_mag + 1e-8)

    lsd_per_frame = np.sqrt(np.mean((log_X - log_Y) ** 2, axis=0))
    lsd = np.mean(lsd_per_frame)

    return float(lsd)

def evaluate_samples(input_root):
    
    results =[]

    for folder in os.listdir(input_root):
        subdir = os.path.join(input_root, folder)
        if not os.path.isdir(subdir):
            continue

        WAV_path = os.path.join(subdir, f"{folder}.wav")
        gen_path = os.path.join(subdir, f"{folder}_generated.wav")

        if not os.path.exists(WAV_path) or not os.path.exists(gen_path):
            print(f"⚠️ Missing files for {folder}, skipping.")
            continue
        
        original_audio, sr_p = T.load(WAV_path)
        predicted_audio, sr_p = T.load(gen_path)
        #print(predicted_audio.shape)
        #original_audio= original_audio.reshape(1, 12000)

        min_len = min(original_audio.shape[-1], predicted_audio.shape[-1])
        original_audio = original_audio[..., :min_len]
        predicted_audio = predicted_audio[..., :min_len]
        #pred_len = predicted_audio.shape[-1]
        #predicted_audio = predicted_audio[..., 150:pred_len-150]
        # Ensure float tensor
        original_audio = torch.as_tensor(original_audio, dtype=torch.float32)
        predicted_audio = torch.as_tensor(predicted_audio, dtype=torch.float32)

        # Ensure at least 1D
        original_audio = original_audio.reshape(1, -1)
        predicted_audio = predicted_audio.reshape(1, -1)

        # === Compute metrics ===
        metrics = {
            "sample_name": folder,
            "log_mel_mse": float(log_mel_mse(original_audio, predicted_audio, sr_o)),
            "LSD": log_spectral_distance(original_audio, predicted_audio, sr_o),
            "MCD": float(mel_cepstral_distortion(original_audio, predicted_audio, sr_o))
        }

        if torch.is_tensor(original_audio):
            original_audio = original_audio.squeeze().cpu().numpy()
        if torch.is_tensor(predicted_audio):
            predicted_audio = predicted_audio.squeeze().cpu().numpy()

        time = np.linspace(0, len(original_audio) / sr_o, num=len(original_audio))
        title="Audio Waveform Comparison"

        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(time, original_audio, color='blue')
        plt.title("Original Audio Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        plt.subplot(2, 1, 2)
        plt.plot(time, predicted_audio, color='orange')
        plt.title("Predicted Audio Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        save_path = os.path.join(subdir, 'waveforms.png')
        plt.savefig(save_path)
        plt.close()

        results.append(metrics)

        print("evaluated and saved waveforms for sample:",folder)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch evaluation for WaveGrad")
    parser.add_argument("input_dir", help="Folder containing .wav and _generated.wav files")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu)")
    args = parser.parse_args()

    results= evaluate_samples(args.input_dir)
    df = pd.DataFrame(results)

    avg_row = {
        "sample_name": "AVERAGE",
        "log_mel_mse": df["log_mel_mse"].mean(),
        "LSD": df["LSD"].mean(),
        "MCD": df["MCD"].mean()
    }

    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    # === Save results to Excel ===
    save_path = os.path.join(args.input_dir, "evaluation_results.xlsx")
    df.to_excel(save_path, index=False, sheet_name="Results")


    print(f"\n📊 Saved results (with averages) to: {save_path}")
