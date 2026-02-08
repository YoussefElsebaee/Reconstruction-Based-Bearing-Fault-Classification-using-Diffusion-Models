# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os
import torch
import torchaudio
import shutil
from tqdm import tqdm
from argparse import ArgumentParser

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from training_files.params import AttrDict, params as base_params
from training_files.model import WaveGrad

def load_model(model_dir, device, noise_schedule):
    print(f"🔄 Loading model from {model_dir} ...")
    if os.path.exists(f'{model_dir}/weights.pt'):
        checkpoint = torch.load(f'{model_dir}/weights.pt', map_location=device)
    else:
        checkpoint = torch.load(model_dir, map_location=device, weights_only=False)

    model = WaveGrad(AttrDict(base_params)).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print("✅ Model loaded successfully.")

    # ✅ Load custom noise schedule if provided
    if noise_schedule is not None:
        print(f"🎛️ Using custom noise schedule with {len(noise_schedule)} steps.")
        model.params.noise_schedule = noise_schedule.tolist()
        
    return model

def predict_audio(spectrogram, model, device):
  beta = np.array(model.params.noise_schedule)
  alpha = 1 - beta
  alpha_cum = np.cumprod(alpha)

  # Expand rank 2 tensors by adding a batch dimension.
  if len(spectrogram.shape) == 2:
    spectrogram = spectrogram.unsqueeze(0)
  spectrogram = spectrogram.to(device)

  audio = torch.randn(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1], device=device)
  noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)

  with torch.no_grad():
    for n in range(len(alpha) - 1, -1, -1):
      c1 = 1 / alpha[n]**0.5
      c2 = (1 - alpha[n]) / (1 - alpha_cum[n])**0.5
      audio = c1 * (audio - c2 * model(audio, spectrogram, noise_scale[n]).squeeze(1))
      if n > 0:
        noise = torch.randn_like(audio)
        sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
        audio += sigma * noise
      audio = torch.clamp(audio, -1.0, 1.0)
  return audio.cpu(), model.params.sample_rate

def process_folder(input_dir, output_dir, model_dir, device, noise_schedule_path):
    os.makedirs(output_dir, exist_ok=True)
    model = load_model(model_dir, device, noise_schedule_path)

    MAT_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    print(f"🎧 Found {len(MAT_files)} WAV files in {input_dir}")

    for MAT_file in tqdm(MAT_files, desc="Processing files"):
        base_name = MAT_file[:-4]  # remove .wav
        npy_file = base_name + ".wav.spec.npy"
        npy_path = os.path.join(input_dir, npy_file)
        MAT_path = os.path.join(input_dir, MAT_file)

        # Skip if .npy file is missing
        if not os.path.exists(npy_path):
            print(f"⚠️ Missing spectrogram for {MAT_file}, skipping.")
            continue

        # Load spectrogram
        spectrogram = torch.from_numpy(np.load(npy_path))

        # Generate new audio
        gen_audio, sr = predict_audio(spectrogram, model, device)

        # Create output subfolder
        subfolder = os.path.join(output_dir, base_name)
        os.makedirs(subfolder, exist_ok=True)

        # Copy original files
        shutil.copy2(MAT_path, subfolder)
        shutil.copy2(npy_path, subfolder)


        # Save generated audio
        gen_path = os.path.join(subfolder, f"{base_name}_generated.wav")
        torchaudio.save(gen_path, gen_audio, sample_rate=sr)
        
        del gen_audio
        del spectrogram
        print(f"✅ Saved: {gen_path}")

# ---------- Main ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch inference for WaveGrad")
    #parser.add_argument("model_dir", help="Path to trained WaveGrad model or weights.pt")
    #parser.add_argument("output_dir", help="Folder to save output folders and generated audios")
    #parser.add_argument("input_dir", help="Folder containing .wav and .spec.npy files")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda or cpu)")
    args = parser.parse_args()
    noise_schedule= 0.5*(1-np.cos(np.linspace(0,np.pi,1000)))*(0.01-1e-6) + 1e-6

    model_dir= r"C:\Users\yosef\wavegrad\wavegrad\final\trained_models\Ball\model (batch= 1, cosine_NS_1000_steps, 2e-4_LR, epochs= 160) Padded-norm.-wav\weights-160.pt"
    output_dir= r"C:\Users\yosef\wavegrad\wavegrad\final\trained_models\Ball\model (batch= 1, cosine_NS_1000_steps, 2e-4_LR, epochs= 160) Padded-norm.-wav\inferences (cosine_1000)"
    input_dir= r"C:\Users\yosef\wavegrad\wavegrad\final\dataset\norm_wav\ball\test-val_data_padded"
    process_folder(input_dir, output_dir, model_dir, args.device, noise_schedule)
    # steps=1000, beta_min=1e-6, beta_max=0.01
    # linear: np.linspace(1e-6, 0.01, 1000)
    # cosine: 0.5*(1-np.cos(np.linspace(0,np.pi,1000)))*(0.01-1e-6) + 1e-6