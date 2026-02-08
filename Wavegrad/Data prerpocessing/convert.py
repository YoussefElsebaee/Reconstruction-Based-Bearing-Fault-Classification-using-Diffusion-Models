import os
import scipy.io
import torch
import torchaudio

mat_dir = r"C:\Users\yosef\Bachelor\CRWU Dataset\Faulty Data\Drive-end Bearing Fault Data\12KHz Data\Ball\0.028"
wav_output_dir = r"C:\Users\yosef\wavegrad\wavegrad\datasets\Ball_fault DE Data\raw_audio\0.028"  # corrected directory name
os.makedirs(wav_output_dir, exist_ok=True)

K = 1
for mat_filename in sorted(os.listdir(mat_dir)):
    if mat_filename.endswith('.mat'):
        mat_path = os.path.join(mat_dir, mat_filename)
        data = scipy.io.loadmat(mat_path)

        if K == 1:
            column = "X048_DE_time"
        elif K == 2:
            column = "X049_DE_time"
        elif K == 3:
            column = "X050_DE_time"
        elif K == 4:
            column = "X051_DE_time"
        else:
            print(f"No column mapping for file number {K}")
            break
        K += 1

        audio_array = data[column]
        audio_tensor = torch.tensor(audio_array).float().squeeze()  # remove extra dimensions

        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # (1, samples)

        max_val = audio_tensor.abs().max()
        if max_val > 1:
            audio_tensor /= max_val  # normalize to [-1, 1]

        wav_filename = mat_filename.replace('.mat', '.wav')
        wav_path = os.path.join(wav_output_dir, wav_filename)
        torchaudio.save(wav_path, audio_tensor, sample_rate=12000)
        print(f"Saved {wav_path}, shape: {audio_tensor.shape}")
