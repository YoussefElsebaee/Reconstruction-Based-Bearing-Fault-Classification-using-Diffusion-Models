# =======================================================================================================
# This pyhton file is used to generate the log mel spectrogram of .wav files.
# First, .wav files, found in the directory passed to the python code as an argument, are extracted.
# Then, each .wav file is loaded and padded with 874 zeros on each side (end).
# Next, the mel spectrogram is generated using STFT, with neither normalization nor centering.
# After that, a log of the spectrogram is taken and we clamp the log mel spectrogram between [0,1] 
# Finally, the log mel spectrogram is saved with the name of the .wav file + a suffix (.spec.npy)
# =======================================================================================================

import numpy as np
import torch
import torchaudio as T
import torchaudio.transforms as TT
import torch.nn.functional as F

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm
import scipy.io as sio

def extract_signal(wav_dict):
    for k in wav_dict.keys():
        if 'DE_time' in str(k):
            sig = wav_dict[k].squeeze()
            if sig.ndim == 1:
                return sig
    raise KeyError(f"Could not find signal in MAT file. Keys: {wav_dict.keys()}")

def transform(filename):
  #print(T.info(filename))
  audio, sr = T.load(filename)

  # padding the audio with zeros:
  pad_left  = 874
  pad_right = 874
  audio = F.pad(audio, (pad_left, pad_right))
  #print(audio.max(), audio.min())
  #print(audio.shape)

  #extracting the log-mel-spectrogram
  sr=12000
  hop = 300
  win = hop * 4
  n_fft = 2**((win-1).bit_length())
  f_max = sr / 2.0
  mel_spec_transform = TT.MelSpectrogram(sample_rate=sr, n_fft=n_fft, win_length=win, hop_length=hop, f_min=20.0, f_max=f_max, power=1.0, normalized=False, center=False)
  with torch.no_grad():
    spectrogram = mel_spec_transform(audio)
    spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
    spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
    spectrogram = spectrogram.squeeze(0)
    #print(spectrogram.shape)
    np.save(f'{filename}.spec.npy', spectrogram.cpu().numpy().astype(np.float32))

def main(args):
  filenames = glob(f'{args.dir}/**/*.wav', recursive=True)
  with ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(transform, filenames), desc='Preprocessing', total=len(filenames)))

if __name__ == '__main__':
  parser = ArgumentParser(description='prepares a dataset to train WaveGrad')
  parser.add_argument('dir',
      help='directory containing .wav files for training')
  main(parser.parse_args())