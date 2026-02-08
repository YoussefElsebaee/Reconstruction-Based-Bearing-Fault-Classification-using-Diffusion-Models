import os
import random
import shutil
from argparse import ArgumentParser

def split_wav_dataset(source_dir, train_dir, val_dir, train_ratio=0.8, seed=42):

    # Make sure directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Collect all .wav files
    wav_files = [f for f in os.listdir(source_dir) if f.lower().endswith(".wav")]
    print(f"Found {len(wav_files)} .wav files in '{source_dir}'")

    # Shuffle for randomness
    random.seed(seed)
    random.shuffle(wav_files)

    # Split
    split_idx = int(len(wav_files) * train_ratio)
    train_files = wav_files[:split_idx]
    val_files = wav_files[split_idx:]

    # Copy to destination
    for f in train_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(train_dir, f))
    for f in val_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(val_dir, f))

    print(f"✅ Split complete:")
    print(f"   Train set: {len(train_files)} files → {train_dir}")
    print(f"   Test set:  {len(val_files)} files → {val_dir}")

def main(args):
    source = r"C:\Users\yosef\wavegrad\wavegrad\datasets\norm_wav\Normal\processed_audio_1s_(97.5%_overlap)\Normal_3"
    train_out = r"C:\Users\yosef\wavegrad\wavegrad\datasets\norm_wav\Normal\train_data_padded"
    val_out = r"C:\Users\yosef\wavegrad\wavegrad\datasets\norm_wav\Normal\val data_padded"
    split_wav_dataset(source, train_out, val_out)


if __name__ == "__main__":
    parser = ArgumentParser(description="split .wav files randomly into 80% train 20% validation.")
    #parser.add_argument("source_dir", help="Directory containing original .wav files")
    #parser.add_argument("train_dir", help="Directory to save train files")
    #parser.add_argument("val_dir", help="Directory to save val files")
    args = parser.parse_args()

    main(args)