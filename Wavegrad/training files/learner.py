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
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from wavegrad.data_preprocessing_files.wav.dataset import from_path as dataset_from_path
from wavegrad.training_files.model import WaveGrad


def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)


class WaveGradLearner:
  def __init__(self, model_dir, model, dataset, optimizer, params, *args, **kwargs):
    os.makedirs(model_dir, exist_ok=True)
    self.model_dir = model_dir
    self.model = model
    self.dataset = dataset
    self.optimizer = optimizer
    self.params = params
    self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
    self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
    self.step = 0
    self.is_master = True

    beta = np.array(self.params.noise_schedule)
    noise_level = np.cumprod(1 - beta)**0.5
    noise_level = np.concatenate([[1.0], noise_level], axis=0)
    self.noise_level = torch.tensor(noise_level.astype(np.float32))
    self.loss_fn = nn.L1Loss()
    self.summary_writer = None

  def state_dict(self):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      model_state = self.model.module.state_dict()
    else:
      model_state = self.model.state_dict()
    return {
        'step': self.step,
        'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
        'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
        'params': dict(self.params),
        'scaler': self.scaler.state_dict(),
    }

  def load_state_dict(self, state_dict):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      self.model.module.load_state_dict(state_dict['model'])
    else:
      self.model.load_state_dict(state_dict['model'])
    self.optimizer.load_state_dict(state_dict['optimizer'])
    self.scaler.load_state_dict(state_dict['scaler'])
    self.step = state_dict['step']

  def save_to_checkpoint(self, filename='weights'):
    epoch= int(self.step/3518)
    save_basename = f'{filename}-{epoch}.pt'
    save_name = f'{self.model_dir}/{save_basename}'
    link_name = f'{self.model_dir}/{filename}.pt'
    torch.save(self.state_dict(), save_name)
    if os.name == 'nt':
      torch.save(self.state_dict(), link_name)
    else:
      if os.path.islink(link_name):
        os.unlink(link_name)
      os.symlink(save_basename, link_name)

  def restore_from_checkpoint(self, filename='weights'):
    try:
        # Find the latest checkpoint automatically
        checkpoints = [f for f in os.listdir(self.model_dir) if f.startswith(f'{filename}-') and f.endswith('.pt')]
        if not checkpoints:
            print("⚠️ No checkpoint found, starting from scratch.")
            return False

        # Sort by step number (after 'weights-')
        checkpoints.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))
        latest = checkpoints[-1]
        print(f"🔄 Restoring from latest checkpoint: {latest}")
        
        checkpoint = torch.load(os.path.join(self.model_dir, latest), weights_only= False)
        self.load_state_dict(checkpoint)
        return True
    except Exception as e:
        print(f"⚠️ Failed to load checkpoint: {e}")
        return False


  def train_step(self, features):
    for param in self.model.parameters():
      param.grad = None

    audio = features['audio']
    spectrogram = features['spectrogram']

    N, T = audio.shape
    S = len(self.noise_level) - 1
    device = audio.device
    self.noise_level = self.noise_level.to(device)

    with self.autocast:
      s = torch.randint(1, S + 1, [N], device=audio.device)
      l_a, l_b = self.noise_level[s-1], self.noise_level[s]
      noise_scale = l_a + torch.rand(N, device=audio.device) * (l_b - l_a)
      noise_scale = noise_scale.unsqueeze(1)
      noise = torch.randn_like(audio)
      noisy_audio = noise_scale * audio + (1.0 - noise_scale**2)**0.5 * noise
      predicted = self.model(noisy_audio, spectrogram, noise_scale.squeeze(1))
      loss = self.loss_fn(noise, predicted.squeeze(1))

    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    return loss
  
  def run_epoch(self, dataset, training=True):
    """
    Runs one full epoch on the given dataset.
    Returns average loss.
    """
    mode = "train" if training else "val"
    self.model.train(training)
    total_loss = 0.0
    device = next(self.model.parameters()).device

    data_iter = tqdm(dataset, desc=f"{mode.capitalize()} Epoch", disable=not self.is_master)

    with torch.no_grad() if not training else torch.enable_grad():
        for features in data_iter:
            features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
            loss = self.train_step(features) if training else self.eval_step(features)
            total_loss += loss.item()
            if training:
              self.step +=1
    avg_loss = total_loss / len(dataset)
    if self.is_master:
        print(f"{mode.capitalize()} Loss: {avg_loss:.6f}")
    return avg_loss

  def eval_step(self, features):
      """
      Computes loss on validation batch (no gradient).
      """
      audio = features['audio']
      spectrogram = features['spectrogram']
      N, T = audio.shape
      S = len(self.noise_level) - 1
      device = audio.device
      self.noise_level = self.noise_level.to(device)

      s = torch.randint(1, S + 1, [N], device=audio.device)
      l_a, l_b = self.noise_level[s-1], self.noise_level[s]
      noise_scale = l_a + torch.rand(N, device=audio.device) * (l_b - l_a)
      noise_scale = noise_scale.unsqueeze(1)
      noise = torch.randn_like(audio)
      noisy_audio = noise_scale * audio + (1.0 - noise_scale**2)**0.5 * noise

      predicted = self.model(noisy_audio, spectrogram, noise_scale.squeeze(1))
      loss = self.loss_fn(noise, predicted.squeeze(1))
      return loss

  def plot_losses(self, train_losses, val_losses):
      """
      Plots training and validation losses.
      """
      import matplotlib.pyplot as plt
      plt.figure(figsize=(8,5))
      plt.plot(train_losses, label='Train Loss')
      plt.plot(val_losses, label='Validation Loss')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.title('Training vs Validation Loss')
      plt.legend()
      plt.grid(True)
      plt.tight_layout()
      plt.savefig(os.path.join(self.model_dir, 'loss_curve.png'))
      plt.show()

      save_path = os.path.join(self.model_dir, 'loss_curve.png')
      plt.savefig(save_path)
      plt.close()
      print(f"✅ Loss curve saved to: {save_path}")


  def _write_summary(self, step, features, loss):
    writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
    writer.add_audio('audio/reference', features['audio'][0], step, sample_rate=self.params.sample_rate)
    writer.add_scalar('train/loss', loss, step)
    writer.add_scalar('train/grad_norm', self.grad_norm, step)
    writer.flush()
    self.summary_writer = writer


def _train_impl(replica_id, model, train_dataset, val_dataset, args, params):
  torch.backends.cudnn.benchmark = True
  opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

  learner = WaveGradLearner(args.model_dir, model, train_dataset, opt, params, fp16=args.fp16)
  
  learner.is_master = (replica_id == 0)
  learner.restore_from_checkpoint()

  train_losses = []
  val_losses = []
  num_epochs = args.max_steps // len(train_dataset)

  for epoch in range(num_epochs):
      print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")

      # ---- TRAIN ----
      epoch_train_loss = learner.run_epoch(train_dataset, training=True)
      train_losses.append(epoch_train_loss)

      # ---- VALIDATE ----
      epoch_val_loss = learner.run_epoch(val_dataset, training=False)
      val_losses.append(epoch_val_loss)

      # ---- SAVE ----
      if learner.is_master and (epoch + 1) > 158:
          learner.save_to_checkpoint()
      
  if learner.is_master:
      learner.plot_losses(train_losses, val_losses)



def train(args, params):
  
  train_dataset = dataset_from_path(args.train_data_dirs, params)
  val_dataset = dataset_from_path(args.val_data_dirs, params)
  print("Starting to build datasets...")
  print("Train dataset length:", len(dataset_from_path(args.train_data_dirs, params)))
  print("Val dataset length:", len(dataset_from_path(args.val_data_dirs, params)))

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = WaveGrad(params).cuda()
  _train_impl(0, model, train_dataset, val_dataset, args, params)

