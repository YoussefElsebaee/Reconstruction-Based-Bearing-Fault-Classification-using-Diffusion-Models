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


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
      super(AttrDict, self).__init__(*args, **kwargs)
      self.__dict__ = self

  def override(self, attrs):
    if isinstance(attrs, dict):
      self.__dict__.update(**attrs)
    elif isinstance(attrs, (list, tuple, set)):
      for attr in attrs:
        self.override(attr)
    elif attrs is None:
      pass
    else:
      raise NotImplementedError
    return self


params = AttrDict(
    # Training params
    batch_size=1,
    learning_rate=2e-4, #try a smaller one
    max_grad_norm=1.0,

    # Data params
    sample_rate=12000,
    hop_samples=300,  # Don't change this. Really.
    crop_mel_frames=40,

    # Model params
    noise_schedule = 0.5*(1-np.cos(np.linspace(0,np.pi,1000)))*(0.01-1e-6) + 1e-6
)
# steps=1000, beta_min=1e-6, beta_max=0.01
# linear: np.linspace(1e-6, 0.01, 1000)
# cosine: 0.5*(1-np.cos(np.linspace(0,np.pi,1000)))*(0.01-1e-6) + 1e-6
# quadratic: (np.linspace(0,1,1000)**2)*(0.01-1e-6) + 1e-6
# exponential: 1e-6 * ((0.01/1e-6) ** np.linspace(0,1,1000))
# sigmoid: 1e-6 + (0.01 - 1e-6) * (1 / (1 + np.exp(-5 * (np.linspace(0, 1, 1000) - 0.5))))
# linear_sigmoid: 1e-6 + (0.01 - 1e-6) * (0.5 * (1 / (1 + np.exp(-4 * (np.linspace(0, 1, 1000) - 0.5)))) + 0.5 * np.linspace(0, 1, 1000))