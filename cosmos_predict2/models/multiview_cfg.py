# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def project(
    v0: torch.Tensor,  # [B, C, H, W]
    v1: torch.Tensor,  # [B, C, H, W]
):
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    ndims = v0.ndim
    v1 = torch.nn.functional.normalize(v1, dim=list(range(1, ndims)))
    v0_parallel = (v0 * v1).sum(dim=list(range(1, ndims)), keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def adaptive_projected_guidance(
    pred_cond: torch.Tensor,  # [B, C, H, W]
    pred_uncond: torch.Tensor,  # [B, C, H, W]
    guidance_scale: float,
    momentum_buffer: MomentumBuffer = None,
    eta: float = 1.0,
    norm_threshold: float = 0.0,
):
    diff = pred_cond - pred_uncond
    ndims = diff.ndim
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=list(range(1, ndims)), keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    diff_parallel, diff_orthogonal = project(diff, pred_cond)
    normalized_update = diff_orthogonal + eta * diff_parallel
    pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
    return pred_guided
