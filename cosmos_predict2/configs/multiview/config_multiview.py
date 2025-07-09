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

from dataclasses import field
from typing import Tuple

import attrs

from configs.multiview.defaults.conditioner import ConditionLocationList
from configs.base.config_video2world import Video2WorldPipelineConfig
from imaginaire.config import make_freezable


@make_freezable
@attrs.define(slots=False)
class MultiviewVideo2WorldModelConfig(Video2WorldPipelineConfig):
    min_num_conditional_frames_per_view: int = 1
    max_num_conditional_frames_per_view: int = 2
    train_sample_views_range: Tuple[int, int] | None = None
    condition_locations: ConditionLocationList = field(default_factory=lambda: ConditionLocationList([]))
    state_t: int = 0
    view_condition_dropout_max: int = 0
    apg_momentum: float = -0.75
    apg_eta: float = 0.0
    apg_norm_threshold: float = 7.5
