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

import random
from typing import Callable, Optional, Tuple, cast

import torch
import torch.distributed as dist
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor
from torch.distributed import get_process_group_ranks

from cosmos_predict2.conditioner import DataType
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline, IS_PREPROCESSED_KEY, NUM_CONDITIONAL_FRAMES_KEY
from cosmos_predict2.utils.context_parallel import broadcast, broadcast_split_tensor
from configs.multiview.defaults.conditioner import MultiViewCondition
from cosmos_predict2.models.multiview_cfg import MomentumBuffer, adaptive_projected_guidance
from imaginaire.utils import log
from module.res_sampler import COMMON_SOLVER_OPTIONS

TRAIN_SAMPLE_N_VIEWS_KEY = "train_sample_n_views"
TRAIN_SAMPLING_APPLIED_KEY = "train_sampling_applied"
USE_APG_KEY = "use_apg"


class MultiviewVideo2WorldPipeline(Video2WorldPipeline):
    def __init__(self, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)
    
    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        n_views = state.shape[2] // self.tokenizer.get_pixel_num_frames(self.config.state_t)
        if n_views > 4:
            return self.encode_cp(state)
        state = rearrange(state, "B C (V T) H W -> (B V) C T H W", V=n_views)
        encoded_state = super().encode(state)
        encoded_state = rearrange(encoded_state, "(B V) C T H W -> B C (V T) H W", V=n_views)
        return encoded_state

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        n_views = latent.shape[2] // self.config.state_t
        if n_views > 4:
            return self.decode_cp(latent)
        latent = rearrange(latent, "B C (V T) H W -> (B V) C T H W", V=n_views)
        decoded_state = super().decode(latent)
        decoded_state = rearrange(decoded_state, "(B V) C T H W -> B C (V T) H W", V=n_views)
        return decoded_state

    @torch.no_grad()
    def encode_cp(self, state: torch.Tensor) -> torch.Tensor:
        cp_size = len(get_process_group_ranks(parallel_state.get_context_parallel_group()))
        cp_group = parallel_state.get_context_parallel_group()
        n_views = state.shape[2] // self.tokenizer.get_pixel_num_frames(self.config.state_t)
        assert n_views < cp_size, f"n_views must be less than cp_size, got n_views={n_views} and cp_size={cp_size}"
        state_V_B_C_T_H_W = rearrange(state, "B C (V T) H W -> V B C T H W", V=n_views)
        state_input = torch.zeros((cp_size, *state_V_B_C_T_H_W.shape[1:]), **self.tensor_kwargs)
        state_input[0:n_views] = state_V_B_C_T_H_W
        local_state_V_B_C_T_H_W = broadcast_split_tensor(state_input, seq_dim=0, process_group=cp_group)
        local_state = rearrange(local_state_V_B_C_T_H_W, "V B C T H W -> (B V) C T H W")
        encoded_state = super().encode(local_state)
        encoded_state_list = [torch.empty_like(encoded_state) for _ in range(cp_size)]
        dist.all_gather(encoded_state_list, encoded_state, group=cp_group)
        encoded_state = torch.cat(encoded_state_list[0:n_views], dim=2)  # [B, C, V * T, H, W]
        return encoded_state

    @torch.no_grad()
    def decode_cp(self, latent: torch.Tensor) -> torch.Tensor:
        cp_size = len(get_process_group_ranks(parallel_state.get_context_parallel_group()))
        cp_group = parallel_state.get_context_parallel_group()
        n_views = latent.shape[2] // self.config.state_t
        latent_V_B_C_T_H_W = rearrange(latent, "B C (V T) H W -> V B C T H W", V=n_views)
        latent_input = torch.zeros((cp_size, *latent_V_B_C_T_H_W.shape[1:]), **self.tensor_kwargs)
        latent_input[0:n_views] = latent_V_B_C_T_H_W
        local_latent_V_B_C_T_H_W = broadcast_split_tensor(latent_input, seq_dim=0, process_group=cp_group)
        local_latent = rearrange(local_latent_V_B_C_T_H_W, "V B C T H W -> (B V) C T H W")
        decoded_state = super().decode(local_latent)
        decoded_state_list = [torch.empty_like(decoded_state) for _ in range(cp_size)]
        dist.all_gather(decoded_state_list, decoded_state, group=cp_group)
        decoded_state = torch.cat(decoded_state_list[0:n_views], dim=2)  # [B, C, V * T, H, W]
        return decoded_state

    def broadcast_split_for_model_parallelsim(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: MultiViewCondition,
        epsilon_B_C_T_H_W: torch.Tensor,
        sigma_B_T: torch.Tensor,
    ):
        n_views = x0_B_C_T_H_W.shape[2] // self.config.state_t
        x0_B_C_T_H_W = rearrange(x0_B_C_T_H_W, "B C (V T) H W -> (B V) C T H W", V=n_views).contiguous()
        if epsilon_B_C_T_H_W is not None:
            epsilon_B_C_T_H_W = rearrange(epsilon_B_C_T_H_W, "B C (V T) H W -> (B V) C T H W", V=n_views).contiguous()
        reshape_sigma_B_T = False
        if sigma_B_T is not None:
            assert sigma_B_T.ndim == 2, "sigma_B_T should be 2D tensor"
            if sigma_B_T.shape[-1] != 1:
                assert (
                    sigma_B_T.shape[-1] % n_views == 0
                ), f"sigma_B_T temporal dimension T must either be 1 or a multiple of sample_n_views. Got T={sigma_B_T.shape[-1]} and sample_n_views={n_views}"
                sigma_B_T = rearrange(sigma_B_T, "B (V T) -> (B V) T", V=n_views).contiguous()
                reshape_sigma_B_T = True
        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T = super().broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
        )
        x0_B_C_T_H_W = rearrange(x0_B_C_T_H_W, "(B V) C T H W -> B C (V T) H W", V=n_views)
        if epsilon_B_C_T_H_W is not None:
            epsilon_B_C_T_H_W = rearrange(epsilon_B_C_T_H_W, "(B V) C T H W -> B C (V T) H W", V=n_views)
        if reshape_sigma_B_T:
            sigma_B_T = rearrange(sigma_B_T, "(B V) T -> B (V T)", V=n_views)
        return x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T

    def get_data_batch_with_latent_view_indices(self, data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        data_batch = self._preprocess_databatch(
            data_batch
        )  # TODO: move get_data_batch_with_latent_view_indices inside _preprocess_databatch
        num_video_frames_per_view = int(data_batch["num_video_frames_per_view"].cpu().item())
        n_views = data_batch["view_indices"].shape[1] // num_video_frames_per_view
        view_indices_B_V_T = rearrange(data_batch["view_indices"], "B (V T) -> B V T", V=n_views)
        # TODO: Add a check to ensure that view indices are the same for all views or that T=1 for data_batch["view_indices"]
        latent_view_indices_B_V_T = view_indices_B_V_T[:, :, 0 : self.config.state_t]
        latent_view_indices_B_T = rearrange(latent_view_indices_B_V_T, "B V T -> B (V T)")
        data_batch_with_latent_view_indices = data_batch.copy()
        data_batch_with_latent_view_indices["latent_view_indices_B_T"] = latent_view_indices_B_T
        return data_batch_with_latent_view_indices

    def sample_first_n_views_from_data_batch(self, data_batch, n_views):
        new_data_batch = {}
        num_video_frames_per_view = data_batch["num_video_frames_per_view"].cpu().item()
        log.debug(f"Sampling {n_views} views out of {data_batch['sample_n_views'].cpu().item()}")
        log.debug(f"num_video_frames_per_view: {num_video_frames_per_view}")
        new_total_frames = num_video_frames_per_view * n_views
        new_total_t5_dim = 512 * n_views  # TODO: Remove hardcoded value
        new_data_batch["video"] = data_batch["video"][:, :, 0:new_total_frames]
        new_data_batch["view_indices"] = data_batch["view_indices"][:, 0:new_total_frames]
        new_data_batch["sample_n_views"] = 0 * data_batch["sample_n_views"] + n_views
        new_data_batch["fps"] = data_batch["fps"]
        new_data_batch["t5_text_embeddings"] = data_batch["t5_text_embeddings"][:, 0:new_total_t5_dim]
        new_data_batch["t5_text_mask"] = data_batch["t5_text_mask"][:, 0:new_total_t5_dim]
        split_captions = data_batch["ai_caption"][0].split(" -- ")
        assert len(split_captions) == 7, f"Expected 7 view captions, got {len(split_captions)}"
        new_data_batch["ai_caption"] = [" -- ".join(split_captions[0:n_views])]
        new_data_batch["n_orig_video_frames_per_view"] = data_batch["n_orig_video_frames_per_view"]
        assert (
            data_batch["ref_cam_view_idx_sample_position"].item() == -1
        ), f"ref_cam_view_idx_sample_position is not supported by batch sampling, got {data_batch['ref_cam_view_idx_sample_position']}"
        new_data_batch["ref_cam_view_idx_sample_position"] = data_batch["ref_cam_view_idx_sample_position"]
        new_data_batch["camera_keys_selection"] = data_batch["camera_keys_selection"][0:n_views]
        new_data_batch["view_indices_selection"] = data_batch["view_indices_selection"]
        for key in [
            "__url__",
            "__key__",
            "__t5_url__",
            "image_size",
            "num_video_frames_per_view",
            "aspect_ratio",
            "padding_mask",
        ]:
            new_data_batch[key] = data_batch[key]
        old_keys = set(list(data_batch.keys()))
        new_keys = set(list(new_data_batch.keys()))
        diff = old_keys.difference(new_keys)
        assert old_keys == new_keys, f"Expected old keys to equal new keys. Difference {diff}"
        # misc.to(new_data_batch, **self.tensor_kwargs)
        return new_data_batch

    def _preprocess_databatch(self, data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # TODO: consider moving latent_view_indices creation to this function
        if TRAIN_SAMPLING_APPLIED_KEY in data_batch and data_batch[TRAIN_SAMPLING_APPLIED_KEY] is True:
            return data_batch
        if self.config.train_sample_views_range is not None:
            min_views, max_views = self.config.train_sample_views_range
            log.debug(f"Randomly sampling {min_views} to {max_views} views")
            if TRAIN_SAMPLE_N_VIEWS_KEY in data_batch:
                train_sample_n_views = data_batch[TRAIN_SAMPLE_N_VIEWS_KEY]
                log.debug(f"Using {TRAIN_SAMPLE_N_VIEWS_KEY} from data batch: {train_sample_n_views}")
                if train_sample_n_views < 1:  # No sampling is applied
                    return data_batch
            else:
                # Sample and broadcast n_views across cp_groups so all ranks in the same group sample the same number of views
                train_sample_n_views = random.randint(min_views, max_views)
                train_sample_n_views = broadcast(train_sample_n_views, parallel_state.get_context_parallel_group())
                log.debug(f"Sampled and broadcasted {train_sample_n_views=} ")
            log.debug(f"Sampling {train_sample_n_views} views out of {data_batch['sample_n_views'].cpu().item()}")
            data_batch = self.sample_first_n_views_from_data_batch(data_batch, train_sample_n_views)
            data_batch[TRAIN_SAMPLING_APPLIED_KEY] = True
        return data_batch

    def _normalize_video_databatch_inplace(self, data_batch: dict[str, Tensor], input_key: str = None) -> None:
        data_batch = self._preprocess_databatch(data_batch)
        input_key = self.input_data_key if input_key is None else input_key
        is_preprocessed = IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True
        # TODO: Consider simply returning the data_batch if it is already preprocessed. Assuming self.config.resize_online is already applied
        num_video_frames_per_view = (
            self.tokenizer.get_pixel_num_frames(self.config.state_t)
            if is_preprocessed
            else data_batch["num_video_frames_per_view"]
        )
        if isinstance(num_video_frames_per_view, torch.Tensor):
            num_video_frames_per_view = int(num_video_frames_per_view.cpu().item())
        n_views = data_batch[input_key].shape[2] // num_video_frames_per_view
        if input_key in data_batch:
            data_batch[input_key] = rearrange(data_batch[input_key], "B C (V T) H W -> (B V) C T H W", V=n_views)
            super()._normalize_video_databatch_inplace(data_batch, input_key)
            data_batch[input_key] = rearrange(data_batch[input_key], "(B V) C T H W -> B C (V T) H W", V=n_views)

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor]) -> Tuple[Tensor, Tensor, MultiViewCondition]:
        data_batch_with_latent_view_indices = self.get_data_batch_with_latent_view_indices(data_batch)
        raw_state, latent_state, condition = super(Video2WorldPipeline, self).get_data_and_condition(
            data_batch_with_latent_view_indices
        )
        condition = cast(MultiViewCondition, condition)
        condition = condition.set_video_condition(
            state_t=self.config.state_t,
            gt_frames=latent_state.to(**self.tensor_kwargs),
            condition_locations=self.config.condition_locations,
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames_per_view,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames_per_view,
            num_conditional_frames_per_view=None,
            view_condition_dropout_max=self.config.view_condition_dropout_max,
        )
        return raw_state, latent_state, condition

    def get_x0_fn_from_batch(
        self,
        data_batch: dict[str, torch.Tensor],
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        # TODO: update so latent_view_indices_B_T is already in data_batch when passed to this function
        data_batch_with_latent_view_indices = self.get_data_batch_with_latent_view_indices(data_batch)
        if NUM_CONDITIONAL_FRAMES_KEY in data_batch_with_latent_view_indices:
            num_conditional_frames = data_batch_with_latent_view_indices[NUM_CONDITIONAL_FRAMES_KEY]
        else:
            num_conditional_frames = 1

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(
                data_batch_with_latent_view_indices
            )
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch_with_latent_view_indices)

        is_image_batch = self.is_image_batch(data_batch_with_latent_view_indices)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        _, x0, _ = self.get_data_and_condition(data_batch_with_latent_view_indices)
        # override condition with inference mode; num_conditional_frames used Here!
        condition = condition.set_video_condition(
            state_t=self.config.state_t,
            gt_frames=x0,
            condition_locations=self.config.condition_locations,
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames_per_view,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames_per_view,
            num_conditional_frames_per_view=num_conditional_frames,
            view_condition_dropout_max=0,
        )
        uncondition = uncondition.set_video_condition(
            state_t=self.config.state_t,
            gt_frames=x0,
            condition_locations=self.config.condition_locations,
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames_per_view,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames_per_view,
            num_conditional_frames_per_view=num_conditional_frames,
            view_condition_dropout_max=0,
        )
        condition = condition.edit_for_inference(
            is_cfg_conditional=True,
            condition_locations=self.config.condition_locations,
            num_conditional_frames_per_view=num_conditional_frames,
        )
        uncondition = uncondition.edit_for_inference(
            is_cfg_conditional=False,
            condition_locations=self.config.condition_locations,
            num_conditional_frames_per_view=num_conditional_frames,
        )
        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)

        if parallel_state.is_initialized():
            pass
        else:
            assert (
                not self.net.is_context_parallel_enabled
            ), "parallel_state is not initialized, context parallel should be turned off."
        momentum_buffer = MomentumBuffer(self.config.apg_momentum)

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            cond_x0 = self.denoise(noise_x, sigma, condition).x0
            uncond_x0 = self.denoise(noise_x, sigma, uncondition).x0

            if (
                USE_APG_KEY in data_batch_with_latent_view_indices
                and data_batch_with_latent_view_indices[USE_APG_KEY] is True
            ):
                raw_x0 = adaptive_projected_guidance(
                    cond_x0,
                    uncond_x0,
                    guidance,
                    momentum_buffer=momentum_buffer,
                    eta=self.config.apg_eta,
                    norm_threshold=self.config.apg_norm_threshold,
                )
            else:
                raw_x0 = cond_x0 + guidance * (cond_x0 - uncond_x0)
            if "guided_image" in data_batch_with_latent_view_indices:
                # replacement trick that enables inpainting with base model
                assert (
                    "guided_mask" in data_batch_with_latent_view_indices
                ), "guided_mask should be in data_batch if guided_image is present"
                guide_image = data_batch_with_latent_view_indices["guided_image"]
                guide_mask = data_batch_with_latent_view_indices["guided_mask"]
                raw_x0 = guide_mask * guide_image + (1 - guide_mask) * raw_x0
            return raw_x0

        return x0_fn

    def generate_samples_from_batch(
        self,
        data_batch: dict[str, torch.Tensor],
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        solver_option: COMMON_SOLVER_OPTIONS = "2ab",
        x_sigma_max: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        data_batch_with_latent_view_indices = self.get_data_batch_with_latent_view_indices(data_batch)
        process_group = parallel_state.get_context_parallel_group()
        cp_size = len(get_process_group_ranks(process_group))
        samples_B_C_T_H_W = super().generate_samples_from_batch(
            data_batch_with_latent_view_indices,
            guidance,
            seed,
            state_shape,
            n_sample,
            is_negative_prompt,
            num_steps,
            solver_option,
            x_sigma_max,
        )
        if cp_size > 1:
            samples_B_C_T_H_W = rearrange(
                samples_B_C_T_H_W, "B C (c V T) H W -> B C (V c T) H W", c=cp_size, T=self.config.state_t // cp_size
            )
        return samples_B_C_T_H_W
