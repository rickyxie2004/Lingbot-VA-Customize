# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import os
import sys
import time
from contextlib import contextmanager
from functools import partial
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model
from distributed.util import _configure_model, init_distributed
from modules.utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    run_async_server_mode,
    save_async,
)


class LatentMSESpeculativeVerifier:

    def __init__(self, threshold):
        self.threshold = float(threshold)

    def verify(self, real_latent, pred_latent):
        score = torch.mean((real_latent.float() - pred_latent.float()) ** 2).item()
        return score, bool(score < self.threshold)


class ActionDenoiseSpeculativeVerifier:

    def __init__(self, threshold):
        self.threshold = float(threshold)

    def verify(self, pred_velocity, actual_velocity, action_mask=None):
        pred_velocity = pred_velocity.float()
        actual_velocity = actual_velocity.float()
        if action_mask is not None:
            pred_velocity = pred_velocity[:, action_mask]
            actual_velocity = actual_velocity[:, action_mask]
        score = torch.mean((pred_velocity - actual_velocity) ** 2).item()
        return score, bool(score < self.threshold)


class _CudaActionLatencyProfiler:

    def __init__(self, enabled, device, profile_steps=False):
        self.enabled = bool(enabled) and torch.cuda.is_available()
        self.device = device
        self.profile_steps = bool(profile_steps)
        self._records = {}
        self._order = []

    @contextmanager
    def record(self, name):
        if not self.enabled:
            yield
            return
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(self.device):
            start_event.record()
            try:
                yield
            finally:
                end_event.record()
                end_event.synchronize()
                self._add(name, start_event.elapsed_time(end_event))

    def _add(self, name, elapsed_ms):
        if name not in self._records:
            self._records[name] = [0.0, 0]
            self._order.append(name)
        self._records[name][0] += float(elapsed_ms)
        self._records[name][1] += 1

    def summary(self, name, **metadata):
        if not self.enabled:
            return None
        records = []
        for record_name in self._order:
            total_ms, count = self._records[record_name]
            records.append({
                'name': record_name,
                'total_ms': round(total_ms, 3),
                'count': count,
                'avg_ms': round(total_ms / max(count, 1), 3),
            })
        total_ms = sum(
            record['total_ms'] for record in records
            if not record['name'].endswith('.loop')
        )
        return {
            'name': name,
            'device': str(self.device),
            'total_recorded_cuda_ms': round(total_ms, 3),
            'records': records,
            **metadata,
        }


class VA_Server:

    def __init__(self, job_config):
        self.cache_name = 'pos'
        self.job_config = job_config
        self.save_root = job_config.save_root
        self.dtype = job_config.param_dtype
        self.device = torch.device(f"cuda:{job_config.local_rank}")
        self.enable_offload = getattr(job_config, 'enable_offload', True)  # offload vae & text_encoder to save vram

        self.scheduler = FlowMatchScheduler(shift=self.job_config.snr_shift,
                                            sigma_min=0.0,
                                            extra_one_step=True)
        self.action_scheduler = FlowMatchScheduler(
            shift=self.job_config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        self.vae = load_vae(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'vae'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)

        self.tokenizer = load_tokenizer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'tokenizer'), )

        self.text_encoder = load_text_encoder(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'text_encoder'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )
        transformer_base_path = getattr(job_config, 'trained_transformer_path', None)
        if transformer_base_path is not None:
            logger.info('Using trained transformer!')
        else:
            transformer_base_path = job_config.wan22_pretrained_model_name_or_path
        self.transformer = load_transformer(
            os.path.join(transformer_base_path,
                         'transformer'),
            torch_dtype=self.dtype,
            torch_device=self.device,
            attn_mode="torch"
        )
        shard_fn = shard_model
        self.transformer = _configure_model(model=self.transformer,
                                            shard_fn=shard_fn,
                                            param_dtype=self.dtype,
                                            device=self.device,
                                            eval_mode=True,
                                            )

        self.env_type = job_config.env_type
        self.streaming_vae_half = None
        if self.env_type == 'robotwin_tshape':
            vae_half = load_vae(
                os.path.join(job_config.wan22_pretrained_model_name_or_path,
                             'vae'),
                torch_dtype=self.dtype,
                torch_device='cpu' if self.enable_offload else self.device,
            )
            self.streaming_vae_half = WanVAEStreamingWrapper(vae_half)

        if getattr(job_config, 'compile_infer', False):
            logger.info("Enable torch.compile for VA_Server._infer")
            self._encode_obs = torch.compiler.disable(
                self._encode_obs,
                reason="Keep read-only numpy websocket inputs out of Dynamo.",
            )
            self.postprocess_action = torch.compiler.disable(
                self.postprocess_action,
                reason="Keep numpy action output conversion out of Dynamo.",
            )
            self._decode_pred_video = torch.compiler.disable(
                self._decode_pred_video,
                reason="Keep optional VAE video decoding out of Dynamo.",
            )
            self._infer = torch.compile(
                self._infer,
                dynamic=False,
                fullgraph=False,
                mode="default",
            )

    def _get_t5_prompt_embeds(
        self,
        prompt=None,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        text_encoder_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(text_input_ids.to(text_encoder_device),
                                          mask.to(text_encoder_device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack([
            torch.cat(
                [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
                                    dim=0)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt,
                                           seq_len, -1)

        return prompt_embeds.to(device)

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=226,
        device=None,
        dtype=None,
    ):
        r"""
        TODO
        """
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(
                negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(
                    negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}.")
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        return prompt_embeds, negative_prompt_embeds

    def normalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1,
                                         1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1,
                                       1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    def preprocess_action(self, action):
        action_model_input = torch.from_numpy(action)
        CA, FA, HA = action_model_input.shape  # C, F, H
        action_model_input_paded = F.pad(action_model_input,
                                         [0, 0, 0, 0, 0, 1],
                                         mode='constant',
                                         value=0)

        action_model_input = action_model_input_paded[
            self.job_config.inverse_used_action_channel_ids]

        if self.action_norm_method == 'quantiles':
            action_model_input = (action_model_input - self.actions_q01) / (
                self.actions_q99 - self.actions_q01 + 1e-6) * 2. - 1.
        else:
            raise NotImplementedError
        return action_model_input.unsqueeze(0).unsqueeze(-1)  # B, C, F, H, W

    def postprocess_action(self, action):
        action = action.cpu()  # B, C, F, H, W

        action = action[0, ..., 0]  #C, F, H
        if self.action_norm_method == 'quantiles':
            action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 +
                                         1e-6) + self.actions_q01
        else:
            raise NotImplementedError
        action = action.squeeze(0).detach().cpu().numpy()
        return action[self.job_config.used_action_channel_ids]
    
    def _postprocess_action_batch(self, action):
        action = action.cpu()[..., 0]  # B, C, F, H
        if self.action_norm_method == 'quantiles':
            actions_q01 = self.actions_q01.cpu().view(1, -1, 1, 1)
            actions_q99 = self.actions_q99.cpu().view(1, -1, 1, 1)
            action = (action + 1) / 2 * (actions_q99 - actions_q01 + 1e-6) + actions_q01
        else:
            raise NotImplementedError
        action = action.detach().cpu().numpy()
        return action[:, self.job_config.used_action_channel_ids]

    def _clone_transformer_cache(self):
        cache_snapshot = []
        for block in self.transformer.blocks:
            attn_caches = block.attn1.attn_caches
            cache = None if attn_caches is None else attn_caches.get(self.cache_name)
            if cache is None:
                cache_snapshot.append(None)
            else:
                cache_snapshot.append({
                    key: value.clone() if torch.is_tensor(value) else value
                    for key, value in cache.items()
                })
        return cache_snapshot

    def _restore_transformer_cache(self, cache_snapshot):
        for block, cache in zip(self.transformer.blocks, cache_snapshot):
            if block.attn1.attn_caches is not None:
                block.attn1.attn_caches[self.cache_name] = cache

    def _expand_transformer_cache_batch(self, batch_size):
        cache_batch_size = batch_size * (2 if self.use_cfg else 1)
        for block in self.transformer.blocks:
            attn_caches = block.attn1.attn_caches
            if attn_caches is None or attn_caches.get(self.cache_name) is None:
                continue
            cache = attn_caches[self.cache_name]
            if cache.get('k') is not None:
                if self.use_cfg and cache['k'].shape[0] >= 2:
                    cache['k'] = torch.cat([
                        cache['k'][:1].clone().repeat(batch_size, 1, 1, 1),
                        cache['k'][1:2].clone().repeat(batch_size, 1, 1, 1),
                    ], dim=0)
                else:
                    cache['k'] = cache['k'][:1].clone().repeat(cache_batch_size, 1, 1, 1)
            if cache.get('v') is not None:
                if self.use_cfg and cache['v'].shape[0] >= 2:
                    cache['v'] = torch.cat([
                        cache['v'][:1].clone().repeat(batch_size, 1, 1, 1),
                        cache['v'][1:2].clone().repeat(batch_size, 1, 1, 1),
                    ], dim=0)
                else:
                    cache['v'] = cache['v'][:1].clone().repeat(cache_batch_size, 1, 1, 1)

    def _repeat_input_for_cfg(self, input_dict):
        batch_size = input_dict['noisy_latents'].shape[0]
        if self.use_cfg:
            input_dict['noisy_latents'] = torch.cat(
                [input_dict['noisy_latents'], input_dict['noisy_latents']], dim=0)
            input_dict['text_emb'] = torch.cat([
                self.prompt_embeds.to(self.dtype).clone().repeat(batch_size, 1, 1),
                self.negative_prompt_embeds.to(self.dtype).clone().repeat(batch_size, 1, 1),
            ], dim=0)
            input_dict['grid_id'] = input_dict['grid_id'][None].repeat(batch_size * 2, 1, 1)
            input_dict['timesteps'] = input_dict['timesteps'][None].repeat(batch_size * 2, 1)
        else:
            input_dict['text_emb'] = self.prompt_embeds.to(self.dtype).clone().repeat(batch_size, 1, 1)
            input_dict['grid_id'] = input_dict['grid_id'][None].repeat(batch_size, 1, 1)
            input_dict['timesteps'] = input_dict['timesteps'][None].repeat(batch_size, 1)
        return input_dict

    def _prepare_latent_input(self,
                              latent_model_input,
                              action_model_input,
                              latent_t=0,
                              action_t=0,
                              latent_cond=None,
                              action_cond=None,
                              frame_st_id=0,
                              patch_size=(1, 2, 2)):
        logger.info(f"FRAME START ID: {frame_st_id}")
        input_dict = dict()
        if latent_model_input is not None:
            input_dict['latent_res_lst'] = {
                'noisy_latents':
                latent_model_input,
                'timesteps':
                torch.ones([latent_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * latent_t,
                'grid_id':
                get_mesh_id(latent_model_input.shape[-3] // patch_size[0],
                            latent_model_input.shape[-2] // patch_size[1],
                            latent_model_input.shape[-1] // patch_size[2], 0,
                            1, frame_st_id).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                input_dict['latent_res_lst'][
                    'noisy_latents'][:, :, 0:1] = latent_cond[:, :, 0:1]
                input_dict['latent_res_lst']['timesteps'][0:1] *= 0

        if action_model_input is not None:
            input_dict['action_res_lst'] = {
                'noisy_latents':
                action_model_input,
                'timesteps':
                torch.ones([action_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * action_t,
                'grid_id':
                get_mesh_id(action_model_input.shape[-3],
                            action_model_input.shape[-2],
                            action_model_input.shape[-1],
                            1,
                            1,
                            frame_st_id,
                            action=True).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }

            if action_cond is not None:
                input_dict['action_res_lst'][
                    'noisy_latents'][:, :, 0:1] = action_cond[:, :, 0:1]
                input_dict['action_res_lst']['timesteps'][0:1] *= 0
            input_dict['action_res_lst']['noisy_latents'][:, ~self.
                                                          action_mask] *= 0
        return input_dict

    def _encode_obs(self, obs):
        images = obs['obs']
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            return None
        videos = []
        for k_i, k in enumerate(self.job_config.obs_cam_keys):
            if self.env_type == 'robotwin_tshape':
                if k_i == 0:  # camera high
                    height_i, width_i = self.height, self.width
                else:
                    height_i, width_i = self.height // 2, self.width // 2
            else:
                height_i, width_i = self.height, self.width

            history_video_k = torch.from_numpy(
                np.stack([each[k]
                          for each in images])).float().permute(3, 0, 1, 2)
            history_video_k = F.interpolate(history_video_k,
                                            size=(height_i, width_i),
                                            mode='bilinear',
                                            align_corners=False).unsqueeze(0)
            videos.append(history_video_k)

        if self.env_type == 'robotwin_tshape':
            videos_high = videos[0] / 255.0 * 2.0 - 1.0
            videos_left_and_right = torch.cat(videos[1:],
                                              dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            enc_out_high = self.streaming_vae.encode_chunk(
                videos_high.to(vae_device).to(self.dtype))
            enc_out_left_and_right = self.streaming_vae_half.encode_chunk(
                videos_left_and_right.to(vae_device).to(self.dtype))
            enc_out = torch.cat([
                torch.cat(enc_out_left_and_right.split(1, dim=0), dim=-1),
                enc_out_high
            ],
                                dim=-2)
        else:
            videos = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            videos_chunk = videos.to(vae_device).to(self.dtype)
            enc_out = self.streaming_vae.encode_chunk(videos_chunk)

        mu, logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
        return video_latent.to(self.device)

    def _reset(self, prompt=None):
        logger.info('Reset.')
        self.use_cfg = (self.job_config.guidance_scale > 1) or (self.job_config.action_guidance_scale > 1)
        #### Reset all parameters
        self.frame_st_id = 0
        self.init_latent = None
        #### clean vae and transformer cache
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()

        self.action_per_frame = self.job_config.action_per_frame
        self.height, self.width = self.job_config.height, self.job_config.width

        if self.env_type == 'robotwin_tshape':
            self.latent_height, self.latent_width = (
                (self.height // 16) * 3) // 2, self.width // 16
            self.streaming_vae_half.clear_cache()
        else:
            self.latent_height, self.latent_width = self.height // 16, self.width // 16 * len(
                self.job_config.obs_cam_keys)

        patch_size = self.job_config.patch_size
        cache_frame_chunk_size = self.job_config.frame_chunk_size
        if getattr(self.job_config, 'enable_speculative_verifier', False):
            speculative_frame_chunk_size = int(getattr(
                self.job_config, 'speculative_frame_chunk_size', cache_frame_chunk_size))
            replan_frame_chunk_size = int(getattr(
                self.job_config, 'speculative_replan_frame_chunk_size', -1))
            if replan_frame_chunk_size <= 0:
                replan_frame_chunk_size = speculative_frame_chunk_size
            speculative_cache_chunk_size = max(
                speculative_frame_chunk_size, replan_frame_chunk_size) + 1
            cache_frame_chunk_size = max(cache_frame_chunk_size, speculative_cache_chunk_size)
        latent_token_per_chunk = (cache_frame_chunk_size *
                                  self.latent_height * self.latent_width) // (
                                      patch_size[0] * patch_size[1] *
                                      patch_size[2])
        action_token_per_chunk = cache_frame_chunk_size * self.action_per_frame
        self.transformer.create_empty_cache(self.cache_name,
                                            self.job_config.attn_window,
                                            latent_token_per_chunk,
                                            action_token_per_chunk,
                                            dtype=self.dtype,
                                            device=self.device,
                                            batch_size = 2 if self.use_cfg else 1
                                            )

        self.action_mask = torch.zeros([self.job_config.action_dim]).bool()
        self.action_mask[self.job_config.used_action_channel_ids] = True

        self.actions_q01 = torch.tensor(self.job_config.norm_stat['q01'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(self.job_config.norm_stat['q99'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.action_norm_method = self.job_config.action_norm_method

        ##### get prompt
        if prompt is None:
            self.prompt_embeds = self.negative_prompt_embeds = None
        else:
            self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=None,
                do_classifier_free_guidance=self.job_config.guidance_scale > 1,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=512,
                device=self.device,
                dtype=self.dtype,
            )

        self.exp_name = f"{prompt}_{time.strftime('%Y%m%d_%H%M%S')}" if prompt else "default"
        self.exp_save_root = os.path.join(self.save_root, 'real', self.exp_name)
        os.makedirs(self.exp_save_root, exist_ok=True)
        torch.cuda.empty_cache()

    def _new_latency_profiler(self):
        return _CudaActionLatencyProfiler(
            getattr(self.job_config, 'profile_action_latency', False),
            self.device,
            getattr(self.job_config, 'profile_action_latency_steps', False),
        )

    def _log_latency_profile(self, profile):
        if not profile:
            return
        parts = []
        for record in profile['records']:
            suffix = f" x{record['count']}" if record['count'] > 1 else ''
            parts.append(f"{record['name']}={record['total_ms']:.2f}ms{suffix}")
        logger.info(
            f"[ActionProfile] {profile['name']} frame_st_id={profile.get('frame_st_id')} "
            f"recorded_cuda={profile['total_recorded_cuda_ms']:.2f}ms | " +
            ' | '.join(parts)
        )

    def _infer(
        self,
        obs,
        frame_st_id=0,
        video_num_inference_steps=None,
        action_num_inference_steps=None,
        frame_chunk_size=None,
    ):
        profiler = self._new_latency_profiler()
        frame_chunk_size = int(frame_chunk_size or self.job_config.frame_chunk_size)
        if frame_st_id == 0:
            with profiler.record('obs.encode'):
                init_latent = self._encode_obs(obs)
            self.init_latent = init_latent

        with profiler.record('noise.init'):
            latents = torch.randn(1,
                                  48,
                                  frame_chunk_size,
                                  self.latent_height,
                                  self.latent_width,
                                  device=self.device,
                                  dtype=self.dtype)
            actions = torch.randn(1,
                                  self.job_config.action_dim,
                                  frame_chunk_size,
                                  self.action_per_frame,
                                  1,
                                  device=self.device,
                                  dtype=self.dtype)

        video_inference_step = (
            int(video_num_inference_steps)
            if video_num_inference_steps is not None
            else self.job_config.num_inference_steps
        )
        action_inference_step = (
            int(action_num_inference_steps)
            if action_num_inference_steps is not None
            else self.job_config.action_num_inference_steps
        )
        video_step = self.job_config.video_exec_step

        self.scheduler.set_timesteps(video_inference_step)
        self.action_scheduler.set_timesteps(action_inference_step)
        timesteps = self.scheduler.timesteps
        action_timesteps = self.action_scheduler.timesteps

        timesteps = F.pad(timesteps, (0, 1), mode='constant', value=0)

        if video_step != -1:
            timesteps = timesteps[:video_step]

        action_timesteps = F.pad(
            action_timesteps,
            (0,
             1),  # pad 1 element at the end (right side) of the last dimension
            mode='constant',
            value=0)

        with (
                torch.no_grad(),
        ):
            # 1. Video Generation Loop
            with profiler.record('video.loop'):
                for i, t in enumerate(tqdm(timesteps)):
                    step_suffix = f'.step_{i:02d}' if profiler.profile_steps else ''
                    last_step = i == len(timesteps) - 1
                    latent_cond = init_latent[:, :, 0:1].to(
                        self.dtype) if frame_st_id == 0 else None
                    with profiler.record(f'video.prepare_input{step_suffix}'):
                        input_dict = self._prepare_latent_input(
                            latents,
                            None,
                            t,
                            t,
                            latent_cond,
                            None,
                            frame_st_id=frame_st_id)

                    with profiler.record(f'video.transformer{step_suffix}'):
                        video_noise_pred = self.transformer(
                            self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                            update_cache=1 if last_step else 0,
                            cache_name=self.cache_name,
                            action_mode=False)

                    if not last_step or video_step != -1:
                        with profiler.record(f'video.scheduler_step{step_suffix}'):
                            video_noise_pred = data_seq_to_patch(
                                self.job_config.patch_size, video_noise_pred,
                                frame_chunk_size, self.latent_height,
                                self.latent_width, batch_size=2 if self.use_cfg else 1)
                            if self.job_config.guidance_scale > 1:
                                video_noise_pred = video_noise_pred[1:] + self.job_config.guidance_scale * (video_noise_pred[:1] - video_noise_pred[1:])
                            else:
                                video_noise_pred = video_noise_pred[:1]
                            latents = self.scheduler.step(video_noise_pred,
                                                          t,
                                                          latents,
                                                          return_dict=False)

                    latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else latents[:, :, 0:1]

            with profiler.record('action.loop'):
                for i, t in enumerate(tqdm(action_timesteps)):
                    step_suffix = f'.step_{i:02d}' if profiler.profile_steps else ''
                    last_step = i == len(action_timesteps) - 1
                    action_cond = torch.zeros(
                        [
                            1, self.job_config.action_dim, 1,
                            self.action_per_frame, 1
                        ],
                        device=self.device,
                        dtype=self.dtype) if frame_st_id == 0 else None

                    with profiler.record(f'action.prepare_input{step_suffix}'):
                        input_dict = self._prepare_latent_input(
                            None,
                            actions,
                            t,
                            t,
                            None,
                            action_cond,
                            frame_st_id=frame_st_id)
                    with profiler.record(f'action.transformer{step_suffix}'):
                        action_noise_pred = self.transformer(
                            self._repeat_input_for_cfg(input_dict['action_res_lst']),
                            update_cache=1 if last_step else 0,
                            cache_name=self.cache_name,
                            action_mode=True)

                    if not last_step:
                        with profiler.record(f'action.scheduler_step{step_suffix}'):
                            action_noise_pred = rearrange(action_noise_pred,
                                                          'b (f n) c -> b c f n 1',
                                                          f=frame_chunk_size)
                            if self.job_config.action_guidance_scale > 1:
                                action_noise_pred = action_noise_pred[1:] + self.job_config.action_guidance_scale * (action_noise_pred[:1] - action_noise_pred[1:])
                            else:
                                action_noise_pred = action_noise_pred[:1]
                            actions = self.action_scheduler.step(action_noise_pred,
                                                                 t,
                                                                 actions,
                                                                 return_dict=False)

                    actions[:, :, 0:1] = action_cond if frame_st_id == 0 else actions[:, :, 0:1]

        with profiler.record('action.mask'):
            actions[:, ~self.action_mask] *= 0

        save_async(latents, os.path.join(self.exp_save_root, f'latents_{frame_st_id}.pt'))
        save_async(actions, os.path.join(self.exp_save_root, f'actions_{frame_st_id}.pt'))

        with profiler.record('action.postprocess'):
            actions = self.postprocess_action(actions)
        pred_video = None
        with profiler.record('video.decode'):
            pred_video = self._decode_pred_video(latents)
        with profiler.record('cuda.empty_cache'):
            torch.cuda.empty_cache()

        profile = profiler.summary('action_generation', frame_st_id=frame_st_id)
        self._log_latency_profile(profile)
        return actions, latents, profile, pred_video

    def _infer_video_branch_for_action_diversity(self, obs, frame_st_id=0, batch_size=1):
        profiler = self._new_latency_profiler()
        frame_chunk_size = self.job_config.frame_chunk_size
        if frame_st_id == 0 and self.init_latent is None:
            with profiler.record('diversity.obs.encode'):
                self.init_latent = self._encode_obs(obs)

        with profiler.record('diversity.video_noise.init'):
            latents = torch.randn(batch_size,
                                  48,
                                  frame_chunk_size,
                                  self.latent_height,
                                  self.latent_width,
                                  device=self.device,
                                  dtype=self.dtype)

        video_inference_step = self.job_config.num_inference_steps
        video_step = self.job_config.video_exec_step
        self.scheduler.set_timesteps(video_inference_step)
        timesteps = F.pad(self.scheduler.timesteps, (0, 1), mode='constant', value=0)
        if video_step != -1:
            timesteps = timesteps[:video_step]

        with torch.no_grad():
            with profiler.record('diversity.video.loop'):
                for i, t in enumerate(tqdm(timesteps)):
                    step_suffix = f'.step_{i:02d}' if profiler.profile_steps else ''
                    last_step = i == len(timesteps) - 1
                    latent_cond = self.init_latent[:, :, 0:1].to(
                        self.dtype).repeat(batch_size, 1, 1, 1, 1) if frame_st_id == 0 else None
                    with profiler.record(f'diversity.video.prepare_input{step_suffix}'):
                        input_dict = self._prepare_latent_input(
                            latents,
                            None,
                            t,
                            t,
                            latent_cond,
                            None,
                            frame_st_id=frame_st_id)
                    with profiler.record(f'diversity.video.transformer{step_suffix}'):
                        video_noise_pred = self.transformer(
                            self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                            update_cache=1 if last_step else 0,
                            cache_name=self.cache_name,
                            action_mode=False)
                    if not last_step or video_step != -1:
                        with profiler.record(f'diversity.video.scheduler_step{step_suffix}'):
                            video_noise_pred = data_seq_to_patch(
                                self.job_config.patch_size, video_noise_pred,
                                frame_chunk_size, self.latent_height,
                                self.latent_width, batch_size=batch_size * (2 if self.use_cfg else 1))
                            if self.job_config.guidance_scale > 1:
                                video_noise_pred = video_noise_pred[batch_size:] + self.job_config.guidance_scale * (video_noise_pred[:batch_size] - video_noise_pred[batch_size:])
                            else:
                                video_noise_pred = video_noise_pred[:batch_size]
                            latents = self.scheduler.step(video_noise_pred,
                                                          t,
                                                          latents,
                                                          return_dict=False)
                    latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else latents[:, :, 0:1]
        profile = profiler.summary('video_branch_for_action_diversity', frame_st_id=frame_st_id)
        self._log_latency_profile(profile)
        return latents, profile

    def _compute_action_diversity_metrics(self, actions, frame_st_id):
        action_arr = np.stack([np.asarray(action, dtype=np.float32) for action in actions], axis=0)
        flat = action_arr.reshape(action_arr.shape[0], -1)
        std = np.std(action_arr, axis=0)
        pairwise_rmse = []
        pairwise_abs_mean = []
        pairwise_l2 = []
        for i in range(flat.shape[0]):
            for j in range(i + 1, flat.shape[0]):
                diff = flat[i] - flat[j]
                pairwise_rmse.append(float(np.sqrt(np.mean(diff ** 2))))
                pairwise_abs_mean.append(float(np.mean(np.abs(diff))))
                pairwise_l2.append(float(np.linalg.norm(diff)))
        metrics = {
            'frame_st_id': int(frame_st_id),
            'num_branches': int(action_arr.shape[0]),
            'action_shape': list(action_arr.shape[1:]),
            'std_mean': float(np.mean(std)),
            'std_max': float(np.max(std)),
            'std_p95': float(np.percentile(std, 95)),
        }
        if pairwise_rmse:
            metrics.update({
                'pairwise_rmse_mean': float(np.mean(pairwise_rmse)),
                'pairwise_rmse_min': float(np.min(pairwise_rmse)),
                'pairwise_rmse_max': float(np.max(pairwise_rmse)),
                'pairwise_abs_mean_mean': float(np.mean(pairwise_abs_mean)),
                'pairwise_abs_mean_max': float(np.max(pairwise_abs_mean)),
                'pairwise_l2_mean': float(np.mean(pairwise_l2)),
                'pairwise_l2_max': float(np.max(pairwise_l2)),
            })
        else:
            metrics.update({
                'pairwise_rmse_mean': 0.0,
                'pairwise_rmse_min': 0.0,
                'pairwise_rmse_max': 0.0,
                'pairwise_abs_mean_mean': 0.0,
                'pairwise_abs_mean_max': 0.0,
                'pairwise_l2_mean': 0.0,
                'pairwise_l2_max': 0.0,
            })
        return metrics

    def _log_video_branch_action_diversity(self, metrics):
        message = (
            f"[VideoBranchActionDiversity] frame_st_id={metrics['frame_st_id']} "
            f"branches={metrics['num_branches']} action_shape={metrics['action_shape']} "
            f"std_mean={metrics['std_mean']:.6f} std_max={metrics['std_max']:.6f} "
            f"std_p95={metrics['std_p95']:.6f} "
            f"pairwise_rmse_mean={metrics['pairwise_rmse_mean']:.6f} "
            f"pairwise_rmse_min={metrics['pairwise_rmse_min']:.6f} "
            f"pairwise_rmse_max={metrics['pairwise_rmse_max']:.6f} "
            f"pairwise_abs_mean_mean={metrics['pairwise_abs_mean_mean']:.6f} "
            f"pairwise_l2_mean={metrics['pairwise_l2_mean']:.6f}"
        )
        logger.info(message)
        log_path = getattr(self.job_config, 'video_branch_action_diversity_log_path', None)
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(message + '\n')

    def _compute_video_branch_action_diversity(self, obs, frame_st_id=0):
        num_branches = int(getattr(self.job_config, 'video_branch_action_diversity_num', 8))
        if num_branches <= 0:
            return None
        logger.info(
            f"[VideoBranchActionDiversity] Generate {num_branches} batched branches "
            f"at frame_st_id={frame_st_id}"
        )
        cache_snapshot = self._clone_transformer_cache()
        try:
            self.transformer.clear_pred_cache(self.cache_name)
            self._expand_transformer_cache_batch(num_branches)
            self._infer_video_branch_for_action_diversity(
                obs, frame_st_id=frame_st_id, batch_size=num_branches)
            actions, _ = self._infer_action_only(
                frame_st_id=frame_st_id, batch_size=num_branches)
        finally:
            self._restore_transformer_cache(cache_snapshot)
        metrics = self._compute_action_diversity_metrics(actions, frame_st_id)
        self._log_video_branch_action_diversity(metrics)
        return metrics

    def _infer_action_only(self, frame_st_id=0, initial_actions=None, batch_size=1):
        profiler = self._new_latency_profiler()
        frame_chunk_size = self.job_config.frame_chunk_size
        with profiler.record('compare.action_noise.init'):
            if initial_actions is None:
                actions = torch.randn(batch_size,
                                      self.job_config.action_dim,
                                      frame_chunk_size,
                                      self.action_per_frame,
                                      1,
                                      device=self.device,
                                      dtype=self.dtype)
            else:
                actions = initial_actions.clone().to(device=self.device, dtype=self.dtype)

        action_inference_step = self.job_config.action_num_inference_steps
        self.action_scheduler.set_timesteps(action_inference_step)
        action_timesteps = F.pad(
            self.action_scheduler.timesteps,
            (0, 1),
            mode='constant',
            value=0)

        with torch.no_grad():
            with profiler.record('compare.action.loop'):
                for i, t in enumerate(tqdm(action_timesteps)):
                    step_suffix = f'.step_{i:02d}' if profiler.profile_steps else ''
                    last_step = i == len(action_timesteps) - 1
                    action_cond = torch.zeros(
                        [
                            batch_size, self.job_config.action_dim, 1,
                            self.action_per_frame, 1
                        ],
                        device=self.device,
                        dtype=self.dtype) if frame_st_id == 0 else None
                    with profiler.record(f'compare.action.prepare_input{step_suffix}'):
                        input_dict = self._prepare_latent_input(
                            None,
                            actions,
                            t,
                            t,
                            None,
                            action_cond,
                            frame_st_id=frame_st_id)
                    with profiler.record(f'compare.action.transformer{step_suffix}'):
                        action_noise_pred = self.transformer(
                            self._repeat_input_for_cfg(input_dict['action_res_lst']),
                            update_cache=1 if last_step else 0,
                            cache_name=self.cache_name,
                            action_mode=True)
                    if not last_step:
                        with profiler.record(f'compare.action.scheduler_step{step_suffix}'):
                            action_noise_pred = rearrange(action_noise_pred,
                                                          'b (f n) c -> b c f n 1',
                                                          f=frame_chunk_size)
                            if self.job_config.action_guidance_scale > 1:
                                action_noise_pred = action_noise_pred[batch_size:] + self.job_config.action_guidance_scale * (action_noise_pred[:batch_size] - action_noise_pred[batch_size:])
                            else:
                                action_noise_pred = action_noise_pred[:batch_size]
                            actions = self.action_scheduler.step(action_noise_pred,
                                                                 t,
                                                                 actions,
                                                                 return_dict=False)
                    actions[:, :, 0:1] = action_cond if frame_st_id == 0 else actions[:, :, 0:1]

        with profiler.record('compare.action.mask'):
            actions[:, ~self.action_mask] *= 0
        with profiler.record('compare.action.postprocess'):
            actions = self.postprocess_action(actions) if batch_size == 1 else self._postprocess_action_batch(actions)
        profile = profiler.summary('compare_action_generation', frame_st_id=frame_st_id)
        self._log_latency_profile(profile)
        return actions, profile

    def _encode_obs_preserve_vae_cache(self, obs):
        vae_feat_cache = [x.clone() if torch.is_tensor(x) else x for x in self.streaming_vae.feat_cache]
        vae_half_feat_cache = None
        if hasattr(self, 'streaming_vae_half'):
            vae_half_feat_cache = [x.clone() if torch.is_tensor(x) else x for x in self.streaming_vae_half.feat_cache]
        try:
            with torch.no_grad():
                return self._encode_obs(obs)
        finally:
            self.streaming_vae.feat_cache = vae_feat_cache
            if vae_half_feat_cache is not None:
                self.streaming_vae_half.feat_cache = vae_half_feat_cache

    def _infer_speculative_draft(self, obs):
        base_future_frame_chunk_size = int(getattr(
            self.job_config, 'speculative_frame_chunk_size', self.job_config.frame_chunk_size))
        full_denoise_replan = bool(obs.get('speculative_full_denoise', False))
        future_frame_chunk_size = base_future_frame_chunk_size
        if full_denoise_replan:
            replan_frame_chunk_size = int(getattr(
                self.job_config, 'speculative_replan_frame_chunk_size', -1))
            if replan_frame_chunk_size > 0:
                future_frame_chunk_size = replan_frame_chunk_size
            video_steps = int(self.job_config.num_inference_steps)
            action_steps = int(self.job_config.action_num_inference_steps)
        else:
            video_steps = int(getattr(
                self.job_config, 'speculative_video_num_inference_steps', self.job_config.num_inference_steps))
            action_steps = int(getattr(
                self.job_config, 'speculative_action_num_inference_steps', self.job_config.action_num_inference_steps))
        frame_chunk_size = future_frame_chunk_size + 1 if self.frame_st_id == 0 else future_frame_chunk_size
        logger.info(
            f"[SpeculativeDraft] frame_st_id={self.frame_st_id} full_denoise_replan={full_denoise_replan} "
            f"video_steps={video_steps} action_steps={action_steps} frame_chunk_size={frame_chunk_size} "
            f"future_frame_chunk_size={future_frame_chunk_size} base_future_frame_chunk_size={base_future_frame_chunk_size}"
        )
        action, latents, profile, pred_video = self._infer(
            obs,
            frame_st_id=self.frame_st_id,
            video_num_inference_steps=video_steps,
            action_num_inference_steps=action_steps,
            frame_chunk_size=frame_chunk_size,
        )
        # Do not let speculative future tokens become committed history.
        self.transformer.clear_pred_cache(self.cache_name)
        result = {
            'action': action,
            'speculative_pred_latents': latents.detach().float().cpu().numpy(),
            'speculative_frame_chunk_size': int(base_future_frame_chunk_size),
            'speculative_replan_frame_chunk_size_used': int(future_frame_chunk_size),
            'speculative_draft_frame_chunk_size': int(frame_chunk_size),
            'speculative_segment_action_steps': int(getattr(
                self.job_config, 'speculative_segment_action_steps', self.action_per_frame)),
            'speculative_full_denoise_replan': bool(full_denoise_replan),
            'speculative_video_num_inference_steps_used': int(video_steps),
            'speculative_action_num_inference_steps_used': int(action_steps),
        }
        if pred_video is not None:
            result['video'] = pred_video
        if profile:
            result['profile_action_latency'] = profile
        return result

    def _to_single_frame_pred_latent(self, pred_latent, ref_latent):
        pred_latent = torch.from_numpy(np.asarray(pred_latent)).to(
            device=ref_latent.device, dtype=ref_latent.dtype)
        if pred_latent.ndim == 3:
            pred_latent = pred_latent[None, :, None]
        elif pred_latent.ndim == 4:
            pred_latent = pred_latent[None]
        if pred_latent.shape[2] != 1:
            pred_latent = pred_latent[:, :, -1:]
        return pred_latent.to(ref_latent)

    def _verify_speculative(self, obs):
        mode = getattr(self.job_config, 'speculative_verifier_mode', 'action_denoise')
        if mode == 'latent_mse':
            return self._verify_speculative_latent(obs)
        if mode == 'action_denoise':
            return self._verify_speculative_action_denoise(obs)
        raise ValueError(f'Unknown speculative_verifier_mode: {mode}')

    def _verify_speculative_action_denoise(self, obs):
        real_obs = obs.get('obs', None)
        pred_latent = obs.get('pred_latent', None)
        draft_action = obs.get('draft_action', None)
        segment_index = int(obs.get('segment_index', -1))
        verify_frame_offset = int(obs.get('verify_frame_offset', 0))
        verify_frame_st_id = int(self.frame_st_id + verify_frame_offset)
        if real_obs is None or pred_latent is None or draft_action is None:
            raise ValueError('action_denoise speculative_verify requires obs, pred_latent, and draft_action')

        real_obs_list = real_obs if isinstance(real_obs, list) else [real_obs]
        real_latent = self._encode_obs_preserve_vae_cache({'obs': real_obs_list})
        current_latent = real_latent[:, :, -1:].to(self.device).to(self.dtype)
        pred_latent = self._to_single_frame_pred_latent(pred_latent, current_latent)
        latent_condition = torch.cat([current_latent, pred_latent], dim=2)

        action_model_input = self.preprocess_action(np.asarray(draft_action)).to(
            device=self.device, dtype=self.dtype)
        verifier_sigma = float(getattr(
            self.job_config, 'speculative_verifier_action_noise_sigma', 0.15))
        verifier_sigma = max(0.0, min(1.0, verifier_sigma))
        self.action_scheduler.set_timesteps(self.action_scheduler.num_train_timesteps)
        sigma_id = torch.argmin((self.action_scheduler.sigmas - verifier_sigma).abs())
        sigma = self.action_scheduler.sigmas[sigma_id].to(
            device=self.device, dtype=action_model_input.dtype)
        action_t = self.action_scheduler.timesteps[sigma_id].to(self.device)
        noise = torch.randn_like(action_model_input)
        noisy_action = ((1 - sigma) * action_model_input + sigma * noise).to(self.dtype)

        profiler = self._new_latency_profiler()
        threshold = float(getattr(self.job_config, 'speculative_verifier_threshold', 0.5))
        cache_snapshot = self._clone_transformer_cache()
        try:
            self.transformer.clear_pred_cache(self.cache_name)
            with profiler.record('verify.cache_video_condition'):
                self._cache_pred_latent_chunk(latent_condition, verify_frame_st_id)
            with profiler.record('verify.prepare_action'):
                input_dict = self._prepare_latent_input(
                    None,
                    noisy_action,
                    action_t=action_t,
                    frame_st_id=verify_frame_st_id)
            with torch.no_grad():
                with profiler.record('verify.action_transformer'):
                    pred_velocity = self.transformer(
                        self._repeat_input_for_cfg(input_dict['action_res_lst']),
                        update_cache=0,
                        cache_name=self.cache_name,
                        action_mode=True)
            pred_velocity = rearrange(pred_velocity,
                                      'b (f n) c -> b c f n 1',
                                      f=action_model_input.shape[2])
            batch_size = action_model_input.shape[0]
            if self.job_config.action_guidance_scale > 1:
                pred_velocity = pred_velocity[batch_size:] + self.job_config.action_guidance_scale * (
                    pred_velocity[:batch_size] - pred_velocity[batch_size:])
            else:
                pred_velocity = pred_velocity[:batch_size]
            actual_velocity = noise - action_model_input
            pred_velocity[:, ~self.action_mask] *= 0
            actual_velocity[:, ~self.action_mask] *= 0
            verifier = ActionDenoiseSpeculativeVerifier(threshold)
            score, passed = verifier.verify(pred_velocity, actual_velocity, self.action_mask)
        finally:
            self._restore_transformer_cache(cache_snapshot)

        profile = profiler.summary('speculative_action_denoise_verify', frame_st_id=verify_frame_st_id)
        self._log_latency_profile(profile)
        message = (
            f"[SpeculativeVerifier] mode=action_denoise frame_st_id={self.frame_st_id} "
            f"verify_frame_st_id={verify_frame_st_id} segment={segment_index} "
            f"score={score:.6f} threshold={threshold:.6f} "
            f"passed={passed} action_t={float(action_t.detach().cpu()):.6f} "
            f"sigma={float(sigma.detach().cpu()):.6f} target=actual_velocity "
            f"obs_frames={len(real_obs_list)} condition_frames={latent_condition.shape[2]}"
        )
        logger.info(message)
        log_path = getattr(self.job_config, 'speculative_verifier_log_path', None)
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
        return {
            'speculative_verify_score': float(score),
            'speculative_verify_passed': bool(passed),
            'speculative_verify_threshold': float(threshold),
            'speculative_verifier_mode': 'action_denoise',
            'segment_index': int(segment_index),
        }

    def _verify_speculative_latent(self, obs):
        real_obs = obs.get('obs', None)
        pred_latent = obs.get('pred_latent', None)
        segment_index = int(obs.get('segment_index', -1))
        if real_obs is None or pred_latent is None:
            raise ValueError('speculative_verify requires obs and pred_latent')
        real_obs_list = real_obs if isinstance(real_obs, list) else [real_obs]
        real_latent = self._encode_obs_preserve_vae_cache({'obs': real_obs_list})
        pred_latent = self._to_single_frame_pred_latent(pred_latent, real_latent)

        if real_latent.shape[2] != pred_latent.shape[2]:
            real_latent = real_latent[:, :, -pred_latent.shape[2]:]
        pred_latent = pred_latent.to(real_latent)
        threshold = float(getattr(self.job_config, 'speculative_verifier_threshold', 0.5))
        verifier = LatentMSESpeculativeVerifier(threshold)
        score, passed = verifier.verify(real_latent, pred_latent)
        message = (
            f"[SpeculativeVerifier] mode=latent_mse frame_st_id={self.frame_st_id} segment={segment_index} "
            f"score={score:.6f} threshold={threshold:.6f} passed={passed}"
        )
        logger.info(message)
        log_path = getattr(self.job_config, 'speculative_verifier_log_path', None)
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
        return {
            'speculative_verify_score': float(score),
            'speculative_verify_passed': bool(passed),
            'speculative_verify_threshold': float(threshold),
            'speculative_verifier_mode': 'latent_mse',
            'segment_index': int(segment_index),
        }


    def _compare_action_with_real_obs(self, obs):
        real_obs_chunk = obs.get('real_obs_chunk', None)
        reference_action = obs.get('reference_action', None)
        if real_obs_chunk is None or reference_action is None:
            raise ValueError('compare_action_with_real_obs requires real_obs_chunk and reference_action')
        self.transformer.clear_pred_cache(self.cache_name)
        vae_feat_cache = [x.clone() if torch.is_tensor(x) else x for x in self.streaming_vae.feat_cache]
        vae_half_feat_cache = None
        if hasattr(self, 'streaming_vae_half'):
            vae_half_feat_cache = [x.clone() if torch.is_tensor(x) else x for x in self.streaming_vae_half.feat_cache]
        compare_obs = {'obs': real_obs_chunk}
        try:
            with torch.no_grad():
                latent_model_input = self._encode_obs(compare_obs)
        finally:
            self.streaming_vae.feat_cache = vae_feat_cache
            if vae_half_feat_cache is not None:
                self.streaming_vae_half.feat_cache = vae_half_feat_cache

        self._cache_pred_latent_chunk(
            latent_model_input.to(self.device).to(self.dtype),
            self.frame_st_id)
        compare_action, profile = self._infer_action_only(frame_st_id=self.frame_st_id)
        self.transformer.clear_pred_cache(self.cache_name)

        reference_action = np.asarray(reference_action)
        compare_action = np.asarray(compare_action)
        common_shape = tuple(min(a, b) for a, b in zip(reference_action.shape, compare_action.shape))
        slices = tuple(slice(0, n) for n in common_shape)
        ref = reference_action[slices].astype(np.float32)
        cmp = compare_action[slices].astype(np.float32)
        diff = cmp - ref
        ref_l2 = float(np.sqrt(np.mean(ref ** 2)) + 1e-8)
        metrics = {
            'shape': list(common_shape),
            'abs_mean': float(np.mean(np.abs(diff))),
            'abs_max': float(np.max(np.abs(diff))),
            'rmse': float(np.sqrt(np.mean(diff ** 2))),
            'rel_l2': float(np.sqrt(np.mean(diff ** 2)) / ref_l2),
        }
        logger.info(
            f"[ActionCompare] frame_st_id={self.frame_st_id} "
            f"abs_mean={metrics['abs_mean']:.6f} abs_max={metrics['abs_max']:.6f} "
            f"rmse={metrics['rmse']:.6f} rel_l2={metrics['rel_l2']:.6f}"
        )
        return metrics, profile

    def _cache_pred_action_chunk(self, action, frame_st_id):
        profiler = self._new_latency_profiler()
        self.transformer.clear_pred_cache(self.cache_name)
        with profiler.record('async.preprocess_current_action'):
            action_model_input = self.preprocess_action(action).to(self.device).to(self.dtype)
        with profiler.record('async.prepare_current_action'):
            input_dict = self._prepare_latent_input(
                None,
                action_model_input,
                frame_st_id=frame_st_id)
        with torch.no_grad():
            with profiler.record('async.cache_current_action'):
                self.transformer(
                    self._repeat_input_for_cfg(input_dict['action_res_lst']),
                    update_cache=1,
                    cache_name=self.cache_name,
                    action_mode=True)
        profile = profiler.summary('async_cache_current_action', frame_st_id=frame_st_id)
        self._log_latency_profile(profile)
        return profile

    def _cache_pred_latent_chunk(self, latent_model_input, frame_st_id):
        profiler = self._new_latency_profiler()
        with profiler.record('sync_fdm.prepare_real_latent'):
            input_dict = self._prepare_latent_input(
                latent_model_input,
                None,
                frame_st_id=frame_st_id)
        with torch.no_grad():
            with profiler.record('sync_fdm.cache_real_latent'):
                self.transformer(
                    self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                    update_cache=1,
                    cache_name=self.cache_name,
                    action_mode=False)
        profile = profiler.summary('sync_fdm_cache_real_latent', frame_st_id=frame_st_id)
        self._log_latency_profile(profile)
        return profile

    def _infer_fdm_video(self, frame_st_id):
        profiler = self._new_latency_profiler()
        frame_chunk_size = self.job_config.frame_chunk_size

        with profiler.record('fdm.noise.init'):
            latents = torch.randn(1,
                                  48,
                                  frame_chunk_size,
                                  self.latent_height,
                                  self.latent_width,
                                  device=self.device,
                                  dtype=self.dtype)

        video_inference_step = self.job_config.num_inference_steps
        video_step = self.job_config.video_exec_step
        self.scheduler.set_timesteps(video_inference_step)
        timesteps = F.pad(self.scheduler.timesteps, (0, 1), mode='constant', value=0)
        if video_step != -1:
            timesteps = timesteps[:video_step]

        with torch.no_grad():
            with profiler.record('fdm.video.loop'):
                for i, t in enumerate(tqdm(timesteps)):
                    step_suffix = f'.step_{i:02d}' if profiler.profile_steps else ''
                    last_step = i == len(timesteps) - 1
                    with profiler.record(f'fdm.video.prepare_input{step_suffix}'):
                        input_dict = self._prepare_latent_input(
                            latents,
                            None,
                            t,
                            t,
                            None,
                            None,
                            frame_st_id=frame_st_id)
                    with profiler.record(f'fdm.video.transformer{step_suffix}'):
                        video_noise_pred = self.transformer(
                            self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                            update_cache=1 if last_step else 0,
                            cache_name=self.cache_name,
                            action_mode=False)
                    if not last_step or video_step != -1:
                        with profiler.record(f'fdm.video.scheduler_step{step_suffix}'):
                            video_noise_pred = data_seq_to_patch(
                                self.job_config.patch_size, video_noise_pred,
                                frame_chunk_size, self.latent_height,
                                self.latent_width, batch_size=2 if self.use_cfg else 1)
                            if self.job_config.guidance_scale > 1:
                                video_noise_pred = video_noise_pred[1:] + self.job_config.guidance_scale * (video_noise_pred[:1] - video_noise_pred[1:])
                            else:
                                video_noise_pred = video_noise_pred[:1]
                            latents = self.scheduler.step(video_noise_pred,
                                                          t,
                                                          latents,
                                                          return_dict=False)

        save_async(latents, os.path.join(self.exp_save_root, f'fdm_latents_{frame_st_id}.pt'))
        with profiler.record('cuda.empty_cache'):
            torch.cuda.empty_cache()
        profile = profiler.summary('fdm_grounding', frame_st_id=frame_st_id)
        self._log_latency_profile(profile)
        return latents, profile

    def _async_prefetch(self, obs):
        profiler = self._new_latency_profiler()
        feedback_obs = obs.get('feedback_obs', None)
        feedback_state = obs.get('feedback_state', None)
        current_action = obs.get('current_action', None)
        if current_action is None:
            raise ValueError('async_prefetch requires current_action')

        feedback_profile = None
        if feedback_obs is not None and feedback_state is not None:
            with profiler.record('async.feedback_kv_cache'):
                feedback_profile = self._compute_kv_cache({
                    'obs': feedback_obs,
                    'state': feedback_state,
                })
        else:
            self.transformer.clear_pred_cache(self.cache_name)

        current_frame_st_id = self.frame_st_id
        current_action_profile = self._cache_pred_action_chunk(current_action, current_frame_st_id)
        _, fdm_profile = self._infer_fdm_video(current_frame_st_id)
        action, _, action_profile, pred_video = self._infer(
            obs,
            frame_st_id=current_frame_st_id + self.job_config.frame_chunk_size)

        profile = profiler.summary('async_prefetch', frame_st_id=current_frame_st_id)
        self._log_latency_profile(profile)
        return action, pred_video, {
            'async_prefetch': profile,
            'feedback_kv_cache': feedback_profile,
            'cache_current_action': current_action_profile,
            'fdm_grounding': fdm_profile,
            'action_generation': action_profile,
        }

    def _compute_kv_cache(self, obs):
        profiler = self._new_latency_profiler()
        ### optional async save obs for debug
        self.transformer.clear_pred_cache(self.cache_name)
        save_async(obs['obs'], os.path.join(self.exp_save_root, f'obs_data_{self.frame_st_id}.pt'))
        with profiler.record('kv.obs_encode'):
            latent_model_input = self._encode_obs(obs)
        if self.frame_st_id == 0:
            latent_model_input = torch.cat(
                [self.init_latent, latent_model_input],
                dim=2) if latent_model_input is not None else self.init_latent

        with profiler.record('kv.preprocess_action'):
            action_model_input = self.preprocess_action(obs['state'])
            action_model_input = action_model_input.to(latent_model_input)

        if (getattr(self.job_config, 'sync_fdm_recompose_kv_cache', False)
                and self.job_config.frame_chunk_size == 2
                and self.frame_st_id > 0
                and latent_model_input is not None
                and latent_model_input.shape[2] >= 2):
            real_first_latent = latent_model_input[:, :, 0:1].to(self.device).to(self.dtype)
            self.transformer.clear_pred_cache(self.cache_name)
            self._cache_pred_action_chunk(obs['state'], self.frame_st_id)
            self._cache_pred_latent_chunk(real_first_latent, self.frame_st_id)
            fdm_latents, _ = self._infer_fdm_video(self.frame_st_id)
            fdm_second_latent = fdm_latents[:, :, 1:2].to(latent_model_input)
            latent_model_input = torch.cat([latent_model_input[:, :, 0:1], fdm_second_latent], dim=2)
            # latent_model_input = latent_model_input[:, :, 0:2]
            self.transformer.clear_pred_cache(self.cache_name)
            logger.info(
                f"[SyncFDM] Recompose KV latent chunk with real first frame and FDM second frame at frame_st_id={self.frame_st_id}"
            )

        logger.info(
            f"get KV cache obs: {latent_model_input.shape} {action_model_input.shape}"
        )
        with profiler.record('kv.prepare_input'):
            input_dict = self._prepare_latent_input(latent_model_input,
                                                    action_model_input,
                                                    frame_st_id=self.frame_st_id)

        with (
                torch.no_grad(),
        ):
            with profiler.record('kv.video_transformer'):
                self.transformer(self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                                 update_cache=2,
                                 cache_name=self.cache_name,
                                 action_mode=False)

            with profiler.record('kv.action_transformer'):
                self.transformer(self._repeat_input_for_cfg(input_dict['action_res_lst']),
                                 update_cache=2,
                                 cache_name=self.cache_name,
                                 action_mode=True)
        with profiler.record('cuda.empty_cache'):
            torch.cuda.empty_cache()
        profile = profiler.summary('kv_cache', frame_st_id=self.frame_st_id)
        self._log_latency_profile(profile)
        self.frame_st_id += latent_model_input.shape[2]
        return profile

    @torch.no_grad()
    def infer(self, obs):
        reset = obs.get('reset', False)
        prompt = obs.get('prompt', None)
        compute_kv_cache = obs.get('compute_kv_cache', False)
        async_prefetch = obs.get('async_prefetch', False)
        compare_action_with_real_obs = obs.get('compare_action_with_real_obs', False)
        speculative_verify = obs.get('speculative_verify', False)

        if reset:
            logger.info(f"******************* Reset server ******************")
            self._reset(prompt=prompt)
            return dict()
        elif compute_kv_cache:
            logger.info(
                f"################# Compute KV Cache #################")
            profile = self._compute_kv_cache(obs)
            result = dict()
            if profile:
                result['profile_action_latency'] = profile
            return result
        elif async_prefetch:
            logger.info(
                f"################# Async Prefetch #################")
            action, pred_video, profile = self._async_prefetch(obs)
            result = dict(action=action)
            if pred_video is not None:
                result['video'] = pred_video
            if profile:
                result['profile_action_latency'] = profile
            return result
        elif compare_action_with_real_obs:
            logger.info(
                f"################# Compare Action With Real Obs #################")
            metrics, profile = self._compare_action_with_real_obs(obs)
            result = dict(action_compare_metrics=metrics)
            if profile:
                result['profile_action_latency'] = profile
            return result
        elif speculative_verify:
            logger.info(f"################# Speculative Verify #################")
            return self._verify_speculative(obs)
        else:
            if getattr(self.job_config, 'enable_speculative_verifier', False):
                logger.info(f"################# Infer Speculative Draft #################")
                return self._infer_speculative_draft(obs)

            logger.info(f"################# Infer One Chunk #################")
            action, _, profile, pred_video = self._infer(obs, frame_st_id=self.frame_st_id)
            diversity_metrics = None
            if getattr(self.job_config, 'enable_video_branch_action_diversity', False):
                diversity_metrics = self._compute_video_branch_action_diversity(
                    obs, frame_st_id=self.frame_st_id)
            result = dict(action=action)
            if pred_video is not None:
                result['video'] = pred_video
            if profile:
                result['profile_action_latency'] = profile
            if diversity_metrics is not None:
                result['video_branch_action_diversity_metrics'] = diversity_metrics
            return result
    
    def _decode_pred_video(self, latents):
        if not getattr(self.job_config, 'return_pred_video', False):
            return None
        if not hasattr(self, 'video_processor'):
            self.video_processor = VideoProcessor(vae_scale_factor=1)
        return self.decode_one_video(latents, 'np')[0]

    def decode_one_video(self, latents, output_type):
        vae_device = next(self.vae.parameters()).device
        latents = latents.to(device=vae_device, dtype=self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        return video
    
    def load_init_obs(self):
        imf_dict = {v: np.array(Image.open(os.path.join(self.job_config.input_img_path, f"{v}.png")).convert("RGB")) for v in self.job_config.obs_cam_keys}
        init_obs = {}
        init_obs['obs'] = [imf_dict]
        return init_obs
    
    @torch.no_grad()
    def generate(self):
        self.video_processor = VideoProcessor(vae_scale_factor=1)
        self._reset(self.job_config.prompt)
        init_obs = self.load_init_obs()
        pred_latent_lst = []
        pred_action_lst = []
        for chunk_id in range(self.job_config.num_chunks_to_infer):
            actions, latents, _, _ = self._infer(init_obs, frame_st_id=(chunk_id * self.job_config.frame_chunk_size))
            actions = torch.from_numpy(actions)
            pred_latent_lst.append(latents)
            pred_action_lst.append(actions)
        pred_latent = torch.cat(pred_latent_lst, dim=2)
        pred_action = torch.cat(pred_action_lst, dim=1).flatten(1)
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()
        if self.streaming_vae_half:
            self.streaming_vae_half.clear_cache()
        del self.transformer
        del self.streaming_vae_half
        del self.text_encoder
        torch.cuda.empty_cache()
        
        # Move VAE to GPU for decoding
        if self.enable_offload:
            self.vae = self.vae.to(self.device).to(self.dtype)
        
        decoded_video = self.decode_one_video(pred_latent, 'np')[0]
        export_to_video(decoded_video, os.path.join(self.save_root, "demo.mp4"), fps=10)

def run(args):    
    
    config = VA_CONFIGS[args.config_name]
    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root
    if getattr(args, 'profile_action_latency', False):
        config.profile_action_latency = True
    if getattr(args, 'profile_action_latency_steps', False):
        config.profile_action_latency = True
        config.profile_action_latency_steps = True
    if getattr(args, 'return_pred_video', False):
        config.return_pred_video = True
    if getattr(args, 'compile_infer', False):
        config.compile_infer = True
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    model = VA_Server(config)
    if config.infer_mode == 'i2va':
        logger.info(f"******************************USE I2AV mode******************************")
        model.generate()
    elif config.infer_mode == 'server':
        logger.info(f"******************************USE Server mode******************************")
        run_async_server_mode(model, local_rank, config.host, port)
    else:
        raise ValueError(f"Unknown infer mode: {config.infer_mode}")

def main():
    """
    TODO
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        type=str,
        required=False,
        default='robotwin',
        help="config name.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help='(start) port'
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=None,
        help='save root'
    )
    parser.add_argument(
        "--profile-action-latency",
        action='store_true',
        help='profile each server-side action generation with CUDA events'
    )
    parser.add_argument(
        "--profile-action-latency-steps",
        action='store_true',
        help='also log per-denoising-step CUDA timings for action generation'
    )
    parser.add_argument(
        "--return-pred-video",
        action='store_true',
        help='decode predicted video chunks and return them in websocket responses'
    )
    parser.add_argument(
        "--compile-infer",
        action='store_true',
        help='enable torch.compile on VA_Server._infer for repeated-shape inference'
    )
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    init_logger()
    main()
