# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_robotwin_cfg = EasyDict(__name__='Config: VA robotwin')
va_robotwin_cfg.update(va_shared_cfg)

va_robotwin_cfg.wan22_pretrained_model_name_or_path = "/mnt/public/xieruiqi/models/lingbot-va/robotwin"
# va_robotwin_cfg.trained_transformer_path = "/mnt/public/ns-t-te-b905754427352261-427-bk/fs/home/xieruiqi/lingbot-va_0607/train_out/checkpoints/checkpoint_step_9000"

va_robotwin_cfg.attn_window = 72
va_robotwin_cfg.frame_chunk_size = 2
va_robotwin_cfg.env_type = 'robotwin_tshape'

va_robotwin_cfg.height = 256
va_robotwin_cfg.width = 320
va_robotwin_cfg.action_dim = 30
va_robotwin_cfg.action_per_frame = 16
va_robotwin_cfg.obs_cam_keys = [
    'observation.images.cam_high', 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]
va_robotwin_cfg.guidance_scale = 1
va_robotwin_cfg.action_guidance_scale = 1

va_robotwin_cfg.num_inference_steps = 3
va_robotwin_cfg.action_num_inference_steps = 3
# Analyze Exp
va_robotwin_cfg.video_exec_step = -1

va_robotwin_cfg.enable_async_inference = False
va_robotwin_cfg.sync_fdm_recompose_kv_cache = False
va_robotwin_cfg.compare_action_with_real_obs = False
va_robotwin_cfg.enable_action_without_future_video = False

va_robotwin_cfg.enable_video_branch_action_diversity = False
va_robotwin_cfg.video_branch_action_diversity_num = 8
# Speculative
va_robotwin_cfg.enable_speculative_verifier = True
va_robotwin_cfg.speculative_frame_chunk_size = 4
va_robotwin_cfg.speculative_replan_frame_chunk_size = 2
va_robotwin_cfg.speculative_video_num_inference_steps = 5
va_robotwin_cfg.speculative_action_num_inference_steps = 3
va_robotwin_cfg.speculative_segment_action_steps = 16
va_robotwin_cfg.speculative_verifier_mode = "action_denoise"
va_robotwin_cfg.speculative_verifier_action_noise_sigma = 0.3
va_robotwin_cfg.speculative_verifier_threshold = 0.01
va_robotwin_cfg.speculative_verifier_log_path = "/mnt/public/ns-t-te-b905754427352261-427-bk/fs/home/xieruiqi/Lingbot-VA-Customize/speculative_verifier.txt"
va_robotwin_cfg.video_branch_action_diversity_log_path = "/mnt/public/ns-t-te-b905754427352261-427-bk/fs/home/xieruiqi/Lingbot-VA-Customize/video_branch_action_diversity.txt"
va_robotwin_cfg.reset_policy_at_half_eval_steps = False

va_robotwin_cfg.snr_shift = 5.0
va_robotwin_cfg.action_snr_shift = 1.0

va_robotwin_cfg.used_action_channel_ids = list(range(0, 7)) + list(
    range(28, 29)) + list(range(7, 14)) + list(range(29, 30))
inverse_used_action_channel_ids = [
    len(va_robotwin_cfg.used_action_channel_ids)
] * va_robotwin_cfg.action_dim
for i, j in enumerate(va_robotwin_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_robotwin_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_robotwin_cfg.action_norm_method = 'quantiles'
va_robotwin_cfg.norm_stat = {
    "q01": [
        -0.06172713458538055, -3.6716461181640625e-05, -0.08783501386642456,
        -1, -1, -1, -1, -0.3547105032205582, -1.3113021850585938e-06,
        -0.11975435614585876, -1, -1, -1, -1
    ] + [0.] * 16,
    "q99": [
        0.3462600058317184, 0.39966784834861746, 0.14745532035827624, 1, 1, 1,
        1, 0.034201726913452024, 0.39142737388610793, 0.1792279863357542, 1, 1,
        1, 1
    ] + [0.] * 14 + [1.0, 1.0],
}
