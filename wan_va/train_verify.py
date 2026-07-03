#!/usr/bin/env python3
"""Train a RoboTwin action-chunk verifier from cached WAN-VA latents.

The native WAM trainer uses MultiLatentLeRobotDataset, which reads
latents/*.pth files containing VAE-encoded visual latents and T5 text_emb.
This verifier follows the same contract and does not load or run a VAE/text
encoder during training. It also mirrors WAM's grid-id/RoPE-style self-attention
position handling for latent and action tokens.

Verifier input semantics:
  1. Cached real-observation latents up to the current time, max window 6.
  2. The candidate action chunk that will be executed next.
  3. Cached text_emb from the same latent file used by standard WAM training.

Class ids exposed by the model:
  1 -> correct action chunk
  2 -> wrong action chunk, should trigger replan

Internally CrossEntropy uses labels 0/1, where 0 maps to class id 1 and
1 maps to class id 2.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


CAM_HIGH = "observation.images.cam_high"
CAM_LEFT = "observation.images.cam_left_wrist"
CAM_RIGHT = "observation.images.cam_right_wrist"
CAM_KEYS = [CAM_HIGH, CAM_LEFT, CAM_RIGHT]
ACTION_DIM_RAW = 16
ACTION_CHUNK_SIZE = 16
CLASS_CORRECT = 1
CLASS_REPLAN = 2

ROBOTWIN_USED_ACTION_CHANNEL_IDS = (
    list(range(0, 7)) + [28] + list(range(7, 14)) + [29]
)
ROBOTWIN_INVERSE_USED_ACTION_CHANNEL_IDS = [len(ROBOTWIN_USED_ACTION_CHANNEL_IDS)] * 30
for _i, _j in enumerate(ROBOTWIN_USED_ACTION_CHANNEL_IDS):
    ROBOTWIN_INVERSE_USED_ACTION_CHANNEL_IDS[_j] = _i

ROBOTWIN_Q01 = [
    -0.06172713458538055, -3.6716461181640625e-05, -0.08783501386642456,
    -1, -1, -1, -1, -0.3547105032205582, -1.3113021850585938e-06,
    -0.11975435614585876, -1, -1, -1, -1
] + [0.0] * 16
ROBOTWIN_Q99 = [
    0.3462600058317184, 0.39966784834861746, 0.14745532035827624,
    1, 1, 1, 1, 0.034201726913452024, 0.39142737388610793,
    0.1792279863357542, 1, 1, 1, 1
] + [0.0] * 14 + [1.0, 1.0]


@dataclass(frozen=True)
class EpisodeRecord:
    repo_rel: str
    episode_index: int
    length: int
    action_configs: tuple[dict[str, Any], ...]
    data_path: str
    chunks_size: int


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_lerobot_repos(root: Path) -> list[Path]:
    return sorted(path.parent.parent for path in root.rglob("meta/info.json"))


def episode_chunk(info_chunks_size: int, episode_index: int) -> int:
    return episode_index // int(info_chunks_size)


def format_episode_data_path(record: EpisodeRecord) -> str:
    return record.data_path.format(
        episode_chunk=episode_chunk(record.chunks_size, record.episode_index),
        episode_index=record.episode_index,
    )



def latent_files_exist(repo: Path, chunks_size: int, episode_index: int, start_frame: int, end_frame: int) -> bool:
    episode_chunk_id = episode_chunk(chunks_size, episode_index)
    for cam_key in CAM_KEYS:
        latent_file = (
            repo
            / "latents"
            / f"chunk-{episode_chunk_id:03d}"
            / cam_key
            / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
        )
        if not latent_file.exists():
            return False
    return True


def build_success_index(success_root: Path, max_repos: int | None = None) -> list[EpisodeRecord]:
    repo_roots = iter_lerobot_repos(success_root)
    if max_repos is not None:
        repo_roots = repo_roots[:max_repos]

    records: list[EpisodeRecord] = []
    total_action_configs = 0
    kept_action_configs = 0
    for repo in tqdm(repo_roots, desc="index success repos", dynamic_ncols=True):
        info = read_json(repo / "meta" / "info.json")
        repo_rel = repo.relative_to(success_root).as_posix()
        chunks_size = int(info.get("chunks_size", 1000))
        for ep in read_jsonl(repo / "meta" / "episodes.jsonl"):
            episode_index = int(ep["episode_index"])
            valid_action_configs = []
            for acfg in ep.get("action_config", []):
                total_action_configs += 1
                start_frame = int(acfg.get("start_frame", 0))
                end_frame = int(acfg.get("end_frame", ep.get("length", 0)))
                if latent_files_exist(repo, chunks_size, episode_index, start_frame, end_frame):
                    valid_action_configs.append(acfg)
                    kept_action_configs += 1
            if not valid_action_configs:
                continue
            records.append(
                EpisodeRecord(
                    repo_rel=repo_rel,
                    episode_index=episode_index,
                    length=int(ep.get("length", 0)),
                    action_configs=tuple(valid_action_configs),
                    data_path=info["data_path"],
                    chunks_size=chunks_size,
                )
            )
    if not records:
        raise FileNotFoundError(f"No episodes with complete cached latents found under {success_root}")
    print(
        f"kept {kept_action_configs}/{total_action_configs} action_config ranges with complete cached latents "
        f"across {len(records)} episodes"
    )
    return records

def normalize_quat(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    return quat / norm


def get_relative_pose(pose: np.ndarray) -> np.ndarray:
    pose = pose.astype(np.float64)
    quat = normalize_quat(pose[:, 3:7])
    first_quat = np.repeat(quat[:1], len(pose), axis=0)
    rot = R.from_quat(quat)
    first_rot = R.from_quat(first_quat)
    rel_trans = pose[:, :3] - pose[:1, :3]
    rel_quat = (first_rot.inv() * rot).as_quat()
    return np.concatenate([rel_trans, rel_quat], axis=1).astype(np.float32)


class RobotwinActionPreprocessor:
    """Match the robotwin_tshape action preprocessing used for VA training/inference."""

    def __init__(self):
        self.inverse_used_action_channel_ids = np.asarray(
            ROBOTWIN_INVERSE_USED_ACTION_CHANNEL_IDS, dtype=np.int64
        )
        self.q01 = np.asarray(ROBOTWIN_Q01, dtype=np.float32)[None, :]
        self.q99 = np.asarray(ROBOTWIN_Q99, dtype=np.float32)[None, :]

    def __call__(self, action_chunk: np.ndarray) -> np.ndarray:
        if action_chunk.shape != (ACTION_CHUNK_SIZE, ACTION_DIM_RAW):
            raise ValueError(f"Expected action chunk {(ACTION_CHUNK_SIZE, ACTION_DIM_RAW)}, got {action_chunk.shape}")
        left = get_relative_pose(action_chunk[:, :7])
        right = get_relative_pose(action_chunk[:, 8:15])
        rel = np.concatenate([left, action_chunk[:, 7:8], right, action_chunk[:, 15:16]], axis=1)
        rel_padded = np.pad(rel, ((0, 0), (0, 1)), mode="constant", constant_values=0)
        aligned = rel_padded[:, self.inverse_used_action_channel_ids]
        aligned = (aligned - self.q01) / (self.q99 - self.q01 + 1e-6) * 2.0 - 1.0
        aligned = np.clip(aligned, -1.5, 1.5)
        return aligned.astype(np.float32)  # [16, 30]



def choose_chunk(action_start: int, action_end: int, chunk_size: int, rng: random.Random) -> tuple[int, int, int, int]:
    if action_end <= action_start:
        raise ValueError(f"Invalid action range [{action_start}, {action_end})")
    num_chunks = math.ceil((action_end - action_start) / chunk_size)
    chunk_index = rng.randrange(num_chunks)
    chunk_start = action_start + chunk_index * chunk_size
    chunk_end = chunk_start + chunk_size
    valid_end = min(chunk_end, action_end)
    return chunk_index, chunk_start, chunk_end, valid_end


def make_padded_action_chunk(actions: np.ndarray, start: int, end: int, valid_end: int) -> np.ndarray:
    raw = actions[start:valid_end]
    if len(raw) <= 0:
        raise ValueError(f"Empty action chunk start={start}, valid_end={valid_end}")
    if valid_end < end:
        pad = np.repeat(raw[-1:], end - valid_end, axis=0)
        raw = np.concatenate([raw, pad], axis=0)
    if raw.shape != (ACTION_CHUNK_SIZE, ACTION_DIM_RAW):
        raise ValueError(f"Expected padded raw chunk {(ACTION_CHUNK_SIZE, ACTION_DIM_RAW)}, got {raw.shape}")
    return raw.astype(np.float32)


def get_mesh_id(f: int, h: int, w: int, t: int, f_w: float = 1.0, f_shift: float = 0.0, action: bool = False) -> torch.Tensor:
    f_idx = torch.arange(f, dtype=torch.float32) * float(f_w) + float(f_shift)
    h_idx = torch.arange(h, dtype=torch.float32)
    w_idx = torch.arange(w, dtype=torch.float32)
    ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing="ij")
    if action:
        ff_offset = (torch.ones([h], dtype=torch.float32).cumsum(0) / (h + 1)).view(1, -1, 1)
        ff = ff + ff_offset
        hh = torch.ones_like(hh) * -1
        ww = torch.ones_like(ww) * -1
    grid_id = torch.cat([ff.unsqueeze(0), hh.unsqueeze(0), ww.unsqueeze(0)], dim=0).flatten(1)
    return torch.cat([grid_id, torch.full_like(grid_id[:1], float(t))], dim=0)


class VerifyDataset(Dataset):
    def __init__(
        self,
        success_root: Path,
        failure_root: Path,
        split: str = "train",
        obs_window: int = 6,
        action_chunk_size: int = 16,
        positive_prob: float = 0.5,
        seed: int = 0,
        max_success_repos: int | None = None,
        max_failure_records: int | None = None,
    ):
        if action_chunk_size != ACTION_CHUNK_SIZE:
            raise ValueError("This verifier currently assumes action_chunk_size=16 to match RoboTwin 1:16 video/action alignment.")
        self.success_root = success_root
        self.failure_root = failure_root
        self.obs_window = obs_window
        self.action_chunk_size = action_chunk_size
        self.positive_prob = positive_prob
        self.rng = random.Random(seed + (0 if split == "train" else 100000))
        self.action_preprocess = RobotwinActionPreprocessor()

        self.success_records = build_success_index(success_root, max_repos=max_success_repos)
        self.success_by_key = {(r.repo_rel, r.episode_index): r for r in self.success_records}

        manifest = failure_root / "verify_failure_manifest.jsonl"
        raw_failure_records = read_jsonl(manifest)
        self.failure_records = [
            neg for neg in raw_failure_records
            if self.failure_record_has_cached_latents(neg)
        ]
        print(
            f"kept {len(self.failure_records)}/{len(raw_failure_records)} failure records "
            "whose source action_config has complete cached latents"
        )
        if max_failure_records is not None:
            self.failure_records = self.failure_records[:max_failure_records]
        if not self.failure_records:
            raise FileNotFoundError(f"No usable failure records found in {manifest}")

    def __len__(self) -> int:
        return max(len(self.success_records), len(self.failure_records))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.rng.random() < self.positive_prob:
            return self.make_positive()
        return self.make_negative(idx)

    def failure_record_has_cached_latents(self, neg: dict[str, Any]) -> bool:
        key = (neg.get("source_repo"), int(neg.get("episode_index", -1)))
        record = self.success_by_key.get(key)
        if record is None:
            return False
        current_frame = int(neg.get("current_observation_frame", neg.get("bad_action_chunk_start", -1)))
        return any(
            int(acfg.get("start_frame", 0)) <= current_frame < int(acfg.get("end_frame", record.length))
            for acfg in record.action_configs
        )

    def load_actions(self, root: Path, record: EpisodeRecord) -> np.ndarray:
        path = root / record.repo_rel / format_episode_data_path(record)
        return np.asarray(pq.read_table(path, columns=["action"])["action"].to_pylist(), dtype=np.float32)

    def make_positive(self) -> dict[str, Any]:
        record = self.rng.choice(self.success_records)
        action_configs = list(record.action_configs)
        if action_configs:
            acfg = self.rng.choice(action_configs)
            action_start = int(acfg.get("start_frame", 0))
            action_end = int(acfg.get("end_frame", record.length))
        else:
            action_start, action_end = 0, record.length
        _, chunk_start, chunk_end, valid_end = choose_chunk(action_start, action_end, self.action_chunk_size, self.rng)
        actions = self.load_actions(self.success_root, record)
        raw_chunk = make_padded_action_chunk(actions, chunk_start, chunk_end, valid_end)
        return self.pack_sample(record, chunk_start, raw_chunk, label=0)

    def make_negative(self, idx: int) -> dict[str, Any]:
        neg = self.failure_records[idx % len(self.failure_records)]
        record = self.success_by_key[(neg["source_repo"], int(neg["episode_index"]))]
        chunk_path = self.failure_root / neg["candidate_action_chunk_path"]
        raw_chunk = np.load(chunk_path)["actions"].astype(np.float32)
        chunk_start = int(neg["current_observation_frame"])
        return self.pack_sample(record, chunk_start, raw_chunk, label=1)

    def latent_range_for_frame(self, record: EpisodeRecord, current_action_frame: int) -> tuple[int, int]:
        for acfg in record.action_configs:
            start = int(acfg.get("start_frame", 0))
            end = int(acfg.get("end_frame", record.length))
            if start <= current_action_frame < end:
                return start, end
        raise FileNotFoundError(
            f"No valid cached-latent action_config for {record.repo_rel} "
            f"episode={record.episode_index} frame={current_action_frame}"
        )

    def pack_sample(self, record: EpisodeRecord, chunk_start: int, raw_chunk: np.ndarray, label: int) -> dict[str, Any]:
        latent_start, latent_end = self.latent_range_for_frame(record, chunk_start)
        return {
            "actions": torch.from_numpy(self.action_preprocess(raw_chunk)),
            "labels": torch.tensor(label, dtype=torch.long),
            "class_ids": torch.tensor(label + 1, dtype=torch.long),
            "current_action_frame": torch.tensor(chunk_start, dtype=torch.long),
            "repo_rel": record.repo_rel,
            "episode_index": record.episode_index,
            "chunks_size": record.chunks_size,
            "latent_start": latent_start,
            "latent_end": latent_end,
        }


def verify_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "actions": torch.stack([s["actions"] for s in samples], dim=0),
        "labels": torch.stack([s["labels"] for s in samples], dim=0),
        "class_ids": torch.stack([s["class_ids"] for s in samples], dim=0),
        "current_action_frame": torch.stack([s["current_action_frame"] for s in samples], dim=0),
        "repo_rel": [s["repo_rel"] for s in samples],
        "episode_index": torch.tensor([s["episode_index"] for s in samples], dtype=torch.long),
        "chunks_size": torch.tensor([s["chunks_size"] for s in samples], dtype=torch.long),
        "latent_start": torch.tensor([s["latent_start"] for s in samples], dtype=torch.long),
        "latent_end": torch.tensor([s["latent_end"] for s in samples], dtype=torch.long),
    }


class CachedConditionEncoder:
    """Load WAN-VA cached robotwin_tshape latents/text_emb produced for standard training."""

    def __init__(
        self,
        success_root: Path,
        obs_window: int = 6,
        action_chunk_size: int = 16,
        cache_items: int = 128,
    ):
        self.success_root = success_root
        self.obs_window = obs_window
        self.action_chunk_size = action_chunk_size
        self.cache_items = cache_items
        self.cache: OrderedDict[Path, dict[str, Any]] = OrderedDict()

    def get_episode_chunk(self, chunks_size: int, episode_index: int) -> int:
        return int(episode_index) // int(chunks_size)

    def latent_file_path(
        self,
        repo_rel: str,
        episode_index: int,
        chunks_size: int,
        cam_key: str,
        start_frame: int,
        end_frame: int,
    ) -> Path:
        episode_chunk_id = self.get_episode_chunk(chunks_size, episode_index)
        return (
            self.success_root
            / repo_rel
            / "latents"
            / f"chunk-{episode_chunk_id:03d}"
            / cam_key
            / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
        )

    def load_latent_file(self, path: Path) -> dict[str, Any]:
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        data = torch.load(path, weights_only=False, map_location="cpu")
        self.cache[path] = data
        if len(self.cache) > self.cache_items:
            self.cache.popitem(last=False)
        return data

    def reshape_cam_latent(self, data: dict[str, Any]) -> torch.Tensor:
        latent = data["latent"].float()
        f = int(data["latent_num_frames"])
        h = int(data["latent_height"])
        w = int(data["latent_width"])
        return latent.reshape(f, h, w, latent.shape[-1])

    def load_full_tshape_latent(
        self,
        repo_rel: str,
        episode_index: int,
        chunks_size: int,
        start_frame: int,
        end_frame: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cam_data = []
        for cam_key in CAM_KEYS:
            path = self.latent_file_path(repo_rel, episode_index, chunks_size, cam_key, start_frame, end_frame)
            if not path.exists():
                raise FileNotFoundError(f"Missing cached latent file: {path}")
            cam_data.append(self.load_latent_file(path))
        latents = [self.reshape_cam_latent(data) for data in cam_data]
        wrist_latent = torch.cat(latents[1:], dim=2)
        cat_latent = torch.cat([wrist_latent, latents[0]], dim=1)  # [F, 24, 20, 48]
        text_emb = cam_data[0]["text_emb"].float()
        return cat_latent.permute(3, 0, 1, 2).contiguous(), text_emb  # [48, F, H, W], [512, 4096]

    def latent_window_indices(self, current_action_frame: int, start_frame: int, latent_num_frames: int) -> list[int]:
        current_idx = (int(current_action_frame) - int(start_frame)) // self.action_chunk_size
        current_idx = max(0, min(current_idx, latent_num_frames - 1))
        ids = [current_idx - i for i in reversed(range(self.obs_window))]
        return [max(0, min(i, latent_num_frames - 1)) for i in ids]

    @torch.no_grad()
    def encode_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        obs_latents = []
        text_embs = []
        text_masks = []
        for i, repo_rel in enumerate(batch["repo_rel"]):
            episode_index = int(batch["episode_index"][i].item())
            chunks_size = int(batch["chunks_size"][i].item())
            start_frame = int(batch["latent_start"][i].item())
            end_frame = int(batch["latent_end"][i].item())
            current_frame = int(batch["current_action_frame"][i].item())
            full_latent, text_emb = self.load_full_tshape_latent(
                repo_rel,
                episode_index,
                chunks_size,
                start_frame,
                end_frame,
            )
            frame_ids = self.latent_window_indices(current_frame, start_frame, full_latent.shape[1])
            obs_latents.append(full_latent[:, frame_ids])
            text_embs.append(text_emb)
            text_masks.append(text_emb.abs().sum(dim=-1) > 0)
        return {
            "obs_latents": torch.stack(obs_latents, dim=0),
            "text_emb": torch.stack(text_embs, dim=0),
            "text_mask": torch.stack(text_masks, dim=0),
        }


class VerifyRotaryPosEmbed(nn.Module):
    """Small RoPE module following WAM's WanRotaryPosEmbed grid convention."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")
        h_dim = (head_dim // 3) // 2 * 2
        w_dim = h_dim
        f_dim = head_dim - h_dim - w_dim
        if f_dim % 2 != 0:
            f_dim -= 1
            h_dim += 1
        if h_dim % 2 != 0 or w_dim % 2 != 0 or f_dim <= 0:
            raise ValueError(f"Unsupported head_dim for 3D RoPE: {head_dim}")
        self.f_dim = f_dim
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.theta = theta
        self.register_buffer("f_freqs_base", self._freqs_base(f_dim), persistent=False)
        self.register_buffer("h_freqs_base", self._freqs_base(h_dim), persistent=False)
        self.register_buffer("w_freqs_base", self._freqs_base(w_dim), persistent=False)

    def _freqs_base(self, dim: int) -> torch.Tensor:
        return 1.0 / (self.theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))

    def forward(self, grid_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            grid_ids = grid_ids.to(dtype=torch.float32)
            f_freqs = grid_ids[:, 0, :].unsqueeze(-1) * self.f_freqs_base.to(grid_ids.device)
            h_freqs = grid_ids[:, 1, :].unsqueeze(-1) * self.h_freqs_base.to(grid_ids.device)
            w_freqs = grid_ids[:, 2, :].unsqueeze(-1) * self.w_freqs_base.to(grid_ids.device)
            freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).float()
            return torch.polar(torch.ones_like(freqs), freqs)[:, :, None]


class RotarySelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        self.norm_q = nn.RMSNorm(d_model)
        self.norm_k = nn.RMSNorm(d_model)
        self.out = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(dropout))
        self.dropout = dropout
        self.rope = VerifyRotaryPosEmbed(self.head_dim)

    def apply_rotary(self, x: torch.Tensor, rotary_emb: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rot = torch.view_as_real(x_complex * rotary_emb).flatten(-2)
        return x_rot.to(dtype=x.dtype)

    def forward(self, x: torch.Tensor, grid_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        q = self.norm_q(self.to_q(x)).view(b, l, self.nhead, self.head_dim)
        k = self.norm_k(self.to_k(x)).view(b, l, self.nhead, self.head_dim)
        v = self.to_v(x).view(b, l, self.nhead, self.head_dim)
        rotary_emb = self.rope(grid_ids.to(x.device))
        q = self.apply_rotary(q, rotary_emb)
        k = self.apply_rotary(k, rotary_emb)
        float_mask = torch.zeros((l, l), device=x.device, dtype=q.dtype)
        float_mask = float_mask.masked_fill(attn_mask.to(x.device), float("-inf"))
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=float_mask[None, None],
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(b, l, -1)
        return self.out(out)


class VerifyBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attn = RotarySelfAttention(d_model, nhead, dropout)
        self.text_norm = nn.LayerNorm(d_model)
        self.text_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        grid_ids: torch.Tensor,
        self_mask: torch.Tensor,
        text_emb: torch.Tensor,
        text_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.self_norm(x)
        x = x + self.self_attn(h, grid_ids, self_mask)
        h = self.text_norm(x)
        text_out, _ = self.text_attn(
            h,
            text_emb,
            text_emb,
            key_padding_mask=text_key_padding_mask,
            need_weights=False,
        )
        x = x + text_out
        return x + self.mlp(self.mlp_norm(x))


class VerifyTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 5,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        latent_channels: int = 48,
        text_dim: int = 4096,
        max_tokens: int = 8192,
        max_latent_frames: int = 32,
        latent_token_mode: str = "spatial",
    ):
        super().__init__()
        if latent_token_mode not in {"spatial", "pooled"}:
            raise ValueError("latent_token_mode must be 'spatial' or 'pooled'")
        self.latent_token_mode = latent_token_mode
        self.latent_patch_size = (1, 2, 2)
        self.latent_patch_embed = nn.Linear(latent_channels * 4, d_model)
        self.latent_pool_embed = nn.Linear(latent_channels, d_model)
        self.action_embed = nn.Linear(30, d_model)
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.type_embed = nn.Embedding(4, d_model)
        self.blocks = nn.ModuleList([
            VerifyBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)
        self.max_tokens = max_tokens
        self.max_latent_frames = max_latent_frames

    def latent_to_tokens(
        self,
        latents: torch.Tensor,
        token_type: int,
        frame_shift: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.latent_token_mode == "pooled":
            b, c, f, _, _ = latents.shape
            x = latents.mean(dim=(-1, -2)).permute(0, 2, 1).contiguous()  # [B, F, C]
            grid_ids = get_mesh_id(f, 1, 1, t=token_type, f_shift=frame_shift).to(latents.device)
            frame_ids = grid_ids[0].long()
            return self.latent_pool_embed(x), frame_ids, grid_ids
        b, c, f, h, w = latents.shape
        p_t, p_h, p_w = self.latent_patch_size
        if f % p_t != 0 or h % p_h != 0 or w % p_w != 0:
            raise ValueError(f"Latent shape {(f, h, w)} is not divisible by WAM patch_size {self.latent_patch_size}")
        # Match WAM inference latent branch:
        # rearrange 'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)'.
        x = latents.reshape(b, c, f // p_t, p_t, h // p_h, p_h, w // p_w, p_w)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(b, (f // p_t) * (h // p_h) * (w // p_w), c * p_t * p_h * p_w)
        grid_ids = get_mesh_id(f // p_t, h // p_h, w // p_w, t=token_type, f_shift=frame_shift).to(latents.device)
        frame_ids = grid_ids[0].long()
        return self.latent_patch_embed(x), frame_ids, grid_ids

    def action_grid_ids(self, action_tokens: torch.Tensor, current_frame: int) -> tuple[torch.Tensor, torch.Tensor]:
        grid_ids = get_mesh_id(1, action_tokens.shape[1], 1, t=1, f_shift=current_frame, action=True).to(action_tokens.device)
        frame_ids = torch.full((action_tokens.shape[1],), current_frame, dtype=torch.long, device=action_tokens.device)
        return frame_ids, grid_ids

    def build_visibility_mask(
        self,
        obs_frame_ids: torch.Tensor,
        n_action: int,
        device: torch.device,
    ) -> torch.Tensor:
        n_obs = int(obs_frame_ids.numel())
        cls_idx = 0
        obs_start = 1
        action_start = obs_start + n_obs
        total = action_start + n_action
        mask = torch.ones(total, total, dtype=torch.bool, device=device)

        mask[cls_idx, :] = False
        for q in range(obs_start, action_start):
            t_q = obs_frame_ids[q - obs_start]
            mask[q, cls_idx] = False
            mask[q, obs_start:action_start] = ~(obs_frame_ids <= t_q)

        for q in range(action_start, total):
            mask[q, cls_idx] = False
            mask[q, obs_start:action_start] = False
            mask[q, action_start:total] = False
        return mask

    def forward(
        self,
        obs_latents: torch.Tensor,
        actions: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        b = obs_latents.shape[0]
        obs_tokens, obs_frame_ids, obs_grid_ids = self.latent_to_tokens(obs_latents, token_type=0)
        current_frame = int(obs_frame_ids.max().item()) if obs_frame_ids.numel() else 0
        action_tokens = self.action_embed(actions)
        action_frame_ids, action_grid_ids = self.action_grid_ids(action_tokens, current_frame)
        text_tokens = self.text_proj(text_emb.float()).to(dtype=obs_tokens.dtype)

        cls = self.cls.repeat(b, 1, 1).to(dtype=obs_tokens.dtype)
        tokens = torch.cat([cls, obs_tokens, action_tokens], dim=1)
        if tokens.shape[1] > self.max_tokens:
            raise ValueError(f"Token count {tokens.shape[1]} exceeds --max-tokens {self.max_tokens}")
        if int(obs_frame_ids.max().item()) >= self.max_latent_frames:
            raise ValueError("Latent frame count exceeds --max-latent-frames")

        type_ids = torch.cat([
            torch.zeros(b, 1, dtype=torch.long, device=tokens.device),
            torch.ones(b, obs_tokens.shape[1], dtype=torch.long, device=tokens.device),
            torch.full((b, action_tokens.shape[1]), 2, dtype=torch.long, device=tokens.device),
        ], dim=1)
        cls_grid_ids = torch.tensor([[current_frame], [0.0], [0.0], [2.0]], device=tokens.device, dtype=torch.float32)
        grid_ids = torch.cat([cls_grid_ids, obs_grid_ids, action_grid_ids], dim=1)
        tokens = tokens + self.type_embed(type_ids)

        self_mask = self.build_visibility_mask(
            obs_frame_ids,
            action_tokens.shape[1],
            tokens.device,
        )
        text_key_padding_mask = ~text_mask.to(tokens.device).bool()
        for block in self.blocks:
            tokens = block(tokens, grid_ids[None], self_mask, text_tokens, text_key_padding_mask)
        return torch.sigmoid(self.head(self.final_norm(tokens[:, 0])))


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = VerifyDataset(
        success_root=args.success_root,
        failure_root=args.failure_root,
        obs_window=args.obs_window,
        action_chunk_size=args.action_chunk_size,
        positive_prob=args.positive_prob,
        seed=args.seed,
        max_success_repos=args.max_success_repos,
        max_failure_records=args.max_failure_records,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=verify_collate,
    )

    condition_encoder = CachedConditionEncoder(
        success_root=args.success_root,
        obs_window=args.obs_window,
        action_chunk_size=args.action_chunk_size,
        cache_items=args.cache_latent_items,
    )
    model = VerifyTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=5,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        latent_channels=args.latent_channels,
        text_dim=args.text_dim,
        max_tokens=args.max_tokens,
        max_latent_frames=args.max_latent_frames,
        latent_token_mode=args.latent_token_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.save_root.mkdir(parents=True, exist_ok=True)
    step = 0
    data_iter = iter(loader)
    pbar = tqdm(total=args.steps, desc="verify train", dynamic_ncols=True)
    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        cond = condition_encoder.encode_batch(batch)
        obs_latents = cond["obs_latents"].to(device=device, dtype=torch.float32)
        text_emb = cond["text_emb"].to(device=device)
        text_mask = cond["text_mask"].to(device=device)
        actions = batch["actions"].to(device=device, dtype=torch.float32, non_blocking=True)
        labels = batch["labels"].to(device=device, non_blocking=True)

        logits = model(obs_latents, actions, text_emb, text_mask)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            acc = (pred == labels).float().mean().item()
        step += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.3f}")

        if step % args.save_interval == 0 or step == args.steps:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "class_mapping": {"1": "correct", "2": "wrong_replan"},
                "input_contract": {
                    "obs_latents": "cached robotwin_tshape latent from latents/*.pth, same source as MultiLatentLeRobotDataset",
                    "text_emb": "cached text_emb from latents/*.pth, same source as standard WAM training",
                    "action_chunk": "16 raw actions -> relative pose/gripper -> 30 normalized channels",
                },
            }
            torch.save(ckpt, args.save_root / f"verify_step_{step}.pt")
    pbar.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a RoboTwin action-chunk verifier binary classifier.")
    parser.add_argument("--success-root", type=Path, default=Path("/mnt/public/xieruiqi/datasets/lingbot-va/robotwin/robotwin-clean-and-aug-lerobot"))
    parser.add_argument("--failure-root", type=Path, default=Path("/mnt/public/xieruiqi/datasets/lingbot-va/robotwin/robotwin-verify-failure-lerobot"))
    parser.add_argument("--save-root", type=Path, default=Path("./verify_train_out"))
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=20260630)

    parser.add_argument("--obs-window", type=int, default=6)
    parser.add_argument("--action-chunk-size", type=int, default=16)
    parser.add_argument("--positive-prob", type=float, default=0.5)
    parser.add_argument("--cache-latent-items", type=int, default=128)

    parser.add_argument("--latent-channels", type=int, default=48)
    parser.add_argument("--text-dim", type=int, default=4096)
    parser.add_argument("--latent-token-mode", type=str, default="spatial", choices=["spatial", "pooled"])

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-latent-frames", type=int, default=32)

    parser.add_argument("--max-success-repos", type=int, default=None)
    parser.add_argument("--max-failure-records", type=int, default=None)
    args = parser.parse_args()
    if args.obs_window < 1 or args.obs_window > 6:
        raise ValueError("--obs-window must be in [1, 6]")
    if args.action_chunk_size != 16:
        raise ValueError("--action-chunk-size must be 16 for current RoboTwin 1:16 alignment")
    if not 0 < args.positive_prob < 1:
        raise ValueError("--positive-prob must be in (0, 1)")
    return args


if __name__ == "__main__":
    train(parse_args())
