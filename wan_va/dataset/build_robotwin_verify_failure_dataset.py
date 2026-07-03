#!/usr/bin/env python3
"""Build negative RoboTwin trajectories for verifier training.

The source dataset is assumed to use RoboTwin ee actions:

    [left_xyz, left_quat_xyzw, left_gripper,
     right_xyz, right_quat_xyzw, right_gripper]

This matches the official RoboTwin deployment comment for `action_type='ee'`
and Lingbot's Robotwin evaluation visualization helper. The generated dataset
keeps observations/videos/latents/instructions unchanged and corrupts one
future action chunk in each LeRobot episode parquet. This makes the resulting
data useful as "would require replan" negatives against the original successful
trajectories while preserving the temporal relation:

    current observation frame t -> candidate action chunk [t, t + 16)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


ACTION_DIM = 16
LEFT_POSE_DIMS = list(range(0, 7))
LEFT_GRIPPER_DIM = 7
RIGHT_POSE_DIMS = list(range(8, 15))
RIGHT_GRIPPER_DIM = 15
POSE_DIMS = LEFT_POSE_DIMS + RIGHT_POSE_DIMS
GRIPPER_DIMS = [LEFT_GRIPPER_DIM, RIGHT_GRIPPER_DIM]
DEFAULT_ACTION_CHUNK_SIZE = 16


@dataclass
class AugmentRecord:
    label: int
    needs_replan: bool
    failure_type: str
    source_repo: str
    output_repo: str
    episode_index: int
    chunk_index: int
    current_observation_frame: int
    bad_action_chunk_start: int
    bad_action_chunk_end: int
    valid_action_chunk_end: int
    action_chunk_size: int
    valid_action_steps: int
    padded_action_steps: int
    candidate_action_chunk_path: str
    source_episode_path: str
    output_episode_path: str
    seed: int
    params: dict


def _iter_repo_roots(root: Path) -> list[Path]:
    return sorted(path.parent.parent for path in root.rglob("meta/info.json"))


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)
        f.write("\n")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _relative_to_root(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src, dst, target_is_directory=src.is_dir())


def _copy_or_link_tree(src: Path, dst: Path, mode: str) -> None:
    if not src.exists():
        return
    if mode == "symlink":
        _safe_symlink(src.resolve(), dst)
    elif mode == "copy":
        if dst.exists():
            return
        shutil.copytree(src, dst, symlinks=True)
    elif mode == "skip":
        return
    else:
        raise ValueError(f"Unsupported link/copy mode: {mode}")


def _copy_metadata(src_repo: Path, dst_repo: Path, repo_records: list[AugmentRecord]) -> None:
    src_meta = src_repo / "meta"
    dst_meta = dst_repo / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)

    records_by_episode = {record.episode_index: record for record in repo_records}

    for src_file in src_meta.iterdir():
        dst_file = dst_meta / src_file.name
        if src_file.name == "episodes.jsonl":
            episodes = _read_jsonl(src_file)
            for episode in episodes:
                episode_index = int(episode["episode_index"])
                record = records_by_episode.get(episode_index)
                if record is None:
                    continue
                episode["verify_label"] = 0
                episode["needs_replan"] = True
                episode["verify_source"] = "augmented_failure"
                episode["failure_type"] = record.failure_type
                episode["failure_params"] = record.params
                episode["verify_bad_action_chunk"] = {
                    "chunk_index": record.chunk_index,
                    "current_observation_frame": record.current_observation_frame,
                    "start_frame": record.bad_action_chunk_start,
                    "end_frame": record.bad_action_chunk_end,
                    "valid_end_frame": record.valid_action_chunk_end,
                    "action_chunk_size": record.action_chunk_size,
                    "valid_action_steps": record.valid_action_steps,
                    "padded_action_steps": record.padded_action_steps,
                    "candidate_action_chunk_path": record.candidate_action_chunk_path,
                    "padding": "repeat_last_action" if record.padded_action_steps > 0 else "none",
                    "relation": (
                        "Use observations up to current_observation_frame, then judge "
                        "the future candidate action chunk [start_frame, end_frame)."
                    ),
                }
                for action_config in episode.get("action_config", []):
                    action_config["verify_label"] = 0
                    action_config["needs_replan"] = True
                    action_config["failure_type"] = record.failure_type
                    action_config["failure_params"] = record.params
                    action_config["verify_bad_action_chunk"] = episode["verify_bad_action_chunk"]
            _write_jsonl(dst_file, episodes)
        elif src_file.is_file():
            shutil.copy2(src_file, dst_file)

    _write_jsonl(dst_meta / "verify_labels.jsonl", (asdict(r) for r in repo_records))


def _episode_path_from_info(repo: Path, info: dict, episode_index: int) -> Path:
    episode_chunk = episode_index // int(info.get("chunks_size", 1000))
    rel = info["data_path"].format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )
    return repo / rel


def _ensure_action_array(table: pa.Table, episode_path: Path) -> np.ndarray:
    if "action" not in table.column_names:
        raise ValueError(f"No action column in {episode_path}")
    actions = table["action"].to_pylist()
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != ACTION_DIM:
        raise ValueError(
            f"Expected action shape (T, {ACTION_DIM}) in {episode_path}, got {arr.shape}"
        )
    return arr


def _replace_action_column(table: pa.Table, actions: np.ndarray) -> pa.Table:
    field = table.schema.field("action")
    action_array = pa.array(actions.astype(np.float32).tolist(), type=field.type)
    action_idx = table.column_names.index("action")
    return table.set_column(action_idx, field, action_array)


def temporal_swap(actions: np.ndarray, rng: np.random.Generator, pair_width: int = 2,
                  swaps: int = 1) -> tuple[np.ndarray, dict]:
    out = actions.copy()
    horizon = len(out)
    applied = []
    if horizon < pair_width * 2:
        return out, {"pair_width": pair_width, "swaps": 0, "pairs": applied}

    for _ in range(swaps):
        max_start = horizon - pair_width
        for _attempt in range(32):
            i = int(rng.integers(0, max_start + 1))
            j = int(rng.integers(0, max_start + 1))
            if abs(i - j) >= pair_width:
                break
        else:
            continue
        tmp = out[i:i + pair_width].copy()
        out[i:i + pair_width] = out[j:j + pair_width]
        out[j:j + pair_width] = tmp
        applied.append([i, j])

    return out, {"pair_width": pair_width, "swaps": len(applied), "pairs": applied}


def gripper_flip(actions: np.ndarray, rng: np.random.Generator,
                 dims: list[int] | None = None, mode: str = "invert01") -> tuple[np.ndarray, dict]:
    del rng
    out = actions.copy()
    dims = GRIPPER_DIMS if dims is None else dims
    if mode == "negate":
        out[:, dims] = -out[:, dims]
    elif mode == "invert01":
        out[:, dims] = 1.0 - out[:, dims]
    else:
        raise ValueError(f"Unsupported gripper flip mode: {mode}")
    return out, {"dims": dims, "mode": mode}


def late_stage_gaussian_noise(actions: np.ndarray, rng: np.random.Generator,
                              pos_std: float = 0.04,
                              quat_std: float = 0.08,
                              gripper_std: float = 0.0,
                              start_fraction: float = 0.5) -> tuple[np.ndarray, dict]:
    out = actions.copy()
    start = int(round(len(out) * start_fraction))
    if start >= len(out):
        start = max(0, len(out) - 1)

    pos_dims = [0, 1, 2, 8, 9, 10]
    quat_dims = [3, 4, 5, 6, 11, 12, 13, 14]

    out[start:, pos_dims] += rng.normal(0.0, pos_std, size=(len(out) - start, len(pos_dims)))
    out[start:, quat_dims] += rng.normal(0.0, quat_std, size=(len(out) - start, len(quat_dims)))
    if gripper_std > 0:
        out[start:, GRIPPER_DIMS] += rng.normal(
            0.0, gripper_std, size=(len(out) - start, len(GRIPPER_DIMS))
        )

    return out, {
        "start": start,
        "start_fraction": start_fraction,
        "pos_std": pos_std,
        "quat_std": quat_std,
        "gripper_std": gripper_std,
    }


def tail_scaling(actions: np.ndarray, rng: np.random.Generator,
                 min_suffix_fraction: float = 0.2,
                 max_suffix_fraction: float = 0.6,
                 min_scale: float = 0.15,
                 max_scale: float = 0.7) -> tuple[np.ndarray, dict]:
    out = actions.copy()
    horizon = len(out)
    if horizon < 2:
        return out, {
            "start": 0,
            "scale": 1.0,
            "suffix_fraction": 1.0,
            "pose_dims": POSE_DIMS,
        }

    suffix_fraction = float(rng.uniform(min_suffix_fraction, max_suffix_fraction))
    suffix_len = max(1, int(round(horizon * suffix_fraction)))
    start = max(0, horizon - suffix_len)
    scale = float(rng.uniform(min_scale, max_scale))

    anchor = out[start:start + 1, POSE_DIMS].copy()
    out[start:, POSE_DIMS] = anchor + scale * (out[start:, POSE_DIMS] - anchor)

    return out, {
        "start": start,
        "scale": scale,
        "suffix_fraction": suffix_fraction,
        "pose_dims": POSE_DIMS,
    }


def choose_action_chunk(
    action_start: int,
    action_end: int,
    action_chunk_size: int,
    rng: np.random.Generator,
) -> tuple[int, int, int, int]:
    if action_chunk_size < 1:
        raise ValueError("action_chunk_size must be >= 1")
    if action_end <= action_start:
        raise ValueError(f"Invalid action range [{action_start}, {action_end})")

    num_chunks = int(np.ceil((action_end - action_start) / action_chunk_size))
    chunk_index = int(rng.integers(0, num_chunks))
    chunk_start = action_start + chunk_index * action_chunk_size
    chunk_end = chunk_start + action_chunk_size
    valid_chunk_end = min(chunk_end, action_end)
    return chunk_index, chunk_start, chunk_end, valid_chunk_end


AUGMENT_FNS = {
    "temporal_swap": temporal_swap,
    "gripper_flip": gripper_flip,
    "late_stage_gaussian_noise": late_stage_gaussian_noise,
    "tail_scaling": tail_scaling,
}


def _choose_method(methods: list[str], rng: np.random.Generator) -> str:
    return methods[int(rng.integers(0, len(methods)))]


def _write_augmented_episode(
    src_episode_path: Path,
    dst_episode_path: Path,
    candidate_chunk_path: Path,
    method: str,
    chunk_start: int,
    chunk_end: int,
    valid_chunk_end: int,
    seed: int,
    args: argparse.Namespace,
) -> tuple[str, dict, int, int]:
    rng = np.random.default_rng(seed)
    table = pq.read_table(src_episode_path)
    actions = _ensure_action_array(table, src_episode_path)
    if chunk_start < 0 or chunk_start >= len(actions) or valid_chunk_end > len(actions):
        raise ValueError(
            f"Invalid chunk [{chunk_start}, {valid_chunk_end}) for action length {len(actions)} "
            f"in {src_episode_path}"
        )
    if valid_chunk_end <= chunk_start or chunk_end <= chunk_start:
        raise ValueError(f"Invalid chunk bounds: start={chunk_start}, end={chunk_end}, valid_end={valid_chunk_end}")

    raw_chunk = actions[chunk_start:valid_chunk_end]
    valid_action_steps = len(raw_chunk)
    padded_action_steps = chunk_end - valid_chunk_end
    if padded_action_steps > 0:
        pad = np.repeat(raw_chunk[-1:], padded_action_steps, axis=0)
        action_chunk = np.concatenate([raw_chunk, pad], axis=0)
    else:
        action_chunk = raw_chunk.copy()

    if len(action_chunk) != args.action_chunk_size:
        raise ValueError(
            f"Expected padded action chunk length {args.action_chunk_size}, got {len(action_chunk)}"
        )

    if method == "temporal_swap":
        aug_chunk, params = temporal_swap(
            action_chunk,
            rng,
            pair_width=args.temporal_pair_width,
            swaps=args.temporal_swaps,
        )
    elif method == "gripper_flip":
        aug_chunk, params = gripper_flip(action_chunk, rng, mode=args.gripper_flip_mode)
    elif method == "late_stage_gaussian_noise":
        aug_chunk, params = late_stage_gaussian_noise(
            action_chunk,
            rng,
            pos_std=args.noise_pos_std,
            quat_std=args.noise_quat_std,
            gripper_std=args.noise_gripper_std,
            start_fraction=args.noise_start_fraction,
        )
    elif method == "tail_scaling":
        aug_chunk, params = tail_scaling(
            action_chunk,
            rng,
            min_suffix_fraction=args.tail_min_suffix_fraction,
            max_suffix_fraction=args.tail_max_suffix_fraction,
            min_scale=args.tail_min_scale,
            max_scale=args.tail_max_scale,
        )
    else:
        raise ValueError(f"Unknown augmentation method: {method}")

    valid_prefix_changed = np.any(
        np.abs(aug_chunk[:valid_action_steps] - action_chunk[:valid_action_steps]) > 1e-7
    )
    if not valid_prefix_changed:
        fallback_dims = [0, 1, 2, 8, 9, 10]
        fallback_noise = rng.normal(
            0.0,
            args.valid_prefix_fallback_pos_std,
            size=(valid_action_steps, len(fallback_dims)),
        )
        aug_chunk[:valid_action_steps, fallback_dims] += fallback_noise
        params["valid_prefix_fallback"] = "gaussian_position_noise"
        params["valid_prefix_fallback_dims"] = fallback_dims
        params["valid_prefix_fallback_pos_std"] = args.valid_prefix_fallback_pos_std
    else:
        params["valid_prefix_fallback"] = "none"

    aug_actions = actions.copy()
    aug_actions[chunk_start:valid_chunk_end] = aug_chunk[:valid_action_steps]
    params.update({
        "local_chunk_start": chunk_start,
        "local_chunk_end": chunk_end,
        "valid_local_chunk_end": valid_chunk_end,
        "action_chunk_size": args.action_chunk_size,
        "valid_action_steps": valid_action_steps,
        "padded_action_steps": padded_action_steps,
        "padding": "repeat_last_action" if padded_action_steps > 0 else "none",
        "unchanged_prefix_frames": chunk_start,
        "unchanged_suffix_frames": len(actions) - valid_chunk_end,
    })

    dst_episode_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_replace_action_column(table, aug_actions), dst_episode_path)

    candidate_chunk_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        candidate_chunk_path,
        actions=aug_chunk.astype(np.float32),
        source_actions=action_chunk.astype(np.float32),
        valid_action_steps=np.asarray(valid_action_steps, dtype=np.int64),
        padded_action_steps=np.asarray(padded_action_steps, dtype=np.int64),
        chunk_start=np.asarray(chunk_start, dtype=np.int64),
        chunk_end=np.asarray(chunk_end, dtype=np.int64),
        valid_chunk_end=np.asarray(valid_chunk_end, dtype=np.int64),
    )
    return method, params, valid_action_steps, padded_action_steps


def _build_repo(
    src_root: Path,
    dst_root: Path,
    src_repo: Path,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[AugmentRecord]:
    rel_repo = src_repo.relative_to(src_root)
    dst_repo = dst_root / rel_repo
    info = _read_json(src_repo / "meta" / "info.json")
    episodes = _read_jsonl(src_repo / "meta" / "episodes.jsonl")

    if args.max_episodes_per_repo is not None:
        episodes = episodes[:args.max_episodes_per_repo]

    repo_records: list[AugmentRecord] = []
    episode_iter = tqdm(
        episodes,
        desc=f"episodes {src_repo.name}",
        leave=False,
        dynamic_ncols=True,
        disable=not args.progress,
    )
    for episode in episode_iter:
        episode_index = int(episode["episode_index"])
        src_episode_path = _episode_path_from_info(src_repo, info, episode_index)
        dst_episode_path = _episode_path_from_info(dst_repo, info, episode_index)
        method = _choose_method(args.methods, rng)
        episode_seed = int(rng.integers(0, 2**31 - 1))
        action_configs = episode.get("action_config", [])
        if not action_configs:
            action_start = 0
            action_end = int(episode.get("length", info.get("total_frames", 0)))
        else:
            action_config = action_configs[int(rng.integers(0, len(action_configs)))]
            action_start = int(action_config.get("start_frame", 0))
            action_end = int(action_config.get("end_frame", episode.get("length", 0)))
        chunk_index, chunk_start, chunk_end, valid_chunk_end = choose_action_chunk(
            action_start,
            action_end,
            args.action_chunk_size,
            rng,
        )
        candidate_chunk_path = (
            dst_repo
            / "verify_action_chunks"
            / f"episode_{episode_index:06d}_chunk_{chunk_index:06d}.npz"
        )

        failure_type, params, valid_action_steps, padded_action_steps = _write_augmented_episode(
            src_episode_path,
            dst_episode_path,
            candidate_chunk_path,
            method,
            chunk_start,
            chunk_end,
            valid_chunk_end,
            episode_seed,
            args,
        )

        repo_records.append(
            AugmentRecord(
                label=0,
                needs_replan=True,
                failure_type=failure_type,
                source_repo=rel_repo.as_posix(),
                output_repo=rel_repo.as_posix(),
                episode_index=episode_index,
                chunk_index=chunk_index,
                current_observation_frame=chunk_start,
                bad_action_chunk_start=chunk_start,
                bad_action_chunk_end=chunk_end,
                valid_action_chunk_end=valid_chunk_end,
                action_chunk_size=args.action_chunk_size,
                valid_action_steps=valid_action_steps,
                padded_action_steps=padded_action_steps,
                candidate_action_chunk_path=candidate_chunk_path.relative_to(dst_root).as_posix(),
                source_episode_path=src_episode_path.relative_to(src_root).as_posix(),
                output_episode_path=dst_episode_path.relative_to(dst_root).as_posix(),
                seed=episode_seed,
                params=params,
            )
        )

    _copy_metadata(src_repo, dst_repo, repo_records)
    _copy_or_link_tree(src_repo / "videos", dst_repo / "videos", args.media_mode)
    _copy_or_link_tree(src_repo / "latents", dst_repo / "latents", args.latent_mode)

    return repo_records


def build_failure_dataset(args: argparse.Namespace) -> None:
    src_root = args.src_root.resolve()
    dst_root = args.dst_root.resolve()
    if src_root == dst_root:
        raise ValueError("src_root and dst_root must be different")
    if not (src_root / "empty_emb.pt").exists():
        raise FileNotFoundError(f"Expected empty_emb.pt under source root: {src_root}")

    dst_root.mkdir(parents=True, exist_ok=True)
    if args.copy_root_files:
        for name in ["README.md", "empty_emb.pt"]:
            src_file = src_root / name
            if src_file.exists():
                shutil.copy2(src_file, dst_root / name)

    repo_roots = _iter_repo_roots(src_root)
    if args.max_repos is not None:
        repo_roots = repo_roots[:args.max_repos]
    if not repo_roots:
        raise FileNotFoundError(f"No LeRobot repos found under {src_root}")

    rng = np.random.default_rng(args.seed)
    all_records: list[AugmentRecord] = []

    repo_iter = tqdm(
        repo_roots,
        desc="repos",
        dynamic_ncols=True,
        disable=not args.progress,
    )
    for repo_id, src_repo in enumerate(repo_iter):
        records = _build_repo(src_root, dst_root, src_repo, args, rng)
        all_records.extend(records)
        if args.verbose:
            print(
                f"[{repo_id + 1}/{len(repo_roots)}] {src_repo.relative_to(src_root)}: "
                f"{len(records)} negative episodes"
            )

    manifest_path = dst_root / "verify_failure_manifest.jsonl"
    _write_jsonl(manifest_path, (asdict(r) for r in all_records))

    summary = {
        "source_root": str(src_root),
        "output_root": str(dst_root),
        "label": 0,
        "num_repos": len(repo_roots),
        "num_episodes": len(all_records),
        "methods": args.methods,
        "media_mode": args.media_mode,
        "latent_mode": args.latent_mode,
        "seed": args.seed,
        "action_layout": {
            "0:3": "left_xyz",
            "3:7": "left_quat_xyzw",
            "7": "left_gripper",
            "8:11": "right_xyz",
            "11:15": "right_quat_xyzw",
            "15": "right_gripper",
        },
        "temporal_relation": {
            "current_observation_frame": "t",
            "candidate_action_chunk": "[t, t + action_chunk_size)",
            "tail_padding": "If the final chunk is shorter than action_chunk_size, repeat the last real action to form a full candidate chunk.",
            "candidate_action_chunk_path": "NPZ file containing full padded actions and source_actions chunks.",
            "label_0_meaning": "the candidate action chunk is corrupted and should trigger replan",
            "label_1_meaning": "the candidate action chunk is from the original successful trajectory",
        },
    }
    _write_json(dst_root / "verify_failure_summary.json", summary)
    print(f"Wrote {len(all_records)} negative episodes to {dst_root}")
    print(f"Manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate verifier negative RoboTwin LeRobot data by corrupting successful actions."
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("/mnt/public/xieruiqi/datasets/lingbot-va/robotwin/robotwin-clean-and-aug-lerobot"),
        help="Source successful RoboTwin LeRobot dataset root.",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        required=True,
        help="Output root for the failure dataset.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(AUGMENT_FNS.keys()),
        choices=list(AUGMENT_FNS.keys()),
        help="Augmentation methods to sample from.",
    )
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--max-repos", type=int, default=None)
    parser.add_argument("--max-episodes-per-repo", type=int, default=None)
    parser.add_argument(
        "--action-chunk-size",
        type=int,
        default=DEFAULT_ACTION_CHUNK_SIZE,
        help="Number of future raw action steps judged by the verifier at one current observation.",
    )
    parser.add_argument(
        "--media-mode",
        choices=["symlink", "copy", "skip"],
        default="symlink",
        help="How to place videos/ in the output repos.",
    )
    parser.add_argument(
        "--latent-mode",
        choices=["symlink", "copy", "skip"],
        default="symlink",
        help="How to place latents/ in the output repos.",
    )
    parser.add_argument("--copy-root-files", action="store_true", default=True)
    parser.add_argument("--no-copy-root-files", action="store_false", dest="copy_root_files")
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", action="store_false", dest="progress")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--temporal-pair-width", type=int, default=2)
    parser.add_argument("--temporal-swaps", type=int, default=2)
    parser.add_argument("--gripper-flip-mode", choices=["negate", "invert01"], default="invert01")
    parser.add_argument("--noise-pos-std", type=float, default=0.1)
    parser.add_argument("--noise-quat-std", type=float, default=0.15)
    parser.add_argument("--noise-gripper-std", type=float, default=0.0)
    parser.add_argument("--noise-start-fraction", type=float, default=0.5)
    parser.add_argument("--tail-min-suffix-fraction", type=float, default=0.2)
    parser.add_argument("--tail-max-suffix-fraction", type=float, default=0.6)
    parser.add_argument("--tail-min-scale", type=float, default=0.15)
    parser.add_argument("--tail-max-scale", type=float, default=0.7)
    parser.add_argument(
        "--valid-prefix-fallback-pos-std",
        type=float,
        default=0.04,
        help="Fallback XYZ noise std used when an augmentation only changes padded steps.",
    )

    args = parser.parse_args()
    if args.temporal_pair_width < 1:
        raise ValueError("--temporal-pair-width must be >= 1")
    if args.temporal_swaps < 1:
        raise ValueError("--temporal-swaps must be >= 1")
    if args.action_chunk_size < 1:
        raise ValueError("--action-chunk-size must be >= 1")
    if args.valid_prefix_fallback_pos_std <= 0:
        raise ValueError("--valid-prefix-fallback-pos-std must be > 0")
    if not 0 <= args.noise_start_fraction <= 1:
        raise ValueError("--noise-start-fraction must be in [0, 1]")
    if not 0 < args.tail_min_suffix_fraction <= args.tail_max_suffix_fraction <= 1:
        raise ValueError("tail suffix fractions must satisfy 0 < min <= max <= 1")
    if not 0 <= args.tail_min_scale <= args.tail_max_scale:
        raise ValueError("tail scales must satisfy 0 <= min <= max")
    return args


if __name__ == "__main__":
    build_failure_dataset(parse_args())
