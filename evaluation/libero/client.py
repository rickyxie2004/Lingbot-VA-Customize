import numpy as np
from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import WebsocketClientPolicy
import argparse
from libero.libero import benchmark
import time
from libero.libero.envs import OffScreenRenderEnv
from pathlib import Path
from tqdm import tqdm
from lerobot.datasets.utils import write_json
import os
import imageio
import cv2
from concurrent.futures import ThreadPoolExecutor
from wan_va.configs.va_libero_cfg import va_libero_cfg


def save_video(real_obs_list, save_path, fps=15, video_names=["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]):
    if not real_obs_list:
        print("❌ No real observation frames")
        return

    first_obs = real_obs_list[0]
    base_h, width_base = first_obs[video_names[0]].shape[:2]
    target_size = (width_base, base_h)
    
    print(f"Saving video: {len(real_obs_list)} frames...")

    final_frames = [
        np.hstack([cv2.resize(obs[name], target_size) for name in video_names]).astype(np.uint8)
        for obs in real_obs_list
    ]

    imageio.mimsave(save_path, final_frames, fps=fps)
    print(f"✅ Video saved to: {save_path}")


def construct_single_env(env_args):
    count = 0
    env = None
    env_creation = False
    while not env_creation and count < 5:
        try:
            env = OffScreenRenderEnv(**env_args)
            env_creation = True
        except Exception as e:
            print(f"Error!!!  construct env failed: {e}")
            time.sleep(5)
            count += 1
    if count >= 5:
        return None
    return env


def _extract_obs(obs):
    """
    Extract agentview and eye_in_hand images from raw env obs dict.

    Avoids torch round-trip: the env already returns uint8 numpy arrays [H, W, C].
    We just flip the vertical axis ([::-1]) and make a contiguous copy once.
    """
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1])
    eye_in_hand = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    return {"observation.images.agentview_rgb": agentview, "observation.images.eye_in_hand_rgb": eye_in_hand}


def init_single_env(env_in, init_state):
    env_in.reset()
    env_in.set_init_state(init_state)
    for _ in range(5):
        obs, _, _, _ = env_in.step([0.] * 7)
    return _extract_obs(obs)


def env_one_step(env_in, action):
    obs, _, done, _ = env_in.step(action)
    return _extract_obs(obs), done


def _format_profile(profile):
    parts = []
    for record in profile.get('records', []):
        suffix = f" x{record['count']}" if record.get('count', 1) > 1 else ''
        parts.append(f"{record['name']}={record['total_ms']:.2f}ms{suffix}")
    return (
        f"name={profile.get('name')} frame_st_id={profile.get('frame_st_id')} "
        f"recorded_cuda={profile.get('total_recorded_cuda_ms', 0):.2f}ms | " +
        ' | '.join(parts)
    )


def _print_profile(ret, task_idx, episode_idx, infer_idx):
    profile = ret.get('profile_action_latency') if isinstance(ret, dict) else None
    if not profile:
        return
    prefix = f"[ActionProfile] task={task_idx} episode={episode_idx} infer={infer_idx} "
    if 'records' in profile:
        print(prefix + _format_profile(profile))
    else:
        for name, sub_profile in profile.items():
            if sub_profile:
                print(prefix + f"{name}: " + _format_profile(sub_profile))


def _start_async_prefetch(executor, model, current_action, feedback_obs=None, feedback_state=None):
    payload = dict(async_prefetch=True, current_action=current_action)
    if feedback_obs is not None and feedback_state is not None:
        payload['feedback_obs'] = feedback_obs
        payload['feedback_state'] = feedback_state
    return executor.submit(model.infer, payload)


def run_one(model, libero_benchmark, task_idx, out_dir, episode_idx, print_profile=False):
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[libero_benchmark]()
    num_tasks = benchmark_instance.get_num_tasks()
    assert task_idx < num_tasks, f"Error: error id must smaller than {num_tasks}"
    prompt = benchmark_instance.get_task(task_idx).language
    env_args = {
                "bddl_file_name": benchmark_instance.get_task_bddl_file_path(task_idx),
                "camera_heights": 128,
                "camera_widths": 128,
            }
    init_states = benchmark_instance.get_task_init_states(task_idx)

    cur_env = construct_single_env(env_args)
    first_obs = init_single_env(cur_env, init_states[episode_idx % init_states.shape[0]])

    ret = model.infer(dict(reset=True, prompt=prompt))

    full_obs_list = []
    done = False
    first = True
    infer_idx = 0
    async_enabled = getattr(va_libero_cfg, 'enable_async_inference', False)
    async_executor = ThreadPoolExecutor(max_workers=1) if async_enabled else None
    async_future = None
    pending_feedback_obs = None
    pending_feedback_state = None
    current_ret = None
    try:
        while cur_env.env.timestep < 800:
            if current_ret is None:
                current_ret = model.infer(dict(obs=first_obs, prompt=prompt))
                if print_profile:
                    _print_profile(current_ret, task_idx, episode_idx, infer_idx)
            action = current_ret['action']

            if async_enabled and not first:
                async_future = _start_async_prefetch(
                    async_executor,
                    model,
                    action,
                    feedback_obs=pending_feedback_obs,
                    feedback_state=pending_feedback_state)
                pending_feedback_obs = None
                pending_feedback_state = None

            key_frame_list = []
            assert action.shape[2] % 4 == 0
            action_per_frame = action.shape[2] // 4
            start_idx = 1 if first else 0
            for i in range(start_idx, action.shape[1]):
                for j in range(action.shape[2]):
                    ee_action = action[:, i, j]
                    observes, done = env_one_step(cur_env, ee_action)
                    if done:
                        break
                    if (j+1) % action_per_frame == 0:
                        full_obs_list.append(observes)
                        key_frame_list.append(observes)

                if done:
                    break

            was_first = first
            first = False

            if done:
                break

            infer_idx += 1
            if async_enabled and not was_first:
                current_ret = async_future.result()
                async_future = None
                if print_profile:
                    _print_profile(current_ret, task_idx, episode_idx, infer_idx)
                pending_feedback_obs = key_frame_list
                pending_feedback_state = action
                infer_idx += 1
            else:
                ret = model.infer(dict(obs=key_frame_list, compute_kv_cache=True, imagine=False, state=action))
                if print_profile:
                    _print_profile(ret, task_idx, episode_idx, infer_idx)
                infer_idx += 1
                current_ret = None
    finally:
        if async_future is not None:
            async_future.result()
        if async_executor is not None:
            async_executor.shutdown(wait=True, cancel_futures=True)

    out_file = Path(out_dir) / libero_benchmark / f"{task_idx}_{prompt.replace(' ', '_')}" / f"{episode_idx}_{done}.mp4"
    out_file.parent.mkdir(exist_ok=True, parents=True)

    save_video(
        real_obs_list=full_obs_list,
        save_path=out_file,
        fps=60,
        video_names=["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]
    )

    cur_env.close()
    return done


def run(libero_benchmark, port, out_dir, test_num, task_range=None, host="127.0.0.1", print_profile=False):
    '''
        task_range: [start, end) for splitting tasks
    '''
    if task_range is None:
        benchmark_dict = benchmark.get_benchmark_dict()
        benchmark_instance = benchmark_dict[libero_benchmark]()
        num_tasks = benchmark_instance.get_num_tasks()
        progress_bar = tqdm(range(num_tasks), total=num_tasks)
    else:
        assert len(task_range) == 2, f'task_range: [start, end) for splitting tasks, however, task_range: {task_range}'
        num_tasks = task_range[1] - task_range[0]
        progress_bar = tqdm(range(task_range[0], task_range[1]), total=num_tasks)

    print(f"#################### Use benchmark: {libero_benchmark}, num_tasks: {num_tasks} #############")
    print(f"#################### Connect policy server: {host}:{port} #############")
    model = WebsocketClientPolicy(host=host, port=port)

    video_save_root_dict = None

    episode_list = range(test_num)
    for task_idx in progress_bar:
        if video_save_root_dict is not None and task_idx in video_save_root_dict:
            video_save_list = os.listdir(os.path.join(out_dir, libero_benchmark, video_save_root_dict[task_idx]))
            video_states = [1 for file in video_save_list if file.split('_')[1].split('.')[0] == 'True']
            succ_num = float(len(video_states))
            total_num = len(video_save_list)
            episode_list = range(len(video_save_list), test_num)
        else:
            succ_num = 0.
            total_num = 0

        out_file = Path(out_dir) / f"{libero_benchmark}_{task_idx}.json"
        out_file.parent.mkdir(exist_ok=True, parents=True)
        for episode_idx in tqdm(episode_list, total=len(episode_list)):
            res_i = run_one(model, libero_benchmark, task_idx, out_dir, episode_idx, print_profile=print_profile)
            succ_num += res_i
            total_num = episode_idx + 1
            succ_rate = succ_num / total_num
            print(f"Success rate: {succ_rate}, success num: {succ_num}, total num: {total_num}")
            write_json({
                "succ_num": float(succ_num),
                "total_num": float(total_num),
                "succ_rate": float(succ_rate),
                "task_success": bool(succ_num > 0),
                "task_all_success": bool(total_num > 0 and succ_num == total_num),
                }, out_file
            )

        succ_rate = succ_num / total_num if total_num > 0 else 0.0
        task_success = bool(succ_num > 0)
        task_all_success = bool(total_num > 0 and succ_num == total_num)
        print(
            f"#################### Task {task_idx} done: "
            f"success={task_success}, all_success={task_all_success}, "
            f"success_rate={succ_rate}, success_num={succ_num}, total_num={total_num} #############"
        )
        write_json({
            "succ_num": float(succ_num),
            "total_num": float(total_num),
            "succ_rate": float(succ_rate),
            "task_success": bool(task_success),
            "task_all_success": bool(task_all_success),
            }, out_file
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero-benchmark",
        type=str,
        default="libero_10",
        choices=["libero_10", "libero_goal", "libero_spatial", "libero_object"],
        help="Benchmark name",
    )
    parser.add_argument(
        "--task-range",
        type=int,
        nargs="+",
        default=[0, 10],
        help="Task range [start, end) for splitting tasks",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=23908,
        help="WebSocket port",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="WebSocket server host",
    )
    parser.add_argument(
        "--test-num",
        type=int,
        default=50,
        help="Number of test episodes",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/libero",
        help="Output directory for results",
    )
    parser.add_argument(
        "--print-profile",
        action="store_true",
        help="Print server-side action latency profile returned by the policy server",
    )
    args = parser.parse_args()
    run(**vars(args))
    print("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    main()
