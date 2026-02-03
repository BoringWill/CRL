import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from slime_env import SlimeSelfPlayEnv
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager  # 导入共享管理器

# --- 核心配置 ---
CONFIG = {
    "adaptive_model_path": "最强模型集/1.pth",
    "opponents_dir": "最强模型集",
    "device": torch.device("cpu"),  # 多进程并发建议用 CPU，否则显存容易爆
    "init_temp": 1.0,
    "min_temp": 0.1,
    "max_temp": 10.0,
    "inertia": 0.5,
    "ema_alpha": 0.01,
    "max_workers": 10,  # 同时启动的并行进程数 (根据你的 CPU 核心数调整)
}


# --- 1. 逻辑组件 (保持不变) ---
class SmartDifficultyManager:
    def __init__(self, init_min=-0.1, init_max=0.1, init_smooth=0.5):
        self.min_v = init_min
        self.max_v = init_max
        self.smooth_confidence = init_smooth
        self.current_temp = CONFIG["init_temp"]

    def update(self, v_raw, p1_score, p2_score):
        self.min_v = min(self.min_v, v_raw)
        self.max_v = max(self.max_v, v_raw)
        range_v = self.max_v - self.min_v
        instant_conf = (v_raw - self.min_v) / range_v if range_v > 1e-5 else 0.5
        self.smooth_confidence = self.smooth_confidence * (1 - CONFIG["ema_alpha"]) + instant_conf * CONFIG["ema_alpha"]
        target_temp = 0.1 + (self.smooth_confidence * 9.9)
        score_diff = p1_score - p2_score
        if score_diff > 0:
            target_temp = max(CONFIG["min_temp"], target_temp - (score_diff * 1.5))
        diff = target_temp - self.current_temp
        move_speed = CONFIG["inertia"] * (0.5 if self.current_temp < 1.0 else 1.0)
        self.current_temp += diff * move_speed
        self.current_temp = np.clip(self.current_temp, CONFIG["min_temp"], CONFIG["max_temp"])
        return self.current_temp, self.smooth_confidence


class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(nn.Linear(52, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.actor = nn.Sequential(nn.Linear(52, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 4))

    def get_value(self, obs):
        with torch.no_grad():
            t_obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            return self.critic(t_obs).item()

    def get_action(self, obs, temp):
        with torch.no_grad():
            t_obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = self.actor(t_obs)
            probs = F.softmax(logits / temp, dim=-1)
            return torch.distributions.Categorical(probs).sample().item()


def load_weights(model, path):
    if not os.path.exists(path): return False
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt, strict=True)
    return True


# --- 2. 单场对战任务 ---
def play_one_match(opp_name, current_state_snapshot, render=False):
    # 初始化环境
    render_mode = "human" if render else None
    env = SlimeSelfPlayEnv(render_mode=render_mode)

    # 初始化模型
    agent_adaptive = Agent()
    load_weights(agent_adaptive, CONFIG["adaptive_model_path"])

    agent_fixed = Agent()
    opp_path = os.path.join(CONFIG["opponents_dir"], opp_name)
    load_weights(agent_fixed, opp_path)

    # 使用传入的快照初始化（继承值）
    diff_manager = SmartDifficultyManager(
        init_min=current_state_snapshot['min_v'],
        init_max=current_state_snapshot['max_v'],
        init_smooth=current_state_snapshot['smooth_confidence']
    )

    p1_dq, p2_dq = deque(maxlen=4), deque(maxlen=4)
    raw_obs_p1, _ = env.reset();
    raw_obs_p2 = env._get_obs(2)
    for _ in range(4): p1_dq.append(raw_obs_p1); p2_dq.append(raw_obs_p2)

    done = False
    while not done:
        curr_temp = diff_manager.current_temp
        obs_p1 = np.concatenate(list(p1_dq))
        obs_p2 = np.concatenate(list(p2_dq))

        a1 = agent_fixed.get_action(obs_p1, 0.01)
        a2 = agent_adaptive.get_action(obs_p2, curr_temp)

        n_p1, n_p2, _, term, trunc, _ = env.step((a1, a2))
        p1_dq.append(n_p1);
        p2_dq.append(n_p2)

        v_next = 0.0 if (term or trunc) else agent_adaptive.get_value(np.concatenate(list(p2_dq)))
        diff_manager.update(v_next, env.p1_score, env.p2_score)

        if term or trunc:
            done = True

    result = {
        "opponent": opp_name,
        "p1_score": env.p1_score,
        "p2_score": env.p2_score,
        "win": env.p2_score > env.p1_score,
        "final_state": {
            "min_v": diff_manager.min_v,
            "max_v": diff_manager.max_v,
            "smooth_confidence": diff_manager.smooth_confidence
        }
    }
    return result


# --- 3. 多线程控制器 ---
def run_fast_tournament():
    opp_files = [f for f in os.listdir(CONFIG["opponents_dir"]) if f.endswith(".pth")]
    opp_files.sort()

    print(f"🚀 开始并发赛模式 | 并行数: {CONFIG['max_workers']} | 对手总数: {len(opp_files)}")
    print("=" * 60)

    all_results = []

    # 建立进程间共享的状态字典
    with Manager() as manager:
        shared_state = manager.dict({"min_v": -0.1, "max_v": 0.1, "smooth_confidence": 0.5})

        with ProcessPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
            # 提交任务时，传入 shared_state 的当前拷贝
            futures = {executor.submit(play_one_match, name, dict(shared_state), False): name for name in opp_files}

            for future in as_completed(futures):
                res = future.result()
                # 关键：一个任务结束，立即更新共享字典，供后续还没开始的任务读取
                shared_state.update(res["final_state"])

                all_results.append(res)
                status = "🏆 WIN" if res['win'] else "❌ LOSS"
                print(f"[{status}] {res['opponent'].ljust(25)} | P1: {res['p1_score']} vs P2: {res['p2_score']}")

    # 打印最终战绩汇总
    print("\n" + "=" * 60)
    print("📊 最终战绩统计")
    print("-" * 60)
    wins = sum(1 for r in all_results if r['win'])
    total = len(all_results)
    print(f"总场次: {total} | 胜率: {wins / total:.2%} | 胜: {wins} / 负: {total - wins}")
    print("=" * 60)


if __name__ == "__main__":
    mode = input("1. 快速模式 (并发+无渲染)\n2. 观战模式 (单线程+有渲染)\n请选择: ")

    if mode == "1":
        run_fast_tournament()
    else:
        opp_files = [f for f in os.listdir(CONFIG["opponents_dir"]) if f.endswith(".pth")]
        opp_files.sort()
        # 观战模式也手动维护这个继承状态
        current_state = {"min_v": -0.1, "max_v": 0.1, "smooth_confidence": 0.5}
        for f in opp_files:
            res = play_one_match(f, current_state, render=True)
            current_state = res["final_state"]