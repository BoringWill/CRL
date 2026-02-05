import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import pygame
from collections import deque
from slime_env import SlimeSelfPlayEnv
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

# --- 核心配置 ---
CONFIG = {
    "adaptive_model_path": "模型集_opponent/train_20260204-224621/evolution_v2.pth",
    "opponents_dir": "最强模型集",
    "device": torch.device("cpu"),
    "init_temp": 1.0,
    "min_temp": 1,
    "max_temp": 1,
    "inertia": 0.1,
    "ema_alpha": 0.01,
    "max_workers": 10,
}


# --- 1. 逻辑组件 ---
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


# --- 2. 核心对战逻辑 ---
def play_one_match(opp_name, current_state_snapshot, render=False):
    render_mode = "human" if render else None
    env = SlimeSelfPlayEnv(render_mode=render_mode)

    agent_adaptive = Agent()
    load_weights(agent_adaptive, CONFIG["adaptive_model_path"])

    agent_fixed = Agent()
    opp_path = os.path.join(CONFIG["opponents_dir"], opp_name)
    load_weights(agent_fixed, opp_path)

    diff_manager = SmartDifficultyManager(
        init_min=current_state_snapshot['min_v'],
        init_max=current_state_snapshot['max_v'],
        init_smooth=current_state_snapshot['smooth_confidence']
    )

    match_values = []
    font = None
    if render:
        try:
            font = pygame.font.SysFont("Arial", 20, bold=True)
        except:
            font = pygame.font.Font(None, 24)

    p1_dq, p2_dq = deque(maxlen=4), deque(maxlen=4)
    raw_obs_p1, _ = env.reset()
    raw_obs_p2 = env._get_obs(2)
    for _ in range(4): p1_dq.append(raw_obs_p1); p2_dq.append(raw_obs_p2)

    done = False
    while not done:
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None

        obs_p1 = np.concatenate(list(p1_dq))
        obs_p2 = np.concatenate(list(p2_dq))

        current_val = agent_adaptive.get_value(obs_p2)
        match_values.append(current_val)

        curr_temp, conf = diff_manager.update(current_val, env.p1_score, env.p2_score)

        a1 = agent_fixed.get_action(obs_p1, 0.01)
        a2 = agent_adaptive.get_action(obs_p2, curr_temp)

        n_p1, n_p2, _, term, trunc, _ = env.step((a1, a2))
        p1_dq.append(n_p1);
        p2_dq.append(n_p2)

        if render:
            env.render()
            try:
                screen = pygame.display.get_surface()
                if screen is not None and font is not None:
                    texts = [
                        f"Temp: {curr_temp:.2f}",
                        f"Value: {current_val:.3f}",
                        f"Opponent: {opp_name}"
                    ]
                    for i, text in enumerate(texts):
                        txt_surf = font.render(text, True, (255, 255, 0))
                        screen.blit(txt_surf, (10, 10 + i * 25))
                    pygame.display.flip()
            except:
                pass
            time.sleep(0.01)

        if term or trunc:
            done = True

    # 计算统计数据
    mean_v = np.mean(match_values) if match_values else 0.0
    var_v = np.var(match_values) if match_values else 0.0

    result = {
        "opponent": opp_name,
        "p1_score": env.p1_score,
        "p2_score": env.p2_score,
        "win": env.p2_score > env.p1_score,
        "mean_v": mean_v,
        "var_v": var_v,  # 记录方差
        "final_state": {
            "min_v": diff_manager.min_v,
            "max_v": diff_manager.max_v,
            "smooth_confidence": diff_manager.smooth_confidence
        }
    }
    env.close()
    return result


# --- 3. 控制器 ---
def run_fast_tournament():
    opp_files = [f for f in os.listdir(CONFIG["opponents_dir"]) if f.endswith(".pth")]
    opp_files.sort()

    print(f"🚀 开始并发赛模式 | 并行数: {CONFIG['max_workers']} | 对手总数: {len(opp_files)}")
    print("=" * 100)
    # 增加方差列
    print(f"{'STATUS':<8} | {'OPPONENT':<25} | {'SCORE':<12} | {'MEAN VALUE':<12} | {'VALUE VAR':<10}")
    print("-" * 100)

    all_results = []
    with Manager() as manager:
        shared_state = manager.dict({"min_v": -0.1, "max_v": 0.1, "smooth_confidence": 0.5})
        with ProcessPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
            futures = {executor.submit(play_one_match, name, dict(shared_state), False): name for name in opp_files}
            for future in as_completed(futures):
                res = future.result()
                if res:
                    shared_state.update(res["final_state"])
                    all_results.append(res)
                    status = "🏆 WIN" if res['win'] else "❌ LOSS"
                    score_str = f"{res['p1_score']} vs {res['p2_score']}"
                    print(
                        f"{status:<8} | {res['opponent'].ljust(25)} | {score_str:<12} | {res['mean_v']:>12.4f} | {res['var_v']:>10.4f}")

    print("\n" + "=" * 100)
    print("📊 统计汇总")
    print("-" * 100)
    wins = sum(1 for r in all_results if r['win'])
    total = len(all_results)
    avg_v_total = np.mean([r['mean_v'] for r in all_results]) if all_results else 0
    avg_var_total = np.mean([r['var_v'] for r in all_results]) if all_results else 0

    if total > 0:
        print(f"胜率: {wins / total:.2%} ({wins}/{total})")
        print(f"全局平均 Value: {avg_v_total:.4f}")
        print(f"全局平均 Variance: {avg_var_total:.4f}")
    print("=" * 100)


if __name__ == "__main__":
    mode = input("1. 快速模式\n2. 观战模式\n请选择: ")
    if mode == "1":
        run_fast_tournament()
    else:
        pygame.init()
        opp_files = [f for f in os.listdir(CONFIG["opponents_dir"]) if f.endswith(".pth")]
        opp_files.sort()
        current_state = {"min_v": -0.1, "max_v": 0.1, "smooth_confidence": 0.5}
        for f in opp_files:
            if f == os.path.basename(CONFIG["adaptive_model_path"]): continue
            res = play_one_match(f, current_state, render=True)
            if res is None: break
            current_state = res["final_state"]
            print(f"[{'🏆' if res['win'] else '❌'}] {f} | Mean: {res['mean_v']:.4f} | Var: {res['var_v']:.4f}")
        pygame.quit()