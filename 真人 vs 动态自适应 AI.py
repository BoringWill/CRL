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

# --- 核心配置 ---
CONFIG = {
    "adaptive_model_path": "最强模型集/best_.pth",
    "opponents_dir": "最强模型集",
    "device": torch.device("cpu"),
    "init_temp": 0.1,
    "min_temp": 0.1,
    "max_temp": 10,
    "inertia": 0.05,
    "max_workers": 10,
    "games_per_opponent": 10,
}


# --- 1. 逻辑组件 ---
class SmartDifficultyManager:
    def __init__(self, init_v1_acc=0.0, init_v2_acc=0.0):
        self.v1_acc = init_v1_acc
        self.v2_acc = init_v2_acc
        self.current_temp = CONFIG["init_temp"]

    def update(self, v1, v2):
        self.v1_acc += v1
        self.v2_acc += v2
        instant_diff = self.v1_acc - self.v2_acc
        if instant_diff > 0:
            target_temp = max(CONFIG["min_temp"], CONFIG["init_temp"] - abs(instant_diff) * 0.01)
        else:
            target_temp = min(CONFIG["max_temp"], CONFIG["init_temp"] + abs(instant_diff) * 0.01)
        temp_step = (target_temp - self.current_temp) * CONFIG["inertia"]
        self.current_temp += temp_step
        self.current_temp = np.clip(self.current_temp, CONFIG["min_temp"], CONFIG["max_temp"])
        return self.current_temp, instant_diff


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
            safe_temp = max(temp, 1e-6)
            probs = F.softmax(logits / safe_temp, dim=-1)
            return torch.distributions.Categorical(probs).sample().item()


def load_weights(model, path):
    if not os.path.exists(path): return False
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt, strict=True)
    return True


# --- 2. 核心评估函数 (修复了分数跳变逻辑) ---
def evaluate_opponent(opp_name, render=False):
    render_mode = "human" if render else None
    env = SlimeSelfPlayEnv(render_mode=render_mode)

    agent_adaptive = Agent()
    load_weights(agent_adaptive, CONFIG["adaptive_model_path"])
    agent_fixed = Agent()
    load_weights(agent_fixed, os.path.join(CONFIG["opponents_dir"], opp_name))

    font = None
    if render:
        try:
            font = pygame.font.SysFont("Arial", 20, bold=True)
        except:
            font = pygame.font.Font(None, 24)

    total_p1, total_p2 = 0, 0
    match_v1_all, match_v2_all = [], []
    wins = 0

    for game_idx in range(CONFIG["games_per_opponent"]):
        diff_manager = SmartDifficultyManager()
        p1_dq, p2_dq = deque(maxlen=4), deque(maxlen=4)

        # 重置环境并获取初始观测
        raw_obs_p1, _ = env.reset()
        raw_obs_p2 = env._get_obs(2)
        for _ in range(4): p1_dq.append(raw_obs_p1); p2_dq.append(raw_obs_p2)

        done = False
        while not done:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: pygame.quit(); return None

            obs_p1, obs_p2 = np.concatenate(list(p1_dq)), np.concatenate(list(p2_dq))
            v1, v2 = agent_adaptive.get_value(obs_p1), agent_adaptive.get_value(obs_p2)
            match_v1_all.append(v1)
            match_v2_all.append(v2)

            curr_temp, _ = diff_manager.update(v1, v2)
            a1 = agent_fixed.get_action(obs_p1, curr_temp)
            a2 = agent_adaptive.get_action(obs_p2, curr_temp)

            # 执行动作
            n_p1, n_p2, _, term, trunc, info = env.step((a1, a2))
            p1_dq.append(n_p1)
            p2_dq.append(n_p2)

            if render:
                env.render()
                try:
                    screen = pygame.display.get_surface()
                    if screen:
                        # 此处显示总分 + 本局即时分数
                        txts = [
                            f"Game: {game_idx + 1}/{CONFIG['games_per_opponent']}",
                            f"Overall Score: {total_p1}:{total_p2}",
                            f"Current Round: {env.p1_score}:{env.p2_score}",
                            f"Opp: {opp_name}"
                        ]
                        for i, t in enumerate(txts):
                            screen.blit(font.render(t, True, (255, 255, 0)), (10, 10 + i * 25))
                        pygame.display.flip()
                except:
                    pass

            # --- 核心修复：一旦本局结束，立即锁定分数 ---
            if term or trunc:
                # 方案：直接从 info 或 env 抓取最后那一秒的分数，立刻跳出循环
                # 这样可以防止下一局开始后的 reset 逻辑干扰数据
                this_game_p1 = env.p1_score
                this_game_p2 = env.p2_score

                total_p1 += this_game_p1
                total_p2 += this_game_p2

                if this_game_p2 > this_game_p1:
                    wins += 1

                done = True  # 退出 while 循环，准备进入下一局 reset

    env.close()
    return {
        "opponent": opp_name,
        "p1_total": total_p1,
        "p2_total": total_p2,
        "v1_mean": np.mean(match_v1_all) if match_v1_all else 0,
        "v2_mean": np.mean(match_v2_all) if match_v2_all else 0,
        "win_rate": (wins / CONFIG["games_per_opponent"]) * 100
    }


# --- 3. 控制器 ---
def run_main(render=False):
    opp_files = [f for f in os.listdir(CONFIG["opponents_dir"]) if f.endswith(".pth")]
    opp_files.sort()

    print(f"🚀 启动评估 | 总模型: {len(opp_files)} | 每人对战: {CONFIG['games_per_opponent']} 场")
    print("-" * 115)

    final_results = []
    if not render:
        # 快速模式
        with ProcessPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
            futures = [executor.submit(evaluate_opponent, name, False) for name in opp_files]
            for f in as_completed(futures):
                res = f.result()
                if res:
                    final_results.append(res)
                    status = "🏆 WIN" if res['p2_total'] > res['p1_total'] else "❌ LOSS"
                    print(
                        f"{status:<8} | {res['opponent'].ljust(25)} | {res['p1_total']} vs {res['p2_total']} | V1:{res['v1_mean']:>8.2f} || V2:{res['v2_mean']:>8.2f}")
    else:
        # 观战模式
        for name in opp_files:
            res = evaluate_opponent(name, True)
            if res is None: break
            final_results.append(res)
            status = "🏆 WIN" if res['p2_total'] > res['p1_total'] else "❌ LOSS"
            print(
                f"{status:<8} | {res['opponent'].ljust(25)} | {res['p1_total']} vs {res['p2_total']} | V1:{res['v1_mean']:>8.2f} V2:{res['v2_mean']:>8.2f}")

    print("-" * 115)
    print("📊 最终战绩排名 (按 P2 总分排序)")
    sorted_results = sorted(final_results, key=lambda x: x['p2_total'], reverse=True)
    for r in sorted_results:
        indicator = "✅" if r['p2_total'] > r['p1_total'] else "🔻"
        print(
            f"{indicator} {r['opponent'].ljust(25)} | 总比分: {r['p1_total']}:{r['p2_total']} | 胜率: {r['win_rate']:.1f}%")


if __name__ == "__main__":
    mode = input("1. 快速模式\n2. 观战模式\n请选择: ")
    if mode == "2":
        pygame.init()
        run_main(render=True)
        pygame.quit()
    else:
        run_main(render=False)