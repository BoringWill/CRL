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
    "adaptive_model_path": "最强模型集/best.pth",
    "opponents_dir": "最强模型集",
    "device": torch.device("cpu"),
    # 基础温度：当比分持平时，保持在这个温度（1.0代表有一定的随机性，0.1代表最强）
    "init_temp": 1.0,
    "min_temp": 0.1,
    "max_temp": 10.0,
    # 惯性系数：数值越小，温度变化越平滑，防止温度剧烈跳变
    "inertia": 0.1,
    "max_workers": 10,
    # 分差敏感度：每落后1分，温度降低多少（变得更强）
    "score_sensitivity": 2.0
}


# --- 1. 逻辑组件 (修改部分：纯分差驱动) ---
class ScoreOnlyDifficultyManager:
    def __init__(self):
        self.current_temp = CONFIG["init_temp"]

    def update(self, p1_score, p2_score):
        """
        仅根据分差调整温度：
        - P1 (固定对手) 分数高 -> P2 (自适应) 落后 -> 温度降低 (变强)
        - P2 分数高 -> 领先 -> 温度升高 (变弱/娱乐)
        """
        score_diff = p1_score - p2_score  # 正数代表P2落后，负数代表P2领先

        # 目标温度计算公式：基准温度 - (分差 * 敏感度)
        # 例如：基准1.0，落后2分(diff=2) -> target = 1.0 - 4.0 = -3.0 -> clip到 0.1
        # 例如：基准1.0，领先2分(diff=-2) -> target = 1.0 + 4.0 = 5.0
        target_temp = CONFIG["init_temp"] - (score_diff * CONFIG["score_sensitivity"])

        # 限制范围
        target_temp = np.clip(target_temp, CONFIG["min_temp"], CONFIG["max_temp"])

        # 惯性平滑移动
        diff = target_temp - self.current_temp
        # 如果是降温（变强），反应快一点；如果是升温（变弱），反应慢一点
        move_speed = CONFIG["inertia"] * (1.5 if diff < 0 else 0.5)

        self.current_temp += diff * move_speed
        self.current_temp = np.clip(self.current_temp, CONFIG["min_temp"], CONFIG["max_temp"])

        return self.current_temp, score_diff


class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        # Critic 仍然保留结构以便加载权重，但不参与逻辑计算
        self.critic = nn.Sequential(nn.Linear(52, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.actor = nn.Sequential(nn.Linear(52, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 4))

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


# --- 2. 核心对战逻辑 (修改部分：移除Value传参) ---
def play_one_match(opp_name, render=False):
    render_mode = "human" if render else None
    env = SlimeSelfPlayEnv(render_mode=render_mode)

    agent_adaptive = Agent()
    load_weights(agent_adaptive, CONFIG["adaptive_model_path"])

    agent_fixed = Agent()
    opp_path = os.path.join(CONFIG["opponents_dir"], opp_name)
    load_weights(agent_fixed, opp_path)

    # 使用新的分差管理器
    diff_manager = ScoreOnlyDifficultyManager()

    # 准备字体渲染
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

        # --- 修改：不再计算Value，只根据比分更新温度 ---
        curr_temp, score_diff = diff_manager.update(env.p1_score, env.p2_score)

        # 采样动作
        a1 = agent_fixed.get_action(obs_p1, 0.01)  # 对手保持最强状态
        a2 = agent_adaptive.get_action(obs_p2, curr_temp)

        n_p1, n_p2, _, term, trunc, _ = env.step((a1, a2))
        p1_dq.append(n_p1);
        p2_dq.append(n_p2)

        if render:
            env.render()
            try:
                screen = pygame.display.get_surface()
                if screen is not None and font is not None:
                    # 颜色提示：温度低（强）显示红色，温度高（弱）显示绿色
                    temp_color = (255, 50, 50) if curr_temp < 0.5 else (50, 255, 50)
                    texts = [
                        (f"Temp: {curr_temp:.2f}", temp_color),
                        (f"Score Diff: {score_diff}", (255, 255, 255)),
                        (f"Opponent: {opp_name}", (255, 255, 0))
                    ]
                    for i, (text, color) in enumerate(texts):
                        txt_surf = font.render(text, True, color)
                        screen.blit(txt_surf, (10, 10 + i * 25))
                    pygame.display.flip()
            except:
                pass
            time.sleep(0.015)

        if term or trunc:
            done = True

    result = {
        "opponent": opp_name,
        "p1_score": env.p1_score,
        "p2_score": env.p2_score,
        "win": env.p2_score > env.p1_score
    }
    env.close()
    return result


# --- 3. 控制器 ---
def run_fast_tournament():
    opp_files = [f for f in os.listdir(CONFIG["opponents_dir"]) if f.endswith(".pth")]
    opp_files.sort()

    print(f"🧪 开始消融实验 (仅分数驱动) | 并行数: {CONFIG['max_workers']} | 对手总数: {len(opp_files)}")
    print("=" * 60)

    all_results = []
    # 移除了 Manager 和 shared_state，因为不再需要统计全局Value分布
    with ProcessPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        futures = {executor.submit(play_one_match, name, False): name for name in opp_files}
        for future in as_completed(futures):
            res = future.result()
            if res:
                all_results.append(res)
                status = "🏆 WIN" if res['win'] else "❌ LOSS"
                print(f"[{status}] {res['opponent'].ljust(25)} | P1: {res['p1_score']} vs P2: {res['p2_score']}")

    print("\n" + "=" * 60)
    print("📊 消融实验统计")
    print("-" * 60)
    wins = sum(1 for r in all_results if r['win'])
    total = len(all_results)
    if total > 0:
        print(f"总场次: {total} | 胜率: {wins / total:.2%} | 胜: {wins} / 负: {total - wins}")
    print("=" * 60)


if __name__ == "__main__":
    print("--- 🔬 消融实验模式：仅保留分数影响温度 ---")
    mode = input("1. 快速模式 (并发+无渲染)\n2. 观战模式 (单线程+有渲染)\n请选择: ")

    if mode == "1":
        run_fast_tournament()
    else:
        pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        opp_files = [f for f in os.listdir(CONFIG["opponents_dir"]) if f.endswith(".pth")]
        opp_files.sort()

        print("\n📺 进入观战模式...")
        for f in opp_files:
            if f == os.path.basename(CONFIG["adaptive_model_path"]): continue

            print(f"🎮 正在对战: {f}")
            res = play_one_match(f, render=True)

            if res is None: break

            status = "🏆 WIN" if res['win'] else "❌ LOSS"
            print(f"[{status}] 战局结束 | P1: {res['p1_score']} vs P2: {res['p2_score']}")
            print("-" * 40)

        pygame.quit()