import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import os
from collections import deque
from slime_env import SlimeSelfPlayEnv, FrameStack

# --- 配置 ---
NEW_MODEL_PATH = "最强模型集/best_.pth"
HISTORY_FOLDER = "测试"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 测试参数
NUM_ENVS = 32
GAMES_PER_OPPONENT = 32  # 建议稍微多打几局，结果更准 建议多局数少


# --- 模型结构 ---
class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        # 维度更新：13 (特征数) * 4 (堆叠帧) = 52
        self.critic = nn.Sequential(
            nn.Linear(52, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.actor = nn.Sequential(
            nn.Linear(52, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 4)
        )

    def get_actions(self, obs_batch, device):
        with torch.no_grad():
            t_obs = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
            logits = self.actor(t_obs)
            # --- 修改部分：改用 Categorical 采样 ---
            probs = torch.distributions.Categorical(logits=logits)
            actions = probs.sample()
            return actions.cpu().numpy()


def make_env():
    # 显式关闭渲染提高速度
    return lambda: FrameStack(SlimeSelfPlayEnv(render_mode=None), n_frames=4)


def run_vector_battle(envs, agent_new, agent_hist, num_total_games):
    new_model_wins = 0
    games_finished = 0
    # --- 新增：得分统计 ---
    total_score_new = 0
    total_score_hist = 0

    obs_p1, infos = envs.reset()
    p2_deques = [deque(maxlen=4) for _ in range(NUM_ENVS)]

    p2_raw_initial = infos.get("p2_raw_obs")
    for i in range(NUM_ENVS):
        init_p2 = p2_raw_initial[i] if p2_raw_initial is not None else np.zeros(13)
        for _ in range(4): p2_deques[i].append(init_p2)

    side_swapped = np.random.rand(NUM_ENVS) > 0.5

    while games_finished < num_total_games:
        obs_p2 = np.array([np.concatenate(list(d)) for d in p2_deques])

        t_obs_new = np.where(side_swapped[:, None], obs_p2, obs_p1)
        t_obs_hist = np.where(side_swapped[:, None], obs_p1, obs_p2)

        act_new = agent_new.get_actions(t_obs_new, DEVICE)
        act_hist = agent_hist.get_actions(t_obs_hist, DEVICE)

        env_actions = np.zeros((NUM_ENVS, 2), dtype=np.int32)
        for i in range(NUM_ENVS):
            if not side_swapped[i]:
                env_actions[i] = [act_new[i], act_hist[i]]
            else:
                env_actions[i] = [act_hist[i], act_new[i]]

        obs_p1, _, terms, truncs, infos = envs.step(env_actions)
        p2_raw_batch = infos.get("p2_raw_obs")

        for i in range(NUM_ENVS):
            if terms[i] or truncs[i]:
                games_finished += 1
                s1 = infos["p1_score"][i]
                s2 = infos["p2_score"][i]

                # --- 修改：根据 side_swapped 统计得分和胜场 ---
                if not side_swapped[i]:
                    total_score_new += s1
                    total_score_hist += s2
                    if s1 > s2: new_model_wins += 1
                else:
                    total_score_new += s2
                    total_score_hist += s1
                    if s2 > s1: new_model_wins += 1

                side_swapped[i] = np.random.rand() > 0.5
                p2_deques[i].clear()
                res_p2 = p2_raw_batch[i] if p2_raw_batch is not None else np.zeros(13)
                for _ in range(4): p2_deques[i].append(res_p2)

                if games_finished >= num_total_games: break
            else:
                if p2_raw_batch is not None:
                    p2_deques[i].append(p2_raw_batch[i])

    # --- 修改：返回包含比分的结果 ---
    return new_model_wins, total_score_new, total_score_hist


def safe_load(agent, path):
    """通用的安全加载函数"""
    if not os.path.exists(path):
        return False, "路径不存在"
    try:
        checkpoint = torch.load(path, map_location=DEVICE)
        # 提取 state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            sd = checkpoint["model_state_dict"]
        else:
            sd = checkpoint

        # 使用 strict=False 忽略不匹配的层（如 critic）
        msg = agent.load_state_dict(sd, strict=False)
        return True, msg
    except Exception as e:
        return False, str(e)


def main():
    print(f"正在初始化 {NUM_ENVS} 个并行对战环境...")
    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(NUM_ENVS)])

    agent_new = Agent().to(DEVICE)
    success, info = safe_load(agent_new, NEW_MODEL_PATH)
    if not success:
        print(f"❌ 无法加载新模型: {info}")
        return
    print(f"✅ 新模型已准备就绪: {os.path.basename(NEW_MODEL_PATH)}")

    if not os.path.exists(HISTORY_FOLDER):
        print(f"❌ 找不到文件夹: {HISTORY_FOLDER}")
        return

    history_files = [f for f in os.listdir(HISTORY_FOLDER) if f.lower().endswith('.pth')]
    history_files.sort()

    print("=" * 85)
    print(f"开始历史挑战赛 | 总选手: {len(history_files)} | 每场局数: {GAMES_PER_OPPONENT}")
    print("=" * 85)

    results = []
    for hist_file in history_files:
        hist_path = os.path.join(HISTORY_FOLDER, hist_file)
        agent_hist = Agent().to(DEVICE)

        success, info = safe_load(agent_hist, hist_path)
        if not success:
            print(f"⚠️ 跳过 {hist_file.ljust(25)} | 错误原因: {info}")
            continue

        print(f"正在对阵: {hist_file.ljust(25)}", end=" | ", flush=True)
        agent_hist.eval()
        agent_new.eval()

        # --- 修改：接收并打印比分 ---
        wins, score_new, score_hist = run_vector_battle(envs, agent_new, agent_hist, GAMES_PER_OPPONENT)
        win_rate = (wins / GAMES_PER_OPPONENT) * 100
        score_str = f"{int(score_new)}:{int(score_hist)}"
        results.append((hist_file, win_rate, score_str))
        print(f"胜率: {win_rate:>6.2f}% | 比分: {score_str}")

    # --- 修改：最终汇总打印 ---
    print("\n" + "=" * 85)
    print(f"{'历史版本文件名':<35} | {'胜率':<8} | {'总比分':<12} | {'结论'}")
    print("-" * 85)
    for name, rate, s_ratio in results:
        status = "🟢 胜出" if rate > 50 else "🔴 落败"
        print(f"{name:<35} | {rate:>7.1f}% | {s_ratio:<12} | {status}")
    print("=" * 85)

    envs.close()


if __name__ == "__main__":
    main()