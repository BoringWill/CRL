import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import os
from collections import deque
from slime_env import SlimeSelfPlayEnv, FrameStack

# --- 配置 ---
NEW_MODEL_PATH = "模型集_opponent/train_20260125-013011/fixed_opponent_current.pth"
HISTORY_FOLDER = "模型集_opponent/train_20260125-013011"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 测试参数
NUM_ENVS = 4
GAMES_PER_OPPONENT = 10


# --- 模型结构 ---
class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(48, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.actor = nn.Sequential(
            nn.Linear(48, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 4)
        )

    def get_actions(self, obs_batch, device):
        with torch.no_grad():
            t_obs = torch.FloatTensor(obs_batch).to(device)
            logits = self.actor(t_obs)
            return torch.argmax(logits, dim=1).cpu().numpy()


def make_env():
    return lambda: FrameStack(SlimeSelfPlayEnv(render_mode=None), n_frames=4)


def run_vector_battle(envs, agent_new, agent_hist, num_total_games):
    new_model_wins = 0
    games_finished = 0

    obs_p1, infos = envs.reset()
    p2_deques = [deque(maxlen=4) for _ in range(NUM_ENVS)]

    # 修正初始化拿取逻辑
    p2_raw_initial = infos.get("p2_raw_obs")
    for i in range(NUM_ENVS):
        init_p2 = p2_raw_initial[i] if p2_raw_initial is not None else np.zeros(12)
        for _ in range(4): p2_deques[i].append(init_p2)
    obs_p2 = np.array([np.concatenate(list(d), axis=0) for d in p2_deques])

    side_swapped = np.random.rand(NUM_ENVS) > 0.5

    while games_finished < num_total_games:
        t_obs_agent_new = np.zeros((NUM_ENVS, 48), dtype=np.float32)
        t_obs_agent_hist = np.zeros((NUM_ENVS, 48), dtype=np.float32)

        for i in range(NUM_ENVS):
            if not side_swapped[i]:
                t_obs_agent_new[i] = obs_p1[i]
                t_obs_agent_hist[i] = obs_p2[i]
            else:
                t_obs_agent_new[i] = obs_p2[i]
                t_obs_agent_hist[i] = obs_p1[i]

        actions_new = agent_new.get_actions(t_obs_agent_new, DEVICE)
        actions_hist = agent_hist.get_actions(t_obs_agent_hist, DEVICE)

        env_actions = np.zeros((NUM_ENVS, 2), dtype=np.int32)
        for i in range(NUM_ENVS):
            if not side_swapped[i]:
                env_actions[i] = [actions_new[i], actions_hist[i]]
            else:
                env_actions[i] = [actions_hist[i], actions_new[i]]

        obs_p1, rewards, terms, truncs, infos = envs.step(env_actions)

        # 核心修复：获取当前所有环境的 p2_raw_obs
        p2_raw_batch = infos.get("p2_raw_obs")

        for i in range(NUM_ENVS):
            if terms[i] or truncs[i]:
                games_finished += 1

                p1_won = infos["p1_score"][i] > infos["p2_score"][i]
                p2_won = infos["p2_score"][i] > infos["p1_score"][i]

                if not side_swapped[i]:
                    if p1_won: new_model_wins += 1
                else:
                    if p2_won: new_model_wins += 1

                side_swapped[i] = np.random.rand() > 0.5

                p2_deques[i].clear()
                # 结束帧处理
                res_p2 = p2_raw_batch[i] if p2_raw_batch is not None else np.zeros(12)
                for _ in range(4): p2_deques[i].append(res_p2)
                if games_finished >= num_total_games: break
            else:
                # 普通帧处理
                if p2_raw_batch is not None:
                    p2_deques[i].append(p2_raw_batch[i])
                else:
                    # 如果该帧没有返回 p2_raw_obs，为了维持 4 帧缓存，重用上一帧
                    p2_deques[i].append(p2_deques[i][-1])

        obs_p2 = np.array([np.concatenate(list(d), axis=0) for d in p2_deques])

    return new_model_wins


def main():
    print(f"正在启动 {NUM_ENVS} 个并行环境...")
    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(NUM_ENVS)])

    agent_new = Agent().to(DEVICE)
    if not os.path.exists(NEW_MODEL_PATH):
        print(f"找不到最新模型: {NEW_MODEL_PATH}")
        return
    agent_new.load_state_dict(torch.load(NEW_MODEL_PATH, map_location=DEVICE))
    agent_new.eval()

    if not os.path.exists(HISTORY_FOLDER):
        print(f"找不到文件夹: {HISTORY_FOLDER}")
        return

    history_files = [f for f in os.listdir(HISTORY_FOLDER) if f.endswith('.pth')]
    history_files.sort()

    print("=" * 60)
    print(f"随机角色对抗测试 | 并行数: {NUM_ENVS} | 目标总局数: {GAMES_PER_OPPONENT}")
    print("=" * 60)

    results = []
    for hist_file in history_files:
        hist_path = os.path.join(HISTORY_FOLDER, hist_file)
        agent_hist = Agent().to(DEVICE)
        try:
            agent_hist.load_state_dict(torch.load(hist_path, map_location=DEVICE))
            agent_hist.eval()
        except:
            continue

        print(f"正在对阵: {hist_file.ljust(25)}", end=" | ", flush=True)
        wins = run_vector_battle(envs, agent_new, agent_hist, GAMES_PER_OPPONENT)
        win_rate = (wins / GAMES_PER_OPPONENT) * 100
        results.append((hist_file, win_rate))
        print(f"模型总胜率: {win_rate:>6.2f}%")

    print("\n" + "=" * 60)
    print("历史版本挑战总结 (随机位置):")
    for name, rate in results:
        status = "✅ 强于该版本" if rate > 50 else "❌ 弱于该版本"
        print(f"- {name.ljust(30)}: {rate:>6.2f}% {status}")
    print("=" * 60)
    envs.close()


if __name__ == "__main__":
    main()