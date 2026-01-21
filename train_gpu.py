import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from slime_env import SlimeSelfPlayEnv, FrameStack
from torch.utils.tensorboard import SummaryWriter
from collections import deque

# ==========================================================
# 1. 训练超参数配置 (Hyperparameters)
# ==========================================================
config = {
    # 架构与设备
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "save_path": "slime_ppo_gpu.pth",
    "log_dir": "runs/slime_vector_ppo",

    # 强化学习核心参数
    "total_timesteps": 5000000,
    "num_envs": 16,  # 并行环境数量
    "num_steps": 2048,  # 每个环境每次更新收集的步数
    "update_epochs": 10,  # 每次收集完数据后迭代更新的次数
    "batch_size": 2048,  # 小批次更新大小

    # PPO 特定系数
    "lr": 2e-4,  # 学习率
    "gamma": 0.99,  # 折扣因子
    "gae_lambda": 0.95,  # GAE 系数
    "clip_coef": 0.2,  # PPO 截断范围
    "ent_coef": 0.1,  # 熵系数 (鼓励探索)
    "vf_coef": 0.5,  # 价值损失系数
    "max_grad_norm": 0.5,  # 梯度裁剪阈值
}


# ==========================================================
# 2. 神经网络模型
# ==========================================================
class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        # 输入维度: 48 (12维基础特征 * 4帧堆叠)
        self.critic = nn.Sequential(
            nn.Linear(48, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.actor = nn.Sequential(
            nn.Linear(48, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 4)
        )

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


# ==========================================================
# 3. 训练主逻辑
# ==========================================================
def make_env():
    return lambda: FrameStack(SlimeSelfPlayEnv(render_mode=None), n_frames=4)


def train():
    # 环境与资源初始化
    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(config["num_envs"])])
    agent = Agent().to(config["device"])
    optimizer = optim.Adam(agent.parameters(), lr=config["lr"], eps=1e-5)
    writer = SummaryWriter(config["log_dir"])

    # 显存预分配 Buffer
    obs_buf = torch.zeros((config["num_steps"], config["num_envs"] * 2, 48)).to(config["device"])
    act_buf = torch.zeros((config["num_steps"], config["num_envs"] * 2)).to(config["device"])
    logp_buf = torch.zeros((config["num_steps"], config["num_envs"] * 2)).to(config["device"])
    rew_buf = torch.zeros((config["num_steps"], config["num_envs"] * 2)).to(config["device"])
    done_buf = torch.zeros((config["num_steps"], config["num_envs"] * 2)).to(config["device"])
    val_buf = torch.zeros((config["num_steps"], config["num_envs"] * 2)).to(config["device"])

    # 初始状态
    obs_p1, _ = envs.reset()
    p2_deques = [deque(maxlen=4) for _ in range(config["num_envs"])]

    # 初始化 P2 帧堆叠
    temp_env = SlimeSelfPlayEnv()
    temp_env.reset()
    init_p2_raw = temp_env._get_obs(2)
    for d in p2_deques:
        [d.append(init_p2_raw) for _ in range(4)]
    obs_p2 = np.array([np.concatenate(list(d), axis=0) for d in p2_deques])

    global_step = 0
    ep_reward_history = deque(maxlen=100)

    print(f">>> 训练启动 | 设备: {config['device']} | 环境数: {config['num_envs']}")

    while global_step < config["total_timesteps"]:
        # --- A. 数据收集阶段 (Rollout) ---
        agent.eval()
        for step in range(config["num_steps"]):
            global_step += config["num_envs"]

            # 合并 P1 和 P2 观测进行批处理推理
            t_obs = torch.from_numpy(np.concatenate([obs_p1, obs_p2])).float().to(config["device"])
            with torch.no_grad():
                actions, logprobs, _, values = agent.get_action_and_value(t_obs)

            # 拆分动作执行
            p1_acts = actions[:config["num_envs"]].cpu().numpy()
            p2_acts = actions[config["num_envs"]:].cpu().numpy()
            n_obs_p1, reward, term, trunc, infos = envs.step(np.stack([p1_acts, p2_acts], axis=1))

            # 更新 P2 观测队列
            for i in range(config["num_envs"]):
                p2_deques[i].append(infos["p2_raw_obs"][i])
            n_obs_p2 = np.array([np.concatenate(list(d), axis=0) for d in p2_deques])

            # 存入 Buffer
            obs_buf[step], act_buf[step], logp_buf[step], val_buf[step] = t_obs, actions, logprobs, values.flatten()
            rew_buf[step, :config["num_envs"]] = torch.from_numpy(reward).to(config["device"])
            rew_buf[step, config["num_envs"]:] = torch.from_numpy(-reward).to(config["device"])  # 对等奖励
            done_buf[step] = torch.from_numpy((term | trunc).astype(np.float32)).repeat(2).to(config["device"])

            obs_p1, obs_p2 = n_obs_p1, n_obs_p2

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "scores" in info:
                        ep_reward_history.append(info["scores"][0])

        # --- B. 优势计算阶段 (GAE) ---
        with torch.no_grad():
            next_obs = torch.from_numpy(np.concatenate([obs_p1, obs_p2])).float().to(config["device"])
            _, _, _, next_val = agent.get_action_and_value(next_obs)
            adv = torch.zeros_like(rew_buf).to(config["device"])
            lastgae = 0
            for t in reversed(range(config["num_steps"])):
                nt = 1.0 - done_buf[t]
                nv = next_val.flatten() if t == config["num_steps"] - 1 else val_buf[t + 1]
                delta = rew_buf[t] + config["gamma"] * nv * nt - val_buf[t]
                adv[t] = lastgae = delta + config["gamma"] * config["gae_lambda"] * nt * lastgae
            ret = adv + val_buf

        # --- C. 参数更新阶段 (PPO Update) ---
        agent.train()
        b_obs, b_logp, b_act = obs_buf.reshape(-1, 48), logp_buf.reshape(-1), act_buf.reshape(-1)
        b_adv, b_ret = adv.reshape(-1), ret.reshape(-1)

        indices = np.arange(config["num_steps"] * config["num_envs"] * 2)
        pg_losses, v_losses, entropies = [], [], []

        for _ in range(config["update_epochs"]):
            np.random.shuffle(indices)
            for s in range(0, len(indices), config["batch_size"]):
                mb = indices[s: s + config["batch_size"]]

                _, newlogp, ent, newv = agent.get_action_and_value(b_obs[mb], b_act[mb])

                # 策略损失 (Clipped Objective)
                ratio = (newlogp - b_logp[mb]).exp()
                m_adv = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std() + 1e-8)
                pg_loss1 = -m_adv * ratio
                pg_loss2 = -m_adv * torch.clamp(ratio, 1 - config["clip_coef"], 1 + config["clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # 价值损失 (MSE)
                v_loss = 0.5 * ((newv.flatten() - b_ret[mb]) ** 2).mean()

                # 总损失
                entropy = ent.mean()
                loss = pg_loss - config["ent_coef"] * entropy + v_loss * config["vf_coef"]

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config["max_grad_norm"])
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy.item())

        # --- D. 记录与保存 ---
        avg_reward = np.mean(ep_reward_history) if ep_reward_history else 0
        writer.add_scalar("charts/avg_reward", avg_reward, global_step)
        writer.add_scalar("losses/policy_loss", np.mean(pg_losses), global_step)
        writer.add_scalar("losses/value_loss", np.mean(v_losses), global_step)
        writer.add_scalar("losses/entropy", np.mean(entropies), global_step)

        print(
            f"步数: {global_step:7d} | 平均奖励: {avg_reward:6.2f} | 进度: {100 * global_step / config['total_timesteps']:3.1f}%")
        torch.save(agent.state_dict(), config["save_path"])

    envs.close()
    writer.close()


if __name__ == "__main__":
    train()