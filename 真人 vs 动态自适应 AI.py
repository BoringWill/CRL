import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pygame
from collections import deque
from slime_env import SlimeSelfPlayEnv, FrameStack

# --- 配置 ---
CONFIG = {
    "model_path": "模型集_opponent/train_20260125-013011/fixed_opponent_current.pth",  # 替换为你最好的模型
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # 动态难度参数
    "init_temp": 1.0,  # 初始温度
    "min_temp": 0.05,  # 最低温度 (接近0，代表最强/最认真)
    "max_temp": 5.0,  # 最高温度 (代表非常随机/在乱玩)
    "temp_step": 0.05,  # 每次调整的幅度
    "gamma": 0.99,  # 折扣因子 (虽然你公式是 r+V'-V，但通常V包含gamma，这里保留选项)
}


# --- 1. 模型结构 (必须包含 Actor 和 Critic) ---
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

    def get_value(self, obs, device):
        """获取状态价值 V(s)"""
        with torch.no_grad():
            t_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
            if t_obs.dim() == 1: t_obs = t_obs.unsqueeze(0)
            return self.critic(t_obs).item()

    def get_action_with_temp(self, obs, temp, device):
        """根据温度采样动作"""
        with torch.no_grad():
            t_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
            if t_obs.dim() == 1: t_obs = t_obs.unsqueeze(0)

            logits = self.actor(t_obs)

            # --- 关键：应用温度系数 ---
            # 温度越低 -> 分布越尖锐 -> 越接近 argmax (认真)
            # 温度越高 -> 分布越平坦 -> 越接近均匀分布 (随机/放水)
            if temp < 1e-3:  # 防止除以0
                action = torch.argmax(logits, dim=1)
            else:
                # Logits 除以温度
                probs = F.softmax(logits / temp, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()

            return action.cpu().numpy()[0]


def load_weights(model, path, device):
    if not os.path.exists(path):
        print(f"❌ 模型文件不存在: {path}")
        return False
    try:
        ckpt = torch.load(path, map_location=device)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
        print(f"✅ 成功加载模型: {path}")
        return True
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False


# --- 2. 动态调整主循环 ---
def run_adaptive_game():
    pygame.init()
    pygame.font.init()
    # 设置大一点的字体以便观察数据
    font = pygame.font.SysFont('Arial', 24, bold=True)

    # 初始化环境
    raw_env = SlimeSelfPlayEnv(render_mode="human")
    env = FrameStack(raw_env, n_frames=4)

    # 加载 AI
    ai_agent = Agent().to(CONFIG["device"])
    if not load_weights(ai_agent, CONFIG["model_path"], CONFIG["device"]):
        return

    ai_agent.eval()

    # 初始化变量
    current_temp = CONFIG["init_temp"]
    running = True
    clock = pygame.time.Clock()

    # 队列初始化
    p1_dq = deque([np.zeros(12) for _ in range(4)], maxlen=4)  # 真人
    p2_dq = deque([np.zeros(12) for _ in range(4)], maxlen=4)  # AI

    # 初始重置
    obs, _ = env.reset()
    # 这里的obs是FrameStack后的，我们需要手动维护队列来模拟 step
    # 为了简单起见，我们直接利用 env 内部状态或手动步进
    # 这里我们采用标准流程，初始时 obs 已经堆叠好了

    # 分解初始 obs (Gym vector env 可能会有不同，这里假设 standard Box)
    # SlimeSelfPlayEnv 的 reset 返回的是 (obs1, obs2) 还是单边？
    # 根据之前的代码，FrameStack 通常包装后返回单个 obs。
    # 这里为了确保逻辑正确，我们手动维护 obs 队列（就像你之前的脚本一样）

    raw_obs_p1 = raw_env._get_obs(1)
    raw_obs_p2 = raw_env._get_obs(2)
    for _ in range(4):
        p1_dq.append(raw_obs_p1)
        p2_dq.append(raw_obs_p2)

    # 记录上一帧的 AI 价值 V(s)
    last_val_p2 = 0.0
    td_error = 0.0

    print("\n>>> 🎮 动态难度 AI 已就绪！")
    print(">>> 观察左上角数据：温度(Temp)越高代表AI越傻，越低代表AI越强。")

    while running:
        # 1. 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 2. 获取真人动作
        keys = pygame.key.get_pressed()
        action_p1 = 0
        if keys[pygame.K_a]: action_p1 = 1  # 左
        if keys[pygame.K_d]: action_p1 = 2  # 右
        if keys[pygame.K_w]: action_p1 = 3  # 跳

        # 3. 准备 AI 观测数据 (s)
        obs_p2_stack = np.concatenate(list(p2_dq))

        # 4. 计算当前状态价值 V(s)
        current_val_p2 = ai_agent.get_value(obs_p2_stack, CONFIG["device"])

        # 5. AI 根据当前温度行动
        action_p2 = ai_agent.get_action_with_temp(obs_p2_stack, current_temp, CONFIG["device"])

        # 6. 环境步进
        # step 返回: obs, reward, terminated, truncated, info
        # 注意: SlimeSelfPlayEnv 返回的是两个智能体的 observation
        # 但我们手动维护 deque，所以只需要 reward 和 info
        obs_pair, rewards, term, trunc, info = env.step((action_p1, action_p2))

        # 获取 AI 的奖励 (P2)
        # rewards 通常是 (rew1, rew2) 或者根据环境定义
        # 假设 step 返回的是 (obs1, obs2), (rew1, rew2)...
        # 如果 env wrapper 改变了返回格式，这里需要适配。
        # 按照 SlimeSelfPlayEnv 原生逻辑：
        reward_p2 = rewards if isinstance(rewards, (int, float)) else rewards[1]
        # 如果是 self-play wrapper，通常 reward 是针对 P1 的，P2 = -reward
        # 让我们假设 reward 是针对 P1 的：
        if isinstance(rewards, (float, int)):
            # 如果返回单值，通常是 P1 的 reward
            reward_p2 = -rewards
        else:
            reward_p2 = rewards[1]

        # 7. 更新观测队列 -> 得到 (s')
        raw_obs_p1_new = raw_env._get_obs(1)
        raw_obs_p2_new = raw_env._get_obs(2)
        p1_dq.append(raw_obs_p1_new)
        p2_dq.append(raw_obs_p2_new)

        obs_p2_next_stack = np.concatenate(list(p2_dq))

        # 8. 计算下一状态价值 V(s')
        # 如果游戏结束，V(s') 应当为 0
        if term or trunc:
            next_val_p2 = 0.0
        else:
            next_val_p2 = ai_agent.get_value(obs_p2_next_stack, CONFIG["device"])

        # ==========================================
        # 9. 核心逻辑：计算 TD Error 并调整温度
        # 公式: TD = r + V' - V (忽略 gamma 或设 gamma=1.0 以严格匹配你的描述)
        # ==========================================

        # 为了让效果更明显，我们可以给 reward 加一点权重，或者保留原样
        td_error = reward_p2 + next_val_p2 - current_val_p2

        # --- 动态调整规则 ---
        if td_error >= 0:
            # 局面比预期好 (V' > V) 或者 得分了 (r > 0)
            # AI: "优势在我，我要浪一点" -> 温度升高
            current_temp += CONFIG["temp_step"]
        else:
            # 局面比预期差 (V' < V) 或者 丢分了 (r < 0)
            # AI: "情况不妙，我要认真了" -> 温度降低
            current_temp -= CONFIG["temp_step"]

        # 限制温度范围
        current_temp = max(CONFIG["min_temp"], min(CONFIG["max_temp"], current_temp))

        # ==========================================

        # 10. 渲染画面与数据
        raw_env.render()

        # 在屏幕上绘制数据
        screen = pygame.display.get_surface()
        if screen:
            # 绘制背景框
            pygame.draw.rect(screen, (0, 0, 0), (10, 10, 350, 100))

            # 1. 显示温度 (AI 智商状态)
            if current_temp < 0.2:
                status = "Serious (Try Hard)"
                color = (255, 50, 50)  # 红 - 认真
            elif current_temp < 1.5:
                status = "Normal"
                color = (255, 255, 0)  # 黄 - 正常
            else:
                status = "Relaxed (Random)"
                color = (50, 255, 50)  # 绿 - 休闲

            txt_temp = font.render(f"AI Temp: {current_temp:.2f} | {status}", True, color)
            screen.blit(txt_temp, (20, 20))

            # 2. 显示 TD Error
            txt_td = font.render(f"TD Error: {td_error:.4f} (r={reward_p2})", True, (200, 200, 255))
            screen.blit(txt_td, (20, 50))

            # 3. 显示价值估计
            txt_val = font.render(f"Value(s): {current_val_p2:.3f}", True, (200, 200, 255))
            screen.blit(txt_val, (20, 80))

            pygame.display.flip()

        clock.tick(60)

        # 局间重置
        if term or trunc:
            print(f"本局结束 | Temp: {current_temp:.2f}")
            p1_dq = deque([np.zeros(12) for _ in range(4)], maxlen=4)
            p2_dq = deque([np.zeros(12) for _ in range(4)], maxlen=4)
            raw_obs_p1 = raw_env._get_obs(1)
            raw_obs_p2 = raw_env._get_obs(2)
            for _ in range(4):
                p1_dq.append(raw_obs_p1)
                p2_dq.append(raw_obs_p2)
            env.reset()

    pygame.quit()


if __name__ == "__main__":
    run_adaptive_game()