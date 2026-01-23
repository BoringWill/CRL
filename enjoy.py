import torch
import torch.nn as nn
import numpy as np
import pygame
import sys
from collections import deque
from slime_env import SlimeSelfPlayEnv, FrameStack


# 确保与训练架构一致
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


def enjoy_game():
    # 让用户选择角色
    print("请选择你的角色:")
    print("1: 人类控制 P1 (左侧), AI 控制 P2 (右侧)")
    print("2: AI 控制 P1 (左侧), 人类控制 P2 (右侧)")
    choice = input("输入序号 (1 或 2): ")
    human_side = 1 if choice == "1" else 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device} | 你控制的是 P{human_side}")

    # 1. 初始化环境和包装器
    raw_env = SlimeSelfPlayEnv(render_mode="human")
    env = FrameStack(raw_env, n_frames=4)
    clock = pygame.time.Clock()

    # 2. 加载模型
    agent = Agent().to(device)
    try:
        # 路径请根据实际情况修改
        state_dict = torch.load("模型集/slime_ppo_gpu_v4.pth", map_location=device)
        agent.load_state_dict(state_dict)
        agent.eval()
        print(">>> 模型加载成功，开始对战！")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 3. 初始重置逻辑
    def reset_all():
        o1, _ = env.reset()
        p2_dq = deque(maxlen=4)
        init_p2_raw = raw_env._get_obs(2)
        for _ in range(4): p2_dq.append(init_p2_raw)
        o2 = np.concatenate(list(p2_dq), axis=0)
        return o1, o2, p2_dq

    obs_p1, obs_p2, p2_frames = reset_all()
    raw_env.render()

    run = True
    while run:
        # A. 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # 获取键盘输入 (用于人类控制)
        keys = pygame.key.get_pressed()
        human_action = 0
        if keys[pygame.K_a]: human_action = 1
        elif keys[pygame.K_d]: human_action = 2
        if keys[pygame.K_w]: human_action = 3

        # B. 决策逻辑
        with torch.no_grad():
            # P1 决策逻辑
            if human_side == 1:
                action_p1 = human_action
            else:
                input_p1 = torch.FloatTensor(obs_p1).unsqueeze(0).to(device)
                logits_p1 = agent.actor(input_p1)
                probs_p1 = torch.distributions.Categorical(logits=logits_p1)
                action_p1 = probs_p1.sample().item()

            # P2 决策逻辑
            if human_side == 2:
                action_p2 = human_action
            else:
                input_p2 = torch.FloatTensor(obs_p2).unsqueeze(0).to(device)
                logits_p2 = agent.actor(input_p2)
                probs_p2 = torch.distributions.Categorical(logits=logits_p2)
                action_p2 = probs_p2.sample().item()

        # C. 执行动作
        obs_p1, reward, term, trunc, info = env.step((action_p1, action_p2))

        # D. 渲染并锁定帧率
        raw_env.render()
        clock.tick(60)

        # E. 更新 P2 观测数据
        n_obs_p2_raw = info["p2_raw_obs"]
        p2_frames.append(n_obs_p2_raw)
        obs_p2 = np.concatenate(list(p2_frames), axis=0)

        # F. 游戏结束彻底重置
        if term or trunc:
            obs_p1, obs_p2, p2_frames = reset_all()
            raw_env.render()

    pygame.quit()


if __name__ == "__main__":
    enjoy_game()