import torch
import numpy as np
import pygame
import sys
from collections import deque
from slime_env import SlimeSelfPlayEnv, FrameStack
from train_gpu import Agent


def enjoy_game(human_mode=False):
    pygame.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    raw_env = SlimeSelfPlayEnv(render_mode="human")
    env = FrameStack(raw_env, n_frames=4)

    agent = Agent().to(device)
    try:
        # 确保文件名与训练保存的文件名一致
        state_dict = torch.load("slime_ppo_gpu.pth", map_location=device)
        agent.load_state_dict(state_dict)
        agent.eval()
        print(">>> 模型加载成功，开始对战！")
    except Exception as e:
        print(f"加载模型失败: {e}")
        pygame.quit()
        return

    obs_p1, _ = env.reset()
    p2_frames = deque(maxlen=4)
    init_p2_raw = raw_env._get_obs(2)
    for _ in range(4): p2_frames.append(init_p2_raw)
    obs_p2 = np.concatenate(list(p2_frames), axis=0)

    clock = pygame.time.Clock()
    run = True

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
        if not run: break

        # B. 决策
        if human_mode:
            keys = pygame.key.get_pressed()
            action_p1 = 0
            if keys[pygame.K_a]:
                action_p1 = 1
            elif keys[pygame.K_d]:
                action_p1 = 2
            if keys[pygame.K_w]: action_p1 = 3
        else:
            with torch.no_grad():
                input_p1 = torch.FloatTensor(obs_p1).unsqueeze(0).to(device)
                logits_p1 = agent.actor(input_p1)
                action_p1 = torch.argmax(logits_p1, dim=1).item()

        with torch.no_grad():
            input_p2 = torch.FloatTensor(obs_p2).unsqueeze(0).to(device)
            logits_p2 = agent.actor(input_p2)
            action_p2 = torch.argmax(logits_p2, dim=1).item()

        # C. 推进环境
        # 传入元组动作，接收 5 个返回值
        obs_p1, reward, term, trunc, info = env.step((action_p1, action_p2))

        # 显式渲染画面
        env.render()

        n_obs_p2_raw = info["p2_raw_obs"]
        p2_frames.append(n_obs_p2_raw)
        obs_p2 = np.concatenate(list(p2_frames), axis=0)

        if term or trunc:
            obs_p1, _ = env.reset()
            init_p2_raw = raw_env._get_obs(2)
            p2_frames.clear()
            for _ in range(4): p2_frames.append(init_p2_raw)
            obs_p2 = np.concatenate(list(p2_frames), axis=0)

        clock.tick(60)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    enjoy_game(human_mode=False)