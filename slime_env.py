import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from entities import Entity, SlimeBall
from constants import *
from collections import deque

class SlimeSelfPlayEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        # 维度：12个物理状态 + 1个“可否击球”标签 = 13
        self.observation_space = spaces.Box(low=-1, high=1, shape=(13,), dtype=np.float32)

        self.screen = None
        self.clock = None
        self.font = None

        self.p1_score = 0
        self.p2_score = 0
        self.win_score = 10
        self.global_step_in_episode = 0
        self.ball_speed_multiplier = 1.0

        # --- 连击判定变量 ---
        self.last_touched_by = None
        self.collision_cooldown = 0

    def _get_obs(self, player_id):
        p1, p2, b = self.p1, self.p2, self.ball
        # 标签定义：1.0 = 可以合法接球；-1.0 = 已接过一次，再碰违例
        can_hit = 1.0 if self.last_touched_by != player_id else -1.0

        if player_id == 1:
            obs = [p1.x, p1.y, p1.vx, p1.vy, p2.x, p2.y, p2.vx, p2.vy, b.x, b.y, b.vx, b.vy]
            obs_norm = np.array(obs, dtype=np.float32)
            obs_norm[[0, 4, 8]] /= WIDTH
            obs_norm[[1, 5, 9]] /= HEIGHT
            obs_norm[[2, 3, 6, 7, 10, 11]] /= 15.0
            final_obs = (obs_norm * 2.0) - 1.0
            return np.append(final_obs, can_hit)
        else:
            obs = [WIDTH - p2.x, p2.y, -p2.vx, p2.vy, WIDTH - p1.x, p1.y, -p1.vx, p1.vy, WIDTH - b.x, b.y, -b.vx, b.vy]
            obs_norm = np.array(obs, dtype=np.float32)
            obs_norm[[0, 4, 8]] /= WIDTH
            obs_norm[[1, 5, 9]] /= HEIGHT
            obs_norm[[2, 3, 6, 7, 10, 11]] /= 15.0
            final_obs = (obs_norm * 2.0) - 1.0
            return np.append(final_obs, can_hit)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.p1_score = 0
        self.p2_score = 0
        self.global_step_in_episode = 0
        self._internal_point_reset(full_reset=True)
        return self._get_obs(1), {}

    def _internal_point_reset(self, full_reset=False):
        self.global_step_in_episode = 0
        self.last_touched_by = None
        self.collision_cooldown = 0
        self.p1 = Entity(200, GROUND_Y, SLIME_RADIUS, COLOR_P1)
        self.p2 = Entity(800, GROUND_Y, SLIME_RADIUS, COLOR_P2)
        total_points = self.p1_score + self.p2_score
        spawn_x = 200 if total_points % 2 == 0 else 800
        self.ball = SlimeBall(spawn_x, 150, BALL_RADIUS, COLOR_BALL)
        self.ball.speed_multiplier = self.ball_speed_multiplier
        self.ball.vx = 0
        self.ball.vy = 1.0 * self.ball_speed_multiplier

    def step(self, actions):
        action_p1, action_p2 = actions
        self.global_step_in_episode += 1
        reward_p1 = 0.0
        terminated = False
        truncated = False

        # 1. 执行动作
        for p, a in [(self.p1, action_p1), (self.p2, action_p2)]:
            p.vx = 0
            if a == 1: p.vx = -PLAYER_SPEED
            elif a == 2: p.vx = PLAYER_SPEED
            if a == 3 and p.vy == 0: p.vy = JUMP_POWER

        # 2. 物理更新
        self.p1.apply_physics()
        self.p2.apply_physics()
        self.ball.update()
        self._custom_net_collision()

        # 3. 碰撞检测逻辑（违例判定）
        if self.collision_cooldown > 0:
            self.collision_cooldown -= 1
        else:
            collision_threshold_sq = (self.ball.radius + self.p1.radius) ** 2
            dist_p1_sq = (self.ball.x - self.p1.x) ** 2 + (self.ball.y - self.p1.y) ** 2
            dist_p2_sq = (self.ball.x - self.p2.x) ** 2 + (self.ball.y - self.p2.y) ** 2

            foul_detected = False
            if dist_p1_sq <= collision_threshold_sq:
                if self.last_touched_by == 1:
                    reward_p1 = -2.0
                    self.p2_score += 1
                    foul_detected = True
                else:
                    self.last_touched_by = 1
                    self.collision_cooldown = 6
            elif dist_p2_sq <= collision_threshold_sq:
                if self.last_touched_by == 2:
                    reward_p1 = 2.0
                    self.p1_score += 1
                    foul_detected = True
                else:
                    self.last_touched_by = 2
                    self.collision_cooldown = 6

            if foul_detected:
                if self.p1_score >= self.win_score or self.p2_score >= self.win_score:
                    terminated = True
                else:
                    self._internal_point_reset(full_reset=False)
                return self._return_step_data(reward_p1, terminated, truncated)

        # 4. 物理反弹处理
        self.ball.check_player_collision(self.p1)
        self.ball.check_player_collision(self.p2)

        # 5. 区域限制
        self.p1.x = max(self.p1.radius, min(NET_X - NET_WIDTH / 2 - self.p1.radius - 2, self.p1.x))
        self.p2.x = max(NET_X + NET_WIDTH / 2 + self.p2.radius + 2, min(WIDTH - self.p2.radius, self.p2.x))

        # --- 新增：墙壁出界判定 (打手出界/击球出界) ---
        hit_wall = False
        if self.ball.x <= self.ball.radius:  # 撞击左墙 (P1侧)
            hit_wall = True
            if self.last_touched_by == 1:
                reward_p1, self.p2_score = -2.0, self.p2_score + 1 # P1接球失误出界
            else:
                reward_p1, self.p1_score = 2.0, self.p1_score + 1  # P2击球用力过猛出界
        elif self.ball.x >= WIDTH - self.ball.radius:  # 撞击右墙 (P2侧)
            hit_wall = True
            if self.last_touched_by == 2:
                reward_p1, self.p1_score = 2.0, self.p1_score + 1  # P2接球失误出界
            else:
                reward_p1, self.p2_score = -2.0, self.p2_score + 1 # P1击球用力过猛出界

        if hit_wall:
            if self.p1_score >= self.win_score or self.p2_score >= self.win_score:
                terminated = True
            else:
                self._internal_point_reset(full_reset=False)
            return self._return_step_data(reward_p1, terminated, truncated)

        # 6. 落地得分判定
        if self.ball.y >= GROUND_Y - self.ball.radius:
            if self.ball.x < WIDTH / 2:
                reward_p1 = -2.0
                self.p2_score += 1
            else:
                reward_p1 = 2.0
                self.p1_score += 1

            if self.p1_score >= self.win_score or self.p2_score >= self.win_score:
                terminated = True
            else:
                self._internal_point_reset(full_reset=False)

        return self._return_step_data(reward_p1, terminated, truncated)

    def _return_step_data(self, reward_p1, terminated, truncated):
        return (self._get_obs(1), self._get_obs(2), reward_p1, terminated, truncated,
                {"p2_raw_obs": self._get_obs(2), "p1_score": self.p1_score, "p2_score": self.p2_score,
                 "episode_steps": self.global_step_in_episode})

    def _custom_net_collision(self):
        b = self.ball
        nl, nr = NET_X - NET_WIDTH / 2, NET_X + NET_WIDTH / 2
        if b.y + b.radius >= NET_Y and b.y - b.vy < NET_Y:
            if nl < b.x < nr:
                b.vy = -abs(b.vy) * 0.8
                b.y = NET_Y - b.radius
                return
        if b.y >= NET_Y:
            if nl - b.radius < b.x < NET_X and b.vx > 0:
                b.vx = -abs(b.vx); b.x = nl - b.radius
            elif NET_X < b.x < nr + b.radius and b.vx < 0:
                b.vx = abs(b.vx); b.x = nr + b.radius

    def render(self):
        if self.render_mode != "human": return
        if self.screen is None:
            pygame.init(); self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Slime Volleyball RL")
            self.font = pygame.font.SysFont("Arial", 24); self.clock = pygame.time.Clock()
        self.screen.fill(COLOR_BG)
        pygame.draw.rect(self.screen, COLOR_GROUND, (0, GROUND_Y, WIDTH, 50))
        pygame.draw.rect(self.screen, COLOR_NET, (NET_X - NET_WIDTH / 2, NET_Y, NET_WIDTH, NET_HEIGHT))
        score_txt = self.font.render(
            f"P1: {self.p1_score} | P2: {self.p2_score}", True, (0, 0, 0))
        self.screen.blit(score_txt, (WIDTH // 2 - 55 , 20))
        self.p1.draw_slime(self.screen); self.p2.draw_slime(self.screen); self.ball.draw_ball(self.screen)
        pygame.display.flip(); self.clock.tick(60)

class FrameStack(gym.Wrapper):
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
        self.observation_space = spaces.Box(
            low=np.tile(env.observation_space.low, n_frames),
            high=np.tile(env.observation_space.high, n_frames),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.frames.clear()
        for _ in range(self.n_frames): self.frames.append(obs)
        return np.concatenate(list(self.frames), axis=0), info

    def step(self, actions):
        obs_p1, obs_p2, reward, term, trunc, info = self.env.step(actions)
        self.frames.append(obs_p1)
        info["p2_raw_obs"] = obs_p2
        return np.concatenate(list(self.frames), axis=0), reward, term, trunc, info