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
        self.observation_space = spaces.Box(low=-1, high=1, shape=(13,), dtype=np.float32)

        self.screen = None
        self.clock = None
        self.font = None

        self.p1_score = 0
        self.p2_score = 0
        self.win_score = 100
        self.total_rounds = 0
        self.global_step_in_episode = 0
        self.ball_speed_multiplier = 1.0

        self.max_steps_per_point = 2500
        self.steps_in_current_point = 0

        self.last_touched_by = None
        self.is_serving = False
        self.serve_timer = 0
        self.server_id = 1

    def _get_obs(self, player_id):
        p1, p2, b = self.p1, self.p2, self.ball
        can_hit = 1.0

        if player_id == 1:
            obs = [p1.x, p1.y, p1.vx, p1.vy, p2.x, p2.y, p2.vx, p2.vy, b.x, b.y, b.vx, b.vy]
        else:
            obs = [
                WIDTH - p2.x, p2.y, -p2.vx, p2.vy,
                WIDTH - p1.x, p1.y, -p1.vx, p1.vy,
                WIDTH - b.x, b.y, -b.vx, b.vy
            ]

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
        self.total_rounds = 0
        self.global_step_in_episode = 0
        self.steps_in_current_point = 0

        self.server_id = np.random.choice([1, 2])
        self._internal_point_reset(full_reset=True)
        return self._get_obs(1), {}

    def _internal_point_reset(self, full_reset=False):
        self.last_touched_by = None
        self.steps_in_current_point = 0

        if full_reset:
            self.p1 = Entity(200, GROUND_Y, SLIME_RADIUS, COLOR_P1)
            self.p2 = Entity(800, GROUND_Y, SLIME_RADIUS, COLOR_P2)
        else:
            self.p1.vx, self.p1.vy = 0, 0
            self.p2.vx, self.p2.vy = 0, 0

        self.ball = SlimeBall(0, 0, BALL_RADIUS, COLOR_BALL)
        self.ball.speed_multiplier = self.ball_speed_multiplier
        self.is_serving = True
        self.serve_timer = 45

    def _restrict_player_area(self):
        # 玩家可以紧贴球网
        p1_min_x = self.p1.radius
        p1_max_x = NET_X - NET_WIDTH / 2 - self.p1.radius

        p2_min_x = NET_X + NET_WIDTH / 2 + self.p2.radius
        p2_max_x = WIDTH - self.p2.radius

        FORBIDDEN_ZONE_WIDTH = 100
        if self.is_serving:
            if self.server_id == 1:
                p1_max_x -= FORBIDDEN_ZONE_WIDTH
            else:
                p2_min_x += FORBIDDEN_ZONE_WIDTH

        self.p1.x = np.clip(self.p1.x, p1_min_x, p1_max_x)
        self.p2.x = np.clip(self.p2.x, p2_min_x, p2_max_x)

    def _apply_player_actions(self, action_p1, action_p2):
        self.p1.vx = 0
        if action_p1 == 1:
            self.p1.vx = -PLAYER_SPEED
        elif action_p1 == 2:
            self.p1.vx = PLAYER_SPEED
        if action_p1 == 3 and (self.p1.y + self.p1.radius >= GROUND_Y - 2.0) and self.p1.vy == 0:
            self.p1.vy = JUMP_POWER

        self.p2.vx = 0
        if action_p2 == 1:
            self.p2.vx = PLAYER_SPEED
        elif action_p2 == 2:
            self.p2.vx = -PLAYER_SPEED
        if action_p2 == 3 and (self.p2.y + self.p2.radius >= GROUND_Y - 2.0) and self.p2.vy == 0:
            self.p2.vy = JUMP_POWER

    def step(self, actions):
        action_p1, action_p2 = actions
        self.global_step_in_episode += 1
        self.steps_in_current_point += 1

        reward_p1 = 0.0
        terminated = False
        truncated = False

        if self.steps_in_current_point >= self.max_steps_per_point:
            self.p1_score += 1
            self.p2_score += 1
            return self._handle_score_change(0.0)

        # 1. 玩家物理更新
        self._apply_player_actions(action_p1, action_p2)
        self.p1.apply_physics()
        self.p2.apply_physics()
        self._restrict_player_area()

        # 2. 球的物理更新
        if self.is_serving:
            self.serve_timer -= 1
            current_server = self.p1 if self.server_id == 1 else self.p2
            self.ball.x, self.ball.y = current_server.x, current_server.y - 180
            if self.serve_timer <= 0:
                self.is_serving = False
                self.ball.vy = 2.0
        else:
            SUB_STEPS = 4  # 每一帧细分为4次检测
            for _ in range(SUB_STEPS):
                # 应用一小部分重力和位移
                self.ball.x += self.ball.vx / SUB_STEPS
                self.ball.y += self.ball.vy / SUB_STEPS

                # --- 核心修改：移除左右墙壁的强制反弹检测 ---
                # 之前这里的代码会强制 ball.x = radius 并反转速度
                # 现在移除后，球可以飞出边界 (x < 0 或 x > WIDTH)，从而触发下方的判分逻辑

                # 仅保留地面重力逻辑
                if self.ball.y + self.ball.radius < GROUND_Y:
                    self.ball.vy += GRAVITY / SUB_STEPS

                # 碰撞解析顺序：玩家碰撞 -> 球网检测
                if self.ball.check_player_collision(self.p1):
                    self.last_touched_by = 1
                    self.ball.check_net_collision()
                if self.ball.check_player_collision(self.p2):
                    self.last_touched_by = 2
                    self.ball.check_net_collision()

                # 最终球网碰撞兜底
                self.ball.check_net_collision()

        # 3. 判分逻辑
        if not self.is_serving:
            # 允许检测球是否真的出界了（因为上面删除了反弹逻辑，这里可以正确捕获）
            hit_wall_left = self.ball.x <= self.ball.radius
            hit_wall_right = self.ball.x >= WIDTH - self.ball.radius
            hit_ground = (self.ball.y >= GROUND_Y - self.ball.radius)

            if hit_wall_left or hit_wall_right or hit_ground:
                if hit_wall_left:
                    # 左墙判定：如果是 P1 最后碰的，算 P1 失误（P2得分）；否则是 P2 打出界（P1得分）
                    reward_p1, self.p2_score = (-2.0, self.p2_score + 1) if self.last_touched_by == 1 else (
                    2.0, self.p1_score + 1)
                elif hit_wall_right:
                    # 右墙判定
                    reward_p1, self.p1_score = (2.0, self.p1_score + 1) if self.last_touched_by == 2 else (
                    -2.0, self.p2_score + 1)
                elif hit_ground:
                    if self.ball.x < WIDTH / 2:
                        reward_p1, self.p2_score = -2.0, self.p2_score + 1
                    else:
                        reward_p1, self.p1_score = 2.0, self.p1_score + 1
                return self._handle_score_change(reward_p1)

        return self._return_step_data(reward_p1, terminated, truncated)

    def _handle_score_change(self, reward_p1):
        terminated = False
        self.total_rounds += 1
        self.server_id = 3 - self.server_id
        p1_reaches = self.p1_score >= self.win_score
        p2_reaches = self.p2_score >= self.win_score

        if p1_reaches or p2_reaches:
            terminated = True
            if p1_reaches and p2_reaches: reward_p1 = 0.0
        else:
            self._internal_point_reset(full_reset=False)
        return self._return_step_data(reward_p1, terminated, False)

    def _return_step_data(self, reward_p1, terminated, truncated):
        return (self._get_obs(1), self._get_obs(2), reward_p1, terminated, truncated,
                {"p2_raw_obs": self._get_obs(2), "p1_score": self.p1_score, "p2_score": self.p2_score})

    def render(self):
        if self.render_mode != "human": return
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Slime Volleyball Symmetry Mode")
            self.font = pygame.font.SysFont("Arial", 36, bold=True)
            self.clock = pygame.time.Clock()

        self.screen.fill(COLOR_BG)
        pygame.draw.rect(self.screen, COLOR_GROUND, (0, GROUND_Y, WIDTH, 50))
        pygame.draw.rect(self.screen, COLOR_NET, (NET_X - NET_WIDTH / 2, NET_Y, NET_WIDTH, NET_HEIGHT))

        if self.ball.y > -50: self.ball.draw_ball(self.screen)
        self.p1.draw_slime(self.screen)
        self.p2.draw_slime(self.screen)

        score_str = f"{self.p1_score}   -   {self.p2_score}"
        score_surface = self.font.render(score_str, True, (50, 50, 50))
        score_rect = score_surface.get_rect(center=(WIDTH // 2, 40))
        self.screen.blit(score_surface, score_rect)

        pygame.display.flip()
        self.clock.tick(60)


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
        obs_p1, info = self.env.reset(seed=seed, options=options)
        self.frames.clear()
        for _ in range(self.n_frames): self.frames.append(obs_p1)
        return np.concatenate(list(self.frames), axis=0), info

    def step(self, actions):
        obs_p1, obs_p2, reward, term, trunc, info = self.env.step(actions)
        self.frames.append(obs_p1)
        info["p2_raw_obs"] = obs_p2
        return np.concatenate(list(self.frames), axis=0), reward, term, trunc, info