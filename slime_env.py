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
        self.win_score = 10
        self.total_rounds = 0
        self.global_step_in_episode = 0
        self.ball_speed_multiplier = 1.0

        # 步数限制逻辑
        self.max_steps_per_point = 3000
        self.max_total_steps = self.max_steps_per_point * self.win_score
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
            # 【大局重置】：重置所有位置到初始点
            self.p1 = Entity(200, GROUND_Y, SLIME_RADIUS, COLOR_P1)
            self.p2 = Entity(800, GROUND_Y, SLIME_RADIUS, COLOR_P2)
        else:
            # 【小局重置】：只重置速度，不重置位置（满足你的需求）
            self.p1.vx, self.p1.vy = 0, 0
            self.p2.vx, self.p2.vy = 0, 0
            # 下面这两行被注释掉了，因此玩家位置不会被重置
            # self.p1.x, self.p1.y = 200, GROUND_Y - SLIME_RADIUS
            # self.p2.x, self.p2.y = 800, GROUND_Y - SLIME_RADIUS

        # 球始终重置
        self.ball = SlimeBall(0, 0, BALL_RADIUS, COLOR_BALL)
        self.ball.speed_multiplier = self.ball_speed_multiplier
        self.is_serving = True
        self.serve_timer = 45

    def _restrict_player_area(self):
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
        def process_physics(entity, action, is_p1):
            if is_p1:
                move_left = (action == 1)
                move_right = (action == 2)
            else:
                move_left = (action == 2)
                move_right = (action == 1)

            if move_left:
                entity.vx -= PLAYER_ACCEL
            elif move_right:
                entity.vx += PLAYER_ACCEL
            else:
                entity.vx *= PLAYER_FRICTION
                if abs(entity.vx) < 0.1: entity.vx = 0
            entity.vx = np.clip(entity.vx, -PLAYER_MAX_SPEED, PLAYER_MAX_SPEED)
            if action == 3 and (entity.y + entity.radius >= GROUND_Y - 2.0) and entity.vy == 0:
                entity.vy = JUMP_POWER

        process_physics(self.p1, action_p1, True)
        process_physics(self.p2, action_p2, False)

    def step(self, actions):
        action_p1, action_p2 = actions
        self.global_step_in_episode += 1
        self.steps_in_current_point += 1

        reward_p1 = 0.0
        terminated = False
        truncated = False

        # --- 全局步数限制 ---
        if self.global_step_in_episode >= self.max_total_steps:
            if self.p1_score == self.p2_score:
                truncated = True
            else:
                terminated = True
            return self._return_step_data(0.0, terminated, truncated)

        # --- 小局步数限制 ---
        if self.steps_in_current_point >= self.max_steps_per_point:
            self._internal_point_reset(full_reset=False)
            return self._return_step_data(0.0, False, False)

        # 物理更新
        self._apply_player_actions(action_p1, action_p2)
        self.p1.apply_physics()
        self.p2.apply_physics()
        self._restrict_player_area()

        if self.is_serving:
            self.serve_timer -= 1
            current_server = self.p1 if self.server_id == 1 else self.p2
            self.ball.x, self.ball.y = current_server.x, current_server.y - 180
            if self.serve_timer <= 0:
                self.is_serving = False
                self.ball.vy = 2.0
        else:
            SUB_STEPS = 4
            for _ in range(SUB_STEPS):
                self.ball.x += self.ball.vx / SUB_STEPS
                self.ball.y += self.ball.vy / SUB_STEPS
                if self.ball.y + self.ball.radius < GROUND_Y:
                    self.ball.vy += GRAVITY / SUB_STEPS
                if self.ball.check_player_collision(self.p1):
                    self.last_touched_by = 1
                    self.ball.check_net_collision()
                if self.ball.check_player_collision(self.p2):
                    self.last_touched_by = 2
                    self.ball.check_net_collision()
                self.ball.check_net_collision()

        # --- 判分逻辑 (已修复分数跳变 BUG) ---
        if not self.is_serving:
            hit_wall_left = self.ball.x <= self.ball.radius
            hit_wall_right = self.ball.x >= WIDTH - self.ball.radius
            hit_ground = (self.ball.y >= GROUND_Y - self.ball.radius)

            if hit_wall_left or hit_wall_right or hit_ground:
                # 逻辑简化：无论谁最后碰到球，只要球出界/落地，该得分的一方就得分

                if hit_wall_left:
                    # 球击中左墙（P1后方），P1输，P2得1分
                    reward_p1 = -2.0
                    self.p2_score += 1
                elif hit_wall_right:
                    # 球击中右墙（P2后方），P2输，P1得1分
                    reward_p1 = 2.0
                    self.p1_score += 1
                elif hit_ground:
                    if self.ball.x < WIDTH / 2:
                        # 球落在左侧（P1侧），P1输，P2得1分
                        reward_p1 = -2.0
                        self.p2_score += 1
                    else:
                        # 球落在右侧（P2侧），P2输，P1得1分
                        reward_p1 = 2.0
                        self.p1_score += 1

                return self._handle_score_change(reward_p1)

        return self._return_step_data(reward_p1, terminated, truncated)

    def _handle_score_change(self, reward_p1):
        terminated = False
        self.total_rounds += 1
        self.server_id = 3 - self.server_id  # 交换发球权

        p1_reaches = self.p1_score >= self.win_score
        p2_reaches = self.p2_score >= self.win_score

        if p1_reaches or p2_reaches:
            terminated = True
        else:
            # 只有这里调用 False，表示是小局结束，不重置位置
            self._internal_point_reset(full_reset=False)

        return self._return_step_data(reward_p1, terminated, False)

    def _return_step_data(self, reward_p1, terminated, truncated):
        return (self._get_obs(1), self._get_obs(2), reward_p1, terminated, truncated,
                {"p1_score": int(self.p1_score), "p2_score": int(self.p2_score)})

    def render(self):
        if self.render_mode != "human": return
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Slime Volleyball")
            self.font = pygame.font.SysFont("Arial", 36, bold=True)
            self.clock = pygame.time.Clock()

        self.screen.fill(COLOR_BG)
        pygame.draw.rect(self.screen, COLOR_GROUND, (0, GROUND_Y, WIDTH, 50))
        pygame.draw.rect(self.screen, COLOR_NET, (NET_X - NET_WIDTH / 2, NET_Y, NET_WIDTH, NET_HEIGHT))

        if self.ball.y > -50: self.ball.draw_ball(self.screen)
        self.p1.draw_slime(self.screen)
        self.p2.draw_slime(self.screen)

        score_surface = self.font.render(f"{self.p1_score}  -  {self.p2_score}", True, (50, 50, 50))
        self.screen.blit(score_surface, (WIDTH // 2 - 50, 20))

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