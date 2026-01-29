# entities.py
import pygame
import math
from constants import *


class Entity:
    def __init__(self, x, y, radius, color):
        self.x, self.y = x, y
        self.radius = radius
        self.color = color
        self.vx, self.vy = 0, 0

    def apply_physics(self):
        self.x += self.vx
        self.y += self.vy
        if self.y + self.radius < GROUND_Y:
            self.vy += GRAVITY
        else:
            self.y = GROUND_Y - self.radius
            self.vy = 0

    def draw_slime(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.rect(screen, COLOR_BG, (self.x - self.radius, self.y, self.radius * 2, self.radius))


class SlimeBall(Entity):
    def __init__(self, x, y, radius, color):
        super().__init__(x, y, radius, color)
        self.speed_multiplier = 1.0

    def draw_ball(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

    def update(self):
        self.apply_physics()
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = abs(self.vx) * 0.8
        elif self.x + self.radius > WIDTH:
            self.x = WIDTH - self.radius
            self.vx = -abs(self.vx) * 0.8

    def check_player_collision(self, slime):
        dx = self.x - slime.x
        dy = self.y - slime.y
        dist = math.sqrt(dx ** 2 + dy ** 2)

        if dist < (self.radius + slime.radius) and self.y < slime.y:
            angle = math.atan2(dy, dx)

            # --- 核心修改：让物理反馈更细腻 ---
            # 1. 基础反弹力降低，让 AI 有机会“卸力”或“吊球”
            base_bounce = 2.5 * self.speed_multiplier

            # 2. 玩家速度贡献降低 (从2.0降到1.0)，防止一碰就到最大速度
            player_impact_x = slime.vx * 1.0

            # 3. 计算新的速度：X由撞击点和玩家移动共同决定，Y主要由撞击位置决定
            self.vx = math.cos(angle) * base_bounce * 2.0 + player_impact_x
            # 给予一个向上的基础升力，角度越正上方，升力越大
            self.vy = math.sin(angle) * base_bounce - (6.5 * self.speed_multiplier) + (slime.vy * 0.5)

            current_speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
            dynamic_max_speed = BALL_MAX_SPEED * self.speed_multiplier

            # 限制最高速
            if current_speed > dynamic_max_speed:
                scale = dynamic_max_speed / current_speed
                self.vx *= scale
                self.vy *= scale

            # 防止穿透
            overlap = (self.radius + slime.radius) - dist
            self.x += math.cos(angle) * overlap
            self.y += math.sin(angle) * overlap
            return True
        return False

    def check_net_collision(self):
        # 保持原有的网碰撞逻辑
        if abs(self.x - NET_X) < (self.radius + NET_WIDTH / 2) and self.y > NET_Y:
            if self.x < NET_X:
                self.x = NET_X - NET_WIDTH / 2 - self.radius
                self.vx = -abs(self.vx) * 0.8
            else:
                self.x = NET_X + NET_WIDTH / 2 + self.radius
                self.vx = abs(self.vx) * 0.8