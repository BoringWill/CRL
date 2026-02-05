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
        # 落地判定：如果底部超过地面
        if self.y + self.radius >= GROUND_Y:
            self.y = GROUND_Y - self.radius  # 修正位置：确保圆心在地面上方一个半径距离
            self.vy = 0  # 速度清零
        else:
            self.vy += GRAVITY  # 在空中才应用重力

    def draw_slime(self, screen):
        # 视觉上依然保持半圆绘制
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

        if dist < (self.radius + slime.radius):
            angle = math.atan2(dy, dx)
            speed = 15.0 * self.speed_multiplier

            # 速度更新
            self.vx = math.cos(angle) * speed + (slime.vx * 0.5)
            self.vy = math.sin(angle) * speed + (slime.vy * 0.5)

            # 速度上限限制
            current_speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
            dynamic_max_speed = BALL_MAX_SPEED * self.speed_multiplier
            if current_speed > dynamic_max_speed:
                scale = dynamic_max_speed / current_speed
                self.vx *= scale
                self.vy *= scale

            # --- 关键修复：硬位移修正 ---
            overlap = (self.radius + slime.radius) - dist
            # 将球推开，防止下一帧还在圆内
            self.x += math.cos(angle) * overlap
            self.y += math.sin(angle) * overlap
            return True
        return False

    def check_net_collision(self):
        # 定义网的矩形边界
        nl = NET_X - NET_WIDTH / 2
        nr = NET_X + NET_WIDTH / 2
        nt = NET_Y
        nb = HEIGHT  # 假设网一直延伸到地下

        # 1. 寻找矩形内距离球心最近的点 (Clamp)
        closest_x = max(nl, min(self.x, nr))
        closest_y = max(nt, min(self.y, nb))

        # 2. 计算球心到这个最近点的距离
        dx = self.x - closest_x
        dy = self.y - closest_y
        distance = math.sqrt(dx ** 2 + dy ** 2)

        # 3. 如果距离小于半径，发生碰撞
        if distance < self.radius and distance > 0:
            overlap = self.radius - distance
            # 计算碰撞法线
            nx = dx / distance
            ny = dy / distance

            # 修正位移：将球推离网
            self.x += nx * overlap
            self.y += ny * overlap

            # 修正速度：反弹（带有摩擦/损耗）
            # 计算点积确定反弹方向
            dot = self.vx * nx + self.vy * ny
            if dot < 0:  # 只有当球向网移动时才反弹
                self.vx -= 1.8 * dot * nx  # 1.8 代表 0.8 的恢复系数
                self.vy -= 1.8 * dot * ny
            return True

        # 特殊处理：如果球心完全进入了网内部 (distance 为 0 的极端情况)
        elif distance == 0:
            # 暴力推开：哪边近推向哪边
            if self.x < NET_X:
                self.x = nl - self.radius
            else:
                self.x = nr + self.radius
            self.vx = -self.vx * 0.5
            return True

        return False