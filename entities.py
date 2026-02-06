import pygame
import math
from constants import *


class Entity:
    def __init__(self, x, y, radius, color):
        self.x, self.y = x, y
        self.radius = radius
        self.color = color
        self.vx, self.vy = 0, 0

        # [新增] 跳跃蓄力计时器
        # > 0 表示还可以继续按住跳跃键来获得升力
        self.jump_timer = 0

    def apply_physics(self):
        self.x += self.vx
        self.y += self.vy

        # 落地判定
        if self.y + self.radius >= GROUND_Y:
            self.y = GROUND_Y - self.radius
            self.vy = 0
        else:
            self.vy += GRAVITY  # 应用重力

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
        pass

    def check_player_collision(self, slime):
        dx = self.x - slime.x
        dy = self.y - slime.y
        dist = math.sqrt(dx ** 2 + dy ** 2)

        if dist < (self.radius + slime.radius):
            angle = math.atan2(dy, dx)
            speed = 15.0 * self.speed_multiplier

            self.vx = math.cos(angle) * speed + (slime.vx * 0.5)
            self.vy = math.sin(angle) * speed + (slime.vy * 0.5)

            current_speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
            dynamic_max_speed = BALL_MAX_SPEED * self.speed_multiplier
            if current_speed > dynamic_max_speed:
                scale = dynamic_max_speed / current_speed
                self.vx *= scale
                self.vy *= scale

            overlap = (self.radius + slime.radius) - dist
            self.x += math.cos(angle) * overlap
            self.y += math.sin(angle) * overlap
            return True
        return False

    def check_net_collision(self):
        nl = NET_X - NET_WIDTH / 2
        nr = NET_X + NET_WIDTH / 2
        nt = NET_Y
        nb = HEIGHT

        closest_x = max(nl, min(self.x, nr))
        closest_y = max(nt, min(self.y, nb))

        dx = self.x - closest_x
        dy = self.y - closest_y
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if distance < self.radius and distance > 0:
            overlap = self.radius - distance
            nx = dx / distance
            ny = dy / distance

            self.x += nx * overlap
            self.y += ny * overlap

            dot = self.vx * nx + self.vy * ny
            if dot < 0:
                self.vx -= 1.8 * dot * nx
                self.vy -= 1.8 * dot * ny
            return True
        elif distance == 0:
            if self.x < NET_X:
                self.x = nl - self.radius
            else:
                self.x = nr + self.radius
            self.vx = -self.vx * 0.5
            return True
        return False