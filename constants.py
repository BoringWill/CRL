# constants.py

# 窗口设置
WIDTH, HEIGHT = 1200, 600
FPS = 60

# --- 物理参数优化 ---
GRAVITY = 0.45          # 重力
BALL_MAX_SPEED = 15    # 最高球速
JUMP_POWER = -10        # 固定跳跃力度 (恢复原值)

# [保留] 玩家移动物理参数 (实现水平惯性)
PLAYER_MAX_SPEED = 10  # 最大水平奔跑速度
PLAYER_ACCEL = 2    # 加速度：起步感
PLAYER_FRICTION = 0.4 # 摩擦力：滑行感

# --- 尺寸设置 ---
BALL_RADIUS = 9
SLIME_RADIUS = 27

# --- 网的设置 ---
NET_WIDTH = 10
NET_HEIGHT = 80
NET_X = WIDTH // 2
NET_Y = HEIGHT - 50 - NET_HEIGHT

# --- 颜色 ---
COLOR_P1 = (255, 100, 100)
COLOR_P2 = (100, 100, 255)
COLOR_BALL = (255, 255, 0)
COLOR_GROUND = (120, 120, 120)
COLOR_NET = (200, 200, 200)
COLOR_BG = (30, 30, 30)
GROUND_Y = HEIGHT - 50