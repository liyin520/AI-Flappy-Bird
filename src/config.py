"""Game configuration constants."""

# Window
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60

# Bird physics
BIRD_X = 200
BIRD_START_Y = 300
BIRD_RADIUS = 15
GRAVITY = 0.5
JUMP_VELOCITY = -8
BIRD_MAX_VEL = 10

# Pipe settings
PIPE_WIDTH = 60
PIPE_GAP = 200        # vertical gap between top and bottom pipe
PIPE_SPEED = 4
PIPE_SPAWN_INTERVAL = 90   # frames between new pipe pairs
PIPE_MIN_Y = 100           # min gap center Y
PIPE_MAX_Y = 500           # max gap center Y (WINDOW_HEIGHT - GROUND_HEIGHT - 100)

# Ground
GROUND_HEIGHT = 80

# Colors (RGB)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
GREEN_DARK = (0, 150, 0)
BLUE = (100, 150, 255)
BLUE_SKY = (135, 206, 235)
YELLOW = (255, 255, 0)
RED = (200, 0, 0)
GRAY = (120, 120, 120)
BROWN = (139, 90, 43)
ORANGE = (255, 165, 0)

# NEAT configuration
NEAT_POP_SIZE = 100
NEAT_MAX_FRAMES = 10000

# Training
TRAIN_GENERATIONS = 50
