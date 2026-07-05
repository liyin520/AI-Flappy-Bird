"""Flappy Bird game core logic.

Provides FlappyBirdGame class that can be driven by
both human keyboard input and NEAT AI.
"""

import random
import math
import pygame
from src import config


class Bird:
    """The player bird with simple physics."""

    def __init__(self):
        self.x = config.BIRD_X
        self.y = config.BIRD_START_Y
        self.vel_y = 0
        self.radius = config.BIRD_RADIUS

    def reset(self):
        self.y = config.BIRD_START_Y
        self.vel_y = 0

    def jump(self):
        self.vel_y = config.JUMP_VELOCITY

    def update(self):
        self.vel_y += config.GRAVITY
        if self.vel_y > config.BIRD_MAX_VEL:
            self.vel_y = config.BIRD_MAX_VEL
        self.y += self.vel_y

    def get_rect(self):
        """Return collision rect (slightly smaller than visual for fairness)."""
        shrink = 2
        r = self.radius - shrink
        return pygame.Rect(
            self.x - r,
            self.y - r,
            r * 2,
            r * 2,
        )

    def draw(self, surface):
        """Draw the bird as a circle."""
        # Body
        pygame.draw.circle(surface, config.YELLOW,
                           (int(self.x), int(self.y)), self.radius)
        # Outline
        pygame.draw.circle(surface, config.ORANGE,
                           (int(self.x), int(self.y)), self.radius, 2)
        # Eye
        eye_x = int(self.x) + 5
        eye_y = int(self.y) - 4
        pygame.draw.circle(surface, config.WHITE, (eye_x, eye_y), 5)
        pygame.draw.circle(surface, config.BLACK, (eye_x, eye_y), 2)
        # Beak
        beak_points = [
            (int(self.x) + self.radius, int(self.y)),
            (int(self.x) + self.radius + 8, int(self.y) - 2),
            (int(self.x) + self.radius, int(self.y) + 4),
        ]
        pygame.draw.polygon(surface, config.ORANGE, beak_points)


class PipePair:
    """A pair of top and bottom pipes with a gap."""

    def __init__(self, x):
        self.x = x
        self.width = config.PIPE_WIDTH
        self.gap = config.PIPE_GAP
        self.gap_y = random.randint(
            config.PIPE_MIN_Y + self.gap // 2,
            config.PIPE_MAX_Y - self.gap // 2,
        )
        self.passed = False
        self.scored = False

    @property
    def top_height(self):
        return self.gap_y - self.gap // 2

    @property
    def bottom_y(self):
        return self.gap_y + self.gap // 2

    def get_top_rect(self):
        return pygame.Rect(self.x, 0, self.width, self.top_height)

    def get_bottom_rect(self):
        return pygame.Rect(
            self.x, self.bottom_y,
            self.width,
            config.WINDOW_HEIGHT - config.GROUND_HEIGHT - self.bottom_y,
        )

    def collides_with(self, bird_rect):
        return bird_rect.colliderect(self.get_top_rect()) or \
               bird_rect.colliderect(self.get_bottom_rect())

    def is_off_screen(self):
        return self.x + self.width < 0

    def update(self):
        self.x -= config.PIPE_SPEED

    def draw(self, surface):
        """Draw the pipe pair with a simple pillar look."""
        top_rect = self.get_top_rect()
        bottom_rect = self.get_bottom_rect()

        # Pipes
        pygame.draw.rect(surface, config.GREEN, top_rect)
        pygame.draw.rect(surface, config.GREEN, bottom_rect)

        # Pipe cap (wider rim at the gap end)
        cap_height = 20
        cap_extra = 4  # extra width on each side

        # Top pipe cap (bottom of top pipe)
        top_cap_rect = pygame.Rect(
            self.x - cap_extra,
            self.top_height - cap_height,
            self.width + cap_extra * 2,
            cap_height,
        )
        pygame.draw.rect(surface, config.GREEN_DARK, top_cap_rect)

        # Bottom pipe cap (top of bottom pipe)
        bottom_cap_rect = pygame.Rect(
            self.x - cap_extra,
            self.bottom_y,
            self.width + cap_extra * 2,
            cap_height,
        )
        pygame.draw.rect(surface, config.GREEN_DARK, bottom_cap_rect)

        # Dark outline
        pygame.draw.rect(surface, config.BLACK, top_rect, 2)
        pygame.draw.rect(surface, config.BLACK, bottom_rect, 2)
        pygame.draw.rect(surface, config.BLACK, top_cap_rect, 2)
        pygame.draw.rect(surface, config.BLACK, bottom_cap_rect, 2)


class FlappyBirdGame:
    """Core game logic, usable by both manual and AI modes."""

    def __init__(self):
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        self.alive = True
        self.frame_count = 0
        self.pipe_timer = 0

    def reset(self):
        """Reset the game to initial state."""
        self.bird.reset()
        self.pipes.clear()
        self.score = 0
        self.alive = True
        self.frame_count = 0
        self.pipe_timer = 0

    def step(self, jump=False):
        """Advance one frame.

        Args:
            jump: Whether the bird should jump this frame.
        """
        if not self.alive:
            return

        self.frame_count += 1

        # Bird
        if jump:
            self.bird.jump()
        self.bird.update()

        # Spawn pipes
        self.pipe_timer += 1
        if self.pipe_timer >= config.PIPE_SPAWN_INTERVAL:
            self.pipes.append(PipePair(config.WINDOW_WIDTH))
            self.pipe_timer = 0

        # Update pipes
        for pipe in self.pipes:
            pipe.update()

        # Remove off-screen pipes
        self.pipes = [p for p in self.pipes if not p.is_off_screen()]

        # ---- Collision detection ----
        bird_rect = self.bird.get_rect()

        # Ground / ceiling collision
        ground_y = config.WINDOW_HEIGHT - config.GROUND_HEIGHT
        if bird_rect.top <= 0 or bird_rect.bottom >= ground_y:
            self.alive = False
            return

        # Pipe collision
        for pipe in self.pipes:
            if pipe.collides_with(bird_rect):
                self.alive = False
                return

        # Scoring
        for pipe in self.pipes:
            if not pipe.scored and pipe.x + pipe.width < self.bird.x:
                pipe.scored = True
                self.score += 1

    def get_inputs(self):
        """Return 6 normalized inputs for the NEAT neural network.

        Returns:
            tuple of 6 floats: (bird_y, velocity, dx_to_pipe, gap_y, gap_size, dy_to_gap)
        """
        next_pipe = None
        for pipe in self.pipes:
            if not pipe.scored:
                next_pipe = pipe
                break

        if next_pipe:
            dx = next_pipe.x - self.bird.x
            gap_y = next_pipe.gap_y
        else:
            dx = config.WINDOW_WIDTH
            gap_y = config.WINDOW_HEIGHT // 2

        h = config.WINDOW_HEIGHT
        return (
            self.bird.y / h,                    # [0, 1] bird vertical position
            self.bird.vel_y / config.BIRD_MAX_VEL,  # [-1, 1] velocity
            dx / config.WINDOW_WIDTH,            # [0, ~1] horizontal distance to next pipe
            gap_y / h,                           # [0, 1] pipe gap center
            config.PIPE_GAP / h,                 # [0, 1] pipe gap size
            (self.bird.y - gap_y) / h,           # [-1, 1] vertical offset from gap center
        )

    def get_fitness(self):
        """Calculate fitness for NEAT evaluation.

        Fitness rewards both survival time and scoring.
        """
        return self.score * 100 + self.frame_count

    def draw(self, surface, show_debug=False):
        """Render the current game state.

        Args:
            surface: Pygame surface to draw on.
            show_debug: If True, draw AI debug info.
        """
        # Sky background
        surface.fill(config.BLUE_SKY)

        # Pipes
        for pipe in self.pipes:
            pipe.draw(surface)

        # Ground
        ground_y = config.WINDOW_HEIGHT - config.GROUND_HEIGHT
        ground_rect = pygame.Rect(0, ground_y, config.WINDOW_WIDTH, config.GROUND_HEIGHT)
        pygame.draw.rect(surface, config.BROWN, ground_rect)
        pygame.draw.rect(surface, config.GREEN_DARK,
                         (0, ground_y, config.WINDOW_WIDTH, 5))

        # Bird (only when alive)
        if self.alive:
            self.bird.draw(surface)

        # Score
        font = pygame.font.SysFont('Arial', 48)
        score_surf = font.render(str(self.score), True, config.WHITE)
        score_rect = score_surf.get_rect(center=(config.WINDOW_WIDTH // 2, 50))
        # Shadow for readability
        shadow_surf = font.render(str(self.score), True, config.BLACK)
        shadow_rect = shadow_surf.get_rect(center=(config.WINDOW_WIDTH // 2 + 2, 52))
        surface.blit(shadow_surf, shadow_rect)
        surface.blit(score_surf, score_rect)

        # Debug overlay
        if show_debug and self.alive:
            inputs = self.get_inputs()
            font_small = pygame.font.SysFont('Arial', 16)
            labels = ['bird_y', 'velocity', 'pipe_dx', 'gap_y', 'gap_size', 'dy_to_gap']
            for i, (label, val) in enumerate(zip(labels, inputs)):
                text = font_small.render(f'{label}: {val:.3f}', True, config.WHITE)
                surface.blit(text, (10, 10 + i * 20))

            # Draw line to next pipe
            next_pipe = None
            for pipe in self.pipes:
                if not pipe.scored:
                    next_pipe = pipe
                    break
            if next_pipe:
                pygame.draw.line(
                    surface, config.RED,
                    (int(self.bird.x), int(self.bird.y)),
                    (int(next_pipe.x), next_pipe.gap_y),
                    1,
                )
